"""
ML Model Ensemble for Trading Predictions.

Combines predictions from multiple models using:
- Weighted voting based on recent accuracy
- Confidence-weighted aggregation
- Automatic model selection based on market regime
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction from a single model."""

    model_name: str
    signal: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble."""

    signal: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    agreement_score: float
    contributing_models: List[str]
    model_predictions: List[ModelPrediction]
    voting_method: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "contributing_models": self.contributing_models,
            "voting_method": self.voting_method,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelEnsemble:
    """
    Ensemble of ML models for trading predictions.

    Features:
    - Multiple voting strategies (majority, weighted, confidence)
    - Automatic weight adjustment based on recent performance
    - Model disagreement detection
    - Regime-specific model selection
    """

    def __init__(
        self,
        min_agreement: float = 0.5,
        decay_factor: float = 0.95,
        performance_window: int = 50,
    ):
        """
        Initialize ensemble.

        Args:
            min_agreement: Minimum agreement threshold for valid signal
            decay_factor: Exponential decay for historical weights
            performance_window: Number of predictions to track for performance
        """
        self.min_agreement = min_agreement
        self.decay_factor = decay_factor
        self.performance_window = performance_window

        # Registered models
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}

        # Performance tracking
        self._predictions_history: List[Dict] = []
        self._model_accuracy: Dict[str, List[float]] = {}

    def register_model(
        self,
        name: str,
        model: Any,
        initial_weight: float = 1.0,
    ) -> None:
        """
        Register a model with the ensemble.

        Args:
            name: Unique model identifier
            model: Model instance (must have predict method)
            initial_weight: Starting weight for this model
        """
        self.models[name] = model
        self.model_weights[name] = initial_weight
        self._model_accuracy[name] = []
        logger.info(f"Registered model: {name} (weight={initial_weight})")

    def unregister_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        self.models.pop(name, None)
        self.model_weights.pop(name, None)
        self._model_accuracy.pop(name, None)

    def predict(
        self,
        features: np.ndarray,
        voting_method: Literal["majority", "weighted", "confidence"] = "weighted",
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction.

        Args:
            features: Input features for prediction
            voting_method: How to combine predictions
                - majority: Simple majority vote
                - weighted: Weighted by model performance
                - confidence: Weighted by prediction confidence

        Returns:
            EnsemblePrediction with combined signal
        """
        if not self.models:
            raise ValueError("No models registered in ensemble")

        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            try:
                pred = self._get_model_prediction(name, model, features)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not predictions:
            # Return FLAT if no predictions available
            return EnsemblePrediction(
                signal="FLAT",
                confidence=0.0,
                agreement_score=0.0,
                contributing_models=[],
                model_predictions=[],
                voting_method=voting_method,
            )

        # Combine predictions
        if voting_method == "majority":
            signal, confidence, agreement = self._majority_vote(predictions)
        elif voting_method == "weighted":
            signal, confidence, agreement = self._weighted_vote(predictions)
        else:  # confidence
            signal, confidence, agreement = self._confidence_vote(predictions)

        # Check agreement threshold
        if agreement < self.min_agreement:
            signal = "FLAT"
            confidence *= 0.5  # Reduce confidence on disagreement

        return EnsemblePrediction(
            signal=signal,
            confidence=confidence,
            agreement_score=agreement,
            contributing_models=[p.model_name for p in predictions],
            model_predictions=predictions,
            voting_method=voting_method,
        )

    def _get_model_prediction(
        self,
        name: str,
        model: Any,
        features: np.ndarray,
    ) -> Optional[ModelPrediction]:
        """Get prediction from a single model."""
        try:
            # Handle different model interfaces
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features.reshape(1, -1))[0]
                pred_class = np.argmax(proba)

                # Map class to signal
                signal_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
                signal = signal_map.get(pred_class, "FLAT")
                confidence = float(proba[pred_class])

                return ModelPrediction(
                    model_name=name,
                    signal=signal,
                    confidence=confidence,
                    probabilities={
                        "SHORT": float(proba[0]) if len(proba) > 0 else 0,
                        "FLAT": float(proba[1]) if len(proba) > 1 else 0,
                        "LONG": float(proba[2]) if len(proba) > 2 else 0,
                    },
                )

            elif hasattr(model, "predict"):
                pred = model.predict(features.reshape(1, -1))[0]

                # Convert numeric prediction to signal
                if isinstance(pred, (int, np.integer)):
                    signal_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
                    signal = signal_map.get(int(pred), "FLAT")
                else:
                    signal = str(pred).upper()
                    if signal not in ["LONG", "SHORT", "FLAT"]:
                        signal = "FLAT"

                return ModelPrediction(
                    model_name=name,
                    signal=signal,
                    confidence=0.6,  # Default confidence for non-proba models
                )

            else:
                logger.warning(f"Model {name} has no predict method")
                return None

        except Exception as e:
            logger.warning(f"Error getting prediction from {name}: {e}")
            return None

    def _majority_vote(
        self,
        predictions: List[ModelPrediction],
    ) -> Tuple[str, float, float]:
        """Simple majority voting."""
        votes = {"LONG": 0, "SHORT": 0, "FLAT": 0}

        for pred in predictions:
            votes[pred.signal] += 1

        total = len(predictions)
        winning_signal = max(votes.keys(), key=lambda k: votes[k])
        agreement = votes[winning_signal] / total

        # Average confidence of winning signal
        winning_preds = [p for p in predictions if p.signal == winning_signal]
        avg_confidence = np.mean([p.confidence for p in winning_preds])

        return winning_signal, avg_confidence, agreement

    def _weighted_vote(
        self,
        predictions: List[ModelPrediction],
    ) -> Tuple[str, float, float]:
        """Voting weighted by model performance."""
        votes = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}

        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 1.0)
            votes[pred.signal] += weight * pred.confidence

        total_weight = sum(votes.values())
        if total_weight == 0:
            return "FLAT", 0.0, 0.0

        winning_signal = max(votes.keys(), key=lambda k: votes[k])
        confidence = votes[winning_signal] / total_weight
        agreement = votes[winning_signal] / total_weight

        return winning_signal, confidence, agreement

    def _confidence_vote(
        self,
        predictions: List[ModelPrediction],
    ) -> Tuple[str, float, float]:
        """Voting weighted by prediction confidence."""
        votes = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}

        for pred in predictions:
            votes[pred.signal] += pred.confidence

        total = sum(votes.values())
        if total == 0:
            return "FLAT", 0.0, 0.0

        winning_signal = max(votes.keys(), key=lambda k: votes[k])
        confidence = votes[winning_signal] / total
        agreement = votes[winning_signal] / total

        return winning_signal, confidence, agreement

    def update_performance(
        self,
        predictions: List[ModelPrediction],
        actual_signal: str,
    ) -> None:
        """
        Update model weights based on actual outcome.

        Args:
            predictions: Predictions that were made
            actual_signal: What actually happened (LONG, SHORT, FLAT)
        """
        for pred in predictions:
            name = pred.model_name
            correct = 1.0 if pred.signal == actual_signal else 0.0

            # Track accuracy
            if name not in self._model_accuracy:
                self._model_accuracy[name] = []
            self._model_accuracy[name].append(correct)

            # Keep only recent history
            if len(self._model_accuracy[name]) > self.performance_window:
                self._model_accuracy[name] = self._model_accuracy[name][-self.performance_window :]

            # Update weight based on recent accuracy
            recent_accuracy = np.mean(self._model_accuracy[name])
            self.model_weights[name] = max(0.1, recent_accuracy * 2)  # Scale to 0.1-2.0

        logger.debug(f"Updated weights: {self.model_weights}")

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics for all registered models."""
        stats = {}
        for name in self.models:
            accuracy_history = self._model_accuracy.get(name, [])
            stats[name] = {
                "weight": self.model_weights.get(name, 1.0),
                "recent_accuracy": np.mean(accuracy_history) if accuracy_history else 0.0,
                "predictions_tracked": len(accuracy_history),
            }
        return stats


class RegimeBasedEnsemble(ModelEnsemble):
    """
    Ensemble that selects models based on market regime.

    Different models may perform better in different market conditions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regime_models: Dict[str, List[str]] = {
            "bull": [],
            "bear": [],
            "sideways": [],
            "volatile": [],
        }

    def register_model(
        self,
        name: str,
        model: Any,
        initial_weight: float = 1.0,
        suitable_regimes: Optional[List[str]] = None,
    ) -> None:
        """Register model with regime preferences."""
        super().register_model(name, model, initial_weight)

        # Assign to regimes
        if suitable_regimes:
            for regime in suitable_regimes:
                if regime in self.regime_models:
                    self.regime_models[regime].append(name)
        else:
            # Default: suitable for all regimes
            for regime in self.regime_models:
                self.regime_models[regime].append(name)

    def predict_for_regime(
        self,
        features: np.ndarray,
        regime: str,
        voting_method: str = "weighted",
    ) -> EnsemblePrediction:
        """
        Get prediction using only regime-appropriate models.

        Args:
            features: Input features
            regime: Current market regime
            voting_method: Voting method to use

        Returns:
            EnsemblePrediction from regime-specific models
        """
        # Get models for this regime
        regime_model_names = self.regime_models.get(regime.lower(), [])

        if not regime_model_names:
            # Fall back to all models
            return self.predict(features, voting_method)

        # Filter to regime-specific models
        original_models = self.models.copy()
        self.models = {k: v for k, v in original_models.items() if k in regime_model_names}

        try:
            result = self.predict(features, voting_method)
        finally:
            self.models = original_models

        return result
