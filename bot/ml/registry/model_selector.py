"""
Model Selector for regime-based dynamic model switching.

Automatically selects the best model based on market conditions and regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..models.base import BaseMLModel, ModelPrediction
from .model_registry import ModelRegistry, RegisteredModel


class ModelSelectionStrategy(Enum):
    """Strategy for selecting models."""

    SINGLE_BEST = "single_best"  # Always use best accuracy model
    REGIME_BASED = "regime_based"  # Select based on market regime
    ENSEMBLE = "ensemble"  # Ensemble multiple models
    ADAPTIVE = "adaptive"  # Learn which model works best


@dataclass
class ModelPriority:
    """Priority configuration for model selection by regime."""

    # Regime -> ordered list of preferred model types
    CRYPTO_PRIORITIES: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "volatile": ["transformer", "lstm", "xgboost"],
            "strong_bull": ["xgboost", "random_forest", "lstm"],
            "bull": ["xgboost", "random_forest", "lstm"],
            "sideways": ["lstm", "xgboost", "transformer"],
            "bear": ["lstm", "xgboost", "transformer"],
            "strong_bear": ["transformer", "lstm", "xgboost"],
        }
    )

    COMMODITY_PRIORITIES: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "volatile": ["lstm", "xgboost", "transformer"],
            "strong_bull": ["random_forest", "xgboost", "lstm"],
            "bull": ["random_forest", "xgboost", "lstm"],
            "sideways": ["xgboost", "lstm", "random_forest"],
            "bear": ["lstm", "xgboost", "random_forest"],
            "strong_bear": ["lstm", "xgboost", "transformer"],
        }
    )

    STOCK_PRIORITIES: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "volatile": ["transformer", "lstm", "xgboost"],
            "strong_bull": ["xgboost", "transformer", "random_forest"],
            "bull": ["xgboost", "random_forest", "lstm"],
            "sideways": ["random_forest", "xgboost", "lstm"],
            "bear": ["lstm", "transformer", "xgboost"],
            "strong_bear": ["transformer", "lstm", "xgboost"],
        }
    )

    def get_priorities(self, market_type: str, regime: str) -> List[str]:
        """Get model priorities for given market and regime."""
        regime = regime.lower()
        priorities_map = {
            "crypto": self.CRYPTO_PRIORITIES,
            "commodity": self.COMMODITY_PRIORITIES,
            "stock": self.STOCK_PRIORITIES,
        }
        priorities = priorities_map.get(market_type, self.CRYPTO_PRIORITIES)
        return priorities.get(regime, ["xgboost", "random_forest", "lstm"])


@dataclass
class SelectionResult:
    """Result from model selection."""

    model: Optional[BaseMLModel]
    model_type: str
    selection_reason: str
    alternatives_available: List[str]
    regime_used: str
    confidence_in_selection: float


class ModelSelector:
    """
    Intelligent model selector based on market conditions.

    Features:
    - Regime-based model selection
    - Ensemble predictions with weighted voting
    - Fallback to best available model
    - Performance tracking for adaptive selection

    Usage:
        selector = ModelSelector(registry)
        model = selector.select_model("BTC/USDT", "crypto", regime="volatile")
        prediction = selector.get_prediction("BTC/USDT", "crypto", X, regime="bull")
    """

    def __init__(
        self,
        registry: ModelRegistry,
        strategy: ModelSelectionStrategy = ModelSelectionStrategy.REGIME_BASED,
        model_priorities: Optional[ModelPriority] = None,
    ):
        self.registry = registry
        self.strategy = strategy
        self.priorities = model_priorities or ModelPriority()

        # Cache loaded models
        self._model_cache: Dict[str, BaseMLModel] = {}

        # Performance tracking for adaptive selection
        self._performance_history: Dict[str, List[float]] = {}

    def select_model(
        self,
        symbol: str,
        market_type: str = "crypto",
        regime: str = "sideways",
    ) -> SelectionResult:
        """
        Select the best model for given conditions.

        Args:
            symbol: Trading symbol
            market_type: Market type
            regime: Current market regime

        Returns:
            SelectionResult with selected model and metadata
        """
        # Get all available models for this symbol
        available_models = self.registry.list_models(
            symbol=symbol,
            market_type=market_type,
            active_only=True,
        )

        if not available_models:
            return SelectionResult(
                model=None,
                model_type="none",
                selection_reason="No models available for this symbol",
                alternatives_available=[],
                regime_used=regime,
                confidence_in_selection=0.0,
            )

        available_types = list(set(m.model_type for m in available_models))

        # Select based on strategy
        if self.strategy == ModelSelectionStrategy.SINGLE_BEST:
            selected = self._select_best_accuracy(available_models)
            reason = "Selected model with highest validation accuracy"

        elif self.strategy == ModelSelectionStrategy.REGIME_BASED:
            selected = self._select_by_regime(
                available_models,
                market_type,
                regime,
            )
            reason = f"Selected based on {regime} regime priority"

        elif self.strategy == ModelSelectionStrategy.ADAPTIVE:
            selected = self._select_adaptive(available_models, symbol)
            reason = "Selected based on historical performance"

        else:
            selected = self._select_best_accuracy(available_models)
            reason = "Default selection by accuracy"

        # Load the selected model
        model = self._load_cached_model(selected)

        alternatives = [t for t in available_types if t != selected.model_type]

        return SelectionResult(
            model=model,
            model_type=selected.model_type,
            selection_reason=reason,
            alternatives_available=alternatives,
            regime_used=regime,
            confidence_in_selection=selected.val_accuracy,
        )

    def _select_best_accuracy(
        self,
        models: List[RegisteredModel],
    ) -> RegisteredModel:
        """Select model with highest validation accuracy."""
        return max(models, key=lambda m: m.val_accuracy)

    def _select_by_regime(
        self,
        models: List[RegisteredModel],
        market_type: str,
        regime: str,
    ) -> RegisteredModel:
        """Select model based on regime priorities."""
        priorities = self.priorities.get_priorities(market_type, regime)
        available_types = {m.model_type: m for m in models}

        # Find first available model type in priority order
        for model_type in priorities:
            if model_type in available_types:
                return available_types[model_type]

        # Fallback to best accuracy
        return self._select_best_accuracy(models)

    def _select_adaptive(
        self,
        models: List[RegisteredModel],
        symbol: str,
    ) -> RegisteredModel:
        """Select based on historical performance tracking."""
        # Check performance history
        for model in models:
            key = f"{symbol}_{model.model_type}"
            if key in self._performance_history:
                history = self._performance_history[key]
                if history:
                    # Use recent accuracy
                    model.metadata["recent_accuracy"] = np.mean(history[-10:])

        # Sort by recent performance if available, otherwise val_accuracy
        return max(
            models,
            key=lambda m: m.metadata.get("recent_accuracy", m.val_accuracy),
        )

    def _load_cached_model(self, registered: RegisteredModel) -> Optional[BaseMLModel]:
        """Load model with caching."""
        cache_key = f"{registered.symbol}_{registered.model_type}_{registered.version}"

        if cache_key not in self._model_cache:
            model = self.registry.load_model(
                symbol=registered.symbol,
                model_type=registered.model_type,
                market_type=registered.market_type,
                version=registered.version,
            )
            if model:
                self._model_cache[cache_key] = model

        return self._model_cache.get(cache_key)

    def get_prediction(
        self,
        symbol: str,
        market_type: str,
        X: np.ndarray,
        regime: str = "sideways",
    ) -> ModelPrediction:
        """
        Get prediction using selected model.

        Args:
            symbol: Trading symbol
            market_type: Market type
            X: Input features
            regime: Current market regime

        Returns:
            ModelPrediction from selected model
        """
        selection = self.select_model(symbol, market_type, regime)

        if selection.model is None:
            # Return default prediction
            return ModelPrediction(
                action="FLAT",
                confidence=0.33,
                probability_long=0.33,
                probability_short=0.33,
                probability_flat=0.34,
                expected_return=0.0,
                model_name="none",
                model_type="none",
                metadata={"error": "No model available"},
            )

        prediction = selection.model.predict(X)
        prediction.metadata["selection_reason"] = selection.selection_reason
        prediction.metadata["regime"] = regime

        return prediction

    def get_ensemble_prediction(
        self,
        symbol: str,
        market_type: str,
        X: np.ndarray,
        model_types: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> ModelPrediction:
        """
        Get ensemble prediction from multiple models.

        Args:
            symbol: Trading symbol
            market_type: Market type
            X: Input features
            model_types: List of model types to ensemble (default: all available)
            weights: Optional weights for each model type

        Returns:
            Ensemble ModelPrediction
        """
        available_models = self.registry.list_models(
            symbol=symbol,
            market_type=market_type,
            active_only=True,
        )

        if not available_models:
            return self.get_prediction(symbol, market_type, X)

        # Filter to requested types
        if model_types:
            available_models = [m for m in available_models if m.model_type in model_types]

        if not available_models:
            return self.get_prediction(symbol, market_type, X)

        # Collect predictions
        predictions = []
        model_weights = []

        for registered in available_models:
            model = self._load_cached_model(registered)
            if model is None:
                continue

            pred = model.predict(X)
            predictions.append(pred)

            # Determine weight
            if weights and registered.model_type in weights:
                w = weights[registered.model_type]
            else:
                w = registered.val_accuracy
            model_weights.append(w)

        if not predictions:
            return self.get_prediction(symbol, market_type, X)

        # Weighted average of probabilities
        total_weight = sum(model_weights)
        if total_weight == 0:
            total_weight = 1

        avg_prob_long = (
            sum(p.probability_long * w for p, w in zip(predictions, model_weights)) / total_weight
        )
        avg_prob_short = (
            sum(p.probability_short * w for p, w in zip(predictions, model_weights)) / total_weight
        )
        avg_prob_flat = (
            sum(p.probability_flat * w for p, w in zip(predictions, model_weights)) / total_weight
        )

        # Normalize
        total_prob = avg_prob_long + avg_prob_short + avg_prob_flat
        if total_prob > 0:
            avg_prob_long /= total_prob
            avg_prob_short /= total_prob
            avg_prob_flat /= total_prob

        # Determine action
        probs = {"LONG": avg_prob_long, "SHORT": avg_prob_short, "FLAT": avg_prob_flat}
        action = max(probs, key=probs.get)
        confidence = probs[action]

        return ModelPrediction(
            action=action,
            confidence=confidence,
            probability_long=avg_prob_long,
            probability_short=avg_prob_short,
            probability_flat=avg_prob_flat,
            expected_return=0.0,
            model_name="ensemble",
            model_type="ensemble",
            metadata={
                "models_used": [m.model_type for m in available_models],
                "weights": dict(zip([m.model_type for m in available_models], model_weights)),
            },
        )

    def record_performance(
        self,
        symbol: str,
        model_type: str,
        accuracy: float,
    ) -> None:
        """
        Record model performance for adaptive selection.

        Args:
            symbol: Trading symbol
            model_type: Model type
            accuracy: Achieved accuracy
        """
        key = f"{symbol}_{model_type}"
        if key not in self._performance_history:
            self._performance_history[key] = []

        self._performance_history[key].append(accuracy)

        # Keep only last 100 records
        if len(self._performance_history[key]) > 100:
            self._performance_history[key] = self._performance_history[key][-100:]

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get selector status."""
        return {
            "strategy": self.strategy.value,
            "cached_models": len(self._model_cache),
            "performance_tracked_symbols": len(self._performance_history),
        }
