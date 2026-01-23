"""
Uncertainty Quantification - Confidence intervals on ML predictions.

Provides calibrated uncertainty estimates for model predictions using
multiple techniques: ensemble disagreement, conformal prediction,
and Bayesian approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a prediction."""

    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    std_deviation: float
    entropy: float  # Prediction entropy (higher = more uncertain)
    method: str
    is_reliable: bool  # Whether prediction is within reliable range
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "prediction": round(self.prediction, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "confidence_level": self.confidence_level,
            "std_deviation": round(self.std_deviation, 4),
            "entropy": round(self.entropy, 4),
            "method": self.method,
            "is_reliable": self.is_reliable,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def interval_width(self) -> float:
        """Width of confidence interval."""
        return self.upper_bound - self.lower_bound


@dataclass
class CalibrationResult:
    """Model calibration metrics."""

    expected_confidence: List[float]
    observed_accuracy: List[float]
    calibration_error: float  # ECE
    reliability_diagram: Dict[str, List[float]]
    is_well_calibrated: bool
    calibration_curve: List[Tuple[float, float]]

    def to_dict(self) -> Dict:
        return {
            "expected_confidence": [round(c, 4) for c in self.expected_confidence],
            "observed_accuracy": [round(a, 4) for a in self.observed_accuracy],
            "calibration_error": round(self.calibration_error, 4),
            "is_well_calibrated": self.is_well_calibrated,
            "calibration_curve": [(round(e, 4), round(o, 4)) for e, o in self.calibration_curve],
        }


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""

    # Confidence level for intervals
    confidence_level: float = 0.95

    # Ensemble settings
    n_bootstrap: int = 100
    bootstrap_ratio: float = 0.8

    # Conformal prediction settings
    calibration_size: float = 0.2

    # Reliability thresholds
    max_entropy_threshold: float = 0.7
    min_confidence_threshold: float = 0.6


class UncertaintyQuantifier:
    """
    Quantify prediction uncertainty using multiple methods.

    Methods:
    1. Bootstrap ensemble disagreement
    2. Monte Carlo dropout (for neural networks)
    3. Conformal prediction
    4. Calibration analysis
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self._calibration_scores: List[float] = []
        self._calibration_labels: List[int] = []
        self._conformal_nonconformity: List[float] = []

    def estimate_bootstrap_uncertainty(
        self,
        models: List[Any],
        X: Union[np.ndarray, pd.DataFrame],
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from ensemble of bootstrap models.

        Args:
            models: List of trained models (bootstrap ensemble)
            X: Input features (single sample)

        Returns:
            UncertaintyEstimate from ensemble disagreement
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = []

        for model in models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[0, 1]
            else:
                pred = model.predict(X)[0]
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # Confidence interval
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(predictions, alpha / 2 * 100)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100)

        # Entropy (measure of prediction spread)
        entropy = self._calculate_entropy(predictions)

        # Check reliability
        is_reliable = std_pred < 0.3 and entropy < self.config.max_entropy_threshold

        return UncertaintyEstimate(
            prediction=float(mean_pred),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.config.confidence_level,
            std_deviation=float(std_pred),
            entropy=float(entropy),
            method="bootstrap_ensemble",
            is_reliable=is_reliable,
        )

    def estimate_dropout_uncertainty(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        n_forward_passes: int = 50,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using MC Dropout (for neural networks).

        Args:
            model: Neural network with dropout layers
            X: Input features
            n_forward_passes: Number of stochastic forward passes

        Returns:
            UncertaintyEstimate from dropout samples
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Check if model supports training mode for dropout
        has_train_mode = hasattr(model, "train") and hasattr(model, "eval")

        predictions = []

        for _ in range(n_forward_passes):
            if has_train_mode:
                model.train()  # Enable dropout

            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[0, 1]
            else:
                pred = model.predict(X)[0]
            predictions.append(pred)

        if has_train_mode:
            model.eval()  # Disable dropout

        predictions = np.array(predictions)

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        alpha = 1 - self.config.confidence_level
        lower = np.percentile(predictions, alpha / 2 * 100)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100)

        entropy = self._calculate_entropy(predictions)

        is_reliable = std_pred < 0.25

        return UncertaintyEstimate(
            prediction=float(mean_pred),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.config.confidence_level,
            std_deviation=float(std_pred),
            entropy=float(entropy),
            method="mc_dropout",
            is_reliable=is_reliable,
        )

    def fit_conformal(
        self,
        model: Any,
        X_cal: Union[np.ndarray, pd.DataFrame],
        y_cal: np.ndarray,
    ):
        """
        Fit conformal predictor on calibration set.

        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        if isinstance(X_cal, pd.DataFrame):
            X_cal = X_cal.values

        # Calculate nonconformity scores
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_cal)
            # Nonconformity = 1 - probability of true class
            nonconformity = []
            for i, y in enumerate(y_cal):
                if y < len(proba[i]):
                    nonconformity.append(1 - proba[i, int(y)])
                else:
                    nonconformity.append(1.0)
        else:
            preds = model.predict(X_cal)
            nonconformity = np.abs(preds - y_cal).tolist()

        self._conformal_nonconformity = sorted(nonconformity)
        logger.info(f"Fitted conformal predictor on {len(y_cal)} samples")

    def estimate_conformal_uncertainty(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using conformal prediction.

        Args:
            model: Trained model
            X: Input features

        Returns:
            UncertaintyEstimate with conformal bounds
        """
        if not self._conformal_nonconformity:
            raise ValueError("Must call fit_conformal first")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get prediction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            pred = proba[1] if len(proba) == 2 else np.max(proba)
        else:
            pred = model.predict(X)[0]

        # Get threshold from nonconformity scores
        alpha = 1 - self.config.confidence_level
        n = len(self._conformal_nonconformity)
        threshold_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        threshold_idx = min(threshold_idx, n - 1)
        threshold = self._conformal_nonconformity[threshold_idx]

        # Compute prediction interval
        lower = max(0, pred - threshold)
        upper = min(1, pred + threshold)

        # Estimate entropy from threshold
        entropy = threshold / 2  # Simplified

        is_reliable = threshold < 0.4

        return UncertaintyEstimate(
            prediction=float(pred),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.config.confidence_level,
            std_deviation=float(threshold / 2),
            entropy=float(entropy),
            method="conformal",
            is_reliable=is_reliable,
        )

    def calibrate_predictions(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """
        Analyze model calibration.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            CalibrationResult with calibration metrics
        """
        # Store for later use
        self._calibration_scores = y_proba.tolist()
        self._calibration_labels = y_true.tolist()

        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        expected_confidence = []
        observed_accuracy = []
        calibration_curve = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_confidence = np.mean(y_proba[mask])
                avg_accuracy = np.mean(y_true[mask])
                expected_confidence.append(avg_confidence)
                observed_accuracy.append(avg_accuracy)
                calibration_curve.append((avg_confidence, avg_accuracy))

        # Calculate Expected Calibration Error (ECE)
        ece = 0
        for i in range(n_bins):
            mask = bin_indices == i
            bin_size = np.sum(mask)
            if bin_size > 0:
                avg_confidence = np.mean(y_proba[mask])
                avg_accuracy = np.mean(y_true[mask])
                ece += (bin_size / len(y_true)) * abs(avg_accuracy - avg_confidence)

        # Well calibrated if ECE < 0.1
        is_well_calibrated = ece < 0.1

        return CalibrationResult(
            expected_confidence=expected_confidence,
            observed_accuracy=observed_accuracy,
            calibration_error=float(ece),
            reliability_diagram={
                "bins": expected_confidence,
                "accuracy": observed_accuracy,
            },
            is_well_calibrated=is_well_calibrated,
            calibration_curve=calibration_curve,
        )

    def _calculate_entropy(self, predictions: np.ndarray) -> float:
        """Calculate entropy of prediction distribution."""
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

        # For binary classification probabilities
        entropy = -np.mean(
            predictions * np.log(predictions) + (1 - predictions) * np.log(1 - predictions)
        )

        # Normalize to 0-1 range
        max_entropy = np.log(2)  # Maximum entropy for binary
        return float(min(entropy / max_entropy, 1.0))

    def get_prediction_with_uncertainty(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = "bootstrap",
        models: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get prediction with uncertainty estimate.

        Args:
            model: Main model
            X: Input features
            method: "bootstrap", "dropout", or "conformal"
            models: List of models for bootstrap method

        Returns:
            Dict with prediction and uncertainty info
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get base prediction
        if hasattr(model, "predict_proba"):
            base_pred = model.predict_proba(X)[0, 1]
        else:
            base_pred = model.predict(X)[0]

        # Get uncertainty estimate
        if method == "bootstrap" and models:
            uncertainty = self.estimate_bootstrap_uncertainty(models, X)
        elif method == "dropout":
            uncertainty = self.estimate_dropout_uncertainty(model, X)
        elif method == "conformal" and self._conformal_nonconformity:
            uncertainty = self.estimate_conformal_uncertainty(model, X)
        else:
            # Fallback to simple variance estimation
            uncertainty = UncertaintyEstimate(
                prediction=float(base_pred),
                lower_bound=max(0, float(base_pred) - 0.1),
                upper_bound=min(1, float(base_pred) + 0.1),
                confidence_level=self.config.confidence_level,
                std_deviation=0.1,
                entropy=0.5,
                method="default",
                is_reliable=True,
            )

        # Determine trading signal with uncertainty
        signal = "HOLD"
        if uncertainty.is_reliable:
            if uncertainty.lower_bound > 0.6:
                signal = "STRONG_BUY"
            elif uncertainty.prediction > 0.55 and uncertainty.lower_bound > 0.45:
                signal = "BUY"
            elif uncertainty.upper_bound < 0.4:
                signal = "STRONG_SELL"
            elif uncertainty.prediction < 0.45 and uncertainty.upper_bound < 0.55:
                signal = "SELL"

        return {
            "prediction": uncertainty.prediction,
            "signal": signal,
            "confidence_interval": [uncertainty.lower_bound, uncertainty.upper_bound],
            "std_deviation": uncertainty.std_deviation,
            "entropy": uncertainty.entropy,
            "is_reliable": uncertainty.is_reliable,
            "method": uncertainty.method,
        }


class EnsembleUncertainty:
    """
    Uncertainty estimation using ensemble of different model types.

    Combines predictions from multiple model architectures to
    estimate epistemic (model) uncertainty.
    """

    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        """
        Initialize with dictionary of models.

        Args:
            models: Dict mapping model name to model instance
            feature_names: List of feature names
        """
        self.models = models
        self.feature_names = feature_names
        self._weights: Dict[str, float] = {name: 1.0 for name in models}

    def set_weights(self, weights: Dict[str, float]):
        """Set model weights based on validation performance."""
        self._weights = weights

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Get weighted ensemble prediction with uncertainty.

        Args:
            X: Input features

        Returns:
            Dict with ensemble prediction and disagreement metrics
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = {}
        weighted_sum = 0
        weight_total = 0

        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[0, 1]
            else:
                pred = model.predict(X)[0]

            predictions[name] = float(pred)
            weight = self._weights.get(name, 1.0)
            weighted_sum += pred * weight
            weight_total += weight

        # Weighted average prediction
        ensemble_pred = weighted_sum / weight_total if weight_total > 0 else 0.5

        # Model disagreement (epistemic uncertainty)
        pred_values = list(predictions.values())
        disagreement = np.std(pred_values)

        # Prediction range
        pred_min = min(pred_values)
        pred_max = max(pred_values)

        # Agreement score (1 = all models agree)
        agreement = 1 - (pred_max - pred_min)

        return {
            "ensemble_prediction": float(ensemble_pred),
            "model_predictions": predictions,
            "disagreement": float(disagreement),
            "prediction_range": [float(pred_min), float(pred_max)],
            "agreement_score": float(agreement),
            "is_consensus": agreement > 0.7,
        }


def create_uncertainty_quantifier(
    config: Optional[UncertaintyConfig] = None,
) -> UncertaintyQuantifier:
    """Factory function to create uncertainty quantifier."""
    return UncertaintyQuantifier(config=config)
