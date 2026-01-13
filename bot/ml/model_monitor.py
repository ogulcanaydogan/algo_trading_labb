"""
ML Model Monitoring and Drift Detection Module.

Features:
- Data drift detection (feature distribution shifts)
- Concept drift detection (model performance degradation)
- Prediction calibration
- Automatic retraining triggers
- Model confidence assessment
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"           # Feature distributions changed
    CONCEPT_DRIFT = "concept_drift"     # Relationship between features and target changed
    LABEL_DRIFT = "label_drift"         # Target distribution changed
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution changed


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    # Detection thresholds
    ks_threshold_low: float = 0.1        # KS statistic threshold for low drift
    ks_threshold_high: float = 0.2       # KS statistic threshold for high drift
    psi_threshold_low: float = 0.1       # PSI threshold for low drift
    psi_threshold_high: float = 0.25     # PSI threshold for high drift

    # Performance degradation thresholds
    accuracy_drop_threshold: float = 0.05    # 5% accuracy drop
    sharpe_drop_threshold: float = 0.3       # 0.3 Sharpe ratio drop
    win_rate_drop_threshold: float = 0.05    # 5% win rate drop

    # Monitoring windows
    reference_window: int = 500          # Number of samples for reference distribution
    detection_window: int = 100          # Number of samples for drift detection
    min_samples: int = 50                # Minimum samples for detection

    # Calibration settings
    calibration_window: int = 200        # Samples for calibration
    calibration_bins: int = 10           # Number of bins for calibration

    # Retraining triggers
    auto_retrain_on_drift: bool = True
    retrain_cooldown_hours: int = 24     # Minimum time between retrains
    max_retrains_per_week: int = 3


@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    drift_type: DriftType = DriftType.DATA_DRIFT
    severity: DriftSeverity = DriftSeverity.NONE
    feature_name: Optional[str] = None
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    psi_value: float = 0.0
    mean_shift: float = 0.0
    std_shift: float = 0.0
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "feature_name": self.feature_name,
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "psi_value": round(self.psi_value, 4),
            "mean_shift": round(self.mean_shift, 4),
            "std_shift": round(self.std_shift, 4),
            "details": self.details,
        }


@dataclass
class CalibrationMetrics:
    """Model calibration metrics."""
    expected_calibration_error: float = 0.0    # ECE
    max_calibration_error: float = 0.0         # MCE
    brier_score: float = 0.0
    reliability_curve: List[Tuple[float, float]] = field(default_factory=list)
    is_calibrated: bool = True

    def to_dict(self) -> Dict:
        return {
            "ece": round(self.expected_calibration_error, 4),
            "mce": round(self.max_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "reliability_curve": [(round(x, 2), round(y, 2)) for x, y in self.reliability_curve],
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class PerformanceMetrics:
    """Model performance tracking metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    window_size: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "window_size": self.window_size,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_confidence": round(self.avg_confidence, 4),
        }


class ModelMonitor:
    """
    Monitor ML model health and detect drift.

    Tracks:
    - Feature distribution changes (data drift)
    - Model performance degradation (concept drift)
    - Prediction confidence trends
    - Calibration quality

    Provides:
    - Drift alerts with severity levels
    - Retraining recommendations
    - Calibration adjustments
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        data_dir: str = "data/ml_monitoring",
    ):
        self.config = config or DriftConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Reference distributions (from training)
        self._reference_features: Dict[str, np.ndarray] = {}
        self._reference_predictions: np.ndarray = np.array([])
        self._reference_labels: np.ndarray = np.array([])

        # Current window data
        self._current_features: Dict[str, List[float]] = {}
        self._current_predictions: List[float] = []
        self._current_labels: List[int] = []
        self._current_returns: List[float] = []
        self._current_confidences: List[float] = []

        # History
        self._drift_history: List[DriftMetrics] = []
        self._performance_history: List[PerformanceMetrics] = []
        self._last_retrain: Optional[datetime] = None
        self._retrain_count_this_week: int = 0

        # Calibration
        self._calibration_map: Dict[float, float] = {}  # predicted -> calibrated

        # Load state
        self._load_state()

    def set_reference_data(
        self,
        features: Dict[str, np.ndarray],
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Set reference distributions from training data.

        Args:
            features: Dict of feature_name -> values array
            predictions: Model predictions array
            labels: True labels array
        """
        self._reference_features = {
            name: values[-self.config.reference_window:]
            for name, values in features.items()
        }
        self._reference_predictions = predictions[-self.config.reference_window:]
        self._reference_labels = labels[-self.config.reference_window:]

        # Calculate reference statistics
        self._save_state()

    def add_sample(
        self,
        features: Dict[str, float],
        prediction: float,
        confidence: float,
        label: Optional[int] = None,
        pnl: Optional[float] = None,
    ):
        """
        Add a new prediction sample for monitoring.

        Args:
            features: Feature values for this sample
            prediction: Model prediction (probability or class)
            confidence: Model confidence score
            label: True label (if known)
            pnl: P&L from this prediction (if trade executed)
        """
        # Add to current window
        for name, value in features.items():
            if name not in self._current_features:
                self._current_features[name] = []
            self._current_features[name].append(value)

        self._current_predictions.append(prediction)
        self._current_confidences.append(confidence)

        if label is not None:
            self._current_labels.append(label)

        if pnl is not None:
            self._current_returns.append(pnl)

        # Trim to detection window
        max_size = self.config.detection_window
        for name in self._current_features:
            if len(self._current_features[name]) > max_size:
                self._current_features[name] = self._current_features[name][-max_size:]

        if len(self._current_predictions) > max_size:
            self._current_predictions = self._current_predictions[-max_size:]
            self._current_confidences = self._current_confidences[-max_size:]

        if len(self._current_labels) > max_size:
            self._current_labels = self._current_labels[-max_size:]

        if len(self._current_returns) > max_size:
            self._current_returns = self._current_returns[-max_size:]

    def check_drift(self) -> List[DriftMetrics]:
        """
        Check for all types of drift.

        Returns:
            List of drift metrics for any detected drift
        """
        results = []

        # Check data drift for each feature
        if len(self._current_features) > 0:
            for feature_name, current_values in self._current_features.items():
                if feature_name in self._reference_features:
                    drift = self._check_feature_drift(
                        feature_name,
                        self._reference_features[feature_name],
                        np.array(current_values),
                    )
                    if drift.severity != DriftSeverity.NONE:
                        results.append(drift)
                        self._drift_history.append(drift)

        # Check prediction drift
        if len(self._current_predictions) >= self.config.min_samples:
            if len(self._reference_predictions) > 0:
                pred_drift = self._check_prediction_drift()
                if pred_drift.severity != DriftSeverity.NONE:
                    results.append(pred_drift)
                    self._drift_history.append(pred_drift)

        # Check concept drift (performance degradation)
        if len(self._current_labels) >= self.config.min_samples:
            concept_drift = self._check_concept_drift()
            if concept_drift.severity != DriftSeverity.NONE:
                results.append(concept_drift)
                self._drift_history.append(concept_drift)

        # Trim history
        if len(self._drift_history) > 1000:
            self._drift_history = self._drift_history[-1000:]

        self._save_state()
        return results

    def _check_feature_drift(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftMetrics:
        """Check drift for a single feature using KS test and PSI."""
        if len(current) < self.config.min_samples:
            return DriftMetrics(feature_name=feature_name)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)

        # Population Stability Index
        psi = self._calculate_psi(reference, current)

        # Mean and std shifts
        ref_mean, ref_std = np.mean(reference), np.std(reference)
        cur_mean, cur_std = np.mean(current), np.std(current)

        mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
        std_shift = abs(cur_std - ref_std) / (ref_std + 1e-8)

        # Determine severity
        severity = DriftSeverity.NONE

        if ks_stat >= self.config.ks_threshold_high or psi >= self.config.psi_threshold_high:
            severity = DriftSeverity.HIGH
        elif ks_stat >= self.config.ks_threshold_low or psi >= self.config.psi_threshold_low:
            severity = DriftSeverity.MODERATE
        elif ks_pvalue < 0.05:
            severity = DriftSeverity.LOW

        return DriftMetrics(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            feature_name=feature_name,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            psi_value=psi,
            mean_shift=mean_shift,
            std_shift=std_shift,
            details={
                "ref_mean": float(ref_mean),
                "ref_std": float(ref_std),
                "cur_mean": float(cur_mean),
                "cur_std": float(cur_std),
            },
        )

    def _check_prediction_drift(self) -> DriftMetrics:
        """Check drift in model predictions."""
        current = np.array(self._current_predictions)
        reference = self._reference_predictions

        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        psi = self._calculate_psi(reference, current)

        severity = DriftSeverity.NONE
        if ks_stat >= self.config.ks_threshold_high or psi >= self.config.psi_threshold_high:
            severity = DriftSeverity.HIGH
        elif ks_stat >= self.config.ks_threshold_low or psi >= self.config.psi_threshold_low:
            severity = DriftSeverity.MODERATE

        return DriftMetrics(
            drift_type=DriftType.PREDICTION_DRIFT,
            severity=severity,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            psi_value=psi,
        )

    def _check_concept_drift(self) -> DriftMetrics:
        """Check for concept drift via performance degradation."""
        if len(self._current_labels) < self.config.min_samples:
            return DriftMetrics(drift_type=DriftType.CONCEPT_DRIFT)

        predictions = np.array(self._current_predictions[-len(self._current_labels):])
        labels = np.array(self._current_labels)

        # Calculate current accuracy
        pred_classes = (predictions > 0.5).astype(int)
        current_accuracy = np.mean(pred_classes == labels)

        # Reference accuracy (from training)
        ref_preds = (self._reference_predictions > 0.5).astype(int)
        ref_labels = self._reference_labels[:len(ref_preds)]
        if len(ref_labels) > 0:
            ref_accuracy = np.mean(ref_preds[:len(ref_labels)] == ref_labels)
        else:
            ref_accuracy = 0.6  # Default expected accuracy

        accuracy_drop = ref_accuracy - current_accuracy

        # Calculate win rate from returns
        win_rate = 0.0
        sharpe = 0.0
        if len(self._current_returns) > 0:
            returns = np.array(self._current_returns)
            win_rate = np.mean(returns > 0)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Determine severity
        severity = DriftSeverity.NONE

        if accuracy_drop >= self.config.accuracy_drop_threshold * 2:
            severity = DriftSeverity.CRITICAL
        elif accuracy_drop >= self.config.accuracy_drop_threshold:
            severity = DriftSeverity.HIGH
        elif accuracy_drop >= self.config.accuracy_drop_threshold * 0.5:
            severity = DriftSeverity.MODERATE

        return DriftMetrics(
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=severity,
            details={
                "current_accuracy": float(current_accuracy),
                "reference_accuracy": float(ref_accuracy),
                "accuracy_drop": float(accuracy_drop),
                "win_rate": float(win_rate),
                "sharpe_ratio": float(sharpe),
            },
        )

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins from reference distribution
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)

        # Avoid division by zero
        ref_props = np.clip(ref_props, 1e-6, None)
        cur_props = np.clip(cur_props, 1e-6, None)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return psi

    # =====================
    # CALIBRATION
    # =====================

    def calibrate_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics and build calibration map.

        Args:
            predictions: Model probability predictions
            labels: True binary labels

        Returns:
            Calibration metrics
        """
        n_bins = self.config.calibration_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate reliability
        reliability_curve = []
        calibration_errors = []

        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_pred = np.mean(predictions[mask])
                bin_true = np.mean(labels[mask])
                bin_count = np.sum(mask)

                reliability_curve.append((bin_pred, bin_true))
                calibration_errors.append(abs(bin_pred - bin_true) * bin_count)

                # Build calibration map
                self._calibration_map[bin_centers[i]] = bin_true
            else:
                reliability_curve.append((bin_centers[i], bin_centers[i]))

        # Expected Calibration Error (ECE)
        total_samples = len(predictions)
        ece = sum(calibration_errors) / total_samples if total_samples > 0 else 0

        # Maximum Calibration Error (MCE)
        mce = max([abs(p - t) for p, t in reliability_curve]) if reliability_curve else 0

        # Brier Score
        brier = np.mean((predictions - labels) ** 2)

        # Is model well-calibrated?
        is_calibrated = ece < 0.1 and mce < 0.2

        metrics = CalibrationMetrics(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            reliability_curve=reliability_curve,
            is_calibrated=is_calibrated,
        )

        return metrics

    def get_calibrated_prediction(self, raw_prediction: float) -> float:
        """
        Apply calibration to a raw model prediction.

        Args:
            raw_prediction: Raw model probability

        Returns:
            Calibrated probability
        """
        if not self._calibration_map:
            return raw_prediction

        # Find nearest bin
        bin_centers = sorted(self._calibration_map.keys())
        nearest_bin = min(bin_centers, key=lambda x: abs(x - raw_prediction))

        # Linear interpolation between bins
        idx = bin_centers.index(nearest_bin)

        if raw_prediction <= bin_centers[0]:
            return self._calibration_map[bin_centers[0]]
        elif raw_prediction >= bin_centers[-1]:
            return self._calibration_map[bin_centers[-1]]
        elif raw_prediction < nearest_bin and idx > 0:
            # Interpolate with previous bin
            prev_bin = bin_centers[idx - 1]
            alpha = (raw_prediction - prev_bin) / (nearest_bin - prev_bin)
            return (1 - alpha) * self._calibration_map[prev_bin] + alpha * self._calibration_map[nearest_bin]
        elif raw_prediction > nearest_bin and idx < len(bin_centers) - 1:
            # Interpolate with next bin
            next_bin = bin_centers[idx + 1]
            alpha = (raw_prediction - nearest_bin) / (next_bin - nearest_bin)
            return (1 - alpha) * self._calibration_map[nearest_bin] + alpha * self._calibration_map[next_bin]
        else:
            return self._calibration_map[nearest_bin]

    # =====================
    # PERFORMANCE TRACKING
    # =====================

    def update_performance(self) -> PerformanceMetrics:
        """
        Calculate and store current performance metrics.

        Returns:
            Current performance metrics
        """
        metrics = PerformanceMetrics(
            window_size=len(self._current_labels),
        )

        if len(self._current_labels) < self.config.min_samples:
            return metrics

        predictions = np.array(self._current_predictions[-len(self._current_labels):])
        labels = np.array(self._current_labels)
        pred_classes = (predictions > 0.5).astype(int)

        # Basic metrics
        metrics.accuracy = np.mean(pred_classes == labels)

        # Precision, Recall, F1
        tp = np.sum((pred_classes == 1) & (labels == 1))
        fp = np.sum((pred_classes == 1) & (labels == 0))
        fn = np.sum((pred_classes == 0) & (labels == 1))

        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = (
            2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            if (metrics.precision + metrics.recall) > 0 else 0
        )

        # Trading metrics
        if len(self._current_returns) > 0:
            returns = np.array(self._current_returns)
            metrics.win_rate = np.mean(returns > 0)

            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

            wins = returns[returns > 0]
            losses = abs(returns[returns < 0])
            if len(losses) > 0 and np.sum(losses) > 0:
                metrics.profit_factor = np.sum(wins) / np.sum(losses) if len(wins) > 0 else 0

        metrics.avg_confidence = np.mean(self._current_confidences) if self._current_confidences else 0

        self._performance_history.append(metrics)
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]

        return metrics

    def get_performance_trend(self, window: int = 10) -> Dict:
        """Get recent performance trend."""
        if len(self._performance_history) < 2:
            return {"trend": "unknown", "change": 0}

        recent = self._performance_history[-window:]
        if len(recent) < 2:
            return {"trend": "unknown", "change": 0}

        # Compare first half to second half
        mid = len(recent) // 2
        first_half_acc = np.mean([m.accuracy for m in recent[:mid]])
        second_half_acc = np.mean([m.accuracy for m in recent[mid:]])

        change = second_half_acc - first_half_acc

        if change > 0.02:
            trend = "improving"
        elif change < -0.02:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": round(change, 4),
            "current_accuracy": round(second_half_acc, 4),
            "previous_accuracy": round(first_half_acc, 4),
        }

    # =====================
    # RETRAINING
    # =====================

    def should_retrain(self) -> Tuple[bool, List[str]]:
        """
        Check if model should be retrained.

        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []

        if not self.config.auto_retrain_on_drift:
            return False, reasons

        # Check cooldown
        if self._last_retrain:
            cooldown = timedelta(hours=self.config.retrain_cooldown_hours)
            if datetime.now() - self._last_retrain < cooldown:
                return False, ["Cooldown period active"]

        # Check weekly limit
        if self._retrain_count_this_week >= self.config.max_retrains_per_week:
            return False, ["Weekly retrain limit reached"]

        # Check for high severity drift
        recent_drift = [d for d in self._drift_history[-20:]]

        high_drift = [d for d in recent_drift if d.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]]
        if high_drift:
            reasons.append(f"High severity drift detected in {len(high_drift)} checks")

        # Check performance degradation
        if len(self._performance_history) >= 5:
            recent_perf = self._performance_history[-5:]
            avg_accuracy = np.mean([p.accuracy for p in recent_perf])
            if avg_accuracy < 0.5:
                reasons.append(f"Model accuracy below 50%: {avg_accuracy:.2%}")

        return len(reasons) > 0, reasons

    def record_retrain(self):
        """Record that a retrain was performed."""
        self._last_retrain = datetime.now()
        self._retrain_count_this_week += 1

        # Reset weekly counter on Monday
        if datetime.now().weekday() == 0:
            self._retrain_count_this_week = 1

        self._save_state()

    # =====================
    # STATE PERSISTENCE
    # =====================

    def _save_state(self):
        """Save monitoring state to disk."""
        state_file = self.data_dir / "monitor_state.json"
        state = {
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "retrain_count_this_week": self._retrain_count_this_week,
            "calibration_map": {str(k): v for k, v in self._calibration_map.items()},
            "drift_history_count": len(self._drift_history),
            "performance_history_count": len(self._performance_history),
            "updated_at": datetime.now().isoformat(),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load monitoring state from disk."""
        state_file = self.data_dir / "monitor_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            if state.get("last_retrain"):
                self._last_retrain = datetime.fromisoformat(state["last_retrain"])
            self._retrain_count_this_week = state.get("retrain_count_this_week", 0)

            cal_map = state.get("calibration_map", {})
            self._calibration_map = {float(k): v for k, v in cal_map.items()}

        except (json.JSONDecodeError, KeyError):
            pass

    def get_monitoring_summary(self) -> Dict:
        """Get comprehensive monitoring summary."""
        drift_results = self.check_drift()
        performance = self.update_performance()
        should_retrain, retrain_reasons = self.should_retrain()
        trend = self.get_performance_trend()

        return {
            "drift_detected": len(drift_results) > 0,
            "drift_severity": max([d.severity.value for d in drift_results], default="none"),
            "drift_count": len(drift_results),
            "performance": performance.to_dict(),
            "performance_trend": trend,
            "should_retrain": should_retrain,
            "retrain_reasons": retrain_reasons,
            "samples_monitored": len(self._current_predictions),
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
        }
