"""
Automated Model Retraining Pipeline.

Monitors model performance, detects degradation, and triggers retraining.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Reasons for triggering model retraining."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    MODEL_AGE = "model_age"
    REGIME_SHIFT = "regime_shift"


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class DriftMetrics:
    """Data drift detection metrics."""
    feature_drift_score: float  # 0-1, higher = more drift
    target_drift_score: float
    covariate_shift_detected: bool
    label_shift_detected: bool
    drifted_features: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_drift_score": self.feature_drift_score,
            "target_drift_score": self.target_drift_score,
            "covariate_shift_detected": self.covariate_shift_detected,
            "label_shift_detected": self.label_shift_detected,
            "drifted_features": self.drifted_features,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RetrainingConfig:
    """Configuration for retraining pipeline."""
    # Performance thresholds
    min_accuracy: float = 0.52
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.45
    max_drawdown_threshold: float = 0.15

    # Drift thresholds
    drift_threshold: float = 0.3

    # Scheduling
    max_model_age_days: int = 30
    min_retraining_interval_hours: int = 24
    scheduled_retraining_day: int = 0  # Monday
    scheduled_retraining_hour: int = 2  # 2 AM

    # Training parameters
    min_training_samples: int = 1000
    validation_split: float = 0.2
    walk_forward_folds: int = 5

    # Validation requirements
    min_improvement_pct: float = 0.02  # 2% improvement required
    require_validation_pass: bool = True


@dataclass
class RetrainingResult:
    """Result of a retraining attempt."""
    symbol: str
    model_type: str
    success: bool
    trigger: RetrainingTrigger
    old_metrics: Optional[PerformanceMetrics]
    new_metrics: Optional[PerformanceMetrics]
    improvement_pct: float
    deployed: bool
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "success": self.success,
            "trigger": self.trigger.value,
            "old_metrics": self.old_metrics.to_dict() if self.old_metrics else None,
            "new_metrics": self.new_metrics.to_dict() if self.new_metrics else None,
            "improvement_pct": self.improvement_pct,
            "deployed": self.deployed,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelPerformanceMonitor:
    """Monitors model performance and detects degradation."""

    def __init__(
        self,
        data_dir: str = "data/model_monitoring",
        lookback_window: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_window = lookback_window

        # Performance history per model
        self._history: Dict[str, List[PerformanceMetrics]] = {}
        self._load_history()

    def _get_history_file(self) -> Path:
        return self.data_dir / "performance_history.json"

    def _load_history(self) -> None:
        """Load performance history from disk."""
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    for key, metrics_list in data.items():
                        self._history[key] = [
                            PerformanceMetrics.from_dict(m) for m in metrics_list
                        ]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load performance history: {e}")

    def _save_history(self) -> None:
        """Save performance history to disk."""
        data = {
            key: [m.to_dict() for m in metrics_list[-self.lookback_window:]]
            for key, metrics_list in self._history.items()
        }
        with open(self._get_history_file(), "w") as f:
            json.dump(data, f, indent=2)

    def record_performance(
        self,
        symbol: str,
        model_type: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """Record model performance metrics."""
        key = f"{symbol}_{model_type}"
        if key not in self._history:
            self._history[key] = []

        self._history[key].append(metrics)

        # Keep only recent history
        if len(self._history[key]) > self.lookback_window:
            self._history[key] = self._history[key][-self.lookback_window:]

        self._save_history()

    def get_performance_trend(
        self,
        symbol: str,
        model_type: str,
        window: int = 20,
    ) -> Optional[float]:
        """
        Calculate performance trend.

        Returns:
            Trend value: positive = improving, negative = degrading
        """
        key = f"{symbol}_{model_type}"
        if key not in self._history or len(self._history[key]) < window:
            return None

        recent = self._history[key][-window:]
        accuracies = [m.accuracy for m in recent]

        # Simple linear regression slope
        x = np.arange(len(accuracies))
        slope, _ = np.polyfit(x, accuracies, 1)

        return float(slope)

    def check_degradation(
        self,
        symbol: str,
        model_type: str,
        config: RetrainingConfig,
    ) -> Tuple[bool, str]:
        """
        Check if model performance has degraded.

        Returns:
            Tuple of (is_degraded, reason)
        """
        key = f"{symbol}_{model_type}"
        if key not in self._history or len(self._history[key]) < 5:
            return False, "Insufficient history"

        recent = self._history[key][-5:]
        avg_accuracy = np.mean([m.accuracy for m in recent])
        avg_sharpe = np.mean([m.sharpe_ratio for m in recent])
        avg_win_rate = np.mean([m.win_rate for m in recent])
        latest_drawdown = recent[-1].max_drawdown

        if avg_accuracy < config.min_accuracy:
            return True, f"Accuracy ({avg_accuracy:.3f}) below threshold ({config.min_accuracy})"

        if avg_sharpe < config.min_sharpe_ratio:
            return True, f"Sharpe ratio ({avg_sharpe:.2f}) below threshold ({config.min_sharpe_ratio})"

        if avg_win_rate < config.min_win_rate:
            return True, f"Win rate ({avg_win_rate:.2%}) below threshold ({config.min_win_rate:.2%})"

        if latest_drawdown > config.max_drawdown_threshold:
            return True, f"Drawdown ({latest_drawdown:.2%}) exceeds threshold ({config.max_drawdown_threshold:.2%})"

        # Check trend
        trend = self.get_performance_trend(symbol, model_type)
        if trend is not None and trend < -0.01:  # Declining more than 1% per period
            return True, f"Performance declining (trend: {trend:.4f})"

        return False, "Performance acceptable"

    def get_latest_metrics(
        self,
        symbol: str,
        model_type: str,
    ) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        key = f"{symbol}_{model_type}"
        if key in self._history and self._history[key]:
            return self._history[key][-1]
        return None


class DataDriftDetector:
    """Detects data drift to trigger retraining."""

    def __init__(self, reference_window: int = 500, test_window: int = 100):
        self.reference_window = reference_window
        self.test_window = test_window

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str],
    ) -> DriftMetrics:
        """
        Detect drift between reference and current data.

        Uses Population Stability Index (PSI) for drift detection.
        """
        drifted_features = []
        feature_psi_scores = []

        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) < 50 or len(curr_values) < 50:
                continue

            psi = self._calculate_psi(ref_values, curr_values)
            feature_psi_scores.append(psi)

            if psi > 0.2:  # PSI > 0.2 indicates significant drift
                drifted_features.append(feature)

        avg_feature_drift = np.mean(feature_psi_scores) if feature_psi_scores else 0.0

        # Target drift (for labeled data)
        target_drift = 0.0
        if "target" in reference_data.columns and "target" in current_data.columns:
            ref_target = reference_data["target"].value_counts(normalize=True)
            curr_target = current_data["target"].value_counts(normalize=True)

            # Compare class distributions
            common_classes = set(ref_target.index) & set(curr_target.index)
            if common_classes:
                target_drift = sum(
                    abs(ref_target.get(c, 0) - curr_target.get(c, 0))
                    for c in common_classes
                ) / len(common_classes)

        return DriftMetrics(
            feature_drift_score=avg_feature_drift,
            target_drift_score=target_drift,
            covariate_shift_detected=avg_feature_drift > 0.15,
            label_shift_detected=target_drift > 0.1,
            drifted_features=drifted_features,
        )

    def _calculate_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from expected distribution
        try:
            _, bin_edges = np.histogram(expected, bins=bins)
        except ValueError:
            return 0.0

        # Calculate proportions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero
        expected_pct = np.maximum(expected_pct, 0.0001)
        actual_pct = np.maximum(actual_pct, 0.0001)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)


class RetrainingPipeline:
    """
    Automated model retraining pipeline.

    Features:
    - Performance monitoring and degradation detection
    - Data drift detection
    - Scheduled retraining
    - Walk-forward validation
    - Safe model deployment

    Usage:
        pipeline = RetrainingPipeline(config)

        # Check if retraining is needed
        if pipeline.should_retrain("BTC/USDT", "lstm"):
            result = pipeline.retrain("BTC/USDT", "lstm", training_data)
    """

    def __init__(
        self,
        config: Optional[RetrainingConfig] = None,
        data_dir: str = "data/retraining",
    ):
        self.config = config or RetrainingConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = ModelPerformanceMonitor(str(self.data_dir / "monitoring"))
        self.drift_detector = DataDriftDetector()

        # Track last retraining times
        self._last_retrain: Dict[str, datetime] = {}
        self._retraining_history: List[RetrainingResult] = []

        self._load_state()

    def _get_state_file(self) -> Path:
        return self.data_dir / "pipeline_state.json"

    def _load_state(self) -> None:
        """Load pipeline state from disk."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self._last_retrain = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get("last_retrain", {}).items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load pipeline state: {e}")

    def _save_state(self) -> None:
        """Save pipeline state to disk."""
        data = {
            "last_retrain": {
                k: v.isoformat() for k, v in self._last_retrain.items()
            },
        }
        with open(self._get_state_file(), "w") as f:
            json.dump(data, f, indent=2)

    def should_retrain(
        self,
        symbol: str,
        model_type: str,
        model_created: Optional[datetime] = None,
    ) -> Tuple[bool, RetrainingTrigger, str]:
        """
        Check if a model should be retrained.

        Returns:
            Tuple of (should_retrain, trigger_reason, explanation)
        """
        key = f"{symbol}_{model_type}"
        now = datetime.now()

        # Check minimum interval
        last_retrain = self._last_retrain.get(key)
        if last_retrain:
            hours_since = (now - last_retrain).total_seconds() / 3600
            if hours_since < self.config.min_retraining_interval_hours:
                return False, RetrainingTrigger.SCHEDULED, f"Too recent ({hours_since:.1f}h ago)"

        # Check model age
        if model_created:
            age_days = (now - model_created).days
            if age_days >= self.config.max_model_age_days:
                return True, RetrainingTrigger.MODEL_AGE, f"Model is {age_days} days old"

        # Check scheduled retraining
        if (now.weekday() == self.config.scheduled_retraining_day and
            now.hour == self.config.scheduled_retraining_hour):
            if last_retrain is None or (now - last_retrain).days >= 7:
                return True, RetrainingTrigger.SCHEDULED, "Weekly scheduled retraining"

        # Check performance degradation
        is_degraded, reason = self.monitor.check_degradation(
            symbol, model_type, self.config
        )
        if is_degraded:
            return True, RetrainingTrigger.PERFORMANCE_DEGRADATION, reason

        return False, RetrainingTrigger.SCHEDULED, "No retraining needed"

    def check_drift(
        self,
        symbol: str,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str],
    ) -> Tuple[bool, DriftMetrics]:
        """
        Check for data drift.

        Returns:
            Tuple of (drift_detected, drift_metrics)
        """
        metrics = self.drift_detector.detect_drift(
            reference_data, current_data, features
        )

        drift_detected = (
            metrics.feature_drift_score > self.config.drift_threshold or
            metrics.covariate_shift_detected or
            metrics.label_shift_detected
        )

        return drift_detected, metrics

    def retrain(
        self,
        symbol: str,
        model_type: str,
        training_data: pd.DataFrame,
        trigger: RetrainingTrigger,
        train_func: Callable[[pd.DataFrame], Any],
        validate_func: Callable[[Any, pd.DataFrame], PerformanceMetrics],
    ) -> RetrainingResult:
        """
        Retrain a model.

        Args:
            symbol: Trading symbol
            model_type: Type of model
            training_data: Data for training
            trigger: Reason for retraining
            train_func: Function that trains the model and returns it
            validate_func: Function that validates model and returns metrics

        Returns:
            RetrainingResult with outcome details
        """
        key = f"{symbol}_{model_type}"
        old_metrics = self.monitor.get_latest_metrics(symbol, model_type)

        logger.info(f"Starting retraining for {symbol} {model_type}")
        logger.info(f"Trigger: {trigger.value}")

        try:
            # Check minimum samples
            if len(training_data) < self.config.min_training_samples:
                return RetrainingResult(
                    symbol=symbol,
                    model_type=model_type,
                    success=False,
                    trigger=trigger,
                    old_metrics=old_metrics,
                    new_metrics=None,
                    improvement_pct=0.0,
                    deployed=False,
                    error_message=f"Insufficient training data ({len(training_data)} < {self.config.min_training_samples})",
                )

            # Split data
            split_idx = int(len(training_data) * (1 - self.config.validation_split))
            train_df = training_data.iloc[:split_idx]
            val_df = training_data.iloc[split_idx:]

            # Train model
            logger.info(f"Training on {len(train_df)} samples")
            new_model = train_func(train_df)

            # Validate model
            logger.info(f"Validating on {len(val_df)} samples")
            new_metrics = validate_func(new_model, val_df)

            # Check improvement
            improvement_pct = 0.0
            if old_metrics:
                improvement_pct = (new_metrics.accuracy - old_metrics.accuracy) / old_metrics.accuracy

            # Decide on deployment
            should_deploy = True
            if self.config.require_validation_pass:
                if new_metrics.accuracy < self.config.min_accuracy:
                    should_deploy = False
                    logger.warning(f"New model accuracy ({new_metrics.accuracy:.3f}) below threshold")

                if old_metrics and improvement_pct < self.config.min_improvement_pct:
                    should_deploy = False
                    logger.warning(f"Improvement ({improvement_pct:.2%}) below threshold ({self.config.min_improvement_pct:.2%})")

            # Record performance
            if should_deploy:
                self.monitor.record_performance(symbol, model_type, new_metrics)
                self._last_retrain[key] = datetime.now()
                self._save_state()

            result = RetrainingResult(
                symbol=symbol,
                model_type=model_type,
                success=True,
                trigger=trigger,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                improvement_pct=improvement_pct,
                deployed=should_deploy,
            )

            logger.info(f"Retraining complete. Deployed: {should_deploy}, Improvement: {improvement_pct:.2%}")

            self._retraining_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return RetrainingResult(
                symbol=symbol,
                model_type=model_type,
                success=False,
                trigger=trigger,
                old_metrics=old_metrics,
                new_metrics=None,
                improvement_pct=0.0,
                deployed=False,
                error_message=str(e),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "config": {
                "min_accuracy": self.config.min_accuracy,
                "max_model_age_days": self.config.max_model_age_days,
                "drift_threshold": self.config.drift_threshold,
            },
            "last_retrains": {
                k: v.isoformat() for k, v in self._last_retrain.items()
            },
            "recent_retrainings": [
                r.to_dict() for r in self._retraining_history[-10:]
            ],
        }

    def get_retraining_schedule(
        self,
        symbols: List[str],
        model_types: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming retraining schedule.

        Returns list of models with their retraining status.
        """
        schedule = []

        for symbol in symbols:
            for model_type in model_types:
                should, trigger, reason = self.should_retrain(symbol, model_type)

                key = f"{symbol}_{model_type}"
                last = self._last_retrain.get(key)

                schedule.append({
                    "symbol": symbol,
                    "model_type": model_type,
                    "should_retrain": should,
                    "trigger": trigger.value,
                    "reason": reason,
                    "last_retrained": last.isoformat() if last else None,
                })

        return schedule
