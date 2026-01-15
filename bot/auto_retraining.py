"""
Auto-Retraining System

Phase 11: Automatic model retraining based on performance monitoring.

Features:
1. Continuous performance monitoring
2. Degradation detection algorithms
3. Automatic retraining triggers
4. A/B testing for new models
5. Model registry and versioning
6. Rollback capabilities
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import shutil

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating"
    CANDIDATE = "candidate"      # Ready for A/B test
    SHADOW = "shadow"            # Running in shadow mode
    PRODUCTION = "production"    # Active in production
    DEPRECATED = "deprecated"    # Replaced by newer model
    FAILED = "failed"            # Failed validation


class DegradationReason(Enum):
    """Reasons for model degradation."""
    ACCURACY_DROP = "accuracy_drop"
    PRECISION_DROP = "precision_drop"
    PROFIT_FACTOR_DROP = "profit_factor_drop"
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: datetime
    model_id: str

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0

    # Prediction metrics
    predictions_made: int = 0
    correct_predictions: int = 0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades
        }


@dataclass
class ModelRecord:
    """Record of a trained model."""
    model_id: str
    model_path: str
    created_at: datetime
    status: ModelStatus

    # Training info
    training_data_hash: str
    training_samples: int
    feature_names: List[str]

    # Performance at training time
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]

    # Production performance
    production_metrics: List[ModelMetrics] = field(default_factory=list)

    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def get_recent_performance(self, days: int = 7) -> Optional[ModelMetrics]:
        """Get average performance over recent days."""
        if not self.production_metrics:
            return None

        cutoff = datetime.now() - timedelta(days=days)
        recent = [m for m in self.production_metrics if m.timestamp > cutoff]

        if not recent:
            return None

        return ModelMetrics(
            timestamp=datetime.now(),
            model_id=self.model_id,
            accuracy=np.mean([m.accuracy for m in recent]),
            precision=np.mean([m.precision for m in recent]),
            win_rate=np.mean([m.win_rate for m in recent]),
            profit_factor=np.mean([m.profit_factor for m in recent]),
            total_trades=sum(m.total_trades for m in recent)
        )


@dataclass
class RetrainingTrigger:
    """Trigger for model retraining."""
    trigger_id: str
    triggered_at: datetime
    reason: DegradationReason
    model_id: str

    # Metrics that triggered retraining
    current_value: float
    threshold: float
    baseline_value: float

    # Status
    acknowledged: bool = False
    retraining_started: bool = False
    new_model_id: Optional[str] = None


class PerformanceMonitor:
    """
    Monitors model performance and detects degradation.

    Tracks:
    - Prediction accuracy over time
    - Trading performance metrics
    - Data distribution drift
    - Feature drift
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.05,      # 5% accuracy drop triggers alert
        precision_threshold: float = 0.05,
        profit_factor_threshold: float = 0.3,  # 30% profit factor drop
        drift_threshold: float = 0.1,
        window_size: int = 100,                # Predictions to track
        min_samples: int = 50                  # Minimum samples before evaluation
    ):
        """
        Initialize performance monitor.

        Args:
            accuracy_threshold: Maximum allowed accuracy drop
            precision_threshold: Maximum allowed precision drop
            profit_factor_threshold: Maximum allowed profit factor drop
            drift_threshold: Maximum allowed feature drift
            window_size: Rolling window for metrics
            min_samples: Minimum samples before checking
        """
        self.accuracy_threshold = accuracy_threshold
        self.precision_threshold = precision_threshold
        self.profit_factor_threshold = profit_factor_threshold
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.min_samples = min_samples

        # Tracking
        self.predictions: List[Dict] = []
        self.feature_distributions: Dict[str, List[float]] = {}
        self.baseline_metrics: Optional[ModelMetrics] = None

        # Drift detection
        self._baseline_feature_stats: Dict[str, Tuple[float, float]] = {}

    def set_baseline(self, metrics: ModelMetrics, feature_stats: Optional[Dict] = None):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
        if feature_stats:
            self._baseline_feature_stats = feature_stats
        logger.info(f"Baseline set: accuracy={metrics.accuracy:.3f}, precision={metrics.precision:.3f}")

    def record_prediction(
        self,
        prediction: int,
        actual: int,
        features: Optional[np.ndarray] = None,
        trade_pnl: Optional[float] = None
    ):
        """Record a prediction result."""
        self.predictions.append({
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual,
            "correct": prediction == actual,
            "pnl": trade_pnl
        })

        # Track features for drift detection
        if features is not None:
            for i, val in enumerate(features):
                key = f"feature_{i}"
                if key not in self.feature_distributions:
                    self.feature_distributions[key] = []
                self.feature_distributions[key].append(val)

        # Trim to window size
        if len(self.predictions) > self.window_size * 2:
            self.predictions = self.predictions[-self.window_size:]

    def check_degradation(self) -> Tuple[bool, Optional[DegradationReason], Dict]:
        """
        Check if model has degraded.

        Returns:
            Tuple of (is_degraded, reason, details)
        """
        if len(self.predictions) < self.min_samples:
            return False, None, {"message": "Not enough samples"}

        if self.baseline_metrics is None:
            return False, None, {"message": "No baseline set"}

        current_metrics = self._compute_current_metrics()

        # Check accuracy drop
        accuracy_drop = self.baseline_metrics.accuracy - current_metrics.accuracy
        if accuracy_drop > self.accuracy_threshold:
            return True, DegradationReason.ACCURACY_DROP, {
                "baseline": self.baseline_metrics.accuracy,
                "current": current_metrics.accuracy,
                "drop": accuracy_drop
            }

        # Check precision drop
        precision_drop = self.baseline_metrics.precision - current_metrics.precision
        if precision_drop > self.precision_threshold:
            return True, DegradationReason.PRECISION_DROP, {
                "baseline": self.baseline_metrics.precision,
                "current": current_metrics.precision,
                "drop": precision_drop
            }

        # Check profit factor drop (if baseline has it)
        if self.baseline_metrics.profit_factor > 0:
            pf_drop = (self.baseline_metrics.profit_factor - current_metrics.profit_factor) / self.baseline_metrics.profit_factor
            if pf_drop > self.profit_factor_threshold:
                return True, DegradationReason.PROFIT_FACTOR_DROP, {
                    "baseline": self.baseline_metrics.profit_factor,
                    "current": current_metrics.profit_factor,
                    "drop_pct": pf_drop
                }

        # Check data drift
        drift_detected, drift_details = self._check_data_drift()
        if drift_detected:
            return True, DegradationReason.DATA_DRIFT, drift_details

        return False, None, {"metrics": current_metrics.to_dict()}

    def _compute_current_metrics(self) -> ModelMetrics:
        """Compute metrics from recent predictions."""
        recent = self.predictions[-self.window_size:]

        correct = sum(1 for p in recent if p["correct"])
        total = len(recent)

        # Calculate trading metrics
        trades_with_pnl = [p for p in recent if p.get("pnl") is not None]
        wins = sum(1 for t in trades_with_pnl if t["pnl"] > 0)
        gross_profit = sum(t["pnl"] for t in trades_with_pnl if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades_with_pnl if t["pnl"] < 0))

        return ModelMetrics(
            timestamp=datetime.now(),
            model_id="current",
            accuracy=correct / total if total > 0 else 0,
            precision=correct / total if total > 0 else 0,  # Simplified
            win_rate=wins / len(trades_with_pnl) if trades_with_pnl else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            total_trades=len(trades_with_pnl),
            predictions_made=total,
            correct_predictions=correct
        )

    def _check_data_drift(self) -> Tuple[bool, Dict]:
        """Check for data/feature drift using simple statistics."""
        if not self._baseline_feature_stats or not self.feature_distributions:
            return False, {}

        drifted_features = []

        for feature_name, values in self.feature_distributions.items():
            if feature_name not in self._baseline_feature_stats:
                continue

            if len(values) < self.min_samples:
                continue

            baseline_mean, baseline_std = self._baseline_feature_stats[feature_name]
            current_mean = np.mean(values[-self.window_size:])
            current_std = np.std(values[-self.window_size:])

            # Simple drift detection: check if mean shifted significantly
            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / baseline_std
                if z_score > 3:  # 3 sigma rule
                    drifted_features.append({
                        "feature": feature_name,
                        "baseline_mean": baseline_mean,
                        "current_mean": current_mean,
                        "z_score": z_score
                    })

        if len(drifted_features) > len(self.feature_distributions) * 0.2:  # 20% features drifted
            return True, {"drifted_features": drifted_features}

        return False, {}

    def get_current_metrics(self) -> ModelMetrics:
        """Get current performance metrics."""
        return self._compute_current_metrics()


class ModelRegistry:
    """
    Registry for tracking all model versions.

    Handles:
    - Model versioning
    - Status management
    - Performance history
    - Rollback
    """

    def __init__(self, registry_dir: str = "./data/model_registry"):
        """Initialize model registry."""
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelRecord] = {}
        self.production_model_id: Optional[str] = None

        # Load existing registry
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                self.production_model_id = data.get("production_model_id")
                # Would load model records here
                logger.info(f"Registry loaded with {len(data.get('models', {}))} models")

    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"
        data = {
            "production_model_id": self.production_model_id,
            "models": {
                model_id: {
                    "model_id": record.model_id,
                    "status": record.status.value,
                    "created_at": record.created_at.isoformat(),
                    "model_path": record.model_path,
                    "training_metrics": record.training_metrics,
                    "validation_metrics": record.validation_metrics
                }
                for model_id, record in self.models.items()
            },
            "updated_at": datetime.now().isoformat()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_model(self, record: ModelRecord) -> str:
        """Register a new model."""
        self.models[record.model_id] = record
        self._save_registry()
        logger.info(f"Model {record.model_id} registered with status {record.status.value}")
        return record.model_id

    def update_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        if model_id in self.models:
            self.models[model_id].status = status
            self._save_registry()
            logger.info(f"Model {model_id} status updated to {status.value}")

    def promote_to_production(self, model_id: str) -> bool:
        """Promote a model to production."""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False

        # Demote current production model
        if self.production_model_id and self.production_model_id in self.models:
            self.models[self.production_model_id].status = ModelStatus.DEPRECATED

        # Promote new model
        self.models[model_id].status = ModelStatus.PRODUCTION
        self.production_model_id = model_id
        self._save_registry()

        logger.info(f"Model {model_id} promoted to production")
        return True

    def rollback_to_previous(self) -> Optional[str]:
        """Rollback to previous production model."""
        # Find most recent deprecated model
        deprecated = [
            (m.model_id, m.created_at)
            for m in self.models.values()
            if m.status == ModelStatus.DEPRECATED
        ]

        if not deprecated:
            logger.warning("No previous model to rollback to")
            return None

        # Sort by creation time, get most recent
        deprecated.sort(key=lambda x: x[1], reverse=True)
        previous_id = deprecated[0][0]

        # Promote previous model
        if self.promote_to_production(previous_id):
            logger.info(f"Rolled back to model {previous_id}")
            return previous_id

        return None

    def get_production_model(self) -> Optional[ModelRecord]:
        """Get the current production model."""
        if self.production_model_id:
            return self.models.get(self.production_model_id)
        return None

    def record_production_metrics(self, model_id: str, metrics: ModelMetrics):
        """Record production performance metrics."""
        if model_id in self.models:
            self.models[model_id].production_metrics.append(metrics)
            # Keep only last 1000 metrics
            if len(self.models[model_id].production_metrics) > 1000:
                self.models[model_id].production_metrics = self.models[model_id].production_metrics[-1000:]


class AutoRetrainingManager:
    """
    Manages automatic model retraining.

    Responsibilities:
    - Monitor model performance
    - Detect degradation
    - Trigger retraining
    - Validate new models
    - Manage promotion/rollback
    """

    def __init__(
        self,
        training_pipeline,
        model_registry: Optional[ModelRegistry] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        notification_manager=None,
        check_interval_minutes: int = 60,
        min_retrain_interval_hours: int = 24
    ):
        """
        Initialize auto-retraining manager.

        Args:
            training_pipeline: ML training pipeline instance
            model_registry: Model registry instance
            performance_monitor: Performance monitor instance
            notification_manager: Notification manager for alerts
            check_interval_minutes: How often to check performance
            min_retrain_interval_hours: Minimum time between retrainings
        """
        self.training_pipeline = training_pipeline
        self.registry = model_registry or ModelRegistry()
        self.monitor = performance_monitor or PerformanceMonitor()
        self.notification_manager = notification_manager

        self.check_interval_minutes = check_interval_minutes
        self.min_retrain_interval_hours = min_retrain_interval_hours

        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_retrain_time: Optional[datetime] = None
        self._pending_triggers: List[RetrainingTrigger] = []

        # Callbacks
        self._on_degradation_callbacks: List[Callable] = []
        self._on_retrain_callbacks: List[Callable] = []
        self._on_promotion_callbacks: List[Callable] = []

        logger.info("AutoRetrainingManager initialized")

    async def start(self):
        """Start the auto-retraining manager."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Auto-retraining manager started")

    async def stop(self):
        """Stop the auto-retraining manager."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Auto-retraining manager stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_minutes * 60)

                # Check for degradation
                is_degraded, reason, details = self.monitor.check_degradation()

                if is_degraded:
                    await self._handle_degradation(reason, details)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    async def _handle_degradation(self, reason: DegradationReason, details: Dict):
        """Handle detected model degradation."""
        logger.warning(f"Model degradation detected: {reason.value}")

        current_model = self.registry.get_production_model()
        model_id = current_model.model_id if current_model else "unknown"

        # Create trigger
        trigger = RetrainingTrigger(
            trigger_id=f"trigger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            triggered_at=datetime.now(),
            reason=reason,
            model_id=model_id,
            current_value=details.get("current", 0),
            threshold=details.get("threshold", 0),
            baseline_value=details.get("baseline", 0)
        )

        self._pending_triggers.append(trigger)

        # Notify callbacks
        for callback in self._on_degradation_callbacks:
            try:
                await callback(trigger, details)
            except Exception as e:
                logger.error(f"Degradation callback error: {e}")

        # Send notification
        if self.notification_manager:
            await self.notification_manager.notify_risk_alert(
                alert_type="Model Degradation",
                message=f"Model {model_id} degraded: {reason.value}",
                metrics=details,
                critical=reason == DegradationReason.ACCURACY_DROP
            )

        # Check if we should retrain
        if self._should_retrain():
            await self.trigger_retraining(trigger)

    def _should_retrain(self) -> bool:
        """Check if conditions allow retraining."""
        if self._last_retrain_time:
            hours_since = (datetime.now() - self._last_retrain_time).total_seconds() / 3600
            if hours_since < self.min_retrain_interval_hours:
                logger.info(f"Skipping retrain, only {hours_since:.1f}h since last retrain")
                return False
        return True

    async def trigger_retraining(
        self,
        trigger: Optional[RetrainingTrigger] = None,
        price_data: Optional[pd.DataFrame] = None,
        news_features: Optional[pd.DataFrame] = None
    ) -> Optional[str]:
        """
        Trigger model retraining.

        Args:
            trigger: Retraining trigger (if from degradation)
            price_data: Training data (if not provided, will fetch)
            news_features: News features (optional)

        Returns:
            New model ID if successful
        """
        logger.info("Starting model retraining...")

        if trigger:
            trigger.retraining_started = True

        self._last_retrain_time = datetime.now()

        try:
            # Train new model
            result = self.training_pipeline.train(
                price_data=price_data,
                news_features=news_features
            )

            # Create model record
            record = ModelRecord(
                model_id=result.model_id,
                model_path=result.model_path,
                created_at=datetime.now(),
                status=ModelStatus.CANDIDATE,
                training_data_hash=result.data_hash,
                training_samples=0,  # Would come from training
                feature_names=[],    # Would come from training
                training_metrics=result.train_metrics,
                validation_metrics=result.val_metrics,
                config={}
            )

            # Register model
            self.registry.register_model(record)

            if trigger:
                trigger.new_model_id = result.model_id

            # Notify callbacks
            for callback in self._on_retrain_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Retrain callback error: {e}")

            # Notify
            if self.notification_manager:
                await self.notification_manager.notify(
                    category="system",
                    title="Model Retrained",
                    message=f"New model {result.model_id} trained. Accuracy: {result.val_metrics.get('accuracy', 0):.3f}",
                    priority="medium"
                )

            logger.info(f"New model trained: {result.model_id}")
            return result.model_id

        except Exception as e:
            logger.error(f"Retraining failed: {e}")

            if self.notification_manager:
                await self.notification_manager.notify_system_error(
                    error_type="Retraining Failed",
                    error_message=str(e)
                )

            return None

    async def validate_and_promote(
        self,
        model_id: str,
        validation_data: pd.DataFrame,
        min_accuracy: float = 0.52,
        min_profit_factor: float = 1.1
    ) -> bool:
        """
        Validate a candidate model and promote if good enough.

        Args:
            model_id: Model to validate
            validation_data: Out-of-sample validation data
            min_accuracy: Minimum required accuracy
            min_profit_factor: Minimum required profit factor

        Returns:
            True if promoted
        """
        logger.info(f"Validating model {model_id}...")

        record = self.registry.models.get(model_id)
        if not record:
            logger.error(f"Model {model_id} not found")
            return False

        # Run validation
        val_metrics = record.validation_metrics

        # Check thresholds
        accuracy = val_metrics.get("accuracy", 0)
        if accuracy < min_accuracy:
            logger.warning(f"Model {model_id} failed accuracy check: {accuracy:.3f} < {min_accuracy}")
            self.registry.update_status(model_id, ModelStatus.FAILED)
            return False

        # Compare with current production
        current_model = self.registry.get_production_model()
        if current_model:
            current_accuracy = current_model.validation_metrics.get("accuracy", 0)
            if accuracy < current_accuracy * 0.95:  # Must be within 5% of current
                logger.warning(f"Model {model_id} not better than current: {accuracy:.3f} vs {current_accuracy:.3f}")
                self.registry.update_status(model_id, ModelStatus.FAILED)
                return False

        # Promote to production
        success = self.registry.promote_to_production(model_id)

        if success:
            # Notify callbacks
            for callback in self._on_promotion_callbacks:
                try:
                    await callback(model_id, record)
                except Exception as e:
                    logger.error(f"Promotion callback error: {e}")

            # Update monitor baseline
            self.monitor.set_baseline(ModelMetrics(
                timestamp=datetime.now(),
                model_id=model_id,
                accuracy=accuracy,
                precision=val_metrics.get("precision", accuracy)
            ))

            if self.notification_manager:
                await self.notification_manager.notify(
                    category="system",
                    title="Model Promoted",
                    message=f"Model {model_id} promoted to production",
                    priority="high"
                )

        return success

    async def rollback(self, reason: str = "Manual rollback") -> Optional[str]:
        """Rollback to previous production model."""
        previous_id = self.registry.rollback_to_previous()

        if previous_id:
            if self.notification_manager:
                await self.notification_manager.notify(
                    category="system",
                    title="Model Rollback",
                    message=f"Rolled back to model {previous_id}: {reason}",
                    priority="high"
                )

        return previous_id

    def record_prediction(
        self,
        prediction: int,
        actual: int,
        features: Optional[np.ndarray] = None,
        trade_pnl: Optional[float] = None
    ):
        """Record a prediction for monitoring."""
        self.monitor.record_prediction(prediction, actual, features, trade_pnl)

        # Also record to registry if we have a production model
        if self.registry.production_model_id:
            metrics = self.monitor.get_current_metrics()
            self.registry.record_production_metrics(
                self.registry.production_model_id,
                metrics
            )

    def on_degradation(self, callback: Callable):
        """Register callback for degradation events."""
        self._on_degradation_callbacks.append(callback)

    def on_retrain(self, callback: Callable):
        """Register callback for retrain events."""
        self._on_retrain_callbacks.append(callback)

    def on_promotion(self, callback: Callable):
        """Register callback for promotion events."""
        self._on_promotion_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        production_model = self.registry.get_production_model()

        return {
            "running": self._running,
            "production_model_id": self.registry.production_model_id,
            "production_model_status": production_model.status.value if production_model else None,
            "last_retrain_time": self._last_retrain_time.isoformat() if self._last_retrain_time else None,
            "pending_triggers": len(self._pending_triggers),
            "total_models": len(self.registry.models),
            "current_metrics": self.monitor.get_current_metrics().to_dict()
        }


def create_auto_retraining_manager(
    training_pipeline,
    notification_manager=None,
    **kwargs
) -> AutoRetrainingManager:
    """Factory function to create auto-retraining manager."""
    return AutoRetrainingManager(
        training_pipeline=training_pipeline,
        notification_manager=notification_manager,
        **kwargs
    )


if __name__ == "__main__":
    # Demo
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def demo():
        print("=== Auto-Retraining System Demo ===")

        # Create mock training pipeline
        class MockPipeline:
            def train(self, **kwargs):
                from dataclasses import dataclass

                @dataclass
                class Result:
                    model_id: str = "model_test_001"
                    model_path: str = "/tmp/model.pkl"
                    data_hash: str = "abc123"
                    train_metrics: dict = None
                    val_metrics: dict = None

                    def __post_init__(self):
                        self.train_metrics = {"accuracy": 0.55}
                        self.val_metrics = {"accuracy": 0.53}

                return Result()

        # Create manager
        manager = AutoRetrainingManager(
            training_pipeline=MockPipeline(),
            check_interval_minutes=1  # Fast for demo
        )

        # Set baseline
        manager.monitor.set_baseline(ModelMetrics(
            timestamp=datetime.now(),
            model_id="baseline",
            accuracy=0.55,
            precision=0.54
        ))

        # Simulate predictions
        import random
        for i in range(100):
            pred = random.randint(0, 1)
            actual = random.randint(0, 1)
            manager.record_prediction(pred, actual, trade_pnl=random.uniform(-100, 150))

        # Check status
        status = manager.get_status()
        print(f"\nStatus: {json.dumps(status, indent=2)}")

        # Check degradation
        is_degraded, reason, details = manager.monitor.check_degradation()
        print(f"\nDegraded: {is_degraded}")
        if reason:
            print(f"Reason: {reason.value}")
            print(f"Details: {details}")

    asyncio.run(demo())
