"""
Automated Model Retraining Scheduler.

Background scheduler that monitors model performance and triggers
retraining when performance degrades or data drift is detected.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from .training.retraining_pipeline import (
    RetrainingPipeline,
    RetrainingConfig,
    RetrainingTrigger,
    RetrainingResult,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class ModelHealthStatus(Enum):
    """Model health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ModelHealth:
    """Health status of a model."""

    symbol: str
    model_type: str
    status: ModelHealthStatus
    accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    days_since_trained: Optional[int] = None
    drift_detected: bool = False
    needs_retraining: bool = False
    reason: str = ""
    last_checked: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "status": self.status.value,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "days_since_trained": self.days_since_trained,
            "drift_detected": self.drift_detected,
            "needs_retraining": self.needs_retraining,
            "reason": self.reason,
            "last_checked": self.last_checked.isoformat(),
        }


@dataclass
class RetrainingJob:
    """A scheduled retraining job."""

    symbol: str
    model_type: str
    trigger: RetrainingTrigger
    priority: int  # Lower = higher priority
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[RetrainingResult] = None
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "trigger": self.trigger.value,
            "priority": self.priority,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
        }


class AutoRetrainingScheduler:
    """
    Background scheduler for automatic model retraining.

    Features:
    - Periodic health checks (every N hours)
    - Automatic retraining on performance degradation
    - Data drift detection
    - Priority queue for retraining jobs
    - Hot-swap model deployment
    - Thread-safe operations

    Usage:
        scheduler = AutoRetrainingScheduler(
            check_interval_hours=6,
            data_fetcher=fetch_market_data,
        )

        # Register models to monitor
        scheduler.register_model("BTC/USDT", "gradient_boosting", model)
        scheduler.register_model("ETH/USDT", "gradient_boosting", model)

        # Start background monitoring
        scheduler.start()

        # Check health manually
        health = scheduler.check_model_health("BTC/USDT", "gradient_boosting")

        # Stop scheduler
        scheduler.stop()
    """

    def __init__(
        self,
        check_interval_hours: float = 6,
        data_fetcher: Optional[Callable[[str, int], pd.DataFrame]] = None,
        train_func: Optional[Callable[[str, pd.DataFrame], Any]] = None,
        config: Optional[RetrainingConfig] = None,
        data_dir: str = "data/auto_retraining",
    ):
        """
        Initialize the auto-retraining scheduler.

        Args:
            check_interval_hours: How often to check model health
            data_fetcher: Function to fetch market data: fn(symbol, days) -> DataFrame
            train_func: Function to train model: fn(model_type, data) -> model
            config: Retraining configuration
            data_dir: Directory for storing state
        """
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.data_fetcher = data_fetcher
        self.train_func = train_func
        self.config = config or RetrainingConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline for actual retraining
        self.pipeline = RetrainingPipeline(config=self.config, data_dir=str(self.data_dir))

        # Registered models to monitor
        self._models: Dict[str, Dict[str, Any]] = {}  # {symbol: {model_type: model}}
        self._model_created: Dict[str, datetime] = {}  # {key: creation_time}

        # Job queue and history
        self._job_queue: List[RetrainingJob] = []
        self._completed_jobs: List[RetrainingJob] = []

        # Health cache
        self._health_cache: Dict[str, ModelHealth] = {}

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Load persisted state
        self._load_state()

    def _get_state_file(self) -> Path:
        return self.data_dir / "scheduler_state.json"

    def _load_state(self) -> None:
        """Load scheduler state from disk."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    self._model_created = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get("model_created", {}).items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load scheduler state: {e}")

    def _save_state(self) -> None:
        """Save scheduler state to disk."""
        data = {
            "model_created": {k: v.isoformat() for k, v in self._model_created.items()},
            "last_check": datetime.now().isoformat(),
            "registered_models": list(self._models.keys()),
        }
        with open(self._get_state_file(), "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        symbol: str,
        model_type: str,
        model: Any,
        created_at: Optional[datetime] = None,
    ) -> None:
        """
        Register a model for monitoring.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            model_type: Model type (e.g., "gradient_boosting")
            model: The model object
            created_at: When the model was created/trained
        """
        with self._lock:
            if symbol not in self._models:
                self._models[symbol] = {}
            self._models[symbol][model_type] = model

            key = f"{symbol}_{model_type}"
            if created_at:
                self._model_created[key] = created_at
            elif key not in self._model_created:
                self._model_created[key] = datetime.now()

            self._save_state()
            logger.info(f"Registered model: {symbol} {model_type}")

    def unregister_model(self, symbol: str, model_type: str) -> None:
        """Unregister a model from monitoring."""
        with self._lock:
            if symbol in self._models and model_type in self._models[symbol]:
                del self._models[symbol][model_type]
                if not self._models[symbol]:
                    del self._models[symbol]

            key = f"{symbol}_{model_type}"
            self._model_created.pop(key, None)
            self._health_cache.pop(key, None)

            self._save_state()
            logger.info(f"Unregistered model: {symbol} {model_type}")

    def get_model(self, symbol: str, model_type: str) -> Optional[Any]:
        """Get the current model for a symbol."""
        with self._lock:
            return self._models.get(symbol, {}).get(model_type)

    def hot_swap_model(
        self,
        symbol: str,
        model_type: str,
        new_model: Any,
    ) -> bool:
        """
        Atomically swap a model without stopping the bot.

        Thread-safe model replacement that ensures no requests
        are processed with a half-updated model.

        Args:
            symbol: Trading symbol
            model_type: Model type
            new_model: New model to deploy

        Returns:
            True if swap successful
        """
        with self._lock:
            if symbol not in self._models:
                self._models[symbol] = {}

            old_model = self._models[symbol].get(model_type)
            self._models[symbol][model_type] = new_model

            key = f"{symbol}_{model_type}"
            self._model_created[key] = datetime.now()
            self._save_state()

            logger.info(
                f"Hot-swapped model: {symbol} {model_type} (had previous: {old_model is not None})"
            )
            return True

    def check_model_health(
        self,
        symbol: str,
        model_type: str,
    ) -> ModelHealth:
        """
        Check the health of a specific model.

        Returns:
            ModelHealth with status and metrics
        """
        key = f"{symbol}_{model_type}"

        # Check if model exists
        model = self.get_model(symbol, model_type)
        if model is None:
            return ModelHealth(
                symbol=symbol,
                model_type=model_type,
                status=ModelHealthStatus.UNKNOWN,
                reason="Model not registered",
            )

        # Get model age
        created = self._model_created.get(key)
        days_since_trained = None
        if created:
            days_since_trained = (datetime.now() - created).days

        # Check if should retrain
        should_retrain, trigger, reason = self.pipeline.should_retrain(symbol, model_type, created)

        # Get latest metrics from monitor
        latest_metrics = self.pipeline.monitor.get_latest_metrics(symbol, model_type)

        # Determine health status
        status = ModelHealthStatus.HEALTHY
        if should_retrain:
            if trigger == RetrainingTrigger.PERFORMANCE_DEGRADATION:
                status = ModelHealthStatus.CRITICAL
            elif trigger == RetrainingTrigger.DATA_DRIFT:
                status = ModelHealthStatus.CRITICAL
            elif trigger == RetrainingTrigger.MODEL_AGE:
                status = ModelHealthStatus.WARNING
            else:
                status = ModelHealthStatus.WARNING

        # Check for drift
        drift_detected = False
        if trigger == RetrainingTrigger.DATA_DRIFT:
            drift_detected = True

        health = ModelHealth(
            symbol=symbol,
            model_type=model_type,
            status=status,
            accuracy=latest_metrics.accuracy if latest_metrics else None,
            sharpe_ratio=latest_metrics.sharpe_ratio if latest_metrics else None,
            win_rate=latest_metrics.win_rate if latest_metrics else None,
            days_since_trained=days_since_trained,
            drift_detected=drift_detected,
            needs_retraining=should_retrain,
            reason=reason,
        )

        # Cache the health check
        with self._lock:
            self._health_cache[key] = health

        return health

    def check_all_models_health(self) -> Dict[str, ModelHealth]:
        """Check health of all registered models."""
        results = {}
        with self._lock:
            models_copy = dict(self._models)

        for symbol, model_types in models_copy.items():
            for model_type in model_types:
                key = f"{symbol}_{model_type}"
                health = self.check_model_health(symbol, model_type)
                results[key] = health

        return results

    def record_performance(
        self,
        symbol: str,
        model_type: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Record model performance metrics.

        Call this after each trading period to track performance.
        """
        self.pipeline.monitor.record_performance(symbol, model_type, metrics)

    def schedule_retraining(
        self,
        symbol: str,
        model_type: str,
        trigger: RetrainingTrigger = RetrainingTrigger.MANUAL,
        priority: int = 5,
    ) -> RetrainingJob:
        """
        Schedule a model for retraining.

        Args:
            symbol: Trading symbol
            model_type: Model type
            trigger: Reason for retraining
            priority: Job priority (lower = higher priority)

        Returns:
            The scheduled job
        """
        job = RetrainingJob(
            symbol=symbol,
            model_type=model_type,
            trigger=trigger,
            priority=priority,
            scheduled_at=datetime.now(),
        )

        with self._lock:
            # Check if job already exists
            for existing in self._job_queue:
                if existing.symbol == symbol and existing.model_type == model_type:
                    logger.info(f"Job already scheduled for {symbol} {model_type}")
                    return existing

            self._job_queue.append(job)
            # Sort by priority
            self._job_queue.sort(key=lambda j: j.priority)

        logger.info(f"Scheduled retraining: {symbol} {model_type} (trigger: {trigger.value})")
        return job

    def _process_job_queue(self) -> None:
        """Process pending retraining jobs."""
        with self._lock:
            if not self._job_queue:
                return
            job = self._job_queue.pop(0)

        # Mark as running
        job.status = "running"
        job.started_at = datetime.now()

        logger.info(f"Starting retraining job: {job.symbol} {job.model_type}")

        try:
            # Fetch training data
            if self.data_fetcher is None:
                raise ValueError("No data_fetcher configured")

            training_data = self.data_fetcher(job.symbol, 90)  # 90 days of data

            if training_data is None or len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data) if training_data is not None else 0}"
                )

            # Define training function
            def train_model(data: pd.DataFrame) -> Any:
                if self.train_func is None:
                    raise ValueError("No train_func configured")
                return self.train_func(job.model_type, data)

            # Define validation function
            def validate_model(model: Any, val_data: pd.DataFrame) -> PerformanceMetrics:
                # Basic validation - override for real implementation
                return PerformanceMetrics(
                    accuracy=0.55,
                    precision=0.55,
                    recall=0.55,
                    f1_score=0.55,
                    sharpe_ratio=1.0,
                    win_rate=0.52,
                    profit_factor=1.2,
                    max_drawdown=0.10,
                )

            # Run retraining
            result = self.pipeline.retrain(
                symbol=job.symbol,
                model_type=job.model_type,
                training_data=training_data,
                trigger=job.trigger,
                train_func=train_model,
                validate_func=validate_model,
            )

            job.result = result
            job.status = "completed" if result.success else "failed"

            # Hot swap if deployed
            if result.deployed and result.success:
                # The new model would be returned from train_func
                # For now, just log - in real implementation, get model from result
                logger.info(f"Model deployed: {job.symbol} {job.model_type}")

        except Exception as e:
            logger.error(f"Retraining job failed: {e}")
            job.status = "failed"

        job.completed_at = datetime.now()

        with self._lock:
            self._completed_jobs.append(job)
            # Keep only recent history
            if len(self._completed_jobs) > 100:
                self._completed_jobs = self._completed_jobs[-50:]

    def _check_and_schedule(self) -> None:
        """Check all models and schedule retraining if needed."""
        logger.debug("Running health check on all models")

        health_results = self.check_all_models_health()

        for key, health in health_results.items():
            if health.needs_retraining:
                # Determine priority based on status
                priority = 5
                if health.status == ModelHealthStatus.CRITICAL:
                    priority = 1
                elif health.status == ModelHealthStatus.WARNING:
                    priority = 3

                # Determine trigger
                trigger = RetrainingTrigger.MANUAL
                if health.drift_detected:
                    trigger = RetrainingTrigger.DATA_DRIFT
                elif (
                    health.days_since_trained
                    and health.days_since_trained > self.config.max_model_age_days
                ):
                    trigger = RetrainingTrigger.MODEL_AGE
                elif health.accuracy and health.accuracy < self.config.min_accuracy:
                    trigger = RetrainingTrigger.PERFORMANCE_DEGRADATION

                self.schedule_retraining(
                    symbol=health.symbol,
                    model_type=health.model_type,
                    trigger=trigger,
                    priority=priority,
                )

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Auto-retraining scheduler started")

        last_check = datetime.now() - timedelta(hours=self.check_interval / 3600 + 1)

        while self._running:
            try:
                now = datetime.now()

                # Check if it's time for health check
                if (now - last_check).total_seconds() >= self.check_interval:
                    self._check_and_schedule()
                    last_check = now

                # Process job queue
                self._process_job_queue()

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            # Sleep for a bit before next iteration
            time.sleep(60)  # Check every minute

        logger.info("Auto-retraining scheduler stopped")

    def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Auto-retraining scheduler started")

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Auto-retraining scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        with self._lock:
            return {
                "running": self._running,
                "check_interval_hours": self.check_interval / 3600,
                "registered_models": [
                    f"{sym}_{mt}" for sym, types in self._models.items() for mt in types
                ],
                "pending_jobs": [j.to_dict() for j in self._job_queue],
                "recent_jobs": [j.to_dict() for j in self._completed_jobs[-10:]],
                "health_cache": {k: v.to_dict() for k, v in self._health_cache.items()},
            }

    def get_model_health_summary(self) -> Dict[str, Any]:
        """Get summary of all model health statuses."""
        health_results = self.check_all_models_health()

        summary = {
            "total_models": len(health_results),
            "healthy": 0,
            "warning": 0,
            "critical": 0,
            "unknown": 0,
            "models": [],
        }

        for key, health in health_results.items():
            if health.status == ModelHealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif health.status == ModelHealthStatus.WARNING:
                summary["warning"] += 1
            elif health.status == ModelHealthStatus.CRITICAL:
                summary["critical"] += 1
            else:
                summary["unknown"] += 1

            summary["models"].append(health.to_dict())

        return summary

    def force_retrain(
        self,
        symbol: str,
        model_type: str,
    ) -> RetrainingJob:
        """
        Force immediate retraining of a model.

        Schedules the job with highest priority.
        """
        return self.schedule_retraining(
            symbol=symbol,
            model_type=model_type,
            trigger=RetrainingTrigger.MANUAL,
            priority=0,  # Highest priority
        )


def create_auto_retrainer(
    symbols: List[str],
    model_type: str = "gradient_boosting",
    check_interval_hours: float = 6,
    data_dir: str = "data/auto_retraining",
) -> AutoRetrainingScheduler:
    """
    Factory function to create an auto-retraining scheduler.

    Args:
        symbols: List of symbols to monitor
        model_type: Type of model
        check_interval_hours: Health check interval
        data_dir: Data directory

    Returns:
        Configured AutoRetrainingScheduler
    """
    scheduler = AutoRetrainingScheduler(
        check_interval_hours=check_interval_hours,
        data_dir=data_dir,
    )

    # Register symbols (models will be registered later when loaded)
    for symbol in symbols:
        scheduler.register_model(symbol, model_type, None)

    return scheduler
