"""
Tests for Auto-Retraining System.

Tests performance monitoring, degradation detection, model registry, and retraining.
"""

from __future__ import annotations

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
import numpy as np

from bot.auto_retraining import (
    ModelStatus,
    DegradationReason,
    ModelMetrics,
    ModelRecord,
    RetrainingTrigger,
    PerformanceMonitor,
    ModelRegistry,
    AutoRetrainingManager,
    create_auto_retraining_manager,
)


class TestModelStatus:
    """Test ModelStatus enum."""

    def test_statuses_defined(self):
        """Test all model statuses are defined."""
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.VALIDATING.value == "validating"
        assert ModelStatus.CANDIDATE.value == "candidate"
        assert ModelStatus.SHADOW.value == "shadow"
        assert ModelStatus.PRODUCTION.value == "production"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.FAILED.value == "failed"


class TestDegradationReason:
    """Test DegradationReason enum."""

    def test_reasons_defined(self):
        """Test all degradation reasons are defined."""
        assert DegradationReason.ACCURACY_DROP
        assert DegradationReason.PRECISION_DROP
        assert DegradationReason.PROFIT_FACTOR_DROP
        assert DegradationReason.CONCEPT_DRIFT
        assert DegradationReason.DATA_DRIFT
        assert DegradationReason.REGIME_CHANGE
        assert DegradationReason.MANUAL


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_id="test_model",
        )

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0

    def test_custom_metrics(self):
        """Test custom metrics."""
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_id="test_model",
            accuracy=0.65,
            precision=0.60,
            win_rate=0.55,
            profit_factor=1.5,
            total_trades=100,
        )

        assert metrics.accuracy == 0.65
        assert metrics.precision == 0.60
        assert metrics.profit_factor == 1.5

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            model_id="test_model",
            accuracy=0.65,
        )

        data = metrics.to_dict()

        assert data["model_id"] == "test_model"
        assert data["accuracy"] == 0.65
        assert "timestamp" in data


class TestModelRecord:
    """Test ModelRecord dataclass."""

    def test_record_creation(self):
        """Test model record creation."""
        record = ModelRecord(
            model_id="model_001",
            model_path="/models/model_001.pkl",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="abc123",
            training_samples=1000,
            feature_names=["feature_1", "feature_2"],
            training_metrics={"accuracy": 0.65},
            validation_metrics={"accuracy": 0.60},
        )

        assert record.model_id == "model_001"
        assert record.status == ModelStatus.CANDIDATE
        assert record.training_samples == 1000

    def test_get_recent_performance_empty(self):
        """Test getting recent performance with no data."""
        record = ModelRecord(
            model_id="test",
            model_path="/test",
            created_at=datetime.now(),
            status=ModelStatus.PRODUCTION,
            training_data_hash="hash",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )

        result = record.get_recent_performance(days=7)
        assert result is None

    def test_get_recent_performance(self):
        """Test getting recent performance."""
        record = ModelRecord(
            model_id="test",
            model_path="/test",
            created_at=datetime.now(),
            status=ModelStatus.PRODUCTION,
            training_data_hash="hash",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )

        # Add some metrics
        for i in range(10):
            record.production_metrics.append(ModelMetrics(
                timestamp=datetime.now(),
                model_id="test",
                accuracy=0.55 + i * 0.01,
                win_rate=0.5,
            ))

        result = record.get_recent_performance(days=7)

        assert result is not None
        assert result.accuracy > 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor."""
        return PerformanceMonitor(
            accuracy_threshold=0.05,
            precision_threshold=0.05,
            window_size=50,
            min_samples=20,
        )

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.accuracy_threshold == 0.05
        assert monitor.window_size == 50
        assert monitor.min_samples == 20
        assert monitor.baseline_metrics is None

    def test_set_baseline(self, monitor):
        """Test setting baseline metrics."""
        baseline = ModelMetrics(
            timestamp=datetime.now(),
            model_id="baseline",
            accuracy=0.60,
            precision=0.55,
        )

        monitor.set_baseline(baseline)

        assert monitor.baseline_metrics is not None
        assert monitor.baseline_metrics.accuracy == 0.60

    def test_record_prediction(self, monitor):
        """Test recording predictions."""
        monitor.record_prediction(
            prediction=1,
            actual=1,
            trade_pnl=100.0,
        )

        assert len(monitor.predictions) == 1
        assert monitor.predictions[0]["correct"] is True

    def test_check_degradation_insufficient_samples(self, monitor):
        """Test degradation check with insufficient samples."""
        is_degraded, reason, details = monitor.check_degradation()

        assert is_degraded is False
        assert reason is None
        assert "Not enough samples" in details["message"]

    def test_check_degradation_no_baseline(self, monitor):
        """Test degradation check without baseline."""
        # Add some samples
        for i in range(30):
            monitor.record_prediction(1, 1)

        is_degraded, reason, details = monitor.check_degradation()

        assert is_degraded is False
        assert "No baseline" in details["message"]

    def test_check_degradation_accuracy_drop(self, monitor):
        """Test detecting accuracy degradation."""
        # Set high baseline
        monitor.set_baseline(ModelMetrics(
            timestamp=datetime.now(),
            model_id="baseline",
            accuracy=0.70,
            precision=0.65,
        ))

        # Record poor predictions
        for i in range(50):
            monitor.record_prediction(
                prediction=i % 2,
                actual=(i + 1) % 2,  # Always wrong
            )

        is_degraded, reason, details = monitor.check_degradation()

        assert is_degraded is True
        assert reason == DegradationReason.ACCURACY_DROP

    def test_check_no_degradation(self, monitor):
        """Test when no degradation detected."""
        # Set baseline
        monitor.set_baseline(ModelMetrics(
            timestamp=datetime.now(),
            model_id="baseline",
            accuracy=0.55,
            precision=0.50,
        ))

        # Record good predictions
        for i in range(50):
            # 60% correct
            correct = i % 5 != 0
            monitor.record_prediction(
                prediction=1 if correct else 0,
                actual=1,
            )

        is_degraded, reason, details = monitor.check_degradation()

        assert is_degraded is False

    def test_get_current_metrics(self, monitor):
        """Test getting current metrics."""
        for i in range(30):
            monitor.record_prediction(
                prediction=i % 2,
                actual=i % 2,  # All correct
                trade_pnl=100 if i % 2 == 0 else -50,
            )

        metrics = monitor.get_current_metrics()

        assert metrics.predictions_made == 30
        assert metrics.accuracy == 1.0


class TestModelRegistry:
    """Test ModelRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create model registry in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelRegistry(registry_dir=tmpdir)

    def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None
        assert registry.registry_dir.exists()
        assert len(registry.models) == 0

    def test_register_model(self, registry):
        """Test registering a model."""
        record = ModelRecord(
            model_id="model_001",
            model_path="/models/model_001.pkl",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="abc123",
            training_samples=1000,
            feature_names=["f1", "f2"],
            training_metrics={"accuracy": 0.65},
            validation_metrics={"accuracy": 0.60},
        )

        model_id = registry.register_model(record)

        assert model_id == "model_001"
        assert "model_001" in registry.models

    def test_update_status(self, registry):
        """Test updating model status."""
        record = ModelRecord(
            model_id="model_001",
            model_path="/test",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="hash",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )
        registry.register_model(record)

        registry.update_status("model_001", ModelStatus.SHADOW)

        assert registry.models["model_001"].status == ModelStatus.SHADOW

    def test_promote_to_production(self, registry):
        """Test promoting model to production."""
        record = ModelRecord(
            model_id="model_001",
            model_path="/test",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="hash",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )
        registry.register_model(record)

        success = registry.promote_to_production("model_001")

        assert success is True
        assert registry.production_model_id == "model_001"
        assert registry.models["model_001"].status == ModelStatus.PRODUCTION

    def test_rollback_to_previous(self, registry):
        """Test rolling back to previous model."""
        # Register and promote first model
        record1 = ModelRecord(
            model_id="model_001",
            model_path="/test1",
            created_at=datetime.now() - timedelta(days=1),
            status=ModelStatus.CANDIDATE,
            training_data_hash="hash1",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )
        registry.register_model(record1)
        registry.promote_to_production("model_001")

        # Register and promote second model
        record2 = ModelRecord(
            model_id="model_002",
            model_path="/test2",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="hash2",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )
        registry.register_model(record2)
        registry.promote_to_production("model_002")

        # Rollback
        previous_id = registry.rollback_to_previous()

        assert previous_id == "model_001"
        assert registry.production_model_id == "model_001"

    def test_get_production_model(self, registry):
        """Test getting production model."""
        record = ModelRecord(
            model_id="model_001",
            model_path="/test",
            created_at=datetime.now(),
            status=ModelStatus.CANDIDATE,
            training_data_hash="hash",
            training_samples=100,
            feature_names=[],
            training_metrics={},
            validation_metrics={},
        )
        registry.register_model(record)
        registry.promote_to_production("model_001")

        model = registry.get_production_model()

        assert model is not None
        assert model.model_id == "model_001"


class TestAutoRetrainingManager:
    """Test AutoRetrainingManager class."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock training pipeline."""
        pipeline = MagicMock()
        pipeline.train.return_value = MagicMock(
            model_id="new_model_001",
            model_path="/models/new_model.pkl",
            data_hash="newhash",
            train_metrics={"accuracy": 0.65},
            val_metrics={"accuracy": 0.62},
        )
        return pipeline

    @pytest.fixture
    def manager(self, mock_pipeline):
        """Create auto-retraining manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)
            monitor = PerformanceMonitor()

            yield AutoRetrainingManager(
                training_pipeline=mock_pipeline,
                model_registry=registry,
                performance_monitor=monitor,
                check_interval_minutes=1,
                min_retrain_interval_hours=0,  # No minimum for tests
            )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager._running is False
        assert manager.training_pipeline is not None

    def test_get_status(self, manager):
        """Test getting manager status."""
        status = manager.get_status()

        assert "running" in status
        assert "production_model_id" in status
        assert "current_metrics" in status

    def test_record_prediction(self, manager):
        """Test recording predictions."""
        manager.record_prediction(
            prediction=1,
            actual=1,
            trade_pnl=50.0,
        )

        metrics = manager.monitor.get_current_metrics()
        assert metrics.predictions_made == 1

    def test_should_retrain_respects_interval(self, manager):
        """Test retraining interval is respected."""
        manager._last_retrain_time = datetime.now()
        manager.min_retrain_interval_hours = 24

        assert manager._should_retrain() is False

    def test_on_degradation_callback(self, manager):
        """Test degradation callback registration."""
        callback = MagicMock()
        manager.on_degradation(callback)

        assert callback in manager._on_degradation_callbacks


class TestAutoRetrainingManagerAsync:
    """Test async methods of AutoRetrainingManager."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock training pipeline."""
        pipeline = MagicMock()
        pipeline.train.return_value = MagicMock(
            model_id="new_model_001",
            model_path="/models/new_model.pkl",
            data_hash="newhash",
            train_metrics={"accuracy": 0.65},
            val_metrics={"accuracy": 0.62},
        )
        return pipeline

    @pytest.fixture
    def manager(self, mock_pipeline):
        """Create auto-retraining manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            yield AutoRetrainingManager(
                training_pipeline=mock_pipeline,
                model_registry=registry,
                min_retrain_interval_hours=0,
            )

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Test starting and stopping manager."""
        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_trigger_retraining(self, manager):
        """Test triggering retraining."""
        model_id = await manager.trigger_retraining()

        assert model_id == "new_model_001"
        assert "new_model_001" in manager.registry.models

    @pytest.mark.asyncio
    async def test_validate_and_promote(self, manager):
        """Test validating and promoting a model."""
        # First, create a model
        model_id = await manager.trigger_retraining()

        # Validate and promote
        import pandas as pd
        val_data = pd.DataFrame({"x": [1, 2, 3]})

        success = await manager.validate_and_promote(
            model_id=model_id,
            validation_data=val_data,
            min_accuracy=0.50,  # Lower threshold for test
        )

        assert success is True
        assert manager.registry.production_model_id == model_id

    @pytest.mark.asyncio
    async def test_rollback(self, manager):
        """Test model rollback."""
        # Create and promote two models
        manager.mock_pipeline = manager.training_pipeline  # Reference

        await manager.trigger_retraining()
        manager.registry.promote_to_production("new_model_001")

        # Change the mock to return different ID
        manager.training_pipeline.train.return_value = MagicMock(
            model_id="new_model_002",
            model_path="/models/new_model2.pkl",
            data_hash="hash2",
            train_metrics={"accuracy": 0.65},
            val_metrics={"accuracy": 0.60},
        )

        await manager.trigger_retraining()
        manager.registry.promote_to_production("new_model_002")

        # Rollback
        previous_id = await manager.rollback("Test rollback")

        assert previous_id == "new_model_001"


class TestRetrainingTrigger:
    """Test RetrainingTrigger dataclass."""

    def test_trigger_creation(self):
        """Test trigger creation."""
        trigger = RetrainingTrigger(
            trigger_id="trigger_001",
            triggered_at=datetime.now(),
            reason=DegradationReason.ACCURACY_DROP,
            model_id="model_001",
            current_value=0.50,
            threshold=0.05,
            baseline_value=0.60,
        )

        assert trigger.trigger_id == "trigger_001"
        assert trigger.reason == DegradationReason.ACCURACY_DROP
        assert trigger.acknowledged is False
        assert trigger.retraining_started is False
