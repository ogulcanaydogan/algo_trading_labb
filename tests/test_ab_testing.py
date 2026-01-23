"""
Tests for the A/B testing module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from bot.ml.ab_testing import (
    ABExperiment,
    ABTestManager,
    ExperimentOutcome,
    ExperimentStatus,
    VariantMetrics,
    WinnerDecision,
    create_ab_test_manager,
)


class TestABExperiment:
    """Tests for ABExperiment dataclass."""

    def test_creation(self):
        """Test creating an experiment."""
        exp = ABExperiment(
            name="test_exp",
            description="Test experiment",
            control_model=MagicMock(),
            treatment_model=MagicMock(),
            traffic_split=0.3,
        )

        assert exp.name == "test_exp"
        assert exp.traffic_split == 0.3
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.min_samples == 100

    def test_to_dict(self):
        """Test serialization."""
        exp = ABExperiment(
            name="test_exp",
            description="Test",
            control_model=None,
            treatment_model=None,
            traffic_split=0.5,
        )
        data = exp.to_dict()

        assert data["name"] == "test_exp"
        assert data["traffic_split"] == 0.5
        assert data["status"] == "draft"


class TestVariantMetrics:
    """Tests for VariantMetrics dataclass."""

    def test_defaults(self):
        """Test default values."""
        metrics = VariantMetrics()
        assert metrics.total_samples == 0
        assert metrics.accuracy == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_to_dict(self):
        """Test serialization."""
        metrics = VariantMetrics(
            total_samples=100,
            correct_predictions=55,
            accuracy=0.55,
            total_pnl=500.0,
            avg_pnl=5.0,
        )
        data = metrics.to_dict()

        assert data["total_samples"] == 100
        assert data["accuracy"] == 0.55
        assert data["avg_pnl"] == 5.0


class TestABTestManager:
    """Tests for ABTestManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager."""
        return ABTestManager(data_dir=str(tmp_path / "ab_testing"))

    @pytest.fixture
    def control_model(self):
        """Mock control model."""
        return MagicMock(name="control")

    @pytest.fixture
    def treatment_model(self):
        """Mock treatment model."""
        return MagicMock(name="treatment")

    def test_create_experiment(self, manager, control_model, treatment_model):
        """Test creating an experiment."""
        exp = manager.create_experiment(
            name="test_exp",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.2,
            description="Testing new model",
        )

        assert exp.name == "test_exp"
        assert exp.traffic_split == 0.2
        assert exp.control_model == control_model
        assert exp.treatment_model == treatment_model
        assert exp.status == ExperimentStatus.DRAFT

    def test_create_duplicate_experiment_raises(self, manager, control_model, treatment_model):
        """Test that duplicate names raise error."""
        manager.create_experiment(
            name="duplicate",
            control_model=control_model,
            treatment_model=treatment_model,
        )

        with pytest.raises(ValueError, match="already exists"):
            manager.create_experiment(
                name="duplicate",
                control_model=control_model,
                treatment_model=treatment_model,
            )

    def test_invalid_traffic_split_raises(self, manager, control_model, treatment_model):
        """Test that invalid splits raise error."""
        with pytest.raises(ValueError, match="traffic_split"):
            manager.create_experiment(
                name="invalid",
                control_model=control_model,
                treatment_model=treatment_model,
                traffic_split=1.5,
            )

    def test_start_experiment(self, manager, control_model, treatment_model):
        """Test starting an experiment."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
        )

        manager.start_experiment("exp")
        exp = manager.get_experiment("exp")

        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

    def test_pause_experiment(self, manager, control_model, treatment_model):
        """Test pausing an experiment."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
        )
        manager.start_experiment("exp")
        manager.pause_experiment("exp")

        exp = manager.get_experiment("exp")
        assert exp.status == ExperimentStatus.PAUSED

    def test_stop_experiment_with_winner(self, manager, control_model, treatment_model):
        """Test stopping with winner declaration."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
        )
        manager.start_experiment("exp")
        manager.stop_experiment("exp", winner=WinnerDecision.TREATMENT)

        exp = manager.get_experiment("exp")
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.winner == WinnerDecision.TREATMENT
        assert exp.ended_at is not None

    def test_get_model_returns_control_when_not_running(self, manager, control_model, treatment_model):
        """Test that non-running experiments return control."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.5,
        )

        model, variant = manager.get_model("exp", symbol="BTC/USDT")
        assert model == control_model
        assert variant == "control"

    def test_get_model_traffic_splitting(self, manager, control_model, treatment_model):
        """Test traffic splitting when experiment is running."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.5,
        )
        manager.start_experiment("exp")

        # Run many times and check distribution
        control_count = 0
        treatment_count = 0
        for i in range(1000):
            model, variant = manager.get_model("exp", symbol=f"symbol_{i}")
            if variant == "control":
                control_count += 1
            else:
                treatment_count += 1

        # Should be roughly 50/50 with some variance
        assert 400 < control_count < 600
        assert 400 < treatment_count < 600

    def test_consistent_assignment(self, manager, control_model, treatment_model):
        """Test that same symbol always gets same variant."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.5,
        )
        manager.start_experiment("exp")

        # Same symbol should always get same variant
        _, first_variant = manager.get_model("exp", symbol="BTC/USDT")
        for _ in range(10):
            _, variant = manager.get_model("exp", symbol="BTC/USDT")
            assert variant == first_variant

    def test_record_outcome(self, manager, control_model, treatment_model):
        """Test recording outcomes."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
        )
        manager.start_experiment("exp")

        manager.record_outcome(
            experiment_name="exp",
            symbol="BTC/USDT",
            predicted=1,
            actual=1,
            pnl=50.0,
            variant="control",
        )

        exp = manager.get_experiment("exp")
        assert len(exp.outcomes) == 1
        assert exp.outcomes[0].correct is True
        assert exp.outcomes[0].pnl == 50.0

    def test_analyze_experiment_insufficient_samples(self, manager, control_model, treatment_model):
        """Test analysis with insufficient samples."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            min_samples=100,
        )
        manager.start_experiment("exp")

        # Add only a few outcomes
        for i in range(10):
            manager.record_outcome(
                experiment_name="exp",
                symbol="BTC/USDT",
                predicted=1,
                actual=1,
                pnl=10.0,
                variant="control",
            )

        results = manager.analyze_experiment("exp")
        assert results.winner == WinnerDecision.INCONCLUSIVE
        assert "Need more samples" in results.recommendation

    def test_analyze_experiment_with_data(self, manager, control_model, treatment_model):
        """Test analysis with sufficient data."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            min_samples=100,  # Set high to prevent auto-conclude
        )
        manager.start_experiment("exp")

        # Add control outcomes (50% accuracy, avg pnl = 10)
        for i in range(30):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_{i}",
                predicted=1,
                actual=1 if i % 2 == 0 else 0,
                pnl=20.0 if i % 2 == 0 else -10.0,
                variant="control",
            )

        # Add treatment outcomes (70% accuracy, avg pnl = 20)
        for i in range(30):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_{i}",
                predicted=1,
                actual=1 if i % 10 != 0 else 0,
                pnl=30.0 if i % 10 != 0 else -10.0,
                variant="treatment",
            )

        results = manager.analyze_experiment("exp")

        assert results.control_metrics.total_samples == 30
        assert results.treatment_metrics.total_samples == 30
        assert results.treatment_metrics.accuracy > results.control_metrics.accuracy

    def test_list_experiments(self, manager, control_model, treatment_model):
        """Test listing experiments."""
        manager.create_experiment(
            name="exp1",
            control_model=control_model,
            treatment_model=treatment_model,
        )
        manager.create_experiment(
            name="exp2",
            control_model=control_model,
            treatment_model=treatment_model,
        )
        manager.start_experiment("exp1")

        all_exps = manager.list_experiments()
        assert len(all_exps) == 2

        running_exps = manager.list_experiments(status=ExperimentStatus.RUNNING)
        assert len(running_exps) == 1
        assert running_exps[0].name == "exp1"

    def test_delete_experiment(self, manager, control_model, treatment_model):
        """Test deleting an experiment."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
        )

        result = manager.delete_experiment("exp")
        assert result is True

        assert manager.get_experiment("exp") is None
        assert manager.delete_experiment("nonexistent") is False

    def test_register_models(self, manager, control_model, treatment_model):
        """Test registering models for an experiment."""
        manager.create_experiment(
            name="exp",
            control_model=None,
            treatment_model=None,
        )

        manager.register_models("exp", control_model, treatment_model)

        exp = manager.get_experiment("exp")
        assert exp.control_model == control_model
        assert exp.treatment_model == treatment_model

    def test_gradual_rollout(self, manager, control_model, treatment_model):
        """Test gradual rollout functionality."""
        manager.create_experiment(
            name="exp",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.1,
        )

        manager.gradual_rollout("exp", 0.5)

        exp = manager.get_experiment("exp")
        assert exp.traffic_split == 0.5

    def test_persistence(self, tmp_path, control_model, treatment_model):
        """Test that experiments persist across manager instances."""
        # Create experiment with first manager
        manager1 = ABTestManager(data_dir=str(tmp_path / "ab_testing"))
        manager1.create_experiment(
            name="persistent",
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=0.3,
        )

        # Load with second manager
        manager2 = ABTestManager(data_dir=str(tmp_path / "ab_testing"))

        exp = manager2.get_experiment("persistent")
        assert exp is not None
        assert exp.name == "persistent"
        assert exp.traffic_split == 0.3

    def test_factory_function(self, tmp_path):
        """Test factory function."""
        manager = create_ab_test_manager(
            data_dir=str(tmp_path / "ab"),
            default_significance=0.01,
        )

        assert manager.default_significance == 0.01


class TestStatisticalAnalysis:
    """Tests for statistical significance calculations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager."""
        return ABTestManager(data_dir=str(tmp_path / "ab_testing"))

    def test_chi_squared_test(self, manager):
        """Test chi-squared test for accuracy difference."""
        manager.create_experiment(
            name="exp",
            control_model=MagicMock(),
            treatment_model=MagicMock(),
            min_samples=500,  # Set very high to prevent auto-conclude
            significance_level=0.05,
        )
        manager.start_experiment("exp")

        # Control: 50% accuracy (50/100)
        for i in range(100):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_{i}",
                predicted=1,
                actual=1 if i < 50 else 0,
                pnl=10.0 if i < 50 else -10.0,
                variant="control",
            )

        # Treatment: 75% accuracy (75/100)
        for i in range(100):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_T_{i}",
                predicted=1,
                actual=1 if i < 75 else 0,
                pnl=10.0 if i < 75 else -10.0,
                variant="treatment",
            )

        results = manager.analyze_experiment("exp")

        # Should show significant difference with larger sample
        assert results.control_metrics.accuracy == 0.5
        assert results.treatment_metrics.accuracy == 0.75
        assert results.accuracy_p_value < 0.05  # Should be significant with n=100

    def test_ttest_for_pnl(self, manager):
        """Test t-test for PnL difference."""
        manager.create_experiment(
            name="exp",
            control_model=MagicMock(),
            treatment_model=MagicMock(),
            min_samples=500,  # Set high to prevent auto-conclude
        )
        manager.start_experiment("exp")

        # Control: low PnL
        for i in range(30):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_{i}",
                predicted=1,
                actual=1,
                pnl=5.0 + (i % 3),  # 5, 6, 7, 5, 6, 7...
                variant="control",
            )

        # Treatment: high PnL
        for i in range(30):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_T_{i}",
                predicted=1,
                actual=1,
                pnl=50.0 + (i % 3),  # 50, 51, 52, ...
                variant="treatment",
            )

        results = manager.analyze_experiment("exp")

        assert results.control_metrics.avg_pnl < 10
        assert results.treatment_metrics.avg_pnl > 40
        assert results.pnl_p_value < 0.05  # Should be very significant


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager."""
        return ABTestManager(data_dir=str(tmp_path / "ab_testing"))

    def test_empty_outcomes(self, manager):
        """Test analysis with no outcomes."""
        manager.create_experiment(
            name="exp",
            control_model=MagicMock(),
            treatment_model=MagicMock(),
        )

        results = manager.analyze_experiment("exp")

        assert results.control_metrics.total_samples == 0
        assert results.treatment_metrics.total_samples == 0

    def test_only_control_outcomes(self, manager):
        """Test analysis with only control outcomes."""
        manager.create_experiment(
            name="exp",
            control_model=MagicMock(),
            treatment_model=MagicMock(),
            min_samples=5,
        )
        manager.start_experiment("exp")

        for i in range(10):
            manager.record_outcome(
                experiment_name="exp",
                symbol=f"SYM_{i}",
                predicted=1,
                actual=1,
                pnl=10.0,
                variant="control",
            )

        results = manager.analyze_experiment("exp")

        assert results.control_metrics.total_samples == 10
        assert results.treatment_metrics.total_samples == 0
        assert results.winner == WinnerDecision.INCONCLUSIVE

    def test_nonexistent_experiment(self, manager):
        """Test operations on nonexistent experiment."""
        with pytest.raises(ValueError, match="not found"):
            manager.start_experiment("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            manager.get_model("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            manager.record_outcome(
                experiment_name="nonexistent",
                symbol="SYM",
                predicted=1,
                actual=1,
                pnl=10.0,
            )
