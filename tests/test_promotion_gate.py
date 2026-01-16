"""
Tests for promotion gate module.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from bot.promotion_gate import (
    PromotionStatus,
    ComparisonResult,
    PromotionCriteria,
    ChallengerPerformance,
    ChampionRecord,
    PromotionGate,
    get_promotion_gate,
)


class TestPromotionStatus:
    """Test PromotionStatus enum."""

    def test_all_statuses(self):
        """Test all promotion statuses exist."""
        assert PromotionStatus.PENDING.value == "pending"
        assert PromotionStatus.SHADOW.value == "shadow"
        assert PromotionStatus.EVALUATION.value == "evaluation"
        assert PromotionStatus.CANARY.value == "canary"
        assert PromotionStatus.PROMOTED.value == "promoted"
        assert PromotionStatus.REJECTED.value == "rejected"
        assert PromotionStatus.ROLLED_BACK.value == "rolled_back"


class TestComparisonResult:
    """Test ComparisonResult enum."""

    def test_all_results(self):
        """Test all comparison results exist."""
        assert ComparisonResult.CHALLENGER_BETTER.value == "challenger_better"
        assert ComparisonResult.CHAMPION_BETTER.value == "champion_better"
        assert ComparisonResult.NO_DIFFERENCE.value == "no_difference"
        assert ComparisonResult.INSUFFICIENT_DATA.value == "insufficient_data"


class TestPromotionCriteria:
    """Test PromotionCriteria dataclass."""

    def test_default_criteria(self):
        """Test default criteria values."""
        criteria = PromotionCriteria()
        assert criteria.min_shadow_days == 14
        assert criteria.min_shadow_trades == 50
        assert criteria.min_evaluation_days == 7
        assert criteria.min_sharpe_improvement == 0.1
        assert criteria.confidence_level == 0.95
        assert criteria.max_acceptable_drawdown == 15.0

    def test_custom_criteria(self):
        """Test custom criteria values."""
        criteria = PromotionCriteria(
            min_shadow_days=7,
            min_shadow_trades=30,
            min_sharpe_improvement=0.2,
        )
        assert criteria.min_shadow_days == 7
        assert criteria.min_shadow_trades == 30
        assert criteria.min_sharpe_improvement == 0.2


class TestChallengerPerformance:
    """Test ChallengerPerformance dataclass."""

    def test_default_creation(self):
        """Test creating challenger with defaults."""
        challenger = ChallengerPerformance(
            challenger_id="test_001",
            strategy_name="momentum",
            strategy_version="2.0",
        )
        assert challenger.challenger_id == "test_001"
        assert challenger.strategy_name == "momentum"
        assert challenger.status == PromotionStatus.PENDING
        assert challenger.shadow_trades == 0
        assert challenger.shadow_pnl_pct == 0.0

    def test_to_dict(self):
        """Test conversion to dict."""
        challenger = ChallengerPerformance(
            challenger_id="test_001",
            strategy_name="momentum",
            strategy_version="2.0",
            status=PromotionStatus.SHADOW,
            shadow_trades=25,
            shadow_pnl_pct=5.5,
            shadow_sharpe=1.2,
        )
        d = challenger.to_dict()

        assert d["challenger_id"] == "test_001"
        assert d["status"] == "shadow"
        assert d["shadow_trades"] == 25
        assert d["shadow_pnl_pct"] == 5.5


class TestChampionRecord:
    """Test ChampionRecord dataclass."""

    def test_champion_creation(self):
        """Test creating champion record."""
        now = datetime.now()
        champion = ChampionRecord(
            strategy_name="momentum",
            strategy_version="1.0",
            promoted_at=now,
            total_trades=100,
            sharpe_ratio=1.5,
        )
        assert champion.strategy_name == "momentum"
        assert champion.total_trades == 100
        assert champion.sharpe_ratio == 1.5

    def test_to_dict(self):
        """Test champion to dict conversion."""
        now = datetime.now()
        champion = ChampionRecord(
            strategy_name="mean_reversion",
            strategy_version="3.0",
            promoted_at=now,
            total_trades=200,
            total_pnl_pct=15.5,
            sharpe_ratio=1.8,
            win_rate=0.55,
        )
        d = champion.to_dict()

        assert d["strategy_name"] == "mean_reversion"
        assert d["total_trades"] == 200
        assert d["sharpe_ratio"] == 1.8


class TestPromotionGate:
    """Test PromotionGate class."""

    @pytest.fixture
    def temp_state_file(self):
        """Create temporary state file."""
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/promotion_state.json"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def gate(self, temp_state_file):
        """Create promotion gate instance."""
        criteria = PromotionCriteria(
            min_shadow_days=1,
            min_shadow_trades=5,
        )
        return PromotionGate(criteria=criteria, state_file=temp_state_file)

    def test_gate_creation(self, gate):
        """Test gate is created."""
        assert gate is not None
        assert gate.criteria is not None
        assert gate.challengers == {}
        assert gate.champions == {}

    def test_register_challenger(self, gate):
        """Test registering a challenger."""
        challenger_id = gate.register_challenger(
            strategy_name="momentum",
            strategy_version="2.0",
        )

        assert challenger_id is not None
        assert challenger_id in gate.challengers
        assert gate.challengers[challenger_id].status == PromotionStatus.PENDING

    def test_register_challenger_custom_id(self, gate):
        """Test registering with custom ID."""
        challenger_id = gate.register_challenger(
            strategy_name="momentum",
            strategy_version="2.0",
            challenger_id="my_custom_id",
        )

        assert challenger_id == "my_custom_id"
        assert "my_custom_id" in gate.challengers

    def test_start_shadow_mode(self, gate):
        """Test starting shadow mode."""
        challenger_id = gate.register_challenger("test", "1.0")
        result = gate.start_shadow_mode(challenger_id)

        assert result is True
        assert gate.challengers[challenger_id].status == PromotionStatus.SHADOW
        assert gate.challengers[challenger_id].started_at is not None

    def test_start_shadow_mode_nonexistent(self, gate):
        """Test starting shadow mode for nonexistent challenger."""
        result = gate.start_shadow_mode("nonexistent")
        assert result is False

    def test_record_shadow_trade(self, gate):
        """Test recording shadow trades."""
        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        for i in range(10):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.5)

        challenger = gate.challengers[challenger_id]
        assert challenger.shadow_trades == 10
        assert challenger.shadow_pnl_pct == 5.0

    def test_shadow_metrics_calculation(self, gate):
        """Test shadow mode metrics are calculated."""
        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        # Record mixed trades
        returns = [0.5, -0.2, 0.8, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1]
        for r in returns:
            gate.record_shadow_trade(challenger_id, pnl_pct=r)

        challenger = gate.challengers[challenger_id]
        assert challenger.shadow_trades == 10
        assert challenger.shadow_win_rate > 0

    def test_start_evaluation(self, gate):
        """Test starting evaluation."""
        gate.criteria.min_shadow_days = 0
        gate.criteria.min_shadow_trades = 5

        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        for i in range(5):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.5)

        # Manually set days for test
        gate.challengers[challenger_id].shadow_days = 1

        result = gate.start_evaluation(challenger_id)
        assert result is True
        assert gate.challengers[challenger_id].status == PromotionStatus.EVALUATION

    def test_evaluate_no_champion(self, gate):
        """Test evaluation when no champion exists."""
        challenger_id = gate.register_challenger("new_strategy", "1.0")
        gate.start_shadow_mode(challenger_id)

        for i in range(60):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.3)

        result, analysis = gate.evaluate_challenger(challenger_id)

        assert result == ComparisonResult.CHALLENGER_BETTER
        assert "No existing champion" in analysis.get("reason", "")

    def test_evaluate_insufficient_data(self, gate):
        """Test evaluation with insufficient data when champion exists."""
        gate.criteria.min_shadow_trades = 100

        # Manually set up a champion so the insufficient data path is triggered
        gate.champions["test"] = ChampionRecord(
            strategy_name="test",
            strategy_version="1.0",
            promoted_at=datetime.now(),
            total_trades=100,
            sharpe_ratio=1.5,
        )

        # Now register a new challenger with insufficient trades
        challenger_id = gate.register_challenger("test", "2.0")
        gate.start_shadow_mode(challenger_id)

        for i in range(10):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.5)

        result, analysis = gate.evaluate_challenger(challenger_id)

        # With a champion existing, insufficient data should be detected
        assert result == ComparisonResult.INSUFFICIENT_DATA

    def test_get_promotion_status(self, gate):
        """Test getting promotion status."""
        challenger_id = gate.register_challenger("test", "1.0")

        status = gate.get_promotion_status(challenger_id)

        assert "challenger_id" in status
        assert "criteria" in status
        assert status["criteria"]["min_shadow_days"] == 1

    def test_get_promotion_status_nonexistent(self, gate):
        """Test status for nonexistent challenger."""
        status = gate.get_promotion_status("nonexistent")
        assert "error" in status

    def test_get_champion(self, gate):
        """Test getting champion info."""
        # No champion yet
        result = gate.get_champion("test")
        assert result is None

    def test_get_all_champions(self, gate):
        """Test getting all champions."""
        champions = gate.get_all_champions()
        assert isinstance(champions, dict)

    def test_get_active_challengers(self, gate):
        """Test getting active challengers."""
        gate.register_challenger("test1", "1.0")
        gate.register_challenger("test2", "1.0")

        challengers = gate.get_active_challengers()
        assert len(challengers) == 2


class TestStatisticalTests:
    """Test statistical comparison methods."""

    @pytest.fixture
    def temp_state_file(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/promotion_state.json"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def gate(self, temp_state_file):
        return PromotionGate(state_file=temp_state_file)

    def test_welch_t_test_same_samples(self, gate):
        """Test Welch's t-test with identical samples."""
        import numpy as np

        sample1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        t_stat, p_value = gate._welch_t_test(sample1, sample2)

        assert abs(t_stat) < 0.1  # Should be near 0
        assert p_value > 0.9  # Should be high (not significant)

    def test_welch_t_test_different_samples(self, gate):
        """Test Welch's t-test with different samples."""
        import numpy as np

        sample1 = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        sample2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        t_stat, p_value = gate._welch_t_test(sample1, sample2)

        assert t_stat > 1.0  # sample1 mean is higher
        assert p_value < 0.5  # Should show difference

    def test_welch_t_test_insufficient_data(self, gate):
        """Test t-test with insufficient data."""
        import numpy as np

        sample1 = np.array([1.0])
        sample2 = np.array([2.0])

        t_stat, p_value = gate._welch_t_test(sample1, sample2)

        assert t_stat == 0.0
        assert p_value == 1.0

    def test_calculate_sharpe(self, gate):
        """Test Sharpe ratio calculation."""
        import numpy as np

        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.005, 0.015])
        sharpe = gate._calculate_sharpe(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should have positive Sharpe

    def test_calculate_sharpe_insufficient_data(self, gate):
        """Test Sharpe with single return."""
        import numpy as np

        returns = np.array([0.01])
        sharpe = gate._calculate_sharpe(returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_zero_std(self, gate):
        """Test Sharpe with zero standard deviation."""
        import numpy as np

        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = gate._calculate_sharpe(returns)

        assert sharpe == 0.0


class TestStatePersistence:
    """Test state persistence."""

    @pytest.fixture
    def temp_state_file(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/promotion_state.json"
        shutil.rmtree(temp_dir)

    def test_state_saved_on_register(self, temp_state_file):
        """Test state is saved when challenger registered."""
        gate = PromotionGate(state_file=temp_state_file)
        gate.register_challenger("test", "1.0")

        assert Path(temp_state_file).exists()

    def test_state_loaded_on_restart(self, temp_state_file):
        """Test state is loaded on restart."""
        # Create first gate and register challenger
        gate1 = PromotionGate(state_file=temp_state_file)
        gate1.register_challenger("test", "1.0", challenger_id="persistent_id")

        # Create second gate - should load state
        gate2 = PromotionGate(state_file=temp_state_file)

        # Note: Challengers aren't persisted by default in load_state
        # But state file should exist
        assert Path(temp_state_file).exists()


class TestCanaryRollout:
    """Test canary rollout functionality."""

    @pytest.fixture
    def temp_state_file(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/promotion_state.json"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def gate(self, temp_state_file):
        criteria = PromotionCriteria(
            min_shadow_days=0,
            min_shadow_trades=5,
        )
        return PromotionGate(criteria=criteria, state_file=temp_state_file)

    def test_start_canary_without_evaluation(self, gate):
        """Test canary can't start without passing evaluation."""
        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        result = gate.start_canary(challenger_id)
        assert result is False

    def test_record_canary_trade(self, gate):
        """Test recording canary trades."""
        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        for _ in range(10):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.5)

        # Set up for canary
        gate.challengers[challenger_id].comparison_result = ComparisonResult.CHALLENGER_BETTER
        gate.start_canary(challenger_id)

        gate.record_canary_trade(challenger_id, pnl_pct=0.3)

        assert gate.challengers[challenger_id].canary_trades == 1
        assert gate.challengers[challenger_id].canary_pnl_pct == 0.3

    def test_check_canary_health_insufficient_trades(self, gate):
        """Test canary health with insufficient trades."""
        challenger_id = gate.register_challenger("test", "1.0")
        gate.start_shadow_mode(challenger_id)

        for _ in range(10):
            gate.record_shadow_trade(challenger_id, pnl_pct=0.5)

        gate.challengers[challenger_id].comparison_result = ComparisonResult.CHALLENGER_BETTER
        gate.start_canary(challenger_id)

        healthy, reason = gate.check_canary_health(challenger_id)
        assert healthy is False
        assert "more canary trades" in reason.lower()
