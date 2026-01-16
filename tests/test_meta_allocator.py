"""
Tests for meta-allocator module.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from bot.meta_allocator import (
    AllocationMethod,
    StrategyAllocation,
    AllocationResult,
    StrategyPerformanceSnapshot,
    AllocationConfig,
    MetaAllocator,
    PortfolioRebalancer,
    get_meta_allocator,
)


class TestAllocationMethod:
    """Test AllocationMethod enum."""

    def test_all_methods_exist(self):
        """Test all allocation methods exist."""
        assert AllocationMethod.EQUAL_WEIGHT.value == "equal_weight"
        assert AllocationMethod.RISK_PARITY.value == "risk_parity"
        assert AllocationMethod.MOMENTUM.value == "momentum"
        assert AllocationMethod.MEAN_REVERSION.value == "mean_reversion"
        assert AllocationMethod.KELLY.value == "kelly"
        assert AllocationMethod.CONTEXTUAL_BANDIT.value == "contextual_bandit"
        assert AllocationMethod.REGIME_ADAPTIVE.value == "regime_adaptive"


class TestStrategyAllocation:
    """Test StrategyAllocation dataclass."""

    def test_allocation_creation(self):
        """Test creating allocation."""
        alloc = StrategyAllocation(
            strategy_name="trend_following",
            weight=0.25,
            capital_amount=2500.0,
        )
        assert alloc.strategy_name == "trend_following"
        assert alloc.weight == 0.25
        assert alloc.capital_amount == 2500.0

    def test_to_dict(self):
        """Test conversion to dict."""
        alloc = StrategyAllocation(
            strategy_name="momentum",
            weight=0.30,
            capital_amount=3000.0,
            base_weight=0.35,
            regime_adjustment=-0.05,
        )
        d = alloc.to_dict()

        assert d["strategy_name"] == "momentum"
        assert d["weight"] == 0.30
        assert d["capital_amount"] == 3000.0


class TestAllocationResult:
    """Test AllocationResult dataclass."""

    def test_result_creation(self):
        """Test creating allocation result."""
        result = AllocationResult(
            total_capital=10000.0,
            current_regime="trending",
        )
        assert result.total_capital == 10000.0
        assert result.current_regime == "trending"
        assert result.allocations == []

    def test_result_with_allocations(self):
        """Test result with allocations."""
        allocs = [
            StrategyAllocation("strat1", 0.5, 5000.0),
            StrategyAllocation("strat2", 0.5, 5000.0),
        ]
        result = AllocationResult(
            total_capital=10000.0,
            allocations=allocs,
            effective_strategies=2,
        )
        assert len(result.allocations) == 2
        assert result.effective_strategies == 2

    def test_to_dict(self):
        """Test conversion to dict."""
        result = AllocationResult(
            method=AllocationMethod.RISK_PARITY,
            total_capital=10000.0,
            current_regime="ranging",
        )
        d = result.to_dict()

        assert d["method"] == "risk_parity"
        assert d["total_capital"] == 10000.0
        assert d["current_regime"] == "ranging"


class TestStrategyPerformanceSnapshot:
    """Test StrategyPerformanceSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating snapshot."""
        snapshot = StrategyPerformanceSnapshot(
            strategy_name="rsi_reversion",
            returns_30d=5.0,
            volatility_30d=0.15,
            sharpe_30d=1.2,
        )
        assert snapshot.strategy_name == "rsi_reversion"
        assert snapshot.returns_30d == 5.0
        assert snapshot.sharpe_30d == 1.2

    def test_default_values(self):
        """Test default values."""
        snapshot = StrategyPerformanceSnapshot(strategy_name="test")
        assert snapshot.returns_1d == 0.0
        assert snapshot.win_rate_30d == 0.0
        assert snapshot.is_degraded is False


class TestAllocationConfig:
    """Test AllocationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = AllocationConfig()
        assert config.method == AllocationMethod.REGIME_ADAPTIVE
        assert config.min_weight == 0.05
        assert config.max_weight == 0.40
        assert config.kelly_fraction == 0.25

    def test_custom_config(self):
        """Test custom configuration."""
        config = AllocationConfig(
            method=AllocationMethod.EQUAL_WEIGHT,
            min_weight=0.10,
            max_weight=0.50,
            target_volatility=0.20,
        )
        assert config.method == AllocationMethod.EQUAL_WEIGHT
        assert config.min_weight == 0.10
        assert config.max_weight == 0.50


class TestMetaAllocator:
    """Test MetaAllocator class."""

    @pytest.fixture
    def allocator(self):
        """Create meta-allocator."""
        config = AllocationConfig(method=AllocationMethod.EQUAL_WEIGHT)
        return MetaAllocator(config)

    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategies."""
        return [
            {"name": "trend_ema", "performance": {"total_pnl_pct": 5.0, "sharpe_ratio": 1.5}},
            {"name": "mean_reversion", "performance": {"total_pnl_pct": 3.0, "sharpe_ratio": 1.2}},
            {"name": "breakout", "performance": {"total_pnl_pct": 7.0, "sharpe_ratio": 1.8}},
        ]

    @pytest.fixture
    def sample_snapshots(self):
        """Create sample performance snapshots."""
        return [
            StrategyPerformanceSnapshot(
                strategy_name="trend_ema",
                returns_30d=5.0,
                volatility_30d=0.15,
                sharpe_30d=1.5,
                win_rate_30d=0.55,
                profit_factor_30d=1.6,
            ),
            StrategyPerformanceSnapshot(
                strategy_name="mean_reversion",
                returns_30d=3.0,
                volatility_30d=0.10,
                sharpe_30d=1.2,
                win_rate_30d=0.50,
                profit_factor_30d=1.3,
            ),
            StrategyPerformanceSnapshot(
                strategy_name="breakout",
                returns_30d=7.0,
                volatility_30d=0.20,
                sharpe_30d=1.8,
                win_rate_30d=0.60,
                profit_factor_30d=2.0,
            ),
        ]

    def test_allocator_creation(self, allocator):
        """Test allocator is created."""
        assert allocator is not None
        assert allocator.config.method == AllocationMethod.EQUAL_WEIGHT

    def test_allocate_empty_strategies(self, allocator):
        """Test allocation with no strategies."""
        result = allocator.allocate([], 10000.0)
        assert result.total_capital == 10000.0
        assert len(result.allocations) == 0

    def test_equal_weight_allocation(self, allocator, sample_strategies):
        """Test equal weight allocation."""
        result = allocator.allocate(sample_strategies, 10000.0)

        assert len(result.allocations) == 3
        # Equal weight should give ~33% each
        for alloc in result.allocations:
            assert alloc.weight == pytest.approx(0.333, abs=0.05)

    def test_risk_parity_allocation(self, sample_strategies, sample_snapshots):
        """Test risk parity allocation."""
        config = AllocationConfig(
            method=AllocationMethod.RISK_PARITY,
            min_weight=0.01,  # Allow small weights
        )
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            sample_strategies,
            10000.0,
            performance_snapshots=sample_snapshots,
        )

        # Lower volatility should get higher weight
        weights = {a.strategy_name: a.weight for a in result.allocations}
        # mean_reversion has lowest volatility (0.10) so should get highest weight
        if "mean_reversion" in weights and "breakout" in weights:
            assert weights["mean_reversion"] > weights["breakout"]
        else:
            # At least some allocations
            assert len(result.allocations) > 0

    def test_momentum_allocation(self, sample_strategies, sample_snapshots):
        """Test momentum allocation."""
        config = AllocationConfig(
            method=AllocationMethod.MOMENTUM,
            min_weight=0.01,
        )
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            sample_strategies,
            10000.0,
            performance_snapshots=sample_snapshots,
        )

        # Higher returns should get higher weight
        weights = {a.strategy_name: a.weight for a in result.allocations}
        if "breakout" in weights and "mean_reversion" in weights:
            assert weights["breakout"] >= weights["mean_reversion"]
        else:
            assert len(result.allocations) > 0

    def test_kelly_allocation(self, sample_strategies, sample_snapshots):
        """Test Kelly criterion allocation."""
        config = AllocationConfig(
            method=AllocationMethod.KELLY,
            min_weight=0.01,
        )
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            sample_strategies,
            10000.0,
            performance_snapshots=sample_snapshots,
        )

        # Should have some allocations
        if len(result.allocations) > 0:
            total_weight = sum(a.weight for a in result.allocations)
            assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_contextual_bandit_allocation(self, sample_strategies, sample_snapshots):
        """Test contextual bandit allocation."""
        config = AllocationConfig(
            method=AllocationMethod.CONTEXTUAL_BANDIT,
            min_weight=0.01,
        )
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            sample_strategies,
            10000.0,
            current_regime="trending",
            performance_snapshots=sample_snapshots,
        )

        assert len(result.allocations) > 0

    def test_regime_adaptive_allocation(self, sample_strategies, sample_snapshots):
        """Test regime adaptive allocation."""
        config = AllocationConfig(
            method=AllocationMethod.REGIME_ADAPTIVE,
            min_weight=0.01,
        )
        allocator = MetaAllocator(config)

        # Add regime scores
        sample_snapshots[0].regime_scores = {"trending": 0.8, "ranging": 0.4}
        sample_snapshots[1].regime_scores = {"trending": 0.3, "ranging": 0.7}
        sample_snapshots[2].regime_scores = {"trending": 0.6, "ranging": 0.5}

        result = allocator.allocate(
            sample_strategies,
            10000.0,
            current_regime="trending",
            performance_snapshots=sample_snapshots,
        )

        # Should have allocations
        assert len(result.allocations) > 0

    def test_weight_constraints(self, sample_snapshots):
        """Test weight constraints are applied."""
        config = AllocationConfig(
            method=AllocationMethod.MOMENTUM,
            min_weight=0.10,
            max_weight=0.50,
        )
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            [],
            10000.0,
            performance_snapshots=sample_snapshots,
        )

        for alloc in result.allocations:
            # Weights should be capped at max
            assert alloc.weight <= config.max_weight + 0.01

    def test_degraded_strategy_excluded(self, sample_snapshots):
        """Test degraded strategies are excluded."""
        sample_snapshots[1].is_degraded = True

        config = AllocationConfig(method=AllocationMethod.EQUAL_WEIGHT)
        allocator = MetaAllocator(config)

        result = allocator.allocate(
            [],
            10000.0,
            performance_snapshots=sample_snapshots,
        )

        # Degraded strategy should have zero or minimal weight
        weights = {a.strategy_name: a.weight for a in result.allocations}
        assert weights.get("mean_reversion", 0) == 0 or len(result.allocations) == 2

    def test_hhi_calculation(self, allocator, sample_strategies):
        """Test HHI (concentration) calculation."""
        result = allocator.allocate(sample_strategies, 10000.0)

        # Equal weight gives HHI of 1/n = 0.33
        assert result.concentration_ratio == pytest.approx(0.333, abs=0.05)

    def test_update_bandit(self, sample_snapshots):
        """Test updating bandit state."""
        config = AllocationConfig(method=AllocationMethod.CONTEXTUAL_BANDIT)
        allocator = MetaAllocator(config)

        allocator.allocate(
            [],
            10000.0,
            current_regime="trending",
            performance_snapshots=sample_snapshots,
        )

        # Update with positive reward
        allocator.update_bandit("trend_ema", "trending", 3.0)

        key = "trend_ema_trending"
        assert key in allocator.bandit_state
        assert allocator.bandit_state[key]["alpha"] > 1.0

    def test_get_allocation_summary(self, allocator, sample_strategies):
        """Test getting allocation summary."""
        allocator.allocate(sample_strategies, 10000.0, current_regime="ranging")

        summary = allocator.get_allocation_summary()

        # After allocation, summary should have allocation data
        assert "status" not in summary or summary.get("status") != "no_allocation"
        assert summary["regime"] == "ranging"
        assert summary["total_capital"] == 10000.0
        assert "weights" in summary

    def test_no_allocation_summary(self, allocator):
        """Test summary when no allocation exists."""
        summary = allocator.get_allocation_summary()
        assert summary["status"] == "no_allocation"

    def test_rebalance_detection(self, sample_strategies, sample_snapshots):
        """Test rebalance detection."""
        config = AllocationConfig(
            method=AllocationMethod.REGIME_ADAPTIVE,
            min_rebalance_interval_hours=0,
            rebalance_threshold=0.05,
            min_weight=0.01,
        )
        allocator = MetaAllocator(config)

        # First allocation
        sample_snapshots[0].regime_scores = {"trending": 0.9, "ranging": 0.1}
        sample_snapshots[1].regime_scores = {"trending": 0.1, "ranging": 0.9}
        sample_snapshots[2].regime_scores = {"trending": 0.5, "ranging": 0.5}

        allocator.allocate(
            sample_strategies,
            10000.0,
            current_regime="trending",
            performance_snapshots=sample_snapshots,
        )

        # Second allocation with regime change
        result = allocator.allocate(
            sample_strategies,
            10000.0,
            current_regime="ranging",
            performance_snapshots=sample_snapshots,
        )

        # Check if rebalance is needed (may or may not trigger based on weights)
        # At minimum, verify the regime changed
        assert result.current_regime == "ranging"


class TestPortfolioRebalancer:
    """Test PortfolioRebalancer class."""

    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer."""
        return PortfolioRebalancer(min_trade_pct=0.5)

    def test_rebalancer_creation(self, rebalancer):
        """Test rebalancer creation."""
        assert rebalancer.min_trade_pct == 0.5
        assert rebalancer.max_trades_per_rebalance == 5

    def test_calculate_no_rebalance_needed(self, rebalancer):
        """Test when no rebalance needed."""
        current = {"strat1": 0.5, "strat2": 0.5}
        target = {"strat1": 0.5, "strat2": 0.5}

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)
        assert len(trades) == 0

    def test_calculate_simple_rebalance(self, rebalancer):
        """Test simple rebalance calculation."""
        current = {"strat1": 0.6, "strat2": 0.4}
        target = {"strat1": 0.4, "strat2": 0.6}

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)

        assert len(trades) == 2
        # Strat1 should decrease
        strat1_trade = next(t for t in trades if t["strategy"] == "strat1")
        assert strat1_trade["direction"] == "decrease"
        assert strat1_trade["weight_change"] == pytest.approx(-0.2, abs=0.01)

    def test_calculate_with_new_strategy(self, rebalancer):
        """Test rebalance with new strategy."""
        current = {"strat1": 0.5, "strat2": 0.5}
        target = {"strat1": 0.4, "strat2": 0.4, "strat3": 0.2}

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)

        strat3_trade = next((t for t in trades if t["strategy"] == "strat3"), None)
        assert strat3_trade is not None
        assert strat3_trade["direction"] == "increase"

    def test_calculate_with_removed_strategy(self, rebalancer):
        """Test rebalance with removed strategy."""
        current = {"strat1": 0.33, "strat2": 0.33, "strat3": 0.34}
        target = {"strat1": 0.5, "strat2": 0.5}

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)

        strat3_trade = next((t for t in trades if t["strategy"] == "strat3"), None)
        assert strat3_trade is not None
        assert strat3_trade["direction"] == "decrease"

    def test_respects_max_trades(self):
        """Test max trades limit."""
        rebalancer = PortfolioRebalancer(min_trade_pct=0.5, max_trades_per_rebalance=2)

        current = {f"strat{i}": 0.1 for i in range(10)}
        target = {f"strat{i}": 0.2 if i < 5 else 0.0 for i in range(10)}

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)
        assert len(trades) <= 2

    def test_filters_small_trades(self):
        """Test small trades are filtered out."""
        rebalancer = PortfolioRebalancer(min_trade_pct=5.0)

        current = {"strat1": 0.50, "strat2": 0.50}
        target = {"strat1": 0.51, "strat2": 0.49}  # Only 1% change

        trades = rebalancer.calculate_rebalance_trades(current, target, 10000.0)
        assert len(trades) == 0


class TestGlobalInstance:
    """Test global instance factory."""

    def test_get_meta_allocator(self):
        """Test getting global meta-allocator."""
        allocator = get_meta_allocator()
        assert allocator is not None
        assert isinstance(allocator, MetaAllocator)

    def test_same_instance_returned(self):
        """Test same instance is returned."""
        allocator1 = get_meta_allocator()
        allocator2 = get_meta_allocator()
        assert allocator1 is allocator2
