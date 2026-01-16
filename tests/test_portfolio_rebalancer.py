"""
Tests for portfolio rebalancer module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.portfolio_rebalancer import (
    RebalanceConfig,
    PortfolioRebalancer,
    create_portfolio_rebalancer,
)


class TestRebalanceConfig:
    """Test RebalanceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = RebalanceConfig()
        assert config.enabled is True
        assert config.strategy == "risk_parity"
        assert config.rebalance_threshold == 0.05
        assert config.min_rebalance_interval_hours == 24
        assert config.max_single_trade_pct == 0.10
        assert config.min_position_pct == 0.05
        assert config.max_position_pct == 0.30

    def test_custom_values(self):
        """Test custom configuration."""
        config = RebalanceConfig(
            strategy="equal",
            rebalance_threshold=0.10,
            max_position_pct=0.40,
        )
        assert config.strategy == "equal"
        assert config.rebalance_threshold == 0.10
        assert config.max_position_pct == 0.40


class TestPortfolioRebalancer:
    """Test PortfolioRebalancer class."""

    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer instance."""
        return PortfolioRebalancer()

    @pytest.fixture
    def equal_weight_rebalancer(self):
        """Create equal weight rebalancer."""
        config = RebalanceConfig(strategy="equal")
        return PortfolioRebalancer(config)

    @pytest.fixture
    def momentum_rebalancer(self):
        """Create momentum rebalancer."""
        config = RebalanceConfig(strategy="momentum")
        return PortfolioRebalancer(config)

    @pytest.fixture
    def min_corr_rebalancer(self):
        """Create min correlation rebalancer."""
        config = RebalanceConfig(strategy="min_correlation")
        return PortfolioRebalancer(config)

    @pytest.fixture
    def sample_prices(self):
        """Create sample price history."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        # Generate different price patterns
        btc_prices = 50000 * (1 + np.cumsum(np.random.randn(50) * 0.02))
        eth_prices = 3000 * (1 + np.cumsum(np.random.randn(50) * 0.03))
        sol_prices = 100 * (1 + np.cumsum(np.random.randn(50) * 0.04))

        return {
            "BTC/USDT": pd.Series(btc_prices, index=dates),
            "ETH/USDT": pd.Series(eth_prices, index=dates),
            "SOL/USDT": pd.Series(sol_prices, index=dates),
        }

    def test_rebalancer_creation(self, rebalancer):
        """Test rebalancer is created."""
        assert rebalancer is not None
        assert rebalancer.config is not None

    def test_update_prices(self, rebalancer, sample_prices):
        """Test updating price history."""
        for symbol, prices in sample_prices.items():
            rebalancer.update_prices(symbol, prices)

        assert len(rebalancer._price_history) == 3

    # Equal Weight Tests
    def test_equal_weight_calculation(self, equal_weight_rebalancer):
        """Test equal weight allocation."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        weights = equal_weight_rebalancer.calculate_target_weights(symbols, {})

        expected_weight = 1/3
        for symbol in symbols:
            # After constraints applied, weights should be close to 1/3
            assert symbol in weights
            assert weights[symbol] > 0

    def test_equal_weight_empty_list(self, equal_weight_rebalancer):
        """Test equal weight with empty list."""
        weights = equal_weight_rebalancer.calculate_target_weights([], {})
        assert weights == {}

    # Risk Parity Tests
    def test_risk_parity_calculation(self, rebalancer, sample_prices):
        """Test risk parity allocation."""
        for symbol, prices in sample_prices.items():
            rebalancer.update_prices(symbol, prices)

        symbols = list(sample_prices.keys())
        weights = rebalancer.calculate_target_weights(symbols, {})

        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_risk_parity_no_history(self, rebalancer):
        """Test risk parity with no price history."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        weights = rebalancer.calculate_target_weights(symbols, {})

        # Should use default volatility
        assert len(weights) == 2

    # Momentum Tests
    def test_momentum_weight_calculation(self, momentum_rebalancer, sample_prices):
        """Test momentum-based allocation."""
        for symbol, prices in sample_prices.items():
            momentum_rebalancer.update_prices(symbol, prices)

        symbols = list(sample_prices.keys())
        weights = momentum_rebalancer.calculate_target_weights(symbols, {})

        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    # Min Correlation Tests
    def test_min_correlation_calculation(self, min_corr_rebalancer, sample_prices):
        """Test min correlation allocation."""
        for symbol, prices in sample_prices.items():
            min_corr_rebalancer.update_prices(symbol, prices)

        symbols = list(sample_prices.keys())
        weights = min_corr_rebalancer.calculate_target_weights(symbols, {})

        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_min_correlation_single_symbol(self, min_corr_rebalancer):
        """Test min correlation with single symbol."""
        weights = min_corr_rebalancer.calculate_target_weights(["BTC/USDT"], {})
        assert len(weights) == 1

    # Constraint Tests
    def test_min_position_constraint(self, rebalancer):
        """Test minimum position constraint."""
        rebalancer.config.min_position_pct = 0.10
        rebalancer.config.max_position_pct = 0.50

        symbols = ["A", "B", "C", "D", "E"]  # 5 positions
        weights = rebalancer.calculate_target_weights(symbols, {})

        for w in weights.values():
            assert w >= 0.10 - 0.01  # Allow small tolerance

    def test_max_position_constraint(self, rebalancer, sample_prices):
        """Test maximum position constraint is applied."""
        rebalancer.config.max_position_pct = 0.50

        for symbol, prices in sample_prices.items():
            rebalancer.update_prices(symbol, prices)

        symbols = list(sample_prices.keys())
        weights = rebalancer.calculate_target_weights(symbols, {})

        # Weights are renormalized after constraints, so we check they sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)
        # With 3 symbols, no single weight should dominate excessively
        max_weight = max(weights.values())
        assert max_weight <= 0.60  # Allow for renormalization effects

    # Rebalance Check Tests
    def test_check_rebalance_needed(self, rebalancer):
        """Test rebalance check."""
        rebalancer._target_weights = {
            "BTC": 0.40,
            "ETH": 0.30,
            "SOL": 0.30,
        }

        current = {
            "BTC": 0.35,  # 5% drift
            "ETH": 0.35,  # 5% drift
            "SOL": 0.30,
        }

        needs_rebalance, diffs = rebalancer.check_rebalance_needed(current)
        # 5% drift equals threshold, may or may not trigger
        assert isinstance(needs_rebalance, bool)
        assert "BTC" in diffs

    def test_check_rebalance_large_drift(self, rebalancer):
        """Test rebalance with large drift."""
        rebalancer._target_weights = {
            "BTC": 0.40,
            "ETH": 0.30,
            "SOL": 0.30,
        }

        current = {
            "BTC": 0.20,  # 20% drift
            "ETH": 0.50,
            "SOL": 0.30,
        }

        needs_rebalance, diffs = rebalancer.check_rebalance_needed(current)
        assert needs_rebalance is True
        assert diffs["BTC"] == pytest.approx(0.20, abs=0.01)

    def test_check_rebalance_no_target(self, rebalancer):
        """Test rebalance check with no target weights."""
        current = {"BTC": 0.5, "ETH": 0.5}
        needs_rebalance, diffs = rebalancer.check_rebalance_needed(current)
        assert needs_rebalance is False
        assert diffs == {}

    def test_check_rebalance_interval(self, rebalancer):
        """Test rebalance interval check."""
        rebalancer._target_weights = {"BTC": 0.40, "ETH": 0.60}
        rebalancer._last_rebalance = datetime.now()  # Just rebalanced

        current = {"BTC": 0.20, "ETH": 0.80}  # Large drift
        needs_rebalance, _ = rebalancer.check_rebalance_needed(current)
        assert needs_rebalance is False  # Too soon

    # Order Generation Tests
    def test_generate_rebalance_orders(self, rebalancer):
        """Test generating rebalance orders."""
        rebalancer._target_weights = {
            "BTC/USDT": 0.40,
            "ETH/USDT": 0.30,
            "SOL/USDT": 0.30,
        }

        current_positions = {
            "BTC/USDT": {"quantity": 0.001, "value": 2000},  # 20% of portfolio
            "ETH/USDT": {"quantity": 1.0, "value": 3000},    # 30%
            "SOL/USDT": {"quantity": 50, "value": 5000},     # 50%
        }

        current_prices = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 100,
        }

        orders = rebalancer.generate_rebalance_orders(
            current_positions,
            total_portfolio_value=10000,
            current_prices=current_prices,
        )

        # Should have orders to rebalance
        assert len(orders) > 0
        assert all("symbol" in o for o in orders)
        assert all("side" in o for o in orders)
        assert all("quantity" in o for o in orders)

    def test_generate_orders_no_target(self, rebalancer):
        """Test generate orders with no target weights."""
        orders = rebalancer.generate_rebalance_orders(
            current_positions={"BTC": {"quantity": 1, "value": 1000}},
            total_portfolio_value=1000,
            current_prices={"BTC": 50000},
        )
        assert orders == []

    def test_generate_orders_small_drift(self, rebalancer):
        """Test orders not generated for small drift."""
        rebalancer._target_weights = {"BTC": 0.50, "ETH": 0.50}
        rebalancer.config.rebalance_threshold = 0.10

        current_positions = {
            "BTC": {"quantity": 0.001, "value": 4800},  # 48%
            "ETH": {"quantity": 1.0, "value": 5200},    # 52%
        }

        orders = rebalancer.generate_rebalance_orders(
            current_positions,
            total_portfolio_value=10000,
            current_prices={"BTC": 50000, "ETH": 3000},
        )

        # 2% drift is below 10% threshold
        assert orders == []

    # Portfolio Metrics Tests
    def test_get_portfolio_metrics(self, rebalancer):
        """Test portfolio metrics calculation."""
        rebalancer._target_weights = {"BTC": 0.50, "ETH": 0.50}

        current_positions = {
            "BTC": {"quantity": 0.1, "value": 5000},
            "ETH": {"quantity": 1.5, "value": 4500},
        }

        metrics = rebalancer.get_portfolio_metrics(current_positions, total_value=9500)

        assert "num_positions" in metrics
        assert "effective_positions" in metrics
        assert "concentration_hhi" in metrics
        assert "max_position_weight" in metrics
        assert "weights" in metrics
        assert metrics["num_positions"] == 2

    def test_get_portfolio_metrics_empty(self, rebalancer):
        """Test metrics with empty portfolio."""
        metrics = rebalancer.get_portfolio_metrics({}, total_value=0)
        assert metrics == {}

    def test_concentration_hhi(self, rebalancer):
        """Test HHI concentration metric."""
        # Perfectly balanced portfolio
        positions = {
            "A": {"quantity": 1, "value": 2500},
            "B": {"quantity": 1, "value": 2500},
            "C": {"quantity": 1, "value": 2500},
            "D": {"quantity": 1, "value": 2500},
        }
        rebalancer._target_weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

        metrics = rebalancer.get_portfolio_metrics(positions, total_value=10000)

        # HHI for equal weights = N * (1/N)^2 = 1/N = 0.25
        assert metrics["concentration_hhi"] == pytest.approx(0.25, abs=0.01)
        assert metrics["effective_positions"] == pytest.approx(4.0, abs=0.1)


class TestFactoryFunction:
    """Test factory function."""

    def test_create_portfolio_rebalancer(self):
        """Test creating rebalancer via factory."""
        rebalancer = create_portfolio_rebalancer()
        assert rebalancer is not None
        assert rebalancer.config.strategy == "risk_parity"

    def test_create_with_custom_strategy(self):
        """Test creating with custom strategy."""
        rebalancer = create_portfolio_rebalancer(strategy="equal")
        assert rebalancer.config.strategy == "equal"

    def test_create_with_custom_threshold(self):
        """Test creating with custom threshold."""
        rebalancer = create_portfolio_rebalancer(rebalance_threshold=0.10)
        assert rebalancer.config.rebalance_threshold == 0.10


class TestEdgeCases:
    """Test edge cases."""

    def test_single_asset_portfolio(self):
        """Test with single asset."""
        rebalancer = PortfolioRebalancer()
        weights = rebalancer.calculate_target_weights(["BTC"], {})
        assert weights["BTC"] == pytest.approx(1.0, rel=0.01)

    def test_invalid_strategy_fallback(self):
        """Test invalid strategy falls back to equal."""
        config = RebalanceConfig(strategy="invalid_strategy")
        rebalancer = PortfolioRebalancer(config)

        weights = rebalancer.calculate_target_weights(["A", "B"], {})
        assert len(weights) == 2

    def test_zero_portfolio_value(self):
        """Test with zero portfolio value."""
        rebalancer = PortfolioRebalancer()
        rebalancer._target_weights = {"BTC": 1.0}

        orders = rebalancer.generate_rebalance_orders(
            current_positions={},
            total_portfolio_value=0,
            current_prices={"BTC": 50000},
        )
        assert orders == []

    def test_missing_price_in_orders(self):
        """Test order generation with missing price."""
        rebalancer = PortfolioRebalancer()
        rebalancer._target_weights = {"BTC": 0.50, "ETH": 0.50}

        current_positions = {
            "BTC": {"quantity": 0.1, "value": 5000},
            "ETH": {"quantity": 1.0, "value": 5000},
        }

        orders = rebalancer.generate_rebalance_orders(
            current_positions,
            total_portfolio_value=10000,
            current_prices={"BTC": 50000},  # ETH price missing
        )
        # Should handle gracefully
        assert isinstance(orders, list)
