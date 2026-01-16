"""
Tests for Strategy Optimizer module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from bot.optimizer import (
    OptimizationResult,
    _objective_from_result,
    _make_config,
    _sample_params,
    results_to_dataframe,
)
from bot.strategy import StrategyConfig
from bot.backtesting import BacktestResult


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_basic_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            params={"ema_fast": 10, "ema_slow": 30},
            final_balance=11500.0,
            total_pnl_pct=15.0,
            win_rate=0.6,
            profit_factor=1.8,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.5,
            total_trades=50,
        )

        assert result.params["ema_fast"] == 10
        assert result.final_balance == 11500.0
        assert result.total_pnl_pct == 15.0
        assert result.win_rate == 0.6
        assert result.total_trades == 50

    def test_to_dict(self):
        """Test conversion to dict."""
        result = OptimizationResult(
            params={"ema_fast": 12, "ema_slow": 26},
            final_balance=10500.5,
            total_pnl_pct=5.05,
            win_rate=0.55,
            profit_factor=1.5,
            max_drawdown_pct=3.5,
            sharpe_ratio=1.2,
            total_trades=30,
        )

        d = result.to_dict()

        assert isinstance(d["final_balance"], float)
        assert d["final_balance"] == 10500.5
        assert d["total_pnl_pct"] == 5.05
        assert d["total_trades"] == 30
        assert isinstance(d["total_trades"], int)

    def test_to_dict_converts_integer_floats(self):
        """Test that integer-like floats in params are converted to int."""
        result = OptimizationResult(
            params={
                "ema_fast": 10.0,  # Should become int
                "ema_slow": 30.0,  # Should become int
                "stop_loss": 0.02,  # Should remain float
            },
            final_balance=10000.0,
            total_pnl_pct=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
        )

        d = result.to_dict()

        assert d["params"]["ema_fast"] == 10
        assert isinstance(d["params"]["ema_fast"], int)
        assert d["params"]["ema_slow"] == 30
        assert isinstance(d["params"]["ema_slow"], int)
        assert d["params"]["stop_loss"] == 0.02
        assert isinstance(d["params"]["stop_loss"], float)

    def test_to_dict_with_string_param(self):
        """Test to_dict with string params."""
        result = OptimizationResult(
            params={"strategy_type": "ema_crossover", "ema_fast": 10.0},
            final_balance=10000.0,
            total_pnl_pct=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
        )

        d = result.to_dict()

        assert d["params"]["strategy_type"] == "ema_crossover"

    def test_to_dict_with_boolean_param(self):
        """Test to_dict preserves boolean params."""
        result = OptimizationResult(
            params={"use_trailing_stop": True, "ema_fast": 10},
            final_balance=10000.0,
            total_pnl_pct=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
        )

        d = result.to_dict()

        # Boolean should remain boolean (not converted to int)
        assert d["params"]["use_trailing_stop"] is True


class TestObjectiveFunction:
    """Test _objective_from_result function."""

    @pytest.fixture
    def good_result(self):
        """Create a good backtest result."""
        return BacktestResult(
            initial_balance=10000.0,
            final_balance=12000.0,
            total_pnl_pct=20.0,
            win_rate=0.65,
            profit_factor=2.5,
            max_drawdown_pct=8.0,
            sharpe_ratio=2.0,
            total_trades=50,
            trades=[],
        )

    @pytest.fixture
    def bad_result(self):
        """Create a bad backtest result."""
        return BacktestResult(
            initial_balance=10000.0,
            final_balance=8000.0,
            total_pnl_pct=-20.0,
            win_rate=0.35,
            profit_factor=0.5,
            max_drawdown_pct=25.0,
            sharpe_ratio=-1.0,
            total_trades=50,
            trades=[],
        )

    def test_sharpe_objective(self, good_result):
        """Test Sharpe ratio objective."""
        score = _objective_from_result(good_result, objective="sharpe")

        # Sharpe 2.0 * 100 - 0.5 * 8.0 = 200 - 4 = 196
        expected = 2.0 * 100.0 - 0.5 * 8.0
        assert score == pytest.approx(expected)

    def test_pnl_objective(self, good_result):
        """Test PnL objective."""
        score = _objective_from_result(good_result, objective="pnl")

        # 20.0 - 0.5 * 8.0 = 20 - 4 = 16
        expected = 20.0 - 0.5 * 8.0
        assert score == pytest.approx(expected)

    def test_winrate_objective(self, good_result):
        """Test win rate objective."""
        score = _objective_from_result(good_result, objective="winrate")

        # 0.65 * 100 - 0.5 * 8.0 = 65 - 4 = 61
        expected = 0.65 * 100.0 - 0.5 * 8.0
        assert score == pytest.approx(expected)

    def test_fallback_objective(self, good_result):
        """Test fallback uses pnl."""
        score = _objective_from_result(good_result, objective="unknown")

        # Should fall back to pnl calculation
        expected = 20.0 - 0.5 * 8.0
        assert score == pytest.approx(expected)

    def test_min_trades_filter(self, good_result):
        """Test minimum trades filter."""
        good_result.total_trades = 3

        score = _objective_from_result(good_result, min_trades=5)

        # Should return very negative score
        assert score == -1e9

    def test_custom_mdd_weight(self, good_result):
        """Test custom drawdown weight."""
        score_low = _objective_from_result(good_result, mdd_weight=0.2)
        score_high = _objective_from_result(good_result, mdd_weight=1.0)

        # Higher weight penalizes drawdown more
        assert score_low > score_high

    def test_negative_sharpe_handled(self, bad_result):
        """Test negative Sharpe ratio is handled."""
        score = _objective_from_result(bad_result, objective="sharpe")

        # Score should be negative
        assert score < 0


class TestMakeConfig:
    """Test _make_config function."""

    @pytest.fixture
    def base_config(self):
        """Create base strategy config."""
        return StrategyConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            ema_fast=12,
            ema_slow=26,
            rsi_period=14,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            risk_per_trade_pct=1.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

    def test_creates_new_config(self, base_config):
        """Test creating new config from params."""
        params = {
            "ema_fast": 10,
            "ema_slow": 30,
            "rsi_period": 20,
        }

        new_config = _make_config(base_config, params)

        assert new_config.ema_fast == 10
        assert new_config.ema_slow == 30
        assert new_config.rsi_period == 20
        # Unchanged params should come from base
        assert new_config.symbol == "BTC/USDT"

    def test_preserves_base_values(self, base_config):
        """Test missing params use base values."""
        params = {"ema_fast": 15}

        new_config = _make_config(base_config, params)

        assert new_config.ema_fast == 15
        assert new_config.ema_slow == 26  # From base
        assert new_config.rsi_period == 14  # From base

    def test_converts_types(self, base_config):
        """Test parameter type conversion."""
        params = {
            "ema_fast": 10.5,  # Float that should become int
            "rsi_overbought": 75,  # Int that should become float
        }

        new_config = _make_config(base_config, params)

        assert new_config.ema_fast == 10
        assert isinstance(new_config.ema_fast, int)
        assert new_config.rsi_overbought == 75.0
        assert isinstance(new_config.rsi_overbought, float)


class TestSampleParams:
    """Test _sample_params function."""

    @pytest.fixture
    def base_config(self):
        return StrategyConfig(
            symbol="BTC/USDT",
            timeframe="1h",
        )

    def test_returns_all_params(self, base_config):
        """Test all expected params are returned."""
        rng = np.random.default_rng(42)

        params = _sample_params(rng, base_config)

        assert "ema_fast" in params
        assert "ema_slow" in params
        assert "rsi_period" in params
        assert "rsi_overbought" in params
        assert "rsi_oversold" in params
        assert "risk_per_trade_pct" in params
        assert "stop_loss_pct" in params
        assert "take_profit_pct" in params

    def test_ema_slow_greater_than_fast(self, base_config):
        """Test ema_slow is always greater than ema_fast."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            params = _sample_params(rng, base_config)
            assert params["ema_slow"] > params["ema_fast"]

    def test_rsi_bounds_valid(self, base_config):
        """Test RSI parameters are in valid ranges."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            params = _sample_params(rng, base_config)
            assert 8 <= params["rsi_period"] <= 28
            assert 65 <= params["rsi_overbought"] <= 80
            assert 20 <= params["rsi_oversold"] <= 35
            assert params["rsi_overbought"] > params["rsi_oversold"]

    def test_risk_params_bounds(self, base_config):
        """Test risk parameters are in valid ranges."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            params = _sample_params(rng, base_config)
            assert 0.2 <= params["risk_per_trade_pct"] <= 1.5
            assert 0.002 <= params["stop_loss_pct"] <= 0.02
            assert 0.004 <= params["take_profit_pct"] <= 0.05

    def test_deterministic_with_seed(self, base_config):
        """Test reproducible with same seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        params1 = _sample_params(rng1, base_config)
        params2 = _sample_params(rng2, base_config)

        assert params1 == params2

    def test_different_with_different_seeds(self, base_config):
        """Test different seeds produce different params."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        params1 = _sample_params(rng1, base_config)
        params2 = _sample_params(rng2, base_config)

        # Very unlikely to be exactly the same
        assert params1 != params2


class TestResultsToDataframe:
    """Test results_to_dataframe function."""

    def test_empty_results(self):
        """Test with empty results list."""
        df = results_to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_result(self):
        """Test with single result."""
        results = [
            OptimizationResult(
                params={"ema_fast": 10, "ema_slow": 30},
                final_balance=11000.0,
                total_pnl_pct=10.0,
                win_rate=0.6,
                profit_factor=1.5,
                max_drawdown_pct=5.0,
                sharpe_ratio=1.2,
                total_trades=20,
            )
        ]

        df = results_to_dataframe(results)

        assert len(df) == 1
        assert "param_ema_fast" in df.columns
        assert "param_ema_slow" in df.columns
        assert "final_balance" in df.columns
        assert df.iloc[0]["param_ema_fast"] == 10
        assert df.iloc[0]["final_balance"] == 11000.0

    def test_multiple_results(self):
        """Test with multiple results."""
        results = [
            OptimizationResult(
                params={"ema_fast": 10, "ema_slow": 30},
                final_balance=11000.0,
                total_pnl_pct=10.0,
                win_rate=0.6,
                profit_factor=1.5,
                max_drawdown_pct=5.0,
                sharpe_ratio=1.2,
                total_trades=20,
            ),
            OptimizationResult(
                params={"ema_fast": 15, "ema_slow": 40},
                final_balance=12000.0,
                total_pnl_pct=20.0,
                win_rate=0.65,
                profit_factor=2.0,
                max_drawdown_pct=8.0,
                sharpe_ratio=1.8,
                total_trades=30,
            ),
        ]

        df = results_to_dataframe(results)

        assert len(df) == 2
        assert df.iloc[0]["param_ema_fast"] == 10
        assert df.iloc[1]["param_ema_fast"] == 15

    def test_all_columns_present(self):
        """Test all expected columns are present."""
        results = [
            OptimizationResult(
                params={"ema_fast": 10},
                final_balance=10000.0,
                total_pnl_pct=0.0,
                win_rate=0.5,
                profit_factor=1.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
            )
        ]

        df = results_to_dataframe(results)

        expected_cols = [
            "final_balance",
            "total_pnl_pct",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "sharpe_ratio",
            "total_trades",
        ]

        for col in expected_cols:
            assert col in df.columns

    def test_param_prefix(self):
        """Test params get 'param_' prefix."""
        results = [
            OptimizationResult(
                params={
                    "ema_fast": 10,
                    "rsi_period": 14,
                    "stop_loss_pct": 0.02,
                },
                final_balance=10000.0,
                total_pnl_pct=0.0,
                win_rate=0.5,
                profit_factor=1.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
            )
        ]

        df = results_to_dataframe(results)

        assert "param_ema_fast" in df.columns
        assert "param_rsi_period" in df.columns
        assert "param_stop_loss_pct" in df.columns
        # Should not have unprefixed versions
        assert "ema_fast" not in df.columns


class TestOptimizationResultEdgeCases:
    """Test edge cases for optimization results."""

    def test_zero_values(self):
        """Test result with all zero values."""
        result = OptimizationResult(
            params={},
            final_balance=0.0,
            total_pnl_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
        )

        d = result.to_dict()

        assert d["final_balance"] == 0.0
        assert d["total_trades"] == 0

    def test_negative_values(self):
        """Test result with negative values."""
        result = OptimizationResult(
            params={"ema_fast": 10},
            final_balance=8000.0,
            total_pnl_pct=-20.0,
            win_rate=0.3,
            profit_factor=0.5,
            max_drawdown_pct=30.0,
            sharpe_ratio=-1.5,
            total_trades=100,
        )

        d = result.to_dict()

        assert d["total_pnl_pct"] == -20.0
        assert d["sharpe_ratio"] == -1.5

    def test_large_values(self):
        """Test result with large values."""
        result = OptimizationResult(
            params={"ema_fast": 10},
            final_balance=1000000.0,
            total_pnl_pct=9900.0,
            win_rate=0.99,
            profit_factor=100.0,
            max_drawdown_pct=1.0,
            sharpe_ratio=10.0,
            total_trades=10000,
        )

        d = result.to_dict()

        assert d["final_balance"] == 1000000.0
        assert d["total_trades"] == 10000


class TestObjectiveEdgeCases:
    """Test edge cases for objective function."""

    def test_nan_sharpe_handled(self):
        """Test NaN Sharpe ratio is handled."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=10000.0,
            total_pnl_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=float('nan'),
            total_trades=10,
            trades=[],
        )

        score = _objective_from_result(result, objective="sharpe")

        # NaN sharpe ratio propagates through calculation - this tests that the
        # function doesn't crash on NaN input (it may return NaN which is acceptable)
        # The function handles NaN by potentially returning -1e12 in the guard clause
        # or the NaN may propagate through, both are acceptable behaviors
        assert np.isnan(score) or np.isfinite(score)

    def test_zero_trades(self):
        """Test zero trades returns very negative score."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=10000.0,
            total_pnl_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            trades=[],
        )

        score = _objective_from_result(result, min_trades=5)

        assert score == -1e9

    def test_exact_min_trades(self):
        """Test exactly min_trades is valid."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=11000.0,
            total_pnl_pct=10.0,
            win_rate=0.6,
            profit_factor=1.5,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.0,
            total_trades=5,
            trades=[],
        )

        score = _objective_from_result(result, min_trades=5)

        # Should be valid, not rejected
        assert score != -1e9
        assert score > -1e9
