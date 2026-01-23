"""
Tests for strategy optimizer module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from bot.strategy_optimizer import (
    OptimizationResult,
    WalkForwardWindow,
    StrategyOptimizer,
    optimize_strategy,
)
from bot.strategies.base import BaseStrategy, StrategySignal


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, param1: int = 10, param2: float = 0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    @property
    def name(self) -> str:
        return "mock_strategy"

    @property
    def description(self) -> str:
        return "A mock strategy for testing"

    @property
    def suitable_regimes(self) -> list:
        return ["sideways", "bull"]

    def get_required_indicators(self) -> list:
        return ["close", "volume"]

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < 10:
            return StrategySignal(
                decision="FLAT",
                confidence=0.0,
                strategy_name=self.name,
                reason="Not enough data",
            )

        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-10]

        if last_close > prev_close * 1.02:
            return StrategySignal(
                decision="LONG",
                confidence=0.7,
                strategy_name=self.name,
                reason="Price up",
                entry_price=last_close,
                stop_loss=last_close * 0.98,
                take_profit=last_close * 1.04,
            )
        elif last_close < prev_close * 0.98:
            return StrategySignal(
                decision="SHORT",
                confidence=0.7,
                strategy_name=self.name,
                reason="Price down",
                entry_price=last_close,
                stop_loss=last_close * 1.02,
                take_profit=last_close * 0.96,
            )
        return StrategySignal(
            decision="FLAT",
            confidence=0.5,
            strategy_name=self.name,
            reason="Neutral",
        )


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_result_creation(self):
        """Test creating an optimization result."""
        result = OptimizationResult(
            strategy_name="test_strategy",
            best_params={"param1": 10, "param2": 0.5},
            best_score=1.5,
            objective="sharpe",
            in_sample_metrics={"sharpe": 1.6},
            out_of_sample_metrics={"sharpe": 1.5},
        )
        assert result.strategy_name == "test_strategy"
        assert result.best_score == 1.5
        assert result.objective == "sharpe"

    def test_to_dict(self):
        """Test conversion to dict."""
        result = OptimizationResult(
            strategy_name="test",
            best_params={"param1": 10},
            best_score=1.234567,
            objective="sharpe",
            in_sample_metrics={"sharpe": 1.5},
            out_of_sample_metrics={"sharpe": 1.4},
            optimization_time=10.5,
            windows_tested=5,
        )
        d = result.to_dict()

        assert d["strategy_name"] == "test"
        assert d["best_score"] == 1.2346  # Rounded
        assert "best_params" in d
        assert d["optimization_time"] == 10.5
        assert d["windows_tested"] == 5


class TestWalkForwardWindow:
    """Test WalkForwardWindow dataclass."""

    def test_window_creation(self):
        """Test creating a window."""
        window = WalkForwardWindow(
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 6, 30),
            test_start=datetime(2024, 7, 1),
            test_end=datetime(2024, 7, 31),
            window_index=0,
        )
        assert window.window_index == 0
        assert window.train_start == datetime(2024, 1, 1)
        assert window.test_end == datetime(2024, 7, 31)


class TestStrategyOptimizer:
    """Test StrategyOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return StrategyOptimizer(
            train_window_days=90,
            test_window_days=30,
            step_days=30,
            min_trades=5,
        )

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        base_price = 100.0
        returns = np.random.randn(n) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.001),
                "high": prices * (1 + np.abs(np.random.randn(n)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(n)) * 0.01),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    @pytest.fixture
    def short_ohlcv(self):
        """Create short OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        return pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [102] * 50,
                "low": [98] * 50,
                "close": [101] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_optimizer_creation(self, optimizer):
        """Test optimizer is created."""
        assert optimizer is not None
        assert optimizer.train_window_days == 90
        assert optimizer.test_window_days == 30
        assert optimizer.step_days == 30

    def test_objectives_defined(self, optimizer):
        """Test all objectives are defined."""
        assert "sharpe" in optimizer.OBJECTIVES
        assert "sortino" in optimizer.OBJECTIVES
        assert "profit_factor" in optimizer.OBJECTIVES
        assert "win_rate" in optimizer.OBJECTIVES
        assert "total_return" in optimizer.OBJECTIVES

    def test_generate_windows(self, optimizer):
        """Test window generation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)

        windows = optimizer.generate_windows(start, end)

        assert len(windows) > 0
        for window in windows:
            assert window.train_end == window.test_start
            assert window.train_start < window.train_end

    def test_generate_windows_short_period(self, optimizer):
        """Test window generation with short period."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)  # Only 1 month

        windows = optimizer.generate_windows(start, end)
        assert len(windows) == 0

    def test_optimize_invalid_objective(self, optimizer, sample_ohlcv):
        """Test optimize with invalid objective."""
        with pytest.raises(ValueError, match="Unknown objective"):
            optimizer.optimize(
                MockStrategy,
                sample_ohlcv,
                param_grid={"param1": [10]},
                objective="invalid",
            )

    def test_optimize_non_datetime_index(self, optimizer):
        """Test optimize with non-datetime index."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1000],
            }
        )

        with pytest.raises(ValueError, match="DatetimeIndex"):
            optimizer.optimize(
                MockStrategy,
                df,
                param_grid={"param1": [10]},
            )

    def test_optimize_insufficient_data(self, optimizer, short_ohlcv):
        """Test optimize with insufficient data."""
        with pytest.raises(ValueError, match="Not enough data"):
            optimizer.optimize(
                MockStrategy,
                short_ohlcv,
                param_grid={"param1": [10]},
            )

    def test_calculate_metrics_few_trades(self, optimizer):
        """Test metrics with few trades."""
        trades = [
            {"pnl_pct": 0.02},
            {"pnl_pct": -0.01},
        ]
        optimizer.min_trades = 10

        metrics = optimizer._calculate_metrics(trades)
        assert metrics["sharpe"] == 0.0
        assert metrics["num_trades"] == 2

    def test_calculate_metrics_sufficient_trades(self, optimizer):
        """Test metrics with sufficient trades."""
        optimizer.min_trades = 5
        trades = [
            {"pnl_pct": 0.02},
            {"pnl_pct": 0.03},
            {"pnl_pct": -0.01},
            {"pnl_pct": 0.015},
            {"pnl_pct": -0.005},
            {"pnl_pct": 0.01},
        ]

        metrics = optimizer._calculate_metrics(trades)

        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "profit_factor" in metrics
        assert "win_rate" in metrics
        assert "total_return" in metrics
        assert metrics["num_trades"] == 6

    def test_calculate_metrics_all_wins(self, optimizer):
        """Test metrics with all winning trades."""
        optimizer.min_trades = 3
        trades = [
            {"pnl_pct": 0.02},
            {"pnl_pct": 0.03},
            {"pnl_pct": 0.01},
        ]

        metrics = optimizer._calculate_metrics(trades)
        assert metrics["win_rate"] == 1.0

    def test_calculate_metrics_all_losses(self, optimizer):
        """Test metrics with all losing trades."""
        optimizer.min_trades = 3
        trades = [
            {"pnl_pct": -0.02},
            {"pnl_pct": -0.03},
            {"pnl_pct": -0.01},
        ]

        metrics = optimizer._calculate_metrics(trades)
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_get_objective_value(self, optimizer):
        """Test getting objective value."""
        metrics = {"sharpe": 1.5, "sortino": 2.0, "profit_factor": 2.5}

        assert optimizer._get_objective_value(metrics, "sharpe") == 1.5
        assert optimizer._get_objective_value(metrics, "sortino") == 2.0
        assert optimizer._get_objective_value(metrics, "missing") == 0.0

    def test_create_strategy(self, optimizer):
        """Test strategy creation."""
        strategy = optimizer._create_strategy(MockStrategy, {"param1": 20, "param2": 0.8})
        assert strategy is not None
        assert strategy.param1 == 20
        assert strategy.param2 == 0.8

    def test_create_strategy_fallback(self, optimizer):
        """Test strategy creation with fallback."""

        class NoParamStrategy(BaseStrategy):
            @property
            def name(self):
                return "no_param"

            @property
            def description(self):
                return "test"

            @property
            def suitable_regimes(self):
                return []

            def get_required_indicators(self):
                return []

            def add_indicators(self, df):
                return df

            def generate_signal(self, df):
                return StrategySignal(
                    decision="FLAT", confidence=0, strategy_name="test", reason="test"
                )

        strategy = optimizer._create_strategy(
            NoParamStrategy,
            {"nonexistent_param": 123},  # Strategy doesn't accept this
        )
        assert strategy is not None


class TestBacktest:
    """Test backtest functionality."""

    @pytest.fixture
    def optimizer(self):
        return StrategyOptimizer(min_trades=2)

    @pytest.fixture
    def trending_ohlcv(self):
        """Create trending OHLCV data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        # Uptrend
        prices = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 2

        return pd.DataFrame(
            {
                "open": prices - 0.5,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_backtest_generates_trades(self, optimizer, trending_ohlcv):
        """Test backtest generates trades."""
        strategy = MockStrategy()
        trades = optimizer._backtest(strategy, trending_ohlcv)

        # Should have some trades in trending data
        assert isinstance(trades, list)

    def test_backtest_short_data(self, optimizer):
        """Test backtest with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        short_df = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [102] * 50,
                "low": [98] * 50,
                "close": [101] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        strategy = MockStrategy()
        trades = optimizer._backtest(strategy, short_df)
        assert trades == []


class TestConvenienceFunction:
    """Test convenience function."""

    def test_optimize_strategy_function(self):
        """Test optimize_strategy convenience function."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

        ohlcv = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

        result = optimize_strategy(
            MockStrategy,
            ohlcv,
            param_grid={"param1": [5, 10]},
            objective="sharpe",
        )

        assert isinstance(result, OptimizationResult)
        assert result.strategy_name == "MockStrategy"
