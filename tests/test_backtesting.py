"""
Tests for backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import sys

from bot.backtesting import (
    Trade,
    BacktestResult,
    Backtester,
    save_backtest_results,
)
from bot.strategy import StrategyConfig


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="LONG",
            entry_price=100.0,
            size=1.0,
        )
        assert trade.direction == "LONG"
        assert trade.entry_price == 100.0
        assert trade.size == 1.0
        assert trade.exit_time is None
        assert trade.pnl == 0.0

    def test_trade_with_exit(self):
        """Test trade with exit."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            direction="LONG",
            entry_price=100.0,
            exit_price=110.0,
            size=1.0,
            pnl=10.0,
            pnl_pct=10.0,
            exit_reason="Take Profit",
        )
        assert trade.exit_price == 110.0
        assert trade.pnl == 10.0
        assert trade.exit_reason == "Take Profit"

    def test_trade_to_dict(self):
        """Test trade to dict conversion."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            direction="SHORT",
            entry_price=100.0,
            exit_price=95.0,
            size=2.0,
            pnl=10.0,
            pnl_pct=5.0,
            stop_loss=105.0,
            take_profit=90.0,
            exit_reason="Take Profit",
            confidence=0.75,
        )
        d = trade.to_dict()

        assert d["direction"] == "SHORT"
        assert d["entry_price"] == 100.0
        assert d["exit_price"] == 95.0
        assert d["pnl"] == 10.0
        assert d["pnl_pct"] == 5.0
        assert d["confidence"] == 0.75

    def test_trade_to_dict_no_exit(self):
        """Test trade to dict with no exit."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="LONG",
            entry_price=100.0,
            size=1.0,
        )
        d = trade.to_dict()
        assert d["exit_time"] is None
        assert d["exit_price"] is None


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_result_creation(self):
        """Test creating a backtest result."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=12000.0,
        )
        assert result.initial_balance == 10000.0
        assert result.final_balance == 12000.0
        assert result.total_trades == 0
        assert result.trades == []

    def test_result_with_trades(self):
        """Test result with trades."""
        trade1 = Trade(
            entry_time=datetime(2024, 1, 1),
            direction="LONG",
            entry_price=100.0,
            size=1.0,
            pnl=10.0,
        )
        trade2 = Trade(
            entry_time=datetime(2024, 1, 2),
            direction="SHORT",
            entry_price=110.0,
            size=1.0,
            pnl=-5.0,
        )

        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=10005.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            total_pnl=5.0,
            total_pnl_pct=0.05,
            win_rate=0.5,
            trades=[trade1, trade2],
        )

        assert result.total_trades == 2
        assert result.win_rate == 0.5
        assert len(result.trades) == 2

    def test_result_to_dict(self):
        """Test result to dict conversion."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=11500.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=1500.0,
            total_pnl_pct=15.0,
            win_rate=0.6,
            avg_win=350.0,
            avg_loss=200.0,
            profit_factor=2.1,
            max_drawdown=500.0,
            max_drawdown_pct=4.5,
            sharpe_ratio=1.8,
        )

        d = result.to_dict()

        assert d["initial_balance"] == 10000.0
        assert d["final_balance"] == 11500.0
        assert d["total_trades"] == 10
        assert d["win_rate"] == 0.6
        assert d["profit_factor"] == 2.1

    def test_print_summary(self, capsys):
        """Test print summary output."""
        result = BacktestResult(
            initial_balance=10000.0,
            final_balance=12000.0,
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            total_pnl=2000.0,
            total_pnl_pct=20.0,
            win_rate=0.6,
            avg_win=800.0,
            avg_loss=100.0,
            profit_factor=8.0,
            max_drawdown=300.0,
            max_drawdown_pct=2.5,
            sharpe_ratio=1.5,
        )

        result.print_summary()
        captured = capsys.readouterr()

        assert "BACKTEST RESULTS" in captured.out
        assert "$10,000.00" in captured.out
        assert "$12,000.00" in captured.out


class TestBacktester:
    """Test Backtester class."""

    @pytest.fixture
    def config(self):
        """Create strategy config."""
        return StrategyConfig()

    @pytest.fixture
    def backtester(self, config):
        """Create backtester instance."""
        return Backtester(config, initial_balance=10000.0)

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")

        base_price = 100.0
        returns = np.random.randn(n) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.001),
                "high": prices * (1 + np.abs(np.random.randn(n)) * 0.005),
                "low": prices * (1 - np.abs(np.random.randn(n)) * 0.005),
                "close": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

    def test_backtester_creation(self, backtester):
        """Test backtester is created."""
        assert backtester is not None
        assert backtester.initial_balance == 10000.0
        assert backtester.balance == 10000.0
        assert backtester.position is None
        assert backtester.trades == []

    def test_backtester_run(self, backtester, sample_ohlcv):
        """Test running backtest."""
        result = backtester.run(sample_ohlcv)

        assert isinstance(result, BacktestResult)
        assert result.initial_balance == 10000.0
        assert result.final_balance is not None

    def test_backtester_equity_curve(self, backtester, sample_ohlcv):
        """Test equity curve is generated."""
        result = backtester.run(sample_ohlcv)

        assert len(result.equity_curve) > 0
        for point in result.equity_curve:
            assert "timestamp" in point
            assert "balance" in point
            assert "price" in point


class TestBacktesterInternals:
    """Test backtester internal methods."""

    @pytest.fixture
    def config(self):
        return StrategyConfig(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

    @pytest.fixture
    def backtester(self, config):
        return Backtester(config, initial_balance=10000.0)

    def test_open_long_position(self, backtester):
        """Test opening long position."""
        bar = pd.Series(
            {
                "close": 100.0,
                "high": 101.0,
                "low": 99.0,
            }
        )
        bar.name = datetime(2024, 1, 1, 10, 0)

        signal = {
            "decision": "LONG",
            "confidence": 0.75,
        }

        backtester._open_position(bar, signal)

        assert backtester.position is not None
        assert backtester.position.direction == "LONG"
        assert backtester.position.entry_price == 100.0
        assert backtester.position.stop_loss < 100.0
        assert backtester.position.take_profit > 100.0

    def test_open_short_position(self, backtester):
        """Test opening short position."""
        bar = pd.Series(
            {
                "close": 100.0,
                "high": 101.0,
                "low": 99.0,
            }
        )
        bar.name = datetime(2024, 1, 1, 10, 0)

        signal = {
            "decision": "SHORT",
            "confidence": 0.75,
        }

        backtester._open_position(bar, signal)

        assert backtester.position is not None
        assert backtester.position.direction == "SHORT"
        assert backtester.position.stop_loss > 100.0
        assert backtester.position.take_profit < 100.0

    def test_close_position_long_profit(self, backtester):
        """Test closing long position with profit."""
        backtester.position = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="LONG",
            entry_price=100.0,
            size=1.0,
        )

        backtester._close_position(110.0, datetime(2024, 1, 1, 12, 0), "Take Profit")

        assert backtester.position is None
        assert len(backtester.trades) == 1
        assert backtester.trades[0].pnl > 0
        assert backtester.balance > 10000.0

    def test_close_position_long_loss(self, backtester):
        """Test closing long position with loss."""
        backtester.position = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="LONG",
            entry_price=100.0,
            size=1.0,
        )

        backtester._close_position(95.0, datetime(2024, 1, 1, 12, 0), "Stop Loss")

        assert backtester.position is None
        assert len(backtester.trades) == 1
        assert backtester.trades[0].pnl < 0
        assert backtester.balance < 10000.0

    def test_close_position_short_profit(self, backtester):
        """Test closing short position with profit."""
        backtester.position = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="SHORT",
            entry_price=100.0,
            size=1.0,
        )

        backtester._close_position(90.0, datetime(2024, 1, 1, 12, 0), "Take Profit")

        assert backtester.trades[0].pnl > 0

    def test_close_position_short_loss(self, backtester):
        """Test closing short position with loss."""
        backtester.position = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            direction="SHORT",
            entry_price=100.0,
            size=1.0,
        )

        backtester._close_position(110.0, datetime(2024, 1, 1, 12, 0), "Stop Loss")

        assert backtester.trades[0].pnl < 0


class TestCalculateResults:
    """Test result calculation."""

    @pytest.fixture
    def backtester(self):
        config = StrategyConfig()
        bt = Backtester(config, initial_balance=10000.0)
        return bt

    def test_calculate_empty_results(self, backtester):
        """Test calculating results with no trades."""
        result = backtester._calculate_results()

        assert result.total_trades == 0
        assert result.win_rate == 0
        assert result.profit_factor == 0

    def test_calculate_with_winning_trades(self, backtester):
        """Test calculating results with winning trades."""
        backtester.trades = [
            Trade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                direction="LONG",
                entry_price=100.0,
                exit_price=105.0,
                size=1.0,
                pnl=5.0,
                pnl_pct=5.0,
            ),
            Trade(
                entry_time=datetime(2024, 1, 3),
                exit_time=datetime(2024, 1, 4),
                direction="LONG",
                entry_price=105.0,
                exit_price=110.0,
                size=1.0,
                pnl=5.0,
                pnl_pct=4.76,
            ),
        ]
        backtester.balance = 10010.0

        result = backtester._calculate_results()

        assert result.total_trades == 2
        assert result.winning_trades == 2
        assert result.losing_trades == 0
        assert result.win_rate == 1.0

    def test_calculate_with_mixed_trades(self, backtester):
        """Test calculating results with mixed trades."""
        backtester.trades = [
            Trade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                direction="LONG",
                entry_price=100.0,
                exit_price=110.0,
                size=1.0,
                pnl=10.0,
                pnl_pct=10.0,
            ),
            Trade(
                entry_time=datetime(2024, 1, 3),
                exit_time=datetime(2024, 1, 4),
                direction="LONG",
                entry_price=110.0,
                exit_price=105.0,
                size=1.0,
                pnl=-5.0,
                pnl_pct=-4.55,
            ),
        ]
        backtester.balance = 10005.0
        backtester.equity_curve = [
            {"timestamp": datetime(2024, 1, 1), "balance": 10000.0, "price": 100.0},
            {"timestamp": datetime(2024, 1, 2), "balance": 10010.0, "price": 110.0},
            {"timestamp": datetime(2024, 1, 3), "balance": 10010.0, "price": 110.0},
            {"timestamp": datetime(2024, 1, 4), "balance": 10005.0, "price": 105.0},
        ]

        result = backtester._calculate_results()

        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == 0.5
        assert result.avg_win == 10.0
        assert result.avg_loss == 5.0

    def test_max_drawdown_calculation(self, backtester):
        """Test max drawdown calculation."""
        backtester.trades = [
            Trade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 2),
                direction="LONG",
                entry_price=100.0,
                exit_price=105.0,
                size=1.0,
                pnl=500.0,
                pnl_pct=5.0,
            ),
        ]
        backtester.balance = 10500.0
        backtester.equity_curve = [
            {"timestamp": datetime(2024, 1, 1), "balance": 10000.0, "price": 100.0},
            {"timestamp": datetime(2024, 1, 2), "balance": 10500.0, "price": 105.0},
            {"timestamp": datetime(2024, 1, 3), "balance": 10200.0, "price": 102.0},
            {"timestamp": datetime(2024, 1, 4), "balance": 10500.0, "price": 105.0},
        ]

        result = backtester._calculate_results()

        assert result.max_drawdown == 300.0  # 10500 - 10200
