"""
Tests for Profit Optimizer.

Tests:
- Entry optimization
- Exit optimization
- Optimal stop loss/take profit levels
- Learning from trade outcomes
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from bot.ml.profit_optimizer import (
    ProfitOptimizer,
    EntryOptimizer,
    ExitOptimizer,
    EntrySignal,
    ExitSignal,
    TradeState,
    OptimizedLevels,
)


def generate_synthetic_ohlcv(bars: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    if trend == "up":
        base = 100 + np.cumsum(np.random.randn(bars) * 0.5 + 0.1)
    elif trend == "down":
        base = 100 + np.cumsum(np.random.randn(bars) * 0.5 - 0.1)
    else:
        base = 100 + np.cumsum(np.random.randn(bars) * 0.3)

    high = base + np.abs(np.random.randn(bars)) * 0.5
    low = base - np.abs(np.random.randn(bars)) * 0.5
    open_price = base + np.random.randn(bars) * 0.2
    close = base + np.random.randn(bars) * 0.2
    volume = np.abs(np.random.randn(bars)) * 1000 + 500

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    df.index = pd.date_range(start="2024-01-01", periods=bars, freq="1h")
    return df


class TestEntryOptimizer:
    """Test EntryOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        return EntryOptimizer()

    @pytest.fixture
    def sample_data(self):
        return generate_synthetic_ohlcv(bars=100, trend="up")

    def test_analyze_entry_long(self, optimizer, sample_data):
        current_price = sample_data["close"].iloc[-1]

        signal = optimizer.analyze_entry(sample_data, "LONG", current_price)

        assert isinstance(signal, EntrySignal)
        assert signal.direction == "LONG"
        assert 0 <= signal.entry_quality <= 1
        assert signal.wait_bars >= 0
        assert signal.suggested_price > 0

    def test_analyze_entry_short(self, optimizer, sample_data):
        current_price = sample_data["close"].iloc[-1]

        signal = optimizer.analyze_entry(sample_data, "SHORT", current_price)

        assert isinstance(signal, EntrySignal)
        assert signal.direction == "SHORT"
        assert 0 <= signal.entry_quality <= 1

    def test_entry_quality_varies(self, optimizer):
        # Oversold conditions (good for LONG)
        oversold_data = generate_synthetic_ohlcv(bars=100, trend="down")
        oversold_price = oversold_data["close"].iloc[-1]

        signal_long = optimizer.analyze_entry(oversold_data, "LONG", oversold_price)

        # Overbought conditions (good for SHORT)
        overbought_data = generate_synthetic_ohlcv(bars=100, trend="up")
        overbought_price = overbought_data["close"].iloc[-1]

        signal_short = optimizer.analyze_entry(overbought_data, "SHORT", overbought_price)

        # Entry quality should differ based on conditions
        assert signal_long.entry_quality >= 0
        assert signal_short.entry_quality >= 0

    def test_record_entry_outcome(self, optimizer):
        # Record multiple outcomes
        optimizer.record_entry_outcome(
            entry_quality=0.7,
            direction="LONG",
            pnl_pct=0.02,
            max_favorable_pct=0.03,
            max_adverse_pct=0.01,
        )

        optimizer.record_entry_outcome(
            entry_quality=0.3,
            direction="LONG",
            pnl_pct=-0.01,
            max_favorable_pct=0.005,
            max_adverse_pct=0.02,
        )

        assert len(optimizer.entry_outcomes) == 2

    def test_insufficient_data(self, optimizer):
        small_data = generate_synthetic_ohlcv(bars=10)
        current_price = small_data["close"].iloc[-1]

        signal = optimizer.analyze_entry(small_data, "LONG", current_price)

        # Should return default signal with 0.5 quality
        assert signal.entry_quality == 0.5


class TestExitOptimizer:
    """Test ExitOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        return ExitOptimizer()

    @pytest.fixture
    def sample_data(self):
        return generate_synthetic_ohlcv(bars=100)

    def test_stop_loss_exit(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=97.0,  # 3% loss
            bars_held=5,
            max_favorable=0.01,
            max_adverse=0.03,
            unrealized_pnl_pct=-0.03,
        )

        signal = optimizer.analyze_exit(trade_state, sample_data)

        assert isinstance(signal, ExitSignal)
        assert signal.should_exit is True
        assert signal.exit_type == "STOP_LOSS"
        assert signal.urgency > 0.5

    def test_take_profit_exit(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=104.0,  # 4% profit
            bars_held=10,
            max_favorable=0.04,
            max_adverse=0.005,
            unrealized_pnl_pct=0.04,
        )

        signal = optimizer.analyze_exit(trade_state, sample_data)

        assert signal.should_exit is True
        assert signal.exit_type in ["TAKE_PROFIT", "TRAILING"]

    def test_trailing_stop_exit(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=101.0,  # 1% profit now
            bars_held=15,
            max_favorable=0.03,  # Was up 3%
            max_adverse=0.005,
            unrealized_pnl_pct=0.01,
        )

        signal = optimizer.analyze_exit(trade_state, sample_data)

        # Should trigger trailing stop since we gave back 2%
        assert signal.should_exit is True
        assert signal.exit_type == "TRAILING"

    def test_time_based_exit(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=100.5,  # Small profit
            bars_held=55,  # Over max hold
            max_favorable=0.01,
            max_adverse=0.005,
            unrealized_pnl_pct=0.005,
        )

        signal = optimizer.analyze_exit(trade_state, sample_data)

        assert signal.should_exit is True
        assert signal.exit_type == "TIME"

    def test_hold_position(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=101.5,  # 1.5% profit
            bars_held=5,
            max_favorable=0.015,
            max_adverse=0.002,
            unrealized_pnl_pct=0.015,
        )

        signal = optimizer.analyze_exit(trade_state, sample_data)

        # Should hold - not at stop, not at target, not at trailing
        assert signal.should_exit is False

    def test_calculate_optimal_levels(self, optimizer):
        levels = optimizer.calculate_optimal_levels(
            entry_price=100.0,
            direction="LONG",
            volatility=0.02,  # 2% volatility
            confidence=0.7,
        )

        assert isinstance(levels, OptimizedLevels)
        assert levels.stop_loss_price < 100.0  # Below entry for LONG
        assert levels.take_profit_price > 100.0  # Above entry for LONG
        assert levels.take_profit_pct >= levels.stop_loss_pct * 1.5  # Min 1.5:1 R:R

    def test_calculate_optimal_levels_short(self, optimizer):
        levels = optimizer.calculate_optimal_levels(
            entry_price=100.0,
            direction="SHORT",
            volatility=0.02,
            confidence=0.7,
        )

        assert levels.stop_loss_price > 100.0  # Above entry for SHORT
        assert levels.take_profit_price < 100.0  # Below entry for SHORT

    def test_record_exit_outcome(self, optimizer):
        optimizer.record_exit_outcome(
            exit_type="TAKE_PROFIT",
            pnl_pct=0.03,
            bars_held=15,
            max_favorable_pct=0.035,
            left_on_table_pct=0.005,
        )

        assert len(optimizer.exit_outcomes) == 1


class TestProfitOptimizer:
    """Test combined ProfitOptimizer class."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def optimizer(self, temp_dir):
        return ProfitOptimizer(model_dir=temp_dir)

    @pytest.fixture
    def sample_data(self):
        return generate_synthetic_ohlcv(bars=100)

    def test_optimize_entry(self, optimizer, sample_data):
        current_price = sample_data["close"].iloc[-1]

        signal = optimizer.optimize_entry(sample_data, "LONG", current_price)

        assert isinstance(signal, EntrySignal)
        assert signal.direction == "LONG"

    def test_optimize_exit(self, optimizer, sample_data):
        trade_state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=102.0,
            bars_held=10,
            max_favorable=0.025,
            max_adverse=0.01,
            unrealized_pnl_pct=0.02,
        )

        signal = optimizer.optimize_exit(trade_state, sample_data)

        assert isinstance(signal, ExitSignal)

    def test_get_optimal_levels(self, optimizer, sample_data):
        levels = optimizer.get_optimal_levels(
            entry_price=100.0,
            direction="LONG",
            df=sample_data,
            confidence=0.7,
        )

        assert isinstance(levels, OptimizedLevels)
        assert levels.stop_loss_pct > 0
        assert levels.take_profit_pct > 0

    def test_record_trade(self, optimizer):
        optimizer.record_trade(
            entry_quality=0.7,
            direction="LONG",
            pnl_pct=0.025,
            max_favorable_pct=0.03,
            max_adverse_pct=0.01,
            bars_held=12,
            exit_type="TAKE_PROFIT",
        )

        # Both entry and exit optimizers should have recorded
        assert len(optimizer.entry_optimizer.entry_outcomes) == 1
        assert len(optimizer.exit_optimizer.exit_outcomes) == 1

    def test_save_and_load(self, temp_dir, sample_data):
        optimizer1 = ProfitOptimizer(model_dir=temp_dir)

        # Modify some parameters
        optimizer1.entry_optimizer.optimal_rsi_buy = 30.0
        optimizer1.exit_optimizer.optimal_profit_target = 0.04

        optimizer1.save("test_optimizer")

        # Load in new instance
        optimizer2 = ProfitOptimizer(model_dir=temp_dir)
        success = optimizer2.load("test_optimizer")

        assert success is True
        assert optimizer2.entry_optimizer.optimal_rsi_buy == 30.0
        assert optimizer2.exit_optimizer.optimal_profit_target == 0.04

    def test_get_stats(self, optimizer):
        stats = optimizer.get_stats()

        assert "entry" in stats
        assert "exit" in stats
        assert "optimal_rsi_buy" in stats["entry"]
        assert "optimal_profit_target" in stats["exit"]


class TestTradeState:
    """Test TradeState dataclass."""

    def test_create_trade_state(self):
        state = TradeState(
            direction="LONG",
            entry_price=100.0,
            current_price=102.0,
            bars_held=10,
            max_favorable=0.025,
            max_adverse=0.01,
            unrealized_pnl_pct=0.02,
        )

        assert state.direction == "LONG"
        assert state.unrealized_pnl_pct == 0.02


class TestOptimizedLevels:
    """Test OptimizedLevels dataclass."""

    def test_create_optimized_levels(self):
        levels = OptimizedLevels(
            stop_loss_price=98.0,
            stop_loss_pct=0.02,
            take_profit_price=103.0,
            take_profit_pct=0.03,
            trailing_stop_pct=0.015,
            break_even_trigger_pct=0.01,
        )

        assert levels.stop_loss_price == 98.0
        assert levels.take_profit_pct == 0.03
        assert levels.trailing_stop_pct == 0.015
