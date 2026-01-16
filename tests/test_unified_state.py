"""
Tests for unified state module.
"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

from bot.unified_state import (
    PositionState,
    TradeRecord,
    EquityPoint,
    UnifiedState,
    UnifiedStateStore,
)
from bot.trading_mode import TradingMode, TradingStatus


class TestPositionState:
    """Test PositionState dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time="2024-01-15T10:00:00",
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.quantity == 0.1
        assert pos.entry_price == 50000.0
        assert pos.side == "long"
        assert pos.stop_loss is None
        assert pos.unrealized_pnl == 0.0

    def test_position_with_optional_fields(self):
        """Test position with optional fields."""
        pos = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=3000.0,
            side="short",
            entry_time="2024-01-15T10:00:00",
            stop_loss=3100.0,
            take_profit=2800.0,
            unrealized_pnl=-50.0,
            current_price=3050.0,
        )
        assert pos.stop_loss == 3100.0
        assert pos.take_profit == 2800.0
        assert pos.unrealized_pnl == -50.0
        assert pos.current_price == 3050.0

    def test_to_dict(self):
        """Test conversion to dict."""
        pos = PositionState(
            symbol="SOL/USDT",
            quantity=10.0,
            entry_price=100.0,
            side="long",
            entry_time="2024-01-15T10:00:00",
            unrealized_pnl=50.0,
        )
        d = pos.to_dict()

        assert d["symbol"] == "SOL/USDT"
        assert d["quantity"] == 10.0
        assert d["entry_price"] == 100.0
        assert d["side"] == "long"
        assert d["unrealized_pnl"] == 50.0

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "symbol": "AVAX/USDT",
            "quantity": 5.0,
            "entry_price": 40.0,
            "side": "long",
            "entry_time": "2024-01-15T10:00:00",
            "stop_loss": 38.0,
            "take_profit": 45.0,
            "unrealized_pnl": 10.0,
            "current_price": 42.0,
        }
        pos = PositionState.from_dict(data)

        assert pos.symbol == "AVAX/USDT"
        assert pos.quantity == 5.0
        assert pos.stop_loss == 38.0


class TestTradeRecord:
    """Test TradeRecord dataclass."""

    def test_trade_creation(self):
        """Test creating a trade record."""
        trade = TradeRecord(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            exit_price=52000.0,
            pnl=200.0,
            pnl_pct=4.0,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T14:00:00",
            exit_reason="take_profit",
        )
        assert trade.id == "trade_001"
        assert trade.pnl == 200.0
        assert trade.exit_reason == "take_profit"
        assert trade.commission == 0.0
        assert trade.mode == "paper"

    def test_trade_with_all_fields(self):
        """Test trade with all fields."""
        trade = TradeRecord(
            id="trade_002",
            symbol="ETH/USDT",
            side="sell",
            quantity=1.0,
            entry_price=3000.0,
            exit_price=2800.0,
            pnl=200.0,
            pnl_pct=6.67,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T12:00:00",
            exit_reason="signal",
            commission=5.0,
            mode="live",
            signal_confidence=0.85,
            signal_reasons=["RSI oversold", "MACD cross"],
        )
        assert trade.commission == 5.0
        assert trade.mode == "live"
        assert trade.signal_confidence == 0.85
        assert len(trade.signal_reasons) == 2

    def test_to_dict(self):
        """Test conversion to dict."""
        trade = TradeRecord(
            id="trade_003",
            symbol="SOL/USDT",
            side="buy",
            quantity=10.0,
            entry_price=100.0,
            exit_price=110.0,
            pnl=100.0,
            pnl_pct=10.0,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T16:00:00",
            exit_reason="manual",
        )
        d = trade.to_dict()

        assert d["id"] == "trade_003"
        assert d["pnl"] == 100.0
        assert d["mode"] == "paper"

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "id": "trade_004",
            "symbol": "AVAX/USDT",
            "side": "buy",
            "quantity": 5.0,
            "entry_price": 40.0,
            "exit_price": 38.0,
            "pnl": -10.0,
            "pnl_pct": -5.0,
            "entry_time": "2024-01-15T10:00:00",
            "exit_time": "2024-01-15T11:00:00",
            "exit_reason": "stop_loss",
            "commission": 2.0,
            "mode": "paper_synthetic",
            "signal_confidence": 0.7,
            "signal_reasons": ["trend following"],
        }
        trade = TradeRecord.from_dict(data)

        assert trade.id == "trade_004"
        assert trade.pnl == -10.0
        assert trade.signal_confidence == 0.7


class TestEquityPoint:
    """Test EquityPoint dataclass."""

    def test_equity_point_creation(self):
        """Test creating an equity point."""
        point = EquityPoint(
            timestamp="2024-01-15T10:00:00",
            balance=10000.0,
            positions_value=500.0,
            total_equity=10500.0,
        )
        assert point.timestamp == "2024-01-15T10:00:00"
        assert point.balance == 10000.0
        assert point.positions_value == 500.0
        assert point.total_equity == 10500.0
        assert point.unrealized_pnl == 0.0

    def test_equity_point_with_unrealized(self):
        """Test equity point with unrealized P&L."""
        point = EquityPoint(
            timestamp="2024-01-15T10:00:00",
            balance=10000.0,
            positions_value=500.0,
            total_equity=10500.0,
            unrealized_pnl=50.0,
        )
        assert point.unrealized_pnl == 50.0


class TestUnifiedState:
    """Test UnifiedState dataclass."""

    def test_state_creation(self):
        """Test creating a unified state."""
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=10500.0,
            peak_balance=10500.0,
        )
        assert state.mode == TradingMode.PAPER_SYNTHETIC
        assert state.status == TradingStatus.ACTIVE
        assert state.current_balance == 10500.0
        assert state.total_trades == 0
        assert state.positions == {}

    def test_state_with_positions(self):
        """Test state with positions."""
        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time="2024-01-15T10:00:00",
        )
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=10000.0,
            peak_balance=10000.0,
            positions={"BTC/USDT": pos},
        )
        assert len(state.positions) == 1
        assert "BTC/USDT" in state.positions

    def test_win_rate_calculation(self):
        """Test win rate property."""
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=10500.0,
            peak_balance=10500.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
        )
        assert state.win_rate == 0.6

    def test_win_rate_zero_trades(self):
        """Test win rate with no trades."""
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.STOPPED,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=10000.0,
            peak_balance=10000.0,
        )
        assert state.win_rate == 0.0

    def test_total_equity_calculation(self):
        """Test total equity property."""
        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time="2024-01-15T10:00:00",
            current_price=52000.0,
        )
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=5000.0,  # Used 5000 to buy
            peak_balance=10000.0,
            positions={"BTC/USDT": pos},
        )
        # Total equity = 5000 (balance) + 0.1 * 52000 (position)
        assert state.total_equity == 5000.0 + 0.1 * 52000.0

    def test_total_equity_no_current_price(self):
        """Test total equity when position has no current price."""
        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time="2024-01-15T10:00:00",
        )
        state = UnifiedState(
            mode=TradingMode.PAPER_SYNTHETIC,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=5000.0,
            peak_balance=10000.0,
            positions={"BTC/USDT": pos},
        )
        # Uses entry price when no current price
        assert state.total_equity == 5000.0 + 0.1 * 50000.0

    def test_to_dict(self):
        """Test state to dict conversion."""
        state = UnifiedState(
            mode=TradingMode.LIVE_FULL,
            status=TradingStatus.ACTIVE,
            timestamp="2024-01-15T10:00:00",
            initial_capital=10000.0,
            current_balance=12000.0,
            peak_balance=12500.0,
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            total_pnl=2000.0,
            max_drawdown_pct=0.04,
        )
        d = state.to_dict()

        assert d["mode"] == "live_full"
        assert d["status"] == "active"
        assert d["current_balance"] == 12000.0
        assert d["total_trades"] == 20
        assert d["max_drawdown_pct"] == 0.04

    def test_from_dict(self):
        """Test state from dict."""
        data = {
            "mode": "paper_synthetic",
            "status": "stopped",
            "timestamp": "2024-01-15T10:00:00",
            "initial_capital": 10000.0,
            "current_balance": 11000.0,
            "peak_balance": 11000.0,
            "positions": {},
            "total_trades": 5,
            "winning_trades": 3,
            "losing_trades": 2,
            "total_pnl": 1000.0,
            "max_drawdown_pct": 0.02,
            "mode_started_at": "2024-01-10T00:00:00",
            "days_in_mode": 5,
            "daily_trades": 2,
            "daily_pnl": 100.0,
            "daily_date": "2024-01-15",
        }
        state = UnifiedState.from_dict(data)

        assert state.mode == TradingMode.PAPER_SYNTHETIC
        assert state.status == TradingStatus.STOPPED
        assert state.total_trades == 5
        assert state.win_rate == 0.6


class TestUnifiedStateStore:
    """Test UnifiedStateStore class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "unified_trading"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def store(self, temp_data_dir):
        """Create state store."""
        return UnifiedStateStore(data_dir=temp_data_dir)

    def test_store_creation(self, store):
        """Test store is created."""
        assert store is not None
        assert store.data_dir.exists()

    def test_initialize_new_state(self, store):
        """Test initializing new state."""
        state = store.initialize(
            mode=TradingMode.PAPER_SYNTHETIC,
            initial_capital=10000.0,
            resume=False,
        )

        assert state is not None
        assert state.mode == TradingMode.PAPER_SYNTHETIC
        assert state.initial_capital == 10000.0
        assert state.current_balance == 10000.0
        assert state.status == TradingStatus.STOPPED

    def test_initialize_resumes_existing(self, temp_data_dir):
        """Test initializing resumes existing state."""
        # Create first store and initialize
        store1 = UnifiedStateStore(data_dir=temp_data_dir)
        state1 = store1.initialize(
            mode=TradingMode.PAPER_SYNTHETIC,
            initial_capital=10000.0,
        )
        # Modify state
        store1.update_state(current_balance=11000.0)

        # Create second store - should resume
        store2 = UnifiedStateStore(data_dir=temp_data_dir)
        state2 = store2.initialize(
            mode=TradingMode.LIVE_FULL,  # Different mode
            initial_capital=5000.0,  # Different capital
            resume=True,
        )

        # Should have loaded the saved state
        assert state2.current_balance == 11000.0

    def test_get_state(self, store):
        """Test getting state."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        state = store.get_state()

        assert state is not None
        assert state.mode == TradingMode.PAPER_SYNTHETIC

    def test_get_state_not_initialized(self, store):
        """Test getting state before initialization."""
        state = store.get_state()
        assert state is None

    def test_update_state(self, store):
        """Test updating state."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        state = store.update_state(
            current_balance=11000.0,
            total_trades=5,
        )

        assert state.current_balance == 11000.0
        assert state.total_trades == 5

    def test_update_state_updates_peak_balance(self, store):
        """Test peak balance is updated."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        store.update_state(current_balance=12000.0)
        state = store.get_state()
        assert state.peak_balance == 12000.0

        # Lower balance shouldn't update peak
        store.update_state(current_balance=11500.0)
        state = store.get_state()
        assert state.peak_balance == 12000.0

    def test_update_state_calculates_drawdown(self, store):
        """Test drawdown is calculated."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        # Go up then down
        store.update_state(current_balance=12000.0)
        store.update_state(current_balance=10800.0)  # 10% drawdown

        state = store.get_state()
        assert state.max_drawdown_pct == pytest.approx(0.1, rel=0.01)

    def test_update_state_not_initialized(self, store):
        """Test update raises when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            store.update_state(current_balance=11000.0)

    def test_update_position(self, store):
        """Test updating a position."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time=datetime.now().isoformat(),
        )
        store.update_position("BTC/USDT", pos)

        state = store.get_state()
        assert "BTC/USDT" in state.positions
        assert state.positions["BTC/USDT"].quantity == 0.1

    def test_update_position_remove(self, store):
        """Test removing a position."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time=datetime.now().isoformat(),
        )
        store.update_position("BTC/USDT", pos)
        assert "BTC/USDT" in store.get_state().positions

        # Remove position
        store.update_position("BTC/USDT", None)
        assert "BTC/USDT" not in store.get_state().positions

    def test_record_trade(self, store):
        """Test recording a trade."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        trade = TradeRecord(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            exit_price=52000.0,
            pnl=200.0,
            pnl_pct=4.0,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T14:00:00",
            exit_reason="take_profit",
        )
        store.record_trade(trade)

        state = store.get_state()
        assert state.total_trades == 1
        assert state.winning_trades == 1
        assert state.total_pnl == 200.0

        trades = store.get_trades()
        assert len(trades) == 1
        assert trades[0].id == "trade_001"

    def test_record_losing_trade(self, store):
        """Test recording a losing trade."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        trade = TradeRecord(
            id="trade_002",
            symbol="ETH/USDT",
            side="buy",
            quantity=1.0,
            entry_price=3000.0,
            exit_price=2800.0,
            pnl=-200.0,
            pnl_pct=-6.67,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T12:00:00",
            exit_reason="stop_loss",
        )
        store.record_trade(trade)

        state = store.get_state()
        assert state.total_trades == 1
        assert state.losing_trades == 1
        assert state.total_pnl == -200.0

    def test_record_equity_point(self, store):
        """Test recording equity point."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        store.record_equity_point()

        equity_curve = store.get_equity_curve()
        assert len(equity_curve) == 1
        assert equity_curve[0].balance == 10000.0

    def test_record_equity_point_with_position(self, store):
        """Test recording equity point with position."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.1,
            entry_price=50000.0,
            side="long",
            entry_time=datetime.now().isoformat(),
            current_price=52000.0,
            unrealized_pnl=200.0,
        )
        store.update_position("BTC/USDT", pos)
        store.record_equity_point()

        equity_curve = store.get_equity_curve()
        assert len(equity_curve) == 1
        assert equity_curve[0].positions_value == 0.1 * 52000.0
        assert equity_curve[0].unrealized_pnl == 200.0

    def test_change_mode(self, store):
        """Test changing mode."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        store.change_mode(
            new_mode=TradingMode.LIVE_FULL,
            reason="Paper validation passed",
            approver="system",
        )

        state = store.get_state()
        assert state.mode == TradingMode.LIVE_FULL

        history = store.get_mode_history()
        assert len(history) == 1
        assert history[0]["from_mode"] == "paper_synthetic"
        assert history[0]["to_mode"] == "live_full"

    def test_get_trades_limit(self, store):
        """Test getting trades with limit."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        for i in range(10):
            trade = TradeRecord(
                id=f"trade_{i}",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                entry_price=50000.0,
                exit_price=51000.0,
                pnl=100.0,
                pnl_pct=2.0,
                entry_time="2024-01-15T10:00:00",
                exit_time="2024-01-15T12:00:00",
                exit_reason="tp",
            )
            store.record_trade(trade)

        trades = store.get_trades(limit=5)
        assert len(trades) == 5

    def test_get_equity_curve_limit(self, store):
        """Test getting equity curve with limit."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        for i in range(20):
            store.record_equity_point()

        curve = store.get_equity_curve(limit=10)
        assert len(curve) == 10

    def test_get_mode_state(self, store):
        """Test getting mode state."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        store.update_state(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=500.0,
        )

        mode_state = store.get_mode_state()

        assert mode_state.mode == TradingMode.PAPER_SYNTHETIC
        assert mode_state.total_trades == 10
        assert mode_state.winning_trades == 6

    def test_get_summary(self, store):
        """Test getting summary."""
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        store.update_state(
            current_balance=11000.0,
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            total_pnl=1000.0,
        )

        summary = store.get_summary()

        assert summary["mode"] == "paper_synthetic"
        assert summary["balance"] == 11000.0
        assert summary["total_pnl"] == 1000.0
        assert summary["total_pnl_pct"] == 10.0  # 1000/10000*100
        assert summary["win_rate"] == 60.0

    def test_get_summary_not_initialized(self, store):
        """Test summary when not initialized."""
        summary = store.get_summary()
        assert summary == {}


class TestStatePersistence:
    """Test state persistence."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "unified_trading"
        shutil.rmtree(temp_dir)

    def test_state_persisted(self, temp_data_dir):
        """Test state is saved to disk."""
        store = UnifiedStateStore(data_dir=temp_data_dir)
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        assert (temp_data_dir / "state.json").exists()

    def test_trades_persisted(self, temp_data_dir):
        """Test trades are saved to disk."""
        store = UnifiedStateStore(data_dir=temp_data_dir)
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        trade = TradeRecord(
            id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            entry_price=50000.0,
            exit_price=52000.0,
            pnl=200.0,
            pnl_pct=4.0,
            entry_time="2024-01-15T10:00:00",
            exit_time="2024-01-15T14:00:00",
            exit_reason="tp",
        )
        store.record_trade(trade)

        assert (temp_data_dir / "trades.json").exists()

        # Verify content
        with open(temp_data_dir / "trades.json") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["id"] == "trade_001"

    def test_equity_persisted(self, temp_data_dir):
        """Test equity curve is saved to disk."""
        store = UnifiedStateStore(data_dir=temp_data_dir)
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        store.record_equity_point()

        assert (temp_data_dir / "equity.json").exists()

    def test_mode_history_persisted(self, temp_data_dir):
        """Test mode history is saved to disk."""
        store = UnifiedStateStore(data_dir=temp_data_dir)
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        store.change_mode(TradingMode.LIVE_FULL, "test")

        assert (temp_data_dir / "mode_history.json").exists()

    def test_state_loaded_on_restart(self, temp_data_dir):
        """Test state is loaded when resuming."""
        # First store
        store1 = UnifiedStateStore(data_dir=temp_data_dir)
        store1.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)
        store1.update_state(current_balance=12000.0, total_trades=5)

        # Second store should load
        store2 = UnifiedStateStore(data_dir=temp_data_dir)
        store2.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0, resume=True)

        state = store2.get_state()
        assert state.current_balance == 12000.0
        assert state.total_trades == 5


class TestMaxLimits:
    """Test max limits for trades and equity."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "unified_trading"
        shutil.rmtree(temp_dir)

    def test_trades_trimmed(self, temp_data_dir):
        """Test trades list is trimmed."""
        store = UnifiedStateStore(
            data_dir=temp_data_dir,
            max_trades=10,
        )
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        for i in range(15):
            trade = TradeRecord(
                id=f"trade_{i}",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                entry_price=50000.0,
                exit_price=51000.0,
                pnl=100.0,
                pnl_pct=2.0,
                entry_time="2024-01-15T10:00:00",
                exit_time="2024-01-15T12:00:00",
                exit_reason="tp",
            )
            store.record_trade(trade)

        trades = store.get_trades(limit=100)
        assert len(trades) == 10
        # Should have the most recent trades
        assert trades[0].id == "trade_5"

    def test_equity_trimmed(self, temp_data_dir):
        """Test equity curve is trimmed."""
        store = UnifiedStateStore(
            data_dir=temp_data_dir,
            max_equity_points=10,
        )
        store.initialize(TradingMode.PAPER_SYNTHETIC, 10000.0)

        for i in range(15):
            store.record_equity_point()

        curve = store.get_equity_curve(limit=100)
        assert len(curve) == 10
