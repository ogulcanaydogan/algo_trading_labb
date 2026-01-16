"""
Tests for state module.
"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime, timezone
from pathlib import Path

from bot.state import (
    BotState,
    SignalEvent,
    EquityPoint,
    StateStore,
)


class TestBotState:
    """Test BotState dataclass."""

    def test_default_state(self):
        """Test creating state with defaults."""
        state = BotState()
        assert state.symbol == "BTC/USDT"
        assert state.position == "FLAT"
        assert state.balance == 10000.0
        assert state.position_size == 0.0

    def test_custom_state(self):
        """Test creating state with custom values."""
        state = BotState(
            symbol="ETH/USDT",
            position="LONG",
            entry_price=3000.0,
            position_size=1.0,
            balance=15000.0,
        )
        assert state.symbol == "ETH/USDT"
        assert state.position == "LONG"
        assert state.entry_price == 3000.0

    def test_state_with_ai_fields(self):
        """Test state with AI fields."""
        state = BotState(
            ai_action="BUY",
            ai_confidence=0.85,
            ai_probability_long=0.7,
            ai_probability_short=0.2,
            ai_probability_flat=0.1,
        )
        assert state.ai_action == "BUY"
        assert state.ai_confidence == 0.85
        assert state.ai_probability_long == 0.7

    def test_state_with_macro_fields(self):
        """Test state with macro fields."""
        state = BotState(
            macro_bias=0.5,
            macro_confidence=0.7,
            macro_summary="Risk-on environment",
            macro_drivers=["Fed policy", "GDP growth"],
        )
        assert state.macro_bias == 0.5
        assert len(state.macro_drivers) == 2

    def test_to_dict(self):
        """Test conversion to dict."""
        state = BotState(
            symbol="SOL/USDT",
            position="SHORT",
            balance=12000.0,
        )
        d = state.to_dict()

        assert d["symbol"] == "SOL/USDT"
        assert d["position"] == "SHORT"
        assert d["balance"] == 12000.0
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "AVAX/USDT",
            "position": "LONG",
            "entry_price": 40.0,
            "position_size": 10.0,
            "balance": 11000.0,
            "initial_balance": 10000.0,
            "unrealized_pnl_pct": 0.0,
            "last_signal": None,
            "last_signal_reason": None,
            "confidence": None,
            "technical_signal": None,
            "technical_confidence": None,
            "technical_reason": None,
            "ai_override_active": False,
            "rsi": None,
            "ema_fast": None,
            "ema_slow": None,
            "risk_per_trade_pct": 0.5,
            "ai_action": None,
            "ai_confidence": None,
            "ai_probability_long": None,
            "ai_probability_short": None,
            "ai_probability_flat": None,
            "ai_expected_move_pct": None,
            "ai_summary": None,
            "ai_features": {},
            "macro_bias": None,
            "macro_confidence": None,
            "macro_summary": None,
            "macro_drivers": [],
            "macro_interest_rate_outlook": None,
            "macro_political_risk": None,
            "macro_events": [],
            "portfolio_playbook": {},
        }
        state = BotState.from_dict(data)

        assert state.symbol == "AVAX/USDT"
        assert state.position == "LONG"
        assert state.entry_price == 40.0

    def test_from_dict_no_timestamp(self):
        """Test from_dict without timestamp."""
        data = {
            "symbol": "BTC/USDT",
            "position": "FLAT",
            "entry_price": None,
            "position_size": 0.0,
            "balance": 10000.0,
            "initial_balance": 10000.0,
            "unrealized_pnl_pct": 0.0,
            "last_signal": None,
            "last_signal_reason": None,
            "confidence": None,
            "technical_signal": None,
            "technical_confidence": None,
            "technical_reason": None,
            "ai_override_active": False,
            "rsi": None,
            "ema_fast": None,
            "ema_slow": None,
            "risk_per_trade_pct": 0.5,
            "ai_action": None,
            "ai_confidence": None,
            "ai_probability_long": None,
            "ai_probability_short": None,
            "ai_probability_flat": None,
            "ai_expected_move_pct": None,
            "ai_summary": None,
            "ai_features": {},
            "macro_bias": None,
            "macro_confidence": None,
            "macro_summary": None,
            "macro_drivers": [],
            "macro_interest_rate_outlook": None,
            "macro_political_risk": None,
            "macro_events": [],
            "portfolio_playbook": {},
        }
        state = BotState.from_dict(data)
        assert state.timestamp is not None


class TestSignalEvent:
    """Test SignalEvent dataclass."""

    def test_signal_creation(self):
        """Test creating a signal."""
        now = datetime.now(timezone.utc)
        signal = SignalEvent(
            timestamp=now,
            symbol="BTC/USDT",
            decision="LONG",
            confidence=0.8,
            reason="RSI oversold",
        )
        assert signal.symbol == "BTC/USDT"
        assert signal.decision == "LONG"
        assert signal.confidence == 0.8

    def test_signal_with_ai_fields(self):
        """Test signal with AI fields."""
        now = datetime.now(timezone.utc)
        signal = SignalEvent(
            timestamp=now,
            symbol="ETH/USDT",
            decision="SHORT",
            confidence=0.75,
            reason="Trend reversal",
            ai_action="SELL",
            ai_confidence=0.8,
            ai_expected_move_pct=-2.5,
        )
        assert signal.ai_action == "SELL"
        assert signal.ai_confidence == 0.8
        assert signal.ai_expected_move_pct == -2.5

    def test_to_dict(self):
        """Test conversion to dict."""
        now = datetime.now(timezone.utc)
        signal = SignalEvent(
            timestamp=now,
            symbol="SOL/USDT",
            decision="LONG",
            confidence=0.9,
            reason="Breakout",
        )
        d = signal.to_dict()

        assert d["symbol"] == "SOL/USDT"
        assert d["decision"] == "LONG"
        assert d["confidence"] == 0.9
        assert "execution_reason" in d  # Legacy field

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "AVAX/USDT",
            "decision": "SHORT",
            "confidence": 0.7,
            "reason": "Resistance hit",
        }
        signal = SignalEvent.from_dict(data)

        assert signal.symbol == "AVAX/USDT"
        assert signal.decision == "SHORT"
        assert signal.reason == "Resistance hit"

    def test_from_dict_legacy_field(self):
        """Test from_dict with legacy execution_reason field."""
        data = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "BTC/USDT",
            "decision": "FLAT",
            "confidence": 0.5,
            "execution_reason": "Take profit hit",
        }
        signal = SignalEvent.from_dict(data)

        assert signal.reason == "Take profit hit"


class TestEquityPoint:
    """Test EquityPoint dataclass."""

    def test_point_creation(self):
        """Test creating an equity point."""
        now = datetime.now(timezone.utc)
        point = EquityPoint(timestamp=now, value=11000.0)

        assert point.value == 11000.0
        assert point.timestamp == now

    def test_to_dict(self):
        """Test conversion to dict."""
        now = datetime.now(timezone.utc)
        point = EquityPoint(timestamp=now, value=12500.0)
        d = point.to_dict()

        assert d["value"] == 12500.0
        assert "timestamp" in d

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "value": 13000.0,
        }
        point = EquityPoint.from_dict(data)

        assert point.value == 13000.0


class TestStateStore:
    """Test StateStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create state store."""
        return StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )

    def test_store_creation(self, store):
        """Test store is created."""
        assert store is not None
        assert store.state is not None
        assert store.signals == []
        assert store.equity_curve == []

    def test_update_state(self, store):
        """Test updating state."""
        store.update_state(
            symbol="ETH/USDT",
            position="LONG",
            balance=12000.0,
        )

        assert store.state.symbol == "ETH/USDT"
        assert store.state.position == "LONG"
        assert store.state.balance == 12000.0

    def test_update_state_persists(self, temp_dir):
        """Test state is persisted."""
        store = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        store.update_state(balance=15000.0)

        # Create new store - should load persisted state
        store2 = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        assert store2.state.balance == 15000.0

    def test_record_signal(self, store):
        """Test recording a signal."""
        now = datetime.now(timezone.utc)
        signal = SignalEvent(
            timestamp=now,
            symbol="BTC/USDT",
            decision="LONG",
            confidence=0.8,
            reason="Test",
        )
        store.record_signal(signal)

        assert len(store.signals) == 1
        assert store.signals[0].decision == "LONG"

    def test_record_signal_limit(self, temp_dir):
        """Test signal recording respects limit."""
        store = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
            max_signals=10,
        )

        for i in range(15):
            signal = SignalEvent(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                decision="LONG" if i % 2 == 0 else "SHORT",
                confidence=0.5,
                reason=f"Test {i}",
            )
            store.record_signal(signal)

        assert len(store.signals) == 10

    def test_record_equity(self, store):
        """Test recording equity points."""
        now = datetime.now(timezone.utc)
        point = EquityPoint(timestamp=now, value=11000.0)
        store.record_equity(point)

        assert len(store.equity_curve) == 1
        assert store.equity_curve[0].value == 11000.0

    def test_get_state_dict(self, store):
        """Test getting state as dict."""
        store.update_state(position="SHORT", balance=9000.0)
        d = store.get_state_dict()

        assert d["position"] == "SHORT"
        assert d["balance"] == 9000.0

    def test_get_signals(self, store):
        """Test getting signals."""
        for i in range(5):
            signal = SignalEvent(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                decision="LONG",
                confidence=0.5 + i * 0.1,
                reason=f"Test {i}",
            )
            store.record_signal(signal)

        signals = store.get_signals()
        assert len(signals) == 5
        # Should be reversed (most recent first)
        assert signals[0]["confidence"] == pytest.approx(0.9, rel=0.01)

    def test_get_signals_with_limit(self, store):
        """Test getting signals with limit."""
        for i in range(10):
            signal = SignalEvent(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                decision="LONG",
                confidence=0.5,
                reason=f"Test {i}",
            )
            store.record_signal(signal)

        signals = store.get_signals(limit=3)
        assert len(signals) == 3

    def test_get_equity_curve(self, store):
        """Test getting equity curve."""
        for i in range(5):
            point = EquityPoint(
                timestamp=datetime.now(timezone.utc),
                value=10000.0 + i * 100,
            )
            store.record_equity(point)

        curve = store.get_equity_curve()
        assert len(curve) == 5
        assert curve[0]["value"] == 10000.0
        assert curve[4]["value"] == 10400.0


class TestStateStorePersistence:
    """Test state store persistence."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_creates_parent_dirs(self, temp_dir):
        """Test parent directories are created."""
        nested = temp_dir / "a" / "b" / "c"
        store = StateStore(
            state_path=nested / "state.json",
            signals_path=nested / "signals.json",
            equity_path=nested / "equity.json",
        )
        assert nested.exists()

    def test_state_file_created(self, temp_dir):
        """Test state file is created."""
        store = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        store.update_state(balance=12000.0)

        assert (temp_dir / "state.json").exists()

    def test_signals_loaded_on_restart(self, temp_dir):
        """Test signals are loaded on restart."""
        store1 = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        signal = SignalEvent(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            decision="LONG",
            confidence=0.9,
            reason="Persistent signal",
        )
        store1.record_signal(signal)

        # Create new store
        store2 = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        assert len(store2.signals) == 1
        assert store2.signals[0].reason == "Persistent signal"

    def test_equity_loaded_on_restart(self, temp_dir):
        """Test equity curve is loaded on restart."""
        store1 = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        point = EquityPoint(
            timestamp=datetime.now(timezone.utc),
            value=15000.0,
        )
        store1.record_equity(point)

        # Create new store
        store2 = StateStore(
            state_path=temp_dir / "state.json",
            signals_path=temp_dir / "signals.json",
            equity_path=temp_dir / "equity.json",
        )
        assert len(store2.equity_curve) == 1
        assert store2.equity_curve[0].value == 15000.0
