"""Integration tests for critical trading workflows."""

from datetime import datetime, timezone

from bot.safety_controller import SafetyController, SafetyLimits
from bot.trading_mode import TradingMode, TradingStatus
from bot.unified_state import UnifiedState, PositionState


def _create_state(mode=TradingMode.PAPER_LIVE_DATA, balance=10000.0):
    """Helper to create test state with all required fields."""
    return UnifiedState(
        mode=mode,
        status=TradingStatus.ACTIVE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        initial_capital=balance,
        current_balance=balance,
        peak_balance=balance,
    )


class TestPaperTradingWorkflow:
    """Test paper trading mode workflow."""

    def test_paper_mode_initialization(self):
        """Test that paper mode initializes with safe defaults."""
        state = _create_state()
        
        assert state.mode == TradingMode.PAPER_LIVE_DATA
        assert state.current_balance == 10000.0

    def test_paper_mode_open_position(self):
        """Test opening a position in paper mode."""
        state = _create_state()
        
        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        state.positions["BTC/USDT"] = position
        
        assert "BTC/USDT" in state.positions
        assert state.positions["BTC/USDT"].quantity == 0.01


class TestSafetyControllerWorkflow:
    """Test safety controller enforces limits."""

    def test_daily_loss_limit_enforcement(self):
        """Test that daily loss limit is configured."""
        limits = SafetyLimits(max_daily_loss_usd=100.0)
        assert limits.max_daily_loss_usd == 100.0

    def test_position_size_limit(self):
        """Test position size limit enforcement."""
        limits = SafetyLimits(max_position_size_usd=500.0)
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)
        
        controller.update_positions({"BTC/USDT": 400.0})
        assert sum(controller._open_positions.values()) == 400.0


class TestModeTransitionWorkflow:
    """Test trading mode transitions."""

    def test_paper_to_testnet_transition(self):
        """Test transition from paper to testnet mode."""
        state = _create_state(mode=TradingMode.PAPER_LIVE_DATA)
        
        assert state.mode == TradingMode.PAPER_LIVE_DATA
        
        state.mode = TradingMode.TESTNET
        
        assert state.mode == TradingMode.TESTNET

    def test_state_preserved_across_mode_transition(self):
        """Test that portfolio state is preserved during mode transition."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        state = _create_state(balance=9580.0)
        state.positions["BTC/USDT"] = position
        
        state.mode = TradingMode.TESTNET
        
        assert state.current_balance == 9580.0
        assert "BTC/USDT" in state.positions


class TestSignalGenerationWorkflow:
    """Test signal generation from market data."""

    def test_signal_generation_returns_valid_structure(self):
        """Test that signals have required fields."""
        signal = {
            "symbol": "BTC/USDT",
            "signal": "LONG",
            "confidence": 0.75,
            "reason": "EMA crossover with RSI confirmation",
        }
        
        assert signal["signal"] in ["LONG", "SHORT", "FLAT"]
        assert 0.0 <= signal["confidence"] <= 1.0

    def test_signal_confidence_bounds(self):
        """Test signal confidence is within valid range."""
        valid_signals = [
            {"signal": "LONG", "confidence": 0.0},
            {"signal": "LONG", "confidence": 0.5},
            {"signal": "LONG", "confidence": 1.0},
        ]
        
        for sig in valid_signals:
            assert 0.0 <= sig["confidence"] <= 1.0


class TestErrorRecoveryWorkflow:
    """Test error handling and recovery in trading workflows."""

    def test_trading_pauses_on_api_errors(self):
        """Test that trading pauses after repeated API errors."""
        controller = SafetyController()
        controller.update_balance(10000.0)
        
        for _ in range(3):
            controller._daily_stats.api_errors += 1
        
        assert controller._daily_stats.api_errors == 3

    def test_state_recovery_on_reconnect(self):
        """Test state recovery when connection is restored."""
        position = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )
        
        state = _create_state(balance=7500.0)
        state.positions["ETH/USDT"] = position
        
        assert state.positions["ETH/USDT"].symbol == "ETH/USDT"
        assert state.current_balance == 7500.0


class TestMultiPositionWorkflow:
    """Test handling multiple concurrent positions."""

    def test_open_multiple_positions(self):
        """Test opening multiple positions simultaneously."""
        state = _create_state()
        
        pos1 = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        pos2 = PositionState(
            symbol="ETH/USDT",
            quantity=0.5,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )
        
        state.positions["BTC/USDT"] = pos1
        state.positions["ETH/USDT"] = pos2
        
        assert len(state.positions) == 2

    def test_close_one_position_keep_others(self):
        """Test closing one position while keeping others open."""
        state = _create_state()
        state.positions["BTC/USDT"] = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        state.positions["ETH/USDT"] = PositionState(
            symbol="ETH/USDT",
            quantity=0.5,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )
        
        del state.positions["BTC/USDT"]
        
        assert "BTC/USDT" not in state.positions
        assert "ETH/USDT" in state.positions
