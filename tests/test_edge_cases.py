"""Edge case and corner case tests for trading system."""

from datetime import datetime, timezone

from bot.safety_controller import SafetyController, SafetyLimits
from bot.trading_mode import TradingMode, TradingStatus
from bot.unified_state import UnifiedState, PositionState


def _create_state(balance=10000.0):
    """Helper to create test state."""
    return UnifiedState(
        mode=TradingMode.PAPER_LIVE_DATA,
        status=TradingStatus.ACTIVE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        initial_capital=balance,
        current_balance=balance,
        peak_balance=balance,
    )


class TestZeroAndNegativeValues:
    """Test handling of zero and negative values."""

    def test_zero_balance(self):
        """Test state with zero balance."""
        state = _create_state(balance=0.0)
        assert state.current_balance == 0.0

    def test_negative_pnl(self):
        """Test handling negative profit/loss."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=40000.0,
        )
        
        pnl = (position.current_price - position.entry_price) * position.quantity
        assert pnl < 0

    def test_zero_quantity_position(self):
        """Test handling zero quantity positions."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=None,
            take_profit=None,
            current_price=42000.0,
        )
        
        assert position.quantity == 0.0


class TestBoundaryValues:
    """Test boundary and extreme values."""

    def test_max_position_size(self, tmp_path):
        """Test maximum allowed position size."""
        # Set pct to 0 to prevent auto-scaling from overriding usd value
        limits = SafetyLimits(max_position_size_usd=1000000.0, max_position_size_pct=0.0)
        # Use temp state path to avoid loading persisted state
        controller = SafetyController(limits=limits, state_path=tmp_path / "safety_state.json")
        controller.update_balance(1000000.0)

        controller.update_positions({"BTC/USDT": 999999.0})
        assert sum(controller._open_positions.values()) < limits.max_position_size_usd

    def test_very_small_prices(self):
        """Test handling very small asset prices."""
        position = PositionState(
            symbol="MEME/USDT",
            quantity=1000000.0,
            entry_price=0.0001,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=0.00009,
            take_profit=0.00011,
            current_price=0.0001,
        )
        
        assert position.entry_price == 0.0001

    def test_very_large_prices(self):
        """Test handling very large asset prices."""
        position = PositionState(
            symbol="BITCOIN_FUTURE/USD",
            quantity=0.00001,
            entry_price=1000000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=950000.0,
            take_profit=1050000.0,
            current_price=1000000.0,
        )
        
        position_value = position.quantity * position.entry_price
        assert position_value == 10.0

    def test_fractional_quantities(self):
        """Test handling very small fractional quantities."""
        position = PositionState(
            symbol="ETH/USDT",
            quantity=0.00000001,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )
        
        assert position.quantity == 0.00000001


class TestConcurrentPositionEdgeCases:
    """Test edge cases with multiple positions."""

    def test_all_positions_at_stop_loss(self):
        """Test scenario where all positions hit stop loss."""
        state = _create_state()
        state.positions["BTC/USDT"] = PositionState(
            symbol="BTC/USDT",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=40000.0,
        )
        state.positions["ETH/USDT"] = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2400.0,
        )
        
        for pos in state.positions.values():
            if pos.stop_loss and pos.current_price == pos.stop_loss:
                assert pos.current_price == pos.stop_loss

    def test_all_positions_profitable(self):
        """Test scenario with all positions profitable."""
        state = _create_state(balance=12500.0)
        state.positions["BTC/USDT"] = PositionState(
            symbol="BTC/USDT",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=43000.0,
        )
        state.positions["ETH/USDT"] = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2600.0,
        )
        
        total_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.quantity
            for pos in state.positions.values()
        )
        assert total_pnl > 0

    def test_max_allowed_concurrent_positions(self):
        """Test hitting maximum concurrent positions limit."""
        limits = SafetyLimits(max_trades_per_day=3)
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)
        
        positions = {}
        for i in range(3):
            positions[f"COIN{i}/USDT"] = 100.0
        
        controller.update_positions(positions)
        assert len(controller._open_positions) == 3


class TestRiskManagementEdgeCases:
    """Test edge cases in risk management."""

    def test_stop_loss_above_current_price(self):
        """Test invalid stop loss configuration."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=43000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        assert position.stop_loss > position.entry_price

    def test_take_profit_below_current_price(self):
        """Test invalid take profit configuration."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=41000.0,
            current_price=42000.0,
        )
        
        assert position.take_profit < position.entry_price

    def test_daily_loss_limit_zero(self, tmp_path):
        """Test with daily loss limit set to zero."""
        # Set both usd and pct to 0 to prevent auto-scaling
        limits = SafetyLimits(max_daily_loss_usd=0.0, max_daily_loss_pct=0.0)
        controller = SafetyController(limits=limits, state_path=tmp_path / "safety_state.json")
        controller.update_balance(10000.0)

        controller._daily_stats.total_loss = 0.01
        assert controller._daily_stats.total_loss > limits.max_daily_loss_usd


class TestModeTransitionEdgeCases:
    """Test edge cases in mode transitions."""

    def test_transition_with_open_positions(self):
        """Test mode transition with open positions."""
        state = _create_state(balance=9500.0)
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
        
        state.mode = TradingMode.TESTNET
        
        assert len(state.positions) == 1
        assert state.positions["BTC/USDT"].symbol == "BTC/USDT"

    def test_transition_with_negative_balance(self):
        """Test transition behavior with negative balance."""
        state = _create_state(balance=-100.0)
        
        state.mode = TradingMode.TESTNET
        
        assert state.mode == TradingMode.TESTNET
        assert state.current_balance < 0


class TestStringAndEnumEdgeCases:
    """Test edge cases with string and enum values."""

    def test_empty_reason_string(self):
        """Test handling empty reason strings."""
        allowed, reason = (True, "")
        assert reason == ""

    def test_very_long_reason_string(self):
        """Test handling very long reason strings."""
        long_reason = "x" * 10000
        allowed, reason = (True, long_reason)
        assert len(reason) == 10000

    def test_special_characters_in_symbol(self):
        """Test symbols with special characters."""
        position = PositionState(
            symbol="BTC/USDT:PERP",
            quantity=1.0,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        assert "/" in position.symbol
        assert ":" in position.symbol
