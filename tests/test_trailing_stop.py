"""
Tests for Trailing Stop Module.
"""

import pytest
from datetime import datetime

from bot.trailing_stop import (
    TrailingStopManager,
    TrailingStopConfig,
    TrailingStopState,
    create_trailing_stop_manager,
)


class TestTrailingStopConfig:
    """Test TrailingStopConfig."""

    def test_default_config(self):
        config = TrailingStopConfig()
        assert config.enabled is True
        assert config.initial_stop_pct == 0.02
        assert config.trailing_pct == 0.015
        assert config.activation_profit_pct == 0.01

    def test_custom_config(self):
        config = TrailingStopConfig(
            initial_stop_pct=0.03,
            trailing_pct=0.02,
        )
        assert config.initial_stop_pct == 0.03
        assert config.trailing_pct == 0.02


class TestTrailingStopState:
    """Test TrailingStopState."""

    def test_state_creation(self):
        state = TrailingStopState(
            symbol="BTC/USDT",
            entry_price=50000,
            current_stop=49000,
            highest_price=50000,
        )
        assert state.symbol == "BTC/USDT"
        assert state.entry_price == 50000
        assert state.is_activated is False

    def test_state_to_dict(self):
        state = TrailingStopState(
            symbol="ETH/USDT",
            entry_price=3000,
            current_stop=2940,
            highest_price=3000,
        )
        d = state.to_dict()
        assert d["symbol"] == "ETH/USDT"
        assert d["entry_price"] == 3000
        assert "last_update" in d

    def test_state_from_dict(self):
        data = {
            "symbol": "SOL/USDT",
            "entry_price": 100,
            "current_stop": 98,
            "highest_price": 100,
            "is_activated": False,
            "is_at_breakeven": False,
            "last_update": datetime.now().isoformat(),
        }
        state = TrailingStopState.from_dict(data)
        assert state.symbol == "SOL/USDT"
        assert state.entry_price == 100


class TestTrailingStopManager:
    """Test TrailingStopManager."""

    def test_add_position(self):
        manager = create_trailing_stop_manager()
        state = manager.add_position("BTC/USDT", entry_price=50000)

        assert state.symbol == "BTC/USDT"
        assert state.entry_price == 50000
        # Initial stop at 2% below entry
        assert state.current_stop == 50000 * 0.98

    def test_update_no_change(self):
        manager = create_trailing_stop_manager()
        manager.add_position("BTC/USDT", entry_price=50000)

        # Price unchanged
        result = manager.update("BTC/USDT", 50000)
        assert result["action"] == "none"

    def test_stop_triggered(self):
        manager = create_trailing_stop_manager()
        manager.add_position("BTC/USDT", entry_price=50000)

        # Price drops below stop
        result = manager.update("BTC/USDT", 48000)
        assert result["action"] == "stop_triggered"

    def test_trailing_activation(self):
        manager = create_trailing_stop_manager(
            activation_profit_pct=0.01,  # Activate at 1% profit
        )
        manager.add_position("BTC/USDT", entry_price=50000)

        # Price rises 2% - should activate trailing
        result = manager.update("BTC/USDT", 51000)
        assert result.get("trailing_activated") is True
        assert manager.stops["BTC/USDT"].is_activated is True

    def test_trailing_stop_raised(self):
        manager = create_trailing_stop_manager(
            activation_profit_pct=0.01,
            trailing_pct=0.015,
        )
        manager.add_position("BTC/USDT", entry_price=50000)

        # Activate trailing
        manager.update("BTC/USDT", 51000)

        # Price rises more
        result = manager.update("BTC/USDT", 52000)

        # Stop should be raised
        assert result["action"] in ("stop_raised", "none")
        assert manager.stops["BTC/USDT"].highest_price == 52000

    def test_move_to_breakeven(self):
        config = TrailingStopConfig(
            move_to_breakeven_pct=0.015,  # Move to BE at 1.5% profit
        )
        manager = TrailingStopManager(config)
        manager.add_position("BTC/USDT", entry_price=50000)

        # Price rises 2%
        result = manager.update("BTC/USDT", 51000)

        assert manager.stops["BTC/USDT"].is_at_breakeven is True

    def test_remove_position(self):
        manager = create_trailing_stop_manager()
        manager.add_position("BTC/USDT", entry_price=50000)

        assert "BTC/USDT" in manager.stops
        manager.remove_position("BTC/USDT")
        assert "BTC/USDT" not in manager.stops

    def test_get_stop_price(self):
        manager = create_trailing_stop_manager()
        manager.add_position("BTC/USDT", entry_price=50000)

        stop = manager.get_stop_price("BTC/USDT")
        assert stop == 50000 * 0.98

        # Non-existent symbol
        assert manager.get_stop_price("UNKNOWN") is None

    def test_should_close_position(self):
        manager = create_trailing_stop_manager()
        manager.add_position("BTC/USDT", entry_price=50000)

        # Price above stop
        assert manager.should_close_position("BTC/USDT", 50000) is False

        # Price below stop
        assert manager.should_close_position("BTC/USDT", 48000) is True

    def test_short_position(self):
        manager = create_trailing_stop_manager()
        state = manager.add_position("BTC/USDT", entry_price=50000, side="short")

        # For short, stop is above entry
        assert state.current_stop == 50000 * 1.02

        # Price rises - stop triggered for short
        result = manager.update("BTC/USDT", 52000, side="short")
        assert result["action"] == "stop_triggered"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_trailing_stop_manager(self):
        manager = create_trailing_stop_manager(
            initial_stop_pct=0.03,
            trailing_pct=0.02,
            activation_profit_pct=0.015,
        )

        assert manager.config.initial_stop_pct == 0.03
        assert manager.config.trailing_pct == 0.02
        assert manager.config.activation_profit_pct == 0.015
