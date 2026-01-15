"""
Tests for DCA Manager Module.
"""

import pytest
from datetime import datetime, timedelta

from bot.dca_manager import (
    DCAManager,
    DCAConfig,
    DCAState,
    create_dca_manager,
)


class TestDCAConfig:
    """Test DCAConfig."""

    def test_default_config(self):
        config = DCAConfig()
        assert config.enabled is True
        assert config.max_dca_orders == 4
        assert config.max_position_multiplier == 3.0
        assert len(config.dca_levels) == 4

    def test_dca_levels(self):
        config = DCAConfig()
        levels = config.dca_levels

        # Check default levels
        assert levels[0]["drawdown_pct"] == 0.03
        assert levels[1]["drawdown_pct"] == 0.05
        assert levels[2]["drawdown_pct"] == 0.08
        assert levels[3]["drawdown_pct"] == 0.12


class TestDCAState:
    """Test DCAState."""

    def test_state_creation(self):
        state = DCAState(
            symbol="BTC/USDT",
            original_entry_price=50000,
            original_quantity=0.1,
            average_entry_price=50000,
            total_quantity=0.1,
            total_cost=5000,
        )
        assert state.symbol == "BTC/USDT"
        assert state.dca_orders_count == 0

    def test_current_size_multiplier(self):
        state = DCAState(
            symbol="BTC/USDT",
            original_entry_price=50000,
            original_quantity=0.1,
            average_entry_price=49000,
            total_quantity=0.2,  # Doubled
            total_cost=9800,
        )
        assert state.current_size_multiplier == 2.0

    def test_state_to_dict(self):
        state = DCAState(
            symbol="ETH/USDT",
            original_entry_price=3000,
            original_quantity=1.0,
            average_entry_price=3000,
            total_quantity=1.0,
            total_cost=3000,
        )
        d = state.to_dict()
        assert d["symbol"] == "ETH/USDT"
        assert "created_at" in d

    def test_state_from_dict(self):
        data = {
            "symbol": "SOL/USDT",
            "original_entry_price": 100,
            "original_quantity": 10,
            "average_entry_price": 100,
            "total_quantity": 10,
            "total_cost": 1000,
            "dca_orders_count": 0,
            "dca_history": [],
            "last_dca_time": None,
            "created_at": datetime.now().isoformat(),
        }
        state = DCAState.from_dict(data)
        assert state.symbol == "SOL/USDT"


class TestDCAManager:
    """Test DCAManager."""

    def test_add_position(self):
        manager = create_dca_manager()
        state = manager.add_position(
            symbol="BTC/USDT",
            entry_price=50000,
            quantity=0.1,
        )

        assert state.symbol == "BTC/USDT"
        assert state.original_entry_price == 50000
        assert state.total_cost == 5000

    def test_no_dca_when_not_in_drawdown(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # Price at entry - no drawdown
        result = manager.check_dca_opportunity("BTC/USDT", 50000, 1000)
        assert result is None

    def test_dca_at_3_percent_drawdown(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # Price 4% below entry
        result = manager.check_dca_opportunity("BTC/USDT", 48000, 1000)

        assert result is not None
        assert result["symbol"] == "BTC/USDT"
        assert result["action"] == "DCA_BUY"
        assert result["level_drawdown"] == 0.03

    def test_dca_at_5_percent_drawdown(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # First DCA at 3%
        manager.execute_dca("BTC/USDT", 0.03, 48500, 0.03)

        # Manually set last_dca_time to past to bypass interval check
        manager.positions["BTC/USDT"].last_dca_time = None

        # Price 6% below original entry
        result = manager.check_dca_opportunity("BTC/USDT", 47000, 1000)

        assert result is not None
        assert result["level_drawdown"] == 0.05

    def test_max_dca_orders_limit(self):
        config = DCAConfig(max_dca_orders=2)
        manager = DCAManager(config)
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # Execute 2 DCAs
        manager.execute_dca("BTC/USDT", 0.03, 48500, 0.03)
        manager.execute_dca("BTC/USDT", 0.04, 47500, 0.05)

        # Third should be blocked
        result = manager.check_dca_opportunity("BTC/USDT", 44000, 1000)
        assert result is None

    def test_max_position_multiplier_limit(self):
        config = DCAConfig(max_position_multiplier=1.5)
        manager = DCAManager(config)
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # Execute DCA that brings total to 1.5x
        manager.execute_dca("BTC/USDT", 0.05, 47500, 0.03)

        # Manually set to max
        manager.positions["BTC/USDT"].total_quantity = 0.15

        # Should be blocked due to max multiplier
        result = manager.check_dca_opportunity("BTC/USDT", 44000, 1000)
        assert result is None

    def test_execute_dca(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        state = manager.execute_dca(
            symbol="BTC/USDT",
            quantity=0.05,
            price=48000,
            level_drawdown=0.03,
        )

        assert state.dca_orders_count == 1
        assert abs(state.total_quantity - 0.15) < 0.0001  # Float tolerance
        assert len(state.dca_history) == 1

    def test_average_entry_calculation(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # DCA at lower price
        manager.execute_dca("BTC/USDT", 0.1, 45000, 0.03)

        state = manager.positions["BTC/USDT"]
        # New average should be between 45000 and 50000
        assert 45000 < state.average_entry_price < 50000

    def test_remove_position(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        assert "BTC/USDT" in manager.positions
        manager.remove_position("BTC/USDT")
        assert "BTC/USDT" not in manager.positions

    def test_get_summary(self):
        manager = create_dca_manager()
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)
        manager.execute_dca("BTC/USDT", 0.05, 48000, 0.03)

        summary = manager.get_summary("BTC/USDT", 49000)

        assert summary is not None
        assert "unrealized_pnl" in summary
        assert "dca_orders" in summary
        assert summary["dca_orders"] == 1

    def test_min_dca_interval(self):
        config = DCAConfig(min_dca_interval=3600)  # 1 hour
        manager = DCAManager(config)
        manager.add_position("BTC/USDT", entry_price=50000, quantity=0.1)

        # First DCA
        manager.execute_dca("BTC/USDT", 0.03, 48500, 0.03)

        # Immediately try another - should be blocked
        result = manager.check_dca_opportunity("BTC/USDT", 47000, 1000)
        assert result is None


class TestFactoryFunction:
    """Test factory function."""

    def test_create_dca_manager(self):
        manager = create_dca_manager(
            max_dca_orders=3,
            max_position_multiplier=2.5,
        )

        assert manager.config.max_dca_orders == 3
        assert manager.config.max_position_multiplier == 2.5
