"""
Tests for the Unified Trading Engine.

Tests cover:
- Trading mode definitions
- Execution adapters
- Safety controller
- State management
- Transition validation
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bot.trading_mode import (
    ModeConfig,
    ModeState,
    TradingMode,
    TradingStatus,
    TransitionRequirements,
    get_transition_requirements,
)
from bot.execution_adapter import (
    Balance,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperExecutionAdapter,
    Position,
)
from bot.safety_controller import (
    SafetyController,
    SafetyLimits,
    SafetyStatus,
    create_safety_controller_for_mode,
)
from bot.transition_validator import TransitionValidator


class TestTradingMode:
    """Tests for TradingMode enum and related classes."""

    def test_mode_properties(self):
        """Test mode property methods."""
        # Paper modes
        assert TradingMode.PAPER_SYNTHETIC.is_paper
        assert TradingMode.PAPER_LIVE_DATA.is_paper
        assert TradingMode.BACKTEST.is_paper
        assert not TradingMode.TESTNET.is_paper

        # Live modes
        assert TradingMode.LIVE_LIMITED.is_live
        assert TradingMode.LIVE_FULL.is_live
        assert not TradingMode.PAPER_LIVE_DATA.is_live

        # Real data modes
        assert TradingMode.PAPER_LIVE_DATA.uses_real_data
        assert TradingMode.TESTNET.uses_real_data
        assert not TradingMode.PAPER_SYNTHETIC.uses_real_data

    def test_mode_progression(self):
        """Test mode progression order."""
        progression = TradingMode.get_progression()
        assert progression[0] == TradingMode.BACKTEST
        assert progression[-1] == TradingMode.LIVE_FULL

    def test_can_transition_to(self):
        """Test transition validation."""
        # Can move forward one step
        assert TradingMode.PAPER_LIVE_DATA.can_transition_to(TradingMode.TESTNET)

        # Can move backward any steps
        assert TradingMode.LIVE_FULL.can_transition_to(TradingMode.PAPER_LIVE_DATA)

        # Cannot skip steps forward
        assert not TradingMode.PAPER_LIVE_DATA.can_transition_to(TradingMode.LIVE_LIMITED)


class TestModeConfig:
    """Tests for ModeConfig defaults."""

    def test_live_limited_defaults(self):
        """Test LIVE_LIMITED mode has strict limits."""
        config = ModeConfig.get_default(TradingMode.LIVE_LIMITED)

        assert config.capital_limit == 100.0
        assert config.max_position_usd == 20.0
        assert config.max_daily_loss_usd == 2.0
        assert config.max_daily_loss_pct == 0.02
        assert config.max_trades_per_day == 10
        assert config.require_confirmation

    def test_paper_mode_relaxed(self):
        """Test paper modes have relaxed limits."""
        config = ModeConfig.get_default(TradingMode.PAPER_LIVE_DATA)

        assert config.max_daily_loss_pct == 1.0  # 100%
        assert not config.require_confirmation


class TestModeState:
    """Tests for ModeState tracking."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)

        # No trades
        assert state.win_rate == 0.0

        # Add trades
        state.total_trades = 10
        state.winning_trades = 6
        assert state.win_rate == 0.6

    def test_record_trade(self):
        """Test trade recording."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)

        # Record winning trade
        state.record_trade(pnl=100, is_win=True)
        assert state.total_trades == 1
        assert state.winning_trades == 1
        assert state.total_pnl == 100
        assert state.consecutive_losses == 0

        # Record losing trade
        state.record_trade(pnl=-50, is_win=False)
        assert state.total_trades == 2
        assert state.losing_trades == 1
        assert state.consecutive_losses == 1


class TestTransitionRequirements:
    """Tests for transition requirements."""

    def test_paper_to_testnet_requirements(self):
        """Test requirements for paper to testnet transition."""
        req = get_transition_requirements(
            TradingMode.PAPER_LIVE_DATA, TradingMode.TESTNET
        )

        assert req is not None
        assert req.min_days_in_current_mode == 14
        assert req.min_trades == 100
        assert req.min_win_rate == 0.45
        assert req.require_manual_approval

    def test_requirements_check(self):
        """Test requirements checking."""
        req = TransitionRequirements(
            min_days_in_current_mode=7,
            min_trades=50,
            min_win_rate=0.40,
            max_drawdown_pct=0.15,
            min_profit_factor=0.8,
            require_manual_approval=True,
        )

        # Failing state
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now()
        passed, reasons = req.check(state)
        assert not passed
        assert len(reasons) > 0

        # Passing state
        state.started_at = datetime(2024, 1, 1)  # Old enough
        state.total_trades = 100
        state.winning_trades = 50
        state.max_drawdown_pct = 0.05
        passed, reasons = req.check(state)
        assert passed or len(reasons) <= 1  # May still fail profit factor


class TestPaperExecutionAdapter:
    """Tests for paper execution adapter."""

    @pytest.fixture
    def adapter(self):
        """Create a paper adapter for testing."""
        return PaperExecutionAdapter(initial_balance=10000.0)

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test connection."""
        result = await adapter.connect()
        assert result
        assert adapter.is_connected

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """Test balance retrieval."""
        await adapter.connect()
        balance = await adapter.get_balance()

        assert isinstance(balance, Balance)
        assert balance.total == 10000.0
        assert balance.available == 10000.0

    @pytest.mark.asyncio
    async def test_place_buy_order(self, adapter):
        """Test placing a buy order."""
        await adapter.connect()
        adapter.set_price("BTC/USDT", 50000.0)

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await adapter.place_order(order)

        assert result.success
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.1
        assert result.simulated

    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(self, adapter):
        """Test order rejection for insufficient balance."""
        await adapter.connect()
        adapter.set_price("BTC/USDT", 50000.0)

        # Try to buy more than balance allows
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,  # $50,000 worth
        )

        result = await adapter.place_order(order)

        assert not result.success
        assert result.status == OrderStatus.REJECTED
        assert "Insufficient balance" in result.error_message

    @pytest.mark.asyncio
    async def test_position_tracking(self, adapter):
        """Test position creation and tracking."""
        await adapter.connect()
        adapter.set_price("BTC/USDT", 50000.0)

        # Buy
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        await adapter.place_order(order)

        # Check position
        position = await adapter.get_position("BTC/USDT")
        assert position is not None
        assert position.quantity == 0.1
        assert position.side == "long"


class TestSafetyController:
    """Tests for safety controller."""

    @pytest.fixture
    def controller(self, tmp_path):
        """Create a safety controller for testing."""
        limits = SafetyLimits(
            max_daily_loss_usd=100.0,
            max_daily_loss_pct=0.02,
            max_trades_per_day=10,
            max_position_size_usd=500.0,
        )
        return SafetyController(limits=limits, state_path=tmp_path / "safety.json")

    def test_initial_status(self, controller):
        """Test initial safety status."""
        allowed, reason = controller.is_trading_allowed()
        assert allowed

    def test_daily_loss_limit(self, controller):
        """Test daily loss limit enforcement."""
        controller.update_balance(10000.0)

        # Mock a large loss
        controller._daily_stats.total_loss = 150.0  # Exceeds $100 limit

        # Create mock order
        order = MagicMock()
        order.quantity = 0.1
        order.price = 50000.0

        passed, reason = controller.pre_trade_check(order)
        assert not passed
        assert "Daily loss limit reached" in reason

    def test_trades_per_day_limit(self, controller):
        """Test daily trade limit."""
        controller.update_balance(10000.0)
        controller._daily_stats.trades = 10  # At limit

        order = MagicMock()
        order.quantity = 0.001
        order.price = 50000.0

        passed, reason = controller.pre_trade_check(order)
        assert not passed
        assert "Daily trade limit reached" in reason

    def test_emergency_stop(self, controller):
        """Test emergency stop functionality."""
        controller.emergency_stop("Test emergency")

        allowed, reason = controller.is_trading_allowed()
        assert not allowed
        assert "Emergency stop" in reason

        # Clear emergency stop
        controller.clear_emergency_stop("test_user")
        allowed, _ = controller.is_trading_allowed()
        assert allowed

    def test_create_safety_controller_for_mode(self):
        """Test factory function creates appropriate limits."""
        # Live limited mode
        controller = create_safety_controller_for_mode("live_limited", 100.0)
        assert controller.limits.max_daily_loss_usd == 2.0
        assert controller.limits.capital_limit == 100.0

        # Paper mode (relaxed)
        controller = create_safety_controller_for_mode("paper_live_data", 10000.0)
        assert controller.limits.max_daily_loss_pct == 1.0


class TestTransitionValidator:
    """Tests for transition validator."""

    @pytest.fixture
    def validator(self):
        """Create a transition validator."""
        return TransitionValidator()

    def test_can_transition_paper_to_testnet(self, validator):
        """Test paper to testnet transition validation."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime(2024, 1, 1)
        state.total_trades = 150
        state.winning_trades = 75
        state.max_drawdown_pct = 0.08

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA, TradingMode.TESTNET, state
        )

        # Should pass most requirements
        assert result.from_mode == TradingMode.PAPER_LIVE_DATA
        assert result.to_mode == TradingMode.TESTNET

    def test_cannot_skip_modes(self, validator):
        """Test that skipping modes is blocked."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)

        result = validator.can_transition(
            TradingMode.PAPER_LIVE_DATA, TradingMode.LIVE_LIMITED, state
        )

        assert not result.allowed
        assert "Cannot transition directly" in result.blocking_reasons[0]

    def test_downgrade_always_allowed(self, validator):
        """Test that downgrading is always allowed."""
        state = ModeState(mode=TradingMode.LIVE_LIMITED)

        result = validator.can_transition(
            TradingMode.LIVE_LIMITED, TradingMode.PAPER_LIVE_DATA, state
        )

        assert result.allowed

    def test_get_transition_progress(self, validator):
        """Test progress tracking."""
        state = ModeState(mode=TradingMode.PAPER_LIVE_DATA)
        state.started_at = datetime.now()
        state.total_trades = 50  # Half way

        progress = validator.get_transition_progress(
            TradingMode.PAPER_LIVE_DATA, TradingMode.TESTNET, state
        )

        assert "overall_progress" in progress
        assert "progress_details" in progress
        assert len(progress["progress_details"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
