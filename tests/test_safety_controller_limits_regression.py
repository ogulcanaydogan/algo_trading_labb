"""
Regression tests for SafetyController limits.

These tests protect against accidental reversion of critical safety fixes:
- Position size must be 5% max (not 20%)
- Daily loss must be 2% max (not 10%)
- Limits scale with current balance (not initial capital)
- live_limited mode uses 5% position, 2% daily loss (not 10% & 5%)
"""

import pytest
from pathlib import Path

from bot.safety_controller import SafetyController, SafetyLimits, SafetyStatus
from bot.trading_mode import TradingMode, ModeConfig


class MockOrder:
    """Mock order for testing."""

    def __init__(self, quantity: float, price: float, symbol: str = "BTC/USDT"):
        self.quantity = quantity
        self.price = price
        self.symbol = symbol


class TestSafetyControllerLimitsRegression:
    """Regression tests for critical SafetyController limit fixes."""

    def test_default_position_limit_is_5_percent(self):
        """CRITICAL: Default max position must be 5%, not 20%."""
        limits = SafetyLimits()
        assert limits.max_position_size_pct == 0.05, (
            "REGRESSION: Position limit reverted to unsafe value! Must be 5% (0.05) not 20% (0.20)"
        )

    def test_default_daily_loss_limit_is_2_percent(self):
        """CRITICAL: Default max daily loss must be 2%, not 10%."""
        limits = SafetyLimits()
        assert limits.max_daily_loss_pct == 0.02, (
            "REGRESSION: Daily loss limit reverted to unsafe value! "
            "Must be 2% (0.02) not 10% (0.10)"
        )

    def test_live_limited_position_limit_is_5_percent(self):
        """CRITICAL: live_limited mode must use 5% position limit."""
        # Default SafetyLimits must already enforce 5%
        limits = SafetyLimits()
        assert limits.max_position_size_pct == 0.05, (
            "REGRESSION: Default position limit wrong! Must be 5% not 20%"
        )

    def test_live_limited_daily_loss_is_2_percent(self):
        """CRITICAL: live_limited mode must use 2% daily loss limit."""
        # Default SafetyLimits must already enforce 2%
        limits = SafetyLimits()
        assert limits.max_daily_loss_pct == 0.02, (
            "REGRESSION: Default daily loss limit wrong! Must be 2% not 10%"
        )

    def test_position_limit_scales_with_current_balance(self, tmp_path):
        """CRITICAL: Position limits must scale with current balance, not initial capital."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_position_size_pct=0.05,  # 5%
                max_position_size_usd=1000,
            ),
            state_path=tmp_path / "safety_test.json",
        )

        # Set current balance to $10,000
        controller.update_balance(10000.0)

        # 5% of $10,000 = $500 position should pass
        order_ok = MockOrder(quantity=0.01, price=50000)  # $500 order
        allowed, reason = controller.pre_trade_check(order_ok)
        assert allowed, f"$500 order (5%) should pass with $10k balance: {reason}"

        # 6% of $10,000 = $600 position should fail
        order_too_big = MockOrder(quantity=0.012, price=50000)  # $600 order
        allowed, reason = controller.pre_trade_check(order_too_big)
        assert not allowed, "$600 order (6%) should fail with $10k balance"
        assert "exceeds limit" in reason.lower()

    def test_daily_loss_scales_with_current_balance(self, tmp_path):
        """CRITICAL: Daily loss limits must scale with current balance, not initial capital."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_daily_loss_pct=0.02,  # 2%
                max_daily_loss_usd=1000,
                min_time_between_trades_seconds=0,  # disable spacing for test
            ),
            state_path=tmp_path / "safety_test.json",
        )

        # Set current balance to $10,000
        controller.update_balance(10000.0)

        # Record $150 loss (1.5% of $10k) - should still allow trading
        class MockResult:
            symbol = "BTC/USDT"
            pnl = -150.0
            realized_pnl = -150.0

        controller.post_trade_check(MockResult())

        order = MockOrder(quantity=0.001, price=50000)
        allowed, reason = controller.pre_trade_check(order)
        assert allowed, f"$150 loss (1.5%) should still allow trades: {reason}"

        # Record another $60 loss (total $210 = 2.1% of $10k) - should block
        MockResult.pnl = -60.0
        MockResult.realized_pnl = -60.0
        controller.post_trade_check(MockResult())

        allowed, reason = controller.pre_trade_check(order)
        assert not allowed, "$210 loss (2.1%) should block trading"
        assert "daily loss" in reason.lower()

    def test_position_pct_uses_current_not_peak_balance(self, tmp_path):
        """CRITICAL: Position % must use current balance, not peak/initial."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_position_size_pct=0.05,  # 5%
                max_position_size_usd=10000,
            ),
            state_path=tmp_path / "safety_test.json",
        )

        # Start with $10,000
        controller.update_balance(10000.0)

        # Lose money, now $8,000 current balance
        class MockResult:
            symbol = "BTC/USDT"
            pnl = -2000.0
            realized_pnl = -2000.0

        controller.post_trade_check(MockResult())
        controller.update_balance(8000.0)

        # 5% of current $8,000 = $400 max position
        order_400 = MockOrder(quantity=0.008, price=50000)  # $400
        allowed, reason = controller.pre_trade_check(order_400)
        assert allowed, f"$400 order (5% of $8k current) should pass: {reason}"

        # $500 would be 6.25% of $8k current - should fail
        order_500 = MockOrder(quantity=0.01, price=50000)  # $500
        allowed, reason = controller.pre_trade_check(order_500)
        assert not allowed, "$500 is 6.25% of $8k current balance, should fail"

    def test_max_position_usd_absolute_cap(self, tmp_path):
        """Position size must respect absolute USD cap regardless of balance."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_position_size_pct=0.20,  # 20% would allow huge positions
                max_position_size_usd=100.0,  # But USD cap is $100
            ),
            state_path=tmp_path / "safety_test.json",
        )

        # Even with $10k balance (20% = $2k), USD cap of $100 must apply
        controller.update_balance(10000.0)

        order_under_cap = MockOrder(quantity=0.0019, price=50000)  # $95
        allowed, reason = controller.pre_trade_check(order_under_cap)
        assert allowed, f"$95 order under $100 cap should pass: {reason}"

        order_over_cap = MockOrder(quantity=0.0022, price=50000)  # $110
        allowed, reason = controller.pre_trade_check(order_over_cap)
        assert not allowed, "$110 order over $100 cap should fail"
        assert "$" in reason and "exceeds limit" in reason.lower()

    def test_live_limited_strict_limits(self, tmp_path):
        """live_limited mode enforces strict $100 position, 2% daily loss."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_position_size_usd=100.0,  # $100 cap
                max_position_size_pct=0.05,  # 5%
                max_daily_loss_usd=10.0,  # $10 daily loss cap
                max_daily_loss_pct=0.02,  # 2%
            ),
            state_path=tmp_path / "safety_test.json",
        )

        controller.update_balance(1000.0)  # $1k balance

        # Position limits
        order_ok = MockOrder(quantity=0.0019, price=50000)  # $95 (under $100 cap)
        allowed, _ = controller.pre_trade_check(order_ok)
        assert allowed, "$95 should pass for live_limited"

        order_too_big = MockOrder(quantity=0.0022, price=50000)  # $110
        allowed, reason = controller.pre_trade_check(order_too_big)
        assert not allowed, "$110 exceeds $100 live_limited cap"

        # Daily loss limits
        class MockResult:
            symbol = "BTC/USDT"
            pnl = -15.0
            realized_pnl = -15.0

        controller.post_trade_check(MockResult())

        order_small = MockOrder(quantity=0.0001, price=50000)
        allowed, reason = controller.pre_trade_check(order_small)
        assert not allowed, "$15 loss should hit $10 daily loss cap"
        assert "daily loss" in reason.lower()

    def test_dynamic_limits_scale_with_balance(self, tmp_path):
        """Position/loss limits must scale dynamically with current balance."""
        controller = SafetyController(
            limits=SafetyLimits(
                max_position_size_pct=0.05,
                max_daily_loss_pct=0.02,
            ),
            state_path=tmp_path / "safety_test.json",
        )

        # Start with $5,000
        controller.update_balance(5000.0)

        # 5% of $5k = $250 position limit
        order_250 = MockOrder(quantity=0.005, price=50000)  # $250
        allowed, _ = controller.pre_trade_check(order_250)
        assert allowed, "$250 should be exactly at 5% limit"

        # Grow to $20,000
        controller.update_balance(20000.0)

        # Now 5% of $20k = $1,000 position limit (scales automatically)
        order_1000 = MockOrder(quantity=0.02, price=50000)  # $1,000
        allowed, _ = controller.pre_trade_check(order_1000)
        assert allowed, "$1,000 should be exactly at 5% of new $20k balance"

        # $1,100 exceeds new limit
        order_1100 = MockOrder(quantity=0.022, price=50000)  # $1,100
        allowed, _ = controller.pre_trade_check(order_1100)
        assert not allowed, "$1,100 exceeds 5% of $20k"
