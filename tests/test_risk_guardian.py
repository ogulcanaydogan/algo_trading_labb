"""
Comprehensive tests for Risk Guardian module.

Tests risk limits, position sizing, drawdown protection, and kill switch functionality.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np

from bot.risk_guardian import (
    RiskGuardian,
    RiskLimits,
    RiskMetrics,
    RiskCheckResult,
    RiskViolationType,
)


class TestRiskLimits:
    """Test RiskLimits dataclass."""

    def test_default_limits(self):
        """Test default risk limits are reasonable."""
        limits = RiskLimits()

        assert limits.max_position_pct > 0
        assert limits.max_daily_loss_pct > 0
        assert limits.max_drawdown_pct > 0
        assert limits.max_correlation > 0
        assert limits.max_trades_per_day > 0

    def test_custom_limits(self):
        """Test custom risk limits."""
        limits = RiskLimits(
            max_position_pct=0.05,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.08,
            max_trades_per_day=50,
        )

        assert limits.max_position_pct == 0.05
        assert limits.max_daily_loss_pct == 0.02
        assert limits.max_drawdown_pct == 0.08
        assert limits.max_trades_per_day == 50


class TestRiskMetrics:
    """Test RiskMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics are properly initialized."""
        metrics = RiskMetrics(
            current_drawdown=0.05,
            daily_pnl=-100.0,
            open_positions=3,
            total_exposure=5000.0,
            var_95=200.0,
        )

        assert metrics.current_drawdown == 0.05
        assert metrics.daily_pnl == -100.0
        assert metrics.open_positions == 3
        assert metrics.total_exposure == 5000.0
        assert metrics.var_95 == 200.0


class TestRiskCheckResult:
    """Test RiskCheckResult dataclass."""

    def test_approved_result(self):
        """Test approved risk check result."""
        result = RiskCheckResult(
            approved=True,
            violations=[],
            adjusted_size=1.0,
        )

        assert result.approved
        assert len(result.violations) == 0
        assert result.adjusted_size == 1.0

    def test_rejected_result(self):
        """Test rejected risk check result with violations."""
        result = RiskCheckResult(
            approved=False,
            violations=[
                RiskViolationType.MAX_DRAWDOWN,
                RiskViolationType.DAILY_LOSS_LIMIT,
            ],
            adjusted_size=0.0,
            reason="Drawdown exceeded",
        )

        assert not result.approved
        assert len(result.violations) == 2
        assert RiskViolationType.MAX_DRAWDOWN in result.violations
        assert result.reason == "Drawdown exceeded"


class TestRiskGuardian:
    """Test RiskGuardian core functionality."""

    @pytest.fixture
    def guardian(self):
        """Create RiskGuardian instance for testing."""
        limits = RiskLimits(
            max_position_pct=0.10,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_trades_per_day=100,
        )
        return RiskGuardian(limits=limits, initial_capital=100000.0)

    def test_initialization(self, guardian):
        """Test RiskGuardian initialization."""
        assert guardian.initial_capital == 100000.0
        assert guardian.limits.max_position_pct == 0.10
        assert not guardian.is_killed
        assert guardian.current_capital == 100000.0

    def test_check_position_size_within_limits(self, guardian):
        """Test position size check within limits."""
        result = guardian.check_position_size(
            symbol="BTC/USDT",
            requested_size=0.05,  # 5% of portfolio
            current_price=50000.0,
        )

        assert result.approved
        assert result.adjusted_size > 0

    def test_check_position_size_exceeds_limit(self, guardian):
        """Test position size check exceeding limits."""
        result = guardian.check_position_size(
            symbol="BTC/USDT",
            requested_size=0.15,  # 15% - exceeds 10% limit
            current_price=50000.0,
        )

        # Should adjust size down to limit
        assert result.adjusted_size <= guardian.limits.max_position_pct

    def test_drawdown_protection(self, guardian):
        """Test drawdown protection triggers correctly."""
        # Simulate drawdown
        guardian.current_capital = 88000.0  # 12% drawdown

        result = guardian.check_trade(
            symbol="BTC/USDT",
            side="BUY",
            size=0.05,
            price=50000.0,
        )

        assert not result.approved
        assert RiskViolationType.MAX_DRAWDOWN in result.violations

    def test_daily_loss_protection(self, guardian):
        """Test daily loss limit protection."""
        # Simulate daily loss
        guardian.daily_pnl = -3500.0  # 3.5% loss

        result = guardian.check_trade(
            symbol="BTC/USDT",
            side="BUY",
            size=0.05,
            price=50000.0,
        )

        assert not result.approved
        assert RiskViolationType.DAILY_LOSS_LIMIT in result.violations

    def test_kill_switch_activation(self, guardian):
        """Test kill switch can be activated."""
        guardian.activate_kill_switch(reason="Manual activation")

        assert guardian.is_killed
        assert guardian.kill_reason == "Manual activation"

        # All trades should be rejected when killed
        result = guardian.check_trade(
            symbol="BTC/USDT",
            side="BUY",
            size=0.01,
            price=50000.0,
        )

        assert not result.approved
        assert RiskViolationType.KILL_SWITCH in result.violations

    def test_kill_switch_deactivation(self, guardian):
        """Test kill switch can be deactivated."""
        guardian.activate_kill_switch(reason="Test")
        assert guardian.is_killed

        guardian.deactivate_kill_switch()
        assert not guardian.is_killed

    def test_trade_count_limit(self, guardian):
        """Test daily trade count limit."""
        guardian.limits.max_trades_per_day = 5

        # Simulate reaching trade limit
        for i in range(5):
            guardian.record_trade(
                symbol="BTC/USDT",
                side="BUY",
                size=0.01,
                price=50000.0,
            )

        result = guardian.check_trade(
            symbol="BTC/USDT",
            side="BUY",
            size=0.01,
            price=50000.0,
        )

        assert not result.approved
        assert RiskViolationType.TRADE_LIMIT in result.violations

    def test_update_capital(self, guardian):
        """Test capital update."""
        guardian.update_capital(110000.0)

        assert guardian.current_capital == 110000.0
        assert guardian.peak_capital == 110000.0

    def test_drawdown_calculation(self, guardian):
        """Test drawdown is calculated correctly."""
        guardian.update_capital(110000.0)  # New peak
        guardian.update_capital(99000.0)   # Drawdown

        drawdown = guardian.get_current_drawdown()
        expected = (110000.0 - 99000.0) / 110000.0

        assert abs(drawdown - expected) < 0.0001

    def test_reset_daily_counters(self, guardian):
        """Test daily counters reset."""
        guardian.daily_pnl = -1000.0
        guardian.daily_trade_count = 50

        guardian.reset_daily()

        assert guardian.daily_pnl == 0.0
        assert guardian.daily_trade_count == 0

    def test_get_risk_metrics(self, guardian):
        """Test getting current risk metrics."""
        guardian.current_capital = 95000.0
        guardian.daily_pnl = -500.0

        metrics = guardian.get_metrics()

        assert isinstance(metrics, RiskMetrics)
        assert metrics.current_drawdown > 0
        assert metrics.daily_pnl == -500.0

    def test_volatility_adjusted_sizing(self, guardian):
        """Test position sizing adjusts for volatility."""
        # High volatility should reduce position size
        result_high_vol = guardian.check_position_size(
            symbol="BTC/USDT",
            requested_size=0.10,
            current_price=50000.0,
            volatility=0.05,  # 5% daily volatility
        )

        result_low_vol = guardian.check_position_size(
            symbol="BTC/USDT",
            requested_size=0.10,
            current_price=50000.0,
            volatility=0.01,  # 1% daily volatility
        )

        # Higher volatility should result in smaller position
        assert result_high_vol.adjusted_size <= result_low_vol.adjusted_size

    def test_correlation_check(self, guardian):
        """Test correlation limit checking."""
        # Add existing correlated position
        guardian.add_position(
            symbol="BTC/USDT",
            size=0.05,
            correlation_group="crypto",
        )

        guardian.add_position(
            symbol="ETH/USDT",
            size=0.05,
            correlation_group="crypto",
        )

        # Check if adding more crypto is allowed
        result = guardian.check_correlation(
            symbol="SOL/USDT",
            correlation_group="crypto",
            proposed_size=0.05,
        )

        # Should flag high correlation exposure
        assert "crypto" in result.get("warnings", []) or result.get("total_correlated", 0) > 0


class TestRiskGuardianAsync:
    """Test async methods of RiskGuardian."""

    @pytest.fixture
    def guardian(self):
        """Create RiskGuardian for async tests."""
        return RiskGuardian(
            limits=RiskLimits(),
            initial_capital=100000.0,
        )

    @pytest.mark.asyncio
    async def test_async_check_trade(self, guardian):
        """Test async trade check."""
        result = await guardian.check_trade_async(
            symbol="BTC/USDT",
            side="BUY",
            size=0.05,
            price=50000.0,
        )

        assert isinstance(result, RiskCheckResult)

    @pytest.mark.asyncio
    async def test_async_close_all_positions(self, guardian):
        """Test async close all positions."""
        # Add some positions
        guardian.add_position("BTC/USDT", 0.05, "crypto")
        guardian.add_position("ETH/USDT", 0.03, "crypto")

        close_orders = await guardian.close_all_positions_async()

        assert len(close_orders) >= 2


class TestRiskGuardianEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def guardian(self):
        """Create RiskGuardian for edge case tests."""
        return RiskGuardian(
            limits=RiskLimits(),
            initial_capital=100000.0,
        )

    def test_zero_capital(self):
        """Test handling of zero capital."""
        with pytest.raises(ValueError):
            RiskGuardian(limits=RiskLimits(), initial_capital=0.0)

    def test_negative_position_size(self, guardian):
        """Test handling of negative position size."""
        result = guardian.check_position_size(
            symbol="BTC/USDT",
            requested_size=-0.05,
            current_price=50000.0,
        )

        # Should handle gracefully (reject or convert to absolute)
        assert result.adjusted_size >= 0

    def test_invalid_symbol(self, guardian):
        """Test handling of invalid symbol."""
        result = guardian.check_trade(
            symbol="",
            side="BUY",
            size=0.05,
            price=50000.0,
        )

        assert not result.approved

    def test_concurrent_checks(self, guardian):
        """Test thread safety of risk checks."""
        import threading

        results = []

        def check_trade():
            result = guardian.check_trade(
                symbol="BTC/USDT",
                side="BUY",
                size=0.01,
                price=50000.0,
            )
            results.append(result)

        threads = [threading.Thread(target=check_trade) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
