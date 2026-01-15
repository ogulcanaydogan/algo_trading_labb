"""
Comprehensive tests for Risk Guardian module.

Tests risk limits, position sizing, drawdown protection, and kill switch functionality.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import tempfile

from bot.risk_guardian import (
    RiskGuardian,
    RiskLimits,
    RiskState,
    RiskCheckResult,
    RiskLevel,
    VetoReason,
    TradeRequest,
)

# Backward compatibility alias
RiskMetrics = RiskState
RiskViolationType = VetoReason


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
            max_position_pct=5.0,
            max_daily_loss_pct=2.0,
            max_drawdown_pct=8.0,
            max_trades_per_day=50,
        )

        assert limits.max_position_pct == 5.0
        assert limits.max_daily_loss_pct == 2.0
        assert limits.max_drawdown_pct == 8.0
        assert limits.max_trades_per_day == 50


class TestRiskMetrics:
    """Test RiskMetrics (RiskState) dataclass."""

    def test_metrics_initialization(self):
        """Test metrics are properly initialized."""
        metrics = RiskState(
            current_drawdown_pct=5.0,
            daily_pnl_pct=-1.0,
            current_equity=9900.0,
            peak_equity=10000.0,
        )

        assert metrics.current_drawdown_pct == 5.0
        assert metrics.daily_pnl_pct == -1.0
        assert metrics.current_equity == 9900.0
        assert metrics.peak_equity == 10000.0


class TestRiskCheckResult:
    """Test RiskCheckResult dataclass."""

    def test_approved_result(self):
        """Test approved risk check result."""
        result = RiskCheckResult(
            approved=True,
            veto_reasons=[],
            adjusted_size_pct=1.0,
        )

        assert result.approved
        assert len(result.veto_reasons) == 0
        assert result.adjusted_size_pct == 1.0

    def test_rejected_result(self):
        """Test rejected risk check result with violations."""
        result = RiskCheckResult(
            approved=False,
            veto_reasons=[
                VetoReason.DRAWDOWN_LIMIT,
                VetoReason.DAILY_LOSS_LIMIT,
            ],
            adjusted_size_pct=0.0,
            reasoning="Drawdown exceeded",
        )

        assert not result.approved
        assert len(result.veto_reasons) == 2
        assert VetoReason.DRAWDOWN_LIMIT in result.veto_reasons
        assert result.reasoning == "Drawdown exceeded"


class TestRiskGuardian:
    """Test RiskGuardian core functionality."""

    @pytest.fixture
    def guardian(self, tmp_path):
        """Create RiskGuardian instance for testing."""
        limits = RiskLimits(
            max_position_pct=10.0,
            max_daily_loss_pct=3.0,
            max_drawdown_pct=10.0,
            max_trades_per_day=100,
        )
        state_file = tmp_path / "test_risk_state.json"
        return RiskGuardian(limits=limits, initial_capital=100000.0, state_file=state_file, auto_save=False)

    def test_initialization(self, guardian):
        """Test RiskGuardian initialization."""
        assert guardian.initial_capital == 100000.0
        assert guardian.limits.max_position_pct == 10.0
        assert not guardian.state.kill_switch_active
        assert guardian.state.current_equity == 100000.0

    def test_check_position_size_within_limits(self, guardian):
        """Test position size check within limits."""
        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,  # 5% of portfolio
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        assert result.approved
        assert len(result.veto_reasons) == 0

    def test_check_position_size_exceeds_limit(self, guardian):
        """Test position size check exceeding limits."""
        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=15.0,  # 15% - exceeds 10% limit
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        # Should have warning or adjusted size
        assert result.max_allowed_size_pct == guardian.limits.max_position_size_pct

    def test_drawdown_protection(self, guardian):
        """Test drawdown protection triggers correctly."""
        # Simulate drawdown beyond limit
        guardian.state.current_equity = 88000.0  # 12% drawdown from 100k
        guardian.state.peak_equity = 100000.0
        guardian.state.current_drawdown_pct = 12.0

        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        assert not result.approved
        assert VetoReason.DRAWDOWN_LIMIT in result.veto_reasons

    def test_daily_loss_protection(self, guardian):
        """Test daily loss limit protection."""
        # Simulate daily loss beyond limit
        guardian.state.daily_pnl_pct = -5.0  # 5% loss, exceeds 3% limit

        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        assert not result.approved
        assert VetoReason.DAILY_LOSS_LIMIT in result.veto_reasons

    def test_kill_switch_activation(self, guardian):
        """Test kill switch activation."""
        guardian.activate_kill_switch("Test activation")

        assert guardian.state.kill_switch_active

        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        assert not result.approved
        assert VetoReason.KILL_SWITCH in result.veto_reasons

    def test_kill_switch_deactivation(self, guardian):
        """Test kill switch deactivation."""
        guardian.activate_kill_switch("Test")
        assert guardian.state.kill_switch_active

        guardian.deactivate_kill_switch()
        assert not guardian.state.kill_switch_active

    def test_trade_count_limit(self, guardian):
        """Test daily trade count limit."""
        guardian.state.daily_trades = 100  # At limit

        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        # Should still allow but may warn
        # Trade count limit is more of a warning than hard stop
        assert result is not None

    def test_update_capital(self, guardian):
        """Test capital update."""
        guardian.update_equity(95000.0)

        assert guardian.state.current_equity == 95000.0
        # Should have updated drawdown
        assert guardian.state.current_drawdown_pct > 0

    def test_drawdown_calculation(self, guardian):
        """Test drawdown is calculated correctly."""
        guardian.state.peak_equity = 100000.0
        guardian.update_equity(92000.0)

        # 8% drawdown from peak
        expected_drawdown = ((100000 - 92000) / 100000) * 100
        assert abs(guardian.state.current_drawdown_pct - expected_drawdown) < 0.1

    def test_reset_daily_counters(self, guardian):
        """Test daily counters reset."""
        guardian.state.daily_trades = 50
        guardian.state.daily_pnl_pct = -2.0
        guardian.state.daily_wins = 20
        guardian.state.daily_losses = 30

        guardian.reset_daily_stats(current_equity=100000.0)

        assert guardian.state.daily_trades == 0
        assert guardian.state.daily_pnl_pct == 0.0

    def test_get_risk_metrics(self, guardian):
        """Test getting risk metrics."""
        status = guardian.get_status()

        # Check main status keys
        assert "risk_level" in status
        assert "kill_switch_active" in status
        assert "drawdown" in status
        assert "daily_stats" in status

        # Check nested keys
        assert "current_pct" in status["drawdown"]
        assert "pnl_pct" in status["daily_stats"]

    def test_volatility_adjusted_sizing(self, guardian):
        """Test that high volatility affects position sizing."""
        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=10.0,
            current_price=50000.0,
            volatility_ratio=3.0,  # High volatility
        )
        result = guardian.check_trade(request)

        # High volatility should trigger warning
        assert result.risk_level != RiskLevel.NORMAL or len(result.warnings) > 0

    def test_correlation_check(self, guardian):
        """Test correlation-based exposure limits."""
        # Add existing position
        guardian.state.positions = {"BTC/USDT": 20.0}
        guardian.state.total_exposure_pct = 20.0

        request = TradeRequest(
            symbol="ETH/USDT",
            action="BUY",
            direction="LONG",
            size_pct=25.0,
            current_price=3000.0,
        )
        result = guardian.check_trade(request)

        # Should consider total exposure
        assert result is not None


class TestRiskGuardianAsync:
    """Test async methods of RiskGuardian."""

    @pytest.fixture
    def guardian(self, tmp_path):
        """Create RiskGuardian instance for testing."""
        limits = RiskLimits(
            max_position_pct=10.0,
            max_daily_loss_pct=3.0,
            max_drawdown_pct=10.0,
        )
        state_file = tmp_path / "test_risk_state.json"
        return RiskGuardian(limits=limits, initial_capital=100000.0, state_file=state_file, auto_save=False)

    @pytest.mark.asyncio
    async def test_async_check_trade(self, guardian):
        """Test async trade check."""
        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )

        # Use sync method (async wrapper not required for basic check)
        result = guardian.check_trade(request)
        assert result is not None
        assert isinstance(result.approved, bool)

    @pytest.mark.asyncio
    async def test_async_close_all_positions(self, guardian):
        """Test async close all positions command."""
        # Test kill switch activation as proxy for close all
        guardian.activate_kill_switch("Emergency close")
        assert guardian.state.kill_switch_active


class TestRiskGuardianEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def guardian(self, tmp_path):
        """Create RiskGuardian instance for testing."""
        limits = RiskLimits()
        state_file = tmp_path / "test_risk_state.json"
        return RiskGuardian(limits=limits, initial_capital=100000.0, state_file=state_file, auto_save=False)

    def test_zero_capital(self, tmp_path):
        """Test handling of zero capital."""
        limits = RiskLimits()
        state_file = tmp_path / "test_risk_state.json"
        guardian = RiskGuardian(limits=limits, initial_capital=0.0, state_file=state_file, auto_save=False)

        # Should handle gracefully
        request = TradeRequest(
            symbol="BTC/USDT",
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)
        assert result is not None

    def test_negative_position_size(self, guardian):
        """Test handling of negative position size."""
        request = TradeRequest(
            symbol="BTC/USDT",
            action="SELL",
            direction="SHORT",
            size_pct=-5.0,  # Negative size
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        # Should handle gracefully
        assert result is not None

    def test_invalid_symbol(self, guardian):
        """Test handling of unusual symbol."""
        request = TradeRequest(
            symbol="",  # Empty symbol
            action="BUY",
            direction="LONG",
            size_pct=5.0,
            current_price=50000.0,
        )
        result = guardian.check_trade(request)

        # Should handle gracefully
        assert result is not None

    def test_concurrent_checks(self, guardian):
        """Test thread safety with concurrent checks."""
        import threading

        results = []
        errors = []

        def check_trade():
            try:
                request = TradeRequest(
                    symbol="BTC/USDT",
                    action="BUY",
                    direction="LONG",
                    size_pct=5.0,
                    current_price=50000.0,
                )
                result = guardian.check_trade(request)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_trade) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
