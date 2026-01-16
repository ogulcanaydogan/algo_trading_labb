"""
Tests for safety controller module.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from bot.safety_controller import (
    SafetyStatus,
    SafetyLimits,
    SafetyCheckResult,
    DailyStats,
    SafetyController,
)


class TestSafetyStatus:
    """Test SafetyStatus enum."""

    def test_all_statuses_exist(self):
        """Test all safety statuses exist."""
        assert SafetyStatus.OK.value == "ok"
        assert SafetyStatus.WARNING.value == "warning"
        assert SafetyStatus.BLOCKED.value == "blocked"
        assert SafetyStatus.EMERGENCY_STOP.value == "emergency_stop"


class TestSafetyLimits:
    """Test SafetyLimits dataclass."""

    def test_default_limits(self):
        """Test default limit values."""
        limits = SafetyLimits()
        assert limits.max_position_size_usd == 20.0
        assert limits.max_daily_loss_usd == 2.0
        assert limits.max_trades_per_day == 10
        assert limits.max_consecutive_losses == 5

    def test_custom_limits(self):
        """Test custom limit values."""
        limits = SafetyLimits(
            max_position_size_usd=100.0,
            max_daily_loss_usd=10.0,
            max_trades_per_day=20,
        )
        assert limits.max_position_size_usd == 100.0
        assert limits.max_daily_loss_usd == 10.0
        assert limits.max_trades_per_day == 20

    def test_position_limits(self):
        """Test position limit fields."""
        limits = SafetyLimits(
            max_position_size_pct=0.10,
            max_open_positions=5,
        )
        assert limits.max_position_size_pct == 0.10
        assert limits.max_open_positions == 5

    def test_balance_protection(self):
        """Test balance protection limits."""
        limits = SafetyLimits(
            min_balance_reserve_pct=0.30,
            min_balance_reserve_usd=50.0,
        )
        assert limits.min_balance_reserve_pct == 0.30
        assert limits.min_balance_reserve_usd == 50.0

    def test_emergency_triggers(self):
        """Test emergency trigger settings."""
        limits = SafetyLimits(
            emergency_stop_loss_pct=0.10,
            max_consecutive_losses=10,
            max_api_errors=5,
        )
        assert limits.emergency_stop_loss_pct == 0.10
        assert limits.max_consecutive_losses == 10


class TestSafetyCheckResult:
    """Test SafetyCheckResult dataclass."""

    def test_passed_result(self):
        """Test passed check result."""
        result = SafetyCheckResult(
            passed=True,
            status=SafetyStatus.OK,
            reason="All checks passed",
        )
        assert result.passed is True
        assert result.status == SafetyStatus.OK

    def test_failed_result(self):
        """Test failed check result."""
        result = SafetyCheckResult(
            passed=False,
            status=SafetyStatus.BLOCKED,
            reason="Daily loss limit exceeded",
            details={"daily_loss": 5.0, "limit": 2.0},
        )
        assert result.passed is False
        assert result.status == SafetyStatus.BLOCKED
        assert result.details["daily_loss"] == 5.0


class TestDailyStats:
    """Test DailyStats dataclass."""

    def test_default_stats(self):
        """Test default daily stats."""
        stats = DailyStats(date="2024-01-15")
        assert stats.date == "2024-01-15"
        assert stats.trades == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.total_pnl == 0.0

    def test_stats_with_data(self):
        """Test stats with trading data."""
        stats = DailyStats(
            date="2024-01-15",
            trades=5,
            wins=3,
            losses=2,
            total_pnl=50.0,
            total_loss=20.0,
            consecutive_losses=0,
        )
        assert stats.trades == 5
        assert stats.wins == 3
        assert stats.losses == 2
        assert stats.total_pnl == 50.0

    def test_hourly_trades_tracking(self):
        """Test hourly trades tracking."""
        stats = DailyStats(
            date="2024-01-15",
            hourly_trades={9: 2, 10: 3, 14: 1},
        )
        assert stats.hourly_trades[9] == 2
        assert stats.hourly_trades[10] == 3
        assert 14 in stats.hourly_trades


class TestSafetyController:
    """Test SafetyController class."""

    @pytest.fixture
    def temp_state_path(self):
        """Create temporary state path."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "safety_state.json"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def controller(self, temp_state_path):
        """Create safety controller."""
        return SafetyController(state_path=temp_state_path)

    def test_controller_creation(self, controller):
        """Test controller is created."""
        assert controller is not None
        assert controller.limits is not None
        assert controller._status == SafetyStatus.OK

    def test_controller_with_custom_limits(self, temp_state_path):
        """Test controller with custom limits."""
        limits = SafetyLimits(
            max_position_size_usd=50.0,
            max_daily_loss_usd=5.0,
        )
        controller = SafetyController(limits=limits, state_path=temp_state_path)

        assert controller.limits.max_position_size_usd == 50.0
        assert controller.limits.max_daily_loss_usd == 5.0

    def test_update_balance(self, controller):
        """Test updating balance."""
        controller.update_balance(1000.0)
        # Check internal state was updated
        assert controller._current_balance == 1000.0

    def test_emergency_stop(self, controller):
        """Test emergency stop activation."""
        controller.emergency_stop("Manual stop for testing")

        assert controller._emergency_stop_active is True
        assert controller._status == SafetyStatus.EMERGENCY_STOP

    def test_emergency_stop_blocks_trading(self, controller):
        """Test emergency stop blocks trading."""
        controller.emergency_stop("Test stop")

        allowed, reason = controller.is_trading_allowed()
        assert allowed is False
        assert "emergency" in reason.lower()

    def test_clear_emergency_stop(self, controller):
        """Test clearing emergency stop."""
        controller.emergency_stop("Test stop")
        assert controller._emergency_stop_active is True

        controller.clear_emergency_stop("admin")
        assert controller._emergency_stop_active is False

    def test_get_status(self, controller):
        """Test getting status."""
        status = controller.get_status()

        assert "status" in status
        assert "emergency_stop_active" in status
        assert "daily_stats" in status

    def test_record_api_error(self, controller):
        """Test recording API errors."""
        controller.record_api_error()

        assert controller._daily_stats.api_errors == 1

    def test_clear_api_errors(self, controller):
        """Test clearing API errors."""
        controller.record_api_error()
        controller.record_api_error()
        assert controller._daily_stats.api_errors == 2

        controller.clear_api_errors()
        assert controller._daily_stats.api_errors == 0

    def test_is_trading_allowed_ok(self, controller):
        """Test is_trading_allowed when OK."""
        allowed, reason = controller.is_trading_allowed()
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)

    def test_get_remaining_capacity(self, controller):
        """Test getting remaining capacity."""
        controller.update_balance(1000.0)
        capacity = controller.get_remaining_capacity()

        assert "trades_remaining" in capacity
        assert "loss_remaining_usd" in capacity


class TestSafetyControllerPersistence:
    """Test state persistence."""

    @pytest.fixture
    def temp_state_path(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "safety_state.json"
        shutil.rmtree(temp_dir)

    def test_emergency_stop_persisted(self, temp_state_path):
        """Test emergency stop is persisted."""
        controller1 = SafetyController(state_path=temp_state_path)
        controller1.emergency_stop("Test persistence")

        # Create new controller - should load persisted state
        controller2 = SafetyController(state_path=temp_state_path)

        assert controller2._emergency_stop_active is True
