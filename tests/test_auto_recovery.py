"""
Tests for Auto-Recovery Service.

Tests automatic failure detection, recovery handlers, and escalation.
"""

from __future__ import annotations

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile

from bot.auto_recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryAttempt,
    RecoveryConfig,
    AutoRecovery,
    create_auto_recovery,
    recover_trading_bot,
    recover_data_feed,
)


class TestRecoveryAction:
    """Test RecoveryAction enum."""

    def test_actions_defined(self):
        """Test all recovery actions are defined."""
        assert RecoveryAction.RESTART_PROCESS.value == "restart_process"
        assert RecoveryAction.RECONNECT.value == "reconnect"
        assert RecoveryAction.CLEAR_CACHE.value == "clear_cache"
        assert RecoveryAction.RESET_STATE.value == "reset_state"
        assert RecoveryAction.FAILOVER.value == "failover"
        assert RecoveryAction.ESCALATE.value == "escalate"


class TestRecoveryResult:
    """Test RecoveryResult enum."""

    def test_results_defined(self):
        """Test all recovery results are defined."""
        assert RecoveryResult.SUCCESS.value == "success"
        assert RecoveryResult.PARTIAL.value == "partial"
        assert RecoveryResult.FAILED.value == "failed"
        assert RecoveryResult.SKIPPED.value == "skipped"


class TestRecoveryAttempt:
    """Test RecoveryAttempt dataclass."""

    def test_attempt_creation(self):
        """Test recovery attempt creation."""
        attempt = RecoveryAttempt(
            component="trading_bot",
            action=RecoveryAction.RESTART_PROCESS,
            result=RecoveryResult.SUCCESS,
            duration_ms=150.0,
        )

        assert attempt.component == "trading_bot"
        assert attempt.action == RecoveryAction.RESTART_PROCESS
        assert attempt.result == RecoveryResult.SUCCESS
        assert attempt.duration_ms == 150.0
        assert attempt.error is None

    def test_attempt_with_error(self):
        """Test recovery attempt with error."""
        attempt = RecoveryAttempt(
            component="data_feed",
            action=RecoveryAction.RECONNECT,
            result=RecoveryResult.FAILED,
            error="Connection timeout",
        )

        assert attempt.result == RecoveryResult.FAILED
        assert attempt.error == "Connection timeout"


class TestRecoveryConfig:
    """Test RecoveryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = RecoveryConfig()

        assert config.max_attempts == 3
        assert config.backoff_base == 2.0
        assert config.backoff_max == 300.0
        assert config.cooldown_period == 60.0
        assert config.escalation_threshold == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = RecoveryConfig(
            max_attempts=5,
            backoff_base=3.0,
            cooldown_period=120.0,
        )

        assert config.max_attempts == 5
        assert config.backoff_base == 3.0
        assert config.cooldown_period == 120.0


class TestAutoRecovery:
    """Test AutoRecovery class."""

    @pytest.fixture
    def recovery(self):
        """Create auto-recovery service."""
        config = RecoveryConfig(
            max_attempts=3,
            cooldown_period=0,  # Disable cooldown for tests
        )
        return AutoRecovery(config=config)

    def test_initialization(self, recovery):
        """Test auto-recovery initialization."""
        assert recovery is not None
        assert len(recovery.recovery_handlers) > 0
        assert recovery.recovery_history == []

    def test_default_handlers_registered(self, recovery):
        """Test default handlers are registered."""
        assert "process" in recovery.recovery_handlers
        assert "connection" in recovery.recovery_handlers
        assert "cache" in recovery.recovery_handlers
        assert "state" in recovery.recovery_handlers
        assert "data_feed" in recovery.recovery_handlers
        assert "api" in recovery.recovery_handlers

    def test_register_custom_handler(self, recovery):
        """Test registering a custom handler."""
        async def custom_handler(component, context):
            return True

        recovery.register_handler("custom", custom_handler)

        assert "custom" in recovery.recovery_handlers

    def test_get_statistics_empty(self, recovery):
        """Test statistics with no attempts."""
        stats = recovery.get_statistics()

        assert stats["total_attempts"] == 0
        assert stats["success_rate"] == 0

    def test_reset_attempts(self, recovery):
        """Test resetting attempt counts."""
        recovery.attempt_counts["test"] = 5
        recovery.last_recovery["test"] = 12345

        recovery.reset_attempts("test")

        assert "test" not in recovery.attempt_counts
        assert "test" not in recovery.last_recovery

    def test_reset_all_attempts(self, recovery):
        """Test resetting all attempt counts."""
        recovery.attempt_counts["a"] = 1
        recovery.attempt_counts["b"] = 2

        recovery.reset_attempts()

        assert len(recovery.attempt_counts) == 0


class TestAutoRecoveryAsync:
    """Test async methods of AutoRecovery."""

    @pytest.fixture
    def recovery(self):
        """Create auto-recovery service."""
        config = RecoveryConfig(
            max_attempts=3,
            cooldown_period=0,
            backoff_base=0,  # No backoff for tests
        )
        return AutoRecovery(config=config)

    @pytest.mark.asyncio
    async def test_recover_unknown_type(self, recovery):
        """Test recovery with unknown component type."""
        result = await recovery.recover(
            component="test",
            component_type="unknown_type",
        )

        assert result == RecoveryResult.FAILED

    @pytest.mark.asyncio
    async def test_recover_success(self, recovery):
        """Test successful recovery."""
        async def success_handler(component, context):
            return True

        recovery.register_handler("test_type", success_handler)

        result = await recovery.recover(
            component="test",
            component_type="test_type",
        )

        assert result == RecoveryResult.SUCCESS
        assert recovery.attempt_counts.get("test", 0) == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_recover_failure(self, recovery):
        """Test failed recovery."""
        async def fail_handler(component, context):
            raise Exception("Test failure")

        recovery.register_handler("test_type", fail_handler)

        result = await recovery.recover(
            component="test",
            component_type="test_type",
        )

        assert result == RecoveryResult.FAILED
        assert recovery.attempt_counts["test"] == 1

    @pytest.mark.asyncio
    async def test_max_attempts_reached(self, recovery):
        """Test behavior when max attempts reached."""
        escalated = []

        async def escalation_callback(component, reason):
            escalated.append((component, reason))

        recovery.escalation_callback = escalation_callback
        recovery.attempt_counts["test"] = 3  # Max attempts

        async def fail_handler(component, context):
            return True

        recovery.register_handler("test_type", fail_handler)

        result = await recovery.recover(
            component="test",
            component_type="test_type",
        )

        assert result == RecoveryResult.FAILED
        assert len(escalated) == 1
        assert escalated[0][0] == "test"

    @pytest.mark.asyncio
    async def test_cooldown_period(self):
        """Test cooldown period prevents rapid recovery."""
        config = RecoveryConfig(cooldown_period=60.0)
        recovery = AutoRecovery(config=config)

        # Set last recovery time
        import time
        recovery.last_recovery["test"] = time.time()

        async def handler(component, context):
            return True

        recovery.register_handler("test_type", handler)

        result = await recovery.recover(
            component="test",
            component_type="test_type",
        )

        assert result == RecoveryResult.SKIPPED

    @pytest.mark.asyncio
    async def test_notification_callback(self, recovery):
        """Test notification callback is called."""
        notifications = []

        async def notify(message, context):
            notifications.append((message, context))

        recovery.notification_callback = notify

        async def success_handler(component, context):
            return True

        recovery.register_handler("test_type", success_handler)

        await recovery.recover(
            component="test",
            component_type="test_type",
        )

        assert len(notifications) == 1
        assert "success" in notifications[0][0].lower()

    @pytest.mark.asyncio
    async def test_recover_cache(self, recovery):
        """Test cache recovery handler."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            cache_path = f.name

        result = await recovery.recover(
            component="test_cache",
            component_type="cache",
            context={"cache_path": cache_path},
        )

        assert result == RecoveryResult.SUCCESS
        # Cache file should be cleared
        assert not Path(cache_path).exists()

    @pytest.mark.asyncio
    async def test_recover_state(self, recovery):
        """Test state recovery handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text('{"old": "data"}')

            result = await recovery.recover(
                component="test_state",
                component_type="state",
                context={
                    "state_file": str(state_file),
                    "default_state": {"new": "state"},
                },
            )

            assert result == RecoveryResult.SUCCESS
            # Backup should exist
            assert (Path(tmpdir) / "state.backup").exists()


class TestRecoveryStatistics:
    """Test recovery statistics tracking."""

    @pytest.fixture
    def recovery_with_history(self):
        """Create recovery service with some history."""
        recovery = AutoRecovery()

        # Add some history
        recovery.recovery_history = [
            RecoveryAttempt(component="a", action=RecoveryAction.RESTART_PROCESS, result=RecoveryResult.SUCCESS, duration_ms=100),
            RecoveryAttempt(component="a", action=RecoveryAction.RESTART_PROCESS, result=RecoveryResult.SUCCESS, duration_ms=150),
            RecoveryAttempt(component="b", action=RecoveryAction.RECONNECT, result=RecoveryResult.FAILED, duration_ms=200, error="Error"),
            RecoveryAttempt(component="b", action=RecoveryAction.RECONNECT, result=RecoveryResult.SUCCESS, duration_ms=180),
        ]

        return recovery

    def test_statistics_total_attempts(self, recovery_with_history):
        """Test total attempts count."""
        stats = recovery_with_history.get_statistics()
        assert stats["total_attempts"] == 4

    def test_statistics_success_rate(self, recovery_with_history):
        """Test success rate calculation."""
        stats = recovery_with_history.get_statistics()
        assert stats["success_rate"] == 0.75  # 3 out of 4

    def test_statistics_by_result(self, recovery_with_history):
        """Test statistics by result."""
        stats = recovery_with_history.get_statistics()
        assert stats["by_result"]["success"] == 3
        assert stats["by_result"]["failed"] == 1

    def test_statistics_by_component(self, recovery_with_history):
        """Test statistics by component."""
        stats = recovery_with_history.get_statistics()
        assert stats["by_component"]["a"]["total"] == 2
        assert stats["by_component"]["a"]["success"] == 2
        assert stats["by_component"]["b"]["failed"] == 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_create_auto_recovery(self):
        """Test create_auto_recovery factory."""
        recovery = await create_auto_recovery()

        assert recovery is not None
        assert isinstance(recovery, AutoRecovery)
        assert recovery.config.max_attempts == 3
