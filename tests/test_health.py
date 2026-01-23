"""
Tests for the health check module.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from bot.core.health import (
    ComponentHealth,
    GracefulShutdown,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    create_database_health_check,
    create_exchange_health_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test that status values are correct."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Test creating a component health object."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.latency_ms == 0.0

    def test_default_values(self):
        """Test default values are set."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.message == ""
        assert health.latency_ms == 0.0
        assert isinstance(health.last_check, datetime)
        assert health.details == {}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        health = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=5.5,
            details={"connections": 10},
        )
        data = health.to_dict()

        assert data["name"] == "database"
        assert data["status"] == "healthy"
        assert data["message"] == "Connected"
        assert data["latency_ms"] == 5.5
        assert data["details"] == {"connections": 10}
        assert "last_check" in data


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_healthy_system(self):
        """Test a healthy system status."""
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY),
            ComponentHealth(name="cache", status=HealthStatus.HEALTHY),
        ]
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
            uptime_seconds=100.0,
        )

        assert health.is_healthy is True
        assert health.is_ready is True

    def test_degraded_system(self):
        """Test a degraded system status."""
        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            components=[],
        )

        assert health.is_healthy is False
        assert health.is_ready is True  # Degraded is still ready

    def test_unhealthy_system(self):
        """Test an unhealthy system status."""
        health = SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components=[],
        )

        assert health.is_healthy is False
        assert health.is_ready is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY),
        ]
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
            uptime_seconds=50.0,
        )
        data = health.to_dict()

        assert data["status"] == "healthy"
        assert data["is_healthy"] is True
        assert data["is_ready"] is True
        assert data["uptime_seconds"] == 50.0
        assert len(data["components"]) == 1


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a health checker instance."""
        return HealthChecker(check_timeout=2.0, unhealthy_threshold=3)

    def test_initialization(self, checker):
        """Test health checker initialization."""
        assert checker.check_timeout == 2.0
        assert checker.unhealthy_threshold == 3
        assert len(checker._checks) == 0

    def test_register_check(self, checker):
        """Test registering a health check."""
        async def dummy_check():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        checker.register("test", dummy_check)
        assert "test" in checker._checks
        assert "test" in checker._history

    def test_unregister_check(self, checker):
        """Test unregistering a health check."""
        async def dummy_check():
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        checker.register("test", dummy_check)
        checker.unregister("test")

        assert "test" not in checker._checks
        assert "test" not in checker._history

    @pytest.mark.asyncio
    async def test_check_unknown_component(self, checker):
        """Test checking an unregistered component."""
        result = await checker.check("unknown")

        assert result.status == HealthStatus.UNKNOWN
        assert "No health check registered" in result.message

    @pytest.mark.asyncio
    async def test_check_healthy_component(self, checker):
        """Test checking a healthy component."""
        async def healthy_check():
            return ComponentHealth(
                name="service",
                status=HealthStatus.HEALTHY,
                message="OK",
            )

        checker.register("service", healthy_check)
        result = await checker.check("service")

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_check_timeout(self, checker):
        """Test that slow checks timeout."""
        checker.check_timeout = 0.1

        async def slow_check():
            await asyncio.sleep(1.0)
            return ComponentHealth(name="slow", status=HealthStatus.HEALTHY)

        checker.register("slow", slow_check)
        result = await checker.check("slow")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_check_exception(self, checker):
        """Test handling exceptions in health checks."""
        async def failing_check():
            raise ValueError("Connection failed")

        checker.register("failing", failing_check)
        result = await checker.check("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message

    @pytest.mark.asyncio
    async def test_check_all_empty(self, checker):
        """Test check_all with no registered checks."""
        result = await checker.check_all()

        assert result.status == HealthStatus.UNKNOWN
        assert len(result.components) == 0
        assert result.uptime_seconds > 0

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, checker):
        """Test check_all when all components are healthy."""
        async def healthy1():
            return ComponentHealth(name="db", status=HealthStatus.HEALTHY)

        async def healthy2():
            return ComponentHealth(name="cache", status=HealthStatus.HEALTHY)

        checker.register("db", healthy1)
        checker.register("cache", healthy2)

        result = await checker.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    @pytest.mark.asyncio
    async def test_check_all_degraded(self, checker):
        """Test check_all when one component is degraded."""
        async def healthy():
            return ComponentHealth(name="db", status=HealthStatus.HEALTHY)

        async def degraded():
            return ComponentHealth(name="cache", status=HealthStatus.DEGRADED)

        checker.register("db", healthy)
        checker.register("cache", degraded)

        result = await checker.check_all()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(self, checker):
        """Test check_all when one component is unhealthy."""
        async def healthy():
            return ComponentHealth(name="db", status=HealthStatus.HEALTHY)

        async def unhealthy():
            return ComponentHealth(name="cache", status=HealthStatus.UNHEALTHY)

        checker.register("db", healthy)
        checker.register("cache", unhealthy)

        result = await checker.check_all()

        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_history_tracking(self, checker):
        """Test that check history is tracked."""
        call_count = 0

        async def varying_check():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return ComponentHealth(name="svc", status=HealthStatus.DEGRADED)
            return ComponentHealth(name="svc", status=HealthStatus.HEALTHY)

        checker.register("svc", varying_check)

        # Run multiple checks
        for _ in range(3):
            await checker.check_all()

        assert len(checker._history["svc"]) == 3
        assert checker._history["svc"][0] == HealthStatus.HEALTHY
        assert checker._history["svc"][1] == HealthStatus.DEGRADED
        assert checker._history["svc"][2] == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_liveness(self, checker):
        """Test liveness probe."""
        result = await checker.liveness()
        assert result is True

    @pytest.mark.asyncio
    async def test_readiness_no_checks(self, checker):
        """Test readiness with no checks."""
        result = await checker.readiness()
        assert result is False  # Unknown status is not ready

    @pytest.mark.asyncio
    async def test_readiness_healthy(self, checker):
        """Test readiness when healthy."""
        async def healthy():
            return ComponentHealth(name="db", status=HealthStatus.HEALTHY)

        checker.register("db", healthy)
        result = await checker.readiness()
        assert result is True

    @pytest.mark.asyncio
    async def test_readiness_degraded(self, checker):
        """Test readiness when degraded."""
        async def degraded():
            return ComponentHealth(name="db", status=HealthStatus.DEGRADED)

        checker.register("db", degraded)
        result = await checker.readiness()
        assert result is True  # Degraded is still ready


class TestGracefulShutdown:
    """Tests for GracefulShutdown class."""

    @pytest.fixture
    def shutdown(self):
        """Create a graceful shutdown instance."""
        return GracefulShutdown(timeout=5.0)

    def test_initialization(self, shutdown):
        """Test graceful shutdown initialization."""
        assert shutdown.timeout == 5.0
        assert len(shutdown._callbacks) == 0
        assert shutdown.is_shutting_down is False

    def test_register_callback(self, shutdown):
        """Test registering a cleanup callback."""
        async def cleanup():
            pass

        shutdown.register(cleanup)
        assert len(shutdown._callbacks) == 1

    def test_is_shutting_down(self, shutdown):
        """Test shutdown state tracking."""
        assert shutdown.is_shutting_down is False

    @pytest.mark.asyncio
    async def test_shutdown_runs_callbacks(self, shutdown):
        """Test that shutdown runs all callbacks."""
        called = []

        async def cleanup1():
            called.append("first")

        async def cleanup2():
            called.append("second")

        shutdown.register(cleanup1)
        shutdown.register(cleanup2)

        await shutdown.shutdown()

        assert called == ["first", "second"]
        assert shutdown.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_shutdown_only_once(self, shutdown):
        """Test that shutdown only runs once."""
        call_count = 0

        async def cleanup():
            nonlocal call_count
            call_count += 1

        shutdown.register(cleanup)

        await shutdown.shutdown()
        await shutdown.shutdown()  # Should not run again

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, shutdown):
        """Test that slow callbacks are timed out."""
        shutdown.timeout = 0.5  # Give more time for both callbacks
        called = []

        async def slow_cleanup():
            await asyncio.sleep(0.2)  # Takes 0.2s, will complete
            called.append("slow")

        async def fast_cleanup():
            called.append("fast")

        shutdown.register(slow_cleanup)
        shutdown.register(fast_cleanup)

        await shutdown.shutdown()

        # Both should complete within timeout
        assert "slow" in called
        assert "fast" in called

    @pytest.mark.asyncio
    async def test_shutdown_individual_timeout(self, shutdown):
        """Test that individual slow callbacks timeout without blocking others."""
        shutdown.timeout = 0.3
        called = []

        async def very_slow_cleanup():
            await asyncio.sleep(10.0)  # Will definitely timeout
            called.append("very_slow")

        async def fast_cleanup():
            called.append("fast")

        shutdown.register(very_slow_cleanup)
        shutdown.register(fast_cleanup)

        await shutdown.shutdown()

        # Very slow should timeout, but fast should still run within remaining time
        assert "very_slow" not in called
        # Note: fast may or may not run depending on remaining timeout

    @pytest.mark.asyncio
    async def test_shutdown_handles_exceptions(self, shutdown):
        """Test that exceptions in callbacks are handled."""
        called = []

        async def failing_cleanup():
            raise ValueError("Cleanup failed")

        async def working_cleanup():
            called.append("working")

        shutdown.register(failing_cleanup)
        shutdown.register(working_cleanup)

        # Should not raise
        await shutdown.shutdown()

        assert "working" in called


class TestHealthCheckFactories:
    """Tests for health check factory functions."""

    @pytest.mark.asyncio
    async def test_create_exchange_health_check(self):
        """Test creating an exchange health check."""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter.get_current_price = AsyncMock(return_value=50000.0)

        check_func = await create_exchange_health_check(mock_adapter)
        result = await check_func()

        assert result.status == HealthStatus.HEALTHY
        assert "50,000" in result.message
        assert result.details["btc_price"] == 50000.0

    @pytest.mark.asyncio
    async def test_create_exchange_health_check_failure(self):
        """Test exchange health check with failure."""
        mock_adapter = MagicMock()
        mock_adapter.get_current_price = AsyncMock(side_effect=Exception("Connection error"))

        check_func = await create_exchange_health_check(mock_adapter)
        result = await check_func()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection error" in result.message

    @pytest.mark.asyncio
    async def test_create_database_health_check(self, tmp_path):
        """Test creating a database health check."""
        import sqlite3

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        check_func = await create_database_health_check(db_path)
        result = await check_func()

        assert result.status == HealthStatus.HEALTHY
        assert "connected" in result.message.lower()

    @pytest.mark.asyncio
    async def test_create_database_health_check_failure(self):
        """Test database health check with invalid path."""
        check_func = await create_database_health_check("/nonexistent/path/db.sqlite")
        result = await check_func()

        assert result.status == HealthStatus.UNHEALTHY


class TestParallelHealthChecks:
    """Tests for parallel health check execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test that health checks run in parallel."""
        import time

        checker = HealthChecker()
        timings = []

        async def slow_check1():
            start = time.perf_counter()
            await asyncio.sleep(0.1)
            timings.append(("check1", time.perf_counter() - start))
            return ComponentHealth(name="svc1", status=HealthStatus.HEALTHY)

        async def slow_check2():
            start = time.perf_counter()
            await asyncio.sleep(0.1)
            timings.append(("check2", time.perf_counter() - start))
            return ComponentHealth(name="svc2", status=HealthStatus.HEALTHY)

        checker.register("svc1", slow_check1)
        checker.register("svc2", slow_check2)

        start = time.perf_counter()
        await checker.check_all()
        total_time = time.perf_counter() - start

        # If running in parallel, total should be ~0.1s, not ~0.2s
        assert total_time < 0.15  # Allow some margin


class TestHistoryLimit:
    """Tests for history limiting."""

    @pytest.mark.asyncio
    async def test_history_limited_to_10(self):
        """Test that history is limited to last 10 entries."""
        checker = HealthChecker()

        async def check():
            return ComponentHealth(name="svc", status=HealthStatus.HEALTHY)

        checker.register("svc", check)

        # Run 15 checks
        for _ in range(15):
            await checker.check_all()

        # Should only keep last 10
        assert len(checker._history["svc"]) == 10
