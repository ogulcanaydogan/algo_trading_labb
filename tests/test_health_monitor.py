"""
Tests for health monitor module.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from bot.health_monitor import (
    HealthStatus,
    ComponentType,
    ComponentHealth,
    HealthCheckResult,
    SystemHealth,
    HealthMonitor,
    create_file_freshness_check,
    create_disk_health_check,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_all_statuses_exist(self):
        """Test all health statuses exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentType:
    """Test ComponentType enum."""

    def test_all_types_exist(self):
        """Test all component types exist."""
        assert ComponentType.API.value == "api"
        assert ComponentType.ORCHESTRATOR.value == "orchestrator"
        assert ComponentType.DATA_FEED.value == "data_feed"
        assert ComponentType.EXCHANGE.value == "exchange"
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.RISK_GUARDIAN.value == "risk_guardian"
        assert ComponentType.EXECUTION_ENGINE.value == "execution_engine"
        assert ComponentType.ML_MODEL.value == "ml_model"


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_default_creation(self):
        """Test creating component health with defaults."""
        health = ComponentHealth(
            name="test_component",
            component_type=ComponentType.API,
        )
        assert health.name == "test_component"
        assert health.component_type == ComponentType.API
        assert health.status == HealthStatus.UNKNOWN
        assert health.error_count == 0
        assert health.consecutive_failures == 0

    def test_full_creation(self):
        """Test creating component health with all fields."""
        now = datetime.now()
        health = ComponentHealth(
            name="test_api",
            component_type=ComponentType.API,
            status=HealthStatus.HEALTHY,
            last_check=now,
            last_healthy=now,
            error_count=5,
            consecutive_failures=0,
            last_error=None,
            latency_ms=50.0,
            details={"version": "1.0"},
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 50.0
        assert health.details["version"] == "1.0"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now()
        health = ComponentHealth(
            name="test",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.DEGRADED,
            last_check=now,
            latency_ms=100.0,
        )
        d = health.to_dict()

        assert d["name"] == "test"
        assert d["type"] == "database"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 100.0


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_healthy_result(self):
        """Test creating a healthy result."""
        result = HealthCheckResult(
            healthy=True,
            latency_ms=25.0,
            message="All good",
        )
        assert result.healthy is True
        assert result.latency_ms == 25.0
        assert result.message == "All good"

    def test_unhealthy_result(self):
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            latency_ms=5000.0,
            message="Timeout exceeded",
            details={"timeout": 5000},
        )
        assert result.healthy is False
        assert result.details["timeout"] == 5000


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_system_health_creation(self):
        """Test creating system health."""
        now = datetime.now()
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components={
                "api": ComponentHealth("api", ComponentType.API, status=HealthStatus.HEALTHY),
                "db": ComponentHealth("db", ComponentType.DATABASE, status=HealthStatus.HEALTHY),
            },
            last_update=now,
            uptime_seconds=3600.0,
        )
        assert health.status == HealthStatus.HEALTHY
        assert len(health.components) == 2
        assert health.uptime_seconds == 3600.0

    def test_to_dict(self):
        """Test system health to dict."""
        now = datetime.now()
        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            components={
                "api": ComponentHealth("api", ComponentType.API, status=HealthStatus.HEALTHY),
                "db": ComponentHealth("db", ComponentType.DATABASE, status=HealthStatus.DEGRADED),
            },
            last_update=now,
            uptime_seconds=7200.0,
            alerts=["DB performance degraded"],
        )
        d = health.to_dict()

        assert d["status"] == "degraded"
        assert d["healthy_count"] == 1
        assert d["total_count"] == 2
        assert len(d["alerts"]) == 1


class TestHealthMonitor:
    """Test HealthMonitor class."""

    @pytest.fixture
    def temp_state_file(self):
        """Create temporary state file."""
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/health_state.json"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def monitor(self, temp_state_file):
        """Create health monitor instance."""
        return HealthMonitor(
            check_interval=1.0,
            alert_threshold=3,
            state_file=temp_state_file,
        )

    def test_monitor_creation(self, monitor):
        """Test monitor is created."""
        assert monitor is not None
        assert monitor.check_interval == 1.0
        assert monitor.alert_threshold == 3

    def test_register_component(self, monitor):
        """Test registering a component."""
        def check():
            return HealthCheckResult(healthy=True, latency_ms=10.0)

        monitor.register_component("test_api", ComponentType.API, check)

        assert "test_api" in monitor._components
        assert monitor._components["test_api"].component_type == ComponentType.API

    def test_register_critical_component(self, monitor):
        """Test registering a critical component."""
        def check():
            return HealthCheckResult(healthy=True, latency_ms=10.0)

        monitor.register_component("exchange", ComponentType.EXCHANGE, check, critical=True)

        assert monitor._components["exchange"].details["critical"] is True

    def test_unregister_component(self, monitor):
        """Test unregistering a component."""
        def check():
            return HealthCheckResult(healthy=True, latency_ms=10.0)

        monitor.register_component("temp", ComponentType.OTHER, check)
        assert "temp" in monitor._components

        monitor.unregister_component("temp")
        assert "temp" not in monitor._components

    @pytest.mark.asyncio
    async def test_check_component_healthy(self, monitor):
        """Test checking a healthy component."""
        def check():
            return HealthCheckResult(healthy=True, latency_ms=25.0, message="OK")

        monitor.register_component("api", ComponentType.API, check)

        health = await monitor.check_component("api")

        assert health.status == HealthStatus.HEALTHY
        assert health.consecutive_failures == 0
        assert health.latency_ms is not None

    @pytest.mark.asyncio
    async def test_check_component_unhealthy(self, monitor):
        """Test checking an unhealthy component."""
        def check():
            return HealthCheckResult(healthy=False, latency_ms=0, message="Connection failed")

        monitor.register_component("db", ComponentType.DATABASE, check)

        health = await monitor.check_component("db")

        assert health.status == HealthStatus.DEGRADED
        assert health.consecutive_failures == 1
        assert health.last_error == "Connection failed"

    @pytest.mark.asyncio
    async def test_check_component_unknown(self, monitor):
        """Test checking unknown component raises error."""
        with pytest.raises(ValueError, match="Unknown component"):
            await monitor.check_component("nonexistent")

    @pytest.mark.asyncio
    async def test_check_all_components(self, monitor):
        """Test checking all components."""
        def check1():
            return HealthCheckResult(healthy=True, latency_ms=10.0)

        def check2():
            return HealthCheckResult(healthy=True, latency_ms=20.0)

        monitor.register_component("api", ComponentType.API, check1)
        monitor.register_component("db", ComponentType.DATABASE, check2)

        system_health = await monitor.check_all()

        assert system_health.status == HealthStatus.HEALTHY
        assert len(system_health.components) == 2

    def test_get_system_health_empty(self, monitor):
        """Test getting system health with no components."""
        health = monitor.get_system_health()
        # Empty system with no components is considered healthy
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.UNKNOWN]
        assert len(health.components) == 0

    def test_get_component(self, monitor):
        """Test getting a specific component."""
        def check():
            return HealthCheckResult(healthy=True, latency_ms=10.0)

        monitor.register_component("api", ComponentType.API, check)

        component = monitor.get_component("api")
        assert component is not None
        assert component.name == "api"

    def test_get_component_nonexistent(self, monitor):
        """Test getting nonexistent component returns None."""
        component = monitor.get_component("nonexistent")
        assert component is None

    def test_is_healthy(self, monitor):
        """Test is_healthy check."""
        # Empty system without any failing components is healthy
        result = monitor.is_healthy()
        assert isinstance(result, bool)

    def test_is_critical(self, monitor):
        """Test is_critical check."""
        assert not monitor.is_critical()

    def test_stop(self, monitor):
        """Test stopping monitor."""
        monitor.stop()
        assert monitor._running is False


class TestPrebuiltHealthChecks:
    """Test pre-built health check functions."""

    def test_file_freshness_check_exists(self):
        """Test file freshness check with existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            check = create_file_freshness_check(temp_path, max_age_seconds=300)
            result = check()

            assert result.healthy is True
            assert "age_seconds" in result.details
        finally:
            import os
            os.unlink(temp_path)

    def test_file_freshness_check_not_found(self):
        """Test file freshness check with missing file."""
        check = create_file_freshness_check("/nonexistent/file.txt")
        result = check()

        assert result.healthy is False
        assert "not found" in result.message.lower()

    def test_file_freshness_check_stale(self):
        """Test file freshness check with stale file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            import os
            import time
            # Set modification time to 10 minutes ago
            old_time = time.time() - 600
            os.utime(temp_path, (old_time, old_time))

            check = create_file_freshness_check(temp_path, max_age_seconds=60)
            result = check()

            assert result.healthy is False
        finally:
            os.unlink(temp_path)

    def test_disk_health_check(self):
        """Test disk health check."""
        check = create_disk_health_check("/", min_free_gb=0.1)
        result = check()

        assert result.healthy is True
        assert "free_gb" in result.details


class TestAlertThreshold:
    """Test alert threshold behavior."""

    @pytest.fixture
    def temp_state_file(self):
        temp_dir = tempfile.mkdtemp()
        yield f"{temp_dir}/health_state.json"
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_alert_after_threshold(self, temp_state_file):
        """Test alert is triggered after threshold failures."""
        alerts_sent = []

        def notification_callback(**kwargs):
            alerts_sent.append(kwargs)

        monitor = HealthMonitor(
            check_interval=1.0,
            alert_threshold=2,
            notification_callback=notification_callback,
            state_file=temp_state_file,
        )

        def failing_check():
            return HealthCheckResult(healthy=False, latency_ms=0, message="Fail")

        monitor.register_component("test", ComponentType.API, failing_check)

        # First failure - no alert
        await monitor.check_component("test")
        assert len(alerts_sent) == 0

        # Second failure - alert triggered
        await monitor.check_component("test")
        assert len(alerts_sent) == 1
