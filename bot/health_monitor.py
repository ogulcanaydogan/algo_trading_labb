"""
Production Health Monitoring System.

Monitors all system components and triggers alerts/recovery when issues detected.
Critical for 24/7 live trading operation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    API = "api"
    ORCHESTRATOR = "orchestrator"
    DATA_FEED = "data_feed"
    EXCHANGE = "exchange"
    DATABASE = "database"
    RISK_GUARDIAN = "risk_guardian"
    EXECUTION_ENGINE = "execution_engine"
    NOTIFICATION = "notification"
    ML_MODEL = "ml_model"
    OTHER = "other"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    component_type: ComponentType
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_healthy: Optional[datetime] = None
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    latency_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    last_update: datetime
    uptime_seconds: float
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "last_update": self.last_update.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "alerts": self.alerts,
            "healthy_count": sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY),
            "total_count": len(self.components),
        }


class HealthMonitor:
    """
    Production Health Monitoring System.

    Features:
    - Continuous health checking of all components
    - Automatic alerting on failures
    - Integration with notification system
    - Recovery trigger support
    - Metrics collection for dashboards

    Usage:
        monitor = HealthMonitor()
        monitor.register_component("binance", ComponentType.EXCHANGE, check_binance)
        await monitor.start()
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        alert_threshold: int = 3,
        recovery_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None,
        state_file: str = "data/health_state.json",
    ):
        """
        Initialize health monitor.

        Args:
            check_interval: Seconds between health checks
            alert_threshold: Consecutive failures before alerting
            recovery_callback: Called when recovery is needed
            notification_callback: Called to send alerts
            state_file: File to persist health state
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.recovery_callback = recovery_callback
        self.notification_callback = notification_callback
        self.state_file = Path(state_file)

        self._components: Dict[str, ComponentHealth] = {}
        self._check_functions: Dict[str, Callable] = {}
        self._start_time = datetime.now()
        self._running = False
        self._lock = threading.Lock()
        self._alerted_components: Set[str] = set()

        # Ensure state directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        check_function: Callable[[], HealthCheckResult],
        critical: bool = False,
    ) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Unique component name
            component_type: Type of component
            check_function: Async function that returns HealthCheckResult
            critical: If True, system is critical when this fails
        """
        with self._lock:
            self._components[name] = ComponentHealth(
                name=name,
                component_type=component_type,
            )
            self._check_functions[name] = check_function
            self._components[name].details["critical"] = critical

        logger.info(f"Registered health check for: {name} ({component_type.value})")

    def unregister_component(self, name: str) -> None:
        """Remove a component from monitoring."""
        with self._lock:
            self._components.pop(name, None)
            self._check_functions.pop(name, None)

    async def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component."""
        if name not in self._components:
            raise ValueError(f"Unknown component: {name}")

        component = self._components[name]
        check_func = self._check_functions[name]

        start_time = time.monotonic()

        try:
            # Run health check
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check_func
                )

            latency = (time.monotonic() - start_time) * 1000

            with self._lock:
                component.last_check = datetime.now()
                component.latency_ms = latency

                if result.healthy:
                    component.status = HealthStatus.HEALTHY
                    component.last_healthy = datetime.now()
                    component.consecutive_failures = 0
                    component.details.update(result.details)

                    # Clear alert if was alerted
                    if name in self._alerted_components:
                        self._alerted_components.remove(name)
                        await self._send_recovery_notification(name)
                else:
                    component.consecutive_failures += 1
                    component.error_count += 1
                    component.last_error = result.message
                    component.details.update(result.details)

                    # Determine status based on failures
                    if component.consecutive_failures >= self.alert_threshold * 2:
                        component.status = HealthStatus.CRITICAL
                    elif component.consecutive_failures >= self.alert_threshold:
                        component.status = HealthStatus.UNHEALTHY
                    else:
                        component.status = HealthStatus.DEGRADED

                    # Alert if threshold reached
                    if (component.consecutive_failures >= self.alert_threshold
                            and name not in self._alerted_components):
                        self._alerted_components.add(name)
                        await self._send_alert(name, component)

        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000

            with self._lock:
                component.last_check = datetime.now()
                component.latency_ms = latency
                component.consecutive_failures += 1
                component.error_count += 1
                component.last_error = str(e)
                component.status = HealthStatus.UNHEALTHY

                logger.error(f"Health check failed for {name}: {e}")

                if (component.consecutive_failures >= self.alert_threshold
                        and name not in self._alerted_components):
                    self._alerted_components.add(name)
                    await self._send_alert(name, component)

        return component

    async def check_all(self) -> SystemHealth:
        """Check health of all registered components."""
        tasks = [self.check_component(name) for name in self._components]
        await asyncio.gather(*tasks, return_exceptions=True)

        return self.get_system_health()

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        with self._lock:
            components = dict(self._components)

        # Determine overall status
        statuses = [c.status for c in components.values()]
        alerts = []

        if any(s == HealthStatus.CRITICAL for s in statuses):
            overall = HealthStatus.CRITICAL
            alerts.append("CRITICAL: One or more components in critical state")
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
            alerts.append("WARNING: One or more components unhealthy")
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Add specific component alerts
        for name, comp in components.items():
            if comp.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                alerts.append(f"{name}: {comp.last_error or 'Unknown error'}")

        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall,
            components=components,
            last_update=datetime.now(),
            uptime_seconds=uptime,
            alerts=alerts,
        )

    async def _send_alert(self, name: str, component: ComponentHealth) -> None:
        """Send alert for unhealthy component."""
        message = (
            f"HEALTH ALERT: {name}\n"
            f"Status: {component.status.value}\n"
            f"Failures: {component.consecutive_failures}\n"
            f"Error: {component.last_error}"
        )

        logger.warning(message)

        if self.notification_callback:
            try:
                if asyncio.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(
                        title=f"Health Alert: {name}",
                        message=message,
                        priority="high",
                        category="health",
                    )
                else:
                    self.notification_callback(
                        title=f"Health Alert: {name}",
                        message=message,
                        priority="high",
                        category="health",
                    )
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")

        # Trigger recovery if critical
        if component.details.get("critical") and self.recovery_callback:
            try:
                if asyncio.iscoroutinefunction(self.recovery_callback):
                    await self.recovery_callback(name, component)
                else:
                    self.recovery_callback(name, component)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

    async def _send_recovery_notification(self, name: str) -> None:
        """Send notification that component recovered."""
        message = f"RECOVERED: {name} is now healthy"

        logger.info(message)

        if self.notification_callback:
            try:
                if asyncio.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(
                        title=f"Recovered: {name}",
                        message=message,
                        priority="low",
                        category="health",
                    )
                else:
                    self.notification_callback(
                        title=f"Recovered: {name}",
                        message=message,
                        priority="low",
                        category="health",
                    )
            except Exception as e:
                logger.error(f"Failed to send recovery notification: {e}")

    async def start(self) -> None:
        """Start continuous health monitoring."""
        self._running = True
        self._start_time = datetime.now()

        logger.info(f"Health monitor started (interval: {self.check_interval}s)")

        while self._running:
            try:
                await self.check_all()
                self._save_state()
            except Exception as e:
                logger.error(f"Health check cycle failed: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        logger.info("Health monitor stopped")

    def _save_state(self) -> None:
        """Save current health state to file."""
        try:
            health = self.get_system_health()
            with open(self.state_file, "w") as f:
                json.dump(health.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save health state: {e}")

    def get_component(self, name: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component."""
        return self._components.get(name)

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        health = self.get_system_health()
        return health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def is_critical(self) -> bool:
        """Check if system is in critical state."""
        health = self.get_system_health()
        return health.status == HealthStatus.CRITICAL


# =============================================================================
# Pre-built Health Checks
# =============================================================================


def create_exchange_health_check(exchange_name: str, exchange_client: Any) -> Callable:
    """Create health check for exchange connection."""

    async def check() -> HealthCheckResult:
        try:
            start = time.monotonic()

            if hasattr(exchange_client, 'fetch_ticker'):
                # Try to fetch a common ticker
                await exchange_client.fetch_ticker("BTC/USDT")
            elif hasattr(exchange_client, 'ping'):
                await exchange_client.ping()

            latency = (time.monotonic() - start) * 1000

            return HealthCheckResult(
                healthy=True,
                latency_ms=latency,
                message="Exchange responsive",
                details={"exchange": exchange_name},
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"exchange": exchange_name, "error": str(e)},
            )

    return check


def create_api_health_check(url: str, timeout: float = 5.0) -> Callable:
    """Create health check for API endpoint."""

    def check() -> HealthCheckResult:
        import urllib.request
        import urllib.error

        try:
            start = time.monotonic()

            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = response.status
                latency = (time.monotonic() - start) * 1000

            return HealthCheckResult(
                healthy=status == 200,
                latency_ms=latency,
                message=f"Status: {status}",
                details={"url": url, "status": status},
            )
        except urllib.error.URLError as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"url": url, "error": str(e)},
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"url": url, "error": str(e)},
            )

    return check


def create_file_freshness_check(
    file_path: str,
    max_age_seconds: float = 300,
) -> Callable:
    """Create health check for file freshness (e.g., bot state)."""

    def check() -> HealthCheckResult:
        try:
            path = Path(file_path)

            if not path.exists():
                return HealthCheckResult(
                    healthy=False,
                    latency_ms=0,
                    message="File not found",
                    details={"path": str(path)},
                )

            mtime = path.stat().st_mtime
            age = time.time() - mtime

            healthy = age <= max_age_seconds

            return HealthCheckResult(
                healthy=healthy,
                latency_ms=0,
                message=f"Age: {age:.0f}s (max: {max_age_seconds}s)",
                details={
                    "path": str(path),
                    "age_seconds": age,
                    "max_age_seconds": max_age_seconds,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"path": file_path, "error": str(e)},
            )

    return check


def create_process_health_check(process_name: str) -> Callable:
    """Create health check for a running process."""

    def check() -> HealthCheckResult:
        import subprocess

        try:
            result = subprocess.run(
                ["pgrep", "-f", process_name],
                capture_output=True,
                timeout=5,
            )

            healthy = result.returncode == 0
            pid_count = len(result.stdout.decode().strip().split('\n')) if healthy else 0

            return HealthCheckResult(
                healthy=healthy,
                latency_ms=0,
                message=f"Running: {pid_count} process(es)" if healthy else "Not running",
                details={"process": process_name, "count": pid_count},
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"process": process_name, "error": str(e)},
            )

    return check


def create_memory_health_check(max_usage_percent: float = 90.0) -> Callable:
    """Create health check for memory usage."""

    def check() -> HealthCheckResult:
        try:
            import psutil
            mem = psutil.virtual_memory()

            healthy = mem.percent < max_usage_percent

            return HealthCheckResult(
                healthy=healthy,
                latency_ms=0,
                message=f"Memory: {mem.percent:.1f}% used",
                details={
                    "percent": mem.percent,
                    "available_gb": mem.available / (1024**3),
                    "total_gb": mem.total / (1024**3),
                },
            )
        except ImportError:
            return HealthCheckResult(
                healthy=True,
                latency_ms=0,
                message="psutil not available",
                details={},
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"error": str(e)},
            )

    return check


def create_disk_health_check(
    path: str = "/",
    min_free_gb: float = 5.0,
) -> Callable:
    """Create health check for disk space."""

    def check() -> HealthCheckResult:
        try:
            import shutil
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024**3)

            healthy = free_gb >= min_free_gb

            return HealthCheckResult(
                healthy=healthy,
                latency_ms=0,
                message=f"Free: {free_gb:.1f} GB",
                details={
                    "path": path,
                    "free_gb": free_gb,
                    "total_gb": usage.total / (1024**3),
                    "used_percent": (usage.used / usage.total) * 100,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                latency_ms=0,
                message=str(e),
                details={"path": path, "error": str(e)},
            )

    return check
