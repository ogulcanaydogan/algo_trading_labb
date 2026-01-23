"""
Health Check Module.

Provides health monitoring for the trading system:
- Component health checks
- Readiness/liveness probes
- Dependency status tracking
- Graceful shutdown coordination

Usage:
    from bot.core.health import HealthChecker, health_check

    # Create health checker
    checker = HealthChecker()
    checker.register("database", db_health_check)
    checker.register("exchange", exchange_health_check)

    # Check health
    status = await checker.check_all()
    if status.is_healthy:
        print("System healthy")
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [c.to_dict() for c in self.components],
        }


HealthCheckFunc = Callable[[], Awaitable[ComponentHealth]]


class HealthChecker:
    """
    System health checker with component monitoring.

    Features:
    - Register custom health checks
    - Async parallel health checks
    - Configurable timeouts
    - Health history tracking
    """

    def __init__(
        self,
        check_timeout: float = 5.0,
        unhealthy_threshold: int = 3,
    ):
        """
        Initialize health checker.

        Args:
            check_timeout: Timeout for individual health checks
            unhealthy_threshold: Number of failed checks before unhealthy
        """
        self.check_timeout = check_timeout
        self.unhealthy_threshold = unhealthy_threshold

        self._checks: Dict[str, HealthCheckFunc] = {}
        self._history: Dict[str, List[HealthStatus]] = {}
        self._start_time = time.monotonic()

    def register(
        self,
        name: str,
        check_func: HealthCheckFunc,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Component name
            check_func: Async function returning ComponentHealth
        """
        self._checks[name] = check_func
        self._history[name] = []
        logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._history.pop(name, None)

    async def check(self, name: str) -> ComponentHealth:
        """
        Run a single health check.

        Args:
            name: Component name

        Returns:
            ComponentHealth result
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
            )

        start = time.perf_counter()
        try:
            async with asyncio.timeout(self.check_timeout):
                result = await self._checks[name]()
                result.latency_ms = (time.perf_counter() - start) * 1000
                return result
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.check_timeout}s",
                latency_ms=self.check_timeout * 1000,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def check_all(self) -> SystemHealth:
        """
        Run all health checks in parallel.

        Returns:
            SystemHealth with all component statuses
        """
        if not self._checks:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                components=[],
                uptime_seconds=time.monotonic() - self._start_time,
            )

        # Run all checks in parallel
        tasks = [self.check(name) for name in self._checks]
        components = await asyncio.gather(*tasks)

        # Update history
        for component in components:
            history = self._history.get(component.name, [])
            history.append(component.status)
            # Keep last N checks
            self._history[component.name] = history[-10:]

        # Determine overall status
        statuses = [c.status for c in components]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall,
            components=list(components),
            uptime_seconds=time.monotonic() - self._start_time,
        )

    async def liveness(self) -> bool:
        """
        Liveness probe - is the system alive?

        Returns:
            True if system is responding
        """
        return True

    async def readiness(self) -> bool:
        """
        Readiness probe - is the system ready to serve?

        Returns:
            True if all critical components are healthy
        """
        health = await self.check_all()
        return health.is_ready


class GracefulShutdown:
    """
    Graceful shutdown coordinator.

    Handles:
    - Signal handling (SIGTERM, SIGINT)
    - Cleanup callback registration
    - Ordered shutdown sequence
    - Timeout for cleanup

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register(close_database)
        shutdown.register(stop_trading_engine)

        # In main
        await shutdown.wait_for_signal()
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize shutdown handler.

        Args:
            timeout: Maximum time to wait for cleanup
        """
        self.timeout = timeout
        self._callbacks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False

    def register(
        self,
        callback: Callable[[], Awaitable[None]],
        priority: int = 0,
    ) -> None:
        """
        Register a cleanup callback.

        Args:
            callback: Async cleanup function
            priority: Lower priority runs first
        """
        self._callbacks.append(callback)

    def setup_signals(self) -> None:
        """Setup signal handlers."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler, sig)

        logger.info("Signal handlers registered for graceful shutdown")

    def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        if not self._is_shutting_down:
            logger.info(f"Received signal {sig.name}, initiating shutdown")
            self._shutdown_event.set()

    async def wait_for_signal(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """
        Execute graceful shutdown.

        Runs all registered callbacks with timeout.
        """
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        logger.info("Starting graceful shutdown...")

        start = time.monotonic()

        for callback in self._callbacks:
            try:
                remaining = self.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    logger.warning("Shutdown timeout exceeded")
                    break

                async with asyncio.timeout(remaining):
                    await callback()
                    logger.debug(f"Cleanup callback completed: {callback.__name__}")
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup callback timed out: {callback.__name__}")
            except Exception as e:
                logger.error(f"Cleanup callback failed: {callback.__name__}: {e}")

        elapsed = time.monotonic() - start
        logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down


# Convenience functions for creating common health checks


async def create_exchange_health_check(
    adapter,
    name: str = "exchange",
) -> HealthCheckFunc:
    """Create health check for exchange adapter."""

    async def check() -> ComponentHealth:
        try:
            # Try to get current price
            price = await adapter.get_current_price("BTC/USDT")
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"Connected, BTC/USDT = ${price:,.2f}",
                details={"btc_price": price},
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


async def create_database_health_check(
    db_path: str,
    name: str = "database",
) -> HealthCheckFunc:
    """Create health check for SQLite database."""
    import sqlite3

    async def check() -> ComponentHealth:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Database connected",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


# Global instances
health_checker = HealthChecker()
graceful_shutdown = GracefulShutdown()
