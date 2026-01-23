"""
Circuit breaker pattern implementation for resilience and fault tolerance.
Prevents cascading failures by temporarily disabling failing services.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass, field

from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    timeout: int = 60  # Seconds to stay open
    success_threshold: int = 2  # Successes to close again
    expected_exception: type = Exception  # Exception type to count as failure


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        metrics_collector: Optional["MetricsCollector"] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = metrics_collector

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        async with self._lock:
            await self._check_state()

            if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset():
                    self._record_failure("CIRCUIT_OPEN")
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN for {self.config.timeout}s"
                    )

                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")

        try:
            start_time = time.time()
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
            duration = time.time() - start_time

            async with self._lock:
                await self._record_success(duration)

            return result

        except self.config.expected_exception as e:
            async with self._lock:
                await self._record_failure(str(e))

            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {e}")
            raise

    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        if self.state == CircuitState.OPEN and self._should_attempt_reset():
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and time.time() - self.last_failure_time >= self.config.timeout
        )

    async def _record_success(self, duration: float) -> None:
        """Record successful operation."""
        self.total_requests += 1
        self.total_successes += 1
        self.success_count += 1

        if self.metrics:
            self.metrics.record_success(self.name, duration)

        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' closed after successful test")

    async def _record_failure(self, error_message: str) -> None:
        """Record failed operation."""
        self.total_requests += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.metrics:
            self.metrics.record_failure(self.name, error_message)

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures"
                )

        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' re-OPENED during HALF_OPEN test")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        uptime_ratio = self.total_successes / max(self.total_requests, 1)

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "uptime_ratio": round(uptime_ratio, 3),
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout": self.config.timeout,
                "success_threshold": self.config.success_threshold,
            },
        }

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register(self, circuit_breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        self._circuit_breakers[circuit_breaker.name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {circuit_breaker.name}")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._circuit_breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            await cb.reset()


# Global registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    return _global_registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: int = 60,
    success_threshold: int = 2,
    expected_exception: type = Exception,
) -> Callable:
    """
    Decorator to apply circuit breaker to a function.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        timeout: Seconds to stay open
        success_threshold: Successes to close again
        expected_exception: Exception type to count as failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get or create circuit breaker
        registry = get_circuit_breaker_registry()
        cb = registry.get(name)

        if cb is None:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                timeout=timeout,
                success_threshold=success_threshold,
                expected_exception=expected_exception,
            )
            cb = CircuitBreaker(name, config)
            registry.register(cb)

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await cb.call(func, *args, **kwargs)
            except CircuitBreakerOpenError as e:
                logger.error(f"Circuit breaker open for {name}: {e}")
                raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {e}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return asyncio.run(cb.call(func, *args, **kwargs))
            except CircuitBreakerOpenError as e:
                logger.error(f"Circuit breaker open for {name}: {e}")
                raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {e}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Pre-configured circuit breakers for common services
class CommonCircuitBreakers:
    """Pre-configured circuit breakers for common trading services."""

    EXCHANGE_API = "exchange_api"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    ML_PREDICTION = "ml_prediction"
    RISK_MANAGEMENT = "risk_management"

    @classmethod
    def create_default_breakers(cls) -> None:
        """Create default circuit breakers with appropriate configurations."""
        registry = get_circuit_breaker_registry()

        # Exchange API - sensitive to failures
        registry.register(
            CircuitBreaker(
                cls.EXCHANGE_API,
                CircuitBreakerConfig(failure_threshold=3, timeout=30, success_threshold=2),
            )
        )

        # Database - more tolerant
        registry.register(
            CircuitBreaker(
                cls.DATABASE,
                CircuitBreakerConfig(failure_threshold=5, timeout=60, success_threshold=3),
            )
        )

        # External APIs - sensitive to rate limits
        registry.register(
            CircuitBreaker(
                cls.EXTERNAL_API,
                CircuitBreakerConfig(failure_threshold=3, timeout=120, success_threshold=2),
            )
        )

        # ML Prediction - moderate tolerance
        registry.register(
            CircuitBreaker(
                cls.ML_PREDICTION,
                CircuitBreakerConfig(failure_threshold=4, timeout=45, success_threshold=2),
            )
        )

        # Risk Management - critical, low tolerance
        registry.register(
            CircuitBreaker(
                cls.RISK_MANAGEMENT,
                CircuitBreakerConfig(failure_threshold=2, timeout=15, success_threshold=3),
            )
        )


# Metrics collector for monitoring
class MetricsCollector:
    """Collects circuit breaker metrics for monitoring."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def record_success(self, circuit_name: str, duration: float) -> None:
        """Record successful operation."""
        if circuit_name not in self.metrics:
            self.metrics[circuit_name] = {
                "success_count": 0,
                "failure_count": 0,
                "total_duration": 0.0,
            }

        self.metrics[circuit_name]["success_count"] += 1
        self.metrics[circuit_name]["total_duration"] += duration

    def record_failure(self, circuit_name: str, error_message: str) -> None:
        """Record failed operation."""
        if circuit_name not in self.metrics:
            self.metrics[circuit_name] = {
                "success_count": 0,
                "failure_count": 0,
                "total_duration": 0.0,
                "errors": [],
            }

        self.metrics[circuit_name]["failure_count"] += 1
        if len(self.metrics[circuit_name]["errors"]) < 10:
            self.metrics[circuit_name]["errors"].append(error_message)

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get collected metrics."""
        result = {}
        for name, data in self.metrics.items():
            success_count = data["success_count"]
            total_ops = success_count + data["failure_count"]
            avg_duration = data["total_duration"] / max(success_count, 1)

            result[name] = {
                "success_count": success_count,
                "failure_count": data["failure_count"],
                "total_operations": total_ops,
                "success_rate": success_count / max(total_ops, 1),
                "average_duration": avg_duration,
                "recent_errors": data.get("errors", [])[-5:],  # Last 5 errors
            }

        return result
