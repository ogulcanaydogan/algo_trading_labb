"""
Circuit Breaker for Production Trading

Prevents cascade failures by stopping operations when error rates exceed thresholds.
Implements the circuit breaker pattern for fault tolerance.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Circuit tripped, requests blocked
- HALF_OPEN: Testing if service recovered
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking all calls
    HALF_OPEN = "half_open"  # Testing recovery


class TripReason(Enum):
    """Reasons for circuit trip."""

    ERROR_RATE = "error_rate"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    LATENCY = "latency"
    MANUAL = "manual"
    EXTERNAL = "external"


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker."""

    # Trip thresholds
    error_rate_threshold: float = 0.5  # 50% error rate
    consecutive_failures_threshold: int = 5
    latency_threshold_ms: float = 5000  # 5 seconds

    # Timing
    window_size_seconds: float = 60.0
    reset_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3

    # Minimum samples before tripping
    min_samples: int = 10


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker decisions."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    total_latency_ms: float = 0

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.failed_calls / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.total_latency_ms / self.total_calls

    def reset(self) -> None:
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.consecutive_failures = 0
        self.total_latency_ms = 0


@dataclass
class CircuitEvent:
    """Event in circuit breaker history."""

    timestamp: datetime
    event_type: str
    state_from: CircuitState
    state_to: CircuitState
    reason: Optional[str] = None
    metrics: Optional[dict] = None


class CircuitBreaker:
    """
    Circuit breaker for protecting trading operations.

    Features:
    - Error rate monitoring
    - Consecutive failure detection
    - Latency threshold monitoring
    - Automatic recovery testing
    - Event history for debugging
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
        on_state_change: Optional[Callable] = None,
        on_trip: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or CircuitConfig()
        self.on_state_change = on_state_change
        self.on_trip = on_trip

        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.half_open_calls = 0

        self.last_state_change = time.time()
        self.last_failure_time: Optional[float] = None
        self.trip_reason: Optional[TripReason] = None

        self.call_timestamps: list[float] = []
        self.event_history: list[CircuitEvent] = []

        self._lock = asyncio.Lock()

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Function result if circuit allows

        Raises:
            CircuitOpenError if circuit is open
        """
        async with self._lock:
            if not await self._can_execute():
                raise CircuitOpenError(
                    f"Circuit {self.name} is OPEN",
                    reason=self.trip_reason,
                )

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

        start_time = time.time()
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            latency_ms = (time.time() - start_time) * 1000
            await self._record_success(latency_ms)
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            await self._record_failure(latency_ms, str(e))
            raise

    async def _can_execute(self) -> bool:
        """Check if call can be executed."""
        self._clean_old_calls()

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if reset timeout passed
            elapsed = time.time() - self.last_state_change
            if elapsed >= self.config.reset_timeout_seconds:
                await self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    async def _record_success(self, latency_ms: float) -> None:
        """Record successful call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_failures = 0
            self.metrics.total_latency_ms += latency_ms
            self.call_timestamps.append(time.time())

            # Check latency threshold
            if latency_ms > self.config.latency_threshold_ms:
                logger.warning(
                    f"Circuit {self.name}: High latency {latency_ms:.0f}ms"
                )

            # If half-open and successful, close circuit
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    await self._transition_to(CircuitState.CLOSED)
                    self.metrics.reset()

    async def _record_failure(self, latency_ms: float, error: str) -> None:
        """Record failed call."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.total_latency_ms += latency_ms
            self.call_timestamps.append(time.time())
            self.last_failure_time = time.time()

            # If half-open and failed, open circuit
            if self.state == CircuitState.HALF_OPEN:
                await self._trip(TripReason.CONSECUTIVE_FAILURES, error)
                return

            # Check trip conditions
            await self._check_trip_conditions()

    async def _check_trip_conditions(self) -> None:
        """Check if circuit should trip."""
        # Consecutive failures
        if self.metrics.consecutive_failures >= self.config.consecutive_failures_threshold:
            await self._trip(
                TripReason.CONSECUTIVE_FAILURES,
                f"{self.metrics.consecutive_failures} consecutive failures",
            )
            return

        # Error rate (need minimum samples)
        if self.metrics.total_calls >= self.config.min_samples:
            if self.metrics.error_rate >= self.config.error_rate_threshold:
                await self._trip(
                    TripReason.ERROR_RATE,
                    f"Error rate {self.metrics.error_rate:.1%}",
                )
                return

        # Latency
        if self.metrics.avg_latency_ms > self.config.latency_threshold_ms:
            await self._trip(
                TripReason.LATENCY,
                f"Avg latency {self.metrics.avg_latency_ms:.0f}ms",
            )

    async def _trip(self, reason: TripReason, details: str) -> None:
        """Trip the circuit breaker."""
        self.trip_reason = reason
        await self._transition_to(CircuitState.OPEN)

        logger.warning(f"Circuit {self.name} TRIPPED: {reason.value} - {details}")

        if self.on_trip:
            try:
                if inspect.iscoroutinefunction(self.on_trip):
                    await self.on_trip(self.name, reason, details)
                else:
                    self.on_trip(self.name, reason, details)
            except Exception as e:
                logger.error(f"Trip callback failed: {e}")

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0

        if new_state == CircuitState.CLOSED:
            self.trip_reason = None

        # Record event
        event = CircuitEvent(
            timestamp=datetime.now(),
            event_type="state_change",
            state_from=old_state,
            state_to=new_state,
            reason=self.trip_reason.value if self.trip_reason else None,
            metrics={
                "error_rate": self.metrics.error_rate,
                "consecutive_failures": self.metrics.consecutive_failures,
                "avg_latency_ms": self.metrics.avg_latency_ms,
            },
        )
        self.event_history.append(event)

        # Keep only last 50 events
        if len(self.event_history) > 50:
            self.event_history = self.event_history[-50:]

        logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")

        if self.on_state_change:
            try:
                if inspect.iscoroutinefunction(self.on_state_change):
                    await self.on_state_change(self.name, old_state, new_state)
                else:
                    self.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _clean_old_calls(self) -> None:
        """Remove calls outside the window."""
        cutoff = time.time() - self.config.window_size_seconds
        self.call_timestamps = [t for t in self.call_timestamps if t > cutoff]

    async def force_open(self, reason: str = "Manual trip") -> None:
        """Manually open the circuit."""
        async with self._lock:
            await self._trip(TripReason.MANUAL, reason)

    async def force_close(self) -> None:
        """Manually close the circuit."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self.metrics.reset()

    async def reset(self) -> None:
        """Reset circuit to initial state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics.reset()
            self.half_open_calls = 0
            self.trip_reason = None
            self.last_state_change = time.time()
            self.call_timestamps.clear()

    def get_status(self) -> dict[str, Any]:
        """Get current circuit status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "trip_reason": self.trip_reason.value if self.trip_reason else None,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "error_rate": f"{self.metrics.error_rate:.1%}",
                "consecutive_failures": self.metrics.consecutive_failures,
                "avg_latency_ms": f"{self.metrics.avg_latency_ms:.0f}",
            },
            "config": {
                "error_rate_threshold": self.config.error_rate_threshold,
                "consecutive_failures_threshold": self.config.consecutive_failures_threshold,
                "latency_threshold_ms": self.config.latency_threshold_ms,
                "reset_timeout_seconds": self.config.reset_timeout_seconds,
            },
            "last_state_change": datetime.fromtimestamp(
                self.last_state_change
            ).isoformat(),
            "time_in_state_seconds": time.time() - self.last_state_change,
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open."""

    def __init__(self, message: str, reason: Optional[TripReason] = None):
        super().__init__(message)
        self.reason = reason


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized control and monitoring of all circuits.
    """

    def __init__(
        self,
        on_any_trip: Optional[Callable] = None,
    ):
        self.circuits: dict[str, CircuitBreaker] = {}
        self.on_any_trip = on_any_trip

    def create(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ) -> CircuitBreaker:
        """Create and register a circuit breaker."""

        async def on_trip(circuit_name: str, reason: TripReason, details: str):
            if self.on_any_trip:
                if inspect.iscoroutinefunction(self.on_any_trip):
                    await self.on_any_trip(circuit_name, reason, details)
                else:
                    self.on_any_trip(circuit_name, reason, details)

        circuit = CircuitBreaker(
            name=name,
            config=config,
            on_trip=on_trip,
        )

        self.circuits[name] = circuit
        return circuit

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuits.get(name)

    async def force_open_all(self, reason: str = "Emergency stop") -> None:
        """Open all circuits (emergency stop)."""
        for circuit in self.circuits.values():
            await circuit.force_open(reason)

    async def force_close_all(self) -> None:
        """Close all circuits."""
        for circuit in self.circuits.values():
            await circuit.force_close()

    async def reset_all(self) -> None:
        """Reset all circuits."""
        for circuit in self.circuits.values():
            await circuit.reset()

    def get_all_status(self) -> dict[str, Any]:
        """Get status of all circuits."""
        return {
            "total_circuits": len(self.circuits),
            "open_circuits": sum(
                1 for c in self.circuits.values()
                if c.state == CircuitState.OPEN
            ),
            "half_open_circuits": sum(
                1 for c in self.circuits.values()
                if c.state == CircuitState.HALF_OPEN
            ),
            "circuits": {
                name: circuit.get_status()
                for name, circuit in self.circuits.items()
            },
        }


# ==================== Pre-configured Circuit Breakers ====================

def create_trading_circuit_breaker(
    name: str = "trading",
    on_trip: Optional[Callable] = None,
) -> CircuitBreaker:
    """Create circuit breaker optimized for trading operations."""
    config = CircuitConfig(
        error_rate_threshold=0.3,  # 30% - stricter for trading
        consecutive_failures_threshold=3,  # Trip faster
        latency_threshold_ms=10000,  # 10s max
        window_size_seconds=60.0,
        reset_timeout_seconds=60.0,  # Wait 1 min before retry
        half_open_max_calls=2,
        min_samples=5,
    )

    return CircuitBreaker(name=name, config=config, on_trip=on_trip)


def create_data_feed_circuit_breaker(
    name: str = "data_feed",
    on_trip: Optional[Callable] = None,
) -> CircuitBreaker:
    """Create circuit breaker for data feed connections."""
    config = CircuitConfig(
        error_rate_threshold=0.5,
        consecutive_failures_threshold=5,
        latency_threshold_ms=5000,
        window_size_seconds=30.0,
        reset_timeout_seconds=30.0,
        half_open_max_calls=3,
        min_samples=10,
    )

    return CircuitBreaker(name=name, config=config, on_trip=on_trip)


def create_api_circuit_breaker(
    name: str = "api",
    on_trip: Optional[Callable] = None,
) -> CircuitBreaker:
    """Create circuit breaker for external API calls."""
    config = CircuitConfig(
        error_rate_threshold=0.4,
        consecutive_failures_threshold=4,
        latency_threshold_ms=3000,
        window_size_seconds=60.0,
        reset_timeout_seconds=45.0,
        half_open_max_calls=2,
        min_samples=8,
    )

    return CircuitBreaker(name=name, config=config, on_trip=on_trip)


# ==================== Usage Example ====================

async def example_usage():
    """Example of using circuit breakers."""

    # Create manager
    manager = CircuitBreakerManager(
        on_any_trip=lambda name, reason, details: print(
            f"ALERT: Circuit {name} tripped - {reason}: {details}"
        )
    )

    # Create circuits
    trading = manager.create("trading", CircuitConfig(
        error_rate_threshold=0.3,
        consecutive_failures_threshold=3,
    ))

    data_feed = manager.create("data_feed", CircuitConfig(
        error_rate_threshold=0.5,
        consecutive_failures_threshold=5,
    ))

    # Use circuit breaker
    async def fetch_data():
        # Simulated data fetch
        import random
        if random.random() < 0.3:
            raise Exception("Simulated failure")
        return {"price": 50000}

    for _ in range(20):
        try:
            result = await data_feed.call(fetch_data)
            print(f"Got data: {result}")
        except CircuitOpenError as e:
            print(f"Circuit open: {e}")
        except Exception as e:
            print(f"Call failed: {e}")

        await asyncio.sleep(0.1)

    # Check status
    print("\nCircuit Status:")
    print(manager.get_all_status())


if __name__ == "__main__":
    asyncio.run(example_usage())
