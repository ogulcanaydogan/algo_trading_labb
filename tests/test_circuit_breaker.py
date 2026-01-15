"""
Tests for Circuit Breaker module.

Tests circuit breaker states, transitions, and failure detection.
"""

from __future__ import annotations

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from bot.circuit_breaker import (
    CircuitState,
    TripReason,
    CircuitConfig,
    CircuitMetrics,
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitOpenError,
    create_trading_circuit_breaker,
    create_data_feed_circuit_breaker,
    create_api_circuit_breaker,
)


class TestCircuitState:
    """Test CircuitState enum."""

    def test_states_defined(self):
        """Test all states are defined."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestTripReason:
    """Test TripReason enum."""

    def test_reasons_defined(self):
        """Test all trip reasons are defined."""
        assert TripReason.ERROR_RATE
        assert TripReason.CONSECUTIVE_FAILURES
        assert TripReason.LATENCY
        assert TripReason.MANUAL
        assert TripReason.EXTERNAL


class TestCircuitConfig:
    """Test CircuitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitConfig()

        assert config.error_rate_threshold == 0.5
        assert config.consecutive_failures_threshold == 5
        assert config.latency_threshold_ms == 5000
        assert config.reset_timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.min_samples == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitConfig(
            error_rate_threshold=0.3,
            consecutive_failures_threshold=3,
            latency_threshold_ms=10000,
        )

        assert config.error_rate_threshold == 0.3
        assert config.consecutive_failures_threshold == 3
        assert config.latency_threshold_ms == 10000


class TestCircuitMetrics:
    """Test CircuitMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = CircuitMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.consecutive_failures == 0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = CircuitMetrics(
            total_calls=100,
            failed_calls=30,
        )

        assert metrics.error_rate == 0.3

    def test_error_rate_zero_calls(self):
        """Test error rate with zero calls."""
        metrics = CircuitMetrics()
        assert metrics.error_rate == 0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = CircuitMetrics(
            total_calls=10,
            total_latency_ms=1000,
        )

        assert metrics.avg_latency_ms == 100

    def test_reset(self):
        """Test metrics reset."""
        metrics = CircuitMetrics(
            total_calls=100,
            failed_calls=30,
            consecutive_failures=5,
        )

        metrics.reset()

        assert metrics.total_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.consecutive_failures == 0


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.fixture
    def circuit(self):
        """Create a circuit breaker."""
        return CircuitBreaker(name="test_circuit")

    def test_initialization(self, circuit):
        """Test circuit breaker initialization."""
        assert circuit.name == "test_circuit"
        assert circuit.state == CircuitState.CLOSED
        assert circuit.metrics.total_calls == 0

    def test_get_status(self, circuit):
        """Test getting circuit status."""
        status = circuit.get_status()

        assert status["name"] == "test_circuit"
        assert status["state"] == "closed"
        assert "metrics" in status
        assert "config" in status

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit):
        """Test successful call through circuit."""
        async def success_func():
            return "success"

        result = await circuit.call(success_func)

        assert result == "success"
        assert circuit.metrics.total_calls == 1
        assert circuit.metrics.successful_calls == 1
        assert circuit.metrics.failed_calls == 0

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit):
        """Test failed call through circuit."""
        async def fail_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await circuit.call(fail_func)

        assert circuit.metrics.total_calls == 1
        assert circuit.metrics.failed_calls == 1
        assert circuit.metrics.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_trips_on_consecutive_failures(self):
        """Test circuit trips after consecutive failures."""
        config = CircuitConfig(consecutive_failures_threshold=3)
        circuit = CircuitBreaker(name="test", config=config)

        async def fail_func():
            raise ValueError("Test error")

        # Trigger failures
        for _ in range(3):
            try:
                await circuit.call(fail_func)
            except ValueError:
                pass

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self):
        """Test circuit blocks calls when open."""
        circuit = CircuitBreaker(name="test")

        # Force open
        await circuit.force_open("Test")

        async def some_func():
            return "result"

        with pytest.raises(CircuitOpenError):
            await circuit.call(some_func)

    @pytest.mark.asyncio
    async def test_force_open(self, circuit):
        """Test force opening circuit."""
        await circuit.force_open("Manual test")

        assert circuit.state == CircuitState.OPEN
        assert circuit.trip_reason == TripReason.MANUAL

    @pytest.mark.asyncio
    async def test_force_close(self, circuit):
        """Test force closing circuit."""
        await circuit.force_open("Test")
        await circuit.force_close()

        assert circuit.state == CircuitState.CLOSED
        assert circuit.trip_reason is None

    @pytest.mark.asyncio
    async def test_reset(self, circuit):
        """Test resetting circuit."""
        await circuit.force_open("Test")
        await circuit.reset()

        assert circuit.state == CircuitState.CLOSED
        assert circuit.metrics.total_calls == 0

    @pytest.mark.asyncio
    async def test_callback_on_trip(self):
        """Test callback is called when circuit trips."""
        callback_called = []

        async def on_trip(name, reason, details):
            callback_called.append((name, reason, details))

        config = CircuitConfig(consecutive_failures_threshold=2)
        circuit = CircuitBreaker(name="test", config=config, on_trip=on_trip)

        async def fail_func():
            raise ValueError("Error")

        for _ in range(2):
            try:
                await circuit.call(fail_func)
            except ValueError:
                pass

        assert len(callback_called) == 1
        assert callback_called[0][0] == "test"
        assert callback_called[0][1] == TripReason.CONSECUTIVE_FAILURES

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        """Test successful calls in half-open state close circuit."""
        config = CircuitConfig(
            consecutive_failures_threshold=2,
            reset_timeout_seconds=0,  # Immediate transition
            half_open_max_calls=1,
        )
        circuit = CircuitBreaker(name="test", config=config)

        # Trip the circuit
        async def fail_func():
            raise ValueError("Error")

        for _ in range(2):
            try:
                await circuit.call(fail_func)
            except ValueError:
                pass

        assert circuit.state == CircuitState.OPEN

        # Wait for half-open
        await asyncio.sleep(0.1)

        # Successful call should close
        async def success_func():
            return "ok"

        result = await circuit.call(success_func)
        assert result == "ok"
        assert circuit.state == CircuitState.CLOSED


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager class."""

    @pytest.fixture
    def manager(self):
        """Create circuit breaker manager."""
        return CircuitBreakerManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.circuits) == 0

    def test_create_circuit(self, manager):
        """Test creating a circuit."""
        circuit = manager.create("trading")

        assert "trading" in manager.circuits
        assert circuit.name == "trading"

    def test_get_circuit(self, manager):
        """Test getting a circuit."""
        manager.create("trading")

        circuit = manager.get("trading")
        assert circuit is not None

        missing = manager.get("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_force_open_all(self, manager):
        """Test opening all circuits."""
        manager.create("trading")
        manager.create("data_feed")

        await manager.force_open_all("Emergency")

        assert manager.circuits["trading"].state == CircuitState.OPEN
        assert manager.circuits["data_feed"].state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_force_close_all(self, manager):
        """Test closing all circuits."""
        circuit1 = manager.create("trading")
        circuit2 = manager.create("data_feed")

        await circuit1.force_open("Test")
        await circuit2.force_open("Test")

        await manager.force_close_all()

        assert circuit1.state == CircuitState.CLOSED
        assert circuit2.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_all(self, manager):
        """Test resetting all circuits."""
        circuit = manager.create("trading")
        await circuit.force_open("Test")

        await manager.reset_all()

        assert circuit.state == CircuitState.CLOSED

    def test_get_all_status(self, manager):
        """Test getting status of all circuits."""
        manager.create("trading")
        manager.create("data_feed")

        status = manager.get_all_status()

        assert status["total_circuits"] == 2
        assert "circuits" in status
        assert "trading" in status["circuits"]
        assert "data_feed" in status["circuits"]


class TestPreConfiguredCircuitBreakers:
    """Test pre-configured circuit breaker factories."""

    def test_create_trading_circuit_breaker(self):
        """Test trading circuit breaker factory."""
        circuit = create_trading_circuit_breaker()

        assert circuit.name == "trading"
        assert circuit.config.error_rate_threshold == 0.3
        assert circuit.config.consecutive_failures_threshold == 3

    def test_create_data_feed_circuit_breaker(self):
        """Test data feed circuit breaker factory."""
        circuit = create_data_feed_circuit_breaker()

        assert circuit.name == "data_feed"
        assert circuit.config.error_rate_threshold == 0.5
        assert circuit.config.consecutive_failures_threshold == 5

    def test_create_api_circuit_breaker(self):
        """Test API circuit breaker factory."""
        circuit = create_api_circuit_breaker()

        assert circuit.name == "api"
        assert circuit.config.error_rate_threshold == 0.4
        assert circuit.config.latency_threshold_ms == 3000


class TestCircuitOpenError:
    """Test CircuitOpenError exception."""

    def test_error_creation(self):
        """Test error creation."""
        error = CircuitOpenError("Circuit is open", reason=TripReason.ERROR_RATE)

        assert str(error) == "Circuit is open"
        assert error.reason == TripReason.ERROR_RATE

    def test_error_without_reason(self):
        """Test error without reason."""
        error = CircuitOpenError("Circuit is open")

        assert error.reason is None
