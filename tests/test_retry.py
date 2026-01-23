"""Unit tests for retry and circuit breaker utilities."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from bot.retry import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RetryError,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test that failures trigger retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = failing_then_succeeding()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Test that RetryError is raised after max attempts."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")

        with pytest.raises(RetryError) as exc_info:
            always_failing()

        assert call_count == 3
        assert "failed after 3 attempts" in str(exc_info.value)
        assert exc_info.value.last_exception is not None

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            retryable_exceptions=(ConnectionError,),
            base_delay=0.01,
        )
        def raising_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            raising_value_error()

        assert call_count == 1  # No retries

    def test_on_retry_callback(self):
        """Test that on_retry callback is called on each retry."""
        retry_events = []

        def on_retry(exc, attempt):
            retry_events.append((str(exc), attempt))

        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            base_delay=0.01,
            on_retry=on_retry,
        )
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Failed")
            return "success"

        result = failing_then_succeeding()
        assert result == "success"
        assert len(retry_events) == 2  # Two retries before success

    def test_exponential_backoff_timing(self):
        """Test that delays increase exponentially."""
        delays = []
        original_sleep = time.sleep

        def mock_sleep(duration):
            delays.append(duration)
            # Don't actually sleep, just record the delay

        call_count = 0

        @retry_with_backoff(
            max_attempts=4,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable delays
        )
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Failed")

        with patch("time.sleep", mock_sleep):
            with pytest.raises(RetryError):
                always_failing()

        # Check delays: 1.0, 2.0, 4.0 (base * 2^attempt)
        assert len(delays) == 3
        assert delays[0] == pytest.approx(1.0, rel=0.1)
        assert delays[1] == pytest.approx(2.0, rel=0.1)
        assert delays[2] == pytest.approx(4.0, rel=0.1)

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        delays = []

        def mock_sleep(duration):
            delays.append(duration)

        @retry_with_backoff(
            max_attempts=5,
            base_delay=10.0,
            max_delay=15.0,
            exponential_base=2.0,
            jitter=False,
        )
        def always_failing():
            raise ConnectionError("Failed")

        with patch("time.sleep", mock_sleep):
            with pytest.raises(RetryError):
                always_failing()

        # All delays should be capped at 15.0
        for delay in delays:
            assert delay <= 15.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test that circuit breaker starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitBreaker.CLOSED

    def test_successful_calls_keep_circuit_closed(self):
        """Test that successful calls keep circuit closed."""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker
        def successful():
            return "success"

        for _ in range(10):
            result = successful()
            assert result == "success"

        assert breaker.state == CircuitBreaker.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        @breaker
        def failing():
            raise Exception("Failed")

        # Cause enough failures to open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                failing()

        assert breaker.state == CircuitBreaker.OPEN

    def test_open_circuit_rejects_calls(self):
        """Test that open circuit rejects calls immediately."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        call_count = 0

        @breaker
        def function():
            nonlocal call_count
            call_count += 1
            raise Exception("Failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                function()

        assert call_count == 2
        assert breaker.state == CircuitBreaker.OPEN

        # Further calls should be rejected without invoking function
        with pytest.raises(CircuitBreakerOpenError):
            function()

        assert call_count == 2  # Function not called

    def test_circuit_transitions_to_half_open(self):
        """Test that circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # Very short timeout for testing
        )

        @breaker
        def failing():
            raise Exception("Failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                failing()

        assert breaker.state == CircuitBreaker.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # State should transition to half-open
        assert breaker.state == CircuitBreaker.HALF_OPEN

    def test_successful_half_open_closes_circuit(self):
        """Test that successful call in half-open state closes circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        call_count = 0

        @breaker
        def function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failed")
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                function()

        assert breaker.state == CircuitBreaker.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)
        assert breaker.state == CircuitBreaker.HALF_OPEN

        # Successful call should close circuit
        result = function()
        assert result == "success"
        assert breaker.state == CircuitBreaker.CLOSED

    def test_failed_half_open_reopens_circuit(self):
        """Test that failed call in half-open state reopens circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        @breaker
        def always_failing():
            raise Exception("Failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                always_failing()

        assert breaker.state == CircuitBreaker.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)
        assert breaker.state == CircuitBreaker.HALF_OPEN

        # Failed call should reopen circuit
        with pytest.raises(Exception):
            always_failing()

        assert breaker.state == CircuitBreaker.OPEN

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def failing():
            raise Exception("Failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                failing()

        assert breaker.state == CircuitBreaker.OPEN

        # Reset manually
        breaker.reset()

        assert breaker.state == CircuitBreaker.CLOSED

    def test_half_open_max_calls_limit(self):
        """Test that half-open state limits concurrent test calls."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )

        call_count = 0

        @breaker
        def function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Initial failures")
            # Simulate slow recovery
            time.sleep(0.1)
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                function()

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitBreaker.HALF_OPEN

        # First call should be allowed
        result = function()
        assert result == "success"
