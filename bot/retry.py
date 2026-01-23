"""Retry utilities with exponential backoff for resilient operations."""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Sequence, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including the first one)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types that should trigger retry
        on_retry: Optional callback called on each retry with (exception, attempt_number)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, retryable_exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc

                    if attempt == max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise RetryError(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            last_exception=exc,
                        ) from exc

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "Function %s attempt %d/%d failed: %s. Retrying in %.2fs...",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )

                    if on_retry:
                        on_retry(exc, attempt)

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise RetryError(
                f"Function {func.__name__} failed after {max_attempts} attempts",
                last_exception=last_exception,
            )

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for failing services.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests go through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        @breaker
        def call_external_service():
            return requests.get("https://api.example.com")
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open state
            half_open_max_calls: Number of test calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == self.OPEN:
            # Check if we should transition to half-open
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    def _record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            logger.info("Circuit breaker closed after successful recovery")

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            logger.warning("Circuit breaker opened again after half-open failure")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning("Circuit breaker opened after %d failures", self._failure_count)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use circuit breaker as a decorator."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_state = self.state

            if current_state == self.OPEN:
                raise CircuitBreakerOpenError(f"Circuit breaker is open for {func.__name__}")

            if current_state == self.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker half-open limit reached for {func.__name__}"
                    )
                self._half_open_calls += 1

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as exc:
                self._record_failure()
                raise

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    pass


# Pre-configured retry decorators for common use cases
def retry_network_operation(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator configured for network operations."""
    return retry_with_backoff(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )(func)


def retry_api_call(func: Callable[..., T]) -> Callable[..., T]:
    """Retry decorator configured for API calls."""
    return retry_with_backoff(
        max_attempts=5,
        base_delay=0.5,
        max_delay=15.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )(func)
