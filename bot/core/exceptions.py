"""
Exception handling utilities for the trading bot.

Provides:
- Custom exception hierarchy for trading operations
- Exception handling decorators
- Safe execution context managers
- Error classification and logging
"""

from __future__ import annotations

import functools
import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"  # Recoverable, log and continue
    MEDIUM = "medium"  # May need attention, but not critical
    HIGH = "high"  # Requires attention, may affect operations
    CRITICAL = "critical"  # System-level issue, may need shutdown


class ErrorCategory(Enum):
    """Categories for error classification."""

    DATA = "data"  # Data fetching/parsing errors
    NETWORK = "network"  # Network/API connectivity issues
    EXCHANGE = "exchange"  # Exchange-specific errors
    VALIDATION = "validation"  # Input validation failures
    EXECUTION = "execution"  # Trade execution errors
    MODEL = "model"  # ML model errors
    SYSTEM = "system"  # System-level errors
    CONFIGURATION = "configuration"  # Configuration errors
    UNKNOWN = "unknown"  # Unclassified errors


# Custom Exception Hierarchy


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class DataError(TradingBotError):
    """Error related to data fetching or parsing."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.MEDIUM),
            category=ErrorCategory.DATA,
            **kwargs,
        )


class NetworkError(TradingBotError):
    """Error related to network connectivity."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=ErrorCategory.NETWORK,
            **kwargs,
        )


class ExchangeError(TradingBotError):
    """Error from exchange operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=ErrorCategory.EXCHANGE,
            **kwargs,
        )


class ValidationError(TradingBotError):
    """Error from input validation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.LOW),
            category=ErrorCategory.VALIDATION,
            **kwargs,
        )


class ExecutionError(TradingBotError):
    """Error during trade execution."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=ErrorCategory.EXECUTION,
            **kwargs,
        )


class ModelError(TradingBotError):
    """Error related to ML models."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.MEDIUM),
            category=ErrorCategory.MODEL,
            **kwargs,
        )


class ConfigurationError(TradingBotError):
    """Error in configuration."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=kwargs.pop("severity", ErrorSeverity.HIGH),
            category=ErrorCategory.CONFIGURATION,
            **kwargs,
        )


# Exception Mapping for Classification

EXCEPTION_MAPPING: Dict[Type[Exception], tuple[ErrorCategory, ErrorSeverity]] = {
    # Network errors
    ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
    TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
    # Value errors
    ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
    TypeError: (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
    KeyError: (ErrorCategory.DATA, ErrorSeverity.LOW),
    # File errors
    FileNotFoundError: (ErrorCategory.DATA, ErrorSeverity.MEDIUM),
    PermissionError: (ErrorCategory.SYSTEM, ErrorSeverity.HIGH),
    # System errors
    MemoryError: (ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL),
    OSError: (ErrorCategory.SYSTEM, ErrorSeverity.HIGH),
}


def classify_exception(exc: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
    """Classify an exception into category and severity."""
    # Check direct mapping
    for exc_type, (category, severity) in EXCEPTION_MAPPING.items():
        if isinstance(exc, exc_type):
            return category, severity

    # Check if it's a TradingBotError
    if isinstance(exc, TradingBotError):
        return exc.category, exc.severity

    # Default classification
    return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    traceback_str: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": type(self.exception).__name__,
            "message": str(self.exception),
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_str,
        }


class ErrorTracker:
    """Track and aggregate errors."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._errors: List[ErrorRecord] = []
        self._error_counts: Dict[str, int] = {}

    def record(
        self,
        exc: Exception,
        context: str = "",
        include_traceback: bool = True,
    ) -> ErrorRecord:
        """Record an error."""
        category, severity = classify_exception(exc)

        record = ErrorRecord(
            exception=exc,
            category=category,
            severity=severity,
            context=context,
            traceback_str=traceback.format_exc() if include_traceback else None,
        )

        self._errors.append(record)
        if len(self._errors) > self.max_history:
            self._errors = self._errors[-self.max_history:]

        # Update counts
        error_key = f"{type(exc).__name__}:{context}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

        return record

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return [e.to_dict() for e in self._errors[-limit:]]

    def get_counts(self) -> Dict[str, int]:
        """Get error counts by type."""
        return self._error_counts.copy()

    def get_by_severity(self, severity: ErrorSeverity) -> List[ErrorRecord]:
        """Get errors by severity."""
        return [e for e in self._errors if e.severity == severity]

    def clear(self) -> None:
        """Clear error history."""
        self._errors.clear()
        self._error_counts.clear()


# Global error tracker
error_tracker = ErrorTracker()


# Decorators for exception handling


def handle_exceptions(
    default: T = None,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    log_level: int = logging.ERROR,
    reraise: bool = False,
    context: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for handling exceptions with proper logging.

    Args:
        default: Default value to return on exception
        exceptions: Tuple of exception types to catch
        log_level: Logging level for caught exceptions
        reraise: Whether to re-raise the exception after logging
        context: Context string for error tracking

    Example:
        @handle_exceptions(default=[], exceptions=(ValueError, KeyError))
        def fetch_data():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                ctx = context or func.__name__
                category, severity = classify_exception(e)

                logger.log(
                    log_level,
                    f"[{category.value.upper()}] {ctx}: {type(e).__name__}: {e}",
                )

                error_tracker.record(e, ctx)

                if reraise:
                    raise

                return default

        return wrapper

    return decorator


def handle_exceptions_async(
    default: T = None,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    log_level: int = logging.ERROR,
    reraise: bool = False,
    context: str = "",
) -> Callable:
    """Async version of handle_exceptions decorator."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                ctx = context or func.__name__
                category, severity = classify_exception(e)

                logger.log(
                    log_level,
                    f"[{category.value.upper()}] {ctx}: {type(e).__name__}: {e}",
                )

                error_tracker.record(e, ctx)

                if reraise:
                    raise

                return default

        return wrapper

    return decorator


@contextmanager
def safe_execution(
    context: str = "",
    default: T = None,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    log_level: int = logging.ERROR,
    reraise: bool = False,
):
    """
    Context manager for safe execution with exception handling.

    Example:
        with safe_execution("fetching prices", default={}):
            prices = fetch_prices()
    """
    try:
        yield
    except exceptions as e:
        category, severity = classify_exception(e)

        logger.log(
            log_level,
            f"[{category.value.upper()}] {context}: {type(e).__name__}: {e}",
        )

        error_tracker.record(e, context)

        if reraise:
            raise


def log_exception(
    exc: Exception,
    context: str = "",
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
) -> ErrorRecord:
    """
    Log an exception with proper classification.

    Args:
        exc: The exception to log
        context: Context string describing where the error occurred
        log_level: Logging level to use
        include_traceback: Whether to include the traceback

    Returns:
        ErrorRecord for the logged exception
    """
    category, severity = classify_exception(exc)

    message = f"[{category.value.upper()}] {context}: {type(exc).__name__}: {exc}"

    if include_traceback:
        logger.log(log_level, message, exc_info=True)
    else:
        logger.log(log_level, message)

    return error_tracker.record(exc, context, include_traceback)


def retry_on_exception(
    max_retries: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    context: str = "",
) -> Callable:
    """
    Decorator for retrying functions on exception.

    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        context: Context string for logging
    """
    import time

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            ctx = context or func.__name__
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(
                            f"{ctx}: Failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{ctx}: Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # Should never reach here
            raise RuntimeError(f"{ctx}: Unexpected retry loop exit")

        return wrapper

    return decorator


def retry_on_exception_async(
    max_retries: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    context: str = "",
) -> Callable:
    """Async version of retry_on_exception decorator."""
    import asyncio

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            ctx = context or func.__name__
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(
                            f"{ctx}: Failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{ctx}: Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise RuntimeError(f"{ctx}: Unexpected retry loop exit")

        return wrapper

    return decorator
