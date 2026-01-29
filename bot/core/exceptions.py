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


# Additional Exception Types for Trading

class InsufficientFundsError(ExecutionError):
    """Error when account has insufficient funds for a trade."""

    def __init__(self, message: str, required: float = 0, available: float = 0, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.details["required"] = required
        self.details["available"] = available


class OrderRejectedError(ExecutionError):
    """Error when an order is rejected by the exchange."""

    def __init__(self, message: str, order_id: str = "", reason: str = "", **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.details["order_id"] = order_id
        self.details["reason"] = reason


class RateLimitError(NetworkError):
    """Error when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = 0, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.details["retry_after"] = retry_after


class PositionLimitError(ValidationError):
    """Error when position limits are exceeded."""

    def __init__(self, message: str, max_allowed: float = 0, requested: float = 0, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.details["max_allowed"] = max_allowed
        self.details["requested"] = requested


class MarketClosedError(ExchangeError):
    """Error when trying to trade on a closed market."""

    def __init__(self, message: str, market: str = "", **kwargs):
        super().__init__(message, severity=ErrorSeverity.LOW, **kwargs)
        self.details["market"] = market


class PredictionError(ModelError):
    """Error during ML prediction."""

    def __init__(self, message: str, model_name: str = "", **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.details["model_name"] = model_name


class CircuitBreakerOpenError(TradingBotError):
    """Error when circuit breaker is open."""

    def __init__(self, message: str, breaker_name: str = "", cooldown_remaining: float = 0, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            **kwargs,
        )
        self.details["breaker_name"] = breaker_name
        self.details["cooldown_remaining"] = cooldown_remaining


# Error Alerting System

class ErrorAlertHandler:
    """Handler for sending alerts on critical errors."""

    def __init__(self):
        self._alert_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._async_alert_callbacks: List[Callable[[ErrorRecord], Any]] = []
        self._severity_threshold = ErrorSeverity.HIGH
        self._error_cooldowns: Dict[str, datetime] = {}
        self._cooldown_seconds = 300  # 5 minutes between same alerts

    def set_severity_threshold(self, severity: ErrorSeverity) -> None:
        """Set minimum severity level for alerts."""
        self._severity_threshold = severity

    def set_cooldown(self, seconds: int) -> None:
        """Set cooldown between alerts of the same type."""
        self._cooldown_seconds = seconds

    def register_callback(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Register a synchronous alert callback."""
        self._alert_callbacks.append(callback)

    def register_async_callback(self, callback: Callable[[ErrorRecord], Any]) -> None:
        """Register an asynchronous alert callback."""
        self._async_alert_callbacks.append(callback)

    def should_alert(self, record: ErrorRecord) -> bool:
        """Check if an alert should be sent for this error."""
        # Check severity threshold
        severity_order = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        if severity_order.index(record.severity) < severity_order.index(self._severity_threshold):
            return False

        # Check cooldown
        error_key = f"{type(record.exception).__name__}:{record.context}"
        last_alert = self._error_cooldowns.get(error_key)
        if last_alert:
            elapsed = (datetime.utcnow() - last_alert).total_seconds()
            if elapsed < self._cooldown_seconds:
                return False

        self._error_cooldowns[error_key] = datetime.utcnow()
        return True

    def alert(self, record: ErrorRecord) -> None:
        """Send alert for an error."""
        if not self.should_alert(record):
            return

        for callback in self._alert_callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

    async def alert_async(self, record: ErrorRecord) -> None:
        """Send async alert for an error."""
        if not self.should_alert(record):
            return

        import asyncio

        for callback in self._async_alert_callbacks:
            try:
                result = callback(record)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Async alert callback failed: {e}")


# Global alert handler
alert_handler = ErrorAlertHandler()


# Enhanced Error Tracker with Alerting

class EnhancedErrorTracker(ErrorTracker):
    """Enhanced error tracker with alerting and aggregation."""

    def __init__(self, max_history: int = 1000):
        super().__init__(max_history)
        self._alert_handler = alert_handler
        self._error_rates: Dict[str, List[datetime]] = {}
        self._rate_window_seconds = 60  # 1 minute window for rate calculation

    def record(
        self,
        exc: Exception,
        context: str = "",
        include_traceback: bool = True,
        send_alert: bool = True,
    ) -> ErrorRecord:
        """Record an error with optional alerting."""
        record = super().record(exc, context, include_traceback)

        # Track error rate
        error_key = f"{type(exc).__name__}:{context}"
        if error_key not in self._error_rates:
            self._error_rates[error_key] = []
        self._error_rates[error_key].append(datetime.utcnow())

        # Clean old entries
        cutoff = datetime.utcnow()
        self._error_rates[error_key] = [
            t for t in self._error_rates[error_key]
            if (cutoff - t).total_seconds() < self._rate_window_seconds
        ]

        # Send alert if needed
        if send_alert:
            self._alert_handler.alert(record)

        return record

    async def record_async(
        self,
        exc: Exception,
        context: str = "",
        include_traceback: bool = True,
        send_alert: bool = True,
    ) -> ErrorRecord:
        """Async version of record with async alerting."""
        record = super().record(exc, context, include_traceback)

        if send_alert:
            await self._alert_handler.alert_async(record)

        return record

    def get_error_rate(self, error_type: str, context: str = "") -> int:
        """Get the error rate (count per minute) for an error type."""
        error_key = f"{error_type}:{context}"
        if error_key not in self._error_rates:
            return 0
        return len(self._error_rates[error_key])

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top errors by frequency."""
        sorted_errors = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"error": key, "count": count}
            for key, count in sorted_errors[:limit]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of error statistics."""
        by_severity = {}
        by_category = {}

        for record in self._errors:
            sev = record.severity.value
            cat = record.category.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_errors": len(self._errors),
            "by_severity": by_severity,
            "by_category": by_category,
            "top_errors": self.get_top_errors(5),
            "unique_error_types": len(self._error_counts),
        }


# Replace global error tracker with enhanced version
enhanced_error_tracker = EnhancedErrorTracker()


# Convenience functions

def setup_error_alerts(
    telegram_callback: Optional[Callable] = None,
    discord_callback: Optional[Callable] = None,
    severity_threshold: ErrorSeverity = ErrorSeverity.HIGH,
) -> None:
    """
    Set up error alerting with notification callbacks.

    Example:
        setup_error_alerts(
            telegram_callback=send_telegram_alert,
            severity_threshold=ErrorSeverity.CRITICAL,
        )
    """
    if telegram_callback:
        alert_handler.register_callback(telegram_callback)
    if discord_callback:
        alert_handler.register_callback(discord_callback)
    alert_handler.set_severity_threshold(severity_threshold)


def get_error_summary() -> Dict[str, Any]:
    """Get a summary of recent errors."""
    return enhanced_error_tracker.get_summary()


def handle_critical_error(
    exc: Exception,
    context: str,
    shutdown_on_critical: bool = False,
) -> ErrorRecord:
    """
    Handle a critical error with full logging and alerting.

    Args:
        exc: The exception that occurred
        context: Description of where the error occurred
        shutdown_on_critical: Whether to trigger shutdown for critical errors
    """
    record = enhanced_error_tracker.record(exc, context, include_traceback=True)

    if record.severity == ErrorSeverity.CRITICAL:
        logger.critical(
            f"CRITICAL ERROR in {context}: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        if shutdown_on_critical:
            logger.critical("Initiating shutdown due to critical error")
            # Could trigger graceful shutdown here

    return record
