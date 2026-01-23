"""
Enhanced logging configuration with correlation IDs and structured logging.
Provides centralized logging setup with request tracking and performance monitoring.
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None


# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord with structured data support."""

    def __init__(
        self, name, level, pathname, lineno, msg, args, exc_info, func=None, sinfo=None, **kwargs
    ):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, func, sinfo)
        self.extra_data = kwargs


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation context to all log messages."""

    def process(self, msg, kwargs):
        # Get correlation data from context
        correlation_data = {
            "correlation_id": correlation_id_var.get(),
            "user_id": user_id_var.get(),
            "request_id": request_id_var.get(),
            "timestamp": datetime.utcnow().isoformat(),
            "logger_name": self.logger.name,
        }

        # Extract extra data if present
        extra = kwargs.get("extra", {})

        # Merge correlation data with existing extra data
        if isinstance(extra, dict):
            extra.update(correlation_data)
        else:
            kwargs["extra"] = {**correlation_data, "structured_data": extra}
            kwargs["extra"].update(correlation_data)

        return msg, kwargs


class PerformanceLogger:
    """Tracks performance metrics for operations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.operations: Dict[str, list] = {}

    @contextmanager
    def log_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for logging operation performance."""
        start_time = time.time()
        correlation_id = correlation_id_var.get() or str(uuid.uuid4())

        # Set correlation ID in context
        correlation_id_var.set(correlation_id)

        # Log operation start
        self.logger.info(
            f"Starting {operation_name}",
            extra={
                "operation": operation_name,
                "phase": "start",
                "context": context or {},
                "correlation_id": correlation_id,
            },
        )

        try:
            yield
            duration_ms = (time.time() - start_time) * 1000

            # Log successful completion
            self.logger.info(
                f"Completed {operation_name}",
                extra={
                    "operation": operation_name,
                    "phase": "complete",
                    "duration_ms": round(duration_ms, 2),
                    "context": context or {},
                    "correlation_id": correlation_id,
                },
            )

            # Track performance metrics
            if operation_name not in self.operations:
                self.operations[operation_name] = []

            self.operations[operation_name].append(duration_ms)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log failure
            self.logger.error(
                f"Failed {operation_name}: {str(e)}",
                extra={
                    "operation": operation_name,
                    "phase": "error",
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "context": context or {},
                    "correlation_id": correlation_id,
                },
                exc_info=True,
            )

            # Track failed operations
            if operation_name not in self.operations:
                self.operations[operation_name] = []

            self.operations[operation_name].append(duration_ms)
            raise

    def get_performance_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operations."""
        if operation_name:
            if operation_name not in self.operations:
                return {}

            durations = self.operations[operation_name]
            return {
                "operation": operation_name,
                "total_calls": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": self._percentile(durations, 95),
                "p99_duration_ms": self._percentile(durations, 99),
            }

        # Return stats for all operations
        return {name: self.get_performance_stats(name) for name in self.operations.keys()}

    def _percentile(self, data: list, percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index % 1)


def setup_structured_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    log_file: Optional[str] = None,
) -> Dict[str, logging.Logger]:
    """Setup structured logging with correlation IDs."""

    loggers = {}

    # Default structured format
    if log_format is None:
        log_format = (
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "correlation_id=%(correlation_id)s user_id=%(user_id)s "
            "request_id=%(request_id)s"
        )

    # JSON formatter for structured logging
    if jsonlogger is not None:
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "correlation_id=%(correlation_id)s user_id=%(user_id)s "
            "request_id=%(request_id)s operation=%(operation)s "
            "duration_ms=%(duration_ms)s"
        )
    else:
        # Fallback to regular formatter
        json_formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "correlation_id=%(correlation_id)s user_id=%(user_id)s "
            "request_id=%(request_id)s operation=%(operation)s "
            "duration_ms=%(duration_ms)s"
        )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(log_format) if not enable_file else json_formatter
    )
    console_handler.setLevel(getattr(logging, level.upper()))

    # Setup file handler if enabled
    file_handler = None
    if enable_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(getattr(logging, level.upper()))

    # Configure trading logger
    trading_logger = logging.getLogger("trading")
    trading_logger.setLevel(getattr(logging, level.upper()))
    trading_logger.addHandler(console_handler)
    if file_handler:
        trading_logger.addHandler(file_handler)
    trading_logger.propagate = False

    # Configure API logger
    api_logger = logging.getLogger("api")
    api_logger.setLevel(getattr(logging, level.upper()))
    api_logger.addHandler(console_handler)
    if file_handler:
        api_logger.addHandler(file_handler)
    api_logger.propagate = False

    # Configure system logger
    system_logger = logging.getLogger("system")
    system_logger.setLevel(getattr(logging, level.upper()))
    system_logger.addHandler(console_handler)
    if file_handler:
        system_logger.addHandler(file_handler)
    system_logger.propagate = False

    # Return structured logger adapters
    loggers["trading"] = StructuredLoggerAdapter(trading_logger, {})
    loggers["api"] = StructuredLoggerAdapter(api_logger, {})
    loggers["system"] = StructuredLoggerAdapter(system_logger, {})

    return loggers


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """Context manager for setting correlation variables."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    # Set context variables
    token_correlation = correlation_id_var.set(correlation_id)
    token_user = user_id_var.set(user_id)
    token_request = request_id_var.set(request_id)

    try:
        yield {
            "correlation_id": correlation_id,
            "user_id": user_id,
            "request_id": request_id,
        }
    finally:
        # Reset context variables
        correlation_id_var.reset(token_correlation)
        user_id_var.reset(token_user)
        request_id_var.reset(token_request)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get() or str(uuid.uuid4())


def with_correlation_id(func):
    """Decorator to add correlation ID to function calls."""

    def wrapper(*args, **kwargs):
        correlation_id = get_correlation_id()
        with correlation_context(correlation_id):
            return func(*args, **kwargs)

    return wrapper


class RequestLogger:
    """Logs HTTP requests with correlation tracking."""

    def __init__(self, logger: logging.Logger):
        self.logger = StructuredLoggerAdapter(logger, {})

    def log_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """Log incoming HTTP request."""
        with correlation_context(correlation_id=correlation_id, user_id=user_id):
            self.logger.info(
                f"HTTP {method} {path}",
                extra={
                    "event_type": "http_request",
                    "method": method,
                    "path": path,
                    "user_agent": headers.get("User-Agent", ""),
                    "content_type": headers.get("Content-Type", ""),
                },
            )

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        response_size: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ):
        """Log HTTP response."""
        with correlation_context(correlation_id=correlation_id):
            level = logging.ERROR if status_code >= 400 else logging.INFO

            self.logger.log(
                level,
                f"HTTP {method} {path} -> {status_code}",
                extra={
                    "event_type": "http_response",
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "response_size": response_size,
                    "is_error": status_code >= 400,
                },
            )

    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        duration_ms: float,
        correlation_id: Optional[str] = None,
    ):
        """Log HTTP error."""
        with correlation_context(correlation_id=correlation_id):
            self.logger.error(
                f"HTTP {method} {path} -> ERROR: {str(error)}",
                extra={
                    "event_type": "http_error",
                    "method": method,
                    "path": path,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "duration_ms": duration_ms,
                },
                exc_info=True,
            )


# Global performance logger instance
_performance_logger = None


def get_performance_logger(logger_name: str = "trading") -> PerformanceLogger:
    """Get performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        base_logger = logging.getLogger(logger_name)
        _performance_logger = PerformanceLogger(base_logger)
    return _performance_logger


def log_trading_operation(operation_name: str):
    """Decorator to log trading operations with correlation tracking."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            perf_logger = get_performance_logger("trading")
            with perf_logger.log_operation(
                operation_name, {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Initialize structured logging on module import
loggers = setup_structured_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    enable_file=bool(os.getenv("LOG_TO_FILE", "false").lower() == "true"),
    log_file=os.getenv("LOG_FILE", "logs/trading.log"),
)
