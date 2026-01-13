"""Structured JSON logging configuration for the trading bot."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing by log aggregation tools.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_location: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data: Dict[str, Any] = {}

        # Add timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()

        # Add log level
        if self.include_level:
            log_data["level"] = record.levelname
            log_data["level_num"] = record.levelno

        # Add logger name
        if self.include_logger:
            log_data["logger"] = record.name

        # Add message
        log_data["message"] = record.getMessage()

        # Add location info for debugging
        if self.include_location:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields from the record
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data["data"] = record.extra_data

        # Add static extra fields
        log_data.update(self.extra_fields)

        # Add any additional attributes set on the record
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "extra_data",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                if key not in log_data:
                    log_data[key] = value

        return json.dumps(log_data, default=str)


class StructuredLogger(logging.Logger):
    """
    Extended logger that supports structured logging with extra data.

    Example:
        logger = get_structured_logger("trading")
        logger.info_with_data("Signal generated", symbol="BTC/USDT", decision="LONG")
    """

    def _log_with_data(
        self,
        level: int,
        msg: str,
        args: tuple = (),
        exc_info: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Internal method to log with structured data."""
        extra = extra or {}
        extra["extra_data"] = kwargs
        self.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
        )

    def debug_with_data(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self._log_with_data(logging.DEBUG, msg, stacklevel=2, **kwargs)

    def info_with_data(self, msg: str, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self._log_with_data(logging.INFO, msg, stacklevel=2, **kwargs)

    def warning_with_data(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self._log_with_data(logging.WARNING, msg, stacklevel=2, **kwargs)

    def error_with_data(self, msg: str, **kwargs: Any) -> None:
        """Log error message with structured data."""
        self._log_with_data(logging.ERROR, msg, stacklevel=2, **kwargs)

    def critical_with_data(self, msg: str, **kwargs: Any) -> None:
        """Log critical message with structured data."""
        self._log_with_data(logging.CRITICAL, msg, stacklevel=2, **kwargs)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format (recommended for production)
        log_file: Optional file path for logging output
        extra_fields: Additional fields to include in every log record
    """
    # Set the custom logger class
    logging.setLoggerClass(StructuredLogger)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    if json_format:
        formatter = JSONFormatter(extra_fields=extra_fields)
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)
    return logger  # type: ignore


def configure_from_env() -> None:
    """Configure logging from environment variables."""
    level = os.getenv("LOG_LEVEL", "INFO")
    json_format = os.getenv("LOG_FORMAT", "json").lower() == "json"
    log_file = os.getenv("LOG_FILE")

    extra_fields: Dict[str, Any] = {}

    # Add service identification
    service_name = os.getenv("SERVICE_NAME", "algo-trading-bot")
    extra_fields["service"] = service_name

    # Add environment
    environment = os.getenv("ENVIRONMENT", "development")
    extra_fields["environment"] = environment

    setup_logging(
        level=level,
        json_format=json_format,
        log_file=log_file,
        extra_fields=extra_fields,
    )
