"""
Structured Logging Configuration.

Provides enhanced logging with:
- JSON structured output for production
- Colored console output for development
- Trade-specific logging
- Performance metrics tracking
- Context managers for operation tracking

Usage:
    from bot.core.logging_config import setup_logging, get_logger, log_trade

    # Setup at application start
    setup_logging(level="INFO", json_format=False)

    # Get a logger
    logger = get_logger(__name__)

    # Log a trade
    log_trade(symbol="BTC/USDT", side="buy", quantity=0.1, price=50000)

    # Track operation timing
    with log_operation("fetch_data"):
        data = fetch_data()
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add location
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Format extra data if present
        extra = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra_str = " | ".join(f"{k}={v}" for k, v in record.extra_data.items())
            extra = f" [{extra_str}]"

        formatted = super().format(record)
        return formatted + extra


class ExtraDataAdapter(logging.LoggerAdapter):
    """Logger adapter that supports extra data."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if "extra_data" not in extra:
            extra["extra_data"] = {}
        extra["extra_data"].update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
    include_trade_logger: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for structured logging
        log_file: Optional file path for logging
        include_trade_logger: Setup separate trade logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Trade logger
    if include_trade_logger:
        trade_logger = logging.getLogger("trades")
        trade_logger.setLevel(logging.INFO)

        trade_handler = logging.FileHandler("data/logs/trades.jsonl")
        trade_handler.setFormatter(JSONFormatter())
        trade_logger.addHandler(trade_handler)


def get_logger(name: str, **extra) -> logging.Logger:
    """
    Get a logger with optional extra context.

    Args:
        name: Logger name (usually __name__)
        **extra: Extra context to include in all logs

    Returns:
        Logger or LoggerAdapter with extra context
    """
    logger = logging.getLogger(name)
    if extra:
        return ExtraDataAdapter(logger, extra)
    return logger


@dataclass
class TradeLogEntry:
    """Structured trade log entry."""

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = "market"
    order_id: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: float = 0.0
    slippage_bps: float = 0.0
    execution_time_ms: float = 0.0
    strategy: Optional[str] = None
    regime: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type,
            "notional": self.quantity * self.price,
        }

        if self.order_id:
            data["order_id"] = self.order_id
        if self.pnl is not None:
            data["pnl"] = self.pnl
            data["pnl_pct"] = self.pnl_pct
        if self.fees:
            data["fees"] = self.fees
        if self.slippage_bps:
            data["slippage_bps"] = self.slippage_bps
        if self.execution_time_ms:
            data["execution_time_ms"] = self.execution_time_ms
        if self.strategy:
            data["strategy"] = self.strategy
        if self.regime:
            data["regime"] = self.regime
        if self.extra:
            data.update(self.extra)

        return data


_trade_logger = logging.getLogger("trades")


def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    **kwargs,
) -> None:
    """
    Log a trade with structured data.

    Args:
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Trade quantity
        price: Execution price
        **kwargs: Additional trade data
    """
    entry = TradeLogEntry(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        **kwargs,
    )

    record = logging.LogRecord(
        name="trades",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=f"TRADE: {side.upper()} {quantity} {symbol} @ {price}",
        args=(),
        exc_info=None,
    )
    record.extra_data = entry.to_dict()

    _trade_logger.handle(record)


@dataclass
class OperationMetrics:
    """Metrics for an operation."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@contextmanager
def log_operation(
    name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """
    Context manager to log operation timing.

    Args:
        name: Operation name
        logger: Logger to use (defaults to root)
        level: Log level

    Yields:
        OperationMetrics

    Example:
        with log_operation("fetch_prices") as metrics:
            prices = fetch_prices()
        print(f"Took {metrics.duration_ms}ms")
    """
    log = logger or logging.getLogger()
    metrics = OperationMetrics(name=name, start_time=time.perf_counter())

    try:
        yield metrics
        metrics.success = True
    except Exception as e:
        metrics.success = False
        metrics.error = str(e)
        raise
    finally:
        metrics.end_time = time.perf_counter()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000

        record = logging.LogRecord(
            name=log.name,
            level=level,
            pathname="",
            lineno=0,
            msg=f"Operation '{name}' completed in {metrics.duration_ms:.2f}ms",
            args=(),
            exc_info=None,
        )
        record.extra_data = metrics.to_dict()
        log.handle(record)


def log_timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use
        level: Log level

    Example:
        @log_timed()
        def expensive_function():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log = logger or logging.getLogger(func.__module__)
            with log_operation(func.__name__, log, level):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_timed_async(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """
    Decorator to log async function execution time.

    Args:
        logger: Logger to use
        level: Log level
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            start = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                record = logging.LogRecord(
                    name=log.name,
                    level=level,
                    pathname="",
                    lineno=0,
                    msg=f"Async '{func.__name__}' completed in {duration_ms:.2f}ms",
                    args=(),
                    exc_info=None,
                )
                record.extra_data = {
                    "operation": func.__name__,
                    "duration_ms": duration_ms,
                    "success": success,
                    "error": error,
                }
                log.handle(record)

            return result

        return wrapper

    return decorator


class MetricsCollector:
    """
    Simple metrics collector for observability.

    Example:
        metrics = MetricsCollector()
        metrics.increment("trades_executed")
        metrics.gauge("portfolio_value", 10000)
        metrics.timing("order_latency_ms", 45.2)
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timings: Dict[str, list] = {}

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value

    def timing(self, name: str, value_ms: float) -> None:
        """Record a timing."""
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(value_ms)
        # Keep last 1000 timings
        if len(self._timings[name]) > 1000:
            self._timings[name] = self._timings[name][-1000:]

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        result = {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "timings": {},
        }

        for name, values in self._timings.items():
            if values:
                result["timings"][name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }

        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._timings.clear()

    def get_counters(self) -> Dict[str, int]:
        """Get all counter values."""
        return self._counters.copy()

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge values."""
        return self._gauges.copy()

    def get_timings(self) -> Dict[str, list]:
        """Get all timing values."""
        return {k: v.copy() for k, v in self._timings.items()}


# Global metrics collector
metrics = MetricsCollector()
