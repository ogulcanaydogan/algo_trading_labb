"""
Core module for dependency injection and shared infrastructure.
"""

from .container import (
    Container,
    Lifetime,
    Scope,
    container,
    get,
    inject,
    register,
)

from .validation import (
    DataQuality,
    OHLCVValidator,
    OrderValidator,
    PositionValidator,
    ValidationError,
    ValidationResult,
    validate_ohlcv,
    validate_order,
    validate_position,
)

from .async_utils import (
    AsyncTaskManager,
    RateLimiter,
    Semaphore,
    TimeoutContext,
    create_safe_callback,
    retry_async,
    run_with_semaphore,
    safe_timeout,
    wait_with_timeout,
)

from .logging_config import (
    MetricsCollector,
    OperationMetrics,
    TradeLogEntry,
    get_logger,
    log_operation,
    log_timed,
    log_timed_async,
    log_trade,
    metrics,
    setup_logging,
)

__all__ = [
    # Container
    "Container",
    "Lifetime",
    "Scope",
    "container",
    "get",
    "inject",
    "register",
    # Validation
    "DataQuality",
    "OHLCVValidator",
    "OrderValidator",
    "PositionValidator",
    "ValidationError",
    "ValidationResult",
    "validate_ohlcv",
    "validate_order",
    "validate_position",
    # Async utilities
    "AsyncTaskManager",
    "RateLimiter",
    "Semaphore",
    "TimeoutContext",
    "create_safe_callback",
    "retry_async",
    "run_with_semaphore",
    "safe_timeout",
    "wait_with_timeout",
    # Logging
    "MetricsCollector",
    "OperationMetrics",
    "TradeLogEntry",
    "get_logger",
    "log_operation",
    "log_timed",
    "log_timed_async",
    "log_trade",
    "metrics",
    "setup_logging",
]
