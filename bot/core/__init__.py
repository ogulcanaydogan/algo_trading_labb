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

from .health import (
    ComponentHealth,
    GracefulShutdown,
    HealthChecker,
    HealthStatus,
    SystemHealth,
    create_database_health_check,
    create_exchange_health_check,
    graceful_shutdown,
    health_checker,
)

from .prometheus import (
    Counter,
    Gauge,
    Histogram,
    PrometheusRegistry,
    Summary,
    counter,
    gauge,
    histogram,
    summary,
    default_registry,
    trading_registry,
    trades_total,
    trade_pnl,
    position_size,
    order_latency,
    data_fetch_errors,
    model_predictions,
    signal_strength,
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
    # Health
    "ComponentHealth",
    "GracefulShutdown",
    "HealthChecker",
    "HealthStatus",
    "SystemHealth",
    "create_database_health_check",
    "create_exchange_health_check",
    "graceful_shutdown",
    "health_checker",
    # Prometheus
    "Counter",
    "Gauge",
    "Histogram",
    "PrometheusRegistry",
    "Summary",
    "counter",
    "gauge",
    "histogram",
    "summary",
    "default_registry",
    "trading_registry",
    "trades_total",
    "trade_pnl",
    "position_size",
    "order_latency",
    "data_fetch_errors",
    "model_predictions",
    "signal_strength",
]
