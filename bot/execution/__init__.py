"""
Execution module for order management and execution quality.

Provides:
- Slippage tracking and analytics
- Smart order execution (TWAP/VWAP)
- Iceberg orders
- Smart order routing
"""

from .slippage_tracker import (
    SlippageTracker,
    SlippageRecord,
    SlippageStats,
    get_slippage_tracker,
)

from .smart_order import (
    SmartOrderExecutor,
    SmartOrderResult,
    SliceOrder,
    ExecutionAlgorithm,
    TWAPConfig,
    VWAPConfig,
    create_smart_executor,
)

from .iceberg_order import (
    IcebergExecutor,
    IcebergOrder,
    IcebergSlice,
    IcebergState,
    IcebergConfig,
    create_iceberg_executor,
)

from .smart_router import (
    SmartOrderRouter,
    Venue,
    VenueType,
    VenueQuote,
    RouteDecision,
    RoutingStrategy,
    RoutingConfig,
    create_smart_router,
)

__all__ = [
    # Slippage tracking
    "SlippageTracker",
    "SlippageRecord",
    "SlippageStats",
    "get_slippage_tracker",
    # Smart order execution
    "SmartOrderExecutor",
    "SmartOrderResult",
    "SliceOrder",
    "ExecutionAlgorithm",
    "TWAPConfig",
    "VWAPConfig",
    "create_smart_executor",
    # Iceberg orders
    "IcebergExecutor",
    "IcebergOrder",
    "IcebergSlice",
    "IcebergState",
    "IcebergConfig",
    "create_iceberg_executor",
    # Smart routing
    "SmartOrderRouter",
    "Venue",
    "VenueType",
    "VenueQuote",
    "RouteDecision",
    "RoutingStrategy",
    "RoutingConfig",
    "create_smart_router",
]
