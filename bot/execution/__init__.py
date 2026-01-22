"""
Execution module for order management and execution quality.

Provides:
- Slippage tracking and analytics
- Smart order execution (TWAP/VWAP)
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
]
