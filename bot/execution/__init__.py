"""
Execution module for order management and execution quality.

Provides:
- Slippage tracking and analytics
- Smart order execution (TWAP/VWAP)
- Iceberg orders
- Smart order routing
- Execution simulation (slippage, fees, partial fills)
- Reconciliation and idempotency
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

from .execution_algos import (
    AlgoStatus,
    UrgencyLevel,
    AlgoOrder,
    AlgoSlice,
    AlgoExecution,
    ExecutionAlgorithm as BaseExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    POVAlgorithm,
    ImplementationShortfallAlgorithm,
    IcebergAlgorithm,
    AdaptiveAlgorithm,
    AlgorithmFactory,
    create_execution_algorithm,
)

from .execution_simulator import (
    ExecutionResult,
    ExecutionSimulator,
    FeeSchedule,
    LatencyModel,
    PartialFillModel,
    SimulatorConfig,
    SlippageModel,
    get_execution_simulator,
    reset_execution_simulator,
)

from .reconciler import (
    PositionRecord,
    ReconciliationResult,
    ReconciliationStatus,
    Reconciler,
    ReconcilerConfig,
    TransactionRecord,
    TransactionState,
    get_reconciler,
    reset_reconciler,
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
    # Advanced Execution Algorithms
    "AlgoStatus",
    "UrgencyLevel",
    "AlgoOrder",
    "AlgoSlice",
    "AlgoExecution",
    "BaseExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "POVAlgorithm",
    "ImplementationShortfallAlgorithm",
    "IcebergAlgorithm",
    "AdaptiveAlgorithm",
    "AlgorithmFactory",
    "create_execution_algorithm",
    # Execution Simulator
    "ExecutionResult",
    "ExecutionSimulator",
    "FeeSchedule",
    "LatencyModel",
    "PartialFillModel",
    "SimulatorConfig",
    "SlippageModel",
    "get_execution_simulator",
    "reset_execution_simulator",
    # Reconciliation
    "PositionRecord",
    "ReconciliationResult",
    "ReconciliationStatus",
    "Reconciler",
    "ReconcilerConfig",
    "TransactionRecord",
    "TransactionState",
    "get_reconciler",
    "reset_reconciler",
]
