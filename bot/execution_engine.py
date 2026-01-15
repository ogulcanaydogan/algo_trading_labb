"""
Execution Engine Module - Alias for unified_execution.py

This module re-exports everything from unified_execution.py for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Re-export everything from unified_execution
from bot.unified_execution import (
    # Enums
    ExecutionMode,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    # Models
    FeeStructure,
    SlippageModel,
    Order,
    Fill,
    Position,
    ExecutionResult,
    # Engines
    ExecutionEngine,
    BacktestExecutionEngine,
    PaperExecutionEngine,
    LiveExecutionEngine,
    # Factory
    create_execution_engine,
    get_execution_engine,
    set_execution_engine,
)


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    mode: ExecutionMode = ExecutionMode.PAPER
    paper_mode: bool = True  # Backward compatibility alias
    fee_structure: Optional[FeeStructure] = None
    slippage_model: Optional[SlippageModel] = None

    # Risk settings
    max_slippage_pct: float = 0.5
    max_retries: int = 3
    retry_delay_ms: float = 100.0

    # Order settings
    default_time_in_force: str = "GTC"
    enable_partial_fills: bool = True

    # Connection settings
    timeout_ms: float = 5000.0

    # Metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Sync paper_mode with mode after initialization."""
        if self.paper_mode:
            self.mode = ExecutionMode.PAPER
        else:
            self.mode = ExecutionMode.LIVE


__all__ = [
    # Enums
    "ExecutionMode",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSide",
    # Models
    "FeeStructure",
    "SlippageModel",
    "Order",
    "Fill",
    "Position",
    "ExecutionResult",
    "ExecutionConfig",
    # Engines
    "ExecutionEngine",
    "BacktestExecutionEngine",
    "PaperExecutionEngine",
    "LiveExecutionEngine",
    # Factory
    "create_execution_engine",
    "get_execution_engine",
    "set_execution_engine",
]
