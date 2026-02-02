"""
Safety Module.

Provides capital protection and risk management:
- Capital preservation mode with graduated response
- Automatic detection of performance degradation
- Dynamic trading restrictions
"""

from .capital_preservation import (
    CapitalPreservationMode,
    PreservationConfig,
    PreservationLevel,
    PreservationState,
    TradeMetrics,
    get_capital_preservation,
    reset_capital_preservation,
)

__all__ = [
    "CapitalPreservationMode",
    "PreservationConfig",
    "PreservationLevel",
    "PreservationState",
    "TradeMetrics",
    "get_capital_preservation",
    "reset_capital_preservation",
]
