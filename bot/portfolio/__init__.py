"""
Portfolio module.

Provides:
- Portfolio runner for multi-asset deployments (PortfolioRunner, PortfolioConfig)
- Kelly Criterion and Risk Parity allocation strategies (PortfolioAllocator)
"""

# Re-export original portfolio runner classes
from bot.portfolio_runner import (
    PortfolioAssetConfig,
    PortfolioConfig,
    PortfolioRunner,
)

# New allocator module exports
from bot.portfolio.allocator import (
    PortfolioAllocator,
    AllocationMethod,
    AssetAllocation,
    PortfolioAllocationResult,
    BacktestStats,
    kelly_fraction,
    half_kelly_fraction,
    quarter_kelly_fraction,
    calculate_edge,
    create_allocator_from_backtests,
)

# Integration helpers
from bot.portfolio.integration import (
    PortfolioPositionSizer,
    PositionSizeResult,
    calculate_position_size_with_kelly,
    extract_stats_from_backtest_result,
    integrate_with_trading_engine,
)

__all__ = [
    # Original portfolio runner classes
    "PortfolioAssetConfig",
    "PortfolioConfig",
    "PortfolioRunner",
    # New allocator classes
    "PortfolioAllocator",
    "AllocationMethod",
    "AssetAllocation",
    "PortfolioAllocationResult",
    "BacktestStats",
    "kelly_fraction",
    "half_kelly_fraction",
    "quarter_kelly_fraction",
    "calculate_edge",
    "create_allocator_from_backtests",
    # Integration helpers
    "PortfolioPositionSizer",
    "PositionSizeResult",
    "calculate_position_size_with_kelly",
    "extract_stats_from_backtest_result",
    "integrate_with_trading_engine",
]
