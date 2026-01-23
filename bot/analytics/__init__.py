"""
Analytics module for comprehensive trading performance analysis.

Key components:
- PerformanceMetrics: Expectancy, Sharpe, Sortino, Calmar, Profit Factor
- TradeJournal: Learning from trades, pattern recognition
- PnLAttributor: P&L attribution by strategy, asset, factor
"""

from .performance_metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_expectancy,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    analyze_trading_performance,
)

from .trade_journal import (
    TradeJournal,
    TradeAnalysis,
    TradeJournalSummary,
    PatternInsight,
    analyze_trades,
)

from .pnl_attribution import (
    PnLAttributor,
    AttributionResult,
    AttributionMethod,
    AttributionPeriod,
    Trade as AttributionTrade,
    FactorExposure,
    create_pnl_attributor,
)

__all__ = [
    # Performance metrics
    "PerformanceMetrics",
    "calculate_all_metrics",
    "calculate_expectancy",
    "calculate_profit_factor",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "analyze_trading_performance",
    # Trade journal
    "TradeJournal",
    "TradeAnalysis",
    "TradeJournalSummary",
    "PatternInsight",
    "analyze_trades",
    # P&L Attribution
    "PnLAttributor",
    "AttributionResult",
    "AttributionMethod",
    "AttributionPeriod",
    "AttributionTrade",
    "FactorExposure",
    "create_pnl_attributor",
]
