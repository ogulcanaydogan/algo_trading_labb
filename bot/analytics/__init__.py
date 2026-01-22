"""
Analytics module for comprehensive trading performance analysis.

Key components:
- PerformanceMetrics: Expectancy, Sharpe, Sortino, Calmar, Profit Factor
- TradeJournal: Learning from trades, pattern recognition
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

__all__ = [
    # Performance metrics
    'PerformanceMetrics',
    'calculate_all_metrics',
    'calculate_expectancy',
    'calculate_profit_factor',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'analyze_trading_performance',
    # Trade journal
    'TradeJournal',
    'TradeAnalysis',
    'TradeJournalSummary',
    'PatternInsight',
    'analyze_trades',
]
