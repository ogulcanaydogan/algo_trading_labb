"""
Meta-Learning Module.

Provides self-improvement capabilities:
- Failure analysis
- Strategy evolution
- Pattern discovery
- Post-trade forensics (MAE/MFE analysis)
"""

from .failure_analyzer import (
    FailureAnalyzer,
    FailureCategory,
    FailurePattern,
    ImprovementRecommendation,
    TradeAnalysis,
    get_failure_analyzer,
)
from .strategy_evolver import (
    EvolutionConfig,
    StrategyEvolver,
    StrategyGene,
    get_strategy_evolver,
)
from .trade_forensics import (
    AggregateStats,
    EntryQuality,
    ExitQuality,
    ForensicsConfig,
    ForensicsResult,
    StopQuality,
    TradeForensics,
    get_trade_forensics,
    reset_trade_forensics,
)

__all__ = [
    # Failure Analyzer
    "FailureAnalyzer",
    "FailureCategory",
    "FailurePattern",
    "ImprovementRecommendation",
    "TradeAnalysis",
    "get_failure_analyzer",
    # Strategy Evolver
    "EvolutionConfig",
    "StrategyEvolver",
    "StrategyGene",
    "get_strategy_evolver",
    # Trade Forensics
    "AggregateStats",
    "EntryQuality",
    "ExitQuality",
    "ForensicsConfig",
    "ForensicsResult",
    "StopQuality",
    "TradeForensics",
    "get_trade_forensics",
    "reset_trade_forensics",
]
