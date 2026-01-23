"""
Regime-Aware Trading Module.

Components:
- RegimeDetector: Detects market regimes (Bull, Bear, Crash, Sideways, HighVol)
- RegimeRiskEngine: Enforces risk constraints based on regime
- RegimeStrategySelector: Selects appropriate strategies per regime
- RegimePositionManager: Manages position sizing based on regime
- RegimeTradingEngine: Main orchestrator for regime-based trading
"""

from .regime_detector import (
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeIndicators,
    RegimeState,
)
from .regime_position_manager import (
    MultiAssetPositionManager,
    PositionAction,
    PositionRecommendation,
    PositionSizingConfig,
    RegimePositionManager,
)
from .regime_risk_engine import (
    PortfolioState,
    RegimeLimits,
    RegimeRiskEngine,
    RiskCheckResult,
    RiskConfig,
    TradeRequest,
)
from .regime_strategies import (
    RegimeStrategy,
    RegimeStrategySelector,
    StrategySignal,
)
from .regime_trading_engine import (
    RegimeTradingEngine,
    SimplePaperAdapter,
    TradingConfig,
    TradingMode,
    TradingState,
    TradeRecord,
)
from .strategy_tracker import (
    StrategyTracker,
    StrategyPerformance,
    StrategyStatus,
    TradeEntry,
    get_tracker,
    register_new_strategy,
    list_strategy_types,
    get_best_strategies,
    get_strategy_summary,
)
from .multi_timeframe_detector import (
    MultiTimeframeDetector,
    MultiTimeframeConfig,
    MultiTimeframeState,
    create_multi_tf_detector,
)
from .dynamic_risk_manager import (
    DynamicRiskManager,
    DynamicRiskConfig,
    RiskAction,
    RiskState,
    StopLossResult,
)
from .performance_tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    get_performance_tracker,
)
from .daily_reporter import (
    DailyReporter,
    ReporterConfig,
    get_daily_reporter,
    setup_daily_reporter,
)
from .strategy_selector import (
    StrategyType,
    StrategyConfig as AdvancedStrategyConfig,
    StrategyPerformance as AdvancedStrategyPerformance,
    SelectionResult,
    SelectorConfig,
    RegimeStrategySelector as AdvancedRegimeStrategySelector,
    create_default_strategies,
    create_regime_strategy_selector,
)

__all__ = [
    # Detector
    "MarketRegime",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeIndicators",
    "RegimeState",
    # Risk Engine
    "PortfolioState",
    "RegimeLimits",
    "RegimeRiskEngine",
    "RiskCheckResult",
    "RiskConfig",
    "TradeRequest",
    # Strategies
    "RegimeStrategy",
    "RegimeStrategySelector",
    "StrategySignal",
    # Position Manager
    "MultiAssetPositionManager",
    "PositionAction",
    "PositionRecommendation",
    "PositionSizingConfig",
    "RegimePositionManager",
    # Trading Engine
    "RegimeTradingEngine",
    "SimplePaperAdapter",
    "TradingConfig",
    "TradingMode",
    "TradingState",
    "TradeRecord",
    # Strategy Tracker
    "StrategyTracker",
    "StrategyPerformance",
    "StrategyStatus",
    "TradeEntry",
    "get_tracker",
    "register_new_strategy",
    "list_strategy_types",
    "get_best_strategies",
    "get_strategy_summary",
    # Multi-timeframe
    "MultiTimeframeDetector",
    "MultiTimeframeConfig",
    "MultiTimeframeState",
    "create_multi_tf_detector",
    # Dynamic Risk Manager
    "DynamicRiskManager",
    "DynamicRiskConfig",
    "RiskAction",
    "RiskState",
    "StopLossResult",
    # Performance Tracker
    "PerformanceTracker",
    "PerformanceMetrics",
    "get_performance_tracker",
    # Daily Reporter
    "DailyReporter",
    "ReporterConfig",
    "get_daily_reporter",
    "setup_daily_reporter",
    # Advanced Strategy Selector
    "StrategyType",
    "AdvancedStrategyConfig",
    "AdvancedStrategyPerformance",
    "SelectionResult",
    "SelectorConfig",
    "AdvancedRegimeStrategySelector",
    "create_default_strategies",
    "create_regime_strategy_selector",
]
