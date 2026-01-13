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
]
