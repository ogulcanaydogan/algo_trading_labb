"""
Intelligent Trading Brain Module.

Provides AI-driven intelligence capabilities for smarter trading:
1. Real-time learning from trades
2. Regime detection and strategy switching
3. News reasoning and sentiment analysis
4. Trade decision explanations

Components:
- IntelligentTradingBrain: Central orchestrator
- LLMRouter: Hybrid Claude/Ollama routing
- RealTimeLearner: Continuous learning from trades
- RegimeAdapter: Market regime detection and adaptation
- NewsReasoner: News sentiment analysis
- TradeExplainer: Human-readable explanations
"""

from .llm_router import LLMRouter, LLMRequest, LLMResponse, RequestPriority
from .intelligent_brain import IntelligentTradingBrain, BrainConfig, get_intelligent_brain
from .trade_explainer import TradeExplainer, TradeExplanation
from .pattern_memory import PatternMemory, TradingPattern
from .real_time_learner import RealTimeLearner, LearningResult, TradeOutcome
from .regime_adapter import RegimeAdapter, RegimeStrategy
from .news_reasoner import NewsReasoner, NewsContext

__all__ = [
    # Core
    "IntelligentTradingBrain",
    "BrainConfig",
    "get_intelligent_brain",
    # LLM
    "LLMRouter",
    "LLMRequest",
    "LLMResponse",
    "RequestPriority",
    # Learning
    "RealTimeLearner",
    "LearningResult",
    "TradeOutcome",
    "PatternMemory",
    "TradingPattern",
    # Adaptation
    "RegimeAdapter",
    "RegimeStrategy",
    # News
    "NewsReasoner",
    "NewsContext",
    # Explanation
    "TradeExplainer",
    "TradeExplanation",
]
