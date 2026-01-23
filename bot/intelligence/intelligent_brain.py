"""
Intelligent Trading Brain - Central AI Orchestrator.

Coordinates all AI intelligence capabilities:
1. Real-time learning from trades
2. Regime detection and strategy switching
3. News reasoning and sentiment
4. Trade decision explanations

This is the main entry point for intelligent trading.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .llm_router import LLMRouter, LLMRequest, RequestPriority
from .pattern_memory import PatternMemory
from .real_time_learner import RealTimeLearner, TradeOutcome, LearningResult
from .regime_adapter import RegimeAdapter, RegimeState, MarketRegime
from .news_reasoner import NewsReasoner, NewsContext
from .trade_explainer import TradeExplainer, TradeExplanation

logger = logging.getLogger(__name__)


@dataclass
class BrainConfig:
    """Configuration for the Intelligent Trading Brain."""

    # LLM settings
    daily_budget: float = 5.0
    enable_claude: bool = True
    enable_ollama: bool = True
    ollama_model: str = "llama3"

    # Learning settings
    learning_rate: float = 0.1
    min_patterns_for_learning: int = 5

    # Regime settings
    regime_lookback: int = 50
    volatility_threshold: float = 0.03

    # News settings
    news_cache_minutes: int = 5
    news_lookback_hours: int = 24

    # Telegram settings
    telegram_enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            "daily_budget": self.daily_budget,
            "enable_claude": self.enable_claude,
            "enable_ollama": self.enable_ollama,
            "ollama_model": self.ollama_model,
            "learning_rate": self.learning_rate,
            "min_patterns_for_learning": self.min_patterns_for_learning,
            "regime_lookback": self.regime_lookback,
            "volatility_threshold": self.volatility_threshold,
            "news_cache_minutes": self.news_cache_minutes,
            "news_lookback_hours": self.news_lookback_hours,
            "telegram_enabled": self.telegram_enabled,
        }


@dataclass
class EnrichedSignal:
    """A trading signal enriched with AI intelligence."""

    # Original signal
    symbol: str
    action: str
    original_confidence: float
    price: float

    # AI enrichment
    adjusted_confidence: float
    regime: str
    regime_confidence: float
    news_sentiment: float
    pattern_win_rate: float

    # Recommendations
    position_size_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float

    # Reasoning
    confidence_adjustments: List[str] = field(default_factory=list)
    should_trade: bool = True
    trade_reasoning: str = ""

    # Metadata
    enriched_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "original_confidence": self.original_confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "price": self.price,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "news_sentiment": self.news_sentiment,
            "pattern_win_rate": self.pattern_win_rate,
            "position_size_multiplier": self.position_size_multiplier,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "confidence_adjustments": self.confidence_adjustments,
            "should_trade": self.should_trade,
            "trade_reasoning": self.trade_reasoning,
            "enriched_at": self.enriched_at.isoformat(),
        }


class IntelligentTradingBrain:
    """
    Central orchestrator for AI-driven trading intelligence.

    Provides:
    1. Pre-trade analysis with confidence adjustment
    2. Post-trade learning and pattern storage
    3. Regime-based strategy adaptation
    4. News-informed decision making
    5. Human-readable trade explanations

    Usage:
        brain = IntelligentTradingBrain()

        # Before trading
        enriched = brain.enrich_signal(signal, prices)
        if enriched.should_trade:
            execute_trade(enriched)

        # After trade closes
        brain.learn_from_trade(outcome)
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        """
        Initialize the Intelligent Trading Brain.

        Args:
            config: Brain configuration
        """
        self.config = config or BrainConfig()

        # Initialize components
        self.llm_router = LLMRouter(
            daily_budget=self.config.daily_budget,
            enable_claude=self.config.enable_claude,
            enable_ollama=self.config.enable_ollama,
            default_ollama_model=self.config.ollama_model,
        )

        self.pattern_memory = PatternMemory()

        self.learner = RealTimeLearner(
            pattern_memory=self.pattern_memory,
            learning_rate=self.config.learning_rate,
        )

        self.regime_adapter = RegimeAdapter(
            lookback_period=self.config.regime_lookback,
            volatility_threshold=self.config.volatility_threshold,
        )

        self.news_reasoner = NewsReasoner(
            llm_router=self.llm_router,
            cache_duration_minutes=self.config.news_cache_minutes,
        )

        self.trade_explainer = TradeExplainer(
            llm_router=self.llm_router,
            telegram_enabled=self.config.telegram_enabled,
        )

        # Track state
        self._is_initialized = True
        self._last_regime_check: Optional[datetime] = None

        logger.info("Intelligent Trading Brain initialized")

    def enrich_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        prices: np.ndarray,
        portfolio_context: Optional[Dict] = None,
    ) -> EnrichedSignal:
        """
        Enrich a trading signal with AI intelligence.

        This is called before every trade decision.

        Args:
            symbol: Trading symbol
            action: Proposed action (BUY, SELL)
            confidence: Original signal confidence
            price: Current price
            prices: Array of recent prices for regime detection
            portfolio_context: Portfolio information

        Returns:
            EnrichedSignal with AI adjustments
        """
        portfolio_context = portfolio_context or {}
        adjustments = []

        # 1. Detect regime
        regime_state = self.regime_adapter.detect_regime(prices)
        regime_strategy = self.regime_adapter.get_strategy(regime_state.regime)

        # 2. Get pattern-based confidence adjustment
        pattern_adjustment, pattern_reason = self.learner.get_confidence_adjustment(
            symbol=symbol,
            regime=regime_state.regime.value,
            action=action,
        )
        adjustments.append(f"Pattern history: {pattern_reason}")

        # 3. Get news-based confidence adjustment
        news_context = self.news_reasoner.get_news_context(
            symbols=[symbol],
            hours_lookback=self.config.news_lookback_hours,
        )
        news_adjustment = 1.0 + (news_context.overall_sentiment * 0.15)
        news_adjustment = max(0.85, min(1.15, news_adjustment))
        adjustments.append(
            f"News sentiment: {news_context.sentiment_label} ({news_context.overall_sentiment:+.2f})"
        )

        # 4. Apply regime-based threshold
        regime_threshold = regime_strategy.confidence_threshold
        adjustments.append(
            f"Regime ({regime_state.regime.value}): threshold={regime_threshold:.0%}"
        )

        # 5. Calculate final confidence
        adjusted_confidence = confidence * pattern_adjustment * news_adjustment
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # 6. Determine if we should trade
        should_trade = adjusted_confidence >= regime_threshold

        # Build reasoning
        if should_trade:
            trade_reasoning = (
                f"Signal approved: {adjusted_confidence:.0%} >= {regime_threshold:.0%} threshold"
            )
        else:
            trade_reasoning = (
                f"Signal rejected: {adjusted_confidence:.0%} < {regime_threshold:.0%} threshold"
            )

        # Check for regime-based reduction
        reduce_exposure, reduce_reason = self.regime_adapter.should_reduce_exposure()
        if reduce_exposure:
            adjustments.append(reduce_reason)

        # Get pattern stats for win rate
        pattern_insights = self.learner.get_pattern_insights(
            symbol=symbol,
            regime=regime_state.regime.value,
            action=action,
        )

        return EnrichedSignal(
            symbol=symbol,
            action=action,
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            price=price,
            regime=regime_state.regime.value,
            regime_confidence=regime_state.confidence,
            news_sentiment=news_context.overall_sentiment,
            pattern_win_rate=pattern_insights.get("win_rate", 0.5),
            position_size_multiplier=regime_strategy.position_size_multiplier,
            stop_loss_pct=regime_strategy.stop_loss_pct,
            take_profit_pct=regime_strategy.take_profit_pct,
            confidence_adjustments=adjustments,
            should_trade=should_trade,
            trade_reasoning=trade_reasoning,
        )

    def learn_from_trade(self, outcome: TradeOutcome) -> LearningResult:
        """
        Learn from a completed trade.

        This is called immediately after every trade exit.

        Args:
            outcome: Trade outcome details

        Returns:
            LearningResult with insights
        """
        return self.learner.learn_from_trade(outcome)

    def explain_entry(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        signal: Dict[str, Any],
        enriched: Optional[EnrichedSignal] = None,
        portfolio_context: Optional[Dict] = None,
    ) -> TradeExplanation:
        """
        Generate explanation for a trade entry.

        Args:
            symbol: Trading symbol
            action: BUY or SELL
            price: Entry price
            quantity: Position quantity
            signal: Original signal info
            enriched: Enriched signal (if available)
            portfolio_context: Portfolio info

        Returns:
            TradeExplanation with formatted output
        """
        # Get pattern stats
        pattern_stats = None
        if enriched:
            pattern_stats = self.learner.get_pattern_insights(
                symbol=symbol,
                regime=enriched.regime,
                action=action,
            )

        # Get news summary
        news_context = self.news_reasoner.get_news_context([symbol], hours_lookback=4)

        return self.trade_explainer.explain_entry(
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            signal=signal,
            regime=enriched.regime if enriched else "unknown",
            news_summary=news_context.summary,
            portfolio_context=portfolio_context,
            pattern_stats=pattern_stats,
        )

    def explain_exit(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        hold_duration_minutes: int,
    ) -> TradeExplanation:
        """
        Generate explanation for a trade exit.

        Args:
            symbol: Trading symbol
            action: SELL (close long) or BUY (close short)
            entry_price: Original entry price
            exit_price: Exit price
            quantity: Position quantity
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            exit_reason: Why the exit occurred
            hold_duration_minutes: How long the position was held

        Returns:
            TradeExplanation
        """
        regime = self.regime_adapter.get_current_state()

        return self.trade_explainer.explain_exit(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            hold_duration_minutes=hold_duration_minutes,
            regime=regime.regime.value if regime else "unknown",
        )

    def get_regime_state(self, prices: np.ndarray) -> RegimeState:
        """Get current regime state."""
        return self.regime_adapter.detect_regime(prices)

    def get_news_context(self, symbols: List[str]) -> NewsContext:
        """Get news context for symbols."""
        return self.news_reasoner.get_news_context(symbols)

    def get_pattern_insights(
        self,
        symbol: str,
        regime: str,
        action: str,
    ) -> Dict[str, Any]:
        """Get pattern insights for a potential trade."""
        return self.learner.get_pattern_insights(symbol, regime, action)

    def adapt_parameters(
        self,
        base_position_size: float,
        base_stop_loss: float = 2.0,
        base_take_profit: float = 4.0,
        base_confidence_threshold: float = 0.6,
    ) -> Dict[str, float]:
        """Get regime-adapted trading parameters."""
        return self.regime_adapter.adapt_parameters(
            base_position_size=base_position_size,
            base_stop_loss=base_stop_loss,
            base_take_profit=base_take_profit,
            base_confidence_threshold=base_confidence_threshold,
        )

    def health_check(self) -> Dict[str, Any]:
        """Check health of all brain components."""
        return {
            "status": "healthy",
            "llm_router": self.llm_router.health_check(),
            "pattern_memory": self.pattern_memory.get_summary(),
            "learner": self.learner.get_summary(),
            "regime_adapter": self.regime_adapter.get_summary(),
            "news_reasoner": self.news_reasoner.get_summary(),
            "trade_explainer": self.trade_explainer.get_stats(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive brain summary."""
        return {
            "config": self.config.to_dict(),
            "health": self.health_check(),
            "recent_performance": self.learner.get_recent_performance(),
            "regime_distribution": self.regime_adapter.get_regime_distribution(),
        }

    def log_status(self):
        """Log current brain status."""
        summary = self.get_summary()
        regime = self.regime_adapter.get_current_state()

        logger.info("=== Intelligent Trading Brain Status ===")
        logger.info(f"Current Regime: {regime.regime.value if regime else 'unknown'}")
        logger.info(
            f"Pattern Memory: {summary['health']['pattern_memory']['total_patterns']} patterns"
        )
        logger.info(f"Recent Win Rate: {summary['recent_performance'].get('win_rate', 0):.1%}")
        logger.info(
            f"LLM Router: Claude={summary['health']['llm_router']['claude_available']}, Ollama={summary['health']['llm_router']['ollama_available']}"
        )
        logger.info("=" * 40)


# Convenience function for getting a brain instance
_global_brain: Optional[IntelligentTradingBrain] = None


def get_intelligent_brain(config: Optional[BrainConfig] = None) -> IntelligentTradingBrain:
    """Get or create the global intelligent brain instance."""
    global _global_brain
    if _global_brain is None:
        _global_brain = IntelligentTradingBrain(config)
    return _global_brain
