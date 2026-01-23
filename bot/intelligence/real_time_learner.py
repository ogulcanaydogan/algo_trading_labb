"""
Real-Time Learner - Continuous Learning from Trades.

Learns from every trade immediately after execution:
- Stores patterns in memory
- Updates confidence thresholds
- Adapts to changing market conditions
- Provides learning insights
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .pattern_memory import PatternMemory, TradingPattern, PatternStats

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """Result of learning from a trade."""

    pattern_id: int
    was_profitable: bool
    pnl_pct: float

    # What was learned
    confidence_adjustment: float = 1.0
    new_win_rate: float = 0.0
    patterns_in_memory: int = 0

    # Insights
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    learned_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "was_profitable": self.was_profitable,
            "pnl_pct": self.pnl_pct,
            "confidence_adjustment": self.confidence_adjustment,
            "new_win_rate": self.new_win_rate,
            "patterns_in_memory": self.patterns_in_memory,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "learned_at": self.learned_at.isoformat(),
        }


@dataclass
class TradeOutcome:
    """Outcome of a completed trade for learning."""

    symbol: str
    action: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    hold_duration_minutes: int

    # Market state at entry
    regime: str = "unknown"
    confidence_at_entry: float = 0.0
    rsi: float = 0.0
    macd: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0

    # Outcome details
    exit_reason: str = ""
    max_drawdown_pct: float = 0.0
    max_profit_pct: float = 0.0

    # Timestamps
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None


class RealTimeLearner:
    """
    Learns continuously from trade outcomes.

    Features:
    - Pattern storage and retrieval
    - Confidence threshold adaptation
    - Win rate tracking per regime/symbol
    - Learning insights generation
    """

    # Minimum patterns required for confident adjustments
    MIN_PATTERNS_FOR_LEARNING = 5

    # Max confidence adjustment per trade
    MAX_CONFIDENCE_ADJUSTMENT = 0.05

    def __init__(
        self,
        pattern_memory: Optional[PatternMemory] = None,
        learning_rate: float = 0.1,
    ):
        """
        Initialize the Real-Time Learner.

        Args:
            pattern_memory: Pattern memory storage
            learning_rate: How quickly to adapt (0-1)
        """
        self.pattern_memory = pattern_memory or PatternMemory()
        self.learning_rate = learning_rate

        # Track confidence adjustments
        self._confidence_adjustments: Dict[str, float] = {}

        # Track recent performance
        self._recent_trades: List[TradeOutcome] = []
        self._max_recent = 100

        logger.info(f"Real-Time Learner initialized: learning_rate={learning_rate}")

    def learn_from_trade(self, outcome: TradeOutcome) -> LearningResult:
        """
        Learn from a completed trade.

        This is called immediately after every trade exit.

        Args:
            outcome: The trade outcome to learn from

        Returns:
            LearningResult with insights and adjustments
        """
        logger.info(
            f"Learning from trade: {outcome.symbol} {outcome.action} {outcome.pnl_pct:+.2f}%"
        )

        # 1. Create and store pattern
        pattern = self._create_pattern(outcome)
        pattern_id = self.pattern_memory.store_pattern(pattern)

        # 2. Track in recent trades
        self._recent_trades.append(outcome)
        if len(self._recent_trades) > self._max_recent:
            self._recent_trades = self._recent_trades[-self._max_recent :]

        # 3. Get updated statistics
        stats = self.pattern_memory.get_pattern_stats(
            symbol=outcome.symbol,
            regime=outcome.regime,
            action=outcome.action,
            days_lookback=30,
        )

        # 4. Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(
            outcome=outcome,
            stats=stats,
        )

        # 5. Update stored adjustments
        adjustment_key = f"{outcome.symbol}:{outcome.regime}:{outcome.action}"
        self._confidence_adjustments[adjustment_key] = confidence_adjustment

        # 6. Generate insights
        insights = self._generate_insights(outcome, stats)
        recommendations = self._generate_recommendations(outcome, stats)

        # 7. Build result
        result = LearningResult(
            pattern_id=pattern_id,
            was_profitable=outcome.pnl > 0,
            pnl_pct=outcome.pnl_pct,
            confidence_adjustment=confidence_adjustment,
            new_win_rate=stats.win_rate,
            patterns_in_memory=stats.total_patterns,
            insights=insights,
            recommendations=recommendations,
        )

        logger.info(
            f"Learned pattern #{pattern_id}: win_rate={stats.win_rate:.1%}, adjustment={confidence_adjustment:.3f}"
        )

        return result

    def get_confidence_adjustment(
        self,
        symbol: str,
        regime: str,
        action: str,
    ) -> Tuple[float, str]:
        """
        Get the current confidence adjustment for a signal.

        Args:
            symbol: Trading symbol
            regime: Current market regime
            action: Proposed action (BUY, SELL)

        Returns:
            Tuple of (adjustment_factor, reasoning)
        """
        adjustment_key = f"{symbol}:{regime}:{action}"

        # Check cached adjustment
        if adjustment_key in self._confidence_adjustments:
            adjustment = self._confidence_adjustments[adjustment_key]
            stats = self.pattern_memory.get_pattern_stats(
                symbol=symbol,
                regime=regime,
                action=action,
            )
            reasoning = f"Based on {stats.total_patterns} patterns: {stats.win_rate:.0%} win rate"
            return adjustment, reasoning

        # Calculate from pattern memory
        return self.pattern_memory.get_confidence_adjustment(
            symbol=symbol,
            regime=regime,
            action=action,
            base_confidence=1.0,
        )

    def get_pattern_insights(
        self,
        symbol: str,
        regime: str,
        action: str,
    ) -> Dict[str, Any]:
        """
        Get insights about patterns for a potential trade.

        Args:
            symbol: Trading symbol
            regime: Current market regime
            action: Proposed action

        Returns:
            Insights dictionary
        """
        stats = self.pattern_memory.get_pattern_stats(
            symbol=symbol,
            regime=regime,
            action=action,
        )

        if stats.total_patterns < self.MIN_PATTERNS_FOR_LEARNING:
            return {
                "has_sufficient_data": False,
                "patterns_found": stats.total_patterns,
                "message": f"Need at least {self.MIN_PATTERNS_FOR_LEARNING} patterns for reliable insights",
            }

        return {
            "has_sufficient_data": True,
            "patterns_found": stats.total_patterns,
            "win_rate": stats.win_rate,
            "avg_pnl_pct": stats.avg_pnl_pct,
            "avg_hold_duration_minutes": stats.avg_hold_duration,
            "best_pnl_pct": stats.best_pnl_pct,
            "worst_pnl_pct": stats.worst_pnl_pct,
            "confidence_adjustment": self._confidence_adjustments.get(
                f"{symbol}:{regime}:{action}", 1.0
            ),
            "recommendation": self._get_pattern_recommendation(stats),
        }

    def _create_pattern(self, outcome: TradeOutcome) -> TradingPattern:
        """Create a TradingPattern from a TradeOutcome."""
        return TradingPattern(
            symbol=outcome.symbol,
            regime=outcome.regime,
            action=outcome.action,
            entry_price=outcome.entry_price,
            exit_price=outcome.exit_price,
            pnl_pct=outcome.pnl_pct,
            hold_duration_minutes=outcome.hold_duration_minutes,
            confidence_at_entry=outcome.confidence_at_entry,
            rsi=outcome.rsi,
            macd=outcome.macd,
            volatility=outcome.volatility,
            trend_strength=outcome.trend_strength,
            was_profitable=outcome.pnl > 0,
            max_drawdown_pct=outcome.max_drawdown_pct,
            max_profit_pct=outcome.max_profit_pct,
            timestamp=outcome.exit_time or datetime.now(),
        )

    def _calculate_confidence_adjustment(
        self,
        outcome: TradeOutcome,
        stats: PatternStats,
    ) -> float:
        """Calculate confidence adjustment based on performance."""
        if stats.total_patterns < self.MIN_PATTERNS_FOR_LEARNING:
            return 1.0  # No adjustment without enough data

        # Base adjustment on win rate deviation from 50%
        expected_win_rate = 0.5
        win_rate_deviation = stats.win_rate - expected_win_rate

        # Scale by learning rate
        adjustment = 1.0 + (win_rate_deviation * self.learning_rate * 2)

        # Clamp to reasonable range (0.7 to 1.3)
        return max(0.7, min(1.3, adjustment))

    def _generate_insights(
        self,
        outcome: TradeOutcome,
        stats: PatternStats,
    ) -> List[str]:
        """Generate insights from the trade outcome."""
        insights = []

        # Win/loss insight
        if outcome.pnl > 0:
            insights.append(f"Profitable trade: +{outcome.pnl_pct:.2f}%")
        else:
            insights.append(f"Losing trade: {outcome.pnl_pct:.2f}%")

        # Win rate insight
        if stats.total_patterns >= self.MIN_PATTERNS_FOR_LEARNING:
            if stats.win_rate > 0.6:
                insights.append(f"Strong pattern: {stats.win_rate:.0%} win rate")
            elif stats.win_rate < 0.4:
                insights.append(f"Weak pattern: {stats.win_rate:.0%} win rate")

        # Hold duration insight
        if stats.avg_hold_duration > 0:
            if outcome.hold_duration_minutes < stats.avg_hold_duration * 0.5:
                insights.append("Early exit compared to average")
            elif outcome.hold_duration_minutes > stats.avg_hold_duration * 2:
                insights.append("Extended hold compared to average")

        # Regime insight
        insights.append(f"Regime: {outcome.regime}")

        return insights

    def _generate_recommendations(
        self,
        outcome: TradeOutcome,
        stats: PatternStats,
    ) -> List[str]:
        """Generate recommendations based on the trade."""
        recommendations = []

        if stats.total_patterns < self.MIN_PATTERNS_FOR_LEARNING:
            recommendations.append("Accumulate more trades for reliable recommendations")
            return recommendations

        # Win rate based recommendations
        if stats.win_rate < 0.4:
            recommendations.append(
                f"Consider reducing position size for {outcome.action} in {outcome.regime} regime"
            )

        if stats.win_rate > 0.6:
            recommendations.append(f"Pattern performing well - maintain or increase confidence")

        # PnL based recommendations
        if stats.avg_pnl_pct < -1:
            recommendations.append("Average loss high - review stop loss levels")

        if stats.best_pnl_pct > 5 and stats.avg_pnl_pct < 1:
            recommendations.append("Large winners being offset - consider trailing stops")

        # Drawdown based
        if outcome.max_drawdown_pct > 3:
            recommendations.append("High drawdown observed - tighter stops may help")

        return recommendations

    def _get_pattern_recommendation(self, stats: PatternStats) -> str:
        """Get a single recommendation based on pattern stats."""
        if stats.win_rate > 0.6:
            return "FAVORABLE - Pattern has strong historical performance"
        elif stats.win_rate < 0.4:
            return "CAUTIOUS - Pattern has weak historical performance"
        else:
            return "NEUTRAL - Pattern has average historical performance"

    def get_recent_performance(self, n_trades: int = 10) -> Dict[str, Any]:
        """Get recent trading performance summary."""
        if not self._recent_trades:
            return {"trades": 0, "message": "No recent trades"}

        recent = self._recent_trades[-n_trades:]
        profitable = [t for t in recent if t.pnl > 0]

        return {
            "trades": len(recent),
            "profitable": len(profitable),
            "win_rate": len(profitable) / len(recent),
            "total_pnl_pct": sum(t.pnl_pct for t in recent),
            "avg_pnl_pct": sum(t.pnl_pct for t in recent) / len(recent),
            "best_trade": max(t.pnl_pct for t in recent),
            "worst_trade": min(t.pnl_pct for t in recent),
            "avg_hold_minutes": sum(t.hold_duration_minutes for t in recent) / len(recent),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get learner summary."""
        memory_summary = self.pattern_memory.get_summary()
        recent_performance = self.get_recent_performance()

        return {
            "pattern_memory": memory_summary,
            "recent_performance": recent_performance,
            "confidence_adjustments": len(self._confidence_adjustments),
            "learning_rate": self.learning_rate,
        }

    def update_all_adjustments(self):
        """Update all confidence adjustments based on pattern memory."""
        self.pattern_memory.update_confidence_adjustments()
        logger.info("Updated all confidence adjustments from pattern memory")
