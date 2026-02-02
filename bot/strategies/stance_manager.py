"""
Market Stance Manager.

Determines overall market stance based on:
- News sentiment (30%)
- Fear & Greed Index (20%)
- Regime detection (30%)
- Technical indicators (20%)

Stances:
- BULLISH: Prefer longs, higher leverage
- BEARISH: Prefer shorts, hedge longs
- NEUTRAL: Reduce positions, wait
- CRISIS: Close all, cash position
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MarketStance(Enum):
    """Overall market stance."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CRISIS = "crisis"


@dataclass
class StanceDecision:
    """Stance decision with reasoning."""
    stance: MarketStance
    confidence: float  # 0-1
    reasoning: List[str]
    recommended_exposure: float  # 0-1, how much capital to deploy
    prefer_shorts: bool
    max_leverage: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StanceInputs:
    """Inputs for stance calculation."""
    news_sentiment: float = 0.0  # -1 to 1
    fear_greed: float = 50.0  # 0-100
    regime: str = "unknown"
    trend_strength: float = 0.0  # -1 to 1
    volatility: float = 0.02
    rsi: float = 50.0
    recent_returns_24h: float = 0.0


class StanceManager:
    """
    Manages overall market stance.

    Uses multiple inputs to determine whether to be
    bullish, bearish, neutral, or in crisis mode.
    """

    # Component weights
    WEIGHTS = {
        "news_sentiment": 0.30,
        "fear_greed": 0.20,
        "regime": 0.30,
        "technical": 0.20,
    }

    # Regime scores (-1 to 1)
    REGIME_SCORES = {
        "STRONG_BULL": 0.9,
        "BULL": 0.6,
        "RECOVERY": 0.3,
        "SIDEWAYS": 0.0,
        "LOW_VOL": 0.1,
        "HIGH_VOL": -0.2,
        "BEAR": -0.6,
        "STRONG_BEAR": -0.8,
        "CRASH": -1.0,
        "unknown": 0.0,
    }

    def __init__(
        self,
        crisis_threshold: float = -0.7,
        bearish_threshold: float = -0.2,
        bullish_threshold: float = 0.2,
    ):
        self.crisis_threshold = crisis_threshold
        self.bearish_threshold = bearish_threshold
        self.bullish_threshold = bullish_threshold

        # History
        self._stance_history: List[StanceDecision] = []
        self._current_stance: Optional[StanceDecision] = None

        logger.info("StanceManager initialized")

    def calculate_stance(self, inputs: StanceInputs) -> StanceDecision:
        """
        Calculate market stance from inputs.

        Args:
            inputs: StanceInputs with all factors

        Returns:
            StanceDecision with stance and reasoning
        """
        scores = {}
        reasoning = []

        # 1. News Sentiment Score (-1 to 1)
        scores["news_sentiment"] = inputs.news_sentiment
        if inputs.news_sentiment < -0.5:
            reasoning.append(f"Very negative news: {inputs.news_sentiment:.2f}")
        elif inputs.news_sentiment > 0.5:
            reasoning.append(f"Positive news: {inputs.news_sentiment:.2f}")

        # 2. Fear & Greed Score (convert 0-100 to -1 to 1)
        fg_score = (inputs.fear_greed - 50) / 50  # -1 to 1
        scores["fear_greed"] = fg_score
        if inputs.fear_greed < 25:
            reasoning.append(f"Extreme fear: {inputs.fear_greed:.0f}")
        elif inputs.fear_greed > 75:
            reasoning.append(f"Extreme greed: {inputs.fear_greed:.0f}")

        # 3. Regime Score
        scores["regime"] = self.REGIME_SCORES.get(inputs.regime, 0.0)
        reasoning.append(f"Regime: {inputs.regime}")

        # 4. Technical Score
        tech_score = 0.0

        # Trend contribution
        tech_score += inputs.trend_strength * 0.5

        # RSI contribution (overbought/oversold)
        if inputs.rsi > 70:
            tech_score -= 0.3
            reasoning.append(f"Overbought RSI: {inputs.rsi:.1f}")
        elif inputs.rsi < 30:
            tech_score += 0.3
            reasoning.append(f"Oversold RSI: {inputs.rsi:.1f}")

        # Volatility contribution
        if inputs.volatility > 0.05:
            tech_score -= 0.2
            reasoning.append(f"High volatility: {inputs.volatility*100:.1f}%")

        scores["technical"] = max(-1, min(1, tech_score))

        # Calculate weighted total
        total_score = sum(
            scores[key] * self.WEIGHTS[key]
            for key in scores
        )

        # Determine stance
        if total_score <= self.crisis_threshold:
            stance = MarketStance.CRISIS
            exposure = 0.0
            max_leverage = 1.0
        elif total_score <= self.bearish_threshold:
            stance = MarketStance.BEARISH
            exposure = 0.5
            max_leverage = 2.0
        elif total_score >= self.bullish_threshold:
            stance = MarketStance.BULLISH
            exposure = 0.8
            max_leverage = 3.0
        else:
            stance = MarketStance.NEUTRAL
            exposure = 0.3
            max_leverage = 1.5

        # Add score breakdown to reasoning
        reasoning.append(f"Score: {total_score:.2f}")

        decision = StanceDecision(
            stance=stance,
            confidence=abs(total_score),
            reasoning=reasoning,
            recommended_exposure=exposure,
            prefer_shorts=total_score < 0,
            max_leverage=max_leverage,
        )

        self._current_stance = decision
        self._stance_history.append(decision)

        # Keep history limited
        if len(self._stance_history) > 1000:
            self._stance_history = self._stance_history[-500:]

        logger.info(f"Stance: {stance.value} (score={total_score:.2f}, exposure={exposure:.0%})")

        return decision

    def get_current_stance(self) -> Optional[StanceDecision]:
        """Get current stance."""
        return self._current_stance

    def get_trading_parameters(self) -> Dict:
        """Get trading parameters based on current stance."""
        if not self._current_stance:
            return {
                "max_exposure": 0.5,
                "prefer_longs": True,
                "prefer_shorts": False,
                "max_leverage": 2.0,
                "position_size_multiplier": 1.0,
            }

        stance = self._current_stance

        params = {
            "max_exposure": stance.recommended_exposure,
            "prefer_longs": stance.stance == MarketStance.BULLISH,
            "prefer_shorts": stance.prefer_shorts,
            "max_leverage": stance.max_leverage,
        }

        # Position size multiplier
        if stance.stance == MarketStance.CRISIS:
            params["position_size_multiplier"] = 0.0  # No new positions
        elif stance.stance == MarketStance.BEARISH:
            params["position_size_multiplier"] = 0.5
        elif stance.stance == MarketStance.NEUTRAL:
            params["position_size_multiplier"] = 0.7
        else:  # BULLISH
            params["position_size_multiplier"] = 1.0 + (stance.confidence * 0.3)

        return params

    def should_close_all_positions(self) -> Tuple[bool, str]:
        """Check if all positions should be closed."""
        if not self._current_stance:
            return False, ""

        if self._current_stance.stance == MarketStance.CRISIS:
            return True, "CRISIS mode - close all positions"

        # Check for stance flip from bullish to crisis within short time
        if len(self._stance_history) >= 3:
            recent = self._stance_history[-3:]
            if (recent[0].stance == MarketStance.BULLISH and
                recent[-1].stance == MarketStance.BEARISH):
                return True, "Rapid stance deterioration"

        return False, ""

    def get_stance_history(self, hours: int = 24) -> List[StanceDecision]:
        """Get stance history for last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self._stance_history if s.timestamp > cutoff]

    def get_stance_summary(self) -> Dict:
        """Get summary of recent stance changes."""
        history = self.get_stance_history(24)

        if not history:
            return {"current": None, "changes_24h": 0}

        stance_counts = {}
        for s in history:
            stance_counts[s.stance.value] = stance_counts.get(s.stance.value, 0) + 1

        # Count changes
        changes = 0
        for i in range(1, len(history)):
            if history[i].stance != history[i-1].stance:
                changes += 1

        return {
            "current": self._current_stance.stance.value if self._current_stance else None,
            "confidence": self._current_stance.confidence if self._current_stance else 0,
            "exposure": self._current_stance.recommended_exposure if self._current_stance else 0,
            "changes_24h": changes,
            "stance_distribution": stance_counts,
        }


# Singleton
_stance_manager: Optional[StanceManager] = None


def get_stance_manager() -> StanceManager:
    """Get or create the StanceManager singleton."""
    global _stance_manager
    if _stance_manager is None:
        _stance_manager = StanceManager()
    return _stance_manager
