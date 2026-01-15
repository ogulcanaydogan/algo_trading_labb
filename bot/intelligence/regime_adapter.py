"""
Regime Adapter - Market Regime Detection and Strategy Switching.

Detects market regime changes and adapts trading strategy:
- Bull/Bear/Sideways/Volatile regime detection
- Strategy parameter adjustment per regime
- Position sizing adaptation
- Stop loss/take profit adjustment
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    VOLATILE = "volatile"
    CRASH = "crash"
    UNKNOWN = "unknown"


@dataclass
class RegimeStrategy:
    """Strategy parameters for a specific regime."""
    regime: MarketRegime
    position_size_multiplier: float = 1.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    confidence_threshold: float = 0.6
    max_positions: int = 5
    signal_type: str = "momentum"  # momentum, mean_reversion, trend_following
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "position_size_multiplier": self.position_size_multiplier,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "confidence_threshold": self.confidence_threshold,
            "max_positions": self.max_positions,
            "signal_type": self.signal_type,
            "description": self.description,
        }


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: MarketRegime
    confidence: float
    previous_regime: MarketRegime
    regime_duration_bars: int = 0
    volatility: float = 0.0
    trend_strength: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "previous_regime": self.previous_regime.value,
            "regime_duration_bars": self.regime_duration_bars,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "detected_at": self.detected_at.isoformat(),
        }


class RegimeAdapter:
    """
    Detects market regimes and adapts trading strategy.

    Uses multiple indicators for regime detection:
    - Trend (moving average relationship)
    - Volatility (ATR-based)
    - Momentum (RSI, MACD)

    Adapts strategy parameters based on detected regime.
    """

    # Default strategy parameters per regime
    DEFAULT_STRATEGIES = {
        MarketRegime.STRONG_BULL: RegimeStrategy(
            regime=MarketRegime.STRONG_BULL,
            position_size_multiplier=1.5,
            stop_loss_pct=3.0,
            take_profit_pct=6.0,
            confidence_threshold=0.5,
            max_positions=6,
            signal_type="momentum",
            description="Aggressive momentum following in strong uptrend",
        ),
        MarketRegime.BULL: RegimeStrategy(
            regime=MarketRegime.BULL,
            position_size_multiplier=1.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence_threshold=0.55,
            max_positions=5,
            signal_type="trend_following",
            description="Standard trend following in uptrend",
        ),
        MarketRegime.SIDEWAYS: RegimeStrategy(
            regime=MarketRegime.SIDEWAYS,
            position_size_multiplier=0.75,
            stop_loss_pct=1.5,
            take_profit_pct=2.5,
            confidence_threshold=0.65,
            max_positions=4,
            signal_type="mean_reversion",
            description="Range trading with quick profits",
        ),
        MarketRegime.BEAR: RegimeStrategy(
            regime=MarketRegime.BEAR,
            position_size_multiplier=0.5,
            stop_loss_pct=1.5,
            take_profit_pct=3.0,
            confidence_threshold=0.65,
            max_positions=3,
            signal_type="counter_trend",
            description="Reduced exposure, counter-trend entries",
        ),
        MarketRegime.STRONG_BEAR: RegimeStrategy(
            regime=MarketRegime.STRONG_BEAR,
            position_size_multiplier=0.25,
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            confidence_threshold=0.75,
            max_positions=2,
            signal_type="defensive",
            description="Minimal exposure, defensive positioning",
        ),
        MarketRegime.VOLATILE: RegimeStrategy(
            regime=MarketRegime.VOLATILE,
            position_size_multiplier=0.5,
            stop_loss_pct=2.5,
            take_profit_pct=5.0,
            confidence_threshold=0.7,
            max_positions=3,
            signal_type="volatility_breakout",
            description="Reduced size, wider stops for volatility",
        ),
        MarketRegime.CRASH: RegimeStrategy(
            regime=MarketRegime.CRASH,
            position_size_multiplier=0.25,
            stop_loss_pct=1.0,
            take_profit_pct=2.0,
            confidence_threshold=0.8,
            max_positions=2,
            signal_type="defensive",
            description="Emergency mode - minimal exposure",
        ),
        MarketRegime.UNKNOWN: RegimeStrategy(
            regime=MarketRegime.UNKNOWN,
            position_size_multiplier=0.5,
            stop_loss_pct=2.0,
            take_profit_pct=3.0,
            confidence_threshold=0.7,
            max_positions=3,
            signal_type="neutral",
            description="Unknown regime - conservative approach",
        ),
    }

    def __init__(
        self,
        lookback_period: int = 50,
        volatility_threshold: float = 0.03,
        trend_threshold: float = 0.02,
    ):
        """
        Initialize the Regime Adapter.

        Args:
            lookback_period: Bars to analyze for regime detection
            volatility_threshold: Threshold for high volatility classification
            trend_threshold: Threshold for trend detection
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold

        # Current state
        self._current_state: Optional[RegimeState] = None
        self._regime_history: List[RegimeState] = []
        self._max_history = 100

        # Custom strategies (can override defaults)
        self._strategies = dict(self.DEFAULT_STRATEGIES)

        logger.info(f"Regime Adapter initialized: lookback={lookback_period}")

    def detect_regime(self, prices: np.ndarray) -> RegimeState:
        """
        Detect the current market regime from price data.

        Args:
            prices: Array of closing prices (most recent last)

        Returns:
            RegimeState with detected regime and confidence
        """
        if len(prices) < self.lookback_period:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                previous_regime=MarketRegime.UNKNOWN,
            )

        # Get recent prices
        recent = prices[-self.lookback_period:]

        # Calculate indicators
        returns = np.diff(recent) / recent[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        trend = (recent[-1] - recent[0]) / recent[0]  # Period return

        # Calculate momentum
        momentum = self._calculate_momentum(recent)

        # Determine regime
        regime, confidence = self._classify_regime(
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            returns=returns,
        )

        # Calculate regime duration
        duration = 0
        if self._current_state and self._current_state.regime == regime:
            duration = self._current_state.regime_duration_bars + 1

        # Build state
        previous = self._current_state.regime if self._current_state else MarketRegime.UNKNOWN

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            previous_regime=previous,
            regime_duration_bars=duration,
            volatility=volatility,
            trend_strength=abs(trend),
        )

        # Track history
        self._update_history(state)

        return state

    def get_strategy(self, regime: Optional[MarketRegime] = None) -> RegimeStrategy:
        """
        Get strategy parameters for a regime.

        Args:
            regime: Market regime (uses current if None)

        Returns:
            RegimeStrategy with parameters
        """
        if regime is None:
            regime = self._current_state.regime if self._current_state else MarketRegime.UNKNOWN

        return self._strategies.get(regime, self._strategies[MarketRegime.UNKNOWN])

    def adapt_parameters(
        self,
        base_position_size: float,
        base_stop_loss: float,
        base_take_profit: float,
        base_confidence_threshold: float,
    ) -> Dict[str, float]:
        """
        Adapt trading parameters based on current regime.

        Args:
            base_position_size: Base position size
            base_stop_loss: Base stop loss percentage
            base_take_profit: Base take profit percentage
            base_confidence_threshold: Base confidence threshold

        Returns:
            Adapted parameters dictionary
        """
        strategy = self.get_strategy()

        return {
            "position_size": base_position_size * strategy.position_size_multiplier,
            "stop_loss_pct": strategy.stop_loss_pct,
            "take_profit_pct": strategy.take_profit_pct,
            "confidence_threshold": max(base_confidence_threshold, strategy.confidence_threshold),
            "max_positions": strategy.max_positions,
            "signal_type": strategy.signal_type,
            "regime": strategy.regime.value,
            "regime_description": strategy.description,
        }

    def is_regime_change(self) -> Tuple[bool, Optional[str]]:
        """
        Check if there's been a recent regime change.

        Returns:
            Tuple of (changed, message)
        """
        if not self._current_state:
            return False, None

        if len(self._regime_history) < 2:
            return False, None

        current = self._regime_history[-1]
        previous = self._regime_history[-2]

        if current.regime != previous.regime:
            message = f"Regime changed: {previous.regime.value} -> {current.regime.value}"
            return True, message

        return False, None

    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """
        Check if exposure should be reduced based on regime.

        Returns:
            Tuple of (should_reduce, reason)
        """
        if not self._current_state:
            return False, ""

        high_risk_regimes = [
            MarketRegime.STRONG_BEAR,
            MarketRegime.CRASH,
            MarketRegime.VOLATILE,
        ]

        if self._current_state.regime in high_risk_regimes:
            strategy = self.get_strategy()
            return True, f"High risk regime: {self._current_state.regime.value} - reducing to {strategy.position_size_multiplier}x"

        return False, ""

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum."""
        if len(prices) < 14:
            return 0.0

        # Simple momentum: recent return vs average return
        recent_return = (prices[-1] - prices[-5]) / prices[-5]
        avg_return = (prices[-1] - prices[0]) / prices[0] / len(prices) * 5

        return recent_return - avg_return

    def _classify_regime(
        self,
        trend: float,
        volatility: float,
        momentum: float,
        returns: np.ndarray,
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on indicators.

        Returns:
            Tuple of (regime, confidence)
        """
        # Check for crash (extreme negative return + high volatility)
        if trend < -0.15 and volatility > self.volatility_threshold * 2:
            return MarketRegime.CRASH, 0.9

        # Check for high volatility
        if volatility > self.volatility_threshold * 1.5:
            return MarketRegime.VOLATILE, 0.75

        # Check trend direction
        if trend > self.trend_threshold * 2:
            # Strong uptrend
            if momentum > 0.02:
                return MarketRegime.STRONG_BULL, 0.8
            return MarketRegime.BULL, 0.7

        elif trend < -self.trend_threshold * 2:
            # Strong downtrend
            if momentum < -0.02:
                return MarketRegime.STRONG_BEAR, 0.8
            return MarketRegime.BEAR, 0.7

        elif abs(trend) < self.trend_threshold:
            # Sideways
            return MarketRegime.SIDEWAYS, 0.7

        elif trend > 0:
            return MarketRegime.BULL, 0.6
        else:
            return MarketRegime.BEAR, 0.6

    def _update_history(self, state: RegimeState):
        """Update regime history and current state."""
        self._current_state = state
        self._regime_history.append(state)

        if len(self._regime_history) > self._max_history:
            self._regime_history = self._regime_history[-self._max_history:]

    def set_strategy(self, regime: MarketRegime, strategy: RegimeStrategy):
        """Set custom strategy for a regime."""
        self._strategies[regime] = strategy
        logger.info(f"Set custom strategy for {regime.value}: {strategy.description}")

    def get_current_state(self) -> Optional[RegimeState]:
        """Get current regime state."""
        return self._current_state

    def get_regime_distribution(self, n_bars: int = 50) -> Dict[str, float]:
        """Get distribution of regimes over recent history."""
        if len(self._regime_history) < n_bars:
            recent = self._regime_history
        else:
            recent = self._regime_history[-n_bars:]

        if not recent:
            return {}

        distribution = {}
        for state in recent:
            regime = state.regime.value
            distribution[regime] = distribution.get(regime, 0) + 1

        # Convert to percentages
        total = len(recent)
        return {k: v / total for k, v in distribution.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get adapter summary."""
        current = self._current_state

        return {
            "current_regime": current.regime.value if current else "unknown",
            "confidence": current.confidence if current else 0,
            "volatility": current.volatility if current else 0,
            "trend_strength": current.trend_strength if current else 0,
            "regime_duration_bars": current.regime_duration_bars if current else 0,
            "regime_distribution": self.get_regime_distribution(),
            "current_strategy": self.get_strategy().to_dict() if current else None,
        }
