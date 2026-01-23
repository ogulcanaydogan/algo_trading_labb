"""
Multi-Timeframe Regime Detector.

Confirms regime by requiring agreement across multiple timeframes.
This reduces false signals and whipsaws.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .regime_detector import MarketRegime, RegimeConfig, RegimeDetector, RegimeState

logger = logging.getLogger(__name__)


class TimeframeWeight(Enum):
    """Weights for different timeframes."""

    SHORT = 0.2  # 15m, 1h - fast signals
    MEDIUM = 0.3  # 4h - primary
    LONG = 0.5  # 1d - trend confirmation


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe detection."""

    # Timeframes to analyze (in order: short to long)
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])

    # Weights for each timeframe (must sum to 1.0)
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "1h": 0.2,
            "4h": 0.3,
            "1d": 0.5,
        }
    )

    # Minimum agreement threshold (0-1)
    min_agreement: float = 0.6

    # Require longest timeframe to agree for trend regimes
    require_long_term_confirmation: bool = True

    # Cache duration for each timeframe (seconds)
    cache_duration: Dict[str, int] = field(
        default_factory=lambda: {
            "1h": 300,  # 5 minutes
            "4h": 900,  # 15 minutes
            "1d": 3600,  # 1 hour
        }
    )


@dataclass
class MultiTimeframeState:
    """State from multi-timeframe analysis."""

    regime: MarketRegime
    confidence: float
    agreement_score: float
    timeframe_regimes: Dict[str, MarketRegime]
    timeframe_confidences: Dict[str, float]
    dominant_timeframe: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_confirmed(self) -> bool:
        """Check if regime is confirmed across timeframes."""
        return self.agreement_score >= 0.6

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "timeframe_regimes": {k: v.value for k, v in self.timeframe_regimes.items()},
            "timeframe_confidences": self.timeframe_confidences,
            "dominant_timeframe": self.dominant_timeframe,
            "is_confirmed": self.is_confirmed,
            "timestamp": self.timestamp.isoformat(),
        }


class MultiTimeframeDetector:
    """
    Detects market regime using multiple timeframes for confirmation.

    Strategy:
    - Analyze regime on multiple timeframes (1h, 4h, 1d)
    - Weight longer timeframes more heavily
    - Require agreement threshold for regime confirmation
    - Use longest timeframe for trend direction confirmation
    """

    def __init__(
        self,
        config: Optional[MultiTimeframeConfig] = None,
        regime_config: Optional[RegimeConfig] = None,
    ):
        self.config = config or MultiTimeframeConfig()
        self.regime_config = regime_config or RegimeConfig()

        # Create detector for each timeframe
        self.detectors: Dict[str, RegimeDetector] = {
            tf: RegimeDetector(self.regime_config) for tf in self.config.timeframes
        }

        # Cache for timeframe results
        self._cache: Dict[str, Tuple[RegimeState, datetime]] = {}

        # Current state
        self._current_state: Optional[MultiTimeframeState] = None
        self._state_history: List[MultiTimeframeState] = []

    def detect(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str = "",
    ) -> MultiTimeframeState:
        """
        Detect regime using multiple timeframes.

        Args:
            data: Dict mapping timeframe -> OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            MultiTimeframeState with confirmed regime
        """
        timeframe_regimes: Dict[str, MarketRegime] = {}
        timeframe_confidences: Dict[str, float] = {}

        # Detect regime for each timeframe
        for tf in self.config.timeframes:
            if tf not in data or data[tf] is None or data[tf].empty:
                logger.warning(f"No data for timeframe {tf}")
                continue

            # Check cache
            cached = self._get_cached(tf)
            if cached:
                timeframe_regimes[tf] = cached.regime
                timeframe_confidences[tf] = cached.confidence
            else:
                # Detect and cache
                state = self.detectors[tf].detect(data[tf], symbol, tf)
                timeframe_regimes[tf] = state.regime
                timeframe_confidences[tf] = state.confidence
                self._cache[tf] = (state, datetime.now())

        if not timeframe_regimes:
            # No data available, return unknown
            return MultiTimeframeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                agreement_score=0.0,
                timeframe_regimes={},
                timeframe_confidences={},
                dominant_timeframe="",
            )

        # Calculate weighted regime scores
        regime_scores = self._calculate_regime_scores(timeframe_regimes, timeframe_confidences)

        # Determine final regime
        final_regime, final_confidence = self._determine_regime(
            regime_scores, timeframe_regimes, timeframe_confidences
        )

        # Calculate agreement score
        agreement_score = self._calculate_agreement(timeframe_regimes, final_regime)

        # Find dominant timeframe
        dominant_tf = max(
            timeframe_confidences.keys(),
            key=lambda tf: timeframe_confidences[tf] * self.config.weights.get(tf, 0.1),
        )

        # Create state
        state = MultiTimeframeState(
            regime=final_regime,
            confidence=final_confidence,
            agreement_score=agreement_score,
            timeframe_regimes=timeframe_regimes,
            timeframe_confidences=timeframe_confidences,
            dominant_timeframe=dominant_tf,
        )

        # Log if regime changed
        if self._current_state and self._current_state.regime != state.regime:
            logger.info(
                f"Multi-TF regime change: {self._current_state.regime.value} -> {state.regime.value} "
                f"(agreement: {agreement_score:.1%}, confirmed: {state.is_confirmed})"
            )

        self._current_state = state
        self._state_history.append(state)

        return state

    def _get_cached(self, timeframe: str) -> Optional[RegimeState]:
        """Get cached regime state if still valid."""
        if timeframe not in self._cache:
            return None

        state, timestamp = self._cache[timeframe]
        cache_duration = self.config.cache_duration.get(timeframe, 300)

        if (datetime.now() - timestamp).total_seconds() < cache_duration:
            return state

        return None

    def _calculate_regime_scores(
        self,
        timeframe_regimes: Dict[str, MarketRegime],
        timeframe_confidences: Dict[str, float],
    ) -> Dict[MarketRegime, float]:
        """Calculate weighted scores for each regime."""
        scores: Dict[MarketRegime, float] = {regime: 0.0 for regime in MarketRegime}

        for tf, regime in timeframe_regimes.items():
            weight = self.config.weights.get(tf, 0.1)
            confidence = timeframe_confidences.get(tf, 0.5)
            scores[regime] += weight * confidence

        return scores

    def _determine_regime(
        self,
        regime_scores: Dict[MarketRegime, float],
        timeframe_regimes: Dict[str, MarketRegime],
        timeframe_confidences: Dict[str, float],
    ) -> Tuple[MarketRegime, float]:
        """Determine final regime from scores."""

        # Get regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        best_score = regime_scores[best_regime]

        # Check long-term confirmation for trend regimes
        if self.config.require_long_term_confirmation:
            longest_tf = self.config.timeframes[-1]  # Last is longest
            if longest_tf in timeframe_regimes:
                long_regime = timeframe_regimes[longest_tf]

                # For BULL/BEAR, require long-term agreement
                if best_regime in (MarketRegime.BULL, MarketRegime.BEAR):
                    if long_regime != best_regime:
                        # Downgrade to SIDEWAYS if long-term doesn't confirm
                        if long_regime == MarketRegime.SIDEWAYS:
                            logger.debug(
                                f"Downgrading {best_regime.value} to SIDEWAYS "
                                f"(long-term shows {long_regime.value})"
                            )
                            best_regime = MarketRegime.SIDEWAYS
                            best_score *= 0.7  # Reduce confidence

        # CRASH always takes priority (safety first)
        if MarketRegime.CRASH in timeframe_regimes.values():
            crash_count = sum(1 for r in timeframe_regimes.values() if r == MarketRegime.CRASH)
            if crash_count >= 1:  # Any timeframe showing crash
                best_regime = MarketRegime.CRASH
                best_score = max(
                    timeframe_confidences.get(tf, 0)
                    for tf, r in timeframe_regimes.items()
                    if r == MarketRegime.CRASH
                )

        # Normalize confidence
        confidence = min(1.0, best_score / sum(self.config.weights.values()))

        return best_regime, confidence

    def _calculate_agreement(
        self,
        timeframe_regimes: Dict[str, MarketRegime],
        final_regime: MarketRegime,
    ) -> float:
        """Calculate agreement score (0-1) across timeframes."""
        if not timeframe_regimes:
            return 0.0

        agreeing_weight = 0.0
        total_weight = 0.0

        for tf, regime in timeframe_regimes.items():
            weight = self.config.weights.get(tf, 0.1)
            total_weight += weight

            if regime == final_regime:
                agreeing_weight += weight
            # Partial agreement for related regimes
            elif self._are_related_regimes(regime, final_regime):
                agreeing_weight += weight * 0.5

        return agreeing_weight / total_weight if total_weight > 0 else 0.0

    def _are_related_regimes(self, r1: MarketRegime, r2: MarketRegime) -> bool:
        """Check if two regimes are related (partial agreement)."""
        related_pairs = [
            (MarketRegime.BULL, MarketRegime.SIDEWAYS),
            (MarketRegime.BEAR, MarketRegime.SIDEWAYS),
            (MarketRegime.BEAR, MarketRegime.CRASH),
            (MarketRegime.HIGH_VOL, MarketRegime.CRASH),
            (MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS),
        ]
        return (r1, r2) in related_pairs or (r2, r1) in related_pairs

    def get_status(self) -> Dict:
        """Get current multi-timeframe status."""
        if not self._current_state:
            return {"status": "not_initialized"}

        return {
            "current_state": self._current_state.to_dict(),
            "detector_stats": {
                tf: detector.get_regime_stats() for tf, detector in self.detectors.items()
            },
            "cache_status": {
                tf: {
                    "cached": tf in self._cache,
                    "age_seconds": (datetime.now() - self._cache[tf][1]).total_seconds()
                    if tf in self._cache
                    else None,
                }
                for tf in self.config.timeframes
            },
        }

    def clear_cache(self):
        """Clear all cached states."""
        self._cache.clear()


# Convenience function
def create_multi_tf_detector(
    timeframes: List[str] = None,
    weights: Dict[str, float] = None,
) -> MultiTimeframeDetector:
    """Create a multi-timeframe detector with custom settings."""
    config = MultiTimeframeConfig()

    if timeframes:
        config.timeframes = timeframes
    if weights:
        config.weights = weights

    return MultiTimeframeDetector(config=config)
