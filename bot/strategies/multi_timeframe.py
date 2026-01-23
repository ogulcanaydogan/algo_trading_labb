"""
Multi-Timeframe Signal Fusion - Combine signals across timeframes.

Fuses signals from multiple timeframes (1h, 4h, 1d) for more
robust trading decisions with higher conviction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""

    STRONG_LONG = "strong_long"
    LONG = "long"
    WEAK_LONG = "weak_long"
    NEUTRAL = "neutral"
    WEAK_SHORT = "weak_short"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: str
    direction: Literal["long", "short", "neutral"]
    strength: float  # -1 to 1
    indicators: Dict[str, float]
    trend: Literal["up", "down", "sideways"]
    momentum: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "indicators": {k: round(v, 4) for k, v in self.indicators.items()},
            "trend": self.trend,
            "momentum": round(self.momentum, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FusedSignal:
    """Fused signal from multiple timeframes."""

    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    strength: SignalStrength
    timeframe_signals: Dict[str, TimeframeSignal]
    alignment_score: float  # How aligned are the timeframes
    primary_timeframe: str
    supporting_timeframes: List[str]
    conflicting_timeframes: List[str]
    entry_quality: str  # "excellent", "good", "fair", "poor"
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "strength": self.strength.value,
            "alignment_score": round(self.alignment_score, 4),
            "primary_timeframe": self.primary_timeframe,
            "supporting_timeframes": self.supporting_timeframes,
            "conflicting_timeframes": self.conflicting_timeframes,
            "entry_quality": self.entry_quality,
            "reasoning": self.reasoning,
            "timeframe_signals": {k: v.to_dict() for k, v in self.timeframe_signals.items()},
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MTFConfig:
    """Multi-timeframe fusion configuration."""

    # Timeframes to analyze
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])

    # Weights for each timeframe (higher = more important)
    timeframe_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "1h": 0.3,
            "4h": 0.4,
            "1d": 0.3,
        }
    )

    # Indicator settings
    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14

    # Signal thresholds
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    trend_strength_threshold: float = 0.02  # 2% from EMA

    # Alignment requirements
    min_alignment_score: float = 0.5  # Minimum alignment to trade
    require_higher_tf_confirmation: bool = True


class MultiTimeframeFusion:
    """
    Fuse signals from multiple timeframes for higher-conviction trades.

    Logic:
    1. Analyze each timeframe independently
    2. Calculate alignment between timeframes
    3. Weight signals by timeframe importance
    4. Generate fused signal with confidence score
    """

    def __init__(self, config: Optional[MTFConfig] = None):
        self.config = config or MTFConfig()
        self._signal_cache: Dict[str, Dict[str, TimeframeSignal]] = {}

    def analyze_timeframe(
        self,
        ohlcv: pd.DataFrame,
        timeframe: str,
    ) -> TimeframeSignal:
        """
        Analyze a single timeframe and generate signal.

        Args:
            ohlcv: OHLCV data for the timeframe
            timeframe: Timeframe string (1h, 4h, 1d)

        Returns:
            TimeframeSignal for this timeframe
        """
        if len(ohlcv) < self.config.ema_trend:
            return TimeframeSignal(
                timeframe=timeframe,
                direction="neutral",
                strength=0,
                indicators={},
                trend="sideways",
                momentum=0,
            )

        df = ohlcv.copy()

        # Calculate indicators
        ema_fast = EMAIndicator(df["close"], window=self.config.ema_fast).ema_indicator()
        ema_slow = EMAIndicator(df["close"], window=self.config.ema_slow).ema_indicator()
        ema_trend = EMAIndicator(df["close"], window=self.config.ema_trend).ema_indicator()
        rsi = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()
        atr = AverageTrueRange(
            df["high"], df["low"], df["close"], window=self.config.atr_period
        ).average_true_range()

        macd = MACD(df["close"])
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        macd_hist = macd.macd_diff()

        # Get latest values
        close = df["close"].iloc[-1]
        ema_f = ema_fast.iloc[-1]
        ema_s = ema_slow.iloc[-1]
        ema_t = ema_trend.iloc[-1]
        rsi_val = rsi.iloc[-1]
        atr_val = atr.iloc[-1]
        macd_val = macd_line.iloc[-1]
        macd_sig = macd_signal.iloc[-1]
        macd_h = macd_hist.iloc[-1]

        indicators = {
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "ema_trend": ema_t,
            "rsi": rsi_val,
            "atr": atr_val,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "macd_histogram": macd_h,
        }

        # Determine trend
        price_vs_trend = (close - ema_t) / ema_t
        if price_vs_trend > self.config.trend_strength_threshold:
            trend = "up"
        elif price_vs_trend < -self.config.trend_strength_threshold:
            trend = "down"
        else:
            trend = "sideways"

        # Calculate momentum
        momentum = (macd_h / close * 100) if close > 0 else 0

        # Calculate signal strength
        strength = self._calculate_strength(close, ema_f, ema_s, ema_t, rsi_val, macd_h, trend)

        # Determine direction
        if strength > 0.2:
            direction = "long"
        elif strength < -0.2:
            direction = "short"
        else:
            direction = "neutral"

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            indicators=indicators,
            trend=trend,
            momentum=momentum,
        )

    def _calculate_strength(
        self,
        close: float,
        ema_fast: float,
        ema_slow: float,
        ema_trend: float,
        rsi: float,
        macd_hist: float,
        trend: str,
    ) -> float:
        """Calculate signal strength from -1 to 1."""
        signals = []

        # EMA alignment
        if close > ema_fast > ema_slow > ema_trend:
            signals.append(1.0)  # Strong bullish alignment
        elif close > ema_fast > ema_slow:
            signals.append(0.5)
        elif close < ema_fast < ema_slow < ema_trend:
            signals.append(-1.0)  # Strong bearish alignment
        elif close < ema_fast < ema_slow:
            signals.append(-0.5)
        else:
            signals.append(0)

        # RSI signal
        if rsi < self.config.rsi_oversold:
            signals.append(0.5)  # Oversold = potential long
        elif rsi > self.config.rsi_overbought:
            signals.append(-0.5)  # Overbought = potential short
        else:
            # Normalize RSI to -0.3 to 0.3
            signals.append((50 - rsi) / 100)

        # MACD signal
        if macd_hist > 0:
            signals.append(min(macd_hist / close * 100, 0.5))
        else:
            signals.append(max(macd_hist / close * 100, -0.5))

        # Trend alignment
        if trend == "up":
            signals.append(0.3)
        elif trend == "down":
            signals.append(-0.3)

        # Combine signals
        return np.clip(np.mean(signals), -1, 1)

    def fuse_signals(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        symbol: Optional[str] = None,
    ) -> FusedSignal:
        """
        Fuse signals from multiple timeframes.

        Args:
            data_by_timeframe: Dict mapping timeframe to OHLCV DataFrame
            symbol: Symbol being analyzed (for logging)

        Returns:
            FusedSignal with combined analysis
        """
        timeframe_signals = {}
        reasoning = []

        # Analyze each timeframe
        for tf in self.config.timeframes:
            if tf not in data_by_timeframe:
                continue

            signal = self.analyze_timeframe(data_by_timeframe[tf], tf)
            timeframe_signals[tf] = signal

        if not timeframe_signals:
            return self._neutral_signal(timeframe_signals)

        # Cache signals
        if symbol:
            self._signal_cache[symbol] = timeframe_signals

        # Calculate weighted average strength
        weighted_strength = 0
        total_weight = 0
        for tf, signal in timeframe_signals.items():
            weight = self.config.timeframe_weights.get(tf, 0.33)
            weighted_strength += signal.strength * weight
            total_weight += weight

        if total_weight > 0:
            weighted_strength /= total_weight

        # Determine alignment
        directions = [s.direction for s in timeframe_signals.values()]
        alignment_score = self._calculate_alignment(directions)

        # Find supporting and conflicting timeframes
        primary_direction = (
            "long" if weighted_strength > 0 else "short" if weighted_strength < 0 else "neutral"
        )
        supporting = [tf for tf, s in timeframe_signals.items() if s.direction == primary_direction]
        conflicting = [
            tf
            for tf, s in timeframe_signals.items()
            if s.direction != primary_direction and s.direction != "neutral"
        ]

        # Generate reasoning
        for tf, signal in timeframe_signals.items():
            reasoning.append(
                f"{tf}: {signal.direction} (strength={signal.strength:.2f}, trend={signal.trend})"
            )

        # Check higher timeframe confirmation
        if self.config.require_higher_tf_confirmation:
            higher_tfs = self.config.timeframes[1:]  # Skip lowest timeframe
            higher_confirms = all(
                timeframe_signals.get(
                    tf, TimeframeSignal(tf, "neutral", 0, {}, "sideways", 0)
                ).direction
                == primary_direction
                or timeframe_signals.get(
                    tf, TimeframeSignal(tf, "neutral", 0, {}, "sideways", 0)
                ).direction
                == "neutral"
                for tf in higher_tfs
                if tf in timeframe_signals
            )
            if not higher_confirms and primary_direction != "neutral":
                reasoning.append("Warning: Higher timeframes not confirming signal")
                alignment_score *= 0.5

        # Determine final action
        action, confidence, strength = self._determine_action(
            weighted_strength, alignment_score, supporting, conflicting
        )

        # Rate entry quality
        if alignment_score > 0.8 and len(supporting) >= 2:
            entry_quality = "excellent"
        elif alignment_score > 0.6 and len(supporting) >= 1:
            entry_quality = "good"
        elif alignment_score > 0.4:
            entry_quality = "fair"
        else:
            entry_quality = "poor"

        return FusedSignal(
            action=action,
            confidence=confidence,
            strength=strength,
            timeframe_signals=timeframe_signals,
            alignment_score=alignment_score,
            primary_timeframe=self.config.timeframes[-1],  # Highest TF as primary
            supporting_timeframes=supporting,
            conflicting_timeframes=conflicting,
            entry_quality=entry_quality,
            reasoning=reasoning,
        )

    def _calculate_alignment(self, directions: List[str]) -> float:
        """Calculate alignment score between timeframe directions."""
        if not directions:
            return 0

        # Count direction agreement
        long_count = directions.count("long")
        short_count = directions.count("short")
        neutral_count = directions.count("neutral")
        total = len(directions)

        # Max alignment is all same direction
        max_count = max(long_count, short_count)

        # Neutrals partially count toward alignment
        alignment = (max_count + neutral_count * 0.5) / total

        return alignment

    def _determine_action(
        self,
        weighted_strength: float,
        alignment_score: float,
        supporting: List[str],
        conflicting: List[str],
    ) -> Tuple[str, float, SignalStrength]:
        """Determine final action, confidence, and strength."""
        # Check minimum alignment
        if alignment_score < self.config.min_alignment_score:
            return "FLAT", alignment_score, SignalStrength.NEUTRAL

        # Determine action
        if weighted_strength > 0.3 and len(supporting) >= 2:
            action = "LONG"
            if weighted_strength > 0.6:
                strength = SignalStrength.STRONG_LONG
            elif weighted_strength > 0.4:
                strength = SignalStrength.LONG
            else:
                strength = SignalStrength.WEAK_LONG
        elif weighted_strength < -0.3 and len(supporting) >= 2:
            action = "SHORT"
            if weighted_strength < -0.6:
                strength = SignalStrength.STRONG_SHORT
            elif weighted_strength < -0.4:
                strength = SignalStrength.SHORT
            else:
                strength = SignalStrength.WEAK_SHORT
        else:
            action = "FLAT"
            strength = SignalStrength.NEUTRAL

        # Calculate confidence
        confidence = abs(weighted_strength) * alignment_score

        # Reduce confidence if conflicting signals
        if conflicting:
            confidence *= 1 - 0.2 * len(conflicting)

        return action, max(0, min(1, confidence)), strength

    def _neutral_signal(self, timeframe_signals: Dict) -> FusedSignal:
        """Return neutral signal when insufficient data."""
        return FusedSignal(
            action="FLAT",
            confidence=0,
            strength=SignalStrength.NEUTRAL,
            timeframe_signals=timeframe_signals,
            alignment_score=0,
            primary_timeframe="",
            supporting_timeframes=[],
            conflicting_timeframes=[],
            entry_quality="poor",
            reasoning=["Insufficient data for analysis"],
        )

    def get_cached_signals(self, symbol: str) -> Optional[Dict[str, TimeframeSignal]]:
        """Get cached signals for a symbol."""
        return self._signal_cache.get(symbol)


def create_mtf_fusion(config: Optional[MTFConfig] = None) -> MultiTimeframeFusion:
    """Factory function to create multi-timeframe fusion."""
    return MultiTimeframeFusion(config=config)
