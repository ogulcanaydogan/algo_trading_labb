"""
Advanced Technical Analysis Module

Provides:
- Confluence zone detection (S/R levels where multiple indicators align)
- Candlestick pattern recognition
- Divergence detection (RSI, MACD, Stochastic)
- Support/Resistance identification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator

logger = logging.getLogger(__name__)


# ============================================================================
# CANDLESTICK PATTERNS
# ============================================================================

class CandlePattern(Enum):
    """Recognized candlestick patterns"""
    # Bullish patterns
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    MORNING_STAR = "morning_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    BULLISH_HARAMI = "bullish_harami"
    PIERCING_LINE = "piercing_line"
    DRAGONFLY_DOJI = "dragonfly_doji"

    # Bearish patterns
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    BEARISH_ENGULFING = "bearish_engulfing"
    EVENING_STAR = "evening_star"
    THREE_BLACK_CROWS = "three_black_crows"
    BEARISH_HARAMI = "bearish_harami"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    GRAVESTONE_DOJI = "gravestone_doji"

    # Neutral patterns
    DOJI = "doji"
    SPINNING_TOP = "spinning_top"


@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern: CandlePattern
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    bar_index: int
    description: str


class CandlestickAnalyzer:
    """Detects candlestick patterns in OHLCV data"""

    def __init__(self, body_threshold: float = 0.3, doji_threshold: float = 0.1):
        self.body_threshold = body_threshold  # Min body size as % of range
        self.doji_threshold = doji_threshold  # Max body size for doji

    def analyze(self, ohlcv: pd.DataFrame, lookback: int = 5) -> List[PatternResult]:
        """
        Detect candlestick patterns in recent bars.

        Args:
            ohlcv: OHLCV DataFrame
            lookback: Number of recent bars to analyze

        Returns:
            List of detected patterns
        """
        patterns = []

        if len(ohlcv) < lookback + 3:
            return patterns

        # Analyze each bar in lookback window
        for i in range(-lookback, 0):
            idx = len(ohlcv) + i

            # Single candle patterns
            single_patterns = self._detect_single_candle(ohlcv, idx)
            patterns.extend(single_patterns)

            # Two candle patterns
            if idx >= 1:
                two_patterns = self._detect_two_candle(ohlcv, idx)
                patterns.extend(two_patterns)

            # Three candle patterns
            if idx >= 2:
                three_patterns = self._detect_three_candle(ohlcv, idx)
                patterns.extend(three_patterns)

        return patterns

    def _get_candle_metrics(self, ohlcv: pd.DataFrame, idx: int) -> Dict:
        """Calculate candle metrics"""
        row = ohlcv.iloc[idx]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        body = abs(c - o)
        range_hl = h - l if h > l else 0.0001
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_bullish = c > o

        return {
            "open": o, "high": h, "low": l, "close": c,
            "body": body,
            "range": range_hl,
            "body_pct": body / range_hl,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "upper_wick_pct": upper_wick / range_hl,
            "lower_wick_pct": lower_wick / range_hl,
            "is_bullish": is_bullish,
            "is_bearish": c < o,
            "is_doji": body / range_hl < self.doji_threshold,
        }

    def _detect_single_candle(self, ohlcv: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detect single candle patterns"""
        patterns = []
        m = self._get_candle_metrics(ohlcv, idx)

        # Doji
        if m["is_doji"]:
            if m["lower_wick_pct"] > 0.6:
                patterns.append(PatternResult(
                    pattern=CandlePattern.DRAGONFLY_DOJI,
                    direction="bullish",
                    strength=0.6,
                    bar_index=idx,
                    description="Dragonfly doji - potential bullish reversal"
                ))
            elif m["upper_wick_pct"] > 0.6:
                patterns.append(PatternResult(
                    pattern=CandlePattern.GRAVESTONE_DOJI,
                    direction="bearish",
                    strength=0.6,
                    bar_index=idx,
                    description="Gravestone doji - potential bearish reversal"
                ))
            else:
                patterns.append(PatternResult(
                    pattern=CandlePattern.DOJI,
                    direction="neutral",
                    strength=0.4,
                    bar_index=idx,
                    description="Doji - indecision"
                ))

        # Hammer (bullish) - small body at top, long lower wick
        elif (m["is_bullish"] and
              m["lower_wick_pct"] > 0.6 and
              m["upper_wick_pct"] < 0.1 and
              m["body_pct"] < 0.3):
            patterns.append(PatternResult(
                pattern=CandlePattern.HAMMER,
                direction="bullish",
                strength=0.7,
                bar_index=idx,
                description="Hammer - potential bullish reversal"
            ))

        # Inverted Hammer (bullish) - small body at bottom, long upper wick
        elif (m["is_bullish"] and
              m["upper_wick_pct"] > 0.6 and
              m["lower_wick_pct"] < 0.1 and
              m["body_pct"] < 0.3):
            patterns.append(PatternResult(
                pattern=CandlePattern.INVERTED_HAMMER,
                direction="bullish",
                strength=0.6,
                bar_index=idx,
                description="Inverted hammer - potential bullish reversal"
            ))

        # Shooting Star (bearish) - small body at bottom, long upper wick
        elif (m["is_bearish"] and
              m["upper_wick_pct"] > 0.6 and
              m["lower_wick_pct"] < 0.1 and
              m["body_pct"] < 0.3):
            patterns.append(PatternResult(
                pattern=CandlePattern.SHOOTING_STAR,
                direction="bearish",
                strength=0.7,
                bar_index=idx,
                description="Shooting star - potential bearish reversal"
            ))

        # Hanging Man (bearish) - small body at top, long lower wick (in uptrend)
        elif (m["is_bearish"] and
              m["lower_wick_pct"] > 0.6 and
              m["upper_wick_pct"] < 0.1 and
              m["body_pct"] < 0.3):
            patterns.append(PatternResult(
                pattern=CandlePattern.HANGING_MAN,
                direction="bearish",
                strength=0.6,
                bar_index=idx,
                description="Hanging man - potential bearish reversal"
            ))

        # Spinning Top
        elif (m["body_pct"] < 0.3 and
              m["upper_wick_pct"] > 0.25 and
              m["lower_wick_pct"] > 0.25):
            patterns.append(PatternResult(
                pattern=CandlePattern.SPINNING_TOP,
                direction="neutral",
                strength=0.3,
                bar_index=idx,
                description="Spinning top - indecision"
            ))

        return patterns

    def _detect_two_candle(self, ohlcv: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detect two candle patterns"""
        patterns = []

        m1 = self._get_candle_metrics(ohlcv, idx - 1)  # Previous candle
        m2 = self._get_candle_metrics(ohlcv, idx)      # Current candle

        # Bullish Engulfing
        if (m1["is_bearish"] and m2["is_bullish"] and
            m2["open"] < m1["close"] and m2["close"] > m1["open"] and
            m2["body"] > m1["body"] * 1.1):
            patterns.append(PatternResult(
                pattern=CandlePattern.BULLISH_ENGULFING,
                direction="bullish",
                strength=0.8,
                bar_index=idx,
                description="Bullish engulfing - strong bullish reversal signal"
            ))

        # Bearish Engulfing
        elif (m1["is_bullish"] and m2["is_bearish"] and
              m2["open"] > m1["close"] and m2["close"] < m1["open"] and
              m2["body"] > m1["body"] * 1.1):
            patterns.append(PatternResult(
                pattern=CandlePattern.BEARISH_ENGULFING,
                direction="bearish",
                strength=0.8,
                bar_index=idx,
                description="Bearish engulfing - strong bearish reversal signal"
            ))

        # Bullish Harami
        elif (m1["is_bearish"] and m2["is_bullish"] and
              m2["open"] > m1["close"] and m2["close"] < m1["open"] and
              m2["body"] < m1["body"] * 0.5):
            patterns.append(PatternResult(
                pattern=CandlePattern.BULLISH_HARAMI,
                direction="bullish",
                strength=0.6,
                bar_index=idx,
                description="Bullish harami - potential bullish reversal"
            ))

        # Bearish Harami
        elif (m1["is_bullish"] and m2["is_bearish"] and
              m2["open"] < m1["close"] and m2["close"] > m1["open"] and
              m2["body"] < m1["body"] * 0.5):
            patterns.append(PatternResult(
                pattern=CandlePattern.BEARISH_HARAMI,
                direction="bearish",
                strength=0.6,
                bar_index=idx,
                description="Bearish harami - potential bearish reversal"
            ))

        # Piercing Line
        elif (m1["is_bearish"] and m2["is_bullish"] and
              m2["open"] < m1["low"] and
              m2["close"] > (m1["open"] + m1["close"]) / 2):
            patterns.append(PatternResult(
                pattern=CandlePattern.PIERCING_LINE,
                direction="bullish",
                strength=0.7,
                bar_index=idx,
                description="Piercing line - bullish reversal signal"
            ))

        # Dark Cloud Cover
        elif (m1["is_bullish"] and m2["is_bearish"] and
              m2["open"] > m1["high"] and
              m2["close"] < (m1["open"] + m1["close"]) / 2):
            patterns.append(PatternResult(
                pattern=CandlePattern.DARK_CLOUD_COVER,
                direction="bearish",
                strength=0.7,
                bar_index=idx,
                description="Dark cloud cover - bearish reversal signal"
            ))

        return patterns

    def _detect_three_candle(self, ohlcv: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detect three candle patterns"""
        patterns = []

        m1 = self._get_candle_metrics(ohlcv, idx - 2)
        m2 = self._get_candle_metrics(ohlcv, idx - 1)
        m3 = self._get_candle_metrics(ohlcv, idx)

        # Morning Star
        if (m1["is_bearish"] and m1["body_pct"] > 0.5 and
            m2["body_pct"] < 0.3 and  # Small body (star)
            m3["is_bullish"] and m3["body_pct"] > 0.5 and
            m3["close"] > (m1["open"] + m1["close"]) / 2):
            patterns.append(PatternResult(
                pattern=CandlePattern.MORNING_STAR,
                direction="bullish",
                strength=0.85,
                bar_index=idx,
                description="Morning star - strong bullish reversal"
            ))

        # Evening Star
        elif (m1["is_bullish"] and m1["body_pct"] > 0.5 and
              m2["body_pct"] < 0.3 and  # Small body (star)
              m3["is_bearish"] and m3["body_pct"] > 0.5 and
              m3["close"] < (m1["open"] + m1["close"]) / 2):
            patterns.append(PatternResult(
                pattern=CandlePattern.EVENING_STAR,
                direction="bearish",
                strength=0.85,
                bar_index=idx,
                description="Evening star - strong bearish reversal"
            ))

        # Three White Soldiers
        elif (m1["is_bullish"] and m2["is_bullish"] and m3["is_bullish"] and
              m2["close"] > m1["close"] and m3["close"] > m2["close"] and
              m1["body_pct"] > 0.5 and m2["body_pct"] > 0.5 and m3["body_pct"] > 0.5):
            patterns.append(PatternResult(
                pattern=CandlePattern.THREE_WHITE_SOLDIERS,
                direction="bullish",
                strength=0.9,
                bar_index=idx,
                description="Three white soldiers - strong bullish continuation"
            ))

        # Three Black Crows
        elif (m1["is_bearish"] and m2["is_bearish"] and m3["is_bearish"] and
              m2["close"] < m1["close"] and m3["close"] < m2["close"] and
              m1["body_pct"] > 0.5 and m2["body_pct"] > 0.5 and m3["body_pct"] > 0.5):
            patterns.append(PatternResult(
                pattern=CandlePattern.THREE_BLACK_CROWS,
                direction="bearish",
                strength=0.9,
                bar_index=idx,
                description="Three black crows - strong bearish continuation"
            ))

        return patterns


# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

class DivergenceType(Enum):
    """Types of divergence"""
    REGULAR_BULLISH = "regular_bullish"    # Price lower low, indicator higher low
    REGULAR_BEARISH = "regular_bearish"    # Price higher high, indicator lower high
    HIDDEN_BULLISH = "hidden_bullish"      # Price higher low, indicator lower low
    HIDDEN_BEARISH = "hidden_bearish"      # Price lower high, indicator higher high


@dataclass
class DivergenceResult:
    """Result of divergence detection"""
    divergence_type: DivergenceType
    indicator: str  # "rsi", "macd", "stochastic"
    strength: float  # 0-1
    price_point1: Tuple[int, float]  # (bar_index, price)
    price_point2: Tuple[int, float]
    indicator_point1: Tuple[int, float]  # (bar_index, value)
    indicator_point2: Tuple[int, float]
    description: str


class DivergenceDetector:
    """Detects divergences between price and momentum indicators"""

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_period: int = 14,
        lookback: int = 30,
        min_swing_pct: float = 0.5,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_period = stoch_period
        self.lookback = lookback
        self.min_swing_pct = min_swing_pct

    def detect_all(self, ohlcv: pd.DataFrame) -> List[DivergenceResult]:
        """Detect divergences across all indicators"""
        divergences = []

        # Calculate indicators
        df = ohlcv.copy()
        df["rsi"] = RSIIndicator(close=df["close"], window=self.rsi_period).rsi()

        macd = MACD(
            close=df["close"],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        df["macd"] = macd.macd()
        df["macd_hist"] = macd.macd_diff()

        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.stoch_period
        )
        df["stoch_k"] = stoch.stoch()

        # Find swing points
        highs, lows = self._find_swing_points(df)

        # Detect divergences for each indicator
        for indicator in ["rsi", "macd_hist", "stoch_k"]:
            indicator_name = indicator.replace("_hist", "").replace("_k", "")

            # Regular bullish: price lower low, indicator higher low
            div = self._check_bullish_divergence(df, lows, indicator)
            if div:
                div.indicator = indicator_name
                divergences.append(div)

            # Regular bearish: price higher high, indicator lower high
            div = self._check_bearish_divergence(df, highs, indicator)
            if div:
                div.indicator = indicator_name
                divergences.append(div)

            # Hidden bullish: price higher low, indicator lower low
            div = self._check_hidden_bullish(df, lows, indicator)
            if div:
                div.indicator = indicator_name
                divergences.append(div)

            # Hidden bearish: price lower high, indicator higher high
            div = self._check_hidden_bearish(df, highs, indicator)
            if div:
                div.indicator = indicator_name
                divergences.append(div)

        return divergences

    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        highs = []
        lows = []

        close = df["close"].values
        n = len(close)

        # Look for local maxima/minima
        for i in range(2, n - 2):
            # Swing high
            if (close[i] > close[i-1] and close[i] > close[i-2] and
                close[i] > close[i+1] and close[i] > close[i+2]):
                highs.append(i)

            # Swing low
            if (close[i] < close[i-1] and close[i] < close[i-2] and
                close[i] < close[i+1] and close[i] < close[i+2]):
                lows.append(i)

        # Keep only recent swings
        highs = [h for h in highs if h >= n - self.lookback]
        lows = [l for l in lows if l >= n - self.lookback]

        return highs, lows

    def _check_bullish_divergence(
        self, df: pd.DataFrame, lows: List[int], indicator: str
    ) -> Optional[DivergenceResult]:
        """Check for regular bullish divergence (price lower low, indicator higher low)"""
        if len(lows) < 2:
            return None

        close = df["close"].values
        ind = df[indicator].values

        # Check most recent two lows
        l1, l2 = lows[-2], lows[-1]

        # Price made lower low
        if close[l2] >= close[l1]:
            return None

        # Indicator made higher low
        if pd.isna(ind[l1]) or pd.isna(ind[l2]) or ind[l2] <= ind[l1]:
            return None

        strength = min(1.0, abs(close[l1] - close[l2]) / close[l1] * 10)

        return DivergenceResult(
            divergence_type=DivergenceType.REGULAR_BULLISH,
            indicator=indicator,
            strength=strength,
            price_point1=(l1, close[l1]),
            price_point2=(l2, close[l2]),
            indicator_point1=(l1, ind[l1]),
            indicator_point2=(l2, ind[l2]),
            description=f"Regular bullish divergence: Price made lower low but {indicator} made higher low"
        )

    def _check_bearish_divergence(
        self, df: pd.DataFrame, highs: List[int], indicator: str
    ) -> Optional[DivergenceResult]:
        """Check for regular bearish divergence (price higher high, indicator lower high)"""
        if len(highs) < 2:
            return None

        close = df["close"].values
        ind = df[indicator].values

        h1, h2 = highs[-2], highs[-1]

        # Price made higher high
        if close[h2] <= close[h1]:
            return None

        # Indicator made lower high
        if pd.isna(ind[h1]) or pd.isna(ind[h2]) or ind[h2] >= ind[h1]:
            return None

        strength = min(1.0, abs(close[h2] - close[h1]) / close[h1] * 10)

        return DivergenceResult(
            divergence_type=DivergenceType.REGULAR_BEARISH,
            indicator=indicator,
            strength=strength,
            price_point1=(h1, close[h1]),
            price_point2=(h2, close[h2]),
            indicator_point1=(h1, ind[h1]),
            indicator_point2=(h2, ind[h2]),
            description=f"Regular bearish divergence: Price made higher high but {indicator} made lower high"
        )

    def _check_hidden_bullish(
        self, df: pd.DataFrame, lows: List[int], indicator: str
    ) -> Optional[DivergenceResult]:
        """Check for hidden bullish divergence (price higher low, indicator lower low)"""
        if len(lows) < 2:
            return None

        close = df["close"].values
        ind = df[indicator].values

        l1, l2 = lows[-2], lows[-1]

        # Price made higher low
        if close[l2] <= close[l1]:
            return None

        # Indicator made lower low
        if pd.isna(ind[l1]) or pd.isna(ind[l2]) or ind[l2] >= ind[l1]:
            return None

        strength = min(1.0, abs(close[l2] - close[l1]) / close[l1] * 10) * 0.8

        return DivergenceResult(
            divergence_type=DivergenceType.HIDDEN_BULLISH,
            indicator=indicator,
            strength=strength,
            price_point1=(l1, close[l1]),
            price_point2=(l2, close[l2]),
            indicator_point1=(l1, ind[l1]),
            indicator_point2=(l2, ind[l2]),
            description=f"Hidden bullish divergence: Price made higher low but {indicator} made lower low (trend continuation)"
        )

    def _check_hidden_bearish(
        self, df: pd.DataFrame, highs: List[int], indicator: str
    ) -> Optional[DivergenceResult]:
        """Check for hidden bearish divergence (price lower high, indicator higher high)"""
        if len(highs) < 2:
            return None

        close = df["close"].values
        ind = df[indicator].values

        h1, h2 = highs[-2], highs[-1]

        # Price made lower high
        if close[h2] >= close[h1]:
            return None

        # Indicator made higher high
        if pd.isna(ind[h1]) or pd.isna(ind[h2]) or ind[h2] <= ind[h1]:
            return None

        strength = min(1.0, abs(close[h1] - close[h2]) / close[h1] * 10) * 0.8

        return DivergenceResult(
            divergence_type=DivergenceType.HIDDEN_BEARISH,
            indicator=indicator,
            strength=strength,
            price_point1=(h1, close[h1]),
            price_point2=(h2, close[h2]),
            indicator_point1=(h1, ind[h1]),
            indicator_point2=(h2, ind[h2]),
            description=f"Hidden bearish divergence: Price made lower high but {indicator} made higher high (trend continuation)"
        )


# ============================================================================
# CONFLUENCE ZONE DETECTION
# ============================================================================

@dataclass
class SupportResistanceLevel:
    """A support or resistance level"""
    price: float
    level_type: str  # "support" or "resistance"
    strength: int  # Number of touches/confirmations
    sources: List[str]  # What identified this level (EMA, pivot, etc.)


@dataclass
class ConfluenceZone:
    """A zone where multiple S/R levels converge"""
    price_low: float
    price_high: float
    center: float
    zone_type: str  # "support", "resistance", "pivot"
    strength: float  # 0-1, based on number of confluent levels
    levels: List[SupportResistanceLevel]
    description: str


class ConfluenceDetector:
    """
    Detects confluence zones where multiple technical levels align.

    A confluence zone is stronger because multiple independent indicators
    point to the same price area as significant.
    """

    def __init__(
        self,
        ema_periods: List[int] = None,
        sma_periods: List[int] = None,
        pivot_lookback: int = 20,
        zone_tolerance_pct: float = 0.5,
        min_confluence: int = 3,
    ):
        self.ema_periods = ema_periods or [20, 50, 100, 200]
        self.sma_periods = sma_periods or [50, 200]
        self.pivot_lookback = pivot_lookback
        self.zone_tolerance_pct = zone_tolerance_pct
        self.min_confluence = min_confluence

    def detect(self, ohlcv: pd.DataFrame) -> List[ConfluenceZone]:
        """
        Detect confluence zones in the price data.

        Analyzes:
        - EMA levels
        - SMA levels
        - Pivot points (swing highs/lows)
        - Round numbers
        - Previous day high/low
        - Fibonacci retracements

        Returns:
            List of confluence zones sorted by proximity to current price
        """
        levels = []
        current_price = ohlcv["close"].iloc[-1]

        # Add EMA levels
        for period in self.ema_periods:
            if len(ohlcv) >= period:
                ema = EMAIndicator(close=ohlcv["close"], window=period).ema_indicator()
                ema_value = ema.iloc[-1]
                if not pd.isna(ema_value):
                    level_type = "support" if ema_value < current_price else "resistance"
                    levels.append(SupportResistanceLevel(
                        price=ema_value,
                        level_type=level_type,
                        strength=2 if period >= 100 else 1,
                        sources=[f"EMA{period}"]
                    ))

        # Add SMA levels
        for period in self.sma_periods:
            if len(ohlcv) >= period:
                sma = SMAIndicator(close=ohlcv["close"], window=period).sma_indicator()
                sma_value = sma.iloc[-1]
                if not pd.isna(sma_value):
                    level_type = "support" if sma_value < current_price else "resistance"
                    levels.append(SupportResistanceLevel(
                        price=sma_value,
                        level_type=level_type,
                        strength=2 if period >= 100 else 1,
                        sources=[f"SMA{period}"]
                    ))

        # Add pivot points (swing highs/lows)
        pivots = self._find_pivots(ohlcv)
        levels.extend(pivots)

        # Add round numbers
        round_levels = self._find_round_numbers(current_price)
        levels.extend(round_levels)

        # Add Fibonacci retracements
        fib_levels = self._calculate_fibonacci(ohlcv)
        levels.extend(fib_levels)

        # Group levels into confluence zones
        zones = self._group_into_zones(levels, current_price)

        # Filter by minimum confluence
        zones = [z for z in zones if len(z.levels) >= self.min_confluence]

        # Sort by proximity to current price
        zones.sort(key=lambda z: abs(z.center - current_price))

        return zones

    def _find_pivots(self, ohlcv: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Find pivot points (swing highs and lows)"""
        levels = []
        n = len(ohlcv)

        if n < self.pivot_lookback + 5:
            return levels

        high = ohlcv["high"].values
        low = ohlcv["low"].values
        current_price = ohlcv["close"].iloc[-1]

        # Find swing highs and lows in lookback period
        for i in range(n - self.pivot_lookback, n - 2):
            # Swing high
            if (high[i] > high[i-1] and high[i] > high[i-2] and
                high[i] > high[i+1] and high[i] > high[i+2]):
                level_type = "support" if high[i] < current_price else "resistance"
                levels.append(SupportResistanceLevel(
                    price=high[i],
                    level_type=level_type,
                    strength=1,
                    sources=["SwingHigh"]
                ))

            # Swing low
            if (low[i] < low[i-1] and low[i] < low[i-2] and
                low[i] < low[i+1] and low[i] < low[i+2]):
                level_type = "support" if low[i] < current_price else "resistance"
                levels.append(SupportResistanceLevel(
                    price=low[i],
                    level_type=level_type,
                    strength=1,
                    sources=["SwingLow"]
                ))

        return levels

    def _find_round_numbers(self, current_price: float) -> List[SupportResistanceLevel]:
        """Find psychologically significant round numbers near current price"""
        levels = []

        # Determine round number increment based on price magnitude
        if current_price > 10000:
            increments = [1000, 5000, 10000]
        elif current_price > 1000:
            increments = [100, 500, 1000]
        elif current_price > 100:
            increments = [10, 50, 100]
        else:
            increments = [1, 5, 10]

        # Find nearby round numbers
        for inc in increments:
            lower = (current_price // inc) * inc
            upper = lower + inc

            for price in [lower, upper]:
                if abs(price - current_price) / current_price < 0.05:  # Within 5%
                    level_type = "support" if price < current_price else "resistance"
                    strength = 2 if inc == increments[-1] else 1  # Higher for bigger round numbers
                    levels.append(SupportResistanceLevel(
                        price=price,
                        level_type=level_type,
                        strength=strength,
                        sources=[f"Round_{int(inc)}"]
                    ))

        return levels

    def _calculate_fibonacci(self, ohlcv: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Calculate Fibonacci retracement levels"""
        levels = []

        if len(ohlcv) < 50:
            return levels

        # Find recent swing high and low
        recent = ohlcv.iloc[-50:]
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        current_price = ohlcv["close"].iloc[-1]

        diff = swing_high - swing_low

        # Standard Fibonacci levels
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

        for ratio in fib_ratios:
            # Retracement from high
            price = swing_high - (diff * ratio)
            level_type = "support" if price < current_price else "resistance"
            levels.append(SupportResistanceLevel(
                price=price,
                level_type=level_type,
                strength=2 if ratio in [0.5, 0.618] else 1,
                sources=[f"Fib_{ratio}"]
            ))

        return levels

    def _group_into_zones(
        self, levels: List[SupportResistanceLevel], current_price: float
    ) -> List[ConfluenceZone]:
        """Group nearby levels into confluence zones"""
        if not levels:
            return []

        # Sort levels by price
        levels.sort(key=lambda x: x.price)

        zones = []
        tolerance = current_price * (self.zone_tolerance_pct / 100)

        # Group levels within tolerance
        used = set()
        for i, level in enumerate(levels):
            if i in used:
                continue

            zone_levels = [level]
            used.add(i)

            # Find nearby levels
            for j, other in enumerate(levels):
                if j in used:
                    continue
                if abs(other.price - level.price) <= tolerance:
                    zone_levels.append(other)
                    used.add(j)

            if len(zone_levels) >= 2:
                prices = [l.price for l in zone_levels]
                center = sum(prices) / len(prices)

                # Determine zone type
                support_count = sum(1 for l in zone_levels if l.level_type == "support")
                resistance_count = len(zone_levels) - support_count

                if support_count > resistance_count:
                    zone_type = "support"
                elif resistance_count > support_count:
                    zone_type = "resistance"
                else:
                    zone_type = "pivot"

                # Calculate strength based on number and quality of levels
                strength = min(1.0, sum(l.strength for l in zone_levels) / 10)

                sources = []
                for l in zone_levels:
                    sources.extend(l.sources)

                zones.append(ConfluenceZone(
                    price_low=min(prices),
                    price_high=max(prices),
                    center=center,
                    zone_type=zone_type,
                    strength=strength,
                    levels=zone_levels,
                    description=f"{zone_type.title()} confluence zone at ${center:.2f} ({len(zone_levels)} levels: {', '.join(set(sources))})"
                ))

        return zones


# ============================================================================
# COMBINED TECHNICAL ANALYZER
# ============================================================================

@dataclass
class TechnicalAnalysisResult:
    """Combined result of all technical analysis"""
    candlestick_patterns: List[PatternResult]
    divergences: List[DivergenceResult]
    confluence_zones: List[ConfluenceZone]
    nearest_support: Optional[ConfluenceZone]
    nearest_resistance: Optional[ConfluenceZone]
    signal_boost: float  # Overall technical signal boost (-1 to 1)
    summary: str


class TechnicalAnalyzer:
    """
    Combined technical analyzer that integrates:
    - Candlestick patterns
    - Divergences
    - Confluence zones
    """

    def __init__(self):
        self.candle_analyzer = CandlestickAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.confluence_detector = ConfluenceDetector()

    def analyze(self, ohlcv: pd.DataFrame) -> TechnicalAnalysisResult:
        """
        Run comprehensive technical analysis.

        Returns combined analysis result with signal boost.
        """
        current_price = ohlcv["close"].iloc[-1]

        # Detect patterns
        patterns = self.candle_analyzer.analyze(ohlcv)
        divergences = self.divergence_detector.detect_all(ohlcv)
        zones = self.confluence_detector.detect(ohlcv)

        # Find nearest support/resistance
        support_zones = [z for z in zones if z.zone_type == "support" and z.center < current_price]
        resistance_zones = [z for z in zones if z.zone_type == "resistance" and z.center > current_price]

        nearest_support = support_zones[0] if support_zones else None
        nearest_resistance = resistance_zones[0] if resistance_zones else None

        # Calculate signal boost
        signal_boost = self._calculate_signal_boost(patterns, divergences, zones, current_price)

        # Generate summary
        summary = self._generate_summary(patterns, divergences, zones, current_price)

        return TechnicalAnalysisResult(
            candlestick_patterns=patterns,
            divergences=divergences,
            confluence_zones=zones,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            signal_boost=signal_boost,
            summary=summary,
        )

    def _calculate_signal_boost(
        self,
        patterns: List[PatternResult],
        divergences: List[DivergenceResult],
        zones: List[ConfluenceZone],
        current_price: float,
    ) -> float:
        """Calculate overall signal boost from technical factors"""
        boost = 0.0

        # Pattern contribution
        for p in patterns[-3:]:  # Consider last 3 patterns
            if p.direction == "bullish":
                boost += p.strength * 0.15
            elif p.direction == "bearish":
                boost -= p.strength * 0.15

        # Divergence contribution
        for d in divergences:
            if d.divergence_type in [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]:
                boost += d.strength * 0.2
            else:
                boost -= d.strength * 0.2

        # Confluence zone proximity contribution
        for z in zones[:3]:  # Consider nearest 3 zones
            distance_pct = abs(z.center - current_price) / current_price * 100

            if distance_pct < 0.5:  # Very close to zone
                if z.zone_type == "support":
                    boost += z.strength * 0.1  # Near support = bullish
                elif z.zone_type == "resistance":
                    boost -= z.strength * 0.1  # Near resistance = bearish

        return max(-1.0, min(1.0, boost))

    def _generate_summary(
        self,
        patterns: List[PatternResult],
        divergences: List[DivergenceResult],
        zones: List[ConfluenceZone],
        current_price: float,
    ) -> str:
        """Generate human-readable summary"""
        parts = []

        # Patterns
        bullish_patterns = [p for p in patterns if p.direction == "bullish"]
        bearish_patterns = [p for p in patterns if p.direction == "bearish"]

        if bullish_patterns:
            names = [p.pattern.value.replace("_", " ").title() for p in bullish_patterns[-2:]]
            parts.append(f"Bullish patterns: {', '.join(names)}")
        if bearish_patterns:
            names = [p.pattern.value.replace("_", " ").title() for p in bearish_patterns[-2:]]
            parts.append(f"Bearish patterns: {', '.join(names)}")

        # Divergences
        bullish_div = [d for d in divergences if "BULLISH" in d.divergence_type.name]
        bearish_div = [d for d in divergences if "BEARISH" in d.divergence_type.name]

        if bullish_div:
            parts.append(f"Bullish {bullish_div[0].indicator} divergence detected")
        if bearish_div:
            parts.append(f"Bearish {bearish_div[0].indicator} divergence detected")

        # Zones
        strong_zones = [z for z in zones if z.strength >= 0.5]
        if strong_zones:
            zone = strong_zones[0]
            parts.append(f"Strong {zone.zone_type} zone at ${zone.center:.2f}")

        return " | ".join(parts) if parts else "No significant technical signals"
