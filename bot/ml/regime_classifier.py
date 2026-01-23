"""
Market Regime Classifier.

Detects whether the market is in a Bull, Bear, or Sideways regime
to help select the optimal trading strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands


class MarketRegime(Enum):
    """Market regime types."""

    STRONG_BULL = "strong_bull"  # Strong uptrend, use aggressive momentum
    BULL = "bull"  # Moderate uptrend, use trend-following
    SIDEWAYS = "sideways"  # Range-bound, use mean-reversion
    BEAR = "bear"  # Moderate downtrend, reduce exposure
    STRONG_BEAR = "strong_bear"  # Strong downtrend, stay flat or short
    VOLATILE = "volatile"  # High volatility, reduce position size


@dataclass
class RegimeAnalysis:
    """Result of market regime analysis."""

    regime: MarketRegime
    confidence: float
    trend_strength: float  # -1 to 1 (negative = bearish, positive = bullish)
    volatility_level: str  # low, normal, high, extreme
    volatility_percentile: float
    adx_value: float
    momentum_score: float
    support_level: float
    resistance_level: float
    regime_duration: int  # Bars in current regime
    recommended_strategy: str
    reasoning: List[str]

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "trend_strength": round(self.trend_strength, 4),
            "volatility_level": self.volatility_level,
            "volatility_percentile": round(self.volatility_percentile, 2),
            "adx_value": round(self.adx_value, 2),
            "momentum_score": round(self.momentum_score, 4),
            "support_level": round(self.support_level, 2),
            "resistance_level": round(self.resistance_level, 2),
            "regime_duration": self.regime_duration,
            "recommended_strategy": self.recommended_strategy,
            "reasoning": self.reasoning,
        }


class MarketRegimeClassifier:
    """
    Classifies market conditions into regimes for strategy selection.

    Uses multiple indicators:
    - EMA slopes for trend direction
    - ADX for trend strength
    - ATR for volatility
    - Bollinger Bands for range analysis
    - Price action patterns

    Outputs regime + recommended strategy type.
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        ema_trend: int = 200,
        adx_period: int = 14,
        atr_period: int = 14,
        lookback: int = 100,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.lookback = lookback

        self._regime_history: List[MarketRegime] = []

    def classify(self, ohlcv: pd.DataFrame) -> RegimeAnalysis:
        """
        Classify current market regime.

        Args:
            ohlcv: OHLCV DataFrame with at least 200 rows

        Returns:
            RegimeAnalysis with regime type and supporting metrics
        """
        if len(ohlcv) < self.ema_trend + 10:
            return self._default_analysis()

        df = ohlcv.copy().sort_index()

        # Calculate indicators
        df = self._add_indicators(df)
        last = df.iloc[-1]

        # Analyze components
        trend_strength = self._calculate_trend_strength(df)
        volatility_level, vol_percentile = self._calculate_volatility(df)
        momentum = self._calculate_momentum(df)
        support, resistance = self._calculate_levels(df)
        adx_value = float(last["adx"]) if "adx" in last else 20.0

        # Determine regime
        regime, confidence, reasoning = self._determine_regime(
            trend_strength, volatility_level, vol_percentile, adx_value, momentum, df
        )

        # Track regime history
        self._regime_history.append(regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

        regime_duration = self._calculate_regime_duration()

        # Recommend strategy
        recommended_strategy = self._recommend_strategy(regime, volatility_level)

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            volatility_percentile=vol_percentile,
            adx_value=adx_value,
            momentum_score=momentum,
            support_level=support,
            resistance_level=resistance,
            regime_duration=regime_duration,
            recommended_strategy=recommended_strategy,
            reasoning=reasoning,
        )

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for regime detection."""
        # EMAs
        df["ema_fast"] = EMAIndicator(df["close"], window=self.ema_fast).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=self.ema_slow).ema_indicator()
        df["ema_trend"] = EMAIndicator(df["close"], window=self.ema_trend).ema_indicator()

        # EMA slopes (rate of change)
        df["ema_fast_slope"] = df["ema_fast"].pct_change(5)
        df["ema_slow_slope"] = df["ema_slow"].pct_change(10)
        df["ema_trend_slope"] = df["ema_trend"].pct_change(20)

        # ADX for trend strength
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=self.adx_period)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # ATR for volatility
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=self.atr_period)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = df["atr"] / df["close"]

        # Bollinger Bands
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

        # Returns
        df["return_5"] = df["close"].pct_change(5)
        df["return_20"] = df["close"].pct_change(20)

        return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength from -1 (strong bear) to 1 (strong bull).
        """
        last = df.iloc[-1]

        # EMA alignment score
        ema_alignment = 0.0
        if last["ema_fast"] > last["ema_slow"] > last["ema_trend"]:
            ema_alignment = 1.0  # Perfect bullish alignment
        elif last["ema_fast"] < last["ema_slow"] < last["ema_trend"]:
            ema_alignment = -1.0  # Perfect bearish alignment
        else:
            # Partial alignment
            if last["ema_fast"] > last["ema_slow"]:
                ema_alignment += 0.3
            else:
                ema_alignment -= 0.3
            if last["close"] > last["ema_trend"]:
                ema_alignment += 0.2
            else:
                ema_alignment -= 0.2

        # EMA slope contribution
        slope_score = 0.0
        slope_score += np.sign(last["ema_fast_slope"]) * min(abs(last["ema_fast_slope"]) * 50, 0.3)
        slope_score += np.sign(last["ema_slow_slope"]) * min(abs(last["ema_slow_slope"]) * 30, 0.2)

        # ADX direction (DI+ vs DI-)
        adx_direction = 0.0
        if last["adx"] > 20:  # Only count if trend is significant
            adx_direction = (last["adx_pos"] - last["adx_neg"]) / 100

        # Combine scores
        trend_strength = ema_alignment * 0.5 + slope_score * 0.3 + adx_direction * 0.2

        return float(np.clip(trend_strength, -1, 1))

    def _calculate_volatility(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Calculate volatility level and percentile."""
        atr_pct = df["atr_pct"].dropna()
        current_atr = float(df["atr_pct"].iloc[-1])

        # Calculate percentile of current volatility
        percentile = float((atr_pct < current_atr).mean() * 100)

        # Classify volatility
        if percentile < 20:
            level = "low"
        elif percentile < 50:
            level = "normal"
        elif percentile < 80:
            level = "high"
        else:
            level = "extreme"

        return level, percentile

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score from -1 to 1."""
        last = df.iloc[-1]

        # Short-term momentum
        short_mom = last["return_5"] * 10  # Scale for sensitivity

        # Medium-term momentum
        med_mom = last["return_20"] * 5

        # Combined momentum
        momentum = short_mom * 0.6 + med_mom * 0.4

        return float(np.clip(momentum, -1, 1))

    def _calculate_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels."""
        lookback_data = df.tail(self.lookback)

        # Simple pivot-based levels
        high = lookback_data["high"].max()
        low = lookback_data["low"].min()
        close = float(df["close"].iloc[-1])

        # Recent swing highs/lows
        resistance = lookback_data["high"].rolling(20).max().iloc[-1]
        support = lookback_data["low"].rolling(20).min().iloc[-1]

        return float(support), float(resistance)

    def _determine_regime(
        self,
        trend_strength: float,
        volatility_level: str,
        vol_percentile: float,
        adx: float,
        momentum: float,
        df: pd.DataFrame,
    ) -> Tuple[MarketRegime, float, List[str]]:
        """Determine the market regime based on all factors."""
        reasoning = []

        # Check for extreme volatility first
        if volatility_level == "extreme" or vol_percentile > 90:
            reasoning.append(f"Extreme volatility (percentile: {vol_percentile:.0f}%)")
            return MarketRegime.VOLATILE, 0.8, reasoning

        # Strong trends (ADX > 25)
        if adx > 25:
            if trend_strength > 0.5:
                reasoning.append(
                    f"Strong ADX ({adx:.1f}) with bullish trend ({trend_strength:.2f})"
                )
                return MarketRegime.STRONG_BULL, min(0.9, adx / 40), reasoning
            elif trend_strength < -0.5:
                reasoning.append(
                    f"Strong ADX ({adx:.1f}) with bearish trend ({trend_strength:.2f})"
                )
                return MarketRegime.STRONG_BEAR, min(0.9, adx / 40), reasoning

        # Moderate trends
        if trend_strength > 0.25 and momentum > 0.1:
            reasoning.append(
                f"Positive trend ({trend_strength:.2f}) with bullish momentum ({momentum:.2f})"
            )
            confidence = 0.6 + trend_strength * 0.2
            return MarketRegime.BULL, confidence, reasoning

        if trend_strength < -0.25 and momentum < -0.1:
            reasoning.append(
                f"Negative trend ({trend_strength:.2f}) with bearish momentum ({momentum:.2f})"
            )
            confidence = 0.6 + abs(trend_strength) * 0.2
            return MarketRegime.BEAR, confidence, reasoning

        # High volatility without clear trend
        if volatility_level == "high":
            reasoning.append(
                f"High volatility ({vol_percentile:.0f}% percentile) without clear trend"
            )
            return MarketRegime.VOLATILE, 0.6, reasoning

        # Default to sideways
        reasoning.append(f"Low trend strength ({trend_strength:.2f}), weak ADX ({adx:.1f})")
        reasoning.append("Market appears range-bound")
        return MarketRegime.SIDEWAYS, 0.7, reasoning

    def _calculate_regime_duration(self) -> int:
        """Count how many bars we've been in the current regime."""
        if len(self._regime_history) < 2:
            return 1

        current = self._regime_history[-1]
        duration = 1

        for regime in reversed(self._regime_history[:-1]):
            if regime == current:
                duration += 1
            else:
                break

        return duration

    def _recommend_strategy(self, regime: MarketRegime, volatility: str) -> str:
        """Recommend strategy type based on regime."""
        strategy_map = {
            MarketRegime.STRONG_BULL: "momentum_aggressive",
            MarketRegime.BULL: "trend_following",
            MarketRegime.SIDEWAYS: "mean_reversion",
            MarketRegime.BEAR: "defensive",
            MarketRegime.STRONG_BEAR: "stay_flat",
            MarketRegime.VOLATILE: "reduced_size",
        }

        base_strategy = strategy_map.get(regime, "default")

        # Adjust for volatility
        if volatility == "extreme":
            return "stay_flat"
        elif volatility == "high" and regime not in [
            MarketRegime.STRONG_BULL,
            MarketRegime.STRONG_BEAR,
        ]:
            return "reduced_size"

        return base_strategy

    def _default_analysis(self) -> RegimeAnalysis:
        """Return default analysis when not enough data."""
        return RegimeAnalysis(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.3,
            trend_strength=0.0,
            volatility_level="normal",
            volatility_percentile=50.0,
            adx_value=20.0,
            momentum_score=0.0,
            support_level=0.0,
            resistance_level=0.0,
            regime_duration=0,
            recommended_strategy="default",
            reasoning=["Insufficient data for analysis"],
        )

    def get_strategy_parameters(self, regime: MarketRegime) -> Dict:
        """
        Get recommended strategy parameters for each regime.

        Returns parameter adjustments relative to defaults.
        """
        params = {
            MarketRegime.STRONG_BULL: {
                "position_size_multiplier": 1.5,
                "stop_loss_multiplier": 1.2,  # Wider stops in trends
                "take_profit_multiplier": 2.0,  # Let winners run
                "entry_threshold": 0.3,  # Lower bar for entries
                "prefer_long": True,
            },
            MarketRegime.BULL: {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.5,
                "entry_threshold": 0.4,
                "prefer_long": True,
            },
            MarketRegime.SIDEWAYS: {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 0.8,  # Tighter stops
                "take_profit_multiplier": 0.8,  # Take profits quickly
                "entry_threshold": 0.6,  # Higher bar for entries
                "prefer_long": None,  # No bias
            },
            MarketRegime.BEAR: {
                "position_size_multiplier": 0.6,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 1.0,
                "entry_threshold": 0.6,
                "prefer_long": False,
            },
            MarketRegime.STRONG_BEAR: {
                "position_size_multiplier": 0.3,
                "stop_loss_multiplier": 0.6,
                "take_profit_multiplier": 0.8,
                "entry_threshold": 0.8,  # Very selective
                "prefer_long": False,
            },
            MarketRegime.VOLATILE: {
                "position_size_multiplier": 0.5,
                "stop_loss_multiplier": 1.5,  # Wider for volatility
                "take_profit_multiplier": 1.5,
                "entry_threshold": 0.7,
                "prefer_long": None,
            },
        }

        return params.get(regime, params[MarketRegime.SIDEWAYS])
