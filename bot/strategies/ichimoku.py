"""
Ichimoku Cloud Trading Strategy.

A comprehensive Japanese technical analysis strategy that provides
support/resistance, trend direction, and momentum signals.

Components:
- Tenkan-sen (Conversion Line): 9-period high+low/2
- Kijun-sen (Base Line): 26-period high+low/2
- Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
- Senkou Span B (Leading Span B): 52-period high+low/2, plotted 26 periods ahead
- Chikou Span (Lagging Span): Close plotted 26 periods behind

Trading Rules:
- LONG: Price above cloud, Tenkan > Kijun, Chikou above price
- SHORT: Price below cloud, Tenkan < Kijun, Chikou below price
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class IchimokuConfig(StrategyConfig):
    """Configuration for Ichimoku strategy."""

    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26
    min_cloud_thickness_pct: float = 0.5


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Cloud trading strategy.

    Generates signals based on:
    - Price position relative to cloud
    - Tenkan-sen / Kijun-sen crossover
    - Chikou Span confirmation
    - Cloud thickness (trend strength)
    """

    def __init__(self, config: Optional[IchimokuConfig] = None):
        self.ichimoku_config = config or IchimokuConfig()
        super().__init__(self.ichimoku_config)

    @property
    def name(self) -> str:
        return "ichimoku_cloud"

    @property
    def description(self) -> str:
        return "Ichimoku Cloud strategy using all five components for trend following"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["TRENDING", "BULL", "BEAR", "BULLISH_TRENDING", "BEARISH_TRENDING"]

    def get_required_indicators(self) -> List[str]:
        return [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
            "cloud_thickness",
        ]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku indicators to dataframe."""
        df = ohlcv.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.ichimoku_config.tenkan_period).max()
        tenkan_low = low.rolling(window=self.ichimoku_config.tenkan_period).min()
        df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.ichimoku_config.kijun_period).max()
        kijun_low = low.rolling(window=self.ichimoku_config.kijun_period).min()
        df["kijun_sen"] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A) - shifted forward
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(
            self.ichimoku_config.displacement
        )

        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b_high = high.rolling(window=self.ichimoku_config.senkou_b_period).max()
        senkou_b_low = low.rolling(window=self.ichimoku_config.senkou_b_period).min()
        df["senkou_span_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(
            self.ichimoku_config.displacement
        )

        # Chikou Span (Lagging Span) - shifted backward
        df["chikou_span"] = close.shift(-self.ichimoku_config.displacement)

        # Cloud boundaries
        df["cloud_top"] = df[["senkou_span_a", "senkou_span_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_span_a", "senkou_span_b"]].min(axis=1)
        df["cloud_thickness"] = (df["cloud_top"] - df["cloud_bottom"]) / close * 100

        # Position relative to cloud
        df["above_cloud"] = close > df["cloud_top"]
        df["below_cloud"] = close < df["cloud_bottom"]
        df["in_cloud"] = ~df["above_cloud"] & ~df["below_cloud"]

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate Ichimoku trading signal."""
        if (
            len(ohlcv)
            < self.ichimoku_config.senkou_b_period + self.ichimoku_config.displacement + 5
        ):
            return self._flat_signal("Insufficient data for Ichimoku")

        df = self.add_indicators(ohlcv)
        current = df.iloc[-1]

        # Gather current indicator values
        close = float(current["close"])
        tenkan = float(current["tenkan_sen"])
        kijun = float(current["kijun_sen"])
        span_a = float(current["senkou_span_a"]) if not pd.isna(current["senkou_span_a"]) else close
        span_b = float(current["senkou_span_b"]) if not pd.isna(current["senkou_span_b"]) else close
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        cloud_thickness = (
            float(current["cloud_thickness"]) if not pd.isna(current["cloud_thickness"]) else 0
        )

        # Check for previous crossover
        prev = df.iloc[-2]
        prev_tenkan = float(prev["tenkan_sen"]) if not pd.isna(prev["tenkan_sen"]) else tenkan
        prev_kijun = float(prev["kijun_sen"]) if not pd.isna(prev["kijun_sen"]) else kijun

        # Crossover detection
        bullish_cross = prev_tenkan <= prev_kijun and tenkan > kijun
        bearish_cross = prev_tenkan >= prev_kijun and tenkan < kijun

        # Chikou Span confirmation (compare with price 26 periods ago)
        chikou_idx = -self.ichimoku_config.displacement - 1
        chikou_price = float(df.iloc[chikou_idx]["close"]) if abs(chikou_idx) < len(df) else close
        chikou_bullish = close > chikou_price
        chikou_bearish = close < chikou_price

        # Build indicators dict
        indicators = {
            "tenkan_sen": tenkan,
            "kijun_sen": kijun,
            "senkou_span_a": span_a,
            "senkou_span_b": span_b,
            "cloud_thickness": cloud_thickness,
            "close": close,
        }

        # Scoring system
        long_score = 0
        short_score = 0

        # Price position relative to cloud (strongest signal)
        if close > cloud_top:
            long_score += 2
        elif close < cloud_bottom:
            short_score += 2

        # Tenkan/Kijun relationship
        if tenkan > kijun:
            long_score += 1
            if bullish_cross:
                long_score += 1
        elif tenkan < kijun:
            short_score += 1
            if bearish_cross:
                short_score += 1

        # Chikou confirmation
        if chikou_bullish:
            long_score += 1
        elif chikou_bearish:
            short_score += 1

        # Cloud color (future trend)
        if span_a > span_b:  # Green cloud (bullish)
            long_score += 0.5
        else:  # Red cloud (bearish)
            short_score += 0.5

        # Determine signal
        max_score = 5.5
        if long_score > short_score and long_score >= 2.5:
            confidence = min(long_score / max_score, 1.0)
            stop_loss = max(kijun, cloud_bottom) * 0.99
            take_profit = close * (1 + self.config.take_profit_pct)

            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason=f"Ichimoku bullish: Price above cloud, TK cross up, score={long_score:.1f}",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        elif short_score > long_score and short_score >= 2.5:
            confidence = min(short_score / max_score, 1.0)
            stop_loss = min(kijun, cloud_top) * 1.01
            take_profit = close * (1 - self.config.take_profit_pct)

            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason=f"Ichimoku bearish: Price below cloud, TK cross down, score={short_score:.1f}",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        return self._flat_signal(
            f"No clear Ichimoku signal (long={long_score:.1f}, short={short_score:.1f})"
        )
