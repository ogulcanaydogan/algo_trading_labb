"""
EMA Crossover Strategy.

Trend-following strategy based on EMA crossovers with RSI confirmation.
Best suited for trending markets (bull/bear regimes).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from .base import BaseStrategy, StrategyConfig, StrategySignal


class EMACrossoverConfig(StrategyConfig):
    """Configuration for EMA Crossover strategy."""

    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend: int = 200  # Long-term trend filter
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    require_trend_alignment: bool = True  # Only trade with the trend


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover with RSI Confirmation.

    Entry conditions:
    - LONG: Fast EMA crosses above Slow EMA, RSI not overbought, price above trend EMA
    - SHORT: Fast EMA crosses below Slow EMA, RSI not oversold, price below trend EMA

    Works best in: Bull, Strong Bull, Bear, Strong Bear regimes.
    """

    def __init__(self, config: Optional[EMACrossoverConfig] = None):
        super().__init__(config or EMACrossoverConfig())
        self.config: EMACrossoverConfig = self.config

    @property
    def name(self) -> str:
        return "ema_crossover"

    @property
    def description(self) -> str:
        return f"EMA({self.config.ema_fast}/{self.config.ema_slow}) crossover with RSI confirmation"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["bull", "strong_bull", "bear", "strong_bear"]

    def get_required_indicators(self) -> List[str]:
        return ["ema_fast", "ema_slow", "ema_trend", "rsi"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add EMA and RSI indicators."""
        df = ohlcv.copy()

        df["ema_fast"] = EMAIndicator(df["close"], window=self.config.ema_fast).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=self.config.ema_slow).ema_indicator()
        df["ema_trend"] = EMAIndicator(df["close"], window=self.config.ema_trend).ema_indicator()
        df["rsi"] = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on EMA crossover."""
        if len(ohlcv) < self.config.ema_trend + 5:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Current values
        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        ema_trend = float(last["ema_trend"])
        rsi = float(last["rsi"])
        close = float(last["close"])

        # Previous values for crossover detection
        prev_ema_fast = float(prev["ema_fast"])
        prev_ema_slow = float(prev["ema_slow"])

        # Detect crossovers
        bullish_cross = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
        bearish_cross = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

        # Trend alignment
        above_trend = close > ema_trend
        below_trend = close < ema_trend

        indicators = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "rsi": rsi,
            "close": close,
        }

        # Generate signals
        if bullish_cross and rsi < self.config.rsi_overbought:
            if self.config.require_trend_alignment and not above_trend:
                return self._flat_signal("Bullish cross but below trend EMA")

            confidence = self._calculate_confidence(ema_fast, ema_slow, close, rsi, is_long=True)
            stop_loss = close * (1 - self.config.stop_loss_pct)
            take_profit = close * (1 + self.config.take_profit_pct)

            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason=f"Bullish EMA cross (RSI: {rsi:.1f})",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        elif bearish_cross and rsi > self.config.rsi_oversold:
            if self.config.require_trend_alignment and not below_trend:
                return self._flat_signal("Bearish cross but above trend EMA")

            confidence = self._calculate_confidence(ema_fast, ema_slow, close, rsi, is_long=False)
            stop_loss = close * (1 + self.config.stop_loss_pct)
            take_profit = close * (1 - self.config.take_profit_pct)

            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason=f"Bearish EMA cross (RSI: {rsi:.1f})",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        return self._flat_signal("No EMA crossover detected")

    def _calculate_confidence(
        self,
        ema_fast: float,
        ema_slow: float,
        close: float,
        rsi: float,
        is_long: bool,
    ) -> float:
        """Calculate signal confidence based on indicator strength."""
        # EMA gap strength (0-0.4)
        ema_gap = abs(ema_fast - ema_slow) / close
        ema_score = min(ema_gap * 200, 0.4)

        # RSI confirmation (0-0.3)
        if is_long:
            rsi_score = max(0, (70 - rsi) / 100) * 0.3
        else:
            rsi_score = max(0, (rsi - 30) / 100) * 0.3

        # Base confidence
        base = 0.3

        return min(base + ema_score + rsi_score, 1.0)
