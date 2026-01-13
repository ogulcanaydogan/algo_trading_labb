"""
RSI Mean Reversion Strategy.

Counter-trend strategy that buys oversold and sells overbought conditions.
Best suited for range-bound/sideways markets.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator

from .base import BaseStrategy, StrategyConfig, StrategySignal


class RSIMeanReversionConfig(StrategyConfig):
    """Configuration for RSI Mean Reversion strategy."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_extreme_oversold: float = 20.0
    rsi_extreme_overbought: float = 80.0
    use_stochastic: bool = True  # Use stochastic as confirmation
    stoch_period: int = 14
    ema_filter_period: int = 50  # Optional trend filter


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Entry conditions:
    - LONG: RSI < oversold threshold (stronger signal at extreme)
    - SHORT: RSI > overbought threshold (stronger signal at extreme)

    Optional confirmations:
    - Stochastic oscillator agreement
    - Price near EMA (mean reversion target)

    Works best in: Sideways markets.
    """

    def __init__(self, config: Optional[RSIMeanReversionConfig] = None):
        super().__init__(config or RSIMeanReversionConfig())
        self.config: RSIMeanReversionConfig = self.config

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    @property
    def description(self) -> str:
        return f"RSI({self.config.rsi_period}) mean reversion with extremes at {self.config.rsi_oversold}/{self.config.rsi_overbought}"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["sideways"]

    def get_required_indicators(self) -> List[str]:
        indicators = ["rsi", "ema_50"]
        if self.config.use_stochastic:
            indicators.extend(["stoch_k", "stoch_d"])
        return indicators

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and optional indicators."""
        df = ohlcv.copy()

        df["rsi"] = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()
        df["ema_50"] = EMAIndicator(df["close"], window=self.config.ema_filter_period).ema_indicator()

        if self.config.use_stochastic:
            stoch = StochasticOscillator(
                df["high"], df["low"], df["close"],
                window=self.config.stoch_period, smooth_window=3
            )
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on RSI extremes."""
        if len(ohlcv) < self.config.ema_filter_period + 10:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        rsi = float(last["rsi"])
        prev_rsi = float(prev["rsi"])
        ema_50 = float(last["ema_50"])

        indicators = {
            "close": close,
            "rsi": rsi,
            "ema_50": ema_50,
        }

        stoch_confirms_long = True
        stoch_confirms_short = True

        if self.config.use_stochastic:
            stoch_k = float(last["stoch_k"])
            stoch_d = float(last["stoch_d"])
            indicators["stoch_k"] = stoch_k
            indicators["stoch_d"] = stoch_d

            stoch_confirms_long = stoch_k < 20 or stoch_k > stoch_d  # Oversold or turning up
            stoch_confirms_short = stoch_k > 80 or stoch_k < stoch_d  # Overbought or turning down

        # Calculate distance from mean (EMA)
        mean_distance_pct = (close - ema_50) / ema_50 * 100

        # LONG signal: RSI oversold
        if rsi <= self.config.rsi_oversold:
            # Check for RSI turning up (confirmation)
            rsi_turning_up = rsi > prev_rsi

            if rsi <= self.config.rsi_extreme_oversold:
                # Extreme oversold - high confidence
                confidence = 0.75
                reason = f"Extreme RSI oversold ({rsi:.1f})"
            else:
                confidence = 0.5
                reason = f"RSI oversold ({rsi:.1f})"

            # Adjust confidence based on confirmations
            if stoch_confirms_long:
                confidence += 0.1
                reason += " + Stoch confirms"
            if rsi_turning_up:
                confidence += 0.05
            if mean_distance_pct < -2:  # Price below mean
                confidence += 0.05

            confidence = min(confidence, 0.9)
            stop_loss = close * (1 - self.config.stop_loss_pct * 1.5)  # Wider stop for MR
            take_profit = ema_50  # Target the mean

            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # SHORT signal: RSI overbought
        if rsi >= self.config.rsi_overbought:
            rsi_turning_down = rsi < prev_rsi

            if rsi >= self.config.rsi_extreme_overbought:
                confidence = 0.75
                reason = f"Extreme RSI overbought ({rsi:.1f})"
            else:
                confidence = 0.5
                reason = f"RSI overbought ({rsi:.1f})"

            if stoch_confirms_short:
                confidence += 0.1
                reason += " + Stoch confirms"
            if rsi_turning_down:
                confidence += 0.05
            if mean_distance_pct > 2:
                confidence += 0.05

            confidence = min(confidence, 0.9)
            stop_loss = close * (1 + self.config.stop_loss_pct * 1.5)
            take_profit = ema_50

            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason=reason,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        return self._flat_signal(f"RSI neutral ({rsi:.1f})")
