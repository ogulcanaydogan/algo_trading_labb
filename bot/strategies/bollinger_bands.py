"""
Bollinger Band Strategy.

Mean-reversion and breakout strategy using Bollinger Bands.
Can work in both trending and range-bound markets depending on configuration.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from dataclasses import dataclass

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class BollingerBandConfig(StrategyConfig):
    """Configuration for Bollinger Band strategy."""

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    mode: str = "mean_reversion"  # "mean_reversion" or "breakout"
    squeeze_threshold: float = 0.02  # BB width below this = squeeze


class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band Strategy.

    Mean Reversion Mode (for sideways markets):
    - LONG: Price touches lower band, RSI oversold
    - SHORT: Price touches upper band, RSI overbought

    Breakout Mode (for volatile markets):
    - LONG: Price breaks above upper band after squeeze
    - SHORT: Price breaks below lower band after squeeze

    Works best in: Sideways (mean reversion), Volatile (breakout).
    """

    def __init__(self, config: Optional[BollingerBandConfig] = None):
        super().__init__(config or BollingerBandConfig())
        self.config: BollingerBandConfig = self.config

    @property
    def name(self) -> str:
        return f"bollinger_bands_{self.config.mode}"

    @property
    def description(self) -> str:
        return (
            f"Bollinger Bands ({self.config.bb_period}, {self.config.bb_std}) - {self.config.mode}"
        )

    @property
    def suitable_regimes(self) -> List[str]:
        if self.config.mode == "mean_reversion":
            return ["sideways"]
        return ["volatile", "strong_bull", "strong_bear"]

    def get_required_indicators(self) -> List[str]:
        return ["bb_high", "bb_low", "bb_mid", "bb_width", "rsi"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands and RSI."""
        df = ohlcv.copy()

        bb = BollingerBands(
            df["close"], window=self.config.bb_period, window_dev=self.config.bb_std
        )
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

        # Percent B (position within bands)
        df["bb_pctb"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

        df["rsi"] = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on Bollinger Bands."""
        if len(ohlcv) < self.config.bb_period + 10:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)

        if self.config.mode == "mean_reversion":
            return self._mean_reversion_signal(df)
        else:
            return self._breakout_signal(df)

    def _mean_reversion_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Generate mean-reversion signals (buy low, sell high)."""
        last = df.iloc[-1]
        close = float(last["close"])
        bb_high = float(last["bb_high"])
        bb_low = float(last["bb_low"])
        bb_mid = float(last["bb_mid"])
        bb_pctb = float(last["bb_pctb"])
        rsi = float(last["rsi"])

        indicators = {
            "close": close,
            "bb_high": bb_high,
            "bb_low": bb_low,
            "bb_mid": bb_mid,
            "bb_pctb": bb_pctb,
            "rsi": rsi,
        }

        # LONG: Price at/below lower band
        if close <= bb_low or bb_pctb < 0.05:
            if rsi < 35:  # RSI confirms oversold
                confidence = self._calculate_mr_confidence(bb_pctb, rsi, is_long=True)
                stop_loss = close * (1 - self.config.stop_loss_pct * 1.5)  # Wider stop for MR
                take_profit = bb_mid  # Target middle band

                return StrategySignal(
                    decision="LONG",
                    confidence=confidence,
                    reason=f"Price at lower BB (RSI: {rsi:.1f})",
                    strategy_name=self.name,
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        # SHORT: Price at/above upper band
        if close >= bb_high or bb_pctb > 0.95:
            if rsi > 65:  # RSI confirms overbought
                confidence = self._calculate_mr_confidence(bb_pctb, rsi, is_long=False)
                stop_loss = close * (1 + self.config.stop_loss_pct * 1.5)
                take_profit = bb_mid

                return StrategySignal(
                    decision="SHORT",
                    confidence=confidence,
                    reason=f"Price at upper BB (RSI: {rsi:.1f})",
                    strategy_name=self.name,
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        return self._flat_signal("Price within Bollinger Bands")

    def _breakout_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Generate breakout signals after squeeze."""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        bb_high = float(last["bb_high"])
        bb_low = float(last["bb_low"])
        bb_width = float(last["bb_width"])
        prev_bb_width = float(prev["bb_width"])

        indicators = {
            "close": close,
            "bb_high": bb_high,
            "bb_low": bb_low,
            "bb_width": bb_width,
        }

        # Check for squeeze (narrow bands)
        was_squeezed = prev_bb_width < self.config.squeeze_threshold
        expanding = bb_width > prev_bb_width * 1.1

        # LONG breakout: Close above upper band after squeeze
        if close > bb_high and was_squeezed and expanding:
            confidence = min(0.8, 0.5 + (bb_width - prev_bb_width) * 10)
            stop_loss = bb_low  # Stop at lower band
            take_profit = close * (1 + self.config.take_profit_pct * 2)  # Extended target

            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason=f"Bullish BB breakout after squeeze",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # SHORT breakout: Close below lower band after squeeze
        if close < bb_low and was_squeezed and expanding:
            confidence = min(0.8, 0.5 + (bb_width - prev_bb_width) * 10)
            stop_loss = bb_high
            take_profit = close * (1 - self.config.take_profit_pct * 2)

            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason=f"Bearish BB breakout after squeeze",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        return self._flat_signal("No breakout conditions met")

    def _calculate_mr_confidence(self, bb_pctb: float, rsi: float, is_long: bool) -> float:
        """Calculate confidence for mean-reversion signals."""
        # Distance from band edge (0-0.4)
        if is_long:
            band_score = max(0, (0.1 - bb_pctb) * 4)
            rsi_score = max(0, (30 - rsi) / 100) * 0.3
        else:
            band_score = max(0, (bb_pctb - 0.9) * 4)
            rsi_score = max(0, (rsi - 70) / 100) * 0.3

        return min(0.3 + band_score + rsi_score, 0.9)
