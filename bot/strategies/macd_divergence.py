"""
MACD Divergence Strategy.

Momentum strategy that detects price-MACD divergences for potential reversals.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator

from .base import BaseStrategy, StrategyConfig, StrategySignal


class MACDDivergenceConfig(StrategyConfig):
    """Configuration for MACD Divergence strategy."""

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    divergence_lookback: int = 20  # Bars to look for divergence
    min_divergence_bars: int = 3  # Minimum bars for valid divergence


class MACDDivergenceStrategy(BaseStrategy):
    """
    MACD Divergence Strategy.

    Detects divergences between price and MACD histogram:
    - Bullish divergence: Price makes lower low, MACD makes higher low
    - Bearish divergence: Price makes higher high, MACD makes lower high

    Also uses MACD crossovers as confirmation.

    Works best in: All regimes, particularly at trend exhaustion points.
    """

    def __init__(self, config: Optional[MACDDivergenceConfig] = None):
        super().__init__(config or MACDDivergenceConfig())
        self.config: MACDDivergenceConfig = self.config

    @property
    def name(self) -> str:
        return "macd_divergence"

    @property
    def description(self) -> str:
        return f"MACD({self.config.macd_fast}/{self.config.macd_slow}/{self.config.macd_signal}) divergence"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["all"]  # Divergences can signal reversals in any regime

    def get_required_indicators(self) -> List[str]:
        return ["macd", "macd_signal", "macd_hist", "rsi"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add MACD and RSI indicators."""
        df = ohlcv.copy()

        macd = MACD(
            df["close"],
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        df["rsi"] = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on MACD divergence."""
        if len(ohlcv) < self.config.macd_slow + self.config.divergence_lookback:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)
        last = df.iloc[-1]

        close = float(last["close"])
        macd = float(last["macd"])
        macd_signal = float(last["macd_signal"])
        macd_hist = float(last["macd_hist"])
        rsi = float(last["rsi"])

        indicators = {
            "close": close,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "rsi": rsi,
        }

        # Check for divergences
        bullish_div, bull_strength = self._detect_bullish_divergence(df)
        bearish_div, bear_strength = self._detect_bearish_divergence(df)

        # MACD crossover confirmation
        prev = df.iloc[-2]
        macd_cross_up = prev["macd"] <= prev["macd_signal"] and macd > macd_signal
        macd_cross_down = prev["macd"] >= prev["macd_signal"] and macd < macd_signal

        # Generate signals
        if bullish_div and (macd_cross_up or macd_hist > 0):
            confidence = self._calculate_confidence(bull_strength, rsi, is_long=True)
            stop_loss = close * (1 - self.config.stop_loss_pct)
            take_profit = close * (1 + self.config.take_profit_pct * 1.5)

            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason=f"Bullish MACD divergence detected",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        if bearish_div and (macd_cross_down or macd_hist < 0):
            confidence = self._calculate_confidence(bear_strength, rsi, is_long=False)
            stop_loss = close * (1 + self.config.stop_loss_pct)
            take_profit = close * (1 - self.config.take_profit_pct * 1.5)

            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason=f"Bearish MACD divergence detected",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # Simple MACD crossover (lower confidence)
        if macd_cross_up and rsi < 60:
            confidence = 0.4
            return StrategySignal(
                decision="LONG",
                confidence=confidence,
                reason="MACD bullish crossover",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=close * (1 - self.config.stop_loss_pct),
                take_profit=close * (1 + self.config.take_profit_pct),
                indicators=indicators,
            )

        if macd_cross_down and rsi > 40:
            confidence = 0.4
            return StrategySignal(
                decision="SHORT",
                confidence=confidence,
                reason="MACD bearish crossover",
                strategy_name=self.name,
                entry_price=close,
                stop_loss=close * (1 + self.config.stop_loss_pct),
                take_profit=close * (1 - self.config.take_profit_pct),
                indicators=indicators,
            )

        return self._flat_signal("No MACD signal")

    def _detect_bullish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detect bullish divergence (price lower low, MACD higher low).
        Returns (is_divergence, strength).
        """
        lookback = df.tail(self.config.divergence_lookback)
        prices = lookback["low"].values
        macd_hist = lookback["macd_hist"].values

        # Find local minima in price
        price_lows = self._find_local_minima(prices)
        macd_lows = self._find_local_minima(-macd_hist)  # Negate for minima

        if len(price_lows) < 2 or len(macd_lows) < 2:
            return False, 0.0

        # Check if price made lower low but MACD made higher low
        recent_price_low_idx = price_lows[-1]
        prev_price_low_idx = price_lows[-2]

        if prices[recent_price_low_idx] < prices[prev_price_low_idx]:
            # Price made lower low
            if macd_hist[recent_price_low_idx] > macd_hist[prev_price_low_idx]:
                # MACD made higher low = bullish divergence
                strength = abs(macd_hist[recent_price_low_idx] - macd_hist[prev_price_low_idx])
                return True, strength

        return False, 0.0

    def _detect_bearish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detect bearish divergence (price higher high, MACD lower high).
        Returns (is_divergence, strength).
        """
        lookback = df.tail(self.config.divergence_lookback)
        prices = lookback["high"].values
        macd_hist = lookback["macd_hist"].values

        # Find local maxima
        price_highs = self._find_local_maxima(prices)
        macd_highs = self._find_local_maxima(macd_hist)

        if len(price_highs) < 2 or len(macd_highs) < 2:
            return False, 0.0

        # Check if price made higher high but MACD made lower high
        recent_price_high_idx = price_highs[-1]
        prev_price_high_idx = price_highs[-2]

        if prices[recent_price_high_idx] > prices[prev_price_high_idx]:
            # Price made higher high
            if macd_hist[recent_price_high_idx] < macd_hist[prev_price_high_idx]:
                # MACD made lower high = bearish divergence
                strength = abs(macd_hist[prev_price_high_idx] - macd_hist[recent_price_high_idx])
                return True, strength

        return False, 0.0

    def _find_local_minima(self, arr: np.ndarray, order: int = 3) -> List[int]:
        """Find indices of local minima."""
        minima = []
        for i in range(order, len(arr) - order):
            if all(arr[i] <= arr[i - j] for j in range(1, order + 1)) and all(
                arr[i] <= arr[i + j] for j in range(1, order + 1)
            ):
                minima.append(i)
        return minima

    def _find_local_maxima(self, arr: np.ndarray, order: int = 3) -> List[int]:
        """Find indices of local maxima."""
        maxima = []
        for i in range(order, len(arr) - order):
            if all(arr[i] >= arr[i - j] for j in range(1, order + 1)) and all(
                arr[i] >= arr[i + j] for j in range(1, order + 1)
            ):
                maxima.append(i)
        return maxima

    def _calculate_confidence(self, divergence_strength: float, rsi: float, is_long: bool) -> float:
        """Calculate signal confidence."""
        # Divergence strength contribution (0-0.4)
        div_score = min(divergence_strength * 100, 0.4)

        # RSI confirmation (0-0.3)
        if is_long:
            rsi_score = max(0, (40 - rsi) / 100) * 0.3
        else:
            rsi_score = max(0, (rsi - 60) / 100) * 0.3

        return min(0.3 + div_score + rsi_score, 0.9)
