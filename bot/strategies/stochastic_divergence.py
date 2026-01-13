"""
Stochastic Divergence Strategy.

Detects divergences between price and Stochastic oscillator for reversal signals.
Works best in ranging markets and at trend exhaustion points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ta
except ImportError:
    ta = None

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class StochasticDivergenceConfig(StrategyConfig):
    """Configuration for Stochastic Divergence strategy."""
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth: int = 3
    oversold: float = 20.0
    overbought: float = 80.0
    divergence_lookback: int = 20
    min_divergence_bars: int = 3
    use_ema_filter: bool = True
    ema_period: int = 50
    atr_period: int = 14
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0


class StochasticDivergenceStrategy(BaseStrategy):
    """
    Stochastic Divergence Strategy.

    Entry Logic:
    - Bullish Divergence: Price makes lower lows while Stochastic makes higher lows
    - Bearish Divergence: Price makes higher highs while Stochastic makes lower highs
    - Optional EMA filter: Only long above EMA, short below

    Exit Logic:
    - ATR-based stop loss and take profit
    - Exit when Stochastic crosses overbought/oversold
    """

    def __init__(self, config: Optional[StochasticDivergenceConfig] = None):
        super().__init__(config or StochasticDivergenceConfig())
        self.stoch_config = config or StochasticDivergenceConfig()

    @property
    def name(self) -> str:
        return "stochastic_divergence"

    @property
    def description(self) -> str:
        return "Detects price-Stochastic divergences for reversal trading"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["sideways", "volatile", "bull", "bear"]

    def get_required_indicators(self) -> List[str]:
        return ["stoch_k", "stoch_d", "ema", "atr"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic and supporting indicators."""
        df = ohlcv.copy()

        if ta is None:
            return df

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.stoch_config.stoch_k_period,
            smooth_window=self.stoch_config.stoch_smooth,
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # EMA for trend filter
        df["ema"] = ta.trend.EMAIndicator(
            close=df["close"],
            window=self.stoch_config.ema_period,
        ).ema_indicator()

        # ATR for stop loss / take profit
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.stoch_config.atr_period,
        ).average_true_range()

        return df

    def _find_local_extremes(self, series: pd.Series, lookback: int) -> tuple:
        """Find local minima and maxima indices."""
        minima = []
        maxima = []

        for i in range(lookback, len(series) - 1):
            window = series.iloc[i - lookback:i + 1]
            if series.iloc[i] == window.min():
                minima.append(i)
            if series.iloc[i] == window.max():
                maxima.append(i)

        return minima, maxima

    def _detect_bullish_divergence(
        self,
        price: pd.Series,
        stoch: pd.Series,
        lookback: int,
    ) -> tuple[bool, float]:
        """
        Detect bullish divergence (price lower low, stoch higher low).

        Returns:
            (is_divergence, confidence)
        """
        if len(price) < lookback * 2:
            return False, 0.0

        recent = price.iloc[-lookback:]
        recent_stoch = stoch.iloc[-lookback:]

        # Find recent lows
        price_lows_idx, _ = self._find_local_extremes(recent, 5)
        stoch_lows_idx, _ = self._find_local_extremes(recent_stoch, 5)

        if len(price_lows_idx) < 2 or len(stoch_lows_idx) < 2:
            return False, 0.0

        # Check last two lows
        p_idx1, p_idx2 = price_lows_idx[-2], price_lows_idx[-1]
        s_idx1, s_idx2 = stoch_lows_idx[-2], stoch_lows_idx[-1]

        # Price making lower low
        price_lower = recent.iloc[p_idx2] < recent.iloc[p_idx1]
        # Stochastic making higher low
        stoch_higher = recent_stoch.iloc[s_idx2] > recent_stoch.iloc[s_idx1]

        if price_lower and stoch_higher:
            # Calculate divergence strength
            price_diff = abs(recent.iloc[p_idx1] - recent.iloc[p_idx2]) / recent.iloc[p_idx1]
            stoch_diff = abs(recent_stoch.iloc[s_idx2] - recent_stoch.iloc[s_idx1])
            confidence = min(0.9, 0.5 + price_diff * 10 + stoch_diff / 100)
            return True, confidence

        return False, 0.0

    def _detect_bearish_divergence(
        self,
        price: pd.Series,
        stoch: pd.Series,
        lookback: int,
    ) -> tuple[bool, float]:
        """
        Detect bearish divergence (price higher high, stoch lower high).

        Returns:
            (is_divergence, confidence)
        """
        if len(price) < lookback * 2:
            return False, 0.0

        recent = price.iloc[-lookback:]
        recent_stoch = stoch.iloc[-lookback:]

        # Find recent highs
        _, price_highs_idx = self._find_local_extremes(recent, 5)
        _, stoch_highs_idx = self._find_local_extremes(recent_stoch, 5)

        if len(price_highs_idx) < 2 or len(stoch_highs_idx) < 2:
            return False, 0.0

        # Check last two highs
        p_idx1, p_idx2 = price_highs_idx[-2], price_highs_idx[-1]
        s_idx1, s_idx2 = stoch_highs_idx[-2], stoch_highs_idx[-1]

        # Price making higher high
        price_higher = recent.iloc[p_idx2] > recent.iloc[p_idx1]
        # Stochastic making lower high
        stoch_lower = recent_stoch.iloc[s_idx2] < recent_stoch.iloc[s_idx1]

        if price_higher and stoch_lower:
            price_diff = abs(recent.iloc[p_idx2] - recent.iloc[p_idx1]) / recent.iloc[p_idx1]
            stoch_diff = abs(recent_stoch.iloc[s_idx1] - recent_stoch.iloc[s_idx2])
            confidence = min(0.9, 0.5 + price_diff * 10 + stoch_diff / 100)
            return True, confidence

        return False, 0.0

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on Stochastic divergence."""
        df = self.add_indicators(ohlcv)

        if len(df) < self.stoch_config.divergence_lookback * 2:
            return self._flat_signal("Insufficient data")

        if "stoch_k" not in df.columns:
            return self._flat_signal("Stochastic indicator not available")

        current_price = df["close"].iloc[-1]
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_d = df["stoch_d"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df.columns else current_price * 0.02
        ema = df["ema"].iloc[-1] if "ema" in df.columns else current_price

        lookback = self.stoch_config.divergence_lookback

        # Detect divergences
        bullish_div, bull_conf = self._detect_bullish_divergence(
            df["close"], df["stoch_k"], lookback
        )
        bearish_div, bear_conf = self._detect_bearish_divergence(
            df["close"], df["stoch_k"], lookback
        )

        indicators = {
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "ema": ema,
            "atr": atr,
            "bullish_divergence": 1.0 if bullish_div else 0.0,
            "bearish_divergence": 1.0 if bearish_div else 0.0,
        }

        # Bullish signal
        if bullish_div and stoch_k < self.stoch_config.oversold + 10:
            # Check EMA filter if enabled
            if self.stoch_config.use_ema_filter and current_price < ema * 0.98:
                # Below EMA but allow if strong divergence
                bull_conf *= 0.7

            # Stochastic crossover confirmation
            if stoch_k > stoch_d:
                bull_conf += 0.1

            stop_loss = current_price - atr * self.stoch_config.stop_loss_atr_mult
            take_profit = current_price + atr * self.stoch_config.take_profit_atr_mult

            return StrategySignal(
                decision="LONG",
                confidence=min(0.95, bull_conf),
                reason=f"Bullish divergence detected, Stoch K={stoch_k:.1f}",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # Bearish signal
        if bearish_div and stoch_k > self.stoch_config.overbought - 10:
            if self.stoch_config.use_ema_filter and current_price > ema * 1.02:
                bear_conf *= 0.7

            if stoch_k < stoch_d:
                bear_conf += 0.1

            stop_loss = current_price + atr * self.stoch_config.stop_loss_atr_mult
            take_profit = current_price - atr * self.stoch_config.take_profit_atr_mult

            return StrategySignal(
                decision="SHORT",
                confidence=min(0.95, bear_conf),
                reason=f"Bearish divergence detected, Stoch K={stoch_k:.1f}",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # Check for simple oversold/overbought with crossover
        if stoch_k < self.stoch_config.oversold and stoch_k > stoch_d:
            if not self.stoch_config.use_ema_filter or current_price > ema:
                stop_loss = current_price - atr * self.stoch_config.stop_loss_atr_mult
                take_profit = current_price + atr * self.stoch_config.take_profit_atr_mult

                return StrategySignal(
                    decision="LONG",
                    confidence=0.5,
                    reason=f"Stochastic oversold crossover, K={stoch_k:.1f}",
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        if stoch_k > self.stoch_config.overbought and stoch_k < stoch_d:
            if not self.stoch_config.use_ema_filter or current_price < ema:
                stop_loss = current_price + atr * self.stoch_config.stop_loss_atr_mult
                take_profit = current_price - atr * self.stoch_config.take_profit_atr_mult

                return StrategySignal(
                    decision="SHORT",
                    confidence=0.5,
                    reason=f"Stochastic overbought crossover, K={stoch_k:.1f}",
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        return StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason="No divergence or crossover signal",
            strategy_name=self.name,
            indicators=indicators,
        )
