"""
Momentum Strategy.

Trend-following strategy that captures strong directional moves
using multiple momentum indicators.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD

from .base import BaseStrategy, StrategyConfig, StrategySignal


class MomentumConfig(StrategyConfig):
    """Configuration for Momentum strategy."""

    rsi_period: int = 14
    rsi_momentum_threshold: float = 50.0  # RSI above/below this for momentum
    adx_period: int = 14
    adx_threshold: float = 25.0  # Minimum ADX for trend strength
    ema_fast: int = 12
    ema_slow: int = 26
    roc_period: int = 10  # Rate of change period
    min_roc: float = 2.0  # Minimum ROC % for entry


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.

    Follows strong trends using multiple momentum confirmations:
    - RSI directional bias (above/below 50)
    - ADX trend strength
    - EMA alignment
    - Rate of change

    Entry conditions:
    - LONG: RSI > 50, ADX > threshold, EMA fast > slow, positive ROC
    - SHORT: RSI < 50, ADX > threshold, EMA fast < slow, negative ROC

    Works best in: Trending markets, high volatility.
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        super().__init__(config or MomentumConfig())
        self.config: MomentumConfig = self.config

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def description(self) -> str:
        return f"Multi-indicator momentum with ADX>{self.config.adx_threshold}"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["bullish", "bearish", "high_volatility"]

    def get_required_indicators(self) -> List[str]:
        return ["rsi", "adx", "ema_fast", "ema_slow", "roc", "macd", "macd_signal"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = ohlcv.copy()

        # RSI
        df["rsi"] = RSIIndicator(df["close"], window=self.config.rsi_period).rsi()

        # ADX
        adx = ADXIndicator(
            df["high"], df["low"], df["close"], window=self.config.adx_period
        )
        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()

        # EMAs
        df["ema_fast"] = EMAIndicator(df["close"], window=self.config.ema_fast).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=self.config.ema_slow).ema_indicator()

        # Rate of Change
        df["roc"] = ROCIndicator(df["close"], window=self.config.roc_period).roc()

        # MACD
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on momentum indicators."""
        min_periods = max(self.config.ema_slow, self.config.adx_period) + 10
        if len(ohlcv) < min_periods:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        rsi = float(last["rsi"])
        adx = float(last["adx"])
        di_plus = float(last["di_plus"])
        di_minus = float(last["di_minus"])
        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        roc = float(last["roc"])
        macd = float(last["macd"])
        macd_signal = float(last["macd_signal"])
        macd_hist = float(last["macd_hist"])
        prev_macd_hist = float(prev["macd_hist"])

        indicators = {
            "close": close,
            "rsi": rsi,
            "adx": adx,
            "di_plus": di_plus,
            "di_minus": di_minus,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "roc": roc,
            "macd": macd,
            "macd_signal": macd_signal,
        }

        # Check trend strength
        has_trend = adx >= self.config.adx_threshold

        # Bullish momentum conditions
        bullish_rsi = rsi > self.config.rsi_momentum_threshold
        bullish_ema = ema_fast > ema_slow
        bullish_di = di_plus > di_minus
        bullish_roc = roc >= self.config.min_roc
        bullish_macd = macd > macd_signal and macd_hist > prev_macd_hist

        # Bearish momentum conditions
        bearish_rsi = rsi < self.config.rsi_momentum_threshold
        bearish_ema = ema_fast < ema_slow
        bearish_di = di_minus > di_plus
        bearish_roc = roc <= -self.config.min_roc
        bearish_macd = macd < macd_signal and macd_hist < prev_macd_hist

        # LONG signal
        if has_trend and bullish_ema:
            confirmations = sum([bullish_rsi, bullish_di, bullish_roc, bullish_macd])

            if confirmations >= 3:
                confidence = 0.60 + (confirmations - 3) * 0.10
                reason = f"Strong bullish momentum (ADX={adx:.1f}, {confirmations}/4 confirms)"
            elif confirmations >= 2:
                confidence = 0.50
                reason = f"Moderate bullish momentum ({confirmations}/4 confirms)"
            else:
                return self._flat_signal(f"Weak bullish momentum ({confirmations}/4)")

            # Higher confidence if MACD histogram expanding
            if macd_hist > prev_macd_hist > 0:
                confidence += 0.05
                reason += " + MACD expanding"

            confidence = min(confidence, 0.90)
            stop_loss = close * (1 - self.config.stop_loss_pct)
            take_profit = close * (1 + self.config.take_profit_pct)

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

        # SHORT signal
        if has_trend and bearish_ema:
            confirmations = sum([bearish_rsi, bearish_di, bearish_roc, bearish_macd])

            if confirmations >= 3:
                confidence = 0.60 + (confirmations - 3) * 0.10
                reason = f"Strong bearish momentum (ADX={adx:.1f}, {confirmations}/4 confirms)"
            elif confirmations >= 2:
                confidence = 0.50
                reason = f"Moderate bearish momentum ({confirmations}/4 confirms)"
            else:
                return self._flat_signal(f"Weak bearish momentum ({confirmations}/4)")

            if macd_hist < prev_macd_hist < 0:
                confidence += 0.05
                reason += " + MACD expanding"

            confidence = min(confidence, 0.90)
            stop_loss = close * (1 + self.config.stop_loss_pct)
            take_profit = close * (1 - self.config.take_profit_pct)

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

        return self._flat_signal(f"No momentum signal (ADX={adx:.1f})")

    def _flat_signal(self, reason: str) -> StrategySignal:
        """Return a flat/no-trade signal."""
        return StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason=reason,
            strategy_name=self.name,
            indicators={},
        )
