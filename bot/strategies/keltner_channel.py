"""
Keltner Channel Breakout Strategy.

Uses ATR-based channels for trend-following breakout trading.
More responsive than Bollinger Bands in trending markets.
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
class KeltnerChannelConfig(StrategyConfig):
    """Configuration for Keltner Channel strategy."""

    ema_period: int = 20
    atr_period: int = 10
    atr_multiplier: float = 2.0
    use_squeeze_filter: bool = True
    squeeze_bb_period: int = 20
    squeeze_bb_std: float = 2.0
    trend_ema_period: int = 50
    volume_confirmation: bool = True
    volume_threshold: float = 1.2
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0


class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner Channel Breakout Strategy.

    Entry Logic:
    - Long: Close above upper Keltner channel (breakout)
    - Short: Close below lower Keltner channel (breakdown)
    - Optional squeeze filter: Wait for Bollinger inside Keltner (low volatility)
    - Volume confirmation: Requires above-average volume

    Exit Logic:
    - Price returns inside channel
    - ATR-based stop loss
    - Trailing stop using middle band (EMA)
    """

    def __init__(self, config: Optional[KeltnerChannelConfig] = None):
        super().__init__(config or KeltnerChannelConfig())
        self.kc_config = config or KeltnerChannelConfig()

    @property
    def name(self) -> str:
        return "keltner_channel"

    @property
    def description(self) -> str:
        return "ATR-based channel breakout strategy for trending markets"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["bull", "strong_bull", "bear", "strong_bear", "volatile"]

    def get_required_indicators(self) -> List[str]:
        return ["kc_upper", "kc_middle", "kc_lower", "atr", "trend_ema"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add Keltner Channel and supporting indicators."""
        df = ohlcv.copy()

        if ta is None:
            return df

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.kc_config.ema_period,
            window_atr=self.kc_config.atr_period,
            multiplier=self.kc_config.atr_multiplier,
        )
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_middle"] = kc.keltner_channel_mband()
        df["kc_lower"] = kc.keltner_channel_lband()

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.kc_config.atr_period,
        ).average_true_range()

        # Trend EMA
        df["trend_ema"] = ta.trend.EMAIndicator(
            close=df["close"],
            window=self.kc_config.trend_ema_period,
        ).ema_indicator()

        # Bollinger Bands for squeeze detection
        if self.kc_config.use_squeeze_filter:
            bb = ta.volatility.BollingerBands(
                close=df["close"],
                window=self.kc_config.squeeze_bb_period,
                window_dev=self.kc_config.squeeze_bb_std,
            )
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()

            # Squeeze: BB inside KC
            df["squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])

        # Volume
        if self.kc_config.volume_confirmation:
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Channel width for volatility assessment
        df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / df["kc_middle"] * 100

        return df

    def _is_squeeze_release(self, df: pd.DataFrame) -> bool:
        """Check if we just released from a squeeze."""
        if not self.kc_config.use_squeeze_filter:
            return True

        if "squeeze" not in df.columns or len(df) < 5:
            return True

        recent = df["squeeze"].iloc[-5:]
        # Was in squeeze recently and now released
        was_squeezed = recent.iloc[:-1].any()
        now_released = not recent.iloc[-1]

        return was_squeezed and now_released

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on Keltner Channel breakout."""
        df = self.add_indicators(ohlcv)

        if len(df) < max(self.kc_config.ema_period, self.kc_config.trend_ema_period) + 10:
            return self._flat_signal("Insufficient data")

        if "kc_upper" not in df.columns:
            return self._flat_signal("Keltner Channel not available")

        current_price = df["close"].iloc[-1]
        prev_price = df["close"].iloc[-2]
        kc_upper = df["kc_upper"].iloc[-1]
        kc_middle = df["kc_middle"].iloc[-1]
        kc_lower = df["kc_lower"].iloc[-1]
        atr = df["atr"].iloc[-1]
        trend_ema = df["trend_ema"].iloc[-1]
        kc_width = df["kc_width"].iloc[-1]

        # Volume check
        volume_ok = True
        volume_ratio = 1.0
        if self.kc_config.volume_confirmation and "volume_ratio" in df.columns:
            volume_ratio = df["volume_ratio"].iloc[-1]
            volume_ok = volume_ratio >= self.kc_config.volume_threshold

        # Squeeze release check
        squeeze_release = self._is_squeeze_release(df)

        indicators = {
            "kc_upper": kc_upper,
            "kc_middle": kc_middle,
            "kc_lower": kc_lower,
            "atr": atr,
            "trend_ema": trend_ema,
            "kc_width": kc_width,
            "volume_ratio": volume_ratio,
            "squeeze_release": 1.0 if squeeze_release else 0.0,
        }

        # Bullish breakout
        if current_price > kc_upper and prev_price <= kc_upper:
            # Confirm trend direction
            trend_aligned = current_price > trend_ema

            if not volume_ok and self.kc_config.volume_confirmation:
                return StrategySignal(
                    decision="FLAT",
                    confidence=0.3,
                    reason=f"Breakout above KC but low volume ({volume_ratio:.2f}x)",
                    strategy_name=self.name,
                    indicators=indicators,
                )

            confidence = 0.6
            if trend_aligned:
                confidence += 0.15
            if squeeze_release:
                confidence += 0.15
            if volume_ratio > 1.5:
                confidence += 0.1

            stop_loss = kc_middle - atr * self.kc_config.stop_loss_atr_mult
            take_profit = current_price + atr * self.kc_config.take_profit_atr_mult

            return StrategySignal(
                decision="LONG",
                confidence=min(0.95, confidence),
                reason=f"Keltner Channel breakout (width={kc_width:.1f}%)",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # Bearish breakdown
        if current_price < kc_lower and prev_price >= kc_lower:
            trend_aligned = current_price < trend_ema

            if not volume_ok and self.kc_config.volume_confirmation:
                return StrategySignal(
                    decision="FLAT",
                    confidence=0.3,
                    reason=f"Breakdown below KC but low volume ({volume_ratio:.2f}x)",
                    strategy_name=self.name,
                    indicators=indicators,
                )

            confidence = 0.6
            if trend_aligned:
                confidence += 0.15
            if squeeze_release:
                confidence += 0.15
            if volume_ratio > 1.5:
                confidence += 0.1

            stop_loss = kc_middle + atr * self.kc_config.stop_loss_atr_mult
            take_profit = current_price - atr * self.kc_config.take_profit_atr_mult

            return StrategySignal(
                decision="SHORT",
                confidence=min(0.95, confidence),
                reason=f"Keltner Channel breakdown (width={kc_width:.1f}%)",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        # Trend continuation: Price holding above middle band in uptrend
        if current_price > kc_middle and current_price < kc_upper:
            if current_price > trend_ema and prev_price < kc_middle:
                # Bounce off middle band
                stop_loss = kc_lower
                take_profit = kc_upper + atr

                return StrategySignal(
                    decision="LONG",
                    confidence=0.5,
                    reason="Bounce off Keltner middle band (uptrend)",
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        # Trend continuation: Price holding below middle band in downtrend
        if current_price < kc_middle and current_price > kc_lower:
            if current_price < trend_ema and prev_price > kc_middle:
                stop_loss = kc_upper
                take_profit = kc_lower - atr

                return StrategySignal(
                    decision="SHORT",
                    confidence=0.5,
                    reason="Rejection at Keltner middle band (downtrend)",
                    strategy_name=self.name,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators=indicators,
                )

        return StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason="No breakout or bounce signal",
            strategy_name=self.name,
            indicators=indicators,
        )
