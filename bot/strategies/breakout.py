"""
Breakout Trading Strategy.

Detects and trades price breakouts from consolidation patterns.

Types of breakouts detected:
- Support/Resistance level breakouts
- Donchian Channel breakouts
- Volatility expansion breakouts (ATR-based)
- Volume-confirmed breakouts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class BreakoutConfig(StrategyConfig):
    """Configuration for Breakout strategy."""
    lookback_period: int = 20  # Period for support/resistance
    atr_period: int = 14
    atr_multiplier: float = 1.5  # ATR expansion threshold
    volume_multiplier: float = 1.5  # Required volume for confirmation
    consolidation_threshold: float = 0.03  # Max range for consolidation (3%)
    min_consolidation_bars: int = 5  # Min bars in consolidation


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy.

    Identifies consolidation patterns and trades breakouts with
    volume and volatility confirmation.
    """

    def __init__(self, config: Optional[BreakoutConfig] = None):
        self.breakout_config = config or BreakoutConfig()
        super().__init__(self.breakout_config)

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def description(self) -> str:
        return "Breakout strategy trading support/resistance breaks with volume confirmation"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["RANGING", "CONSOLIDATION", "VOLATILE", "TRANSITION"]

    def get_required_indicators(self) -> List[str]:
        return [
            "resistance",
            "support",
            "donchian_high",
            "donchian_low",
            "atr",
            "atr_ratio",
            "volume_ratio",
            "in_consolidation",
        ]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add breakout indicators."""
        df = ohlcv.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        period = self.breakout_config.lookback_period

        # Donchian Channels
        df["donchian_high"] = high.rolling(window=period).max()
        df["donchian_low"] = low.rolling(window=period).min()
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2

        # Support and Resistance (using pivot points)
        df["resistance"] = high.rolling(window=period).max()
        df["support"] = low.rolling(window=period).min()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=self.breakout_config.atr_period).mean()

        # ATR expansion (current ATR vs average)
        df["atr_sma"] = df["atr"].rolling(window=50).mean()
        df["atr_ratio"] = df["atr"] / df["atr_sma"]

        # Volume ratio
        df["volume_sma"] = volume.rolling(window=20).mean()
        df["volume_ratio"] = volume / df["volume_sma"]

        # Consolidation detection
        range_pct = (df["donchian_high"] - df["donchian_low"]) / df["donchian_mid"]
        df["range_pct"] = range_pct

        # Check if in consolidation (tight range for N bars)
        threshold = self.breakout_config.consolidation_threshold
        df["tight_range"] = range_pct < threshold

        # Count consecutive tight range bars
        df["consolidation_bars"] = df["tight_range"].rolling(
            window=self.breakout_config.min_consolidation_bars
        ).sum()

        df["in_consolidation"] = (
            df["consolidation_bars"] >= self.breakout_config.min_consolidation_bars - 1
        )

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate breakout signal."""
        min_periods = max(self.breakout_config.lookback_period, 50) + 5
        if len(ohlcv) < min_periods:
            return self._flat_signal("Insufficient data for breakout detection")

        df = self.add_indicators(ohlcv)
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Get current values
        close = float(current["close"])
        prev_close = float(prev["close"])
        resistance = float(current["resistance"]) if not pd.isna(current["resistance"]) else close
        support = float(current["support"]) if not pd.isna(current["support"]) else close
        donchian_high = float(current["donchian_high"]) if not pd.isna(current["donchian_high"]) else close
        donchian_low = float(current["donchian_low"]) if not pd.isna(current["donchian_low"]) else close
        atr = float(current["atr"]) if not pd.isna(current["atr"]) else 0
        atr_ratio = float(current["atr_ratio"]) if not pd.isna(current["atr_ratio"]) else 1.0
        volume_ratio = float(current["volume_ratio"]) if not pd.isna(current["volume_ratio"]) else 1.0
        in_consolidation = bool(current["in_consolidation"]) if not pd.isna(current["in_consolidation"]) else False
        range_pct = float(current["range_pct"]) if not pd.isna(current["range_pct"]) else 0

        # Previous values for breakout detection
        prev_resistance = float(prev["resistance"]) if not pd.isna(prev["resistance"]) else resistance
        prev_support = float(prev["support"]) if not pd.isna(prev["support"]) else support

        indicators = {
            "resistance": resistance,
            "support": support,
            "donchian_high": donchian_high,
            "donchian_low": donchian_low,
            "atr": atr,
            "atr_ratio": atr_ratio,
            "volume_ratio": volume_ratio,
            "in_consolidation": in_consolidation,
            "range_pct": range_pct,
            "close": close,
        }

        # Volume confirmation
        volume_confirmed = volume_ratio >= self.breakout_config.volume_multiplier

        # ATR expansion (volatility confirmation)
        volatility_expanding = atr_ratio >= self.breakout_config.atr_multiplier

        # Bullish breakout: Close above resistance (previous high)
        bullish_breakout = close > prev_resistance and prev_close <= prev_resistance

        # Bearish breakout: Close below support (previous low)
        bearish_breakout = close < prev_support and prev_close >= prev_support

        # Alternative: Donchian channel breakout
        donchian_bull = close > df.iloc[-2]["donchian_high"] if len(df) > 1 else False
        donchian_bear = close < df.iloc[-2]["donchian_low"] if len(df) > 1 else False

        # Calculate confidence
        def calculate_confidence(volume_conf: bool, volatility_conf: bool, consolidation: bool) -> float:
            base_confidence = 0.4
            if volume_conf:
                base_confidence += 0.2
            if volatility_conf:
                base_confidence += 0.15
            if consolidation:
                base_confidence += 0.15  # Breakouts from consolidation are more reliable
            return min(base_confidence, 1.0)

        # LONG: Bullish breakout
        if bullish_breakout or donchian_bull:
            confidence = calculate_confidence(volume_confirmed, volatility_expanding, in_consolidation)

            if confidence >= self.config.min_confidence:
                stop_loss = max(support, close - atr * 2)
                take_profit = close + atr * 3

                reason = f"Bullish breakout above {resistance:.2f}"
                if volume_confirmed:
                    reason += " (volume confirmed)"
                if volatility_expanding:
                    reason += " (volatility expanding)"

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

        # SHORT: Bearish breakout
        elif bearish_breakout or donchian_bear:
            confidence = calculate_confidence(volume_confirmed, volatility_expanding, in_consolidation)

            if confidence >= self.config.min_confidence:
                stop_loss = min(resistance, close + atr * 2)
                take_profit = close - atr * 3

                reason = f"Bearish breakout below {support:.2f}"
                if volume_confirmed:
                    reason += " (volume confirmed)"
                if volatility_expanding:
                    reason += " (volatility expanding)"

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

        # No breakout detected
        return self._flat_signal(
            f"No breakout (price between {support:.2f} and {resistance:.2f})"
        )
