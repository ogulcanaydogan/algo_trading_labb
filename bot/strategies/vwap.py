"""
VWAP Deviation Trading Strategy.

Volume Weighted Average Price (VWAP) with standard deviation bands.
Useful for mean reversion and institutional trading levels.

Strategy:
- LONG when price is significantly below VWAP (oversold)
- SHORT when price is significantly above VWAP (overbought)
- Confirmation from volume and momentum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class VWAPConfig(StrategyConfig):
    """Configuration for VWAP strategy."""
    std_multiplier: float = 2.0  # Standard deviation bands
    lookback_periods: int = 50   # Rolling window for VWAP
    volume_threshold: float = 1.2  # Require above-average volume
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70


class VWAPStrategy(BaseStrategy):
    """
    VWAP Deviation trading strategy.

    Uses VWAP as the mean and trades deviations from it.
    Enhanced with RSI confirmation and volume filters.
    """

    def __init__(self, config: Optional[VWAPConfig] = None):
        self.vwap_config = config or VWAPConfig()
        super().__init__(self.vwap_config)

    @property
    def name(self) -> str:
        return "vwap_deviation"

    @property
    def description(self) -> str:
        return "VWAP deviation strategy with RSI confirmation for mean reversion"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["RANGING", "MEAN_REVERTING", "LOW_VOLATILITY", "FLAT"]

    def get_required_indicators(self) -> List[str]:
        return [
            "vwap",
            "vwap_upper",
            "vwap_lower",
            "vwap_deviation",
            "rsi",
            "volume_ratio",
        ]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP and related indicators."""
        df = ohlcv.copy()

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).rolling(
            window=self.vwap_config.lookback_periods
        ).sum()
        cumulative_vol = df["volume"].rolling(
            window=self.vwap_config.lookback_periods
        ).sum()
        df["vwap"] = cumulative_tp_vol / cumulative_vol

        # Standard deviation bands
        tp_squared = ((typical_price - df["vwap"]) ** 2 * df["volume"]).rolling(
            window=self.vwap_config.lookback_periods
        ).sum()
        variance = tp_squared / cumulative_vol
        std = np.sqrt(variance)

        df["vwap_std"] = std
        df["vwap_upper"] = df["vwap"] + (std * self.vwap_config.std_multiplier)
        df["vwap_lower"] = df["vwap"] - (std * self.vwap_config.std_multiplier)

        # VWAP deviation (z-score)
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / std

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.vwap_config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.vwap_config.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume ratio
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate VWAP deviation signal."""
        min_periods = max(self.vwap_config.lookback_periods, 20) + 5
        if len(ohlcv) < min_periods:
            return self._flat_signal("Insufficient data for VWAP")

        df = self.add_indicators(ohlcv)
        current = df.iloc[-1]

        # Get current values
        close = float(current["close"])
        vwap = float(current["vwap"]) if not pd.isna(current["vwap"]) else close
        vwap_upper = float(current["vwap_upper"]) if not pd.isna(current["vwap_upper"]) else close
        vwap_lower = float(current["vwap_lower"]) if not pd.isna(current["vwap_lower"]) else close
        vwap_deviation = float(current["vwap_deviation"]) if not pd.isna(current["vwap_deviation"]) else 0
        rsi = float(current["rsi"]) if not pd.isna(current["rsi"]) else 50
        volume_ratio = float(current["volume_ratio"]) if not pd.isna(current["volume_ratio"]) else 1.0

        indicators = {
            "vwap": vwap,
            "vwap_upper": vwap_upper,
            "vwap_lower": vwap_lower,
            "vwap_deviation": vwap_deviation,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "close": close,
        }

        # Check for extreme deviations
        is_below_lower = close < vwap_lower
        is_above_upper = close > vwap_upper
        strong_volume = volume_ratio >= self.vwap_config.volume_threshold

        # LONG signal: Price below lower band + RSI oversold
        if is_below_lower and rsi < self.vwap_config.rsi_oversold:
            # Confidence based on deviation and RSI
            deviation_score = min(abs(vwap_deviation) / 3, 1.0)
            rsi_score = (self.vwap_config.rsi_oversold - rsi) / self.vwap_config.rsi_oversold
            volume_bonus = 0.1 if strong_volume else 0

            confidence = min((deviation_score + rsi_score) / 2 + volume_bonus, 1.0)

            if confidence >= self.config.min_confidence:
                return StrategySignal(
                    decision="LONG",
                    confidence=confidence,
                    reason=f"VWAP oversold: deviation={vwap_deviation:.2f}, RSI={rsi:.1f}",
                    strategy_name=self.name,
                    entry_price=close,
                    stop_loss=close * (1 - self.config.stop_loss_pct),
                    take_profit=vwap,  # Target: Return to VWAP
                    indicators=indicators,
                )

        # SHORT signal: Price above upper band + RSI overbought
        elif is_above_upper and rsi > self.vwap_config.rsi_overbought:
            deviation_score = min(abs(vwap_deviation) / 3, 1.0)
            rsi_score = (rsi - self.vwap_config.rsi_overbought) / (100 - self.vwap_config.rsi_overbought)
            volume_bonus = 0.1 if strong_volume else 0

            confidence = min((deviation_score + rsi_score) / 2 + volume_bonus, 1.0)

            if confidence >= self.config.min_confidence:
                return StrategySignal(
                    decision="SHORT",
                    confidence=confidence,
                    reason=f"VWAP overbought: deviation={vwap_deviation:.2f}, RSI={rsi:.1f}",
                    strategy_name=self.name,
                    entry_price=close,
                    stop_loss=close * (1 + self.config.stop_loss_pct),
                    take_profit=vwap,  # Target: Return to VWAP
                    indicators=indicators,
                )

        return self._flat_signal(
            f"Price within VWAP bands (deviation={vwap_deviation:.2f})"
        )
