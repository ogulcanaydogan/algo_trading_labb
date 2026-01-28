"""
Z-Score Mean Reversion Strategy.

Statistical arbitrage strategy using z-score to identify extreme
price deviations from the rolling mean.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategySignal


class ZScoreReversionConfig(StrategyConfig):
    """Configuration for Z-Score Mean Reversion strategy."""

    lookback_period: int = 20  # Rolling window for mean/std
    entry_zscore: float = 2.0  # Enter when z-score exceeds this
    exit_zscore: float = 0.5  # Exit when z-score drops below this
    extreme_zscore: float = 2.5  # Higher confidence at extreme
    use_volume_filter: bool = True  # Filter by volume
    volume_threshold: float = 1.2  # Volume > 1.2x average


class ZScoreReversionStrategy(BaseStrategy):
    """
    Z-Score Mean Reversion Strategy.

    Uses statistical z-score to identify when price has deviated
    significantly from its rolling mean. Assumes prices revert to mean.

    Entry conditions:
    - LONG: Z-score < -entry_zscore (price significantly below mean)
    - SHORT: Z-score > entry_zscore (price significantly above mean)

    Exit conditions:
    - When z-score reverts toward 0

    Works best in: Range-bound markets, low volatility regimes.
    """

    def __init__(self, config: Optional[ZScoreReversionConfig] = None):
        super().__init__(config or ZScoreReversionConfig())
        self.config: ZScoreReversionConfig = self.config

    @property
    def name(self) -> str:
        return "zscore_reversion"

    @property
    def description(self) -> str:
        return f"Z-Score({self.config.lookback_period}) mean reversion at +/-{self.config.entry_zscore}"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["sideways", "low_volatility"]

    def get_required_indicators(self) -> List[str]:
        return ["zscore", "rolling_mean", "rolling_std", "volume_ratio"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add z-score and related indicators."""
        df = ohlcv.copy()

        # Calculate rolling statistics
        df["rolling_mean"] = df["close"].rolling(window=self.config.lookback_period).mean()
        df["rolling_std"] = df["close"].rolling(window=self.config.lookback_period).std()

        # Z-score
        df["zscore"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]

        # Volume analysis
        df["volume_ma"] = df["volume"].rolling(window=self.config.lookback_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Price momentum (rate of change)
        df["roc"] = df["close"].pct_change(periods=5) * 100

        # Bollinger Band position (0-1)
        df["bb_upper"] = df["rolling_mean"] + 2 * df["rolling_std"]
        df["bb_lower"] = df["rolling_mean"] - 2 * df["rolling_std"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on z-score extremes."""
        min_periods = self.config.lookback_period + 10
        if len(ohlcv) < min_periods:
            return self._flat_signal("Insufficient data")

        df = self.add_indicators(ohlcv)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        zscore = float(last["zscore"])
        prev_zscore = float(prev["zscore"])
        rolling_mean = float(last["rolling_mean"])
        rolling_std = float(last["rolling_std"])
        volume_ratio = float(last["volume_ratio"])
        roc = float(last["roc"])

        indicators = {
            "close": close,
            "zscore": zscore,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "volume_ratio": volume_ratio,
            "roc": roc,
        }

        # Volume filter
        volume_ok = not self.config.use_volume_filter or volume_ratio >= self.config.volume_threshold

        # LONG signal: Price significantly below mean
        if zscore <= -self.config.entry_zscore:
            # Check if z-score is turning up (mean reversion starting)
            zscore_turning = zscore > prev_zscore

            if zscore <= -self.config.extreme_zscore:
                confidence = 0.70
                reason = f"Extreme negative z-score ({zscore:.2f})"
            else:
                confidence = 0.55
                reason = f"Z-score oversold ({zscore:.2f})"

            # Adjust confidence
            if zscore_turning:
                confidence += 0.10
                reason += " + reversing"
            if volume_ok:
                confidence += 0.05
            if roc < -1:  # Price was falling - potential reversal
                confidence += 0.05

            confidence = min(confidence, 0.90)

            # Dynamic stop/target based on volatility
            stop_loss = close - 2.5 * rolling_std
            take_profit = rolling_mean  # Target the mean

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

        # SHORT signal: Price significantly above mean
        if zscore >= self.config.entry_zscore:
            zscore_turning = zscore < prev_zscore

            if zscore >= self.config.extreme_zscore:
                confidence = 0.70
                reason = f"Extreme positive z-score ({zscore:.2f})"
            else:
                confidence = 0.55
                reason = f"Z-score overbought ({zscore:.2f})"

            if zscore_turning:
                confidence += 0.10
                reason += " + reversing"
            if volume_ok:
                confidence += 0.05
            if roc > 1:
                confidence += 0.05

            confidence = min(confidence, 0.90)

            stop_loss = close + 2.5 * rolling_std
            take_profit = rolling_mean

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

        return self._flat_signal(f"Z-score neutral ({zscore:.2f})")

    def _flat_signal(self, reason: str) -> StrategySignal:
        """Return a flat/no-trade signal."""
        return StrategySignal(
            decision="FLAT",
            confidence=0.0,
            reason=reason,
            strategy_name=self.name,
            indicators={},
        )
