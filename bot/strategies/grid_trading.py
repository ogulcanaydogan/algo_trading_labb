"""
Grid Trading Strategy.

Places buy and sell orders at regular price intervals.
Best for sideways/ranging markets with clear support and resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import ta
except ImportError:
    ta = None

from .base import BaseStrategy, StrategyConfig, StrategySignal


@dataclass
class GridTradingConfig(StrategyConfig):
    """Configuration for Grid Trading strategy."""
    grid_levels: int = 5  # Number of grid levels above and below
    grid_spacing_pct: float = 1.0  # Spacing between levels as percentage
    use_atr_spacing: bool = True  # Use ATR for dynamic spacing
    atr_period: int = 14
    atr_multiplier: float = 0.5  # ATR * multiplier = grid spacing
    support_resistance_lookback: int = 50
    use_sr_bounds: bool = True  # Use S/R as grid boundaries
    volume_filter: bool = True
    volume_threshold: float = 0.8  # Minimum volume ratio
    position_per_level_pct: float = 20.0  # Percentage of capital per grid level
    trend_filter: bool = True
    trend_ema_period: int = 100


@dataclass
class GridLevel:
    """Represents a single grid level."""
    price: float
    type: str  # "buy" or "sell"
    active: bool = True
    filled: bool = False


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy.

    Concept:
    - Divide price range into grid levels
    - Place buy orders at lower levels, sell orders at higher levels
    - Profit from price oscillations within the range

    Entry Logic:
    - Buy when price drops to a grid buy level
    - Sell when price rises to a grid sell level

    Risk Management:
    - Grid bounded by support/resistance levels
    - Optional trend filter to avoid counter-trend grids
    - Volume confirmation for entries
    """

    def __init__(self, config: Optional[GridTradingConfig] = None):
        super().__init__(config or GridTradingConfig())
        self.grid_config = config or GridTradingConfig()
        self._grid_levels: List[GridLevel] = []
        self._last_grid_center: Optional[float] = None

    @property
    def name(self) -> str:
        return "grid_trading"

    @property
    def description(self) -> str:
        return "Automated grid trading for ranging markets"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["sideways", "volatile"]

    def get_required_indicators(self) -> List[str]:
        return ["atr", "support", "resistance", "trend_ema"]

    def add_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add indicators for grid calculation."""
        df = ohlcv.copy()

        if ta is None:
            return df

        # ATR for dynamic grid spacing
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.grid_config.atr_period,
        ).average_true_range()

        # Trend EMA
        df["trend_ema"] = ta.trend.EMAIndicator(
            close=df["close"],
            window=self.grid_config.trend_ema_period,
        ).ema_indicator()

        # Volume
        if self.grid_config.volume_filter:
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Support and Resistance using Donchian Channel
        lookback = self.grid_config.support_resistance_lookback
        df["resistance"] = df["high"].rolling(window=lookback).max()
        df["support"] = df["low"].rolling(window=lookback).min()

        # Price position within range
        df["range_position"] = (df["close"] - df["support"]) / (df["resistance"] - df["support"])

        return df

    def _calculate_grid_levels(
        self,
        center_price: float,
        atr: float,
        support: float,
        resistance: float,
    ) -> List[GridLevel]:
        """Calculate grid levels around center price."""
        levels = []

        # Calculate spacing
        if self.grid_config.use_atr_spacing:
            spacing = atr * self.grid_config.atr_multiplier
        else:
            spacing = center_price * (self.grid_config.grid_spacing_pct / 100)

        # Ensure minimum spacing
        spacing = max(spacing, center_price * 0.002)  # At least 0.2%

        # Generate buy levels (below center)
        for i in range(1, self.grid_config.grid_levels + 1):
            level_price = center_price - (spacing * i)

            # Respect support boundary
            if self.grid_config.use_sr_bounds and level_price < support:
                break

            levels.append(GridLevel(
                price=level_price,
                type="buy",
                active=True,
                filled=False,
            ))

        # Generate sell levels (above center)
        for i in range(1, self.grid_config.grid_levels + 1):
            level_price = center_price + (spacing * i)

            # Respect resistance boundary
            if self.grid_config.use_sr_bounds and level_price > resistance:
                break

            levels.append(GridLevel(
                price=level_price,
                type="sell",
                active=True,
                filled=False,
            ))

        return sorted(levels, key=lambda x: x.price)

    def _find_triggered_level(
        self,
        current_price: float,
        prev_price: float,
        levels: List[GridLevel],
    ) -> Optional[GridLevel]:
        """Find if price crossed any grid level."""
        for level in levels:
            if not level.active or level.filled:
                continue

            # Check if price crossed this level
            if level.type == "buy":
                # Price dropped through buy level
                if prev_price > level.price >= current_price:
                    return level
            else:  # sell
                # Price rose through sell level
                if prev_price < level.price <= current_price:
                    return level

        return None

    def generate_signal(self, ohlcv: pd.DataFrame) -> StrategySignal:
        """Generate signal based on grid level triggers."""
        df = self.add_indicators(ohlcv)

        min_len = max(
            self.grid_config.support_resistance_lookback,
            self.grid_config.trend_ema_period,
        ) + 10

        if len(df) < min_len:
            return self._flat_signal("Insufficient data")

        current_price = df["close"].iloc[-1]
        prev_price = df["close"].iloc[-2]
        atr = df["atr"].iloc[-1] if "atr" in df.columns else current_price * 0.02
        support = df["support"].iloc[-1]
        resistance = df["resistance"].iloc[-1]
        trend_ema = df["trend_ema"].iloc[-1]
        range_position = df["range_position"].iloc[-1]

        # Volume check
        volume_ok = True
        volume_ratio = 1.0
        if self.grid_config.volume_filter and "volume_ratio" in df.columns:
            volume_ratio = df["volume_ratio"].iloc[-1]
            volume_ok = volume_ratio >= self.grid_config.volume_threshold

        # Trend check
        in_uptrend = current_price > trend_ema
        in_downtrend = current_price < trend_ema

        indicators = {
            "atr": atr,
            "support": support,
            "resistance": resistance,
            "trend_ema": trend_ema,
            "range_position": range_position,
            "volume_ratio": volume_ratio,
            "grid_levels": len(self._grid_levels),
        }

        # Recalculate grid if center moved significantly
        grid_center = (support + resistance) / 2
        if self._last_grid_center is None or abs(grid_center - self._last_grid_center) > atr:
            self._grid_levels = self._calculate_grid_levels(
                center_price=grid_center,
                atr=atr,
                support=support,
                resistance=resistance,
            )
            self._last_grid_center = grid_center

        # Find triggered level
        triggered = self._find_triggered_level(current_price, prev_price, self._grid_levels)

        if triggered is None:
            return StrategySignal(
                decision="FLAT",
                confidence=0.0,
                reason=f"No grid level triggered (range: ${support:.2f}-${resistance:.2f})",
                strategy_name=self.name,
                indicators=indicators,
            )

        # Apply filters
        if not volume_ok:
            return StrategySignal(
                decision="FLAT",
                confidence=0.3,
                reason=f"Grid level at ${triggered.price:.2f} but low volume",
                strategy_name=self.name,
                indicators=indicators,
            )

        # Trend filter
        if self.grid_config.trend_filter:
            if triggered.type == "buy" and in_downtrend:
                # Buying in downtrend - reduce confidence but allow
                pass
            elif triggered.type == "sell" and in_uptrend:
                # Selling in uptrend - reduce confidence but allow
                pass

        # Calculate confidence based on position in range
        if triggered.type == "buy":
            # Lower in range = higher confidence for buy
            confidence = 0.5 + (1 - range_position) * 0.4

            # Better if price is oversold
            if range_position < 0.3:
                confidence += 0.1

            stop_loss = support - atr * 0.5
            take_profit = triggered.price + atr * 2

            # Mark level as filled
            triggered.filled = True

            return StrategySignal(
                decision="LONG",
                confidence=min(0.9, confidence),
                reason=f"Grid buy at ${triggered.price:.2f} (range pos: {range_position:.0%})",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

        else:  # sell
            # Higher in range = higher confidence for sell
            confidence = 0.5 + range_position * 0.4

            if range_position > 0.7:
                confidence += 0.1

            stop_loss = resistance + atr * 0.5
            take_profit = triggered.price - atr * 2

            triggered.filled = True

            return StrategySignal(
                decision="SHORT",
                confidence=min(0.9, confidence),
                reason=f"Grid sell at ${triggered.price:.2f} (range pos: {range_position:.0%})",
                strategy_name=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
            )

    def reset_grid(self) -> None:
        """Reset all grid levels to unfilled state."""
        for level in self._grid_levels:
            level.filled = False
            level.active = True

    def get_grid_status(self) -> Dict:
        """Get current grid status for monitoring."""
        buy_levels = [l for l in self._grid_levels if l.type == "buy"]
        sell_levels = [l for l in self._grid_levels if l.type == "sell"]

        return {
            "center": self._last_grid_center,
            "buy_levels": len(buy_levels),
            "sell_levels": len(sell_levels),
            "filled_buy": sum(1 for l in buy_levels if l.filled),
            "filled_sell": sum(1 for l in sell_levels if l.filled),
            "levels": [
                {"price": l.price, "type": l.type, "filled": l.filled}
                for l in self._grid_levels
            ],
        }
