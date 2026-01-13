"""
Advanced Position Management Module.

Features:
- Trailing stops (percentage and ATR-based)
- Partial position exits (scale out)
- Break-even stops
- Time-based exits
- Multiple take-profit targets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class StopType(Enum):
    """Types of stop loss mechanisms."""
    FIXED = "fixed"
    TRAILING_PERCENT = "trailing_percent"
    TRAILING_ATR = "trailing_atr"
    BREAKEVEN = "breakeven"
    TIME_BASED = "time_based"


class ExitReason(Enum):
    """Reasons for position exit."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL = "manual"
    PARTIAL_TP1 = "partial_tp1"
    PARTIAL_TP2 = "partial_tp2"
    PARTIAL_TP3 = "partial_tp3"


@dataclass
class TakeProfitLevel:
    """A single take-profit level."""
    price: float
    exit_percent: float  # Percentage of position to exit (0.0-1.0)
    triggered: bool = False


@dataclass
class PositionConfig:
    """Configuration for position management."""
    # Stop loss settings
    initial_stop_pct: float = 0.02  # 2% initial stop
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5% trailing
    trailing_activation_pct: float = 0.01  # Activate after 1% profit
    use_atr_trailing: bool = False
    atr_multiplier: float = 2.0

    # Break-even settings
    use_breakeven: bool = True
    breakeven_trigger_pct: float = 0.01  # Move to BE after 1% profit
    breakeven_offset_pct: float = 0.001  # Small buffer above entry

    # Take profit settings
    use_multiple_tp: bool = True
    tp1_pct: float = 0.02  # 2% - exit 33%
    tp1_exit_pct: float = 0.33
    tp2_pct: float = 0.04  # 4% - exit 33%
    tp2_exit_pct: float = 0.33
    tp3_pct: float = 0.06  # 6% - exit remaining
    tp3_exit_pct: float = 0.34

    # Time-based settings
    use_time_exit: bool = False
    max_hold_hours: int = 48

    # Scaling settings
    allow_scale_in: bool = False
    max_scale_ins: int = 2
    scale_in_threshold_pct: float = -0.01  # Scale in if down 1%


@dataclass
class Position:
    """Represents an open position with advanced tracking."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    initial_size: float
    current_size: float

    # Stop loss tracking
    initial_stop: float
    current_stop: float
    stop_type: StopType = StopType.FIXED
    highest_price: float = 0.0  # For trailing (LONG)
    lowest_price: float = float('inf')  # For trailing (SHORT)

    # Take profit tracking
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)

    # Status tracking
    is_breakeven: bool = False
    trailing_activated: bool = False
    scale_in_count: int = 0
    partial_exits: List[Dict] = field(default_factory=list)

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if self.direction == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.current_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.current_size

    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Get unrealized P&L as percentage."""
        if self.direction == "LONG":
            return (current_price / self.entry_price - 1) * 100
        else:
            return (self.entry_price / current_price - 1) * 100

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "initial_size": self.initial_size,
            "current_size": self.current_size,
            "current_stop": self.current_stop,
            "stop_type": self.stop_type.value,
            "is_breakeven": self.is_breakeven,
            "trailing_activated": self.trailing_activated,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "partial_exits": self.partial_exits,
        }


class PositionManager:
    """
    Advanced Position Manager.

    Handles:
    - Dynamic stop loss management (trailing, ATR-based)
    - Multiple take-profit targets with partial exits
    - Break-even stops
    - Position scaling (in/out)
    - Time-based exits
    """

    def __init__(self, config: Optional[PositionConfig] = None):
        self.config = config or PositionConfig()
        self.positions: Dict[str, Position] = {}
        self._atr_cache: Dict[str, float] = {}

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        atr: Optional[float] = None,
    ) -> Position:
        """
        Open a new position with proper stop loss and take profit levels.

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            entry_price: Entry price
            size: Position size
            atr: Current ATR value (optional, for ATR-based stops)

        Returns:
            Position object
        """
        # Calculate initial stop loss
        if direction == "LONG":
            initial_stop = entry_price * (1 - self.config.initial_stop_pct)
            if self.config.use_atr_trailing and atr:
                initial_stop = entry_price - (atr * self.config.atr_multiplier)
        else:
            initial_stop = entry_price * (1 + self.config.initial_stop_pct)
            if self.config.use_atr_trailing and atr:
                initial_stop = entry_price + (atr * self.config.atr_multiplier)

        # Create take profit levels
        tp_levels = []
        if self.config.use_multiple_tp:
            if direction == "LONG":
                tp_levels = [
                    TakeProfitLevel(entry_price * (1 + self.config.tp1_pct), self.config.tp1_exit_pct),
                    TakeProfitLevel(entry_price * (1 + self.config.tp2_pct), self.config.tp2_exit_pct),
                    TakeProfitLevel(entry_price * (1 + self.config.tp3_pct), self.config.tp3_exit_pct),
                ]
            else:
                tp_levels = [
                    TakeProfitLevel(entry_price * (1 - self.config.tp1_pct), self.config.tp1_exit_pct),
                    TakeProfitLevel(entry_price * (1 - self.config.tp2_pct), self.config.tp2_exit_pct),
                    TakeProfitLevel(entry_price * (1 - self.config.tp3_pct), self.config.tp3_exit_pct),
                ]

        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            initial_size=size,
            current_size=size,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            take_profit_levels=tp_levels,
        )

        self.positions[symbol] = position
        if atr:
            self._atr_cache[symbol] = atr

        return position

    def update(
        self,
        symbol: str,
        current_price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> Tuple[Optional[ExitReason], float, float]:
        """
        Update position and check for exit conditions.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            high: Current bar high (for accurate stop checking)
            low: Current bar low (for accurate stop checking)
            atr: Current ATR (for dynamic trailing)

        Returns:
            Tuple of (exit_reason, exit_price, exit_size) or (None, 0, 0)
        """
        if symbol not in self.positions:
            return None, 0.0, 0.0

        position = self.positions[symbol]
        high = high or current_price
        low = low or current_price

        if atr:
            self._atr_cache[symbol] = atr

        # Update price tracking
        position.highest_price = max(position.highest_price, high)
        position.lowest_price = min(position.lowest_price, low)
        position.update_unrealized_pnl(current_price)

        # Check exit conditions in order of priority

        # 1. Check stop loss hit
        exit_reason, exit_price = self._check_stop_loss(position, high, low)
        if exit_reason:
            return exit_reason, exit_price, position.current_size

        # 2. Check take profit levels (partial exits)
        exit_reason, exit_price, exit_size = self._check_take_profits(position, current_price)
        if exit_reason:
            return exit_reason, exit_price, exit_size

        # 3. Check time-based exit
        if self.config.use_time_exit:
            exit_reason = self._check_time_exit(position)
            if exit_reason:
                return exit_reason, current_price, position.current_size

        # 4. Update trailing stop
        self._update_trailing_stop(position, current_price)

        # 5. Check and apply break-even
        self._update_breakeven(position, current_price)

        return None, 0.0, 0.0

    def _check_stop_loss(
        self,
        position: Position,
        high: float,
        low: float,
    ) -> Tuple[Optional[ExitReason], float]:
        """Check if stop loss was hit."""
        if position.direction == "LONG":
            if low <= position.current_stop:
                exit_reason = (
                    ExitReason.TRAILING_STOP if position.trailing_activated
                    else ExitReason.BREAKEVEN_STOP if position.is_breakeven
                    else ExitReason.STOP_LOSS
                )
                return exit_reason, position.current_stop
        else:  # SHORT
            if high >= position.current_stop:
                exit_reason = (
                    ExitReason.TRAILING_STOP if position.trailing_activated
                    else ExitReason.BREAKEVEN_STOP if position.is_breakeven
                    else ExitReason.STOP_LOSS
                )
                return exit_reason, position.current_stop

        return None, 0.0

    def _check_take_profits(
        self,
        position: Position,
        current_price: float,
    ) -> Tuple[Optional[ExitReason], float, float]:
        """Check take profit levels and handle partial exits."""
        for i, tp in enumerate(position.take_profit_levels):
            if tp.triggered:
                continue

            hit = False
            if position.direction == "LONG" and current_price >= tp.price:
                hit = True
            elif position.direction == "SHORT" and current_price <= tp.price:
                hit = True

            if hit:
                tp.triggered = True
                exit_size = position.current_size * tp.exit_percent

                # Record partial exit
                position.partial_exits.append({
                    "price": tp.price,
                    "size": exit_size,
                    "level": i + 1,
                    "timestamp": datetime.now().isoformat(),
                })

                # Update position size
                position.current_size -= exit_size

                # Calculate realized P&L for this exit
                if position.direction == "LONG":
                    pnl = (tp.price - position.entry_price) * exit_size
                else:
                    pnl = (position.entry_price - tp.price) * exit_size
                position.realized_pnl += pnl

                exit_reasons = [ExitReason.PARTIAL_TP1, ExitReason.PARTIAL_TP2, ExitReason.PARTIAL_TP3]
                return exit_reasons[i], tp.price, exit_size

        return None, 0.0, 0.0

    def _check_time_exit(self, position: Position) -> Optional[ExitReason]:
        """Check if position has exceeded max hold time."""
        hold_duration = datetime.now() - position.entry_time
        if hold_duration > timedelta(hours=self.config.max_hold_hours):
            return ExitReason.TIME_EXIT
        return None

    def _update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop based on price movement."""
        if not self.config.use_trailing_stop:
            return

        pnl_pct = position.get_unrealized_pnl_pct(current_price) / 100

        # Check if trailing should be activated
        if pnl_pct >= self.config.trailing_activation_pct:
            position.trailing_activated = True
            position.stop_type = StopType.TRAILING_PERCENT

        if not position.trailing_activated:
            return

        # Calculate new trailing stop
        if self.config.use_atr_trailing and position.symbol in self._atr_cache:
            atr = self._atr_cache[position.symbol]
            if position.direction == "LONG":
                new_stop = position.highest_price - (atr * self.config.atr_multiplier)
            else:
                new_stop = position.lowest_price + (atr * self.config.atr_multiplier)
            position.stop_type = StopType.TRAILING_ATR
        else:
            if position.direction == "LONG":
                new_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
            else:
                new_stop = position.lowest_price * (1 + self.config.trailing_stop_pct)

        # Only move stop in favorable direction
        if position.direction == "LONG":
            position.current_stop = max(position.current_stop, new_stop)
        else:
            position.current_stop = min(position.current_stop, new_stop)

    def _update_breakeven(self, position: Position, current_price: float):
        """Move stop to break-even if conditions met."""
        if not self.config.use_breakeven or position.is_breakeven:
            return

        pnl_pct = position.get_unrealized_pnl_pct(current_price) / 100

        if pnl_pct >= self.config.breakeven_trigger_pct:
            position.is_breakeven = True
            position.stop_type = StopType.BREAKEVEN

            # Set stop to entry + small buffer
            if position.direction == "LONG":
                be_stop = position.entry_price * (1 + self.config.breakeven_offset_pct)
                position.current_stop = max(position.current_stop, be_stop)
            else:
                be_stop = position.entry_price * (1 - self.config.breakeven_offset_pct)
                position.current_stop = min(position.current_stop, be_stop)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: ExitReason = ExitReason.MANUAL,
    ) -> Optional[Dict]:
        """
        Fully close a position.

        Returns:
            Dictionary with position summary or None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate final P&L
        if position.direction == "LONG":
            final_pnl = (exit_price - position.entry_price) * position.current_size
        else:
            final_pnl = (position.entry_price - exit_price) * position.current_size

        position.realized_pnl += final_pnl

        result = {
            "symbol": symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "initial_size": position.initial_size,
            "final_size": position.current_size,
            "realized_pnl": round(position.realized_pnl, 2),
            "exit_reason": reason.value,
            "partial_exits": position.partial_exits,
            "was_breakeven": position.is_breakeven,
            "trailing_activated": position.trailing_activated,
        }

        del self.positions[symbol]
        return result

    def scale_in(
        self,
        symbol: str,
        additional_size: float,
        price: float,
    ) -> bool:
        """
        Add to an existing position.

        Returns:
            True if scale-in was successful
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        if not self.config.allow_scale_in:
            return False

        if position.scale_in_count >= self.config.max_scale_ins:
            return False

        # Check if conditions for scale-in are met
        pnl_pct = position.get_unrealized_pnl_pct(price) / 100
        if pnl_pct > self.config.scale_in_threshold_pct:
            return False  # Don't scale in unless position is down

        # Calculate new average entry
        total_cost = (position.entry_price * position.current_size) + (price * additional_size)
        new_size = position.current_size + additional_size
        position.entry_price = total_cost / new_size
        position.current_size = new_size
        position.initial_size += additional_size
        position.scale_in_count += 1

        # Recalculate stop loss from new entry
        if position.direction == "LONG":
            position.initial_stop = position.entry_price * (1 - self.config.initial_stop_pct)
        else:
            position.initial_stop = position.entry_price * (1 + self.config.initial_stop_pct)
        position.current_stop = position.initial_stop

        return True

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self.positions

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self.positions.copy()

    def get_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        return sum(
            p.current_size * p.entry_price
            for p in self.positions.values()
        )

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())
