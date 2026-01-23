"""
Trailing Stop Loss Module.

Implements trailing stop functionality that moves stop loss
upward as price increases, locking in profits.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop."""

    # Basic settings
    enabled: bool = True
    initial_stop_pct: float = 0.02  # 2% initial stop loss
    trailing_pct: float = 0.015  # 1.5% trailing distance

    # Activation settings
    activation_profit_pct: float = 0.01  # Activate after 1% profit

    # Step trailing (moves in steps rather than continuously)
    use_step_trailing: bool = True
    step_size_pct: float = 0.005  # 0.5% steps

    # Breakeven settings
    move_to_breakeven_pct: float = 0.015  # Move to breakeven after 1.5% profit
    breakeven_buffer_pct: float = 0.001  # Small buffer above breakeven


@dataclass
class TrailingStopState:
    """State for a single position's trailing stop."""

    symbol: str
    entry_price: float
    current_stop: float
    highest_price: float
    is_activated: bool = False
    is_at_breakeven: bool = False
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "current_stop": self.current_stop,
            "highest_price": self.highest_price,
            "is_activated": self.is_activated,
            "is_at_breakeven": self.is_at_breakeven,
            "last_update": self.last_update.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrailingStopState":
        return cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            current_stop=data["current_stop"],
            highest_price=data["highest_price"],
            is_activated=data.get("is_activated", False),
            is_at_breakeven=data.get("is_at_breakeven", False),
            last_update=datetime.fromisoformat(data["last_update"]),
        )


class TrailingStopManager:
    """
    Manages trailing stop losses for multiple positions.

    Features:
    - Initial stop loss at entry
    - Move to breakeven after X% profit
    - Trail stop upward as price increases
    - Optional step-based trailing
    """

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or TrailingStopConfig()
        self.stops: Dict[str, TrailingStopState] = {}

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        side: str = "long",
    ) -> TrailingStopState:
        """
        Add a new position with trailing stop.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: "long" or "short"

        Returns:
            TrailingStopState for the position
        """
        # Calculate initial stop
        if side == "long":
            initial_stop = entry_price * (1 - self.config.initial_stop_pct)
        else:
            initial_stop = entry_price * (1 + self.config.initial_stop_pct)

        state = TrailingStopState(
            symbol=symbol,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_price=entry_price,
        )

        self.stops[symbol] = state
        logger.info(
            f"Trailing stop added: {symbol} entry=${entry_price:.2f}, "
            f"stop=${initial_stop:.2f} ({self.config.initial_stop_pct * 100:.1f}%)"
        )

        return state

    def update(
        self,
        symbol: str,
        current_price: float,
        side: str = "long",
    ) -> Dict[str, Any]:
        """
        Update trailing stop based on current price.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            side: "long" or "short"

        Returns:
            Dict with stop status and any actions needed
        """
        if symbol not in self.stops:
            return {"action": "none", "reason": "no_stop_configured"}

        state = self.stops[symbol]
        result = {
            "action": "none",
            "symbol": symbol,
            "current_price": current_price,
            "stop_price": state.current_stop,
            "entry_price": state.entry_price,
        }

        # Check if stop hit
        if side == "long" and current_price <= state.current_stop:
            result["action"] = "stop_triggered"
            result["reason"] = "price_below_stop"
            logger.warning(
                f"STOP TRIGGERED: {symbol} price=${current_price:.2f} "
                f"<= stop=${state.current_stop:.2f}"
            )
            return result
        elif side == "short" and current_price >= state.current_stop:
            result["action"] = "stop_triggered"
            result["reason"] = "price_above_stop"
            return result

        # Calculate profit percentage
        if side == "long":
            profit_pct = (current_price - state.entry_price) / state.entry_price
        else:
            profit_pct = (state.entry_price - current_price) / state.entry_price

        result["profit_pct"] = profit_pct

        # Update highest price (for long) or lowest (for short)
        if side == "long":
            if current_price > state.highest_price:
                state.highest_price = current_price
                result["new_high"] = True
        else:
            if current_price < state.highest_price:
                state.highest_price = current_price
                result["new_low"] = True

        # Move to breakeven check
        if not state.is_at_breakeven and profit_pct >= self.config.move_to_breakeven_pct:
            if side == "long":
                new_stop = state.entry_price * (1 + self.config.breakeven_buffer_pct)
            else:
                new_stop = state.entry_price * (1 - self.config.breakeven_buffer_pct)

            if (side == "long" and new_stop > state.current_stop) or (
                side == "short" and new_stop < state.current_stop
            ):
                state.current_stop = new_stop
                state.is_at_breakeven = True
                result["action"] = "moved_to_breakeven"
                result["new_stop"] = new_stop
                logger.info(f"Moved to breakeven: {symbol} stop=${new_stop:.2f}")

        # Trailing stop activation check
        if not state.is_activated and profit_pct >= self.config.activation_profit_pct:
            state.is_activated = True
            result["trailing_activated"] = True
            logger.info(f"Trailing stop activated: {symbol} at {profit_pct * 100:.1f}% profit")

        # Trail the stop if activated
        if state.is_activated:
            if side == "long":
                new_trail_stop = state.highest_price * (1 - self.config.trailing_pct)

                # Apply step trailing if enabled
                if self.config.use_step_trailing:
                    step = state.entry_price * self.config.step_size_pct
                    new_trail_stop = (
                        state.current_stop
                        + int((new_trail_stop - state.current_stop) / step) * step
                    )

                if new_trail_stop > state.current_stop:
                    old_stop = state.current_stop
                    state.current_stop = new_trail_stop
                    result["action"] = "stop_raised"
                    result["old_stop"] = old_stop
                    result["new_stop"] = new_trail_stop
                    logger.info(
                        f"Trailing stop raised: {symbol} ${old_stop:.2f} -> ${new_trail_stop:.2f}"
                    )
            else:
                new_trail_stop = state.highest_price * (1 + self.config.trailing_pct)

                if self.config.use_step_trailing:
                    step = state.entry_price * self.config.step_size_pct
                    new_trail_stop = (
                        state.current_stop
                        - int((state.current_stop - new_trail_stop) / step) * step
                    )

                if new_trail_stop < state.current_stop:
                    old_stop = state.current_stop
                    state.current_stop = new_trail_stop
                    result["action"] = "stop_lowered"
                    result["old_stop"] = old_stop
                    result["new_stop"] = new_trail_stop

        state.last_update = datetime.now()
        result["stop_price"] = state.current_stop

        return result

    def remove_position(self, symbol: str) -> bool:
        """Remove trailing stop for a position."""
        if symbol in self.stops:
            del self.stops[symbol]
            logger.info(f"Trailing stop removed: {symbol}")
            return True
        return False

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Get current stop price for a symbol."""
        if symbol in self.stops:
            return self.stops[symbol].current_stop
        return None

    def get_all_stops(self) -> Dict[str, Dict[str, Any]]:
        """Get all trailing stop states."""
        return {symbol: state.to_dict() for symbol, state in self.stops.items()}

    def load_state(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        """Load trailing stop states from dict."""
        self.stops = {
            symbol: TrailingStopState.from_dict(data) for symbol, data in state_dict.items()
        }
        logger.info(f"Loaded {len(self.stops)} trailing stop states")

    def should_close_position(
        self,
        symbol: str,
        current_price: float,
        side: str = "long",
    ) -> bool:
        """Check if position should be closed due to stop."""
        result = self.update(symbol, current_price, side)
        return result.get("action") == "stop_triggered"


def create_trailing_stop_manager(
    enabled: bool = True,
    initial_stop_pct: float = 0.02,
    trailing_pct: float = 0.015,
    activation_profit_pct: float = 0.01,
) -> TrailingStopManager:
    """Factory function to create trailing stop manager."""
    config = TrailingStopConfig(
        enabled=enabled,
        initial_stop_pct=initial_stop_pct,
        trailing_pct=trailing_pct,
        activation_profit_pct=activation_profit_pct,
    )
    return TrailingStopManager(config)
