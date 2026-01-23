"""
Trading Operations Manager

This module manages buy/sell operations on Binance Testnet or real exchange.
"""

import json
import logging
from typing import Optional, Dict, Literal, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

from .state import create_state_store, SignalEvent

logger = logging.getLogger(__name__)


@dataclass
class PositionManagementState:
    """Persistent state for position management features."""

    # Trailing stop state
    peak_price: Optional[float] = None
    trailing_active: bool = False
    current_atr: Optional[float] = None

    # Break-even state
    break_even_activated: bool = False

    # Partial profit state
    partial_exits_completed: list = None
    original_size: float = 0.0
    initial_risk: float = 0.0

    def __post_init__(self):
        if self.partial_exits_completed is None:
            self.partial_exits_completed = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_price": self.peak_price,
            "trailing_active": self.trailing_active,
            "current_atr": self.current_atr,
            "break_even_activated": self.break_even_activated,
            "partial_exits_completed": self.partial_exits_completed,
            "original_size": self.original_size,
            "initial_risk": self.initial_risk,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionManagementState":
        """Create from dictionary."""
        return cls(
            peak_price=data.get("peak_price"),
            trailing_active=data.get("trailing_active", False),
            current_atr=data.get("current_atr"),
            break_even_activated=data.get("break_even_activated", False),
            partial_exits_completed=data.get("partial_exits_completed", []),
            original_size=data.get("original_size", 0.0),
            initial_risk=data.get("initial_risk", 0.0),
        )


@dataclass
class OrderResult:
    """Order result"""

    success: bool
    order_id: Optional[str] = None
    price: float = 0.0
    quantity: float = 0.0
    side: str = ""  # BUY or SELL
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop"""

    enabled: bool = True
    activation_pct: float = 1.0  # Start trailing after 1% profit
    trail_pct: float = 0.5  # Trail by 0.5% from peak
    use_atr: bool = True  # Use ATR for trail distance
    atr_multiplier: float = 1.5  # Trail distance = ATR * multiplier


@dataclass
class BreakEvenConfig:
    """Configuration for break-even stop management"""

    enabled: bool = True
    activation_pct: float = 0.5  # Move to break-even after 0.5% profit
    buffer_pct: float = 0.05  # Add small buffer above/below entry (0.05%)


@dataclass
class PartialProfitConfig:
    """Configuration for scaling out of positions"""

    enabled: bool = True
    # Take profit levels as multiples of risk (R)
    # Default: Take 25% at 1R, 25% at 2R, 25% at 3R, let 25% run
    levels: tuple = (
        (1.0, 0.25),  # At 1R profit, close 25%
        (2.0, 0.25),  # At 2R profit, close another 25%
        (3.0, 0.25),  # At 3R profit, close another 25%
        # Remaining 25% runs with trailing stop
    )


@dataclass
class AdvancedPositionConfig:
    """Combined configuration for advanced position management"""

    trailing_stop: TrailingStopConfig = None
    break_even: BreakEvenConfig = None
    partial_profit: PartialProfitConfig = None

    def __post_init__(self):
        if self.trailing_stop is None:
            self.trailing_stop = TrailingStopConfig()
        if self.break_even is None:
            self.break_even = BreakEvenConfig()
        if self.partial_profit is None:
            self.partial_profit = PartialProfitConfig()


class TradingManager:
    """
    Class that manages real trading operations

    This class handles order submission on Binance Testnet or real exchange.
    Includes risk management and position control.
    """

    def __init__(
        self,
        exchange_client,
        symbol: str = "BTC/USDT",
        max_position_size: float = 0.1,  # Maximum position size (BTC)
        min_order_size: float = 0.001,  # Minimum order size
        dry_run: bool = True,  # True = only log, don't send real orders
        trailing_stop_config: Optional[TrailingStopConfig] = None,
        break_even_config: Optional[BreakEvenConfig] = None,
        partial_profit_config: Optional[PartialProfitConfig] = None,
        data_dir: Optional[Path] = None,  # Directory for state persistence
    ):
        self.client = exchange_client
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.min_order_size = min_order_size
        self.dry_run = dry_run
        self.trailing_config = trailing_stop_config or TrailingStopConfig()
        self.break_even_config = break_even_config or BreakEvenConfig()
        self.partial_profit_config = partial_profit_config or PartialProfitConfig()
        self.data_dir = data_dir or Path("./data")

        self.current_position: Optional[Dict] = None
        self.pending_orders: Dict[str, Dict] = {}

        # Initialize position management state
        self._pm_state = PositionManagementState()

        # Legacy attribute aliases for backward compatibility
        self._peak_price: Optional[float] = None
        self._trailing_active: bool = False
        self._current_atr: Optional[float] = None
        self._break_even_activated: bool = False
        self._partial_exits_completed: list = []
        self._original_size: float = 0.0
        self._initial_risk: float = 0.0

        # Load persisted state if available
        self._load_position_state()

        logger.info(
            f"TradingManager initialized | Symbol: {symbol} | Dry Run: {dry_run} | "
            f"Trailing: {self.trailing_config.enabled} | Break-Even: {self.break_even_config.enabled} | "
            f"Partial Profit: {self.partial_profit_config.enabled}"
        )

    def _get_state_file_path(self) -> Path:
        """Get the path for position management state file."""
        safe_symbol = self.symbol.replace("/", "_")
        return self.data_dir / f"position_state_{safe_symbol}.json"

    def _save_position_state(self) -> None:
        """Persist position management state to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            state_data = {
                "position_management": self._pm_state.to_dict(),
                "current_position": self.current_position,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            state_file = self._get_state_file_path()
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2, default=str)
            logger.debug(f"Position state saved to {state_file}")
        except Exception as e:
            logger.warning(f"Failed to save position state: {e}")

    def _load_position_state(self) -> None:
        """Load position management state from disk."""
        try:
            state_file = self._get_state_file_path()
            if state_file.exists():
                with open(state_file) as f:
                    state_data = json.load(f)

                # Restore position management state
                pm_data = state_data.get("position_management", {})
                self._pm_state = PositionManagementState.from_dict(pm_data)

                # Sync legacy attributes
                self._peak_price = self._pm_state.peak_price
                self._trailing_active = self._pm_state.trailing_active
                self._current_atr = self._pm_state.current_atr
                self._break_even_activated = self._pm_state.break_even_activated
                self._partial_exits_completed = self._pm_state.partial_exits_completed
                self._original_size = self._pm_state.original_size
                self._initial_risk = self._pm_state.initial_risk

                # Restore current position
                self.current_position = state_data.get("current_position")

                logger.info(f"Position state loaded from {state_file}")
        except Exception as e:
            logger.debug(f"No position state loaded: {e}")

    def _sync_state_to_pm(self) -> None:
        """Sync legacy attributes to position management state."""
        self._pm_state.peak_price = self._peak_price
        self._pm_state.trailing_active = self._trailing_active
        self._pm_state.current_atr = self._current_atr
        self._pm_state.break_even_activated = self._break_even_activated
        self._pm_state.partial_exits_completed = self._partial_exits_completed
        self._pm_state.original_size = self._original_size
        self._pm_state.initial_risk = self._initial_risk

    def _reset_position_state(self) -> None:
        """Reset position management state when closing a position."""
        self._peak_price = None
        self._trailing_active = False
        self._current_atr = None
        self._break_even_activated = False
        self._partial_exits_completed = []
        self._original_size = 0.0
        self._initial_risk = 0.0

        # Sync and persist
        self._pm_state = PositionManagementState()
        self._save_position_state()

    def open_position(
        self,
        direction: Literal["LONG", "SHORT"],
        size: float,
        stop_loss: float,
        take_profit: float,
        signal_info: Optional[Dict] = None,
        atr: Optional[float] = None,
    ) -> OrderResult:
        """
        Open new position

        Args:
            direction: LONG or SHORT
            size: Position size (BTC)
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_info: Signal information (optional)
            atr: Average True Range for trailing stop (optional)

        Returns:
            OrderResult: Operation result
        """
        # Reset trailing stop state
        self._peak_price = None
        self._trailing_active = False
        self._current_atr = atr

        # Reset break-even state
        self._break_even_activated = False

        # Reset partial profit state
        self._partial_exits_completed = []
        self._original_size = size
        self._initial_risk = abs(stop_loss - (stop_loss / (1 - 0.02)))  # Will be recalculated

        # Check existing position
        if self.current_position:
            logger.warning("Already have an open position!")
            return OrderResult(success=False, error="Existing position already open")

        # Check position size
        if size > self.max_position_size:
            logger.warning(f"Position too large: {size} > {self.max_position_size}")
            size = self.max_position_size

        if size < self.min_order_size:
            logger.warning(f"Position too small: {size} < {self.min_order_size}")
            return OrderResult(success=False, error=f"Position size too small: {size}")

        # Get market price
        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Could not get price: {e}")
            return OrderResult(success=False, error=str(e))

        # Send order
        side = "BUY" if direction == "LONG" else "SELL"

        if self.dry_run:
            logger.info(f"[DRY RUN] {side} order: {size} {self.symbol} @ ${current_price:.2f}")
            logger.info(f"[DRY RUN] Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")

            # Calculate initial risk (1R) for partial profit targets
            self._initial_risk = abs(current_price - stop_loss)
            self._original_size = size

            self.current_position = {
                "direction": direction,
                "entry_price": current_price,
                "size": size,
                "original_size": size,
                "stop_loss": stop_loss,
                "original_stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(timezone.utc),
                "signal_info": signal_info,
                "initial_risk": self._initial_risk,
            }

            # Persist simulated execution to StateStore so UI/dashboard reflects DRY RUN trades
            try:
                store = create_state_store(Path("./data"))
                store.update_state(
                    symbol=self.symbol,
                    position=direction,
                    entry_price=current_price,
                    position_size=size,
                    last_signal=direction,
                    last_signal_reason=signal_info.get("reason")
                    if isinstance(signal_info, dict)
                    else "Simulated DRY RUN",
                    confidence=signal_info.get("confidence")
                    if isinstance(signal_info, dict)
                    else None,
                )
                # record a signal event for history
                evt = SignalEvent(
                    timestamp=datetime.now(timezone.utc),
                    symbol=self.symbol,
                    decision=direction,
                    confidence=float(signal_info.get("confidence", 0.0))
                    if isinstance(signal_info, dict)
                    else 0.0,
                    reason=signal_info.get("reason", "Simulated DRY RUN")
                    if isinstance(signal_info, dict)
                    else "Simulated DRY RUN",
                )
                store.record_signal(evt)
            except (IOError, ValueError, KeyError) as e:
                logger.exception(f"Failed to persist DRY RUN execution to StateStore: {e}")

            # Persist position management state
            self._sync_state_to_pm()
            self._save_position_state()

            return OrderResult(
                success=True,
                order_id=f"DRY_RUN_{datetime.now().timestamp()}",
                price=current_price,
                quantity=size,
                side=side,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            # Real order submission
            try:
                order = self.client.client.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=size,
                )

                logger.info(f"{side} order executed: {order['id']}")

                # Send stop loss and take profit orders
                self._place_stop_loss(direction, size, stop_loss)
                self._place_take_profit(direction, size, take_profit)

                self.current_position = {
                    "direction": direction,
                    "entry_price": order.get("price", current_price),
                    "size": size,
                    "original_size": size,
                    "stop_loss": stop_loss,
                    "original_stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(timezone.utc),
                    "order_id": order["id"],
                    "signal_info": signal_info,
                    "initial_risk": abs(order.get("price", current_price) - stop_loss),
                }

                # Update and persist position management state
                self._initial_risk = abs(order.get("price", current_price) - stop_loss)
                self._original_size = size
                self._sync_state_to_pm()
                self._save_position_state()

                return OrderResult(
                    success=True,
                    order_id=order["id"],
                    price=order.get("price", current_price),
                    quantity=size,
                    side=side,
                    timestamp=datetime.now(timezone.utc),
                )

            except Exception as e:
                logger.error(f"Order failed: {e}")
                return OrderResult(success=False, error=str(e))

    def close_position(self, reason: str = "Manual close") -> OrderResult:
        """
        Close current position

        Args:
            reason: Close reason

        Returns:
            OrderResult: Operation result
        """
        if not self.current_position:
            logger.warning("No position to close!")
            return OrderResult(success=False, error="No position to close")

        direction = self.current_position["direction"]
        size = self.current_position["size"]

        # Send order in opposite direction
        side = "SELL" if direction == "LONG" else "BUY"

        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Could not get price: {e}")
            return OrderResult(success=False, error=str(e))

        if self.dry_run:
            entry_price = self.current_position["entry_price"]
            if direction == "LONG":
                pnl = (current_price - entry_price) * size
                pnl_pct = (current_price / entry_price - 1) * 100
            else:
                pnl = (entry_price - current_price) * size
                pnl_pct = (entry_price / current_price - 1) * 100

            logger.info(
                f"[DRY RUN] Position closed: {side} {size} @ ${current_price:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
            )

            self.current_position = None

            # Persist simulated close to StateStore so UI/dashboard reflects the DRY RUN close
            try:
                store = create_state_store(Path("./data"))
                store.update_state(
                    symbol=self.symbol,
                    position="FLAT",
                    entry_price=None,
                    position_size=0.0,
                    last_signal="FLAT",
                    last_signal_reason=reason,
                )
                evt = SignalEvent(
                    timestamp=datetime.now(timezone.utc),
                    symbol=self.symbol,
                    decision="FLAT",
                    confidence=0.0,
                    reason=reason,
                )
                store.record_signal(evt)
            except (IOError, ValueError, KeyError) as e:
                logger.exception(f"Failed to persist DRY RUN close to StateStore: {e}")

            # Reset and persist position management state
            self._reset_position_state()

            return OrderResult(
                success=True,
                order_id=f"DRY_RUN_{datetime.now().timestamp()}",
                price=current_price,
                quantity=size,
                side=side,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            try:
                # Cancel pending stop/tp orders
                self._cancel_pending_orders()

                # Close position
                order = self.client.client.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=size,
                )

                logger.info(f"Position closed: {order['id']} | Reason: {reason}")

                self.current_position = None

                # Reset and persist position management state
                self._reset_position_state()

                return OrderResult(
                    success=True,
                    order_id=order["id"],
                    price=order.get("price", current_price),
                    quantity=size,
                    side=side,
                    timestamp=datetime.now(timezone.utc),
                )

            except Exception as e:
                logger.error(f"Could not close position: {e}")
                return OrderResult(success=False, error=str(e))

    def check_position_exit(self, current_price: float) -> Optional[str]:
        """
        Check position exit (stop loss / take profit / trailing stop)

        Args:
            current_price: Current price

        Returns:
            Exit reason or None
        """
        if not self.current_position:
            return None

        direction = self.current_position["direction"]
        stop_loss = self.current_position["stop_loss"]
        take_profit = self.current_position["take_profit"]

        # Update break-even stop first (before trailing)
        if self.break_even_config.enabled:
            self._update_break_even_stop(current_price)

        # Update trailing stop
        if self.trailing_config.enabled:
            self._update_trailing_stop(current_price)

        # Use updated stop loss
        stop_loss = self.current_position["stop_loss"]

        if direction == "LONG":
            if current_price <= stop_loss:
                return "Trailing Stop Hit" if self._trailing_active else "Stop Loss Hit"
            elif current_price >= take_profit:
                return "Take Profit Hit"
        else:  # SHORT
            if current_price >= stop_loss:
                return "Trailing Stop Hit" if self._trailing_active else "Stop Loss Hit"
            elif current_price <= take_profit:
                return "Take Profit Hit"

        return None

    def _update_trailing_stop(self, current_price: float) -> None:
        """
        Update trailing stop based on current price.

        The trailing stop:
        1. Activates when position is in profit by activation_pct
        2. Trails the price by trail_pct or ATR * multiplier
        3. Only moves in favorable direction (never backward)
        """
        if not self.current_position:
            return

        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]

        # Calculate current profit percentage
        if direction == "LONG":
            profit_pct = (current_price / entry_price - 1) * 100
            # Check if trailing should activate
            if profit_pct >= self.trailing_config.activation_pct:
                self._trailing_active = True

            if self._trailing_active:
                # Update peak price
                if self._peak_price is None or current_price > self._peak_price:
                    self._peak_price = current_price

                    # Calculate new stop loss
                    if self.trailing_config.use_atr and self._current_atr:
                        trail_distance = self._current_atr * self.trailing_config.atr_multiplier
                    else:
                        trail_distance = self._peak_price * (self.trailing_config.trail_pct / 100)

                    new_stop = self._peak_price - trail_distance

                    # Only move stop up, never down
                    if new_stop > self.current_position["stop_loss"]:
                        old_stop = self.current_position["stop_loss"]
                        self.current_position["stop_loss"] = new_stop
                        logger.info(
                            f"Trailing stop updated: ${old_stop:.2f} -> ${new_stop:.2f} "
                            f"(peak: ${self._peak_price:.2f})"
                        )
                        # Persist state change
                        self._sync_state_to_pm()
                        self._save_position_state()

        else:  # SHORT
            profit_pct = (1 - current_price / entry_price) * 100

            if profit_pct >= self.trailing_config.activation_pct:
                self._trailing_active = True

            if self._trailing_active:
                # Update peak price (lowest for short)
                if self._peak_price is None or current_price < self._peak_price:
                    self._peak_price = current_price

                    # Calculate new stop loss
                    if self.trailing_config.use_atr and self._current_atr:
                        trail_distance = self._current_atr * self.trailing_config.atr_multiplier
                    else:
                        trail_distance = self._peak_price * (self.trailing_config.trail_pct / 100)

                    new_stop = self._peak_price + trail_distance

                    # Only move stop down, never up
                    if new_stop < self.current_position["stop_loss"]:
                        old_stop = self.current_position["stop_loss"]
                        self.current_position["stop_loss"] = new_stop
                        logger.info(
                            f"Trailing stop updated: ${old_stop:.2f} -> ${new_stop:.2f} "
                            f"(peak: ${self._peak_price:.2f})"
                        )
                        # Persist state change
                        self._sync_state_to_pm()
                        self._save_position_state()

    def update_atr(self, atr: float) -> None:
        """Update the current ATR value for trailing stop calculations."""
        self._current_atr = atr

    def _update_break_even_stop(self, current_price: float) -> bool:
        """
        Check and update stop to break-even after profit threshold.

        Returns:
            True if break-even was just activated
        """
        if not self.break_even_config.enabled or self._break_even_activated:
            return False

        if not self.current_position:
            return False

        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]

        # Calculate profit percentage
        if direction == "LONG":
            profit_pct = (current_price / entry_price - 1) * 100
        else:
            profit_pct = (1 - current_price / entry_price) * 100

        # Check if we should move to break-even
        if profit_pct >= self.break_even_config.activation_pct:
            # Calculate break-even price with buffer
            buffer = entry_price * (self.break_even_config.buffer_pct / 100)

            if direction == "LONG":
                new_stop = entry_price + buffer
                # Only move stop up
                if new_stop > self.current_position["stop_loss"]:
                    old_stop = self.current_position["stop_loss"]
                    self.current_position["stop_loss"] = new_stop
                    self._break_even_activated = True
                    logger.info(
                        f"Break-even activated: Stop moved ${old_stop:.2f} -> ${new_stop:.2f} "
                        f"(entry: ${entry_price:.2f})"
                    )
                    # Persist state change
                    self._sync_state_to_pm()
                    self._save_position_state()
                    return True
            else:  # SHORT
                new_stop = entry_price - buffer
                # Only move stop down
                if new_stop < self.current_position["stop_loss"]:
                    old_stop = self.current_position["stop_loss"]
                    self.current_position["stop_loss"] = new_stop
                    self._break_even_activated = True
                    logger.info(
                        f"Break-even activated: Stop moved ${old_stop:.2f} -> ${new_stop:.2f} "
                        f"(entry: ${entry_price:.2f})"
                    )
                    # Persist state change
                    self._sync_state_to_pm()
                    self._save_position_state()
                    return True

        return False

    def check_partial_profit(self, current_price: float) -> Optional[Dict]:
        """
        Check if we should take partial profit at current price.

        Returns:
            Dict with partial exit info if triggered, None otherwise
        """
        if not self.partial_profit_config.enabled:
            return None

        if not self.current_position:
            return None

        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        current_size = self.current_position["size"]
        initial_risk = self.current_position.get("initial_risk", self._initial_risk)

        if initial_risk <= 0:
            return None

        # Calculate current R multiple
        if direction == "LONG":
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price

        current_r = profit / initial_risk

        # Check each profit level
        for r_target, exit_pct in self.partial_profit_config.levels:
            if r_target in self._partial_exits_completed:
                continue

            if current_r >= r_target:
                # Calculate exit size
                exit_size = self._original_size * exit_pct
                exit_size = min(exit_size, current_size)  # Don't exit more than we have

                if exit_size >= self.min_order_size:
                    self._partial_exits_completed.append(r_target)

                    return {
                        "r_target": r_target,
                        "exit_pct": exit_pct,
                        "exit_size": exit_size,
                        "remaining_size": current_size - exit_size,
                        "current_r": current_r,
                        "profit_pct": (current_r * initial_risk / entry_price) * 100,
                    }

        return None

    def execute_partial_exit(self, exit_info: Dict) -> OrderResult:
        """
        Execute a partial position exit.

        Args:
            exit_info: Dict from check_partial_profit()

        Returns:
            OrderResult
        """
        if not self.current_position:
            return OrderResult(success=False, error="No position to partially exit")

        exit_size = exit_info["exit_size"]
        remaining_size = exit_info["remaining_size"]
        r_target = exit_info["r_target"]

        direction = self.current_position["direction"]
        side = "SELL" if direction == "LONG" else "BUY"

        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]
        except Exception as e:
            logger.error(f"Could not get price for partial exit: {e}")
            return OrderResult(success=False, error=str(e))

        if self.dry_run:
            # Calculate P&L for this partial exit
            entry_price = self.current_position["entry_price"]
            if direction == "LONG":
                pnl = (current_price - entry_price) * exit_size
            else:
                pnl = (entry_price - current_price) * exit_size

            logger.info(
                f"[DRY RUN] Partial exit at {r_target}R: {side} {exit_size:.6f} @ ${current_price:.2f} | "
                f"P&L: ${pnl:.2f} | Remaining: {remaining_size:.6f}"
            )

            # Update position size
            self.current_position["size"] = remaining_size

            # Persist state change
            self._sync_state_to_pm()
            self._save_position_state()

            return OrderResult(
                success=True,
                order_id=f"DRY_RUN_PARTIAL_{r_target}R_{datetime.now().timestamp()}",
                price=current_price,
                quantity=exit_size,
                side=side,
                timestamp=datetime.now(timezone.utc),
            )
        else:
            try:
                order = self.client.client.create_market_order(
                    symbol=self.symbol,
                    side=side,
                    amount=exit_size,
                )

                logger.info(
                    f"Partial exit at {r_target}R executed: {order['id']} | "
                    f"Size: {exit_size:.6f} | Remaining: {remaining_size:.6f}"
                )

                self.current_position["size"] = remaining_size

                # Persist state change
                self._sync_state_to_pm()
                self._save_position_state()

                return OrderResult(
                    success=True,
                    order_id=order["id"],
                    price=order.get("price", current_price),
                    quantity=exit_size,
                    side=side,
                    timestamp=datetime.now(timezone.utc),
                )

            except Exception as e:
                logger.error(f"Partial exit failed: {e}")
                return OrderResult(success=False, error=str(e))

    def get_trailing_stop_info(self) -> Optional[Dict]:
        """Get current trailing stop information."""
        if not self.current_position:
            return None

        return {
            "enabled": self.trailing_config.enabled,
            "active": self._trailing_active,
            "peak_price": self._peak_price,
            "current_stop": self.current_position.get("stop_loss"),
            "activation_pct": self.trailing_config.activation_pct,
            "trail_pct": self.trailing_config.trail_pct,
            "atr": self._current_atr,
        }

    def get_position_info(self) -> Optional[Dict]:
        """Return current position information"""
        if not self.current_position:
            return None

        try:
            ticker = self._get_ticker()
            current_price = ticker["last"]

            entry_price = self.current_position["entry_price"]
            size = self.current_position["size"]
            direction = self.current_position["direction"]

            if direction == "LONG":
                unrealized_pnl = (current_price - entry_price) * size
                unrealized_pnl_pct = (current_price / entry_price - 1) * 100
            else:
                unrealized_pnl = (entry_price - current_price) * size
                unrealized_pnl_pct = (entry_price / current_price - 1) * 100

            return {
                **self.current_position,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            }

        except Exception as e:
            logger.error(f"Could not get position info: {e}")
            return self.current_position

    def _get_ticker(self) -> Dict:
        """Get current price information"""
        return self.client.client.fetch_ticker(self.symbol)

    def _place_stop_loss(self, direction: str, size: float, stop_price: float):
        """Send stop loss order"""
        try:
            side = "SELL" if direction == "LONG" else "BUY"
            order = self.client.client.create_order(
                symbol=self.symbol,
                type="STOP_LOSS_LIMIT",
                side=side,
                amount=size,
                price=stop_price,
                params={"stopPrice": stop_price},
            )
            self.pending_orders[order["id"]] = order
            logger.info(f"Stop Loss order sent: {order['id']} @ ${stop_price:.2f}")
        except Exception as e:
            logger.error(f"Stop Loss order failed: {e}")

    def _place_take_profit(self, direction: str, size: float, tp_price: float):
        """Send take profit order"""
        try:
            side = "SELL" if direction == "LONG" else "BUY"
            order = self.client.client.create_limit_order(
                symbol=self.symbol,
                side=side,
                amount=size,
                price=tp_price,
            )
            self.pending_orders[order["id"]] = order
            logger.info(f"Take Profit order sent: {order['id']} @ ${tp_price:.2f}")
        except Exception as e:
            logger.error(f"Take Profit order failed: {e}")

    def _cancel_pending_orders(self):
        """Cancel pending orders"""
        for order_id in list(self.pending_orders.keys()):
            try:
                self.client.client.cancel_order(order_id, self.symbol)
                logger.info(f"Order cancelled: {order_id}")
                del self.pending_orders[order_id]
            except Exception as e:
                logger.error(f"Could not cancel order {order_id}: {e}")
