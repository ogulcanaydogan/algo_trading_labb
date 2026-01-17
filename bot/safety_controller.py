"""
Safety Controller Module.

Provides comprehensive safety checks, kill switches, and emergency stops
for trading operations. This is a critical safety layer that prevents
catastrophic losses.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Current safety status."""

    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits configuration - user selected focus on daily loss and position caps."""

    # Position limits (user priority)
    max_position_size_usd: float = 20.0  # $20 max for LIVE_LIMITED
    max_position_size_pct: float = 0.05  # 5% of capital per position (CRITICAL: was 20% - unsafe!)
    max_open_positions: int = 3

    # Daily loss limits (user priority)
    max_daily_loss_usd: float = 2.0  # $2 max for LIVE_LIMITED
    max_daily_loss_pct: float = 0.02  # 2% max daily loss

    # Single trade limits
    max_single_trade_loss_pct: float = 0.01  # 1% max per trade
    max_single_trade_loss_usd: float = 1.0  # $1 max per trade

    # Trading frequency
    max_trades_per_day: int = 10
    max_trades_per_hour: int = 5
    min_time_between_trades_seconds: int = 60

    # Balance protection
    min_balance_reserve_pct: float = 0.20  # Keep 20% as reserve
    min_balance_reserve_usd: float = 20.0  # $20 minimum reserve

    # Emergency triggers
    emergency_stop_loss_pct: float = 0.05  # 5% total loss triggers emergency
    max_consecutive_losses: int = 5
    max_api_errors: int = 3

    # Capital cap
    capital_limit: Optional[float] = 100.0  # $100 for LIVE_LIMITED


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    passed: bool
    status: SafetyStatus
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_loss: float = 0.0  # Absolute value of losses
    last_trade_time: Optional[datetime] = None
    consecutive_losses: int = 0
    api_errors: int = 0
    hourly_trades: Dict[int, int] = field(default_factory=dict)


class SafetyController:
    """
    Safety controller for trading operations.

    Implements pre-trade checks, post-trade monitoring, and emergency stops.
    User priorities: max daily loss limit and position size caps.
    """

    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
        state_path: Optional[Path] = None,
    ):
        self.limits = limits or SafetyLimits()
        self.state_path = state_path or Path("data/safety_state.json")

        self._lock = Lock()
        self._status = SafetyStatus.OK
        self._emergency_stop_active = False
        self._emergency_stop_reason: Optional[str] = None
        self._daily_stats = DailyStats(date=self._today())
        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0
        self._open_positions: Dict[str, float] = {}  # symbol -> value

        self._load_state()

    def _today(self) -> str:
        """Get today's date string."""
        return datetime.now().strftime("%Y-%m-%d")

    def _current_hour(self) -> int:
        """Get current hour."""
        return datetime.now().hour

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                    if data.get("date") == self._today():
                        self._daily_stats = DailyStats(**data.get("daily_stats", {}))
                    self._peak_balance = data.get("peak_balance", 0.0)
                    self._emergency_stop_active = data.get("emergency_stop_active", False)
                    self._emergency_stop_reason = data.get("emergency_stop_reason")
                    logger.info("Loaded safety state from disk")
            except Exception as e:
                logger.warning(f"Failed to load safety state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(
                    {
                        "date": self._today(),
                        "daily_stats": {
                            "date": self._daily_stats.date,
                            "trades": self._daily_stats.trades,
                            "wins": self._daily_stats.wins,
                            "losses": self._daily_stats.losses,
                            "total_pnl": self._daily_stats.total_pnl,
                            "total_loss": self._daily_stats.total_loss,
                            "consecutive_losses": self._daily_stats.consecutive_losses,
                            "api_errors": self._daily_stats.api_errors,
                        },
                        "peak_balance": self._peak_balance,
                        "emergency_stop_active": self._emergency_stop_active,
                        "emergency_stop_reason": self._emergency_stop_reason,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save safety state: {e}")

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats if it's a new day."""
        if self._daily_stats.date != self._today():
            logger.info("New day detected, resetting daily stats")
            self._daily_stats = DailyStats(date=self._today())
            self._save_state()

    def update_balance(self, balance: float) -> None:
        """Update current balance and peak tracking."""
        with self._lock:
            self._current_balance = balance
            if balance > self._peak_balance:
                self._peak_balance = balance
                self._save_state()
            
            # Recalculate dynamic position limits based on current balance
            if self.limits.max_position_size_pct > 0:
                self.limits.max_position_size_usd = balance * self.limits.max_position_size_pct
            if self.limits.max_daily_loss_pct > 0:
                self.limits.max_daily_loss_usd = balance * self.limits.max_daily_loss_pct

    def update_positions(self, positions: Dict[str, float]) -> None:
        """Update open positions. positions = {symbol: value_usd}."""
        with self._lock:
            self._open_positions = positions.copy()

    def pre_trade_check(self, order: Any) -> Tuple[bool, str]:
        """
        Perform pre-trade safety checks.

        Returns (passed, reason).
        User priorities: daily loss limit and position size caps.
        """
        with self._lock:
            self._reset_daily_stats_if_needed()

            # Emergency stop check (highest priority)
            if self._emergency_stop_active:
                return False, f"Emergency stop active: {self._emergency_stop_reason}"

            # Check trading status
            if self._status == SafetyStatus.EMERGENCY_STOP:
                return False, "Trading halted due to emergency stop"

            if self._status == SafetyStatus.BLOCKED:
                return False, "Trading blocked by safety controller"

            # Daily loss limit check (USER PRIORITY)
            if self._daily_stats.total_loss >= self.limits.max_daily_loss_usd:
                self._status = SafetyStatus.BLOCKED
                return (
                    False,
                    f"Daily loss limit reached: ${self._daily_stats.total_loss:.2f} "
                    f">= ${self.limits.max_daily_loss_usd:.2f}",
                )

            daily_loss_pct = (
                self._daily_stats.total_loss / self._current_balance
                if self._current_balance > 0
                else 0
            )
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                self._status = SafetyStatus.BLOCKED
                return (
                    False,
                    f"Daily loss % limit reached: {daily_loss_pct:.1%} "
                    f">= {self.limits.max_daily_loss_pct:.1%}",
                )

            # Position size check (USER PRIORITY)
            order_qty = getattr(order, "quantity", 0) or 0
            order_price = getattr(order, "price", None)
            order_value = order_qty * order_price if order_price else 0
            # Use current price estimate if not available
            if order_value == 0:
                order_value = order_qty * 50000  # Estimate for market orders

            if order_value > self.limits.max_position_size_usd:
                return (
                    False,
                    f"Position size ${order_value:.2f} exceeds limit "
                    f"${self.limits.max_position_size_usd:.2f}",
                )

            position_pct = (
                order_value / self._current_balance if self._current_balance > 0 else 1
            )
            if position_pct > self.limits.max_position_size_pct:
                return (
                    False,
                    f"Position size {position_pct:.1%} exceeds limit "
                    f"{self.limits.max_position_size_pct:.1%}",
                )

            # Daily trades limit check
            if self._daily_stats.trades >= self.limits.max_trades_per_day:
                return (
                    False,
                    f"Daily trade limit reached: {self._daily_stats.trades} "
                    f">= {self.limits.max_trades_per_day}",
                )

            # Hourly trades limit check
            current_hour = self._current_hour()
            hourly_trades = self._daily_stats.hourly_trades.get(current_hour, 0)
            if hourly_trades >= self.limits.max_trades_per_hour:
                return (
                    False,
                    f"Hourly trade limit reached: {hourly_trades} "
                    f">= {self.limits.max_trades_per_hour}",
                )

            # Time between trades check
            if self._daily_stats.last_trade_time:
                time_since_last = (
                    datetime.now() - self._daily_stats.last_trade_time
                ).total_seconds()
                if time_since_last < self.limits.min_time_between_trades_seconds:
                    return (
                        False,
                        f"Too soon since last trade: {time_since_last:.0f}s "
                        f"< {self.limits.min_time_between_trades_seconds}s",
                    )

            # Open positions limit check
            if len(self._open_positions) >= self.limits.max_open_positions:
                symbol = getattr(order, "symbol", "unknown")
                if symbol not in self._open_positions:
                    return (
                        False,
                        f"Max open positions reached: {len(self._open_positions)} "
                        f">= {self.limits.max_open_positions}",
                    )

            # Balance reserve check
            available_for_trading = self._current_balance * (
                1 - self.limits.min_balance_reserve_pct
            )
            total_position_value = sum(self._open_positions.values())
            if total_position_value + order_value > available_for_trading:
                return (
                    False,
                    f"Would exceed available balance: "
                    f"${total_position_value + order_value:.2f} > "
                    f"${available_for_trading:.2f}",
                )

            # Capital limit check
            if self.limits.capital_limit:
                if self._current_balance > self.limits.capital_limit * 1.1:
                    logger.warning(
                        f"Balance ${self._current_balance:.2f} exceeds capital limit "
                        f"${self.limits.capital_limit:.2f}"
                    )

            # Consecutive losses check
            if (
                self._daily_stats.consecutive_losses
                >= self.limits.max_consecutive_losses
            ):
                return (
                    False,
                    f"Too many consecutive losses: "
                    f"{self._daily_stats.consecutive_losses} "
                    f">= {self.limits.max_consecutive_losses}",
                )

            return True, "All safety checks passed"

    def post_trade_check(self, result: Any) -> None:
        """
        Post-trade safety check and stat updates.

        Called after every trade execution.
        """
        with self._lock:
            self._reset_daily_stats_if_needed()

            # Update trade count
            self._daily_stats.trades += 1
            current_hour = self._current_hour()
            self._daily_stats.hourly_trades[current_hour] = (
                self._daily_stats.hourly_trades.get(current_hour, 0) + 1
            )
            self._daily_stats.last_trade_time = datetime.now()

            # Calculate P&L if available
            pnl = getattr(result, "pnl", 0)
            if pnl is None:
                pnl = 0

            self._daily_stats.total_pnl += pnl

            if pnl >= 0:
                self._daily_stats.wins += 1
                self._daily_stats.consecutive_losses = 0
            else:
                self._daily_stats.losses += 1
                self._daily_stats.consecutive_losses += 1
                self._daily_stats.total_loss += abs(pnl)

                # Check if we should trigger emergency stop
                if self._daily_stats.total_loss >= self.limits.max_daily_loss_usd:
                    self.emergency_stop(
                        f"Daily loss limit exceeded: ${self._daily_stats.total_loss:.2f}"
                    )
                elif (
                    self._daily_stats.consecutive_losses
                    >= self.limits.max_consecutive_losses
                ):
                    self.emergency_stop(
                        f"Too many consecutive losses: "
                        f"{self._daily_stats.consecutive_losses}"
                    )

            self._save_state()

    def record_api_error(self) -> None:
        """Record an API error."""
        with self._lock:
            self._daily_stats.api_errors += 1
            if self._daily_stats.api_errors >= self.limits.max_api_errors:
                self.emergency_stop(
                    f"Too many API errors: {self._daily_stats.api_errors}"
                )
            self._save_state()

    def clear_api_errors(self) -> None:
        """Clear API error count after successful operation."""
        with self._lock:
            self._daily_stats.api_errors = 0

    def emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop.

        This immediately halts all trading until manually cleared.
        """
        with self._lock:
            self._emergency_stop_active = True
            self._emergency_stop_reason = reason
            self._status = SafetyStatus.EMERGENCY_STOP
            self._save_state()

        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

        # Could also send notifications here
        try:
            from bot.notifications import NotificationManager

            notifier = NotificationManager()
            notifier.send_critical(f"EMERGENCY STOP: {reason}")
        except Exception:
            pass

    def clear_emergency_stop(self, approver: str = "manual") -> bool:
        """
        Clear emergency stop.

        Requires explicit approval to resume trading.
        """
        with self._lock:
            if not self._emergency_stop_active:
                return True

            logger.warning(f"Clearing emergency stop. Approved by: {approver}")
            self._emergency_stop_active = False
            self._emergency_stop_reason = None
            self._status = SafetyStatus.OK
            self._save_state()

        return True

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed."""
        with self._lock:
            if self._emergency_stop_active:
                return False, f"Emergency stop: {self._emergency_stop_reason}"

            if self._status == SafetyStatus.BLOCKED:
                return False, "Trading blocked by safety limits"

            if self._status == SafetyStatus.EMERGENCY_STOP:
                return False, "Emergency stop active"

            return True, "Trading allowed"

    def get_status(self) -> Dict[str, Any]:
        """Get current safety status for monitoring/dashboard."""
        with self._lock:
            return {
                "status": self._status.value,
                "emergency_stop_active": self._emergency_stop_active,
                "emergency_stop_reason": self._emergency_stop_reason,
                "daily_stats": {
                    "date": self._daily_stats.date,
                    "trades": self._daily_stats.trades,
                    "wins": self._daily_stats.wins,
                    "losses": self._daily_stats.losses,
                    "total_pnl": self._daily_stats.total_pnl,
                    "total_loss": self._daily_stats.total_loss,
                    "consecutive_losses": self._daily_stats.consecutive_losses,
                    "api_errors": self._daily_stats.api_errors,
                },
                "limits": {
                    "max_daily_loss_usd": self.limits.max_daily_loss_usd,
                    "max_position_size_usd": self.limits.max_position_size_usd,
                    "max_trades_per_day": self.limits.max_trades_per_day,
                    "daily_loss_remaining": self.limits.max_daily_loss_usd
                    - self._daily_stats.total_loss,
                    "trades_remaining": self.limits.max_trades_per_day
                    - self._daily_stats.trades,
                },
                "current_balance": self._current_balance,
                "peak_balance": self._peak_balance,
                "open_positions": len(self._open_positions),
            }

    def get_remaining_capacity(self) -> Dict[str, Any]:
        """Get remaining trading capacity for the day."""
        with self._lock:
            return {
                "trades_remaining": max(
                    0, self.limits.max_trades_per_day - self._daily_stats.trades
                ),
                "loss_remaining_usd": max(
                    0, self.limits.max_daily_loss_usd - self._daily_stats.total_loss
                ),
                "position_slots_remaining": max(
                    0, self.limits.max_open_positions - len(self._open_positions)
                ),
                "available_capital": self._current_balance
                * (1 - self.limits.min_balance_reserve_pct)
                - sum(self._open_positions.values()),
            }


def create_safety_controller_for_mode(
    mode: str, capital: float = 10000.0
) -> SafetyController:
    """
    Create a safety controller with appropriate limits for the trading mode.
    """
    from bot.trading_mode import TradingMode

    mode_enum = TradingMode(mode) if isinstance(mode, str) else mode

    if mode_enum == TradingMode.LIVE_LIMITED:
        # Most restrictive - enforce 5% position size limit (scales with balance)
        limits = SafetyLimits(
            max_position_size_usd=capital * 0.05,  # 5% of current capital
            max_position_size_pct=0.05,  # Will auto-scale with balance
            max_daily_loss_usd=capital * 0.02,  # 2% of current capital
            max_daily_loss_pct=0.02,  # Will auto-scale with balance
            max_trades_per_day=20,
            max_open_positions=3,
            capital_limit=None,  # Remove hard capital limit, use percentage-based
        )
    elif mode_enum == TradingMode.LIVE_FULL:
        limits = SafetyLimits(
            max_position_size_usd=capital * 0.20,
            max_position_size_pct=0.20,
            max_daily_loss_usd=capital * 0.05,
            max_daily_loss_pct=0.05,
            max_trades_per_day=50,
            max_open_positions=10,
            capital_limit=None,
        )
    elif mode_enum == TradingMode.TESTNET:
        limits = SafetyLimits(
            max_position_size_usd=1000.0,
            max_position_size_pct=0.25,
            max_daily_loss_usd=1000.0,
            max_daily_loss_pct=0.10,
            max_trades_per_day=50,
            max_open_positions=5,
            capital_limit=10000.0,
        )
    else:
        # Paper trading - relaxed limits
        limits = SafetyLimits(
            max_position_size_usd=float("inf"),
            max_position_size_pct=1.0,
            max_daily_loss_usd=float("inf"),
            max_daily_loss_pct=1.0,
            max_trades_per_day=1000,
            max_open_positions=100,
            capital_limit=None,
        )

    return SafetyController(limits=limits)
