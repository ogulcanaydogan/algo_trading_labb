"""
Trading Mode Definitions.

Defines the trading modes and their configurations for the unified trading engine.
Supports seamless transitions from paper trading to live trading with safety gates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class TradingMode(Enum):
    """Trading mode state machine."""

    BACKTEST = "backtest"
    PAPER_SYNTHETIC = "paper_synthetic"
    PAPER_LIVE_DATA = "paper_live_data"
    TESTNET = "testnet"
    LIVE_LIMITED = "live_limited"
    LIVE_FULL = "live_full"

    @property
    def is_paper(self) -> bool:
        """Check if this is a paper trading mode."""
        return self in (
            TradingMode.BACKTEST,
            TradingMode.PAPER_SYNTHETIC,
            TradingMode.PAPER_LIVE_DATA,
        )

    @property
    def is_live(self) -> bool:
        """Check if this uses real money."""
        return self in (TradingMode.LIVE_LIMITED, TradingMode.LIVE_FULL)

    @property
    def uses_real_data(self) -> bool:
        """Check if this mode uses real market data."""
        return self not in (TradingMode.BACKTEST, TradingMode.PAPER_SYNTHETIC)

    @property
    def executes_real_orders(self) -> bool:
        """Check if this mode executes real orders."""
        return self in (
            TradingMode.TESTNET,
            TradingMode.LIVE_LIMITED,
            TradingMode.LIVE_FULL,
        )

    @classmethod
    def get_progression(cls) -> List["TradingMode"]:
        """Get the standard mode progression path."""
        return [
            cls.BACKTEST,
            cls.PAPER_SYNTHETIC,
            cls.PAPER_LIVE_DATA,
            cls.TESTNET,
            cls.LIVE_LIMITED,
            cls.LIVE_FULL,
        ]

    def can_transition_to(self, target: "TradingMode") -> bool:
        """Check if direct transition to target mode is allowed."""
        progression = self.get_progression()
        current_idx = progression.index(self)
        target_idx = progression.index(target)
        # Can only move forward one step or backward any number of steps
        return target_idx <= current_idx + 1


class TradingStatus(Enum):
    """Current trading status within a mode."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"
    TRANSITIONING = "transitioning"


@dataclass
class ModeConfig:
    """Configuration specific to each trading mode."""

    mode: TradingMode
    max_position_usd: float
    max_daily_loss_pct: float
    max_daily_loss_usd: float
    max_trades_per_day: int
    max_open_positions: int
    require_confirmation: bool
    allowed_symbols: List[str]
    capital_limit: Optional[float] = None
    min_balance_reserve_pct: float = 0.20

    # Safety settings
    emergency_stop_loss_pct: float = 0.05
    max_single_trade_loss_pct: float = 0.02
    auto_stop_on_consecutive_losses: int = 5

    @classmethod
    def get_default(cls, mode: TradingMode) -> "ModeConfig":
        """Get default configuration for a mode."""
        defaults = {
            TradingMode.BACKTEST: cls(
                mode=mode,
                max_position_usd=float("inf"),
                max_daily_loss_pct=1.0,
                max_daily_loss_usd=float("inf"),
                max_trades_per_day=1000,
                max_open_positions=100,
                require_confirmation=False,
                allowed_symbols=["*"],
                capital_limit=None,
            ),
            TradingMode.PAPER_SYNTHETIC: cls(
                mode=mode,
                max_position_usd=float("inf"),
                max_daily_loss_pct=1.0,
                max_daily_loss_usd=float("inf"),
                max_trades_per_day=1000,
                max_open_positions=100,
                require_confirmation=False,
                allowed_symbols=["*"],
                capital_limit=None,
            ),
            TradingMode.PAPER_LIVE_DATA: cls(
                mode=mode,
                max_position_usd=float("inf"),
                max_daily_loss_pct=1.0,
                max_daily_loss_usd=float("inf"),
                max_trades_per_day=100,
                max_open_positions=10,
                require_confirmation=False,
                allowed_symbols=["*"],
                capital_limit=None,
            ),
            TradingMode.TESTNET: cls(
                mode=mode,
                max_position_usd=1000.0,
                max_daily_loss_pct=0.10,
                max_daily_loss_usd=1000.0,
                max_trades_per_day=50,
                max_open_positions=5,
                require_confirmation=False,
                allowed_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                capital_limit=10000.0,
            ),
            TradingMode.LIVE_LIMITED: cls(
                mode=mode,
                max_position_usd=20.0,  # Max $20 per position with $100 capital
                max_daily_loss_pct=0.02,  # 2% max daily loss
                max_daily_loss_usd=2.0,  # Max $2 daily loss
                max_trades_per_day=10,
                max_open_positions=3,
                require_confirmation=True,
                allowed_symbols=["BTC/USDT", "ETH/USDT"],
                capital_limit=100.0,  # User selected $100
            ),
            TradingMode.LIVE_FULL: cls(
                mode=mode,
                max_position_usd=5000.0,
                max_daily_loss_pct=0.05,
                max_daily_loss_usd=500.0,
                max_trades_per_day=50,
                max_open_positions=10,
                require_confirmation=True,
                allowed_symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
                capital_limit=None,  # User-defined
            ),
        }
        return defaults.get(mode, defaults[TradingMode.PAPER_LIVE_DATA])


@dataclass
class ModeState:
    """Current state of a trading mode."""

    mode: TradingMode
    status: TradingStatus = TradingStatus.STOPPED
    started_at: Optional[datetime] = None

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    peak_balance: float = 0.0
    max_drawdown_pct: float = 0.0

    # Daily tracking
    daily_trades: int = 0
    daily_pnl: float = 0.0
    daily_loss: float = 0.0
    last_trade_date: Optional[str] = None
    consecutive_losses: int = 0

    # Safety state
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None
    emergency_stop_active: bool = False
    emergency_stop_reason: Optional[str] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        # This is a simplified version - would need to track gross profit/loss separately
        if self.total_pnl <= 0:
            return 0.0
        return 1.0 + (self.total_pnl / max(abs(self.total_pnl), 1))

    @property
    def days_in_mode(self) -> int:
        """Calculate days in current mode."""
        if self.started_at is None:
            return 0
        return (datetime.now() - self.started_at).days

    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_loss = 0.0
        self.last_trade_date = datetime.now().strftime("%Y-%m-%d")

    def record_trade(self, pnl: float, is_win: bool) -> None:
        """Record a trade result."""
        self.total_trades += 1
        self.daily_trades += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl

        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.daily_loss += abs(pnl)

        self.last_trade_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class TransitionRequirements:
    """Requirements to transition between modes."""

    min_days_in_current_mode: int
    min_trades: int
    min_win_rate: float
    max_drawdown_pct: float
    min_profit_factor: float
    require_manual_approval: bool

    def check(self, state: ModeState) -> tuple[bool, List[str]]:
        """Check if requirements are met. Returns (passed, reasons)."""
        failures = []

        if state.days_in_mode < self.min_days_in_current_mode:
            failures.append(
                f"Need {self.min_days_in_current_mode} days in mode, "
                f"have {state.days_in_mode}"
            )

        if state.total_trades < self.min_trades:
            failures.append(
                f"Need {self.min_trades} trades, have {state.total_trades}"
            )

        if state.win_rate < self.min_win_rate:
            failures.append(
                f"Need {self.min_win_rate:.0%} win rate, "
                f"have {state.win_rate:.0%}"
            )

        if state.max_drawdown_pct > self.max_drawdown_pct:
            failures.append(
                f"Max drawdown {self.max_drawdown_pct:.0%} exceeded, "
                f"have {state.max_drawdown_pct:.0%}"
            )

        if state.profit_factor < self.min_profit_factor:
            failures.append(
                f"Need profit factor {self.min_profit_factor}, "
                f"have {state.profit_factor:.2f}"
            )

        return len(failures) == 0, failures


# Strict transition requirements (user selected)
TRANSITION_REQUIREMENTS: Dict[tuple, TransitionRequirements] = {
    (TradingMode.PAPER_SYNTHETIC, TradingMode.PAPER_LIVE_DATA): TransitionRequirements(
        min_days_in_current_mode=3,
        min_trades=30,
        min_win_rate=0.0,
        max_drawdown_pct=0.25,
        min_profit_factor=0.0,
        require_manual_approval=False,
    ),
    (TradingMode.PAPER_LIVE_DATA, TradingMode.TESTNET): TransitionRequirements(
        min_days_in_current_mode=14,
        min_trades=100,
        min_win_rate=0.45,
        max_drawdown_pct=0.12,
        min_profit_factor=1.0,
        require_manual_approval=True,
    ),
    (TradingMode.TESTNET, TradingMode.LIVE_LIMITED): TransitionRequirements(
        min_days_in_current_mode=14,
        min_trades=100,
        min_win_rate=0.45,
        max_drawdown_pct=0.10,
        min_profit_factor=1.0,
        require_manual_approval=True,
    ),
    (TradingMode.LIVE_LIMITED, TradingMode.LIVE_FULL): TransitionRequirements(
        min_days_in_current_mode=30,
        min_trades=200,
        min_win_rate=0.50,
        max_drawdown_pct=0.08,
        min_profit_factor=1.2,
        require_manual_approval=True,
    ),
}


def get_transition_requirements(
    from_mode: TradingMode, to_mode: TradingMode
) -> Optional[TransitionRequirements]:
    """Get transition requirements between two modes."""
    return TRANSITION_REQUIREMENTS.get((from_mode, to_mode))
