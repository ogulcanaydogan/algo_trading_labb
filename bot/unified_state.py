"""
Unified State Management Module.

Provides consistent state persistence across all trading modes with
support for positions, trades, equity tracking, and mode state.

Features:
- Sync JSON persistence for quick state recovery
- Async SQLite database for trade analytics and history
- Connection pooling for high-performance writes
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from bot.trading_mode import ModeState, TradingMode, TradingStatus

# Async database for high-performance trade storage
try:
    from bot.async_database import AsyncTradingDatabase, get_async_database
    ASYNC_DB_AVAILABLE = True
except ImportError:
    ASYNC_DB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Position state for persistence."""

    symbol: str
    quantity: float
    entry_price: float
    side: str  # "long" or "short"
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    current_price: Optional[float] = None
    signal_confidence: Optional[float] = None
    signal_reason: Optional[str] = None
    signal_meta: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.symbol, self.entry_time))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    commission: float = 0.0
    mode: str = "paper"
    signal_confidence: Optional[float] = None
    signal_reasons: List[str] = field(default_factory=list)
    signal_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EquityPoint:
    """Point on the equity curve."""

    timestamp: str
    balance: float
    positions_value: float
    total_equity: float
    unrealized_pnl: float = 0.0


@dataclass
class UnifiedState:
    """Complete unified state for the trading engine."""

    # Core state
    mode: TradingMode
    status: TradingStatus
    timestamp: str

    # Balance
    initial_capital: float
    current_balance: float
    peak_balance: float

    # Positions
    positions: Dict[str, PositionState] = field(default_factory=dict)

    # Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0

    # Mode tracking
    mode_started_at: Optional[str] = None
    days_in_mode: int = 0

    # Daily stats
    daily_trades: int = 0
    daily_pnl: float = 0.0
    daily_date: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "initial_capital": self.initial_capital,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "mode_started_at": self.mode_started_at,
            "days_in_mode": self.days_in_mode,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "daily_date": self.daily_date,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UnifiedState":
        """Create from dictionary."""
        positions = {k: PositionState.from_dict(v) for k, v in data.get("positions", {}).items()}
        return cls(
            mode=TradingMode(data["mode"]),
            status=TradingStatus(data.get("status", "stopped")),
            timestamp=data["timestamp"],
            initial_capital=data["initial_capital"],
            current_balance=data["current_balance"],
            peak_balance=data.get("peak_balance", data["current_balance"]),
            positions=positions,
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            total_pnl=data.get("total_pnl", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            mode_started_at=data.get("mode_started_at"),
            days_in_mode=data.get("days_in_mode", 0),
            daily_trades=data.get("daily_trades", 0),
            daily_pnl=data.get("daily_pnl", 0.0),
            daily_date=data.get("daily_date", ""),
        )

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_equity(self) -> float:
        """Calculate total equity including unrealized P&L."""
        positions_value = sum(
            pos.quantity * (pos.current_price or pos.entry_price) for pos in self.positions.values()
        )
        return self.current_balance + positions_value


class UnifiedStateStore:
    """
    Unified state store for all trading data.

    Handles persistence for:
    - Current state (balance, positions)
    - Trade history
    - Equity curve
    - Mode state

    Features:
    - Sync JSON for quick state recovery on restart
    - Async SQLite database for trade analytics (via AsyncTradingDatabase)
    - Background async writes don't block trading loop
    """

    def __init__(
        self,
        data_dir: Path = Path("data/unified_trading"),
        max_trades: int = 1000,
        max_equity_points: int = 10000,
        use_async_db: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_trades = max_trades
        self.max_equity_points = max_equity_points
        self.use_async_db = use_async_db and ASYNC_DB_AVAILABLE

        self._lock = Lock()
        self._state: Optional[UnifiedState] = None
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[EquityPoint] = []
        self._mode_history: List[Dict] = []

        # Async database for high-performance trade storage
        self._async_db: Optional[AsyncTradingDatabase] = None
        self._async_db_initialized = False
        self._pending_async_tasks: List[asyncio.Task] = []

        # File paths
        self.state_path = self.data_dir / "state.json"
        self.trades_path = self.data_dir / "trades.json"
        self.equity_path = self.data_dir / "equity.json"
        self.mode_history_path = self.data_dir / "mode_history.json"

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        mode: TradingMode,
        initial_capital: float,
        resume: bool = True,
    ) -> UnifiedState:
        """
        Initialize state store.

        If resume=True and state exists, load it. Otherwise create new state.
        """
        with self._lock:
            if resume and self.state_path.exists():
                try:
                    self._load_all()
                    if self._state:
                        logger.info(
                            f"Resumed state from disk: mode={self._state.mode.value}, "
                            f"balance=${self._state.current_balance:.2f}"
                        )
                        return self._state
                except Exception as e:
                    logger.warning(f"Failed to load state: {e}")

            # Create new state
            self._state = UnifiedState(
                mode=mode,
                status=TradingStatus.STOPPED,
                timestamp=datetime.now().isoformat(),
                initial_capital=initial_capital,
                current_balance=initial_capital,
                peak_balance=initial_capital,
                mode_started_at=datetime.now().isoformat(),
                daily_date=datetime.now().strftime("%Y-%m-%d"),
            )
            self._save_state()

            logger.info(f"Initialized new state: mode={mode.value}, capital=${initial_capital:.2f}")
            return self._state

    def _load_all(self) -> None:
        """Load all persisted data."""
        # Load state
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data: Dict[str, Any] = json.load(f)
                    self._state = UnifiedState.from_dict(data)
            except (OSError, json.JSONDecodeError, ValueError):
                logger.warning(f"Failed to load state from {self.state_path}")

        # Load trades
        if self.trades_path.exists():
            try:
                with open(self.trades_path, "r") as f:
                    data = json.load(f)
                    self._trades = [TradeRecord.from_dict(t) for t in data]
            except (OSError, json.JSONDecodeError, ValueError):
                logger.warning(f"Failed to load trades from {self.trades_path}")

        # Load equity curve
        if self.equity_path.exists():
            try:
                with open(self.equity_path, "r") as f:
                    data = json.load(f)
                    self._equity_curve = [EquityPoint(**p) for p in data[-self.max_equity_points :]]
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                logger.warning(f"Failed to load equity curve from {self.equity_path}")

        # Load mode history
        if self.mode_history_path.exists():
            try:
                with open(self.mode_history_path, "r") as f:
                    self._mode_history = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to load mode history from {self.mode_history_path}")

    def _save_state(self) -> None:
        """Save current state to disk."""
        if self._state:
            with open(self.state_path, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)

    def _save_trades(self) -> None:
        """Save trade history to disk."""
        with open(self.trades_path, "w") as f:
            json.dump([t.to_dict() for t in self._trades[-self.max_trades :]], f, indent=2)

    def _save_equity(self) -> None:
        """Save equity curve to disk."""
        with open(self.equity_path, "w") as f:
            points = [asdict(p) for p in self._equity_curve[-self.max_equity_points :]]
            json.dump(points, f, indent=2)

    def _save_mode_history(self) -> None:
        """Save mode transition history."""
        with open(self.mode_history_path, "w") as f:
            json.dump(self._mode_history, f, indent=2)

    def get_state(self) -> Optional[UnifiedState]:
        """Get current state."""
        with self._lock:
            return self._state

    def update_state(self, **kwargs) -> UnifiedState:
        """Update state with new values."""
        with self._lock:
            if not self._state:
                raise RuntimeError("State not initialized")

            self._state.timestamp = datetime.now().isoformat()

            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

            # Update peak balance
            if self._state.current_balance > self._state.peak_balance:
                self._state.peak_balance = self._state.current_balance

            # Calculate drawdown
            if self._state.peak_balance > 0:
                drawdown = (
                    self._state.peak_balance - self._state.current_balance
                ) / self._state.peak_balance
                if drawdown > self._state.max_drawdown_pct:
                    self._state.max_drawdown_pct = drawdown

            # Check for new day
            today = datetime.now().strftime("%Y-%m-%d")
            if today != self._state.daily_date:
                self._state.daily_date = today
                self._state.daily_trades = 0
                self._state.daily_pnl = 0.0

            self._save_state()
            return self._state

    def update_position(self, symbol: str, position: Optional[PositionState]) -> None:
        """Update or remove a position."""
        with self._lock:
            if not self._state:
                raise RuntimeError("State not initialized")

            if position is None:
                if symbol in self._state.positions:
                    del self._state.positions[symbol]
            else:
                self._state.positions[symbol] = position

            self._state.timestamp = datetime.now().isoformat()
            self._save_state()

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade."""
        with self._lock:
            if not self._state:
                raise RuntimeError("State not initialized")

            self._trades.append(trade)

            # Update state stats
            self._state.total_trades += 1
            self._state.daily_trades += 1
            self._state.total_pnl += trade.pnl
            self._state.daily_pnl += trade.pnl

            if trade.pnl >= 0:
                self._state.winning_trades += 1
            else:
                self._state.losing_trades += 1

            # Trim trades list if needed
            if len(self._trades) > self.max_trades:
                self._trades = self._trades[-self.max_trades :]

            self._save_trades()
            self._save_state()

    def record_equity_point(self) -> None:
        """Record current equity point."""
        with self._lock:
            if not self._state:
                return

            positions_value = sum(
                pos.quantity * (pos.current_price or pos.entry_price)
                for pos in self._state.positions.values()
            )
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self._state.positions.values())

            point = EquityPoint(
                timestamp=datetime.now().isoformat(),
                balance=self._state.current_balance,
                positions_value=positions_value,
                total_equity=self._state.current_balance + positions_value,
                unrealized_pnl=unrealized_pnl,
            )
            self._equity_curve.append(point)

            # Trim if needed
            if len(self._equity_curve) > self.max_equity_points:
                self._equity_curve = self._equity_curve[-self.max_equity_points :]

            self._save_equity()

    def change_mode(self, new_mode: TradingMode, reason: str = "", approver: str = "") -> None:
        """Record a mode change."""
        with self._lock:
            if not self._state:
                raise RuntimeError("State not initialized")

            old_mode = self._state.mode

            # Record in history
            self._mode_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "from_mode": old_mode.value,
                    "to_mode": new_mode.value,
                    "reason": reason,
                    "approver": approver,
                    "stats_at_transition": {
                        "total_trades": self._state.total_trades,
                        "win_rate": self._state.win_rate,
                        "total_pnl": self._state.total_pnl,
                        "max_drawdown": self._state.max_drawdown_pct,
                        "days_in_mode": self._state.days_in_mode,
                    },
                }
            )

            # Update state
            self._state.mode = new_mode
            self._state.mode_started_at = datetime.now().isoformat()
            self._state.days_in_mode = 0

            self._save_state()
            self._save_mode_history()

            logger.info(f"Mode changed: {old_mode.value} -> {new_mode.value} (reason: {reason})")

    def get_trades(self, limit: int = 100) -> List[TradeRecord]:
        """Get recent trades."""
        with self._lock:
            return self._trades[-limit:]

    def get_equity_curve(self, limit: int = 1000) -> List[EquityPoint]:
        """Get recent equity points."""
        with self._lock:
            return self._equity_curve[-limit:]

    def get_mode_history(self) -> List[Dict[str, Any]]:
        """Get mode transition history."""
        with self._lock:
            return self._mode_history.copy()

    def get_mode_state(self) -> ModeState:
        """Get current mode state for transition validation."""
        with self._lock:
            if not self._state:
                raise RuntimeError("State not initialized")

            mode_state = ModeState(
                mode=self._state.mode,
                status=self._state.status,
                total_trades=self._state.total_trades,
                winning_trades=self._state.winning_trades,
                losing_trades=self._state.losing_trades,
                total_pnl=self._state.total_pnl,
                peak_balance=self._state.peak_balance,
                max_drawdown_pct=self._state.max_drawdown_pct,
                daily_trades=self._state.daily_trades,
                daily_pnl=self._state.daily_pnl,
            )

            # Calculate days in mode
            if self._state.mode_started_at:
                started = datetime.fromisoformat(self._state.mode_started_at)
                mode_state.started_at = started
            else:
                mode_state.started_at = datetime.now()

            return mode_state

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        with self._lock:
            if not self._state:
                return {}

            return {
                "mode": self._state.mode.value,
                "status": self._state.status.value,
                "balance": self._state.current_balance,
                "initial_capital": self._state.initial_capital,
                "total_pnl": self._state.total_pnl,
                "total_pnl_pct": (
                    self._state.total_pnl / self._state.initial_capital * 100
                    if self._state.initial_capital > 0
                    else 0
                ),
                "total_trades": self._state.total_trades,
                "win_rate": self._state.win_rate * 100,
                "max_drawdown": self._state.max_drawdown_pct * 100,
                "days_in_mode": self._state.days_in_mode,
                "open_positions": len(self._state.positions),
                "daily_trades": self._state.daily_trades,
                "daily_pnl": self._state.daily_pnl,
            }

    # ========== Async Database Methods ==========

    async def initialize_async_db(self) -> bool:
        """
        Initialize async database for high-performance trade storage.

        Returns True if initialized successfully.
        """
        if not self.use_async_db:
            return False

        if self._async_db_initialized:
            return True

        try:
            db_path = str(self.data_dir / "portfolio.db")
            self._async_db = await get_async_database(db_path)
            self._async_db_initialized = True
            logger.info(f"Async database initialized: {db_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize async database: {e}")
            self.use_async_db = False
            return False

    async def record_trade_async(self, trade: TradeRecord) -> Optional[int]:
        """
        Record trade to async database (non-blocking).

        Returns the trade ID if successful.
        """
        if not self._async_db or not self._async_db_initialized:
            return None

        try:
            trade_data = {
                "symbol": trade.symbol,
                "direction": trade.side,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "size": trade.quantity,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "commission": trade.commission,
                "slippage": 0,
                "exit_reason": trade.exit_reason,
                "strategy": "unified_engine",
                "regime": trade.signal_metadata.get("regime", "unknown"),
                "confidence": trade.signal_confidence,
                "metadata": trade.signal_metadata,
            }
            trade_id = await self._async_db.insert_trade(trade_data)
            logger.debug(f"Trade recorded to async DB: id={trade_id}")
            return trade_id
        except Exception as e:
            logger.warning(f"Failed to record trade to async DB: {e}")
            return None

    async def record_equity_async(self) -> None:
        """Record equity snapshot to async database."""
        if not self._async_db or not self._async_db_initialized or not self._state:
            return

        try:
            positions_value = sum(
                pos.quantity * (pos.current_price or pos.entry_price)
                for pos in self._state.positions.values()
            )
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self._state.positions.values())

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "balance": self._state.current_balance,
                "equity": self._state.current_balance + positions_value,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": self._state.total_pnl,
                "open_positions": len(self._state.positions),
                "mode": self._state.mode.value,
            }
            await self._async_db.insert_equity_snapshot(snapshot)
        except Exception as e:
            logger.debug(f"Failed to record equity to async DB: {e}")

    async def get_trade_stats_async(self, days: int = 30) -> Dict[str, Any]:
        """Get trade statistics from async database."""
        if not self._async_db or not self._async_db_initialized:
            return {}

        try:
            return await self._async_db.get_trade_stats(days)
        except Exception as e:
            logger.warning(f"Failed to get trade stats: {e}")
            return {}

    async def close_async_db(self) -> None:
        """Close async database connection pool."""
        if self._async_db:
            try:
                await self._async_db.close()
                self._async_db_initialized = False
                logger.info("Async database closed")
            except Exception as e:
                logger.warning(f"Error closing async database: {e}")
