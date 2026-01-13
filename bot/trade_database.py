"""
Trade Database Module.

SQLite-based storage for trades, signals, and performance metrics.
Provides efficient querying and historical analysis capabilities.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade status enum."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    PARTIALLY_CLOSED = "partially_closed"


class SignalType(Enum):
    """Signal type enum."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradeRecord:
    """Trade record for database storage."""
    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""  # LONG or SHORT
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"
    strategy: str = ""
    regime: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: Optional[str] = None
    confidence: float = 0.0
    fees: float = 0.0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.entry_time:
            result["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            result["exit_time"] = self.exit_time.isoformat()
        return result


@dataclass
class SignalRecord:
    """Signal record for database storage."""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    symbol: str = ""
    signal_type: str = ""  # LONG, SHORT, FLAT
    confidence: float = 0.0
    strategy: str = ""
    regime: str = ""
    price: float = 0.0
    indicators: Optional[Dict] = None
    reason: str = ""
    executed: bool = False


@dataclass
class EquityRecord:
    """Equity point for database storage."""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    market_type: str = ""  # crypto, stock, commodity, forex
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    drawdown: float = 0.0


class TradeDatabase:
    """
    SQLite database for trading data.

    Features:
    - Trade logging with full lifecycle tracking
    - Signal history with indicators
    - Equity curve storage
    - Performance metrics calculation
    - Efficient querying with indices
    """

    def __init__(self, db_path: str = "data/trades.db"):
        """
        Initialize trade database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL,
                    pnl_pct REAL,
                    status TEXT DEFAULT 'open',
                    strategy TEXT,
                    regime TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    exit_reason TEXT,
                    confidence REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    strategy TEXT,
                    regime TEXT,
                    price REAL,
                    indicators TEXT,
                    reason TEXT,
                    executed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Equity table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    market_type TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL DEFAULT 0,
                    positions_value REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    drawdown REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    market_type TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL,
                    profit_factor REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, market_type)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_market ON equity(market_type)")

            logger.info(f"Trade database initialized at {self.db_path}")

    # Trade operations
    def insert_trade(self, trade: TradeRecord) -> int:
        """Insert a new trade record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    symbol, direction, entry_time, exit_time, entry_price,
                    exit_price, quantity, pnl, pnl_pct, status, strategy,
                    regime, stop_loss, take_profit, exit_reason, confidence,
                    fees, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.direction, trade.entry_time, trade.exit_time,
                trade.entry_price, trade.exit_price, trade.quantity, trade.pnl,
                trade.pnl_pct, trade.status, trade.strategy, trade.regime,
                trade.stop_loss, trade.take_profit, trade.exit_reason,
                trade.confidence, trade.fees,
                json.dumps(trade.metadata) if trade.metadata else None
            ))
            return cursor.lastrowid

    def update_trade(self, trade_id: int, **kwargs) -> bool:
        """Update an existing trade."""
        if not kwargs:
            return False

        # Handle metadata serialization
        if "metadata" in kwargs and kwargs["metadata"] is not None:
            kwargs["metadata"] = json.dumps(kwargs["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values())
        values.append(trade_id)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE trades SET {set_clause} WHERE id = ?",
                values
            )
            return cursor.rowcount > 0

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str = "manual",
        fees: float = 0.0,
    ) -> bool:
        """Close an open trade."""
        trade = self.get_trade(trade_id)
        if not trade or trade.status != "open":
            return False

        exit_time = datetime.now(timezone.utc)

        # Calculate P&L
        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.quantity - fees
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity - fees

        pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100

        return self.update_trade(
            trade_id,
            exit_time=exit_time,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            status="closed",
            exit_reason=exit_reason,
            fees=fees,
        )

    def get_trade(self, trade_id: int) -> Optional[TradeRecord]:
        """Get a trade by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_trade(row)
        return None

    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TradeRecord]:
        """Query trades with filters."""
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if start_date:
            conditions.append("entry_time >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("entry_time <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT * FROM trades WHERE {where_clause}
                   ORDER BY entry_time DESC LIMIT ? OFFSET ?""",
                params + [limit, offset]
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_open_trades(self, symbol: Optional[str] = None) -> List[TradeRecord]:
        """Get all open trades."""
        return self.get_trades(symbol=symbol, status="open", limit=1000)

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """Convert database row to TradeRecord."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        return TradeRecord(
            id=row["id"],
            symbol=row["symbol"],
            direction=row["direction"],
            entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            quantity=row["quantity"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            status=row["status"],
            strategy=row["strategy"],
            regime=row["regime"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            exit_reason=row["exit_reason"],
            confidence=row["confidence"],
            fees=row["fees"],
            metadata=metadata,
        )

    # Signal operations
    def insert_signal(self, signal: SignalRecord) -> int:
        """Insert a new signal record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, signal_type, confidence, strategy,
                    regime, price, indicators, reason, executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.timestamp, signal.symbol, signal.signal_type,
                signal.confidence, signal.strategy, signal.regime,
                signal.price,
                json.dumps(signal.indicators) if signal.indicators else None,
                signal.reason, 1 if signal.executed else 0
            ))
            return cursor.lastrowid

    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SignalRecord]:
        """Query signals with filters."""
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if signal_type:
            conditions.append("signal_type = ?")
            params.append(signal_type)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT * FROM signals WHERE {where_clause}
                   ORDER BY timestamp DESC LIMIT ?""",
                params + [limit]
            )
            return [self._row_to_signal(row) for row in cursor.fetchall()]

    def _row_to_signal(self, row: sqlite3.Row) -> SignalRecord:
        """Convert database row to SignalRecord."""
        indicators = json.loads(row["indicators"]) if row["indicators"] else None
        return SignalRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            symbol=row["symbol"],
            signal_type=row["signal_type"],
            confidence=row["confidence"],
            strategy=row["strategy"],
            regime=row["regime"],
            price=row["price"],
            indicators=indicators,
            reason=row["reason"],
            executed=bool(row["executed"]),
        )

    # Equity operations
    def insert_equity(self, equity: EquityRecord) -> int:
        """Insert an equity record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO equity (
                    timestamp, market_type, total_value, cash_balance,
                    positions_value, unrealized_pnl, realized_pnl, drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                equity.timestamp, equity.market_type, equity.total_value,
                equity.cash_balance, equity.positions_value,
                equity.unrealized_pnl, equity.realized_pnl, equity.drawdown
            ))
            return cursor.lastrowid

    def get_equity_curve(
        self,
        market_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EquityRecord]:
        """Get equity curve data."""
        conditions = []
        params = []

        if market_type:
            conditions.append("market_type = ?")
            params.append(market_type)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT * FROM equity WHERE {where_clause}
                   ORDER BY timestamp ASC LIMIT ?""",
                params + [limit]
            )
            return [self._row_to_equity(row) for row in cursor.fetchall()]

    def _row_to_equity(self, row: sqlite3.Row) -> EquityRecord:
        """Convert database row to EquityRecord."""
        return EquityRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            market_type=row["market_type"],
            total_value=row["total_value"],
            cash_balance=row["cash_balance"],
            positions_value=row["positions_value"],
            unrealized_pnl=row["unrealized_pnl"],
            realized_pnl=row["realized_pnl"],
            drawdown=row["drawdown"],
        )

    # Performance metrics
    def calculate_metrics(
        self,
        market_type: str = "all",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """Calculate performance metrics for closed trades."""
        conditions = ["status = 'closed'"]
        params = []

        if market_type != "all":
            # Infer market from symbol patterns
            pass  # Would need symbol mapping

        if start_date:
            conditions.append("exit_time >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("exit_time <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Basic metrics
            cursor.execute(
                f"""SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade,
                    SUM(fees) as total_fees
                FROM trades WHERE {where_clause}""",
                params
            )
            row = cursor.fetchone()

            if not row or row["total_trades"] == 0:
                return {"error": "No trades found"}

            total_trades = row["total_trades"]
            winning = row["winning"] or 0
            losing = row["losing"] or 0
            total_pnl = row["total_pnl"] or 0
            avg_win = row["avg_win"] or 0
            avg_loss = abs(row["avg_loss"]) if row["avg_loss"] else 0

            win_rate = winning / total_trades if total_trades > 0 else 0
            profit_factor = (avg_win * winning) / (avg_loss * losing) if losing > 0 and avg_loss > 0 else float("inf")

            return {
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(win_rate, 4),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(row["avg_pnl"] or 0, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "best_trade": round(row["best_trade"] or 0, 2),
                "worst_trade": round(row["worst_trade"] or 0, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
                "total_fees": round(row["total_fees"] or 0, 2),
            }

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict:
        """Get daily trading summary."""
        if date is None:
            date = datetime.now(timezone.utc).date()
        else:
            date = date.date()

        start = datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        return self.calculate_metrics(start_date=start, end_date=end)


# Singleton instance
_db_instance: Optional[TradeDatabase] = None


def get_trade_database(db_path: str = "data/trades.db") -> TradeDatabase:
    """Get or create the trade database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradeDatabase(db_path)
    return _db_instance
