"""
SQLite Database Storage Module.

Provides persistent storage for:
- Trade history
- Equity curve
- Signals
- ML model metadata
- Performance metrics
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class Trade:
    """Trade record."""

    id: Optional[int] = None
    symbol: str = ""
    direction: str = ""  # LONG or SHORT
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    strategy: str = ""
    regime: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "exit_reason": self.exit_reason,
            "strategy": self.strategy,
            "regime": self.regime,
            "confidence": self.confidence,
        }


class TradingDatabase:
    """
    SQLite database for trading data persistence.

    Tables:
    - trades: Complete trade history
    - equity: Equity curve snapshots
    - signals: All generated signals
    - metrics: Daily/weekly/monthly performance metrics
    - models: ML model metadata and performance
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables."""
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
                    size REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    exit_reason TEXT,
                    strategy TEXT,
                    regime TEXT,
                    confidence REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Equity curve table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    balance REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    total_equity REAL NOT NULL,
                    drawdown_pct REAL DEFAULT 0,
                    positions_count INTEGER DEFAULT 0,
                    symbol TEXT
                )
            """)

            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    price REAL,
                    regime TEXT,
                    strategy TEXT,
                    ml_action TEXT,
                    ml_confidence REAL,
                    executed INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)

            # Daily metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    starting_balance REAL,
                    ending_balance REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    trades_count INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown_pct REAL,
                    sharpe_ratio REAL,
                    volume REAL
                )
            """)

            # ML models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    symbol TEXT,
                    trained_at TIMESTAMP NOT NULL,
                    accuracy REAL,
                    cross_val_score REAL,
                    train_samples INTEGER,
                    test_samples INTEGER,
                    feature_count INTEGER,
                    parameters TEXT,
                    performance_oos TEXT,
                    file_path TEXT,
                    is_active INTEGER DEFAULT 0
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")

    # ========================
    # Trade Methods
    # ========================

    def insert_trade(self, trade: Trade) -> int:
        """Insert a new trade and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    symbol, direction, entry_time, exit_time, entry_price,
                    exit_price, size, pnl, pnl_pct, commission, slippage,
                    exit_reason, strategy, regime, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.symbol,
                    trade.direction,
                    trade.entry_time,
                    trade.exit_time,
                    trade.entry_price,
                    trade.exit_price,
                    trade.size,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.commission,
                    trade.slippage,
                    trade.exit_reason,
                    trade.strategy,
                    trade.regime,
                    trade.confidence,
                    json.dumps(trade.metadata) if trade.metadata else None,
                ),
            )
            return cursor.lastrowid

    # Valid column names for Trade table (prevents SQL injection)
    _VALID_TRADE_COLUMNS = frozenset(
        {
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "size",
            "pnl",
            "pnl_pct",
            "commission",
            "slippage",
            "exit_reason",
            "strategy",
            "regime",
            "confidence",
            "metadata",
        }
    )

    def update_trade(self, trade_id: int, **updates) -> bool:
        """Update a trade by ID."""
        if not updates:
            return False

        # Validate column names against whitelist to prevent SQL injection
        invalid_columns = set(updates.keys()) - self._VALID_TRADE_COLUMNS
        if invalid_columns:
            raise ValueError(f"Invalid column names: {invalid_columns}")

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [trade_id]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE trades SET {set_clause} WHERE id = ?", values)
            return cursor.rowcount > 0

    def get_trade(self, trade_id: int) -> Optional[Trade]:
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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Trade]:
        """Get trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)

        # Ensure limit is a valid integer to prevent SQL injection
        safe_limit = int(limit)
        if safe_limit < 0:
            safe_limit = 1000
        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(safe_limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_trade_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get trade statistics."""
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = start_date.replace(day=start_date.day - days)

        trades = self.get_trades(symbol=symbol, start_date=start_date, limit=10000)

        if not trades:
            return {}

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "total_pnl": sum(pnls),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "profit_factor": sum(wins) / sum(losses) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": max(losses) if losses else 0,
        }

    def _row_to_trade(self, row: sqlite3.Row) -> Trade:
        """Convert database row to Trade object."""
        return Trade(
            id=row["id"],
            symbol=row["symbol"],
            direction=row["direction"],
            entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            size=row["size"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            commission=row["commission"],
            slippage=row["slippage"],
            exit_reason=row["exit_reason"],
            strategy=row["strategy"],
            regime=row["regime"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )

    # ========================
    # Equity Methods
    # ========================

    def insert_equity_point(
        self,
        timestamp: datetime,
        balance: float,
        unrealized_pnl: float = 0.0,
        positions_count: int = 0,
        symbol: Optional[str] = None,
    ):
        """Insert an equity curve point."""
        total_equity = balance + unrealized_pnl

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO equity (
                    timestamp, balance, unrealized_pnl, total_equity,
                    positions_count, symbol
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (timestamp, balance, unrealized_pnl, total_equity, positions_count, symbol),
            )

    def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        """Get equity curve data."""
        query = "SELECT * FROM equity WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += f" ORDER BY timestamp ASC LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ========================
    # Signal Methods
    # ========================

    def insert_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        ml_action: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Insert a trading signal."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signals (
                    timestamp, symbol, signal, confidence, price, regime,
                    strategy, ml_action, ml_confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    symbol,
                    signal,
                    confidence,
                    price,
                    regime,
                    strategy,
                    ml_action,
                    ml_confidence,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def mark_signal_executed(self, signal_id: int):
        """Mark a signal as executed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE signals SET executed = 1 WHERE id = ?", (signal_id,))

    def get_signals(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """Get signals with optional filters."""
        query = "SELECT * FROM signals WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ========================
    # Daily Metrics Methods
    # ========================

    def upsert_daily_metrics(
        self,
        date: datetime,
        metrics: Dict[str, Any],
    ):
        """Insert or update daily metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_metrics (
                    date, starting_balance, ending_balance, pnl, pnl_pct,
                    trades_count, winning_trades, losing_trades, win_rate,
                    profit_factor, max_drawdown_pct, sharpe_ratio, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date.date() if isinstance(date, datetime) else date,
                    metrics.get("starting_balance"),
                    metrics.get("ending_balance"),
                    metrics.get("pnl"),
                    metrics.get("pnl_pct"),
                    metrics.get("trades_count"),
                    metrics.get("winning_trades"),
                    metrics.get("losing_trades"),
                    metrics.get("win_rate"),
                    metrics.get("profit_factor"),
                    metrics.get("max_drawdown_pct"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("volume"),
                ),
            )

    def get_daily_metrics(self, days: int = 30) -> List[Dict]:
        """Get recent daily metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM daily_metrics
                ORDER BY date DESC
                LIMIT {days}
            """)
            return [dict(row) for row in cursor.fetchall()]

    # ========================
    # ML Model Methods
    # ========================

    def insert_model(
        self,
        name: str,
        model_type: str,
        symbol: Optional[str],
        accuracy: float,
        cross_val_score: float,
        train_samples: int,
        test_samples: int,
        feature_count: int,
        parameters: Dict,
        file_path: str,
    ) -> int:
        """Insert ML model record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ml_models (
                    name, model_type, symbol, trained_at, accuracy,
                    cross_val_score, train_samples, test_samples,
                    feature_count, parameters, file_path, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
                (
                    name,
                    model_type,
                    symbol,
                    datetime.now(),
                    accuracy,
                    cross_val_score,
                    train_samples,
                    test_samples,
                    feature_count,
                    json.dumps(parameters),
                    file_path,
                ),
            )
            return cursor.lastrowid

    def get_active_model(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Get the active ML model."""
        query = "SELECT * FROM ml_models WHERE is_active = 1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY trained_at DESC LIMIT 1"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_model_history(self, limit: int = 20) -> List[Dict]:
        """Get ML model training history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM ml_models
                ORDER BY trained_at DESC
                LIMIT {limit}
            """)
            return [dict(row) for row in cursor.fetchall()]

    # ========================
    # Utility Methods
    # ========================

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a custom query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}
            for table in ["trades", "equity", "signals", "daily_metrics", "ml_models"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            return stats

    def vacuum(self):
        """Optimize database by running VACUUM."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def backup(self, backup_path: str):
        """Create database backup."""
        import shutil

        shutil.copy2(self.db_path, backup_path)
