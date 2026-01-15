"""
Trade Ledger - SQLite Audit Trail

A comprehensive trade logging system that stores every trade
for audit, analysis, and AI learning.

Features:
- Complete trade history with all execution details
- Performance attribution by strategy, regime, leverage
- AI learning data export
- Risk event logging
- Automatic data retention and cleanup
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class TradeType(Enum):
    """Trade type classification"""
    ENTRY = "entry"
    EXIT = "exit"
    INCREASE = "increase"
    DECREASE = "decrease"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    LIQUIDATION = "liquidation"


class RiskEventType(Enum):
    """Risk event types"""
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MARGIN_WARNING = "margin_warning"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRADE_REJECTED = "trade_rejected"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class TradeRecord:
    """Complete trade record for ledger"""
    # Identifiers
    trade_id: str
    order_id: str
    symbol: str

    # Execution details
    side: str  # long, short
    trade_type: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    leverage: float

    # PnL
    realized_pnl: float
    realized_pnl_pct: float
    fees_paid: float
    slippage_cost: float

    # Timing
    entry_time: datetime
    exit_time: Optional[datetime]
    holding_duration_seconds: Optional[int]

    # Market context
    regime: str  # trending, ranging, volatile
    volatility: float
    volume_24h: float

    # Strategy context
    strategy_name: str
    strategy_version: str
    signal_strength: float
    confidence: float

    # AI context
    ai_recommended: bool
    ai_confidence: float
    rl_action: Optional[str]
    meta_allocation_pct: float

    # Execution mode
    execution_mode: str  # backtest, paper, live

    # Outcome
    outcome: str  # win, loss, breakeven
    max_favorable_excursion: float
    max_adverse_excursion: float

    # Tags and metadata
    tags: str  # JSON string
    metadata: str  # JSON string


@dataclass
class RiskEventRecord:
    """Risk event record for audit"""
    event_id: str
    event_type: str
    timestamp: datetime
    severity: str  # info, warning, critical

    # Context
    equity_before: float
    equity_after: float
    drawdown_pct: float
    daily_pnl_pct: float

    # Details
    message: str
    triggered_by: str
    action_taken: str

    # Related trades
    related_trade_ids: str  # JSON array

    # Metadata
    metadata: str  # JSON string


class TradeLedger:
    """
    SQLite-based trade ledger for comprehensive audit trail.
    """

    def __init__(
        self,
        db_path: str = "data/trade_ledger.db",
        retention_days: int = 365,
        auto_vacuum: bool = True
    ):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.auto_vacuum = auto_vacuum

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize database
        self._init_db()

        logger.info(f"Trade Ledger initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for cursor with automatic commit"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

    def _init_db(self):
        """Initialize database schema"""
        with self._cursor() as cursor:
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,

                    side TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    leverage REAL DEFAULT 1.0,

                    realized_pnl REAL DEFAULT 0.0,
                    realized_pnl_pct REAL DEFAULT 0.0,
                    fees_paid REAL DEFAULT 0.0,
                    slippage_cost REAL DEFAULT 0.0,

                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    holding_duration_seconds INTEGER,

                    regime TEXT,
                    volatility REAL,
                    volume_24h REAL,

                    strategy_name TEXT,
                    strategy_version TEXT,
                    signal_strength REAL,
                    confidence REAL,

                    ai_recommended INTEGER DEFAULT 0,
                    ai_confidence REAL DEFAULT 0.0,
                    rl_action TEXT,
                    meta_allocation_pct REAL DEFAULT 0.0,

                    execution_mode TEXT NOT NULL,
                    outcome TEXT,
                    max_favorable_excursion REAL DEFAULT 0.0,
                    max_adverse_excursion REAL DEFAULT 0.0,

                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Risk events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    severity TEXT NOT NULL,

                    equity_before REAL,
                    equity_after REAL,
                    drawdown_pct REAL,
                    daily_pnl_pct REAL,

                    message TEXT,
                    triggered_by TEXT,
                    action_taken TEXT,

                    related_trade_ids TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance snapshots table (daily)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    execution_mode TEXT NOT NULL,

                    starting_equity REAL,
                    ending_equity REAL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,

                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,

                    gross_profit REAL DEFAULT 0.0,
                    gross_loss REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,

                    max_drawdown REAL DEFAULT 0.0,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,

                    avg_leverage REAL DEFAULT 1.0,
                    total_fees REAL DEFAULT 0.0,

                    best_trade_pnl REAL DEFAULT 0.0,
                    worst_trade_pnl REAL DEFAULT 0.0,
                    avg_trade_duration_seconds INTEGER,

                    by_regime TEXT DEFAULT '{}',
                    by_strategy TEXT DEFAULT '{}',
                    by_leverage TEXT DEFAULT '{}',

                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # AI learning data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,

                    state_features TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    reward REAL NOT NULL,
                    next_state_features TEXT,

                    is_terminal INTEGER DEFAULT 0,
                    episode_id TEXT,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(execution_mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_events_time ON risk_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_learning_trade ON ai_learning_data(trade_id)")

    def record_trade(self, trade: TradeRecord) -> bool:
        """Record a trade to the ledger"""
        try:
            with self._cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, order_id, symbol,
                        side, trade_type, quantity, entry_price, exit_price, leverage,
                        realized_pnl, realized_pnl_pct, fees_paid, slippage_cost,
                        entry_time, exit_time, holding_duration_seconds,
                        regime, volatility, volume_24h,
                        strategy_name, strategy_version, signal_strength, confidence,
                        ai_recommended, ai_confidence, rl_action, meta_allocation_pct,
                        execution_mode, outcome, max_favorable_excursion, max_adverse_excursion,
                        tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id, trade.order_id, trade.symbol,
                    trade.side, trade.trade_type, trade.quantity,
                    trade.entry_price, trade.exit_price, trade.leverage,
                    trade.realized_pnl, trade.realized_pnl_pct,
                    trade.fees_paid, trade.slippage_cost,
                    trade.entry_time.isoformat(), trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.holding_duration_seconds,
                    trade.regime, trade.volatility, trade.volume_24h,
                    trade.strategy_name, trade.strategy_version,
                    trade.signal_strength, trade.confidence,
                    1 if trade.ai_recommended else 0, trade.ai_confidence,
                    trade.rl_action, trade.meta_allocation_pct,
                    trade.execution_mode, trade.outcome,
                    trade.max_favorable_excursion, trade.max_adverse_excursion,
                    trade.tags, trade.metadata
                ))
            logger.debug(f"Trade recorded: {trade.trade_id}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Trade already exists: {trade.trade_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return False

    def record_risk_event(self, event: RiskEventRecord) -> bool:
        """Record a risk event"""
        try:
            with self._cursor() as cursor:
                cursor.execute("""
                    INSERT INTO risk_events (
                        event_id, event_type, timestamp, severity,
                        equity_before, equity_after, drawdown_pct, daily_pnl_pct,
                        message, triggered_by, action_taken,
                        related_trade_ids, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id, event.event_type,
                    event.timestamp.isoformat(), event.severity,
                    event.equity_before, event.equity_after,
                    event.drawdown_pct, event.daily_pnl_pct,
                    event.message, event.triggered_by, event.action_taken,
                    event.related_trade_ids, event.metadata
                ))
            logger.info(f"Risk event recorded: {event.event_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to record risk event: {e}")
            return False

    def record_ai_learning_data(
        self,
        trade_id: str,
        state_features: Dict[str, float],
        action_taken: str,
        reward: float,
        next_state_features: Optional[Dict[str, float]] = None,
        is_terminal: bool = False,
        episode_id: Optional[str] = None
    ) -> bool:
        """Record AI learning experience"""
        try:
            with self._cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ai_learning_data (
                        trade_id, timestamp, state_features, action_taken,
                        reward, next_state_features, is_terminal, episode_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, datetime.now().isoformat(),
                    json.dumps(state_features), action_taken, reward,
                    json.dumps(next_state_features) if next_state_features else None,
                    1 if is_terminal else 0, episode_id
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to record AI learning data: {e}")
            return False

    def update_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float,
        realized_pnl_pct: float,
        outcome: str,
        max_favorable: float = 0.0,
        max_adverse: float = 0.0
    ) -> bool:
        """Update trade with exit information"""
        try:
            with self._cursor() as cursor:
                # Get entry time for duration calculation
                cursor.execute(
                    "SELECT entry_time FROM trades WHERE trade_id = ?",
                    (trade_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                entry_time = datetime.fromisoformat(row["entry_time"])
                duration = int((exit_time - entry_time).total_seconds())

                cursor.execute("""
                    UPDATE trades SET
                        exit_price = ?,
                        exit_time = ?,
                        holding_duration_seconds = ?,
                        realized_pnl = ?,
                        realized_pnl_pct = ?,
                        outcome = ?,
                        max_favorable_excursion = ?,
                        max_adverse_excursion = ?
                    WHERE trade_id = ?
                """, (
                    exit_price, exit_time.isoformat(), duration,
                    realized_pnl, realized_pnl_pct, outcome,
                    max_favorable, max_adverse, trade_id
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to update trade exit: {e}")
            return False

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a trade by ID"""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        execution_mode: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query trades with filters"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
        if execution_mode:
            query += " AND execution_mode = ?"
            params.append(execution_mode)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)

        query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        execution_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance summary"""
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing_trades,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl_pct) as avg_pnl_pct,
                SUM(fees_paid) as total_fees,
                AVG(leverage) as avg_leverage,
                AVG(holding_duration_seconds) as avg_duration,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade,
                SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) ELSE 0 END) as gross_loss
            FROM trades
            WHERE exit_time IS NOT NULL
        """
        params = []

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
        if execution_mode:
            query += " AND execution_mode = ?"
            params.append(execution_mode)

        with self._cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row or row["total_trades"] == 0:
                return {"total_trades": 0}

            result = dict(row)

            # Calculate derived metrics
            result["win_rate"] = (
                result["winning_trades"] / result["total_trades"] * 100
                if result["total_trades"] > 0 else 0
            )
            result["profit_factor"] = (
                result["gross_profit"] / result["gross_loss"]
                if result["gross_loss"] > 0 else float("inf")
            )
            result["avg_duration_hours"] = (
                result["avg_duration"] / 3600
                if result["avg_duration"] else 0
            )

            return result

    def get_performance_by_regime(
        self,
        execution_mode: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by regime"""
        query = """
            SELECT
                regime,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl_pct) as avg_pnl_pct
            FROM trades
            WHERE exit_time IS NOT NULL AND regime IS NOT NULL
        """
        params = []

        if execution_mode:
            query += " AND execution_mode = ?"
            params.append(execution_mode)

        query += " GROUP BY regime"

        with self._cursor() as cursor:
            cursor.execute(query, params)
            result = {}
            for row in cursor.fetchall():
                regime = row["regime"]
                result[regime] = {
                    "total_trades": row["total_trades"],
                    "wins": row["wins"],
                    "win_rate": row["wins"] / row["total_trades"] * 100 if row["total_trades"] > 0 else 0,
                    "total_pnl": row["total_pnl"],
                    "avg_pnl_pct": row["avg_pnl_pct"]
                }
            return result

    def get_performance_by_leverage(
        self,
        execution_mode: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by leverage level"""
        query = """
            SELECT
                CAST(leverage AS INTEGER) as leverage_level,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl_pct) as avg_pnl_pct
            FROM trades
            WHERE exit_time IS NOT NULL
        """
        params = []

        if execution_mode:
            query += " AND execution_mode = ?"
            params.append(execution_mode)

        query += " GROUP BY leverage_level ORDER BY leverage_level"

        with self._cursor() as cursor:
            cursor.execute(query, params)
            result = {}
            for row in cursor.fetchall():
                level = f"{row['leverage_level']}x"
                result[level] = {
                    "total_trades": row["total_trades"],
                    "wins": row["wins"],
                    "win_rate": row["wins"] / row["total_trades"] * 100 if row["total_trades"] > 0 else 0,
                    "total_pnl": row["total_pnl"],
                    "avg_pnl_pct": row["avg_pnl_pct"]
                }
            return result

    def get_performance_by_strategy(
        self,
        execution_mode: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy"""
        query = """
            SELECT
                strategy_name,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl_pct) as avg_pnl_pct,
                AVG(leverage) as avg_leverage
            FROM trades
            WHERE exit_time IS NOT NULL AND strategy_name IS NOT NULL
        """
        params = []

        if execution_mode:
            query += " AND execution_mode = ?"
            params.append(execution_mode)

        query += " GROUP BY strategy_name ORDER BY total_pnl DESC"

        with self._cursor() as cursor:
            cursor.execute(query, params)
            result = {}
            for row in cursor.fetchall():
                strategy = row["strategy_name"]
                result[strategy] = {
                    "total_trades": row["total_trades"],
                    "wins": row["wins"],
                    "win_rate": row["wins"] / row["total_trades"] * 100 if row["total_trades"] > 0 else 0,
                    "total_pnl": row["total_pnl"],
                    "avg_pnl_pct": row["avg_pnl_pct"],
                    "avg_leverage": row["avg_leverage"]
                }
            return result

    def get_risk_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query risk events"""
        query = "SELECT * FROM risk_events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_ai_learning_data(
        self,
        limit: int = 10000,
        episode_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get AI learning data for training"""
        query = """
            SELECT
                trade_id, timestamp, state_features, action_taken,
                reward, next_state_features, is_terminal, episode_id
            FROM ai_learning_data
        """
        params = []

        if episode_id:
            query += " WHERE episode_id = ?"
            params.append(episode_id)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        with self._cursor() as cursor:
            cursor.execute(query, params)
            result = []
            for row in cursor.fetchall():
                data = dict(row)
                data["state_features"] = json.loads(data["state_features"])
                if data["next_state_features"]:
                    data["next_state_features"] = json.loads(data["next_state_features"])
                result.append(data)
            return result

    def export_for_training(
        self,
        output_path: str,
        execution_mode: Optional[str] = None,
        min_trades: int = 100
    ) -> bool:
        """Export trades data for AI training"""
        try:
            trades = self.get_trades(
                execution_mode=execution_mode,
                limit=100000
            )

            if len(trades) < min_trades:
                logger.warning(f"Not enough trades for training: {len(trades)} < {min_trades}")
                return False

            # Prepare training data
            training_data = []
            for trade in trades:
                if trade["exit_time"] is None:
                    continue

                training_data.append({
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "leverage": trade["leverage"],
                    "regime": trade["regime"],
                    "volatility": trade["volatility"],
                    "signal_strength": trade["signal_strength"],
                    "confidence": trade["confidence"],
                    "holding_duration": trade["holding_duration_seconds"],
                    "pnl_pct": trade["realized_pnl_pct"],
                    "outcome": trade["outcome"],
                    "max_favorable": trade["max_favorable_excursion"],
                    "max_adverse": trade["max_adverse_excursion"],
                    "strategy": trade["strategy_name"]
                })

            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(training_data, f, indent=2)

            logger.info(f"Exported {len(training_data)} trades for training to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return False

    def save_daily_snapshot(
        self,
        date: datetime,
        execution_mode: str,
        equity_start: float,
        equity_end: float
    ) -> bool:
        """Save daily performance snapshot"""
        try:
            # Get daily trades
            start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)

            summary = self.get_performance_summary(
                start_date=start,
                end_date=end,
                execution_mode=execution_mode
            )

            by_regime = self.get_performance_by_regime(execution_mode)
            by_strategy = self.get_performance_by_strategy(execution_mode)
            by_leverage = self.get_performance_by_leverage(execution_mode)

            with self._cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_snapshots (
                        date, execution_mode,
                        starting_equity, ending_equity,
                        daily_pnl, daily_pnl_pct,
                        total_trades, winning_trades, losing_trades, win_rate,
                        gross_profit, gross_loss, profit_factor,
                        avg_leverage, total_fees,
                        best_trade_pnl, worst_trade_pnl,
                        avg_trade_duration_seconds,
                        by_regime, by_strategy, by_leverage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date.date().isoformat(), execution_mode,
                    equity_start, equity_end,
                    summary.get("total_pnl", 0),
                    (equity_end - equity_start) / equity_start * 100 if equity_start > 0 else 0,
                    summary.get("total_trades", 0),
                    summary.get("winning_trades", 0),
                    summary.get("losing_trades", 0),
                    summary.get("win_rate", 0),
                    summary.get("gross_profit", 0),
                    summary.get("gross_loss", 0),
                    summary.get("profit_factor", 0),
                    summary.get("avg_leverage", 1),
                    summary.get("total_fees", 0),
                    summary.get("best_trade", 0),
                    summary.get("worst_trade", 0),
                    summary.get("avg_duration", 0),
                    json.dumps(by_regime),
                    json.dumps(by_strategy),
                    json.dumps(by_leverage)
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to save daily snapshot: {e}")
            return False

    def cleanup_old_data(self) -> int:
        """Remove data older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM trades WHERE entry_time < ?",
                (cutoff.isoformat(),)
            )
            trades_deleted = cursor.rowcount

            cursor.execute(
                "DELETE FROM risk_events WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            events_deleted = cursor.rowcount

            cursor.execute(
                "DELETE FROM ai_learning_data WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            learning_deleted = cursor.rowcount

            if self.auto_vacuum:
                cursor.execute("VACUUM")

        total = trades_deleted + events_deleted + learning_deleted
        logger.info(f"Cleaned up {total} old records (trades: {trades_deleted}, events: {events_deleted}, learning: {learning_deleted})")
        return total

    def get_statistics(self) -> Dict[str, Any]:
        """Get ledger statistics"""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM trades")
            trades_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM risk_events")
            events_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM ai_learning_data")
            learning_count = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM performance_snapshots")
            snapshots_count = cursor.fetchone()["count"]

            cursor.execute("SELECT MIN(entry_time) as oldest, MAX(entry_time) as newest FROM trades")
            row = cursor.fetchone()
            oldest = row["oldest"]
            newest = row["newest"]

        # Get file size
        file_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return {
            "total_trades": trades_count,
            "total_risk_events": events_count,
            "total_learning_records": learning_count,
            "total_snapshots": snapshots_count,
            "oldest_trade": oldest,
            "newest_trade": newest,
            "database_size_mb": round(file_size_mb, 2),
            "retention_days": self.retention_days
        }


# Global ledger instance
_trade_ledger: Optional[TradeLedger] = None


def get_trade_ledger(db_path: str = "data/trade_ledger.db") -> TradeLedger:
    """Get or create global trade ledger instance"""
    global _trade_ledger
    if _trade_ledger is None:
        _trade_ledger = TradeLedger(db_path=db_path)
    return _trade_ledger


__all__ = [
    # Enums
    "TradeOutcome",
    "TradeType",
    "RiskEventType",
    # Records
    "TradeRecord",
    "RiskEventRecord",
    # Main class
    "TradeLedger",
    "get_trade_ledger",
]
