"""
Unified Learning Database.

Comprehensive storage for all trade experiences with full context:
- Feature vectors for experience replay
- Regime context at entry/exit
- News sentiment snapshots
- Leverage and short position tracking
- Win/loss streak context
- RL replay buffer integration

Schema designed for:
- Fast retrieval by regime, symbol, strategy
- Efficient batch sampling for model training
- Time-decay weighted statistics
- Drift detection metrics
"""

import json
import logging
import pickle
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """
    Complete trade record for learning.

    Captures full context at entry and exit for effective learning.
    Includes execution quality, risk management, and forensics data.
    """

    # Primary identifiers
    trade_id: Optional[int] = None
    transaction_id: str = ""  # Idempotency key from reconciler
    symbol: str = ""
    side: str = "LONG"  # LONG or SHORT

    # Timing
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    hold_duration_seconds: int = 0

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Position sizing
    quantity: float = 0.0
    leverage: float = 1.0
    position_value: float = 0.0

    # Performance
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Regime context at entry
    regime_at_entry: str = "unknown"
    regime_at_exit: str = "unknown"
    regime_confidence_at_entry: float = 0.0
    volatility_at_entry: float = 0.0
    trend_at_entry: str = "neutral"

    # Signal context
    signal_source: str = ""  # ML, scalping, momentum, etc.
    signal_confidence: float = 0.0
    signal_reason: str = ""

    # News/Sentiment at entry
    news_sentiment_score: float = 0.0  # -1 to 1
    news_urgency: int = 0  # 1-10
    fear_greed_index: float = 50.0  # 0-100

    # Technical indicators at entry
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_position: float = 0.5  # 0-1, position in Bollinger Bands
    volume_ratio: float = 1.0  # vs average
    atr: float = 0.0

    # Feature vector (for RL replay)
    feature_vector: Optional[np.ndarray] = None

    # Streak context
    win_streak_at_entry: int = 0
    loss_streak_at_entry: int = 0
    daily_pnl_at_entry: float = 0.0

    # Exit reason
    exit_reason: str = ""  # TP, SL, trailing, signal_flip, manual

    # Metadata
    model_version: str = ""
    was_profitable: bool = False

    # === NEW PHASE 1 PRODUCTION FIELDS ===

    # Trade Gate fields
    gate_score: float = 0.0  # Overall gate score (0-1)
    gate_decision: str = ""  # APPROVED, REJECTED, DEFERRED
    gate_rejection_reason: str = ""  # Why rejected/deferred

    # Execution Quality fields
    expected_entry_price: float = 0.0  # Price expected at signal
    expected_exit_price: float = 0.0
    entry_slippage_pct: float = 0.0  # (actual - expected) / expected
    exit_slippage_pct: float = 0.0
    total_slippage_pct: float = 0.0  # Combined impact
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    total_fees: float = 0.0
    execution_latency_ms: float = 0.0  # Signal to fill time
    partial_fill_pct: float = 1.0  # 1.0 = full fill

    # Risk Budget fields
    risk_budget_pct: float = 0.0  # % of capital allocated
    risk_budget_usd: float = 0.0  # USD allocated
    max_leverage_allowed: float = 1.0  # From risk budget engine
    kelly_fraction: float = 0.0  # Kelly criterion suggestion
    var_at_entry: float = 0.0  # Value at Risk
    cvar_at_entry: float = 0.0  # Conditional VaR

    # Capital Preservation fields
    preservation_level: str = "normal"  # normal, cautious, defensive, critical, lockdown
    leverage_multiplier_applied: float = 1.0  # Applied restriction
    confidence_threshold_at_entry: float = 0.5  # Required confidence

    # Forensics fields (populated after close)
    mae_price: float = 0.0  # Maximum Adverse Excursion price
    mae_pct: float = 0.0  # MAE as percentage
    mae_time_seconds: int = 0  # Time to MAE from entry
    mfe_price: float = 0.0  # Maximum Favorable Excursion price
    mfe_pct: float = 0.0  # MFE as percentage
    mfe_time_seconds: int = 0  # Time to MFE from entry
    capture_ratio: float = 0.0  # realized / MFE
    pain_ratio: float = 0.0  # MAE / MFE
    entry_quality_score: float = 0.0  # 0-1 score
    exit_quality_score: float = 0.0  # 0-1 score
    stop_quality_score: float = 0.0  # 0-1 score

    # Portfolio context at entry
    portfolio_equity_at_entry: float = 0.0
    portfolio_drawdown_at_entry: float = 0.0
    open_positions_at_entry: int = 0
    correlation_with_portfolio: float = 0.0  # -1 to 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "trade_id": self.trade_id,
            "transaction_id": self.transaction_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "hold_duration_seconds": self.hold_duration_seconds,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "position_value": self.position_value,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "max_profit_pct": self.max_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "regime_at_entry": self.regime_at_entry,
            "regime_at_exit": self.regime_at_exit,
            "regime_confidence_at_entry": self.regime_confidence_at_entry,
            "volatility_at_entry": self.volatility_at_entry,
            "trend_at_entry": self.trend_at_entry,
            "signal_source": self.signal_source,
            "signal_confidence": self.signal_confidence,
            "signal_reason": self.signal_reason,
            "news_sentiment_score": self.news_sentiment_score,
            "news_urgency": self.news_urgency,
            "fear_greed_index": self.fear_greed_index,
            "rsi": self.rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "bb_position": self.bb_position,
            "volume_ratio": self.volume_ratio,
            "atr": self.atr,
            "win_streak_at_entry": self.win_streak_at_entry,
            "loss_streak_at_entry": self.loss_streak_at_entry,
            "daily_pnl_at_entry": self.daily_pnl_at_entry,
            "exit_reason": self.exit_reason,
            "model_version": self.model_version,
            "was_profitable": self.was_profitable,
            # New Phase 1 fields
            "gate_score": self.gate_score,
            "gate_decision": self.gate_decision,
            "gate_rejection_reason": self.gate_rejection_reason,
            "expected_entry_price": self.expected_entry_price,
            "expected_exit_price": self.expected_exit_price,
            "entry_slippage_pct": self.entry_slippage_pct,
            "exit_slippage_pct": self.exit_slippage_pct,
            "total_slippage_pct": self.total_slippage_pct,
            "entry_fees": self.entry_fees,
            "exit_fees": self.exit_fees,
            "total_fees": self.total_fees,
            "execution_latency_ms": self.execution_latency_ms,
            "partial_fill_pct": self.partial_fill_pct,
            "risk_budget_pct": self.risk_budget_pct,
            "risk_budget_usd": self.risk_budget_usd,
            "max_leverage_allowed": self.max_leverage_allowed,
            "kelly_fraction": self.kelly_fraction,
            "var_at_entry": self.var_at_entry,
            "cvar_at_entry": self.cvar_at_entry,
            "preservation_level": self.preservation_level,
            "leverage_multiplier_applied": self.leverage_multiplier_applied,
            "confidence_threshold_at_entry": self.confidence_threshold_at_entry,
            "mae_price": self.mae_price,
            "mae_pct": self.mae_pct,
            "mae_time_seconds": self.mae_time_seconds,
            "mfe_price": self.mfe_price,
            "mfe_pct": self.mfe_pct,
            "mfe_time_seconds": self.mfe_time_seconds,
            "capture_ratio": self.capture_ratio,
            "pain_ratio": self.pain_ratio,
            "entry_quality_score": self.entry_quality_score,
            "exit_quality_score": self.exit_quality_score,
            "stop_quality_score": self.stop_quality_score,
            "portfolio_equity_at_entry": self.portfolio_equity_at_entry,
            "portfolio_drawdown_at_entry": self.portfolio_drawdown_at_entry,
            "open_positions_at_entry": self.open_positions_at_entry,
            "correlation_with_portfolio": self.correlation_with_portfolio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        """Create from dictionary."""
        record = cls()
        for key, value in data.items():
            if hasattr(record, key):
                if key in ("entry_time", "exit_time") and value:
                    value = datetime.fromisoformat(value)
                setattr(record, key, value)
        return record


@dataclass
class LearningMetrics:
    """Aggregated learning metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_hold_time_minutes: float = 0.0

    # Regime breakdown
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


class LearningDatabase:
    """
    Unified Learning Database for adaptive trading.

    Features:
    - SQLite storage for persistence and efficient queries
    - Feature vector storage for experience replay
    - Time-decay weighting for relevance
    - Regime-aware statistics
    - Drift detection metrics
    """

    DECAY_HALFLIFE_DAYS = 14  # Patterns lose half relevance in 14 days

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the learning database.

        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = str(
                Path(__file__).parent.parent.parent / "data" / "learning" / "learning_db.sqlite"
            )

        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_database()

        logger.info(f"Learning Database initialized: {db_path}")

    def _migrate_schema(self, cursor):
        """
        Migrate existing database schema to add new Phase 1 columns.

        This method safely adds new columns to existing tables without data loss.
        """
        # Get existing columns
        cursor.execute("PRAGMA table_info(trades)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # New columns to add with their defaults
        new_columns = [
            ("transaction_id", "TEXT", "''"),
            ("regime_confidence_at_entry", "REAL", "0.0"),
            ("gate_score", "REAL", "0.0"),
            ("gate_decision", "TEXT", "''"),
            ("gate_rejection_reason", "TEXT", "''"),
            ("expected_entry_price", "REAL", "0.0"),
            ("expected_exit_price", "REAL", "0.0"),
            ("entry_slippage_pct", "REAL", "0.0"),
            ("exit_slippage_pct", "REAL", "0.0"),
            ("total_slippage_pct", "REAL", "0.0"),
            ("entry_fees", "REAL", "0.0"),
            ("exit_fees", "REAL", "0.0"),
            ("total_fees", "REAL", "0.0"),
            ("execution_latency_ms", "REAL", "0.0"),
            ("partial_fill_pct", "REAL", "1.0"),
            ("risk_budget_pct", "REAL", "0.0"),
            ("risk_budget_usd", "REAL", "0.0"),
            ("max_leverage_allowed", "REAL", "1.0"),
            ("kelly_fraction", "REAL", "0.0"),
            ("var_at_entry", "REAL", "0.0"),
            ("cvar_at_entry", "REAL", "0.0"),
            ("preservation_level", "TEXT", "'normal'"),
            ("leverage_multiplier_applied", "REAL", "1.0"),
            ("confidence_threshold_at_entry", "REAL", "0.5"),
            ("mae_price", "REAL", "0.0"),
            ("mae_pct", "REAL", "0.0"),
            ("mae_time_seconds", "INTEGER", "0"),
            ("mfe_price", "REAL", "0.0"),
            ("mfe_pct", "REAL", "0.0"),
            ("mfe_time_seconds", "INTEGER", "0"),
            ("capture_ratio", "REAL", "0.0"),
            ("pain_ratio", "REAL", "0.0"),
            ("entry_quality_score", "REAL", "0.0"),
            ("exit_quality_score", "REAL", "0.0"),
            ("stop_quality_score", "REAL", "0.0"),
            ("portfolio_equity_at_entry", "REAL", "0.0"),
            ("portfolio_drawdown_at_entry", "REAL", "0.0"),
            ("open_positions_at_entry", "INTEGER", "0"),
            ("correlation_with_portfolio", "REAL", "0.0"),
        ]

        # Add missing columns
        for col_name, col_type, default in new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE trades ADD COLUMN {col_name} {col_type} DEFAULT {default}"
                    )
                    logger.info(f"Added column {col_name} to trades table")
                except sqlite3.OperationalError as e:
                    # Column may already exist in some edge cases
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"Could not add column {col_name}: {e}")

    def _init_database(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main trades table with Phase 1 production fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                hold_duration_seconds INTEGER DEFAULT 0,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity REAL,
                leverage REAL DEFAULT 1.0,
                position_value REAL,
                pnl REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                max_profit_pct REAL DEFAULT 0.0,
                max_drawdown_pct REAL DEFAULT 0.0,
                regime_at_entry TEXT,
                regime_at_exit TEXT,
                regime_confidence_at_entry REAL DEFAULT 0.0,
                volatility_at_entry REAL,
                trend_at_entry TEXT,
                signal_source TEXT,
                signal_confidence REAL,
                signal_reason TEXT,
                news_sentiment_score REAL DEFAULT 0.0,
                news_urgency INTEGER DEFAULT 0,
                fear_greed_index REAL DEFAULT 50.0,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bb_position REAL,
                volume_ratio REAL,
                atr REAL,
                win_streak_at_entry INTEGER DEFAULT 0,
                loss_streak_at_entry INTEGER DEFAULT 0,
                daily_pnl_at_entry REAL DEFAULT 0.0,
                exit_reason TEXT,
                model_version TEXT,
                was_profitable INTEGER DEFAULT 0,
                -- Phase 1 Production Fields --
                -- Trade Gate
                gate_score REAL DEFAULT 0.0,
                gate_decision TEXT,
                gate_rejection_reason TEXT,
                -- Execution Quality
                expected_entry_price REAL DEFAULT 0.0,
                expected_exit_price REAL DEFAULT 0.0,
                entry_slippage_pct REAL DEFAULT 0.0,
                exit_slippage_pct REAL DEFAULT 0.0,
                total_slippage_pct REAL DEFAULT 0.0,
                entry_fees REAL DEFAULT 0.0,
                exit_fees REAL DEFAULT 0.0,
                total_fees REAL DEFAULT 0.0,
                execution_latency_ms REAL DEFAULT 0.0,
                partial_fill_pct REAL DEFAULT 1.0,
                -- Risk Budget
                risk_budget_pct REAL DEFAULT 0.0,
                risk_budget_usd REAL DEFAULT 0.0,
                max_leverage_allowed REAL DEFAULT 1.0,
                kelly_fraction REAL DEFAULT 0.0,
                var_at_entry REAL DEFAULT 0.0,
                cvar_at_entry REAL DEFAULT 0.0,
                -- Capital Preservation
                preservation_level TEXT DEFAULT 'normal',
                leverage_multiplier_applied REAL DEFAULT 1.0,
                confidence_threshold_at_entry REAL DEFAULT 0.5,
                -- Forensics
                mae_price REAL DEFAULT 0.0,
                mae_pct REAL DEFAULT 0.0,
                mae_time_seconds INTEGER DEFAULT 0,
                mfe_price REAL DEFAULT 0.0,
                mfe_pct REAL DEFAULT 0.0,
                mfe_time_seconds INTEGER DEFAULT 0,
                capture_ratio REAL DEFAULT 0.0,
                pain_ratio REAL DEFAULT 0.0,
                entry_quality_score REAL DEFAULT 0.0,
                exit_quality_score REAL DEFAULT 0.0,
                stop_quality_score REAL DEFAULT 0.0,
                -- Portfolio Context
                portfolio_equity_at_entry REAL DEFAULT 0.0,
                portfolio_drawdown_at_entry REAL DEFAULT 0.0,
                open_positions_at_entry INTEGER DEFAULT 0,
                correlation_with_portfolio REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add new columns to existing tables if they don't exist (schema migration)
        self._migrate_schema(cursor)

        # Feature vectors table (separate for efficiency)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_vectors (
                trade_id INTEGER PRIMARY KEY,
                feature_blob BLOB,
                feature_shape TEXT,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # Regime statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_stats (
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_hold_seconds INTEGER DEFAULT 0,
                last_updated TEXT,
                PRIMARY KEY (regime, symbol)
            )
        """)

        # Strategy statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_stats (
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                last_updated TEXT,
                PRIMARY KEY (strategy, symbol, regime)
            )
        """)

        # Daily performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                dominant_regime TEXT,
                notes TEXT
            )
        """)

        # Drift detection table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL,
                baseline_value REAL,
                drift_score REAL,
                window_size INTEGER,
                requires_retrain INTEGER DEFAULT 0
            )
        """)

        # RL recommendations table for counterfactual analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime TEXT,
                preservation_level TEXT DEFAULT 'normal',
                -- Strategy preferences (JSON)
                strategy_preferences TEXT,
                -- Directional bias
                directional_bias TEXT,
                bias_confidence REAL DEFAULT 0.5,
                -- Suggested action
                suggested_action TEXT,
                action_confidence REAL DEFAULT 0.5,
                primary_agent TEXT,
                agent_reasoning TEXT,
                -- All agent votes (JSON)
                agent_votes TEXT,
                -- Application status
                was_applied INTEGER DEFAULT 0,
                application_reason TEXT,
                -- Actual outcome (filled in after trade)
                actual_action TEXT,
                actual_pnl REAL,
                -- Link to trade if applicable
                trade_id INTEGER,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # Create indexes for efficient queries
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime_at_entry)",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trades_signal_source ON trades(signal_source)",
            "CREATE INDEX IF NOT EXISTS idx_trades_profitable ON trades(was_profitable)",
            "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side)",
        ]
        for idx_sql in indexes:
            cursor.execute(idx_sql)

        conn.commit()
        conn.close()

    def record_trade(self, record: TradeRecord) -> int:
        """
        Record a completed trade.

        Args:
            record: TradeRecord with full trade context

        Returns:
            trade_id of the inserted record
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert main record with all Phase 1 fields
            cursor.execute("""
                INSERT INTO trades (
                    transaction_id, symbol, side, entry_time, exit_time, hold_duration_seconds,
                    entry_price, exit_price, stop_loss, take_profit,
                    quantity, leverage, position_value, pnl, pnl_pct,
                    max_profit_pct, max_drawdown_pct, regime_at_entry,
                    regime_at_exit, regime_confidence_at_entry, volatility_at_entry, trend_at_entry,
                    signal_source, signal_confidence, signal_reason,
                    news_sentiment_score, news_urgency, fear_greed_index,
                    rsi, macd, macd_signal, bb_position, volume_ratio, atr,
                    win_streak_at_entry, loss_streak_at_entry, daily_pnl_at_entry,
                    exit_reason, model_version, was_profitable,
                    gate_score, gate_decision, gate_rejection_reason,
                    expected_entry_price, expected_exit_price,
                    entry_slippage_pct, exit_slippage_pct, total_slippage_pct,
                    entry_fees, exit_fees, total_fees, execution_latency_ms, partial_fill_pct,
                    risk_budget_pct, risk_budget_usd, max_leverage_allowed,
                    kelly_fraction, var_at_entry, cvar_at_entry,
                    preservation_level, leverage_multiplier_applied, confidence_threshold_at_entry,
                    mae_price, mae_pct, mae_time_seconds,
                    mfe_price, mfe_pct, mfe_time_seconds,
                    capture_ratio, pain_ratio, entry_quality_score, exit_quality_score, stop_quality_score,
                    portfolio_equity_at_entry, portfolio_drawdown_at_entry,
                    open_positions_at_entry, correlation_with_portfolio
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record.transaction_id, record.symbol, record.side,
                record.entry_time.isoformat() if record.entry_time else None,
                record.exit_time.isoformat() if record.exit_time else None,
                record.hold_duration_seconds,
                record.entry_price, record.exit_price, record.stop_loss, record.take_profit,
                record.quantity, record.leverage, record.position_value,
                record.pnl, record.pnl_pct, record.max_profit_pct, record.max_drawdown_pct,
                record.regime_at_entry, record.regime_at_exit, record.regime_confidence_at_entry,
                record.volatility_at_entry, record.trend_at_entry,
                record.signal_source, record.signal_confidence, record.signal_reason,
                record.news_sentiment_score, record.news_urgency, record.fear_greed_index,
                record.rsi, record.macd, record.macd_signal,
                record.bb_position, record.volume_ratio, record.atr,
                record.win_streak_at_entry, record.loss_streak_at_entry, record.daily_pnl_at_entry,
                record.exit_reason, record.model_version,
                1 if record.was_profitable else 0,
                record.gate_score, record.gate_decision, record.gate_rejection_reason,
                record.expected_entry_price, record.expected_exit_price,
                record.entry_slippage_pct, record.exit_slippage_pct, record.total_slippage_pct,
                record.entry_fees, record.exit_fees, record.total_fees,
                record.execution_latency_ms, record.partial_fill_pct,
                record.risk_budget_pct, record.risk_budget_usd, record.max_leverage_allowed,
                record.kelly_fraction, record.var_at_entry, record.cvar_at_entry,
                record.preservation_level, record.leverage_multiplier_applied,
                record.confidence_threshold_at_entry,
                record.mae_price, record.mae_pct, record.mae_time_seconds,
                record.mfe_price, record.mfe_pct, record.mfe_time_seconds,
                record.capture_ratio, record.pain_ratio,
                record.entry_quality_score, record.exit_quality_score, record.stop_quality_score,
                record.portfolio_equity_at_entry, record.portfolio_drawdown_at_entry,
                record.open_positions_at_entry, record.correlation_with_portfolio,
            ))

            trade_id = cursor.lastrowid

            # Store feature vector if present
            if record.feature_vector is not None:
                feature_blob = pickle.dumps(record.feature_vector)
                feature_shape = json.dumps(list(record.feature_vector.shape))
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_vectors (trade_id, feature_blob, feature_shape)
                    VALUES (?, ?, ?)
                """, (trade_id, feature_blob, feature_shape))

            # Update regime statistics
            self._update_regime_stats(cursor, record)

            # Update strategy statistics
            self._update_strategy_stats(cursor, record)

            # Update daily performance
            self._update_daily_performance(cursor, record)

            conn.commit()
            conn.close()

            logger.debug(
                f"Recorded trade #{trade_id}: {record.symbol} {record.side} "
                f"PnL: {record.pnl_pct:+.2f}%"
            )

            return trade_id

    def _update_regime_stats(self, cursor, record: TradeRecord):
        """Update regime statistics after a trade."""
        regime = record.regime_at_entry or "unknown"
        symbol = record.symbol

        cursor.execute("""
            INSERT INTO regime_stats (regime, symbol, total_trades, wins, losses, total_pnl, avg_hold_seconds, last_updated)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(regime, symbol) DO UPDATE SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                avg_hold_seconds = (avg_hold_seconds * total_trades + ?) / (total_trades + 1),
                last_updated = ?
        """, (
            regime, symbol,
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            record.hold_duration_seconds,
            datetime.now().isoformat(),
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            record.hold_duration_seconds,
            datetime.now().isoformat(),
        ))

    def _update_strategy_stats(self, cursor, record: TradeRecord):
        """Update strategy statistics after a trade."""
        strategy = record.signal_source or "unknown"
        symbol = record.symbol
        regime = record.regime_at_entry or "unknown"

        cursor.execute("""
            INSERT INTO strategy_stats (strategy, symbol, regime, total_trades, wins, losses, total_pnl, avg_confidence, last_updated)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy, symbol, regime) DO UPDATE SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                avg_confidence = (avg_confidence * total_trades + ?) / (total_trades + 1),
                last_updated = ?
        """, (
            strategy, symbol, regime,
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            record.signal_confidence,
            datetime.now().isoformat(),
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            record.signal_confidence,
            datetime.now().isoformat(),
        ))

    def _update_daily_performance(self, cursor, record: TradeRecord):
        """Update daily performance after a trade."""
        if record.exit_time:
            date = record.exit_time.date().isoformat()
        else:
            date = datetime.now().date().isoformat()

        cursor.execute("""
            INSERT INTO daily_performance (date, total_trades, wins, losses, total_pnl, dominant_regime)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                win_rate = CAST(wins + ? AS REAL) / (total_trades + 1)
        """, (
            date,
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            record.regime_at_entry or "unknown",
            1 if record.was_profitable else 0,
            0 if record.was_profitable else 1,
            record.pnl,
            1 if record.was_profitable else 0,
        ))

    def get_trades(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        side: Optional[str] = None,
        days_lookback: int = 30,
        limit: int = 500,
        include_features: bool = False,
    ) -> List[TradeRecord]:
        """
        Retrieve trades with optional filtering.

        Args:
            symbol: Filter by symbol
            regime: Filter by regime at entry
            strategy: Filter by signal source
            side: Filter by LONG or SHORT
            days_lookback: Number of days to look back
            limit: Maximum number of records
            include_features: Whether to load feature vectors

        Returns:
            List of TradeRecord objects
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if regime:
                query += " AND regime_at_entry = ?"
                params.append(regime)

            if strategy:
                query += " AND signal_source = ?"
                params.append(strategy)

            if side:
                query += " AND side = ?"
                params.append(side)

            cutoff = (datetime.now() - timedelta(days=days_lookback)).isoformat()
            query += " AND entry_time > ?"
            params.append(cutoff)

            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            trades = []
            for row in rows:
                record = TradeRecord(
                    trade_id=row["id"],
                    symbol=row["symbol"],
                    side=row["side"],
                    entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
                    exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
                    hold_duration_seconds=row["hold_duration_seconds"] or 0,
                    entry_price=row["entry_price"] or 0,
                    exit_price=row["exit_price"] or 0,
                    stop_loss=row["stop_loss"] or 0,
                    take_profit=row["take_profit"] or 0,
                    quantity=row["quantity"] or 0,
                    leverage=row["leverage"] or 1.0,
                    position_value=row["position_value"] or 0,
                    pnl=row["pnl"] or 0,
                    pnl_pct=row["pnl_pct"] or 0,
                    max_profit_pct=row["max_profit_pct"] or 0,
                    max_drawdown_pct=row["max_drawdown_pct"] or 0,
                    regime_at_entry=row["regime_at_entry"] or "unknown",
                    regime_at_exit=row["regime_at_exit"] or "unknown",
                    volatility_at_entry=row["volatility_at_entry"] or 0,
                    trend_at_entry=row["trend_at_entry"] or "neutral",
                    signal_source=row["signal_source"] or "",
                    signal_confidence=row["signal_confidence"] or 0,
                    signal_reason=row["signal_reason"] or "",
                    news_sentiment_score=row["news_sentiment_score"] or 0,
                    news_urgency=row["news_urgency"] or 0,
                    fear_greed_index=row["fear_greed_index"] or 50,
                    rsi=row["rsi"] or 50,
                    macd=row["macd"] or 0,
                    macd_signal=row["macd_signal"] or 0,
                    bb_position=row["bb_position"] or 0.5,
                    volume_ratio=row["volume_ratio"] or 1,
                    atr=row["atr"] or 0,
                    win_streak_at_entry=row["win_streak_at_entry"] or 0,
                    loss_streak_at_entry=row["loss_streak_at_entry"] or 0,
                    daily_pnl_at_entry=row["daily_pnl_at_entry"] or 0,
                    exit_reason=row["exit_reason"] or "",
                    model_version=row["model_version"] or "",
                    was_profitable=bool(row["was_profitable"]),
                    # Phase 1 Production Fields
                    transaction_id=row["transaction_id"] or "",
                    regime_confidence_at_entry=row["regime_confidence_at_entry"] or 0.0,
                    gate_score=row["gate_score"] or 0.0,
                    gate_decision=row["gate_decision"] or "",
                    gate_rejection_reason=row["gate_rejection_reason"] or "",
                    expected_entry_price=row["expected_entry_price"] or 0.0,
                    expected_exit_price=row["expected_exit_price"] or 0.0,
                    entry_slippage_pct=row["entry_slippage_pct"] or 0.0,
                    exit_slippage_pct=row["exit_slippage_pct"] or 0.0,
                    total_slippage_pct=row["total_slippage_pct"] or 0.0,
                    entry_fees=row["entry_fees"] or 0.0,
                    exit_fees=row["exit_fees"] or 0.0,
                    total_fees=row["total_fees"] or 0.0,
                    execution_latency_ms=row["execution_latency_ms"] or 0.0,
                    partial_fill_pct=row["partial_fill_pct"] or 1.0,
                    risk_budget_pct=row["risk_budget_pct"] or 0.0,
                    risk_budget_usd=row["risk_budget_usd"] or 0.0,
                    max_leverage_allowed=row["max_leverage_allowed"] or 1.0,
                    kelly_fraction=row["kelly_fraction"] or 0.0,
                    var_at_entry=row["var_at_entry"] or 0.0,
                    cvar_at_entry=row["cvar_at_entry"] or 0.0,
                    preservation_level=row["preservation_level"] or "normal",
                    leverage_multiplier_applied=row["leverage_multiplier_applied"] or 1.0,
                    confidence_threshold_at_entry=row["confidence_threshold_at_entry"] or 0.5,
                    mae_price=row["mae_price"] or 0.0,
                    mae_pct=row["mae_pct"] or 0.0,
                    mae_time_seconds=row["mae_time_seconds"] or 0,
                    mfe_price=row["mfe_price"] or 0.0,
                    mfe_pct=row["mfe_pct"] or 0.0,
                    mfe_time_seconds=row["mfe_time_seconds"] or 0,
                    capture_ratio=row["capture_ratio"] or 0.0,
                    pain_ratio=row["pain_ratio"] or 0.0,
                    entry_quality_score=row["entry_quality_score"] or 0.0,
                    exit_quality_score=row["exit_quality_score"] or 0.0,
                    stop_quality_score=row["stop_quality_score"] or 0.0,
                    portfolio_equity_at_entry=row["portfolio_equity_at_entry"] or 0.0,
                    portfolio_drawdown_at_entry=row["portfolio_drawdown_at_entry"] or 0.0,
                    open_positions_at_entry=row["open_positions_at_entry"] or 0,
                    correlation_with_portfolio=row["correlation_with_portfolio"] or 0.0,
                )

                # Load feature vector if requested
                if include_features:
                    cursor.execute(
                        "SELECT feature_blob FROM feature_vectors WHERE trade_id = ?",
                        (row["id"],)
                    )
                    fv_row = cursor.fetchone()
                    if fv_row and fv_row[0]:
                        record.feature_vector = pickle.loads(fv_row[0])

                trades.append(record)

            conn.close()
            return trades

    def get_regime_stats(self, regime: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by regime."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if regime:
                cursor.execute(
                    "SELECT * FROM regime_stats WHERE regime = ?",
                    (regime,)
                )
            else:
                cursor.execute("SELECT * FROM regime_stats")

            rows = cursor.fetchall()
            conn.close()

            stats = {}
            for row in rows:
                key = f"{row['regime']}_{row['symbol']}"
                total = row["total_trades"] or 1
                stats[key] = {
                    "regime": row["regime"],
                    "symbol": row["symbol"],
                    "total_trades": total,
                    "wins": row["wins"] or 0,
                    "losses": row["losses"] or 0,
                    "win_rate": (row["wins"] or 0) / total,
                    "total_pnl": row["total_pnl"] or 0,
                    "avg_pnl": (row["total_pnl"] or 0) / total,
                    "avg_hold_minutes": (row["avg_hold_seconds"] or 0) / 60,
                }

            return stats

    def get_strategy_stats(self, strategy: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by strategy."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if strategy:
                cursor.execute(
                    "SELECT * FROM strategy_stats WHERE strategy = ?",
                    (strategy,)
                )
            else:
                cursor.execute("SELECT * FROM strategy_stats")

            rows = cursor.fetchall()
            conn.close()

            stats = {}
            for row in rows:
                key = f"{row['strategy']}_{row['symbol']}_{row['regime']}"
                total = row["total_trades"] or 1
                stats[key] = {
                    "strategy": row["strategy"],
                    "symbol": row["symbol"],
                    "regime": row["regime"],
                    "total_trades": total,
                    "wins": row["wins"] or 0,
                    "losses": row["losses"] or 0,
                    "win_rate": (row["wins"] or 0) / total,
                    "total_pnl": row["total_pnl"] or 0,
                    "avg_pnl": (row["total_pnl"] or 0) / total,
                    "avg_confidence": row["avg_confidence"] or 0,
                }

            return stats

    def get_daily_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily performance for the last N days."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()
            cursor.execute(
                "SELECT * FROM daily_performance WHERE date > ? ORDER BY date DESC",
                (cutoff,)
            )

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def get_learning_metrics(self, days_lookback: int = 30) -> LearningMetrics:
        """Calculate comprehensive learning metrics."""
        trades = self.get_trades(days_lookback=days_lookback, limit=5000)

        if not trades:
            return LearningMetrics()

        # Basic metrics
        total = len(trades)
        wins = [t for t in trades if t.was_profitable]
        losses = [t for t in trades if not t.was_profitable]

        total_win_pnl = sum(t.pnl for t in wins)
        total_loss_pnl = abs(sum(t.pnl for t in losses))

        # Calculate streaks
        max_wins = max_losses = current_wins = current_losses = 0
        for t in sorted(trades, key=lambda x: x.entry_time or datetime.min):
            if t.was_profitable:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        # Regime breakdown
        regime_perf = {}
        for t in trades:
            regime = t.regime_at_entry or "unknown"
            if regime not in regime_perf:
                regime_perf[regime] = {"total": 0, "wins": 0, "pnl": 0.0}
            regime_perf[regime]["total"] += 1
            regime_perf[regime]["wins"] += 1 if t.was_profitable else 0
            regime_perf[regime]["pnl"] += t.pnl

        for regime in regime_perf:
            total_r = regime_perf[regime]["total"]
            regime_perf[regime]["win_rate"] = regime_perf[regime]["wins"] / total_r if total_r > 0 else 0

        # Strategy breakdown
        strategy_perf = {}
        for t in trades:
            strat = t.signal_source or "unknown"
            if strat not in strategy_perf:
                strategy_perf[strat] = {"total": 0, "wins": 0, "pnl": 0.0}
            strategy_perf[strat]["total"] += 1
            strategy_perf[strat]["wins"] += 1 if t.was_profitable else 0
            strategy_perf[strat]["pnl"] += t.pnl

        for strat in strategy_perf:
            total_s = strategy_perf[strat]["total"]
            strategy_perf[strat]["win_rate"] = strategy_perf[strat]["wins"] / total_s if total_s > 0 else 0

        return LearningMetrics(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / total if total > 0 else 0,
            avg_pnl_pct=sum(t.pnl_pct for t in trades) / total if total > 0 else 0,
            avg_win_pct=sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
            avg_loss_pct=sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
            profit_factor=total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_hold_time_minutes=sum(t.hold_duration_seconds for t in trades) / total / 60 if total > 0 else 0,
            regime_performance=regime_perf,
            strategy_performance=strategy_perf,
        )

    def get_features_for_replay(
        self,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        n_samples: int = 1000,
        prioritize_recent: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get feature vectors and labels for experience replay.

        Args:
            regime: Filter by regime
            strategy: Filter by strategy
            n_samples: Number of samples to retrieve
            prioritize_recent: Weight recent samples higher

        Returns:
            Tuple of (features, labels, rewards) arrays
        """
        trades = self.get_trades(
            regime=regime,
            strategy=strategy,
            days_lookback=90,
            limit=n_samples * 2,
            include_features=True,
        )

        # Filter to trades with feature vectors
        trades_with_features = [t for t in trades if t.feature_vector is not None]

        if not trades_with_features:
            return np.array([]), np.array([]), np.array([])

        # Apply time-decay weighting if prioritizing recent
        if prioritize_recent:
            weights = []
            now = datetime.now()
            for t in trades_with_features:
                age_days = (now - (t.entry_time or now)).total_seconds() / 86400
                weight = 0.5 ** (age_days / self.DECAY_HALFLIFE_DAYS)
                weights.append(weight)
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Sample with replacement based on weights
            n_actual = min(n_samples, len(trades_with_features))
            indices = np.random.choice(
                len(trades_with_features),
                size=n_actual,
                replace=True,
                p=weights,
            )
            trades_with_features = [trades_with_features[i] for i in indices]
        else:
            trades_with_features = trades_with_features[:n_samples]

        # Build arrays
        features = np.array([t.feature_vector for t in trades_with_features])

        # Labels: 0=SHORT, 1=FLAT, 2=LONG
        labels = np.array([
            0 if t.side == "SHORT" else 2 if t.side == "LONG" else 1
            for t in trades_with_features
        ])

        # Rewards based on PnL percentage
        rewards = np.array([t.pnl_pct * 100 for t in trades_with_features])

        return features, labels, rewards

    def record_drift_metric(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        window_size: int = 50,
    ) -> bool:
        """
        Record a drift detection metric.

        Args:
            metric_name: Name of the metric (e.g., "win_rate", "avg_pnl")
            current_value: Current rolling value
            baseline_value: Baseline value for comparison
            window_size: Window size used for calculation

        Returns:
            True if drift detected (requires retrain)
        """
        # Calculate drift score (normalized difference)
        if baseline_value == 0:
            drift_score = abs(current_value) if current_value != 0 else 0
        else:
            drift_score = abs(current_value - baseline_value) / abs(baseline_value)

        # Threshold for significant drift
        requires_retrain = drift_score > 0.2  # 20% drift threshold

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO drift_metrics (timestamp, metric_name, current_value, baseline_value, drift_score, window_size, requires_retrain)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_name,
                current_value,
                baseline_value,
                drift_score,
                window_size,
                1 if requires_retrain else 0,
            ))

            conn.commit()
            conn.close()

        if requires_retrain:
            logger.warning(
                f"Drift detected for {metric_name}: current={current_value:.4f}, "
                f"baseline={baseline_value:.4f}, drift={drift_score:.2%}"
            )

        return requires_retrain

    def get_streak_context(self) -> Tuple[int, int, float]:
        """
        Get current win/loss streak and daily PnL.

        Returns:
            Tuple of (win_streak, loss_streak, daily_pnl)
        """
        trades = self.get_trades(days_lookback=7, limit=100)

        if not trades:
            return 0, 0, 0.0

        # Sort by time
        trades = sorted(trades, key=lambda t: t.exit_time or t.entry_time or datetime.min, reverse=True)

        # Calculate streak
        win_streak = loss_streak = 0
        for t in trades:
            if t.was_profitable:
                if loss_streak > 0:
                    break
                win_streak += 1
            else:
                if win_streak > 0:
                    break
                loss_streak += 1

        # Calculate daily PnL
        today = datetime.now().date()
        daily_pnl = sum(
            t.pnl for t in trades
            if t.exit_time and t.exit_time.date() == today
        )

        return win_streak, loss_streak, daily_pnl

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning database."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trades WHERE was_profitable = 1")
            profitable_trades = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(pnl) FROM trades")
            total_pnl = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trades")
            unique_symbols = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT regime_at_entry) FROM trades")
            unique_regimes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT signal_source) FROM trades")
            unique_strategies = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(entry_time), MAX(entry_time) FROM trades")
            time_range = cursor.fetchone()

            conn.close()

        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": profitable_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "unique_symbols": unique_symbols,
            "unique_regimes": unique_regimes,
            "unique_strategies": unique_strategies,
            "first_trade": time_range[0] if time_range else None,
            "last_trade": time_range[1] if time_range else None,
        }

    def cleanup_old_data(self, days_to_keep: int = 180):
        """Remove data older than specified days."""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get IDs of old trades
            cursor.execute("SELECT id FROM trades WHERE entry_time < ?", (cutoff,))
            old_ids = [row[0] for row in cursor.fetchall()]

            if old_ids:
                # Delete feature vectors
                cursor.execute(
                    f"DELETE FROM feature_vectors WHERE trade_id IN ({','.join('?' * len(old_ids))})",
                    old_ids
                )

                # Delete trades
                cursor.execute("DELETE FROM trades WHERE entry_time < ?", (cutoff,))
                deleted = cursor.rowcount

                # Delete old drift metrics
                cursor.execute("DELETE FROM drift_metrics WHERE timestamp < ?", (cutoff,))

                conn.commit()
                logger.info(f"Cleaned up {deleted} old trades (older than {days_to_keep} days)")

            conn.close()

    # =========================================================================
    # RL RECOMMENDATION LOGGING (Phase 2A)
    # =========================================================================

    def record_rl_recommendation(
        self,
        symbol: str,
        regime: str,
        preservation_level: str,
        strategy_preferences: Dict[str, float],
        directional_bias: str,
        bias_confidence: float,
        suggested_action: str,
        action_confidence: float,
        primary_agent: str,
        agent_reasoning: str,
        agent_votes: Dict[str, Any],
        was_applied: bool,
        application_reason: str,
        trade_id: Optional[int] = None,
    ) -> int:
        """
        Record an RL recommendation for counterfactual analysis.

        Args:
            symbol: Trading symbol
            regime: Market regime
            preservation_level: Capital preservation level
            strategy_preferences: Dict of strategy -> weight
            directional_bias: "long", "short", or "neutral"
            bias_confidence: Confidence in bias (0-1)
            suggested_action: Suggested action from RL
            action_confidence: Confidence in action (0-1)
            primary_agent: Agent that generated recommendation
            agent_reasoning: Reasoning for recommendation
            agent_votes: All agent votes (dict)
            was_applied: Whether recommendation was used
            application_reason: Why/why not applied
            trade_id: Optional linked trade ID

        Returns:
            recommendation_id
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO rl_recommendations (
                    timestamp, symbol, regime, preservation_level,
                    strategy_preferences, directional_bias, bias_confidence,
                    suggested_action, action_confidence, primary_agent,
                    agent_reasoning, agent_votes, was_applied, application_reason,
                    trade_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                symbol,
                regime,
                preservation_level,
                json.dumps(strategy_preferences),
                directional_bias,
                bias_confidence,
                suggested_action,
                action_confidence,
                primary_agent,
                agent_reasoning,
                json.dumps(agent_votes),
                1 if was_applied else 0,
                application_reason,
                trade_id,
            ))

            rec_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.debug(f"Recorded RL recommendation {rec_id} for {symbol}")
            return rec_id

    def update_rl_recommendation_outcome(
        self,
        recommendation_id: int,
        actual_action: str,
        actual_pnl: float,
        trade_id: Optional[int] = None,
    ):
        """
        Update RL recommendation with actual outcome.

        Args:
            recommendation_id: ID of recommendation to update
            actual_action: Action actually taken
            actual_pnl: Actual profit/loss
            trade_id: Optional trade ID to link
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE rl_recommendations
                SET actual_action = ?, actual_pnl = ?, trade_id = ?
                WHERE id = ?
            """, (actual_action, actual_pnl, trade_id, recommendation_id))

            conn.commit()
            conn.close()

    def get_rl_recommendations(
        self,
        symbol: Optional[str] = None,
        days_lookback: int = 7,
        limit: int = 100,
        only_counterfactuals: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve RL recommendations for analysis.

        Args:
            symbol: Optional symbol filter
            days_lookback: Days to look back
            limit: Max records
            only_counterfactuals: Only return recommendations with outcomes

        Returns:
            List of recommendation dicts
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT * FROM rl_recommendations
                WHERE timestamp > ?
            """
            cutoff = (datetime.now() - timedelta(days=days_lookback)).isoformat()
            params = [cutoff]

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if only_counterfactuals:
                query += " AND actual_action IS NOT NULL"

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            recommendations = []
            for row in rows:
                rec = dict(row)
                # Parse JSON fields
                if rec.get("strategy_preferences"):
                    rec["strategy_preferences"] = json.loads(rec["strategy_preferences"])
                if rec.get("agent_votes"):
                    rec["agent_votes"] = json.loads(rec["agent_votes"])
                recommendations.append(rec)

            return recommendations

    def get_rl_counterfactual_analysis(
        self,
        days_lookback: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze RL recommendation performance vs actual outcomes.

        Returns counterfactual metrics showing what would have happened
        if RL recommendations were followed.

        Returns:
            Dict with counterfactual analysis metrics
        """
        recs = self.get_rl_recommendations(
            days_lookback=days_lookback,
            limit=1000,
            only_counterfactuals=True,
        )

        if not recs:
            return {"error": "No counterfactual data available"}

        # Calculate metrics
        rl_would_win = 0
        rl_would_lose = 0
        actual_wins = 0
        actual_losses = 0
        agreement_count = 0
        disagreement_count = 0

        for rec in recs:
            suggested = rec.get("suggested_action", "").lower()
            actual = rec.get("actual_action", "").lower()
            pnl = rec.get("actual_pnl", 0) or 0

            # Normalize actions
            action_map = {"buy": "long", "sell": "short", "long": "long", "short": "short"}
            suggested_norm = action_map.get(suggested, suggested)
            actual_norm = action_map.get(actual, actual)

            # Track agreement
            if suggested_norm == actual_norm:
                agreement_count += 1
            else:
                disagreement_count += 1

            # Track outcomes
            if pnl > 0:
                actual_wins += 1
            else:
                actual_losses += 1

            # Estimate what RL would have done
            # (This is a rough estimate - proper counterfactual requires backtesting)
            if suggested_norm == actual_norm and pnl > 0:
                rl_would_win += 1
            elif suggested_norm != actual_norm and pnl < 0:
                rl_would_win += 1  # RL suggested differently, actual lost
            else:
                rl_would_lose += 1

        total = len(recs)
        return {
            "total_recommendations": total,
            "agreement_rate": agreement_count / total if total > 0 else 0,
            "actual_win_rate": actual_wins / total if total > 0 else 0,
            "rl_estimated_win_rate": rl_would_win / total if total > 0 else 0,
            "avg_action_confidence": sum(r.get("action_confidence", 0) for r in recs) / total if total > 0 else 0,
            "top_agents": self._count_top_agents(recs),
        }

    def _count_top_agents(self, recs: List[Dict]) -> Dict[str, int]:
        """Count recommendations by agent."""
        counts = {}
        for rec in recs:
            agent = rec.get("primary_agent", "unknown")
            counts[agent] = counts.get(agent, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
