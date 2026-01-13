"""
Learning Database - Persistent storage for all AI learning data.

Stores:
- Trade history with outcomes
- Strategy performance metrics
- Parameter optimization results
- RL agent experiences
- Evolved strategy genes
- Model weights and checkpoints
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Default database location
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DB_PATH = DATA_DIR / "ai_learning.db"


@dataclass
class TradeRecord:
    """A single trade record for learning."""
    id: Optional[int]
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    hold_duration_mins: Optional[int]
    regime: str
    strategy_id: str
    indicators: Dict[str, float]  # Indicator values at entry
    outcome: Optional[str]  # WIN, LOSS, BREAKEVEN

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['indicators'] = json.dumps(d['indicators'])
        return d


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    total_trades: int
    win_rate: float
    avg_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    regime: str  # Which regime this performance is for
    last_updated: str


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    id: Optional[int]
    timestamp: str
    symbol: str
    regime: str
    parameters: Dict[str, Any]
    sharpe_ratio: float
    win_rate: float
    total_return: float
    max_drawdown: float
    num_trades: int
    backtest_period_days: int


@dataclass
class EvolvedStrategy:
    """A strategy discovered by genetic evolution."""
    id: Optional[int]
    generation: int
    fitness: float  # Sharpe ratio or other metric
    genes: Dict[str, Any]  # Strategy parameters/rules
    created_at: str
    regime: str
    is_active: bool
    performance_live: Optional[float]  # Live trading performance


class LearningDatabase:
    """
    Centralized database for all AI learning data.

    Uses SQLite for simplicity and portability.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trade history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                hold_duration_mins INTEGER,
                regime TEXT,
                strategy_id TEXT,
                indicators TEXT,
                outcome TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Strategy performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                regime TEXT NOT NULL,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl_pct REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                profit_factor REAL,
                last_updated TEXT,
                UNIQUE(strategy_id, regime)
            )
        """)

        # Parameter optimization results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                parameters TEXT NOT NULL,
                sharpe_ratio REAL,
                win_rate REAL,
                total_return REAL,
                max_drawdown REAL,
                num_trades INTEGER,
                backtest_period_days INTEGER
            )
        """)

        # Evolved strategies (genetic algorithm)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolved_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                fitness REAL NOT NULL,
                genes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                regime TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                performance_live REAL
            )
        """)

        # RL agent experiences (for replay buffer)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                state TEXT NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                next_state TEXT NOT NULL,
                done INTEGER NOT NULL,
                symbol TEXT NOT NULL
            )
        """)

        # Model checkpoints
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                metrics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_best INTEGER DEFAULT 0
            )
        """)

        # Regime transitions (for learning regime patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                from_regime TEXT NOT NULL,
                to_regime TEXT NOT NULL,
                indicators_at_transition TEXT,
                outcome_after_24h REAL
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Learning database initialized at {self.db_path}")

    # ==================== Trade Records ====================

    def record_trade(self, trade: TradeRecord) -> int:
        """Record a trade for learning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                timestamp, symbol, action, entry_price, exit_price,
                quantity, pnl, pnl_pct, hold_duration_mins, regime,
                strategy_id, indicators, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp, trade.symbol, trade.action, trade.entry_price,
            trade.exit_price, trade.quantity, trade.pnl, trade.pnl_pct,
            trade.hold_duration_mins, trade.regime, trade.strategy_id,
            json.dumps(trade.indicators), trade.outcome
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: int = 1000,
        outcome: Optional[str] = None
    ) -> List[TradeRecord]:
        """Get trade records with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(TradeRecord(
                id=row[0],
                timestamp=row[1],
                symbol=row[2],
                action=row[3],
                entry_price=row[4],
                exit_price=row[5],
                quantity=row[6],
                pnl=row[7],
                pnl_pct=row[8],
                hold_duration_mins=row[9],
                regime=row[10],
                strategy_id=row[11],
                indicators=json.loads(row[12]) if row[12] else {},
                outcome=row[13]
            ))

        return trades

    def get_trade_statistics(
        self,
        strategy_id: Optional[str] = None,
        regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregate statistics for trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(pnl) as total_pnl,
                MAX(pnl_pct) as best_trade,
                MIN(pnl_pct) as worst_trade,
                AVG(hold_duration_mins) as avg_hold_mins
            FROM trades WHERE pnl IS NOT NULL
        """
        params = []

        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if regime:
            query += " AND regime = ?"
            params.append(regime)

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {"total_trades": 0}

        total_trades, wins, losses = row[0], row[1] or 0, row[2] or 0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "avg_pnl_pct": row[3] or 0,
            "total_pnl": row[4] or 0,
            "best_trade": row[5] or 0,
            "worst_trade": row[6] or 0,
            "avg_hold_mins": row[7] or 0,
        }

    # ==================== Strategy Performance ====================

    def update_strategy_performance(self, perf: StrategyPerformance):
        """Update or insert strategy performance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO strategy_performance (
                strategy_id, regime, total_trades, win_rate, avg_pnl_pct,
                sharpe_ratio, max_drawdown, profit_factor, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            perf.strategy_id, perf.regime, perf.total_trades, perf.win_rate,
            perf.avg_pnl_pct, perf.sharpe_ratio, perf.max_drawdown,
            perf.profit_factor, perf.last_updated
        ))

        conn.commit()
        conn.close()

    def get_best_strategies(
        self,
        regime: Optional[str] = None,
        metric: str = "sharpe_ratio",
        limit: int = 10
    ) -> List[StrategyPerformance]:
        """Get best performing strategies."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"""
            SELECT * FROM strategy_performance
            WHERE total_trades >= 10
        """
        params = []

        if regime:
            query += " AND regime = ?"
            params.append(regime)

        query += f" ORDER BY {metric} DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            StrategyPerformance(
                strategy_id=row[1],
                regime=row[2],
                total_trades=row[3],
                win_rate=row[4],
                avg_pnl_pct=row[5],
                sharpe_ratio=row[6],
                max_drawdown=row[7],
                profit_factor=row[8],
                last_updated=row[9]
            )
            for row in rows
        ]

    # ==================== Optimization Results ====================

    def save_optimization_result(self, result: OptimizationResult) -> int:
        """Save parameter optimization result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO optimization_results (
                timestamp, symbol, regime, parameters, sharpe_ratio,
                win_rate, total_return, max_drawdown, num_trades,
                backtest_period_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.timestamp, result.symbol, result.regime,
            json.dumps(result.parameters), result.sharpe_ratio,
            result.win_rate, result.total_return, result.max_drawdown,
            result.num_trades, result.backtest_period_days
        ))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_best_parameters(
        self,
        symbol: str,
        regime: str
    ) -> Optional[Dict[str, Any]]:
        """Get best parameters for a symbol/regime combination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT parameters, sharpe_ratio FROM optimization_results
            WHERE symbol = ? AND regime = ?
            ORDER BY sharpe_ratio DESC LIMIT 1
        """, (symbol, regime))

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    # ==================== Evolved Strategies ====================

    def save_evolved_strategy(self, strategy: EvolvedStrategy) -> int:
        """Save an evolved strategy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO evolved_strategies (
                generation, fitness, genes, created_at, regime,
                is_active, performance_live
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy.generation, strategy.fitness,
            json.dumps(strategy.genes), strategy.created_at,
            strategy.regime, strategy.is_active,
            strategy.performance_live
        ))

        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return strategy_id

    def get_top_evolved_strategies(
        self,
        regime: Optional[str] = None,
        limit: int = 10,
        active_only: bool = True
    ) -> List[EvolvedStrategy]:
        """Get top evolved strategies by fitness."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM evolved_strategies WHERE 1=1"
        params = []

        if regime:
            query += " AND regime = ?"
            params.append(regime)
        if active_only:
            query += " AND is_active = 1"

        query += f" ORDER BY fitness DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            EvolvedStrategy(
                id=row[0],
                generation=row[1],
                fitness=row[2],
                genes=json.loads(row[3]),
                created_at=row[4],
                regime=row[5],
                is_active=bool(row[6]),
                performance_live=row[7]
            )
            for row in rows
        ]

    # ==================== RL Experiences ====================

    def save_rl_experience(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
        symbol: str
    ):
        """Save RL experience for replay buffer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rl_experiences (
                timestamp, state, action, reward, next_state, done, symbol
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(state),
            action,
            reward,
            json.dumps(next_state),
            int(done),
            symbol
        ))

        conn.commit()
        conn.close()

    def get_rl_experiences(
        self,
        symbol: Optional[str] = None,
        limit: int = 10000
    ) -> List[Tuple[List[float], int, float, List[float], bool]]:
        """Get RL experiences for training."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT state, action, reward, next_state, done FROM rl_experiences"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += f" ORDER BY RANDOM() LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            (json.loads(row[0]), row[1], row[2], json.loads(row[3]), bool(row[4]))
            for row in rows
        ]

    # ==================== Regime Transitions ====================

    def record_regime_transition(
        self,
        symbol: str,
        from_regime: str,
        to_regime: str,
        indicators: Dict[str, float],
        outcome_after_24h: Optional[float] = None
    ):
        """Record a regime transition for pattern learning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regime_transitions (
                timestamp, symbol, from_regime, to_regime,
                indicators_at_transition, outcome_after_24h
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, from_regime, to_regime,
            json.dumps(indicators), outcome_after_24h
        ))

        conn.commit()
        conn.close()

    def get_regime_transition_patterns(
        self,
        to_regime: str
    ) -> List[Dict[str, Any]]:
        """Get historical patterns leading to a regime."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT from_regime, indicators_at_transition, outcome_after_24h
            FROM regime_transitions
            WHERE to_regime = ? AND outcome_after_24h IS NOT NULL
        """, (to_regime,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "from_regime": row[0],
                "indicators": json.loads(row[1]) if row[1] else {},
                "outcome_after_24h": row[2]
            }
            for row in rows
        ]

    # ==================== Summary ====================

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count records in each table
        tables = [
            'trades', 'strategy_performance', 'optimization_results',
            'evolved_strategies', 'rl_experiences', 'regime_transitions'
        ]

        counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()

        return {
            "total_trades": counts['trades'],
            "strategies_tracked": counts['strategy_performance'],
            "optimizations_run": counts['optimization_results'],
            "evolved_strategies": counts['evolved_strategies'],
            "rl_experiences": counts['rl_experiences'],
            "regime_transitions": counts['regime_transitions'],
            "database_path": str(self.db_path),
        }


# Global instance
_db: Optional[LearningDatabase] = None


def get_learning_db() -> LearningDatabase:
    """Get or create the global learning database."""
    global _db
    if _db is None:
        _db = LearningDatabase()
    return _db
