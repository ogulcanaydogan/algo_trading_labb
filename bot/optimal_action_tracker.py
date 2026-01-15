"""
Optimal Action Tracker - Learn the best action for each market state.

This module tracks:
1. Market state at decision time (regime, indicators, patterns)
2. Action taken (BUY/SELL/HOLD, size)
3. Outcome (profit/loss, max drawdown, holding time)

Over time, builds a decision matrix showing optimal actions per state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"  # Short
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class MarketState:
    """Snapshot of market conditions at decision time."""

    # Regime
    regime: str  # bull, bear, crash, sideways, etc.
    regime_confidence: float = 0.0

    # Trend indicators
    trend_direction: str = "neutral"  # up, down, neutral
    trend_strength: float = 0.0  # 0-1

    # Technical indicators
    rsi: float = 50.0
    macd_signal: str = "neutral"  # bullish, bearish, neutral
    bb_position: float = 0.5  # 0=lower band, 1=upper band

    # Volatility
    atr_normalized: float = 0.0
    volatility_regime: str = "normal"  # low, normal, high, extreme

    # Pattern recognition
    candlestick_pattern: str = "none"  # doji, hammer, engulfing, etc.
    fibonacci_level: str = "none"  # 0.236, 0.382, 0.5, 0.618, 0.786
    support_resistance: str = "none"  # at_support, at_resistance, between

    # Volume
    volume_trend: str = "normal"  # low, normal, high, climax

    # Time context
    session: str = "unknown"  # asian, european, us, overlap
    day_of_week: int = 0

    def to_feature_vector(self) -> List[float]:
        """Convert to numeric features for ML."""
        regime_map = {"strong_bull": 2, "bull": 1, "sideways": 0, "bear": -1, "strong_bear": -2, "crash": -3}
        trend_map = {"up": 1, "neutral": 0, "down": -1}
        vol_map = {"low": 0, "normal": 1, "high": 2, "extreme": 3}

        return [
            regime_map.get(self.regime, 0),
            self.regime_confidence,
            trend_map.get(self.trend_direction, 0),
            self.trend_strength,
            (self.rsi - 50) / 50,  # Normalize to -1 to 1
            {"bullish": 1, "neutral": 0, "bearish": -1}.get(self.macd_signal, 0),
            self.bb_position * 2 - 1,  # Normalize to -1 to 1
            self.atr_normalized,
            vol_map.get(self.volatility_regime, 1),
        ]

    def to_state_key(self) -> str:
        """Create a discrete state key for lookup table."""
        rsi_bucket = "oversold" if self.rsi < 30 else "overbought" if self.rsi > 70 else "neutral"
        return f"{self.regime}|{self.trend_direction}|{rsi_bucket}|{self.volatility_regime}"


@dataclass
class ActionOutcome:
    """Result of an action."""

    action: ActionType
    entry_price: float
    exit_price: float = 0.0
    position_size: float = 0.0

    # Outcomes
    pnl: float = 0.0
    pnl_percent: float = 0.0
    max_favorable: float = 0.0  # Best unrealized P&L
    max_adverse: float = 0.0  # Worst unrealized P&L (drawdown)
    holding_time_hours: float = 0.0

    # Risk metrics
    risk_reward_achieved: float = 0.0

    # Timing
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: Optional[datetime] = None

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    @property
    def sharpe_contribution(self) -> float:
        """Estimate contribution to Sharpe ratio."""
        if self.holding_time_hours == 0:
            return 0
        daily_return = self.pnl_percent / (self.holding_time_hours / 24)
        return daily_return / max(abs(self.max_adverse), 0.01)


@dataclass
class StateActionRecord:
    """Complete record of state, action, and outcome."""

    id: Optional[int] = None
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    state: MarketState = field(default_factory=MarketState)
    outcome: ActionOutcome = field(default_factory=ActionOutcome)

    # ML model info
    model_prediction: str = ""
    model_confidence: float = 0.0

    # Strategy info
    strategy_used: str = ""
    signal_reason: str = ""


class OptimalActionTracker:
    """
    Tracks and learns optimal actions for each market state.

    Uses a combination of:
    1. Lookup table (Q-table style) for discrete states
    2. Historical outcome analysis
    3. Expected value calculation
    """

    def __init__(self, db_path: Path = Path("data/optimal_actions.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory Q-table for fast lookups
        self._q_table: Dict[str, Dict[str, float]] = {}
        self._load_q_table()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                state_key TEXT,
                state_json TEXT,
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                pnl REAL,
                pnl_percent REAL,
                max_favorable REAL,
                max_adverse REAL,
                holding_time_hours REAL,
                model_prediction TEXT,
                model_confidence REAL,
                strategy_used TEXT,
                signal_reason TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS q_table (
                state_key TEXT,
                action TEXT,
                expected_value REAL,
                count INTEGER,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                last_updated TEXT,
                PRIMARY KEY (state_key, action)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_state_key ON action_records(state_key)
        """)

        conn.commit()
        conn.close()

    def _load_q_table(self):
        """Load Q-table from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT state_key, action, expected_value FROM q_table")
            for row in cursor.fetchall():
                state_key, action, ev = row
                if state_key not in self._q_table:
                    self._q_table[state_key] = {}
                self._q_table[state_key][action] = ev

            conn.close()
            logger.info(f"Loaded Q-table with {len(self._q_table)} states")
        except Exception as e:
            logger.warning(f"Could not load Q-table: {e}")

    def record_action(self, record: StateActionRecord) -> int:
        """Record an action taken."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        state_key = record.state.to_state_key()
        state_json = json.dumps(asdict(record.state))

        cursor.execute("""
            INSERT INTO action_records
            (symbol, timestamp, state_key, state_json, action, entry_price,
             position_size, model_prediction, model_confidence, strategy_used, signal_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.symbol,
            record.timestamp.isoformat(),
            state_key,
            state_json,
            record.outcome.action.value,
            record.outcome.entry_price,
            record.outcome.position_size,
            record.model_prediction,
            record.model_confidence,
            record.strategy_used,
            record.signal_reason,
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def record_outcome(self, record_id: int, outcome: ActionOutcome):
        """Record the outcome of a previously recorded action."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE action_records SET
                exit_price = ?,
                pnl = ?,
                pnl_percent = ?,
                max_favorable = ?,
                max_adverse = ?,
                holding_time_hours = ?
            WHERE id = ?
        """, (
            outcome.exit_price,
            outcome.pnl,
            outcome.pnl_percent,
            outcome.max_favorable,
            outcome.max_adverse,
            outcome.holding_time_hours,
            record_id,
        ))

        conn.commit()
        conn.close()

        # Update Q-table
        self._update_q_table_from_record(record_id)

    def _update_q_table_from_record(self, record_id: int):
        """Update Q-table based on a completed trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT state_key, action, pnl_percent FROM action_records WHERE id = ?
        """, (record_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        state_key, action, pnl_percent = row

        # Get existing stats
        cursor.execute("""
            SELECT expected_value, count, win_rate, avg_win, avg_loss
            FROM q_table WHERE state_key = ? AND action = ?
        """, (state_key, action))

        existing = cursor.fetchone()

        if existing:
            old_ev, count, win_rate, avg_win, avg_loss = existing
            new_count = count + 1

            # Update win rate
            if pnl_percent > 0:
                new_win_rate = (win_rate * count + 1) / new_count
                new_avg_win = (avg_win * (win_rate * count) + pnl_percent) / max(1, new_win_rate * new_count)
                new_avg_loss = avg_loss
            else:
                new_win_rate = (win_rate * count) / new_count
                new_avg_win = avg_win
                loss_count = (1 - win_rate) * count
                new_avg_loss = (avg_loss * loss_count + abs(pnl_percent)) / max(1, loss_count + 1)

            # Calculate expected value
            new_ev = new_win_rate * new_avg_win - (1 - new_win_rate) * new_avg_loss

            cursor.execute("""
                UPDATE q_table SET
                    expected_value = ?,
                    count = ?,
                    win_rate = ?,
                    avg_win = ?,
                    avg_loss = ?,
                    last_updated = ?
                WHERE state_key = ? AND action = ?
            """, (new_ev, new_count, new_win_rate, new_avg_win, new_avg_loss,
                  datetime.now(timezone.utc).isoformat(), state_key, action))
        else:
            # First record for this state-action pair
            win_rate = 1.0 if pnl_percent > 0 else 0.0
            avg_win = pnl_percent if pnl_percent > 0 else 0.0
            avg_loss = abs(pnl_percent) if pnl_percent <= 0 else 0.0
            ev = pnl_percent

            cursor.execute("""
                INSERT INTO q_table
                (state_key, action, expected_value, count, win_rate, avg_win, avg_loss, last_updated)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            """, (state_key, action, ev, win_rate, avg_win, avg_loss,
                  datetime.now(timezone.utc).isoformat()))

        conn.commit()
        conn.close()

        # Update in-memory cache
        if state_key not in self._q_table:
            self._q_table[state_key] = {}
        self._q_table[state_key][action] = pnl_percent  # Simplified for now

    def get_optimal_action(self, state: MarketState) -> Tuple[ActionType, float]:
        """
        Get the optimal action for a given market state.

        Returns:
            Tuple of (action, expected_value)
        """
        state_key = state.to_state_key()

        if state_key in self._q_table:
            actions = self._q_table[state_key]
            if actions:
                best_action = max(actions.keys(), key=lambda a: actions[a])
                return ActionType(best_action), actions[best_action]

        # No data for this exact state - use heuristics
        return self._heuristic_action(state)

    def _heuristic_action(self, state: MarketState) -> Tuple[ActionType, float]:
        """Fallback heuristic when no historical data exists."""

        # Simple regime-based heuristics
        if state.regime in ("strong_bull", "bull"):
            if state.rsi < 30:  # Oversold in uptrend = buy
                return ActionType.BUY, 0.6
            elif state.rsi > 70:  # Overbought = hold/reduce
                return ActionType.HOLD, 0.3
            else:
                return ActionType.BUY, 0.4

        elif state.regime in ("strong_bear", "bear", "crash"):
            if state.rsi > 70:  # Overbought in downtrend = sell
                return ActionType.SELL, 0.6
            elif state.rsi < 30:  # Oversold = hold/cover
                return ActionType.HOLD, 0.3
            else:
                return ActionType.SELL, 0.4

        else:  # Sideways/unknown
            if state.rsi < 30:
                return ActionType.BUY, 0.3
            elif state.rsi > 70:
                return ActionType.SELL, 0.3
            else:
                return ActionType.HOLD, 0.5

    def get_action_stats(self, state_key: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for actions, optionally filtered by state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if state_key:
            cursor.execute("""
                SELECT action,
                       COUNT(*) as count,
                       AVG(pnl_percent) as avg_return,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM action_records
                WHERE state_key = ? AND pnl IS NOT NULL
                GROUP BY action
            """, (state_key,))
        else:
            cursor.execute("""
                SELECT action,
                       COUNT(*) as count,
                       AVG(pnl_percent) as avg_return,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM action_records
                WHERE pnl IS NOT NULL
                GROUP BY action
            """)

        results = {}
        for row in cursor.fetchall():
            action, count, avg_return, win_rate = row
            results[action] = {
                "count": count,
                "avg_return": round(avg_return or 0, 4),
                "win_rate": round(win_rate or 0, 4),
            }

        conn.close()
        return results

    def get_best_states(self, action: str = "buy", min_count: int = 5) -> List[Dict]:
        """Get market states where an action performs best."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT state_key,
                   COUNT(*) as count,
                   AVG(pnl_percent) as avg_return,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
            FROM action_records
            WHERE action = ? AND pnl IS NOT NULL
            GROUP BY state_key
            HAVING count >= ?
            ORDER BY avg_return DESC
            LIMIT 20
        """, (action, min_count))

        results = []
        for row in cursor.fetchall():
            state_key, count, avg_return, win_rate = row
            results.append({
                "state": state_key,
                "count": count,
                "avg_return": round(avg_return, 4),
                "win_rate": round(win_rate, 4),
            })

        conn.close()
        return results


# Singleton instance
_tracker: Optional[OptimalActionTracker] = None

def get_tracker() -> OptimalActionTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = OptimalActionTracker()
    return _tracker
