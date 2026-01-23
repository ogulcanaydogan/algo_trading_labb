"""
Pattern Memory - Learning Storage.

Stores learned trading patterns for continuous learning:
- Market state -> Action -> Outcome
- Regime-specific performance
- Symbol-specific patterns
- Time-based patterns

Uses SQLite for persistence with exponential decay for recent relevance.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradingPattern:
    """A learned trading pattern."""

    pattern_id: Optional[int] = None
    symbol: str = ""
    regime: str = ""
    action: str = ""  # BUY, SELL, HOLD
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    hold_duration_minutes: int = 0
    confidence_at_entry: float = 0.0

    # Market state at entry
    rsi: float = 0.0
    macd: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0

    # Outcome
    was_profitable: bool = False
    max_drawdown_pct: float = 0.0
    max_profit_pct: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    decay_weight: float = 1.0  # Exponential decay for relevance

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "regime": self.regime,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_pct": self.pnl_pct,
            "hold_duration_minutes": self.hold_duration_minutes,
            "confidence_at_entry": self.confidence_at_entry,
            "rsi": self.rsi,
            "macd": self.macd,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "was_profitable": self.was_profitable,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_profit_pct": self.max_profit_pct,
            "timestamp": self.timestamp.isoformat(),
            "decay_weight": self.decay_weight,
        }


@dataclass
class PatternStats:
    """Statistics for a pattern category."""

    total_patterns: int = 0
    profitable_patterns: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_hold_duration: float = 0.0
    avg_confidence: float = 0.0
    best_pnl_pct: float = 0.0
    worst_pnl_pct: float = 0.0


class PatternMemory:
    """
    SQLite-based pattern memory for learning.

    Stores and retrieves trading patterns with:
    - Exponential decay weighting (recent patterns more important)
    - Filtering by regime, symbol, action
    - Performance statistics calculation
    """

    DECAY_HALFLIFE_DAYS = 7  # Patterns lose half their weight every 7 days

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(
                Path(__file__).parent.parent.parent / "data" / "intelligence" / "pattern_memory.db"
            )

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

        logger.info(f"Pattern Memory initialized: {db_path}")

    def _init_database(self):
        """Initialize the SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                pnl_pct REAL,
                hold_duration_minutes INTEGER,
                confidence_at_entry REAL,
                rsi REAL,
                macd REAL,
                volatility REAL,
                trend_strength REAL,
                was_profitable INTEGER,
                max_drawdown_pct REAL,
                max_profit_pct REAL,
                timestamp TEXT NOT NULL,
                decay_weight REAL DEFAULT 1.0
            )
        """)

        # Indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON patterns(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_regime ON patterns(regime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_action ON patterns(action)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_profitable ON patterns(was_profitable)"
        )

        # Confidence adjustments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                regime TEXT,
                action TEXT,
                adjustment_factor REAL DEFAULT 1.0,
                based_on_patterns INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)

        conn.commit()
        conn.close()

    def store_pattern(self, pattern: TradingPattern) -> int:
        """Store a new trading pattern."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO patterns (
                    symbol, regime, action, entry_price, exit_price, pnl_pct,
                    hold_duration_minutes, confidence_at_entry, rsi, macd,
                    volatility, trend_strength, was_profitable, max_drawdown_pct,
                    max_profit_pct, timestamp, decay_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern.symbol,
                    pattern.regime,
                    pattern.action,
                    pattern.entry_price,
                    pattern.exit_price,
                    pattern.pnl_pct,
                    pattern.hold_duration_minutes,
                    pattern.confidence_at_entry,
                    pattern.rsi,
                    pattern.macd,
                    pattern.volatility,
                    pattern.trend_strength,
                    1 if pattern.was_profitable else 0,
                    pattern.max_drawdown_pct,
                    pattern.max_profit_pct,
                    pattern.timestamp.isoformat(),
                    pattern.decay_weight,
                ),
            )

            pattern_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.debug(
                f"Stored pattern #{pattern_id}: {pattern.symbol} {pattern.action} {pattern.pnl_pct:+.2f}%"
            )
            return pattern_id

    def get_similar_patterns(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        min_confidence: float = 0.0,
        days_lookback: int = 90,
    ) -> List[TradingPattern]:
        """Get similar patterns from memory."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM patterns WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if regime:
                query += " AND regime = ?"
                params.append(regime)

            if action:
                query += " AND action = ?"
                params.append(action)

            if min_confidence > 0:
                query += " AND confidence_at_entry >= ?"
                params.append(min_confidence)

            # Time filter
            cutoff = (datetime.now() - timedelta(days=days_lookback)).isoformat()
            query += " AND timestamp > ?"
            params.append(cutoff)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

        # Convert to TradingPattern objects with decay weighting
        patterns = []
        for row in rows:
            pattern = TradingPattern(
                pattern_id=row[0],
                symbol=row[1],
                regime=row[2],
                action=row[3],
                entry_price=row[4] or 0,
                exit_price=row[5] or 0,
                pnl_pct=row[6] or 0,
                hold_duration_minutes=row[7] or 0,
                confidence_at_entry=row[8] or 0,
                rsi=row[9] or 0,
                macd=row[10] or 0,
                volatility=row[11] or 0,
                trend_strength=row[12] or 0,
                was_profitable=bool(row[13]),
                max_drawdown_pct=row[14] or 0,
                max_profit_pct=row[15] or 0,
                timestamp=datetime.fromisoformat(row[16]),
            )
            # Apply decay
            pattern.decay_weight = self._calculate_decay(pattern.timestamp)
            patterns.append(pattern)

        return patterns

    def get_pattern_stats(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        action: Optional[str] = None,
        days_lookback: int = 30,
    ) -> PatternStats:
        """Get statistics for a pattern category."""
        patterns = self.get_similar_patterns(
            symbol=symbol,
            regime=regime,
            action=action,
            limit=1000,
            days_lookback=days_lookback,
        )

        if not patterns:
            return PatternStats()

        # Calculate weighted statistics
        total_weight = sum(p.decay_weight for p in patterns)
        if total_weight == 0:
            return PatternStats(total_patterns=len(patterns))

        profitable = [p for p in patterns if p.was_profitable]

        weighted_pnl = sum(p.pnl_pct * p.decay_weight for p in patterns) / total_weight
        weighted_duration = (
            sum(p.hold_duration_minutes * p.decay_weight for p in patterns) / total_weight
        )
        weighted_confidence = (
            sum(p.confidence_at_entry * p.decay_weight for p in patterns) / total_weight
        )

        return PatternStats(
            total_patterns=len(patterns),
            profitable_patterns=len(profitable),
            win_rate=len(profitable) / len(patterns) if patterns else 0,
            avg_pnl_pct=weighted_pnl,
            avg_hold_duration=weighted_duration,
            avg_confidence=weighted_confidence,
            best_pnl_pct=max(p.pnl_pct for p in patterns) if patterns else 0,
            worst_pnl_pct=min(p.pnl_pct for p in patterns) if patterns else 0,
        )

    def get_confidence_adjustment(
        self,
        symbol: str,
        regime: str,
        action: str,
        base_confidence: float,
    ) -> Tuple[float, str]:
        """
        Get confidence adjustment based on historical performance.

        Returns:
            Tuple of (adjusted_confidence, reasoning)
        """
        stats = self.get_pattern_stats(symbol=symbol, regime=regime, action=action)

        if stats.total_patterns < 5:
            # Not enough data
            return base_confidence, "Insufficient pattern history"

        # Calculate adjustment factor based on win rate
        expected_win_rate = 0.5  # Baseline
        win_rate_deviation = stats.win_rate - expected_win_rate

        # Adjust confidence: boost if winning, reduce if losing
        # Max adjustment is +/- 20%
        adjustment = 1.0 + (win_rate_deviation * 0.4)
        adjustment = max(0.8, min(1.2, adjustment))

        adjusted_confidence = base_confidence * adjustment

        reasoning = (
            f"Historical: {stats.win_rate:.0%} win rate over {stats.total_patterns} patterns, "
            f"avg PnL {stats.avg_pnl_pct:+.2f}%"
        )

        return adjusted_confidence, reasoning

    def update_confidence_adjustments(self):
        """Update stored confidence adjustments based on recent performance."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get unique combinations
            cursor.execute(
                """
                SELECT DISTINCT symbol, regime, action FROM patterns
                WHERE timestamp > ?
            """,
                ((datetime.now() - timedelta(days=30)).isoformat(),),
            )

            combinations = cursor.fetchall()

            for symbol, regime, action in combinations:
                stats = self.get_pattern_stats(
                    symbol=symbol,
                    regime=regime,
                    action=action,
                    days_lookback=30,
                )

                if stats.total_patterns >= 5:
                    # Calculate adjustment factor
                    adjustment = 1.0 + ((stats.win_rate - 0.5) * 0.4)
                    adjustment = max(0.8, min(1.2, adjustment))

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO confidence_adjustments
                        (symbol, regime, action, adjustment_factor, based_on_patterns, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            regime,
                            action,
                            adjustment,
                            stats.total_patterns,
                            datetime.now().isoformat(),
                        ),
                    )

            conn.commit()
            conn.close()

        logger.info(f"Updated confidence adjustments for {len(combinations)} combinations")

    def _calculate_decay(self, timestamp: datetime) -> float:
        """Calculate exponential decay weight for a pattern."""
        age_days = (datetime.now() - timestamp).total_seconds() / 86400
        # Exponential decay: weight = 0.5^(age/halflife)
        return 0.5 ** (age_days / self.DECAY_HALFLIFE_DAYS)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pattern memory."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM patterns")
            total_patterns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM patterns WHERE was_profitable = 1")
            profitable_patterns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM patterns")
            unique_symbols = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT regime) FROM patterns")
            unique_regimes = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(pnl_pct) FROM patterns")
            avg_pnl = cursor.fetchone()[0] or 0

            conn.close()

        return {
            "total_patterns": total_patterns,
            "profitable_patterns": profitable_patterns,
            "overall_win_rate": profitable_patterns / total_patterns if total_patterns else 0,
            "unique_symbols": unique_symbols,
            "unique_regimes": unique_regimes,
            "average_pnl_pct": avg_pnl,
        }

    def cleanup_old_patterns(self, days_to_keep: int = 180):
        """Remove patterns older than specified days."""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patterns WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()

        logger.info(f"Cleaned up {deleted} old patterns (older than {days_to_keep} days)")
        return deleted
