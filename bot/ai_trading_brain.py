"""
AI Trading Brain - Intelligent Self-Learning Trading System

Goal: Achieve minimum 1% daily gains while minimizing losses.

This system:
1. Learns from ALL historical market data what conditions lead to profits
2. Analyzes every trade we make to understand what worked and what didn't
3. Develops and backtests strategies automatically
4. Monitors execution in real-time to optimize entry/exit
5. Tracks daily target progress and adjusts behavior accordingly

Core Philosophy:
- Learn from history, but adapt to current conditions
- Cut losses FAST, let winners run
- Never risk more than we can afford to lose
- Compound gains, protect capital
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MarketCondition(Enum):
    """Market condition classifications."""
    STRONG_BULL = "strong_bull"      # Clear uptrend, high momentum
    BULL = "bull"                     # Uptrend
    WEAK_BULL = "weak_bull"          # Slight uptrend
    SIDEWAYS = "sideways"            # No clear direction
    WEAK_BEAR = "weak_bear"          # Slight downtrend
    BEAR = "bear"                     # Downtrend
    STRONG_BEAR = "strong_bear"      # Clear downtrend, high momentum
    CRASH = "crash"                   # Rapid decline
    VOLATILE = "volatile"            # High volatility, unclear direction


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    timestamp: datetime
    symbol: str
    price: float

    # Trend indicators
    trend_1h: str = "neutral"    # up/down/neutral
    trend_4h: str = "neutral"
    trend_1d: str = "neutral"

    # Momentum
    rsi: float = 50.0
    macd_histogram: float = 0.0
    momentum_score: float = 0.0   # -1 to 1

    # Volatility
    atr_percent: float = 0.0
    volatility_percentile: float = 50.0  # Where current vol is vs history

    # Volume
    volume_ratio: float = 1.0     # vs average

    # Support/Resistance
    distance_to_support: float = 0.0   # % from nearest support
    distance_to_resistance: float = 0.0

    # Overall
    condition: MarketCondition = MarketCondition.SIDEWAYS
    confidence: float = 0.5


@dataclass
class TradeLesson:
    """What we learned from a single trade."""
    trade_id: str
    symbol: str

    # Entry conditions
    entry_condition: MarketCondition
    entry_rsi: float
    entry_trend: str
    entry_volatility: float

    # What we did
    action: str  # buy, sell, short
    entry_price: float
    exit_price: float
    position_size: float

    # Outcome
    pnl_percent: float
    max_profit_percent: float    # Best unrealized gain
    max_drawdown_percent: float  # Worst unrealized loss
    holding_hours: float

    # Lessons
    should_have_entered: bool = True
    optimal_entry_price: Optional[float] = None
    optimal_exit_price: Optional[float] = None
    optimal_stop_loss: Optional[float] = None
    lesson_notes: str = ""


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_percent: float = 0.0
    avg_win_percent: float = 0.0
    avg_loss_percent: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0

    # Per condition performance
    performance_by_condition: Dict[str, Dict] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def expectancy(self) -> float:
        """Expected return per trade."""
        if self.total_trades == 0:
            return 0.0
        return (self.win_rate * self.avg_win_percent -
                (1 - self.win_rate) * abs(self.avg_loss_percent))


@dataclass
class DailyTarget:
    """Daily performance target tracking."""
    date: str
    target_percent: float = 1.0
    current_percent: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0

    # Risk budget
    max_daily_loss: float = 2.0   # Max we can lose today
    current_drawdown: float = 0.0

    # Status
    target_achieved: bool = False
    risk_budget_used: float = 0.0  # 0-100%

    @property
    def remaining_to_target(self) -> float:
        return max(0, self.target_percent - self.current_percent)

    @property
    def can_still_trade(self) -> bool:
        return self.current_drawdown < self.max_daily_loss


# =============================================================================
# HISTORICAL PATTERN LEARNER
# =============================================================================

class HistoricalPatternLearner:
    """
    Analyzes historical data to learn what conditions lead to profits.

    Builds a knowledge base of:
    - What market conditions precede big moves
    - What indicator combinations work best
    - What time patterns exist (day of week, time of day)
    - What sequences of conditions lead to profits
    """

    def __init__(self, db_path: Path = Path("data/ai_brain/patterns.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Pattern knowledge base (in memory for fast access)
        self.profitable_patterns: Dict[str, float] = {}
        self.losing_patterns: Dict[str, float] = {}
        self._load_patterns()

    def _init_db(self):
        """Initialize database for pattern storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_key TEXT UNIQUE,
                condition TEXT,
                rsi_bucket TEXT,
                trend TEXT,
                volatility TEXT,
                total_occurrences INTEGER DEFAULT 0,
                profitable_occurrences INTEGER DEFAULT 0,
                avg_return_percent REAL DEFAULT 0,
                best_action TEXT,
                avg_holding_hours REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                condition TEXT,
                rsi REAL,
                volatility REAL,
                next_1h_return REAL,
                next_4h_return REAL,
                next_24h_return REAL,
                optimal_action TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_patterns(self):
        """Load known patterns from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT pattern_key, avg_return_percent, best_action
                FROM market_patterns
                WHERE total_occurrences >= 10
            """)

            for row in cursor.fetchall():
                pattern_key, avg_return, best_action = row
                if avg_return > 0:
                    self.profitable_patterns[pattern_key] = avg_return
                else:
                    self.losing_patterns[pattern_key] = avg_return

            conn.close()
            logger.info(f"Loaded {len(self.profitable_patterns)} profitable patterns, "
                       f"{len(self.losing_patterns)} losing patterns")
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")

    def create_pattern_key(self, snapshot: MarketSnapshot) -> str:
        """Create a unique key for market conditions."""
        rsi_bucket = "oversold" if snapshot.rsi < 30 else "overbought" if snapshot.rsi > 70 else "neutral"
        vol_bucket = "low" if snapshot.volatility_percentile < 30 else "high" if snapshot.volatility_percentile > 70 else "normal"

        return f"{snapshot.condition.value}|{rsi_bucket}|{snapshot.trend_1h}|{vol_bucket}"

    def learn_from_movement(
        self,
        snapshot: MarketSnapshot,
        next_1h_return: float,
        next_4h_return: float,
        next_24h_return: float
    ):
        """Learn from an observed price movement."""
        pattern_key = self.create_pattern_key(snapshot)

        # Determine optimal action based on returns
        if next_4h_return > 1.0:
            optimal_action = "buy"
        elif next_4h_return < -1.0:
            optimal_action = "sell"
        else:
            optimal_action = "hold"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Record the movement
        cursor.execute("""
            INSERT INTO price_movements
            (timestamp, symbol, condition, rsi, volatility,
             next_1h_return, next_4h_return, next_24h_return, optimal_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.timestamp.isoformat(),
            snapshot.symbol,
            snapshot.condition.value,
            snapshot.rsi,
            snapshot.volatility_percentile,
            next_1h_return,
            next_4h_return,
            next_24h_return,
            optimal_action
        ))

        # Update pattern statistics
        is_profitable = next_4h_return > 0

        cursor.execute("""
            INSERT INTO market_patterns
            (pattern_key, condition, rsi_bucket, trend, volatility,
             total_occurrences, profitable_occurrences, avg_return_percent,
             best_action, last_updated)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(pattern_key) DO UPDATE SET
                total_occurrences = total_occurrences + 1,
                profitable_occurrences = profitable_occurrences + ?,
                avg_return_percent = (avg_return_percent * total_occurrences + ?) / (total_occurrences + 1),
                best_action = ?,
                last_updated = ?
        """, (
            pattern_key,
            snapshot.condition.value,
            "oversold" if snapshot.rsi < 30 else "overbought" if snapshot.rsi > 70 else "neutral",
            snapshot.trend_1h,
            "low" if snapshot.volatility_percentile < 30 else "high" if snapshot.volatility_percentile > 70 else "normal",
            1 if is_profitable else 0,
            next_4h_return,
            optimal_action,
            datetime.now(timezone.utc).isoformat(),
            1 if is_profitable else 0,
            next_4h_return,
            optimal_action,
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()

        # Update in-memory cache
        if next_4h_return > 0:
            self.profitable_patterns[pattern_key] = next_4h_return
        else:
            self.losing_patterns[pattern_key] = next_4h_return

    def get_pattern_expectation(self, snapshot: MarketSnapshot) -> Tuple[str, float, float]:
        """
        Get expected outcome for current market conditions.

        Returns:
            (recommended_action, expected_return, confidence)
        """
        pattern_key = self.create_pattern_key(snapshot)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT best_action, avg_return_percent, total_occurrences, profitable_occurrences
            FROM market_patterns
            WHERE pattern_key = ?
        """, (pattern_key,))

        row = cursor.fetchone()
        conn.close()

        if row:
            best_action, avg_return, total, profitable = row
            confidence = min(1.0, total / 100)  # More data = more confidence
            win_rate = profitable / total if total > 0 else 0.5

            # Adjust recommendation based on win rate
            if win_rate < 0.4:
                best_action = "hold"  # Don't trade low win rate patterns

            return best_action, avg_return, confidence * win_rate

        # No data for this pattern - return neutral
        return "hold", 0.0, 0.0

    def get_best_conditions_for_action(self, action: str, min_samples: int = 20) -> List[Dict]:
        """Get market conditions where an action performs best."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pattern_key, avg_return_percent, total_occurrences,
                   profitable_occurrences * 1.0 / total_occurrences as win_rate
            FROM market_patterns
            WHERE best_action = ? AND total_occurrences >= ?
            ORDER BY avg_return_percent DESC
            LIMIT 20
        """, (action, min_samples))

        results = []
        for row in cursor.fetchall():
            pattern_key, avg_return, total, win_rate = row
            parts = pattern_key.split("|")
            results.append({
                "condition": parts[0] if len(parts) > 0 else "unknown",
                "rsi": parts[1] if len(parts) > 1 else "neutral",
                "trend": parts[2] if len(parts) > 2 else "neutral",
                "volatility": parts[3] if len(parts) > 3 else "normal",
                "avg_return": round(avg_return, 2),
                "win_rate": round(win_rate, 2),
                "samples": total
            })

        conn.close()
        return results


# =============================================================================
# TRADE OUTCOME ANALYZER
# =============================================================================

class TradeOutcomeAnalyzer:
    """
    Analyzes every trade we make to learn what works.

    For each trade, asks:
    - Should we have entered at all?
    - Did we enter at the right price?
    - Did we exit at the right time?
    - What was the optimal stop loss?
    - What could we have done better?
    """

    def __init__(self, db_path: Path = Path("data/ai_brain/trade_analysis.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Aggregated lessons
        self.lessons_learned: List[str] = []
        self.optimal_parameters: Dict[str, Any] = {}

    def _init_db(self):
        """Initialize trade analysis database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT,
                timestamp TEXT,

                -- Entry conditions
                entry_condition TEXT,
                entry_rsi REAL,
                entry_trend TEXT,
                entry_volatility REAL,

                -- Trade details
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,

                -- Outcomes
                pnl_percent REAL,
                max_profit_percent REAL,
                max_drawdown_percent REAL,
                holding_hours REAL,

                -- Analysis
                should_have_entered INTEGER,
                optimal_entry_price REAL,
                optimal_exit_price REAL,
                optimal_stop_loss REAL,
                lesson_notes TEXT,

                -- Grades
                entry_grade TEXT,
                exit_grade TEXT,
                overall_grade TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                insight_key TEXT,
                insight_value TEXT,
                confidence REAL,
                sample_count INTEGER,
                last_updated TEXT,
                UNIQUE(insight_type, insight_key)
            )
        """)

        conn.commit()
        conn.close()

    def analyze_trade(
        self,
        trade_id: str,
        symbol: str,
        entry_snapshot: MarketSnapshot,
        action: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        price_history: List[float],  # Prices during the trade
        holding_hours: float
    ) -> TradeLesson:
        """
        Perform deep analysis of a completed trade.
        """
        # Calculate outcomes
        if action == "buy":
            pnl_percent = (exit_price - entry_price) / entry_price * 100
        else:  # sell/short
            pnl_percent = (entry_price - exit_price) / entry_price * 100

        # Find max profit and drawdown during trade
        if price_history:
            if action == "buy":
                max_price = max(price_history)
                min_price = min(price_history)
                max_profit_percent = (max_price - entry_price) / entry_price * 100
                max_drawdown_percent = (entry_price - min_price) / entry_price * 100
            else:
                max_price = max(price_history)
                min_price = min(price_history)
                max_profit_percent = (entry_price - min_price) / entry_price * 100
                max_drawdown_percent = (max_price - entry_price) / entry_price * 100
        else:
            max_profit_percent = max(0, pnl_percent)
            max_drawdown_percent = max(0, -pnl_percent)

        # Determine if we should have entered
        should_have_entered = True
        lesson_notes = []

        # Rule: Don't enter in extreme volatility
        if entry_snapshot.volatility_percentile > 90:
            should_have_entered = False
            lesson_notes.append("Entered during extreme volatility - avoid")

        # Rule: Don't buy overbought, don't sell oversold
        if action == "buy" and entry_snapshot.rsi > 75:
            should_have_entered = False
            lesson_notes.append("Bought when overbought (RSI > 75)")
        elif action in ("sell", "short") and entry_snapshot.rsi < 25:
            should_have_entered = False
            lesson_notes.append("Sold when oversold (RSI < 25)")

        # Analyze entry timing
        optimal_entry_price = None
        if price_history:
            if action == "buy":
                optimal_entry_price = min(price_history[:len(price_history)//4] or [entry_price])
                if optimal_entry_price < entry_price * 0.99:
                    lesson_notes.append(f"Could have entered {((entry_price - optimal_entry_price) / entry_price * 100):.1f}% lower")
            else:
                optimal_entry_price = max(price_history[:len(price_history)//4] or [entry_price])
                if optimal_entry_price > entry_price * 1.01:
                    lesson_notes.append(f"Could have entered {((optimal_entry_price - entry_price) / entry_price * 100):.1f}% higher")

        # Analyze exit timing
        optimal_exit_price = None
        if price_history:
            if action == "buy":
                optimal_exit_price = max(price_history)
                if optimal_exit_price > exit_price * 1.02:
                    missed_profit = (optimal_exit_price - exit_price) / exit_price * 100
                    lesson_notes.append(f"Exited too early - missed {missed_profit:.1f}% more profit")
            else:
                optimal_exit_price = min(price_history)
                if optimal_exit_price < exit_price * 0.98:
                    missed_profit = (exit_price - optimal_exit_price) / exit_price * 100
                    lesson_notes.append(f"Exited too early - missed {missed_profit:.1f}% more profit")

        # Calculate optimal stop loss (based on max drawdown)
        optimal_stop_loss = None
        if max_drawdown_percent > 2:
            # Stop loss should have been tighter
            if action == "buy":
                optimal_stop_loss = entry_price * 0.98  # 2% stop
            else:
                optimal_stop_loss = entry_price * 1.02
            lesson_notes.append(f"Max drawdown was {max_drawdown_percent:.1f}% - use tighter stops")

        # Grade the trade
        entry_grade = self._grade_entry(pnl_percent, should_have_entered)
        exit_grade = self._grade_exit(pnl_percent, max_profit_percent)
        overall_grade = self._grade_overall(pnl_percent, max_drawdown_percent)

        # Create lesson
        lesson = TradeLesson(
            trade_id=trade_id,
            symbol=symbol,
            entry_condition=entry_snapshot.condition,
            entry_rsi=entry_snapshot.rsi,
            entry_trend=entry_snapshot.trend_1h,
            entry_volatility=entry_snapshot.volatility_percentile,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl_percent=pnl_percent,
            max_profit_percent=max_profit_percent,
            max_drawdown_percent=max_drawdown_percent,
            holding_hours=holding_hours,
            should_have_entered=should_have_entered,
            optimal_entry_price=optimal_entry_price,
            optimal_exit_price=optimal_exit_price,
            optimal_stop_loss=optimal_stop_loss,
            lesson_notes="; ".join(lesson_notes)
        )

        # Save to database
        self._save_lesson(lesson, entry_grade, exit_grade, overall_grade)

        return lesson

    def _grade_entry(self, pnl: float, should_have: bool) -> str:
        if not should_have and pnl < 0:
            return "F"  # Shouldn't have entered and lost
        if pnl > 2:
            return "A"
        if pnl > 0:
            return "B"
        if pnl > -1:
            return "C"
        return "D"

    def _grade_exit(self, pnl: float, max_profit: float) -> str:
        if max_profit == 0:
            return "C"
        capture_ratio = pnl / max_profit if max_profit > 0 else 0
        if capture_ratio > 0.8:
            return "A"
        if capture_ratio > 0.6:
            return "B"
        if capture_ratio > 0.3:
            return "C"
        return "D"

    def _grade_overall(self, pnl: float, max_dd: float) -> str:
        if pnl > 2 and max_dd < 1:
            return "A"
        if pnl > 0 and max_dd < 2:
            return "B"
        if pnl > -1:
            return "C"
        return "D"

    def _save_lesson(self, lesson: TradeLesson, entry_grade: str, exit_grade: str, overall_grade: str):
        """Save lesson to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trade_lessons
            (trade_id, symbol, timestamp, entry_condition, entry_rsi, entry_trend,
             entry_volatility, action, entry_price, exit_price, position_size,
             pnl_percent, max_profit_percent, max_drawdown_percent, holding_hours,
             should_have_entered, optimal_entry_price, optimal_exit_price,
             optimal_stop_loss, lesson_notes, entry_grade, exit_grade, overall_grade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lesson.trade_id,
            lesson.symbol,
            datetime.now(timezone.utc).isoformat(),
            lesson.entry_condition.value,
            lesson.entry_rsi,
            lesson.entry_trend,
            lesson.entry_volatility,
            lesson.action,
            lesson.entry_price,
            lesson.exit_price,
            lesson.position_size,
            lesson.pnl_percent,
            lesson.max_profit_percent,
            lesson.max_drawdown_percent,
            lesson.holding_hours,
            1 if lesson.should_have_entered else 0,
            lesson.optimal_entry_price,
            lesson.optimal_exit_price,
            lesson.optimal_stop_loss,
            lesson.lesson_notes,
            entry_grade,
            exit_grade,
            overall_grade
        ))

        conn.commit()
        conn.close()

    def get_insights(self) -> Dict[str, Any]:
        """Get aggregated insights from all trade analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Best conditions to trade
        cursor.execute("""
            SELECT entry_condition, AVG(pnl_percent), COUNT(*) as cnt
            FROM trade_lessons
            GROUP BY entry_condition
            HAVING cnt >= 5
            ORDER BY AVG(pnl_percent) DESC
        """)
        best_conditions = [{"condition": r[0], "avg_pnl": r[1], "trades": r[2]}
                          for r in cursor.fetchall()]

        # Worst conditions (avoid)
        cursor.execute("""
            SELECT entry_condition, AVG(pnl_percent), COUNT(*) as cnt
            FROM trade_lessons
            WHERE pnl_percent < 0
            GROUP BY entry_condition
            HAVING cnt >= 3
            ORDER BY AVG(pnl_percent) ASC
            LIMIT 5
        """)
        avoid_conditions = [{"condition": r[0], "avg_loss": r[1], "trades": r[2]}
                           for r in cursor.fetchall()]

        # Optimal holding time
        cursor.execute("""
            SELECT AVG(holding_hours) FROM trade_lessons WHERE pnl_percent > 0
        """)
        row = cursor.fetchone()
        optimal_hold_time = row[0] if row and row[0] else 4.0

        # Entry timing issues
        cursor.execute("""
            SELECT COUNT(*) FROM trade_lessons
            WHERE optimal_entry_price IS NOT NULL
            AND ABS(optimal_entry_price - entry_price) / entry_price > 0.01
        """)
        entry_timing_issues = cursor.fetchone()[0]

        # Exit timing issues
        cursor.execute("""
            SELECT COUNT(*) FROM trade_lessons
            WHERE optimal_exit_price IS NOT NULL
            AND ABS(optimal_exit_price - exit_price) / exit_price > 0.02
        """)
        exit_timing_issues = cursor.fetchone()[0]

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*),
                AVG(pnl_percent),
                SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                AVG(max_drawdown_percent)
            FROM trade_lessons
        """)
        stats = cursor.fetchone()

        conn.close()

        return {
            "total_trades_analyzed": stats[0] if stats else 0,
            "average_pnl": round(stats[1], 2) if stats and stats[1] else 0,
            "win_rate": round(stats[2] * 100, 1) if stats and stats[2] else 0,
            "avg_max_drawdown": round(stats[3], 2) if stats and stats[3] else 0,
            "best_conditions": best_conditions[:5],
            "avoid_conditions": avoid_conditions,
            "optimal_holding_hours": round(optimal_hold_time, 1),
            "entry_timing_issues": entry_timing_issues,
            "exit_timing_issues": exit_timing_issues,
            "recommendations": self._generate_recommendations(
                best_conditions, avoid_conditions, optimal_hold_time,
                entry_timing_issues, exit_timing_issues
            )
        }

    def _generate_recommendations(
        self,
        best_conditions: List,
        avoid_conditions: List,
        optimal_hold: float,
        entry_issues: int,
        exit_issues: int
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if best_conditions:
            top = best_conditions[0]
            recommendations.append(
                f"Focus on {top['condition']} conditions (avg {top['avg_pnl']:.1f}% return)"
            )

        if avoid_conditions:
            worst = avoid_conditions[0]
            recommendations.append(
                f"AVOID trading in {worst['condition']} (avg {worst['avg_loss']:.1f}% loss)"
            )

        recommendations.append(f"Optimal holding time: ~{optimal_hold:.0f} hours")

        if entry_issues > 5:
            recommendations.append("Work on entry timing - often entering too early/late")

        if exit_issues > 5:
            recommendations.append("Work on exit timing - leaving money on the table")

        return recommendations


# =============================================================================
# DAILY TARGET TRACKER
# =============================================================================

class DailyTargetTracker:
    """
    Tracks progress toward daily 1% goal.

    Adjusts behavior based on:
    - Progress toward target
    - Risk budget used
    - Time remaining in day
    """

    def __init__(
        self,
        target_percent: float = 1.0,
        max_daily_loss: float = 2.0,
        data_dir: Path = Path("data/ai_brain")
    ):
        self.target_percent = target_percent
        self.max_daily_loss = max_daily_loss
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.current_day: Optional[DailyTarget] = None
        self._load_today()

    def _load_today(self):
        """Load or create today's target."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_file = self.data_dir / f"daily_target_{today}.json"

        if target_file.exists():
            try:
                with open(target_file) as f:
                    data = json.load(f)
                    self.current_day = DailyTarget(
                        date=data["date"],
                        target_percent=data.get("target_percent", self.target_percent),
                        current_percent=data.get("current_percent", 0.0),
                        trades_today=data.get("trades_today", 0),
                        wins_today=data.get("wins_today", 0),
                        losses_today=data.get("losses_today", 0),
                        max_daily_loss=data.get("max_daily_loss", self.max_daily_loss),
                        current_drawdown=data.get("current_drawdown", 0.0),
                        target_achieved=data.get("target_achieved", False),
                        risk_budget_used=data.get("risk_budget_used", 0.0)
                    )
            except Exception as e:
                logger.warning(f"Could not load daily target: {e}")
                self._create_new_day(today)
        else:
            self._create_new_day(today)

    def _create_new_day(self, date: str):
        """Create new daily target."""
        self.current_day = DailyTarget(
            date=date,
            target_percent=self.target_percent,
            max_daily_loss=self.max_daily_loss
        )
        self._save()

    def _save(self):
        """Save current day to file."""
        if not self.current_day:
            return

        target_file = self.data_dir / f"daily_target_{self.current_day.date}.json"
        with open(target_file, "w") as f:
            json.dump(asdict(self.current_day), f, indent=2)

    def record_trade(self, pnl_percent: float) -> Dict[str, Any]:
        """Record a trade and update daily progress."""
        if not self.current_day:
            self._load_today()

        # Check if new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.current_day.date != today:
            self._create_new_day(today)

        # Update stats
        self.current_day.trades_today += 1
        self.current_day.current_percent += pnl_percent

        if pnl_percent > 0:
            self.current_day.wins_today += 1
        else:
            self.current_day.losses_today += 1
            self.current_day.current_drawdown = max(
                self.current_day.current_drawdown,
                abs(pnl_percent)
            )

        # Check target
        if self.current_day.current_percent >= self.current_day.target_percent:
            self.current_day.target_achieved = True

        # Update risk budget
        self.current_day.risk_budget_used = (
            self.current_day.current_drawdown / self.current_day.max_daily_loss * 100
        )

        self._save()

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get current daily status."""
        if not self.current_day:
            self._load_today()

        return {
            "date": self.current_day.date,
            "target": f"{self.current_day.target_percent}%",
            "target_percent": self.current_day.target_percent,
            "current": f"{self.current_day.current_percent:.2f}%",
            "current_pnl_percent": self.current_day.current_percent,
            "remaining": f"{self.current_day.remaining_to_target:.2f}%",
            "target_achieved": self.current_day.target_achieved,
            "loss_limit_hit": not self.current_day.can_still_trade,
            "trades": self.current_day.trades_today,
            "wins": self.current_day.wins_today,
            "losses": self.current_day.losses_today,
            "win_rate": f"{(self.current_day.wins_today / self.current_day.trades_today * 100):.0f}%" if self.current_day.trades_today > 0 else "N/A",
            "risk_budget_used": f"{self.current_day.risk_budget_used:.0f}%",
            "can_still_trade": self.current_day.can_still_trade,
            "recommendation": self._get_recommendation()
        }

    def should_auto_pause(self) -> Tuple[bool, str]:
        """Check if trading should be auto-paused."""
        if not self.current_day:
            self._load_today()

        if not self.current_day.can_still_trade:
            return True, f"DAILY LOSS LIMIT HIT ({self.current_day.current_percent:.2f}%)"

        if self.current_day.target_achieved and self.current_day.trades_today >= 5:
            return True, f"TARGET ACHIEVED ({self.current_day.current_percent:.2f}%) - Protecting gains"

        return False, ""

    def _get_recommendation(self) -> str:
        """Get recommendation based on current status."""
        if not self.current_day:
            return "Start trading"

        if self.current_day.target_achieved:
            return "TARGET ACHIEVED! Consider reducing position sizes or stopping for the day."

        if not self.current_day.can_still_trade:
            return "STOP TRADING - Daily loss limit reached. Protect capital."

        if self.current_day.risk_budget_used > 75:
            return "CAUTION - Risk budget nearly exhausted. Only take high-confidence trades."

        if self.current_day.risk_budget_used > 50:
            return "Be selective - take only A-grade setups."

        remaining = self.current_day.remaining_to_target
        if remaining > 0.5:
            return f"Need {remaining:.2f}% more. Look for momentum opportunities."
        elif remaining > 0:
            return f"Almost there! {remaining:.2f}% to go. Be patient, don't force trades."

        return "On track - maintain discipline."

    def should_trade(self, signal_confidence: float) -> Tuple[bool, str]:
        """Determine if we should take a trade given current daily status."""
        if not self.current_day:
            self._load_today()

        # Never trade if loss limit reached
        if not self.current_day.can_still_trade:
            return False, "Daily loss limit reached"

        # If target achieved, only take very high confidence trades
        if self.current_day.target_achieved:
            if signal_confidence < 0.8:
                return False, "Target achieved - only taking 80%+ confidence trades"

        # Adjust confidence requirement based on risk budget
        min_confidence = 0.5
        if self.current_day.risk_budget_used > 75:
            min_confidence = 0.75
        elif self.current_day.risk_budget_used > 50:
            min_confidence = 0.65

        if signal_confidence < min_confidence:
            return False, f"Need {min_confidence:.0%} confidence (have {signal_confidence:.0%})"

        return True, "Trade approved"


# =============================================================================
# STRATEGY GENERATOR & TESTER
# =============================================================================

@dataclass
class TradingStrategy:
    """A trading strategy with entry/exit rules."""
    name: str
    description: str

    # Entry conditions
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    # Exit conditions
    exit_conditions: Dict[str, Any] = field(default_factory=dict)

    # Risk parameters
    position_size: float = 1.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 3.0
    max_holding_hours: float = 24.0

    # Performance metrics
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    live_results: Dict[str, Any] = field(default_factory=dict)

    # Status
    is_active: bool = False
    created_at: str = ""
    last_trade: Optional[str] = None


class StrategyGenerator:
    """
    Generates and tests trading strategies based on learned patterns.

    Creates strategies by:
    1. Analyzing best performing conditions from pattern learner
    2. Combining conditions into entry/exit rules
    3. Backtesting on historical data
    4. Ranking and selecting best strategies
    """

    def __init__(
        self,
        pattern_learner: HistoricalPatternLearner,
        data_dir: Path = Path("data/ai_brain")
    ):
        self.pattern_learner = pattern_learner
        self.data_dir = data_dir
        self.strategies_file = data_dir / "strategies.json"

        # Active strategies
        self.strategies: List[TradingStrategy] = []
        self._load_strategies()

    def _load_strategies(self):
        """Load existing strategies."""
        try:
            if self.strategies_file.exists():
                with open(self.strategies_file) as f:
                    data = json.load(f)
                    for s in data.get("strategies", []):
                        self.strategies.append(TradingStrategy(
                            name=s["name"],
                            description=s["description"],
                            entry_conditions=s.get("entry_conditions", {}),
                            exit_conditions=s.get("exit_conditions", {}),
                            position_size=s.get("position_size", 1.0),
                            stop_loss_percent=s.get("stop_loss_percent", 2.0),
                            take_profit_percent=s.get("take_profit_percent", 3.0),
                            max_holding_hours=s.get("max_holding_hours", 24.0),
                            backtest_results=s.get("backtest_results", {}),
                            live_results=s.get("live_results", {}),
                            is_active=s.get("is_active", False),
                            created_at=s.get("created_at", ""),
                            last_trade=s.get("last_trade")
                        ))
        except Exception as e:
            logger.warning(f"Could not load strategies: {e}")

    def _save_strategies(self):
        """Save strategies to file."""
        try:
            data = {
                "strategies": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "entry_conditions": s.entry_conditions,
                        "exit_conditions": s.exit_conditions,
                        "position_size": s.position_size,
                        "stop_loss_percent": s.stop_loss_percent,
                        "take_profit_percent": s.take_profit_percent,
                        "max_holding_hours": s.max_holding_hours,
                        "backtest_results": s.backtest_results,
                        "live_results": s.live_results,
                        "is_active": s.is_active,
                        "created_at": s.created_at,
                        "last_trade": s.last_trade
                    }
                    for s in self.strategies
                ],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.strategies_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save strategies: {e}")

    def generate_strategy_from_patterns(self) -> Optional[TradingStrategy]:
        """Generate a new strategy based on best performing patterns."""
        # Get best buy and sell conditions
        best_buy = self.pattern_learner.get_best_conditions_for_action("buy", 10)
        best_sell = self.pattern_learner.get_best_conditions_for_action("sell", 10)

        if not best_buy or not best_sell:
            logger.warning("Not enough pattern data to generate strategy")
            return None

        top_buy = best_buy[0]
        top_sell = best_sell[0] if best_sell else {"condition": "overbought"}

        # Create strategy
        strategy = TradingStrategy(
            name=f"Pattern_{top_buy['condition']}_{datetime.now().strftime('%Y%m%d')}",
            description=f"Buy in {top_buy['condition']} conditions, exit on {top_sell.get('condition', 'reversal')}",
            entry_conditions={
                "market_condition": top_buy["condition"],
                "rsi_state": top_buy.get("rsi", "neutral"),
                "trend": top_buy.get("trend", "up"),
                "min_confidence": 0.6
            },
            exit_conditions={
                "take_profit": top_buy.get("avg_return", 2.0),
                "stop_loss": max(1.0, abs(top_buy.get("avg_return", 2.0)) * 0.5),
                "rsi_overbought": 75,
                "max_hold_hours": 24
            },
            position_size=1.0 if top_buy.get("win_rate", 0.5) > 0.6 else 0.5,
            stop_loss_percent=max(1.5, abs(top_buy.get("avg_return", 2.0)) * 0.5),
            take_profit_percent=max(1.0, top_buy.get("avg_return", 2.0)),
            created_at=datetime.now(timezone.utc).isoformat(),
            is_active=False
        )

        self.strategies.append(strategy)
        self._save_strategies()

        logger.info(f"Generated new strategy: {strategy.name}")
        return strategy

    def evaluate_strategy_for_conditions(
        self,
        snapshot: MarketSnapshot,
        signal_confidence: float
    ) -> Optional[Tuple[TradingStrategy, str, float]]:
        """
        Find the best strategy for current market conditions.

        Returns: (strategy, action, confidence) or None
        """
        best_match = None
        best_score = 0.0

        for strategy in self.strategies:
            if not strategy.is_active:
                continue

            # Check if conditions match
            entry_cond = strategy.entry_conditions
            score = 0.0

            # Market condition match
            if entry_cond.get("market_condition") == snapshot.condition.value:
                score += 0.4
            elif entry_cond.get("market_condition", "").split("_")[0] in snapshot.condition.value:
                score += 0.2  # Partial match

            # RSI match
            rsi_state = entry_cond.get("rsi_state", "neutral")
            if rsi_state == "oversold" and snapshot.rsi < 30:
                score += 0.2
            elif rsi_state == "overbought" and snapshot.rsi > 70:
                score += 0.2
            elif rsi_state == "neutral" and 30 <= snapshot.rsi <= 70:
                score += 0.1

            # Trend match
            if entry_cond.get("trend") == snapshot.trend_1h:
                score += 0.2

            # Confidence requirement
            min_conf = entry_cond.get("min_confidence", 0.5)
            if signal_confidence >= min_conf:
                score += 0.2

            # Check backtest performance
            backtest = strategy.backtest_results
            if backtest.get("win_rate", 0) > 0.5:
                score *= (1 + backtest.get("win_rate", 0.5) - 0.5)

            if score > best_score:
                best_score = score
                best_match = strategy

        if best_match and best_score > 0.5:
            # Determine action based on strategy
            action = "buy" if "bull" in best_match.entry_conditions.get("market_condition", "") else "sell"
            return best_match, action, best_score

        return None

    def record_strategy_trade(
        self,
        strategy_name: str,
        pnl_percent: float
    ):
        """Record a trade result for a strategy."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                # Update live results
                live = strategy.live_results
                live["total_trades"] = live.get("total_trades", 0) + 1
                live["total_pnl"] = live.get("total_pnl", 0) + pnl_percent

                if pnl_percent > 0:
                    live["wins"] = live.get("wins", 0) + 1
                else:
                    live["losses"] = live.get("losses", 0) + 1

                live["win_rate"] = live["wins"] / live["total_trades"] if live["total_trades"] > 0 else 0
                live["avg_pnl"] = live["total_pnl"] / live["total_trades"]

                strategy.last_trade = datetime.now(timezone.utc).isoformat()
                strategy.live_results = live

                self._save_strategies()
                break

    def get_active_strategies(self) -> List[Dict]:
        """Get list of active strategies."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "position_size": s.position_size,
                "stop_loss": f"{s.stop_loss_percent}%",
                "take_profit": f"{s.take_profit_percent}%",
                "backtest_win_rate": s.backtest_results.get("win_rate", 0),
                "live_trades": s.live_results.get("total_trades", 0),
                "live_pnl": f"{s.live_results.get('total_pnl', 0):.2f}%"
            }
            for s in self.strategies if s.is_active
        ]

    def backtest_strategy(self, name: str, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Backtest a strategy on historical data before activation.

        Args:
            name: Strategy name to backtest
            historical_data: List of dicts with 'snapshot' and future returns

        Returns:
            Backtest results dict
        """
        strategy = None
        for s in self.strategies:
            if s.name == name:
                strategy = s
                break

        if not strategy:
            return {"error": f"Strategy '{name}' not found"}

        # Simulate trades
        trades = []
        wins = 0
        losses = 0
        total_pnl = 0.0

        for data in historical_data:
            snapshot = data.get('snapshot')
            if not snapshot:
                continue

            # Check if entry conditions match
            entry_cond = strategy.entry_conditions
            condition_match = entry_cond.get("market_condition") == snapshot.condition.value

            # Check RSI
            rsi_state = entry_cond.get("rsi_state", "neutral")
            rsi_match = (
                (rsi_state == "oversold" and snapshot.rsi < 30) or
                (rsi_state == "overbought" and snapshot.rsi > 70) or
                (rsi_state == "neutral" and 30 <= snapshot.rsi <= 70)
            )

            # Check trend
            trend_match = entry_cond.get("trend") == snapshot.trend_1h

            # Entry signal
            if condition_match and (rsi_match or trend_match):
                # Simulate trade outcome
                future_return = data.get('return_4h', 0)

                # Apply stop loss and take profit
                take_profit = strategy.take_profit_percent
                stop_loss = strategy.stop_loss_percent

                if future_return >= take_profit:
                    pnl = take_profit
                elif future_return <= -stop_loss:
                    pnl = -stop_loss
                else:
                    pnl = future_return

                trades.append({
                    'timestamp': snapshot.timestamp.isoformat() if hasattr(snapshot.timestamp, 'isoformat') else str(snapshot.timestamp),
                    'entry_price': snapshot.price,
                    'pnl_percent': pnl
                })

                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        # Calculate metrics
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Calculate max drawdown
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for trade in trades:
            cumulative += trade['pnl_percent']
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)

        results = {
            "strategy": name,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_estimate": avg_pnl / (max_drawdown + 0.1) if max_drawdown > 0 else avg_pnl * 10,
            "profitable": total_pnl > 0 and win_rate > 0.4
        }

        # Store results in strategy
        strategy.backtest_results = results
        self._save_strategies()

        logger.info(f"Backtest {name}: {total_trades} trades, {win_rate:.0%} win rate, {total_pnl:.2f}% total PnL")
        return results

    def activate_strategy(self, name: str, require_backtest: bool = True) -> Tuple[bool, str]:
        """
        Activate a strategy for live trading.

        Args:
            name: Strategy name
            require_backtest: If True, only activate if backtest is profitable

        Returns:
            (success, message)
        """
        for s in self.strategies:
            if s.name == name:
                # Check backtest results if required
                if require_backtest:
                    backtest = s.backtest_results
                    if not backtest:
                        return False, "Strategy must be backtested before activation"
                    if not backtest.get("profitable", False):
                        return False, f"Strategy not profitable in backtest (Win rate: {backtest.get('win_rate', 0):.0%}, PnL: {backtest.get('total_pnl', 0):.2f}%)"

                s.is_active = True
                self._save_strategies()
                logger.info(f"Activated strategy: {name}")
                return True, f"Strategy '{name}' activated"
        return False, f"Strategy '{name}' not found"

    def deactivate_strategy(self, name: str) -> bool:
        """Deactivate a strategy."""
        for s in self.strategies:
            if s.name == name:
                s.is_active = False
                self._save_strategies()
                logger.info(f"Deactivated strategy: {name}")
                return True
        return False


# =============================================================================
# REAL-TIME EXECUTION OPTIMIZER
# =============================================================================

class ExecutionOptimizer:
    """
    Optimizes trade execution in real-time.

    Responsibilities:
    1. Find optimal entry price (wait for dip/spike)
    2. Manage partial fills and scaling
    3. Adjust stops dynamically based on price action
    4. Implement trailing stops for winners
    5. Cut losses quickly, let winners run
    """

    def __init__(self, data_dir: Path = Path("data/ai_brain")):
        self.data_dir = data_dir

        # Active position tracking
        self.active_positions: Dict[str, Dict] = {}

        # Execution parameters learned from history
        self.optimal_entry_delay: float = 0.0  # Seconds to wait for better price
        self.avg_slippage: float = 0.1  # Percent

        # Trailing stop settings
        self.trailing_stop_activation: float = 1.0  # Activate after 1% profit
        self.trailing_stop_distance: float = 0.5   # Trail by 0.5%

    def should_enter_now(
        self,
        symbol: str,
        action: str,
        current_price: float,
        snapshot: MarketSnapshot,
        urgency: float = 0.5  # 0-1, higher = enter sooner
    ) -> Tuple[bool, str, float]:
        """
        Determine if we should enter now or wait for better price.

        Returns: (should_enter, reason, suggested_price)
        """
        # High urgency = enter now
        if urgency > 0.8:
            return True, "High urgency - entering now", current_price

        # Check if price is favorable
        if action == "buy":
            # Don't buy at resistance
            if snapshot.distance_to_resistance < 0.5:
                return False, "Too close to resistance - wait for pullback", current_price * 0.995

            # Wait if RSI is high
            if snapshot.rsi > 65:
                return False, f"RSI {snapshot.rsi:.0f} - wait for dip", current_price * 0.99

            # Good entry conditions
            if snapshot.rsi < 40 and snapshot.distance_to_support < 1.0:
                return True, "Near support with low RSI - good entry", current_price

        else:  # sell/short
            # Don't short at support
            if snapshot.distance_to_support < 0.5:
                return False, "Too close to support - wait for bounce", current_price * 1.005

            # Wait if RSI is low
            if snapshot.rsi < 35:
                return False, f"RSI {snapshot.rsi:.0f} - wait for bounce", current_price * 1.01

            # Good short conditions
            if snapshot.rsi > 60 and snapshot.distance_to_resistance < 1.0:
                return True, "Near resistance with high RSI - good entry", current_price

        # Default: enter with small delay expectation
        return True, "Conditions acceptable", current_price

    def calculate_dynamic_stop(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        current_price: float,
        snapshot: MarketSnapshot,
        initial_stop: float
    ) -> float:
        """Calculate dynamic stop loss based on current conditions."""

        if action == "buy":
            current_profit_pct = (current_price - entry_price) / entry_price * 100

            # Implement trailing stop once in profit
            if current_profit_pct > self.trailing_stop_activation:
                # Trail the stop
                trailing_stop = current_price * (1 - self.trailing_stop_distance / 100)
                return max(initial_stop, trailing_stop)

            # Tighten stop if volatility increases
            if snapshot.volatility_percentile > 80:
                return entry_price * 0.985  # Tighter stop in high vol

        else:  # short
            current_profit_pct = (entry_price - current_price) / entry_price * 100

            if current_profit_pct > self.trailing_stop_activation:
                trailing_stop = current_price * (1 + self.trailing_stop_distance / 100)
                return min(initial_stop, trailing_stop)

            if snapshot.volatility_percentile > 80:
                return entry_price * 1.015

        return initial_stop

    def should_exit_now(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        current_price: float,
        snapshot: MarketSnapshot,
        take_profit: float,
        stop_loss: float,
        holding_hours: float
    ) -> Tuple[bool, str]:
        """
        Determine if we should exit now.

        Returns: (should_exit, reason)
        """
        if action == "buy":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # Always respect stop loss
        if pnl_pct <= -abs(stop_loss):
            return True, f"STOP LOSS HIT: {pnl_pct:.2f}%"

        # Take profit hit
        if pnl_pct >= take_profit:
            return True, f"TAKE PROFIT: {pnl_pct:.2f}%"

        # Exit on momentum reversal while in profit
        if pnl_pct > 0.5:
            if action == "buy" and snapshot.trend_1h == "down" and snapshot.rsi > 70:
                return True, f"Momentum reversal (in profit): {pnl_pct:.2f}%"
            elif action != "buy" and snapshot.trend_1h == "up" and snapshot.rsi < 30:
                return True, f"Momentum reversal (in profit): {pnl_pct:.2f}%"

        # Time-based exit - cut losers faster
        if holding_hours > 4 and pnl_pct < -0.5:
            return True, f"Time stop (losing position): {pnl_pct:.2f}%"

        # Let winners run, cut losers
        if holding_hours > 24:
            if pnl_pct < 0.5:
                return True, f"Max hold time reached: {pnl_pct:.2f}%"

        return False, "Hold position"

    def get_position_status(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        current_price: float,
        entry_time: datetime
    ) -> Dict[str, Any]:
        """Get detailed status for an active position."""
        if action == "buy":
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_direction = "profit" if pnl_pct > 0 else "loss"
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            pnl_direction = "profit" if pnl_pct > 0 else "loss"

        holding_time = datetime.now(timezone.utc) - entry_time
        holding_hours = holding_time.total_seconds() / 3600

        return {
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_percent": round(pnl_pct, 2),
            "pnl_direction": pnl_direction,
            "holding_hours": round(holding_hours, 1),
            "status": "winning" if pnl_pct > 0.5 else "losing" if pnl_pct < -0.5 else "flat"
        }


# =============================================================================
# AI TRADING BRAIN - MAIN COORDINATOR
# =============================================================================

class AITradingBrain:
    """
    Main AI Trading Brain that coordinates all learning and decision systems.

    Goal: Achieve minimum 1% daily gains while minimizing losses.
    """

    def __init__(self, data_dir: Path = Path("data/ai_brain")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self.pattern_learner = HistoricalPatternLearner(data_dir / "patterns.db")
        self.trade_analyzer = TradeOutcomeAnalyzer(data_dir / "trade_analysis.db")
        self.daily_tracker = DailyTargetTracker(
            target_percent=1.0,
            max_daily_loss=2.0,
            data_dir=data_dir
        )

        # Initialize strategy and execution components
        self.strategy_generator = StrategyGenerator(
            pattern_learner=self.pattern_learner,
            data_dir=data_dir
        )
        self.execution_optimizer = ExecutionOptimizer(data_dir=data_dir)

        # Current state
        self.current_strategy: str = "Conservative"
        self.confidence_level: float = 0.5
        self.active_positions: Dict[str, Dict] = {}

        logger.info("AI Trading Brain initialized - Target: 1% daily gain")

    async def analyze_opportunity(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a trading opportunity using all available intelligence.

        Returns comprehensive recommendation.
        """
        # Get pattern-based expectation
        pattern_action, pattern_return, pattern_confidence = \
            self.pattern_learner.get_pattern_expectation(snapshot)

        # Get daily status
        daily_status = self.daily_tracker.get_status()

        # Check if we should trade at all
        signal_confidence = signal.get("confidence", 0.5)
        combined_confidence = (signal_confidence + pattern_confidence) / 2

        should_trade, reason = self.daily_tracker.should_trade(combined_confidence)

        # Build recommendation
        recommendation = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # Market analysis
            "market_condition": snapshot.condition.value,
            "pattern_expectation": {
                "action": pattern_action,
                "expected_return": f"{pattern_return:.2f}%",
                "confidence": f"{pattern_confidence:.0%}"
            },

            # Signal analysis
            "signal": {
                "action": signal.get("action", "hold"),
                "confidence": f"{signal_confidence:.0%}",
                "reason": signal.get("reason", "")
            },

            # Combined analysis
            "combined_confidence": f"{combined_confidence:.0%}",

            # Decision
            "should_trade": should_trade,
            "decision_reason": reason,

            # Daily context
            "daily_progress": daily_status,

            # Risk guidance
            "position_size_recommendation": self._recommend_position_size(
                combined_confidence, daily_status
            ),
            "stop_loss_recommendation": self._recommend_stop_loss(
                snapshot, signal.get("action", "hold")
            ),
            "take_profit_recommendation": self._recommend_take_profit(
                pattern_return, daily_status
            )
        }

        return recommendation

    def _recommend_position_size(
        self,
        confidence: float,
        daily_status: Dict
    ) -> str:
        """Recommend position size based on confidence and daily status."""
        base_size = 1.0  # Normal position

        # Adjust for confidence
        if confidence > 0.8:
            base_size = 1.5
        elif confidence > 0.6:
            base_size = 1.0
        elif confidence > 0.4:
            base_size = 0.5
        else:
            base_size = 0.25

        # Adjust for daily status
        risk_budget = float(daily_status.get("risk_budget_used", "0").replace("%", ""))
        if risk_budget > 75:
            base_size *= 0.25
        elif risk_budget > 50:
            base_size *= 0.5

        if daily_status.get("target_achieved"):
            base_size *= 0.5  # Reduce size after target achieved

        if base_size >= 1.5:
            return "LARGE (1.5x normal)"
        elif base_size >= 1.0:
            return "NORMAL"
        elif base_size >= 0.5:
            return "SMALL (0.5x normal)"
        else:
            return "MINIMAL (0.25x normal)"

    def _recommend_stop_loss(
        self,
        snapshot: MarketSnapshot,
        action: str
    ) -> str:
        """Recommend stop loss based on volatility."""
        # Base stop on ATR/volatility
        if snapshot.volatility_percentile > 70:
            stop_pct = 3.0  # Wider stop in high vol
        elif snapshot.volatility_percentile > 30:
            stop_pct = 2.0  # Normal stop
        else:
            stop_pct = 1.5  # Tighter stop in low vol

        return f"{stop_pct}% from entry"

    def _recommend_take_profit(
        self,
        expected_return: float,
        daily_status: Dict
    ) -> str:
        """Recommend take profit level."""
        remaining = daily_status.get("remaining", "1%")
        remaining_pct = float(remaining.replace("%", ""))

        if expected_return > 2:
            return f"Target: {expected_return:.1f}% (let it run with trailing stop)"
        elif remaining_pct > 0:
            return f"Target: {max(remaining_pct, 0.5):.1f}% (daily target completion)"
        else:
            return "Target: 0.5% minimum (target already achieved)"

    def record_trade_result(
        self,
        trade_id: str,
        symbol: str,
        entry_snapshot: MarketSnapshot,
        action: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        price_history: List[float],
        holding_hours: float
    ) -> Dict[str, Any]:
        """Record and analyze a completed trade."""

        # Calculate PnL
        if action == "buy":
            pnl_percent = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_percent = (entry_price - exit_price) / entry_price * 100

        # Update daily tracker
        daily_result = self.daily_tracker.record_trade(pnl_percent)

        # Analyze the trade for lessons
        lesson = self.trade_analyzer.analyze_trade(
            trade_id=trade_id,
            symbol=symbol,
            entry_snapshot=entry_snapshot,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            price_history=price_history,
            holding_hours=holding_hours
        )

        # Learn from the price movement
        if price_history:
            # Calculate what happened after entry
            next_prices = price_history[1:] if len(price_history) > 1 else [exit_price]
            if next_prices:
                next_1h = (next_prices[min(12, len(next_prices)-1)] - entry_price) / entry_price * 100 if len(next_prices) > 0 else 0
                next_4h = (next_prices[min(48, len(next_prices)-1)] - entry_price) / entry_price * 100 if len(next_prices) > 0 else 0
                next_24h = (next_prices[-1] - entry_price) / entry_price * 100

                self.pattern_learner.learn_from_movement(
                    entry_snapshot, next_1h, next_4h, next_24h
                )

        return {
            "trade_id": trade_id,
            "pnl_percent": round(pnl_percent, 2),
            "lesson": {
                "should_have_entered": lesson.should_have_entered,
                "notes": lesson.lesson_notes
            },
            "daily_status": daily_result,
            "insights": self.trade_analyzer.get_insights()
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """Get overall AI brain status."""
        daily = self.daily_tracker.get_status()
        insights = self.trade_analyzer.get_insights()

        # Get best patterns
        best_buy = self.pattern_learner.get_best_conditions_for_action("buy", 10)
        best_sell = self.pattern_learner.get_best_conditions_for_action("sell", 10)

        # Get active strategies
        active_strategies = self.strategy_generator.get_active_strategies()

        return {
            "status": "ACTIVE",
            "goal": "1% daily gain",
            "daily_progress": daily,
            "learning_stats": {
                "patterns_learned": len(self.pattern_learner.profitable_patterns) + len(self.pattern_learner.losing_patterns),
                "trades_analyzed": insights.get("total_trades_analyzed", 0),
                "average_pnl": insights.get("average_pnl", 0),
                "win_rate": insights.get("win_rate", 0)
            },
            "active_strategies": active_strategies,
            "total_strategies": len(self.strategy_generator.strategies),
            "best_buy_conditions": best_buy[:3],
            "best_sell_conditions": best_sell[:3],
            "recommendations": insights.get("recommendations", []),
            "current_strategy": self.current_strategy,
            "active_positions": len(self.active_positions)
        }

    # =========================================================================
    # STRATEGY MANAGEMENT
    # =========================================================================

    def generate_new_strategy(self) -> Optional[Dict]:
        """Generate a new strategy based on learned patterns."""
        strategy = self.strategy_generator.generate_strategy_from_patterns()
        if strategy:
            return {
                "name": strategy.name,
                "description": strategy.description,
                "entry_conditions": strategy.entry_conditions,
                "exit_conditions": strategy.exit_conditions,
                "is_active": strategy.is_active
            }
        return None

    def activate_strategy(self, name: str) -> bool:
        """Activate a strategy for live trading."""
        return self.strategy_generator.activate_strategy(name)

    def deactivate_strategy(self, name: str) -> bool:
        """Deactivate a strategy."""
        return self.strategy_generator.deactivate_strategy(name)

    def get_all_strategies(self) -> List[Dict]:
        """Get all strategies (active and inactive)."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "is_active": s.is_active,
                "position_size": s.position_size,
                "stop_loss": f"{s.stop_loss_percent}%",
                "take_profit": f"{s.take_profit_percent}%",
                "backtest_results": s.backtest_results,
                "live_results": s.live_results,
                "created_at": s.created_at
            }
            for s in self.strategy_generator.strategies
        ]

    # =========================================================================
    # EXECUTION OPTIMIZATION
    # =========================================================================

    def optimize_entry(
        self,
        symbol: str,
        action: str,
        current_price: float,
        snapshot: MarketSnapshot,
        urgency: float = 0.5
    ) -> Dict[str, Any]:
        """Get optimized entry recommendation."""
        should_enter, reason, suggested_price = self.execution_optimizer.should_enter_now(
            symbol, action, current_price, snapshot, urgency
        )
        return {
            "should_enter_now": should_enter,
            "reason": reason,
            "suggested_price": suggested_price,
            "current_price": current_price,
            "price_difference": f"{((suggested_price - current_price) / current_price * 100):.2f}%"
        }

    def track_position(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        current_price: float,
        snapshot: MarketSnapshot,
        take_profit: float,
        stop_loss: float,
        entry_time: datetime
    ) -> Dict[str, Any]:
        """Track and optimize an active position."""
        holding_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        # Get dynamic stop
        dynamic_stop = self.execution_optimizer.calculate_dynamic_stop(
            symbol, action, entry_price, current_price, snapshot, stop_loss
        )

        # Check if should exit
        should_exit, exit_reason = self.execution_optimizer.should_exit_now(
            symbol, action, entry_price, current_price, snapshot,
            take_profit, stop_loss, holding_hours
        )

        # Get position status
        status = self.execution_optimizer.get_position_status(
            symbol, action, entry_price, current_price, entry_time
        )

        return {
            **status,
            "dynamic_stop": dynamic_stop,
            "should_exit": should_exit,
            "exit_reason": exit_reason,
            "take_profit_target": take_profit,
            "original_stop_loss": stop_loss
        }

    # =========================================================================
    # LEARNING FROM MARKET DATA
    # =========================================================================

    def learn_from_price_action(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        next_1h_return: float,
        next_4h_return: float,
        next_24h_return: float
    ):
        """Learn from observed price movements."""
        self.pattern_learner.learn_from_movement(
            snapshot, next_1h_return, next_4h_return, next_24h_return
        )

    def get_pattern_prediction(
        self,
        snapshot: MarketSnapshot
    ) -> Dict[str, Any]:
        """Get prediction based on current market pattern."""
        action, expected_return, confidence = self.pattern_learner.get_pattern_expectation(snapshot)
        return {
            "recommended_action": action,
            "expected_return": f"{expected_return:.2f}%",
            "confidence": f"{confidence:.0%}",
            "pattern_key": self.pattern_learner.create_pattern_key(snapshot)
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_brain: Optional[AITradingBrain] = None

def get_ai_brain() -> AITradingBrain:
    """Get or create the global AI Trading Brain."""
    global _brain
    if _brain is None:
        _brain = AITradingBrain()
    return _brain
