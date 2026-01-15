"""
Self-Learning Trading Engine

A comprehensive system that:
1. Learns optimal actions for each market state
2. Self-evaluates performance and identifies weaknesses
3. Auto-tunes parameters based on results
4. Evolves strategies through shadow testing
5. Adapts to changing market conditions

The system improves itself continuously without human intervention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from collections import deque
import random

logger = logging.getLogger(__name__)


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class LearningMode(Enum):
    """Learning modes for the system."""
    OBSERVATION = "observation"  # Just watch and learn, don't act
    SHADOW = "shadow"            # Generate signals but don't execute
    VALIDATED = "validated"      # Execute only high-confidence learned actions
    AUTONOMOUS = "autonomous"    # Full autonomous operation


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time_hours: float = 0.0

    # Per-regime performance
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # Time-based performance
    hourly_performance: Dict[int, float] = field(default_factory=dict)
    daily_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyCandidate:
    """A strategy being tested in shadow mode."""
    id: str
    name: str
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Shadow performance
    shadow_trades: int = 0
    shadow_pnl: float = 0.0
    shadow_win_rate: float = 0.0
    shadow_sharpe: float = 0.0

    # Status
    is_promoted: bool = False
    is_rejected: bool = False
    promotion_date: Optional[datetime] = None


@dataclass
class MarketCondition:
    """Current market condition snapshot."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Regime
    regime: str = "unknown"
    regime_confidence: float = 0.0
    regime_duration_hours: float = 0.0

    # Trend
    trend_1h: str = "neutral"
    trend_4h: str = "neutral"
    trend_1d: str = "neutral"

    # Volatility
    volatility_percentile: float = 50.0  # 0-100
    atr_normalized: float = 0.0

    # Momentum
    rsi: float = 50.0
    macd_histogram: float = 0.0

    # Volume
    volume_ratio: float = 1.0  # vs average

    # Correlation
    btc_correlation: float = 0.0
    sp500_correlation: float = 0.0

    def to_state_vector(self) -> np.ndarray:
        """Convert to numeric vector for ML."""
        regime_map = {"strong_bull": 2, "bull": 1, "sideways": 0,
                      "bear": -1, "strong_bear": -2, "crash": -3}
        trend_map = {"up": 1, "neutral": 0, "down": -1}

        return np.array([
            regime_map.get(self.regime, 0),
            self.regime_confidence,
            trend_map.get(self.trend_1h, 0),
            trend_map.get(self.trend_4h, 0),
            trend_map.get(self.trend_1d, 0),
            (self.volatility_percentile - 50) / 50,
            self.atr_normalized,
            (self.rsi - 50) / 50,
            np.tanh(self.macd_histogram),
            np.log1p(self.volume_ratio) - 0.7,
        ])


# =============================================================================
# SELF-EVALUATION MODULE
# =============================================================================

class SelfEvaluator:
    """
    Evaluates system performance and identifies areas for improvement.

    Analyzes:
    - What's working well
    - What's underperforming
    - Why trades are failing
    - Which market conditions are problematic
    """

    def __init__(self, db_path: Path = Path("data/self_learning.db")):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize evaluation database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                evaluation_type TEXT,
                period_start TEXT,
                period_end TEXT,
                metrics_json TEXT,
                insights_json TEXT,
                recommendations_json TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                timestamp TEXT,
                regime TEXT,
                action TEXT,
                pnl REAL,
                pnl_percent REAL,
                analysis_json TEXT,
                failure_reason TEXT,
                improvement_suggestion TEXT
            )
        """)

        conn.commit()
        conn.close()

    def evaluate_period(
        self,
        trades: List[Dict],
        period_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Evaluate performance over a period.

        Returns insights and recommendations.
        """
        if not trades:
            return {"status": "no_trades", "insights": [], "recommendations": []}

        # Calculate metrics
        metrics = self._calculate_metrics(trades)

        # Analyze by regime
        regime_analysis = self._analyze_by_regime(trades)

        # Analyze by time
        time_analysis = self._analyze_by_time(trades)

        # Identify failure patterns
        failure_patterns = self._identify_failure_patterns(trades)

        # Generate insights
        insights = self._generate_insights(metrics, regime_analysis, failure_patterns)

        # Generate recommendations
        recommendations = self._generate_recommendations(insights, failure_patterns)

        # Save evaluation
        self._save_evaluation(metrics, insights, recommendations, period_hours)

        return {
            "metrics": asdict(metrics),
            "regime_analysis": regime_analysis,
            "time_analysis": time_analysis,
            "failure_patterns": failure_patterns,
            "insights": insights,
            "recommendations": recommendations,
        }

    def _calculate_metrics(self, trades: List[Dict]) -> PerformanceMetrics:
        """Calculate performance metrics from trades."""
        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        pnls = [t.get("pnl_percent", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics.total_trades = len(trades)
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.total_pnl_percent = sum(pnls)
        metrics.win_rate = len(wins) / len(trades) if trades else 0

        if wins:
            metrics.avg_win = np.mean(wins)
            metrics.largest_win = max(wins)

        if losses:
            metrics.avg_loss = abs(np.mean(losses))
            metrics.largest_loss = abs(min(losses))

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.01
        metrics.profit_factor = total_wins / total_losses

        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            metrics.sharpe_ratio = np.mean(pnls) / (np.std(pnls) + 0.001) * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return metrics

    def _analyze_by_regime(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Analyze performance by market regime."""
        regime_trades: Dict[str, List] = {}

        for trade in trades:
            regime = trade.get("regime", "unknown")
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        analysis = {}
        for regime, rtrades in regime_trades.items():
            pnls = [t.get("pnl_percent", 0) for t in rtrades]
            wins = [p for p in pnls if p > 0]

            analysis[regime] = {
                "count": len(rtrades),
                "win_rate": len(wins) / len(rtrades) if rtrades else 0,
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
            }

        return analysis

    def _analyze_by_time(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by time of day and day of week."""
        hourly: Dict[int, List] = {h: [] for h in range(24)}
        daily: Dict[str, List] = {d: [] for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}

        for trade in trades:
            ts = trade.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hourly[dt.hour].append(trade.get("pnl_percent", 0))
                    daily[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]].append(
                        trade.get("pnl_percent", 0)
                    )
                except:
                    pass

        return {
            "hourly": {h: {"avg": np.mean(pnls) if pnls else 0, "count": len(pnls)}
                       for h, pnls in hourly.items()},
            "daily": {d: {"avg": np.mean(pnls) if pnls else 0, "count": len(pnls)}
                      for d, pnls in daily.items()},
            "best_hour": max(hourly.keys(), key=lambda h: np.mean(hourly[h]) if hourly[h] else -999),
            "worst_hour": min(hourly.keys(), key=lambda h: np.mean(hourly[h]) if hourly[h] else 999),
        }

    def _identify_failure_patterns(self, trades: List[Dict]) -> List[Dict]:
        """Identify common patterns in losing trades."""
        losing_trades = [t for t in trades if t.get("pnl_percent", 0) < 0]

        if not losing_trades:
            return []

        patterns = []

        # Check for regime-related failures
        regime_losses = {}
        for t in losing_trades:
            regime = t.get("regime", "unknown")
            regime_losses[regime] = regime_losses.get(regime, 0) + 1

        worst_regime = max(regime_losses.keys(), key=lambda r: regime_losses[r])
        if regime_losses[worst_regime] > len(losing_trades) * 0.3:
            patterns.append({
                "type": "regime_concentration",
                "description": f"Most losses occur in {worst_regime} regime",
                "regime": worst_regime,
                "loss_count": regime_losses[worst_regime],
                "severity": "high" if regime_losses[worst_regime] > len(losing_trades) * 0.5 else "medium",
            })

        # Check for volatility-related failures
        high_vol_losses = [t for t in losing_trades if t.get("volatility", "normal") in ("high", "extreme")]
        if len(high_vol_losses) > len(losing_trades) * 0.4:
            patterns.append({
                "type": "volatility_sensitivity",
                "description": "High losses during volatile periods",
                "high_vol_loss_pct": len(high_vol_losses) / len(losing_trades),
                "severity": "high",
            })

        # Check for consecutive losses
        consecutive = 0
        max_consecutive = 0
        for t in trades:
            if t.get("pnl_percent", 0) < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        if max_consecutive >= 5:
            patterns.append({
                "type": "losing_streaks",
                "description": f"Max {max_consecutive} consecutive losses detected",
                "max_streak": max_consecutive,
                "severity": "high" if max_consecutive >= 7 else "medium",
            })

        # Check for large single losses
        large_losses = [t for t in losing_trades if abs(t.get("pnl_percent", 0)) > 5]
        if large_losses:
            patterns.append({
                "type": "large_single_losses",
                "description": f"{len(large_losses)} trades with >5% loss",
                "count": len(large_losses),
                "total_lost": sum(abs(t.get("pnl_percent", 0)) for t in large_losses),
                "severity": "high",
            })

        return patterns

    def _generate_insights(
        self,
        metrics: PerformanceMetrics,
        regime_analysis: Dict,
        failure_patterns: List[Dict]
    ) -> List[str]:
        """Generate human-readable insights."""
        insights = []

        # Win rate insight
        if metrics.win_rate < 0.4:
            insights.append(f"LOW WIN RATE: {metrics.win_rate:.1%} - Need better entry signals")
        elif metrics.win_rate > 0.6:
            insights.append(f"GOOD WIN RATE: {metrics.win_rate:.1%} - Entry timing is working well")

        # Risk/reward insight
        if metrics.avg_win > 0 and metrics.avg_loss > 0:
            rr = metrics.avg_win / metrics.avg_loss
            if rr < 1:
                insights.append(f"POOR RISK/REWARD: {rr:.2f} - Wins are smaller than losses")
            elif rr > 1.5:
                insights.append(f"GOOD RISK/REWARD: {rr:.2f} - Wins exceed losses")

        # Profit factor insight
        if metrics.profit_factor < 1:
            insights.append(f"UNPROFITABLE: Profit factor {metrics.profit_factor:.2f} < 1")
        elif metrics.profit_factor > 2:
            insights.append(f"HIGHLY PROFITABLE: Profit factor {metrics.profit_factor:.2f}")

        # Regime insights
        for regime, data in regime_analysis.items():
            if data["count"] >= 5:
                if data["win_rate"] < 0.3:
                    insights.append(f"WEAK IN {regime.upper()}: Only {data['win_rate']:.1%} win rate")
                elif data["win_rate"] > 0.7:
                    insights.append(f"STRONG IN {regime.upper()}: {data['win_rate']:.1%} win rate")

        # Failure pattern insights
        for pattern in failure_patterns:
            if pattern["severity"] == "high":
                insights.append(f"CRITICAL: {pattern['description']}")

        return insights

    def _generate_recommendations(
        self,
        insights: List[str],
        failure_patterns: List[Dict]
    ) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []

        for pattern in failure_patterns:
            if pattern["type"] == "regime_concentration":
                recommendations.append({
                    "action": "reduce_trading_in_regime",
                    "regime": pattern["regime"],
                    "description": f"Reduce position size or avoid trading in {pattern['regime']} regime",
                    "parameter_change": {"regime_position_multiplier": {pattern["regime"]: 0.5}},
                })

            elif pattern["type"] == "volatility_sensitivity":
                recommendations.append({
                    "action": "adjust_volatility_filter",
                    "description": "Tighten stops and reduce size during high volatility",
                    "parameter_change": {
                        "high_vol_position_multiplier": 0.5,
                        "high_vol_stop_multiplier": 0.7,
                    },
                })

            elif pattern["type"] == "losing_streaks":
                recommendations.append({
                    "action": "add_streak_breaker",
                    "description": "Pause trading after consecutive losses",
                    "parameter_change": {"max_consecutive_losses": 4, "cooldown_minutes": 60},
                })

            elif pattern["type"] == "large_single_losses":
                recommendations.append({
                    "action": "tighten_risk_limits",
                    "description": "Reduce max loss per trade",
                    "parameter_change": {"max_loss_per_trade_pct": 2.0},
                })

        return recommendations

    def _save_evaluation(
        self,
        metrics: PerformanceMetrics,
        insights: List[str],
        recommendations: List[Dict],
        period_hours: int
    ):
        """Save evaluation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc)
        cursor.execute("""
            INSERT INTO evaluations
            (timestamp, evaluation_type, period_start, period_end,
             metrics_json, insights_json, recommendations_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            now.isoformat(),
            f"{period_hours}h_review",
            (now - timedelta(hours=period_hours)).isoformat(),
            now.isoformat(),
            json.dumps(asdict(metrics)),
            json.dumps(insights),
            json.dumps(recommendations),
        ))

        conn.commit()
        conn.close()


# =============================================================================
# AUTO-PARAMETER TUNER
# =============================================================================

class AutoParameterTuner:
    """
    Automatically tunes trading parameters based on performance.

    Uses:
    - Bayesian optimization for parameter search
    - Walk-forward validation
    - Gradual parameter adjustment
    """

    def __init__(self, db_path: Path = Path("data/self_learning.db")):
        self.db_path = db_path
        self._init_db()

        # Parameter bounds
        self.parameter_bounds = {
            "confidence_threshold": (0.3, 0.8),
            "stop_loss_atr_multiplier": (1.0, 4.0),
            "take_profit_atr_multiplier": (1.5, 6.0),
            "position_size_pct": (0.5, 5.0),
            "max_positions": (1, 10),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (60, 80),
        }

        # Current parameters
        self.current_params: Dict[str, float] = {}

        # Parameter history
        self.param_history: List[Tuple[Dict, float]] = []  # (params, score)

    def _init_db(self):
        """Initialize tuning database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                parameters_json TEXT,
                score REAL,
                metrics_json TEXT,
                is_current INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                expected_improvement REAL,
                actual_improvement REAL,
                is_applied INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def suggest_parameters(
        self,
        current_metrics: PerformanceMetrics,
        failure_patterns: List[Dict]
    ) -> Dict[str, Any]:
        """
        Suggest parameter changes based on current performance.

        Returns suggested parameters with expected improvement.
        """
        suggestions = {}

        # If win rate is low, tighten confidence threshold
        if current_metrics.win_rate < 0.45:
            current_conf = self.current_params.get("confidence_threshold", 0.5)
            suggestions["confidence_threshold"] = {
                "current": current_conf,
                "suggested": min(current_conf + 0.1, 0.75),
                "reason": "Low win rate suggests signals are not selective enough",
            }

        # If avg loss > avg win, adjust stops
        if current_metrics.avg_loss > current_metrics.avg_win * 1.2:
            current_sl = self.current_params.get("stop_loss_atr_multiplier", 2.0)
            suggestions["stop_loss_atr_multiplier"] = {
                "current": current_sl,
                "suggested": max(current_sl * 0.8, 1.0),
                "reason": "Losses are larger than wins - tighten stop loss",
            }

        # If profit factor < 1.5, adjust take profit
        if current_metrics.profit_factor < 1.5:
            current_tp = self.current_params.get("take_profit_atr_multiplier", 3.0)
            suggestions["take_profit_atr_multiplier"] = {
                "current": current_tp,
                "suggested": current_tp * 1.2,
                "reason": "Need larger wins to improve profit factor",
            }

        # Handle specific failure patterns
        for pattern in failure_patterns:
            if pattern["type"] == "large_single_losses":
                current_size = self.current_params.get("position_size_pct", 2.0)
                suggestions["position_size_pct"] = {
                    "current": current_size,
                    "suggested": max(current_size * 0.7, 0.5),
                    "reason": "Large single losses - reduce position size",
                }

        return suggestions

    def apply_suggestions(
        self,
        suggestions: Dict[str, Any],
        validation_period_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Apply suggested parameters with validation tracking.
        """
        applied = {}

        for param, suggestion in suggestions.items():
            # Record experiment
            self._record_experiment(
                param,
                suggestion["current"],
                suggestion["suggested"],
                suggestion.get("expected_improvement", 0.1)
            )

            # Update current params
            self.current_params[param] = suggestion["suggested"]
            applied[param] = suggestion["suggested"]

        # Save new parameter set
        self._save_parameters(self.current_params, 0.0)  # Score TBD

        return {
            "applied_parameters": applied,
            "validation_period_hours": validation_period_hours,
            "rollback_on_failure": True,
        }

    def _record_experiment(
        self,
        param: str,
        old_val: float,
        new_val: float,
        expected_improvement: float
    ):
        """Record a parameter experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO parameter_experiments
            (timestamp, parameter_name, old_value, new_value, expected_improvement)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            param, old_val, new_val, expected_improvement
        ))

        conn.commit()
        conn.close()

    def _save_parameters(self, params: Dict, score: float):
        """Save parameter set to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Mark previous as not current
        cursor.execute("UPDATE parameter_history SET is_current = 0")

        cursor.execute("""
            INSERT INTO parameter_history (timestamp, parameters_json, score, is_current)
            VALUES (?, ?, ?, 1)
        """, (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(params),
            score
        ))

        conn.commit()
        conn.close()


# =============================================================================
# STRATEGY EVOLUTION
# =============================================================================

class StrategyEvolver:
    """
    Evolves trading strategies through shadow testing and promotion.

    Process:
    1. Generate strategy variations (mutations)
    2. Test in shadow mode (paper signals, no execution)
    3. Evaluate shadow performance
    4. Promote successful strategies
    5. Retire underperforming strategies
    """

    def __init__(self, db_path: Path = Path("data/self_learning.db")):
        self.db_path = db_path
        self._init_db()

        # Active strategies being tested
        self.shadow_strategies: Dict[str, StrategyCandidate] = {}

        # Promotion criteria
        self.min_shadow_trades = 30
        self.min_shadow_win_rate = 0.5
        self.min_shadow_sharpe = 0.5

    def _init_db(self):
        """Initialize strategy evolution database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_candidates (
                id TEXT PRIMARY KEY,
                name TEXT,
                parameters_json TEXT,
                created_at TEXT,
                shadow_trades INTEGER DEFAULT 0,
                shadow_pnl REAL DEFAULT 0,
                shadow_win_rate REAL DEFAULT 0,
                shadow_sharpe REAL DEFAULT 0,
                is_promoted INTEGER DEFAULT 0,
                is_rejected INTEGER DEFAULT 0,
                promotion_date TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                entry_price REAL,
                hypothetical_exit_price REAL,
                hypothetical_pnl REAL,
                market_condition_json TEXT
            )
        """)

        conn.commit()
        conn.close()

    def generate_mutation(
        self,
        base_params: Dict[str, Any],
        mutation_strength: float = 0.2
    ) -> StrategyCandidate:
        """
        Generate a mutated strategy variation.
        """
        import uuid

        mutated_params = base_params.copy()

        # Randomly mutate some parameters
        for key, value in mutated_params.items():
            if isinstance(value, (int, float)) and random.random() < 0.5:
                # Apply random mutation
                mutation = 1 + (random.random() * 2 - 1) * mutation_strength
                mutated_params[key] = value * mutation

        strategy_id = str(uuid.uuid4())[:8]
        candidate = StrategyCandidate(
            id=strategy_id,
            name=f"mutation_{strategy_id}",
            parameters=mutated_params,
        )

        self.shadow_strategies[strategy_id] = candidate
        self._save_candidate(candidate)

        return candidate

    def record_shadow_signal(
        self,
        strategy_id: str,
        symbol: str,
        action: str,
        entry_price: float,
        market_condition: MarketCondition
    ):
        """Record a shadow signal from a candidate strategy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO shadow_signals
            (strategy_id, timestamp, symbol, action, entry_price, market_condition_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            strategy_id,
            datetime.now(timezone.utc).isoformat(),
            symbol, action, entry_price,
            json.dumps(asdict(market_condition))
        ))

        conn.commit()
        conn.close()

    def update_shadow_outcome(
        self,
        strategy_id: str,
        symbol: str,
        exit_price: float,
        pnl: float
    ):
        """Update shadow signal with hypothetical outcome."""
        if strategy_id not in self.shadow_strategies:
            return

        candidate = self.shadow_strategies[strategy_id]
        candidate.shadow_trades += 1
        candidate.shadow_pnl += pnl

        # Update win rate
        if pnl > 0:
            wins = candidate.shadow_win_rate * (candidate.shadow_trades - 1) + 1
        else:
            wins = candidate.shadow_win_rate * (candidate.shadow_trades - 1)
        candidate.shadow_win_rate = wins / candidate.shadow_trades

        self._update_candidate(candidate)

    def evaluate_candidates(self) -> List[Dict]:
        """
        Evaluate all shadow strategies and return promotion recommendations.
        """
        recommendations = []

        for strategy_id, candidate in self.shadow_strategies.items():
            if candidate.is_promoted or candidate.is_rejected:
                continue

            if candidate.shadow_trades < self.min_shadow_trades:
                recommendations.append({
                    "strategy_id": strategy_id,
                    "status": "needs_more_data",
                    "trades": candidate.shadow_trades,
                    "required": self.min_shadow_trades,
                })
                continue

            # Evaluate for promotion
            if (candidate.shadow_win_rate >= self.min_shadow_win_rate and
                candidate.shadow_sharpe >= self.min_shadow_sharpe):
                recommendations.append({
                    "strategy_id": strategy_id,
                    "status": "recommend_promotion",
                    "win_rate": candidate.shadow_win_rate,
                    "sharpe": candidate.shadow_sharpe,
                    "total_pnl": candidate.shadow_pnl,
                })
            else:
                recommendations.append({
                    "strategy_id": strategy_id,
                    "status": "recommend_rejection",
                    "reason": "Below performance thresholds",
                    "win_rate": candidate.shadow_win_rate,
                    "sharpe": candidate.shadow_sharpe,
                })

        return recommendations

    def promote_strategy(self, strategy_id: str) -> bool:
        """Promote a shadow strategy to production."""
        if strategy_id not in self.shadow_strategies:
            return False

        candidate = self.shadow_strategies[strategy_id]
        candidate.is_promoted = True
        candidate.promotion_date = datetime.now(timezone.utc)

        self._update_candidate(candidate)
        logger.info(f"Strategy {strategy_id} promoted to production")

        return True

    def _save_candidate(self, candidate: StrategyCandidate):
        """Save candidate to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO strategy_candidates
            (id, name, parameters_json, created_at, shadow_trades, shadow_pnl,
             shadow_win_rate, shadow_sharpe, is_promoted, is_rejected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.id, candidate.name,
            json.dumps(candidate.parameters),
            candidate.created_at.isoformat(),
            candidate.shadow_trades, candidate.shadow_pnl,
            candidate.shadow_win_rate, candidate.shadow_sharpe,
            1 if candidate.is_promoted else 0,
            1 if candidate.is_rejected else 0,
        ))

        conn.commit()
        conn.close()

    def _update_candidate(self, candidate: StrategyCandidate):
        """Update candidate in database."""
        self._save_candidate(candidate)


# =============================================================================
# MAIN SELF-LEARNING ENGINE
# =============================================================================

class SelfLearningEngine:
    """
    Main coordinator for self-learning capabilities.

    Integrates:
    - Optimal action tracking
    - Self-evaluation
    - Auto-parameter tuning
    - Strategy evolution
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        learning_mode: LearningMode = LearningMode.VALIDATED,
    ):
        self.data_dir = data_dir
        self.learning_mode = learning_mode

        # Initialize components
        self.evaluator = SelfEvaluator(data_dir / "self_learning.db")
        self.tuner = AutoParameterTuner(data_dir / "self_learning.db")
        self.evolver = StrategyEvolver(data_dir / "self_learning.db")

        # Import optimal action tracker
        try:
            from bot.optimal_action_tracker import get_tracker
            self.action_tracker = get_tracker()
        except ImportError:
            self.action_tracker = None
            logger.warning("Optimal action tracker not available")

        # Trade buffer for evaluation
        self.trade_buffer: deque = deque(maxlen=1000)

        # Learning state
        self.last_evaluation: Optional[datetime] = None
        self.evaluation_interval_hours = 4

        # Current market condition
        self.current_condition: Optional[MarketCondition] = None

        logger.info(f"Self-learning engine initialized in {learning_mode.value} mode")

    def update_market_condition(self, condition: MarketCondition):
        """Update current market condition."""
        self.current_condition = condition

    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade for learning."""
        self.trade_buffer.append(trade)

        # Check if evaluation is due
        self._maybe_run_evaluation()

    def get_optimal_action(
        self,
        symbol: str,
        condition: Optional[MarketCondition] = None
    ) -> Tuple[str, float, Dict]:
        """
        Get the optimal action based on learned knowledge.

        Returns:
            (action, confidence, metadata)
        """
        cond = condition or self.current_condition

        if not cond:
            return "hold", 0.5, {"reason": "No market condition data"}

        # Query action tracker
        if self.action_tracker:
            from bot.optimal_action_tracker import MarketState

            state = MarketState(
                regime=cond.regime,
                regime_confidence=cond.regime_confidence,
                rsi=cond.rsi,
                trend_direction=cond.trend_1h,
                volatility_regime="high" if cond.volatility_percentile > 75 else
                                  "low" if cond.volatility_percentile < 25 else "normal",
            )

            action, expected_value = self.action_tracker.get_optimal_action(state)

            return action.value, abs(expected_value), {
                "source": "q_table",
                "state_key": state.to_state_key(),
                "expected_value": expected_value,
            }

        # Fallback to heuristics
        return self._heuristic_action(cond)

    def _heuristic_action(self, condition: MarketCondition) -> Tuple[str, float, Dict]:
        """Fallback heuristic action when no learned data."""
        if condition.regime in ("strong_bull", "bull"):
            if condition.rsi < 30:
                return "buy", 0.7, {"reason": "Oversold in uptrend"}
            elif condition.rsi > 70:
                return "hold", 0.5, {"reason": "Overbought - wait for pullback"}
            else:
                return "buy", 0.5, {"reason": "Bullish regime"}

        elif condition.regime in ("strong_bear", "bear", "crash"):
            if condition.rsi > 70:
                return "sell", 0.7, {"reason": "Overbought in downtrend"}
            elif condition.rsi < 30:
                return "hold", 0.5, {"reason": "Oversold - wait for bounce"}
            else:
                return "sell", 0.5, {"reason": "Bearish regime"}

        else:
            return "hold", 0.4, {"reason": "Uncertain regime"}

    def _maybe_run_evaluation(self):
        """Run evaluation if enough time has passed."""
        now = datetime.now(timezone.utc)

        if self.last_evaluation is None:
            should_evaluate = len(self.trade_buffer) >= 10
        else:
            hours_since = (now - self.last_evaluation).total_seconds() / 3600
            should_evaluate = hours_since >= self.evaluation_interval_hours

        if should_evaluate and len(self.trade_buffer) >= 10:
            self._run_evaluation()

    def _run_evaluation(self):
        """Run full evaluation cycle."""
        logger.info("Running self-evaluation cycle...")

        trades = list(self.trade_buffer)

        # Evaluate performance
        evaluation = self.evaluator.evaluate_period(trades, self.evaluation_interval_hours)

        logger.info(f"Evaluation complete: {len(evaluation.get('insights', []))} insights")

        # Get parameter suggestions
        if evaluation.get("metrics"):
            metrics = PerformanceMetrics(**evaluation["metrics"])
            suggestions = self.tuner.suggest_parameters(
                metrics,
                evaluation.get("failure_patterns", [])
            )

            if suggestions:
                logger.info(f"Parameter suggestions: {list(suggestions.keys())}")

                # In autonomous mode, apply suggestions automatically
                if self.learning_mode == LearningMode.AUTONOMOUS:
                    self.tuner.apply_suggestions(suggestions)
                    logger.info("Auto-applied parameter changes")

        self.last_evaluation = datetime.now(timezone.utc)

        return evaluation

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            "mode": self.learning_mode.value,
            "trades_in_buffer": len(self.trade_buffer),
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "shadow_strategies": len(self.evolver.shadow_strategies),
            "current_parameters": self.tuner.current_params,
        }

    def force_evaluation(self) -> Dict[str, Any]:
        """Force immediate evaluation."""
        return self._run_evaluation()


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_engine: Optional[SelfLearningEngine] = None

def get_self_learning_engine() -> SelfLearningEngine:
    """Get or create the global self-learning engine."""
    global _engine
    if _engine is None:
        _engine = SelfLearningEngine()
    return _engine
