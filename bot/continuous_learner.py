"""
Continuous Learning System - Self-Improving Trading AI.

Implements continuous adaptation and improvement:
1. Real-time performance monitoring
2. Automatic model retraining triggers
3. A/B testing of strategies
4. Hyperparameter optimization
5. Feature importance tracking
6. Concept drift detection and adaptation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceWindow:
    """Rolling window of performance metrics."""

    trades: List[Dict] = field(default_factory=list)
    window_size: int = 100

    def add_trade(self, trade: Dict):
        """Add trade and maintain window size."""
        self.trades.append(trade)
        if len(self.trades) > self.window_size:
            self.trades.pop(0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.get("pnl", 0) for t in self.trades])

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 10:
            return 0.0
        returns = [t.get("pnl_pct", 0) for t in self.trades]
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized

    @property
    def profit_factor(self) -> float:
        if not self.trades:
            return 1.0
        gross_profit = sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else gross_profit


@dataclass
class ModelVersion:
    """Tracks a model version and its performance."""

    version_id: str
    created_at: datetime
    model_path: str
    symbol: str
    model_type: str

    # Performance metrics
    train_accuracy: float = 0.0
    cv_accuracy: float = 0.0
    live_win_rate: float = 0.0
    live_sharpe: float = 0.0
    live_trades: int = 0

    # Metadata
    features_used: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def live_score(self) -> float:
        """Overall live performance score."""
        if self.live_trades < 10:
            return -1  # Not enough data
        return self.live_win_rate * 0.4 + self.live_sharpe * 0.3 + (self.cv_accuracy * 0.3)


@dataclass
class ABTest:
    """A/B test between two strategies or models."""

    test_id: str
    name: str
    start_time: datetime
    variant_a: str  # Model/strategy ID
    variant_b: str

    # Results
    a_trades: int = 0
    a_wins: int = 0
    a_pnl: float = 0.0

    b_trades: int = 0
    b_wins: int = 0
    b_pnl: float = 0.0

    min_trades: int = 50  # Minimum trades before concluding
    end_time: Optional[datetime] = None
    winner: Optional[str] = None

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.a_trades if self.a_trades > 0 else 0.0

    @property
    def b_win_rate(self) -> float:
        return self.b_wins / self.b_trades if self.b_trades > 0 else 0.0

    def record_result(self, variant: str, won: bool, pnl: float):
        """Record a trade result."""
        if variant == "A":
            self.a_trades += 1
            self.a_pnl += pnl
            if won:
                self.a_wins += 1
        else:
            self.b_trades += 1
            self.b_pnl += pnl
            if won:
                self.b_wins += 1

    def is_conclusive(self) -> bool:
        """Check if test has enough data."""
        return self.a_trades >= self.min_trades and self.b_trades >= self.min_trades

    def get_winner(self) -> Optional[str]:
        """Determine winner if conclusive."""
        if not self.is_conclusive():
            return None

        # Score by win rate (60%) and profit factor (40%)
        a_score = self.a_win_rate * 0.6 + (self.a_pnl / max(self.a_trades, 1)) * 0.4
        b_score = self.b_win_rate * 0.6 + (self.b_pnl / max(self.b_trades, 1)) * 0.4

        # Need significant difference (5%)
        if a_score > b_score * 1.05:
            return "A"
        elif b_score > a_score * 1.05:
            return "B"
        return "TIE"


class ConceptDriftDetector:
    """
    Detects concept drift in market data.

    When the underlying data distribution changes,
    model performance degrades and retraining is needed.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.reference_stats: Dict[str, Dict] = {}
        self.current_window: Dict[str, deque] = {}
        self.drift_scores: Dict[str, float] = {}

    def update(self, symbol: str, features: np.ndarray, prediction_correct: bool):
        """Update drift detector with new observation."""
        if symbol not in self.current_window:
            self.current_window[symbol] = deque(maxlen=self.window_size)
            self.reference_stats[symbol] = None

        self.current_window[symbol].append({"features": features, "correct": prediction_correct})

        # Check for drift if we have enough data
        if len(self.current_window[symbol]) >= self.window_size:
            self._check_drift(symbol)

    def _check_drift(self, symbol: str):
        """Check for concept drift."""
        window = list(self.current_window[symbol])

        # Calculate current statistics
        features = np.array([w["features"] for w in window])
        accuracy = np.mean([w["correct"] for w in window])

        current_stats = {
            "mean": features.mean(axis=0),
            "std": features.std(axis=0),
            "accuracy": accuracy,
        }

        # Initialize reference if needed
        if self.reference_stats[symbol] is None:
            self.reference_stats[symbol] = current_stats
            self.drift_scores[symbol] = 0.0
            return

        ref = self.reference_stats[symbol]

        # Calculate drift score
        # 1. Feature distribution drift (mean shift)
        mean_shift = np.abs(current_stats["mean"] - ref["mean"]).mean()
        std_shift = np.abs(current_stats["std"] - ref["std"]).mean()

        # 2. Performance drift
        accuracy_drop = max(0, ref["accuracy"] - current_stats["accuracy"])

        # Combined drift score (0-1)
        drift_score = mean_shift * 0.3 + std_shift * 0.2 + accuracy_drop * 0.5

        self.drift_scores[symbol] = min(1.0, drift_score * 2)  # Scale to 0-1

    def get_drift_level(self, symbol: str) -> str:
        """Get drift severity level."""
        score = self.drift_scores.get(symbol, 0)
        if score > 0.7:
            return "CRITICAL"
        elif score > 0.5:
            return "HIGH"
        elif score > 0.3:
            return "MODERATE"
        elif score > 0.1:
            return "LOW"
        return "NONE"

    def should_retrain(self, symbol: str) -> bool:
        """Check if model should be retrained."""
        return self.drift_scores.get(symbol, 0) > 0.5


class HyperparameterTuner:
    """
    Automatic hyperparameter optimization.

    Uses Bayesian-like approach to find optimal parameters.
    """

    def __init__(self):
        self.param_history: Dict[str, List[Dict]] = {}  # symbol -> history
        self.best_params: Dict[str, Dict] = {}

    def suggest_params(self, symbol: str, model_type: str = "random_forest") -> Dict[str, Any]:
        """Suggest hyperparameters based on history."""
        if symbol in self.best_params:
            # Start from best known and mutate
            base = self.best_params[symbol].copy()
            return self._mutate_params(base, model_type)

        # Default starting params
        if model_type == "random_forest":
            return {
                "n_estimators": np.random.choice([50, 100, 150, 200]),
                "max_depth": np.random.choice([5, 8, 10, 15, None]),
                "min_samples_split": np.random.choice([2, 5, 10]),
                "min_samples_leaf": np.random.choice([1, 2, 4]),
                "max_features": np.random.choice(["sqrt", "log2", 0.5, 0.8]),
            }
        elif model_type == "xgboost":
            return {
                "n_estimators": np.random.choice([50, 100, 150]),
                "max_depth": np.random.choice([3, 5, 7, 9]),
                "learning_rate": np.random.choice([0.01, 0.05, 0.1, 0.2]),
                "subsample": np.random.choice([0.6, 0.8, 1.0]),
                "colsample_bytree": np.random.choice([0.6, 0.8, 1.0]),
            }
        else:
            return {}

    def _mutate_params(self, params: Dict, model_type: str) -> Dict:
        """Mutate parameters slightly."""
        mutated = params.copy()

        if model_type == "random_forest":
            if np.random.random() < 0.3:
                mutated["n_estimators"] = max(
                    50, params["n_estimators"] + np.random.randint(-50, 51)
                )
            if np.random.random() < 0.3 and params["max_depth"]:
                mutated["max_depth"] = max(3, params["max_depth"] + np.random.randint(-2, 3))

        return mutated

    def record_result(self, symbol: str, params: Dict, score: float):
        """Record parameter set and its performance."""
        if symbol not in self.param_history:
            self.param_history[symbol] = []

        self.param_history[symbol].append(
            {"params": params, "score": score, "timestamp": datetime.now().isoformat()}
        )

        # Update best if improved
        if symbol not in self.best_params:
            self.best_params[symbol] = params
        elif score > max(h["score"] for h in self.param_history[symbol][:-1]):
            self.best_params[symbol] = params
            logger.info(f"New best params for {symbol}: {params} (score: {score:.4f})")


class ContinuousLearner:
    """
    Main continuous learning orchestrator.

    Monitors performance, triggers retraining,
    and continuously improves the trading system.
    """

    def __init__(self, data_dir: str = "data/continuous_learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.drift_detector = ConceptDriftDetector()
        self.hyperparameter_tuner = HyperparameterTuner()

        # Performance tracking per symbol
        self.performance_windows: Dict[str, PerformanceWindow] = {}

        # Model versions
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.active_versions: Dict[str, str] = {}  # symbol -> active version_id

        # A/B tests
        self.ab_tests: Dict[str, ABTest] = {}

        # Retraining management
        self.last_retrain: Dict[str, datetime] = {}
        self.retrain_cooldown = timedelta(hours=24)
        self.min_trades_for_retrain = 50

        # Callbacks
        self.retrain_callback: Optional[Callable] = None

        # Load state
        self._load_state()

        logger.info("Continuous learner initialized")

    def _load_state(self):
        """Load saved state."""
        state_path = self.data_dir / "learner_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.last_retrain = {
                    k: datetime.fromisoformat(v) for k, v in state.get("last_retrain", {}).items()
                }
                self.active_versions = state.get("active_versions", {})
                logger.info(f"Loaded learner state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save learner state."""
        state = {
            "last_retrain": {k: v.isoformat() for k, v in self.last_retrain.items()},
            "active_versions": self.active_versions,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.data_dir / "learner_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def set_retrain_callback(self, callback: Callable):
        """Set callback function for retraining."""
        self.retrain_callback = callback

    def record_prediction(
        self,
        symbol: str,
        features: np.ndarray,
        predicted_action: str,
        actual_outcome: str,
        pnl: float,
        pnl_pct: float,
        model_version: Optional[str] = None,
    ):
        """
        Record a prediction and its outcome.

        Called after every trade closes.
        """
        # Track performance
        if symbol not in self.performance_windows:
            self.performance_windows[symbol] = PerformanceWindow()

        self.performance_windows[symbol].add_trade(
            {
                "action": predicted_action,
                "outcome": actual_outcome,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "correct": pnl > 0,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Update drift detector
        self.drift_detector.update(symbol, features, pnl > 0)

        # Update model version stats if tracking
        if model_version and symbol in self.model_versions:
            for mv in self.model_versions[symbol]:
                if mv.version_id == model_version:
                    mv.live_trades += 1
                    if pnl > 0:
                        mv.live_win_rate = (
                            mv.live_win_rate * (mv.live_trades - 1) + 1
                        ) / mv.live_trades
                    else:
                        mv.live_win_rate = (
                            mv.live_win_rate * (mv.live_trades - 1)
                        ) / mv.live_trades

        # Check if retraining needed
        self._check_retrain_triggers(symbol)

        # Save state periodically
        if sum(pw.trades for pw in self.performance_windows.values()) % 10 == 0:
            self._save_state()

    def _check_retrain_triggers(self, symbol: str):
        """Check if model retraining should be triggered."""
        window = self.performance_windows.get(symbol)
        if not window or len(window.trades) < self.min_trades_for_retrain:
            return

        # Check cooldown
        last = self.last_retrain.get(symbol)
        if last and datetime.now() - last < self.retrain_cooldown:
            return

        should_retrain = False
        reason = ""

        # Trigger 1: Concept drift detected
        if self.drift_detector.should_retrain(symbol):
            should_retrain = True
            reason = (
                f"Concept drift detected (level: {self.drift_detector.get_drift_level(symbol)})"
            )

        # Trigger 2: Performance degradation
        if window.win_rate < 0.4:  # Below 40% win rate
            should_retrain = True
            reason = f"Poor performance (win rate: {window.win_rate:.1%})"

        # Trigger 3: Negative Sharpe ratio
        if window.sharpe_ratio < -0.5:
            should_retrain = True
            reason = f"Negative risk-adjusted returns (Sharpe: {window.sharpe_ratio:.2f})"

        if should_retrain:
            logger.warning(f"[{symbol}] Retraining triggered: {reason}")
            self._trigger_retrain(symbol, reason)

    def _trigger_retrain(self, symbol: str, reason: str):
        """Trigger model retraining."""
        self.last_retrain[symbol] = datetime.now()

        if self.retrain_callback:
            # Get suggested hyperparameters
            params = self.hyperparameter_tuner.suggest_params(symbol)

            # Call retraining function
            try:
                asyncio.create_task(self._async_retrain(symbol, reason, params))
            except Exception as e:
                logger.error(f"Retrain trigger failed: {e}")

    async def _async_retrain(self, symbol: str, reason: str, params: Dict):
        """Async wrapper for retraining."""
        if self.retrain_callback:
            result = await self.retrain_callback(symbol, params)

            if result:
                # Record the params and score
                score = result.get("accuracy", 0) * 0.5 + result.get("cv_accuracy", 0) * 0.5
                self.hyperparameter_tuner.record_result(symbol, params, score)

                # Create new model version
                self._register_model_version(symbol, result)

    def _register_model_version(self, symbol: str, train_result: Dict):
        """Register a new model version."""
        if symbol not in self.model_versions:
            self.model_versions[symbol] = []

        version = ModelVersion(
            version_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            model_path=train_result.get("model_path", ""),
            symbol=symbol,
            model_type=train_result.get("model_type", "random_forest"),
            train_accuracy=train_result.get("accuracy", 0),
            cv_accuracy=train_result.get("cv_accuracy", 0),
            features_used=train_result.get("features", []),
            hyperparameters=train_result.get("params", {}),
        )

        self.model_versions[symbol].append(version)
        self.active_versions[symbol] = version.version_id

        logger.info(f"Registered new model version: {version.version_id}")

    def start_ab_test(
        self, name: str, symbol: str, variant_a: str, variant_b: str, min_trades: int = 50
    ) -> str:
        """Start an A/B test between two model/strategy variants."""
        test_id = f"ab_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.ab_tests[test_id] = ABTest(
            test_id=test_id,
            name=name,
            start_time=datetime.now(),
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=min_trades,
        )

        logger.info(f"Started A/B test: {name} ({variant_a} vs {variant_b})")
        return test_id

    def record_ab_result(self, test_id: str, variant: str, won: bool, pnl: float):
        """Record an A/B test result."""
        if test_id not in self.ab_tests:
            return

        test = self.ab_tests[test_id]
        test.record_result(variant, won, pnl)

        # Check if conclusive
        if test.is_conclusive() and test.winner is None:
            winner = test.get_winner()
            test.winner = winner
            test.end_time = datetime.now()
            logger.info(f"A/B test {test.name} concluded. Winner: {winner}")

    def get_performance_summary(self, symbol: str) -> Dict[str, Any]:
        """Get performance summary for a symbol."""
        window = self.performance_windows.get(symbol)
        if not window:
            return {"error": "No data for symbol"}

        drift_level = self.drift_detector.get_drift_level(symbol)
        drift_score = self.drift_detector.drift_scores.get(symbol, 0)

        return {
            "symbol": symbol,
            "trades": len(window.trades),
            "win_rate": window.win_rate,
            "avg_pnl": window.avg_pnl,
            "sharpe_ratio": window.sharpe_ratio,
            "profit_factor": window.profit_factor,
            "drift_level": drift_level,
            "drift_score": drift_score,
            "should_retrain": drift_score > 0.5 or window.win_rate < 0.4,
            "last_retrain": self.last_retrain.get(symbol, "Never"),
            "active_version": self.active_versions.get(symbol, "default"),
        }

    def get_all_summaries(self) -> Dict[str, Any]:
        """Get performance summaries for all symbols."""
        summaries = {}
        for symbol in self.performance_windows:
            summaries[symbol] = self.get_performance_summary(symbol)
        return summaries


# Global instance
_continuous_learner: Optional[ContinuousLearner] = None


def get_continuous_learner() -> ContinuousLearner:
    """Get or create the ContinuousLearner instance."""
    global _continuous_learner
    if _continuous_learner is None:
        _continuous_learner = ContinuousLearner()
    return _continuous_learner
