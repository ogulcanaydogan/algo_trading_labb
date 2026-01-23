"""
ML Model Performance Tracker

Tracks and analyzes ML model predictions vs actual outcomes to identify
which models perform best under different market conditions.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models being tracked."""

    LSTM = "lstm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    RL_PPO = "rl_ppo"
    RL_A2C = "rl_a2c"
    PATTERN_LEARNER = "pattern_learner"


@dataclass
class ModelPrediction:
    """Record of a model prediction."""

    timestamp: datetime
    model_type: str
    symbol: str
    prediction: str  # "buy", "sell", "hold"
    confidence: float
    predicted_return: Optional[float]  # Expected return %
    market_condition: str  # bull, bear, sideways, volatile
    volatility: float

    # Filled in later when outcome is known
    actual_return: Optional[float] = None
    was_correct: Optional[bool] = None
    outcome_recorded_at: Optional[datetime] = None


class MLPerformanceTracker:
    """
    Tracks ML model performance across different market conditions.

    Stores predictions and outcomes in SQLite for historical analysis.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "ml_performance.db")

        self.db_path = db_path
        self._init_database()

        # In-memory cache for recent predictions awaiting outcomes
        self._pending_predictions: Dict[str, ModelPrediction] = {}

    def _init_database(self):
        """Initialize SQLite database for performance tracking."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_return REAL,
                market_condition TEXT NOT NULL,
                volatility REAL NOT NULL,
                actual_return REAL,
                was_correct INTEGER,
                outcome_recorded_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_return REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                UNIQUE(model_type, market_condition, period_start)
            )
        """)

        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_model_condition
            ON predictions(model_type, market_condition)
        """)

        conn.commit()
        conn.close()

    def record_prediction(
        self,
        model_type: str,
        symbol: str,
        prediction: str,
        confidence: float,
        market_condition: str,
        volatility: float,
        predicted_return: Optional[float] = None,
    ) -> str:
        """
        Record a new model prediction.

        Returns prediction_id to link with outcome later.
        """
        timestamp = datetime.now()

        pred = ModelPrediction(
            timestamp=timestamp,
            model_type=model_type,
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            predicted_return=predicted_return,
            market_condition=market_condition,
            volatility=volatility,
        )

        # Generate unique ID
        pred_id = f"{model_type}_{symbol}_{timestamp.isoformat()}"
        self._pending_predictions[pred_id] = pred

        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (timestamp, model_type, symbol, prediction, confidence,
             predicted_return, market_condition, volatility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                timestamp.isoformat(),
                model_type,
                symbol,
                prediction,
                confidence,
                predicted_return,
                market_condition,
                volatility,
            ),
        )

        pred_id_db = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(
            f"Recorded {model_type} prediction: {prediction} ({confidence:.1%}) for {symbol}"
        )

        return str(pred_id_db)

    def record_outcome(
        self, prediction_id: str, actual_return: float, was_correct: Optional[bool] = None
    ):
        """Record the actual outcome of a prediction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Auto-determine correctness if not provided
        if was_correct is None:
            cursor.execute("SELECT prediction FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            if row:
                prediction = row[0]
                if prediction == "buy":
                    was_correct = actual_return > 0
                elif prediction == "sell":
                    was_correct = actual_return < 0
                else:  # hold
                    was_correct = abs(actual_return) < 0.5  # Correct if sideways

        cursor.execute(
            """
            UPDATE predictions
            SET actual_return = ?, was_correct = ?, outcome_recorded_at = ?
            WHERE id = ?
        """,
            (actual_return, 1 if was_correct else 0, datetime.now().isoformat(), prediction_id),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Recorded outcome for prediction {prediction_id}: {actual_return:.2f}%")

    def get_model_performance(
        self, model_type: str, market_condition: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(confidence) as avg_confidence,
                AVG(actual_return) as avg_return,
                SUM(CASE WHEN actual_return > 0 THEN actual_return ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN actual_return < 0 THEN actual_return ELSE 0 END)) as gross_loss
            FROM predictions
            WHERE model_type = ?
            AND timestamp >= ?
            AND was_correct IS NOT NULL
        """

        params = [model_type, since]

        if market_condition:
            query += " AND market_condition = ?"
            params.append(market_condition)

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {
                "model_type": model_type,
                "market_condition": market_condition or "all",
                "total_predictions": 0,
                "accuracy": 0,
                "avg_confidence": 0,
                "avg_return": 0,
                "profit_factor": 0,
                "days": days,
            }

        total, correct, avg_conf, avg_ret, gross_profit, gross_loss = row

        return {
            "model_type": model_type,
            "market_condition": market_condition or "all",
            "total_predictions": total,
            "correct_predictions": correct or 0,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "avg_confidence": avg_conf or 0,
            "avg_return": avg_ret or 0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "days": days,
        }

    def get_best_model_for_condition(
        self, market_condition: str, min_predictions: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Find the best performing model for a specific market condition."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                model_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(actual_return) as avg_return
            FROM predictions
            WHERE market_condition = ?
            AND was_correct IS NOT NULL
            GROUP BY model_type
            HAVING COUNT(*) >= ?
            ORDER BY (SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) DESC
            LIMIT 1
        """,
            (market_condition, min_predictions),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        model_type, total, correct, avg_return = row

        return {
            "model_type": model_type,
            "market_condition": market_condition,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "total_predictions": total,
            "avg_return": avg_return or 0,
        }

    def get_model_ranking(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get ranking of all models by performance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                model_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(confidence) as avg_confidence,
                AVG(actual_return) as avg_return,
                SUM(CASE WHEN actual_return > 0 THEN actual_return ELSE 0 END) as gross_profit,
                ABS(SUM(CASE WHEN actual_return < 0 THEN actual_return ELSE 0 END)) as gross_loss
            FROM predictions
            WHERE timestamp >= ?
            AND was_correct IS NOT NULL
            GROUP BY model_type
            ORDER BY AVG(actual_return) DESC
        """,
            (since,),
        )

        rows = cursor.fetchall()
        conn.close()

        ranking = []
        for row in rows:
            model_type, total, correct, avg_conf, avg_ret, gross_profit, gross_loss = row
            ranking.append(
                {
                    "model_type": model_type,
                    "total_predictions": total,
                    "accuracy": (correct / total * 100) if total > 0 else 0,
                    "avg_confidence": avg_conf or 0,
                    "avg_return": avg_ret or 0,
                    "profit_factor": (gross_profit / gross_loss)
                    if gross_loss > 0
                    else float("inf"),
                }
            )

        return ranking

    def get_condition_performance_matrix(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Get a matrix of model performance by market condition.

        Returns: {model_type: {condition: metrics}}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT
                model_type,
                market_condition,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(actual_return) as avg_return
            FROM predictions
            WHERE timestamp >= ?
            AND was_correct IS NOT NULL
            GROUP BY model_type, market_condition
        """,
            (since,),
        )

        rows = cursor.fetchall()
        conn.close()

        matrix: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for row in rows:
            model_type, condition, total, correct, avg_return = row
            matrix[model_type][condition] = {
                "total": total,
                "accuracy": (correct / total * 100) if total > 0 else 0,
                "avg_return": avg_return or 0,
            }

        return dict(matrix)

    def get_recommendation(self, market_condition: str) -> Dict[str, Any]:
        """
        Get model recommendation for current market condition.

        Returns recommended model and confidence in recommendation.
        """
        best = self.get_best_model_for_condition(market_condition)

        if not best:
            # Fall back to overall best
            ranking = self.get_model_ranking(days=30)
            if ranking:
                best = ranking[0]
                best["market_condition"] = "overall"

        if not best:
            return {
                "recommended_model": "ensemble",
                "confidence": 0.5,
                "reason": "No historical data - using default ensemble",
            }

        # Calculate confidence based on sample size and accuracy
        sample_confidence = min(best["total_predictions"] / 50, 1.0)
        accuracy_confidence = best["accuracy"] / 100

        return {
            "recommended_model": best["model_type"],
            "confidence": (sample_confidence + accuracy_confidence) / 2,
            "accuracy": best["accuracy"],
            "total_samples": best["total_predictions"],
            "avg_return": best.get("avg_return", 0),
            "reason": f"Best performer for {market_condition} with {best['accuracy']:.1f}% accuracy",
        }

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of ML model performance."""
        ranking = self.get_model_ranking(days)
        matrix = self.get_condition_performance_matrix(days)

        # Find best model overall
        best_overall = ranking[0] if ranking else None

        # Find best model per condition
        conditions = ["bull", "bear", "sideways", "volatile"]
        best_per_condition = {}
        for cond in conditions:
            best = self.get_best_model_for_condition(cond)
            if best:
                best_per_condition[cond] = best["model_type"]

        return {
            "period_days": days,
            "total_models_tracked": len(ranking),
            "best_overall": best_overall,
            "best_per_condition": best_per_condition,
            "model_ranking": ranking,
            "performance_matrix": matrix,
        }


# Global instance
_tracker: Optional[MLPerformanceTracker] = None


def get_ml_tracker() -> MLPerformanceTracker:
    """Get the global ML performance tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MLPerformanceTracker()
    return _tracker


def track_prediction(
    model_type: str,
    symbol: str,
    prediction: str,
    confidence: float,
    market_condition: str,
    volatility: float = 50.0,
    predicted_return: Optional[float] = None,
) -> str:
    """Convenience function to track a prediction."""
    return get_ml_tracker().record_prediction(
        model_type=model_type,
        symbol=symbol,
        prediction=prediction,
        confidence=confidence,
        market_condition=market_condition,
        volatility=volatility,
        predicted_return=predicted_return,
    )


def track_outcome(prediction_id: str, actual_return: float):
    """Convenience function to track an outcome."""
    get_ml_tracker().record_outcome(prediction_id, actual_return)
