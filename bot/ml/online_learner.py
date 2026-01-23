"""
Online Learning Module for Real-Time Model Adaptation.

Updates model weights based on recent prediction performance.
Uses exponential decay to prioritize recent data.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""

    timestamp: datetime
    symbol: str
    prediction: int  # 0=SHORT, 1=HOLD, 2=LONG
    confidence: float
    actual_return: Optional[float] = None
    actual_label: Optional[int] = None
    correct: Optional[bool] = None


@dataclass
class ModelPerformance:
    """Track model performance over time."""

    model_name: str
    predictions: int = 0
    correct: int = 0
    recent_accuracy: float = 0.5
    accuracy_ema: float = 0.5  # Exponential moving average
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy(self) -> float:
        return self.correct / self.predictions if self.predictions > 0 else 0.5


class OnlineLearner:
    """
    Adaptive online learning system.

    Features:
    - Tracks prediction accuracy in real-time
    - Adjusts model weights based on recent performance
    - Implements exponential decay for recency bias
    - Provides ensemble weight recommendations
    """

    def __init__(
        self,
        lookback_hours: int = 168,  # 1 week
        ema_alpha: float = 0.1,
        min_predictions: int = 20,
        data_dir: Path = Path("data/ml_online"),
    ):
        self.lookback_hours = lookback_hours
        self.ema_alpha = ema_alpha
        self.min_predictions = min_predictions
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Track performance per model per symbol
        self.model_performance: Dict[str, Dict[str, ModelPerformance]] = {}

        # Pending predictions waiting for outcomes
        self.pending_predictions: List[PredictionRecord] = []

        # Load saved state
        self._load_state()

        logger.info(f"OnlineLearner initialized: lookback={lookback_hours}h, alpha={ema_alpha}")

    def record_prediction(
        self, symbol: str, model_name: str, prediction: int, confidence: float
    ) -> None:
        """Record a new prediction to be evaluated later."""
        record = PredictionRecord(
            timestamp=datetime.now(), symbol=symbol, prediction=prediction, confidence=confidence
        )

        # Store with model info
        record.model_name = model_name  # type: ignore
        self.pending_predictions.append(record)

        # Initialize tracking if needed
        if symbol not in self.model_performance:
            self.model_performance[symbol] = {}
        if model_name not in self.model_performance[symbol]:
            self.model_performance[symbol][model_name] = ModelPerformance(model_name=model_name)

    def update_outcomes(self, symbol: str, current_price: float, lookback_hours: int = 6) -> int:
        """
        Update predictions with actual outcomes.

        Checks pending predictions that are old enough to evaluate.
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        updated = 0

        for record in self.pending_predictions[:]:
            if record.symbol != symbol:
                continue

            if record.timestamp > cutoff:
                continue  # Too recent to evaluate

            # Calculate actual return since prediction
            if hasattr(record, "entry_price") and record.entry_price:
                actual_return = (current_price - record.entry_price) / record.entry_price
            else:
                # Skip if no entry price recorded
                self.pending_predictions.remove(record)
                continue

            # Determine actual label
            if actual_return > 0.003:
                actual_label = 2  # LONG was correct
            elif actual_return < -0.003:
                actual_label = 0  # SHORT was correct
            else:
                actual_label = 1  # HOLD was correct

            record.actual_return = actual_return
            record.actual_label = actual_label
            record.correct = record.prediction == actual_label

            # Update model performance
            model_name = getattr(record, "model_name", "unknown")
            if symbol in self.model_performance and model_name in self.model_performance[symbol]:
                perf = self.model_performance[symbol][model_name]
                perf.predictions += 1
                if record.correct:
                    perf.correct += 1

                # Update EMA
                perf.accuracy_ema = (
                    self.ema_alpha * (1.0 if record.correct else 0.0)
                    + (1 - self.ema_alpha) * perf.accuracy_ema
                )
                perf.last_updated = datetime.now()

            self.pending_predictions.remove(record)
            updated += 1

        if updated > 0:
            self._save_state()

        return updated

    def get_model_weights(self, symbol: str) -> Dict[str, float]:
        """
        Get recommended ensemble weights based on recent performance.

        Models with higher recent accuracy get higher weights.
        """
        if symbol not in self.model_performance:
            return {}

        weights = {}
        total_score = 0.0

        for model_name, perf in self.model_performance[symbol].items():
            if perf.predictions < self.min_predictions:
                # Not enough data, use default weight
                score = 0.5
            else:
                # Use EMA accuracy as score
                score = perf.accuracy_ema

            # Apply softmax-like scaling
            score = np.exp(score * 3)  # Scale factor
            weights[model_name] = score
            total_score += score

        # Normalize
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}

        return weights

    def get_performance_report(self, symbol: str) -> Dict:
        """Get detailed performance report for a symbol."""
        if symbol not in self.model_performance:
            return {"symbol": symbol, "models": {}}

        report = {
            "symbol": symbol,
            "models": {},
            "pending_predictions": len([p for p in self.pending_predictions if p.symbol == symbol]),
        }

        for model_name, perf in self.model_performance[symbol].items():
            report["models"][model_name] = {
                "predictions": perf.predictions,
                "correct": perf.correct,
                "accuracy": perf.accuracy,
                "accuracy_ema": perf.accuracy_ema,
                "last_updated": perf.last_updated.isoformat(),
            }

        return report

    def should_retrain(self, symbol: str, model_name: str, threshold: float = 0.35) -> bool:
        """Check if a model should be retrained due to poor performance."""
        if symbol not in self.model_performance:
            return False
        if model_name not in self.model_performance[symbol]:
            return False

        perf = self.model_performance[symbol][model_name]

        # Need minimum predictions
        if perf.predictions < self.min_predictions * 2:
            return False

        # Check if accuracy is below threshold
        return perf.accuracy_ema < threshold

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "model_performance": {
                symbol: {
                    model: {
                        "predictions": perf.predictions,
                        "correct": perf.correct,
                        "accuracy_ema": perf.accuracy_ema,
                        "last_updated": perf.last_updated.isoformat(),
                    }
                    for model, perf in models.items()
                }
                for symbol, models in self.model_performance.items()
            },
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.data_dir / "online_learner_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.data_dir / "online_learner_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for symbol, models in state.get("model_performance", {}).items():
                self.model_performance[symbol] = {}
                for model_name, data in models.items():
                    self.model_performance[symbol][model_name] = ModelPerformance(
                        model_name=model_name,
                        predictions=data["predictions"],
                        correct=data["correct"],
                        accuracy_ema=data["accuracy_ema"],
                        last_updated=datetime.fromisoformat(data["last_updated"]),
                    )

            logger.info("Loaded online learner state")
        except Exception as e:
            logger.warning(f"Failed to load online learner state: {e}")


# Global instance
_online_learner: Optional[OnlineLearner] = None


def get_online_learner() -> OnlineLearner:
    """Get or create online learner instance."""
    global _online_learner
    if _online_learner is None:
        _online_learner = OnlineLearner()
    return _online_learner
