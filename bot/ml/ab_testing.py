"""
A/B Testing Framework for Trading Models.

Enables controlled experiments to compare model performance:
- Traffic splitting between control and treatment models
- Statistical significance testing
- Automatic winner selection
- Gradual rollout support

Usage:
    from bot.ml.ab_testing import ABExperiment, ABTestManager

    # Create experiment
    manager = ABTestManager()
    experiment = manager.create_experiment(
        name="new_lstm_model",
        control_model=current_model,
        treatment_model=new_model,
        traffic_split=0.2,  # 20% to treatment
    )

    # Get model for prediction
    model = manager.get_model("new_lstm_model", symbol="BTC/USDT")

    # Record outcome
    manager.record_outcome("new_lstm_model", "BTC/USDT", predicted=1, actual=1, pnl=50.0)

    # Check results
    results = manager.analyze_experiment("new_lstm_model")
"""

from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    DRAFT = "draft"           # Not started
    RUNNING = "running"       # Actively running
    PAUSED = "paused"         # Temporarily paused
    COMPLETED = "completed"   # Finished, winner selected
    STOPPED = "stopped"       # Stopped without winner


class WinnerDecision(Enum):
    """Decision on experiment winner."""
    CONTROL = "control"
    TREATMENT = "treatment"
    NO_WINNER = "no_winner"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ExperimentOutcome:
    """Single outcome from an experiment."""
    timestamp: datetime
    symbol: str
    variant: str  # "control" or "treatment"
    predicted: int  # Prediction (1=up, 0=down, -1=sell)
    actual: int  # Actual outcome
    pnl: float  # Profit/loss from this prediction
    correct: bool


@dataclass
class VariantMetrics:
    """Aggregated metrics for a variant."""
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    sharpe_ratio: float = 0.0
    pnl_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.avg_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "wins": self.wins,
            "losses": self.losses,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "pnl_std": round(self.pnl_std, 4),
        }


@dataclass
class ExperimentResults:
    """Results of an A/B experiment analysis."""
    experiment_name: str
    status: ExperimentStatus
    control_metrics: VariantMetrics
    treatment_metrics: VariantMetrics

    # Statistical tests
    accuracy_p_value: float = 1.0
    pnl_p_value: float = 1.0
    accuracy_significant: bool = False
    pnl_significant: bool = False

    # Winner decision
    winner: WinnerDecision = WinnerDecision.INCONCLUSIVE
    confidence_level: float = 0.0
    recommendation: str = ""

    # Relative improvement
    accuracy_lift: float = 0.0
    pnl_lift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "control": self.control_metrics.to_dict(),
            "treatment": self.treatment_metrics.to_dict(),
            "statistical_tests": {
                "accuracy_p_value": round(self.accuracy_p_value, 4),
                "pnl_p_value": round(self.pnl_p_value, 4),
                "accuracy_significant": self.accuracy_significant,
                "pnl_significant": self.pnl_significant,
            },
            "decision": {
                "winner": self.winner.value,
                "confidence_level": round(self.confidence_level, 4),
                "recommendation": self.recommendation,
            },
            "lift": {
                "accuracy_lift_pct": round(self.accuracy_lift * 100, 2),
                "pnl_lift_pct": round(self.pnl_lift * 100, 2),
            },
        }


@dataclass
class ABExperiment:
    """An A/B testing experiment configuration."""
    name: str
    description: str
    control_model: Any
    treatment_model: Any
    traffic_split: float  # Fraction going to treatment (0-1)

    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Configuration
    min_samples: int = 100
    max_duration_days: int = 14
    significance_level: float = 0.05
    min_effect_size: float = 0.02  # 2% improvement required

    # Outcomes storage
    outcomes: List[ExperimentOutcome] = field(default_factory=list)

    # Winner
    winner: WinnerDecision = WinnerDecision.INCONCLUSIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "traffic_split": self.traffic_split,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "min_samples": self.min_samples,
            "max_duration_days": self.max_duration_days,
            "significance_level": self.significance_level,
            "total_outcomes": len(self.outcomes),
            "winner": self.winner.value,
        }


class ABTestManager:
    """
    Manages A/B testing experiments for trading models.

    Features:
    - Multiple concurrent experiments
    - Statistical significance testing
    - Automatic traffic splitting
    - Winner selection with configurable criteria
    - Gradual rollout support
    """

    def __init__(
        self,
        data_dir: str = "data/ab_testing",
        default_significance: float = 0.05,
    ):
        """
        Initialize AB Test Manager.

        Args:
            data_dir: Directory for persisting experiment data
            default_significance: Default p-value threshold
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.default_significance = default_significance

        self._experiments: Dict[str, ABExperiment] = {}
        self._lock = threading.RLock()

        self._load_state()

    def _get_state_file(self) -> Path:
        return self.data_dir / "ab_experiments.json"

    def _load_state(self) -> None:
        """Load experiment state from disk."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    # Restore basic experiment metadata
                    for exp_data in data.get("experiments", []):
                        # Note: models are not persisted, must be re-registered
                        exp = ABExperiment(
                            name=exp_data["name"],
                            description=exp_data.get("description", ""),
                            control_model=None,
                            treatment_model=None,
                            traffic_split=exp_data.get("traffic_split", 0.5),
                            status=ExperimentStatus(exp_data.get("status", "draft")),
                            min_samples=exp_data.get("min_samples", 100),
                            max_duration_days=exp_data.get("max_duration_days", 14),
                        )
                        if exp_data.get("created_at"):
                            exp.created_at = datetime.fromisoformat(exp_data["created_at"])
                        if exp_data.get("started_at"):
                            exp.started_at = datetime.fromisoformat(exp_data["started_at"])
                        self._experiments[exp.name] = exp
                logger.info(f"Loaded {len(self._experiments)} experiments")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load AB test state: {e}")

    def _save_state(self) -> None:
        """Save experiment state to disk."""
        data = {
            "experiments": [exp.to_dict() for exp in self._experiments.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._get_state_file(), "w") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        name: str,
        control_model: Any,
        treatment_model: Any,
        traffic_split: float = 0.5,
        description: str = "",
        min_samples: int = 100,
        max_duration_days: int = 14,
        significance_level: Optional[float] = None,
        min_effect_size: float = 0.02,
    ) -> ABExperiment:
        """
        Create a new A/B experiment.

        Args:
            name: Unique experiment name
            control_model: Current production model
            treatment_model: New model to test
            traffic_split: Fraction of traffic to treatment (0-1)
            description: Experiment description
            min_samples: Minimum samples before analysis
            max_duration_days: Maximum experiment duration
            significance_level: P-value threshold for significance
            min_effect_size: Minimum improvement to declare winner

        Returns:
            Created ABExperiment
        """
        if not 0 < traffic_split < 1:
            raise ValueError("traffic_split must be between 0 and 1")

        with self._lock:
            if name in self._experiments:
                raise ValueError(f"Experiment '{name}' already exists")

            experiment = ABExperiment(
                name=name,
                description=description,
                control_model=control_model,
                treatment_model=treatment_model,
                traffic_split=traffic_split,
                min_samples=min_samples,
                max_duration_days=max_duration_days,
                significance_level=significance_level or self.default_significance,
                min_effect_size=min_effect_size,
            )

            self._experiments[name] = experiment
            self._save_state()

        logger.info(f"Created experiment: {name} (split: {traffic_split})")
        return experiment

    def start_experiment(self, name: str) -> None:
        """Start an experiment."""
        with self._lock:
            exp = self._experiments.get(name)
            if not exp:
                raise ValueError(f"Experiment '{name}' not found")

            if exp.status == ExperimentStatus.RUNNING:
                logger.warning(f"Experiment '{name}' already running")
                return

            exp.status = ExperimentStatus.RUNNING
            exp.started_at = datetime.now()
            self._save_state()

        logger.info(f"Started experiment: {name}")

    def pause_experiment(self, name: str) -> None:
        """Pause an experiment."""
        with self._lock:
            exp = self._experiments.get(name)
            if not exp:
                raise ValueError(f"Experiment '{name}' not found")

            exp.status = ExperimentStatus.PAUSED
            self._save_state()

        logger.info(f"Paused experiment: {name}")

    def stop_experiment(self, name: str, winner: Optional[WinnerDecision] = None) -> None:
        """Stop an experiment with optional winner declaration."""
        with self._lock:
            exp = self._experiments.get(name)
            if not exp:
                raise ValueError(f"Experiment '{name}' not found")

            exp.status = ExperimentStatus.COMPLETED if winner else ExperimentStatus.STOPPED
            exp.ended_at = datetime.now()
            if winner:
                exp.winner = winner
            self._save_state()

        logger.info(f"Stopped experiment: {name} (winner: {winner})")

    def get_model(
        self,
        experiment_name: str,
        symbol: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """
        Get model for a request based on traffic split.

        Uses consistent hashing on symbol/user_id for sticky assignment.

        Args:
            experiment_name: Name of the experiment
            symbol: Trading symbol (for consistent assignment)
            user_id: User identifier (for consistent assignment)

        Returns:
            Tuple of (model, variant_name)
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            if exp.status != ExperimentStatus.RUNNING:
                # Return control for non-running experiments
                return exp.control_model, "control"

            # Determine variant using consistent hashing
            hash_key = f"{symbol}_{user_id}" if symbol or user_id else str(random.random())
            hash_value = hash(hash_key) % 1000 / 1000.0

            if hash_value < exp.traffic_split:
                return exp.treatment_model, "treatment"
            else:
                return exp.control_model, "control"

    def record_outcome(
        self,
        experiment_name: str,
        symbol: str,
        predicted: int,
        actual: int,
        pnl: float,
        variant: Optional[str] = None,
    ) -> None:
        """
        Record an outcome from an experiment.

        Args:
            experiment_name: Name of the experiment
            symbol: Trading symbol
            predicted: Predicted direction
            actual: Actual direction
            pnl: Profit/loss from this prediction
            variant: Variant used (if known)
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            if exp.status != ExperimentStatus.RUNNING:
                logger.debug(f"Experiment '{experiment_name}' not running, skipping outcome")
                return

            # Determine variant if not provided
            if variant is None:
                _, variant = self.get_model(experiment_name, symbol)

            outcome = ExperimentOutcome(
                timestamp=datetime.now(),
                symbol=symbol,
                variant=variant,
                predicted=predicted,
                actual=actual,
                pnl=pnl,
                correct=predicted == actual,
            )

            exp.outcomes.append(outcome)

            # Check if we should auto-analyze
            if len(exp.outcomes) % 50 == 0:
                self._auto_analyze(exp)

    def _auto_analyze(self, exp: ABExperiment) -> None:
        """Automatically analyze experiment and potentially end it."""
        results = self._analyze_experiment(exp)

        # Check if we have enough samples and clear winner
        control_n = results.control_metrics.total_samples
        treatment_n = results.treatment_metrics.total_samples

        if min(control_n, treatment_n) < exp.min_samples:
            return

        # Check duration
        if exp.started_at:
            days_running = (datetime.now() - exp.started_at).days
            if days_running >= exp.max_duration_days:
                # Force conclusion
                self._conclude_experiment(exp, results)
                return

        # Check for clear winner with high confidence
        if results.confidence_level >= 0.95:
            if results.winner in [WinnerDecision.CONTROL, WinnerDecision.TREATMENT]:
                self._conclude_experiment(exp, results)

    def _conclude_experiment(self, exp: ABExperiment, results: ExperimentResults) -> None:
        """Conclude an experiment with results."""
        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = datetime.now()
        exp.winner = results.winner
        self._save_state()

        logger.info(
            f"Experiment '{exp.name}' concluded: {results.winner.value} "
            f"(confidence: {results.confidence_level:.1%})"
        )

    def analyze_experiment(self, experiment_name: str) -> ExperimentResults:
        """
        Analyze an experiment's results.

        Args:
            experiment_name: Name of the experiment

        Returns:
            ExperimentResults with statistical analysis
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            return self._analyze_experiment(exp)

    def _analyze_experiment(self, exp: ABExperiment) -> ExperimentResults:
        """Internal analysis method."""
        # Separate outcomes by variant
        control_outcomes = [o for o in exp.outcomes if o.variant == "control"]
        treatment_outcomes = [o for o in exp.outcomes if o.variant == "treatment"]

        # Calculate metrics
        control_metrics = self._calculate_variant_metrics(control_outcomes)
        treatment_metrics = self._calculate_variant_metrics(treatment_outcomes)

        # Statistical tests
        accuracy_p_value = 1.0
        pnl_p_value = 1.0

        if control_metrics.total_samples >= 20 and treatment_metrics.total_samples >= 20:
            # Chi-squared test for accuracy
            try:
                contingency = [
                    [control_metrics.correct_predictions,
                     control_metrics.total_samples - control_metrics.correct_predictions],
                    [treatment_metrics.correct_predictions,
                     treatment_metrics.total_samples - treatment_metrics.correct_predictions],
                ]
                _, accuracy_p_value, _, _ = stats.chi2_contingency(contingency)
            except (ValueError, ZeroDivisionError):
                accuracy_p_value = 1.0

            # T-test for PnL
            control_pnls = [o.pnl for o in control_outcomes]
            treatment_pnls = [o.pnl for o in treatment_outcomes]

            if len(control_pnls) >= 20 and len(treatment_pnls) >= 20:
                try:
                    _, pnl_p_value = stats.ttest_ind(control_pnls, treatment_pnls)
                except (ValueError, ZeroDivisionError):
                    pnl_p_value = 1.0

        # Determine significance
        accuracy_significant = accuracy_p_value < exp.significance_level
        pnl_significant = pnl_p_value < exp.significance_level

        # Calculate lifts
        accuracy_lift = 0.0
        if control_metrics.accuracy > 0:
            accuracy_lift = (treatment_metrics.accuracy - control_metrics.accuracy) / control_metrics.accuracy

        pnl_lift = 0.0
        if control_metrics.avg_pnl != 0:
            pnl_lift = (treatment_metrics.avg_pnl - control_metrics.avg_pnl) / abs(control_metrics.avg_pnl)

        # Determine winner
        winner = WinnerDecision.INCONCLUSIVE
        confidence_level = 0.0
        recommendation = ""

        if min(control_metrics.total_samples, treatment_metrics.total_samples) < exp.min_samples:
            recommendation = f"Need more samples (min: {exp.min_samples})"
        else:
            # Calculate confidence (1 - min p-value)
            confidence_level = 1 - min(accuracy_p_value, pnl_p_value)

            if pnl_significant and accuracy_significant:
                # Both metrics significant
                if treatment_metrics.avg_pnl > control_metrics.avg_pnl * (1 + exp.min_effect_size):
                    winner = WinnerDecision.TREATMENT
                    recommendation = "Deploy treatment model - significant improvement in both metrics"
                elif control_metrics.avg_pnl > treatment_metrics.avg_pnl * (1 + exp.min_effect_size):
                    winner = WinnerDecision.CONTROL
                    recommendation = "Keep control model - treatment underperformed"
                else:
                    winner = WinnerDecision.NO_WINNER
                    recommendation = "No significant difference - keep control"
            elif pnl_significant:
                # Only PnL significant
                if treatment_metrics.avg_pnl > control_metrics.avg_pnl * (1 + exp.min_effect_size):
                    winner = WinnerDecision.TREATMENT
                    recommendation = "Deploy treatment model - significant PnL improvement"
                else:
                    winner = WinnerDecision.CONTROL
                    recommendation = "Keep control model"
            else:
                winner = WinnerDecision.INCONCLUSIVE
                recommendation = "Results not yet statistically significant"

        return ExperimentResults(
            experiment_name=exp.name,
            status=exp.status,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            accuracy_p_value=accuracy_p_value,
            pnl_p_value=pnl_p_value,
            accuracy_significant=accuracy_significant,
            pnl_significant=pnl_significant,
            winner=winner,
            confidence_level=confidence_level,
            recommendation=recommendation,
            accuracy_lift=accuracy_lift,
            pnl_lift=pnl_lift,
        )

    def _calculate_variant_metrics(self, outcomes: List[ExperimentOutcome]) -> VariantMetrics:
        """Calculate metrics for a variant's outcomes."""
        if not outcomes:
            return VariantMetrics()

        total = len(outcomes)
        correct = sum(1 for o in outcomes if o.correct)
        pnls = [o.pnl for o in outcomes]
        wins = sum(1 for o in outcomes if o.pnl > 0)
        losses = sum(1 for o in outcomes if o.pnl <= 0)

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total if total > 0 else 0
        pnl_std = np.std(pnls) if len(pnls) > 1 else 0

        # Sharpe ratio (annualized assuming daily)
        sharpe_ratio = 0.0
        if pnl_std > 0:
            sharpe_ratio = (avg_pnl / pnl_std) * np.sqrt(252)

        return VariantMetrics(
            total_samples=total,
            correct_predictions=correct,
            accuracy=correct / total if total > 0 else 0,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            win_rate=wins / total if total > 0 else 0,
            wins=wins,
            losses=losses,
            sharpe_ratio=sharpe_ratio,
            pnl_std=pnl_std,
        )

    def get_experiment(self, name: str) -> Optional[ABExperiment]:
        """Get an experiment by name."""
        return self._experiments.get(name)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[ABExperiment]:
        """List all experiments, optionally filtered by status."""
        with self._lock:
            experiments = list(self._experiments.values())
            if status:
                experiments = [e for e in experiments if e.status == status]
            return experiments

    def delete_experiment(self, name: str) -> bool:
        """Delete an experiment."""
        with self._lock:
            if name in self._experiments:
                del self._experiments[name]
                self._save_state()
                logger.info(f"Deleted experiment: {name}")
                return True
            return False

    def get_all_results(self) -> Dict[str, ExperimentResults]:
        """Get analysis results for all experiments."""
        results = {}
        for name in self._experiments:
            try:
                results[name] = self.analyze_experiment(name)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
        return results

    def register_models(
        self,
        experiment_name: str,
        control_model: Any,
        treatment_model: Any,
    ) -> None:
        """
        Register/update models for an experiment.

        Use this to re-register models after loading from disk.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            exp.control_model = control_model
            exp.treatment_model = treatment_model

        logger.info(f"Registered models for experiment: {experiment_name}")

    def gradual_rollout(
        self,
        experiment_name: str,
        new_split: float,
    ) -> None:
        """
        Gradually increase/decrease traffic to treatment.

        Args:
            experiment_name: Name of the experiment
            new_split: New traffic split (0-1)
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            if not 0 <= new_split <= 1:
                raise ValueError("Split must be between 0 and 1")

            old_split = exp.traffic_split
            exp.traffic_split = new_split
            self._save_state()

        logger.info(
            f"Updated traffic split for {experiment_name}: {old_split:.1%} -> {new_split:.1%}"
        )


def create_ab_test_manager(
    data_dir: str = "data/ab_testing",
    default_significance: float = 0.05,
) -> ABTestManager:
    """Factory function to create AB test manager."""
    return ABTestManager(
        data_dir=data_dir,
        default_significance=default_significance,
    )
