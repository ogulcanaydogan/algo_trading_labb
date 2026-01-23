"""
A/B Testing Framework for Trading Strategies.

Provides statistical testing to compare strategy variants
and determine which performs better with significance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class AllocationMethod(Enum):
    """Traffic allocation method."""

    RANDOM = "random"
    HASH = "hash"  # Deterministic based on user/session ID
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"


@dataclass
class Variant:
    """Experiment variant (treatment or control)."""

    name: str
    weight: float = 0.5  # Traffic allocation weight
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "config": self.config,
            "is_control": self.is_control,
        }


@dataclass
class MetricResult:
    """Result for a single metric."""

    name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    absolute_diff: float
    relative_diff_pct: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "control_mean": round(self.control_mean, 6),
            "treatment_mean": round(self.treatment_mean, 6),
            "control_std": round(self.control_std, 6),
            "treatment_std": round(self.treatment_std, 6),
            "absolute_diff": round(self.absolute_diff, 6),
            "relative_diff_pct": round(self.relative_diff_pct, 2),
            "p_value": round(self.p_value, 6),
            "confidence_interval": (
                round(self.confidence_interval[0], 6),
                round(self.confidence_interval[1], 6),
            ),
            "is_significant": self.is_significant,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
        }


@dataclass
class ExperimentResult:
    """Complete experiment results."""

    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    metrics: Dict[str, MetricResult]
    winner: Optional[str]
    recommendation: str
    confidence_level: float
    total_samples: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": round(self.duration_hours, 2),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "winner": self.winner,
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level,
            "total_samples": self.total_samples,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Experiment:
    """A/B test experiment."""

    id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[str]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    allocation_method: AllocationMethod = AllocationMethod.HASH
    min_sample_size: int = 100
    max_duration_hours: float = 168  # 1 week
    significance_level: float = 0.05
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Data storage
    _observations: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    _assignments: Dict[str, str] = field(default_factory=dict)
    _assignment_counter: int = 0

    def __post_init__(self):
        if not self._observations:
            self._observations = {v.name: {m: [] for m in self.metrics} for v in self.variants}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "metrics": self.metrics,
            "status": self.status.value,
            "allocation_method": self.allocation_method.value,
            "min_sample_size": self.min_sample_size,
            "max_duration_hours": self.max_duration_hours,
            "significance_level": self.significance_level,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class ABTestingFramework:
    """
    A/B Testing Framework for Trading Strategies.

    Features:
    - Multiple variants support
    - Statistical significance testing
    - Sequential analysis (early stopping)
    - Metric tracking and analysis
    - Experiment lifecycle management
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        metrics: List[str],
        allocation_method: AllocationMethod = AllocationMethod.HASH,
        min_sample_size: int = 100,
        max_duration_hours: float = 168,
        significance_level: float = 0.05,
    ) -> Experiment:
        """Create a new A/B test experiment."""
        # Validate
        if len(variants) < 2:
            raise ValueError("At least 2 variants required")

        control_count = sum(1 for v in variants if v.is_control)
        if control_count != 1:
            raise ValueError("Exactly one control variant required")

        # Normalize weights
        total_weight = sum(v.weight for v in variants)
        for v in variants:
            v.weight = v.weight / total_weight

        experiment = Experiment(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            allocation_method=allocation_method,
            min_sample_size=min_sample_size,
            max_duration_hours=max_duration_hours,
            significance_level=significance_level,
        )

        self._experiments[experiment.id] = experiment
        logger.info(f"Created experiment: {experiment.id} - {name}")

        return experiment

    def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if exp.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Can only start DRAFT experiments, current: {exp.status}")

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now()
        logger.info(f"Started experiment: {experiment_id}")

    def pause_experiment(self, experiment_id: str):
        """Pause an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only pause RUNNING experiments")

        exp.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment: {experiment_id}")

    def resume_experiment(self, experiment_id: str):
        """Resume a paused experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if exp.status != ExperimentStatus.PAUSED:
            raise ValueError("Can only resume PAUSED experiments")

        exp.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed experiment: {experiment_id}")

    def stop_experiment(self, experiment_id: str):
        """Stop an experiment early."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if exp.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError("Can only stop RUNNING or PAUSED experiments")

        exp.status = ExperimentStatus.STOPPED
        exp.ended_at = datetime.now()
        logger.info(f"Stopped experiment: {experiment_id}")

    def assign_variant(self, experiment_id: str, entity_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Assign an entity to a variant.

        Args:
            experiment_id: Experiment ID
            entity_id: Unique entity identifier (user, session, trade, etc.)

        Returns:
            Tuple of (variant_name, variant_config)
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if exp.status != ExperimentStatus.RUNNING:
            # Return control for non-running experiments
            control = next(v for v in exp.variants if v.is_control)
            return control.name, control.config

        # Check existing assignment
        if entity_id in exp._assignments:
            variant_name = exp._assignments[entity_id]
            variant = next(v for v in exp.variants if v.name == variant_name)
            return variant.name, variant.config

        # Assign based on method
        if exp.allocation_method == AllocationMethod.HASH:
            variant = self._hash_assignment(exp, entity_id)
        elif exp.allocation_method == AllocationMethod.RANDOM:
            variant = self._random_assignment(exp)
        elif exp.allocation_method == AllocationMethod.ROUND_ROBIN:
            variant = self._round_robin_assignment(exp)
        else:
            variant = self._weighted_assignment(exp)

        exp._assignments[entity_id] = variant.name
        return variant.name, variant.config

    def _hash_assignment(self, exp: Experiment, entity_id: str) -> Variant:
        """Deterministic assignment based on hash."""
        hash_input = f"{exp.id}:{entity_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant

        return exp.variants[-1]

    def _random_assignment(self, exp: Experiment) -> Variant:
        """Random assignment based on weights."""
        return random.choices(exp.variants, weights=[v.weight for v in exp.variants])[0]

    def _round_robin_assignment(self, exp: Experiment) -> Variant:
        """Round-robin assignment."""
        exp._assignment_counter += 1
        idx = exp._assignment_counter % len(exp.variants)
        return exp.variants[idx]

    def _weighted_assignment(self, exp: Experiment) -> Variant:
        """Weighted assignment (same as random)."""
        return self._random_assignment(exp)

    def record_metric(self, experiment_id: str, entity_id: str, metric_name: str, value: float):
        """
        Record a metric observation.

        Args:
            experiment_id: Experiment ID
            entity_id: Entity that generated the metric
            metric_name: Name of the metric
            value: Metric value
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        if metric_name not in exp.metrics:
            return

        # Get variant assignment
        variant_name = exp._assignments.get(entity_id)
        if not variant_name:
            return

        # Record observation
        if variant_name in exp._observations:
            if metric_name in exp._observations[variant_name]:
                exp._observations[variant_name][metric_name].append(value)

    def analyze_experiment(
        self, experiment_id: str, primary_metric: Optional[str] = None
    ) -> ExperimentResult:
        """
        Analyze experiment results.

        Args:
            experiment_id: Experiment ID
            primary_metric: Primary metric for winner determination

        Returns:
            ExperimentResult with statistical analysis
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Get control variant
        control = next(v for v in exp.variants if v.is_control)
        control_data = exp._observations.get(control.name, {})

        # Analyze each metric against control
        metric_results = {}

        for metric_name in exp.metrics:
            control_values = control_data.get(metric_name, [])

            # Compare each treatment against control
            for variant in exp.variants:
                if variant.is_control:
                    continue

                treatment_data = exp._observations.get(variant.name, {})
                treatment_values = treatment_data.get(metric_name, [])

                result = self._analyze_metric(
                    metric_name, control_values, treatment_values, exp.significance_level
                )

                key = f"{metric_name}_{variant.name}"
                metric_results[key] = result

        # Determine winner
        primary = primary_metric or exp.metrics[0]
        winner = None
        recommendation = "No significant difference found"

        for key, result in metric_results.items():
            if primary in key and result.is_significant:
                if result.relative_diff_pct > 0:
                    variant_name = key.replace(f"{primary}_", "")
                    winner = variant_name
                    recommendation = (
                        f"{variant_name} outperforms control by {result.relative_diff_pct:.1f}%"
                    )
                else:
                    winner = control.name
                    recommendation = (
                        f"Control outperforms treatment by {abs(result.relative_diff_pct):.1f}%"
                    )

        # Calculate duration
        if exp.started_at:
            end = exp.ended_at or datetime.now()
            duration = (end - exp.started_at).total_seconds() / 3600
        else:
            duration = 0

        total_samples = sum(
            len(exp._observations.get(v.name, {}).get(exp.metrics[0], [])) for v in exp.variants
        )

        return ExperimentResult(
            experiment_id=exp.id,
            experiment_name=exp.name,
            status=exp.status,
            start_time=exp.started_at or exp.created_at,
            end_time=exp.ended_at,
            duration_hours=duration,
            metrics=metric_results,
            winner=winner,
            recommendation=recommendation,
            confidence_level=1 - exp.significance_level,
            total_samples=total_samples,
        )

    def _analyze_metric(
        self,
        metric_name: str,
        control_values: List[float],
        treatment_values: List[float],
        significance_level: float,
    ) -> MetricResult:
        """Perform statistical analysis on a metric."""
        # Handle empty data
        if not control_values or not treatment_values:
            return MetricResult(
                name=metric_name,
                control_mean=0,
                treatment_mean=0,
                control_std=0,
                treatment_std=0,
                absolute_diff=0,
                relative_diff_pct=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                is_significant=False,
                sample_size_control=len(control_values),
                sample_size_treatment=len(treatment_values),
            )

        control_arr = np.array(control_values)
        treatment_arr = np.array(treatment_values)

        control_mean = np.mean(control_arr)
        treatment_mean = np.mean(treatment_arr)
        control_std = np.std(control_arr, ddof=1) if len(control_arr) > 1 else 0
        treatment_std = np.std(treatment_arr, ddof=1) if len(treatment_arr) > 1 else 0

        absolute_diff = treatment_mean - control_mean
        relative_diff = (absolute_diff / control_mean * 100) if control_mean != 0 else 0

        # Welch's t-test (unequal variances)
        if len(control_arr) > 1 and len(treatment_arr) > 1:
            t_stat, p_value = stats.ttest_ind(treatment_arr, control_arr, equal_var=False)
        else:
            p_value = 1.0

        # Confidence interval for difference
        se_diff = (
            np.sqrt((control_std**2 / len(control_arr)) + (treatment_std**2 / len(treatment_arr)))
            if len(control_arr) > 0 and len(treatment_arr) > 0
            else 0
        )

        z = stats.norm.ppf(1 - significance_level / 2)
        ci_lower = absolute_diff - z * se_diff
        ci_upper = absolute_diff + z * se_diff

        return MetricResult(
            name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            absolute_diff=absolute_diff,
            relative_diff_pct=relative_diff,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < significance_level,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
        )

    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
    ) -> Dict[str, int]:
        """
        Calculate required sample size for experiment.

        Args:
            baseline_rate: Expected baseline conversion/metric rate
            minimum_detectable_effect: Minimum relative effect to detect (e.g., 0.05 for 5%)
            significance_level: Type I error rate (alpha)
            power: Statistical power (1 - Type II error)
            ratio: Ratio of treatment to control size

        Returns:
            Required sample sizes per variant
        """
        # For proportions
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        p_avg = (p1 + p2) / 2

        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n1 = (
            (
                z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
                + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
            )
            ** 2
        ) / ((p1 - p2) ** 2)

        n2 = n1 * ratio

        return {
            "control": int(np.ceil(n1)),
            "treatment": int(np.ceil(n2)),
            "total": int(np.ceil(n1 + n2)),
        }

    def check_early_stopping(
        self, experiment_id: str, method: str = "obrien_fleming"
    ) -> Dict[str, Any]:
        """
        Check if experiment can be stopped early.

        Uses sequential analysis methods:
        - obrien_fleming: O'Brien-Fleming boundaries
        - pocock: Pocock boundaries
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        result = self.analyze_experiment(experiment_id)

        # Calculate information fraction
        total_samples = result.total_samples
        target_samples = exp.min_sample_size * len(exp.variants)
        info_fraction = min(1.0, total_samples / target_samples)

        # Adjust significance level based on method
        if method == "obrien_fleming":
            # More conservative early, relaxed later
            adjusted_alpha = exp.significance_level * (
                2
                - 2
                * stats.norm.cdf(
                    stats.norm.ppf(1 - exp.significance_level / 2) / np.sqrt(info_fraction)
                )
            )
        else:  # pocock
            # Constant boundary
            adjusted_alpha = exp.significance_level / max(1, int(1 / info_fraction))

        can_stop = False
        reason = "Insufficient data"

        if info_fraction >= 0.5:  # Minimum 50% of data
            for key, metric_result in result.metrics.items():
                if metric_result.p_value < adjusted_alpha:
                    can_stop = True
                    reason = f"Significant result for {key} (p={metric_result.p_value:.4f} < {adjusted_alpha:.4f})"
                    break

        return {
            "experiment_id": experiment_id,
            "can_stop_early": can_stop,
            "reason": reason,
            "info_fraction": round(info_fraction, 3),
            "adjusted_alpha": round(adjusted_alpha, 6),
            "current_samples": total_samples,
            "target_samples": target_samples,
        }

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments


class StrategyABTest:
    """
    Specialized A/B testing for trading strategies.

    Metrics tracked:
    - Return
    - Sharpe Ratio
    - Win Rate
    - Max Drawdown
    - Trade Count
    """

    STRATEGY_METRICS = [
        "return",
        "sharpe_ratio",
        "win_rate",
        "max_drawdown",
        "trade_count",
        "avg_trade_pnl",
    ]

    def __init__(self, framework: Optional[ABTestingFramework] = None):
        self.framework = framework or ABTestingFramework()

    def create_strategy_test(
        self,
        name: str,
        control_config: Dict[str, Any],
        treatment_configs: List[Dict[str, Any]],
        min_trades: int = 100,
    ) -> Experiment:
        """
        Create a strategy A/B test.

        Args:
            name: Test name
            control_config: Control strategy configuration
            treatment_configs: List of treatment strategy configurations
            min_trades: Minimum trades before analysis
        """
        variants = [Variant(name="control", weight=0.5, config=control_config, is_control=True)]

        weight_per_treatment = 0.5 / len(treatment_configs)
        for i, config in enumerate(treatment_configs):
            variants.append(
                Variant(
                    name=f"treatment_{i + 1}",
                    weight=weight_per_treatment,
                    config=config,
                    is_control=False,
                )
            )

        return self.framework.create_experiment(
            name=name,
            description=f"Strategy A/B test: {name}",
            variants=variants,
            metrics=self.STRATEGY_METRICS,
            min_sample_size=min_trades,
        )

    def record_trade(
        self, experiment_id: str, trade_id: str, pnl: float, return_pct: float, win: bool
    ):
        """Record a trade result."""
        exp = self.framework.get_experiment(experiment_id)
        if not exp:
            return

        self.framework.record_metric(experiment_id, trade_id, "return", return_pct)
        self.framework.record_metric(experiment_id, trade_id, "avg_trade_pnl", pnl)
        self.framework.record_metric(experiment_id, trade_id, "win_rate", 1.0 if win else 0.0)
        self.framework.record_metric(experiment_id, trade_id, "trade_count", 1.0)


def create_ab_testing_framework() -> ABTestingFramework:
    """Factory function to create A/B testing framework."""
    return ABTestingFramework()


def create_strategy_ab_test(framework: Optional[ABTestingFramework] = None) -> StrategyABTest:
    """Factory function to create strategy A/B test."""
    return StrategyABTest(framework)
