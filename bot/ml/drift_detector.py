"""
Model Drift Detection - Auto-detect when models need retraining.

Monitors feature distributions and model performance to detect:
- Data drift (input distribution changes)
- Concept drift (target relationship changes)
- Performance degradation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected."""

    NONE = "none"
    DATA_DRIFT = "data_drift"  # Feature distribution changed
    CONCEPT_DRIFT = "concept_drift"  # Target relationship changed
    PERFORMANCE_DRIFT = "performance"  # Model accuracy degraded
    COMBINED = "combined"  # Multiple drift types


class DriftSeverity(Enum):
    """Severity of detected drift."""

    NONE = "none"
    LOW = "low"  # Monitor closely
    MEDIUM = "medium"  # Consider retraining soon
    HIGH = "high"  # Retrain immediately
    CRITICAL = "critical"  # Stop trading, retrain


@dataclass
class DriftReport:
    """Report of drift detection analysis."""

    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    overall_score: float  # 0-1, higher = more drift
    feature_drift_scores: Dict[str, float]
    drifted_features: List[str]
    performance_metrics: Dict[str, float]
    recommendation: str
    details: Dict

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "overall_score": round(self.overall_score, 4),
            "feature_drift_scores": {k: round(v, 4) for k, v in self.feature_drift_scores.items()},
            "drifted_features": self.drifted_features,
            "performance_metrics": {k: round(v, 4) for k, v in self.performance_metrics.items()},
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Statistical test thresholds
    ks_threshold: float = 0.1  # KS test p-value threshold
    psi_threshold: float = 0.2  # Population Stability Index threshold
    js_threshold: float = 0.1  # Jensen-Shannon divergence threshold

    # Performance thresholds
    accuracy_drop_threshold: float = 0.05  # 5% accuracy drop triggers alert
    sharpe_drop_threshold: float = 0.3  # Sharpe ratio drop threshold

    # Window sizes
    reference_window_days: int = 30  # Days for reference distribution
    detection_window_days: int = 7  # Days for current distribution
    min_samples: int = 100  # Minimum samples for detection

    # Alerting
    check_interval_hours: int = 6  # How often to check for drift


class DriftDetector:
    """
    Detect data drift and concept drift in ML models.

    Methods:
    - KS Test: Compare distributions using Kolmogorov-Smirnov
    - PSI: Population Stability Index
    - JS Divergence: Jensen-Shannon divergence
    - Performance monitoring: Track accuracy over time
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        data_dir: str = "data/drift_monitoring",
    ):
        self.config = config or DriftConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_history: List[DriftReport] = []
        self.performance_history: List[Dict] = []

        self._load_state()

    def _load_state(self):
        """Load saved state."""
        state_file = self.data_dir / "drift_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.reference_stats = state.get("reference_stats", {})
                    self.performance_history = state.get("performance_history", [])
                logger.info("Loaded drift detector state")
            except Exception as e:
                logger.error(f"Error loading drift state: {e}")

    def _save_state(self):
        """Save current state."""
        state_file = self.data_dir / "drift_state.json"
        try:
            state = {
                "reference_stats": self.reference_stats,
                "performance_history": self.performance_history[-1000:],  # Keep last 1000
                "updated_at": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving drift state: {e}")

    def set_reference(self, data: pd.DataFrame, feature_columns: List[str]):
        """
        Set reference distribution for drift detection.

        Call this after training a model to establish baseline.
        """
        self.reference_data = data[feature_columns].copy()

        # Compute reference statistics
        for col in feature_columns:
            values = data[col].dropna().values
            if len(values) < 10:
                continue

            self.reference_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "histogram": np.histogram(values, bins=20)[0].tolist(),
                "bin_edges": np.histogram(values, bins=20)[1].tolist(),
            }

        self._save_state()
        logger.info(f"Reference set with {len(feature_columns)} features")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Detect drift in current data vs reference.

        Args:
            current_data: Recent data to check
            feature_columns: Columns to check (uses reference cols if None)
            predictions: Model predictions (for concept drift)
            actuals: Actual values (for concept drift)

        Returns:
            DriftReport with analysis results
        """
        if not self.reference_stats:
            return DriftReport(
                timestamp=datetime.now(),
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                overall_score=0.0,
                feature_drift_scores={},
                drifted_features=[],
                performance_metrics={},
                recommendation="Set reference data first",
                details={"error": "No reference data"},
            )

        feature_columns = feature_columns or list(self.reference_stats.keys())
        feature_drift_scores = {}
        drifted_features = []

        # Check each feature for drift
        for col in feature_columns:
            if col not in self.reference_stats:
                continue
            if col not in current_data.columns:
                continue

            current_values = current_data[col].dropna().values
            if len(current_values) < self.config.min_samples:
                continue

            # Calculate drift score using multiple methods
            drift_score = self._calculate_feature_drift(col, current_values)
            feature_drift_scores[col] = drift_score

            if drift_score > self.config.psi_threshold:
                drifted_features.append(col)

        # Calculate overall drift score
        if feature_drift_scores:
            overall_score = np.mean(list(feature_drift_scores.values()))
        else:
            overall_score = 0.0

        # Check performance drift if predictions provided
        performance_metrics = {}
        concept_drift = False
        if predictions is not None and actuals is not None:
            performance_metrics = self._check_performance(predictions, actuals)
            if performance_metrics.get("accuracy_drop", 0) > self.config.accuracy_drop_threshold:
                concept_drift = True

        # Determine drift type and severity
        drift_type, severity = self._classify_drift(
            overall_score, len(drifted_features), concept_drift, performance_metrics
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(drift_type, severity, drifted_features)

        report = DriftReport(
            timestamp=datetime.now(),
            drift_type=drift_type,
            severity=severity,
            overall_score=overall_score,
            feature_drift_scores=feature_drift_scores,
            drifted_features=drifted_features,
            performance_metrics=performance_metrics,
            recommendation=recommendation,
            details={
                "num_features_checked": len(feature_drift_scores),
                "num_drifted": len(drifted_features),
                "config": {
                    "psi_threshold": self.config.psi_threshold,
                    "accuracy_drop_threshold": self.config.accuracy_drop_threshold,
                },
            },
        )

        self.drift_history.append(report)
        self._save_state()

        # Log if significant drift
        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            logger.warning(f"DRIFT ALERT: {drift_type.value} ({severity.value}) - {recommendation}")

        return report

    def _calculate_feature_drift(self, col: str, current_values: np.ndarray) -> float:
        """Calculate drift score for a single feature."""
        ref_stats = self.reference_stats[col]

        # Method 1: PSI (Population Stability Index)
        psi = self._calculate_psi(ref_stats, current_values)

        # Method 2: KS Test
        # Reconstruct reference distribution from histogram
        ref_hist = np.array(ref_stats["histogram"])
        ref_edges = np.array(ref_stats["bin_edges"])
        ref_centers = (ref_edges[:-1] + ref_edges[1:]) / 2

        # Sample from reference histogram
        ref_samples = np.random.choice(
            ref_centers,
            size=min(len(current_values), 1000),
            p=ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else None,
        )

        ks_stat, ks_pvalue = stats.ks_2samp(ref_samples, current_values)

        # Combine scores (PSI is primary)
        drift_score = psi * 0.7 + (1 - ks_pvalue) * 0.3

        return float(drift_score)

    def _calculate_psi(self, ref_stats: Dict, current_values: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        ref_hist = np.array(ref_stats["histogram"])
        ref_edges = np.array(ref_stats["bin_edges"])

        # Create histogram for current data using same bins
        current_hist, _ = np.histogram(current_values, bins=ref_edges)

        # Normalize to percentages
        ref_pct = ref_hist / (ref_hist.sum() + 1e-10)
        current_pct = current_hist / (current_hist.sum() + 1e-10)

        # Add small value to avoid log(0)
        ref_pct = np.clip(ref_pct, 1e-10, 1)
        current_pct = np.clip(current_pct, 1e-10, 1)

        # Calculate PSI
        psi = np.sum((current_pct - ref_pct) * np.log(current_pct / ref_pct))

        return float(psi)

    def _check_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Check model performance for degradation."""
        from sklearn.metrics import accuracy_score, f1_score

        accuracy = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        # Compare to historical performance
        accuracy_drop = 0.0
        if self.performance_history:
            recent_acc = np.mean([p["accuracy"] for p in self.performance_history[-10:]])
            accuracy_drop = recent_acc - accuracy

        # Store current performance
        self.performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "accuracy": accuracy,
                "f1": f1,
            }
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "accuracy_drop": accuracy_drop,
        }

    def _classify_drift(
        self,
        overall_score: float,
        num_drifted: int,
        concept_drift: bool,
        performance_metrics: Dict,
    ) -> Tuple[DriftType, DriftSeverity]:
        """Classify drift type and severity."""
        # Determine drift type
        data_drift = overall_score > self.config.psi_threshold / 2

        if data_drift and concept_drift:
            drift_type = DriftType.COMBINED
        elif concept_drift:
            drift_type = DriftType.CONCEPT_DRIFT
        elif data_drift:
            drift_type = DriftType.DATA_DRIFT
        else:
            drift_type = DriftType.NONE

        # Determine severity
        if drift_type == DriftType.NONE:
            severity = DriftSeverity.NONE
        elif overall_score > 0.5 or performance_metrics.get("accuracy_drop", 0) > 0.15:
            severity = DriftSeverity.CRITICAL
        elif overall_score > 0.3 or performance_metrics.get("accuracy_drop", 0) > 0.1:
            severity = DriftSeverity.HIGH
        elif overall_score > 0.15 or num_drifted > 5:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return drift_type, severity

    def _generate_recommendation(
        self,
        drift_type: DriftType,
        severity: DriftSeverity,
        drifted_features: List[str],
    ) -> str:
        """Generate actionable recommendation."""
        if drift_type == DriftType.NONE:
            return "No drift detected. Model is stable."

        recommendations = {
            DriftSeverity.LOW: "Monitor closely. Consider retraining in next scheduled window.",
            DriftSeverity.MEDIUM: "Schedule retraining soon. Reduce position sizes.",
            DriftSeverity.HIGH: "Retrain model immediately. Use conservative trading.",
            DriftSeverity.CRITICAL: "STOP TRADING. Retrain model before resuming.",
        }

        base_rec = recommendations.get(severity, "Unknown severity")

        if drifted_features:
            top_drifted = drifted_features[:3]
            base_rec += f" Top drifted features: {', '.join(top_drifted)}"

        return base_rec

    def get_drift_summary(self, days: int = 7) -> Dict:
        """Get summary of drift detection over period."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in self.drift_history if r.timestamp > cutoff]

        if not recent:
            return {"period_days": days, "checks": 0, "message": "No drift checks in period"}

        severities = [r.severity.value for r in recent]

        return {
            "period_days": days,
            "checks": len(recent),
            "drift_detected": sum(1 for r in recent if r.drift_type != DriftType.NONE),
            "severity_distribution": {s: severities.count(s) for s in set(severities)},
            "avg_drift_score": np.mean([r.overall_score for r in recent]),
            "max_drift_score": max(r.overall_score for r in recent),
            "most_drifted_features": self._get_most_drifted_features(recent),
            "latest_recommendation": recent[-1].recommendation if recent else None,
        }

    def _get_most_drifted_features(self, reports: List[DriftReport], top_n: int = 5) -> List[str]:
        """Get features that drift most frequently."""
        from collections import Counter

        all_drifted = []
        for r in reports:
            all_drifted.extend(r.drifted_features)
        counter = Counter(all_drifted)
        return [f for f, _ in counter.most_common(top_n)]

    def needs_retraining(self) -> Tuple[bool, str]:
        """Check if model needs retraining based on recent drift."""
        if not self.drift_history:
            return False, "No drift data available"

        recent = self.drift_history[-5:]
        high_severity = sum(
            1 for r in recent if r.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        )

        if high_severity >= 2:
            return True, f"{high_severity}/5 recent checks show high/critical drift"

        avg_score = np.mean([r.overall_score for r in recent])
        if avg_score > self.config.psi_threshold:
            return True, f"Average drift score ({avg_score:.3f}) exceeds threshold"

        return False, "Drift within acceptable bounds"


def create_drift_detector(
    config: Optional[DriftConfig] = None,
    data_dir: str = "data/drift_monitoring",
) -> DriftDetector:
    """Factory function to create drift detector."""
    return DriftDetector(config=config, data_dir=data_dir)
