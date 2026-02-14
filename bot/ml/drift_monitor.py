"""
Model Performance Drift Monitor.

Tracks live model predictions, compares to backtest baseline,
detects accuracy degradation, and triggers alerts/retraining.

Architecture:
- PredictionLogger: Records predictions and outcomes
- DriftMonitor: Calculates rolling accuracy and detects drift
- AlertIntegration: Writes to WhatsApp alert queue
- DashboardExporter: Outputs JSON for monitoring UI

Thresholds:
- Warning: 5% accuracy drop from baseline
- Critical: 10% accuracy drop from baseline
- Retrain: Accuracy below random (50%)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import threading
import numpy as np
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class DriftLevel(Enum):
    """Drift severity levels."""
    NONE = "none"
    WARNING = "warning"       # 5% drop from baseline
    CRITICAL = "critical"     # 10% drop from baseline
    RETRAIN = "retrain"       # Below random (50%)


@dataclass
class DriftThresholds:
    """Configurable drift thresholds."""
    warning_drop: float = 0.05       # 5% accuracy drop triggers warning
    critical_drop: float = 0.10      # 10% accuracy drop triggers critical
    random_baseline: float = 0.50    # Below this triggers retrain
    min_predictions: int = 30        # Minimum predictions before checking
    statistical_alpha: float = 0.05  # p-value threshold for significance
    
    # Rolling windows
    window_short: int = 50           # Short-term rolling window
    window_medium: int = 100         # Medium-term rolling window
    window_long: int = 200           # Long-term rolling window
    
    # Check intervals
    check_interval_minutes: int = 15  # How often to check for drift


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Prediction:
    """A single model prediction with outcome tracking."""
    id: str
    model_name: str
    symbol: str
    timestamp: str
    prediction: int           # 1 = up, -1 = down, 0 = hold
    confidence: float
    horizon_minutes: int      # When outcome should be evaluated
    entry_price: float
    
    # Filled in later
    outcome: Optional[int] = None         # 1 = correct, 0 = incorrect
    outcome_timestamp: Optional[str] = None
    exit_price: Optional[float] = None
    actual_direction: Optional[int] = None
    
    def is_resolved(self) -> bool:
        return self.outcome is not None
    
    def is_expired(self) -> bool:
        """Check if horizon has passed."""
        pred_time = datetime.fromisoformat(self.timestamp)
        expiry = pred_time + timedelta(minutes=self.horizon_minutes)
        return datetime.now() >= expiry


@dataclass
class ModelBaseline:
    """Backtest baseline metrics for a model."""
    model_name: str
    symbol: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_trades: int
    backtest_period_days: int
    created_at: str
    updated_at: str
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Drift detection alert."""
    id: str
    model_name: str
    symbol: str
    drift_level: str
    current_accuracy: float
    baseline_accuracy: float
    accuracy_drop: float
    window_size: int
    p_value: float
    recommendation: str
    timestamp: str
    is_statistically_significant: bool


# ============================================================================
# Prediction Logger
# ============================================================================

class PredictionLogger:
    """
    Logs model predictions and tracks outcomes.
    
    Stores predictions in a time-ordered queue and resolves
    outcomes when the prediction horizon passes.
    """
    
    def __init__(
        self,
        data_dir: str = "data/ml_monitoring",
        max_history: int = 10000,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions: Dict[str, deque] = {}  # model_name -> deque of Prediction
        self.max_history = max_history
        self._lock = threading.Lock()
        self._prediction_counter = 0
        
        self._load_state()
    
    def _get_state_file(self) -> Path:
        return self.data_dir / "prediction_log.json"
    
    def _load_state(self):
        """Load saved predictions."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    
                for model_name, preds in data.get("predictions", {}).items():
                    self.predictions[model_name] = deque(maxlen=self.max_history)
                    for p in preds:
                        self.predictions[model_name].append(Prediction(**p))
                
                self._prediction_counter = data.get("counter", 0)
                logger.info(f"Loaded {sum(len(q) for q in self.predictions.values())} predictions")
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
    
    def _save_state(self):
        """Save predictions to file."""
        state_file = self._get_state_file()
        try:
            data = {
                "predictions": {
                    name: [asdict(p) for p in preds]
                    for name, preds in self.predictions.items()
                },
                "counter": self._prediction_counter,
                "updated_at": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def log_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: int,
        confidence: float,
        entry_price: float,
        horizon_minutes: int = 60,
    ) -> str:
        """Log a new prediction."""
        with self._lock:
            self._prediction_counter += 1
            pred_id = f"{model_name}_{self._prediction_counter}"
            
            pred = Prediction(
                id=pred_id,
                model_name=model_name,
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                prediction=prediction,
                confidence=confidence,
                horizon_minutes=horizon_minutes,
                entry_price=entry_price,
            )
            
            if model_name not in self.predictions:
                self.predictions[model_name] = deque(maxlen=self.max_history)
            
            self.predictions[model_name].append(pred)
            self._save_state()
            
            logger.debug(f"Logged prediction: {pred_id} - {symbol} {prediction}")
            return pred_id
    
    def resolve_prediction(
        self,
        pred_id: str,
        exit_price: float,
    ) -> bool:
        """Resolve a prediction with the actual outcome."""
        with self._lock:
            for model_name, preds in self.predictions.items():
                for pred in preds:
                    if pred.id == pred_id and not pred.is_resolved():
                        # Calculate actual direction
                        price_change = exit_price - pred.entry_price
                        if abs(price_change) < pred.entry_price * 0.0001:  # <0.01% change
                            actual_direction = 0  # Hold
                        else:
                            actual_direction = 1 if price_change > 0 else -1
                        
                        # Determine if prediction was correct
                        if pred.prediction == 0:  # Hold prediction
                            outcome = 1 if actual_direction == 0 else 0
                        else:
                            outcome = 1 if pred.prediction == actual_direction else 0
                        
                        pred.outcome = outcome
                        pred.outcome_timestamp = datetime.now().isoformat()
                        pred.exit_price = exit_price
                        pred.actual_direction = actual_direction
                        
                        self._save_state()
                        logger.debug(f"Resolved prediction: {pred_id} - {'correct' if outcome else 'incorrect'}")
                        return True
            
            return False
    
    def resolve_expired_predictions(self, get_price_func) -> int:
        """Resolve all expired predictions using current prices."""
        resolved = 0
        
        with self._lock:
            for model_name, preds in self.predictions.items():
                for pred in preds:
                    if not pred.is_resolved() and pred.is_expired():
                        try:
                            current_price = get_price_func(pred.symbol)
                            if current_price:
                                # Release lock temporarily
                                pass
                        except Exception as e:
                            logger.error(f"Error getting price for {pred.symbol}: {e}")
                            continue
        
        # Resolve outside lock
        for model_name, preds in list(self.predictions.items()):
            for pred in list(preds):
                if not pred.is_resolved() and pred.is_expired():
                    try:
                        current_price = get_price_func(pred.symbol)
                        if current_price:
                            self.resolve_prediction(pred.id, current_price)
                            resolved += 1
                    except Exception as e:
                        logger.error(f"Error resolving {pred.id}: {e}")
        
        return resolved
    
    def get_resolved_predictions(
        self,
        model_name: str,
        window: Optional[int] = None,
    ) -> List[Prediction]:
        """Get resolved predictions for accuracy calculation."""
        with self._lock:
            if model_name not in self.predictions:
                return []
            
            resolved = [p for p in self.predictions[model_name] if p.is_resolved()]
            
            if window:
                resolved = resolved[-window:]
            
            return resolved
    
    def get_all_models(self) -> List[str]:
        """Get all model names with predictions."""
        return list(self.predictions.keys())


# ============================================================================
# Drift Monitor
# ============================================================================

class DriftMonitor:
    """
    Monitors model performance and detects drift.
    
    Compares rolling accuracy against backtest baseline
    and generates alerts when thresholds are crossed.
    """
    
    def __init__(
        self,
        prediction_logger: PredictionLogger,
        thresholds: Optional[DriftThresholds] = None,
        data_dir: str = "data/ml_monitoring",
    ):
        self.logger = prediction_logger
        self.thresholds = thresholds or DriftThresholds()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines: Dict[str, ModelBaseline] = {}
        self.alerts: List[DriftAlert] = []
        self._alert_counter = 0
        self._lock = threading.Lock()
        
        self._load_baselines()
        self._load_alerts()
    
    def _load_baselines(self):
        """Load model baselines from file."""
        baseline_file = self.data_dir / "model_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
                for name, b in data.items():
                    self.baselines[name] = ModelBaseline(**b)
                logger.info(f"Loaded {len(self.baselines)} model baselines")
            except Exception as e:
                logger.error(f"Error loading baselines: {e}")
    
    def _save_baselines(self):
        """Save baselines to file."""
        baseline_file = self.data_dir / "model_baselines.json"
        try:
            data = {name: asdict(b) for name, b in self.baselines.items()}
            with open(baseline_file, "w") as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Error saving baselines: {e}")
    
    def _load_alerts(self):
        """Load alert history."""
        alert_file = self.data_dir / "drift_alerts.json"
        if alert_file.exists():
            try:
                with open(alert_file) as f:
                    data = json.load(f)
                self.alerts = [DriftAlert(**a) for a in data.get("alerts", [])]
                self._alert_counter = data.get("counter", 0)
            except Exception as e:
                logger.error(f"Error loading alerts: {e}")
    
    def _save_alerts(self):
        """Save alerts to file."""
        alert_file = self.data_dir / "drift_alerts.json"
        try:
            data = {
                "alerts": [asdict(a) for a in self.alerts[-1000:]],  # Keep last 1000
                "counter": self._alert_counter,
                "updated_at": datetime.now().isoformat(),
            }
            with open(alert_file, "w") as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def set_baseline(
        self,
        model_name: str,
        symbol: str,
        accuracy: float,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        total_trades: int = 0,
        backtest_period_days: int = 30,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """Set or update baseline metrics for a model."""
        with self._lock:
            key = f"{model_name}_{symbol}"
            now = datetime.now().isoformat()
            
            self.baselines[key] = ModelBaseline(
                model_name=model_name,
                symbol=symbol,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                total_trades=total_trades,
                backtest_period_days=backtest_period_days,
                created_at=self.baselines.get(key, {}).created_at if key in self.baselines else now,
                updated_at=now,
                extra_metrics=extra_metrics or {},
            )
            
            self._save_baselines()
            logger.info(f"Baseline set for {key}: accuracy={accuracy:.2%}")
    
    def get_baseline(self, model_name: str, symbol: str) -> Optional[ModelBaseline]:
        """Get baseline for a model."""
        key = f"{model_name}_{symbol}"
        return self.baselines.get(key)
    
    def calculate_rolling_accuracy(
        self,
        model_name: str,
        window: int,
    ) -> Tuple[Optional[float], int]:
        """Calculate rolling accuracy for last N predictions."""
        predictions = self.logger.get_resolved_predictions(model_name, window)
        
        if not predictions:
            return None, 0
        
        correct = sum(1 for p in predictions if p.outcome == 1)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, total
    
    def calculate_statistical_significance(
        self,
        baseline_accuracy: float,
        current_accuracy: float,
        sample_size: int,
    ) -> Tuple[float, bool]:
        """
        Test if accuracy drop is statistically significant.
        
        Uses a one-proportion z-test to compare current accuracy
        against baseline.
        
        Returns: (p_value, is_significant)
        """
        if sample_size < self.thresholds.min_predictions:
            return 1.0, False
        
        # One-proportion z-test
        # H0: current_accuracy >= baseline_accuracy
        # H1: current_accuracy < baseline_accuracy
        
        observed_success = int(current_accuracy * sample_size)
        expected_success = baseline_accuracy
        
        # Standard error
        se = np.sqrt(baseline_accuracy * (1 - baseline_accuracy) / sample_size)
        
        if se == 0:
            return 1.0, False
        
        # Z-statistic
        z = (current_accuracy - baseline_accuracy) / se
        
        # One-tailed p-value (testing for decrease)
        p_value = stats.norm.cdf(z)
        
        is_significant = p_value < self.thresholds.statistical_alpha
        
        return p_value, is_significant
    
    def detect_drift(
        self,
        model_name: str,
        symbol: str,
    ) -> Optional[DriftAlert]:
        """
        Detect drift for a model by comparing to baseline.
        
        Returns DriftAlert if drift detected, None otherwise.
        """
        key = f"{model_name}_{symbol}"
        baseline = self.baselines.get(key)
        
        if not baseline:
            logger.warning(f"No baseline for {key}")
            return None
        
        # Calculate accuracy at different windows
        windows = [
            self.thresholds.window_short,
            self.thresholds.window_medium,
            self.thresholds.window_long,
        ]
        
        for window in windows:
            accuracy, count = self.calculate_rolling_accuracy(model_name, window)
            
            if accuracy is None or count < self.thresholds.min_predictions:
                continue
            
            # Calculate accuracy drop
            accuracy_drop = baseline.accuracy - accuracy
            
            # Check thresholds
            drift_level = DriftLevel.NONE
            
            if accuracy < self.thresholds.random_baseline:
                drift_level = DriftLevel.RETRAIN
            elif accuracy_drop >= self.thresholds.critical_drop:
                drift_level = DriftLevel.CRITICAL
            elif accuracy_drop >= self.thresholds.warning_drop:
                drift_level = DriftLevel.WARNING
            
            if drift_level == DriftLevel.NONE:
                continue
            
            # Statistical significance
            p_value, is_significant = self.calculate_statistical_significance(
                baseline.accuracy, accuracy, count
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(drift_level, accuracy_drop, accuracy)
            
            # Create alert
            with self._lock:
                self._alert_counter += 1
                alert_id = f"drift_{self._alert_counter}"
            
            alert = DriftAlert(
                id=alert_id,
                model_name=model_name,
                symbol=symbol,
                drift_level=drift_level.value,
                current_accuracy=accuracy,
                baseline_accuracy=baseline.accuracy,
                accuracy_drop=accuracy_drop,
                window_size=window,
                p_value=p_value,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat(),
                is_statistically_significant=is_significant,
            )
            
            with self._lock:
                self.alerts.append(alert)
                self._save_alerts()
            
            logger.warning(
                f"DRIFT DETECTED: {model_name} - {drift_level.value} - "
                f"accuracy={accuracy:.2%} (baseline={baseline.accuracy:.2%})"
            )
            
            return alert
        
        return None
    
    def _generate_recommendation(
        self,
        drift_level: DriftLevel,
        accuracy_drop: float,
        current_accuracy: float,
    ) -> str:
        """Generate actionable recommendation."""
        if drift_level == DriftLevel.RETRAIN:
            return (
                f"URGENT: Model accuracy ({current_accuracy:.1%}) is below random chance. "
                "Immediately retrain the model or pause trading with this model."
            )
        elif drift_level == DriftLevel.CRITICAL:
            return (
                f"Model accuracy dropped {accuracy_drop:.1%} from baseline. "
                "Schedule immediate retraining and reduce position sizes by 50%."
            )
        elif drift_level == DriftLevel.WARNING:
            return (
                f"Model accuracy dropped {accuracy_drop:.1%} from baseline. "
                "Monitor closely and prepare for retraining in the next maintenance window."
            )
        return "No action required."
    
    def check_all_models(self) -> List[DriftAlert]:
        """Check all models for drift."""
        alerts = []
        
        for key, baseline in self.baselines.items():
            alert = self.detect_drift(baseline.model_name, baseline.symbol)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def get_model_status(self, model_name: str, symbol: str) -> Dict[str, Any]:
        """Get comprehensive status for a model."""
        key = f"{model_name}_{symbol}"
        baseline = self.baselines.get(key)
        
        # Calculate accuracies
        acc_short, count_short = self.calculate_rolling_accuracy(
            model_name, self.thresholds.window_short
        )
        acc_medium, count_medium = self.calculate_rolling_accuracy(
            model_name, self.thresholds.window_medium
        )
        acc_long, count_long = self.calculate_rolling_accuracy(
            model_name, self.thresholds.window_long
        )
        
        # Get recent alerts
        recent_alerts = [
            a for a in self.alerts
            if a.model_name == model_name and a.symbol == symbol
        ][-5:]
        
        status = {
            "model_name": model_name,
            "symbol": symbol,
            "baseline": asdict(baseline) if baseline else None,
            "rolling_accuracy": {
                f"last_{self.thresholds.window_short}": {
                    "accuracy": acc_short,
                    "count": count_short,
                },
                f"last_{self.thresholds.window_medium}": {
                    "accuracy": acc_medium,
                    "count": count_medium,
                },
                f"last_{self.thresholds.window_long}": {
                    "accuracy": acc_long,
                    "count": count_long,
                },
            },
            "recent_alerts": [asdict(a) for a in recent_alerts],
            "updated_at": datetime.now().isoformat(),
        }
        
        # Determine health status
        if baseline and acc_short is not None:
            drop = baseline.accuracy - acc_short
            if acc_short < self.thresholds.random_baseline:
                status["health"] = "critical"
            elif drop >= self.thresholds.critical_drop:
                status["health"] = "critical"
            elif drop >= self.thresholds.warning_drop:
                status["health"] = "warning"
            else:
                status["health"] = "healthy"
        else:
            status["health"] = "unknown"
        
        return status


# ============================================================================
# WhatsApp Alert Integration
# ============================================================================

class DriftAlertIntegration:
    """Integrates drift alerts with the WhatsApp alert system."""
    
    def __init__(
        self,
        alert_file: str = "data/trade_alerts.json",
    ):
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def send_drift_alert(self, alert: DriftAlert) -> bool:
        """Write drift alert to the WhatsApp alert queue."""
        try:
            # Format message for WhatsApp
            level_emoji = {
                "warning": "⚠️",
                "critical": "🚨",
                "retrain": "🛑",
            }.get(alert.drift_level, "📊")
            
            significance = "✓ statistically significant" if alert.is_statistically_significant else "○ not significant"
            
            message = f"""{level_emoji} *MODEL DRIFT ALERT*

*Model:* {alert.model_name}
*Symbol:* {alert.symbol}
*Level:* {alert.drift_level.upper()}

📉 *Accuracy Dropped:*
• Current: {alert.current_accuracy:.1%}
• Baseline: {alert.baseline_accuracy:.1%}
• Drop: {alert.accuracy_drop:.1%}

📊 *Analysis:*
• Window: last {alert.window_size} predictions
• p-value: {alert.p_value:.4f}
• {significance}

💡 *Recommendation:*
{alert.recommendation}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
            
            # Create alert entry
            alert_entry = {
                "id": f"drift_{alert.id}",
                "type": "model_drift",
                "priority": "urgent" if alert.drift_level == "retrain" else "high",
                "message": message,
                "created_at": datetime.now().isoformat(),
                "delivered": False,
                "data": asdict(alert),
            }
            
            # Append to alert file
            with self._lock:
                alerts = []
                if self.alert_file.exists():
                    try:
                        with open(self.alert_file) as f:
                            alerts = json.load(f)
                    except:
                        alerts = []
                
                alerts.append(alert_entry)
                
                # Keep last 100
                alerts = alerts[-100:]
                
                with open(self.alert_file, "w") as f:
                    json.dump(alerts, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Drift alert queued for WhatsApp: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}")
            return False


# ============================================================================
# Dashboard Exporter
# ============================================================================

class DashboardExporter:
    """Exports monitoring data as JSON for dashboard consumption."""
    
    def __init__(
        self,
        drift_monitor: DriftMonitor,
        output_dir: str = "data/ml_monitoring/dashboard",
    ):
        self.monitor = drift_monitor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model_status(self, model_name: str, symbol: str):
        """Export status for a single model."""
        status = self.monitor.get_model_status(model_name, symbol)
        
        output_file = self.output_dir / f"{model_name}_{symbol}_status.json"
        with open(output_file, "w") as f:
            json.dump(status, f, indent=2, cls=NumpyEncoder)
        
        return status
    
    def export_all_models(self) -> Dict[str, Any]:
        """Export status for all tracked models."""
        all_status = {}
        
        for key, baseline in self.monitor.baselines.items():
            status = self.monitor.get_model_status(baseline.model_name, baseline.symbol)
            all_status[key] = status
        
        # Summary
        summary = {
            "total_models": len(all_status),
            "healthy": sum(1 for s in all_status.values() if s.get("health") == "healthy"),
            "warning": sum(1 for s in all_status.values() if s.get("health") == "warning"),
            "critical": sum(1 for s in all_status.values() if s.get("health") == "critical"),
            "unknown": sum(1 for s in all_status.values() if s.get("health") == "unknown"),
            "updated_at": datetime.now().isoformat(),
        }
        
        dashboard_data = {
            "summary": summary,
            "models": all_status,
            "thresholds": asdict(self.monitor.thresholds),
        }
        
        # Save to file
        output_file = self.output_dir / "dashboard.json"
        with open(output_file, "w") as f:
            json.dump(dashboard_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Dashboard exported: {len(all_status)} models")
        return dashboard_data
    
    def export_accuracy_trends(self, days: int = 7) -> Dict[str, Any]:
        """Export accuracy trends over time."""
        trends = {}
        cutoff = datetime.now() - timedelta(days=days)
        
        for key, baseline in self.monitor.baselines.items():
            model_name = baseline.model_name
            
            # Get all resolved predictions
            predictions = self.monitor.logger.get_resolved_predictions(model_name)
            
            # Filter to time range
            recent = [
                p for p in predictions
                if datetime.fromisoformat(p.timestamp) >= cutoff
            ]
            
            if not recent:
                continue
            
            # Group by day
            daily_accuracy = {}
            for pred in recent:
                day = pred.timestamp[:10]  # YYYY-MM-DD
                if day not in daily_accuracy:
                    daily_accuracy[day] = {"correct": 0, "total": 0}
                daily_accuracy[day]["total"] += 1
                if pred.outcome == 1:
                    daily_accuracy[day]["correct"] += 1
            
            # Calculate accuracy per day
            trend = []
            for day, counts in sorted(daily_accuracy.items()):
                acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
                trend.append({
                    "date": day,
                    "accuracy": round(acc, 4),
                    "predictions": counts["total"],
                })
            
            trends[key] = {
                "model_name": model_name,
                "symbol": baseline.symbol,
                "baseline_accuracy": baseline.accuracy,
                "trend": trend,
            }
        
        # Save
        output_file = self.output_dir / "accuracy_trends.json"
        with open(output_file, "w") as f:
            json.dump({
                "period_days": days,
                "trends": trends,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2, cls=NumpyEncoder)
        
        return trends


# ============================================================================
# Main Monitor Controller
# ============================================================================

class ModelDriftMonitoringSystem:
    """
    Main controller for the model drift monitoring system.
    
    Usage:
        system = ModelDriftMonitoringSystem()
        
        # Set baseline from backtest results
        system.set_baseline("lstm_btc", "BTC/USDT", accuracy=0.62, ...)
        
        # Log predictions as they happen
        pred_id = system.log_prediction("lstm_btc", "BTC/USDT", 1, 0.75, 42000.0)
        
        # Later, resolve with actual outcome
        system.resolve_prediction(pred_id, 43000.0)
        
        # Check for drift (can be scheduled)
        alerts = system.check_drift()
    """
    
    def __init__(
        self,
        data_dir: str = "data/ml_monitoring",
        thresholds: Optional[DriftThresholds] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = PredictionLogger(data_dir)
        self.monitor = DriftMonitor(self.logger, thresholds, data_dir)
        self.alert_integration = DriftAlertIntegration()
        self.dashboard = DashboardExporter(self.monitor, f"{data_dir}/dashboard")
        
        logger.info("Model Drift Monitoring System initialized")
    
    def set_baseline(
        self,
        model_name: str,
        symbol: str,
        accuracy: float,
        **kwargs,
    ):
        """Set baseline metrics for a model."""
        self.monitor.set_baseline(model_name, symbol, accuracy, **kwargs)
    
    def log_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: int,
        confidence: float,
        entry_price: float,
        horizon_minutes: int = 60,
    ) -> str:
        """Log a prediction."""
        return self.logger.log_prediction(
            model_name, symbol, prediction, confidence, entry_price, horizon_minutes
        )
    
    def resolve_prediction(self, pred_id: str, exit_price: float) -> bool:
        """Resolve a prediction with actual outcome."""
        return self.logger.resolve_prediction(pred_id, exit_price)
    
    def check_drift(self, send_alerts: bool = True) -> List[DriftAlert]:
        """
        Check all models for drift and optionally send alerts.
        
        Returns list of drift alerts.
        """
        alerts = self.monitor.check_all_models()
        
        if send_alerts:
            for alert in alerts:
                self.alert_integration.send_drift_alert(alert)
        
        return alerts
    
    def check_drift_for_model(
        self,
        model_name: str,
        symbol: str,
        send_alert: bool = True,
    ) -> Optional[DriftAlert]:
        """Check drift for a specific model."""
        alert = self.monitor.detect_drift(model_name, symbol)
        
        if alert and send_alert:
            self.alert_integration.send_drift_alert(alert)
        
        return alert
    
    def get_model_status(self, model_name: str, symbol: str) -> Dict[str, Any]:
        """Get status for a model."""
        return self.monitor.get_model_status(model_name, symbol)
    
    def export_dashboard(self) -> Dict[str, Any]:
        """Export dashboard data."""
        return self.dashboard.export_all_models()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of monitoring system status."""
        models = list(self.monitor.baselines.keys())
        
        # Count recent alerts
        recent_alerts = [
            a for a in self.monitor.alerts
            if datetime.fromisoformat(a.timestamp) > datetime.now() - timedelta(days=1)
        ]
        
        return {
            "tracked_models": len(models),
            "total_predictions_logged": sum(
                len(preds) for preds in self.logger.predictions.values()
            ),
            "alerts_last_24h": len(recent_alerts),
            "models_needing_attention": sum(
                1 for m in models
                if any(
                    a.model_name == self.monitor.baselines[m].model_name
                    and a.drift_level in ["critical", "retrain"]
                    for a in recent_alerts
                )
            ),
            "thresholds": {
                "warning": f"{self.monitor.thresholds.warning_drop:.0%} drop",
                "critical": f"{self.monitor.thresholds.critical_drop:.0%} drop",
                "retrain": f"below {self.monitor.thresholds.random_baseline:.0%}",
            },
            "updated_at": datetime.now().isoformat(),
        }


# ============================================================================
# Factory and Convenience Functions
# ============================================================================

_monitoring_system: Optional[ModelDriftMonitoringSystem] = None


def get_monitoring_system() -> ModelDriftMonitoringSystem:
    """Get or create the global monitoring system."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = ModelDriftMonitoringSystem()
    return _monitoring_system


def set_model_baseline(
    model_name: str,
    symbol: str,
    accuracy: float,
    **kwargs,
):
    """Set baseline for a model."""
    return get_monitoring_system().set_baseline(model_name, symbol, accuracy, **kwargs)


def log_model_prediction(
    model_name: str,
    symbol: str,
    prediction: int,
    confidence: float,
    entry_price: float,
    horizon_minutes: int = 60,
) -> str:
    """Log a prediction."""
    return get_monitoring_system().log_prediction(
        model_name, symbol, prediction, confidence, entry_price, horizon_minutes
    )


def resolve_model_prediction(pred_id: str, exit_price: float) -> bool:
    """Resolve a prediction."""
    return get_monitoring_system().resolve_prediction(pred_id, exit_price)


def check_model_drift() -> List[DriftAlert]:
    """Check all models for drift."""
    return get_monitoring_system().check_drift()


# ============================================================================
# Testing
# ============================================================================

def test_drift_monitor():
    """Test the drift monitoring system."""
    import random
    
    print("=" * 60)
    print("Model Drift Monitoring System - Test")
    print("=" * 60)
    
    # Create system
    system = ModelDriftMonitoringSystem(data_dir="data/test_monitoring")
    
    # Set baseline
    print("\n1. Setting baseline...")
    system.set_baseline(
        model_name="test_lstm",
        symbol="BTC/USDT",
        accuracy=0.65,
        precision=0.63,
        recall=0.67,
        f1_score=0.65,
        total_trades=500,
        backtest_period_days=30,
    )
    print("   ✓ Baseline set: 65% accuracy")
    
    # Simulate predictions
    print("\n2. Simulating predictions...")
    pred_ids = []
    
    # First batch: good accuracy (matching baseline)
    for i in range(30):
        pred_id = system.log_prediction(
            model_name="test_lstm",
            symbol="BTC/USDT",
            prediction=random.choice([1, -1]),
            confidence=random.uniform(0.6, 0.8),
            entry_price=42000.0,
            horizon_minutes=60,
        )
        pred_ids.append(pred_id)
    print(f"   ✓ Logged {len(pred_ids)} predictions")
    
    # Resolve with ~65% accuracy (matching baseline)
    print("\n3. Resolving predictions (good accuracy)...")
    for i, pred_id in enumerate(pred_ids):
        # 65% chance of correct prediction
        if random.random() < 0.65:
            exit_price = 43000.0  # Price went up (matches UP prediction)
        else:
            exit_price = 41000.0  # Price went down
        system.resolve_prediction(pred_id, exit_price)
    print("   ✓ Resolved with ~65% accuracy")
    
    # Check drift (should be none)
    print("\n4. Checking for drift...")
    alerts = system.check_drift(send_alerts=False)
    if alerts:
        print(f"   ⚠️ Drift detected: {alerts[0].drift_level}")
    else:
        print("   ✓ No drift detected (as expected)")
    
    # Simulate degradation
    print("\n5. Simulating accuracy degradation...")
    degraded_ids = []
    for i in range(40):
        pred_id = system.log_prediction(
            model_name="test_lstm",
            symbol="BTC/USDT",
            prediction=1,  # Always predict UP
            confidence=random.uniform(0.5, 0.7),
            entry_price=42000.0,
            horizon_minutes=60,
        )
        degraded_ids.append(pred_id)
    
    # Resolve with only 45% accuracy (below random!)
    for pred_id in degraded_ids:
        if random.random() < 0.45:
            exit_price = 43000.0
        else:
            exit_price = 41000.0
        system.resolve_prediction(pred_id, exit_price)
    print("   ✓ Resolved with ~45% accuracy (degraded)")
    
    # Check drift again
    print("\n6. Checking for drift after degradation...")
    alerts = system.check_drift(send_alerts=False)
    if alerts:
        alert = alerts[0]
        print(f"   🚨 DRIFT DETECTED!")
        print(f"      Level: {alert.drift_level}")
        print(f"      Current accuracy: {alert.current_accuracy:.1%}")
        print(f"      Baseline: {alert.baseline_accuracy:.1%}")
        print(f"      Drop: {alert.accuracy_drop:.1%}")
        print(f"      Recommendation: {alert.recommendation}")
    else:
        print("   ✓ No drift detected")
    
    # Get status
    print("\n7. Model status:")
    status = system.get_model_status("test_lstm", "BTC/USDT")
    print(f"   Health: {status['health']}")
    print(f"   Rolling accuracy (50): {status['rolling_accuracy']['last_50']['accuracy']:.1%}")
    
    # Export dashboard
    print("\n8. Exporting dashboard...")
    dashboard = system.export_dashboard()
    print(f"   ✓ Dashboard exported with {dashboard['summary']['total_models']} models")
    
    # Summary
    print("\n9. System summary:")
    summary = system.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_drift_monitor()
