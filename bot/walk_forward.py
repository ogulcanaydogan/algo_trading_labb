"""
Walk-Forward Validation Module for Deep Learning Models.

Provides robust validation that prevents overfitting by:
- Splitting data into training and test periods using time-series aware splits
- Rolling forward through time with configurable windows
- Aggregating out-of-sample results for realistic performance estimates
- Supporting both LSTM and Transformer models

This module complements the existing strategy optimization by providing
specialized validation for deep learning models.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        train_window_days: Number of days for each training window (e.g., 180)
        test_window_days: Number of days for each test window (e.g., 30)
        step_days: Number of days to step forward between windows (e.g., 30)
        min_train_samples: Minimum number of samples required for training
        validation_split: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
    """
    train_window_days: int = 180
    test_window_days: int = 30
    step_days: int = 30
    min_train_samples: int = 500
    validation_split: float = 0.15
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "train_window_days": self.train_window_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
            "min_train_samples": self.min_train_samples,
            "validation_split": self.validation_split,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalkForwardConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WindowMetrics:
    """Metrics for a single validation window.

    Attributes:
        accuracy: Classification accuracy
        precision: Weighted precision
        recall: Weighted recall
        f1_score: Weighted F1 score
        sharpe_ratio: Sharpe ratio based on predictions
        profit_factor: Ratio of winning to losing predictions
        win_rate: Percentage of correct directional predictions
        total_predictions: Number of predictions made
        long_predictions: Number of LONG predictions
        short_predictions: Number of SHORT predictions
        flat_predictions: Number of FLAT predictions
    """
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_predictions: int = 0
    long_predictions: int = 0
    short_predictions: int = 0
    flat_predictions: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "win_rate": round(self.win_rate, 4),
            "total_predictions": self.total_predictions,
            "long_predictions": self.long_predictions,
            "short_predictions": self.short_predictions,
            "flat_predictions": self.flat_predictions,
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass
class WindowResult:
    """Result from a single walk-forward window.

    Attributes:
        window_id: Unique identifier for this window
        train_start: Start date of training period
        train_end: End date of training period
        test_start: Start date of test period
        test_end: End date of test period
        train_samples: Number of training samples
        test_samples: Number of test samples
        train_metrics: Metrics from training
        test_metrics: Metrics from testing
        training_time_seconds: Time taken to train the model
        predictions: Array of predictions made
        actuals: Array of actual values
        probabilities: Array of prediction probabilities
    """
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    train_metrics: WindowMetrics
    test_metrics: WindowMetrics
    training_time_seconds: float = 0.0
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_profitable(self) -> bool:
        """Check if this window was profitable."""
        return self.test_metrics.profit_factor > 1.0 or self.test_metrics.sharpe_ratio > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat() if isinstance(self.train_start, datetime) else str(self.train_start),
            "train_end": self.train_end.isoformat() if isinstance(self.train_end, datetime) else str(self.train_end),
            "test_start": self.test_start.isoformat() if isinstance(self.test_start, datetime) else str(self.test_start),
            "test_end": self.test_end.isoformat() if isinstance(self.test_end, datetime) else str(self.test_end),
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_metrics": self.train_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "training_time_seconds": round(self.training_time_seconds, 2),
            "is_profitable": self.is_profitable,
        }


@dataclass
class WalkForwardResults:
    """Complete walk-forward validation results.

    Attributes:
        windows: List of individual window results
        aggregate_metrics: Aggregated metrics across all windows
        is_robust: True if performance is consistent across windows
        config: Configuration used for validation
        model_type: Type of model validated (lstm/transformer)
        symbol: Trading symbol used
        total_training_time: Total time spent training
        start_date: Start date of entire validation period
        end_date: End date of entire validation period
    """
    windows: List[WindowResult]
    aggregate_metrics: Dict[str, float]
    is_robust: bool
    config: WalkForwardConfig
    model_type: str
    symbol: str
    total_training_time: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    robustness_score: float = 0.0
    consistency_score: float = 0.0
    overfitting_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "summary": {
                "model_type": self.model_type,
                "symbol": self.symbol,
                "total_windows": len(self.windows),
                "profitable_windows": sum(1 for w in self.windows if w.is_profitable),
                "is_robust": self.is_robust,
                "robustness_score": round(self.robustness_score, 4),
                "consistency_score": round(self.consistency_score, 4),
                "overfitting_score": round(self.overfitting_score, 4),
                "total_training_time": round(self.total_training_time, 2),
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
            },
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "config": self.config.to_dict(),
            "windows": [w.to_dict() for w in self.windows],
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the results."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 70)
        print(f"Model Type: {self.model_type.upper()}")
        print(f"Symbol: {self.symbol}")
        print(f"Validation Period: {self.start_date} to {self.end_date}")
        print(f"Total Training Time: {self.total_training_time:.1f}s")
        print("-" * 70)
        print(f"Total Windows: {len(self.windows)}")
        print(f"Profitable Windows: {sum(1 for w in self.windows if w.is_profitable)} ({sum(1 for w in self.windows if w.is_profitable) / len(self.windows) * 100:.1f}%)")
        print("-" * 70)
        print("Aggregate Metrics (Out-of-Sample):")
        for metric, value in self.aggregate_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 70)
        print(f"Robustness Score: {self.robustness_score:.2f}/1.00")
        print(f"Consistency Score: {self.consistency_score:.2f}/1.00")
        print(f"Overfitting Score: {self.overfitting_score:.2f}/1.00 (lower is better)")
        print(f"\nOverall Assessment: {'ROBUST' if self.is_robust else 'NOT ROBUST'}")
        print("=" * 70)

    def save(self, path: Union[str, Path]) -> Path:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Walk-forward results saved to {path}")
        return path


# =============================================================================
# Walk-Forward Validator Class
# =============================================================================


class WalkForwardValidator:
    """
    Walk-Forward Validation Engine for Deep Learning Models.

    Implements proper time-series cross-validation that:
    - Respects temporal ordering (no future data leakage)
    - Uses rolling windows for training and testing
    - Supports both LSTM and Transformer architectures
    - Calculates comprehensive metrics per window
    - Aggregates results for robust performance estimates

    Usage:
        validator = WalkForwardValidator(config)
        results = validator.run_validation(
            model_class=LSTMModel,
            data=ohlcv_dataframe,
            model_config=LSTMConfig()
        )
        results.print_summary()
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        results_dir: str = "data/walk_forward_results",
    ):
        """
        Initialize the walk-forward validator.

        Args:
            config: Walk-forward configuration
            results_dir: Directory to save validation results
        """
        self.config = config or WalkForwardConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Import feature engineer
        try:
            from .ml.feature_engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
        except ImportError:
            logger.warning("FeatureEngineer not available, using raw features")
            self.feature_engineer = None

    def generate_splits(
        self,
        data: pd.DataFrame,
        timeframe: str = "1h",
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate train/test splits for walk-forward validation.

        Args:
            data: DataFrame with DatetimeIndex containing OHLCV data
            timeframe: Timeframe of the data (e.g., '1h', '4h', '1d')

        Yields:
            Tuple of (train_df, test_df) for each validation window
        """
        # Ensure data is sorted by time
        data = data.sort_index()

        # Calculate bars per day based on timeframe
        bars_per_day = self._get_bars_per_day(timeframe)

        # Calculate window sizes in bars
        train_bars = int(self.config.train_window_days * bars_per_day)
        test_bars = int(self.config.test_window_days * bars_per_day)
        step_bars = int(self.config.step_days * bars_per_day)

        total_bars = len(data)
        min_required = train_bars + test_bars

        if total_bars < min_required:
            logger.warning(
                f"Insufficient data: {total_bars} bars < {min_required} required "
                f"(train={train_bars}, test={test_bars})"
            )
            return

        # Generate windows
        start_idx = 0
        window_id = 0

        while start_idx + min_required <= total_bars:
            train_end_idx = start_idx + train_bars
            test_end_idx = train_end_idx + test_bars

            # Ensure we don't exceed data bounds
            if test_end_idx > total_bars:
                break

            train_df = data.iloc[start_idx:train_end_idx].copy()
            test_df = data.iloc[train_end_idx:test_end_idx].copy()

            # Validate minimum samples
            if len(train_df) >= self.config.min_train_samples:
                logger.info(
                    f"Window {window_id + 1}: "
                    f"Train {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars), "
                    f"Test {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)"
                )
                yield train_df, test_df
                window_id += 1
            else:
                logger.warning(
                    f"Skipping window {window_id + 1}: insufficient samples "
                    f"({len(train_df)} < {self.config.min_train_samples})"
                )

            # Step forward
            start_idx += step_bars

    def run_validation(
        self,
        model_class: Type,
        data: pd.DataFrame,
        model_config: Optional[Any] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        verbose: bool = True,
    ) -> WalkForwardResults:
        """
        Run complete walk-forward validation.

        Args:
            model_class: Model class to instantiate (LSTMModel or TransformerModel)
            data: OHLCV DataFrame with DatetimeIndex
            model_config: Configuration for the model
            symbol: Trading symbol for identification
            timeframe: Data timeframe
            verbose: Whether to print progress

        Returns:
            WalkForwardResults with all validation results
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"WALK-FORWARD VALIDATION: {symbol}")
            print(f"Model: {model_class.__name__}")
            print(f"Config: train={self.config.train_window_days}d, "
                  f"test={self.config.test_window_days}d, "
                  f"step={self.config.step_days}d")
            print(f"{'=' * 70}\n")

        window_results: List[WindowResult] = []
        total_training_time = 0.0

        # Generate and process each window
        for window_id, (train_df, test_df) in enumerate(self.generate_splits(data, timeframe)):
            if verbose:
                print(f"\n--- Window {window_id + 1} ---")

            try:
                result = self._process_window(
                    window_id=window_id,
                    train_df=train_df,
                    test_df=test_df,
                    model_class=model_class,
                    model_config=model_config,
                    verbose=verbose,
                )
                window_results.append(result)
                total_training_time += result.training_time_seconds

                if verbose:
                    print(f"  Train Accuracy: {result.train_metrics.accuracy:.4f}")
                    print(f"  Test Accuracy: {result.test_metrics.accuracy:.4f}")
                    print(f"  Test Sharpe: {result.test_metrics.sharpe_ratio:.4f}")

            except Exception as e:
                logger.error(f"Error processing window {window_id + 1}: {e}")
                if verbose:
                    print(f"  ERROR: {e}")
                continue

        if not window_results:
            raise ValueError("No valid windows could be processed")

        # Aggregate results
        aggregate_metrics = self.aggregate_results(window_results)

        # Calculate robustness scores
        robustness_score = self._calculate_robustness_score(window_results)
        consistency_score = self._calculate_consistency_score(window_results)
        overfitting_score = self._calculate_overfitting_score(window_results)

        # Determine if model is robust
        is_robust = (
            robustness_score >= 0.6 and
            consistency_score >= 0.5 and
            overfitting_score <= 0.4
        )

        # Get model type
        model_type = model_class.__name__.lower().replace("model", "")

        results = WalkForwardResults(
            windows=window_results,
            aggregate_metrics=aggregate_metrics,
            is_robust=is_robust,
            config=self.config,
            model_type=model_type,
            symbol=symbol,
            total_training_time=total_training_time,
            start_date=window_results[0].train_start if window_results else None,
            end_date=window_results[-1].test_end if window_results else None,
            robustness_score=robustness_score,
            consistency_score=consistency_score,
            overfitting_score=overfitting_score,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"{symbol.replace('/', '_')}_{model_type}_{timestamp}.json"
        results.save(results_path)

        return results

    def _process_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_class: Type,
        model_config: Optional[Any],
        verbose: bool = True,
    ) -> WindowResult:
        """Process a single validation window."""
        start_time = time.time()

        # Extract features
        if self.feature_engineer is not None:
            train_features = self.feature_engineer.extract_features(train_df)
            test_features = self.feature_engineer.extract_features(test_df)
        else:
            train_features = train_df.copy()
            test_features = test_df.copy()

        # Prepare training data
        feature_cols = [col for col in train_features.columns
                       if col not in ["target_return", "target_direction", "target_class",
                                     "open", "high", "low", "close", "volume"]]

        X_train = train_features[feature_cols].values
        y_train = train_features["target_class"].values if "target_class" in train_features else np.zeros(len(train_features))

        X_test = test_features[feature_cols].values
        y_test = test_features["target_class"].values if "target_class" in test_features else np.zeros(len(test_features))

        # Get returns for Sharpe calculation
        train_returns = train_features["target_return"].values if "target_return" in train_features else np.zeros(len(train_features))
        test_returns = test_features["target_return"].values if "target_return" in test_features else np.zeros(len(test_features))

        # Split training data for validation
        val_size = int(len(X_train) * self.config.validation_split)
        X_train_fit = X_train[:-val_size]
        y_train_fit = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        # Initialize and train model
        model = model_class(config=model_config) if model_config else model_class()

        try:
            training_metrics = model.train(
                X=X_train_fit,
                y=y_train_fit,
                validation_data=(X_val, y_val),
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        training_time = time.time() - start_time

        # Get predictions
        try:
            train_proba = model.predict_proba(X_train)
            test_proba = model.predict_proba(X_test)

            train_preds = np.argmax(train_proba, axis=1)
            test_preds = np.argmax(test_proba, axis=1)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        # Calculate metrics
        train_metrics = self.calculate_metrics(
            predictions=train_preds,
            actuals=y_train,
            probabilities=train_proba,
            returns=train_returns,
        )

        test_metrics = self.calculate_metrics(
            predictions=test_preds,
            actuals=y_test,
            probabilities=test_proba,
            returns=test_returns,
        )

        return WindowResult(
            window_id=window_id,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            training_time_seconds=training_time,
            predictions=test_preds,
            actuals=y_test,
            probabilities=test_proba,
        )

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ) -> WindowMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            predictions: Predicted class labels (0=SHORT, 1=FLAT, 2=LONG)
            actuals: Actual class labels
            probabilities: Prediction probabilities (samples, 3)
            returns: Actual returns for Sharpe calculation

        Returns:
            WindowMetrics with all calculated metrics
        """
        # Basic classification metrics
        accuracy = np.mean(predictions == actuals) if len(actuals) > 0 else 0.0

        # Calculate precision, recall, f1 for each class
        precision_scores = []
        recall_scores = []

        for cls in [0, 1, 2]:
            pred_cls = predictions == cls
            actual_cls = actuals == cls

            tp = np.sum(pred_cls & actual_cls)
            fp = np.sum(pred_cls & ~actual_cls)
            fn = np.sum(~pred_cls & actual_cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_scores.append(prec)
            recall_scores.append(rec)

        # Weighted averages
        class_counts = [np.sum(actuals == cls) for cls in [0, 1, 2]]
        total = sum(class_counts)
        weights = [c / total if total > 0 else 0 for c in class_counts]

        precision = sum(p * w for p, w in zip(precision_scores, weights))
        recall = sum(r * w for r, w in zip(recall_scores, weights))
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Trading metrics
        sharpe_ratio = 0.0
        profit_factor = 0.0
        win_rate = 0.0

        if returns is not None and len(returns) > 0:
            # Calculate strategy returns based on predictions
            # LONG (2) = +1, FLAT (1) = 0, SHORT (0) = -1
            position_multipliers = np.where(predictions == 2, 1, np.where(predictions == 0, -1, 0))
            strategy_returns = position_multipliers * returns

            # Remove NaN values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

            if len(strategy_returns) > 1:
                # Sharpe ratio (annualized assuming hourly data)
                mean_return = np.mean(strategy_returns)
                std_return = np.std(strategy_returns)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 24)  # Annualized

                # Profit factor
                wins = strategy_returns[strategy_returns > 0]
                losses = strategy_returns[strategy_returns < 0]
                if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
                    profit_factor = np.sum(wins) / np.sum(np.abs(losses))
                elif len(wins) > 0:
                    profit_factor = float("inf")

                # Win rate
                non_zero_trades = strategy_returns[strategy_returns != 0]
                if len(non_zero_trades) > 0:
                    win_rate = np.sum(non_zero_trades > 0) / len(non_zero_trades)

        # Prediction distribution
        long_predictions = np.sum(predictions == 2)
        short_predictions = np.sum(predictions == 0)
        flat_predictions = np.sum(predictions == 1)

        # Average confidence
        avg_confidence = 0.0
        if probabilities is not None and len(probabilities) > 0:
            max_probs = np.max(probabilities, axis=1)
            avg_confidence = np.mean(max_probs)

        return WindowMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sharpe_ratio=sharpe_ratio,
            profit_factor=min(profit_factor, 10.0),  # Cap at 10
            win_rate=win_rate,
            total_predictions=len(predictions),
            long_predictions=int(long_predictions),
            short_predictions=int(short_predictions),
            flat_predictions=int(flat_predictions),
            avg_confidence=avg_confidence,
        )

    def aggregate_results(self, results: List[WindowResult]) -> Dict[str, float]:
        """
        Aggregate metrics across all validation windows.

        Args:
            results: List of WindowResult objects

        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {}

        # Collect test metrics from all windows
        metrics_lists = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "sharpe_ratio": [],
            "profit_factor": [],
            "win_rate": [],
            "avg_confidence": [],
        }

        for result in results:
            metrics = result.test_metrics
            metrics_lists["accuracy"].append(metrics.accuracy)
            metrics_lists["precision"].append(metrics.precision)
            metrics_lists["recall"].append(metrics.recall)
            metrics_lists["f1_score"].append(metrics.f1_score)
            metrics_lists["sharpe_ratio"].append(metrics.sharpe_ratio)
            metrics_lists["profit_factor"].append(min(metrics.profit_factor, 10.0))
            metrics_lists["win_rate"].append(metrics.win_rate)
            metrics_lists["avg_confidence"].append(metrics.avg_confidence)

        # Calculate aggregates
        aggregated = {}
        for metric_name, values in metrics_lists.items():
            arr = np.array(values)
            aggregated[f"mean_{metric_name}"] = np.mean(arr)
            aggregated[f"std_{metric_name}"] = np.std(arr)
            aggregated[f"min_{metric_name}"] = np.min(arr)
            aggregated[f"max_{metric_name}"] = np.max(arr)

        # Add summary statistics
        aggregated["total_windows"] = len(results)
        aggregated["profitable_windows"] = sum(1 for r in results if r.is_profitable)
        aggregated["profitable_ratio"] = aggregated["profitable_windows"] / len(results)

        return aggregated

    def _calculate_robustness_score(self, results: List[WindowResult]) -> float:
        """
        Calculate robustness score (0-1).

        Based on:
        - Consistency of profitability across windows
        - Stability of accuracy metrics
        """
        if not results:
            return 0.0

        # Profitability consistency
        profitable_ratio = sum(1 for r in results if r.is_profitable) / len(results)

        # Accuracy stability
        accuracies = [r.test_metrics.accuracy for r in results]
        if len(accuracies) > 1 and np.mean(accuracies) > 0:
            cv = np.std(accuracies) / np.mean(accuracies)
            stability = 1 / (1 + cv)
        else:
            stability = 0.5

        # Combined score
        return 0.6 * profitable_ratio + 0.4 * stability

    def _calculate_consistency_score(self, results: List[WindowResult]) -> float:
        """
        Calculate consistency score (0-1).

        Based on variance of performance metrics across windows.
        """
        if not results or len(results) < 2:
            return 0.5

        sharpes = [r.test_metrics.sharpe_ratio for r in results]
        accuracies = [r.test_metrics.accuracy for r in results]

        # Lower variance = higher consistency
        sharpe_std = np.std(sharpes)
        acc_std = np.std(accuracies)

        sharpe_consistency = 1 / (1 + sharpe_std)
        acc_consistency = 1 / (1 + acc_std * 10)  # Scale accuracy std

        return 0.5 * sharpe_consistency + 0.5 * acc_consistency

    def _calculate_overfitting_score(self, results: List[WindowResult]) -> float:
        """
        Calculate overfitting score (0-1, lower is better).

        Based on gap between training and test performance.
        """
        if not results:
            return 1.0

        gaps = []
        for r in results:
            train_acc = r.train_metrics.accuracy
            test_acc = r.test_metrics.accuracy

            if train_acc > 0:
                gap = (train_acc - test_acc) / train_acc
                gaps.append(max(0, gap))  # Only consider positive gaps

        if not gaps:
            return 0.5

        return min(1.0, np.mean(gaps))

    @staticmethod
    def _get_bars_per_day(timeframe: str) -> float:
        """Get number of bars per day for a given timeframe."""
        tf = timeframe.strip().lower()

        if tf.endswith("m"):
            minutes = int(tf[:-1])
            return (24 * 60) / minutes
        elif tf.endswith("h"):
            hours = int(tf[:-1])
            return 24 / hours
        elif tf.endswith("d"):
            days = int(tf[:-1])
            return 1 / days
        else:
            return 24  # Default to hourly


# =============================================================================
# Convenience Functions
# =============================================================================


def run_walk_forward_validation(
    ohlcv: pd.DataFrame,
    model_type: Literal["lstm", "transformer"] = "lstm",
    symbol: str = "BTC/USDT",
    config: Optional[WalkForwardConfig] = None,
    timeframe: str = "1h",
    verbose: bool = True,
) -> WalkForwardResults:
    """
    Convenience function to run walk-forward validation.

    Args:
        ohlcv: OHLCV DataFrame with DatetimeIndex
        model_type: Type of model to validate ('lstm' or 'transformer')
        symbol: Trading symbol
        config: Optional walk-forward configuration
        timeframe: Data timeframe
        verbose: Whether to print progress

    Returns:
        WalkForwardResults
    """
    # Import model classes
    if model_type == "lstm":
        from .ml.models.deep_learning.lstm import LSTMModel, LSTMConfig
        model_class = LSTMModel
        model_config = LSTMConfig(epochs=50)  # Reduced for faster validation
    elif model_type == "transformer":
        from .ml.models.deep_learning.transformer import TransformerModel, TransformerConfig
        model_class = TransformerModel
        model_config = TransformerConfig(epochs=50)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create validator and run
    validator = WalkForwardValidator(config=config)
    results = validator.run_validation(
        model_class=model_class,
        data=ohlcv,
        model_config=model_config,
        symbol=symbol,
        timeframe=timeframe,
        verbose=verbose,
    )

    if verbose:
        results.print_summary()

    return results


# =============================================================================
# Legacy Support - Keep existing classes for backward compatibility
# =============================================================================


@dataclass
class WalkForwardResult:
    """Legacy: Complete walk-forward optimization result (for backward compatibility)."""
    total_windows: int
    profitable_windows: int
    combined_metrics: Dict[str, float]
    window_results: List[WindowResult]
    final_params: Dict[str, Any]
    robustness_score: float
    overfitting_score: float

    def to_dict(self) -> Dict:
        return {
            "total_windows": self.total_windows,
            "profitable_windows": self.profitable_windows,
            "win_rate_windows": round(self.profitable_windows / self.total_windows, 2) if self.total_windows > 0 else 0,
            "combined_metrics": {k: round(v, 4) for k, v in self.combined_metrics.items()},
            "robustness_score": round(self.robustness_score, 2),
            "overfitting_score": round(self.overfitting_score, 2),
            "final_params": self.final_params,
        }

    def print_summary(self):
        """Print walk-forward results summary."""
        print("\n" + "=" * 60)
        print("WALK-FORWARD OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Total Windows: {self.total_windows}")
        print(f"Profitable Windows: {self.profitable_windows} ({self.profitable_windows / self.total_windows:.0%})")
        print(f"\nCombined Out-of-Sample Metrics:")
        for metric, value in self.combined_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nRobustness Score: {self.robustness_score:.2f}/1.00")
        print(f"Overfitting Score: {self.overfitting_score:.2f}/1.00 (lower is better)")
        print(f"\nFinal Parameters: {self.final_params}")
        print("=" * 60)
