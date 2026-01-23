"""
Walk-Forward Strategy Parameter Optimizer.

Optimizes strategy parameters using walk-forward methodology to ensure
robust, out-of-sample performance. Unlike the ML model walk-forward validator,
this focuses on trading strategy parameters like RSI thresholds, MA periods, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Defines the search space for a parameter."""

    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    values: Optional[List[Any]] = None  # For discrete choices
    param_type: str = "float"  # float, int, categorical

    def get_values(self) -> List[Any]:
        """Get all possible values for this parameter."""
        if self.values is not None:
            return self.values

        if self.param_type == "categorical":
            return self.values or []

        if self.step is not None:
            if self.param_type == "int":
                return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
            return list(np.arange(self.min_value, self.max_value + self.step, self.step))

        # Default: 10 evenly spaced values
        if self.param_type == "int":
            return list(range(int(self.min_value), int(self.max_value) + 1))
        return list(np.linspace(self.min_value, self.max_value, 10))

    def sample_random(self) -> Any:
        """Sample a random value from the parameter space."""
        if self.values is not None:
            return random.choice(self.values)

        if self.param_type == "int":
            return random.randint(int(self.min_value), int(self.max_value))

        return random.uniform(self.min_value, self.max_value)


@dataclass
class WindowOptimizationResult:
    """Result from optimizing a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    train_sharpe: float
    train_win_rate: float
    test_sharpe: float
    test_win_rate: float
    test_return: float
    test_max_drawdown: float
    test_trades: int

    @property
    def is_profitable(self) -> bool:
        return self.test_sharpe > 0 or self.test_return > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat()
            if hasattr(self.train_start, "isoformat")
            else str(self.train_start),
            "train_end": self.train_end.isoformat()
            if hasattr(self.train_end, "isoformat")
            else str(self.train_end),
            "test_start": self.test_start.isoformat()
            if hasattr(self.test_start, "isoformat")
            else str(self.test_start),
            "test_end": self.test_end.isoformat()
            if hasattr(self.test_end, "isoformat")
            else str(self.test_end),
            "best_params": self.best_params,
            "train_sharpe": round(self.train_sharpe, 4),
            "train_win_rate": round(self.train_win_rate, 4),
            "test_sharpe": round(self.test_sharpe, 4),
            "test_win_rate": round(self.test_win_rate, 4),
            "test_return": round(self.test_return, 4),
            "test_max_drawdown": round(self.test_max_drawdown, 4),
            "test_trades": self.test_trades,
            "is_profitable": self.is_profitable,
        }


@dataclass
class WalkForwardOptimizationResults:
    """Complete walk-forward optimization results."""

    windows: List[WindowOptimizationResult]
    parameter_space: List[ParameterSpace]
    strategy_name: str
    symbol: str
    aggregate_metrics: Dict[str, float]
    stable_params: Dict[str, Any]  # Parameters that are stable across windows
    unstable_params: List[str]  # Parameters that vary significantly
    robustness_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "strategy": self.strategy_name,
                "symbol": self.symbol,
                "total_windows": len(self.windows),
                "profitable_windows": sum(1 for w in self.windows if w.is_profitable),
                "robustness_score": round(self.robustness_score, 4),
                "timestamp": self.timestamp.isoformat(),
            },
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "stable_params": self.stable_params,
            "unstable_params": self.unstable_params,
            "windows": [w.to_dict() for w in self.windows],
            "parameter_space": [
                {"name": p.name, "min": p.min_value, "max": p.max_value, "type": p.param_type}
                for p in self.parameter_space
            ],
        }

    def print_summary(self) -> None:
        """Print optimization summary."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD STRATEGY OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"Strategy: {self.strategy_name}")
        print(f"Symbol: {self.symbol}")
        print(f"Total Windows: {len(self.windows)}")
        profitable = sum(1 for w in self.windows if w.is_profitable)
        print(f"Profitable Windows: {profitable} ({profitable / len(self.windows) * 100:.1f}%)")
        print("-" * 70)
        print("Aggregate Out-of-Sample Metrics:")
        for metric, value in self.aggregate_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 70)
        print("Stable Parameters (recommended):")
        for param, value in self.stable_params.items():
            print(f"  {param}: {value}")
        if self.unstable_params:
            print("\nUnstable Parameters (regime-dependent):")
            for param in self.unstable_params:
                print(f"  {param}")
        print("-" * 70)
        print(f"Robustness Score: {self.robustness_score:.2f}/1.00")
        print("=" * 70)

    def save(self, path: Union[str, Path]) -> Path:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class WalkForwardStrategyOptimizer:
    """
    Walk-Forward Strategy Parameter Optimizer.

    Optimizes trading strategy parameters using rolling windows to ensure
    parameters perform well out-of-sample, not just in-sample.

    Features:
    - Rolling window optimization
    - Random search for efficiency
    - Parameter stability analysis
    - Regime-dependent parameter detection

    Usage:
        optimizer = WalkForwardStrategyOptimizer(
            train_days=180,
            test_days=30,
            step_days=30,
        )

        param_space = [
            ParameterSpace("rsi_period", 7, 21, step=2, param_type="int"),
            ParameterSpace("rsi_oversold", 20, 40, step=5, param_type="int"),
            ParameterSpace("rsi_overbought", 60, 80, step=5, param_type="int"),
        ]

        results = optimizer.optimize(
            data=ohlcv,
            strategy_class=RSIStrategy,
            param_space=param_space,
            backtest_func=run_backtest,
        )
    """

    def __init__(
        self,
        train_days: int = 180,
        test_days: int = 30,
        step_days: int = 30,
        n_random_samples: int = 100,
        results_dir: str = "data/walk_forward_strategy",
    ):
        """
        Initialize the optimizer.

        Args:
            train_days: Days for each training window
            test_days: Days for each test window
            step_days: Days to step forward between windows
            n_random_samples: Number of random parameter combinations to try
            results_dir: Directory for saving results
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.n_random_samples = n_random_samples
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_windows(
        self,
        data: pd.DataFrame,
        timeframe: str = "1h",
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """
        Generate train/test windows.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            timeframe: Data timeframe

        Returns:
            List of (train_df, test_df, window_id) tuples
        """
        data = data.sort_index()
        bars_per_day = self._get_bars_per_day(timeframe)

        train_bars = int(self.train_days * bars_per_day)
        test_bars = int(self.test_days * bars_per_day)
        step_bars = int(self.step_days * bars_per_day)

        windows = []
        start_idx = 0
        window_id = 0

        while start_idx + train_bars + test_bars <= len(data):
            train_end = start_idx + train_bars
            test_end = train_end + test_bars

            train_df = data.iloc[start_idx:train_end].copy()
            test_df = data.iloc[train_end:test_end].copy()

            windows.append((train_df, test_df, window_id))
            window_id += 1
            start_idx += step_bars

        return windows

    def optimize(
        self,
        data: pd.DataFrame,
        backtest_func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        param_space: List[ParameterSpace],
        strategy_name: str = "strategy",
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        verbose: bool = True,
    ) -> WalkForwardOptimizationResults:
        """
        Run walk-forward optimization.

        Args:
            data: OHLCV DataFrame
            backtest_func: Function that runs backtest and returns metrics.
                          Signature: fn(data, params) -> {"sharpe": float, "win_rate": float, ...}
            param_space: List of ParameterSpace defining search space
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timeframe: Data timeframe
            verbose: Print progress

        Returns:
            WalkForwardOptimizationResults
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"WALK-FORWARD OPTIMIZATION: {strategy_name}")
            print(f"Symbol: {symbol}")
            print(
                f"Config: train={self.train_days}d, test={self.test_days}d, step={self.step_days}d"
            )
            print(f"Random samples per window: {self.n_random_samples}")
            print(f"{'=' * 70}\n")

        windows = self.generate_windows(data, timeframe)

        if not windows:
            raise ValueError("Not enough data for walk-forward optimization")

        window_results: List[WindowOptimizationResult] = []
        all_best_params: List[Dict[str, Any]] = []

        for train_df, test_df, window_id in windows:
            if verbose:
                print(f"\n--- Window {window_id + 1}/{len(windows)} ---")
                print(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
                print(f"Test: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

            try:
                result = self._optimize_window(
                    window_id=window_id,
                    train_df=train_df,
                    test_df=test_df,
                    backtest_func=backtest_func,
                    param_space=param_space,
                    verbose=verbose,
                )
                window_results.append(result)
                all_best_params.append(result.best_params)

                if verbose:
                    print(f"Best params: {result.best_params}")
                    print(f"Train Sharpe: {result.train_sharpe:.4f}")
                    print(f"Test Sharpe: {result.test_sharpe:.4f}")

            except Exception as e:
                logger.error(f"Window {window_id + 1} failed: {e}")
                if verbose:
                    print(f"ERROR: {e}")

        if not window_results:
            raise ValueError("No windows could be processed")

        # Analyze results
        aggregate_metrics = self._aggregate_metrics(window_results)
        stable_params, unstable_params = self._analyze_parameter_stability(
            all_best_params, param_space
        )
        robustness_score = self._calculate_robustness(window_results)

        results = WalkForwardOptimizationResults(
            windows=window_results,
            parameter_space=param_space,
            strategy_name=strategy_name,
            symbol=symbol,
            aggregate_metrics=aggregate_metrics,
            stable_params=stable_params,
            unstable_params=unstable_params,
            robustness_score=robustness_score,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = (
            self.results_dir / f"{strategy_name}_{symbol.replace('/', '_')}_{timestamp}.json"
        )
        results.save(results_path)

        if verbose:
            results.print_summary()

        return results

    def _optimize_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backtest_func: Callable,
        param_space: List[ParameterSpace],
        verbose: bool = True,
    ) -> WindowOptimizationResult:
        """Optimize parameters for a single window."""
        # Generate random parameter combinations
        param_combinations = self._generate_random_params(param_space, self.n_random_samples)

        best_params = None
        best_train_sharpe = -np.inf
        best_train_metrics = None

        # Search for best parameters on training data
        for params in param_combinations:
            try:
                metrics = backtest_func(train_df, params)
                sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio", 0))

                if sharpe > best_train_sharpe:
                    best_train_sharpe = sharpe
                    best_params = params
                    best_train_metrics = metrics
            except Exception as e:
                logger.debug(f"Backtest failed with params {params}: {e}")
                continue

        if best_params is None:
            # Fallback to first combination
            best_params = param_combinations[0]
            best_train_metrics = {"sharpe": 0, "win_rate": 0}
            best_train_sharpe = 0

        # Test best parameters on test data
        try:
            test_metrics = backtest_func(test_df, best_params)
        except Exception as e:
            logger.error(f"Test backtest failed: {e}")
            test_metrics = {
                "sharpe": 0,
                "win_rate": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "trades": 0,
            }

        return WindowOptimizationResult(
            window_id=window_id,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            best_params=best_params,
            train_sharpe=best_train_sharpe,
            train_win_rate=best_train_metrics.get("win_rate", 0),
            test_sharpe=test_metrics.get("sharpe", test_metrics.get("sharpe_ratio", 0)),
            test_win_rate=test_metrics.get("win_rate", 0),
            test_return=test_metrics.get("total_return", test_metrics.get("return", 0)),
            test_max_drawdown=test_metrics.get("max_drawdown", 0),
            test_trades=test_metrics.get("trades", test_metrics.get("trade_count", 0)),
        )

    def _generate_random_params(
        self,
        param_space: List[ParameterSpace],
        n_samples: int,
    ) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        combinations = []
        for _ in range(n_samples):
            params = {}
            for space in param_space:
                params[space.name] = space.sample_random()
            combinations.append(params)
        return combinations

    def _aggregate_metrics(
        self,
        results: List[WindowOptimizationResult],
    ) -> Dict[str, float]:
        """Aggregate metrics across all windows."""
        metrics = {
            "mean_test_sharpe": np.mean([r.test_sharpe for r in results]),
            "std_test_sharpe": np.std([r.test_sharpe for r in results]),
            "mean_test_win_rate": np.mean([r.test_win_rate for r in results]),
            "mean_test_return": np.mean([r.test_return for r in results]),
            "mean_max_drawdown": np.mean([r.test_max_drawdown for r in results]),
            "total_test_trades": sum(r.test_trades for r in results),
            "profitable_ratio": sum(1 for r in results if r.is_profitable) / len(results),
            "overfitting_gap": np.mean(
                [r.train_sharpe - r.test_sharpe for r in results if r.train_sharpe > 0]
            )
            if any(r.train_sharpe > 0 for r in results)
            else 0,
        }
        return metrics

    def _analyze_parameter_stability(
        self,
        all_params: List[Dict[str, Any]],
        param_space: List[ParameterSpace],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Analyze which parameters are stable across windows.

        Returns:
            Tuple of (stable_params dict, list of unstable param names)
        """
        if not all_params:
            return {}, []

        stable_params = {}
        unstable_params = []

        for space in param_space:
            values = [p[space.name] for p in all_params if space.name in p]
            if not values:
                continue

            if space.param_type == "categorical":
                # For categorical, check if same value appears > 60% of time
                from collections import Counter

                most_common = Counter(values).most_common(1)[0]
                if most_common[1] / len(values) >= 0.6:
                    stable_params[space.name] = most_common[0]
                else:
                    unstable_params.append(space.name)
            else:
                # For numeric, check coefficient of variation
                arr = np.array(values, dtype=float)
                mean_val = np.mean(arr)
                std_val = np.std(arr)

                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                else:
                    cv = 0 if std_val == 0 else 1

                if cv < 0.25:  # Low variation = stable
                    if space.param_type == "int":
                        stable_params[space.name] = int(round(mean_val))
                    else:
                        stable_params[space.name] = round(mean_val, 4)
                else:
                    unstable_params.append(space.name)

        return stable_params, unstable_params

    def _calculate_robustness(
        self,
        results: List[WindowOptimizationResult],
    ) -> float:
        """Calculate robustness score (0-1)."""
        if not results:
            return 0.0

        # Factors:
        # 1. Profitable ratio
        profitable_ratio = sum(1 for r in results if r.is_profitable) / len(results)

        # 2. Sharpe consistency (lower variance = more robust)
        sharpes = [r.test_sharpe for r in results]
        if np.std(sharpes) > 0:
            sharpe_consistency = 1 / (1 + np.std(sharpes))
        else:
            sharpe_consistency = 1.0

        # 3. Overfitting gap (smaller = better)
        gaps = [r.train_sharpe - r.test_sharpe for r in results if r.train_sharpe > r.test_sharpe]
        if gaps:
            avg_gap = np.mean(gaps)
            gap_score = 1 / (1 + avg_gap)
        else:
            gap_score = 1.0

        # Weighted combination
        robustness = 0.5 * profitable_ratio + 0.3 * sharpe_consistency + 0.2 * gap_score
        return min(1.0, max(0.0, robustness))

    @staticmethod
    def _get_bars_per_day(timeframe: str) -> float:
        """Get number of bars per day for a timeframe."""
        tf = timeframe.strip().lower()

        if tf.endswith("m"):
            minutes = int(tf[:-1])
            return (24 * 60) / minutes
        elif tf.endswith("h"):
            hours = int(tf[:-1])
            return 24 / hours
        elif tf.endswith("d"):
            return 1
        else:
            return 24  # Default hourly


@dataclass
class ParameterDriftTracker:
    """
    Tracks how optimal parameters change over time.

    Useful for detecting regime shifts and parameter instability.
    """

    history: List[Tuple[datetime, Dict[str, Any], Dict[str, float]]] = field(default_factory=list)

    def record(
        self,
        params: Dict[str, Any],
        performance: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record optimal params from a validation window."""
        ts = timestamp or datetime.now()
        self.history.append((ts, params.copy(), performance.copy()))

    def detect_drift(self, param_name: str, threshold: float = 0.3) -> bool:
        """
        Detect if a parameter is drifting significantly.

        Args:
            param_name: Name of parameter to check
            threshold: CV threshold above which drift is detected

        Returns:
            True if significant drift detected
        """
        if len(self.history) < 3:
            return False

        values = [h[1].get(param_name) for h in self.history if param_name in h[1]]
        if not values or len(values) < 3:
            return False

        try:
            arr = np.array(values, dtype=float)
            mean_val = np.mean(arr)
            if mean_val == 0:
                return False
            cv = np.std(arr) / abs(mean_val)
            return cv > threshold
        except (ValueError, TypeError):
            return False

    def get_stable_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get stable parameter ranges across all windows.

        Returns:
            Dict of param_name -> (min, max) for stable parameters
        """
        if not self.history:
            return {}

        # Collect all parameter values
        param_values: Dict[str, List[float]] = {}
        for _, params, _ in self.history:
            for name, value in params.items():
                if name not in param_values:
                    param_values[name] = []
                try:
                    param_values[name].append(float(value))
                except (ValueError, TypeError):
                    pass

        # Calculate stable ranges (exclude outliers)
        stable_ranges = {}
        for name, values in param_values.items():
            if len(values) < 2:
                continue

            arr = np.array(values)
            # Use IQR to find stable range
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = max(arr.min(), q1 - 1.5 * iqr)
            upper = min(arr.max(), q3 + 1.5 * iqr)

            stable_ranges[name] = (round(lower, 4), round(upper, 4))

        return stable_ranges

    def get_recent_trend(
        self,
        param_name: str,
        n_windows: int = 5,
    ) -> Optional[float]:
        """
        Get recent trend of a parameter.

        Returns:
            Slope of linear regression (positive = increasing, negative = decreasing)
        """
        if len(self.history) < n_windows:
            return None

        recent = self.history[-n_windows:]
        values = [h[1].get(param_name) for h in recent if param_name in h[1]]

        if len(values) < 3:
            return None

        try:
            arr = np.array(values, dtype=float)
            x = np.arange(len(arr))
            slope, _ = np.polyfit(x, arr, 1)
            return float(slope)
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "history": [
                {
                    "timestamp": ts.isoformat(),
                    "params": params,
                    "performance": perf,
                }
                for ts, params, perf in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterDriftTracker":
        """Create from dictionary."""
        tracker = cls()
        for item in data.get("history", []):
            ts = datetime.fromisoformat(item["timestamp"])
            tracker.history.append((ts, item["params"], item["performance"]))
        return tracker
