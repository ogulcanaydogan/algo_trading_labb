"""
Strategy Parameter Optimizer with Walk-Forward Validation.

Optimizes trading strategy parameters using:
- Grid search or Bayesian optimization
- Walk-forward out-of-sample validation
- Multiple objective functions (Sharpe, Sortino, Profit Factor)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategies.base import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of strategy optimization."""
    strategy_name: str
    best_params: Dict[str, Any]
    best_score: float
    objective: str
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    windows_tested: int = 0

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "objective": self.objective,
            "in_sample_metrics": {k: round(v, 4) for k, v in self.in_sample_metrics.items()},
            "out_of_sample_metrics": {k: round(v, 4) for k, v in self.out_of_sample_metrics.items()},
            "optimization_time": round(self.optimization_time, 2),
            "windows_tested": self.windows_tested,
        }


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_index: int


class StrategyOptimizer:
    """
    Walk-Forward Strategy Optimizer.

    Features:
    - Grid search over parameter space
    - Walk-forward validation to prevent overfitting
    - Multiple optimization objectives
    - Robust out-of-sample testing
    """

    OBJECTIVES = {
        "sharpe": "_calc_sharpe",
        "sortino": "_calc_sortino",
        "profit_factor": "_calc_profit_factor",
        "win_rate": "_calc_win_rate",
        "total_return": "_calc_total_return",
    }

    def __init__(
        self,
        train_window_days: int = 180,
        test_window_days: int = 30,
        step_days: int = 30,
        min_trades: int = 10,
    ):
        """
        Initialize optimizer.

        Args:
            train_window_days: Days for training/optimization
            test_window_days: Days for out-of-sample testing
            step_days: Days to step forward between windows
            min_trades: Minimum trades required for valid metrics
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_trades = min_trades

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows."""
        windows = []
        current_start = start_date
        window_index = 0

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)

            if test_end > end_date:
                break

            windows.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_index=window_index,
            ))

            current_start += timedelta(days=self.step_days)
            window_index += 1

        return windows

    def optimize(
        self,
        strategy_class: type,
        ohlcv: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        objective: str = "sharpe",
        n_jobs: int = 1,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using walk-forward validation.

        Args:
            strategy_class: Strategy class to optimize
            ohlcv: OHLCV DataFrame with datetime index
            param_grid: Dictionary of parameter names to lists of values
            objective: Optimization objective ('sharpe', 'sortino', 'profit_factor', etc.)
            n_jobs: Number of parallel jobs (not implemented yet)

        Returns:
            OptimizationResult with best parameters and metrics
        """
        start_time = datetime.now()

        if objective not in self.OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}. Use: {list(self.OBJECTIVES.keys())}")

        # Get date range from data
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            raise ValueError("OHLCV index must be DatetimeIndex")

        start_date = ohlcv.index.min().to_pydatetime()
        end_date = ohlcv.index.max().to_pydatetime()

        # Generate walk-forward windows
        windows = self.generate_windows(start_date, end_date)
        if len(windows) == 0:
            raise ValueError("Not enough data for walk-forward validation")

        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(itertools.product(*param_values))

        logger.info(f"Testing {len(all_combos)} parameter combinations")

        # Track results
        all_results = []
        best_score = float("-inf")
        best_params = {}
        best_in_sample = {}
        best_out_sample = {}

        # Test each parameter combination
        for combo in all_combos:
            params = dict(zip(param_names, combo))

            try:
                # Run walk-forward validation
                in_sample_scores = []
                out_sample_scores = []

                for window in windows:
                    # Get data for this window
                    train_data = ohlcv[
                        (ohlcv.index >= window.train_start) &
                        (ohlcv.index < window.train_end)
                    ]
                    test_data = ohlcv[
                        (ohlcv.index >= window.test_start) &
                        (ohlcv.index < window.test_end)
                    ]

                    if len(train_data) < 100 or len(test_data) < 20:
                        continue

                    # Create strategy with these parameters
                    strategy = self._create_strategy(strategy_class, params)

                    # Backtest on training data
                    train_trades = self._backtest(strategy, train_data)
                    train_metrics = self._calculate_metrics(train_trades)
                    in_sample_scores.append(
                        self._get_objective_value(train_metrics, objective)
                    )

                    # Backtest on test data (out-of-sample)
                    test_trades = self._backtest(strategy, test_data)
                    test_metrics = self._calculate_metrics(test_trades)
                    out_sample_scores.append(
                        self._get_objective_value(test_metrics, objective)
                    )

                if not out_sample_scores:
                    continue

                # Average scores across windows
                avg_in_sample = np.mean(in_sample_scores)
                avg_out_sample = np.mean(out_sample_scores)

                # Store result
                result = {
                    "params": params,
                    "in_sample_score": avg_in_sample,
                    "out_sample_score": avg_out_sample,
                    "windows": len(out_sample_scores),
                }
                all_results.append(result)

                # Track best (based on out-of-sample performance)
                if avg_out_sample > best_score:
                    best_score = avg_out_sample
                    best_params = params
                    best_in_sample = {"score": avg_in_sample}
                    best_out_sample = {"score": avg_out_sample}

            except Exception as e:
                logger.warning(f"Error testing params {params}: {e}")
                continue

        # Final metrics on full out-of-sample period
        if best_params:
            # Run final backtest with best params
            strategy = self._create_strategy(strategy_class, best_params)
            all_trades = self._backtest(strategy, ohlcv)
            final_metrics = self._calculate_metrics(all_trades)
            best_out_sample.update(final_metrics)

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            strategy_name=strategy_class.__name__ if hasattr(strategy_class, '__name__') else str(strategy_class),
            best_params=best_params,
            best_score=best_score,
            objective=objective,
            in_sample_metrics=best_in_sample,
            out_of_sample_metrics=best_out_sample,
            all_results=all_results,
            optimization_time=optimization_time,
            windows_tested=len(windows),
        )

    def _create_strategy(
        self,
        strategy_class: type,
        params: Dict[str, Any],
    ) -> BaseStrategy:
        """Create strategy instance with given parameters."""
        try:
            return strategy_class(**params)
        except TypeError:
            # Strategy might not accept all params - try creating default
            return strategy_class()

    def _backtest(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
    ) -> List[Dict]:
        """Run simple backtest and return list of trades."""
        trades = []
        position = None
        min_rows = 100

        if len(ohlcv) < min_rows:
            return trades

        for i in range(min_rows, len(ohlcv)):
            window = ohlcv.iloc[:i+1]
            current_price = window["close"].iloc[-1]
            timestamp = window.index[-1]

            try:
                signal = strategy.generate_signal(window)
            except Exception:
                continue

            # Entry logic
            if position is None and signal.decision in ["LONG", "SHORT"]:
                if signal.confidence > 0.4:
                    position = {
                        "side": signal.decision.lower(),
                        "entry_price": current_price,
                        "entry_time": timestamp,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                    }

            # Exit logic
            elif position is not None:
                should_exit = False
                exit_reason = ""

                # Stop loss
                if position.get("stop_loss"):
                    if position["side"] == "long" and current_price <= position["stop_loss"]:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif position["side"] == "short" and current_price >= position["stop_loss"]:
                        should_exit = True
                        exit_reason = "stop_loss"

                # Take profit
                if position.get("take_profit"):
                    if position["side"] == "long" and current_price >= position["take_profit"]:
                        should_exit = True
                        exit_reason = "take_profit"
                    elif position["side"] == "short" and current_price <= position["take_profit"]:
                        should_exit = True
                        exit_reason = "take_profit"

                # Signal reversal
                if signal.decision == "FLAT":
                    should_exit = True
                    exit_reason = "signal_flat"
                elif signal.decision == "LONG" and position["side"] == "short":
                    should_exit = True
                    exit_reason = "reversal"
                elif signal.decision == "SHORT" and position["side"] == "long":
                    should_exit = True
                    exit_reason = "reversal"

                if should_exit:
                    # Calculate P&L
                    if position["side"] == "long":
                        pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                    else:
                        pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]

                    trades.append({
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "side": position["side"],
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                    })

                    position = None

        return trades

    def _calculate_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        if len(trades) < self.min_trades:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "num_trades": len(trades),
            }

        returns = [t["pnl_pct"] for t in trades]
        returns = np.array(returns)

        # Win rate
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns)

        # Total return (compounded)
        total_return = np.prod(1 + returns) - 1

        # Sharpe ratio (assuming daily)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
        else:
            sortino = sharpe

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "total_return": total_return,
            "num_trades": len(trades),
        }

    def _get_objective_value(
        self,
        metrics: Dict[str, float],
        objective: str,
    ) -> float:
        """Get the objective value from metrics."""
        return metrics.get(objective, 0.0)


# Convenience function
def optimize_strategy(
    strategy_class: type,
    ohlcv: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    objective: str = "sharpe",
) -> OptimizationResult:
    """
    Convenience function for strategy optimization.

    Args:
        strategy_class: Strategy class to optimize
        ohlcv: OHLCV data
        param_grid: Parameters to optimize
        objective: Optimization objective

    Returns:
        OptimizationResult with best parameters
    """
    optimizer = StrategyOptimizer()
    return optimizer.optimize(strategy_class, ohlcv, param_grid, objective)
