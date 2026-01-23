"""
Walk-Forward Optimization for Trading Models

Implements rolling window training and testing to simulate
real-world trading conditions and prevent overfitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""

    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: np.ndarray
    actuals: np.ndarray
    probabilities: Optional[np.ndarray] = None


@dataclass
class WalkForwardSummary:
    """Summary of walk-forward optimization."""

    symbol: str
    model_type: str
    n_folds: int
    total_train_samples: int
    total_test_samples: int
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    results: List[WalkForwardResult]


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.

    Implements anchored and rolling walk-forward analysis
    for more realistic backtesting of trading models.
    """

    def __init__(
        self,
        train_size: int = 5000,
        test_size: int = 500,
        step_size: int = 250,
        anchored: bool = False,
        purge_gap: int = 24,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_size: Number of samples for training (or minimum for anchored)
            test_size: Number of samples for testing
            step_size: Number of samples to step forward between folds
            anchored: If True, training window expands from start
            purge_gap: Gap between train and test to prevent data leakage
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.anchored = anchored
        self.purge_gap = purge_gap

    def generate_folds(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward analysis."""
        folds = []

        if self.anchored:
            # Anchored: Training always starts from beginning
            train_end = self.train_size
            while train_end + self.purge_gap + self.test_size <= n_samples:
                train_idx = np.arange(0, train_end)
                test_start = train_end + self.purge_gap
                test_idx = np.arange(test_start, min(test_start + self.test_size, n_samples))

                folds.append((train_idx, test_idx))
                train_end += self.step_size
        else:
            # Rolling: Training window slides forward
            start = 0
            while start + self.train_size + self.purge_gap + self.test_size <= n_samples:
                train_idx = np.arange(start, start + self.train_size)
                test_start = start + self.train_size + self.purge_gap
                test_idx = np.arange(test_start, min(test_start + self.test_size, n_samples))

                folds.append((train_idx, test_idx))
                start += self.step_size

        return folds

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        prices: Optional[pd.Series] = None,
    ) -> WalkForwardSummary:
        """
        Run walk-forward optimization.

        Args:
            X: Feature DataFrame
            y: Target Series
            model: Sklearn-compatible model
            prices: Optional price series for P&L calculation

        Returns:
            WalkForwardSummary with results
        """
        folds = self.generate_folds(len(X))

        if not folds:
            raise ValueError(
                f"Not enough data for walk-forward. Need at least "
                f"{self.train_size + self.purge_gap + self.test_size} samples, got {len(X)}"
            )

        logger.info(f"Running walk-forward with {len(folds)} folds")

        results = []
        all_predictions = []
        all_actuals = []
        all_prices = []

        for i, (train_idx, test_idx) in enumerate(folds):
            # Get train/test data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model (clone to ensure fresh training)
            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = fold_model.predict(X_test_scaled)

            # Get probabilities if available
            probas = None
            if hasattr(fold_model, "predict_proba"):
                probas = fold_model.predict_proba(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Get timestamps if available
            train_start = (
                X.index[train_idx[0]] if hasattr(X.index[0], "isoformat") else train_idx[0]
            )
            train_end = (
                X.index[train_idx[-1]] if hasattr(X.index[0], "isoformat") else train_idx[-1]
            )
            test_start = X.index[test_idx[0]] if hasattr(X.index[0], "isoformat") else test_idx[0]
            test_end = X.index[test_idx[-1]] if hasattr(X.index[0], "isoformat") else test_idx[-1]

            result = WalkForwardResult(
                fold=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                predictions=y_pred,
                actuals=y_test.values,
                probabilities=probas,
            )
            results.append(result)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)

            if prices is not None:
                all_prices.extend(prices.iloc[test_idx].values)

            logger.info(f"  Fold {i + 1}: Accuracy={accuracy:.2%}, F1={f1:.2%}")

        # Calculate summary statistics
        accuracies = [r.accuracy for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1 for r in results]

        # Calculate trading metrics
        sharpe, max_dd, win_rate, profit_factor = self._calculate_trading_metrics(
            np.array(all_predictions),
            np.array(all_actuals),
            np.array(all_prices) if prices is not None else None,
        )

        summary = WalkForwardSummary(
            symbol="",  # Set by caller
            model_type=type(model).__name__,
            n_folds=len(folds),
            total_train_samples=sum(r.train_samples for r in results),
            total_test_samples=sum(r.test_samples for r in results),
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_precision=np.mean(precisions),
            mean_recall=np.mean(recalls),
            mean_f1=np.mean(f1s),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            results=results,
        )

        return summary

    def _calculate_trading_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """Calculate trading-specific metrics."""

        # Simple P&L based on correct predictions
        correct = predictions == actuals

        # Win rate
        trades = predictions != 0  # Exclude HOLD signals
        if trades.sum() == 0:
            return 0.0, 0.0, 0.0, 1.0

        wins = (correct & trades).sum()
        win_rate = wins / trades.sum()

        # Simulated returns (assuming 1% move per correct trade)
        returns = np.where(correct & trades, 0.01, np.where(trades, -0.01, 0))

        # Sharpe ratio (annualized, assuming hourly data)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)  # Hourly to annual
        else:
            sharpe = 0.0

        # Maximum drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = abs(drawdowns.min())

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return sharpe, max_drawdown, win_rate, profit_factor


class AdaptiveWalkForward(WalkForwardOptimizer):
    """
    Adaptive Walk-Forward that adjusts window sizes
    based on market regime changes.
    """

    def __init__(
        self,
        min_train_size: int = 2000,
        max_train_size: int = 10000,
        test_size: int = 500,
        volatility_lookback: int = 100,
    ):
        super().__init__(train_size=min_train_size, test_size=test_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.volatility_lookback = volatility_lookback

    def run_adaptive(
        self, X: pd.DataFrame, y: pd.Series, model: BaseEstimator, returns: pd.Series
    ) -> WalkForwardSummary:
        """
        Run adaptive walk-forward with dynamic window sizing.

        Uses market volatility to adjust training window:
        - High volatility: Smaller window (recent data more relevant)
        - Low volatility: Larger window (more data helps)
        """
        # Calculate rolling volatility
        volatility = returns.rolling(self.volatility_lookback).std()
        vol_percentile = volatility.rank(pct=True)

        results = []
        position = self.min_train_size

        while position + self.test_size < len(X):
            # Get current volatility regime
            current_vol_pct = (
                vol_percentile.iloc[position] if position < len(vol_percentile) else 0.5
            )

            # Adjust window size based on volatility
            # High vol (>75th percentile): use min window
            # Low vol (<25th percentile): use max window
            if current_vol_pct > 0.75:
                current_train_size = self.min_train_size
            elif current_vol_pct < 0.25:
                current_train_size = self.max_train_size
            else:
                # Linear interpolation
                vol_factor = (current_vol_pct - 0.25) / 0.5
                current_train_size = int(
                    self.max_train_size - vol_factor * (self.max_train_size - self.min_train_size)
                )

            # Ensure we don't go before start of data
            train_start = max(0, position - current_train_size)
            train_idx = np.arange(train_start, position)
            test_idx = np.arange(position, min(position + self.test_size, len(X)))

            if len(train_idx) < self.min_train_size // 2:
                position += self.test_size
                continue

            # Train and evaluate
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            logger.debug(
                f"Position {position}: train_size={len(train_idx)}, "
                f"vol_pct={current_vol_pct:.2f}, accuracy={accuracy:.2%}"
            )

            result = WalkForwardResult(
                fold=len(results) + 1,
                train_start=train_start,
                train_end=position,
                test_start=position,
                test_end=position + len(test_idx),
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                accuracy=accuracy,
                precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
                recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
                f1=f1,
                predictions=y_pred,
                actuals=y_test.values,
            )
            results.append(result)

            position += self.test_size

        if not results:
            raise ValueError("Not enough data for adaptive walk-forward")

        accuracies = [r.accuracy for r in results]

        return WalkForwardSummary(
            symbol="",
            model_type=type(model).__name__,
            n_folds=len(results),
            total_train_samples=sum(r.train_samples for r in results),
            total_test_samples=sum(r.test_samples for r in results),
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_precision=np.mean([r.precision for r in results]),
            mean_recall=np.mean([r.recall for r in results]),
            mean_f1=np.mean([r.f1 for r in results]),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=np.mean(accuracies),
            profit_factor=1.0,
            results=results,
        )


def run_walk_forward_analysis(
    symbol: str,
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    prices: Optional[pd.Series] = None,
    train_size: int = 5000,
    test_size: int = 500,
) -> WalkForwardSummary:
    """
    Convenience function to run walk-forward analysis.

    Args:
        symbol: Trading symbol
        X: Feature DataFrame
        y: Target Series
        model: Sklearn model
        prices: Optional price series
        train_size: Training window size
        test_size: Test window size

    Returns:
        WalkForwardSummary with results
    """
    optimizer = WalkForwardOptimizer(
        train_size=train_size, test_size=test_size, step_size=test_size, anchored=False
    )

    summary = optimizer.run(X, y, model, prices)
    summary.symbol = symbol

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Walk-Forward Results for {symbol}")
    logger.info(f"{'=' * 50}")
    logger.info(f"Folds: {summary.n_folds}")
    logger.info(f"Mean Accuracy: {summary.mean_accuracy:.2%} (+/- {summary.std_accuracy:.2%})")
    logger.info(f"Mean F1: {summary.mean_f1:.2%}")
    logger.info(f"Win Rate: {summary.win_rate:.2%}")
    logger.info(f"Sharpe Ratio: {summary.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {summary.max_drawdown:.2%}")
    logger.info(f"Profit Factor: {summary.profit_factor:.2f}")

    return summary
