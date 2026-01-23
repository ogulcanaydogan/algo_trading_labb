"""
Strategy Comparison Module.

Provides side-by-side strategy performance comparison
with comprehensive metrics and visualizations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Comprehensive metrics for a single strategy."""

    name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0
    risk_reward_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_confidence: float = 0.0
    accuracy: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies."""

    strategies: List[StrategyMetrics]
    best_pnl: str
    best_sharpe: str
    best_win_rate: str
    lowest_drawdown: str
    best_profit_factor: str
    ranking: List[Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None


class StrategyComparator:
    """
    Compare trading strategies side-by-side.

    Provides comprehensive performance comparison across multiple
    metrics with ranking and visualization data.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._strategies: Dict[str, StrategyMetrics] = {}

    def load_strategy_results(
        self,
        backtest_dir: str = "data/backtest_results",
        live_dir: str = "data/live_paper_trading",
    ) -> Dict[str, StrategyMetrics]:
        """
        Load strategy results from backtest and live trading directories.

        Returns:
            Dictionary mapping strategy names to their metrics
        """
        strategies = {}

        # Load from backtest results
        backtest_path = Path(backtest_dir)
        if backtest_path.exists():
            for result_file in backtest_path.glob("*.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)

                    strategy_name = data.get("strategy", result_file.stem)
                    metrics = self._extract_metrics_from_backtest(data, strategy_name)
                    strategies[f"{strategy_name}_backtest"] = metrics

                except Exception as e:
                    logger.error(f"Error loading {result_file}: {e}")

        # Load from live/paper trading
        live_path = Path(live_dir)
        if live_path.exists():
            state_file = live_path / "state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data = json.load(f)

                    # Extract strategy-specific results
                    trades = data.get("trades", [])
                    strategy_trades: Dict[str, List] = {}

                    for trade in trades:
                        strategy = trade.get("strategy", "unknown")
                        if strategy not in strategy_trades:
                            strategy_trades[strategy] = []
                        strategy_trades[strategy].append(trade)

                    for strategy, strades in strategy_trades.items():
                        metrics = self._calculate_metrics_from_trades(strades, f"{strategy}_live")
                        strategies[f"{strategy}_live"] = metrics

                except Exception as e:
                    logger.error(f"Error loading live state: {e}")

        self._strategies = strategies
        return strategies

    def _extract_metrics_from_backtest(self, data: Dict, name: str) -> StrategyMetrics:
        """Extract metrics from backtest result data."""
        metrics = data.get("metrics", {})
        trades = data.get("trades", [])

        # Calculate equity curve
        equity_curve = [1.0]
        for trade in trades:
            pnl_pct = trade.get("pnl_percent", 0) / 100
            equity_curve.append(equity_curve[-1] * (1 + pnl_pct))

        # Calculate daily returns
        daily_returns = []
        if len(equity_curve) > 1:
            for i in range(1, len(equity_curve)):
                daily_returns.append((equity_curve[i] / equity_curve[i - 1]) - 1)

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]

        gross_profit = sum(t.get("pnl", 0) for t in wins)
        gross_loss = abs(sum(t.get("pnl", 0) for t in losses))

        return StrategyMetrics(
            name=name,
            total_trades=metrics.get("total_trades", len(trades)),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=metrics.get("win_rate", 0),
            total_pnl=metrics.get("total_pnl", sum(t.get("pnl", 0) for t in trades)),
            avg_pnl=metrics.get("avg_pnl", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            profit_factor=metrics.get(
                "profit_factor", gross_profit / gross_loss if gross_loss > 0 else 0
            ),
            avg_win=gross_profit / len(wins) if wins else 0,
            avg_loss=gross_loss / len(losses) if losses else 0,
            largest_win=max(t.get("pnl", 0) for t in trades) if trades else 0,
            largest_loss=min(t.get("pnl", 0) for t in trades) if trades else 0,
            avg_holding_period=metrics.get("avg_holding_period", 0),
            risk_reward_ratio=metrics.get("risk_reward", 0),
            avg_confidence=np.mean([t.get("confidence", 0) for t in trades]) if trades else 0,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )

    def _calculate_metrics_from_trades(self, trades: List[Dict], name: str) -> StrategyMetrics:
        """Calculate metrics from a list of trade dictionaries."""
        if not trades:
            return StrategyMetrics(name=name)

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        # Calculate equity curve
        equity_curve = [1.0]
        for trade in trades:
            pnl_pct = trade.get("pnl_percent", 0) / 100
            equity_curve.append(equity_curve[-1] * (1 + pnl_pct))

        # Calculate max drawdown
        max_dd = 0.0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_returns.append((equity_curve[i] / equity_curve[i - 1]) - 1)

        sharpe = 0.0
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe = (mean_return / std_return) * np.sqrt(252)

        # Calculate Sortino ratio
        sortino = 0.0
        if daily_returns:
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = (np.mean(daily_returns) / downside_std) * np.sqrt(252)

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return StrategyMetrics(
            name=name,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=(len(wins) / len(trades) * 100) if trades else 0,
            total_pnl=total_pnl,
            avg_pnl=total_pnl / len(trades) if trades else 0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd * 100,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            avg_win=gross_profit / len(wins) if wins else 0,
            avg_loss=gross_loss / len(losses) if losses else 0,
            largest_win=max(pnls) if pnls else 0,
            largest_loss=min(pnls) if pnls else 0,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            avg_confidence=np.mean([t.get("confidence", 0) for t in trades]) if trades else 0,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )

    def add_strategy(self, name: str, metrics: StrategyMetrics) -> None:
        """Add a strategy to the comparison."""
        self._strategies[name] = metrics

    def compare(self, strategy_names: Optional[List[str]] = None) -> ComparisonResult:
        """
        Compare strategies and return comprehensive results.

        Args:
            strategy_names: List of strategy names to compare (all if None)

        Returns:
            ComparisonResult with rankings and best performers
        """
        if strategy_names:
            strategies = [self._strategies[n] for n in strategy_names if n in self._strategies]
        else:
            strategies = list(self._strategies.values())

        if not strategies:
            raise ValueError("No strategies to compare")

        # Find best performers
        best_pnl = max(strategies, key=lambda s: s.total_pnl).name
        best_sharpe = max(strategies, key=lambda s: s.sharpe_ratio).name
        best_win_rate = max(strategies, key=lambda s: s.win_rate).name
        lowest_drawdown = min(strategies, key=lambda s: s.max_drawdown).name
        best_profit_factor = max(
            strategies, key=lambda s: s.profit_factor if s.profit_factor != float("inf") else 0
        ).name

        # Calculate rankings
        ranking = self._calculate_ranking(strategies)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(strategies)

        return ComparisonResult(
            strategies=strategies,
            best_pnl=best_pnl,
            best_sharpe=best_sharpe,
            best_win_rate=best_win_rate,
            lowest_drawdown=lowest_drawdown,
            best_profit_factor=best_profit_factor,
            ranking=ranking,
            correlation_matrix=correlation_matrix,
        )

    def _calculate_ranking(self, strategies: List[StrategyMetrics]) -> List[Dict[str, Any]]:
        """
        Calculate composite ranking based on multiple metrics.

        Uses a weighted scoring system:
        - Sharpe Ratio: 25%
        - Win Rate: 20%
        - Profit Factor: 20%
        - Max Drawdown (inverted): 20%
        - Total PnL: 15%
        """
        scores = []

        # Normalize metrics
        sharpes = [s.sharpe_ratio for s in strategies]
        win_rates = [s.win_rate for s in strategies]
        profit_factors = [min(s.profit_factor, 10) for s in strategies]  # Cap at 10
        drawdowns = [s.max_drawdown for s in strategies]
        pnls = [s.total_pnl for s in strategies]

        def normalize(values: List[float], invert: bool = False) -> List[float]:
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return [0.5] * len(values)
            normalized = [(v - min_v) / (max_v - min_v) for v in values]
            if invert:
                normalized = [1 - n for n in normalized]
            return normalized

        norm_sharpe = normalize(sharpes)
        norm_win_rate = normalize(win_rates)
        norm_pf = normalize(profit_factors)
        norm_dd = normalize(drawdowns, invert=True)  # Lower is better
        norm_pnl = normalize(pnls)

        for i, strategy in enumerate(strategies):
            composite_score = (
                norm_sharpe[i] * 0.25
                + norm_win_rate[i] * 0.20
                + norm_pf[i] * 0.20
                + norm_dd[i] * 0.20
                + norm_pnl[i] * 0.15
            )

            scores.append(
                {
                    "rank": 0,  # Will be assigned after sorting
                    "name": strategy.name,
                    "composite_score": round(composite_score, 3),
                    "sharpe_score": round(norm_sharpe[i], 3),
                    "win_rate_score": round(norm_win_rate[i], 3),
                    "profit_factor_score": round(norm_pf[i], 3),
                    "drawdown_score": round(norm_dd[i], 3),
                    "pnl_score": round(norm_pnl[i], 3),
                    "total_trades": strategy.total_trades,
                    "sharpe_ratio": round(strategy.sharpe_ratio, 2),
                    "win_rate": round(strategy.win_rate, 1),
                    "profit_factor": round(strategy.profit_factor, 2)
                    if strategy.profit_factor != float("inf")
                    else "∞",
                    "max_drawdown": round(strategy.max_drawdown, 1),
                    "total_pnl": round(strategy.total_pnl, 2),
                }
            )

        # Sort by composite score
        scores.sort(key=lambda x: x["composite_score"], reverse=True)

        # Assign ranks
        for i, score in enumerate(scores):
            score["rank"] = i + 1

        return scores

    def _calculate_correlation_matrix(
        self, strategies: List[StrategyMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate return correlation matrix between strategies."""
        if len(strategies) < 2:
            return {}

        correlation = {}

        for s1 in strategies:
            correlation[s1.name] = {}
            for s2 in strategies:
                if len(s1.daily_returns) == len(s2.daily_returns) and len(s1.daily_returns) > 1:
                    corr = np.corrcoef(s1.daily_returns, s2.daily_returns)[0, 1]
                    correlation[s1.name][s2.name] = round(corr, 3) if not np.isnan(corr) else 0
                else:
                    correlation[s1.name][s2.name] = 0

        return correlation

    def get_equity_comparison(
        self, strategy_names: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get equity curves for comparison charting.

        Returns:
            Dictionary with strategy names as keys and equity curve points as values
        """
        if strategy_names:
            strategies = [(n, self._strategies[n]) for n in strategy_names if n in self._strategies]
        else:
            strategies = list(self._strategies.items())

        equity_data = {}
        for name, metrics in strategies:
            equity_data[name] = [
                {"index": i, "equity": round(eq * 100, 2)}
                for i, eq in enumerate(metrics.equity_curve)
            ]

        return equity_data

    def get_metrics_table(self, strategy_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metrics as a table for display."""
        if strategy_names:
            strategies = [self._strategies[n] for n in strategy_names if n in self._strategies]
        else:
            strategies = list(self._strategies.values())

        return [
            {
                "name": s.name,
                "total_trades": s.total_trades,
                "win_rate": round(s.win_rate, 1),
                "total_pnl": round(s.total_pnl, 2),
                "avg_pnl": round(s.avg_pnl, 2),
                "sharpe_ratio": round(s.sharpe_ratio, 2),
                "sortino_ratio": round(s.sortino_ratio, 2),
                "max_drawdown": round(s.max_drawdown, 1),
                "profit_factor": round(s.profit_factor, 2)
                if s.profit_factor != float("inf")
                else "∞",
                "avg_win": round(s.avg_win, 2),
                "avg_loss": round(s.avg_loss, 2),
                "largest_win": round(s.largest_win, 2),
                "largest_loss": round(s.largest_loss, 2),
                "consecutive_wins": s.consecutive_wins,
                "consecutive_losses": s.consecutive_losses,
            }
            for s in strategies
        ]

    def to_api_response(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Convert comparison result to API response format."""
        return {
            "best_performers": {
                "best_pnl": comparison.best_pnl,
                "best_sharpe": comparison.best_sharpe,
                "best_win_rate": comparison.best_win_rate,
                "lowest_drawdown": comparison.lowest_drawdown,
                "best_profit_factor": comparison.best_profit_factor,
            },
            "ranking": comparison.ranking,
            "correlation_matrix": comparison.correlation_matrix,
            "equity_curves": {
                s.name: [round(eq * 100, 2) for eq in s.equity_curve] for s in comparison.strategies
            },
            "metrics_table": [
                {
                    "name": s.name,
                    "total_trades": s.total_trades,
                    "win_rate": round(s.win_rate, 1),
                    "total_pnl": round(s.total_pnl, 2),
                    "sharpe_ratio": round(s.sharpe_ratio, 2),
                    "sortino_ratio": round(s.sortino_ratio, 2),
                    "max_drawdown": round(s.max_drawdown, 1),
                    "profit_factor": round(s.profit_factor, 2)
                    if s.profit_factor != float("inf")
                    else 999,
                }
                for s in comparison.strategies
            ],
        }
