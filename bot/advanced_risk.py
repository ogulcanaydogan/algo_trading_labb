"""
Advanced Risk Management Module.

Implements:
- Kelly Criterion position sizing
- Correlation-based position limits
- Drawdown-based position scaling
- Monte Carlo simulation
- Benchmark comparison
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly criterion calculation result."""

    kelly_fraction: float  # Optimal fraction (0-1)
    half_kelly: float  # Conservative (half Kelly)
    quarter_kelly: float  # Very conservative
    recommended_fraction: float  # Based on risk tolerance
    win_rate: float
    avg_win: float
    avg_loss: float
    edge: float  # Expected edge per trade

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kelly_fraction": round(self.kelly_fraction, 4),
            "half_kelly": round(self.half_kelly, 4),
            "quarter_kelly": round(self.quarter_kelly, 4),
            "recommended_fraction": round(self.recommended_fraction, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "edge": round(self.edge, 4),
        }


@dataclass
class DrawdownScaling:
    """Drawdown-based position scaling result."""

    current_drawdown_pct: float
    scale_factor: float  # 0-1, multiplier for position size
    max_allowed_position_pct: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
            "scale_factor": round(self.scale_factor, 4),
            "max_allowed_position_pct": round(self.max_allowed_position_pct, 2),
            "reason": self.reason,
        }


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""

    num_simulations: int
    num_trades: int
    median_final_balance: float
    percentile_5: float  # Worst 5%
    percentile_25: float
    percentile_75: float
    percentile_95: float  # Best 5%
    probability_of_ruin: float  # % of simulations that hit ruin
    max_drawdown_median: float
    expected_sharpe: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_simulations": self.num_simulations,
            "num_trades": self.num_trades,
            "median_final_balance": round(self.median_final_balance, 2),
            "percentile_5": round(self.percentile_5, 2),
            "percentile_25": round(self.percentile_25, 2),
            "percentile_75": round(self.percentile_75, 2),
            "percentile_95": round(self.percentile_95, 2),
            "probability_of_ruin": round(self.probability_of_ruin, 4),
            "max_drawdown_median": round(self.max_drawdown_median, 4),
            "expected_sharpe": round(self.expected_sharpe, 2),
        }


@dataclass
class BenchmarkComparison:
    """Benchmark comparison result."""

    strategy_return: float
    benchmark_return: float
    alpha: float  # Excess return over benchmark
    beta: float  # Correlation with benchmark
    sharpe_strategy: float
    sharpe_benchmark: float
    max_dd_strategy: float
    max_dd_benchmark: float
    outperformance_pct: float  # % of periods strategy beat benchmark

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_return": round(self.strategy_return, 4),
            "benchmark_return": round(self.benchmark_return, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "sharpe_strategy": round(self.sharpe_strategy, 2),
            "sharpe_benchmark": round(self.sharpe_benchmark, 2),
            "max_dd_strategy": round(self.max_dd_strategy, 4),
            "max_dd_benchmark": round(self.max_dd_benchmark, 4),
            "outperformance_pct": round(self.outperformance_pct, 2),
        }


class AdvancedRiskManager:
    """
    Advanced risk management with Kelly criterion, drawdown scaling,
    and Monte Carlo analysis.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,  # Max 10% per position
        max_portfolio_risk: float = 0.25,  # Max 25% total risk
        ruin_threshold: float = 0.50,  # 50% drawdown = ruin
        risk_tolerance: str = "moderate",  # conservative, moderate, aggressive
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.ruin_threshold = ruin_threshold
        self.risk_tolerance = risk_tolerance

        # Kelly multipliers based on risk tolerance
        self.kelly_multipliers = {
            "conservative": 0.25,  # Quarter Kelly
            "moderate": 0.50,  # Half Kelly
            "aggressive": 0.75,  # Three-quarter Kelly
        }

    def calculate_kelly(
        self,
        trades: List[Dict[str, Any]],
    ) -> KellyResult:
        """
        Calculate Kelly criterion position sizing.

        Kelly formula: f* = (bp - q) / b
        where:
            f* = optimal fraction
            b = odds (avg_win / avg_loss)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        if not trades:
            return KellyResult(
                kelly_fraction=0.0,
                half_kelly=0.0,
                quarter_kelly=0.0,
                recommended_fraction=0.01,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                edge=0.0,
            )

        # Separate wins and losses
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]

        if not wins or not losses:
            # Can't calculate without both wins and losses
            return KellyResult(
                kelly_fraction=0.0,
                half_kelly=0.0,
                quarter_kelly=0.0,
                recommended_fraction=0.01,
                win_rate=len(wins) / len(trades) if trades else 0,
                avg_win=np.mean([t["pnl"] for t in wins]) if wins else 0,
                avg_loss=0.0,
                edge=0.0,
            )

        # Calculate metrics
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([abs(t.get("pnl", 0)) for t in wins])
        avg_loss = np.mean([abs(t.get("pnl", 0)) for t in losses])

        # Kelly formula
        if avg_loss > 0:
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - p

            kelly = (b * p - q) / b
            kelly = max(0, min(1, kelly))  # Clamp to 0-1
        else:
            kelly = 0.0

        # Expected edge per trade
        edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Calculate recommended based on risk tolerance
        multiplier = self.kelly_multipliers.get(self.risk_tolerance, 0.5)
        recommended = kelly * multiplier

        # Cap at max position size
        recommended = min(recommended, self.max_position_pct)

        return KellyResult(
            kelly_fraction=kelly,
            half_kelly=kelly * 0.5,
            quarter_kelly=kelly * 0.25,
            recommended_fraction=recommended,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            edge=edge,
        )

    def calculate_drawdown_scaling(
        self,
        current_balance: float,
        peak_balance: float,
    ) -> DrawdownScaling:
        """
        Calculate position scaling based on current drawdown.

        Reduces position sizes during drawdowns to preserve capital.
        """
        if peak_balance <= 0:
            return DrawdownScaling(
                current_drawdown_pct=0,
                scale_factor=1.0,
                max_allowed_position_pct=self.max_position_pct,
                reason="No peak balance",
            )

        drawdown_pct = (peak_balance - current_balance) / peak_balance
        drawdown_pct = max(0, drawdown_pct)

        # Scaling tiers
        if drawdown_pct < 0.05:  # < 5% drawdown
            scale_factor = 1.0
            reason = "Normal trading - no scaling"
        elif drawdown_pct < 0.10:  # 5-10% drawdown
            scale_factor = 0.75
            reason = "Mild drawdown - 25% position reduction"
        elif drawdown_pct < 0.15:  # 10-15% drawdown
            scale_factor = 0.50
            reason = "Moderate drawdown - 50% position reduction"
        elif drawdown_pct < 0.25:  # 15-25% drawdown
            scale_factor = 0.25
            reason = "Significant drawdown - 75% position reduction"
        else:  # > 25% drawdown
            scale_factor = 0.10
            reason = "Severe drawdown - 90% position reduction (preservation mode)"

        max_allowed = self.max_position_pct * scale_factor

        return DrawdownScaling(
            current_drawdown_pct=drawdown_pct * 100,
            scale_factor=scale_factor,
            max_allowed_position_pct=max_allowed * 100,
            reason=reason,
        )

    def calculate_correlation_limits(
        self,
        positions: Dict[str, Dict],
        correlation_matrix: Dict[str, Dict[str, float]],
        new_symbol: str,
        new_position_pct: float,
    ) -> Tuple[bool, float, str]:
        """
        Check if adding a position would exceed correlation-based limits.

        Returns:
            (allowed, max_allowed_pct, reason)
        """
        if not positions or new_symbol not in correlation_matrix:
            return True, new_position_pct, "No existing correlated positions"

        # Calculate correlated exposure
        correlated_exposure = 0.0
        high_correlation_symbols = []

        for symbol, pos in positions.items():
            if symbol == new_symbol:
                continue

            correlation = correlation_matrix.get(new_symbol, {}).get(symbol, 0)

            if abs(correlation) > 0.5:  # Significantly correlated
                pos_pct = pos.get("value", 0) / pos.get("total_value", 1)
                correlated_exposure += abs(correlation) * pos_pct
                high_correlation_symbols.append(f"{symbol}({correlation:.2f})")

        # Check if adding would exceed limits
        total_after = correlated_exposure + new_position_pct

        if total_after > self.max_portfolio_risk:
            max_allowed = max(0, self.max_portfolio_risk - correlated_exposure)
            reason = f"Correlated with: {', '.join(high_correlation_symbols)}"
            return False, max_allowed, reason

        return True, new_position_pct, "Within correlation limits"

    def run_monte_carlo(
        self,
        trades: List[Dict[str, Any]],
        initial_balance: float = 10000,
        num_simulations: int = 1000,
        num_trades: int = 100,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation to estimate future performance distribution.
        """
        if not trades:
            return MonteCarloResult(
                num_simulations=0,
                num_trades=0,
                median_final_balance=initial_balance,
                percentile_5=initial_balance,
                percentile_25=initial_balance,
                percentile_75=initial_balance,
                percentile_95=initial_balance,
                probability_of_ruin=0,
                max_drawdown_median=0,
                expected_sharpe=0,
            )

        # Extract returns
        returns = [t.get("pnl_pct", 0) / 100 for t in trades if t.get("pnl_pct")]

        if not returns:
            return MonteCarloResult(
                num_simulations=0,
                num_trades=0,
                median_final_balance=initial_balance,
                percentile_5=initial_balance,
                percentile_25=initial_balance,
                percentile_75=initial_balance,
                percentile_95=initial_balance,
                probability_of_ruin=0,
                max_drawdown_median=0,
                expected_sharpe=0,
            )

        final_balances = []
        max_drawdowns = []
        ruin_count = 0

        for _ in range(num_simulations):
            balance = initial_balance
            peak = initial_balance
            max_dd = 0

            # Simulate trades by resampling historical returns
            sim_returns = np.random.choice(returns, size=num_trades, replace=True)

            for ret in sim_returns:
                balance *= 1 + ret
                peak = max(peak, balance)
                dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                # Check for ruin
                if balance < initial_balance * (1 - self.ruin_threshold):
                    ruin_count += 1
                    break

            final_balances.append(balance)
            max_drawdowns.append(max_dd)

        # Calculate statistics
        final_balances = np.array(final_balances)

        # Estimate Sharpe
        if len(returns) > 1:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0

        return MonteCarloResult(
            num_simulations=num_simulations,
            num_trades=num_trades,
            median_final_balance=np.median(final_balances),
            percentile_5=np.percentile(final_balances, 5),
            percentile_25=np.percentile(final_balances, 25),
            percentile_75=np.percentile(final_balances, 75),
            percentile_95=np.percentile(final_balances, 95),
            probability_of_ruin=ruin_count / num_simulations,
            max_drawdown_median=np.median(max_drawdowns),
            expected_sharpe=sharpe,
        )

    def compare_to_benchmark(
        self,
        strategy_equity: List[float],
        benchmark_prices: List[float],
        initial_capital: float = 10000,
    ) -> BenchmarkComparison:
        """
        Compare strategy performance to buy-and-hold benchmark.
        """
        if len(strategy_equity) < 2 or len(benchmark_prices) < 2:
            return BenchmarkComparison(
                strategy_return=0,
                benchmark_return=0,
                alpha=0,
                beta=0,
                sharpe_strategy=0,
                sharpe_benchmark=0,
                max_dd_strategy=0,
                max_dd_benchmark=0,
                outperformance_pct=0,
            )

        # Align lengths
        min_len = min(len(strategy_equity), len(benchmark_prices))
        strategy_equity = strategy_equity[:min_len]
        benchmark_prices = benchmark_prices[:min_len]

        # Calculate returns
        strategy_returns = np.diff(strategy_equity) / strategy_equity[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

        # Total returns
        strategy_total = (strategy_equity[-1] - strategy_equity[0]) / strategy_equity[0]
        benchmark_total = (benchmark_prices[-1] - benchmark_prices[0]) / benchmark_prices[0]

        # Alpha (excess return)
        alpha = strategy_total - benchmark_total

        # Beta (correlation)
        if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
            benchmark_var = np.var(benchmark_returns)
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
        else:
            beta = 0

        # Sharpe ratios
        rf_rate = 0.04 / 252  # ~4% annual risk-free rate

        if np.std(strategy_returns) > 0:
            sharpe_strat = (
                (np.mean(strategy_returns) - rf_rate) / np.std(strategy_returns) * np.sqrt(252)
            )
        else:
            sharpe_strat = 0

        if np.std(benchmark_returns) > 0:
            sharpe_bench = (
                (np.mean(benchmark_returns) - rf_rate) / np.std(benchmark_returns) * np.sqrt(252)
            )
        else:
            sharpe_bench = 0

        # Max drawdowns
        def calc_max_dd(values):
            peak = values[0]
            max_dd = 0
            for v in values:
                peak = max(peak, v)
                dd = (peak - v) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            return max_dd

        max_dd_strat = calc_max_dd(strategy_equity)
        max_dd_bench = calc_max_dd(benchmark_prices)

        # Outperformance %
        strategy_cumret = np.cumsum(strategy_returns)
        bench_cumret = np.cumsum(benchmark_returns)
        outperform_count = sum(s > b for s, b in zip(strategy_cumret, bench_cumret))
        outperform_pct = (
            outperform_count / len(strategy_cumret) * 100 if len(strategy_cumret) > 0 else 0
        )

        return BenchmarkComparison(
            strategy_return=strategy_total,
            benchmark_return=benchmark_total,
            alpha=alpha,
            beta=beta,
            sharpe_strategy=sharpe_strat,
            sharpe_benchmark=sharpe_bench,
            max_dd_strategy=max_dd_strat,
            max_dd_benchmark=max_dd_bench,
            outperformance_pct=outperform_pct,
        )


class FeatureImportanceAnalyzer:
    """Analyze feature importance from ML models."""

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir

    def get_feature_importance(
        self,
        symbol: str,
        model_type: str = "gradient_boosting",
    ) -> Dict[str, float]:
        """Get feature importance from a trained model."""
        import joblib
        from pathlib import Path

        symbol_clean = symbol.replace("/", "_")
        model_path = Path(self.model_dir) / f"{symbol_clean}_{model_type}_model.pkl"
        meta_path = Path(self.model_dir) / f"{symbol_clean}_{model_type}_meta.json"

        if not model_path.exists():
            return {}

        try:
            model = joblib.load(model_path)

            # Get feature names from meta if available
            feature_names = []
            if meta_path.exists():
                import json

                with open(meta_path) as f:
                    meta = json.load(f)
                    feature_names = meta.get("feature_names", [])

            # Get importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                if feature_names and len(feature_names) == len(importances):
                    return dict(
                        sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    )
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importances)}

            return {}

        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}

    def get_top_features(
        self,
        symbol: str,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importance = self.get_feature_importance(symbol)
        return list(importance.items())[:top_n]


def create_advanced_risk_manager(
    risk_tolerance: str = "moderate",
) -> AdvancedRiskManager:
    """Factory function to create advanced risk manager."""
    return AdvancedRiskManager(risk_tolerance=risk_tolerance)
