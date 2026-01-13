"""
Risk Metrics Dashboard Module.

Provides comprehensive risk metrics including VaR, Expected Shortfall,
Sharpe, Sortino, and other risk-adjusted performance measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Return-based metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_volatility: float = 0.0
    annualized_volatility: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0

    # Value at Risk
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0

    # Stability metrics
    stability: float = 0.0  # R-squared of cumulative returns
    beta: float = 0.0  # Beta vs benchmark
    alpha: float = 0.0  # Alpha vs benchmark

    # Timestamps
    calculated_at: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class RiskMetricsCalculator:
    """
    Calculates comprehensive risk metrics for portfolio analysis.

    Provides VaR, Expected Shortfall, Sharpe, Sortino, and other
    risk-adjusted performance measures in real-time.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self._returns: List[float] = []
        self._benchmark_returns: List[float] = []
        self._equity_curve: List[float] = []
        self._dates: List[datetime] = []

    def load_returns(
        self,
        returns: List[float],
        dates: Optional[List[datetime]] = None,
        benchmark_returns: Optional[List[float]] = None,
    ) -> None:
        """Load returns data for analysis."""
        self._returns = returns
        self._dates = dates or []
        self._benchmark_returns = benchmark_returns or []

    def load_equity_curve(
        self,
        equity: List[float],
        dates: Optional[List[datetime]] = None,
    ) -> None:
        """Load equity curve and calculate returns."""
        self._equity_curve = equity
        self._dates = dates or []

        # Calculate returns from equity curve
        if len(equity) > 1:
            self._returns = [
                (equity[i] / equity[i - 1]) - 1
                for i in range(1, len(equity))
            ]

    def calculate(self) -> RiskMetrics:
        """Calculate all risk metrics."""
        if not self._returns or len(self._returns) < 2:
            return RiskMetrics()

        returns = np.array(self._returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return RiskMetrics()

        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        # Risk-adjusted returns
        daily_rf = self.risk_free_rate / 252
        sharpe = self._calculate_sharpe(returns, daily_rf)
        sortino = self._calculate_sortino(returns, daily_rf)
        calmar = self._calculate_calmar(returns, annualized_return)
        omega = self._calculate_omega(returns, daily_rf)

        # Value at Risk
        var_95 = self._calculate_var(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        cvar_99 = self._calculate_cvar(returns, 0.99)

        # Drawdown metrics
        max_dd, avg_dd, max_dd_duration = self._calculate_drawdown_metrics(returns)

        # Distribution metrics
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
        tail_ratio = self._calculate_tail_ratio(returns)

        # Win/Loss metrics
        win_rate, profit_factor, payoff_ratio, expectancy = self._calculate_win_loss_metrics(returns)

        # Stability
        stability = self._calculate_stability(returns)

        # Benchmark comparison
        beta, alpha, info_ratio = self._calculate_benchmark_metrics(returns)

        return RiskMetrics(
            total_return=round(total_return * 100, 2),
            annualized_return=round(annualized_return * 100, 2),
            daily_volatility=round(daily_vol * 100, 2),
            annualized_volatility=round(annual_vol * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            omega_ratio=round(omega, 2),
            information_ratio=round(info_ratio, 2),
            var_95=round(var_95 * 100, 2),
            var_99=round(var_99 * 100, 2),
            cvar_95=round(cvar_95 * 100, 2),
            cvar_99=round(cvar_99 * 100, 2),
            max_drawdown=round(max_dd * 100, 2),
            avg_drawdown=round(avg_dd * 100, 2),
            max_drawdown_duration=max_dd_duration,
            skewness=round(skewness, 2),
            kurtosis=round(kurtosis, 2),
            tail_ratio=round(tail_ratio, 2),
            win_rate=round(win_rate * 100, 1),
            profit_factor=round(profit_factor, 2),
            payoff_ratio=round(payoff_ratio, 2),
            expectancy=round(expectancy * 100, 2),
            stability=round(stability, 2),
            beta=round(beta, 2),
            alpha=round(alpha * 100, 2),
            period_start=self._dates[0] if self._dates else None,
            period_end=self._dates[-1] if self._dates else None,
        )

    def _calculate_sharpe(self, returns: np.ndarray, daily_rf: float) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - daily_rf
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    def _calculate_sortino(self, returns: np.ndarray, daily_rf: float) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        excess_returns = returns - daily_rf
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))

    def _calculate_calmar(self, returns: np.ndarray, annualized_return: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        max_dd, _, _ = self._calculate_drawdown_metrics(returns)

        if max_dd == 0:
            return 0.0

        return annualized_return / max_dd

    def _calculate_omega(self, returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        above = returns[returns > threshold] - threshold
        below = threshold - returns[returns <= threshold]

        if len(below) == 0 or np.sum(below) == 0:
            return float("inf") if len(above) > 0 else 1.0

        return float(np.sum(above) / np.sum(below))

    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        return float(-np.percentile(returns, (1 - confidence) * 100))

    def _calculate_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= -var]

        if len(tail_returns) == 0:
            return var

        return float(-np.mean(tail_returns))

    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Tuple[float, float, int]:
        """Calculate drawdown-related metrics."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max

        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0

        # Calculate max drawdown duration
        in_drawdown = drawdowns > 0
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, avg_dd, max_duration

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        if p5 == 0:
            return 0.0

        return float(abs(p95 / p5))

    def _calculate_win_loss_metrics(self, returns: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate win rate, profit factor, payoff ratio, expectancy."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0

        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        return win_rate, min(profit_factor, 999), min(payoff_ratio, 999), expectancy

    def _calculate_stability(self, returns: np.ndarray) -> float:
        """Calculate stability (R-squared of cumulative returns)."""
        cumulative = np.cumprod(1 + returns)
        x = np.arange(len(cumulative))

        if len(x) < 2:
            return 0.0

        slope, intercept, r_value, _, _ = stats.linregress(x, cumulative)

        return float(r_value ** 2)

    def _calculate_benchmark_metrics(
        self, returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate beta, alpha, and information ratio vs benchmark."""
        if not self._benchmark_returns or len(self._benchmark_returns) != len(returns):
            return 0.0, 0.0, 0.0

        benchmark = np.array(self._benchmark_returns)

        # Beta
        covariance = np.cov(returns, benchmark)[0, 1]
        benchmark_var = np.var(benchmark)
        beta = covariance / benchmark_var if benchmark_var > 0 else 0

        # Alpha (annualized)
        daily_rf = self.risk_free_rate / 252
        alpha_daily = np.mean(returns) - (daily_rf + beta * (np.mean(benchmark) - daily_rf))
        alpha = alpha_daily * 252

        # Information ratio
        tracking_error = np.std(returns - benchmark)
        info_ratio = np.mean(returns - benchmark) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0

        return float(beta), float(alpha), float(info_ratio)

    def get_rolling_metrics(
        self, window: int = 30
    ) -> List[Dict[str, Any]]:
        """Calculate rolling risk metrics."""
        if len(self._returns) < window:
            return []

        returns = np.array(self._returns)
        rolling_metrics = []

        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            daily_rf = self.risk_free_rate / 252

            sharpe = self._calculate_sharpe(window_returns, daily_rf)
            sortino = self._calculate_sortino(window_returns, daily_rf)
            var_95 = self._calculate_var(window_returns, 0.95)
            volatility = float(np.std(window_returns) * np.sqrt(252))

            rolling_metrics.append({
                "index": i,
                "date": self._dates[i - 1].isoformat() if self._dates and i - 1 < len(self._dates) else None,
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2),
                "var_95": round(var_95 * 100, 2),
                "volatility": round(volatility * 100, 2),
            })

        return rolling_metrics

    def get_risk_breakdown(self) -> Dict[str, Any]:
        """Get categorized risk breakdown for display."""
        metrics = self.calculate()

        return {
            "return_metrics": {
                "total_return": {"value": metrics.total_return, "unit": "%", "label": "Total Return"},
                "annualized_return": {"value": metrics.annualized_return, "unit": "%", "label": "Annual Return"},
                "volatility": {"value": metrics.annualized_volatility, "unit": "%", "label": "Volatility"},
            },
            "risk_adjusted": {
                "sharpe_ratio": {"value": metrics.sharpe_ratio, "unit": "", "label": "Sharpe Ratio"},
                "sortino_ratio": {"value": metrics.sortino_ratio, "unit": "", "label": "Sortino Ratio"},
                "calmar_ratio": {"value": metrics.calmar_ratio, "unit": "", "label": "Calmar Ratio"},
                "omega_ratio": {"value": metrics.omega_ratio, "unit": "", "label": "Omega Ratio"},
            },
            "value_at_risk": {
                "var_95": {"value": metrics.var_95, "unit": "%", "label": "VaR 95%"},
                "var_99": {"value": metrics.var_99, "unit": "%", "label": "VaR 99%"},
                "cvar_95": {"value": metrics.cvar_95, "unit": "%", "label": "CVaR 95%"},
                "cvar_99": {"value": metrics.cvar_99, "unit": "%", "label": "CVaR 99%"},
            },
            "drawdown": {
                "max_drawdown": {"value": metrics.max_drawdown, "unit": "%", "label": "Max Drawdown"},
                "avg_drawdown": {"value": metrics.avg_drawdown, "unit": "%", "label": "Avg Drawdown"},
                "max_dd_duration": {"value": metrics.max_drawdown_duration, "unit": "days", "label": "Max DD Duration"},
            },
            "distribution": {
                "skewness": {"value": metrics.skewness, "unit": "", "label": "Skewness"},
                "kurtosis": {"value": metrics.kurtosis, "unit": "", "label": "Kurtosis"},
                "tail_ratio": {"value": metrics.tail_ratio, "unit": "", "label": "Tail Ratio"},
            },
            "trading": {
                "win_rate": {"value": metrics.win_rate, "unit": "%", "label": "Win Rate"},
                "profit_factor": {"value": metrics.profit_factor, "unit": "", "label": "Profit Factor"},
                "expectancy": {"value": metrics.expectancy, "unit": "%", "label": "Expectancy"},
            },
        }

    def to_api_response(self, metrics: Optional[RiskMetrics] = None) -> Dict[str, Any]:
        """Convert metrics to API response format."""
        if metrics is None:
            metrics = self.calculate()

        return {
            "summary": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "annualized_return": metrics.annualized_return,
                "annualized_volatility": metrics.annualized_volatility,
            },
            "returns": {
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "daily_volatility": metrics.daily_volatility,
                "annualized_volatility": metrics.annualized_volatility,
            },
            "risk_adjusted": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "omega_ratio": metrics.omega_ratio,
                "information_ratio": metrics.information_ratio,
            },
            "var": {
                "var_95": metrics.var_95,
                "var_99": metrics.var_99,
                "cvar_95": metrics.cvar_95,
                "cvar_99": metrics.cvar_99,
            },
            "drawdown": {
                "max_drawdown": metrics.max_drawdown,
                "avg_drawdown": metrics.avg_drawdown,
                "max_duration_days": metrics.max_drawdown_duration,
            },
            "distribution": {
                "skewness": metrics.skewness,
                "kurtosis": metrics.kurtosis,
                "tail_ratio": metrics.tail_ratio,
            },
            "trading": {
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "payoff_ratio": metrics.payoff_ratio,
                "expectancy": metrics.expectancy,
            },
            "stability": {
                "stability": metrics.stability,
                "beta": metrics.beta,
                "alpha": metrics.alpha,
            },
            "period": {
                "start": metrics.period_start.isoformat() if metrics.period_start else None,
                "end": metrics.period_end.isoformat() if metrics.period_end else None,
            },
        }
