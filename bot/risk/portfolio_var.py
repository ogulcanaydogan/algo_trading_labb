"""
Portfolio Value-at-Risk (VaR) - Risk measurement and limits.

Calculates VaR using multiple methods to estimate potential losses
and enforce risk limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value-at-Risk calculation result."""

    var_amount: float  # VaR in dollar terms
    var_percent: float  # VaR as percentage of portfolio
    confidence_level: float  # e.g., 0.95 for 95% VaR
    time_horizon_days: int
    method: str
    expected_shortfall: float  # CVaR / ES
    worst_case: float  # Worst historical loss
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "var_amount": round(self.var_amount, 2),
            "var_percent": round(self.var_percent, 4),
            "confidence_level": self.confidence_level,
            "time_horizon_days": self.time_horizon_days,
            "method": self.method,
            "expected_shortfall": round(self.expected_shortfall, 2),
            "worst_case": round(self.worst_case, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""

    portfolio_value: float
    var_95: VaRResult
    var_99: VaRResult
    volatility_daily: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: Optional[float]
    correlation_risk: float  # Concentration in correlated assets
    stress_test_loss: float  # Loss under stress scenario
    risk_adjusted_return: float
    within_limits: bool
    limit_breaches: List[str]

    def to_dict(self) -> Dict:
        return {
            "portfolio_value": round(self.portfolio_value, 2),
            "var_95": self.var_95.to_dict(),
            "var_99": self.var_99.to_dict(),
            "volatility_daily": round(self.volatility_daily, 4),
            "volatility_annual": round(self.volatility_annual, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "beta": round(self.beta, 4) if self.beta else None,
            "correlation_risk": round(self.correlation_risk, 4),
            "stress_test_loss": round(self.stress_test_loss, 2),
            "risk_adjusted_return": round(self.risk_adjusted_return, 4),
            "within_limits": self.within_limits,
            "limit_breaches": self.limit_breaches,
        }


@dataclass
class VaRConfig:
    """Configuration for VaR calculations."""

    # Confidence levels
    confidence_95: float = 0.95
    confidence_99: float = 0.99

    # Time horizons
    time_horizon_days: int = 1

    # Methods
    methods: List[str] = field(default_factory=lambda: ["historical", "parametric", "monte_carlo"])

    # Historical settings
    lookback_days: int = 252  # 1 year

    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_random_seed: Optional[int] = None  # None for true randomness

    # Risk limits
    max_var_percent: float = 0.05  # Max 5% daily VaR
    max_volatility: float = 0.30  # Max 30% annual volatility
    max_drawdown_limit: float = 0.15  # Max 15% drawdown

    # Risk-free rate for Sharpe
    risk_free_rate: float = 0.05  # 5% annual


class PortfolioVaR:
    """
    Calculate portfolio Value-at-Risk and risk metrics.

    Methods:
    1. Historical VaR - Based on actual returns
    2. Parametric VaR - Assumes normal distribution
    3. Monte Carlo VaR - Simulation-based
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config or VaRConfig()
        self._returns_cache: Dict[str, pd.Series] = {}

    def calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> VaRResult:
        """
        Calculate Value-at-Risk for a return series.

        Args:
            returns: Daily return series
            portfolio_value: Current portfolio value
            confidence: Confidence level (e.g., 0.95)
            method: "historical", "parametric", or "monte_carlo"

        Returns:
            VaRResult with calculated metrics
        """
        returns = returns.dropna()

        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return VaRResult(
                var_amount=0,
                var_percent=0,
                confidence_level=confidence,
                time_horizon_days=self.config.time_horizon_days,
                method=method,
                expected_shortfall=0,
                worst_case=0,
            )

        if method == "historical":
            var_pct, es_pct = self._historical_var(returns, confidence)
        elif method == "parametric":
            var_pct, es_pct = self._parametric_var(returns, confidence)
        elif method == "monte_carlo":
            var_pct, es_pct = self._monte_carlo_var(returns, confidence)
        else:
            var_pct, es_pct = self._historical_var(returns, confidence)

        # Scale for time horizon
        sqrt_t = np.sqrt(self.config.time_horizon_days)
        var_pct *= sqrt_t
        es_pct *= sqrt_t

        # Convert to dollar amounts
        var_amount = abs(var_pct) * portfolio_value
        es_amount = abs(es_pct) * portfolio_value
        worst_case = abs(returns.min()) * portfolio_value

        return VaRResult(
            var_amount=var_amount,
            var_percent=abs(var_pct),
            confidence_level=confidence,
            time_horizon_days=self.config.time_horizon_days,
            method=method,
            expected_shortfall=es_amount,
            worst_case=worst_case,
        )

    def _historical_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate historical VaR and ES."""
        # VaR is the percentile of losses
        var_pct = returns.quantile(1 - confidence)

        # Expected Shortfall (CVaR) is mean of losses beyond VaR
        es_pct = returns[returns <= var_pct].mean()

        return float(var_pct), float(es_pct)

    def _parametric_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate parametric (normal) VaR and ES."""
        mu = returns.mean()
        sigma = returns.std()

        # VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence)
        var_pct = mu + z_score * sigma

        # ES for normal distribution
        es_pct = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence)

        return float(var_pct), float(es_pct)

    def _monte_carlo_var(
        self,
        returns: pd.Series,
        confidence: float,
    ) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR and ES."""
        mu = returns.mean()
        sigma = returns.std()

        # Simulate returns (use seed only if configured for reproducibility)
        if self.config.mc_random_seed is not None:
            np.random.seed(self.config.mc_random_seed)
        simulated = np.random.normal(mu, sigma, self.config.mc_simulations)

        # Calculate VaR from simulated distribution
        var_pct = np.percentile(simulated, (1 - confidence) * 100)

        # ES from simulations
        es_pct = simulated[simulated <= var_pct].mean()

        return float(var_pct), float(es_pct)

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],  # symbol -> value
        returns_data: Dict[str, pd.Series],  # symbol -> return series
        weights: Optional[Dict[str, float]] = None,
    ) -> VaRResult:
        """
        Calculate VaR for a multi-asset portfolio.

        Args:
            positions: Current positions {symbol: value}
            returns_data: Historical returns {symbol: return_series}
            weights: Portfolio weights (calculated if not provided)

        Returns:
            Portfolio VaRResult
        """
        portfolio_value = sum(abs(v) for v in positions.values())

        if portfolio_value == 0:
            return VaRResult(
                var_amount=0,
                var_percent=0,
                confidence_level=self.config.confidence_95,
                time_horizon_days=self.config.time_horizon_days,
                method="portfolio",
                expected_shortfall=0,
                worst_case=0,
            )

        # Calculate weights
        if weights is None:
            weights = {s: v / portfolio_value for s, v in positions.items()}

        # Calculate portfolio returns
        portfolio_returns = None
        for symbol, weight in weights.items():
            if symbol in returns_data:
                ret = returns_data[symbol] * abs(weight)
                if portfolio_returns is None:
                    portfolio_returns = ret
                else:
                    portfolio_returns = portfolio_returns.add(ret, fill_value=0)

        if portfolio_returns is None or len(portfolio_returns) < 30:
            logger.warning("Insufficient data for portfolio VaR")
            return VaRResult(
                var_amount=0,
                var_percent=0,
                confidence_level=self.config.confidence_95,
                time_horizon_days=self.config.time_horizon_days,
                method="portfolio",
                expected_shortfall=0,
                worst_case=0,
            )

        return self.calculate_var(
            portfolio_returns,
            portfolio_value,
            self.config.confidence_95,
            "historical",
        )

    def calculate_full_metrics(
        self,
        positions: Dict[str, float],
        returns_data: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: Current positions
            returns_data: Historical returns by symbol
            benchmark_returns: Optional benchmark for beta calculation

        Returns:
            PortfolioRiskMetrics with all risk measures
        """
        portfolio_value = sum(abs(v) for v in positions.values())

        # Calculate portfolio returns
        weights = (
            {s: v / portfolio_value for s, v in positions.items()} if portfolio_value > 0 else {}
        )

        portfolio_returns = None
        for symbol, weight in weights.items():
            if symbol in returns_data:
                ret = returns_data[symbol] * abs(weight)
                if portfolio_returns is None:
                    portfolio_returns = ret
                else:
                    portfolio_returns = portfolio_returns.add(ret, fill_value=0)

        if portfolio_returns is None:
            portfolio_returns = pd.Series([0])

        # VaR calculations
        var_95 = self.calculate_var(portfolio_returns, portfolio_value, 0.95)
        var_99 = self.calculate_var(portfolio_returns, portfolio_value, 0.99)

        # Volatility
        vol_daily = portfolio_returns.std() if len(portfolio_returns) > 1 else 0
        vol_annual = vol_daily * np.sqrt(252)

        # Sharpe ratio
        mean_return = portfolio_returns.mean() * 252  # Annualized
        rf_daily = self.config.risk_free_rate / 252
        excess_return = mean_return - self.config.risk_free_rate
        sharpe = excess_return / vol_annual if vol_annual > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < rf_daily]
        downside_std = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else vol_annual
        )
        sortino = excess_return / downside_std if downside_std > 0 else 0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Beta (if benchmark provided)
        beta = None
        if benchmark_returns is not None and len(benchmark_returns) > 30:
            aligned_port, aligned_bench = portfolio_returns.align(benchmark_returns, join="inner")
            if len(aligned_port) > 30:
                covariance = aligned_port.cov(aligned_bench)
                bench_var = aligned_bench.var()
                beta = covariance / bench_var if bench_var > 0 else 1.0

        # Concentration risk (Herfindahl index)
        if weights:
            herfindahl = sum(w**2 for w in weights.values())
        else:
            herfindahl = 1.0

        # Stress test (3 sigma event)
        stress_loss = portfolio_value * vol_daily * 3

        # Risk-adjusted return
        risk_adj_return = sharpe * vol_annual if vol_annual > 0 else 0

        # Check limits
        limit_breaches = []
        if var_95.var_percent > self.config.max_var_percent:
            limit_breaches.append(
                f"VaR {var_95.var_percent:.1%} exceeds limit {self.config.max_var_percent:.1%}"
            )
        if vol_annual > self.config.max_volatility:
            limit_breaches.append(
                f"Volatility {vol_annual:.1%} exceeds limit {self.config.max_volatility:.1%}"
            )
        if max_dd > self.config.max_drawdown_limit:
            limit_breaches.append(
                f"Drawdown {max_dd:.1%} exceeds limit {self.config.max_drawdown_limit:.1%}"
            )

        return PortfolioRiskMetrics(
            portfolio_value=portfolio_value,
            var_95=var_95,
            var_99=var_99,
            volatility_daily=vol_daily,
            volatility_annual=vol_annual,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            beta=beta,
            correlation_risk=herfindahl,
            stress_test_loss=stress_loss,
            risk_adjusted_return=risk_adj_return,
            within_limits=len(limit_breaches) == 0,
            limit_breaches=limit_breaches,
        )

    def check_risk_limits(
        self,
        positions: Dict[str, float],
        returns_data: Dict[str, pd.Series],
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits.

        Returns:
            Tuple of (within_limits, list_of_breaches)
        """
        metrics = self.calculate_full_metrics(positions, returns_data)
        return metrics.within_limits, metrics.limit_breaches

    def suggest_position_reduction(
        self,
        positions: Dict[str, float],
        returns_data: Dict[str, pd.Series],
        target_var_percent: float = 0.03,
    ) -> Dict[str, float]:
        """
        Suggest position reductions to meet VaR target.

        Returns:
            Dict of suggested position values
        """
        metrics = self.calculate_full_metrics(positions, returns_data)

        if metrics.var_95.var_percent <= target_var_percent:
            return positions  # Already within target

        # Calculate reduction ratio
        current_var = metrics.var_95.var_percent
        reduction_ratio = target_var_percent / current_var if current_var > 0 else 1.0

        # Apply reduction to all positions (simple approach)
        suggested = {s: v * reduction_ratio for s, v in positions.items()}

        logger.info(
            f"Suggested position reduction: {reduction_ratio:.1%} "
            f"(VaR {current_var:.1%} -> {target_var_percent:.1%})"
        )

        return suggested


def create_portfolio_var(config: Optional[VaRConfig] = None) -> PortfolioVaR:
    """Factory function to create portfolio VaR calculator."""
    return PortfolioVaR(config=config)
