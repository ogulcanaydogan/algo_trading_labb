"""
Portfolio Optimization - Black-Litterman and Risk Parity.

Advanced portfolio optimization techniques for optimal
capital allocation across assets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""

    method: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    max_drawdown_estimate: float
    constraints_satisfied: bool
    optimization_message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 4),
            "expected_volatility": round(self.expected_volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "max_drawdown_estimate": round(self.max_drawdown_estimate, 4),
            "constraints_satisfied": self.constraints_satisfied,
            "optimization_message": self.optimization_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BlackLittermanView:
    """Investor view for Black-Litterman model."""

    asset: str
    expected_return: float  # Absolute view
    confidence: float  # 0 to 1
    relative_to: Optional[str] = None  # For relative views

    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "expected_return": self.expected_return,
            "confidence": self.confidence,
            "relative_to": self.relative_to,
        }


@dataclass
class OptimizationConfig:
    """Portfolio optimization configuration."""

    # Risk-free rate
    risk_free_rate: float = 0.05

    # Constraints
    min_weight: float = 0.0
    max_weight: float = 0.4
    max_total_short: float = 0.0  # No shorting by default

    # Black-Litterman parameters
    tau: float = 0.05  # Scaling factor for uncertainty

    # Risk parity parameters
    risk_budget: Optional[Dict[str, float]] = None  # Equal risk if None

    # Optimization settings
    max_iterations: int = 1000
    tolerance: float = 1e-8


class PortfolioOptimizer:
    """
    Advanced portfolio optimization.

    Methods:
    1. Mean-Variance (Markowitz)
    2. Black-Litterman
    3. Risk Parity
    4. Maximum Sharpe
    5. Minimum Volatility
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._returns: Optional[pd.DataFrame] = None
        self._cov_matrix: Optional[np.ndarray] = None
        self._assets: List[str] = []

    def set_returns(self, returns: pd.DataFrame):
        """
        Set historical returns data.

        Args:
            returns: DataFrame with assets as columns, returns as rows
        """
        self._returns = returns
        self._assets = list(returns.columns)
        self._cov_matrix = returns.cov().values

    def set_covariance(self, cov_matrix: np.ndarray, assets: List[str]):
        """Set covariance matrix directly."""
        self._cov_matrix = cov_matrix
        self._assets = assets

    def optimize_mean_variance(
        self,
        target_return: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz).

        Args:
            target_return: Target return constraint (None for max Sharpe)

        Returns:
            OptimizationResult with optimal weights
        """
        if self._returns is None:
            raise ValueError("Returns data not set")

        n_assets = len(self._assets)
        mean_returns = self._returns.mean().values
        cov = self._cov_matrix

        # Initial weights
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda x: np.dot(x, mean_returns) * 252 - target_return}
            )

        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets

        # Objective: minimize variance (or negative Sharpe)
        if target_return is not None:

            def objective(w):
                return np.dot(w.T, np.dot(cov, w)) * 252
        else:

            def objective(w):
                ret = np.dot(w, mean_returns) * 252
                vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) * 252)
                return -(ret - self.config.risk_free_rate) / vol if vol > 0 else 0

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations},
        )

        weights = dict(zip(self._assets, result.x))
        exp_return = np.dot(result.x, mean_returns) * 252
        exp_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)) * 252)
        sharpe = (exp_return - self.config.risk_free_rate) / exp_vol if exp_vol > 0 else 0

        return OptimizationResult(
            method="mean_variance",
            weights=weights,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=self._calc_diversification(result.x),
            max_drawdown_estimate=exp_vol * 2.5,  # Rough estimate
            constraints_satisfied=result.success,
            optimization_message=result.message,
        )

    def optimize_black_litterman(
        self,
        views: List[BlackLittermanView],
        market_caps: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Black-Litterman optimization with investor views.

        Args:
            views: List of investor views
            market_caps: Market capitalizations for equilibrium weights

        Returns:
            OptimizationResult with BL-adjusted weights
        """
        if self._returns is None:
            raise ValueError("Returns data not set")

        n_assets = len(self._assets)
        cov = self._cov_matrix
        tau = self.config.tau

        # Market equilibrium weights
        if market_caps:
            total_cap = sum(market_caps.values())
            eq_weights = np.array(
                [market_caps.get(a, total_cap / n_assets) / total_cap for a in self._assets]
            )
        else:
            eq_weights = np.ones(n_assets) / n_assets

        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 2.5  # Typical value
        pi = risk_aversion * np.dot(cov, eq_weights) * 252

        # Build views matrix
        P = []  # View matrix
        Q = []  # View returns
        omega_diag = []  # View uncertainty

        for view in views:
            if view.asset not in self._assets:
                continue

            idx = self._assets.index(view.asset)
            row = np.zeros(n_assets)

            if view.relative_to and view.relative_to in self._assets:
                # Relative view
                idx2 = self._assets.index(view.relative_to)
                row[idx] = 1
                row[idx2] = -1
            else:
                # Absolute view
                row[idx] = 1

            P.append(row)
            Q.append(view.expected_return)

            # View uncertainty based on confidence
            uncertainty = (1 - view.confidence) * 0.1  # Scale uncertainty
            omega_diag.append(uncertainty)

        if not P:
            # No valid views, return equilibrium
            weights = dict(zip(self._assets, eq_weights))
            exp_return = np.dot(eq_weights, pi)
            exp_vol = np.sqrt(np.dot(eq_weights.T, np.dot(cov, eq_weights)) * 252)

            return OptimizationResult(
                method="black_litterman",
                weights=weights,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=(exp_return - self.config.risk_free_rate) / exp_vol,
                diversification_ratio=1.0,
                max_drawdown_estimate=exp_vol * 2.5,
                constraints_satisfied=True,
                optimization_message="No views provided, using equilibrium",
            )

        P = np.array(P)
        Q = np.array(Q)
        omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_cov = tau * cov
        M = np.linalg.inv(np.linalg.inv(tau_cov) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        bl_returns = np.dot(
            M, (np.dot(np.linalg.inv(tau_cov), pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))
        )

        # Optimize with BL returns
        def objective(w):
            ret = np.dot(w, bl_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) * 252)
            return -(ret - self.config.risk_free_rate) / vol if vol > 0 else 0

        x0 = eq_weights
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = dict(zip(self._assets, result.x))
        exp_return = np.dot(result.x, bl_returns)
        exp_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)) * 252)

        return OptimizationResult(
            method="black_litterman",
            weights=weights,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=(exp_return - self.config.risk_free_rate) / exp_vol,
            diversification_ratio=self._calc_diversification(result.x),
            max_drawdown_estimate=exp_vol * 2.5,
            constraints_satisfied=result.success,
            optimization_message=f"BL optimization with {len(views)} views",
        )

    def optimize_risk_parity(
        self,
        risk_budget: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Risk parity optimization (equal risk contribution).

        Args:
            risk_budget: Target risk contribution per asset (equal if None)

        Returns:
            OptimizationResult with risk parity weights
        """
        if self._cov_matrix is None:
            raise ValueError("Covariance matrix not set")

        n_assets = len(self._assets)
        cov = self._cov_matrix * 252  # Annualized

        # Target risk budget
        if risk_budget:
            budget = np.array([risk_budget.get(a, 1.0 / n_assets) for a in self._assets])
            budget = budget / budget.sum()
        else:
            budget = np.ones(n_assets) / n_assets

        def risk_contribution(w):
            """Calculate risk contribution of each asset."""
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            marginal_risk = np.dot(cov, w) / port_vol
            return w * marginal_risk

        def objective(w):
            """Minimize deviation from target risk budget."""
            rc = risk_contribution(w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            rc_pct = rc / port_vol
            return np.sum((rc_pct - budget) ** 2)

        x0 = np.ones(n_assets) / n_assets
        bounds = [(0.01, self.config.max_weight)] * n_assets
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.config.max_iterations},
        )

        weights = dict(zip(self._assets, result.x))
        exp_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))

        # Expected return (if we have returns data)
        if self._returns is not None:
            exp_return = np.dot(result.x, self._returns.mean().values) * 252
        else:
            exp_return = 0

        return OptimizationResult(
            method="risk_parity",
            weights=weights,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=(exp_return - self.config.risk_free_rate) / exp_vol if exp_vol > 0 else 0,
            diversification_ratio=self._calc_diversification(result.x),
            max_drawdown_estimate=exp_vol * 2.5,
            constraints_satisfied=result.success,
            optimization_message="Risk parity optimization",
        )

    def optimize_minimum_volatility(self) -> OptimizationResult:
        """
        Minimum volatility portfolio.

        Returns:
            OptimizationResult with minimum volatility weights
        """
        if self._cov_matrix is None:
            raise ValueError("Covariance matrix not set")

        n_assets = len(self._assets)
        cov = self._cov_matrix * 252

        def objective(w):
            return np.dot(w.T, np.dot(cov, w))

        x0 = np.ones(n_assets) / n_assets
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = dict(zip(self._assets, result.x))
        exp_vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))

        if self._returns is not None:
            exp_return = np.dot(result.x, self._returns.mean().values) * 252
        else:
            exp_return = 0

        return OptimizationResult(
            method="minimum_volatility",
            weights=weights,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=(exp_return - self.config.risk_free_rate) / exp_vol if exp_vol > 0 else 0,
            diversification_ratio=self._calc_diversification(result.x),
            max_drawdown_estimate=exp_vol * 2.5,
            constraints_satisfied=result.success,
            optimization_message="Minimum volatility optimization",
        )

    def _calc_diversification(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio."""
        if self._cov_matrix is None:
            return 1.0

        cov = self._cov_matrix
        vols = np.sqrt(np.diag(cov))
        weighted_vol = np.dot(weights, vols)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

        return weighted_vol / port_vol if port_vol > 0 else 1.0

    def compare_methods(self) -> Dict[str, OptimizationResult]:
        """Compare all optimization methods."""
        results = {}

        try:
            results["max_sharpe"] = self.optimize_mean_variance()
        except Exception as e:
            logger.warning(f"Max Sharpe optimization failed: {e}")

        try:
            results["min_volatility"] = self.optimize_minimum_volatility()
        except Exception as e:
            logger.warning(f"Min volatility optimization failed: {e}")

        try:
            results["risk_parity"] = self.optimize_risk_parity()
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")

        return results


def create_portfolio_optimizer(config: Optional[OptimizationConfig] = None) -> PortfolioOptimizer:
    """Factory function to create portfolio optimizer."""
    return PortfolioOptimizer(config=config)
