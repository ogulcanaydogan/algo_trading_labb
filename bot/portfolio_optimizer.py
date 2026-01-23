"""
Portfolio Optimizer Module.

Provides portfolio construction and optimization using:
- Correlation analysis
- Risk parity allocation
- Mean-variance optimization
- Minimum volatility portfolio
- Maximum diversification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MIN_VOLATILITY = "min_volatility"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"
    INVERSE_VOLATILITY = "inverse_volatility"


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""

    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    effective_n: float  # Effective number of assets


@dataclass
class AllocationResult:
    """Result from portfolio optimization."""

    weights: Dict[str, float]
    method: OptimizationMethod
    metrics: PortfolioMetrics
    correlation_matrix: pd.DataFrame
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "method": self.method.value,
            "metrics": {
                "expected_return": round(self.metrics.expected_return, 4),
                "volatility": round(self.metrics.volatility, 4),
                "sharpe_ratio": round(self.metrics.sharpe_ratio, 4),
                "max_drawdown": round(self.metrics.max_drawdown, 4),
                "diversification_ratio": round(self.metrics.diversification_ratio, 4),
                "effective_n": round(self.metrics.effective_n, 2),
            },
            "timestamp": self.timestamp.isoformat(),
        }

    def print_summary(self):
        """Print allocation summary."""
        print("\n" + "=" * 60)
        print(f"PORTFOLIO ALLOCATION ({self.method.value.upper()})")
        print("=" * 60)
        print("\nWeights:")
        for symbol, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            print(f"  {symbol}: {weight:.2%}")
        print(f"\nExpected Annual Return: {self.metrics.expected_return:.2%}")
        print(f"Annual Volatility: {self.metrics.volatility:.2%}")
        print(f"Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
        print(f"Diversification Ratio: {self.metrics.diversification_ratio:.2f}")
        print(f"Effective N Assets: {self.metrics.effective_n:.1f}")
        print("=" * 60)


class PortfolioOptimizer:
    """
    Portfolio Optimizer for multi-asset allocation.

    Supports multiple optimization methods:
    - Equal Weight: Simple 1/N allocation
    - Risk Parity: Equal risk contribution from each asset
    - Min Volatility: Minimize portfolio variance
    - Max Sharpe: Maximum risk-adjusted return
    - Max Diversification: Maximize diversification ratio
    - Inverse Volatility: Weight inversely proportional to volatility
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        min_weight: float = 0.0,  # Minimum weight per asset
        max_weight: float = 1.0,  # Maximum weight per asset
        target_volatility: Optional[float] = None,  # Target portfolio volatility
    ):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            min_weight: Minimum allocation per asset (0-1)
            max_weight: Maximum allocation per asset (0-1)
            target_volatility: Target portfolio volatility (optional)
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_volatility = target_volatility

    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> AllocationResult:
        """
        Optimize portfolio allocation.

        Args:
            returns: DataFrame of asset returns (columns = assets, rows = periods)
            method: Optimization method to use
            constraints: Optional per-asset constraints {symbol: (min, max)}

        Returns:
            AllocationResult with optimal weights
        """
        # Validate inputs
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        n_assets = len(returns.columns)
        assets = list(returns.columns)

        # Calculate statistics
        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = returns.cov() * 252  # Annualize
        corr_matrix = returns.corr()
        volatilities = returns.std() * np.sqrt(252)

        # Get optimal weights based on method
        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(n_assets)
        elif method == OptimizationMethod.INVERSE_VOLATILITY:
            weights = self._inverse_volatility(volatilities)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(cov_matrix)
        elif method == OptimizationMethod.MIN_VOLATILITY:
            weights = self._min_volatility(cov_matrix, constraints, assets)
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe(mean_returns, cov_matrix, constraints, assets)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification(volatilities, cov_matrix, constraints, assets)
        else:
            weights = self._equal_weight(n_assets)

        # Create weights dictionary
        weights_dict = {assets[i]: weights[i] for i in range(n_assets)}

        # Calculate portfolio metrics
        metrics = self._calculate_metrics(
            weights, mean_returns.values, cov_matrix.values, volatilities.values, returns
        )

        return AllocationResult(
            weights=weights_dict,
            method=method,
            metrics=metrics,
            correlation_matrix=corr_matrix,
        )

    def _equal_weight(self, n_assets: int) -> np.ndarray:
        """Equal weight allocation (1/N)."""
        return np.ones(n_assets) / n_assets

    def _inverse_volatility(self, volatilities: pd.Series) -> np.ndarray:
        """Inverse volatility weighting."""
        inv_vol = 1.0 / volatilities.values
        return inv_vol / inv_vol.sum()

    def _risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Risk parity allocation.

        Each asset contributes equally to portfolio risk.
        """
        n_assets = len(cov_matrix)
        cov = cov_matrix.values

        def risk_contribution(weights):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(weights @ cov @ weights)
            marginal_contrib = cov @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib

        def objective(weights):
            """Minimize deviation from equal risk contribution."""
            rc = risk_contribution(weights)
            target_rc = 1.0 / n_assets
            return np.sum((rc - target_rc) ** 2)

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        return result.x if result.success else x0

    def _min_volatility(
        self,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Tuple[float, float]]],
        assets: List[str],
    ) -> np.ndarray:
        """Minimum volatility portfolio."""
        n_assets = len(cov_matrix)
        cov = cov_matrix.values

        def portfolio_volatility(weights):
            return np.sqrt(weights @ cov @ weights)

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        opt_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = self._get_bounds(n_assets, constraints, assets)

        # Optimize
        result = minimize(
            portfolio_volatility,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=opt_constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        return result.x if result.success else x0

    def _max_sharpe(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Tuple[float, float]]],
        assets: List[str],
    ) -> np.ndarray:
        """Maximum Sharpe ratio portfolio."""
        n_assets = len(cov_matrix)
        cov = cov_matrix.values
        mu = mean_returns.values

        def neg_sharpe(weights):
            port_return = weights @ mu
            port_vol = np.sqrt(weights @ cov @ weights)
            if port_vol == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_vol

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        opt_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = self._get_bounds(n_assets, constraints, assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=opt_constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        return result.x if result.success else x0

    def _max_diversification(
        self,
        volatilities: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Tuple[float, float]]],
        assets: List[str],
    ) -> np.ndarray:
        """
        Maximum diversification portfolio.

        Maximize ratio of weighted average volatility to portfolio volatility.
        """
        n_assets = len(cov_matrix)
        cov = cov_matrix.values
        vols = volatilities.values

        def neg_diversification_ratio(weights):
            weighted_vol = weights @ vols
            port_vol = np.sqrt(weights @ cov @ weights)
            if port_vol == 0:
                return 0
            return -weighted_vol / port_vol

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        opt_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # Bounds
        bounds = self._get_bounds(n_assets, constraints, assets)

        # Optimize
        result = minimize(
            neg_diversification_ratio,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=opt_constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        return result.x if result.success else x0

    def _get_bounds(
        self,
        n_assets: int,
        constraints: Optional[Dict[str, Tuple[float, float]]],
        assets: List[str],
    ) -> List[Tuple[float, float]]:
        """Get weight bounds for each asset."""
        if constraints is None:
            return [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        bounds = []
        for asset in assets:
            if asset in constraints:
                bounds.append(constraints[asset])
            else:
                bounds.append((self.min_weight, self.max_weight))
        return bounds

    def _calculate_metrics(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        volatilities: np.ndarray,
        returns: pd.DataFrame,
    ) -> PortfolioMetrics:
        """Calculate portfolio metrics."""
        # Expected return
        expected_return = weights @ mean_returns

        # Volatility
        volatility = np.sqrt(weights @ cov_matrix @ weights)

        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown (historical)
        portfolio_returns = returns @ weights
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        # Diversification ratio
        weighted_vol = weights @ volatilities
        diversification_ratio = weighted_vol / volatility if volatility > 0 else 1

        # Effective N (Herfindahl index inverse)
        effective_n = 1.0 / np.sum(weights**2)

        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            diversification_ratio=diversification_ratio,
            effective_n=effective_n,
        )

    def analyze_correlations(
        self,
        returns: pd.DataFrame,
        rolling_window: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between assets.

        Args:
            returns: DataFrame of asset returns
            rolling_window: Window for rolling correlation

        Returns:
            Dictionary with correlation analysis
        """
        corr_matrix = returns.corr()

        # Get correlation pairs
        pairs = []
        n = len(corr_matrix.columns)
        for i in range(n):
            for j in range(i + 1, n):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                pairs.append(
                    {
                        "asset1": asset1,
                        "asset2": asset2,
                        "correlation": round(corr, 4),
                    }
                )

        # Sort by correlation
        pairs.sort(key=lambda x: x["correlation"])

        # Calculate average correlation
        if pairs:
            avg_corr = np.mean([p["correlation"] for p in pairs])
        else:
            avg_corr = 0

        # Rolling correlations (average)
        if len(returns.columns) >= 2:
            rolling_corr = returns.iloc[:, 0].rolling(rolling_window).corr(returns.iloc[:, 1])
            rolling_stats = {
                "mean": float(rolling_corr.mean()),
                "std": float(rolling_corr.std()),
                "current": float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else 0,
            }
        else:
            rolling_stats = {"mean": 0, "std": 0, "current": 0}

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "pairs": pairs,
            "lowest_correlation": pairs[0] if pairs else None,
            "highest_correlation": pairs[-1] if pairs else None,
            "average_correlation": round(avg_corr, 4),
            "rolling_correlation": rolling_stats,
        }

    def suggest_rebalancing(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05,
        current_prices: Optional[Dict[str, float]] = None,
        portfolio_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Suggest rebalancing trades.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Minimum deviation to trigger rebalancing
            current_prices: Current asset prices (optional)
            portfolio_value: Total portfolio value (optional)

        Returns:
            Rebalancing suggestions
        """
        trades = []
        needs_rebalancing = False

        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            diff = target - current

            if abs(diff) >= threshold:
                needs_rebalancing = True
                action = "BUY" if diff > 0 else "SELL"

                trade = {
                    "asset": asset,
                    "action": action,
                    "weight_change": round(diff, 4),
                    "current_weight": round(current, 4),
                    "target_weight": round(target, 4),
                }

                # Add dollar amounts if prices provided
                if current_prices and portfolio_value and asset in current_prices:
                    amount = abs(diff) * portfolio_value
                    quantity = amount / current_prices[asset]
                    trade["amount_usd"] = round(amount, 2)
                    trade["quantity"] = round(quantity, 6)

                trades.append(trade)

        return {
            "needs_rebalancing": needs_rebalancing,
            "trades": trades,
            "total_turnover": sum(abs(t["weight_change"]) for t in trades) / 2,
        }


class MultiAssetPortfolioManager:
    """
    High-level portfolio manager for multi-asset trading.

    Combines portfolio optimization with position management.
    """

    def __init__(
        self,
        optimizer: Optional[PortfolioOptimizer] = None,
        rebalance_threshold: float = 0.05,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
    ):
        """
        Initialize portfolio manager.

        Args:
            optimizer: Portfolio optimizer instance
            rebalance_threshold: Minimum weight deviation to trigger rebalance
            rebalance_frequency: How often to check for rebalancing
        """
        self.optimizer = optimizer or PortfolioOptimizer()
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency = rebalance_frequency
        self.current_allocation: Optional[AllocationResult] = None
        self.allocation_history: List[AllocationResult] = []
        self.last_rebalance: Optional[datetime] = None

    def update_allocation(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        force: bool = False,
    ) -> Optional[AllocationResult]:
        """
        Update portfolio allocation.

        Args:
            returns: Recent returns DataFrame
            method: Optimization method
            force: Force reallocation regardless of timing

        Returns:
            New allocation if updated, None otherwise
        """
        # Check if rebalancing is needed based on timing
        if not force and not self._should_rebalance():
            return None

        # Run optimization
        allocation = self.optimizer.optimize(returns, method)

        # Store result
        self.current_allocation = allocation
        self.allocation_history.append(allocation)
        self.last_rebalance = datetime.now()

        return allocation

    def _should_rebalance(self) -> bool:
        """Check if it's time to rebalance based on frequency."""
        if self.last_rebalance is None:
            return True

        now = datetime.now()
        elapsed = (now - self.last_rebalance).days

        if self.rebalance_frequency == "daily":
            return elapsed >= 1
        elif self.rebalance_frequency == "weekly":
            return elapsed >= 7
        elif self.rebalance_frequency == "monthly":
            return elapsed >= 30
        return True

    def get_target_positions(
        self,
        portfolio_value: float,
        prices: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get target positions based on current allocation.

        Args:
            portfolio_value: Total portfolio value
            prices: Current prices for each asset

        Returns:
            Dictionary with target positions
        """
        if not self.current_allocation:
            return {}

        positions = {}
        for asset, weight in self.current_allocation.weights.items():
            if asset in prices and weight > 0:
                target_value = portfolio_value * weight
                target_quantity = target_value / prices[asset]
                positions[asset] = {
                    "weight": weight,
                    "target_value": round(target_value, 2),
                    "target_quantity": round(target_quantity, 6),
                    "price": prices[asset],
                }

        return positions

    def calculate_portfolio_return(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate portfolio returns using current weights.

        Args:
            returns: Asset returns DataFrame

        Returns:
            Portfolio returns Series
        """
        if not self.current_allocation:
            # Equal weight if no allocation
            return returns.mean(axis=1)

        weights = []
        for col in returns.columns:
            weights.append(self.current_allocation.weights.get(col, 0))

        weights = np.array(weights)
        weights = weights / weights.sum()  # Ensure sums to 1

        return (returns * weights).sum(axis=1)


def run_portfolio_optimization_example():
    """Example usage of portfolio optimizer."""
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    # Simulate correlated returns
    n_assets = 5
    symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]

    # Create correlation structure
    corr = np.array(
        [
            [1.0, 0.7, 0.5, 0.5, 0.6],
            [0.7, 1.0, 0.6, 0.5, 0.5],
            [0.5, 0.6, 1.0, 0.7, 0.6],
            [0.5, 0.5, 0.7, 1.0, 0.5],
            [0.6, 0.5, 0.6, 0.5, 1.0],
        ]
    )

    # Cholesky decomposition for correlated random numbers
    L = np.linalg.cholesky(corr)

    # Generate returns
    daily_vol = np.array([0.04, 0.05, 0.06, 0.07, 0.06])  # Daily volatility
    daily_mean = np.array([0.001, 0.0008, 0.0005, 0.0003, 0.0004])  # Daily mean

    random_returns = np.random.randn(len(dates), n_assets)
    correlated_returns = random_returns @ L.T
    returns_data = correlated_returns * daily_vol + daily_mean

    returns = pd.DataFrame(returns_data, index=dates, columns=symbols)

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 60)
    print(f"\nAssets: {symbols}")
    print(f"Data period: {dates[0].date()} to {dates[-1].date()}")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        risk_free_rate=0.02,
        min_weight=0.05,  # Min 5% per asset
        max_weight=0.40,  # Max 40% per asset
    )

    # Run different optimization methods
    methods = [
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.INVERSE_VOLATILITY,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.MIN_VOLATILITY,
        OptimizationMethod.MAX_SHARPE,
        OptimizationMethod.MAX_DIVERSIFICATION,
    ]

    results = []
    for method in methods:
        result = optimizer.optimize(returns, method)
        results.append(result)
        result.print_summary()

    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    corr_analysis = optimizer.analyze_correlations(returns)
    print(f"\nAverage Correlation: {corr_analysis['average_correlation']:.2f}")
    print(f"Lowest Pair: {corr_analysis['lowest_correlation']}")
    print(f"Highest Pair: {corr_analysis['highest_correlation']}")

    # Best result by Sharpe
    best_result = max(results, key=lambda r: r.metrics.sharpe_ratio)
    print("\n" + "=" * 60)
    print(f"BEST STRATEGY: {best_result.method.value}")
    print(f"Sharpe Ratio: {best_result.metrics.sharpe_ratio:.2f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_portfolio_optimization_example()
