"""
Factor Analysis Module.

Provides performance attribution to market factors
such as momentum, volatility, trend, and sentiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Exposure to a single factor."""
    name: str
    beta: float  # Factor loading/beta
    contribution: float  # Return contribution
    t_stat: float  # Statistical significance
    p_value: float
    is_significant: bool  # p < 0.05


@dataclass
class FactorAnalysisResult:
    """Complete factor analysis results."""
    timestamp: datetime
    total_return: float
    factor_explained_return: float
    alpha: float  # Unexplained return
    r_squared: float  # Factor model fit
    adjusted_r_squared: float
    factors: List[FactorExposure]
    factor_contributions: Dict[str, float]
    residual_analysis: Dict[str, float]


class FactorAnalyzer:
    """
    Analyzes portfolio returns using factor models.

    Decomposes returns into factor contributions and
    calculates alpha (unexplained returns).
    """

    # Standard trading factors
    FACTORS = [
        "momentum",
        "volatility",
        "trend",
        "mean_reversion",
        "volume",
        "sentiment",
        "market",
    ]

    def __init__(self):
        self._returns: np.ndarray = np.array([])
        self._factor_data: Dict[str, np.ndarray] = {}
        self._timestamps: List[datetime] = []

    def load_data(
        self,
        returns: List[float],
        factor_data: Dict[str, List[float]],
        timestamps: Optional[List[datetime]] = None,
    ) -> None:
        """
        Load returns and factor data.

        Args:
            returns: Portfolio returns
            factor_data: Dictionary of factor name to factor returns
            timestamps: Optional timestamps
        """
        self._returns = np.array(returns)
        self._factor_data = {k: np.array(v) for k, v in factor_data.items()}
        self._timestamps = timestamps or []

    def calculate_factors_from_prices(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate standard factors from price data.

        Args:
            prices: Array of prices
            volumes: Optional array of volumes

        Returns:
            Dictionary of factor returns
        """
        if len(prices) < 20:
            return {}

        factors = {}

        # Returns
        returns = np.diff(prices) / prices[:-1]

        # Momentum factor (past 20-period return)
        momentum = np.zeros(len(returns))
        for i in range(20, len(returns)):
            momentum[i] = np.mean(returns[i - 20:i])
        factors["momentum"] = momentum[1:]  # Align with returns

        # Volatility factor (rolling 20-period std)
        volatility = np.zeros(len(returns))
        for i in range(20, len(returns)):
            volatility[i] = np.std(returns[i - 20:i])
        factors["volatility"] = volatility[1:]

        # Trend factor (price vs 20-period SMA)
        sma = np.convolve(prices, np.ones(20) / 20, mode="valid")
        trend = np.zeros(len(returns))
        for i in range(19, len(prices) - 1):
            trend[i] = (prices[i] - sma[i - 19]) / sma[i - 19] if sma[i - 19] > 0 else 0
        factors["trend"] = trend[1:]

        # Mean reversion factor (distance from 20-period mean, inverted)
        mean_rev = -factors["trend"]
        factors["mean_reversion"] = mean_rev

        # Volume factor (if provided)
        if volumes is not None and len(volumes) > 20:
            vol_sma = np.convolve(volumes, np.ones(20) / 20, mode="valid")
            volume_factor = np.zeros(len(returns))
            for i in range(19, len(volumes) - 1):
                if vol_sma[i - 19] > 0:
                    volume_factor[i] = (volumes[i] - vol_sma[i - 19]) / vol_sma[i - 19]
            factors["volume"] = volume_factor[1:]
        else:
            factors["volume"] = np.zeros(len(factors["momentum"]))

        # Market factor (just the return itself for single asset)
        factors["market"] = returns[20:]

        # Sentiment factor (placeholder - would come from news/social data)
        factors["sentiment"] = np.zeros(len(factors["momentum"]))

        return factors

    def run_analysis(self) -> FactorAnalysisResult:
        """
        Run factor analysis on loaded data.

        Returns:
            FactorAnalysisResult with factor exposures and contributions
        """
        if len(self._returns) < 30:
            raise ValueError("Insufficient data for factor analysis")

        # Align data lengths
        min_len = min(len(self._returns), min(len(f) for f in self._factor_data.values()))
        returns = self._returns[-min_len:]

        # Build factor matrix
        factor_names = list(self._factor_data.keys())
        X = np.column_stack([self._factor_data[f][-min_len:] for f in factor_names])

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(returns)), X])

        # Run regression
        try:
            # OLS regression
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, returns, rcond=None)

            # Calculate predictions
            predictions = X_with_const @ beta

            # Calculate R-squared
            ss_res = np.sum((returns - predictions) ** 2)
            ss_tot = np.sum((returns - np.mean(returns)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Adjusted R-squared
            n = len(returns)
            p = len(factor_names)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else 0

            # Calculate t-statistics and p-values
            mse = ss_res / (n - p - 1) if n > p + 1 else 0
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se_beta = np.sqrt(np.maximum(var_beta, 1e-10))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

        except Exception as e:
            logger.error(f"Regression failed: {e}")
            # Return empty result
            return FactorAnalysisResult(
                timestamp=datetime.now(),
                total_return=float(np.sum(returns)) * 100,
                factor_explained_return=0,
                alpha=float(np.sum(returns)) * 100,
                r_squared=0,
                adjusted_r_squared=0,
                factors=[],
                factor_contributions={},
                residual_analysis={},
            )

        # Extract alpha (intercept)
        alpha = beta[0]

        # Build factor exposures
        factors = []
        factor_contributions = {}

        for i, name in enumerate(factor_names):
            factor_beta = beta[i + 1]
            factor_returns = self._factor_data[name][-min_len:]
            contribution = factor_beta * np.sum(factor_returns)

            factor_contributions[name] = round(contribution * 100, 2)

            factors.append(FactorExposure(
                name=name,
                beta=round(float(factor_beta), 4),
                contribution=round(float(contribution) * 100, 2),
                t_stat=round(float(t_stats[i + 1]), 2),
                p_value=round(float(p_values[i + 1]), 4),
                is_significant=float(p_values[i + 1]) < 0.05,
            ))

        # Residual analysis
        residuals = returns - predictions
        residual_analysis = {
            "mean": round(float(np.mean(residuals)) * 100, 4),
            "std": round(float(np.std(residuals)) * 100, 4),
            "skewness": round(float(stats.skew(residuals)), 2),
            "kurtosis": round(float(stats.kurtosis(residuals)), 2),
            "autocorrelation": round(float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]), 3) if len(residuals) > 1 else 0,
        }

        total_return = float(np.sum(returns)) * 100
        factor_explained = total_return - (alpha * len(returns) * 100)

        return FactorAnalysisResult(
            timestamp=datetime.now(),
            total_return=round(total_return, 2),
            factor_explained_return=round(factor_explained, 2),
            alpha=round(alpha * 100, 4),  # Annualized would need *252
            r_squared=round(r_squared, 4),
            adjusted_r_squared=round(adj_r_squared, 4),
            factors=factors,
            factor_contributions=factor_contributions,
            residual_analysis=residual_analysis,
        )

    def get_factor_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between factors."""
        if not self._factor_data:
            return {}

        # Align lengths
        min_len = min(len(f) for f in self._factor_data.values())
        aligned_factors = {k: v[-min_len:] for k, v in self._factor_data.items()}

        factor_names = list(aligned_factors.keys())
        correlations = {}

        for f1 in factor_names:
            correlations[f1] = {}
            for f2 in factor_names:
                corr = np.corrcoef(aligned_factors[f1], aligned_factors[f2])[0, 1]
                correlations[f1][f2] = round(float(corr), 3) if not np.isnan(corr) else 0

        return correlations

    def get_rolling_factor_exposure(
        self, window: int = 60
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Calculate rolling factor exposures over time.

        Args:
            window: Rolling window size

        Returns:
            Dictionary of factor name to list of (timestamp, exposure) pairs
        """
        if len(self._returns) < window + 10:
            return {}

        min_len = min(len(self._returns), min(len(f) for f in self._factor_data.values()))
        returns = self._returns[-min_len:]

        rolling_exposures = {f: [] for f in self._factor_data.keys()}

        for i in range(window, len(returns)):
            window_returns = returns[i - window:i]

            for factor_name, factor_data in self._factor_data.items():
                window_factor = factor_data[-min_len:][i - window:i]

                if len(window_returns) > 2 and len(window_factor) > 2:
                    try:
                        slope, _, _, _, _ = stats.linregress(window_factor, window_returns)
                        rolling_exposures[factor_name].append({
                            "index": i,
                            "timestamp": self._timestamps[i].isoformat() if i < len(self._timestamps) else None,
                            "exposure": round(float(slope), 4),
                        })
                    except Exception:
                        pass

        return rolling_exposures

    def to_api_response(self, result: FactorAnalysisResult) -> Dict[str, Any]:
        """Convert result to API response format."""
        return {
            "summary": {
                "total_return": result.total_return,
                "factor_explained": result.factor_explained_return,
                "alpha": result.alpha,
                "r_squared": result.r_squared,
                "adjusted_r_squared": result.adjusted_r_squared,
            },
            "factors": [
                {
                    "name": f.name,
                    "beta": f.beta,
                    "contribution": f.contribution,
                    "t_stat": f.t_stat,
                    "p_value": f.p_value,
                    "is_significant": f.is_significant,
                }
                for f in sorted(result.factors, key=lambda x: abs(x.contribution), reverse=True)
            ],
            "contributions": result.factor_contributions,
            "residuals": result.residual_analysis,
            "correlations": self.get_factor_correlations(),
            "timestamp": result.timestamp.isoformat(),
        }
