"""
Cross-Market Correlation Analysis.

Analyzes correlations between crypto, stocks, and commodities
to identify hedging opportunities and diversification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime types."""
    HIGH_POSITIVE = "high_positive"  # > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    LOW = "low"  # -0.3 to 0.3
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    HIGH_NEGATIVE = "high_negative"  # < -0.7


@dataclass
class PairCorrelation:
    """Correlation between two assets."""
    asset1: str
    asset2: str
    correlation: float
    p_value: float
    regime: CorrelationRegime
    rolling_correlation: List[float]  # Historical rolling correlation
    is_significant: bool
    market_type1: str
    market_type2: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "regime": self.regime.value,
            "rolling_correlation": self.rolling_correlation[-20:],  # Last 20 values
            "is_significant": self.is_significant,
            "market_type1": self.market_type1,
            "market_type2": self.market_type2,
            "is_cross_market": self.market_type1 != self.market_type2,
        }


@dataclass
class DiversificationScore:
    """Portfolio diversification metrics."""
    overall_score: float  # 0-100, higher = more diversified
    effective_assets: float  # Number of "effective" uncorrelated assets
    max_correlation: float
    avg_correlation: float
    risk_concentration: float
    improvement_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "effective_assets": self.effective_assets,
            "max_correlation": self.max_correlation,
            "avg_correlation": self.avg_correlation,
            "risk_concentration": self.risk_concentration,
            "improvement_suggestions": self.improvement_suggestions,
        }


@dataclass
class CorrelationShift:
    """Detected correlation regime shift."""
    asset1: str
    asset2: str
    old_regime: CorrelationRegime
    new_regime: CorrelationRegime
    old_correlation: float
    new_correlation: float
    shift_magnitude: float
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "old_regime": self.old_regime.value,
            "new_regime": self.new_regime.value,
            "old_correlation": self.old_correlation,
            "new_correlation": self.new_correlation,
            "shift_magnitude": self.shift_magnitude,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class HedgeOpportunity:
    """Identified hedging opportunity."""
    primary_asset: str
    hedge_asset: str
    correlation: float
    hedge_ratio: float  # Optimal hedge ratio
    expected_variance_reduction: float
    cost_estimate: float  # Estimated hedging cost
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_asset": self.primary_asset,
            "hedge_asset": self.hedge_asset,
            "correlation": self.correlation,
            "hedge_ratio": self.hedge_ratio,
            "expected_variance_reduction": self.expected_variance_reduction,
            "cost_estimate": self.cost_estimate,
            "recommendation": self.recommendation,
        }


class CrossMarketAnalyzer:
    """
    Analyzes correlations across different market types.

    Features:
    - Cross-market correlation matrix
    - Rolling correlation tracking
    - Correlation regime detection
    - Hedge opportunity identification
    - Diversification scoring

    Usage:
        analyzer = CrossMarketAnalyzer()

        # Add return data for assets
        analyzer.add_returns("BTC/USDT", returns_btc, "crypto")
        analyzer.add_returns("AAPL", returns_aapl, "stock")
        analyzer.add_returns("XAU/USD", returns_gold, "commodity")

        # Get correlation matrix
        matrix = analyzer.get_correlation_matrix()

        # Find hedge opportunities
        hedges = analyzer.find_hedge_opportunities("BTC/USDT")

        # Get diversification score
        score = analyzer.get_diversification_score(portfolio_weights)
    """

    def __init__(
        self,
        rolling_window: int = 30,
        min_correlation_period: int = 20,
        significance_level: float = 0.05,
        data_dir: str = "data/correlation_analysis",
    ):
        self.rolling_window = rolling_window
        self.min_correlation_period = min_correlation_period
        self.significance_level = significance_level
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Store returns data for each asset
        self._returns: Dict[str, pd.Series] = {}
        self._market_types: Dict[str, str] = {}

        # Cache correlation results
        self._correlation_cache: Dict[str, PairCorrelation] = {}
        self._last_correlation_update: Optional[datetime] = None

        # Track correlation shifts
        self._historical_correlations: Dict[str, List[float]] = {}
        self._correlation_shifts: List[CorrelationShift] = []

    def add_returns(
        self,
        symbol: str,
        returns: pd.Series,
        market_type: str,
    ) -> None:
        """
        Add or update returns data for an asset.

        Args:
            symbol: Asset symbol
            returns: Returns time series (indexed by datetime)
            market_type: Type of market (crypto, stock, commodity)
        """
        self._returns[symbol] = returns
        self._market_types[symbol] = market_type

        # Invalidate correlation cache
        self._correlation_cache.clear()

    def add_price_data(
        self,
        symbol: str,
        prices: pd.Series,
        market_type: str,
    ) -> None:
        """
        Add price data and calculate returns.

        Args:
            symbol: Asset symbol
            prices: Price time series
            market_type: Type of market
        """
        returns = prices.pct_change().dropna()
        self.add_returns(symbol, returns, market_type)

    def _get_correlation_regime(self, corr: float) -> CorrelationRegime:
        """Classify correlation into a regime."""
        if corr > 0.7:
            return CorrelationRegime.HIGH_POSITIVE
        elif corr > 0.3:
            return CorrelationRegime.MODERATE_POSITIVE
        elif corr > -0.3:
            return CorrelationRegime.LOW
        elif corr > -0.7:
            return CorrelationRegime.MODERATE_NEGATIVE
        else:
            return CorrelationRegime.HIGH_NEGATIVE

    def calculate_pair_correlation(
        self,
        symbol1: str,
        symbol2: str,
    ) -> Optional[PairCorrelation]:
        """
        Calculate correlation between two assets.

        Returns:
            PairCorrelation object or None if insufficient data
        """
        if symbol1 not in self._returns or symbol2 not in self._returns:
            return None

        # Align data
        r1 = self._returns[symbol1]
        r2 = self._returns[symbol2]

        # Find common index
        common_idx = r1.index.intersection(r2.index)
        if len(common_idx) < self.min_correlation_period:
            return None

        aligned_r1 = r1.loc[common_idx]
        aligned_r2 = r2.loc[common_idx]

        # Calculate correlation and p-value
        corr, p_value = stats.pearsonr(aligned_r1, aligned_r2)

        # Calculate rolling correlation
        combined = pd.DataFrame({"r1": aligned_r1, "r2": aligned_r2})
        rolling_corr = combined["r1"].rolling(self.rolling_window).corr(combined["r2"])
        rolling_list = rolling_corr.dropna().tolist()

        regime = self._get_correlation_regime(corr)
        is_significant = p_value < self.significance_level

        return PairCorrelation(
            asset1=symbol1,
            asset2=symbol2,
            correlation=corr,
            p_value=p_value,
            regime=regime,
            rolling_correlation=rolling_list,
            is_significant=is_significant,
            market_type1=self._market_types.get(symbol1, "unknown"),
            market_type2=self._market_types.get(symbol2, "unknown"),
        )

    def get_correlation_matrix(
        self,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get correlation matrix for all or specified assets.

        Returns:
            DataFrame with correlation matrix
        """
        if symbols is None:
            symbols = list(self._returns.keys())

        n = len(symbols)
        matrix = np.zeros((n, n))

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    pair_corr = self.calculate_pair_correlation(sym1, sym2)
                    if pair_corr:
                        matrix[i, j] = pair_corr.correlation
                        matrix[j, i] = pair_corr.correlation
                    else:
                        matrix[i, j] = np.nan
                        matrix[j, i] = np.nan

        return pd.DataFrame(matrix, index=symbols, columns=symbols)

    def get_cross_market_correlations(self) -> List[PairCorrelation]:
        """
        Get correlations between assets in different markets.

        Returns:
            List of PairCorrelation for cross-market pairs
        """
        cross_market = []
        symbols = list(self._returns.keys())

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i + 1:], start=i + 1):
                mkt1 = self._market_types.get(sym1)
                mkt2 = self._market_types.get(sym2)

                if mkt1 != mkt2:
                    pair_corr = self.calculate_pair_correlation(sym1, sym2)
                    if pair_corr:
                        cross_market.append(pair_corr)

        # Sort by absolute correlation
        cross_market.sort(key=lambda x: abs(x.correlation), reverse=True)

        return cross_market

    def detect_correlation_shifts(
        self,
        threshold: float = 0.3,
    ) -> List[CorrelationShift]:
        """
        Detect significant changes in correlations.

        Args:
            threshold: Minimum change to consider as a shift

        Returns:
            List of detected correlation shifts
        """
        shifts = []
        symbols = list(self._returns.keys())

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i + 1:], start=i + 1):
                pair_corr = self.calculate_pair_correlation(sym1, sym2)
                if not pair_corr or len(pair_corr.rolling_correlation) < 10:
                    continue

                rolling = pair_corr.rolling_correlation

                # Compare recent vs older correlation
                recent_corr = np.mean(rolling[-5:])
                older_corr = np.mean(rolling[-15:-5])

                shift_magnitude = abs(recent_corr - older_corr)

                if shift_magnitude >= threshold:
                    old_regime = self._get_correlation_regime(older_corr)
                    new_regime = self._get_correlation_regime(recent_corr)

                    if old_regime != new_regime:
                        shift = CorrelationShift(
                            asset1=sym1,
                            asset2=sym2,
                            old_regime=old_regime,
                            new_regime=new_regime,
                            old_correlation=older_corr,
                            new_correlation=recent_corr,
                            shift_magnitude=shift_magnitude,
                        )
                        shifts.append(shift)

        self._correlation_shifts.extend(shifts)
        return shifts

    def find_hedge_opportunities(
        self,
        primary_asset: str,
        min_negative_correlation: float = -0.3,
    ) -> List[HedgeOpportunity]:
        """
        Find assets that can hedge the primary asset.

        Args:
            primary_asset: Asset to hedge
            min_negative_correlation: Minimum negative correlation for hedging

        Returns:
            List of hedge opportunities sorted by effectiveness
        """
        if primary_asset not in self._returns:
            return []

        opportunities = []
        primary_returns = self._returns[primary_asset]
        primary_volatility = primary_returns.std()

        for symbol in self._returns:
            if symbol == primary_asset:
                continue

            pair_corr = self.calculate_pair_correlation(primary_asset, symbol)
            if not pair_corr or not pair_corr.is_significant:
                continue

            corr = pair_corr.correlation

            # Look for negative or low correlation
            if corr > min_negative_correlation:
                continue

            hedge_returns = self._returns[symbol]
            hedge_volatility = hedge_returns.std()

            # Calculate optimal hedge ratio (minimum variance hedge)
            # h* = -Cov(primary, hedge) / Var(hedge)
            common_idx = primary_returns.index.intersection(hedge_returns.index)
            if len(common_idx) < self.min_correlation_period:
                continue

            aligned_primary = primary_returns.loc[common_idx]
            aligned_hedge = hedge_returns.loc[common_idx]

            covariance = aligned_primary.cov(aligned_hedge)
            hedge_variance = aligned_hedge.var()

            if hedge_variance == 0:
                continue

            hedge_ratio = -covariance / hedge_variance

            # Calculate expected variance reduction
            # Var(hedged) = Var(primary) + h^2 * Var(hedge) + 2*h*Cov
            hedged_variance = (
                primary_volatility ** 2 +
                hedge_ratio ** 2 * hedge_variance +
                2 * hedge_ratio * covariance
            )

            variance_reduction = 1 - (hedged_variance / (primary_volatility ** 2))
            variance_reduction = max(0, min(1, variance_reduction))

            # Estimate hedging cost (simplified - assume 0.1% per trade)
            cost_estimate = abs(hedge_ratio) * 0.001

            # Generate recommendation
            if corr < -0.5 and variance_reduction > 0.3:
                recommendation = "Strong hedge - highly recommended"
            elif corr < -0.3 and variance_reduction > 0.2:
                recommendation = "Good hedge - recommended"
            elif variance_reduction > 0.1:
                recommendation = "Moderate hedge - consider for diversification"
            else:
                recommendation = "Weak hedge - limited benefit"

            opportunity = HedgeOpportunity(
                primary_asset=primary_asset,
                hedge_asset=symbol,
                correlation=corr,
                hedge_ratio=hedge_ratio,
                expected_variance_reduction=variance_reduction,
                cost_estimate=cost_estimate,
                recommendation=recommendation,
            )
            opportunities.append(opportunity)

        # Sort by variance reduction
        opportunities.sort(key=lambda x: x.expected_variance_reduction, reverse=True)

        return opportunities

    def get_diversification_score(
        self,
        portfolio_weights: Dict[str, float],
    ) -> DiversificationScore:
        """
        Calculate diversification score for a portfolio.

        Args:
            portfolio_weights: Dict of symbol -> weight

        Returns:
            DiversificationScore with metrics and suggestions
        """
        symbols = [s for s in portfolio_weights if s in self._returns]

        if len(symbols) < 2:
            return DiversificationScore(
                overall_score=0.0,
                effective_assets=len(symbols),
                max_correlation=0.0,
                avg_correlation=0.0,
                risk_concentration=1.0,
                improvement_suggestions=["Add more assets for diversification"],
            )

        # Get correlation matrix
        corr_matrix = self.get_correlation_matrix(symbols)

        # Calculate metrics
        correlations = []
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i + 1:], start=i + 1):
                c = corr_matrix.loc[sym1, sym2]
                if not np.isnan(c):
                    correlations.append(c)

        if not correlations:
            return DiversificationScore(
                overall_score=50.0,
                effective_assets=len(symbols),
                max_correlation=0.0,
                avg_correlation=0.0,
                risk_concentration=1.0 / len(symbols),
                improvement_suggestions=["Insufficient correlation data"],
            )

        max_corr = max(correlations)
        avg_corr = np.mean(correlations)

        # Calculate effective number of assets
        # Using entropy-based measure
        weights = np.array([portfolio_weights.get(s, 0) for s in symbols])
        weights = weights / weights.sum()  # Normalize

        # Effective N = exp(entropy)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        effective_n = np.exp(entropy)

        # Risk concentration (Herfindahl index)
        risk_concentration = np.sum(weights ** 2)

        # Overall score (0-100)
        # Penalize high correlation and concentration
        correlation_penalty = max(0, (avg_corr - 0.3) * 50)
        concentration_penalty = max(0, (risk_concentration - 0.2) * 100)
        cross_market_bonus = self._calculate_cross_market_bonus(symbols) * 20

        overall_score = max(0, min(100, 70 - correlation_penalty - concentration_penalty + cross_market_bonus))

        # Generate suggestions
        suggestions = []

        if avg_corr > 0.5:
            suggestions.append("Portfolio is highly correlated - add negatively correlated assets")

        if risk_concentration > 0.4:
            suggestions.append("Position sizes are concentrated - consider more equal weighting")

        # Check market diversification
        market_types = [self._market_types.get(s) for s in symbols]
        unique_markets = len(set(market_types))
        if unique_markets < 3:
            missing_markets = {"crypto", "stock", "commodity"} - set(market_types)
            if missing_markets:
                suggestions.append(f"Consider adding assets from: {', '.join(missing_markets)}")

        # Find highly correlated pairs
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i + 1:], start=i + 1):
                c = corr_matrix.loc[sym1, sym2]
                if not np.isnan(c) and c > 0.8:
                    suggestions.append(f"High correlation ({c:.2f}) between {sym1} and {sym2}")

        return DiversificationScore(
            overall_score=overall_score,
            effective_assets=effective_n,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
            risk_concentration=risk_concentration,
            improvement_suggestions=suggestions[:5],  # Limit to top 5
        )

    def _calculate_cross_market_bonus(self, symbols: List[str]) -> float:
        """Calculate bonus for cross-market diversification."""
        markets = [self._market_types.get(s) for s in symbols]
        unique_markets = len(set(m for m in markets if m))
        return min(1.0, (unique_markets - 1) / 2)

    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get summary of cross-market correlations.

        Returns:
            Dict with market summary statistics
        """
        cross_market = self.get_cross_market_correlations()

        # Group by market pair
        market_pairs: Dict[Tuple[str, str], List[float]] = {}

        for pc in cross_market:
            pair = tuple(sorted([pc.market_type1, pc.market_type2]))
            if pair not in market_pairs:
                market_pairs[pair] = []
            market_pairs[pair].append(pc.correlation)

        # Calculate average correlation by market pair
        pair_summaries = {}
        for pair, correlations in market_pairs.items():
            pair_summaries[f"{pair[0]}_vs_{pair[1]}"] = {
                "avg_correlation": np.mean(correlations),
                "min_correlation": min(correlations),
                "max_correlation": max(correlations),
                "num_pairs": len(correlations),
            }

        return {
            "total_assets": len(self._returns),
            "total_cross_market_pairs": len(cross_market),
            "market_pair_correlations": pair_summaries,
            "recent_shifts": [s.to_dict() for s in self._correlation_shifts[-5:]],
        }

    def save_analysis(self, filename: str = "correlation_analysis.json") -> None:
        """Save correlation analysis to disk."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "market_summary": self.get_market_summary(),
            "cross_market_correlations": [
                pc.to_dict() for pc in self.get_cross_market_correlations()
            ],
        }

        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved correlation analysis to {filepath}")
