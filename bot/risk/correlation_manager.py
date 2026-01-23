"""
Cross-Market Correlation Manager - Manage correlated positions.

Tracks correlations between assets and limits exposure to
correlated positions to reduce portfolio risk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Correlation between two assets."""

    asset1: str
    asset2: str
    correlation: float
    lookback_days: int
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_highly_correlated(self) -> bool:
        return abs(self.correlation) > 0.7

    @property
    def is_inverse_correlated(self) -> bool:
        return self.correlation < -0.5

    def to_dict(self) -> Dict:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "correlation": round(self.correlation, 4),
            "lookback_days": self.lookback_days,
            "is_highly_correlated": self.is_highly_correlated,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class CorrelationCluster:
    """Group of highly correlated assets."""

    cluster_id: str
    assets: List[str]
    avg_correlation: float
    max_exposure_pct: float  # Maximum combined exposure for cluster

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "assets": self.assets,
            "avg_correlation": round(self.avg_correlation, 4),
            "max_exposure_pct": self.max_exposure_pct,
        }


@dataclass
class ExposureCheck:
    """Result of exposure check."""

    allowed: bool
    reason: str
    current_cluster_exposure: float
    max_cluster_exposure: float
    correlated_positions: List[str]
    suggested_size_reduction: float  # 0-1, multiply position by this

    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "current_cluster_exposure": round(self.current_cluster_exposure, 4),
            "max_cluster_exposure": self.max_cluster_exposure,
            "correlated_positions": self.correlated_positions,
            "suggested_size_reduction": round(self.suggested_size_reduction, 2),
        }


@dataclass
class CorrelationConfig:
    """Configuration for correlation management."""

    # Correlation thresholds
    high_correlation_threshold: float = 0.7  # Considered highly correlated
    inverse_correlation_threshold: float = -0.5  # Inverse correlation

    # Exposure limits
    max_cluster_exposure_pct: float = 0.30  # Max 30% in correlated cluster
    max_single_asset_pct: float = 0.10  # Max 10% in single asset
    max_correlated_positions: int = 3  # Max positions in same cluster

    # Calculation settings
    lookback_days: int = 90
    min_data_points: int = 30
    update_interval_hours: int = 24

    # Default clusters (common correlations)
    default_clusters: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "crypto_majors": ["BTC/USDT", "ETH/USDT"],
            "crypto_alts": ["SOL/USDT", "AVAX/USDT", "DOT/USDT"],
            "tech_stocks": ["AAPL", "MSFT", "GOOGL", "NVDA"],
            "commodities": ["XAU/USD", "XAG/USD"],
            "energy": ["USOIL/USD", "UKOIL/USD", "NATGAS/USD"],
        }
    )


class CorrelationManager:
    """
    Manage cross-market correlations and exposure limits.

    Features:
    - Calculate rolling correlations between assets
    - Identify correlation clusters
    - Limit exposure to correlated positions
    - Dynamic correlation updates
    - Hedging suggestions
    """

    def __init__(
        self,
        config: Optional[CorrelationConfig] = None,
        data_dir: str = "data/correlations",
    ):
        self.config = config or CorrelationConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.clusters: Dict[str, CorrelationCluster] = {}
        self.last_update: Optional[datetime] = None

        # Initialize default clusters
        self._init_default_clusters()
        self._load_state()

    def _init_default_clusters(self):
        """Initialize default correlation clusters."""
        for cluster_id, assets in self.config.default_clusters.items():
            self.clusters[cluster_id] = CorrelationCluster(
                cluster_id=cluster_id,
                assets=assets,
                avg_correlation=0.8,  # Assumed high correlation
                max_exposure_pct=self.config.max_cluster_exposure_pct,
            )

    def _load_state(self):
        """Load saved correlation state."""
        state_file = self.data_dir / "correlation_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.correlation_matrix = state.get("correlation_matrix", {})
                    if state.get("last_update"):
                        self.last_update = datetime.fromisoformat(state["last_update"])
                logger.info("Loaded correlation state")
            except Exception as e:
                logger.error(f"Error loading correlation state: {e}")

    def _save_state(self):
        """Save correlation state."""
        state_file = self.data_dir / "correlation_state.json"
        try:
            state = {
                "correlation_matrix": self.correlation_matrix,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "clusters": {k: v.to_dict() for k, v in self.clusters.items()},
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving correlation state: {e}")

    def update_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """
        Update correlation matrix from price data.

        Args:
            price_data: Dict mapping symbol to DataFrame with 'close' column
        """
        symbols = list(price_data.keys())
        if len(symbols) < 2:
            return

        # Calculate returns
        returns = {}
        for symbol, df in price_data.items():
            if "close" in df.columns and len(df) >= self.config.min_data_points:
                returns[symbol] = df["close"].pct_change().dropna()

        # Calculate correlations
        for i, sym1 in enumerate(returns.keys()):
            if sym1 not in self.correlation_matrix:
                self.correlation_matrix[sym1] = {}

            for sym2 in list(returns.keys())[i + 1 :]:
                # Align data
                r1, r2 = returns[sym1].align(returns[sym2], join="inner")
                if len(r1) < self.config.min_data_points:
                    continue

                corr = r1.corr(r2)
                if not np.isnan(corr):
                    self.correlation_matrix[sym1][sym2] = float(corr)
                    if sym2 not in self.correlation_matrix:
                        self.correlation_matrix[sym2] = {}
                    self.correlation_matrix[sym2][sym1] = float(corr)

        self.last_update = datetime.now()
        self._update_clusters()
        self._save_state()

        logger.info(f"Updated correlations for {len(returns)} assets")

    def _update_clusters(self):
        """Update correlation clusters based on current matrix."""
        # Find highly correlated groups
        visited: Set[str] = set()

        for sym1, correlations in self.correlation_matrix.items():
            if sym1 in visited:
                continue

            # Find all highly correlated assets
            cluster_assets = [sym1]
            for sym2, corr in correlations.items():
                if abs(corr) >= self.config.high_correlation_threshold:
                    cluster_assets.append(sym2)

            if len(cluster_assets) > 1:
                # Calculate average correlation
                corrs = [
                    self.correlation_matrix.get(a, {}).get(b, 0)
                    for a in cluster_assets
                    for b in cluster_assets
                    if a != b
                ]
                avg_corr = np.mean([abs(c) for c in corrs if c]) if corrs else 0.8

                cluster_id = f"dynamic_{sym1}"
                self.clusters[cluster_id] = CorrelationCluster(
                    cluster_id=cluster_id,
                    assets=cluster_assets,
                    avg_correlation=avg_corr,
                    max_exposure_pct=self.config.max_cluster_exposure_pct,
                )
                visited.update(cluster_assets)

    def get_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Get correlation between two assets."""
        if asset1 in self.correlation_matrix:
            return self.correlation_matrix[asset1].get(asset2)
        if asset2 in self.correlation_matrix:
            return self.correlation_matrix[asset2].get(asset1)
        return None

    def get_correlated_assets(
        self, asset: str, min_correlation: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Get all assets correlated with given asset."""
        correlated = []
        if asset in self.correlation_matrix:
            for other, corr in self.correlation_matrix[asset].items():
                if abs(corr) >= min_correlation:
                    correlated.append((other, corr))

        return sorted(correlated, key=lambda x: abs(x[1]), reverse=True)

    def get_cluster_for_asset(self, asset: str) -> Optional[CorrelationCluster]:
        """Get the cluster containing given asset."""
        for cluster in self.clusters.values():
            if asset in cluster.assets:
                return cluster
        return None

    def check_exposure(
        self,
        new_asset: str,
        new_position_value: float,
        current_positions: Dict[str, float],  # symbol -> value
        portfolio_value: float,
    ) -> ExposureCheck:
        """
        Check if adding a position would exceed correlation limits.

        Args:
            new_asset: Asset to add
            new_position_value: Value of new position
            current_positions: Current positions {symbol: value}
            portfolio_value: Total portfolio value

        Returns:
            ExposureCheck with decision and details
        """
        # Check single asset limit
        new_exposure_pct = new_position_value / portfolio_value if portfolio_value > 0 else 0
        if new_exposure_pct > self.config.max_single_asset_pct:
            return ExposureCheck(
                allowed=False,
                reason=f"Single asset exposure ({new_exposure_pct:.1%}) exceeds limit ({self.config.max_single_asset_pct:.1%})",
                current_cluster_exposure=0,
                max_cluster_exposure=self.config.max_single_asset_pct,
                correlated_positions=[],
                suggested_size_reduction=self.config.max_single_asset_pct / new_exposure_pct
                if new_exposure_pct > 0
                else 1,
            )

        # Get cluster for new asset
        cluster = self.get_cluster_for_asset(new_asset)
        if not cluster:
            # Check dynamic correlations
            correlated = self.get_correlated_assets(
                new_asset, self.config.high_correlation_threshold
            )
            if correlated:
                cluster_assets = [new_asset] + [a for a, _ in correlated]
            else:
                # No correlation concerns
                return ExposureCheck(
                    allowed=True,
                    reason="No significant correlations found",
                    current_cluster_exposure=new_exposure_pct,
                    max_cluster_exposure=self.config.max_cluster_exposure_pct,
                    correlated_positions=[],
                    suggested_size_reduction=1.0,
                )
        else:
            cluster_assets = cluster.assets

        # Calculate current cluster exposure
        correlated_positions = []
        current_cluster_value = 0
        for sym, value in current_positions.items():
            if sym in cluster_assets:
                correlated_positions.append(sym)
                current_cluster_value += abs(value)

        current_cluster_exposure = (
            current_cluster_value / portfolio_value if portfolio_value > 0 else 0
        )
        new_cluster_exposure = (
            (current_cluster_value + new_position_value) / portfolio_value
            if portfolio_value > 0
            else 0
        )

        max_exposure = cluster.max_exposure_pct if cluster else self.config.max_cluster_exposure_pct

        # Check cluster exposure limit
        if new_cluster_exposure > max_exposure:
            remaining_capacity = max(0, max_exposure - current_cluster_exposure) * portfolio_value
            suggested_reduction = (
                remaining_capacity / new_position_value if new_position_value > 0 else 0
            )

            return ExposureCheck(
                allowed=False,
                reason=f"Cluster exposure ({new_cluster_exposure:.1%}) would exceed limit ({max_exposure:.1%})",
                current_cluster_exposure=current_cluster_exposure,
                max_cluster_exposure=max_exposure,
                correlated_positions=correlated_positions,
                suggested_size_reduction=suggested_reduction,
            )

        # Check max positions in cluster
        if len(correlated_positions) >= self.config.max_correlated_positions:
            return ExposureCheck(
                allowed=False,
                reason=f"Max correlated positions ({self.config.max_correlated_positions}) already reached",
                current_cluster_exposure=current_cluster_exposure,
                max_cluster_exposure=max_exposure,
                correlated_positions=correlated_positions,
                suggested_size_reduction=0,
            )

        return ExposureCheck(
            allowed=True,
            reason="Within correlation limits",
            current_cluster_exposure=current_cluster_exposure,
            max_cluster_exposure=max_exposure,
            correlated_positions=correlated_positions,
            suggested_size_reduction=1.0,
        )

    def get_hedging_suggestions(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> List[Dict]:
        """
        Suggest hedging opportunities using inverse correlations.

        Args:
            positions: Current positions {symbol: value}
            portfolio_value: Total portfolio value

        Returns:
            List of hedging suggestions
        """
        suggestions = []

        for asset, value in positions.items():
            if value <= 0:  # Only hedge long positions
                continue

            # Find inversely correlated assets
            inverse_correlated = []
            if asset in self.correlation_matrix:
                for other, corr in self.correlation_matrix[asset].items():
                    if corr <= self.config.inverse_correlation_threshold:
                        inverse_correlated.append((other, corr))

            for hedge_asset, correlation in inverse_correlated:
                exposure_pct = abs(value) / portfolio_value if portfolio_value > 0 else 0
                hedge_size = exposure_pct * abs(correlation)  # Proportional to correlation

                suggestions.append(
                    {
                        "original_position": asset,
                        "hedge_asset": hedge_asset,
                        "correlation": correlation,
                        "suggested_hedge_pct": round(hedge_size * 100, 2),
                        "reason": f"{hedge_asset} has {correlation:.2f} correlation with {asset}",
                    }
                )

        return suggestions

    def get_portfolio_correlation_risk(
        self,
        positions: Dict[str, float],
    ) -> Dict:
        """
        Analyze portfolio correlation risk.

        Returns:
            Risk analysis including concentration in correlated assets
        """
        if not positions:
            return {"status": "empty_portfolio"}

        total_value = sum(abs(v) for v in positions.values())
        if total_value == 0:
            return {"status": "zero_value"}

        # Analyze cluster concentrations
        cluster_exposures = {}
        for cluster_id, cluster in self.clusters.items():
            cluster_value = sum(abs(positions.get(asset, 0)) for asset in cluster.assets)
            if cluster_value > 0:
                cluster_exposures[cluster_id] = {
                    "value": cluster_value,
                    "pct": cluster_value / total_value,
                    "assets_held": [a for a in cluster.assets if a in positions],
                }

        # Calculate portfolio concentration
        values = list(positions.values())
        if len(values) > 1:
            herfindahl = sum((v / total_value) ** 2 for v in values if total_value > 0)
        else:
            herfindahl = 1.0

        return {
            "total_positions": len(positions),
            "total_value": round(total_value, 2),
            "cluster_exposures": cluster_exposures,
            "herfindahl_index": round(herfindahl, 4),
            "concentration_risk": "high"
            if herfindahl > 0.25
            else "medium"
            if herfindahl > 0.15
            else "low",
            "diversification_score": round((1 - herfindahl) * 100, 1),
        }

    def get_correlation_matrix_df(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get correlation matrix as DataFrame."""
        if symbols is None:
            symbols = list(self.correlation_matrix.keys())

        matrix = pd.DataFrame(index=symbols, columns=symbols, dtype=float)
        matrix = matrix.fillna(0)

        for i, sym1 in enumerate(symbols):
            matrix.loc[sym1, sym1] = 1.0
            for sym2 in symbols[i + 1 :]:
                corr = self.get_correlation(sym1, sym2)
                if corr is not None:
                    matrix.loc[sym1, sym2] = corr
                    matrix.loc[sym2, sym1] = corr

        return matrix


def create_correlation_manager(
    config: Optional[CorrelationConfig] = None,
    data_dir: str = "data/correlations",
) -> CorrelationManager:
    """Factory function to create correlation manager."""
    return CorrelationManager(config=config, data_dir=data_dir)
