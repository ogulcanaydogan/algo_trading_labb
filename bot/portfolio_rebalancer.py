"""
Portfolio Rebalancing Module.

Implements various portfolio allocation strategies including
risk parity, equal weight, and correlation-based weighting.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RebalanceConfig:
    """Configuration for portfolio rebalancing."""

    enabled: bool = True
    strategy: str = "risk_parity"  # "equal", "risk_parity", "momentum", "min_correlation"
    rebalance_threshold: float = 0.05  # Rebalance when drift > 5%
    min_rebalance_interval_hours: int = 24
    max_single_trade_pct: float = 0.10  # Max 10% of portfolio per trade
    min_position_pct: float = 0.05  # Minimum 5% per position
    max_position_pct: float = 0.30  # Maximum 30% per position


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine.

    Supports multiple allocation strategies:
    - Equal weight: Simple 1/N allocation
    - Risk parity: Weight by inverse volatility
    - Momentum: Overweight recent performers
    - Min correlation: Maximize diversification
    """

    def __init__(self, config: Optional[RebalanceConfig] = None):
        self.config = config or RebalanceConfig()
        self._last_rebalance: Optional[datetime] = None
        self._target_weights: Dict[str, float] = {}
        self._price_history: Dict[str, pd.Series] = {}

    def update_prices(self, symbol: str, prices: pd.Series) -> None:
        """Update price history for a symbol."""
        self._price_history[symbol] = prices

    def calculate_target_weights(
        self,
        symbols: List[str],
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate target portfolio weights based on strategy.

        Args:
            symbols: List of symbols to include
            current_prices: Current prices for each symbol

        Returns:
            Dict of symbol -> target weight (0-1)
        """
        if self.config.strategy == "equal":
            weights = self._equal_weight(symbols)
        elif self.config.strategy == "risk_parity":
            weights = self._risk_parity_weight(symbols)
        elif self.config.strategy == "momentum":
            weights = self._momentum_weight(symbols)
        elif self.config.strategy == "min_correlation":
            weights = self._min_correlation_weight(symbols)
        else:
            weights = self._equal_weight(symbols)

        # Apply min/max constraints
        weights = self._apply_constraints(weights)

        self._target_weights = weights
        return weights

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Simple equal weight allocation."""
        n = len(symbols)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {s: weight for s in symbols}

    def _risk_parity_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Risk parity allocation based on inverse volatility."""
        volatilities = {}

        for symbol in symbols:
            if symbol in self._price_history and len(self._price_history[symbol]) > 20:
                returns = self._price_history[symbol].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # Annualized
                volatilities[symbol] = max(vol, 0.01)  # Floor at 1%
            else:
                volatilities[symbol] = 0.30  # Default 30% vol

        # Inverse volatility weighting
        inv_vols = {s: 1.0 / v for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol == 0:
            return self._equal_weight(symbols)

        return {s: iv / total_inv_vol for s, iv in inv_vols.items()}

    def _momentum_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Momentum-based weighting (overweight recent performers)."""
        momentum_scores = {}

        for symbol in symbols:
            if symbol in self._price_history and len(self._price_history[symbol]) > 20:
                prices = self._price_history[symbol]
                # 20-day momentum
                momentum = (prices.iloc[-1] / prices.iloc[-20]) - 1
                # Normalize to positive
                momentum_scores[symbol] = max(momentum + 0.5, 0.1)
            else:
                momentum_scores[symbol] = 0.5

        total_score = sum(momentum_scores.values())
        if total_score == 0:
            return self._equal_weight(symbols)

        return {s: score / total_score for s, score in momentum_scores.items()}

    def _min_correlation_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Minimize portfolio correlation (maximize diversification)."""
        if len(symbols) < 2:
            return self._equal_weight(symbols)

        # Build returns matrix
        returns_data = {}
        for symbol in symbols:
            if symbol in self._price_history and len(self._price_history[symbol]) > 20:
                returns_data[symbol] = self._price_history[symbol].pct_change().dropna()

        if len(returns_data) < 2:
            return self._equal_weight(symbols)

        # Align data
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 20:
            return self._equal_weight(symbols)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Simple diversification score: inverse of average correlation
        div_scores = {}
        for symbol in returns_df.columns:
            avg_corr = corr_matrix[symbol].drop(symbol).mean()
            div_scores[symbol] = 1.0 - avg_corr  # Higher = less correlated

        total_score = sum(div_scores.values())
        if total_score <= 0:
            return self._equal_weight(symbols)

        weights = {s: score / total_score for s, score in div_scores.items()}

        # Add missing symbols with equal weight
        for symbol in symbols:
            if symbol not in weights:
                weights[symbol] = 1.0 / len(symbols)

        return weights

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max position constraints."""
        constrained = {}

        for symbol, weight in weights.items():
            constrained[symbol] = max(
                self.config.min_position_pct,
                min(self.config.max_position_pct, weight)
            )

        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {s: w / total for s, w in constrained.items()}

        return constrained

    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights (uses cached if not provided)

        Returns:
            Tuple of (needs_rebalance, weight_diffs)
        """
        if target_weights is None:
            target_weights = self._target_weights

        if not target_weights:
            return False, {}

        # Check time since last rebalance
        if self._last_rebalance:
            hours_since = (datetime.now() - self._last_rebalance).total_seconds() / 3600
            if hours_since < self.config.min_rebalance_interval_hours:
                return False, {}

        # Calculate drift
        diffs = {}
        max_drift = 0

        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current
            diffs[symbol] = diff
            max_drift = max(max_drift, abs(diff))

        needs_rebalance = max_drift > self.config.rebalance_threshold

        return needs_rebalance, diffs

    def generate_rebalance_orders(
        self,
        current_positions: Dict[str, Dict[str, float]],
        total_portfolio_value: float,
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Generate orders to rebalance portfolio.

        Args:
            current_positions: Dict of symbol -> {quantity, value}
            total_portfolio_value: Total portfolio value
            current_prices: Current prices

        Returns:
            List of order dicts with symbol, side, quantity
        """
        if not self._target_weights:
            return []

        # Calculate current weights
        current_weights = {
            s: pos["value"] / total_portfolio_value
            for s, pos in current_positions.items()
            if total_portfolio_value > 0
        }

        needs_rebalance, diffs = self.check_rebalance_needed(current_weights)
        if not needs_rebalance:
            return []

        orders = []

        for symbol, diff in diffs.items():
            if abs(diff) < 0.01:  # Skip tiny adjustments
                continue

            target_value = total_portfolio_value * self._target_weights.get(symbol, 0)
            current_value = current_positions.get(symbol, {}).get("value", 0)
            value_change = target_value - current_value

            # Limit single trade size
            max_trade_value = total_portfolio_value * self.config.max_single_trade_pct
            value_change = max(-max_trade_value, min(max_trade_value, value_change))

            if abs(value_change) < 10:  # Skip trades under $10
                continue

            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            quantity = abs(value_change) / price
            side = "buy" if value_change > 0 else "sell"

            orders.append({
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "value": abs(value_change),
                "reason": f"Rebalance: {diff*100:+.1f}% drift",
            })

        # Sort: sells first, then buys
        orders.sort(key=lambda x: (0 if x["side"] == "sell" else 1, -x["value"]))

        self._last_rebalance = datetime.now()
        return orders

    def get_portfolio_metrics(
        self,
        current_positions: Dict[str, Dict[str, float]],
        total_value: float,
    ) -> Dict[str, Any]:
        """Calculate portfolio metrics."""
        if not current_positions or total_value == 0:
            return {}

        weights = {
            s: pos["value"] / total_value
            for s, pos in current_positions.items()
        }

        # Concentration (Herfindahl index)
        hhi = sum(w**2 for w in weights.values())

        # Effective number of positions
        effective_n = 1 / hhi if hhi > 0 else 0

        # Max position
        max_weight = max(weights.values()) if weights else 0

        return {
            "num_positions": len(current_positions),
            "effective_positions": round(effective_n, 1),
            "concentration_hhi": round(hhi, 3),
            "max_position_weight": round(max_weight, 3),
            "weights": {s: round(w, 3) for s, w in weights.items()},
            "target_weights": {s: round(w, 3) for s, w in self._target_weights.items()},
            "strategy": self.config.strategy,
        }


def create_portfolio_rebalancer(
    strategy: str = "risk_parity",
    rebalance_threshold: float = 0.05,
) -> PortfolioRebalancer:
    """Factory function to create portfolio rebalancer."""
    config = RebalanceConfig(
        strategy=strategy,
        rebalance_threshold=rebalance_threshold,
    )
    return PortfolioRebalancer(config)
