"""
Portfolio Agent - Capital Allocation Optimization.

Manages capital allocation across multiple assets using:
- Correlation-aware position sizing
- Risk parity principles
- Kelly criterion for optimal sizing
- Hierarchical RL (portfolio -> asset level)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AssetMetrics:
    """Metrics for a single asset."""
    symbol: str
    current_price: float = 0.0
    returns_history: List[float] = field(default_factory=list)
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    correlation_with_btc: float = 0.0
    current_position_pct: float = 0.0  # Current allocation %

    @property
    def risk_score(self) -> float:
        """Lower is better (less risky)."""
        return self.volatility * (1 - self.win_rate + 0.5)


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: float
    cash: float
    positions: Dict[str, float]  # symbol -> value
    allocations: Dict[str, float]  # symbol -> % of portfolio
    total_risk_exposure: float = 0.0
    correlation_risk: float = 0.0


class PortfolioAgent:
    """
    Manages portfolio-level capital allocation.

    Uses a hierarchical approach:
    1. Portfolio level: Decide overall risk exposure and asset weights
    2. Asset level: Individual agents decide entry/exit within allocation
    """

    def __init__(
        self,
        initial_capital: float = 30000.0,
        max_position_pct: float = 0.25,  # Max 25% in single asset
        max_total_exposure: float = 0.80,  # Max 80% deployed
        min_cash_pct: float = 0.10,  # Keep 10% cash minimum
        rebalance_threshold: float = 0.05,  # Rebalance if allocation drifts 5%
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.min_cash_pct = min_cash_pct
        self.rebalance_threshold = rebalance_threshold

        # Asset tracking
        self.assets: Dict[str, AssetMetrics] = {}
        self._returns_window = 30  # Days of returns to track

        # Correlation matrix
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}

        # Target allocations
        self._target_allocations: Dict[str, float] = {}

        # Performance tracking
        self._allocation_history: deque = deque(maxlen=100)

        logger.info(
            f"PortfolioAgent initialized: capital=${initial_capital:.2f}, "
            f"max_position={max_position_pct*100:.0f}%"
        )

    def register_asset(self, symbol: str, correlation_with_btc: float = 0.5):
        """Register a new asset for portfolio management."""
        if symbol not in self.assets:
            self.assets[symbol] = AssetMetrics(
                symbol=symbol,
                correlation_with_btc=correlation_with_btc,
            )
            logger.debug(f"Registered asset: {symbol}")

    def update_asset_metrics(
        self,
        symbol: str,
        price: float,
        daily_return: float,
        win_rate: float,
        avg_pnl: float,
    ):
        """Update metrics for an asset."""
        if symbol not in self.assets:
            self.register_asset(symbol)

        asset = self.assets[symbol]
        asset.current_price = price
        asset.returns_history.append(daily_return)

        # Keep only recent history
        if len(asset.returns_history) > self._returns_window:
            asset.returns_history = asset.returns_history[-self._returns_window:]

        # Calculate volatility
        if len(asset.returns_history) >= 5:
            asset.volatility = np.std(asset.returns_history) * np.sqrt(252)  # Annualized

        # Update win rate and avg PnL
        asset.win_rate = win_rate
        asset.avg_pnl = avg_pnl

        # Calculate Sharpe ratio
        if asset.volatility > 0:
            mean_return = np.mean(asset.returns_history) * 252  # Annualized
            asset.sharpe_ratio = mean_return / asset.volatility

    def calculate_optimal_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal asset allocations using multiple methods.

        Combines:
        1. Risk parity (equal risk contribution)
        2. Performance-based (higher allocation to winners)
        3. Correlation adjustment (reduce correlated positions)
        """
        if not self.assets:
            return {}

        allocations = {}

        # 1. Risk Parity Component
        # Allocate inversely proportional to volatility
        total_inv_vol = 0
        inv_vols = {}
        for symbol, asset in self.assets.items():
            vol = max(0.01, asset.volatility)  # Minimum volatility
            inv_vols[symbol] = 1.0 / vol
            total_inv_vol += inv_vols[symbol]

        risk_parity = {
            symbol: inv_vol / total_inv_vol
            for symbol, inv_vol in inv_vols.items()
        }

        # 2. Performance Component
        # Higher allocation to assets with better win rate and Sharpe
        total_perf = 0
        perfs = {}
        for symbol, asset in self.assets.items():
            perf = asset.win_rate * (1 + max(0, asset.sharpe_ratio))
            perfs[symbol] = perf
            total_perf += perf

        performance_based = {
            symbol: perf / total_perf if total_perf > 0 else 1.0 / len(self.assets)
            for symbol, perf in perfs.items()
        }

        # 3. Correlation Adjustment
        # Reduce allocation to highly correlated assets
        correlation_adj = {}
        for symbol, asset in self.assets.items():
            # Penalty for high correlation with BTC (dominant asset)
            corr_penalty = 1.0 - (asset.correlation_with_btc * 0.3)
            correlation_adj[symbol] = max(0.5, corr_penalty)

        # Combine methods (weighted average)
        weights = {
            "risk_parity": 0.4,
            "performance": 0.4,
            "correlation": 0.2,
        }

        for symbol in self.assets:
            alloc = (
                risk_parity.get(symbol, 0) * weights["risk_parity"] +
                performance_based.get(symbol, 0) * weights["performance"]
            ) * correlation_adj.get(symbol, 1.0)
            allocations[symbol] = alloc

        # Normalize to sum to max_total_exposure
        total = sum(allocations.values())
        if total > 0:
            allocations = {
                symbol: (alloc / total) * self.max_total_exposure
                for symbol, alloc in allocations.items()
            }

        # Apply individual position caps
        allocations = {
            symbol: min(alloc, self.max_position_pct)
            for symbol, alloc in allocations.items()
        }

        self._target_allocations = allocations
        return allocations

    def get_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        current_regime: str,
    ) -> Tuple[float, float, str]:
        """
        Get recommended position size for a trade.

        Args:
            symbol: Asset symbol
            signal_confidence: ML signal confidence (0-1)
            current_regime: Current market regime

        Returns:
            Tuple of (position_value, position_pct, reasoning)
        """
        if symbol not in self.assets:
            self.register_asset(symbol)

        # Get target allocation for this asset
        if not self._target_allocations:
            self.calculate_optimal_allocations()

        target_alloc = self._target_allocations.get(symbol, 0.1)
        asset = self.assets[symbol]

        # Adjust for signal confidence
        confidence_adj = 0.5 + (signal_confidence * 0.5)  # 0.5 to 1.0

        # Adjust for regime
        regime_adj = {
            "BULL": 1.2,
            "STRONG_BULL": 1.3,
            "BEAR": 0.8,
            "STRONG_BEAR": 0.7,
            "CRASH": 0.5,
            "SIDEWAYS": 1.0,
            "HIGH_VOL": 0.8,
            "LOW_VOL": 1.1,
        }.get(current_regime, 1.0)

        # Adjust for asset performance
        perf_adj = 0.8 + (asset.win_rate * 0.4)  # 0.8 to 1.2

        # Calculate final position size
        position_pct = target_alloc * confidence_adj * regime_adj * perf_adj
        position_pct = min(position_pct, self.max_position_pct)

        position_value = self.current_capital * position_pct

        reasoning = (
            f"Base alloc: {target_alloc*100:.1f}%, "
            f"Conf adj: {confidence_adj:.2f}, "
            f"Regime adj: {regime_adj:.2f}, "
            f"Perf adj: {perf_adj:.2f}"
        )

        return position_value, position_pct, reasoning

    def check_rebalance_needed(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio rebalancing is needed.

        Returns:
            Tuple of (needs_rebalance, adjustments_needed)
        """
        if not self._target_allocations:
            self.calculate_optimal_allocations()

        adjustments = {}
        needs_rebalance = False

        for symbol, asset in self.assets.items():
            target = self._target_allocations.get(symbol, 0)
            current = asset.current_position_pct
            drift = abs(target - current)

            if drift > self.rebalance_threshold:
                needs_rebalance = True
                adjustments[symbol] = target - current  # Positive = buy, negative = sell

        return needs_rebalance, adjustments

    def update_position(self, symbol: str, position_value: float):
        """Update current position for an asset."""
        if symbol not in self.assets:
            self.register_asset(symbol)

        self.assets[symbol].current_position_pct = position_value / self.current_capital

    def update_capital(self, new_capital: float):
        """Update total capital value."""
        self.current_capital = new_capital

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        positions = {
            symbol: asset.current_position_pct * self.current_capital
            for symbol, asset in self.assets.items()
        }
        total_positions = sum(positions.values())

        return PortfolioState(
            total_value=self.current_capital,
            cash=self.current_capital - total_positions,
            positions=positions,
            allocations={s: a.current_position_pct for s, a in self.assets.items()},
            total_risk_exposure=total_positions / self.current_capital,
        )

    def get_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk."""
        if len(self.assets) < 2:
            return 0.0

        # Simple correlation risk based on BTC correlation
        weighted_corr = 0
        total_weight = 0
        for symbol, asset in self.assets.items():
            weight = asset.current_position_pct
            weighted_corr += weight * asset.correlation_with_btc
            total_weight += weight

        return weighted_corr / total_weight if total_weight > 0 else 0.0

    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """Check if overall exposure should be reduced."""
        state = self.get_portfolio_state()

        # Check total exposure
        if state.total_risk_exposure > self.max_total_exposure:
            return True, f"Exposure {state.total_risk_exposure*100:.0f}% > max {self.max_total_exposure*100:.0f}%"

        # Check correlation risk
        corr_risk = self.get_correlation_risk()
        if corr_risk > 0.8:
            return True, f"High correlation risk: {corr_risk:.2f}"

        # Check cash buffer
        cash_pct = state.cash / state.total_value
        if cash_pct < self.min_cash_pct * 0.5:
            return True, f"Cash buffer too low: {cash_pct*100:.1f}%"

        return False, ""

    def get_stats(self) -> Dict:
        """Get portfolio agent statistics."""
        state = self.get_portfolio_state()
        return {
            "total_value": self.current_capital,
            "cash": state.cash,
            "cash_pct": state.cash / state.total_value * 100,
            "total_exposure": state.total_risk_exposure * 100,
            "target_allocations": {
                s: f"{a*100:.1f}%" for s, a in self._target_allocations.items()
            },
            "current_allocations": {
                s: f"{a.current_position_pct*100:.1f}%" for s, a in self.assets.items()
            },
            "correlation_risk": self.get_correlation_risk(),
            "num_assets": len(self.assets),
        }


# Singleton instance
_portfolio_agent: Optional[PortfolioAgent] = None


def get_portfolio_agent(initial_capital: float = 30000.0) -> PortfolioAgent:
    """Get or create the PortfolioAgent singleton."""
    global _portfolio_agent
    if _portfolio_agent is None:
        _portfolio_agent = PortfolioAgent(initial_capital=initial_capital)
    return _portfolio_agent
