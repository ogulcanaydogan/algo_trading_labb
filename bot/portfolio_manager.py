"""
Multi-Asset Portfolio Manager.
Handles position sizing, allocation tracking, and rebalancing logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from bot.portfolio_config import Portfolio, Asset, RebalanceStrategy


@dataclass
class AssetPosition:
    """Current position in a single asset."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in USD."""
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100

    @property
    def position_value(self) -> float:
        """Total position value in USD."""
        return self.quantity * self.current_price


@dataclass
class PortfolioState:
    """Current portfolio state."""

    total_capital: float
    cash: float
    positions: Dict[str, AssetPosition] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_equity(self) -> float:
        """Total portfolio value."""
        return self.cash + sum(p.position_value for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def invested_capital(self) -> float:
        """Total capital invested in positions."""
        return self.total_equity - self.cash

    @property
    def allocation_pcts(self) -> Dict[str, float]:
        """Get current allocation percentages."""
        total = self.total_equity
        if total == 0:
            return {}

        return {
            symbol: (position.position_value / total) * 100
            for symbol, position in self.positions.items()
        }


class PortfolioManager:
    """Manages multi-asset portfolio operations."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.state = PortfolioState(
            total_capital=portfolio.total_capital,
            cash=portfolio.total_capital,
        )
        self.rebalance_history = []

    def calculate_position_size(self, symbol: str, current_price: float) -> float:
        """
        Calculate position size for an asset based on allocation.
        Respects risk limits and capital constraints.
        """
        asset = self.portfolio.get_asset(symbol)
        if not asset:
            raise ValueError(f"Asset {symbol} not found in portfolio")

        # Calculate target capital for this asset
        target_capital = self.state.total_equity * (asset.allocation_pct / 100)

        # Respect min/max position constraints
        if asset.max_position_usd:
            target_capital = min(target_capital, asset.max_position_usd)
        target_capital = max(target_capital, asset.min_position_usd)

        # Calculate quantity
        quantity = target_capital / current_price if current_price > 0 else 0

        return quantity

    def get_allocation_drift(self, symbol: str) -> float:
        """
        Get drift between target and actual allocation.
        Positive = overweight, Negative = underweight
        """
        asset = self.portfolio.get_asset(symbol)
        if not asset:
            return 0

        current_alloc = self.state.allocation_pcts.get(symbol, 0)
        target_alloc = asset.allocation_pct

        return current_alloc - target_alloc

    def needs_rebalancing(self) -> Tuple[bool, List[str]]:
        """
        Determine if portfolio needs rebalancing.
        Returns (needs_rebalancing, assets_to_adjust)
        """
        if self.portfolio.rebalance_strategy == RebalanceStrategy.THRESHOLD:
            assets_to_rebalance = [
                asset.symbol
                for asset in self.portfolio.assets
                if abs(self.get_allocation_drift(asset.symbol))
                > self.portfolio.rebalance_threshold_pct
            ]
            return len(assets_to_rebalance) > 0, assets_to_rebalance

        return False, []

    def calculate_rebalancing_trades(self) -> Dict[str, float]:
        """
        Calculate required trades to rebalance portfolio.
        Returns dict of {symbol: quantity_delta}
        """
        trades = {}

        for asset in self.portfolio.assets:
            if not asset.active:
                continue

            current_position = self.state.positions.get(asset.symbol)
            current_qty = current_position.quantity if current_position else 0

            target_value = self.state.total_equity * (asset.allocation_pct / 100)
            current_price = current_position.current_price if current_position else 0

            if current_price == 0:
                continue

            target_qty = target_value / current_price
            qty_delta = target_qty - current_qty

            if abs(qty_delta) > 0.001:  # Ignore tiny amounts
                trades[asset.symbol] = qty_delta

        return trades

    def calculate_portfolio_correlation(self, returns_data: Dict[str, List[float]]) -> np.ndarray:
        """
        Calculate correlation matrix between assets.
        returns_data: {symbol: [returns]}
        """
        symbols = sorted(returns_data.keys())
        data = np.array([returns_data[s] for s in symbols])

        if data.shape[1] < 2:
            return np.array([[1.0]])

        return np.corrcoef(data)

    def get_portfolio_diversification(self) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI).
        Lower = more diversified. Max 10000 (single asset).
        """
        allocs = list(self.state.allocation_pcts.values())
        if not allocs:
            return 0

        # Normalize to 0-100 scale
        normalized = np.array(allocs) / 100
        hhi = np.sum(normalized**2) * 10000

        return hhi

    def calculate_portfolio_sharpe(
        self, returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate portfolio Sharpe ratio."""
        if len(returns) < 2:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        return (mean_return - risk_free_rate) / std_return

    def update_position(self, symbol: str, quantity: float, price: float):
        """Update or create a position."""
        if quantity == 0 and symbol in self.state.positions:
            del self.state.positions[symbol]
        elif quantity > 0:
            self.state.positions[symbol] = AssetPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                timestamp=datetime.utcnow(),
            )

    def update_prices(self, price_updates: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in price_updates.items():
            if symbol in self.state.positions:
                self.state.positions[symbol].current_price = price

    def record_rebalancing(self, trades: Dict[str, float]):
        """Record a rebalancing event."""
        self.rebalance_history.append(
            {
                "timestamp": datetime.utcnow(),
                "trades": trades,
                "allocations_before": self.state.allocation_pcts.copy(),
            }
        )

    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary."""
        return {
            "total_equity": self.state.total_equity,
            "cash": self.state.cash,
            "invested": self.state.invested_capital,
            "unrealized_pnl": self.state.total_unrealized_pnl,
            "unrealized_pnl_pct": (self.state.total_unrealized_pnl / self.state.total_capital)
            * 100,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "position_value": pos.position_value,
                    "pnl": pos.unrealized_pnl,
                    "pnl_pct": pos.unrealized_pnl_pct,
                }
                for symbol, pos in self.state.positions.items()
            },
            "allocation": self.state.allocation_pcts,
            "diversification_hhi": self.get_portfolio_diversification(),
            "needs_rebalancing": self.needs_rebalancing()[0],
        }
