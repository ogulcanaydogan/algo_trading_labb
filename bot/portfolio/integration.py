"""
Portfolio Allocator Integration with Trading Engine.

Provides helpers to:
- Extract backtest stats for Kelly calculation
- Apply allocation weights to position sizing
- Integrate with enhanced trading engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.portfolio.allocator import (
    PortfolioAllocator,
    AllocationMethod,
    BacktestStats,
    PortfolioAllocationResult,
    kelly_fraction,
    half_kelly_fraction,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result from allocation-based position sizing."""
    
    symbol: str
    allocation_weight: float  # From allocator (0-1)
    dollar_size: float  # Position size in dollars
    risk_adjusted_size: float  # After risk adjustments
    max_allowed_size: float  # Position limit
    final_size: float  # Final recommended size
    kelly_fraction: float
    method: str
    reasoning: str


def extract_stats_from_backtest_result(
    symbol: str,
    backtest_result: Dict[str, Any],
    returns: Optional[pd.Series] = None,
) -> BacktestStats:
    """
    Extract BacktestStats from a backtest result dictionary.
    
    Args:
        symbol: Asset symbol
        backtest_result: Backtest result dict with win_rate, avg_win, etc.
        returns: Optional series of returns for volatility calculation
    
    Returns:
        BacktestStats object
    """
    # Calculate volatility from returns if provided
    if returns is not None and len(returns) > 0:
        volatility = returns.std() * np.sqrt(252)  # Annualize
    else:
        volatility = backtest_result.get("volatility", 0.20)
    
    return BacktestStats(
        symbol=symbol,
        win_rate=backtest_result.get("win_rate", 0.5),
        avg_win=abs(backtest_result.get("avg_win", 0)),
        avg_loss=abs(backtest_result.get("avg_loss", 1)),
        total_trades=backtest_result.get("total_trades", 0),
        volatility=volatility,
        sharpe_ratio=backtest_result.get("sharpe_ratio", 0),
        max_drawdown=backtest_result.get("max_drawdown_pct", 0) / 100 
            if backtest_result.get("max_drawdown_pct") else 0.1,
        profit_factor=backtest_result.get("profit_factor", 1),
    )


def calculate_position_size_with_kelly(
    symbol: str,
    backtest_stats: BacktestStats,
    portfolio_value: float,
    current_price: float,
    method: AllocationMethod = AllocationMethod.HALF_KELLY,
    max_position_pct: float = 0.25,
    current_drawdown: float = 0.0,
    confidence_override: Optional[float] = None,
) -> PositionSizeResult:
    """
    Calculate position size using Kelly-based allocation.
    
    Args:
        symbol: Asset symbol
        backtest_stats: Stats from backtesting
        portfolio_value: Total portfolio value
        current_price: Current asset price
        method: Allocation method
        max_position_pct: Maximum position as fraction of portfolio
        current_drawdown: Current drawdown for risk adjustment
        confidence_override: Override confidence score
    
    Returns:
        PositionSizeResult with sizing details
    """
    # Calculate raw Kelly
    k_full = kelly_fraction(
        backtest_stats.win_rate,
        backtest_stats.avg_win,
        backtest_stats.avg_loss,
    )
    k_half = k_full * 0.5
    
    # Select weight based on method
    if method == AllocationMethod.KELLY:
        allocation_weight = k_full
        method_name = "Full Kelly"
    elif method == AllocationMethod.HALF_KELLY:
        allocation_weight = k_half
        method_name = "Half Kelly"
    elif method == AllocationMethod.QUARTER_KELLY:
        allocation_weight = k_full * 0.25
        method_name = "Quarter Kelly"
    else:
        allocation_weight = k_half
        method_name = "Default Half Kelly"
    
    # Cap at max position
    allocation_weight = min(allocation_weight, max_position_pct)
    
    # Calculate dollar size
    dollar_size = portfolio_value * allocation_weight
    
    # Apply drawdown adjustment
    drawdown_adj = 1.0
    if current_drawdown > 0.10:  # Start reducing after 10% drawdown
        drawdown_adj = max(0.5, 1.0 - (current_drawdown - 0.10) / 0.20)
    
    # Apply confidence adjustment
    if confidence_override is not None:
        confidence = confidence_override
    else:
        # Calculate confidence from stats
        trade_conf = min(backtest_stats.total_trades / 100, 1.0)
        sharpe_conf = min(max(backtest_stats.sharpe_ratio, 0) / 1.5, 1.0)
        confidence = 0.5 * trade_conf + 0.5 * sharpe_conf
    
    conf_adj = confidence if confidence < 0.5 else 1.0
    
    # Risk adjusted size
    risk_adjusted = dollar_size * drawdown_adj * conf_adj
    
    # Max allowed based on portfolio limits
    max_allowed = portfolio_value * max_position_pct
    
    # Final size
    final_size = min(risk_adjusted, max_allowed)
    
    # Calculate units if needed
    units = final_size / current_price if current_price > 0 else 0
    
    reasoning = (
        f"{method_name}: f*={k_full:.2%}, weight={allocation_weight:.2%}, "
        f"dd_adj={drawdown_adj:.2f}, conf={confidence:.2f}"
    )
    
    return PositionSizeResult(
        symbol=symbol,
        allocation_weight=allocation_weight,
        dollar_size=dollar_size,
        risk_adjusted_size=risk_adjusted,
        max_allowed_size=max_allowed,
        final_size=final_size,
        kelly_fraction=k_full,
        method=method_name,
        reasoning=reasoning,
    )


class PortfolioPositionSizer:
    """
    Integrates portfolio allocation with position sizing.
    
    Use with enhanced trading engine for Kelly-based sizing.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = 0.25,
        max_total_allocation: float = 0.90,
        default_method: AllocationMethod = AllocationMethod.HALF_KELLY,
    ):
        """
        Initialize position sizer.
        
        Args:
            portfolio_value: Total portfolio value
            max_position_pct: Max per-asset position
            max_total_allocation: Max total invested (rest cash)
            default_method: Default allocation method
        """
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_total_allocation = max_total_allocation
        self.default_method = default_method
        
        self._allocator = PortfolioAllocator(
            max_position_size=max_position_pct,
            max_total_allocation=max_total_allocation,
        )
        self._current_allocation: Optional[PortfolioAllocationResult] = None
        self._current_drawdown = 0.0
        self._peak_value = portfolio_value
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value and track drawdown."""
        self.portfolio_value = new_value
        
        if new_value > self._peak_value:
            self._peak_value = new_value
            self._current_drawdown = 0.0
        else:
            self._current_drawdown = (self._peak_value - new_value) / self._peak_value
    
    def add_backtest_result(
        self,
        symbol: str,
        backtest_result: Dict[str, Any],
        returns: Optional[pd.Series] = None,
    ) -> None:
        """Add backtest result for an asset."""
        stats = extract_stats_from_backtest_result(symbol, backtest_result, returns)
        self._allocator.add_asset_stats(symbol, stats)
    
    def calculate_allocation(
        self,
        method: Optional[AllocationMethod] = None,
    ) -> PortfolioAllocationResult:
        """Calculate portfolio allocation."""
        m = method or self.default_method
        self._current_allocation = self._allocator.calculate_allocation(m)
        return self._current_allocation
    
    def get_position_size(
        self,
        symbol: str,
        current_price: float,
        method: Optional[AllocationMethod] = None,
    ) -> PositionSizeResult:
        """
        Get position size for a specific symbol.
        
        Args:
            symbol: Asset symbol
            current_price: Current price
            method: Override allocation method
        
        Returns:
            PositionSizeResult with sizing
        """
        if symbol not in self._allocator._asset_stats:
            raise ValueError(f"No backtest stats for {symbol}. Add with add_backtest_result()")
        
        stats = self._allocator._asset_stats[symbol]
        m = method or self.default_method
        
        return calculate_position_size_with_kelly(
            symbol=symbol,
            backtest_stats=stats,
            portfolio_value=self.portfolio_value,
            current_price=current_price,
            method=m,
            max_position_pct=self.max_position_pct,
            current_drawdown=self._current_drawdown,
        )
    
    def get_all_position_sizes(
        self,
        prices: Dict[str, float],
        method: Optional[AllocationMethod] = None,
    ) -> Dict[str, PositionSizeResult]:
        """
        Get position sizes for all assets.
        
        Args:
            prices: Dict of symbol -> current price
            method: Override allocation method
        
        Returns:
            Dict of symbol -> PositionSizeResult
        """
        results = {}
        for symbol in self._allocator._asset_stats:
            if symbol in prices:
                results[symbol] = self.get_position_size(symbol, prices[symbol], method)
        return results
    
    def should_rebalance(
        self,
        current_positions: Dict[str, float],  # symbol -> current value
        threshold: float = 0.05,  # 5% deviation triggers rebalance
    ) -> Tuple[bool, List[str]]:
        """
        Check if rebalancing is needed.
        
        Args:
            current_positions: Dict of symbol -> current position value
            threshold: Deviation threshold to trigger rebalance
        
        Returns:
            Tuple of (should_rebalance, list of symbols needing adjustment)
        """
        if self._current_allocation is None:
            return True, list(current_positions.keys())
        
        total_value = self.portfolio_value
        needs_adjustment = []
        
        for symbol, alloc in self._current_allocation.allocations.items():
            target_weight = alloc.recommended_weight
            current_value = current_positions.get(symbol, 0)
            current_weight = current_value / total_value if total_value > 0 else 0
            
            deviation = abs(target_weight - current_weight)
            if deviation > threshold:
                needs_adjustment.append(symbol)
        
        return len(needs_adjustment) > 0, needs_adjustment
    
    def get_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Get trades needed to rebalance portfolio.
        
        Args:
            current_positions: Current position values
            prices: Current prices
        
        Returns:
            List of trade dictionaries
        """
        if self._current_allocation is None:
            self.calculate_allocation()
        
        trades = []
        total_value = self.portfolio_value
        
        for symbol, alloc in self._current_allocation.allocations.items():
            target_value = total_value * alloc.recommended_weight
            current_value = current_positions.get(symbol, 0)
            diff_value = target_value - current_value
            
            if abs(diff_value) < 10:  # Skip tiny trades
                continue
            
            price = prices.get(symbol, 0)
            if price <= 0:
                continue
            
            quantity = abs(diff_value) / price
            side = "BUY" if diff_value > 0 else "SELL"
            
            trades.append({
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "value": abs(diff_value),
                "target_weight": alloc.recommended_weight,
                "current_weight": current_value / total_value if total_value > 0 else 0,
                "kelly_fraction": alloc.kelly_fraction,
            })
        
        return trades


def integrate_with_trading_engine(
    sizer: PortfolioPositionSizer,
    engine_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update trading engine config with Kelly-based position sizing.
    
    Args:
        sizer: Configured PortfolioPositionSizer
        engine_config: Trading engine configuration dict
    
    Returns:
        Updated config with Kelly-based sizing parameters
    """
    # Calculate allocation
    allocation = sizer.calculate_allocation()
    
    # Update position limits per symbol
    position_limits = {}
    for symbol, alloc in allocation.allocations.items():
        position_limits[symbol] = {
            "max_position_pct": alloc.recommended_weight,
            "kelly_fraction": alloc.kelly_fraction,
            "confidence": alloc.confidence_score,
        }
    
    # Update engine config
    updated_config = engine_config.copy()
    updated_config["position_limits"] = position_limits
    updated_config["allocation_method"] = allocation.method.value
    updated_config["total_allocation"] = allocation.total_weight
    updated_config["cash_reserve"] = allocation.cash_reserve
    
    logger.info(
        f"Updated trading engine config with {allocation.method.value} allocation: "
        f"{allocation.total_weight:.1%} invested, {allocation.cash_reserve:.1%} cash"
    )
    
    return updated_config
