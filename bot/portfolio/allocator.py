"""
Portfolio Allocator Module.

Implements position sizing using:
- Kelly Criterion (optimal fraction based on edge)
- Half-Kelly (conservative Kelly for reduced variance)
- Risk Parity (equal risk contribution)
- Combined Kelly + Risk Parity hybrid

Uses backtest results to calculate optimal allocation per asset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    KELLY_RISK_PARITY_HYBRID = "kelly_rp_hybrid"


@dataclass
class BacktestStats:
    """Statistics from backtesting an asset/strategy."""
    
    symbol: str
    win_rate: float  # 0-1
    avg_win: float  # Average winning trade return (absolute or %)
    avg_loss: float  # Average losing trade (absolute, positive value)
    total_trades: int
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    @classmethod
    def from_backtest_result(cls, symbol: str, result: Dict[str, Any]) -> "BacktestStats":
        """Create from backtest result dictionary."""
        return cls(
            symbol=symbol,
            win_rate=result.get("win_rate", 0.5),
            avg_win=abs(result.get("avg_win", 0)),
            avg_loss=abs(result.get("avg_loss", 1)),
            total_trades=result.get("total_trades", 0),
            volatility=result.get("volatility", 0.2),
            sharpe_ratio=result.get("sharpe_ratio", 0),
            max_drawdown=result.get("max_drawdown_pct", 0) / 100 if result.get("max_drawdown_pct") else 0.1,
            profit_factor=result.get("profit_factor", 1),
        )


@dataclass
class AssetAllocation:
    """Allocation result for a single asset."""
    
    symbol: str
    kelly_fraction: float  # Raw Kelly f*
    half_kelly_fraction: float  # Half Kelly
    risk_parity_weight: float  # Weight from risk parity
    recommended_weight: float  # Final recommended weight
    method_used: AllocationMethod
    edge: float  # Expected edge per trade
    confidence_score: float  # 0-1 confidence in allocation
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "half_kelly_fraction": round(self.half_kelly_fraction, 4),
            "risk_parity_weight": round(self.risk_parity_weight, 4),
            "recommended_weight": round(self.recommended_weight, 4),
            "method_used": self.method_used.value,
            "edge": round(self.edge, 4),
            "confidence_score": round(self.confidence_score, 2),
            "reasoning": self.reasoning,
        }


@dataclass
class PortfolioAllocationResult:
    """Complete portfolio allocation result."""
    
    allocations: Dict[str, AssetAllocation]
    method: AllocationMethod
    total_weight: float
    cash_reserve: float
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    portfolio_sharpe: float
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": {k: v.to_dict() for k, v in self.allocations.items()},
            "method": self.method.value,
            "total_weight": round(self.total_weight, 4),
            "cash_reserve": round(self.cash_reserve, 4),
            "expected_portfolio_return": round(self.expected_portfolio_return, 4),
            "expected_portfolio_volatility": round(self.expected_portfolio_volatility, 4),
            "portfolio_sharpe": round(self.portfolio_sharpe, 4),
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def print_report(self) -> str:
        """Generate printable allocation report."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("PORTFOLIO ALLOCATION REPORT")
        lines.append(f"Method: {self.method.value.upper()}")
        lines.append(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # Kelly fractions table
        lines.append("\n📊 KELLY FRACTIONS PER ASSET:")
        lines.append("-" * 70)
        lines.append(f"{'Asset':<12} {'Kelly f*':>10} {'Half-Kelly':>12} {'Edge':>10} {'Confidence':>12}")
        lines.append("-" * 70)
        
        for symbol, alloc in sorted(self.allocations.items(), key=lambda x: -x[1].kelly_fraction):
            lines.append(
                f"{symbol:<12} {alloc.kelly_fraction:>10.2%} {alloc.half_kelly_fraction:>12.2%} "
                f"{alloc.edge:>10.2%} {alloc.confidence_score:>12.0%}"
            )
        
        # Risk parity weights
        lines.append("\n📈 RISK PARITY WEIGHTS:")
        lines.append("-" * 70)
        lines.append(f"{'Asset':<12} {'Weight':>10} {'Risk Contrib':>14}")
        lines.append("-" * 70)
        
        for symbol, alloc in sorted(self.allocations.items(), key=lambda x: -x[1].risk_parity_weight):
            # Each asset should contribute equally to risk in risk parity
            risk_contrib = 1.0 / len(self.allocations) if self.allocations else 0
            lines.append(
                f"{symbol:<12} {alloc.risk_parity_weight:>10.2%} {risk_contrib:>14.2%}"
            )
        
        # Recommended allocation
        lines.append("\n✅ RECOMMENDED ALLOCATION:")
        lines.append("-" * 70)
        lines.append(f"{'Asset':<12} {'Weight':>10} {'Method':>20} {'Reasoning':<25}")
        lines.append("-" * 70)
        
        for symbol, alloc in sorted(self.allocations.items(), key=lambda x: -x[1].recommended_weight):
            reason_short = alloc.reasoning[:24] + "..." if len(alloc.reasoning) > 25 else alloc.reasoning
            lines.append(
                f"{symbol:<12} {alloc.recommended_weight:>10.2%} {alloc.method_used.value:>20} {reason_short:<25}"
            )
        
        # Portfolio summary
        lines.append("\n📋 PORTFOLIO SUMMARY:")
        lines.append("-" * 70)
        lines.append(f"Total Invested:           {self.total_weight:>10.2%}")
        lines.append(f"Cash Reserve:             {self.cash_reserve:>10.2%}")
        lines.append(f"Expected Annual Return:   {self.expected_portfolio_return:>10.2%}")
        lines.append(f"Expected Annual Vol:      {self.expected_portfolio_volatility:>10.2%}")
        lines.append(f"Portfolio Sharpe Ratio:   {self.portfolio_sharpe:>10.2f}")
        
        # Warnings
        if self.warnings:
            lines.append("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        
        lines.append("=" * 70 + "\n")
        
        report = "\n".join(lines)
        print(report)
        return report


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion optimal fraction.
    
    Formula: f* = (p * b - q) / b
    
    Where:
        p = probability of winning (win_rate)
        q = probability of losing (1 - p)
        b = odds ratio (avg_win / avg_loss)
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive value)
    
    Returns:
        Optimal fraction of capital to bet (f*)
    
    Example:
        >>> kelly_fraction(0.55, 100, 80)
        0.18125  # Bet 18.125% of capital
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    if avg_loss <= 0:
        logger.warning("avg_loss must be positive, returning 0")
        return 0.0
    
    if avg_win <= 0:
        return 0.0
    
    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss
    
    # f* = (p * b - q) / b = p - q/b
    kelly = (p * b - q) / b
    
    # Kelly can be negative if there's no edge
    return max(0.0, kelly)


def half_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Half-Kelly fraction (recommended for real trading).
    
    Full Kelly maximizes geometric growth but has high variance.
    Half-Kelly sacrifices ~25% of expected growth for ~50% less variance.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive value)
    
    Returns:
        Half-Kelly fraction
    """
    return kelly_fraction(win_rate, avg_win, avg_loss) * 0.5


def quarter_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Quarter-Kelly fraction (very conservative).
    
    Use for:
    - New strategies with limited track record
    - High uncertainty environments
    - Risk-averse portfolios
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive value)
    
    Returns:
        Quarter-Kelly fraction
    """
    return kelly_fraction(win_rate, avg_win, avg_loss) * 0.25


def calculate_edge(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate expected edge per trade.
    
    Edge = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    Returns:
        Expected profit per unit risked
    """
    if avg_loss <= 0:
        return 0.0
    
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


class PortfolioAllocator:
    """
    Portfolio allocator using Kelly Criterion and Risk Parity.
    
    Features:
    - Calculates Kelly fraction from backtest results
    - Implements risk parity (inverse volatility weighting)
    - Provides half-Kelly recommendation for safety
    - Respects maximum position limits
    - Considers correlation for portfolio construction
    
    Usage:
        allocator = PortfolioAllocator(
            max_position_size=0.25,
            max_total_allocation=0.90,
        )
        
        # Add backtest results
        allocator.add_asset_stats("BTC/USDT", BacktestStats(...))
        allocator.add_asset_stats("ETH/USDT", BacktestStats(...))
        
        # Calculate allocation
        result = allocator.calculate_allocation(
            method=AllocationMethod.HALF_KELLY
        )
        
        result.print_report()
    """
    
    def __init__(
        self,
        max_position_size: float = 0.25,  # Max 25% per asset
        max_total_allocation: float = 0.90,  # Keep 10% cash
        min_position_size: float = 0.02,  # Min 2% to be meaningful
        target_portfolio_vol: float = 0.15,  # 15% annual target vol
        risk_free_rate: float = 0.02,  # For Sharpe calculation
        kelly_cap: float = 0.50,  # Cap raw Kelly at 50%
    ):
        """
        Initialize portfolio allocator.
        
        Args:
            max_position_size: Maximum allocation per asset
            max_total_allocation: Maximum total invested (rest is cash)
            min_position_size: Minimum allocation to include asset
            target_portfolio_vol: Target portfolio volatility
            risk_free_rate: Risk-free rate for Sharpe ratio
            kelly_cap: Maximum raw Kelly fraction (safety cap)
        """
        self.max_position_size = max_position_size
        self.max_total_allocation = max_total_allocation
        self.min_position_size = min_position_size
        self.target_portfolio_vol = target_portfolio_vol
        self.risk_free_rate = risk_free_rate
        self.kelly_cap = kelly_cap
        
        # Asset statistics from backtests
        self._asset_stats: Dict[str, BacktestStats] = {}
        
        # Correlation matrix (optional)
        self._correlation_matrix: Optional[pd.DataFrame] = None
    
    def add_asset_stats(
        self,
        symbol: str,
        stats: BacktestStats,
    ) -> None:
        """Add or update asset statistics."""
        self._asset_stats[symbol] = stats
        logger.debug(f"Added stats for {symbol}: WR={stats.win_rate:.2%}, PF={stats.profit_factor:.2f}")
    
    def add_asset_from_backtest(
        self,
        symbol: str,
        backtest_result: Dict[str, Any],
    ) -> None:
        """Add asset stats from backtest result dict."""
        stats = BacktestStats.from_backtest_result(symbol, backtest_result)
        self.add_asset_stats(symbol, stats)
    
    def set_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Set correlation matrix for portfolio calculations."""
        self._correlation_matrix = corr_matrix
    
    def calculate_kelly_fractions(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate Kelly and Half-Kelly fractions for all assets.
        
        Returns:
            Dict of symbol -> (kelly, half_kelly)
        """
        fractions = {}
        
        for symbol, stats in self._asset_stats.items():
            k = kelly_fraction(stats.win_rate, stats.avg_win, stats.avg_loss)
            k = min(k, self.kelly_cap)  # Apply safety cap
            fractions[symbol] = (k, k * 0.5)
        
        return fractions
    
    def calculate_risk_parity_weights(self) -> Dict[str, float]:
        """
        Calculate risk parity weights (inverse volatility).
        
        Each asset contributes equally to portfolio risk.
        Higher volatility assets get smaller weights.
        
        Returns:
            Dict of symbol -> weight
        """
        if not self._asset_stats:
            return {}
        
        volatilities = {s: stats.volatility for s, stats in self._asset_stats.items()}
        
        # Handle zero volatility
        for s in volatilities:
            if volatilities[s] <= 0:
                volatilities[s] = 0.01  # Default to 1%
        
        # Inverse volatility
        inv_vols = {s: 1.0 / v for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize to weights
        weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}
        
        # Scale to target volatility
        if self._correlation_matrix is not None:
            # Use covariance matrix for more accurate scaling
            portfolio_vol = self._calculate_portfolio_volatility(weights)
            if portfolio_vol > 0:
                scale = self.target_portfolio_vol / portfolio_vol
                weights = {s: min(w * scale, self.max_position_size) for s, w in weights.items()}
        
        return weights
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility using correlation matrix."""
        if self._correlation_matrix is None:
            # Assume zero correlation
            var = sum(
                (w ** 2) * (self._asset_stats[s].volatility ** 2)
                for s, w in weights.items()
                if s in self._asset_stats
            )
            return np.sqrt(var)
        
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        vols = np.array([self._asset_stats[s].volatility for s in symbols])
        
        # Build covariance matrix
        corr = self._correlation_matrix.loc[symbols, symbols].values
        cov = np.outer(vols, vols) * corr
        
        # Portfolio variance
        portfolio_var = w @ cov @ w
        return np.sqrt(portfolio_var)
    
    def calculate_allocation(
        self,
        method: AllocationMethod = AllocationMethod.HALF_KELLY,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioAllocationResult:
        """
        Calculate portfolio allocation.
        
        Args:
            method: Allocation method to use
            custom_weights: Optional custom weights to override
        
        Returns:
            Complete allocation result with report
        """
        if not self._asset_stats:
            raise ValueError("No asset statistics provided. Use add_asset_stats() first.")
        
        warnings = []
        allocations = {}
        
        # Calculate Kelly fractions
        kelly_fracs = self.calculate_kelly_fractions()
        
        # Calculate risk parity weights
        rp_weights = self.calculate_risk_parity_weights()
        
        # Determine final weights based on method
        for symbol, stats in self._asset_stats.items():
            k_full, k_half = kelly_fracs.get(symbol, (0, 0))
            rp_weight = rp_weights.get(symbol, 0)
            
            # Calculate edge
            edge = calculate_edge(stats.win_rate, stats.avg_win, stats.avg_loss)
            
            # Calculate confidence based on trade count and sharpe
            confidence = self._calculate_confidence(stats)
            
            # Determine recommended weight based on method
            if method == AllocationMethod.KELLY:
                rec_weight = min(k_full, self.max_position_size)
                reasoning = f"Full Kelly: {k_full:.2%} (capped at {self.max_position_size:.0%})"
            
            elif method == AllocationMethod.HALF_KELLY:
                rec_weight = min(k_half, self.max_position_size)
                reasoning = f"Half-Kelly for safety"
            
            elif method == AllocationMethod.QUARTER_KELLY:
                rec_weight = min(k_full * 0.25, self.max_position_size)
                reasoning = f"Quarter-Kelly (conservative)"
            
            elif method == AllocationMethod.RISK_PARITY:
                rec_weight = min(rp_weight, self.max_position_size)
                reasoning = f"Inverse vol weighting"
            
            elif method == AllocationMethod.EQUAL_WEIGHT:
                rec_weight = min(1.0 / len(self._asset_stats), self.max_position_size)
                reasoning = "Equal weight 1/N"
            
            elif method == AllocationMethod.KELLY_RISK_PARITY_HYBRID:
                # Blend: 50% half-kelly, 50% risk parity
                hybrid = 0.5 * k_half + 0.5 * rp_weight
                rec_weight = min(hybrid, self.max_position_size)
                reasoning = "50% Half-Kelly + 50% Risk Parity"
            
            else:
                rec_weight = k_half
                reasoning = "Default: Half-Kelly"
            
            # Apply custom weights if provided
            if custom_weights and symbol in custom_weights:
                rec_weight = custom_weights[symbol]
                reasoning = "Custom weight override"
            
            # Adjust for confidence
            if confidence < 0.5:
                rec_weight *= confidence / 0.5
                reasoning += f" (scaled by confidence {confidence:.0%})"
            
            # Skip if below minimum
            if rec_weight < self.min_position_size:
                warnings.append(f"{symbol}: Weight {rec_weight:.2%} below minimum {self.min_position_size:.0%}")
                rec_weight = 0
            
            allocations[symbol] = AssetAllocation(
                symbol=symbol,
                kelly_fraction=k_full,
                half_kelly_fraction=k_half,
                risk_parity_weight=rp_weight,
                recommended_weight=rec_weight,
                method_used=method,
                edge=edge,
                confidence_score=confidence,
                reasoning=reasoning,
            )
        
        # Calculate total allocation
        total_weight = sum(a.recommended_weight for a in allocations.values())
        
        # Scale down if exceeds maximum
        if total_weight > self.max_total_allocation:
            scale = self.max_total_allocation / total_weight
            warnings.append(f"Scaled weights by {scale:.2%} to respect max allocation")
            for alloc in allocations.values():
                alloc.recommended_weight *= scale
            total_weight = self.max_total_allocation
        
        cash_reserve = 1.0 - total_weight
        
        # Calculate expected portfolio metrics
        weights_dict = {s: a.recommended_weight for s, a in allocations.items()}
        exp_return = self._calculate_expected_return(weights_dict)
        exp_vol = self._calculate_portfolio_volatility(weights_dict)
        sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0
        
        return PortfolioAllocationResult(
            allocations=allocations,
            method=method,
            total_weight=total_weight,
            cash_reserve=cash_reserve,
            expected_portfolio_return=exp_return,
            expected_portfolio_volatility=exp_vol,
            portfolio_sharpe=sharpe,
            warnings=warnings,
        )
    
    def _calculate_confidence(self, stats: BacktestStats) -> float:
        """
        Calculate confidence score for allocation.
        
        Based on:
        - Number of trades (more = higher confidence)
        - Sharpe ratio (higher = more confident)
        - Profit factor (higher = more confident)
        """
        # Trade count contribution (0.3 weight)
        # 100+ trades = full confidence
        trade_conf = min(stats.total_trades / 100, 1.0) * 0.3
        
        # Sharpe contribution (0.4 weight)
        # Sharpe > 1.5 = full confidence
        sharpe_conf = min(max(stats.sharpe_ratio, 0) / 1.5, 1.0) * 0.4
        
        # Profit factor contribution (0.3 weight)
        # PF > 2.0 = full confidence
        pf_conf = min(max(stats.profit_factor - 1, 0) / 1.0, 1.0) * 0.3
        
        return trade_conf + sharpe_conf + pf_conf
    
    def _calculate_expected_return(self, weights: Dict[str, float]) -> float:
        """Calculate expected portfolio return."""
        expected_return = 0.0
        
        for symbol, weight in weights.items():
            if symbol in self._asset_stats:
                stats = self._asset_stats[symbol]
                # Expected return = win_rate * avg_win - loss_rate * avg_loss
                # Annualized (assume ~252 trading days, estimate trades per year)
                edge_per_trade = calculate_edge(stats.win_rate, stats.avg_win, stats.avg_loss)
                # Rough estimate: assume 50 trades per year
                annual_return = edge_per_trade * 50
                expected_return += weight * annual_return
        
        return expected_return
    
    def get_position_sizes(
        self,
        portfolio_value: float,
        method: AllocationMethod = AllocationMethod.HALF_KELLY,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get dollar position sizes for trading.
        
        Args:
            portfolio_value: Total portfolio value in USD
            method: Allocation method
        
        Returns:
            Dict of symbol -> {weight, dollar_amount}
        """
        result = self.calculate_allocation(method)
        
        positions = {}
        for symbol, alloc in result.allocations.items():
            if alloc.recommended_weight > 0:
                positions[symbol] = {
                    "weight": alloc.recommended_weight,
                    "dollar_amount": portfolio_value * alloc.recommended_weight,
                    "kelly_fraction": alloc.kelly_fraction,
                    "confidence": alloc.confidence_score,
                }
        
        return positions
    
    def suggest_method(self) -> Tuple[AllocationMethod, str]:
        """
        Suggest the best allocation method based on available data.
        
        Returns:
            Tuple of (recommended method, reasoning)
        """
        if not self._asset_stats:
            return AllocationMethod.EQUAL_WEIGHT, "No data available"
        
        # Check data quality
        avg_trades = np.mean([s.total_trades for s in self._asset_stats.values()])
        avg_sharpe = np.mean([s.sharpe_ratio for s in self._asset_stats.values()])
        avg_pf = np.mean([s.profit_factor for s in self._asset_stats.values()])
        
        # High confidence: use Half-Kelly
        if avg_trades > 100 and avg_sharpe > 1.0 and avg_pf > 1.5:
            return AllocationMethod.HALF_KELLY, f"Good edge (Sharpe={avg_sharpe:.2f}, PF={avg_pf:.2f})"
        
        # Moderate confidence: use hybrid
        if avg_trades > 50 and avg_sharpe > 0.5:
            return AllocationMethod.KELLY_RISK_PARITY_HYBRID, "Moderate confidence, blending approaches"
        
        # Low confidence: use risk parity (doesn't rely on edge estimates)
        if avg_trades > 20:
            return AllocationMethod.RISK_PARITY, "Limited trade history, using risk-based approach"
        
        # Very low data: equal weight
        return AllocationMethod.EQUAL_WEIGHT, "Insufficient data for sophisticated allocation"


def create_allocator_from_backtests(
    backtest_results: Dict[str, Dict[str, Any]],
    **kwargs,
) -> PortfolioAllocator:
    """
    Create allocator from dictionary of backtest results.
    
    Args:
        backtest_results: Dict of symbol -> backtest result dict
        **kwargs: Additional arguments for PortfolioAllocator
    
    Returns:
        Configured PortfolioAllocator
    """
    allocator = PortfolioAllocator(**kwargs)
    
    for symbol, result in backtest_results.items():
        allocator.add_asset_from_backtest(symbol, result)
    
    return allocator


def run_allocation_example():
    """Example usage of portfolio allocator."""
    print("\n" + "=" * 70)
    print("PORTFOLIO ALLOCATOR EXAMPLE")
    print("=" * 70)
    
    # Simulated backtest results
    # Note: avg_win and avg_loss are in PERCENTAGE terms (e.g., 2.5 = 2.5% gain)
    # This gives realistic expected returns when annualized
    backtest_results = {
        "BTC/USDT": {
            "win_rate": 0.55,        # 55% win rate
            "avg_win": 2.5,          # Average win: 2.5%
            "avg_loss": 1.8,         # Average loss: 1.8%
            "total_trades": 120,     # 120 trades in backtest
            "volatility": 0.65,      # 65% annualized volatility
            "sharpe_ratio": 1.2,     # Good risk-adjusted return
            "max_drawdown_pct": 18,  # 18% max drawdown
            "profit_factor": 1.68,   # Winners / Losers ratio
        },
        "ETH/USDT": {
            "win_rate": 0.52,
            "avg_win": 3.2,
            "avg_loss": 2.5,
            "total_trades": 95,
            "volatility": 0.75,
            "sharpe_ratio": 0.95,
            "max_drawdown_pct": 22,
            "profit_factor": 1.25,
        },
        "SOL/USDT": {
            "win_rate": 0.48,
            "avg_win": 4.5,
            "avg_loss": 3.0,
            "total_trades": 65,
            "volatility": 0.95,
            "sharpe_ratio": 0.75,
            "max_drawdown_pct": 28,
            "profit_factor": 1.15,
        },
        "AVAX/USDT": {
            "win_rate": 0.58,
            "avg_win": 2.8,
            "avg_loss": 2.2,
            "total_trades": 85,
            "volatility": 0.85,
            "sharpe_ratio": 1.05,
            "max_drawdown_pct": 20,
            "profit_factor": 1.45,
        },
    }
    
    # Create allocator
    allocator = create_allocator_from_backtests(
        backtest_results,
        max_position_size=0.30,
        max_total_allocation=0.85,
    )
    
    # Get suggestion
    suggested_method, reason = allocator.suggest_method()
    print(f"\n📌 Suggested Method: {suggested_method.value}")
    print(f"   Reason: {reason}")
    
    # Calculate allocations with different methods
    methods = [
        AllocationMethod.KELLY,
        AllocationMethod.HALF_KELLY,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.KELLY_RISK_PARITY_HYBRID,
    ]
    
    for method in methods:
        result = allocator.calculate_allocation(method)
        result.print_report()
    
    # Get position sizes for $100,000 portfolio
    portfolio_value = 100_000
    positions = allocator.get_position_sizes(portfolio_value, AllocationMethod.HALF_KELLY)
    
    print("\n💰 POSITION SIZES FOR $100,000 PORTFOLIO (Half-Kelly):")
    print("-" * 50)
    for symbol, pos in sorted(positions.items(), key=lambda x: -x[1]["dollar_amount"]):
        print(f"  {symbol}: ${pos['dollar_amount']:,.2f} ({pos['weight']:.1%})")
    
    total = sum(p["dollar_amount"] for p in positions.values())
    print(f"\n  Total Invested: ${total:,.2f}")
    print(f"  Cash Reserve:   ${portfolio_value - total:,.2f}")
    
    return allocator


if __name__ == "__main__":
    run_allocation_example()
