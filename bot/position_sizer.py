"""
Position Sizing Optimization.

Implements various position sizing methods:
- Kelly Criterion
- Volatility-adjusted sizing
- Risk parity sizing
- Maximum drawdown-based sizing
- Confidence-based scaling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods."""

    FIXED_FRACTION = "fixed_fraction"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"
    DRAWDOWN_ADJUSTED = "drawdown_adjusted"
    CONFIDENCE_SCALED = "confidence_scaled"
    OPTIMAL_F = "optimal_f"


@dataclass
class SizingResult:
    """Result from position sizing calculation."""

    symbol: str
    method: SizingMethod
    position_size: float  # Fraction of portfolio (0-1)
    dollar_amount: float
    shares_or_units: float
    confidence_adjustment: float
    volatility_adjustment: float
    drawdown_adjustment: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "method": self.method.value,
            "position_size": round(self.position_size, 4),
            "dollar_amount": round(self.dollar_amount, 2),
            "shares_or_units": round(self.shares_or_units, 6),
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "volatility_adjustment": round(self.volatility_adjustment, 4),
            "drawdown_adjustment": round(self.drawdown_adjustment, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class PortfolioSizing:
    """Position sizing for entire portfolio."""

    positions: Dict[str, SizingResult]
    total_allocation: float
    cash_reserve: float
    leverage: float
    risk_budget_used: float
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "total_allocation": round(self.total_allocation, 4),
            "cash_reserve": round(self.cash_reserve, 4),
            "leverage": round(self.leverage, 4),
            "risk_budget_used": round(self.risk_budget_used, 4),
            "warnings": self.warnings,
        }


class PositionSizer:
    """
    Advanced position sizing calculator.

    Features:
    - Multiple sizing methods (Kelly, volatility, risk parity)
    - Confidence-based scaling
    - Drawdown protection
    - Maximum position limits
    - Correlation adjustment

    Usage:
        sizer = PositionSizer(portfolio_value=10000)

        # Single position
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.KELLY,
            win_rate=0.55,
            win_loss_ratio=1.5,
            price=50000,
        )

        # Portfolio sizing
        portfolio = sizer.calculate_portfolio_sizes(
            signals=[
                {"symbol": "BTC/USDT", "confidence": 0.7, "price": 50000},
                {"symbol": "ETH/USDT", "confidence": 0.6, "price": 3000},
            ],
            method=SizingMethod.VOLATILITY_ADJUSTED,
            volatilities={"BTC/USDT": 0.05, "ETH/USDT": 0.07},
        )
    """

    def __init__(
        self,
        portfolio_value: float,
        max_position_size: float = 0.25,  # Max 25% per position
        max_portfolio_risk: float = 0.02,  # Max 2% risk per trade
        min_position_size: float = 0.01,  # Min 1% per position
        max_leverage: float = 1.0,  # No leverage by default
        risk_free_rate: float = 0.02,  # 2% annual
        min_cash_reserve: float = 0.10,  # Keep 10% in cash
    ):
        self.portfolio_value = portfolio_value
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.min_position_size = min_position_size
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.min_cash_reserve = min_cash_reserve

        # Track current drawdown for adjustments
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

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Calculate Kelly Criterion optimal fraction.

        f* = (p * b - q) / b

        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio

        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / Average loss

        Returns:
            Optimal fraction to bet
        """
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = win_loss_ratio

        kelly = (p * b - q) / b

        return max(0, kelly)

    def calculate_volatility_adjusted_size(
        self,
        volatility: float,
        target_volatility: float = 0.15,  # 15% annual target
    ) -> float:
        """
        Calculate position size based on volatility targeting.

        Args:
            volatility: Asset volatility (annualized)
            target_volatility: Target portfolio volatility

        Returns:
            Position size fraction
        """
        if volatility <= 0:
            return 0.0

        # Scale inversely with volatility
        size = target_volatility / volatility

        return min(1.0, size)

    def calculate_atr_based_size(
        self,
        atr: float,
        price: float,
        risk_per_trade: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate position size based on ATR.

        Args:
            atr: Average True Range
            price: Current asset price
            risk_per_trade: Risk amount per trade (default: max_portfolio_risk)

        Returns:
            Tuple of (position_size, stop_loss_distance)
        """
        if atr <= 0 or price <= 0:
            return 0.0, 0.0

        risk = risk_per_trade or self.max_portfolio_risk
        risk_amount = self.portfolio_value * risk

        # Use 2x ATR as stop loss
        stop_distance = 2 * atr
        stop_pct = stop_distance / price

        # Calculate shares/units
        shares = risk_amount / stop_distance

        # Convert to position size fraction
        position_value = shares * price
        position_size = position_value / self.portfolio_value

        return position_size, stop_pct

    def calculate_risk_parity_size(
        self,
        volatilities: Dict[str, float],
        correlations: Optional[pd.DataFrame] = None,
        target_risk: float = 0.10,  # 10% portfolio risk
    ) -> Dict[str, float]:
        """
        Calculate risk parity position sizes.

        Each asset contributes equally to portfolio risk.

        Args:
            volatilities: Dict of symbol -> volatility
            correlations: Optional correlation matrix
            target_risk: Target portfolio risk

        Returns:
            Dict of symbol -> position size
        """
        symbols = list(volatilities.keys())
        n = len(symbols)

        if n == 0:
            return {}

        vols = np.array([volatilities[s] for s in symbols])

        if correlations is not None:
            # Use full covariance matrix
            corr_matrix = correlations.loc[symbols, symbols].values
            cov_matrix = np.outer(vols, vols) * corr_matrix
        else:
            # Assume zero correlation
            cov_matrix = np.diag(vols**2)

        # Simple risk parity: inverse volatility weighting
        inv_vols = 1.0 / (vols + 1e-8)
        weights = inv_vols / inv_vols.sum()

        # Scale to target risk
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        if port_vol > 0:
            scale = target_risk / port_vol
            weights = weights * min(scale, self.max_leverage)

        return {s: w for s, w in zip(symbols, weights)}

    def calculate_optimal_f(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate Optimal f (Ralph Vince method).

        Maximizes geometric growth rate based on historical returns.

        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for worst case

        Returns:
            Optimal f fraction
        """
        if len(returns) < 20:
            return 0.0

        # Find worst loss
        sorted_returns = np.sort(returns)
        idx = int((1 - confidence_level) * len(sorted_returns))
        worst_loss = sorted_returns[idx]

        if worst_loss >= 0:
            return 1.0  # No losing trades

        # Search for optimal f that maximizes TWR (Terminal Wealth Relative)
        best_f = 0.0
        best_twr = 0.0

        for f in np.arange(0.01, 1.0, 0.01):
            # Calculate TWR
            hpr = 1 + f * (returns / (-worst_loss))
            twr = np.prod(hpr)

            if twr > best_twr:
                best_twr = twr
                best_f = f

        return best_f

    def apply_confidence_scaling(
        self,
        base_size: float,
        confidence: float,
        min_confidence: float = 0.5,
        max_scale: float = 1.5,
    ) -> Tuple[float, float]:
        """
        Scale position size based on signal confidence.

        Args:
            base_size: Base position size
            confidence: Signal confidence (0-1)
            min_confidence: Below this, reduce size
            max_scale: Maximum scaling factor

        Returns:
            Tuple of (scaled_size, scaling_factor)
        """
        if confidence < min_confidence:
            # Reduce size below minimum confidence
            scale = confidence / min_confidence
        elif confidence > 0.7:
            # Increase size for high confidence (up to max_scale)
            scale = 1 + (confidence - 0.7) * (max_scale - 1) / 0.3
        else:
            scale = 1.0

        scaled_size = base_size * scale

        return scaled_size, scale

    def apply_drawdown_protection(
        self,
        base_size: float,
        max_drawdown_reduction: float = 0.5,  # Reduce to 50% at max DD
        drawdown_threshold: float = 0.10,  # Start reducing at 10% DD
    ) -> Tuple[float, float]:
        """
        Reduce position size during drawdowns.

        Args:
            base_size: Base position size
            max_drawdown_reduction: Reduction at max drawdown
            drawdown_threshold: Drawdown level to start reducing

        Returns:
            Tuple of (adjusted_size, adjustment_factor)
        """
        if self._current_drawdown < drawdown_threshold:
            return base_size, 1.0

        # Linear reduction from threshold to 20% drawdown
        max_dd = 0.20
        reduction_range = max_dd - drawdown_threshold

        if self._current_drawdown >= max_dd:
            adjustment = max_drawdown_reduction
        else:
            progress = (self._current_drawdown - drawdown_threshold) / reduction_range
            adjustment = 1 - progress * (1 - max_drawdown_reduction)

        adjusted_size = base_size * adjustment

        return adjusted_size, adjustment

    def calculate_size(
        self,
        symbol: str,
        method: SizingMethod,
        price: float,
        confidence: float = 0.5,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.0,
        returns: Optional[np.ndarray] = None,
    ) -> SizingResult:
        """
        Calculate position size using specified method.

        Args:
            symbol: Asset symbol
            method: Sizing method to use
            price: Current asset price
            confidence: Signal confidence (0-1)
            volatility: Asset volatility (for volatility methods)
            atr: Average True Range (for ATR method)
            win_rate: Historical win rate
            win_loss_ratio: Avg win / Avg loss
            returns: Historical returns (for Optimal f)

        Returns:
            SizingResult with calculated position size
        """
        # Calculate base size based on method
        if method == SizingMethod.FIXED_FRACTION:
            base_size = self.max_position_size
            reasoning = f"Fixed fraction at {self.max_position_size:.1%}"

        elif method == SizingMethod.KELLY:
            base_size = self.calculate_kelly_fraction(win_rate, win_loss_ratio)
            reasoning = f"Kelly: WR={win_rate:.2%}, W/L={win_loss_ratio:.2f}"

        elif method == SizingMethod.HALF_KELLY:
            base_size = self.calculate_kelly_fraction(win_rate, win_loss_ratio) * 0.5
            reasoning = f"Half Kelly: WR={win_rate:.2%}, W/L={win_loss_ratio:.2f}"

        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            if volatility is None or volatility <= 0:
                base_size = self.max_position_size
                reasoning = "No volatility data, using max size"
            else:
                base_size = self.calculate_volatility_adjusted_size(volatility)
                reasoning = f"Vol-adjusted: vol={volatility:.2%}"

        elif method == SizingMethod.ATR_BASED:
            if atr is None or atr <= 0:
                base_size = self.max_position_size
                reasoning = "No ATR data, using max size"
            else:
                base_size, stop_pct = self.calculate_atr_based_size(atr, price)
                reasoning = f"ATR-based: ATR=${atr:.2f}, Stop={stop_pct:.2%}"

        elif method == SizingMethod.OPTIMAL_F:
            if returns is None or len(returns) < 20:
                base_size = self.max_position_size * 0.5
                reasoning = "Insufficient returns history, using conservative size"
            else:
                base_size = self.calculate_optimal_f(returns)
                reasoning = f"Optimal f from {len(returns)} returns"

        elif method == SizingMethod.CONFIDENCE_SCALED:
            base_size = self.max_position_size
            reasoning = "Base size to be scaled by confidence"

        else:
            base_size = self.max_position_size
            reasoning = "Default sizing"

        # Apply confidence scaling
        if method == SizingMethod.CONFIDENCE_SCALED or confidence != 0.5:
            base_size, conf_adj = self.apply_confidence_scaling(base_size, confidence)
        else:
            conf_adj = 1.0

        # Apply drawdown protection
        base_size, dd_adj = self.apply_drawdown_protection(base_size)

        # Apply volatility adjustment if not already done
        vol_adj = 1.0
        if method not in [SizingMethod.VOLATILITY_ADJUSTED, SizingMethod.ATR_BASED]:
            if volatility and volatility > 0.30:  # High volatility
                vol_adj = 0.30 / volatility
                base_size *= vol_adj

        # Apply position size limits
        final_size = max(self.min_position_size, min(self.max_position_size, base_size))

        # Calculate dollar amount and shares
        available_capital = self.portfolio_value * (1 - self.min_cash_reserve)
        dollar_amount = available_capital * final_size
        shares = dollar_amount / price if price > 0 else 0

        return SizingResult(
            symbol=symbol,
            method=method,
            position_size=final_size,
            dollar_amount=dollar_amount,
            shares_or_units=shares,
            confidence_adjustment=conf_adj,
            volatility_adjustment=vol_adj,
            drawdown_adjustment=dd_adj,
            reasoning=reasoning,
        )

    def calculate_portfolio_sizes(
        self,
        signals: List[Dict[str, Any]],
        method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED,
        volatilities: Optional[Dict[str, float]] = None,
        correlations: Optional[pd.DataFrame] = None,
    ) -> PortfolioSizing:
        """
        Calculate position sizes for multiple assets.

        Args:
            signals: List of signal dicts with symbol, confidence, price
            method: Sizing method to use
            volatilities: Dict of symbol -> volatility
            correlations: Optional correlation matrix

        Returns:
            PortfolioSizing with all position sizes
        """
        positions = {}
        warnings = []
        total_allocation = 0.0

        # Use risk parity if correlations provided
        if method == SizingMethod.RISK_PARITY and volatilities:
            rp_sizes = self.calculate_risk_parity_size(
                volatilities,
                correlations,
            )
            for signal in signals:
                symbol = signal["symbol"]
                if symbol in rp_sizes:
                    price = signal.get("price", 1.0)
                    confidence = signal.get("confidence", 0.5)

                    size = rp_sizes[symbol]

                    # Apply confidence and drawdown adjustments
                    size, conf_adj = self.apply_confidence_scaling(size, confidence)
                    size, dd_adj = self.apply_drawdown_protection(size)

                    available_capital = self.portfolio_value * (1 - self.min_cash_reserve)
                    dollar_amount = available_capital * size
                    shares = dollar_amount / price if price > 0 else 0

                    positions[symbol] = SizingResult(
                        symbol=symbol,
                        method=SizingMethod.RISK_PARITY,
                        position_size=size,
                        dollar_amount=dollar_amount,
                        shares_or_units=shares,
                        confidence_adjustment=conf_adj,
                        volatility_adjustment=1.0,
                        drawdown_adjustment=dd_adj,
                        reasoning="Risk parity allocation",
                    )
                    total_allocation += size
        else:
            # Calculate individual sizes
            for signal in signals:
                symbol = signal["symbol"]
                price = signal.get("price", 1.0)
                confidence = signal.get("confidence", 0.5)
                vol = volatilities.get(symbol) if volatilities else None
                atr = signal.get("atr")

                result = self.calculate_size(
                    symbol=symbol,
                    method=method,
                    price=price,
                    confidence=confidence,
                    volatility=vol,
                    atr=atr,
                )

                positions[symbol] = result
                total_allocation += result.position_size

        # Check total allocation
        max_total = 1.0 - self.min_cash_reserve
        if total_allocation > max_total:
            warnings.append(
                f"Total allocation ({total_allocation:.1%}) exceeds max ({max_total:.1%})"
            )

            # Scale down proportionally
            scale = max_total / total_allocation
            for symbol in positions:
                positions[symbol].position_size *= scale
                positions[symbol].dollar_amount *= scale
                positions[symbol].shares_or_units *= scale

            total_allocation = max_total

        # Check leverage
        leverage = total_allocation / (1 - self.min_cash_reserve)
        if leverage > self.max_leverage:
            warnings.append(f"Leverage ({leverage:.2f}x) exceeds max ({self.max_leverage}x)")

        # Calculate risk budget used
        risk_budget = 0.0
        for symbol, pos in positions.items():
            vol = volatilities.get(symbol, 0.20) if volatilities else 0.20
            risk_budget += pos.position_size * vol

        return PortfolioSizing(
            positions=positions,
            total_allocation=total_allocation,
            cash_reserve=1 - total_allocation,
            leverage=leverage,
            risk_budget_used=risk_budget,
            warnings=warnings,
        )

    def suggest_sizing_method(
        self,
        historical_win_rate: Optional[float] = None,
        historical_returns: Optional[np.ndarray] = None,
        current_volatility: Optional[float] = None,
    ) -> Tuple[SizingMethod, str]:
        """
        Suggest the best sizing method based on available data.

        Returns:
            Tuple of (recommended method, reasoning)
        """
        # If we have good win rate data, use Kelly or Half Kelly
        if historical_win_rate is not None and historical_win_rate > 0.5:
            if historical_win_rate > 0.6:
                return (
                    SizingMethod.KELLY,
                    f"Strong edge (WR={historical_win_rate:.1%}) supports Kelly sizing",
                )
            else:
                return SizingMethod.HALF_KELLY, f"Moderate edge supports conservative Half Kelly"

        # If we have returns history, use Optimal f
        if historical_returns is not None and len(historical_returns) >= 100:
            return SizingMethod.OPTIMAL_F, "Sufficient history for Optimal f calculation"

        # If volatility data available, use volatility-adjusted
        if current_volatility is not None:
            if current_volatility > 0.50:  # Very high volatility
                return SizingMethod.ATR_BASED, "High volatility - ATR-based sizing recommended"
            else:
                return (
                    SizingMethod.VOLATILITY_ADJUSTED,
                    "Volatility-adjusted sizing for consistency",
                )

        # Default to fixed fraction with confidence scaling
        return SizingMethod.CONFIDENCE_SCALED, "Using confidence-scaled fixed fraction"
