"""
Portfolio-Level Risk Management Module.

Features:
- Portfolio Value at Risk (VaR)
- Correlation-based risk assessment
- Sector/asset class exposure limits
- Portfolio beta management
- Diversification scoring
- Rebalancing recommendations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .risk_manager import RiskManager, RiskConfig, RiskLevel, TradingStatus


@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk management."""

    # VaR settings
    var_confidence_level: float = 0.95  # 95% VaR
    var_lookback_days: int = 252  # 1 year of daily returns
    max_var_pct: float = 5.0  # Max 5% daily VaR

    # Correlation settings
    correlation_threshold: float = 0.7  # Consider correlated above 0.7
    max_correlated_weight: float = 0.4  # Max 40% in correlated assets

    # Diversification
    min_positions: int = 3  # Minimum positions for diversification
    max_single_asset_weight: float = 0.25  # Max 25% in single asset
    target_hhi: float = 0.15  # Target Herfindahl-Hirschman Index

    # Sector exposure
    max_sector_weight: float = 0.4  # Max 40% in single sector
    sector_mappings: Dict[str, str] = field(default_factory=dict)

    # Beta management
    target_portfolio_beta: float = 1.0
    max_beta: float = 1.5
    min_beta: float = 0.5

    # Rebalancing
    rebalance_threshold_pct: float = 5.0  # Rebalance when >5% drift
    rebalance_frequency_days: int = 7


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics."""

    total_value: float
    var_95: float  # 95% daily Value at Risk
    var_99: float  # 99% daily Value at Risk
    portfolio_volatility: float  # Annualized volatility
    portfolio_beta: float
    sharpe_ratio: float
    diversification_ratio: float
    hhi_concentration: float  # Herfindahl-Hirschman Index
    max_correlation: float
    sector_exposure: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_value": round(self.total_value, 2),
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "portfolio_volatility": round(self.portfolio_volatility, 4),
            "portfolio_beta": round(self.portfolio_beta, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "hhi_concentration": round(self.hhi_concentration, 4),
            "max_correlation": round(self.max_correlation, 4),
            "sector_exposure": {k: round(v, 4) for k, v in self.sector_exposure.items()},
        }


@dataclass
class Position:
    """Portfolio position."""

    symbol: str
    quantity: float
    current_price: float
    avg_cost: float
    market_type: str = "unknown"
    sector: str = "unknown"
    beta: float = 1.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis <= 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "current_price": self.current_price,
            "avg_cost": self.avg_cost,
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "market_type": self.market_type,
            "sector": self.sector,
            "beta": self.beta,
        }


@dataclass
class RebalanceRecommendation:
    """Rebalancing recommendation."""

    symbol: str
    current_weight: float
    target_weight: float
    action: str  # "BUY", "SELL", "HOLD"
    amount: float  # In dollars
    reason: str

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "action": self.action,
            "amount": round(self.amount, 2),
            "reason": self.reason,
        }


class PortfolioRiskManager:
    """
    Portfolio-level risk management.

    Extends individual position risk management with portfolio-wide metrics:
    - Value at Risk (VaR) calculation
    - Correlation-based risk assessment
    - Concentration risk monitoring
    - Rebalancing recommendations
    """

    def __init__(
        self,
        config: Optional[PortfolioRiskConfig] = None,
        risk_manager: Optional[RiskManager] = None,
        data_dir: str = "data/portfolio",
    ):
        self.config = config or PortfolioRiskConfig()
        self.risk_manager = risk_manager
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash_balance: float = 0.0
        self.target_weights: Dict[str, float] = {}

        # Historical data for risk calculations
        self._returns_history: Dict[str, List[float]] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

        # Default sector mappings
        self._init_sector_mappings()

        # Load state
        self._load_state()

    def _init_sector_mappings(self):
        """Initialize default sector mappings."""
        default_mappings = {
            # Crypto
            "BTC/USDT": "crypto",
            "ETH/USDT": "crypto",
            "BTC-USD": "crypto",
            "ETH-USD": "crypto",
            # Commodities
            "GC=F": "commodities",
            "SI=F": "commodities",
            "CL=F": "energy",
            "NG=F": "energy",
            # Tech stocks
            "AAPL": "technology",
            "MSFT": "technology",
            "GOOGL": "technology",
            "AMZN": "technology",
            "NVDA": "technology",
            # Finance
            "JPM": "finance",
            "GS": "finance",
            "BAC": "finance",
        }
        self.config.sector_mappings.update(default_mappings)

    def update_position(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
        avg_cost: Optional[float] = None,
        market_type: str = "unknown",
        beta: float = 1.0,
    ):
        """Update or create a position."""
        if quantity <= 0:
            self.positions.pop(symbol, None)
            return

        sector = self.config.sector_mappings.get(symbol, "other")

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            current_price=current_price,
            avg_cost=avg_cost or current_price,
            market_type=market_type,
            sector=sector,
            beta=beta,
        )
        self._save_state()

    def update_cash(self, cash_balance: float):
        """Update cash balance."""
        self.cash_balance = cash_balance
        self._save_state()

    def set_target_weights(self, weights: Dict[str, float]):
        """Set target portfolio weights."""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            weights = {k: v / total for k, v in weights.items()}
        self.target_weights = weights
        self._save_state()

    def add_returns(self, symbol: str, daily_return: float):
        """Add daily return for correlation/VaR calculation."""
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []

        self._returns_history[symbol].append(daily_return)

        # Keep only lookback period
        max_len = self.config.var_lookback_days
        if len(self._returns_history[symbol]) > max_len:
            self._returns_history[symbol] = self._returns_history[symbol][-max_len:]

        # Recalculate correlation matrix periodically
        if len(self._returns_history[symbol]) % 20 == 0:
            self._update_correlation_matrix()

    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        total_value = self.get_total_value()
        weights = self.get_weights()

        # Default metrics
        var_95 = 0.0
        var_99 = 0.0
        portfolio_vol = 0.0
        portfolio_beta = 1.0
        sharpe = 0.0
        div_ratio = 0.0
        hhi = 1.0
        max_corr = 0.0
        sector_exp = {}

        if not self.positions:
            return PortfolioMetrics(
                total_value=total_value,
                var_95=var_95,
                var_99=var_99,
                portfolio_volatility=portfolio_vol,
                portfolio_beta=portfolio_beta,
                sharpe_ratio=sharpe,
                diversification_ratio=div_ratio,
                hhi_concentration=hhi,
                max_correlation=max_corr,
                sector_exposure=sector_exp,
            )

        # Calculate VaR using historical simulation
        var_95, var_99 = self._calculate_var(total_value, weights)

        # Portfolio volatility and beta
        portfolio_vol = self._calculate_portfolio_volatility(weights)
        portfolio_beta = self._calculate_portfolio_beta(weights)

        # Sharpe ratio (assuming risk-free rate = 0)
        if portfolio_vol > 0:
            avg_return = self._calculate_portfolio_expected_return(weights)
            sharpe = avg_return / portfolio_vol

        # Diversification ratio
        div_ratio = self._calculate_diversification_ratio(weights)

        # HHI concentration
        hhi = sum(w**2 for w in weights.values())

        # Max correlation
        max_corr = self._get_max_correlation()

        # Sector exposure
        sector_exp = self._calculate_sector_exposure()

        return PortfolioMetrics(
            total_value=total_value,
            var_95=var_95,
            var_99=var_99,
            portfolio_volatility=portfolio_vol,
            portfolio_beta=portfolio_beta,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
            hhi_concentration=hhi,
            max_correlation=max_corr,
            sector_exposure=sector_exp,
        )

    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment.

        Returns:
            Dict with risk assessment, warnings, and recommendations
        """
        metrics = self.calculate_portfolio_metrics()
        weights = self.get_weights()

        warnings = []
        risk_level = RiskLevel.NORMAL
        status = TradingStatus.ALLOWED

        # VaR check
        if metrics.var_95 > metrics.total_value * (self.config.max_var_pct / 100):
            warnings.append(
                f"VaR exceeds limit: ${metrics.var_95:.2f} > {self.config.max_var_pct}%"
            )
            risk_level = RiskLevel.ELEVATED

        # Concentration check
        if metrics.hhi_concentration > self.config.target_hhi * 2:
            warnings.append(f"Portfolio too concentrated: HHI={metrics.hhi_concentration:.2f}")
            risk_level = max(risk_level, RiskLevel.ELEVATED, key=lambda x: list(RiskLevel).index(x))

        # Single asset check
        for symbol, weight in weights.items():
            if weight > self.config.max_single_asset_weight:
                warnings.append(f"{symbol} exceeds max weight: {weight:.1%}")
                risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))

        # Correlation check
        if metrics.max_correlation > self.config.correlation_threshold:
            warnings.append(f"High correlation detected: {metrics.max_correlation:.2f}")

        # Beta check
        if metrics.portfolio_beta > self.config.max_beta:
            warnings.append(f"Portfolio beta too high: {metrics.portfolio_beta:.2f}")
            status = TradingStatus.REDUCED
        elif metrics.portfolio_beta < self.config.min_beta:
            warnings.append(f"Portfolio beta too low: {metrics.portfolio_beta:.2f}")

        # Sector exposure check
        for sector, exposure in metrics.sector_exposure.items():
            if exposure > self.config.max_sector_weight:
                warnings.append(f"{sector} sector exposure high: {exposure:.1%}")

        # Diversification check
        if len(self.positions) < self.config.min_positions:
            warnings.append(f"Insufficient diversification: {len(self.positions)} positions")

        return {
            "status": status.value,
            "risk_level": risk_level.value,
            "metrics": metrics.to_dict(),
            "warnings": warnings,
            "positions_count": len(self.positions),
            "total_value": round(metrics.total_value, 2),
        }

    def get_rebalancing_recommendations(self) -> List[RebalanceRecommendation]:
        """
        Generate rebalancing recommendations.

        Returns:
            List of RebalanceRecommendation objects
        """
        if not self.target_weights:
            return []

        recommendations = []
        current_weights = self.get_weights()
        total_value = self.get_total_value()

        for symbol in set(self.target_weights.keys()) | set(current_weights.keys()):
            current = current_weights.get(symbol, 0.0)
            target = self.target_weights.get(symbol, 0.0)
            drift = abs(current - target) * 100

            if drift < self.config.rebalance_threshold_pct:
                continue

            amount = (target - current) * total_value

            if amount > 0:
                action = "BUY"
                reason = f"Under target weight by {drift:.1f}%"
            elif amount < 0:
                action = "SELL"
                reason = f"Over target weight by {drift:.1f}%"
            else:
                continue

            recommendations.append(
                RebalanceRecommendation(
                    symbol=symbol,
                    current_weight=current,
                    target_weight=target,
                    action=action,
                    amount=abs(amount),
                    reason=reason,
                )
            )

        # Sort by absolute drift
        recommendations.sort(key=lambda x: abs(x.current_weight - x.target_weight), reverse=True)

        return recommendations

    def can_add_position(
        self,
        symbol: str,
        value: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding a position is allowed.

        Args:
            symbol: Symbol to add
            value: Proposed position value

        Returns:
            Tuple of (allowed, list of reasons if not allowed)
        """
        reasons = []
        total_value = self.get_total_value()
        new_total = total_value + value

        # Single position limit
        position_weight = value / new_total if new_total > 0 else 1.0
        if position_weight > self.config.max_single_asset_weight:
            reasons.append(
                f"Position would exceed max weight ({position_weight:.1%} > {self.config.max_single_asset_weight:.1%})"
            )

        # Sector limit
        sector = self.config.sector_mappings.get(symbol, "other")
        current_sector_value = sum(
            p.market_value for p in self.positions.values() if p.sector == sector
        )
        new_sector_weight = (current_sector_value + value) / new_total if new_total > 0 else 1.0
        if new_sector_weight > self.config.max_sector_weight:
            reasons.append(
                f"Would exceed {sector} sector limit ({new_sector_weight:.1%} > {self.config.max_sector_weight:.1%})"
            )

        # Correlation check
        max_corr = self._get_correlation_with_portfolio(symbol)
        if max_corr > self.config.correlation_threshold:
            # Check correlated weight
            correlated_value = self._get_correlated_position_value(symbol)
            correlated_weight = (correlated_value + value) / new_total if new_total > 0 else 1.0
            if correlated_weight > self.config.max_correlated_weight:
                reasons.append(f"Would exceed correlated assets limit ({correlated_weight:.1%})")

        return len(reasons) == 0, reasons

    def get_total_value(self) -> float:
        """Get total portfolio value including cash."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return positions_value + self.cash_balance

    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        total = self.get_total_value()
        if total <= 0:
            return {}
        return {symbol: pos.market_value / total for symbol, pos in self.positions.items()}

    def get_positions_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        return [pos.to_dict() for pos in self.positions.values()]

    def _calculate_var(
        self,
        total_value: float,
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """Calculate Value at Risk using historical simulation."""
        if not self._returns_history:
            return 0.0, 0.0

        # Get portfolio returns
        portfolio_returns = []
        min_len = (
            min(len(returns) for returns in self._returns_history.values())
            if self._returns_history
            else 0
        )

        if min_len < 20:  # Need at least 20 data points
            return 0.0, 0.0

        for i in range(min_len):
            portfolio_return = sum(
                weights.get(symbol, 0) * self._returns_history[symbol][i]
                for symbol in self._returns_history.keys()
            )
            portfolio_returns.append(portfolio_return)

        if not portfolio_returns:
            return 0.0, 0.0

        # Calculate VaR as percentile of losses
        var_95 = -np.percentile(portfolio_returns, 5) * total_value
        var_99 = -np.percentile(portfolio_returns, 1) * total_value

        return max(var_95, 0), max(var_99, 0)

    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility."""
        if not self._returns_history:
            return 0.0

        # Get individual volatilities
        volatilities = {}
        for symbol, returns in self._returns_history.items():
            if len(returns) >= 20:
                volatilities[symbol] = np.std(returns) * np.sqrt(252)

        if not volatilities:
            return 0.0

        # Simple weighted average (ignoring correlations for speed)
        portfolio_vol = sum(weights.get(symbol, 0) * vol for symbol, vol in volatilities.items())

        return portfolio_vol

    def _calculate_portfolio_beta(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio beta."""
        return sum(weights.get(symbol, 0) * pos.beta for symbol, pos in self.positions.items())

    def _calculate_portfolio_expected_return(self, weights: Dict[str, float]) -> float:
        """Calculate expected return based on historical average."""
        if not self._returns_history:
            return 0.0

        total_return = 0.0
        for symbol, returns in self._returns_history.items():
            if len(returns) >= 20:
                avg_return = np.mean(returns) * 252  # Annualized
                total_return += weights.get(symbol, 0) * avg_return

        return total_return

    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate diversification ratio."""
        if len(weights) < 2:
            return 0.0

        # Weighted average volatility
        weighted_vol = 0.0
        portfolio_vol = self._calculate_portfolio_volatility(weights)

        for symbol, returns in self._returns_history.items():
            if len(returns) >= 20:
                vol = np.std(returns) * np.sqrt(252)
                weighted_vol += weights.get(symbol, 0) * vol

        if portfolio_vol <= 0:
            return 0.0

        return weighted_vol / portfolio_vol

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure."""
        total = self.get_total_value()
        if total <= 0:
            return {}

        sector_values: Dict[str, float] = {}
        for pos in self.positions.values():
            sector = pos.sector
            sector_values[sector] = sector_values.get(sector, 0) + pos.market_value

        return {sector: value / total for sector, value in sector_values.items()}

    def _update_correlation_matrix(self):
        """Update correlation matrix from returns history."""
        if len(self._returns_history) < 2:
            return

        # Find minimum common length
        min_len = min(len(r) for r in self._returns_history.values())
        if min_len < 20:
            return

        # Build returns DataFrame
        data = {symbol: returns[-min_len:] for symbol, returns in self._returns_history.items()}
        df = pd.DataFrame(data)

        self._correlation_matrix = df.corr()

    def _get_max_correlation(self) -> float:
        """Get maximum pairwise correlation."""
        if self._correlation_matrix is None:
            return 0.0

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(self._correlation_matrix, dtype=bool), k=1)
        upper_tri = self._correlation_matrix.values[mask]

        if len(upper_tri) == 0:
            return 0.0

        return float(np.max(upper_tri))

    def _get_correlation_with_portfolio(self, symbol: str) -> float:
        """Get max correlation of symbol with existing positions."""
        if self._correlation_matrix is None or symbol not in self._correlation_matrix.columns:
            return 0.0

        existing_symbols = [
            s for s in self.positions.keys() if s in self._correlation_matrix.columns
        ]
        if not existing_symbols:
            return 0.0

        correlations = [
            abs(self._correlation_matrix.loc[symbol, s]) for s in existing_symbols if s != symbol
        ]

        return max(correlations) if correlations else 0.0

    def _get_correlated_position_value(self, symbol: str) -> float:
        """Get total value of positions correlated with symbol."""
        if self._correlation_matrix is None or symbol not in self._correlation_matrix.columns:
            return 0.0

        correlated_value = 0.0
        threshold = self.config.correlation_threshold

        for pos_symbol, pos in self.positions.items():
            if pos_symbol in self._correlation_matrix.columns:
                corr = abs(self._correlation_matrix.loc[symbol, pos_symbol])
                if corr >= threshold:
                    correlated_value += pos.market_value

        return correlated_value

    def _save_state(self):
        """Save portfolio state."""
        state_file = self.data_dir / "portfolio_state.json"

        state = {
            "cash_balance": self.cash_balance,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_type": pos.market_type,
                    "beta": pos.beta,
                }
                for symbol, pos in self.positions.items()
            },
            "target_weights": self.target_weights,
            "updated_at": datetime.now().isoformat(),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load portfolio state."""
        state_file = self.data_dir / "portfolio_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self.cash_balance = state.get("cash_balance", 0.0)
            self.target_weights = state.get("target_weights", {})

            for symbol, pos_data in state.get("positions", {}).items():
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=pos_data["quantity"],
                    current_price=0,  # Will be updated
                    avg_cost=pos_data["avg_cost"],
                    market_type=pos_data.get("market_type", "unknown"),
                    sector=self.config.sector_mappings.get(symbol, "other"),
                    beta=pos_data.get("beta", 1.0),
                )
        except (json.JSONDecodeError, KeyError):
            pass
