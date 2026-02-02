"""
Risk Budget Engine.

Production-grade per-trade risk allocation:
- Per-trade risk budget based on confidence and regime
- Leverage caps per regime
- Correlation penalty for portfolio concentration
- VaR and CVaR guardrails
- Dynamic position sizing integration
- Kelly criterion optimization (optional)

Integrates with position_sizer.py, advanced_trading_brain.py, and leverage_manager.py.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


class RiskBudgetTier(Enum):
    """Risk budget tiers based on conditions."""
    FULL = "full"  # Normal risk budget
    REDUCED = "reduced"  # 50% of normal
    MINIMAL = "minimal"  # 25% of normal
    ZERO = "zero"  # No trading


@dataclass
class RiskBudget:
    """Per-trade risk budget allocation."""
    trade_id: str
    symbol: str

    # Position sizing
    max_position_usd: float
    max_position_pct: float
    recommended_position_usd: float

    # Risk metrics
    max_risk_usd: float  # Max $ to risk on this trade
    max_risk_pct: float  # % of portfolio to risk
    stop_distance_pct: float  # Required SL distance

    # Leverage
    max_leverage: float
    recommended_leverage: float

    # Adjustments applied
    tier: RiskBudgetTier
    adjustments: Dict[str, float] = field(default_factory=dict)

    # Context
    regime: str = ""
    confidence: float = 0.0
    correlation_penalty: float = 0.0

    # Guardrails
    var_95_usd: float = 0.0  # 95% VaR
    cvar_95_usd: float = 0.0  # 95% CVaR (Expected Shortfall)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "max_position_usd": self.max_position_usd,
            "max_position_pct": self.max_position_pct,
            "recommended_position_usd": self.recommended_position_usd,
            "max_risk_usd": self.max_risk_usd,
            "max_risk_pct": self.max_risk_pct,
            "max_leverage": self.max_leverage,
            "recommended_leverage": self.recommended_leverage,
            "tier": self.tier.value,
            "adjustments": self.adjustments,
            "regime": self.regime,
            "confidence": self.confidence,
            "correlation_penalty": self.correlation_penalty,
            "var_95_usd": self.var_95_usd,
            "cvar_95_usd": self.cvar_95_usd,
        }


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state for budget calculations."""
    total_equity: float
    current_exposure: float  # Total position value
    exposure_pct: float
    position_count: int
    position_values: Dict[str, float]  # symbol -> value
    position_correlations: Dict[str, float]  # symbol -> correlation with portfolio
    current_drawdown_pct: float
    daily_pnl_pct: float
    win_streak: int
    loss_streak: int
    recent_volatility: float


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budget engine."""

    # Base risk parameters
    base_risk_per_trade_pct: float = 0.01  # 1% base risk
    max_risk_per_trade_pct: float = 0.02  # 2% max risk
    min_risk_per_trade_pct: float = 0.002  # 0.2% min risk

    # Position limits
    max_position_pct: float = 0.20  # 20% max single position
    max_total_exposure_pct: float = 1.0  # 100% max total exposure
    max_correlated_exposure_pct: float = 0.50  # 50% max in correlated positions

    # Leverage limits by regime
    leverage_limits_by_regime: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BULL": 3.0,
        "BULL": 2.5,
        "RECOVERY": 2.0,
        "SIDEWAYS": 1.5,
        "LOW_VOL": 2.0,
        "HIGH_VOL": 1.2,
        "BEAR": 1.5,
        "STRONG_BEAR": 1.2,
        "CRASH": 1.0,
        "unknown": 1.5,
    })

    # Confidence adjustments
    confidence_multipliers: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # (min_conf, max_conf) -> multiplier
        "high": (0.7, 1.0),  # 70%+ confidence = full budget
        "medium": (0.5, 0.7),  # 50-70% = 70% budget
        "low": (0.3, 0.5),  # 30-50% = 40% budget
        "very_low": (0.0, 0.3),  # <30% = no trading
    })

    # Drawdown adjustments
    drawdown_scaling: Dict[float, float] = field(default_factory=lambda: {
        0.02: 1.0,  # <2% DD = full budget
        0.05: 0.7,  # 2-5% DD = 70%
        0.08: 0.4,  # 5-8% DD = 40%
        0.10: 0.1,  # 8-10% DD = 10%
        1.0: 0.0,  # >10% DD = no trading
    })

    # Streak adjustments
    win_streak_boost_threshold: int = 3  # After 3 wins
    win_streak_boost_max: float = 1.2  # Up to 20% boost
    loss_streak_reduce_threshold: int = 2  # After 2 losses
    loss_streak_reduce_factor: float = 0.5  # 50% reduction per loss

    # VaR/CVaR settings
    var_confidence_level: float = 0.95
    var_lookback_days: int = 30
    max_var_pct: float = 0.05  # Max 5% daily VaR

    # Kelly criterion
    use_kelly: bool = False
    kelly_fraction: float = 0.25  # Use 25% Kelly for safety
    min_trades_for_kelly: int = 30

    # Correlation threshold
    high_correlation_threshold: float = 0.7


class RiskBudgetEngine:
    """
    Production-grade risk budget allocation engine.

    Calculates per-trade risk budgets based on:
    - Portfolio state and exposure
    - Market regime
    - Signal confidence
    - Correlation with existing positions
    - Drawdown state
    - Win/loss streaks
    - VaR/CVaR constraints
    """

    def __init__(self, config: Optional[RiskBudgetConfig] = None):
        self.config = config or RiskBudgetConfig()
        self._lock = RLock()

        # Historical returns for VaR calculation
        self._daily_returns: List[float] = []
        self._last_var_calculation: Optional[datetime] = None
        self._cached_var: float = 0.0
        self._cached_cvar: float = 0.0

        # Kelly stats
        self._wins: int = 0
        self._losses: int = 0
        self._total_win_pct: float = 0.0
        self._total_loss_pct: float = 0.0

        logger.info("RiskBudgetEngine initialized")

    def calculate_budget(
        self,
        symbol: str,
        side: str,
        signal_confidence: float,
        regime: str,
        portfolio_state: PortfolioRiskState,
        price: float,
        volatility: float = 0.02,
        correlation_with_portfolio: float = 0.0,
    ) -> RiskBudget:
        """
        Calculate risk budget for a potential trade.

        Args:
            symbol: Trading symbol
            side: BUY/SELL or LONG/SHORT
            signal_confidence: ML/strategy confidence (0-1)
            regime: Current market regime
            portfolio_state: Current portfolio state
            price: Current price
            volatility: Asset volatility (daily)
            correlation_with_portfolio: Correlation with existing positions

        Returns:
            RiskBudget with allocation details
        """
        with self._lock:
            trade_id = f"RB_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            adjustments = {}
            equity = portfolio_state.total_equity

            # 1. Determine base tier
            tier = self._determine_tier(
                signal_confidence, regime, portfolio_state
            )
            adjustments["tier"] = tier.value

            if tier == RiskBudgetTier.ZERO:
                return self._zero_budget(trade_id, symbol, regime, signal_confidence)

            # 2. Calculate base risk amount
            base_risk_pct = self.config.base_risk_per_trade_pct
            base_risk_usd = equity * base_risk_pct

            # 3. Apply confidence adjustment
            conf_multiplier = self._get_confidence_multiplier(signal_confidence)
            adjusted_risk = base_risk_usd * conf_multiplier
            adjustments["confidence_multiplier"] = conf_multiplier

            # 4. Apply drawdown scaling
            dd_multiplier = self._get_drawdown_multiplier(portfolio_state.current_drawdown_pct)
            adjusted_risk *= dd_multiplier
            adjustments["drawdown_multiplier"] = dd_multiplier

            # 5. Apply streak adjustment
            streak_multiplier = self._get_streak_multiplier(
                portfolio_state.win_streak, portfolio_state.loss_streak
            )
            adjusted_risk *= streak_multiplier
            adjustments["streak_multiplier"] = streak_multiplier

            # 6. Apply tier reduction
            tier_multiplier = {
                RiskBudgetTier.FULL: 1.0,
                RiskBudgetTier.REDUCED: 0.5,
                RiskBudgetTier.MINIMAL: 0.25,
            }.get(tier, 1.0)
            adjusted_risk *= tier_multiplier
            adjustments["tier_multiplier"] = tier_multiplier

            # 7. Calculate correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(
                symbol, correlation_with_portfolio, portfolio_state
            )
            adjusted_risk *= (1 - correlation_penalty)
            adjustments["correlation_penalty"] = correlation_penalty

            # 8. Kelly criterion (if enabled)
            if self.config.use_kelly and self._has_kelly_stats():
                kelly_risk = self._calculate_kelly_risk(equity)
                adjusted_risk = min(adjusted_risk, kelly_risk)
                adjustments["kelly_capped"] = kelly_risk < adjusted_risk

            # 9. Apply limits
            max_risk = equity * self.config.max_risk_per_trade_pct
            min_risk = equity * self.config.min_risk_per_trade_pct
            final_risk = max(min_risk, min(max_risk, adjusted_risk))

            # 10. Calculate position sizing
            max_leverage = self._get_leverage_cap(regime, portfolio_state)
            recommended_leverage = max_leverage * conf_multiplier

            # Position size based on stop loss distance
            stop_distance = volatility * 2  # 2x volatility as default stop
            stop_distance = max(0.01, min(0.10, stop_distance))  # 1-10% range

            if stop_distance > 0:
                max_position_usd = final_risk / stop_distance
            else:
                max_position_usd = final_risk / 0.02  # Default 2% stop

            # Apply position limit
            max_position_pct = min(
                self.config.max_position_pct,
                (self.config.max_total_exposure_pct - portfolio_state.exposure_pct)
            )
            max_position_limit = equity * max_position_pct
            max_position_usd = min(max_position_usd, max_position_limit)

            # Recommended size (conservative)
            recommended_position = max_position_usd * 0.7

            # 11. Calculate VaR/CVaR
            var_95, cvar_95 = self._calculate_var_cvar(
                recommended_position, volatility
            )

            # 12. VaR check
            if var_95 > equity * self.config.max_var_pct:
                # Reduce position to meet VaR limit
                scale_factor = (equity * self.config.max_var_pct) / var_95
                recommended_position *= scale_factor
                max_position_usd *= scale_factor
                adjustments["var_limited"] = True

            return RiskBudget(
                trade_id=trade_id,
                symbol=symbol,
                max_position_usd=max_position_usd,
                max_position_pct=max_position_usd / equity if equity > 0 else 0,
                recommended_position_usd=recommended_position,
                max_risk_usd=final_risk,
                max_risk_pct=final_risk / equity if equity > 0 else 0,
                stop_distance_pct=stop_distance,
                max_leverage=max_leverage,
                recommended_leverage=recommended_leverage,
                tier=tier,
                adjustments=adjustments,
                regime=regime,
                confidence=signal_confidence,
                correlation_penalty=correlation_penalty,
                var_95_usd=var_95,
                cvar_95_usd=cvar_95,
            )

    def _determine_tier(
        self,
        confidence: float,
        regime: str,
        state: PortfolioRiskState,
    ) -> RiskBudgetTier:
        """Determine risk budget tier based on conditions."""
        # Very low confidence = no trading
        if confidence < 0.3:
            return RiskBudgetTier.ZERO

        # High drawdown = minimal trading
        if state.current_drawdown_pct > 0.08:
            return RiskBudgetTier.MINIMAL

        # Crash regime = minimal trading
        if regime == "CRASH":
            return RiskBudgetTier.MINIMAL

        # High loss streak = reduced
        if state.loss_streak >= 3:
            return RiskBudgetTier.REDUCED

        # Bearish with low confidence = reduced
        if regime in ["BEAR", "STRONG_BEAR"] and confidence < 0.6:
            return RiskBudgetTier.REDUCED

        # High volatility regime with medium confidence = reduced
        if regime == "HIGH_VOL" and confidence < 0.65:
            return RiskBudgetTier.REDUCED

        # Good conditions
        if confidence >= 0.6 and state.current_drawdown_pct < 0.05:
            return RiskBudgetTier.FULL

        return RiskBudgetTier.REDUCED

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get multiplier based on signal confidence."""
        if confidence >= 0.7:
            return 1.0
        elif confidence >= 0.5:
            return 0.7
        elif confidence >= 0.3:
            return 0.4
        else:
            return 0.0

    def _get_drawdown_multiplier(self, drawdown_pct: float) -> float:
        """Get multiplier based on current drawdown."""
        for threshold, multiplier in sorted(
            self.config.drawdown_scaling.items()
        ):
            if drawdown_pct <= threshold:
                return multiplier
        return 0.0

    def _get_streak_multiplier(self, win_streak: int, loss_streak: int) -> float:
        """Get multiplier based on win/loss streaks."""
        if loss_streak >= self.config.loss_streak_reduce_threshold:
            # Reduce by 50% for each loss beyond threshold
            reduction = (loss_streak - self.config.loss_streak_reduce_threshold + 1)
            return max(0.1, self.config.loss_streak_reduce_factor ** reduction)

        if win_streak >= self.config.win_streak_boost_threshold:
            # Boost up to 20% for hot streaks
            boost = min(
                win_streak - self.config.win_streak_boost_threshold + 1,
                3  # Max 3 boosts
            ) * 0.07
            return min(self.config.win_streak_boost_max, 1.0 + boost)

        return 1.0

    def _calculate_correlation_penalty(
        self,
        symbol: str,
        correlation: float,
        state: PortfolioRiskState,
    ) -> float:
        """Calculate penalty for correlated positions."""
        if not state.position_correlations or symbol in state.position_values:
            return 0.0

        # Check if adding this position would exceed correlation limit
        if abs(correlation) > self.config.high_correlation_threshold:
            # Calculate how much correlated exposure we'd have
            correlated_exposure = sum(
                value for sym, value in state.position_values.items()
                if state.position_correlations.get(sym, 0) > self.config.high_correlation_threshold
            )

            current_ratio = correlated_exposure / max(state.total_equity, 1)

            # Penalty increases as we approach the limit
            if current_ratio > self.config.max_correlated_exposure_pct * 0.8:
                return min(0.5, (current_ratio - self.config.max_correlated_exposure_pct * 0.8) * 2)

        return abs(correlation) * 0.2  # Base correlation penalty

    def _get_leverage_cap(
        self,
        regime: str,
        state: PortfolioRiskState,
    ) -> float:
        """Get maximum leverage for current regime and conditions."""
        base_leverage = self.config.leverage_limits_by_regime.get(regime, 1.5)

        # Reduce in drawdown
        if state.current_drawdown_pct > 0.05:
            base_leverage *= 0.7
        elif state.current_drawdown_pct > 0.03:
            base_leverage *= 0.85

        # Reduce on loss streak
        if state.loss_streak >= 2:
            base_leverage *= 0.8

        return max(1.0, base_leverage)

    def _has_kelly_stats(self) -> bool:
        """Check if we have enough data for Kelly calculation."""
        total_trades = self._wins + self._losses
        return total_trades >= self.config.min_trades_for_kelly

    def _calculate_kelly_risk(self, equity: float) -> float:
        """Calculate Kelly criterion risk amount."""
        total = self._wins + self._losses
        if total < self.config.min_trades_for_kelly:
            return equity * self.config.base_risk_per_trade_pct

        win_rate = self._wins / total
        avg_win = self._total_win_pct / max(self._wins, 1)
        avg_loss = self._total_loss_pct / max(self._losses, 1)

        if avg_loss <= 0:
            return equity * self.config.base_risk_per_trade_pct

        # Kelly formula: f* = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        b = avg_win / avg_loss
        f_star = (b * win_rate - (1 - win_rate)) / b

        # Apply fraction and limits
        kelly_pct = f_star * self.config.kelly_fraction
        kelly_pct = max(0, min(self.config.max_risk_per_trade_pct, kelly_pct))

        return equity * kelly_pct

    def _calculate_var_cvar(
        self,
        position_value: float,
        volatility: float,
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR (Expected Shortfall)."""
        # Parametric VaR using normal distribution
        # Z-score for 95% confidence = 1.645
        z_score = 1.645
        daily_var = position_value * volatility * z_score

        # CVaR (Expected Shortfall) - average loss beyond VaR
        # For normal distribution: ES = μ + σ * φ(Φ^(-1)(α)) / (1-α)
        # Simplified: CVaR ≈ VaR * 1.2 for 95% confidence
        cvar = daily_var * 1.2

        return daily_var, cvar

    def _zero_budget(
        self,
        trade_id: str,
        symbol: str,
        regime: str,
        confidence: float,
    ) -> RiskBudget:
        """Return a zero budget (no trading allowed)."""
        return RiskBudget(
            trade_id=trade_id,
            symbol=symbol,
            max_position_usd=0,
            max_position_pct=0,
            recommended_position_usd=0,
            max_risk_usd=0,
            max_risk_pct=0,
            stop_distance_pct=0.02,
            max_leverage=1.0,
            recommended_leverage=1.0,
            tier=RiskBudgetTier.ZERO,
            adjustments={"reason": "conditions_not_met"},
            regime=regime,
            confidence=confidence,
        )

    def record_trade_result(
        self,
        was_win: bool,
        pnl_pct: float,
    ):
        """Record trade result for Kelly statistics."""
        with self._lock:
            if was_win:
                self._wins += 1
                self._total_win_pct += abs(pnl_pct)
            else:
                self._losses += 1
                self._total_loss_pct += abs(pnl_pct)

    def record_daily_return(self, return_pct: float):
        """Record daily return for VaR calculation."""
        with self._lock:
            self._daily_returns.append(return_pct)
            if len(self._daily_returns) > self.config.var_lookback_days:
                self._daily_returns = self._daily_returns[-self.config.var_lookback_days:]

    def get_stats(self) -> Dict[str, Any]:
        """Get risk budget engine statistics."""
        with self._lock:
            total = self._wins + self._losses
            return {
                "total_trades": total,
                "wins": self._wins,
                "losses": self._losses,
                "win_rate": self._wins / total if total > 0 else 0,
                "avg_win_pct": self._total_win_pct / max(self._wins, 1),
                "avg_loss_pct": self._total_loss_pct / max(self._losses, 1),
                "kelly_available": self._has_kelly_stats(),
                "var_data_points": len(self._daily_returns),
            }


# Singleton instance
_risk_budget_engine: Optional[RiskBudgetEngine] = None


def get_risk_budget_engine(
    config: Optional[RiskBudgetConfig] = None,
) -> RiskBudgetEngine:
    """Get or create the RiskBudgetEngine singleton."""
    global _risk_budget_engine
    if _risk_budget_engine is None:
        _risk_budget_engine = RiskBudgetEngine(config=config)
    return _risk_budget_engine


def reset_risk_budget_engine() -> None:
    """Reset the singleton instance (for testing)."""
    global _risk_budget_engine
    if _risk_budget_engine is not None:
        _risk_budget_engine._initialized = False
    _risk_budget_engine = None
    RiskBudgetEngine._instance = None
