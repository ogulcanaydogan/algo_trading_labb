"""
BTC Turnover Reduction Strategy.

Phase 2B Deliverable:
Addresses BTC edge collapse under realistic friction by reducing turnover
WITHOUT weakening realism.

Strategies implemented:
1. Reduced decision frequency (longer minimum intervals)
2. Minimum expected value threshold (net-of-cost)
3. BTC-specific cooldown rules
4. Prefer maker/limit execution simulation

CRITICAL:
- All changes are reversible
- Must not regress other assets
- Realism costs remain unchanged
- Test coverage required
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# BTC-SPECIFIC TURNOVER REDUCTION CONFIGURATION
# =============================================================================

@dataclass
class BTCTurnoverConfig:
    """
    Configuration for BTC-specific turnover reduction.

    These parameters are MORE RESTRICTIVE than default to reduce
    trading frequency and associated costs.
    """

    # Decision frequency limits
    min_decision_interval_hours: float = 4.0  # Default: 1.0
    max_decisions_per_day: int = 3  # Default: unlimited

    # Minimum expected value threshold (net-of-cost)
    # Trade only if expected profit > expected_cost * multiplier
    min_expected_value_multiplier: float = 2.0  # Need 2x expected costs
    min_expected_profit_usd: float = 50.0  # Minimum $50 expected profit

    # Cost assumptions for BTC
    expected_slippage_bps: float = 10.0  # 10 bps slippage
    expected_fee_bps: float = 10.0  # 10 bps fee
    expected_spread_bps: float = 5.0  # 5 bps spread

    # Cooldown rules
    cooldown_after_loss_hours: float = 6.0  # Default: 2.0
    cooldown_after_consecutive_losses: int = 2  # After 2 losses

    # Execution preference
    prefer_limit_orders: bool = True
    limit_order_timeout_seconds: int = 300  # 5 minutes

    # Position sizing adjustment
    max_position_pct_of_daily_volume: float = 0.001  # 0.1% of daily volume

    # Confidence thresholds (higher than default)
    min_confidence_for_entry: float = 0.75  # Default: 0.6
    min_confidence_for_high_leverage: float = 0.85

    @property
    def total_expected_cost_bps(self) -> float:
        """Total expected cost per trade in basis points."""
        return (
            self.expected_slippage_bps +
            self.expected_fee_bps +
            self.expected_spread_bps
        )


# Default configuration - can be overridden
DEFAULT_BTC_CONFIG = BTCTurnoverConfig()


@dataclass
class DecisionGate:
    """Result of turnover reduction gate check."""
    allowed: bool
    reason: str
    time_until_allowed_minutes: float = 0.0
    expected_cost_usd: float = 0.0
    required_profit_usd: float = 0.0


class BTCTurnoverReducer:
    """
    Applies turnover reduction rules specifically for BTC.

    This is a FILTER that sits before trade execution and can
    reject trades that don't meet the stricter BTC criteria.

    CRITICAL: This does NOT have execution authority.
    It only provides gating decisions to the existing trading system.
    """

    def __init__(self, config: Optional[BTCTurnoverConfig] = None):
        self.config = config or DEFAULT_BTC_CONFIG

        # State tracking
        self._last_decision_time: Optional[datetime] = None
        self._decisions_today: int = 0
        self._last_decision_date: Optional[str] = None
        self._consecutive_losses: int = 0
        self._in_cooldown_until: Optional[datetime] = None

        logger.info(
            f"BTCTurnoverReducer initialized: "
            f"min_interval={self.config.min_decision_interval_hours}h, "
            f"max_daily={self.config.max_decisions_per_day}, "
            f"min_ev_mult={self.config.min_expected_value_multiplier}x"
        )

    def _reset_daily_counters(self):
        """Reset counters for new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_decision_date != today:
            self._decisions_today = 0
            self._last_decision_date = today

    def check_decision_allowed(
        self,
        symbol: str,
        position_value_usd: float,
        expected_profit_pct: float,
        confidence: float,
    ) -> DecisionGate:
        """
        Check if a BTC decision is allowed under turnover reduction rules.

        Args:
            symbol: Trading symbol (should contain "BTC")
            position_value_usd: Proposed position value in USD
            expected_profit_pct: Expected profit as percentage (e.g., 1.0 for 1%)
            confidence: Signal confidence (0-1)

        Returns:
            DecisionGate with allowed status and reason
        """
        if "BTC" not in symbol.upper():
            # Not BTC - allow without restriction
            return DecisionGate(allowed=True, reason="Not BTC - no restriction")

        self._reset_daily_counters()
        now = datetime.now()

        # Check 1: Cooldown period
        if self._in_cooldown_until and now < self._in_cooldown_until:
            remaining = (self._in_cooldown_until - now).total_seconds() / 60
            return DecisionGate(
                allowed=False,
                reason=f"In cooldown after consecutive losses",
                time_until_allowed_minutes=remaining,
            )

        # Check 2: Minimum decision interval
        if self._last_decision_time:
            elapsed = (now - self._last_decision_time).total_seconds() / 3600
            if elapsed < self.config.min_decision_interval_hours:
                remaining = (self.config.min_decision_interval_hours - elapsed) * 60
                return DecisionGate(
                    allowed=False,
                    reason=f"Minimum interval not met ({elapsed:.1f}h < {self.config.min_decision_interval_hours}h)",
                    time_until_allowed_minutes=remaining,
                )

        # Check 3: Max decisions per day
        if self._decisions_today >= self.config.max_decisions_per_day:
            return DecisionGate(
                allowed=False,
                reason=f"Max daily decisions reached ({self._decisions_today}/{self.config.max_decisions_per_day})",
            )

        # Check 4: Confidence threshold
        if confidence < self.config.min_confidence_for_entry:
            return DecisionGate(
                allowed=False,
                reason=f"Confidence too low ({confidence:.1%} < {self.config.min_confidence_for_entry:.1%})",
            )

        # Check 5: Expected value threshold
        expected_cost_bps = self.config.total_expected_cost_bps
        expected_cost_usd = position_value_usd * (expected_cost_bps / 10000)
        expected_profit_usd = position_value_usd * (expected_profit_pct / 100)
        required_profit_usd = max(
            expected_cost_usd * self.config.min_expected_value_multiplier,
            self.config.min_expected_profit_usd,
        )

        if expected_profit_usd < required_profit_usd:
            return DecisionGate(
                allowed=False,
                reason=(
                    f"Expected profit (${expected_profit_usd:.2f}) below required "
                    f"(${required_profit_usd:.2f} = {self.config.min_expected_value_multiplier}x costs)"
                ),
                expected_cost_usd=expected_cost_usd,
                required_profit_usd=required_profit_usd,
            )

        # All checks passed
        return DecisionGate(
            allowed=True,
            reason="All turnover reduction checks passed",
            expected_cost_usd=expected_cost_usd,
            required_profit_usd=required_profit_usd,
        )

    def record_decision(self, symbol: str):
        """Record that a BTC decision was made."""
        if "BTC" not in symbol.upper():
            return

        self._reset_daily_counters()
        self._last_decision_time = datetime.now()
        self._decisions_today += 1

        logger.debug(
            f"BTC decision recorded: {self._decisions_today}/{self.config.max_decisions_per_day} today"
        )

    def record_outcome(self, symbol: str, pnl: float):
        """Record trade outcome for cooldown tracking."""
        if "BTC" not in symbol.upper():
            return

        if pnl < 0:
            self._consecutive_losses += 1

            if self._consecutive_losses >= self.config.cooldown_after_consecutive_losses:
                cooldown_hours = self.config.cooldown_after_loss_hours
                self._in_cooldown_until = datetime.now() + timedelta(hours=cooldown_hours)
                logger.info(
                    f"BTC cooldown triggered: {self._consecutive_losses} consecutive losses, "
                    f"cooldown until {self._in_cooldown_until}"
                )
        else:
            self._consecutive_losses = 0
            self._in_cooldown_until = None

    def get_status(self) -> Dict:
        """Get current turnover reducer status."""
        now = datetime.now()
        return {
            "decisions_today": self._decisions_today,
            "max_daily": self.config.max_decisions_per_day,
            "last_decision": self._last_decision_time.isoformat() if self._last_decision_time else None,
            "consecutive_losses": self._consecutive_losses,
            "in_cooldown": self._in_cooldown_until is not None and now < self._in_cooldown_until,
            "cooldown_until": self._in_cooldown_until.isoformat() if self._in_cooldown_until else None,
            "config": {
                "min_interval_hours": self.config.min_decision_interval_hours,
                "min_ev_multiplier": self.config.min_expected_value_multiplier,
                "min_confidence": self.config.min_confidence_for_entry,
                "total_expected_cost_bps": self.config.total_expected_cost_bps,
            },
        }

    def reset(self):
        """Reset state (for testing)."""
        self._last_decision_time = None
        self._decisions_today = 0
        self._last_decision_date = None
        self._consecutive_losses = 0
        self._in_cooldown_until = None


# =============================================================================
# Singleton instance
# =============================================================================

_reducer: Optional[BTCTurnoverReducer] = None


def get_btc_turnover_reducer(
    config: Optional[BTCTurnoverConfig] = None
) -> BTCTurnoverReducer:
    """Get or create singleton BTCTurnoverReducer."""
    global _reducer
    if _reducer is None:
        _reducer = BTCTurnoverReducer(config)
    return _reducer


def reset_btc_turnover_reducer():
    """Reset singleton reducer."""
    global _reducer
    if _reducer is not None:
        _reducer.reset()
    _reducer = None


# =============================================================================
# Integration helper
# =============================================================================

def should_allow_btc_trade(
    symbol: str,
    position_value_usd: float,
    expected_profit_pct: float,
    confidence: float,
) -> Tuple[bool, str]:
    """
    Convenience function to check if BTC trade should be allowed.

    Returns:
        (allowed, reason)
    """
    reducer = get_btc_turnover_reducer()
    gate = reducer.check_decision_allowed(
        symbol=symbol,
        position_value_usd=position_value_usd,
        expected_profit_pct=expected_profit_pct,
        confidence=confidence,
    )
    return gate.allowed, gate.reason
