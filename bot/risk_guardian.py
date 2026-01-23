"""
Risk Guardian Module - The Safety Layer

This module has VETO POWER over all trading decisions.
It enforces hard limits that cannot be overridden by any AI or strategy.

Key responsibilities:
1. Daily loss limit (hard stop - no more trading today)
2. Max drawdown limit (hard stop - system pauses)
3. Margin usage limits (prevent liquidation)
4. Liquidation distance protection
5. Circuit breakers (consecutive failures, data issues)
6. Kill switch (manual emergency stop)
7. Position exposure limits

RULE: The Risk Guardian can say NO. Always.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class VetoReason(Enum):
    """Reasons for vetoing a trade."""

    NONE = "none"
    DAILY_LOSS_LIMIT = "daily_loss_limit_reached"
    DRAWDOWN_LIMIT = "max_drawdown_reached"
    MARGIN_LIMIT = "margin_usage_too_high"
    LIQUIDATION_RISK = "too_close_to_liquidation"
    POSITION_SIZE_LIMIT = "position_size_exceeds_limit"
    EXPOSURE_LIMIT = "total_exposure_exceeds_limit"
    CONSECUTIVE_LOSSES = "too_many_consecutive_losses"
    DATA_STALE = "market_data_is_stale"
    SPREAD_TOO_WIDE = "spread_exceeds_threshold"
    VOLATILITY_SPIKE = "extreme_volatility_detected"
    CIRCUIT_BREAKER = "circuit_breaker_triggered"
    KILL_SWITCH = "manual_kill_switch_active"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    COOLDOWN_ACTIVE = "trading_cooldown_active"


# Alias for backward compatibility
RiskViolationType = VetoReason


@dataclass
class RiskLimits:
    """Hard risk limits that cannot be exceeded."""

    # Additional attributes for test compatibility
    max_position_pct: float = 25.0  # Alias for max_position_size_pct
    max_correlation: float = 0.8  # Max correlation between positions
    max_trades_per_day: int = 50  # Max trades per day

    # Daily limits
    max_daily_loss_pct: float = 2.0  # Max 2% daily loss
    max_daily_trades: int = 50  # Max trades per day

    # Drawdown limits
    max_drawdown_pct: float = 10.0  # Max 10% drawdown from peak
    drawdown_warning_pct: float = 5.0  # Warning at 5%

    # Position limits
    max_position_size_pct: float = 25.0  # Max 25% per position
    max_total_exposure_pct: float = 100.0  # Max 100% total exposure
    max_single_asset_pct: float = 40.0  # Max 40% in single asset
    max_correlated_exposure_pct: float = 60.0  # Max 60% in correlated assets

    # Leverage limits
    max_leverage: float = 10.0  # Max 10x leverage
    margin_warning_pct: float = 60.0  # Warning at 60% margin used
    margin_critical_pct: float = 80.0  # Critical at 80% margin used
    liquidation_buffer_pct: float = 15.0  # Stay 15% away from liquidation

    # Circuit breaker limits
    max_consecutive_losses: int = 5  # Pause after 5 consecutive losses
    max_consecutive_failures: int = 3  # Pause after 3 execution failures

    # Data quality limits
    max_data_age_seconds: int = 60  # Data older than 60s is stale
    max_spread_pct: float = 1.0  # Max 1% spread
    volatility_spike_threshold: float = 3.0  # 3x normal volatility

    # Cooldown
    loss_cooldown_minutes: int = 30  # Cooldown after hitting loss limit
    failure_cooldown_minutes: int = 5  # Cooldown after failures


@dataclass
class RiskState:
    """Current risk state tracking."""

    # Daily tracking
    daily_pnl_pct: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0
    day_start_equity: float = 0.0

    # Drawdown tracking
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown_pct: float = 0.0

    # Position tracking
    total_exposure_pct: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> exposure %

    # Leverage tracking
    current_leverage: float = 1.0
    margin_used_pct: float = 0.0
    liquidation_distance_pct: float = 100.0

    # Failure tracking
    consecutive_losses: int = 0
    consecutive_failures: int = 0
    last_loss_time: Optional[str] = None
    last_failure_time: Optional[str] = None

    # Status
    kill_switch_active: bool = False
    cooldown_until: Optional[str] = None
    risk_level: str = "normal"
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "day_start_equity": self.day_start_equity,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "total_exposure_pct": round(self.total_exposure_pct, 4),
            "positions": self.positions,
            "current_leverage": round(self.current_leverage, 2),
            "margin_used_pct": round(self.margin_used_pct, 4),
            "liquidation_distance_pct": round(self.liquidation_distance_pct, 4),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_failures": self.consecutive_failures,
            "kill_switch_active": self.kill_switch_active,
            "cooldown_until": self.cooldown_until,
            "risk_level": self.risk_level,
            "last_updated": self.last_updated,
        }


# Alias for backward compatibility
RiskMetrics = RiskState


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    approved: bool
    veto_reasons: List[VetoReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.NORMAL
    adjusted_size_pct: Optional[float] = None  # Suggested reduced size
    adjusted_leverage: Optional[float] = None  # Suggested reduced leverage
    max_allowed_size_pct: float = 0.0
    max_allowed_leverage: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "veto_reasons": [r.value for r in self.veto_reasons],
            "warnings": self.warnings,
            "risk_level": self.risk_level.value,
            "adjusted_size_pct": self.adjusted_size_pct,
            "adjusted_leverage": self.adjusted_leverage,
            "max_allowed_size_pct": round(self.max_allowed_size_pct, 4),
            "max_allowed_leverage": round(self.max_allowed_leverage, 2),
            "reasoning": self.reasoning,
        }


@dataclass
class TradeRequest:
    """A trade request to be checked by Risk Guardian."""

    symbol: str
    action: str  # BUY, SELL, CLOSE
    direction: str  # LONG, SHORT
    size_pct: float  # % of portfolio
    leverage: float = 1.0
    entry_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Market context
    current_price: float = 0.0
    spread_pct: float = 0.0
    volatility_ratio: float = 1.0
    data_age_seconds: int = 0

    # Signal sources
    signal_source: str = "unknown"
    signal_confidence: float = 0.0


class RiskGuardian:
    """
    The Risk Guardian - Enforces hard limits and can veto any trade.

    This is the safety layer that protects the system from catastrophic losses.
    It has absolute authority to reject trades that violate risk rules.
    """

    def __init__(
        self,
        limits: RiskLimits = None,
        state_file: Path = None,
        auto_save: bool = True,
        initial_capital: float = 10000.0,
    ):
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self.state_file = state_file or Path("data/risk_guardian_state.json")
        self.auto_save = auto_save
        self._lock = threading.Lock()
        self.initial_capital = initial_capital

        # Initialize equity tracking with initial capital
        if self.state.current_equity == 0:
            self.state.current_equity = initial_capital
            self.state.peak_equity = initial_capital
            self.state.day_start_equity = initial_capital

        # Load existing state
        self._load_state()

        logger.info(
            "Risk Guardian initialized with limits: "
            f"max_daily_loss={self.limits.max_daily_loss_pct}%, "
            f"max_drawdown={self.limits.max_drawdown_pct}%, "
            f"max_leverage={self.limits.max_leverage}x"
        )

    def check_trade(self, request: TradeRequest) -> RiskCheckResult:
        """
        Check if a trade is allowed under current risk rules.

        This is the main entry point. It runs all risk checks and returns
        whether the trade is approved or vetoed.
        """
        with self._lock:
            result = RiskCheckResult(approved=True)
            result.max_allowed_size_pct = self.limits.max_position_size_pct
            result.max_allowed_leverage = self.limits.max_leverage

            # Run all checks in order of severity
            checks = [
                self._check_kill_switch,
                self._check_cooldown,
                self._check_daily_loss_limit,
                self._check_drawdown_limit,
                self._check_consecutive_losses,
                self._check_data_quality,
                self._check_spread,
                self._check_volatility,
                self._check_position_size,
                self._check_exposure_limits,
                self._check_leverage,
                self._check_margin,
                self._check_liquidation_distance,
            ]

            for check in checks:
                check(request, result)

            # Determine overall risk level
            result.risk_level = self._calculate_risk_level(result)

            # Generate reasoning
            result.reasoning = self._generate_reasoning(request, result)

            # Log the decision
            self._log_decision(request, result)

            return result

    def _check_kill_switch(self, request: TradeRequest, result: RiskCheckResult):
        """Check if kill switch is active."""
        if self.state.kill_switch_active:
            result.approved = False
            result.veto_reasons.append(VetoReason.KILL_SWITCH)

    def _check_cooldown(self, request: TradeRequest, result: RiskCheckResult):
        """Check if in cooldown period."""
        if self.state.cooldown_until:
            cooldown_time = datetime.fromisoformat(self.state.cooldown_until)
            if datetime.now(timezone.utc) < cooldown_time:
                result.approved = False
                result.veto_reasons.append(VetoReason.COOLDOWN_ACTIVE)
                remaining = (cooldown_time - datetime.now(timezone.utc)).total_seconds() / 60
                result.warnings.append(f"Cooldown active for {remaining:.1f} more minutes")

    def _check_daily_loss_limit(self, request: TradeRequest, result: RiskCheckResult):
        """Check daily loss limit."""
        if self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.DAILY_LOSS_LIMIT)
        elif self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct * 0.8:
            result.warnings.append(
                f"Near daily loss limit: {self.state.daily_pnl_pct:.2f}% "
                f"(limit: {-self.limits.max_daily_loss_pct}%)"
            )

    def _check_drawdown_limit(self, request: TradeRequest, result: RiskCheckResult):
        """Check max drawdown limit."""
        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.DRAWDOWN_LIMIT)
        elif self.state.current_drawdown_pct >= self.limits.drawdown_warning_pct:
            result.warnings.append(
                f"Drawdown warning: {self.state.current_drawdown_pct:.2f}% "
                f"(limit: {self.limits.max_drawdown_pct}%)"
            )

    def _check_consecutive_losses(self, request: TradeRequest, result: RiskCheckResult):
        """Check consecutive losses circuit breaker."""
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            result.approved = False
            result.veto_reasons.append(VetoReason.CONSECUTIVE_LOSSES)
        elif self.state.consecutive_losses >= self.limits.max_consecutive_losses - 2:
            result.warnings.append(
                f"Consecutive losses: {self.state.consecutive_losses} "
                f"(limit: {self.limits.max_consecutive_losses})"
            )

    def _check_data_quality(self, request: TradeRequest, result: RiskCheckResult):
        """Check data freshness."""
        if request.data_age_seconds > self.limits.max_data_age_seconds:
            result.approved = False
            result.veto_reasons.append(VetoReason.DATA_STALE)
            result.warnings.append(
                f"Data is {request.data_age_seconds}s old "
                f"(limit: {self.limits.max_data_age_seconds}s)"
            )

    def _check_spread(self, request: TradeRequest, result: RiskCheckResult):
        """Check spread limits."""
        if request.spread_pct > self.limits.max_spread_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.SPREAD_TOO_WIDE)
            result.warnings.append(
                f"Spread too wide: {request.spread_pct:.2f}% (limit: {self.limits.max_spread_pct}%)"
            )

    def _check_volatility(self, request: TradeRequest, result: RiskCheckResult):
        """Check for volatility spikes."""
        if request.volatility_ratio > self.limits.volatility_spike_threshold:
            result.approved = False
            result.veto_reasons.append(VetoReason.VOLATILITY_SPIKE)
            result.warnings.append(
                f"Extreme volatility: {request.volatility_ratio:.1f}x normal "
                f"(limit: {self.limits.volatility_spike_threshold}x)"
            )
        elif request.volatility_ratio > self.limits.volatility_spike_threshold * 0.7:
            # Reduce allowed size in high volatility
            vol_factor = 1.0 / request.volatility_ratio
            result.adjusted_size_pct = request.size_pct * vol_factor
            result.warnings.append(
                f"High volatility ({request.volatility_ratio:.1f}x): "
                f"reducing size to {result.adjusted_size_pct:.1f}%"
            )

    def _check_position_size(self, request: TradeRequest, result: RiskCheckResult):
        """Check position size limits."""
        effective_size = request.size_pct * request.leverage

        if effective_size > self.limits.max_position_size_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.POSITION_SIZE_LIMIT)
            result.max_allowed_size_pct = self.limits.max_position_size_pct / request.leverage

    def _check_exposure_limits(self, request: TradeRequest, result: RiskCheckResult):
        """Check total exposure limits."""
        new_exposure = self.state.total_exposure_pct + (request.size_pct * request.leverage)

        if new_exposure > self.limits.max_total_exposure_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.EXPOSURE_LIMIT)
            available = self.limits.max_total_exposure_pct - self.state.total_exposure_pct
            result.max_allowed_size_pct = max(0, available / request.leverage)

        # Check single asset limit
        current_asset_exposure = self.state.positions.get(request.symbol, 0)
        new_asset_exposure = current_asset_exposure + (request.size_pct * request.leverage)

        if new_asset_exposure > self.limits.max_single_asset_pct:
            result.warnings.append(
                f"Single asset exposure high: {new_asset_exposure:.1f}% "
                f"(limit: {self.limits.max_single_asset_pct}%)"
            )

    def _check_leverage(self, request: TradeRequest, result: RiskCheckResult):
        """Check leverage limits."""
        if request.leverage > self.limits.max_leverage:
            result.approved = False
            result.veto_reasons.append(VetoReason.EXPOSURE_LIMIT)
            result.adjusted_leverage = self.limits.max_leverage
            result.max_allowed_leverage = self.limits.max_leverage

    def _check_margin(self, request: TradeRequest, result: RiskCheckResult):
        """Check margin usage limits."""
        if self.state.margin_used_pct >= self.limits.margin_critical_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.MARGIN_LIMIT)
        elif self.state.margin_used_pct >= self.limits.margin_warning_pct:
            result.warnings.append(
                f"Margin usage high: {self.state.margin_used_pct:.1f}% "
                f"(critical: {self.limits.margin_critical_pct}%)"
            )
            # Reduce allowed leverage
            margin_headroom = (self.limits.margin_critical_pct - self.state.margin_used_pct) / 100
            result.max_allowed_leverage = min(result.max_allowed_leverage, 1 + margin_headroom * 10)

    def _check_liquidation_distance(self, request: TradeRequest, result: RiskCheckResult):
        """Check distance to liquidation."""
        if self.state.liquidation_distance_pct < self.limits.liquidation_buffer_pct:
            result.approved = False
            result.veto_reasons.append(VetoReason.LIQUIDATION_RISK)
            result.warnings.append(
                f"Too close to liquidation: {self.state.liquidation_distance_pct:.1f}% "
                f"(buffer: {self.limits.liquidation_buffer_pct}%)"
            )
        elif self.state.liquidation_distance_pct < self.limits.liquidation_buffer_pct * 2:
            result.warnings.append(
                f"Liquidation distance low: {self.state.liquidation_distance_pct:.1f}%"
            )

    def _calculate_risk_level(self, result: RiskCheckResult) -> RiskLevel:
        """Calculate overall risk level."""
        if result.veto_reasons:
            if VetoReason.KILL_SWITCH in result.veto_reasons:
                return RiskLevel.EMERGENCY
            if VetoReason.LIQUIDATION_RISK in result.veto_reasons:
                return RiskLevel.EMERGENCY
            if VetoReason.DAILY_LOSS_LIMIT in result.veto_reasons:
                return RiskLevel.CRITICAL
            if VetoReason.DRAWDOWN_LIMIT in result.veto_reasons:
                return RiskLevel.CRITICAL
            return RiskLevel.HIGH

        if len(result.warnings) >= 3:
            return RiskLevel.HIGH
        if len(result.warnings) >= 1:
            return RiskLevel.ELEVATED

        return RiskLevel.NORMAL

    def _generate_reasoning(self, request: TradeRequest, result: RiskCheckResult) -> str:
        """Generate human-readable reasoning."""
        if not result.approved:
            reasons = ", ".join(r.value for r in result.veto_reasons)
            return f"TRADE VETOED: {reasons}"

        if result.warnings:
            return (
                f"APPROVED with {len(result.warnings)} warnings: {'; '.join(result.warnings[:3])}"
            )

        return "APPROVED: All risk checks passed"

    def _log_decision(self, request: TradeRequest, result: RiskCheckResult):
        """Log the risk decision."""
        log_msg = (
            f"Risk check for {request.action} {request.symbol}: "
            f"{'APPROVED' if result.approved else 'VETOED'} "
            f"(risk_level={result.risk_level.value})"
        )

        if result.approved:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)

        if result.veto_reasons:
            logger.warning(f"Veto reasons: {[r.value for r in result.veto_reasons]}")
        if result.warnings:
            logger.info(f"Warnings: {result.warnings}")

    # === State Update Methods ===

    def update_equity(self, current_equity: float):
        """Update current equity and recalculate drawdown."""
        with self._lock:
            self.state.current_equity = current_equity

            # Update peak if new high
            if current_equity > self.state.peak_equity:
                self.state.peak_equity = current_equity

            # Calculate drawdown
            if self.state.peak_equity > 0:
                self.state.current_drawdown_pct = (
                    (self.state.peak_equity - current_equity) / self.state.peak_equity * 100
                )

            # Update daily PnL
            if self.state.day_start_equity > 0:
                self.state.daily_pnl_pct = (
                    (current_equity - self.state.day_start_equity)
                    / self.state.day_start_equity
                    * 100
                )

            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            self._update_risk_level()

            if self.auto_save:
                self._save_state()

    def update_position(self, symbol: str, exposure_pct: float, leverage: float = 1.0):
        """Update position tracking."""
        with self._lock:
            if exposure_pct > 0:
                self.state.positions[symbol] = exposure_pct * leverage
            elif symbol in self.state.positions:
                del self.state.positions[symbol]

            self.state.total_exposure_pct = sum(self.state.positions.values())
            self.state.current_leverage = leverage
            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            if self.auto_save:
                self._save_state()

    def update_margin(self, margin_used_pct: float, liquidation_distance_pct: float):
        """Update margin and liquidation status."""
        with self._lock:
            self.state.margin_used_pct = margin_used_pct
            self.state.liquidation_distance_pct = liquidation_distance_pct
            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            self._update_risk_level()

            if self.auto_save:
                self._save_state()

    def record_trade_result(self, pnl_pct: float, is_win: bool):
        """Record a trade result."""
        with self._lock:
            self.state.daily_trades += 1

            if is_win:
                self.state.daily_wins += 1
                self.state.consecutive_losses = 0
            else:
                self.state.daily_losses += 1
                self.state.consecutive_losses += 1
                self.state.last_loss_time = datetime.now(timezone.utc).isoformat()

                # Check if we need to trigger cooldown
                if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
                    self._trigger_cooldown(self.limits.loss_cooldown_minutes)

            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            if self.auto_save:
                self._save_state()

    def record_execution_failure(self):
        """Record an execution failure."""
        with self._lock:
            self.state.consecutive_failures += 1
            self.state.last_failure_time = datetime.now(timezone.utc).isoformat()

            if self.state.consecutive_failures >= self.limits.max_consecutive_failures:
                self._trigger_cooldown(self.limits.failure_cooldown_minutes)

            if self.auto_save:
                self._save_state()

    def record_execution_success(self):
        """Record successful execution (resets failure counter)."""
        with self._lock:
            self.state.consecutive_failures = 0

            if self.auto_save:
                self._save_state()

    def reset_daily_stats(self, current_equity: float):
        """Reset daily statistics (call at start of trading day)."""
        with self._lock:
            self.state.daily_pnl_pct = 0.0
            self.state.daily_trades = 0
            self.state.daily_wins = 0
            self.state.daily_losses = 0
            self.state.day_start_equity = current_equity
            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            if self.auto_save:
                self._save_state()

            logger.info(f"Daily stats reset. Starting equity: ${current_equity:,.2f}")

    # === Kill Switch Methods ===

    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate the kill switch - stops all trading."""
        with self._lock:
            self.state.kill_switch_active = True
            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            if self.auto_save:
                self._save_state()

            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """Deactivate the kill switch."""
        with self._lock:
            self.state.kill_switch_active = False
            self.state.last_updated = datetime.now(timezone.utc).isoformat()

            if self.auto_save:
                self._save_state()

            logger.warning("Kill switch deactivated")

    def _trigger_cooldown(self, minutes: int):
        """Trigger a cooldown period."""
        cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.state.cooldown_until = cooldown_end.isoformat()
        logger.warning(f"Cooldown triggered for {minutes} minutes until {cooldown_end}")

    def _update_risk_level(self):
        """Update the overall risk level based on current state."""
        if self.state.kill_switch_active:
            self.state.risk_level = RiskLevel.EMERGENCY.value
        elif self.state.liquidation_distance_pct < self.limits.liquidation_buffer_pct:
            self.state.risk_level = RiskLevel.EMERGENCY.value
        elif self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
            self.state.risk_level = RiskLevel.CRITICAL.value
        elif self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            self.state.risk_level = RiskLevel.CRITICAL.value
        elif self.state.margin_used_pct >= self.limits.margin_critical_pct:
            self.state.risk_level = RiskLevel.HIGH.value
        elif self.state.consecutive_losses >= self.limits.max_consecutive_losses - 1:
            self.state.risk_level = RiskLevel.HIGH.value
        elif self.state.current_drawdown_pct >= self.limits.drawdown_warning_pct:
            self.state.risk_level = RiskLevel.ELEVATED.value
        elif self.state.margin_used_pct >= self.limits.margin_warning_pct:
            self.state.risk_level = RiskLevel.ELEVATED.value
        else:
            self.state.risk_level = RiskLevel.NORMAL.value

    # === Status Methods ===

    def get_status(self) -> Dict[str, Any]:
        """Get current risk guardian status."""
        with self._lock:
            return {
                "risk_level": self.state.risk_level,
                "kill_switch_active": self.state.kill_switch_active,
                "cooldown_active": self.state.cooldown_until is not None
                and datetime.now(timezone.utc) < datetime.fromisoformat(self.state.cooldown_until)
                if self.state.cooldown_until
                else False,
                "daily_stats": {
                    "pnl_pct": round(self.state.daily_pnl_pct, 4),
                    "trades": self.state.daily_trades,
                    "wins": self.state.daily_wins,
                    "losses": self.state.daily_losses,
                    "win_rate": round(self.state.daily_wins / max(1, self.state.daily_trades), 4),
                    "remaining_loss_budget_pct": round(
                        self.limits.max_daily_loss_pct + self.state.daily_pnl_pct, 4
                    ),
                },
                "drawdown": {
                    "current_pct": round(self.state.current_drawdown_pct, 4),
                    "max_allowed_pct": self.limits.max_drawdown_pct,
                    "remaining_pct": round(
                        self.limits.max_drawdown_pct - self.state.current_drawdown_pct, 4
                    ),
                },
                "exposure": {
                    "total_pct": round(self.state.total_exposure_pct, 4),
                    "max_allowed_pct": self.limits.max_total_exposure_pct,
                    "positions": self.state.positions,
                },
                "leverage": {
                    "current": round(self.state.current_leverage, 2),
                    "max_allowed": self.limits.max_leverage,
                },
                "margin": {
                    "used_pct": round(self.state.margin_used_pct, 4),
                    "warning_pct": self.limits.margin_warning_pct,
                    "critical_pct": self.limits.margin_critical_pct,
                    "liquidation_distance_pct": round(self.state.liquidation_distance_pct, 4),
                    "liquidation_buffer_pct": self.limits.liquidation_buffer_pct,
                },
                "circuit_breakers": {
                    "consecutive_losses": self.state.consecutive_losses,
                    "max_consecutive_losses": self.limits.max_consecutive_losses,
                    "consecutive_failures": self.state.consecutive_failures,
                    "max_consecutive_failures": self.limits.max_consecutive_failures,
                },
                "limits": {
                    "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                    "max_drawdown_pct": self.limits.max_drawdown_pct,
                    "max_position_size_pct": self.limits.max_position_size_pct,
                    "max_leverage": self.limits.max_leverage,
                },
                "last_updated": self.state.last_updated,
            }

    def get_trading_allowed(self) -> Tuple[bool, str]:
        """Quick check if trading is allowed at all."""
        with self._lock:
            if self.state.kill_switch_active:
                return False, "Kill switch is active"

            if self.state.cooldown_until:
                cooldown_time = datetime.fromisoformat(self.state.cooldown_until)
                if datetime.now(timezone.utc) < cooldown_time:
                    remaining = (cooldown_time - datetime.now(timezone.utc)).total_seconds() / 60
                    return False, f"Cooldown active for {remaining:.1f} more minutes"

            if self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
                return False, "Daily loss limit reached"

            if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
                return False, "Max drawdown limit reached"

            if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
                return False, f"Circuit breaker: {self.state.consecutive_losses} consecutive losses"

            return True, "Trading allowed"

    # === Persistence Methods ===

    def _save_state(self):
        """Save state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def _load_state(self):
        """Load state from file."""
        if not self.state_file.exists():
            logger.info("No existing risk state found, starting fresh")
            return

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            self.state.daily_pnl_pct = data.get("daily_pnl_pct", 0)
            self.state.daily_trades = data.get("daily_trades", 0)
            self.state.daily_wins = data.get("daily_wins", 0)
            self.state.daily_losses = data.get("daily_losses", 0)
            self.state.day_start_equity = data.get("day_start_equity", 0)
            self.state.peak_equity = data.get("peak_equity", 0)
            self.state.current_equity = data.get("current_equity", 0)
            self.state.current_drawdown_pct = data.get("current_drawdown_pct", 0)
            self.state.total_exposure_pct = data.get("total_exposure_pct", 0)
            self.state.positions = data.get("positions", {})
            self.state.current_leverage = data.get("current_leverage", 1)
            self.state.margin_used_pct = data.get("margin_used_pct", 0)
            self.state.liquidation_distance_pct = data.get("liquidation_distance_pct", 100)
            self.state.consecutive_losses = data.get("consecutive_losses", 0)
            self.state.consecutive_failures = data.get("consecutive_failures", 0)
            self.state.kill_switch_active = data.get("kill_switch_active", False)
            self.state.cooldown_until = data.get("cooldown_until")
            self.state.risk_level = data.get("risk_level", "normal")
            self.state.last_updated = data.get("last_updated", "")

            logger.info(
                f"Loaded risk state: risk_level={self.state.risk_level}, "
                f"daily_pnl={self.state.daily_pnl_pct:.2f}%"
            )
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")


# Global instance
_guardian: Optional[RiskGuardian] = None


def get_risk_guardian() -> RiskGuardian:
    """Get or create global Risk Guardian."""
    global _guardian
    if _guardian is None:
        _guardian = RiskGuardian()
    return _guardian


def init_risk_guardian(
    limits: RiskLimits = None,
    state_file: Path = None,
) -> RiskGuardian:
    """Initialize Risk Guardian with custom settings."""
    global _guardian
    _guardian = RiskGuardian(limits=limits, state_file=state_file)
    return _guardian
