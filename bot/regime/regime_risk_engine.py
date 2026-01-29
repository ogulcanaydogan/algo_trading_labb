"""
Regime-Aware Risk Engine.

Enforces hard risk constraints before any order execution.
Has ABSOLUTE VETO POWER - can block trades regardless of strategy/ML output.

Safety-first design principles:
1. Risk engine runs BEFORE every order
2. Hard limits cannot be overridden
3. All decisions are logged with reasons
4. Kill switch can halt all trading instantly
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from .regime_detector import MarketRegime, RegimeState

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Risk check decision."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    SIZE_REDUCED = "size_reduced"


class BlockReason(Enum):
    """Reasons for blocking a trade."""

    KILL_SWITCH_ACTIVE = "kill_switch_active"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    PORTFOLIO_HEAT_LIMIT = "portfolio_heat_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    TIME_BETWEEN_TRADES = "time_between_trades"
    VOLATILITY_CIRCUIT_BREAKER = "volatility_circuit_breaker"
    SPREAD_CIRCUIT_BREAKER = "spread_circuit_breaker"
    LIQUIDITY_FILTER = "liquidity_filter"
    REGIME_NOT_ALLOWED = "regime_not_allowed"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_STOP_LOSS = "invalid_stop_loss"
    MAX_TRADES_PER_DAY = "max_trades_per_day"
    NO_TRADE_ZONE = "no_trade_zone"


@dataclass
class RegimeLimits:
    """Risk limits specific to a market regime."""

    max_leverage: float = 3.0
    max_position_pct: float = 0.25  # % of equity per position
    risk_per_trade_pct: float = 0.02  # % of equity risked per trade
    allowed_directions: List[str] = field(default_factory=lambda: ["long", "short"])
    size_multiplier: float = 1.0  # Scale down position sizes
    require_confirmation: bool = False

    @classmethod
    def for_regime(cls, regime: MarketRegime) -> "RegimeLimits":
        """Get default limits for a specific regime."""
        defaults = {
            MarketRegime.BULL: cls(
                max_leverage=3.0,
                max_position_pct=0.25,
                risk_per_trade_pct=0.02,
                allowed_directions=["long", "short"],
                size_multiplier=1.0,
            ),
            MarketRegime.BEAR: cls(
                max_leverage=2.0,
                max_position_pct=0.15,
                risk_per_trade_pct=0.015,
                allowed_directions=["long", "short"],
                size_multiplier=0.75,
            ),
            MarketRegime.CRASH: cls(
                max_leverage=1.0,
                max_position_pct=0.05,
                risk_per_trade_pct=0.005,
                allowed_directions=["short"],  # Long blocked in crash
                size_multiplier=0.25,
                require_confirmation=True,
            ),
            MarketRegime.SIDEWAYS: cls(
                max_leverage=2.0,
                max_position_pct=0.10,
                risk_per_trade_pct=0.01,
                allowed_directions=["long", "short"],
                size_multiplier=0.5,
            ),
            MarketRegime.HIGH_VOL: cls(
                max_leverage=1.5,
                max_position_pct=0.10,
                risk_per_trade_pct=0.01,
                allowed_directions=["long", "short"],
                size_multiplier=0.5,
            ),
            MarketRegime.UNKNOWN: cls(
                max_leverage=2.0,
                max_position_pct=0.15,  # Increased from 0.10 for better capital utilization
                risk_per_trade_pct=0.015,  # Slightly increased for commodities
                allowed_directions=["long", "short"],  # Allow trading to accumulate paper trades
                size_multiplier=0.75,  # Increased from 0.5 for silver/commodities
            ),
        }
        return defaults.get(regime, defaults[MarketRegime.UNKNOWN])


@dataclass
class RiskConfig:
    """Risk engine configuration."""

    # Per-regime limits (can be customized)
    regime_limits: Dict[str, RegimeLimits] = field(default_factory=dict)

    # Hard limits (cannot be overridden by regime)
    max_daily_loss_pct: float = 0.03  # 3%
    max_drawdown_pct: float = 0.10  # 10%
    max_consecutive_losses: int = 5
    max_portfolio_heat: float = 0.06  # 6% total risk
    min_time_between_trades_sec: int = 60
    max_trades_per_day: int = 50

    # Circuit breakers
    volatility_spike_multiplier: float = 3.0
    max_spread_pct: float = 0.005  # 0.5%
    min_liquidity_volume: float = 1000

    # Kill switch
    auto_kill_drawdown: float = 0.15  # 15%
    auto_kill_daily_loss: float = 0.05  # 5%
    cooldown_after_kill_hours: int = 24

    # Position sizing
    default_atr_multiplier: float = 2.0
    min_stop_distance_pct: float = 0.005  # 0.5%
    max_stop_distance_pct: float = 0.05  # 5%

    # No-trade zones (hour ranges in UTC)
    no_trade_hours: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: Dict) -> "RiskConfig":
        """Create config from dictionary."""
        regime_limits = {}
        if "regime_limits" in config:
            for regime_name, limits in config["regime_limits"].items():
                regime_limits[regime_name] = RegimeLimits(**limits)

        return cls(
            regime_limits=regime_limits,
            **{k: v for k, v in config.items() if k != "regime_limits" and hasattr(cls, k)},
        )


@dataclass
class TradeRequest:
    """Trade request to be validated by risk engine."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    quantity: Optional[float] = None  # If None, will be calculated
    signal_confidence: float = 0.5
    signal_reason: str = ""

    @property
    def stop_distance_pct(self) -> float:
        """Calculate stop distance as percentage."""
        if self.entry_price <= 0:
            return 0
        return abs(self.entry_price - self.stop_loss) / self.entry_price

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if not self.take_profit or self.stop_distance_pct == 0:
            return 0
        profit_distance = abs(self.take_profit - self.entry_price)
        return profit_distance / (self.stop_distance_pct * self.entry_price)


@dataclass
class RiskCheckResult:
    """Result of risk check."""

    decision: RiskDecision
    approved_quantity: float = 0.0
    original_quantity: float = 0.0
    block_reasons: List[BlockReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applied_limits: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "approved_quantity": self.approved_quantity,
            "original_quantity": self.original_quantity,
            "block_reasons": [r.value for r in self.block_reasons],
            "warnings": self.warnings,
            "applied_limits": self.applied_limits,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def is_approved(self) -> bool:
        return self.decision in (RiskDecision.APPROVED, RiskDecision.SIZE_REDUCED)

    @property
    def is_blocked(self) -> bool:
        return self.decision == RiskDecision.BLOCKED


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""

    equity: float = 10000.0
    available_balance: float = 10000.0
    positions: Dict[str, Dict] = field(default_factory=dict)  # symbol -> position
    peak_equity: float = 10000.0
    daily_pnl: float = 0.0
    daily_loss: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None

    @property
    def current_drawdown_pct(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_loss_pct(self) -> float:
        """Calculate daily loss percentage."""
        if self.equity <= 0:
            return 0
        return self.daily_loss / self.equity

    @property
    def total_position_value(self) -> float:
        """Calculate total value of open positions."""
        return sum(p.get("value", 0) for p in self.positions.values())

    @property
    def portfolio_heat(self) -> float:
        """Calculate total risk exposure (sum of position risks)."""
        return sum(p.get("risk_pct", 0) for p in self.positions.values())


class RegimeRiskEngine:
    """
    Regime-aware risk engine with hard constraints.

    This engine has ABSOLUTE VETO POWER over all trades.
    Safety is the primary objective.
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        state_path: Optional[Path] = None,
    ):
        self.config = config or RiskConfig()
        self.state_path = state_path or Path("data/regime_risk_state.json")

        self._lock = RLock()
        self._kill_switch_active = False
        self._kill_switch_reason: Optional[str] = None
        self._kill_switch_until: Optional[datetime] = None

        self._portfolio = PortfolioState()
        self._current_regime: Optional[RegimeState] = None
        self._symbol_regimes: Dict[str, RegimeState] = {}  # Per-symbol regime tracking
        self._decision_log: List[Dict] = []

        # Volatility tracking for circuit breaker
        self._normal_volatility: float = 0.0
        self._current_volatility: float = 0.0

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                    self._kill_switch_active = data.get("kill_switch_active", False)
                    self._kill_switch_reason = data.get("kill_switch_reason")
                    if data.get("kill_switch_until"):
                        self._kill_switch_until = datetime.fromisoformat(data["kill_switch_until"])
                    logger.info("Loaded risk engine state from disk")
            except Exception as e:
                logger.warning(f"Failed to load risk state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(
                    {
                        "kill_switch_active": self._kill_switch_active,
                        "kill_switch_reason": self._kill_switch_reason,
                        "kill_switch_until": self._kill_switch_until.isoformat()
                        if self._kill_switch_until
                        else None,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        """Update portfolio state for risk calculations."""
        with self._lock:
            self._portfolio = portfolio

            # Update peak equity
            if portfolio.equity > self._portfolio.peak_equity:
                self._portfolio.peak_equity = portfolio.equity

            # Check for auto kill switch triggers
            self._check_auto_kill_triggers()

    def update_regime(self, regime_state: RegimeState) -> None:
        """Update current regime state."""
        with self._lock:
            # Track per-symbol to prevent false regime change logs
            symbol = regime_state.symbol or "default"
            old_symbol_state = self._symbol_regimes.get(symbol)
            old_regime = old_symbol_state.regime if old_symbol_state else None

            self._symbol_regimes[symbol] = regime_state
            self._current_regime = regime_state  # Global state (backwards compatibility)

            if old_regime != regime_state.regime:
                logger.info(
                    f"Risk engine regime update: {old_regime} -> {regime_state.regime.value}"
                )

    def update_volatility(self, current_vol: float, normal_vol: float) -> None:
        """Update volatility for circuit breaker."""
        with self._lock:
            self._current_volatility = current_vol
            self._normal_volatility = normal_vol

    def check_trade(
        self,
        request: TradeRequest,
        spread_pct: Optional[float] = None,
        volume_24h: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Check if a trade request passes all risk constraints.

        This is the main entry point. Returns approved quantity or blocks.
        """
        with self._lock:
            result = RiskCheckResult(
                decision=RiskDecision.APPROVED,
                original_quantity=request.quantity or 0,
            )

            # Get regime-specific limits
            regime = self._current_regime.regime if self._current_regime else MarketRegime.UNKNOWN
            limits = self._get_regime_limits(regime)
            result.applied_limits["regime"] = regime.value
            result.applied_limits["regime_limits"] = {
                "max_leverage": limits.max_leverage,
                "max_position_pct": limits.max_position_pct,
                "risk_per_trade_pct": limits.risk_per_trade_pct,
            }

            # === HARD CHECKS (block if fail) ===

            # 1. Kill switch check
            if self._kill_switch_active:
                self._check_kill_switch_expiry()
            if self._kill_switch_active:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.KILL_SWITCH_ACTIVE)
                self._log_decision(request, result, "Kill switch is active")
                return result

            # 2. No-trade zone check
            if self._is_no_trade_zone():
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.NO_TRADE_ZONE)
                self._log_decision(request, result, "In no-trade time zone")
                return result

            # 3. Daily loss limit
            if self._portfolio.daily_loss_pct >= self.config.max_daily_loss_pct:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.DAILY_LOSS_LIMIT)
                self._log_decision(
                    request,
                    result,
                    f"Daily loss {self._portfolio.daily_loss_pct:.2%} >= {self.config.max_daily_loss_pct:.2%}",
                )
                return result

            # 4. Drawdown limit
            if self._portfolio.current_drawdown_pct >= self.config.max_drawdown_pct:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.DRAWDOWN_LIMIT)
                self._log_decision(
                    request,
                    result,
                    f"Drawdown {self._portfolio.current_drawdown_pct:.2%} >= {self.config.max_drawdown_pct:.2%}",
                )
                return result

            # 5. Consecutive losses
            if self._portfolio.consecutive_losses >= self.config.max_consecutive_losses:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.CONSECUTIVE_LOSSES)
                self._log_decision(
                    request,
                    result,
                    f"Consecutive losses {self._portfolio.consecutive_losses} >= {self.config.max_consecutive_losses}",
                )
                return result

            # 6. Time between trades
            if self._portfolio.last_trade_time:
                time_since = (datetime.now() - self._portfolio.last_trade_time).total_seconds()
                if time_since < self.config.min_time_between_trades_sec:
                    result.decision = RiskDecision.BLOCKED
                    result.block_reasons.append(BlockReason.TIME_BETWEEN_TRADES)
                    self._log_decision(
                        request,
                        result,
                        f"Only {time_since:.0f}s since last trade (min: {self.config.min_time_between_trades_sec}s)",
                    )
                    return result

            # 7. Max trades per day
            if self._portfolio.daily_trades >= self.config.max_trades_per_day:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.MAX_TRADES_PER_DAY)
                self._log_decision(
                    request,
                    result,
                    f"Daily trades {self._portfolio.daily_trades} >= {self.config.max_trades_per_day}",
                )
                return result

            # 8. Volatility circuit breaker
            if self._normal_volatility > 0:
                vol_ratio = self._current_volatility / self._normal_volatility
                if vol_ratio >= self.config.volatility_spike_multiplier:
                    result.decision = RiskDecision.BLOCKED
                    result.block_reasons.append(BlockReason.VOLATILITY_CIRCUIT_BREAKER)
                    self._log_decision(
                        request,
                        result,
                        f"Volatility spike {vol_ratio:.1f}x >= {self.config.volatility_spike_multiplier}x",
                    )
                    return result

            # 9. Spread circuit breaker
            if spread_pct is not None and spread_pct > self.config.max_spread_pct:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.SPREAD_CIRCUIT_BREAKER)
                self._log_decision(
                    request, result, f"Spread {spread_pct:.4%} > {self.config.max_spread_pct:.4%}"
                )
                return result

            # 10. Liquidity filter
            if volume_24h is not None and volume_24h < self.config.min_liquidity_volume:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.LIQUIDITY_FILTER)
                self._log_decision(
                    request, result, f"Volume {volume_24h:.0f} < {self.config.min_liquidity_volume}"
                )
                return result

            # 11. Direction allowed for regime
            if request.direction not in limits.allowed_directions:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.REGIME_NOT_ALLOWED)
                self._log_decision(
                    request,
                    result,
                    f"Direction '{request.direction}' not allowed in {regime.value} regime",
                )
                return result

            # 12. Stop loss validation
            stop_dist = request.stop_distance_pct
            if stop_dist < self.config.min_stop_distance_pct:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.INVALID_STOP_LOSS)
                result.warnings.append(f"Stop too tight: {stop_dist:.4%}")
                self._log_decision(request, result, "Stop loss too tight")
                return result
            if stop_dist > self.config.max_stop_distance_pct:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.INVALID_STOP_LOSS)
                result.warnings.append(f"Stop too wide: {stop_dist:.4%}")
                self._log_decision(request, result, "Stop loss too wide")
                return result

            # === POSITION SIZING ===
            calculated_size = self._calculate_position_size(request, limits)

            # Apply regime size multiplier
            calculated_size *= limits.size_multiplier

            # Check if calculated size is valid
            if calculated_size <= 0:
                result.decision = RiskDecision.BLOCKED
                result.block_reasons.append(BlockReason.INSUFFICIENT_BALANCE)
                self._log_decision(request, result, "Calculated size is zero")
                return result

            # 13. Portfolio heat limit
            current_heat = self._portfolio.portfolio_heat
            trade_risk = limits.risk_per_trade_pct
            if current_heat + trade_risk > self.config.max_portfolio_heat:
                # Reduce size to fit within heat limit
                available_heat = max(0, self.config.max_portfolio_heat - current_heat)
                if available_heat < limits.risk_per_trade_pct * 0.5:
                    result.decision = RiskDecision.BLOCKED
                    result.block_reasons.append(BlockReason.PORTFOLIO_HEAT_LIMIT)
                    self._log_decision(
                        request,
                        result,
                        f"Portfolio heat {current_heat:.2%} + {trade_risk:.2%} > {self.config.max_portfolio_heat:.2%}",
                    )
                    return result
                else:
                    # Reduce size proportionally
                    reduction = available_heat / limits.risk_per_trade_pct
                    calculated_size *= reduction
                    result.decision = RiskDecision.SIZE_REDUCED
                    result.warnings.append(
                        f"Size reduced by {(1 - reduction):.0%} due to portfolio heat"
                    )

            # Set final approved quantity
            result.approved_quantity = calculated_size
            if request.quantity and calculated_size < request.quantity:
                result.decision = RiskDecision.SIZE_REDUCED
                result.warnings.append(
                    f"Size reduced from {request.quantity:.4f} to {calculated_size:.4f}"
                )

            self._log_decision(request, result, "Trade approved")
            return result

    def _calculate_position_size(
        self,
        request: TradeRequest,
        limits: RegimeLimits,
    ) -> float:
        """Calculate risk-based position size."""

        equity = self._portfolio.equity
        if equity <= 0:
            return 0

        # Risk amount in dollars
        risk_amount = equity * limits.risk_per_trade_pct

        # Position size based on stop distance
        stop_distance = request.stop_distance_pct
        if stop_distance <= 0:
            stop_distance = self.config.min_stop_distance_pct

        # Base size from risk calculation
        position_value = risk_amount / stop_distance

        # Apply position size limit
        max_position_value = equity * limits.max_position_pct
        position_value = min(position_value, max_position_value)

        # Apply leverage limit
        max_leveraged_value = self._portfolio.available_balance * limits.max_leverage
        position_value = min(position_value, max_leveraged_value)

        # Convert to quantity
        if request.entry_price > 0:
            quantity = position_value / request.entry_price
        else:
            quantity = 0

        return quantity

    def _get_regime_limits(self, regime: MarketRegime) -> RegimeLimits:
        """Get limits for regime, with config overrides."""
        regime_name = regime.value
        if regime_name in self.config.regime_limits:
            return self.config.regime_limits[regime_name]
        return RegimeLimits.for_regime(regime)

    def _is_no_trade_zone(self) -> bool:
        """Check if current time is in no-trade zone."""
        if not self.config.no_trade_hours:
            return False

        current_hour = datetime.utcnow().hour
        for start, end in self.config.no_trade_hours:
            if start <= end:
                if start <= current_hour < end:
                    return True
            else:  # Wraps around midnight
                if current_hour >= start or current_hour < end:
                    return True
        return False

    def _check_auto_kill_triggers(self) -> None:
        """Check if auto kill switch should trigger."""
        if self._kill_switch_active:
            return

        # Drawdown trigger
        if self._portfolio.current_drawdown_pct >= self.config.auto_kill_drawdown:
            self.activate_kill_switch(
                f"Auto-triggered: Drawdown {self._portfolio.current_drawdown_pct:.2%} >= {self.config.auto_kill_drawdown:.2%}"
            )
            return

        # Daily loss trigger
        if self._portfolio.daily_loss_pct >= self.config.auto_kill_daily_loss:
            self.activate_kill_switch(
                f"Auto-triggered: Daily loss {self._portfolio.daily_loss_pct:.2%} >= {self.config.auto_kill_daily_loss:.2%}"
            )

    def _check_kill_switch_expiry(self) -> None:
        """Check if kill switch cooldown has expired."""
        if self._kill_switch_until and datetime.now() >= self._kill_switch_until:
            logger.info("Kill switch cooldown expired, deactivating")
            self._kill_switch_active = False
            self._kill_switch_reason = None
            self._kill_switch_until = None
            self._save_state()

    def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch - blocks all new trades."""
        with self._lock:
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            self._kill_switch_until = datetime.now() + timedelta(
                hours=self.config.cooldown_after_kill_hours
            )
            self._save_state()

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

        # Send notification
        try:
            from bot.notifications import NotificationManager

            notifier = NotificationManager()
            notifier.send_critical(f"KILL SWITCH: {reason}")
        except Exception:
            pass

    def deactivate_kill_switch(self, approver: str) -> bool:
        """Manually deactivate kill switch."""
        with self._lock:
            if not self._kill_switch_active:
                return True

            logger.warning(f"Kill switch manually deactivated by: {approver}")
            self._kill_switch_active = False
            self._kill_switch_reason = None
            self._kill_switch_until = None
            self._save_state()

        return True

    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """Record trade result for risk tracking."""
        with self._lock:
            self._portfolio.daily_trades += 1
            self._portfolio.daily_pnl += pnl
            self._portfolio.last_trade_time = datetime.now()

            if is_win:
                self._portfolio.consecutive_losses = 0
            else:
                self._portfolio.consecutive_losses += 1
                self._portfolio.daily_loss += abs(pnl)

            # Update equity
            self._portfolio.equity += pnl
            if self._portfolio.equity > self._portfolio.peak_equity:
                self._portfolio.peak_equity = self._portfolio.equity

            # Check auto kill triggers
            self._check_auto_kill_triggers()

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new day)."""
        with self._lock:
            self._portfolio.daily_pnl = 0.0
            self._portfolio.daily_loss = 0.0
            self._portfolio.daily_trades = 0

    def _log_decision(self, request: TradeRequest, result: RiskCheckResult, message: str) -> None:
        """Log risk decision for audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": request.symbol,
            "direction": request.direction,
            "entry_price": request.entry_price,
            "stop_loss": request.stop_loss,
            "decision": result.decision.value,
            "block_reasons": [r.value for r in result.block_reasons],
            "approved_quantity": result.approved_quantity,
            "message": message,
            "regime": self._current_regime.regime.value if self._current_regime else "unknown",
            "portfolio_equity": self._portfolio.equity,
            "portfolio_drawdown": self._portfolio.current_drawdown_pct,
        }

        self._decision_log.append(entry)
        if len(self._decision_log) > 1000:
            self._decision_log = self._decision_log[-500:]

        level = logging.INFO if result.is_approved else logging.WARNING
        logger.log(level, f"Risk decision: {message} | {result.decision.value}")

    def get_status(self) -> Dict[str, Any]:
        """Get current risk engine status."""
        with self._lock:
            return {
                "kill_switch_active": self._kill_switch_active,
                "kill_switch_reason": self._kill_switch_reason,
                "kill_switch_until": self._kill_switch_until.isoformat()
                if self._kill_switch_until
                else None,
                "current_regime": self._current_regime.regime.value
                if self._current_regime
                else "unknown",
                "portfolio": {
                    "equity": self._portfolio.equity,
                    "available_balance": self._portfolio.available_balance,
                    "drawdown_pct": self._portfolio.current_drawdown_pct,
                    "daily_loss_pct": self._portfolio.daily_loss_pct,
                    "daily_trades": self._portfolio.daily_trades,
                    "consecutive_losses": self._portfolio.consecutive_losses,
                    "portfolio_heat": self._portfolio.portfolio_heat,
                },
                "limits": {
                    "max_daily_loss_pct": self.config.max_daily_loss_pct,
                    "max_drawdown_pct": self.config.max_drawdown_pct,
                    "max_consecutive_losses": self.config.max_consecutive_losses,
                },
            }

    @property
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed."""
        with self._lock:
            if self._kill_switch_active:
                return False, f"Kill switch active: {self._kill_switch_reason}"
            if self._portfolio.daily_loss_pct >= self.config.max_daily_loss_pct:
                return False, "Daily loss limit reached"
            if self._portfolio.current_drawdown_pct >= self.config.max_drawdown_pct:
                return False, "Drawdown limit reached"
            return True, "Trading allowed"
