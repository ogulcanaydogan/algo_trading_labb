"""
Risk Management Module.

Features:
- Daily loss limits
- Drawdown protection
- Position sizing based on volatility
- Exposure limits
- Correlation-based risk limiting
- Circuit breakers
- Market regime detection
- VaR and tail risk monitoring
- Recovery mode for drawdown recovery
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class TradingStatus(Enum):
    """Trading permission status."""
    ALLOWED = "allowed"
    REDUCED = "reduced"  # Only reduced size trades allowed
    HEDGING_ONLY = "hedging_only"  # Only risk-reducing trades
    BLOCKED = "blocked"  # No new trades


class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOL_TRENDING = "low_vol_trending"      # Best for trend following
    LOW_VOL_RANGING = "low_vol_ranging"        # Good for mean reversion
    HIGH_VOL_TRENDING = "high_vol_trending"    # Reduce size, wider stops
    HIGH_VOL_RANGING = "high_vol_ranging"      # Difficult, reduce activity
    CRISIS = "crisis"                          # Extreme vol, minimal activity


class RecoveryPhase(Enum):
    """Drawdown recovery phases."""
    NORMAL = "normal"           # No drawdown
    IN_DRAWDOWN = "in_drawdown" # Currently in drawdown
    RECOVERING = "recovering"   # Recovering from drawdown
    RECOVERED = "recovered"     # Just recovered, still cautious


@dataclass
class CorrelationConfig:
    """Configuration for correlation-based risk management."""
    enabled: bool = True
    max_correlated_exposure_pct: float = 40.0  # Max 40% in correlated assets
    correlation_threshold: float = 0.7         # Consider correlated if > 0.7
    lookback_days: int = 60                    # Correlation calculation window
    correlation_groups: Dict[str, List[str]] = field(default_factory=dict)
    # Default correlation groups (can be overridden)
    # e.g., {"crypto": ["BTC", "ETH", "SOL"], "commodities": ["GOLD", "SILVER"]}


@dataclass
class RegimeConfig:
    """Configuration for regime detection and adjustment."""
    enabled: bool = True
    volatility_lookback: int = 20              # Days for volatility calculation
    trend_lookback: int = 50                   # Days for trend detection
    low_vol_threshold: float = 15.0            # Below 15% annual vol = low vol
    high_vol_threshold: float = 30.0           # Above 30% annual vol = high vol
    crisis_vol_threshold: float = 50.0         # Above 50% = crisis
    trend_threshold: float = 0.6               # ADX > 25 or R² > 0.6 = trending
    regime_size_multipliers: Dict[MarketRegime, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.regime_size_multipliers:
            self.regime_size_multipliers = {
                MarketRegime.LOW_VOL_TRENDING: 1.2,
                MarketRegime.LOW_VOL_RANGING: 1.0,
                MarketRegime.HIGH_VOL_TRENDING: 0.7,
                MarketRegime.HIGH_VOL_RANGING: 0.5,
                MarketRegime.CRISIS: 0.25,
            }


@dataclass
class VaRConfig:
    """Configuration for Value-at-Risk monitoring."""
    enabled: bool = True
    confidence_level: float = 0.95             # 95% VaR
    lookback_days: int = 252                   # 1 year of data
    max_var_pct: float = 5.0                   # Max 5% daily VaR
    max_cvar_pct: float = 7.0                  # Max 7% CVaR (tail risk)
    use_parametric: bool = True                # Parametric VaR
    use_historical: bool = True                # Historical VaR


@dataclass
class RecoveryConfig:
    """Configuration for drawdown recovery mode."""
    enabled: bool = True
    recovery_threshold_pct: float = 50.0       # 50% of drawdown recovered
    full_recovery_threshold_pct: float = 90.0  # 90% = fully recovered
    initial_size_multiplier: float = 0.5       # Start at 50% size
    recovery_increment: float = 0.1            # Increase 10% per win
    max_recovery_multiplier: float = 1.0       # Don't exceed normal size
    min_winning_trades: int = 3                # Need 3 wins to increase


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Daily limits
    max_daily_loss_pct: float = 3.0  # Max 3% daily loss
    max_daily_trades: int = 20
    max_daily_volume: float = 100000.0  # Max daily trading volume

    # Drawdown protection
    max_drawdown_pct: float = 10.0  # Max 10% drawdown from peak
    drawdown_reduce_threshold: float = 5.0  # Reduce size at 5%
    drawdown_pause_threshold: float = 8.0  # Pause trading at 8%

    # Position sizing
    base_risk_per_trade_pct: float = 1.0  # 1% risk per trade
    max_risk_per_trade_pct: float = 2.0  # Max 2% in good conditions
    min_risk_per_trade_pct: float = 0.25  # Min 0.25% in bad conditions
    volatility_adjusted_sizing: bool = True
    target_volatility_pct: float = 15.0  # Annual vol target

    # Exposure limits
    max_single_position_pct: float = 20.0  # Max 20% in single position
    max_total_exposure_pct: float = 100.0  # Max 100% total exposure
    max_correlated_exposure_pct: float = 40.0  # Max 40% in correlated assets

    # Win/loss streak management
    reduce_after_losses: int = 3  # Reduce size after 3 consecutive losses
    increase_after_wins: int = 5  # Allow increase after 5 consecutive wins
    streak_size_adjustment: float = 0.5  # Reduce to 50% after loss streak

    # Circuit breakers
    circuit_breaker_loss_pct: float = 5.0  # Pause after 5% loss in single trade
    circuit_breaker_duration_hours: int = 4  # Pause duration

    # Time-based restrictions
    avoid_news_minutes: int = 30  # Avoid trading around news
    max_overnight_exposure_pct: float = 50.0  # Reduce overnight

    # Advanced risk configs
    correlation_config: CorrelationConfig = field(default_factory=CorrelationConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    var_config: VaRConfig = field(default_factory=VaRConfig)
    recovery_config: RecoveryConfig = field(default_factory=RecoveryConfig)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_balance: float
    current_balance: float
    high_water_mark: float
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_volume: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    @property
    def daily_pnl_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.current_balance / self.starting_balance - 1) * 100

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return self.winning_trades / self.trades_count

    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "starting_balance": self.starting_balance,
            "current_balance": self.current_balance,
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "trades_count": self.trades_count,
            "win_rate": round(self.win_rate, 2),
            "max_drawdown": round(self.max_drawdown, 2),
        }


@dataclass
class VaRMetrics:
    """Value-at-Risk metrics."""
    var_95: float = 0.0           # 95% VaR (parametric)
    var_99: float = 0.0           # 99% VaR
    cvar_95: float = 0.0          # Conditional VaR (Expected Shortfall)
    historical_var_95: float = 0.0 # Historical VaR
    max_loss_pct: float = 0.0     # Largest historical loss
    var_breach: bool = False       # Is current VaR above limit?

    def to_dict(self) -> Dict:
        return {
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "cvar_95": round(self.cvar_95, 4),
            "historical_var_95": round(self.historical_var_95, 4),
            "max_loss_pct": round(self.max_loss_pct, 4),
            "var_breach": self.var_breach,
        }


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    status: TradingStatus
    risk_level: RiskLevel
    allowed_risk_pct: float
    position_size_multiplier: float
    reasons: List[str]
    warnings: List[str]
    daily_stats: Optional[DailyStats] = None
    market_regime: Optional[MarketRegime] = None
    recovery_phase: Optional[RecoveryPhase] = None
    var_metrics: Optional[VaRMetrics] = None
    correlated_exposure_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "risk_level": self.risk_level.value,
            "allowed_risk_pct": round(self.allowed_risk_pct, 4),
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "reasons": self.reasons,
            "warnings": self.warnings,
            "market_regime": self.market_regime.value if self.market_regime else None,
            "recovery_phase": self.recovery_phase.value if self.recovery_phase else None,
            "var_metrics": self.var_metrics.to_dict() if self.var_metrics else None,
            "correlated_exposure_pct": round(self.correlated_exposure_pct, 2),
        }


class RiskManager:
    """
    Comprehensive Risk Management System.

    Tracks:
    - Daily P&L and limits
    - Drawdown from peak
    - Win/loss streaks
    - Exposure limits
    - Circuit breaker conditions

    Provides:
    - Trading permission status
    - Dynamic position sizing
    - Risk-adjusted recommendations
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_balance: float = 10000.0,
        data_dir: str = "data",
    ):
        self.config = config or RiskConfig()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.data_dir = Path(data_dir)

        # Daily tracking
        self._daily_stats: Optional[DailyStats] = None
        self._stats_history: List[DailyStats] = []

        # Circuit breaker state
        self._circuit_breaker_until: Optional[datetime] = None
        self._blocked_reason: Optional[str] = None

        # Correlation data (symbol -> list of correlated symbols)
        self._correlations: Dict[str, List[str]] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._returns_history: Dict[str, List[float]] = {}

        # Regime detection state
        self._current_regime: MarketRegime = MarketRegime.LOW_VOL_RANGING
        self._regime_history: List[Tuple[datetime, MarketRegime]] = []

        # Recovery mode state
        self._recovery_phase: RecoveryPhase = RecoveryPhase.NORMAL
        self._drawdown_peak: float = initial_balance
        self._drawdown_trough: float = initial_balance
        self._recovery_wins: int = 0
        self._recovery_multiplier: float = 1.0

        # VaR tracking
        self._portfolio_returns: List[float] = []
        self._var_metrics: VaRMetrics = VaRMetrics()

        # Load persisted state
        self._load_state()

    def assess_risk(
        self,
        proposed_trade_value: float = 0.0,
        symbol: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None,
        market_data: Optional[pd.DataFrame] = None,
    ) -> RiskAssessment:
        """
        Assess current risk and determine trading permissions.

        Args:
            proposed_trade_value: Value of proposed new trade
            symbol: Symbol for the proposed trade
            current_positions: Dict of symbol -> position value
            market_data: OHLCV data for regime detection

        Returns:
            RiskAssessment with status and recommendations
        """
        self._ensure_daily_stats()
        current_positions = current_positions or {}

        reasons = []
        warnings = []
        status = TradingStatus.ALLOWED
        risk_level = RiskLevel.NORMAL
        size_multiplier = 1.0
        correlated_exposure = 0.0

        # Check circuit breaker
        if self._circuit_breaker_until and datetime.now() < self._circuit_breaker_until:
            remaining = (self._circuit_breaker_until - datetime.now()).seconds // 60
            reasons.append(f"Circuit breaker active for {remaining} more minutes")
            return RiskAssessment(
                status=TradingStatus.BLOCKED,
                risk_level=RiskLevel.CRITICAL,
                allowed_risk_pct=0,
                position_size_multiplier=0,
                reasons=reasons,
                warnings=[self._blocked_reason or "Circuit breaker triggered"],
                daily_stats=self._daily_stats,
                market_regime=self._current_regime,
                recovery_phase=self._recovery_phase,
                var_metrics=self._var_metrics,
            )

        # Check daily loss limit
        if self._daily_stats.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            reasons.append(f"Daily loss limit reached ({self._daily_stats.daily_pnl_pct:.1f}%)")
            status = TradingStatus.BLOCKED
            risk_level = RiskLevel.CRITICAL

        elif self._daily_stats.daily_pnl_pct <= -self.config.max_daily_loss_pct * 0.7:
            warnings.append(f"Approaching daily loss limit ({self._daily_stats.daily_pnl_pct:.1f}%)")
            status = TradingStatus.REDUCED
            risk_level = RiskLevel.HIGH
            size_multiplier *= 0.5

        # Check drawdown from peak
        drawdown_pct = self._calculate_drawdown_pct()

        if drawdown_pct >= self.config.drawdown_pause_threshold:
            reasons.append(f"Drawdown pause threshold reached ({drawdown_pct:.1f}%)")
            status = TradingStatus.BLOCKED
            risk_level = RiskLevel.CRITICAL

        elif drawdown_pct >= self.config.drawdown_reduce_threshold:
            warnings.append(f"In drawdown ({drawdown_pct:.1f}%), reducing position sizes")
            if status == TradingStatus.ALLOWED:
                status = TradingStatus.REDUCED
            risk_level = max(risk_level, RiskLevel.ELEVATED, key=lambda x: list(RiskLevel).index(x))
            size_multiplier *= 0.5

        # Check daily trade count
        if self._daily_stats.trades_count >= self.config.max_daily_trades:
            reasons.append(f"Max daily trades reached ({self._daily_stats.trades_count})")
            status = TradingStatus.BLOCKED

        elif self._daily_stats.trades_count >= self.config.max_daily_trades * 0.8:
            warnings.append(f"Approaching max daily trades ({self._daily_stats.trades_count})")

        # Check consecutive losses
        if self._daily_stats.consecutive_losses >= self.config.reduce_after_losses:
            warnings.append(f"Loss streak ({self._daily_stats.consecutive_losses}), reducing size")
            size_multiplier *= self.config.streak_size_adjustment
            risk_level = max(risk_level, RiskLevel.ELEVATED, key=lambda x: list(RiskLevel).index(x))

        # Check exposure limits
        total_exposure = sum(abs(v) for v in current_positions.values())
        exposure_pct = (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0

        if exposure_pct >= self.config.max_total_exposure_pct:
            reasons.append(f"Max exposure reached ({exposure_pct:.1f}%)")
            status = TradingStatus.HEDGING_ONLY

        elif exposure_pct >= self.config.max_total_exposure_pct * 0.8:
            warnings.append(f"High exposure ({exposure_pct:.1f}%)")

        # Check single position limit
        if symbol and symbol in current_positions:
            position_pct = (abs(current_positions[symbol]) / self.current_balance) * 100
            if position_pct >= self.config.max_single_position_pct:
                reasons.append(f"Position limit for {symbol} reached ({position_pct:.1f}%)")
                status = TradingStatus.BLOCKED

        # === CORRELATION-BASED RISK ===
        if self.config.correlation_config.enabled and symbol and current_positions:
            correlated_exposure = self._check_correlated_exposure(
                symbol, current_positions, proposed_trade_value
            )
            if correlated_exposure >= self.config.correlation_config.max_correlated_exposure_pct:
                reasons.append(f"Correlated exposure limit reached ({correlated_exposure:.1f}%)")
                if status == TradingStatus.ALLOWED:
                    status = TradingStatus.REDUCED
                size_multiplier *= 0.5
            elif correlated_exposure >= self.config.correlation_config.max_correlated_exposure_pct * 0.8:
                warnings.append(f"High correlated exposure ({correlated_exposure:.1f}%)")

        # === REGIME-AWARE ADJUSTMENT ===
        if self.config.regime_config.enabled and market_data is not None:
            self._current_regime = self._detect_market_regime(market_data)
            regime_multiplier = self.config.regime_config.regime_size_multipliers.get(
                self._current_regime, 1.0
            )
            size_multiplier *= regime_multiplier

            if self._current_regime == MarketRegime.CRISIS:
                warnings.append("CRISIS regime detected - minimal position sizing")
                if status == TradingStatus.ALLOWED:
                    status = TradingStatus.REDUCED
            elif self._current_regime == MarketRegime.HIGH_VOL_RANGING:
                warnings.append("High volatility ranging - reduced activity recommended")

        # === VaR CHECK ===
        if self.config.var_config.enabled and len(self._portfolio_returns) > 20:
            self._update_var_metrics()
            if self._var_metrics.var_breach:
                warnings.append(f"VaR limit breached: {self._var_metrics.var_95*100:.2f}%")
                size_multiplier *= 0.5
                if status == TradingStatus.ALLOWED:
                    status = TradingStatus.REDUCED

        # === RECOVERY MODE ===
        if self.config.recovery_config.enabled:
            self._update_recovery_phase()
            if self._recovery_phase in [RecoveryPhase.RECOVERING, RecoveryPhase.RECOVERED]:
                size_multiplier *= self._recovery_multiplier
                warnings.append(f"Recovery mode: {self._recovery_multiplier*100:.0f}% sizing")

        # Calculate allowed risk percentage
        base_risk = self.config.base_risk_per_trade_pct
        allowed_risk = base_risk * size_multiplier
        allowed_risk = max(self.config.min_risk_per_trade_pct,
                         min(self.config.max_risk_per_trade_pct, allowed_risk))

        # Consecutive wins bonus (only if not in recovery)
        if self._daily_stats.consecutive_wins >= self.config.increase_after_wins:
            if risk_level in [RiskLevel.LOW, RiskLevel.NORMAL]:
                if self._recovery_phase == RecoveryPhase.NORMAL:
                    allowed_risk = min(allowed_risk * 1.25, self.config.max_risk_per_trade_pct)
                    warnings.append("Win streak bonus applied")

        return RiskAssessment(
            status=status,
            risk_level=risk_level,
            allowed_risk_pct=allowed_risk,
            position_size_multiplier=size_multiplier,
            reasons=reasons,
            warnings=warnings,
            daily_stats=self._daily_stats,
            market_regime=self._current_regime,
            recovery_phase=self._recovery_phase,
            var_metrics=self._var_metrics if self.config.var_config.enabled else None,
            correlated_exposure_pct=correlated_exposure,
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        volatility: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            volatility: Current volatility (annualized, optional)
            symbol: Trading symbol (optional)

        Returns:
            Tuple of (position_size, risk_amount)
        """
        assessment = self.assess_risk(symbol=symbol)

        if assessment.status == TradingStatus.BLOCKED:
            return 0.0, 0.0

        # Base risk amount
        risk_pct = assessment.allowed_risk_pct / 100
        risk_amount = self.current_balance * risk_pct

        # Volatility adjustment
        if self.config.volatility_adjusted_sizing and volatility:
            vol_ratio = self.config.target_volatility_pct / max(volatility, 1.0)
            vol_ratio = max(0.5, min(2.0, vol_ratio))  # Clamp to 0.5x - 2x
            risk_amount *= vol_ratio

        # Calculate position size from risk and stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0:
            return 0.0, 0.0

        position_size = risk_amount / stop_distance

        # Apply position multiplier
        position_size *= assessment.position_size_multiplier

        return position_size, risk_amount

    def record_trade(
        self,
        pnl: float,
        volume: float,
        is_winner: bool,
    ):
        """
        Record a completed trade.

        Args:
            pnl: Profit/loss amount
            volume: Trade volume
            is_winner: Whether trade was profitable
        """
        self._ensure_daily_stats()

        self._daily_stats.trades_count += 1
        self._daily_stats.total_pnl += pnl
        self._daily_stats.total_volume += volume
        self.current_balance += pnl

        if is_winner:
            self._daily_stats.winning_trades += 1
            self._daily_stats.consecutive_wins += 1
            self._daily_stats.consecutive_losses = 0
        else:
            self._daily_stats.losing_trades += 1
            self._daily_stats.consecutive_losses += 1
            self._daily_stats.consecutive_wins = 0

        # Update high water mark
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        self._daily_stats.current_balance = self.current_balance
        self._daily_stats.high_water_mark = max(
            self._daily_stats.high_water_mark, self.current_balance
        )

        # Update max drawdown
        current_dd = self._calculate_drawdown_pct()
        self._daily_stats.max_drawdown = max(self._daily_stats.max_drawdown, current_dd)

        # Check for circuit breaker
        trade_pnl_pct = (pnl / (self.current_balance - pnl)) * 100 if self.current_balance != pnl else 0
        if trade_pnl_pct <= -self.config.circuit_breaker_loss_pct:
            self._trigger_circuit_breaker(f"Large loss: {trade_pnl_pct:.1f}%")

        self._save_state()

    def update_balance(self, new_balance: float):
        """Update current balance (for unrealized P&L updates)."""
        self.current_balance = new_balance
        self._ensure_daily_stats()
        self._daily_stats.current_balance = new_balance

        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def _calculate_drawdown_pct(self) -> float:
        """Calculate current drawdown from peak as percentage."""
        if self.peak_balance <= 0:
            return 0.0
        return (1 - self.current_balance / self.peak_balance) * 100

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to pause trading."""
        self._circuit_breaker_until = datetime.now() + timedelta(
            hours=self.config.circuit_breaker_duration_hours
        )
        self._blocked_reason = reason
        self._save_state()

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self._circuit_breaker_until = None
        self._blocked_reason = None
        self._save_state()

    def _ensure_daily_stats(self):
        """Ensure daily stats exist for today."""
        today = date.today()

        if self._daily_stats is None or self._daily_stats.date != today:
            # Save previous day stats
            if self._daily_stats:
                self._stats_history.append(self._daily_stats)
                if len(self._stats_history) > 365:
                    self._stats_history = self._stats_history[-365:]

            # Create new daily stats
            self._daily_stats = DailyStats(
                date=today,
                starting_balance=self.current_balance,
                current_balance=self.current_balance,
                high_water_mark=self.current_balance,
            )

    def get_daily_stats(self) -> Optional[DailyStats]:
        """Get current daily statistics."""
        self._ensure_daily_stats()
        return self._daily_stats

    def get_stats_history(self, days: int = 30) -> List[DailyStats]:
        """Get historical daily stats."""
        return self._stats_history[-days:]

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        self._ensure_daily_stats()
        assessment = self.assess_risk()

        return {
            "current_balance": round(self.current_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": round(self._calculate_drawdown_pct(), 2),
            "daily_pnl_pct": round(self._daily_stats.daily_pnl_pct, 2),
            "trades_today": self._daily_stats.trades_count,
            "win_rate_today": round(self._daily_stats.win_rate * 100, 1),
            "consecutive_losses": self._daily_stats.consecutive_losses,
            "consecutive_wins": self._daily_stats.consecutive_wins,
            "trading_status": assessment.status.value,
            "risk_level": assessment.risk_level.value,
            "allowed_risk_pct": round(assessment.allowed_risk_pct, 2),
            "circuit_breaker_active": self._circuit_breaker_until is not None,
            "warnings": assessment.warnings,
        }

    def _save_state(self):
        """Save risk manager state to disk."""
        state_file = self.data_dir / "risk_state.json"
        state = {
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "circuit_breaker_until": self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None,
            "blocked_reason": self._blocked_reason,
            "daily_stats": self._daily_stats.to_dict() if self._daily_stats else None,
            "updated_at": datetime.now().isoformat(),
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load risk manager state from disk."""
        state_file = self.data_dir / "risk_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            self.current_balance = state.get("current_balance", self.initial_balance)
            self.peak_balance = state.get("peak_balance", self.current_balance)

            cb_until = state.get("circuit_breaker_until")
            if cb_until:
                self._circuit_breaker_until = datetime.fromisoformat(cb_until)
            self._blocked_reason = state.get("blocked_reason")

        except (json.JSONDecodeError, KeyError):
            pass  # Use defaults

    # =====================
    # CORRELATION METHODS
    # =====================

    def update_returns(self, symbol: str, returns: List[float]):
        """
        Update returns history for correlation calculation.

        Args:
            symbol: Trading symbol
            returns: List of recent returns
        """
        self._returns_history[symbol] = returns[-self.config.correlation_config.lookback_days:]
        self._update_correlation_matrix()

    def _update_correlation_matrix(self):
        """Recalculate correlation matrix from returns history."""
        if len(self._returns_history) < 2:
            return

        # Build DataFrame from returns
        min_length = min(len(r) for r in self._returns_history.values())
        if min_length < 20:
            return

        data = {}
        for symbol, returns in self._returns_history.items():
            data[symbol] = returns[-min_length:]

        df = pd.DataFrame(data)
        self._correlation_matrix = df.corr()

    def _check_correlated_exposure(
        self,
        new_symbol: str,
        current_positions: Dict[str, float],
        proposed_value: float,
    ) -> float:
        """
        Check correlated exposure for a proposed trade.

        Args:
            new_symbol: Symbol for proposed trade
            current_positions: Current position values
            proposed_value: Value of proposed trade

        Returns:
            Total correlated exposure as percentage of balance
        """
        if self.current_balance <= 0:
            return 0.0

        correlated_exposure = abs(proposed_value)
        threshold = self.config.correlation_config.correlation_threshold

        # Check correlation matrix
        if self._correlation_matrix is not None and new_symbol in self._correlation_matrix.columns:
            for symbol, position_value in current_positions.items():
                if symbol in self._correlation_matrix.columns:
                    corr = self._correlation_matrix.loc[new_symbol, symbol]
                    if abs(corr) >= threshold:
                        # Add weighted exposure based on correlation
                        correlated_exposure += abs(position_value) * abs(corr)

        # Check predefined correlation groups
        for group_name, symbols in self.config.correlation_config.correlation_groups.items():
            if new_symbol in symbols:
                for symbol in symbols:
                    if symbol in current_positions and symbol != new_symbol:
                        correlated_exposure += abs(current_positions[symbol])

        return (correlated_exposure / self.current_balance) * 100

    def get_correlated_symbols(self, symbol: str) -> List[Tuple[str, float]]:
        """
        Get symbols correlated with the given symbol.

        Returns:
            List of (symbol, correlation) tuples
        """
        result = []
        threshold = self.config.correlation_config.correlation_threshold

        if self._correlation_matrix is not None and symbol in self._correlation_matrix.columns:
            correlations = self._correlation_matrix[symbol]
            for other_symbol, corr in correlations.items():
                if other_symbol != symbol and abs(corr) >= threshold:
                    result.append((other_symbol, corr))

        return sorted(result, key=lambda x: abs(x[1]), reverse=True)

    # =====================
    # REGIME DETECTION
    # =====================

    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime from price data.

        Args:
            market_data: OHLCV DataFrame with 'close' column

        Returns:
            Current market regime classification
        """
        if len(market_data) < self.config.regime_config.trend_lookback:
            return MarketRegime.LOW_VOL_RANGING

        close = market_data["close"]
        returns = close.pct_change().dropna()

        # Calculate volatility (annualized)
        vol_lookback = self.config.regime_config.volatility_lookback
        recent_vol = returns.tail(vol_lookback).std() * np.sqrt(252) * 100

        # Determine volatility level
        if recent_vol >= self.config.regime_config.crisis_vol_threshold:
            return MarketRegime.CRISIS

        is_high_vol = recent_vol >= self.config.regime_config.high_vol_threshold
        is_low_vol = recent_vol <= self.config.regime_config.low_vol_threshold

        # Detect trend using linear regression R²
        trend_lookback = self.config.regime_config.trend_lookback
        prices = close.tail(trend_lookback).values
        x = np.arange(len(prices))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        r_squared = r_value ** 2

        is_trending = r_squared >= self.config.regime_config.trend_threshold

        # Classify regime
        if is_high_vol:
            if is_trending:
                regime = MarketRegime.HIGH_VOL_TRENDING
            else:
                regime = MarketRegime.HIGH_VOL_RANGING
        else:  # Low or normal volatility
            if is_trending:
                regime = MarketRegime.LOW_VOL_TRENDING
            else:
                regime = MarketRegime.LOW_VOL_RANGING

        # Track regime changes
        if not self._regime_history or self._regime_history[-1][1] != regime:
            self._regime_history.append((datetime.now(), regime))
            if len(self._regime_history) > 100:
                self._regime_history = self._regime_history[-100:]

        return regime

    def get_regime_history(self, count: int = 10) -> List[Tuple[datetime, MarketRegime]]:
        """Get recent regime history."""
        return self._regime_history[-count:]

    # =====================
    # VaR CALCULATIONS
    # =====================

    def add_portfolio_return(self, daily_return: float):
        """
        Add a daily portfolio return for VaR calculation.

        Args:
            daily_return: Daily return as decimal (e.g., 0.01 for 1%)
        """
        self._portfolio_returns.append(daily_return)
        max_history = self.config.var_config.lookback_days
        if len(self._portfolio_returns) > max_history:
            self._portfolio_returns = self._portfolio_returns[-max_history:]

    def _update_var_metrics(self):
        """Calculate VaR and CVaR metrics from returns history."""
        if len(self._portfolio_returns) < 20:
            return

        returns = np.array(self._portfolio_returns)
        confidence = self.config.var_config.confidence_level

        # Parametric VaR (assuming normal distribution)
        if self.config.var_config.use_parametric:
            mean = np.mean(returns)
            std = np.std(returns)
            z_score_95 = stats.norm.ppf(1 - confidence)
            z_score_99 = stats.norm.ppf(0.01)

            self._var_metrics.var_95 = -(mean + z_score_95 * std)
            self._var_metrics.var_99 = -(mean + z_score_99 * std)

        # Historical VaR
        if self.config.var_config.use_historical:
            self._var_metrics.historical_var_95 = -np.percentile(returns, (1 - confidence) * 100)

        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) > 0:
            self._var_metrics.cvar_95 = -np.mean(tail_losses)
        else:
            self._var_metrics.cvar_95 = self._var_metrics.var_95

        # Maximum loss
        self._var_metrics.max_loss_pct = -np.min(returns)

        # Check if VaR exceeds limits
        self._var_metrics.var_breach = (
            self._var_metrics.var_95 * 100 > self.config.var_config.max_var_pct or
            self._var_metrics.cvar_95 * 100 > self.config.var_config.max_cvar_pct
        )

    def get_var_metrics(self) -> VaRMetrics:
        """Get current VaR metrics."""
        self._update_var_metrics()
        return self._var_metrics

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        horizon_days: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate portfolio VaR for current positions.

        Args:
            positions: Dict of symbol -> position value
            horizon_days: Time horizon in days

        Returns:
            Dict with VaR metrics
        """
        if not positions or self._correlation_matrix is None:
            return {"var_95": 0, "var_99": 0, "cvar_95": 0}

        # Get weights
        total_value = sum(abs(v) for v in positions.values())
        if total_value == 0:
            return {"var_95": 0, "var_99": 0, "cvar_95": 0}

        symbols = list(positions.keys())
        weights = np.array([positions.get(s, 0) / total_value for s in symbols])

        # Get volatilities and correlations
        vols = []
        for symbol in symbols:
            if symbol in self._returns_history:
                returns = np.array(self._returns_history[symbol])
                vols.append(np.std(returns) * np.sqrt(252 * horizon_days))
            else:
                vols.append(0.2)  # Default 20% annual vol

        vols = np.array(vols)

        # Build covariance matrix
        corr_subset = self._correlation_matrix.loc[symbols, symbols].values
        cov_matrix = np.outer(vols, vols) * corr_subset

        # Portfolio volatility
        port_vol = np.sqrt(weights @ cov_matrix @ weights)

        # Calculate VaR
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        var_95 = total_value * port_vol * z_95
        var_99 = total_value * port_vol * z_99
        cvar_95 = var_95 * 1.2  # Approximation

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "portfolio_vol": port_vol,
        }

    # =====================
    # RECOVERY MODE
    # =====================

    def _update_recovery_phase(self):
        """Update recovery phase based on current balance vs drawdown."""
        config = self.config.recovery_config

        # Update drawdown tracking
        if self.current_balance > self._drawdown_peak:
            self._drawdown_peak = self.current_balance

        if self.current_balance < self._drawdown_trough:
            self._drawdown_trough = self.current_balance

        # Calculate drawdown and recovery percentages
        if self._drawdown_peak <= 0:
            return

        drawdown = self._drawdown_peak - self._drawdown_trough
        if drawdown <= 0:
            self._recovery_phase = RecoveryPhase.NORMAL
            self._recovery_multiplier = 1.0
            return

        recovery = self.current_balance - self._drawdown_trough
        recovery_pct = (recovery / drawdown) * 100 if drawdown > 0 else 100

        # Determine phase
        current_dd_pct = self._calculate_drawdown_pct()

        if current_dd_pct < 1.0:  # Less than 1% drawdown
            self._recovery_phase = RecoveryPhase.NORMAL
            self._recovery_multiplier = 1.0
            self._drawdown_trough = self.current_balance  # Reset trough
            self._recovery_wins = 0

        elif recovery_pct >= config.full_recovery_threshold_pct:
            self._recovery_phase = RecoveryPhase.RECOVERED
            self._recovery_multiplier = min(
                config.max_recovery_multiplier,
                config.initial_size_multiplier + self._recovery_wins * config.recovery_increment
            )

        elif recovery_pct >= config.recovery_threshold_pct:
            self._recovery_phase = RecoveryPhase.RECOVERING
            self._recovery_multiplier = min(
                config.max_recovery_multiplier * 0.8,
                config.initial_size_multiplier + self._recovery_wins * config.recovery_increment
            )

        else:
            self._recovery_phase = RecoveryPhase.IN_DRAWDOWN
            self._recovery_multiplier = config.initial_size_multiplier

    def record_recovery_trade(self, is_winner: bool):
        """
        Record a trade result during recovery mode.

        Args:
            is_winner: Whether the trade was profitable
        """
        if self._recovery_phase not in [RecoveryPhase.RECOVERING, RecoveryPhase.RECOVERED]:
            return

        config = self.config.recovery_config

        if is_winner:
            self._recovery_wins += 1
            if self._recovery_wins >= config.min_winning_trades:
                self._recovery_multiplier = min(
                    config.max_recovery_multiplier,
                    self._recovery_multiplier + config.recovery_increment
                )
        else:
            # Reset on loss during recovery
            self._recovery_wins = max(0, self._recovery_wins - 2)
            self._recovery_multiplier = max(
                config.initial_size_multiplier,
                self._recovery_multiplier - config.recovery_increment
            )

    def get_recovery_status(self) -> Dict:
        """Get current recovery mode status."""
        self._update_recovery_phase()
        return {
            "phase": self._recovery_phase.value,
            "multiplier": round(self._recovery_multiplier, 2),
            "recovery_wins": self._recovery_wins,
            "drawdown_peak": round(self._drawdown_peak, 2),
            "drawdown_trough": round(self._drawdown_trough, 2),
            "current_balance": round(self.current_balance, 2),
        }

    # =====================
    # ENHANCED SUMMARY
    # =====================

    def get_advanced_risk_summary(self) -> Dict:
        """Get comprehensive risk summary including all advanced features."""
        self._ensure_daily_stats()
        assessment = self.assess_risk()

        summary = {
            # Basic metrics
            "current_balance": round(self.current_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": round(self._calculate_drawdown_pct(), 2),
            "daily_pnl_pct": round(self._daily_stats.daily_pnl_pct, 2),
            "trades_today": self._daily_stats.trades_count,
            "win_rate_today": round(self._daily_stats.win_rate * 100, 1),

            # Status
            "trading_status": assessment.status.value,
            "risk_level": assessment.risk_level.value,
            "allowed_risk_pct": round(assessment.allowed_risk_pct, 2),
            "position_size_multiplier": round(assessment.position_size_multiplier, 2),

            # Regime
            "market_regime": self._current_regime.value,

            # Recovery
            "recovery_phase": self._recovery_phase.value,
            "recovery_multiplier": round(self._recovery_multiplier, 2),

            # VaR
            "var_95": round(self._var_metrics.var_95 * 100, 2) if self._var_metrics else None,
            "cvar_95": round(self._var_metrics.cvar_95 * 100, 2) if self._var_metrics else None,
            "var_breach": self._var_metrics.var_breach if self._var_metrics else False,

            # Alerts
            "warnings": assessment.warnings,
            "reasons": assessment.reasons,

            # Circuit breaker
            "circuit_breaker_active": self._circuit_breaker_until is not None,
        }

        return summary
