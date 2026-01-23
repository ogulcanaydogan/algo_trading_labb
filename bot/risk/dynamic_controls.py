"""
Dynamic Risk Controls - Circuit Breakers and Position Sizing.

Provides:
- Correlation circuit breaker for detecting correlation breakdown
- Dynamic position sizing based on volatility and regime
- Adaptive risk limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class RiskLevel(Enum):
    """Portfolio risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorrelationAlert:
    """Alert for correlation changes."""
    timestamp: datetime
    asset_pair: Tuple[str, str]
    historical_correlation: float
    current_correlation: float
    change_magnitude: float
    alert_type: str  # "spike", "breakdown", "regime_change"
    severity: str  # "warning", "critical"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_pair": list(self.asset_pair),
            "historical_correlation": round(self.historical_correlation, 4),
            "current_correlation": round(self.current_correlation, 4),
            "change_magnitude": round(self.change_magnitude, 4),
            "alert_type": self.alert_type,
            "severity": self.severity,
        }


@dataclass
class CircuitBreakerStatus:
    """Current status of circuit breaker."""
    state: CircuitBreakerState
    triggered_at: Optional[datetime]
    trigger_reason: Optional[str]
    cooldown_until: Optional[datetime]
    alerts: List[CorrelationAlert]
    risk_reduction_factor: float  # 1.0 = normal, 0 = halt trading

    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "trigger_reason": self.trigger_reason,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "num_alerts": len(self.alerts),
            "risk_reduction_factor": round(self.risk_reduction_factor, 4),
        }


@dataclass
class CorrelationConfig:
    """Correlation circuit breaker configuration."""
    # Thresholds
    correlation_spike_threshold: float = 0.3  # Alert if correlation changes > 30%
    correlation_critical_threshold: float = 0.5  # Trigger if change > 50%
    average_correlation_limit: float = 0.8  # Trigger if avg corr > 80%

    # Windows
    historical_window_days: int = 60
    recent_window_days: int = 5

    # Cooldown
    cooldown_minutes: int = 30

    # Actions
    warning_reduction_factor: float = 0.7
    triggered_reduction_factor: float = 0.3


class CorrelationCircuitBreaker:
    """
    Circuit Breaker for Correlation Breakdown Detection.

    Monitors portfolio correlations and triggers protective actions
    when correlation structure breaks down (typically during market stress).

    Actions:
    - WARNING: Reduce position sizes by warning_reduction_factor
    - TRIGGERED: Severely reduce positions, consider closing hedges
    - COOLDOWN: Gradually return to normal after trigger
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._state = CircuitBreakerState.NORMAL
        self._triggered_at: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._alerts: List[CorrelationAlert] = []
        self._historical_correlations: Dict[Tuple[str, str], float] = {}
        self._listeners: List[Callable[[CircuitBreakerStatus], None]] = []

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state, accounting for cooldown."""
        if self._state == CircuitBreakerState.TRIGGERED:
            if self._cooldown_until and datetime.now() > self._cooldown_until:
                self._state = CircuitBreakerState.COOLDOWN
        elif self._state == CircuitBreakerState.COOLDOWN:
            # Gradually return to normal (could add more logic here)
            pass
        return self._state

    @property
    def risk_reduction_factor(self) -> float:
        """Get current risk reduction factor."""
        state = self.state
        if state == CircuitBreakerState.NORMAL:
            return 1.0
        elif state == CircuitBreakerState.WARNING:
            return self.config.warning_reduction_factor
        elif state == CircuitBreakerState.TRIGGERED:
            return self.config.triggered_reduction_factor
        elif state == CircuitBreakerState.COOLDOWN:
            # Linearly recover from triggered to normal
            if self._cooldown_until and self._triggered_at:
                total_cooldown = (self._cooldown_until - self._triggered_at).total_seconds()
                elapsed = (datetime.now() - self._triggered_at).total_seconds()
                recovery_pct = min(1.0, elapsed / total_cooldown)
                return self.config.triggered_reduction_factor + \
                       (1.0 - self.config.triggered_reduction_factor) * recovery_pct
            return self.config.warning_reduction_factor
        return 1.0

    def update_correlations(
        self,
        returns: pd.DataFrame,
        historical_returns: Optional[pd.DataFrame] = None
    ) -> CircuitBreakerStatus:
        """
        Update correlation monitoring with new returns data.

        Args:
            returns: Recent returns (columns = assets)
            historical_returns: Historical returns for comparison

        Returns:
            Current circuit breaker status
        """
        # Calculate current correlations
        current_corr = returns.corr()

        # Calculate historical correlations
        if historical_returns is not None:
            historical_corr = historical_returns.corr()
        elif self._historical_correlations:
            # Use stored historical correlations
            assets = returns.columns.tolist()
            historical_corr = pd.DataFrame(
                index=assets, columns=assets, dtype=float
            )
            for (a1, a2), corr in self._historical_correlations.items():
                if a1 in assets and a2 in assets:
                    historical_corr.loc[a1, a2] = corr
                    historical_corr.loc[a2, a1] = corr
            for a in assets:
                historical_corr.loc[a, a] = 1.0
        else:
            # No historical data, use current as baseline
            historical_corr = current_corr
            self._store_historical(current_corr)

        # Check for correlation anomalies
        alerts = self._detect_anomalies(current_corr, historical_corr)
        self._alerts.extend(alerts)

        # Update state based on alerts
        self._update_state(current_corr, alerts)

        # Notify listeners
        status = self.get_status()
        for listener in self._listeners:
            try:
                listener(status)
            except Exception as e:
                logger.error(f"Listener error: {e}")

        return status

    def _store_historical(self, corr: pd.DataFrame):
        """Store correlations as historical baseline."""
        for i, asset1 in enumerate(corr.columns):
            for j, asset2 in enumerate(corr.columns):
                if i < j:
                    self._historical_correlations[(asset1, asset2)] = corr.iloc[i, j]

    def _detect_anomalies(
        self,
        current: pd.DataFrame,
        historical: pd.DataFrame
    ) -> List[CorrelationAlert]:
        """Detect correlation anomalies."""
        alerts = []
        now = datetime.now()

        for i, asset1 in enumerate(current.columns):
            for j, asset2 in enumerate(current.columns):
                if i >= j:
                    continue

                current_corr = current.iloc[i, j]
                historical_corr = historical.iloc[i, j]

                # Skip if either is NaN
                if pd.isna(current_corr) or pd.isna(historical_corr):
                    continue

                change = abs(current_corr - historical_corr)

                # Check for spike (correlation moving toward 1)
                if current_corr > historical_corr + self.config.correlation_spike_threshold:
                    severity = "critical" if change > self.config.correlation_critical_threshold else "warning"
                    alerts.append(CorrelationAlert(
                        timestamp=now,
                        asset_pair=(asset1, asset2),
                        historical_correlation=historical_corr,
                        current_correlation=current_corr,
                        change_magnitude=change,
                        alert_type="spike",
                        severity=severity,
                    ))

                # Check for breakdown (correlation moving toward -1 or changing sign)
                elif change > self.config.correlation_spike_threshold:
                    sign_change = (current_corr * historical_corr) < 0
                    alert_type = "regime_change" if sign_change else "breakdown"
                    severity = "critical" if change > self.config.correlation_critical_threshold else "warning"
                    alerts.append(CorrelationAlert(
                        timestamp=now,
                        asset_pair=(asset1, asset2),
                        historical_correlation=historical_corr,
                        current_correlation=current_corr,
                        change_magnitude=change,
                        alert_type=alert_type,
                        severity=severity,
                    ))

        return alerts

    def _update_state(
        self,
        current_corr: pd.DataFrame,
        alerts: List[CorrelationAlert]
    ):
        """Update circuit breaker state based on alerts."""
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "warning"]

        # Calculate average correlation
        upper_triangle = current_corr.values[np.triu_indices_from(current_corr.values, k=1)]
        avg_correlation = np.mean(np.abs(upper_triangle)) if len(upper_triangle) > 0 else 0

        # Determine state
        if len(critical_alerts) >= 2 or avg_correlation > self.config.average_correlation_limit:
            if self._state != CircuitBreakerState.TRIGGERED:
                self._trigger(
                    f"Critical alerts: {len(critical_alerts)}, Avg correlation: {avg_correlation:.2f}"
                )
        elif len(critical_alerts) >= 1 or len(warning_alerts) >= 3:
            if self._state == CircuitBreakerState.NORMAL:
                self._warn(f"Alerts: {len(critical_alerts)} critical, {len(warning_alerts)} warning")
        elif self._state == CircuitBreakerState.WARNING and not alerts:
            self._state = CircuitBreakerState.NORMAL
            logger.info("Correlation circuit breaker returned to NORMAL")

    def _trigger(self, reason: str):
        """Trigger the circuit breaker."""
        self._state = CircuitBreakerState.TRIGGERED
        self._triggered_at = datetime.now()
        self._trigger_reason = reason
        self._cooldown_until = datetime.now() + timedelta(minutes=self.config.cooldown_minutes)
        logger.warning(f"Correlation circuit breaker TRIGGERED: {reason}")

    def _warn(self, reason: str):
        """Set warning state."""
        self._state = CircuitBreakerState.WARNING
        self._trigger_reason = reason
        logger.warning(f"Correlation circuit breaker WARNING: {reason}")

    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = CircuitBreakerState.NORMAL
        self._triggered_at = None
        self._cooldown_until = None
        self._trigger_reason = None
        self._alerts = []
        logger.info("Correlation circuit breaker RESET")

    def add_listener(self, callback: Callable[[CircuitBreakerStatus], None]):
        """Add a listener for state changes."""
        self._listeners.append(callback)

    def get_status(self) -> CircuitBreakerStatus:
        """Get current status."""
        return CircuitBreakerStatus(
            state=self.state,
            triggered_at=self._triggered_at,
            trigger_reason=self._trigger_reason,
            cooldown_until=self._cooldown_until,
            alerts=self._alerts[-20:],  # Last 20 alerts
            risk_reduction_factor=self.risk_reduction_factor,
        )


@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing."""
    # Base parameters
    base_risk_per_trade: float = 0.02  # 2% of capital per trade
    max_position_size: float = 0.20  # 20% max single position

    # Volatility adjustments
    target_volatility: float = 0.15  # 15% annualized
    vol_lookback_days: int = 20
    vol_scaling_enabled: bool = True

    # Regime adjustments
    regime_scaling_enabled: bool = True
    bull_regime_multiplier: float = 1.2
    bear_regime_multiplier: float = 0.8
    crash_regime_multiplier: float = 0.3
    sideways_regime_multiplier: float = 1.0

    # Kelly criterion
    kelly_enabled: bool = True
    kelly_fraction: float = 0.25  # Use 25% of full Kelly

    # Drawdown adjustments
    drawdown_scaling_enabled: bool = True
    drawdown_threshold: float = 0.10  # Start reducing at 10% drawdown
    max_drawdown_multiplier: float = 0.5  # At max drawdown, use 50% size


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    base_size: float
    volatility_adjusted_size: float
    regime_adjusted_size: float
    drawdown_adjusted_size: float
    kelly_suggested_size: float
    final_size: float
    adjustments_applied: Dict[str, float]
    risk_metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "base_size": round(self.base_size, 4),
            "volatility_adjusted_size": round(self.volatility_adjusted_size, 4),
            "regime_adjusted_size": round(self.regime_adjusted_size, 4),
            "drawdown_adjusted_size": round(self.drawdown_adjusted_size, 4),
            "kelly_suggested_size": round(self.kelly_suggested_size, 4),
            "final_size": round(self.final_size, 4),
            "adjustments_applied": {k: round(v, 4) for k, v in self.adjustments_applied.items()},
            "risk_metrics": {
                k: round(v, 4) if isinstance(v, (int, float)) else v
                for k, v in self.risk_metrics.items()
            },
        }


class DynamicPositionSizer:
    """
    Dynamic Position Sizing Based on Multiple Factors.

    Adjusts position sizes based on:
    - Current volatility vs target volatility
    - Market regime (bull, bear, crash, sideways)
    - Current drawdown level
    - Kelly criterion (optional)
    - Circuit breaker status
    """

    def __init__(
        self,
        config: Optional[PositionSizingConfig] = None,
        circuit_breaker: Optional[CorrelationCircuitBreaker] = None
    ):
        self.config = config or PositionSizingConfig()
        self.circuit_breaker = circuit_breaker
        self._portfolio_value: float = 0
        self._current_drawdown: float = 0
        self._peak_value: float = 0

    def set_portfolio_value(self, value: float):
        """Update portfolio value for sizing calculations."""
        self._portfolio_value = value
        self._peak_value = max(self._peak_value, value)
        if self._peak_value > 0:
            self._current_drawdown = (self._peak_value - value) / self._peak_value

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        current_volatility: float,
        regime: str = "unknown",
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            current_volatility: Current asset volatility (annualized)
            regime: Market regime (bull, bear, crash, sideways)
            win_rate: Historical win rate for Kelly
            avg_win_loss_ratio: Average win/loss ratio for Kelly

        Returns:
            PositionSizeResult with sizing details
        """
        adjustments = {}
        risk_metrics = {}

        if self._portfolio_value <= 0:
            return self._empty_result()

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02  # Default 2% stop

        # Base position size (risk-based)
        risk_amount = self._portfolio_value * self.config.base_risk_per_trade
        base_shares = risk_amount / risk_per_share
        base_size = base_shares * entry_price / self._portfolio_value

        # 1. Volatility adjustment
        vol_adjusted_size = base_size
        if self.config.vol_scaling_enabled and current_volatility > 0:
            vol_scalar = self.config.target_volatility / current_volatility
            vol_scalar = max(0.5, min(2.0, vol_scalar))  # Clamp
            vol_adjusted_size = base_size * vol_scalar
            adjustments["volatility"] = vol_scalar
            risk_metrics["current_vol"] = current_volatility

        # 2. Regime adjustment
        regime_adjusted_size = vol_adjusted_size
        if self.config.regime_scaling_enabled:
            regime_multiplier = self._get_regime_multiplier(regime)
            regime_adjusted_size = vol_adjusted_size * regime_multiplier
            adjustments["regime"] = regime_multiplier
            risk_metrics["regime"] = regime

        # 3. Drawdown adjustment
        dd_adjusted_size = regime_adjusted_size
        if self.config.drawdown_scaling_enabled:
            dd_scalar = self._get_drawdown_scalar()
            dd_adjusted_size = regime_adjusted_size * dd_scalar
            adjustments["drawdown"] = dd_scalar
            risk_metrics["current_drawdown"] = self._current_drawdown

        # 4. Kelly criterion
        kelly_size = base_size
        if self.config.kelly_enabled:
            kelly_size = self._calculate_kelly_size(win_rate, avg_win_loss_ratio)
            risk_metrics["kelly_fraction"] = kelly_size

        # 5. Circuit breaker adjustment
        cb_scalar = 1.0
        if self.circuit_breaker:
            cb_scalar = self.circuit_breaker.risk_reduction_factor
            adjustments["circuit_breaker"] = cb_scalar

        # Final size (conservative: use minimum of methods)
        final_size = min(
            dd_adjusted_size * cb_scalar,
            kelly_size * self.config.kelly_fraction,
            self.config.max_position_size
        )

        # Ensure non-negative
        final_size = max(0, final_size)

        return PositionSizeResult(
            base_size=base_size,
            volatility_adjusted_size=vol_adjusted_size,
            regime_adjusted_size=regime_adjusted_size,
            drawdown_adjusted_size=dd_adjusted_size,
            kelly_suggested_size=kelly_size,
            final_size=final_size,
            adjustments_applied=adjustments,
            risk_metrics=risk_metrics,
        )

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier based on regime."""
        regime_lower = regime.lower()

        if "bull" in regime_lower or "strong_bull" in regime_lower:
            return self.config.bull_regime_multiplier
        elif "crash" in regime_lower:
            return self.config.crash_regime_multiplier
        elif "bear" in regime_lower:
            return self.config.bear_regime_multiplier
        elif "sideways" in regime_lower:
            return self.config.sideways_regime_multiplier
        else:
            return 1.0

    def _get_drawdown_scalar(self) -> float:
        """Get position size scalar based on current drawdown."""
        if self._current_drawdown <= self.config.drawdown_threshold:
            return 1.0

        # Linear scaling from threshold to max
        max_dd = 0.25  # Assume 25% max acceptable drawdown
        if self._current_drawdown >= max_dd:
            return self.config.max_drawdown_multiplier

        # Interpolate
        dd_range = max_dd - self.config.drawdown_threshold
        dd_progress = (self._current_drawdown - self.config.drawdown_threshold) / dd_range
        scalar = 1.0 - (1.0 - self.config.max_drawdown_multiplier) * dd_progress

        return scalar

    def _calculate_kelly_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly criterion position size."""
        # Kelly = (p * b - q) / b
        # p = win probability, q = loss probability, b = win/loss ratio
        p = max(0.01, min(0.99, win_rate))
        q = 1 - p
        b = max(0.01, win_loss_ratio)

        kelly = (p * b - q) / b
        kelly = max(0, kelly)  # Can't be negative

        return kelly

    def _empty_result(self) -> PositionSizeResult:
        """Return empty result when portfolio value is 0."""
        return PositionSizeResult(
            base_size=0,
            volatility_adjusted_size=0,
            regime_adjusted_size=0,
            drawdown_adjusted_size=0,
            kelly_suggested_size=0,
            final_size=0,
            adjustments_applied={},
            risk_metrics={"error": "No portfolio value set"},
        )

    def get_portfolio_risk_level(self) -> RiskLevel:
        """Get current portfolio risk level."""
        if self._current_drawdown > 0.20:
            return RiskLevel.CRITICAL
        elif self._current_drawdown > 0.10:
            return RiskLevel.HIGH
        elif self._current_drawdown > 0.05:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


def create_correlation_circuit_breaker(
    config: Optional[CorrelationConfig] = None
) -> CorrelationCircuitBreaker:
    """Factory function to create correlation circuit breaker."""
    return CorrelationCircuitBreaker(config=config)


def create_dynamic_position_sizer(
    config: Optional[PositionSizingConfig] = None,
    circuit_breaker: Optional[CorrelationCircuitBreaker] = None
) -> DynamicPositionSizer:
    """Factory function to create dynamic position sizer."""
    return DynamicPositionSizer(config=config, circuit_breaker=circuit_breaker)
