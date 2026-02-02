"""
Capital Preservation Mode Module.

Automatically activates protective measures when trading performance deteriorates.
Monitors multiple metrics to detect degradation and implements graduated response.

Entry Triggers:
- Rolling edge collapse (profit factor falls below threshold)
- Slippage error accumulation
- Regime confidence collapse
- Drawdown slope acceleration

When Active:
- Reduces maximum leverage
- Restricts to high-confidence trades only
- Increases cooldown between trades
- Reduces position sizing
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PreservationLevel(Enum):
    """Graduated preservation levels."""

    NORMAL = "normal"  # Full trading capability
    CAUTIOUS = "cautious"  # Mild restrictions
    DEFENSIVE = "defensive"  # Significant restrictions
    CRITICAL = "critical"  # Maximum restrictions
    LOCKDOWN = "lockdown"  # No new trades allowed


@dataclass
class PreservationConfig:
    """Configuration for capital preservation triggers and responses."""

    # Rolling window for metrics
    rolling_window_trades: int = 20
    rolling_window_hours: int = 24

    # Edge collapse thresholds (profit factor)
    edge_warning_threshold: float = 1.0  # Break-even
    edge_caution_threshold: float = 0.85  # 15% underwater
    edge_critical_threshold: float = 0.7  # 30% underwater

    # Slippage error thresholds
    slippage_error_warning: float = 0.02  # 2% average slippage error
    slippage_error_critical: float = 0.05  # 5% average slippage error
    max_slippage_errors_streak: int = 3

    # Regime confidence thresholds
    regime_confidence_warning: float = 0.5
    regime_confidence_critical: float = 0.3

    # Drawdown thresholds
    drawdown_warning_pct: float = 0.03  # 3% drawdown
    drawdown_critical_pct: float = 0.05  # 5% drawdown
    drawdown_lockdown_pct: float = 0.08  # 8% triggers lockdown

    # Drawdown acceleration (rate of change)
    drawdown_slope_warning: float = 0.01  # 1% per hour
    drawdown_slope_critical: float = 0.02  # 2% per hour

    # Win rate thresholds
    win_rate_warning: float = 0.45  # Below 45%
    win_rate_critical: float = 0.35  # Below 35%

    # Response parameters by level
    leverage_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "normal": 1.0,
            "cautious": 0.75,
            "defensive": 0.5,
            "critical": 0.25,
            "lockdown": 0.0,
        }
    )

    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "normal": 0.5,  # Accept 50%+ confidence
            "cautious": 0.6,
            "defensive": 0.7,
            "critical": 0.85,
            "lockdown": 1.1,  # Effectively blocks all
        }
    )

    cooldown_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "normal": 1.0,
            "cautious": 1.5,
            "defensive": 2.0,
            "critical": 3.0,
            "lockdown": float("inf"),
        }
    )

    position_size_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "normal": 1.0,
            "cautious": 0.8,
            "defensive": 0.5,
            "critical": 0.25,
            "lockdown": 0.0,
        }
    )

    # Recovery thresholds
    recovery_trades_required: int = 5
    recovery_win_rate_threshold: float = 0.6
    recovery_profit_factor_threshold: float = 1.2

    # State persistence
    state_path: Path = field(default_factory=lambda: Path("data/capital_preservation_state.json"))


@dataclass
class TradeMetrics:
    """Metrics from a single trade for rolling analysis."""

    timestamp: datetime
    pnl: float
    slippage_error: float  # Expected vs actual execution
    regime_confidence: float
    signal_confidence: float
    is_win: bool
    drawdown_at_entry: float


@dataclass
class PreservationState:
    """Current state of capital preservation mode."""

    level: PreservationLevel
    activated_at: Optional[datetime]
    trigger_reasons: List[str]

    # Rolling metrics
    rolling_profit_factor: float
    rolling_win_rate: float
    rolling_slippage_error: float
    rolling_regime_confidence: float

    # Drawdown tracking
    peak_equity: float
    current_equity: float
    current_drawdown_pct: float
    drawdown_slope: float  # Rate per hour

    # Recovery tracking
    recovery_trades: int
    recovery_wins: int

    # Restrictions in effect
    max_leverage_multiplier: float
    min_confidence_required: float
    cooldown_multiplier: float
    position_size_multiplier: float


class CapitalPreservationMode:
    """
    Capital preservation system that monitors trading performance
    and automatically activates protective measures.

    Thread-safe singleton pattern for global access.
    """

    _instance: Optional["CapitalPreservationMode"] = None
    _lock = RLock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: Optional[PreservationConfig] = None,
        initial_equity: float = 10000.0,
    ):
        if self._initialized:
            return

        self._config = config or PreservationConfig()
        self._lock = RLock()

        # Trade history for rolling calculations
        self._trade_history: Deque[TradeMetrics] = deque(
            maxlen=self._config.rolling_window_trades * 2
        )

        # Drawdown tracking with timestamps
        self._equity_history: Deque[Tuple[datetime, float]] = deque(maxlen=1000)

        # Current state
        self._state = PreservationState(
            level=PreservationLevel.NORMAL,
            activated_at=None,
            trigger_reasons=[],
            rolling_profit_factor=1.0,
            rolling_win_rate=0.5,
            rolling_slippage_error=0.0,
            rolling_regime_confidence=0.7,
            peak_equity=initial_equity,
            current_equity=initial_equity,
            current_drawdown_pct=0.0,
            drawdown_slope=0.0,
            recovery_trades=0,
            recovery_wins=0,
            max_leverage_multiplier=1.0,
            min_confidence_required=0.5,
            cooldown_multiplier=1.0,
            position_size_multiplier=1.0,
        )

        # Initialize equity tracking
        self._equity_history.append((datetime.now(), initial_equity))

        self._load_state()
        self._initialized = True

        logger.info(
            f"CapitalPreservationMode initialized at level {self._state.level.value}"
        )

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self._config.state_path.exists():
            try:
                with open(self._config.state_path, "r") as f:
                    data = json.load(f)

                self._state.level = PreservationLevel(data.get("level", "normal"))
                if data.get("activated_at"):
                    self._state.activated_at = datetime.fromisoformat(
                        data["activated_at"]
                    )
                self._state.trigger_reasons = data.get("trigger_reasons", [])
                self._state.peak_equity = data.get("peak_equity", self._state.peak_equity)
                self._state.current_equity = data.get(
                    "current_equity", self._state.current_equity
                )
                self._state.recovery_trades = data.get("recovery_trades", 0)
                self._state.recovery_wins = data.get("recovery_wins", 0)

                self._apply_level_restrictions()
                logger.info(f"Loaded preservation state: level={self._state.level.value}")

            except Exception as e:
                logger.warning(f"Failed to load preservation state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self._config.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config.state_path, "w") as f:
                json.dump(
                    {
                        "level": self._state.level.value,
                        "activated_at": (
                            self._state.activated_at.isoformat()
                            if self._state.activated_at
                            else None
                        ),
                        "trigger_reasons": self._state.trigger_reasons,
                        "peak_equity": self._state.peak_equity,
                        "current_equity": self._state.current_equity,
                        "recovery_trades": self._state.recovery_trades,
                        "recovery_wins": self._state.recovery_wins,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save preservation state: {e}")

    def update_equity(self, equity: float) -> None:
        """Update current equity for drawdown tracking."""
        with self._lock:
            now = datetime.now()
            self._state.current_equity = equity

            # Track peak
            if equity > self._state.peak_equity:
                self._state.peak_equity = equity

            # Add to history
            self._equity_history.append((now, equity))

            # Calculate current drawdown
            if self._state.peak_equity > 0:
                self._state.current_drawdown_pct = (
                    self._state.peak_equity - equity
                ) / self._state.peak_equity

            # Calculate drawdown slope (rate of change per hour)
            self._calculate_drawdown_slope()

            # Check if drawdown alone triggers level change
            self._evaluate_all_triggers()

    def _calculate_drawdown_slope(self) -> None:
        """Calculate rate of drawdown change."""
        if len(self._equity_history) < 2:
            self._state.drawdown_slope = 0.0
            return

        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Get equity values within last hour
        recent = [(ts, eq) for ts, eq in self._equity_history if ts >= cutoff]

        if len(recent) < 2:
            self._state.drawdown_slope = 0.0
            return

        # Calculate slope: change in drawdown % per hour
        first_ts, first_eq = recent[0]
        last_ts, last_eq = recent[-1]

        time_delta_hours = (last_ts - first_ts).total_seconds() / 3600
        if time_delta_hours < 0.01:  # Less than ~30 seconds
            self._state.drawdown_slope = 0.0
            return

        first_dd = (self._state.peak_equity - first_eq) / self._state.peak_equity
        last_dd = (self._state.peak_equity - last_eq) / self._state.peak_equity

        # Positive slope = drawdown increasing (bad)
        self._state.drawdown_slope = (last_dd - first_dd) / time_delta_hours

    def record_trade(
        self,
        pnl: float,
        expected_price: float,
        actual_price: float,
        regime_confidence: float,
        signal_confidence: float,
    ) -> None:
        """Record a completed trade for rolling analysis."""
        with self._lock:
            # Calculate slippage error
            if expected_price > 0:
                slippage_error = abs(actual_price - expected_price) / expected_price
            else:
                slippage_error = 0.0

            # Create metrics record
            metrics = TradeMetrics(
                timestamp=datetime.now(),
                pnl=pnl,
                slippage_error=slippage_error,
                regime_confidence=regime_confidence,
                signal_confidence=signal_confidence,
                is_win=pnl > 0,
                drawdown_at_entry=self._state.current_drawdown_pct,
            )

            self._trade_history.append(metrics)

            # Update rolling metrics
            self._update_rolling_metrics()

            # Track recovery progress if in preservation mode
            if self._state.level != PreservationLevel.NORMAL:
                self._state.recovery_trades += 1
                if pnl > 0:
                    self._state.recovery_wins += 1
                self._check_recovery()

            # Evaluate triggers
            self._evaluate_all_triggers()

            self._save_state()

    def _update_rolling_metrics(self) -> None:
        """Update rolling window metrics."""
        if not self._trade_history:
            return

        # Get recent trades within window
        now = datetime.now()
        cutoff = now - timedelta(hours=self._config.rolling_window_hours)

        recent_trades = [
            t
            for t in self._trade_history
            if t.timestamp >= cutoff
        ]

        if not recent_trades:
            return

        # Limit to configured window size
        recent_trades = recent_trades[-self._config.rolling_window_trades :]

        # Calculate profit factor
        gross_profit = sum(t.pnl for t in recent_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in recent_trades if t.pnl < 0))

        if gross_loss > 0:
            self._state.rolling_profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            self._state.rolling_profit_factor = 2.0  # Positive edge, no losses
        else:
            self._state.rolling_profit_factor = 1.0

        # Calculate win rate
        wins = sum(1 for t in recent_trades if t.is_win)
        self._state.rolling_win_rate = wins / len(recent_trades)

        # Calculate average slippage error
        self._state.rolling_slippage_error = sum(
            t.slippage_error for t in recent_trades
        ) / len(recent_trades)

        # Calculate average regime confidence
        self._state.rolling_regime_confidence = sum(
            t.regime_confidence for t in recent_trades
        ) / len(recent_trades)

    def _evaluate_all_triggers(self) -> None:
        """Evaluate all triggers and determine appropriate level."""
        triggers = []
        max_level = PreservationLevel.NORMAL

        # 1. Edge collapse (profit factor)
        if self._state.rolling_profit_factor < self._config.edge_critical_threshold:
            triggers.append(
                f"Edge collapse: PF={self._state.rolling_profit_factor:.2f} < {self._config.edge_critical_threshold}"
            )
            max_level = max(max_level, PreservationLevel.CRITICAL, key=lambda x: x.value)
        elif self._state.rolling_profit_factor < self._config.edge_caution_threshold:
            triggers.append(
                f"Edge deterioration: PF={self._state.rolling_profit_factor:.2f}"
            )
            max_level = max(max_level, PreservationLevel.DEFENSIVE, key=lambda x: x.value)
        elif self._state.rolling_profit_factor < self._config.edge_warning_threshold:
            triggers.append(f"Edge warning: PF={self._state.rolling_profit_factor:.2f}")
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # 2. Slippage errors
        if self._state.rolling_slippage_error > self._config.slippage_error_critical:
            triggers.append(
                f"Slippage critical: {self._state.rolling_slippage_error:.1%}"
            )
            max_level = max(max_level, PreservationLevel.DEFENSIVE, key=lambda x: x.value)
        elif self._state.rolling_slippage_error > self._config.slippage_error_warning:
            triggers.append(
                f"Slippage elevated: {self._state.rolling_slippage_error:.1%}"
            )
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # 3. Regime confidence collapse
        if self._state.rolling_regime_confidence < self._config.regime_confidence_critical:
            triggers.append(
                f"Regime confidence collapse: {self._state.rolling_regime_confidence:.1%}"
            )
            max_level = max(max_level, PreservationLevel.CRITICAL, key=lambda x: x.value)
        elif self._state.rolling_regime_confidence < self._config.regime_confidence_warning:
            triggers.append(
                f"Regime uncertainty: {self._state.rolling_regime_confidence:.1%}"
            )
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # 4. Drawdown level
        if self._state.current_drawdown_pct >= self._config.drawdown_lockdown_pct:
            triggers.append(
                f"Drawdown lockdown: {self._state.current_drawdown_pct:.1%}"
            )
            max_level = PreservationLevel.LOCKDOWN
        elif self._state.current_drawdown_pct >= self._config.drawdown_critical_pct:
            triggers.append(
                f"Drawdown critical: {self._state.current_drawdown_pct:.1%}"
            )
            max_level = max(max_level, PreservationLevel.CRITICAL, key=lambda x: x.value)
        elif self._state.current_drawdown_pct >= self._config.drawdown_warning_pct:
            triggers.append(
                f"Drawdown warning: {self._state.current_drawdown_pct:.1%}"
            )
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # 5. Drawdown acceleration
        if self._state.drawdown_slope > self._config.drawdown_slope_critical:
            triggers.append(
                f"Drawdown accelerating: {self._state.drawdown_slope:.2%}/hr"
            )
            max_level = max(max_level, PreservationLevel.DEFENSIVE, key=lambda x: x.value)
        elif self._state.drawdown_slope > self._config.drawdown_slope_warning:
            triggers.append(
                f"Drawdown slope warning: {self._state.drawdown_slope:.2%}/hr"
            )
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # 6. Win rate
        if self._state.rolling_win_rate < self._config.win_rate_critical:
            triggers.append(f"Win rate critical: {self._state.rolling_win_rate:.1%}")
            max_level = max(max_level, PreservationLevel.DEFENSIVE, key=lambda x: x.value)
        elif self._state.rolling_win_rate < self._config.win_rate_warning:
            triggers.append(f"Win rate warning: {self._state.rolling_win_rate:.1%}")
            max_level = max(max_level, PreservationLevel.CAUTIOUS, key=lambda x: x.value)

        # Update level if it increased (never automatically decrease)
        level_order = [
            PreservationLevel.NORMAL,
            PreservationLevel.CAUTIOUS,
            PreservationLevel.DEFENSIVE,
            PreservationLevel.CRITICAL,
            PreservationLevel.LOCKDOWN,
        ]

        current_idx = level_order.index(self._state.level)
        new_idx = level_order.index(max_level)

        if new_idx > current_idx:
            old_level = self._state.level
            self._state.level = max_level
            self._state.activated_at = datetime.now()
            self._state.trigger_reasons = triggers
            self._state.recovery_trades = 0
            self._state.recovery_wins = 0

            self._apply_level_restrictions()

            logger.warning(
                f"Capital preservation escalated: {old_level.value} -> {max_level.value}. "
                f"Triggers: {triggers}"
            )

            self._save_state()

    def _apply_level_restrictions(self) -> None:
        """Apply restrictions based on current level."""
        level = self._state.level.value

        self._state.max_leverage_multiplier = self._config.leverage_multipliers.get(
            level, 1.0
        )
        self._state.min_confidence_required = self._config.confidence_thresholds.get(
            level, 0.5
        )
        self._state.cooldown_multiplier = self._config.cooldown_multipliers.get(
            level, 1.0
        )
        self._state.position_size_multiplier = self._config.position_size_multipliers.get(
            level, 1.0
        )

    def _check_recovery(self) -> None:
        """Check if conditions allow level reduction."""
        if self._state.level == PreservationLevel.NORMAL:
            return

        if self._state.recovery_trades < self._config.recovery_trades_required:
            return

        # Calculate recovery metrics
        if self._state.recovery_trades > 0:
            recovery_win_rate = (
                self._state.recovery_wins / self._state.recovery_trades
            )
        else:
            recovery_win_rate = 0.0

        # Check recovery conditions
        can_recover = (
            recovery_win_rate >= self._config.recovery_win_rate_threshold
            and self._state.rolling_profit_factor
            >= self._config.recovery_profit_factor_threshold
            and self._state.current_drawdown_pct < self._config.drawdown_warning_pct
        )

        if can_recover:
            level_order = [
                PreservationLevel.NORMAL,
                PreservationLevel.CAUTIOUS,
                PreservationLevel.DEFENSIVE,
                PreservationLevel.CRITICAL,
                PreservationLevel.LOCKDOWN,
            ]

            current_idx = level_order.index(self._state.level)
            if current_idx > 0:
                old_level = self._state.level
                self._state.level = level_order[current_idx - 1]
                self._state.recovery_trades = 0
                self._state.recovery_wins = 0

                self._apply_level_restrictions()

                logger.info(
                    f"Capital preservation de-escalated: {old_level.value} -> "
                    f"{self._state.level.value} (recovery WR={recovery_win_rate:.1%})"
                )

                self._save_state()

    def can_trade(self, signal_confidence: float) -> Tuple[bool, str]:
        """
        Check if a trade is allowed given current preservation state.

        Returns (allowed, reason).
        """
        with self._lock:
            if self._state.level == PreservationLevel.LOCKDOWN:
                return False, "Lockdown mode active - no new trades"

            if signal_confidence < self._state.min_confidence_required:
                return (
                    False,
                    f"Confidence {signal_confidence:.1%} below required "
                    f"{self._state.min_confidence_required:.1%} ({self._state.level.value} mode)",
                )

            return True, "Trade allowed"

    def get_restrictions(self) -> Dict[str, float]:
        """Get current trading restrictions."""
        with self._lock:
            return {
                "level": self._state.level.value,
                "leverage_multiplier": self._state.max_leverage_multiplier,
                "min_confidence_required": self._state.min_confidence_required,
                "cooldown_multiplier": self._state.cooldown_multiplier,
                "position_size_multiplier": self._state.position_size_multiplier,
            }

    def adjust_leverage(self, requested_leverage: float) -> float:
        """Adjust leverage based on preservation level."""
        with self._lock:
            return requested_leverage * self._state.max_leverage_multiplier

    def adjust_position_size(self, requested_size: float) -> float:
        """Adjust position size based on preservation level."""
        with self._lock:
            return requested_size * self._state.position_size_multiplier

    def adjust_cooldown(self, base_cooldown_seconds: int) -> int:
        """Adjust cooldown period based on preservation level."""
        with self._lock:
            return int(base_cooldown_seconds * self._state.cooldown_multiplier)

    def force_level(self, level: PreservationLevel, reason: str = "manual") -> None:
        """Manually set preservation level (for testing or emergency)."""
        with self._lock:
            old_level = self._state.level
            self._state.level = level
            self._state.activated_at = datetime.now()
            self._state.trigger_reasons = [f"Manual: {reason}"]
            self._state.recovery_trades = 0
            self._state.recovery_wins = 0

            self._apply_level_restrictions()

            logger.warning(
                f"Capital preservation manually set: {old_level.value} -> {level.value}. "
                f"Reason: {reason}"
            )

            self._save_state()

    def reset(self) -> None:
        """Reset to normal level (requires explicit call)."""
        with self._lock:
            old_level = self._state.level
            self._state.level = PreservationLevel.NORMAL
            self._state.activated_at = None
            self._state.trigger_reasons = []
            self._state.recovery_trades = 0
            self._state.recovery_wins = 0
            self._state.peak_equity = self._state.current_equity

            self._apply_level_restrictions()

            logger.info(f"Capital preservation reset from {old_level.value} to NORMAL")

            self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive preservation status."""
        with self._lock:
            return {
                "level": self._state.level.value,
                "activated_at": (
                    self._state.activated_at.isoformat()
                    if self._state.activated_at
                    else None
                ),
                "trigger_reasons": self._state.trigger_reasons,
                "rolling_metrics": {
                    "profit_factor": self._state.rolling_profit_factor,
                    "win_rate": self._state.rolling_win_rate,
                    "slippage_error": self._state.rolling_slippage_error,
                    "regime_confidence": self._state.rolling_regime_confidence,
                },
                "drawdown": {
                    "current_pct": self._state.current_drawdown_pct,
                    "peak_equity": self._state.peak_equity,
                    "current_equity": self._state.current_equity,
                    "slope_per_hour": self._state.drawdown_slope,
                },
                "restrictions": {
                    "leverage_multiplier": self._state.max_leverage_multiplier,
                    "min_confidence_required": self._state.min_confidence_required,
                    "cooldown_multiplier": self._state.cooldown_multiplier,
                    "position_size_multiplier": self._state.position_size_multiplier,
                },
                "recovery": {
                    "trades_since_escalation": self._state.recovery_trades,
                    "wins_since_escalation": self._state.recovery_wins,
                    "win_rate": (
                        self._state.recovery_wins / self._state.recovery_trades
                        if self._state.recovery_trades > 0
                        else 0.0
                    ),
                    "trades_needed": max(
                        0,
                        self._config.recovery_trades_required
                        - self._state.recovery_trades,
                    ),
                },
                "thresholds": {
                    "edge_warning": self._config.edge_warning_threshold,
                    "drawdown_warning": self._config.drawdown_warning_pct,
                    "drawdown_critical": self._config.drawdown_critical_pct,
                    "drawdown_lockdown": self._config.drawdown_lockdown_pct,
                },
            }


# Singleton accessor
_instance: Optional[CapitalPreservationMode] = None


def get_capital_preservation(
    config: Optional[PreservationConfig] = None, initial_equity: float = 10000.0
) -> CapitalPreservationMode:
    """Get or create the singleton CapitalPreservationMode instance."""
    global _instance
    if _instance is None:
        _instance = CapitalPreservationMode(config=config, initial_equity=initial_equity)
    return _instance


def reset_capital_preservation() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    if _instance is not None:
        _instance._initialized = False
    _instance = None
    CapitalPreservationMode._instance = None
