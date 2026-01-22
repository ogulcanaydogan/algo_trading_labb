"""
Drawdown Recovery Mode - Conservative trading after losses.

Automatically reduces risk exposure when drawdown is detected,
then gradually increases back to normal as performance recovers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """Current phase of drawdown recovery."""
    NORMAL = "normal"               # Full trading capacity
    CAUTION = "caution"             # Minor drawdown, slightly reduced
    RECOVERY = "recovery"           # Significant drawdown, reduced exposure
    CRITICAL = "critical"           # Severe drawdown, minimal trading
    HALTED = "halted"              # Trading suspended


@dataclass
class DrawdownState:
    """Current drawdown state."""
    peak_equity: float
    current_equity: float
    drawdown_pct: float
    drawdown_usd: float
    phase: RecoveryPhase
    days_in_drawdown: int
    position_size_multiplier: float
    max_positions_allowed: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "drawdown_usd": round(self.drawdown_usd, 2),
            "phase": self.phase.value,
            "days_in_drawdown": self.days_in_drawdown,
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "max_positions_allowed": self.max_positions_allowed,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RecoveryConfig:
    """Configuration for drawdown recovery."""
    # Drawdown thresholds
    caution_threshold: float = 0.05     # 5% drawdown -> caution
    recovery_threshold: float = 0.10    # 10% drawdown -> recovery mode
    critical_threshold: float = 0.15    # 15% drawdown -> critical
    halt_threshold: float = 0.20        # 20% drawdown -> halt trading

    # Position size multipliers per phase
    normal_multiplier: float = 1.0
    caution_multiplier: float = 0.75
    recovery_multiplier: float = 0.50
    critical_multiplier: float = 0.25

    # Max positions per phase
    normal_max_positions: int = 10
    caution_max_positions: int = 7
    recovery_max_positions: int = 4
    critical_max_positions: int = 2

    # Recovery requirements
    recovery_profit_streak: int = 3     # Consecutive profitable days to recover
    recovery_time_min_days: int = 2     # Minimum days before upgrading phase

    # Additional restrictions
    no_new_positions_after_loss: bool = True  # No new positions after daily loss
    reduce_on_consecutive_losses: int = 3     # Reduce after N consecutive losses


@dataclass
class DailyPerformance:
    """Daily performance tracking."""
    date: str
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_pct: float
    trades: int
    winners: int
    losers: int

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0


class DrawdownRecoveryManager:
    """
    Manage trading during drawdown periods.

    Features:
    - Track drawdown levels and duration
    - Automatically adjust position sizes
    - Limit number of positions
    - Gradual recovery as performance improves
    - Halt trading if drawdown exceeds limits
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        data_dir: str = "data/risk",
        initial_equity: float = 10000.0,
    ):
        self.config = config or RecoveryConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.phase = RecoveryPhase.NORMAL
        self.drawdown_start_date: Optional[datetime] = None
        self.daily_history: List[DailyPerformance] = []
        self.consecutive_losses = 0
        self.consecutive_profits = 0

        self._load_state()

    def _load_state(self):
        """Load saved state."""
        state_file = self.data_dir / "drawdown_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.peak_equity = state.get("peak_equity", self.peak_equity)
                    self.current_equity = state.get("current_equity", self.current_equity)
                    self.phase = RecoveryPhase(state.get("phase", "normal"))
                    if state.get("drawdown_start_date"):
                        self.drawdown_start_date = datetime.fromisoformat(state["drawdown_start_date"])
                    self.consecutive_losses = state.get("consecutive_losses", 0)
                    self.consecutive_profits = state.get("consecutive_profits", 0)
                logger.info(f"Loaded drawdown state: phase={self.phase.value}")
            except Exception as e:
                logger.error(f"Error loading drawdown state: {e}")

    def _save_state(self):
        """Save current state."""
        state_file = self.data_dir / "drawdown_state.json"
        try:
            state = {
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "phase": self.phase.value,
                "drawdown_start_date": self.drawdown_start_date.isoformat() if self.drawdown_start_date else None,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_profits": self.consecutive_profits,
                "updated_at": datetime.now().isoformat(),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving drawdown state: {e}")

    def update_equity(self, new_equity: float) -> DrawdownState:
        """
        Update equity and recalculate drawdown state.

        Args:
            new_equity: Current portfolio equity

        Returns:
            Updated DrawdownState
        """
        old_equity = self.current_equity
        self.current_equity = new_equity

        # Update peak equity if new high
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            self.drawdown_start_date = None

        # Calculate drawdown
        drawdown_usd = self.peak_equity - self.current_equity
        drawdown_pct = drawdown_usd / self.peak_equity if self.peak_equity > 0 else 0

        # Track consecutive wins/losses
        if new_equity > old_equity:
            self.consecutive_profits += 1
            self.consecutive_losses = 0
        elif new_equity < old_equity:
            self.consecutive_losses += 1
            self.consecutive_profits = 0

        # Calculate days in drawdown
        days_in_drawdown = 0
        if drawdown_pct > self.config.caution_threshold:
            if self.drawdown_start_date is None:
                self.drawdown_start_date = datetime.now()
            days_in_drawdown = (datetime.now() - self.drawdown_start_date).days

        # Determine phase
        old_phase = self.phase
        self.phase = self._determine_phase(drawdown_pct, days_in_drawdown)

        # Log phase transitions
        if self.phase != old_phase:
            logger.warning(
                f"Drawdown phase transition: {old_phase.value} -> {self.phase.value} "
                f"(drawdown: {drawdown_pct:.1%})"
            )

        self._save_state()

        return DrawdownState(
            peak_equity=self.peak_equity,
            current_equity=self.current_equity,
            drawdown_pct=drawdown_pct,
            drawdown_usd=drawdown_usd,
            phase=self.phase,
            days_in_drawdown=days_in_drawdown,
            position_size_multiplier=self.get_position_size_multiplier(),
            max_positions_allowed=self.get_max_positions(),
        )

    def _determine_phase(self, drawdown_pct: float, days_in_drawdown: int) -> RecoveryPhase:
        """Determine recovery phase based on drawdown."""
        # Check for halt condition
        if drawdown_pct >= self.config.halt_threshold:
            return RecoveryPhase.HALTED

        # Check for phase upgrade (recovery improvement)
        if self.phase != RecoveryPhase.NORMAL:
            if self._should_upgrade_phase(drawdown_pct):
                return self._get_upgraded_phase()

        # Check for phase downgrade (worsening)
        if drawdown_pct >= self.config.critical_threshold:
            return RecoveryPhase.CRITICAL
        elif drawdown_pct >= self.config.recovery_threshold:
            return RecoveryPhase.RECOVERY
        elif drawdown_pct >= self.config.caution_threshold:
            return RecoveryPhase.CAUTION
        else:
            return RecoveryPhase.NORMAL

    def _should_upgrade_phase(self, drawdown_pct: float) -> bool:
        """Check if conditions met to upgrade (improve) phase."""
        # Need consecutive profits
        if self.consecutive_profits < self.config.recovery_profit_streak:
            return False

        # Check drawdown improved to lower threshold
        if self.phase == RecoveryPhase.CRITICAL:
            return drawdown_pct < self.config.critical_threshold
        elif self.phase == RecoveryPhase.RECOVERY:
            return drawdown_pct < self.config.recovery_threshold
        elif self.phase == RecoveryPhase.CAUTION:
            return drawdown_pct < self.config.caution_threshold

        return False

    def _get_upgraded_phase(self) -> RecoveryPhase:
        """Get the next better phase."""
        upgrades = {
            RecoveryPhase.HALTED: RecoveryPhase.CRITICAL,
            RecoveryPhase.CRITICAL: RecoveryPhase.RECOVERY,
            RecoveryPhase.RECOVERY: RecoveryPhase.CAUTION,
            RecoveryPhase.CAUTION: RecoveryPhase.NORMAL,
        }
        return upgrades.get(self.phase, self.phase)

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on current phase."""
        multipliers = {
            RecoveryPhase.NORMAL: self.config.normal_multiplier,
            RecoveryPhase.CAUTION: self.config.caution_multiplier,
            RecoveryPhase.RECOVERY: self.config.recovery_multiplier,
            RecoveryPhase.CRITICAL: self.config.critical_multiplier,
            RecoveryPhase.HALTED: 0.0,
        }

        multiplier = multipliers.get(self.phase, 1.0)

        # Additional reduction for consecutive losses
        if self.consecutive_losses >= self.config.reduce_on_consecutive_losses:
            multiplier *= 0.5
            logger.info(f"Position size reduced due to {self.consecutive_losses} consecutive losses")

        return multiplier

    def get_max_positions(self) -> int:
        """Get maximum positions allowed based on current phase."""
        max_positions = {
            RecoveryPhase.NORMAL: self.config.normal_max_positions,
            RecoveryPhase.CAUTION: self.config.caution_max_positions,
            RecoveryPhase.RECOVERY: self.config.recovery_max_positions,
            RecoveryPhase.CRITICAL: self.config.critical_max_positions,
            RecoveryPhase.HALTED: 0,
        }
        return max_positions.get(self.phase, 0)

    def can_open_position(self, current_positions: int = 0) -> tuple[bool, str]:
        """
        Check if allowed to open a new position.

        Returns:
            Tuple of (allowed, reason)
        """
        if self.phase == RecoveryPhase.HALTED:
            return False, "Trading halted due to severe drawdown"

        max_positions = self.get_max_positions()
        if current_positions >= max_positions:
            return False, f"Max positions ({max_positions}) reached in {self.phase.value} phase"

        # Check if restricted after loss
        if self.config.no_new_positions_after_loss and self.consecutive_losses > 0:
            # Allow after some recovery
            if self.consecutive_profits < 1:
                return False, f"No new positions after {self.consecutive_losses} consecutive losses"

        return True, "OK"

    def adjust_position_size(self, base_size: float) -> float:
        """
        Adjust position size based on current drawdown state.

        Args:
            base_size: Original position size

        Returns:
            Adjusted position size
        """
        multiplier = self.get_position_size_multiplier()
        adjusted = base_size * multiplier

        if multiplier < 1.0:
            logger.debug(f"Position size adjusted: {base_size} -> {adjusted} (x{multiplier})")

        return adjusted

    def record_daily_performance(
        self,
        starting_equity: float,
        ending_equity: float,
        trades: int,
        winners: int,
        losers: int,
    ):
        """Record daily performance for tracking."""
        perf = DailyPerformance(
            date=datetime.now().strftime("%Y-%m-%d"),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            pnl=ending_equity - starting_equity,
            pnl_pct=(ending_equity - starting_equity) / starting_equity if starting_equity > 0 else 0,
            trades=trades,
            winners=winners,
            losers=losers,
        )
        self.daily_history.append(perf)

        # Keep last 90 days
        self.daily_history = self.daily_history[-90:]

    def get_status(self) -> Dict:
        """Get current drawdown recovery status."""
        drawdown_usd = self.peak_equity - self.current_equity
        drawdown_pct = drawdown_usd / self.peak_equity if self.peak_equity > 0 else 0

        return {
            "phase": self.phase.value,
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "drawdown_pct": round(drawdown_pct * 100, 2),
            "drawdown_usd": round(drawdown_usd, 2),
            "position_size_multiplier": self.get_position_size_multiplier(),
            "max_positions": self.get_max_positions(),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_profits": self.consecutive_profits,
            "can_trade": self.phase != RecoveryPhase.HALTED,
        }

    def get_recovery_progress(self) -> Dict:
        """Get progress towards full recovery."""
        if self.phase == RecoveryPhase.NORMAL:
            return {"status": "fully_recovered", "progress": 100}

        drawdown_usd = self.peak_equity - self.current_equity
        drawdown_pct = drawdown_usd / self.peak_equity if self.peak_equity > 0 else 0

        # Progress is how close we are to caution threshold (full recovery)
        if drawdown_pct > 0:
            recovery_needed = drawdown_pct - self.config.caution_threshold
            if recovery_needed > 0:
                total_recovery_range = self.config.halt_threshold - self.config.caution_threshold
                progress = (1 - recovery_needed / total_recovery_range) * 100
            else:
                progress = 90  # Close to full recovery
        else:
            progress = 100

        return {
            "status": "recovering",
            "phase": self.phase.value,
            "progress": round(max(0, min(100, progress)), 1),
            "profit_streak": self.consecutive_profits,
            "streak_needed": self.config.recovery_profit_streak,
            "amount_to_recover": round(drawdown_usd, 2),
        }


def create_drawdown_manager(
    config: Optional[RecoveryConfig] = None,
    data_dir: str = "data/risk",
    initial_equity: float = 10000.0,
) -> DrawdownRecoveryManager:
    """Factory function to create drawdown recovery manager."""
    return DrawdownRecoveryManager(
        config=config,
        data_dir=data_dir,
        initial_equity=initial_equity,
    )
