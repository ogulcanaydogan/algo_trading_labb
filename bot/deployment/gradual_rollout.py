"""
Gradual Rollout System.

Safe deployment from paper trading to live:
- Shadow mode (signals only)
- Micro live (1% capital)
- Limited live (10% capital)
- Full live (100% capital)

With automatic rollback triggers.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Deployment stages."""
    SHADOW = "shadow"  # Generate signals, don't execute
    MICRO = "micro"  # 1% capital, 1 asset
    LIMITED = "limited"  # 10% capital, 3 assets
    FULL = "full"  # 100% capital, all assets
    PAUSED = "paused"  # Rollback state
    DISABLED = "disabled"  # System disabled


@dataclass
class StageConfig:
    """Configuration for a rollout stage."""
    stage: RolloutStage
    capital_pct: float  # % of total capital to use
    max_assets: int  # Max number of assets to trade
    max_leverage: float  # Max allowed leverage
    max_position_pct: float  # Max single position size
    min_days_in_stage: int  # Minimum days before advancing
    min_trades: int  # Minimum trades before advancing

    # Advancement criteria
    min_win_rate: float  # Required win rate to advance
    min_sharpe: float  # Required Sharpe to advance
    max_daily_loss_pct: float  # Max daily loss before rollback


@dataclass
class RolloutMetrics:
    """Metrics tracked for rollout decisions."""
    stage: RolloutStage
    days_in_stage: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    total_pnl: float
    max_drawdown: float
    daily_losses: List[float]
    started_at: datetime
    last_trade: Optional[datetime] = None


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    from_stage: RolloutStage
    to_stage: RolloutStage
    reason: str
    metrics_at_rollback: Dict
    timestamp: datetime = field(default_factory=datetime.now)


class GradualRolloutSystem:
    """
    Manages gradual deployment from paper to live trading.

    Features:
    - Stage-based capital allocation
    - Automatic advancement
    - Automatic rollback on poor performance
    - Detailed tracking and logging
    """

    # Default stage configurations
    STAGE_CONFIGS = {
        RolloutStage.SHADOW: StageConfig(
            stage=RolloutStage.SHADOW,
            capital_pct=0.0,
            max_assets=99,
            max_leverage=1.0,
            max_position_pct=0.0,
            min_days_in_stage=7,
            min_trades=50,
            min_win_rate=0.50,
            min_sharpe=0.5,
            max_daily_loss_pct=0.0,  # No real loss in shadow
        ),
        RolloutStage.MICRO: StageConfig(
            stage=RolloutStage.MICRO,
            capital_pct=0.01,  # 1%
            max_assets=1,
            max_leverage=1.5,
            max_position_pct=0.01,
            min_days_in_stage=7,
            min_trades=20,
            min_win_rate=0.48,
            min_sharpe=0.3,
            max_daily_loss_pct=0.02,  # 2% of allocated capital
        ),
        RolloutStage.LIMITED: StageConfig(
            stage=RolloutStage.LIMITED,
            capital_pct=0.10,  # 10%
            max_assets=3,
            max_leverage=2.0,
            max_position_pct=0.05,
            min_days_in_stage=14,
            min_trades=50,
            min_win_rate=0.50,
            min_sharpe=0.5,
            max_daily_loss_pct=0.03,  # 3%
        ),
        RolloutStage.FULL: StageConfig(
            stage=RolloutStage.FULL,
            capital_pct=1.0,  # 100%
            max_assets=99,
            max_leverage=3.0,
            max_position_pct=0.10,
            min_days_in_stage=0,  # No advancement from full
            min_trades=0,
            min_win_rate=0.40,
            min_sharpe=0.0,
            max_daily_loss_pct=0.05,  # 5%
        ),
    }

    # Stage progression order
    STAGE_ORDER = [
        RolloutStage.SHADOW,
        RolloutStage.MICRO,
        RolloutStage.LIMITED,
        RolloutStage.FULL,
    ]

    def __init__(
        self,
        total_capital: float = 30000.0,
        state_file: Optional[str] = None,
        advancement_callback: Optional[Callable[[RolloutStage, RolloutStage], None]] = None,
        rollback_callback: Optional[Callable[[RollbackEvent], None]] = None,
    ):
        """
        Initialize gradual rollout system.

        Args:
            total_capital: Total available capital
            state_file: Path to persist state
            advancement_callback: Called when stage advances
            rollback_callback: Called when rollback occurs
        """
        self.total_capital = total_capital
        self.state_file = state_file
        self.advancement_callback = advancement_callback
        self.rollback_callback = rollback_callback

        # Current state
        self._current_stage = RolloutStage.SHADOW
        self._metrics = RolloutMetrics(
            stage=RolloutStage.SHADOW,
            days_in_stage=0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            daily_losses=[],
            started_at=datetime.now(),
        )

        # History
        self._rollback_history: List[RollbackEvent] = []
        self._stage_history: List[Dict] = []

        # Load state if exists
        if state_file and os.path.exists(state_file):
            self._load_state()

        logger.info(f"GradualRolloutSystem initialized at stage: {self._current_stage.value}")

    def get_current_stage(self) -> RolloutStage:
        """Get current rollout stage."""
        return self._current_stage

    def get_stage_config(self) -> StageConfig:
        """Get configuration for current stage."""
        return self.STAGE_CONFIGS[self._current_stage]

    def get_allocated_capital(self) -> float:
        """Get capital allocated for current stage."""
        config = self.get_stage_config()
        return self.total_capital * config.capital_pct

    def get_max_position_size(self) -> float:
        """Get maximum position size for current stage."""
        config = self.get_stage_config()
        return self.total_capital * config.max_position_pct

    def get_allowed_assets(self) -> int:
        """Get number of assets allowed in current stage."""
        return self.get_stage_config().max_assets

    def get_max_leverage(self) -> float:
        """Get maximum leverage for current stage."""
        return self.get_stage_config().max_leverage

    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled in current stage."""
        return self._current_stage not in [RolloutStage.SHADOW, RolloutStage.PAUSED, RolloutStage.DISABLED]

    def record_trade(
        self,
        pnl: float,
        is_win: bool,
        symbol: str,
    ):
        """
        Record a trade for metrics tracking.

        Args:
            pnl: Trade P&L
            is_win: Whether trade was profitable
            symbol: Trading symbol
        """
        self._metrics.total_trades += 1
        self._metrics.total_pnl += pnl
        self._metrics.last_trade = datetime.now()

        # Update win rate
        wins = self._metrics.win_rate * (self._metrics.total_trades - 1)
        if is_win:
            wins += 1
        self._metrics.win_rate = wins / self._metrics.total_trades

        # Track drawdown
        if self._metrics.total_pnl < 0:
            self._metrics.max_drawdown = max(
                self._metrics.max_drawdown,
                abs(self._metrics.total_pnl)
            )

        # Check for rollback triggers
        self._check_rollback_triggers()

        # Save state
        self._save_state()

        logger.debug(
            f"Trade recorded: PnL=${pnl:.2f}, Win rate={self._metrics.win_rate:.2%}, "
            f"Total trades={self._metrics.total_trades}"
        )

    def record_daily_result(self, daily_pnl: float):
        """
        Record daily P&L result.

        Args:
            daily_pnl: Day's total P&L
        """
        self._metrics.daily_losses.append(daily_pnl)
        if len(self._metrics.daily_losses) > 30:
            self._metrics.daily_losses = self._metrics.daily_losses[-30:]

        # Update days in stage
        self._metrics.days_in_stage = (datetime.now() - self._metrics.started_at).days

        # Calculate rolling Sharpe
        if len(self._metrics.daily_losses) >= 7:
            self._metrics.sharpe_ratio = self._calculate_sharpe(
                self._metrics.daily_losses[-20:]
            )

        # Check for daily loss trigger
        config = self.get_stage_config()
        allocated = self.get_allocated_capital()

        if allocated > 0 and daily_pnl < 0:
            loss_pct = abs(daily_pnl) / allocated
            if loss_pct > config.max_daily_loss_pct:
                self._trigger_rollback(f"Daily loss {loss_pct:.1%} exceeded limit {config.max_daily_loss_pct:.1%}")

        # Check for advancement
        self._check_advancement()

        # Save state
        self._save_state()

    def _calculate_sharpe(self, daily_returns: List[float]) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(daily_returns) < 2:
            return 0.0

        import statistics

        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.001

        if std_return == 0:
            return 0.0

        # Annualized Sharpe
        return (mean_return / std_return) * (252 ** 0.5)

    def _check_advancement(self):
        """Check if ready to advance to next stage."""
        if self._current_stage in [RolloutStage.FULL, RolloutStage.PAUSED, RolloutStage.DISABLED]:
            return

        config = self.get_stage_config()

        # Check minimum requirements
        if self._metrics.days_in_stage < config.min_days_in_stage:
            return
        if self._metrics.total_trades < config.min_trades:
            return
        if self._metrics.win_rate < config.min_win_rate:
            return
        if self._metrics.sharpe_ratio < config.min_sharpe:
            return

        # All criteria met - advance
        self._advance_stage()

    def _advance_stage(self):
        """Advance to next rollout stage."""
        current_idx = self.STAGE_ORDER.index(self._current_stage)
        if current_idx >= len(self.STAGE_ORDER) - 1:
            return

        old_stage = self._current_stage
        new_stage = self.STAGE_ORDER[current_idx + 1]

        # Record history
        self._stage_history.append({
            "from": old_stage.value,
            "to": new_stage.value,
            "metrics": self._get_metrics_dict(),
            "timestamp": datetime.now().isoformat(),
        })

        # Update state
        self._current_stage = new_stage
        self._metrics = RolloutMetrics(
            stage=new_stage,
            days_in_stage=0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            daily_losses=[],
            started_at=datetime.now(),
        )

        # Callback
        if self.advancement_callback:
            try:
                self.advancement_callback(old_stage, new_stage)
            except Exception as e:
                logger.error(f"Advancement callback failed: {e}")

        logger.info(f"Advanced from {old_stage.value} to {new_stage.value}")

        # Save state
        self._save_state()

    def _check_rollback_triggers(self):
        """Check if rollback should be triggered."""
        config = self.get_stage_config()

        # Win rate check (after sufficient trades)
        if self._metrics.total_trades >= 20:
            if self._metrics.win_rate < config.min_win_rate - 0.10:  # 10% below threshold
                self._trigger_rollback(
                    f"Win rate {self._metrics.win_rate:.1%} below threshold"
                )
                return

        # Drawdown check
        allocated = self.get_allocated_capital()
        if allocated > 0:
            drawdown_pct = self._metrics.max_drawdown / allocated
            if drawdown_pct > config.max_daily_loss_pct * 3:  # 3x daily limit as total
                self._trigger_rollback(
                    f"Drawdown {drawdown_pct:.1%} exceeded limit"
                )
                return

    def _trigger_rollback(self, reason: str):
        """Trigger rollback to previous stage."""
        if self._current_stage == RolloutStage.SHADOW:
            return  # Can't rollback from shadow

        current_idx = self.STAGE_ORDER.index(self._current_stage)
        new_stage = self.STAGE_ORDER[max(0, current_idx - 1)]

        # Create rollback event
        event = RollbackEvent(
            from_stage=self._current_stage,
            to_stage=new_stage,
            reason=reason,
            metrics_at_rollback=self._get_metrics_dict(),
        )
        self._rollback_history.append(event)

        # Update state
        old_stage = self._current_stage
        self._current_stage = new_stage
        self._metrics = RolloutMetrics(
            stage=new_stage,
            days_in_stage=0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            daily_losses=[],
            started_at=datetime.now(),
        )

        # Callback
        if self.rollback_callback:
            try:
                self.rollback_callback(event)
            except Exception as e:
                logger.error(f"Rollback callback failed: {e}")

        logger.warning(f"ROLLBACK: {old_stage.value} -> {new_stage.value}: {reason}")

        # Save state
        self._save_state()

    def force_stage(self, stage: RolloutStage, reason: str = "Manual override"):
        """Force a specific stage (for manual control)."""
        old_stage = self._current_stage

        self._stage_history.append({
            "from": old_stage.value,
            "to": stage.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        self._current_stage = stage
        self._metrics = RolloutMetrics(
            stage=stage,
            days_in_stage=0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            daily_losses=[],
            started_at=datetime.now(),
        )

        logger.info(f"Forced stage change: {old_stage.value} -> {stage.value}: {reason}")
        self._save_state()

    def pause(self, reason: str = "Manual pause"):
        """Pause trading (emergency)."""
        self.force_stage(RolloutStage.PAUSED, reason)

    def resume(self):
        """Resume from pause to shadow mode."""
        if self._current_stage == RolloutStage.PAUSED:
            self.force_stage(RolloutStage.SHADOW, "Resume from pause")

    def _get_metrics_dict(self) -> Dict:
        """Get metrics as dictionary."""
        return {
            "stage": self._metrics.stage.value,
            "days_in_stage": self._metrics.days_in_stage,
            "total_trades": self._metrics.total_trades,
            "win_rate": self._metrics.win_rate,
            "sharpe_ratio": self._metrics.sharpe_ratio,
            "total_pnl": self._metrics.total_pnl,
            "max_drawdown": self._metrics.max_drawdown,
            "started_at": self._metrics.started_at.isoformat(),
        }

    def _save_state(self):
        """Save state to file."""
        if not self.state_file:
            return

        try:
            state = {
                "current_stage": self._current_stage.value,
                "metrics": self._get_metrics_dict(),
                "daily_losses": self._metrics.daily_losses,
                "rollback_history": [
                    {
                        "from": e.from_stage.value,
                        "to": e.to_stage.value,
                        "reason": e.reason,
                        "timestamp": e.timestamp.isoformat(),
                    }
                    for e in self._rollback_history
                ],
                "stage_history": self._stage_history,
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load state from file."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self._current_stage = RolloutStage(state["current_stage"])

            metrics_data = state["metrics"]
            self._metrics = RolloutMetrics(
                stage=RolloutStage(metrics_data["stage"]),
                days_in_stage=metrics_data["days_in_stage"],
                total_trades=metrics_data["total_trades"],
                win_rate=metrics_data["win_rate"],
                sharpe_ratio=metrics_data["sharpe_ratio"],
                total_pnl=metrics_data["total_pnl"],
                max_drawdown=metrics_data["max_drawdown"],
                daily_losses=state.get("daily_losses", []),
                started_at=datetime.fromisoformat(metrics_data["started_at"]),
            )

            self._stage_history = state.get("stage_history", [])

            logger.info(f"Loaded state: stage={self._current_stage.value}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def get_status(self) -> Dict:
        """Get comprehensive status."""
        config = self.get_stage_config()

        return {
            "current_stage": self._current_stage.value,
            "trading_enabled": self.is_trading_enabled(),
            "allocated_capital": self.get_allocated_capital(),
            "max_position_size": self.get_max_position_size(),
            "max_leverage": self.get_max_leverage(),
            "max_assets": self.get_allowed_assets(),
            "metrics": self._get_metrics_dict(),
            "advancement_criteria": {
                "min_days": config.min_days_in_stage,
                "min_trades": config.min_trades,
                "min_win_rate": config.min_win_rate,
                "min_sharpe": config.min_sharpe,
            },
            "rollback_triggers": {
                "max_daily_loss_pct": config.max_daily_loss_pct,
            },
            "rollback_count": len(self._rollback_history),
        }

    def get_progress_to_next_stage(self) -> Dict:
        """Get progress towards next stage advancement."""
        if self._current_stage == RolloutStage.FULL:
            return {"next_stage": None, "complete": True}

        config = self.get_stage_config()

        return {
            "next_stage": self.STAGE_ORDER[self.STAGE_ORDER.index(self._current_stage) + 1].value,
            "days_progress": f"{self._metrics.days_in_stage}/{config.min_days_in_stage}",
            "trades_progress": f"{self._metrics.total_trades}/{config.min_trades}",
            "win_rate_progress": f"{self._metrics.win_rate:.1%}/{config.min_win_rate:.1%}",
            "sharpe_progress": f"{self._metrics.sharpe_ratio:.2f}/{config.min_sharpe:.2f}",
            "criteria_met": {
                "days": self._metrics.days_in_stage >= config.min_days_in_stage,
                "trades": self._metrics.total_trades >= config.min_trades,
                "win_rate": self._metrics.win_rate >= config.min_win_rate,
                "sharpe": self._metrics.sharpe_ratio >= config.min_sharpe,
            },
        }


# Singleton
_rollout_system: Optional[GradualRolloutSystem] = None


def get_gradual_rollout(
    total_capital: float = 30000.0,
    state_file: Optional[str] = None,
) -> GradualRolloutSystem:
    """Get or create the GradualRolloutSystem singleton."""
    global _rollout_system
    if _rollout_system is None:
        _rollout_system = GradualRolloutSystem(
            total_capital=total_capital,
            state_file=state_file,
        )
    return _rollout_system
