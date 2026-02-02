"""
Reward Shaping for 1% Daily Target.

Phase 2B Enhanced Version:
- NET-OF-COST returns (realistic friction)
- Turnover penalty (excessive trading)
- Slippage surprise penalty (worse than expected execution)
- Drawdown/CVaR penalty (tail risk aversion)
- Capital preservation violation penalty (NEAR INFINITE)

Custom reward functions designed to optimize for:
- 1% daily returns ($300/day on $30,000)
- Consistent profitability over big wins
- Risk-adjusted returns
- Win rate optimization
- LOW TURNOVER unless justified
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# CRITICAL: Capital preservation violation penalty
# This should be effectively infinite to prevent any learning that
# bypasses capital preservation constraints
CAPITAL_PRESERVATION_VIOLATION_PENALTY = -1000.0


@dataclass
class DailyProgress:
    """Track daily progress towards 1% goal."""
    date: str
    target_pnl: float  # $300 for $30k portfolio
    current_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0

    @property
    def progress_pct(self) -> float:
        """Progress towards daily target (0-100+)."""
        return (self.current_pnl / self.target_pnl) * 100 if self.target_pnl > 0 else 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.5

    @property
    def remaining_target(self) -> float:
        return max(0, self.target_pnl - self.current_pnl)


class RewardShaper:
    """
    Shapes rewards to optimize for 1% daily returns.

    Reward Components:
    1. PnL Reward: Base reward from trade profit/loss
    2. Daily Progress Multiplier: Bonus for progress towards daily goal
    3. Win Streak Bonus: Reward consistency
    4. Drawdown Penalty: Penalize large drawdowns
    5. Early Profit Bonus: Reward hitting target early in the day
    6. Risk-Adjusted Bonus: Reward good risk/reward trades
    """

    def __init__(
        self,
        portfolio_value: float = 30000.0,
        daily_target_pct: float = 0.01,  # 1%
        max_daily_drawdown_pct: float = 0.02,  # 2%
    ):
        self.portfolio_value = portfolio_value
        self.daily_target_pct = daily_target_pct
        self.daily_target_usd = portfolio_value * daily_target_pct
        self.max_daily_drawdown = portfolio_value * max_daily_drawdown_pct

        # Daily tracking
        self._daily_progress: Optional[DailyProgress] = None
        self._last_reset_date: Optional[str] = None

        # Streak tracking
        self._win_streak = 0
        self._loss_streak = 0

        # Component weights
        self.weights = {
            "pnl": 1.0,
            "progress": 0.5,
            "streak": 0.3,
            "drawdown": 0.4,
            "early_bonus": 0.2,
            "risk_reward": 0.3,
        }

        logger.info(
            f"RewardShaper initialized: target=${self.daily_target_usd:.2f}/day "
            f"({daily_target_pct*100:.1f}%)"
        )

    def _ensure_daily_progress(self):
        """Ensure daily progress tracker is initialized for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._daily_progress = DailyProgress(
                date=today,
                target_pnl=self.daily_target_usd,
            )
            self._last_reset_date = today
            logger.debug(f"Reset daily progress for {today}")

    def calculate_reward(
        self,
        pnl: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        hold_time_minutes: int,
        regime: str = "unknown",
    ) -> Dict[str, float]:
        """
        Calculate shaped reward for a trade.

        Args:
            pnl: Absolute profit/loss in USD
            pnl_pct: Percentage profit/loss
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            hold_time_minutes: How long position was held
            regime: Current market regime

        Returns:
            Dictionary with reward components and total
        """
        self._ensure_daily_progress()

        rewards = {}

        # 1. Base PnL Reward (normalized)
        # Scale so typical trade gives reward in [-2, 2] range
        rewards["pnl"] = pnl_pct * 100  # 1% = 1.0 reward

        # 2. Daily Progress Multiplier
        # Bonus for making progress towards daily goal
        progress_before = self._daily_progress.progress_pct
        self._daily_progress.current_pnl += pnl
        progress_after = self._daily_progress.progress_pct

        if pnl > 0:
            # Bigger bonus when closer to completing daily target
            if progress_after >= 100:
                rewards["progress"] = 2.0  # Hit daily target!
            elif progress_after >= 75:
                rewards["progress"] = 1.0
            elif progress_after >= 50:
                rewards["progress"] = 0.5
            else:
                rewards["progress"] = 0.2
        else:
            # Penalty scaled by how much it sets us back
            setback = progress_before - progress_after
            rewards["progress"] = -setback / 50  # -1.0 for 50% setback

        # 3. Win Streak Bonus
        if pnl > 0:
            self._win_streak += 1
            self._loss_streak = 0
            self._daily_progress.wins += 1
            # Exponential bonus for streaks
            rewards["streak"] = min(1.5, 0.2 * (1.5 ** (self._win_streak - 1)))
        else:
            self._loss_streak += 1
            self._win_streak = 0
            self._daily_progress.losses += 1
            # Penalty increases with streak
            rewards["streak"] = -0.3 * min(3, self._loss_streak)

        self._daily_progress.trades += 1

        # 4. Drawdown Penalty
        # Track peak PnL and penalize drawdowns
        if self._daily_progress.current_pnl > self._daily_progress.peak_pnl:
            self._daily_progress.peak_pnl = self._daily_progress.current_pnl

        current_drawdown = self._daily_progress.peak_pnl - self._daily_progress.current_pnl
        if current_drawdown > self._daily_progress.max_drawdown:
            self._daily_progress.max_drawdown = current_drawdown

        # Severe penalty for approaching max daily drawdown
        drawdown_pct = current_drawdown / self.max_daily_drawdown
        if drawdown_pct > 0.8:
            rewards["drawdown"] = -2.0  # Severe penalty
        elif drawdown_pct > 0.5:
            rewards["drawdown"] = -1.0
        elif drawdown_pct > 0.25:
            rewards["drawdown"] = -0.3
        else:
            rewards["drawdown"] = 0.0

        # 5. Early Profit Bonus
        # Reward hitting target early in the trading day
        hour = datetime.now().hour
        if progress_after >= 100:
            if hour < 12:
                rewards["early_bonus"] = 1.0  # Hit target in morning
            elif hour < 18:
                rewards["early_bonus"] = 0.5
            else:
                rewards["early_bonus"] = 0.2
        else:
            rewards["early_bonus"] = 0.0

        # 6. Risk/Reward Bonus
        # Reward trades with good risk/reward ratio
        if stop_loss > 0 and take_profit > 0:
            if entry_price > 0:
                risk = abs(entry_price - stop_loss) / entry_price
                reward_potential = abs(take_profit - entry_price) / entry_price
                rr_ratio = reward_potential / risk if risk > 0 else 1.0

                if pnl > 0 and rr_ratio >= 2.0:
                    rewards["risk_reward"] = 0.5  # Good R:R and won
                elif pnl > 0 and rr_ratio >= 1.5:
                    rewards["risk_reward"] = 0.3
                elif pnl < 0 and rr_ratio < 1.0:
                    rewards["risk_reward"] = -0.5  # Bad R:R and lost
                else:
                    rewards["risk_reward"] = 0.0
            else:
                rewards["risk_reward"] = 0.0
        else:
            rewards["risk_reward"] = 0.0

        # Calculate weighted total
        total = sum(
            rewards[key] * self.weights.get(key, 1.0)
            for key in rewards
        )
        rewards["total"] = total

        # Add metadata
        rewards["daily_progress_pct"] = progress_after
        rewards["win_streak"] = self._win_streak
        rewards["loss_streak"] = self._loss_streak

        logger.debug(
            f"Reward calculated: total={total:.2f}, pnl={rewards['pnl']:.2f}, "
            f"progress={rewards['progress']:.2f}, streak={rewards['streak']:.2f}"
        )

        return rewards

    def get_daily_summary(self) -> Dict[str, float]:
        """Get summary of daily performance."""
        self._ensure_daily_progress()
        return {
            "date": self._daily_progress.date,
            "current_pnl": self._daily_progress.current_pnl,
            "target_pnl": self._daily_progress.target_pnl,
            "progress_pct": self._daily_progress.progress_pct,
            "remaining": self._daily_progress.remaining_target,
            "trades": self._daily_progress.trades,
            "win_rate": self._daily_progress.win_rate,
            "max_drawdown": self._daily_progress.max_drawdown,
            "current_win_streak": self._win_streak,
            "current_loss_streak": self._loss_streak,
        }

    def should_stop_trading(self) -> tuple[bool, str]:
        """Check if we should stop trading for the day."""
        self._ensure_daily_progress()

        # Hit daily target
        if self._daily_progress.progress_pct >= 100:
            return True, "Daily target reached!"

        # Max drawdown hit
        if self._daily_progress.max_drawdown >= self.max_daily_drawdown:
            return True, f"Max daily drawdown hit (${self._daily_progress.max_drawdown:.2f})"

        # Too many consecutive losses
        if self._loss_streak >= 5:
            return True, f"Loss streak of {self._loss_streak} - cooling down"

        return False, ""

    def get_position_size_adjustment(self) -> float:
        """
        Get position size adjustment based on daily progress.

        Returns multiplier (0.5 to 1.5) for position sizing.
        """
        self._ensure_daily_progress()

        # Base multiplier
        multiplier = 1.0

        # Reduce size if in drawdown
        drawdown_pct = self._daily_progress.max_drawdown / self.max_daily_drawdown
        if drawdown_pct > 0.5:
            multiplier *= 0.6
        elif drawdown_pct > 0.25:
            multiplier *= 0.8

        # Reduce size on loss streak
        if self._loss_streak >= 3:
            multiplier *= 0.7
        elif self._loss_streak >= 2:
            multiplier *= 0.85

        # Increase size on win streak (carefully)
        if self._win_streak >= 4:
            multiplier *= 1.2
        elif self._win_streak >= 3:
            multiplier *= 1.1

        # Reduce size if close to target (protect gains)
        if self._daily_progress.progress_pct >= 80:
            multiplier *= 0.7
        elif self._daily_progress.progress_pct >= 60:
            multiplier *= 0.85

        return max(0.5, min(1.5, multiplier))

    def update_portfolio_value(self, new_value: float):
        """Update portfolio value (affects daily target)."""
        self.portfolio_value = new_value
        self.daily_target_usd = new_value * self.daily_target_pct
        self.max_daily_drawdown = new_value * 0.02

    def reset_daily(self):
        """Force reset of daily tracking."""
        self._last_reset_date = None
        self._ensure_daily_progress()


@dataclass
class CostAwareConfig:
    """Configuration for cost-aware reward shaping."""
    # Turnover penalty: penalize excessive trading
    turnover_penalty_per_trade: float = 0.05  # -0.05 per trade
    max_trades_per_day_before_penalty: int = 10
    turnover_penalty_multiplier: float = 0.1  # Additional penalty per trade over limit

    # Slippage surprise penalty
    expected_slippage_pct: float = 0.05  # 5 bps expected
    slippage_surprise_multiplier: float = 5.0  # 5x penalty for unexpected slippage

    # Cost thresholds
    max_acceptable_cost_pct: float = 0.3  # 30 bps per trade max
    cost_penalty_multiplier: float = 2.0

    # Drawdown/CVaR penalties
    cvar_threshold_pct: float = 0.05  # 5% CVaR threshold
    cvar_penalty_multiplier: float = 3.0
    tail_risk_lookback: int = 20  # Trades to consider for tail risk

    # Capital preservation
    preservation_violation_penalty: float = CAPITAL_PRESERVATION_VIOLATION_PENALTY

    # Calibration factors (updated from counterfactual analysis)
    cost_awareness_weight: float = 0.5
    turnover_weight: float = 0.3
    tail_risk_weight: float = 0.4


class CostAwareRewardShaper(RewardShaper):
    """
    Cost-aware reward shaper for Phase 2B.

    Extends base RewardShaper with:
    - Net-of-cost returns
    - Turnover penalty
    - Slippage surprise penalty
    - CVaR/tail risk penalty
    - Capital preservation violation penalty

    CRITICAL: Learns to optimize NET returns, not gross.
    """

    def __init__(
        self,
        portfolio_value: float = 30000.0,
        daily_target_pct: float = 0.01,
        max_daily_drawdown_pct: float = 0.02,
        cost_config: Optional[CostAwareConfig] = None,
    ):
        super().__init__(
            portfolio_value=portfolio_value,
            daily_target_pct=daily_target_pct,
            max_daily_drawdown_pct=max_daily_drawdown_pct,
        )

        self.cost_config = cost_config or CostAwareConfig()

        # Track recent trades for tail risk calculation
        self._recent_pnls: List[float] = []

        # Track daily turnover
        self._daily_trades_count = 0
        self._daily_turnover_usd = 0.0

        # Update weights to include new components
        self.weights.update({
            "cost_penalty": self.cost_config.cost_awareness_weight,
            "turnover_penalty": self.cost_config.turnover_weight,
            "slippage_surprise": self.cost_config.cost_awareness_weight,
            "tail_risk": self.cost_config.tail_risk_weight,
            "preservation_violation": 1.0,  # Always full weight
        })

        logger.info(
            f"CostAwareRewardShaper initialized with cost_awareness={self.cost_config.cost_awareness_weight}"
        )

    def calculate_reward(
        self,
        pnl: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        hold_time_minutes: int,
        regime: str = "unknown",
        # New cost-aware parameters
        gross_pnl: Optional[float] = None,
        slippage_cost: float = 0.0,
        fee_cost: float = 0.0,
        expected_slippage: Optional[float] = None,
        position_value: float = 0.0,
        preservation_level: str = "normal",
        preservation_violated: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate cost-aware shaped reward.

        Args:
            pnl: NET profit/loss (after costs)
            pnl_pct: NET percentage profit/loss
            entry_price: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            hold_time_minutes: How long position was held
            regime: Current market regime
            gross_pnl: Gross P&L before costs (if different from pnl)
            slippage_cost: Actual slippage cost
            fee_cost: Fee cost
            expected_slippage: Expected slippage (for surprise calculation)
            position_value: Total position value for cost % calculation
            preservation_level: Current capital preservation level
            preservation_violated: Whether preservation rules were violated

        Returns:
            Dictionary with reward components and total
        """
        # CRITICAL: Check preservation violation FIRST
        if preservation_violated:
            logger.warning("Capital preservation violation detected - applying max penalty")
            return {
                "total": self.cost_config.preservation_violation_penalty,
                "preservation_violation": self.cost_config.preservation_violation_penalty,
                "reason": "Capital preservation rules violated",
            }

        # Get base rewards (using NET pnl)
        rewards = super().calculate_reward(
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            hold_time_minutes=hold_time_minutes,
            regime=regime,
        )

        # Track for tail risk
        self._recent_pnls.append(pnl)
        if len(self._recent_pnls) > self.cost_config.tail_risk_lookback:
            self._recent_pnls = self._recent_pnls[-self.cost_config.tail_risk_lookback:]

        # Track turnover
        self._daily_trades_count += 1
        self._daily_turnover_usd += position_value

        # 1. Cost Penalty (total execution cost as % of position)
        total_cost = slippage_cost + fee_cost
        if position_value > 0:
            cost_pct = total_cost / position_value
            if cost_pct > self.cost_config.max_acceptable_cost_pct / 100:
                excess_cost = cost_pct - (self.cost_config.max_acceptable_cost_pct / 100)
                rewards["cost_penalty"] = -excess_cost * 100 * self.cost_config.cost_penalty_multiplier
            else:
                rewards["cost_penalty"] = 0.0
        else:
            rewards["cost_penalty"] = 0.0

        # 2. Turnover Penalty
        if self._daily_trades_count > self.cost_config.max_trades_per_day_before_penalty:
            excess_trades = self._daily_trades_count - self.cost_config.max_trades_per_day_before_penalty
            rewards["turnover_penalty"] = (
                -self.cost_config.turnover_penalty_per_trade -
                (excess_trades * self.cost_config.turnover_penalty_multiplier)
            )
        else:
            # Small base penalty to discourage unnecessary trades
            rewards["turnover_penalty"] = -self.cost_config.turnover_penalty_per_trade

        # 3. Slippage Surprise Penalty
        if expected_slippage is not None and slippage_cost > 0:
            surprise = slippage_cost - expected_slippage
            if surprise > 0:
                # Worse than expected
                rewards["slippage_surprise"] = (
                    -surprise / max(expected_slippage, 0.01) *
                    self.cost_config.slippage_surprise_multiplier * 0.1
                )
            else:
                # Better than expected - small bonus
                rewards["slippage_surprise"] = 0.1
        else:
            rewards["slippage_surprise"] = 0.0

        # 4. Tail Risk / CVaR Penalty
        if len(self._recent_pnls) >= 5:
            # Calculate CVaR (expected shortfall) at 10th percentile
            sorted_pnls = sorted(self._recent_pnls)
            cvar_idx = max(1, len(sorted_pnls) // 10)
            cvar = np.mean(sorted_pnls[:cvar_idx])

            # Penalty if CVaR exceeds threshold
            cvar_threshold = -self.portfolio_value * self.cost_config.cvar_threshold_pct
            if cvar < cvar_threshold:
                excess_risk = abs(cvar - cvar_threshold) / self.portfolio_value
                rewards["tail_risk"] = -excess_risk * 100 * self.cost_config.cvar_penalty_multiplier
            else:
                rewards["tail_risk"] = 0.0
        else:
            rewards["tail_risk"] = 0.0

        # 5. Preservation level penalty (graduated)
        preservation_penalties = {
            "normal": 0.0,
            "cautious": -0.1,
            "defensive": -0.3,
            "critical": -0.5,
            "lockdown": -2.0,  # Should not be trading in lockdown anyway
        }
        rewards["preservation_penalty"] = preservation_penalties.get(
            preservation_level.lower(), 0.0
        )

        # Recalculate total with new components
        total = sum(
            rewards[key] * self.weights.get(key, 1.0)
            for key in rewards
            if key not in ["total", "daily_progress_pct", "win_streak", "loss_streak", "reason"]
        )
        rewards["total"] = total

        # Add cost metadata
        rewards["total_cost_usd"] = total_cost
        rewards["cost_pct"] = (total_cost / position_value * 100) if position_value > 0 else 0
        rewards["daily_trade_count"] = self._daily_trades_count

        logger.debug(
            f"Cost-aware reward: total={total:.2f}, "
            f"cost_penalty={rewards['cost_penalty']:.2f}, "
            f"turnover={rewards['turnover_penalty']:.2f}, "
            f"tail_risk={rewards['tail_risk']:.2f}"
        )

        return rewards

    def calibrate_from_counterfactual(
        self,
        avg_slippage_pct: float,
        avg_fee_pct: float,
        avg_trades_per_day: float,
        historical_cvar: float,
    ):
        """
        Calibrate reward weights from counterfactual analysis.

        Args:
            avg_slippage_pct: Average slippage as % of position
            avg_fee_pct: Average fees as % of position
            avg_trades_per_day: Average number of trades per day
            historical_cvar: Historical CVaR from backtest
        """
        # Adjust expected slippage based on historical data
        self.cost_config.expected_slippage_pct = avg_slippage_pct * 100  # Convert to bps

        # Adjust max trades before penalty based on historical turnover
        self.cost_config.max_trades_per_day_before_penalty = max(
            5, int(avg_trades_per_day * 1.2)
        )

        # Adjust CVaR threshold based on historical tail risk
        if historical_cvar < 0:
            self.cost_config.cvar_threshold_pct = min(
                0.1, abs(historical_cvar / self.portfolio_value) * 1.5
            )

        logger.info(
            f"Calibrated rewards: expected_slip={self.cost_config.expected_slippage_pct:.2f}bps, "
            f"max_trades={self.cost_config.max_trades_per_day_before_penalty}, "
            f"cvar_threshold={self.cost_config.cvar_threshold_pct:.2%}"
        )

    def reset_daily(self):
        """Reset daily tracking including turnover."""
        super().reset_daily()
        self._daily_trades_count = 0
        self._daily_turnover_usd = 0.0

    def get_cost_summary(self) -> Dict[str, float]:
        """Get summary of cost-related metrics."""
        return {
            "daily_trades": self._daily_trades_count,
            "daily_turnover_usd": self._daily_turnover_usd,
            "recent_cvar": self._calculate_recent_cvar(),
            "expected_slippage_bps": self.cost_config.expected_slippage_pct,
            "max_trades_before_penalty": self.cost_config.max_trades_per_day_before_penalty,
        }

    def _calculate_recent_cvar(self) -> float:
        """Calculate recent CVaR from tracked P&Ls."""
        if len(self._recent_pnls) < 5:
            return 0.0
        sorted_pnls = sorted(self._recent_pnls)
        cvar_idx = max(1, len(sorted_pnls) // 10)
        return np.mean(sorted_pnls[:cvar_idx])


# Singleton instances
_reward_shaper: Optional[RewardShaper] = None
_cost_aware_shaper: Optional[CostAwareRewardShaper] = None


def get_reward_shaper(portfolio_value: float = 30000.0) -> RewardShaper:
    """Get or create the RewardShaper singleton."""
    global _reward_shaper
    if _reward_shaper is None:
        _reward_shaper = RewardShaper(portfolio_value=portfolio_value)
    return _reward_shaper


def get_cost_aware_shaper(
    portfolio_value: float = 30000.0,
    cost_config: Optional[CostAwareConfig] = None,
) -> CostAwareRewardShaper:
    """Get or create the CostAwareRewardShaper singleton."""
    global _cost_aware_shaper
    if _cost_aware_shaper is None:
        _cost_aware_shaper = CostAwareRewardShaper(
            portfolio_value=portfolio_value,
            cost_config=cost_config,
        )
    return _cost_aware_shaper


def reset_reward_shapers():
    """Reset all reward shaper singletons."""
    global _reward_shaper, _cost_aware_shaper
    _reward_shaper = None
    _cost_aware_shaper = None
