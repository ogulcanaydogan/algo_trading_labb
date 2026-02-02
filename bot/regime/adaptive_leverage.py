"""
Adaptive Leverage Engine.

Dynamically adjusts leverage based on:
- Win/loss streak
- Market regime
- Volatility conditions
- Time-of-day liquidity
- Position correlation
- Daily P&L progress
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LeverageConfig:
    """Configuration for adaptive leverage."""
    base_leverage: float = 1.0
    max_leverage: float = 5.0
    min_leverage: float = 0.5

    # Streak adjustments
    win_streak_bonus: float = 0.15  # +15% per win
    loss_streak_penalty: float = 0.25  # -25% per loss
    max_streak_adjustment: float = 0.5  # Max +/- 50%

    # Regime multipliers
    regime_multipliers: Dict[str, float] = None

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                "BULL": 1.3,
                "STRONG_BULL": 1.5,
                "BEAR": 0.8,
                "STRONG_BEAR": 0.6,
                "CRASH": 0.4,
                "SIDEWAYS": 1.0,
                "HIGH_VOL": 0.7,
                "LOW_VOL": 1.2,
                "RECOVERY": 1.1,
                "unknown": 1.0,
            }


class AdaptiveLeverageEngine:
    """
    Calculates optimal leverage based on multiple factors.

    Learns optimal leverage per regime over time.
    """

    def __init__(self, config: Optional[LeverageConfig] = None):
        self.config = config or LeverageConfig()

        # Tracking
        self._win_streak = 0
        self._loss_streak = 0
        self._daily_pnl = 0.0
        self._daily_target = 300.0  # $300 for 1% on $30k
        self._last_date: Optional[str] = None

        # Learned regime leverage (updated from outcomes)
        self._learned_leverage: Dict[str, float] = {}
        self._regime_outcomes: Dict[str, List[Tuple[float, float]]] = {}  # regime -> [(leverage, pnl)]

        logger.info("AdaptiveLeverageEngine initialized")

    def calculate_leverage(
        self,
        base_leverage: float,
        regime: str,
        volatility: float,
        signal_confidence: float,
        current_hour: Optional[int] = None,
        correlation_penalty: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Calculate optimal leverage for a trade.

        Args:
            base_leverage: Starting leverage from strategy
            regime: Current market regime
            volatility: Current volatility level
            signal_confidence: ML signal confidence (0-1)
            current_hour: Hour of day (0-23) for liquidity adjustment
            correlation_penalty: Penalty for correlated positions (0-1)

        Returns:
            Tuple of (adjusted_leverage, reasoning)
        """
        self._check_daily_reset()

        adjustments = []
        leverage = base_leverage

        # 1. Regime Adjustment
        regime_mult = self.config.regime_multipliers.get(regime, 1.0)

        # Use learned leverage if available
        if regime in self._learned_leverage:
            regime_mult = (regime_mult + self._learned_leverage[regime]) / 2

        leverage *= regime_mult
        adjustments.append(f"Regime({regime}): {regime_mult:.2f}x")

        # 2. Streak Adjustment
        streak_adj = 1.0
        if self._win_streak >= 3:
            streak_adj = 1.0 + min(
                self.config.max_streak_adjustment,
                self._win_streak * self.config.win_streak_bonus
            )
            adjustments.append(f"WinStreak({self._win_streak}): +{(streak_adj-1)*100:.0f}%")
        elif self._loss_streak >= 2:
            streak_adj = 1.0 - min(
                self.config.max_streak_adjustment,
                self._loss_streak * self.config.loss_streak_penalty
            )
            adjustments.append(f"LossStreak({self._loss_streak}): {(streak_adj-1)*100:.0f}%")

        leverage *= streak_adj

        # 3. Volatility Adjustment
        if volatility > 0.03:  # High volatility
            vol_adj = 0.7
            adjustments.append(f"HighVol: -30%")
        elif volatility < 0.01:  # Low volatility
            vol_adj = 1.2
            adjustments.append(f"LowVol: +20%")
        else:
            vol_adj = 1.0

        leverage *= vol_adj

        # 4. Confidence Adjustment
        conf_adj = 0.7 + (signal_confidence * 0.6)  # 0.7 to 1.3
        leverage *= conf_adj
        adjustments.append(f"Confidence({signal_confidence:.2f}): {conf_adj:.2f}x")

        # 5. Time-of-Day Adjustment (liquidity)
        if current_hour is not None:
            if 0 <= current_hour < 6:  # Low liquidity hours
                time_adj = 0.7
                adjustments.append("LowLiquidity: -30%")
            elif 14 <= current_hour <= 21:  # Peak hours (US market open)
                time_adj = 1.1
                adjustments.append("PeakHours: +10%")
            else:
                time_adj = 1.0
            leverage *= time_adj

        # 6. Correlation Penalty
        if correlation_penalty > 0:
            corr_adj = 1.0 - (correlation_penalty * 0.3)
            leverage *= corr_adj
            adjustments.append(f"CorrPenalty: -{correlation_penalty*30:.0f}%")

        # 7. Daily Progress Adjustment
        if self._daily_pnl >= self._daily_target * 0.8:
            # Close to target - reduce risk
            progress_adj = 0.6
            adjustments.append("NearTarget: -40%")
        elif self._daily_pnl <= -self._daily_target * 0.5:
            # In drawdown - reduce risk
            progress_adj = 0.7
            adjustments.append("InDrawdown: -30%")
        else:
            progress_adj = 1.0

        leverage *= progress_adj

        # Apply bounds
        leverage = max(self.config.min_leverage, min(self.config.max_leverage, leverage))

        reasoning = " | ".join(adjustments)
        return leverage, reasoning

    def record_outcome(self, regime: str, leverage_used: float, pnl: float):
        """Record trade outcome for learning."""
        # Update streaks
        if pnl > 0:
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0

        # Update daily P&L
        self._daily_pnl += pnl

        # Store for regime learning
        if regime not in self._regime_outcomes:
            self._regime_outcomes[regime] = []
        self._regime_outcomes[regime].append((leverage_used, pnl))

        # Update learned leverage for regime
        self._update_learned_leverage(regime)

    def _update_learned_leverage(self, regime: str):
        """Update learned optimal leverage for a regime."""
        if regime not in self._regime_outcomes:
            return

        outcomes = self._regime_outcomes[regime]
        if len(outcomes) < 10:
            return

        # Find leverage that maximizes risk-adjusted returns
        # Group by leverage ranges and calculate Sharpe-like ratio
        leverage_groups: Dict[float, List[float]] = {}
        for lev, pnl in outcomes[-50:]:  # Last 50 trades
            bucket = round(lev, 1)  # Round to 0.1
            if bucket not in leverage_groups:
                leverage_groups[bucket] = []
            leverage_groups[bucket].append(pnl)

        best_leverage = 1.0
        best_score = -999

        for lev, pnls in leverage_groups.items():
            if len(pnls) >= 3:
                import numpy as np
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls) + 0.001  # Avoid division by zero
                score = mean_pnl / std_pnl  # Sharpe-like

                if score > best_score:
                    best_score = score
                    best_leverage = lev

        # Smooth update
        if regime in self._learned_leverage:
            self._learned_leverage[regime] = (
                0.7 * self._learned_leverage[regime] + 0.3 * best_leverage
            )
        else:
            self._learned_leverage[regime] = best_leverage

        logger.debug(f"Updated learned leverage for {regime}: {self._learned_leverage[regime]:.2f}")

    def _check_daily_reset(self):
        """Reset daily tracking if new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_date != today:
            self._daily_pnl = 0.0
            self._last_date = today

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Get leverage statistics by regime."""
        stats = {}
        for regime, outcomes in self._regime_outcomes.items():
            if outcomes:
                import numpy as np
                leverages = [o[0] for o in outcomes]
                pnls = [o[1] for o in outcomes]
                stats[regime] = {
                    "avg_leverage": np.mean(leverages),
                    "learned_leverage": self._learned_leverage.get(regime, None),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    "trades": len(outcomes),
                }
        return stats

    def set_daily_target(self, target: float):
        """Update daily target for progress adjustment."""
        self._daily_target = target


# Singleton
_leverage_engine: Optional[AdaptiveLeverageEngine] = None


def get_adaptive_leverage_engine() -> AdaptiveLeverageEngine:
    """Get or create the AdaptiveLeverageEngine singleton."""
    global _leverage_engine
    if _leverage_engine is None:
        _leverage_engine = AdaptiveLeverageEngine()
    return _leverage_engine
