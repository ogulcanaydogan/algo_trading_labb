"""
Safe Strategy Weighting Advisor.

Phase 2B Requirement:
Adjusts internal weighting among existing strategies based on RL preferences
and counterfactual evaluator statistics.

CRITICAL CONSTRAINTS:
- NEVER changes position sizing
- NEVER changes leverage caps
- NEVER places orders directly
- Only affects strategy SELECTION weights
- Maximum shift clamped to 10-20% per day
- TradeGate / RiskBudget / CapitalPreservation remain ultimate authority
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LOCKED SAFETY CONSTRAINTS - DO NOT MODIFY
# =============================================================================

MAX_DAILY_WEIGHT_SHIFT = 0.10  # Maximum 10% weight shift per day
MIN_STRATEGY_WEIGHT = 0.05    # Minimum 5% weight (never fully disable)
MAX_STRATEGY_WEIGHT = 0.50    # Maximum 50% weight (no single strategy dominance)
WEIGHT_UPDATE_COOLDOWN_HOURS = 4  # Minimum hours between weight updates


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_name: str
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    avg_cost_pct: float = 0.0  # Average cost as percentage of trade value
    regime_performance: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def quality_score(self) -> float:
        """Calculate overall quality score."""
        if self.trade_count < 10:
            return 0.5  # Insufficient data

        # Weighted combination
        score = (
            self.win_rate * 0.25 +
            min(1.0, max(0.0, (self.sharpe_ratio + 1) / 3)) * 0.30 +  # Normalized Sharpe
            (1.0 - min(1.0, self.max_drawdown / 0.10)) * 0.20 +  # Drawdown penalty
            min(1.0, max(0.0, (0.005 - self.avg_cost_pct) / 0.005)) * 0.15 +  # Cost efficiency
            min(1.0, self.trade_count / 100) * 0.10  # Sample size bonus
        )
        return max(0.1, min(0.9, score))


@dataclass
class WeightingConfig:
    """Configuration for strategy weighting advisor."""
    enabled: bool = True
    max_daily_shift: float = MAX_DAILY_WEIGHT_SHIFT
    min_weight: float = MIN_STRATEGY_WEIGHT
    max_weight: float = MAX_STRATEGY_WEIGHT
    cooldown_hours: float = WEIGHT_UPDATE_COOLDOWN_HOURS

    # Regime-specific adjustments
    regime_weight_boost: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # RL integration
    use_rl_preferences: bool = True
    rl_preference_weight: float = 0.3  # How much RL preferences influence weights

    # Safety
    require_minimum_trades: int = 20  # Minimum trades before adjusting weight

    def __post_init__(self):
        """Lock safety parameters."""
        # LOCKED: Cannot exceed safety limits
        self.max_daily_shift = min(self.max_daily_shift, MAX_DAILY_WEIGHT_SHIFT)
        self.min_weight = max(self.min_weight, MIN_STRATEGY_WEIGHT)
        self.max_weight = min(self.max_weight, MAX_STRATEGY_WEIGHT)

        # Default regime boosts if not specified
        if not self.regime_weight_boost:
            self.regime_weight_boost = {
                "bull": {"TrendFollower": 0.05, "MomentumTrader": 0.03},
                "bear": {"ShortSpecialist": 0.05, "MeanReversion": 0.02},
                "sideways": {"MeanReversion": 0.05, "Scalper": 0.03},
                "volatile": {"Scalper": 0.03},
                "crash": {"ShortSpecialist": 0.05},
            }


@dataclass
class WeightUpdate:
    """Record of a weight update."""
    timestamp: datetime
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    reason: str
    regime: str
    rl_preferences: Dict[str, float]
    clipped: bool = False  # Whether update was clipped by safety limits


class StrategyWeightingAdvisor:
    """
    Safe advisor for strategy weight adjustments.

    CRITICAL: This class ONLY provides weight RECOMMENDATIONS.
    It has NO execution authority.
    It CANNOT change position sizes or leverage.

    The actual weight application is done by the strategy selector,
    which still respects TradeGate and all safety systems.
    """

    def __init__(
        self,
        config: Optional[WeightingConfig] = None,
        state_path: Optional[Path] = None,
    ):
        self.config = config or WeightingConfig()
        self.state_path = state_path or Path("data/rl/strategy_weights.json")

        # Initialize default weights
        self._weights: Dict[str, float] = {
            "TrendFollower": 0.20,
            "MeanReversion": 0.20,
            "MomentumTrader": 0.20,
            "ShortSpecialist": 0.20,
            "Scalper": 0.20,
        }

        # Performance tracking
        self._performance: Dict[str, StrategyPerformance] = {}

        # Update history
        self._update_history: List[WeightUpdate] = []
        self._last_update: Optional[datetime] = None

        # Load saved state
        self._load_state()

        logger.info(
            f"StrategyWeightingAdvisor initialized: "
            f"max_shift={self.config.max_daily_shift}, "
            f"rl_weight={self.config.rl_preference_weight}"
        )

    def _load_state(self):
        """Load saved weights and history."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)

                self._weights = state.get("weights", self._weights)
                self._last_update = (
                    datetime.fromisoformat(state["last_update"])
                    if state.get("last_update")
                    else None
                )

                # Load performance
                for name, perf in state.get("performance", {}).items():
                    self._performance[name] = StrategyPerformance(
                        strategy_name=name,
                        win_rate=perf.get("win_rate", 0.5),
                        avg_pnl=perf.get("avg_pnl", 0.0),
                        sharpe_ratio=perf.get("sharpe_ratio", 0.0),
                        max_drawdown=perf.get("max_drawdown", 0.0),
                        trade_count=perf.get("trade_count", 0),
                        avg_cost_pct=perf.get("avg_cost_pct", 0.0),
                        regime_performance=perf.get("regime_performance", {}),
                    )

                logger.info(f"Loaded strategy weights from {self.state_path}")

            except Exception as e:
                logger.warning(f"Failed to load weight state: {e}")

    def _save_state(self):
        """Save current weights and history."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "weights": self._weights,
                "last_update": self._last_update.isoformat() if self._last_update else None,
                "performance": {
                    name: {
                        "win_rate": perf.win_rate,
                        "avg_pnl": perf.avg_pnl,
                        "sharpe_ratio": perf.sharpe_ratio,
                        "max_drawdown": perf.max_drawdown,
                        "trade_count": perf.trade_count,
                        "avg_cost_pct": perf.avg_cost_pct,
                        "regime_performance": perf.regime_performance,
                    }
                    for name, perf in self._performance.items()
                },
                "history": [
                    {
                        "timestamp": u.timestamp.isoformat(),
                        "old_weights": u.old_weights,
                        "new_weights": u.new_weights,
                        "reason": u.reason,
                        "regime": u.regime,
                        "clipped": u.clipped,
                    }
                    for u in self._update_history[-100:]  # Keep last 100 updates
                ],
            }

            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save weight state: {e}")

    def get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights (read-only copy)."""
        return dict(self._weights)

    def update_strategy_performance(
        self,
        strategy_name: str,
        pnl: float,
        pnl_pct: float,
        regime: str,
        cost_pct: float = 0.0,
        is_win: bool = False,
    ):
        """
        Update performance metrics for a strategy.

        This is called after each trade to track strategy performance.
        """
        if strategy_name not in self._performance:
            self._performance[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )

        perf = self._performance[strategy_name]
        n = perf.trade_count

        # Update running averages
        perf.win_rate = (perf.win_rate * n + (1.0 if is_win else 0.0)) / (n + 1)
        perf.avg_pnl = (perf.avg_pnl * n + pnl) / (n + 1)
        perf.avg_cost_pct = (perf.avg_cost_pct * n + cost_pct) / (n + 1)

        # Update regime performance
        if regime not in perf.regime_performance:
            perf.regime_performance[regime] = 0.0
        regime_n = sum(1 for _ in perf.regime_performance.values())
        perf.regime_performance[regime] = (
            (perf.regime_performance[regime] * regime_n + pnl_pct) / (regime_n + 1)
        )

        # Track drawdown (simplified)
        if pnl_pct < 0:
            perf.max_drawdown = max(perf.max_drawdown, abs(pnl_pct))

        perf.trade_count = n + 1
        perf.last_updated = datetime.now()

    def _calculate_target_weights(
        self,
        regime: str,
        rl_preferences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate target weights based on performance and RL preferences.

        Returns unconstrained target weights (will be clamped later).
        """
        # Start with quality scores
        quality_scores = {}
        for name in self._weights:
            if name in self._performance:
                quality_scores[name] = self._performance[name].quality_score()
            else:
                quality_scores[name] = 0.5  # Default for unknown strategies

        # Apply regime boosts
        regime_boosts = self.config.regime_weight_boost.get(regime, {})
        for name, boost in regime_boosts.items():
            if name in quality_scores:
                quality_scores[name] = min(0.95, quality_scores[name] + boost)

        # Incorporate RL preferences if enabled
        if self.config.use_rl_preferences and rl_preferences:
            rl_weight = self.config.rl_preference_weight
            for name, pref in rl_preferences.items():
                if name in quality_scores:
                    # Blend quality score with RL preference
                    quality_scores[name] = (
                        quality_scores[name] * (1 - rl_weight) +
                        pref * rl_weight
                    )

        # Normalize to weights
        total = sum(quality_scores.values())
        if total > 0:
            target_weights = {
                name: score / total for name, score in quality_scores.items()
            }
        else:
            # Fallback to equal weights
            n = len(self._weights)
            target_weights = {name: 1.0 / n for name in self._weights}

        return target_weights

    def _clamp_weight_changes(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
    ) -> Tuple[Dict[str, float], bool]:
        """
        Clamp weight changes to safety limits.

        Returns (clamped_weights, was_clipped).
        """
        clamped = {}
        was_clipped = False

        for name in current:
            if name not in target:
                clamped[name] = current[name]
                continue

            current_w = current[name]
            target_w = target[name]
            delta = target_w - current_w

            # Clamp delta to max daily shift
            if abs(delta) > self.config.max_daily_shift:
                delta = self.config.max_daily_shift * (1 if delta > 0 else -1)
                was_clipped = True

            new_weight = current_w + delta

            # Clamp to min/max bounds
            new_weight = max(self.config.min_weight, min(self.config.max_weight, new_weight))
            if new_weight != current_w + delta:
                was_clipped = True

            clamped[name] = new_weight

        # Renormalize to sum to 1.0
        total = sum(clamped.values())
        if abs(total - 1.0) > 0.001:
            clamped = {name: w / total for name, w in clamped.items()}

        return clamped, was_clipped

    def recommend_weights(
        self,
        regime: str,
        rl_preferences: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Recommend new strategy weights.

        This method ONLY returns a recommendation.
        It does NOT apply the weights or execute any trades.

        Args:
            regime: Current market regime
            rl_preferences: Strategy preferences from RL system
            force: Bypass cooldown check (for testing only)

        Returns:
            Recommended weights if update is needed, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check cooldown
        if not force and self._last_update:
            hours_since = (datetime.now() - self._last_update).total_seconds() / 3600
            if hours_since < self.config.cooldown_hours:
                logger.debug(
                    f"Weight update skipped: cooldown ({hours_since:.1f}h < "
                    f"{self.config.cooldown_hours}h)"
                )
                return None

        # Calculate target weights
        target = self._calculate_target_weights(regime, rl_preferences)

        # Clamp changes
        clamped, was_clipped = self._clamp_weight_changes(self._weights, target)

        # Check if change is significant
        max_change = max(
            abs(clamped[name] - self._weights[name])
            for name in self._weights
        )

        if max_change < 0.01:  # Less than 1% change
            logger.debug("Weight update skipped: insignificant change")
            return None

        logger.info(
            f"Weight recommendation: regime={regime}, "
            f"max_change={max_change:.2%}, clipped={was_clipped}"
        )

        return clamped

    def apply_weights(
        self,
        new_weights: Dict[str, float],
        regime: str,
        rl_preferences: Optional[Dict[str, float]] = None,
        reason: str = "scheduled_update",
    ):
        """
        Apply new weights (must be called explicitly by strategy selector).

        This is separated from recommend_weights to ensure explicit
        application by the calling code.
        """
        if not self.config.enabled:
            return

        # Validate weights
        if not self._validate_weights(new_weights):
            logger.error("Invalid weights rejected")
            return

        # Record update
        update = WeightUpdate(
            timestamp=datetime.now(),
            old_weights=dict(self._weights),
            new_weights=dict(new_weights),
            reason=reason,
            regime=regime,
            rl_preferences=rl_preferences or {},
            clipped=False,
        )

        # Apply
        self._weights = dict(new_weights)
        self._last_update = datetime.now()
        self._update_history.append(update)

        # Persist
        self._save_state()

        logger.info(f"Weights applied: {new_weights}")

    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate weight constraints."""
        if not weights:
            return False

        # Check bounds
        for name, w in weights.items():
            if w < self.config.min_weight - 0.001:
                logger.warning(f"Weight {name}={w} below minimum")
                return False
            if w > self.config.max_weight + 0.001:
                logger.warning(f"Weight {name}={w} above maximum")
                return False

        # Check sum
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, not 1.0")
            return False

        return True

    def get_regime_adjusted_weights(
        self,
        regime: str,
    ) -> Dict[str, float]:
        """
        Get current weights with regime adjustment applied.

        This is a read-only method that shows what weights would be
        used for the given regime, without modifying stored weights.
        """
        weights = dict(self._weights)

        # Apply regime boosts
        boosts = self.config.regime_weight_boost.get(regime, {})
        for name, boost in boosts.items():
            if name in weights:
                weights[name] = min(
                    self.config.max_weight,
                    weights[name] + boost
                )

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {name: w / total for name, w in weights.items()}

        return weights

    def get_advisor_status(self) -> Dict[str, Any]:
        """Get current advisor status for monitoring."""
        return {
            "enabled": self.config.enabled,
            "current_weights": self._weights,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "updates_today": sum(
                1 for u in self._update_history
                if u.timestamp.date() == datetime.now().date()
            ),
            "performance_tracked": list(self._performance.keys()),
            "config": {
                "max_daily_shift": self.config.max_daily_shift,
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "cooldown_hours": self.config.cooldown_hours,
                "use_rl_preferences": self.config.use_rl_preferences,
                "rl_preference_weight": self.config.rl_preference_weight,
            },
        }

    def get_weight_change_analysis(self) -> Dict[str, Any]:
        """Analyze recent weight changes for reporting."""
        if not self._update_history:
            return {"message": "No weight changes recorded"}

        recent = [
            u for u in self._update_history
            if u.timestamp > datetime.now() - timedelta(days=7)
        ]

        if not recent:
            return {"message": "No weight changes in last 7 days"}

        # Calculate cumulative change per strategy
        cumulative_change = {name: 0.0 for name in self._weights}
        for update in recent:
            for name in cumulative_change:
                if name in update.old_weights and name in update.new_weights:
                    cumulative_change[name] += (
                        update.new_weights[name] - update.old_weights[name]
                    )

        return {
            "period": "7_days",
            "total_updates": len(recent),
            "clipped_updates": sum(1 for u in recent if u.clipped),
            "cumulative_change": {
                name: round(change, 4)
                for name, change in cumulative_change.items()
            },
            "updates_by_regime": {
                regime: sum(1 for u in recent if u.regime == regime)
                for regime in set(u.regime for u in recent)
            },
        }

    def reset(self):
        """Reset to default weights."""
        self._weights = {
            "TrendFollower": 0.20,
            "MeanReversion": 0.20,
            "MomentumTrader": 0.20,
            "ShortSpecialist": 0.20,
            "Scalper": 0.20,
        }
        self._performance = {}
        self._update_history = []
        self._last_update = None
        self._save_state()
        logger.info("StrategyWeightingAdvisor reset to defaults")


# =============================================================================
# Singleton access
# =============================================================================

_advisor: Optional[StrategyWeightingAdvisor] = None


def get_strategy_weighting_advisor(
    config: Optional[WeightingConfig] = None
) -> StrategyWeightingAdvisor:
    """Get or create singleton StrategyWeightingAdvisor."""
    global _advisor
    if _advisor is None:
        _advisor = StrategyWeightingAdvisor(config)
    return _advisor


def reset_strategy_weighting_advisor():
    """Reset singleton advisor."""
    global _advisor
    if _advisor is not None:
        _advisor.reset()
    _advisor = None
