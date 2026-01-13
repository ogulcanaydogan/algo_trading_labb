"""
Regime-Based Position Manager.

Manages position sizing based on detected market regime.
The core principle: Stay invested, but adjust exposure based on risk.

Key features:
- Dynamic position sizing per regime
- Smooth transitions (no sudden 100% to 0% moves)
- Leverage management
- Integration with risk engine for safety
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .regime_detector import MarketRegime, RegimeState

logger = logging.getLogger(__name__)


class PositionAction(Enum):
    """Actions the position manager can recommend."""
    HOLD = "hold"
    INCREASE = "increase"
    DECREASE = "decrease"
    CLOSE_ALL = "close_all"


@dataclass
class PositionSizingConfig:
    """Configuration for regime-based position sizing."""

    # Target allocation per regime (as fraction of capital)
    regime_allocations: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.BULL: 1.0,      # 100% invested
        MarketRegime.SIDEWAYS: 0.8,  # 80% invested
        MarketRegime.BEAR: 0.4,      # 40% invested
        MarketRegime.HIGH_VOL: 0.5,  # 50% invested
        MarketRegime.CRASH: 0.1,     # 10% invested (mostly cash)
        MarketRegime.UNKNOWN: 0.3,   # 30% when uncertain
    })

    # Leverage settings per regime
    regime_leverage: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.BULL: 1.5,      # Can use 1.5x in bull
        MarketRegime.SIDEWAYS: 1.0,  # No leverage in sideways
        MarketRegime.BEAR: 1.0,      # No leverage in bear
        MarketRegime.HIGH_VOL: 0.8,  # Reduce exposure in high vol
        MarketRegime.CRASH: 0.5,     # Minimum exposure in crash
        MarketRegime.UNKNOWN: 0.5,   # Conservative when uncertain
    })

    # Transition settings
    max_position_change_per_hour: float = 0.10  # Max 10% change per hour
    min_position_change: float = 0.05  # Don't bother with <5% changes

    # Safety limits
    max_total_allocation: float = 1.5  # Never more than 150% (with leverage)
    min_total_allocation: float = 0.0  # Can go to 0% (full cash)

    # Rebalancing
    rebalance_threshold: float = 0.10  # Rebalance if off by 10%
    min_time_between_rebalances: int = 3600  # 1 hour minimum

    # Use leverage
    enable_leverage: bool = False  # Disabled by default for safety


@dataclass
class PositionRecommendation:
    """Recommendation from position manager."""

    target_allocation: float  # Target as fraction of capital
    current_allocation: float  # Current as fraction of capital
    change_needed: float  # Difference
    action: PositionAction
    regime: MarketRegime
    confidence: float
    leverage: float  # Recommended leverage

    # Execution details
    should_execute: bool  # Whether change is significant enough
    reason: str

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "target_allocation": self.target_allocation,
            "current_allocation": self.current_allocation,
            "change_needed": self.change_needed,
            "action": self.action.value,
            "regime": self.regime.value,
            "confidence": self.confidence,
            "leverage": self.leverage,
            "should_execute": self.should_execute,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class RegimePositionManager:
    """
    Manages position sizing based on market regime.

    This is the core of the regime-based trading approach:
    - Detects market regime
    - Calculates appropriate position size
    - Recommends gradual transitions
    - Respects risk limits
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()

        # State tracking (per-symbol)
        self._current_allocation: float = 0.0
        self._target_allocation: float = 0.0
        self._current_regime: Optional[MarketRegime] = None
        self._symbol_regimes: Dict[str, MarketRegime] = {}  # Per-symbol regime tracking
        self._last_rebalance: Optional[datetime] = None
        self._allocation_history: List[Dict] = []

        # Transition tracking
        self._transition_start_time: Optional[datetime] = None
        self._transition_start_allocation: float = 0.0
        self._transition_target_allocation: float = 0.0

    def update_current_allocation(self, allocation: float) -> None:
        """Update current allocation from actual portfolio state."""
        self._current_allocation = max(0.0, min(allocation, self.config.max_total_allocation))

    def get_recommendation(
        self,
        regime_state: RegimeState,
        current_equity: float,
        current_position_value: float,
    ) -> PositionRecommendation:
        """
        Get position sizing recommendation based on regime.

        Args:
            regime_state: Current detected regime
            current_equity: Total portfolio equity
            current_position_value: Current position value

        Returns:
            PositionRecommendation with action details
        """

        # Calculate current allocation
        if current_equity > 0:
            current_allocation = current_position_value / current_equity
        else:
            current_allocation = 0.0

        self._current_allocation = current_allocation

        # Get target allocation for this regime
        base_allocation = self.config.regime_allocations.get(
            regime_state.regime,
            self.config.regime_allocations[MarketRegime.UNKNOWN]
        )

        # Apply leverage if enabled
        if self.config.enable_leverage:
            leverage = self.config.regime_leverage.get(
                regime_state.regime,
                1.0
            )
        else:
            leverage = 1.0

        # Calculate target with leverage
        target_allocation = base_allocation * leverage

        # Apply confidence weighting (lower confidence = more conservative)
        if regime_state.confidence < 0.5:
            # Reduce target when uncertain
            target_allocation *= (0.5 + regime_state.confidence)

        # Clamp to limits
        target_allocation = max(
            self.config.min_total_allocation,
            min(target_allocation, self.config.max_total_allocation)
        )

        # Check if regime changed for this symbol (per-symbol tracking to prevent oscillation)
        symbol = regime_state.symbol or "default"
        previous_symbol_regime = self._symbol_regimes.get(symbol)
        regime_changed = previous_symbol_regime != regime_state.regime

        if regime_changed:
            logger.info(
                f"Regime change detected: {previous_symbol_regime} -> {regime_state.regime}"
            )
            self._symbol_regimes[symbol] = regime_state.regime

        # Update global current regime (for backwards compatibility)
        self._current_regime = regime_state.regime

        # Calculate change needed
        change_needed = target_allocation - current_allocation

        # Determine action
        if abs(change_needed) < self.config.min_position_change:
            action = PositionAction.HOLD
            should_execute = False
            reason = f"Change too small ({abs(change_needed)*100:.1f}% < {self.config.min_position_change*100:.1f}%)"
        elif target_allocation < 0.05:
            action = PositionAction.CLOSE_ALL
            should_execute = True
            reason = f"Target near zero in {regime_state.regime.value} regime"
        elif change_needed > 0:
            action = PositionAction.INCREASE
            should_execute = True
            reason = f"Increase for {regime_state.regime.value} regime"
        else:
            action = PositionAction.DECREASE
            should_execute = True
            reason = f"Decrease for {regime_state.regime.value} regime"

        # Check rebalance timing
        if should_execute and self._last_rebalance:
            time_since_rebalance = (datetime.now() - self._last_rebalance).total_seconds()
            if time_since_rebalance < self.config.min_time_between_rebalances:
                # Apply rate limiting
                max_change = self.config.max_position_change_per_hour * (time_since_rebalance / 3600)
                if abs(change_needed) > max_change:
                    # Limit the change
                    old_change = change_needed
                    change_needed = max_change if change_needed > 0 else -max_change
                    target_allocation = current_allocation + change_needed
                    reason += f" (rate limited: {old_change*100:.1f}% -> {change_needed*100:.1f}%)"

        # Update target
        self._target_allocation = target_allocation

        recommendation = PositionRecommendation(
            target_allocation=target_allocation,
            current_allocation=current_allocation,
            change_needed=change_needed,
            action=action,
            regime=regime_state.regime,
            confidence=regime_state.confidence,
            leverage=leverage,
            should_execute=should_execute,
            reason=reason,
        )

        # Log recommendation
        if should_execute:
            logger.info(
                f"Position recommendation: {action.value} | "
                f"Current: {current_allocation*100:.1f}% -> Target: {target_allocation*100:.1f}% | "
                f"Regime: {regime_state.regime.value} | {reason}"
            )

        return recommendation

    def calculate_position_size(
        self,
        recommendation: PositionRecommendation,
        current_equity: float,
        asset_price: float,
    ) -> Tuple[float, float]:
        """
        Calculate actual position size from recommendation.

        Args:
            recommendation: Position recommendation
            current_equity: Total equity
            asset_price: Current asset price

        Returns:
            Tuple of (target_quantity, current_quantity)
        """

        target_value = current_equity * recommendation.target_allocation
        current_value = current_equity * recommendation.current_allocation

        target_quantity = target_value / asset_price if asset_price > 0 else 0
        current_quantity = current_value / asset_price if asset_price > 0 else 0

        return target_quantity, current_quantity

    def record_rebalance(self, recommendation: PositionRecommendation) -> None:
        """Record that a rebalance was executed."""
        self._last_rebalance = datetime.now()
        self._current_allocation = recommendation.target_allocation

        self._allocation_history.append({
            "timestamp": datetime.now().isoformat(),
            "regime": recommendation.regime.value,
            "allocation": recommendation.target_allocation,
            "action": recommendation.action.value,
        })

        # Keep last 1000 entries
        if len(self._allocation_history) > 1000:
            self._allocation_history = self._allocation_history[-1000:]

    def get_status(self) -> Dict:
        """Get current status of position manager."""
        return {
            "current_allocation": self._current_allocation,
            "target_allocation": self._target_allocation,
            "current_regime": self._current_regime.value if self._current_regime else None,
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "config": {
                "regime_allocations": {k.value: v for k, v in self.config.regime_allocations.items()},
                "leverage_enabled": self.config.enable_leverage,
                "max_change_per_hour": self.config.max_position_change_per_hour,
            },
            "history_length": len(self._allocation_history),
        }

    def get_allocation_for_regime(self, regime: MarketRegime) -> float:
        """Get target allocation for a specific regime."""
        base = self.config.regime_allocations.get(regime, 0.5)
        if self.config.enable_leverage:
            leverage = self.config.regime_leverage.get(regime, 1.0)
            return base * leverage
        return base


@dataclass
class MultiAssetAllocation:
    """Allocation recommendation for multiple assets."""

    allocations: Dict[str, float]  # symbol -> allocation fraction
    total_allocation: float
    regime: MarketRegime
    timestamp: datetime = field(default_factory=datetime.now)


class MultiAssetPositionManager(RegimePositionManager):
    """
    Position manager for multiple assets.

    Distributes capital across multiple assets based on regime.
    """

    def __init__(
        self,
        config: Optional[PositionSizingConfig] = None,
        asset_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(config)

        # Default equal weights
        self.asset_weights = asset_weights or {}
        self._asset_allocations: Dict[str, float] = {}

    def set_asset_weights(self, weights: Dict[str, float]) -> None:
        """Set relative weights for each asset."""
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            self.asset_weights = {k: v/total for k, v in weights.items()}
        else:
            self.asset_weights = weights

    def get_multi_asset_recommendation(
        self,
        regime_state: RegimeState,
        current_equity: float,
        current_positions: Dict[str, float],  # symbol -> position value
    ) -> MultiAssetAllocation:
        """
        Get allocation recommendation for multiple assets.

        Args:
            regime_state: Current regime
            current_equity: Total portfolio equity
            current_positions: Current position values per asset

        Returns:
            MultiAssetAllocation with per-asset allocations
        """

        # Get total allocation for regime
        total_position_value = sum(current_positions.values())

        recommendation = self.get_recommendation(
            regime_state,
            current_equity,
            total_position_value,
        )

        target_total = recommendation.target_allocation

        # Distribute across assets by weight
        allocations = {}
        for symbol, weight in self.asset_weights.items():
            allocations[symbol] = target_total * weight

        return MultiAssetAllocation(
            allocations=allocations,
            total_allocation=target_total,
            regime=regime_state.regime,
        )
