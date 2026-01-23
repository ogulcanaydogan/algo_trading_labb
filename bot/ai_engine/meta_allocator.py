"""
Meta-Allocator - Strategy Selection and Capital Allocation

Intelligently allocates capital across multiple strategies based on:
- Historical performance per regime
- Current market conditions
- Diversification requirements
- Risk budgets

Acts as a "fund of strategies" manager.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

from .learning_db import LearningDatabase, get_learning_db
from .online_learner import OnlineLearner, get_online_learner

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy."""

    strategy_id: str
    weight: float  # 0.0 to 1.0
    regime: str
    reason: str
    expected_sharpe: float
    is_active: bool


@dataclass
class AllocationPlan:
    """Complete allocation plan across strategies."""

    timestamp: str
    total_capital: float
    regime: str
    allocations: List[StrategyAllocation]
    diversification_score: float
    expected_portfolio_sharpe: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_capital": self.total_capital,
            "regime": self.regime,
            "allocations": [
                {
                    "strategy_id": a.strategy_id,
                    "weight": round(a.weight, 4),
                    "regime": a.regime,
                    "reason": a.reason,
                    "expected_sharpe": round(a.expected_sharpe, 3),
                    "is_active": a.is_active,
                }
                for a in self.allocations
            ],
            "diversification_score": round(self.diversification_score, 3),
            "expected_portfolio_sharpe": round(self.expected_portfolio_sharpe, 3),
        }


class MetaAllocator:
    """
    Meta-level allocator that manages multiple strategies.

    Features:
    1. Performance-based allocation (more capital to better strategies)
    2. Regime-adaptive allocation (different strategies for different regimes)
    3. Diversification constraints (don't put all eggs in one basket)
    4. Risk budgeting (limit drawdown exposure)
    5. Dynamic rebalancing (adjust as conditions change)
    """

    def __init__(
        self,
        db: LearningDatabase = None,
        online_learner: OnlineLearner = None,
        min_allocation: float = 0.05,  # Min 5% to any active strategy
        max_allocation: float = 0.40,  # Max 40% to any single strategy
        rebalance_threshold: float = 0.1,  # Rebalance if weights drift >10%
    ):
        self.db = db or get_learning_db()
        self.online_learner = online_learner or get_online_learner()
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.rebalance_threshold = rebalance_threshold

        # Strategy performance tracking
        self.strategy_metrics: Dict[str, Dict[str, float]] = {}

        # Current allocation
        self.current_plan: Optional[AllocationPlan] = None

        # Regime-strategy performance matrix
        self.regime_performance: Dict[str, Dict[str, float]] = {}

    def register_strategy(
        self,
        strategy_id: str,
        metrics: Dict[str, float],
    ):
        """
        Register a strategy with its performance metrics.

        Args:
            strategy_id: Unique strategy identifier
            metrics: Dict with keys: sharpe_ratio, win_rate, max_drawdown, profit_factor
        """
        self.strategy_metrics[strategy_id] = {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "win_rate": metrics.get("win_rate", 0.5),
            "max_drawdown": metrics.get("max_drawdown", 0.1),
            "profit_factor": metrics.get("profit_factor", 1.0),
            "trades": metrics.get("trades", 0),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Registered strategy {strategy_id}: "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
            f"WinRate={metrics.get('win_rate', 0):.1%}"
        )

    def update_regime_performance(
        self,
        strategy_id: str,
        regime: str,
        sharpe_ratio: float,
    ):
        """Update strategy's performance in a specific regime."""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}

        self.regime_performance[regime][strategy_id] = sharpe_ratio

    def calculate_allocation(
        self,
        current_regime: str,
        total_capital: float,
        risk_budget: float = 0.02,  # Max 2% daily risk
    ) -> AllocationPlan:
        """
        Calculate optimal allocation across strategies.

        Args:
            current_regime: Current market regime
            total_capital: Total capital to allocate
            risk_budget: Maximum acceptable daily risk (as fraction)

        Returns:
            AllocationPlan with strategy weights
        """
        if not self.strategy_metrics:
            logger.warning("No strategies registered")
            return self._empty_plan(total_capital, current_regime)

        # Get online learner recommendations
        recommendations = self.online_learner.get_strategy_recommendations(current_regime)

        # Score each strategy
        strategy_scores: Dict[str, float] = {}
        for strategy_id, metrics in self.strategy_metrics.items():
            score = self._calculate_strategy_score(
                strategy_id, metrics, current_regime, recommendations
            )
            strategy_scores[strategy_id] = score

        # Filter to positive scores only
        positive_strategies = {k: v for k, v in strategy_scores.items() if v > 0}

        if not positive_strategies:
            logger.warning("No strategies with positive scores")
            return self._empty_plan(total_capital, current_regime)

        # Normalize to weights
        total_score = sum(positive_strategies.values())
        raw_weights = {k: v / total_score for k, v in positive_strategies.items()}

        # Apply constraints
        constrained_weights = self._apply_constraints(raw_weights)

        # Adjust for risk budget
        final_weights = self._adjust_for_risk(constrained_weights, risk_budget)

        # Create allocation plan
        allocations = []
        for strategy_id, weight in final_weights.items():
            metrics = self.strategy_metrics.get(strategy_id, {})
            allocations.append(
                StrategyAllocation(
                    strategy_id=strategy_id,
                    weight=weight,
                    regime=current_regime,
                    reason=self._get_allocation_reason(strategy_id, current_regime, weight),
                    expected_sharpe=metrics.get("sharpe_ratio", 0),
                    is_active=weight >= self.min_allocation,
                )
            )

        # Calculate portfolio metrics
        diversification = self._calculate_diversification(final_weights)
        portfolio_sharpe = self._estimate_portfolio_sharpe(final_weights)

        plan = AllocationPlan(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_capital=total_capital,
            regime=current_regime,
            allocations=allocations,
            diversification_score=diversification,
            expected_portfolio_sharpe=portfolio_sharpe,
        )

        self.current_plan = plan

        logger.info(
            f"Allocation plan: {len(allocations)} strategies, "
            f"diversification={diversification:.2f}, "
            f"expected Sharpe={portfolio_sharpe:.2f}"
        )

        return plan

    def _calculate_strategy_score(
        self,
        strategy_id: str,
        metrics: Dict[str, float],
        regime: str,
        recommendations: List[Dict],
    ) -> float:
        """
        Calculate allocation score for a strategy.

        Higher score = more allocation.
        """
        score = 0.0

        # Base score from Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        score += sharpe * 2.0

        # Win rate bonus
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate > 0.55:
            score += (win_rate - 0.5) * 5.0

        # Drawdown penalty
        max_dd = metrics.get("max_drawdown", 0.1)
        score -= max_dd * 3.0

        # Profit factor bonus
        pf = metrics.get("profit_factor", 1.0)
        if pf > 1.2:
            score += (pf - 1.0) * 2.0

        # Regime-specific performance bonus
        if regime in self.regime_performance:
            regime_sharpe = self.regime_performance[regime].get(strategy_id, 0)
            score += regime_sharpe * 1.5

        # Online learner recommendation bonus
        for rec in recommendations:
            if rec.get("strategy_id") == strategy_id:
                if rec.get("is_healthy", True):
                    score += rec.get("sharpe", 0) * 1.0
                else:
                    score -= 2.0  # Penalty for degraded strategies

        # Minimum trades requirement
        trades = metrics.get("trades", 0)
        if trades < 10:
            score *= 0.5  # Reduce score for untested strategies

        return max(0, score)

    def _apply_constraints(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply min/max allocation constraints."""
        constrained = {}

        # First pass: apply max constraint
        excess = 0.0
        for strategy_id, weight in weights.items():
            if weight > self.max_allocation:
                excess += weight - self.max_allocation
                constrained[strategy_id] = self.max_allocation
            else:
                constrained[strategy_id] = weight

        # Redistribute excess
        if excess > 0:
            eligible = [k for k, v in constrained.items() if v < self.max_allocation]
            if eligible:
                per_strategy = excess / len(eligible)
                for strategy_id in eligible:
                    constrained[strategy_id] = min(
                        self.max_allocation, constrained[strategy_id] + per_strategy
                    )

        # Second pass: apply min constraint
        final = {}
        for strategy_id, weight in constrained.items():
            if weight < self.min_allocation:
                # Either bump to minimum or exclude
                if weight > self.min_allocation / 2:
                    final[strategy_id] = self.min_allocation
                # else: exclude (too small)
            else:
                final[strategy_id] = weight

        # Renormalize
        total = sum(final.values())
        if total > 0:
            final = {k: v / total for k, v in final.items()}

        return final

    def _adjust_for_risk(
        self,
        weights: Dict[str, float],
        risk_budget: float,
    ) -> Dict[str, float]:
        """Adjust weights to stay within risk budget."""
        # Calculate expected portfolio risk
        portfolio_risk = 0.0
        for strategy_id, weight in weights.items():
            metrics = self.strategy_metrics.get(strategy_id, {})
            strategy_risk = metrics.get("max_drawdown", 0.1) / 10  # Daily risk estimate
            portfolio_risk += weight * strategy_risk

        # If exceeds budget, scale down
        if portfolio_risk > risk_budget:
            scale_factor = risk_budget / portfolio_risk
            weights = {k: v * scale_factor for k, v in weights.items()}

            # Put remainder in "cash" (not allocated)
            total = sum(weights.values())
            if total < 1.0:
                logger.info(f"Risk adjustment: {1 - total:.1%} kept in cash")

        return weights

    def _calculate_diversification(
        self,
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate diversification score (0-1).

        1.0 = perfectly diversified
        0.0 = concentrated in one strategy
        """
        if not weights:
            return 0.0

        # Use Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights.values())

        # Convert to diversification score (1 - HHI)
        # Normalize so that equal weights = 1.0
        n = len(weights)
        if n <= 1:
            return 0.0

        min_hhi = 1.0 / n  # Perfect diversification
        max_hhi = 1.0  # Complete concentration

        diversification = 1 - (hhi - min_hhi) / (max_hhi - min_hhi)
        return max(0, min(1, diversification))

    def _estimate_portfolio_sharpe(
        self,
        weights: Dict[str, float],
    ) -> float:
        """Estimate portfolio Sharpe ratio."""
        if not weights:
            return 0.0

        # Weighted average Sharpe (simplified, ignores correlations)
        portfolio_sharpe = 0.0
        for strategy_id, weight in weights.items():
            metrics = self.strategy_metrics.get(strategy_id, {})
            strategy_sharpe = metrics.get("sharpe_ratio", 0)
            portfolio_sharpe += weight * strategy_sharpe

        return portfolio_sharpe

    def _get_allocation_reason(
        self,
        strategy_id: str,
        regime: str,
        weight: float,
    ) -> str:
        """Generate human-readable reason for allocation."""
        metrics = self.strategy_metrics.get(strategy_id, {})
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0.5)

        reasons = []

        if sharpe > 1.5:
            reasons.append(f"Strong Sharpe ({sharpe:.2f})")
        elif sharpe > 1.0:
            reasons.append(f"Good Sharpe ({sharpe:.2f})")

        if win_rate > 0.6:
            reasons.append(f"High win rate ({win_rate:.0%})")

        if regime in self.regime_performance:
            regime_sharpe = self.regime_performance[regime].get(strategy_id, 0)
            if regime_sharpe > 1.0:
                reasons.append(f"Performs well in {regime} regime")

        if not reasons:
            reasons.append("Diversification")

        return "; ".join(reasons)

    def _empty_plan(
        self,
        total_capital: float,
        regime: str,
    ) -> AllocationPlan:
        """Return empty allocation plan."""
        return AllocationPlan(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_capital=total_capital,
            regime=regime,
            allocations=[],
            diversification_score=0.0,
            expected_portfolio_sharpe=0.0,
        )

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Check if portfolio should be rebalanced.

        Returns:
            Tuple of (should_rebalance, reason)
        """
        if not self.current_plan:
            return True, "No current plan"

        target_weights = {a.strategy_id: a.weight for a in self.current_plan.allocations}

        # Check drift from target
        max_drift = 0.0
        drifted_strategy = ""

        for strategy_id, target in target_weights.items():
            current = current_weights.get(strategy_id, 0)
            drift = abs(current - target)

            if drift > max_drift:
                max_drift = drift
                drifted_strategy = strategy_id

        if max_drift > self.rebalance_threshold:
            return True, f"{drifted_strategy} drifted {max_drift:.1%} from target"

        return False, "Within tolerance"

    def get_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        total_capital: float,
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance to target allocation.

        Args:
            current_positions: Current position values by strategy
            total_capital: Total portfolio value

        Returns:
            List of trades to execute
        """
        if not self.current_plan:
            return []

        trades = []

        for allocation in self.current_plan.allocations:
            strategy_id = allocation.strategy_id
            target_value = allocation.weight * total_capital
            current_value = current_positions.get(strategy_id, 0)

            diff = target_value - current_value

            if abs(diff) > total_capital * 0.01:  # Min 1% trade size
                trades.append(
                    {
                        "strategy_id": strategy_id,
                        "action": "BUY" if diff > 0 else "SELL",
                        "amount": abs(diff),
                        "target_weight": allocation.weight,
                        "current_weight": current_value / total_capital if total_capital > 0 else 0,
                    }
                )

        return trades

    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        if not self.current_plan:
            return {
                "has_plan": False,
                "strategies_registered": len(self.strategy_metrics),
            }

        return {
            "has_plan": True,
            "plan_timestamp": self.current_plan.timestamp,
            "regime": self.current_plan.regime,
            "total_capital": self.current_plan.total_capital,
            "num_strategies": len(self.current_plan.allocations),
            "active_strategies": sum(1 for a in self.current_plan.allocations if a.is_active),
            "diversification_score": self.current_plan.diversification_score,
            "expected_sharpe": self.current_plan.expected_portfolio_sharpe,
            "allocations": self.current_plan.to_dict()["allocations"],
        }


# Global instance
_allocator: Optional[MetaAllocator] = None


def get_meta_allocator() -> MetaAllocator:
    """Get or create global meta allocator."""
    global _allocator
    if _allocator is None:
        _allocator = MetaAllocator()
    return _allocator
