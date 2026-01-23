"""
Meta-Allocator - Capital Allocation Across Strategies

The "fund manager" layer that allocates capital to strategies based on:
1. Recent risk-adjusted performance (per regime)
2. Correlation penalties (diversification)
3. Drawdown and tail risk
4. Regime-specific weights

This is what makes "always something working" possible - by dynamically
shifting capital to the best-performing strategies for current conditions.

Allocation methods:
- Equal Weight: Simple 1/N allocation
- Risk Parity: Weight by inverse volatility
- Performance-Based: Weight by recent Sharpe/returns
- Contextual Bandit: Thompson Sampling with regime context
- Kelly-Based: Fractional Kelly with constraints
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Capital allocation methods"""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"  # Follow recent performance
    MEAN_REVERSION = "mean_reversion"  # Bet on underperformers
    KELLY = "kelly"
    CONTEXTUAL_BANDIT = "contextual_bandit"
    REGIME_ADAPTIVE = "regime_adaptive"


@dataclass
class StrategyAllocation:
    """Allocation result for a single strategy"""

    strategy_name: str
    weight: float  # 0.0 to 1.0
    capital_amount: float
    max_leverage: float = 1.0

    # Reasoning
    base_weight: float = 0.0
    regime_adjustment: float = 0.0
    correlation_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    final_weight: float = 0.0

    # Constraints
    hit_max_weight: bool = False
    hit_min_weight: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "weight": round(self.weight, 4),
            "capital_amount": round(self.capital_amount, 2),
            "max_leverage": self.max_leverage,
            "base_weight": round(self.base_weight, 4),
            "regime_adjustment": round(self.regime_adjustment, 4),
            "correlation_penalty": round(self.correlation_penalty, 4),
            "drawdown_penalty": round(self.drawdown_penalty, 4),
        }


@dataclass
class AllocationResult:
    """Complete allocation result"""

    timestamp: datetime = field(default_factory=datetime.now)
    method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    total_capital: float = 0.0
    current_regime: str = "unknown"

    # Per-strategy allocations
    allocations: List[StrategyAllocation] = field(default_factory=list)

    # Portfolio-level metrics
    effective_strategies: int = 0
    concentration_ratio: float = 0.0  # HHI
    expected_sharpe: float = 0.0

    # Rebalance info
    needs_rebalance: bool = False
    rebalance_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "method": self.method.value,
            "total_capital": self.total_capital,
            "current_regime": self.current_regime,
            "allocations": [a.to_dict() for a in self.allocations],
            "effective_strategies": self.effective_strategies,
            "concentration_ratio": round(self.concentration_ratio, 4),
            "expected_sharpe": round(self.expected_sharpe, 3),
            "needs_rebalance": self.needs_rebalance,
            "rebalance_reason": self.rebalance_reason,
        }


@dataclass
class StrategyPerformanceSnapshot:
    """Performance snapshot for allocation decisions"""

    strategy_name: str

    # Returns
    returns_1d: float = 0.0
    returns_7d: float = 0.0
    returns_30d: float = 0.0

    # Risk metrics
    volatility_30d: float = 0.0
    sharpe_30d: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    # Trade quality
    win_rate_30d: float = 0.0
    profit_factor_30d: float = 0.0

    # Status
    is_degraded: bool = False
    consecutive_losses: int = 0

    # Regime performance
    regime_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AllocationConfig:
    """Configuration for the meta-allocator"""

    # Allocation method
    method: AllocationMethod = AllocationMethod.REGIME_ADAPTIVE

    # Weight constraints
    min_weight: float = 0.05  # Minimum 5% per strategy
    max_weight: float = 0.40  # Maximum 40% per strategy
    min_strategies: int = 2  # Minimum strategies to allocate to

    # Performance lookback
    lookback_days: int = 30
    ema_halflife_days: float = 7.0  # For exponential weighting

    # Risk parameters
    target_volatility: float = 0.15  # 15% annual
    max_correlation_penalty: float = 0.3
    drawdown_penalty_factor: float = 2.0

    # Rebalance triggers
    rebalance_threshold: float = 0.10  # 10% drift triggers rebalance
    min_rebalance_interval_hours: int = 24

    # Kelly parameters
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)

    # Regime-specific
    regime_weight_boost: float = 0.3  # Boost for strategies suited to regime


class MetaAllocator:
    """
    Capital allocator across multiple strategies.

    Implements the "fund manager" layer that shifts capital to
    the best strategies for current market conditions.
    """

    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()

        # Performance history per strategy
        self.performance_history: Dict[str, List[StrategyPerformanceSnapshot]] = {}

        # Correlation matrix (estimated from returns)
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

        # Last allocation
        self.last_allocation: Optional[AllocationResult] = None
        self.last_allocation_time: Optional[datetime] = None

        # Thompson sampling state (for contextual bandit)
        self.bandit_state: Dict[str, Dict[str, float]] = {}  # strategy -> {alpha, beta}

        logger.info(f"Meta-Allocator initialized with method: {self.config.method.value}")

    def allocate(
        self,
        strategies: List[Dict[str, Any]],
        total_capital: float,
        current_regime: str = "unknown",
        performance_snapshots: Optional[List[StrategyPerformanceSnapshot]] = None,
    ) -> AllocationResult:
        """
        Calculate optimal capital allocation across strategies.

        Args:
            strategies: List of strategy info dicts with name, performance, etc.
            total_capital: Total capital to allocate
            current_regime: Current market regime
            performance_snapshots: Recent performance data per strategy

        Returns:
            AllocationResult with per-strategy allocations
        """
        if not strategies:
            return AllocationResult(total_capital=total_capital)

        # Build performance snapshots if not provided
        if performance_snapshots is None:
            performance_snapshots = self._build_performance_snapshots(strategies)

        # Store for history
        for snapshot in performance_snapshots:
            if snapshot.strategy_name not in self.performance_history:
                self.performance_history[snapshot.strategy_name] = []
            self.performance_history[snapshot.strategy_name].append(snapshot)

        # Calculate raw weights based on method
        if self.config.method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(performance_snapshots)
        elif self.config.method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity(performance_snapshots)
        elif self.config.method == AllocationMethod.MOMENTUM:
            weights = self._momentum_weight(performance_snapshots)
        elif self.config.method == AllocationMethod.MEAN_REVERSION:
            weights = self._mean_reversion_weight(performance_snapshots)
        elif self.config.method == AllocationMethod.KELLY:
            weights = self._kelly_weight(performance_snapshots)
        elif self.config.method == AllocationMethod.CONTEXTUAL_BANDIT:
            weights = self._contextual_bandit_weight(performance_snapshots, current_regime)
        elif self.config.method == AllocationMethod.REGIME_ADAPTIVE:
            weights = self._regime_adaptive_weight(performance_snapshots, current_regime)
        else:
            weights = self._equal_weight(performance_snapshots)

        # Apply adjustments
        adjusted_weights = self._apply_adjustments(weights, performance_snapshots, current_regime)

        # Apply constraints
        final_weights = self._apply_constraints(adjusted_weights)

        # Build result
        result = AllocationResult(
            method=self.config.method,
            total_capital=total_capital,
            current_regime=current_regime,
        )

        for name, weight in final_weights.items():
            snapshot = next((s for s in performance_snapshots if s.strategy_name == name), None)

            alloc = StrategyAllocation(
                strategy_name=name,
                weight=weight,
                capital_amount=total_capital * weight,
                base_weight=weights.get(name, 0),
                regime_adjustment=adjusted_weights.get(name, 0) - weights.get(name, 0),
                final_weight=weight,
            )
            result.allocations.append(alloc)

        # Calculate portfolio metrics
        result.effective_strategies = sum(1 for a in result.allocations if a.weight > 0.01)
        result.concentration_ratio = self._calculate_hhi(final_weights)

        # Check if rebalance needed
        result.needs_rebalance, result.rebalance_reason = self._check_rebalance_needed(result)

        # Store
        self.last_allocation = result
        self.last_allocation_time = datetime.now()

        logger.info(
            f"Allocated to {result.effective_strategies} strategies, HHI={result.concentration_ratio:.3f}"
        )

        return result

    def _build_performance_snapshots(
        self, strategies: List[Dict[str, Any]]
    ) -> List[StrategyPerformanceSnapshot]:
        """Build performance snapshots from strategy dicts"""
        snapshots = []
        for s in strategies:
            perf = s.get("performance", {})
            snapshot = StrategyPerformanceSnapshot(
                strategy_name=s.get("name", "unknown"),
                returns_30d=perf.get("total_pnl_pct", 0),
                volatility_30d=perf.get("volatility", 0.02),
                sharpe_30d=perf.get("sharpe_ratio", 0),
                max_drawdown=perf.get("max_drawdown", 0),
                current_drawdown=perf.get("current_drawdown", 0),
                win_rate_30d=perf.get("win_rate", 0.5),
                profit_factor_30d=perf.get("profit_factor", 1.0),
                is_degraded=perf.get("is_degraded", False),
                consecutive_losses=perf.get("consecutive_losses", 0),
                regime_scores=s.get("regime_scores", {}),
            )
            snapshots.append(snapshot)
        return snapshots

    def _equal_weight(self, snapshots: List[StrategyPerformanceSnapshot]) -> Dict[str, float]:
        """Simple equal weight allocation"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            active = snapshots

        weight = 1.0 / len(active) if active else 0
        return {s.strategy_name: weight for s in active}

    def _risk_parity(self, snapshots: List[StrategyPerformanceSnapshot]) -> Dict[str, float]:
        """Risk parity: weight by inverse volatility"""
        active = [s for s in snapshots if not s.is_degraded and s.volatility_30d > 0]
        if not active:
            return self._equal_weight(snapshots)

        # Inverse volatility weights
        inv_vols = {s.strategy_name: 1.0 / s.volatility_30d for s in active}
        total_inv_vol = sum(inv_vols.values())

        return {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}

    def _momentum_weight(self, snapshots: List[StrategyPerformanceSnapshot]) -> Dict[str, float]:
        """Momentum: weight by recent performance"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            return self._equal_weight(snapshots)

        # Use exponential moving average of returns
        scores = {}
        for s in active:
            # Combine multiple timeframes
            score = s.returns_1d * 0.2 + s.returns_7d * 0.3 + s.returns_30d * 0.5
            # Adjust for Sharpe
            if s.sharpe_30d > 0:
                score *= 1 + s.sharpe_30d * 0.2
            scores[s.strategy_name] = max(0.01, score + 10)  # Shift to positive

        total = sum(scores.values())
        return {name: score / total for name, score in scores.items()}

    def _mean_reversion_weight(
        self, snapshots: List[StrategyPerformanceSnapshot]
    ) -> Dict[str, float]:
        """Mean reversion: bet on underperformers (contrarian)"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            return self._equal_weight(snapshots)

        # Inverse of recent performance (but not for badly degraded)
        scores = {}
        avg_return = np.mean([s.returns_30d for s in active])

        for s in active:
            # Underperformers get higher weight (but not if deeply negative)
            deviation = avg_return - s.returns_30d
            if s.returns_30d > -10:  # Not catastrophic
                score = max(0.1, 1 + deviation * 0.1)
            else:
                score = 0.1
            scores[s.strategy_name] = score

        total = sum(scores.values())
        return {name: score / total for name, score in scores.items()}

    def _kelly_weight(self, snapshots: List[StrategyPerformanceSnapshot]) -> Dict[str, float]:
        """Kelly criterion-based weighting"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            return self._equal_weight(snapshots)

        kelly_weights = {}
        for s in active:
            # Kelly = (p * b - q) / b
            # where p = win rate, q = 1-p, b = avg_win/avg_loss
            p = s.win_rate_30d
            q = 1 - p

            # Estimate b from profit factor
            if s.profit_factor_30d > 0 and s.win_rate_30d > 0:
                # PF = (p * avg_win) / (q * avg_loss)
                # b = avg_win/avg_loss = PF * q / p
                b = s.profit_factor_30d * q / p if p > 0 else 1
            else:
                b = 1

            kelly = (p * b - q) / b if b > 0 else 0
            kelly = max(0, kelly * self.config.kelly_fraction)  # Fractional Kelly

            kelly_weights[s.strategy_name] = kelly

        # Normalize
        total = sum(kelly_weights.values())
        if total <= 0:
            return self._equal_weight(snapshots)

        return {name: w / total for name, w in kelly_weights.items()}

    def _contextual_bandit_weight(
        self, snapshots: List[StrategyPerformanceSnapshot], current_regime: str
    ) -> Dict[str, float]:
        """Thompson Sampling with regime context"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            return self._equal_weight(snapshots)

        samples = {}
        for s in active:
            name = s.strategy_name
            key = f"{name}_{current_regime}"

            # Initialize bandit state if needed
            if key not in self.bandit_state:
                self.bandit_state[key] = {"alpha": 1.0, "beta": 1.0}

            state = self.bandit_state[key]

            # Thompson sample from Beta distribution
            sample = np.random.beta(state["alpha"], state["beta"])

            # Adjust by regime score if available
            regime_score = s.regime_scores.get(current_regime, 0.5)
            sample *= 0.5 + regime_score

            samples[name] = sample

        # Normalize
        total = sum(samples.values())
        return {name: sample / total for name, sample in samples.items()}

    def update_bandit(self, strategy_name: str, regime: str, reward: float):
        """Update bandit state after observing reward"""
        key = f"{strategy_name}_{regime}"
        if key not in self.bandit_state:
            self.bandit_state[key] = {"alpha": 1.0, "beta": 1.0}

        # Normalize reward to [0, 1]
        norm_reward = min(1, max(0, (reward + 5) / 10))  # -5% to +5% -> 0 to 1

        # Update Beta distribution parameters
        self.bandit_state[key]["alpha"] += norm_reward
        self.bandit_state[key]["beta"] += 1 - norm_reward

    def _regime_adaptive_weight(
        self, snapshots: List[StrategyPerformanceSnapshot], current_regime: str
    ) -> Dict[str, float]:
        """Regime-adaptive allocation (recommended default)"""
        active = [s for s in snapshots if not s.is_degraded]
        if not active:
            return self._equal_weight(snapshots)

        weights = {}
        for s in active:
            # Base weight from risk-adjusted performance
            base = 1.0
            if s.sharpe_30d > 0:
                base *= 1 + s.sharpe_30d * 0.3
            if s.profit_factor_30d > 1:
                base *= 1 + (s.profit_factor_30d - 1) * 0.2

            # Regime boost
            regime_score = s.regime_scores.get(current_regime, 0.5)
            regime_boost = 1 + (regime_score - 0.5) * self.config.regime_weight_boost * 2

            # Drawdown penalty
            dd_penalty = 1 - (s.current_drawdown / 100) * 0.5

            # Consecutive loss penalty
            loss_penalty = 1 - (s.consecutive_losses * 0.1)

            weight = max(0.01, base * regime_boost * dd_penalty * loss_penalty)
            weights[s.strategy_name] = weight

        # Normalize
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}

    def _apply_adjustments(
        self,
        weights: Dict[str, float],
        snapshots: List[StrategyPerformanceSnapshot],
        current_regime: str,
    ) -> Dict[str, float]:
        """Apply correlation and risk adjustments"""
        adjusted = weights.copy()

        # Correlation penalty (reduce weight for correlated strategies)
        # In practice, would use actual return correlations
        # Here we use a simplified approach based on strategy types
        strategy_groups = self._group_strategies(snapshots)

        for group, members in strategy_groups.items():
            if len(members) > 1:
                # Reduce weight for each strategy in a correlated group
                penalty = self.config.max_correlation_penalty * (len(members) - 1) / len(members)
                for name in members:
                    if name in adjusted:
                        adjusted[name] *= 1 - penalty

        # Drawdown penalty
        for s in snapshots:
            if s.strategy_name in adjusted:
                if s.current_drawdown > 5:
                    dd_factor = (
                        1 - (s.current_drawdown - 5) / 100 * self.config.drawdown_penalty_factor
                    )
                    adjusted[s.strategy_name] *= max(0.1, dd_factor)

        # Re-normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {name: w / total for name, w in adjusted.items()}

        return adjusted

    def _group_strategies(
        self, snapshots: List[StrategyPerformanceSnapshot]
    ) -> Dict[str, List[str]]:
        """Group strategies by type for correlation estimation"""
        groups = {
            "trend": [],
            "reversion": [],
            "momentum": [],
            "breakout": [],
            "other": [],
        }

        for s in snapshots:
            name_lower = s.strategy_name.lower()
            if "trend" in name_lower or "ema" in name_lower:
                groups["trend"].append(s.strategy_name)
            elif "reversion" in name_lower or "mean" in name_lower:
                groups["reversion"].append(s.strategy_name)
            elif "momentum" in name_lower or "macd" in name_lower:
                groups["momentum"].append(s.strategy_name)
            elif "breakout" in name_lower:
                groups["breakout"].append(s.strategy_name)
            else:
                groups["other"].append(s.strategy_name)

        return {k: v for k, v in groups.items() if v}

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints"""
        constrained = weights.copy()

        # Apply min/max
        for name in list(constrained.keys()):
            if constrained[name] < self.config.min_weight:
                constrained[name] = 0  # Below min -> remove
            elif constrained[name] > self.config.max_weight:
                constrained[name] = self.config.max_weight

        # Remove strategies below minimum
        constrained = {k: v for k, v in constrained.items() if v > 0}

        # Ensure minimum number of strategies
        if (
            len(constrained) < self.config.min_strategies
            and len(weights) >= self.config.min_strategies
        ):
            # Add back top strategies until we have minimum
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for name, w in sorted_weights:
                if name not in constrained:
                    constrained[name] = self.config.min_weight
                if len(constrained) >= self.config.min_strategies:
                    break

        # Re-normalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {name: w / total for name, w in constrained.items()}

        return constrained

    def _calculate_hhi(self, weights: Dict[str, float]) -> float:
        """Calculate Herfindahl-Hirschman Index (concentration)"""
        return sum(w**2 for w in weights.values())

    def _check_rebalance_needed(self, result: AllocationResult) -> Tuple[bool, str]:
        """Check if portfolio needs rebalancing"""
        if self.last_allocation is None:
            return False, ""

        # Check time since last rebalance
        if self.last_allocation_time:
            hours_since = (datetime.now() - self.last_allocation_time).total_seconds() / 3600
            if hours_since < self.config.min_rebalance_interval_hours:
                return False, "Too soon since last rebalance"

        # Check drift
        max_drift = 0.0
        for alloc in result.allocations:
            old_alloc = next(
                (
                    a
                    for a in self.last_allocation.allocations
                    if a.strategy_name == alloc.strategy_name
                ),
                None,
            )
            if old_alloc:
                drift = abs(alloc.weight - old_alloc.weight)
                max_drift = max(max_drift, drift)

        if max_drift > self.config.rebalance_threshold:
            return True, f"Weight drift of {max_drift:.1%} exceeds threshold"

        # Check regime change
        if result.current_regime != self.last_allocation.current_regime:
            return (
                True,
                f"Regime changed from {self.last_allocation.current_regime} to {result.current_regime}",
            )

        return False, ""

    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocation"""
        if self.last_allocation is None:
            return {"status": "no_allocation"}

        return {
            "timestamp": self.last_allocation.timestamp.isoformat(),
            "method": self.last_allocation.method.value,
            "regime": self.last_allocation.current_regime,
            "total_capital": self.last_allocation.total_capital,
            "num_strategies": self.last_allocation.effective_strategies,
            "concentration": self.last_allocation.concentration_ratio,
            "weights": {
                a.strategy_name: round(a.weight, 4) for a in self.last_allocation.allocations
            },
        }


# ============================================================
# Rebalancer Helper
# ============================================================


class PortfolioRebalancer:
    """
    Helper to execute rebalancing trades.

    Calculates the trades needed to move from current to target allocation.
    """

    def __init__(
        self,
        min_trade_pct: float = 0.5,  # Minimum trade size as % of portfolio
        max_trades_per_rebalance: int = 5,
    ):
        self.min_trade_pct = min_trade_pct
        self.max_trades_per_rebalance = max_trades_per_rebalance

    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_capital: float,
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance.

        Returns list of trades with strategy, direction, amount.
        """
        trades = []

        # Calculate differences
        all_strategies = set(current_weights.keys()) | set(target_weights.keys())

        for strategy in all_strategies:
            current = current_weights.get(strategy, 0)
            target = target_weights.get(strategy, 0)
            diff = target - current

            trade_amount = diff * total_capital
            trade_pct = abs(diff) * 100

            if trade_pct >= self.min_trade_pct:
                trades.append(
                    {
                        "strategy": strategy,
                        "direction": "increase" if diff > 0 else "decrease",
                        "current_weight": current,
                        "target_weight": target,
                        "weight_change": diff,
                        "capital_change": trade_amount,
                        "priority": trade_pct,  # Larger changes first
                    }
                )

        # Sort by priority (largest changes first)
        trades.sort(key=lambda x: x["priority"], reverse=True)

        # Limit number of trades
        return trades[: self.max_trades_per_rebalance]


# Global instance
_meta_allocator: Optional[MetaAllocator] = None


def get_meta_allocator() -> MetaAllocator:
    """Get or create global meta-allocator instance"""
    global _meta_allocator
    if _meta_allocator is None:
        _meta_allocator = MetaAllocator()
    return _meta_allocator


__all__ = [
    # Enums
    "AllocationMethod",
    # Data classes
    "StrategyAllocation",
    "AllocationResult",
    "StrategyPerformanceSnapshot",
    "AllocationConfig",
    # Main classes
    "MetaAllocator",
    "PortfolioRebalancer",
    # Factory
    "get_meta_allocator",
]
