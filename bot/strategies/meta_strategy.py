"""
Meta Strategy - Dynamic strategy allocation and selection.

Allocates capital across multiple strategies based on performance,
market regime, and risk constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Strategy allocation methods."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"
    MEAN_VARIANCE = "mean_variance"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""

    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int
    avg_trade_return: float
    volatility: float
    recent_performance: float  # Last N trades
    regime_performance: Dict[str, float]  # Performance by regime
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "trade_count": self.trade_count,
            "avg_trade_return": round(self.avg_trade_return, 4),
            "volatility": round(self.volatility, 4),
            "recent_performance": round(self.recent_performance, 4),
            "regime_performance": {k: round(v, 4) for k, v in self.regime_performance.items()},
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def score(self) -> float:
        """Calculate overall strategy score."""
        return (
            self.sharpe_ratio * 0.3
            + self.win_rate * 0.2
            + (1 - self.max_drawdown) * 0.2
            + self.recent_performance * 0.3
        )


@dataclass
class StrategyAllocation:
    """Allocation to a strategy."""

    strategy_name: str
    weight: float
    capital_allocated: float
    reason: str
    constraints_applied: List[str]

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "weight": round(self.weight, 4),
            "capital_allocated": round(self.capital_allocated, 2),
            "reason": self.reason,
            "constraints_applied": self.constraints_applied,
        }


@dataclass
class MetaSignal:
    """Combined signal from meta strategy."""

    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    contributing_strategies: List[Dict[str, Any]]
    allocation: Dict[str, float]
    consensus_level: float
    primary_strategy: str
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "contributing_strategies": self.contributing_strategies,
            "allocation": {k: round(v, 4) for k, v in self.allocation.items()},
            "consensus_level": round(self.consensus_level, 4),
            "primary_strategy": self.primary_strategy,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetaStrategyConfig:
    """Meta strategy configuration."""

    # Allocation method
    allocation_method: AllocationMethod = AllocationMethod.ADAPTIVE

    # Constraints
    min_allocation: float = 0.05  # 5% minimum per strategy
    max_allocation: float = 0.40  # 40% maximum per strategy
    max_strategies: int = 5  # Maximum active strategies

    # Performance windows
    short_window_days: int = 7
    long_window_days: int = 30

    # Rebalancing
    rebalance_threshold: float = 0.1  # 10% drift triggers rebalance
    rebalance_frequency_hours: int = 24

    # Risk constraints
    max_correlation: float = 0.7  # Max correlation between strategies
    min_sharpe_for_allocation: float = 0.5

    # Voting settings
    consensus_threshold: float = 0.6  # 60% agreement for strong signal
    min_strategies_for_signal: int = 2


class MetaStrategy:
    """
    Dynamically allocate capital across multiple strategies.

    Features:
    - Performance-based allocation
    - Regime-aware strategy selection
    - Risk parity weighting
    - Momentum-based rebalancing
    - Strategy correlation management
    """

    def __init__(self, config: Optional[MetaStrategyConfig] = None):
        self.config = config or MetaStrategyConfig()
        self._strategies: Dict[str, Any] = {}
        self._performance: Dict[str, StrategyPerformance] = {}
        self._trade_history: Dict[str, List[Dict]] = {}
        self._allocations: Dict[str, float] = {}
        self._last_rebalance: datetime = datetime.now() - timedelta(hours=24)
        self._current_regime: str = "unknown"

    def register_strategy(self, name: str, strategy: Any):
        """Register a strategy for allocation."""
        self._strategies[name] = strategy
        self._trade_history[name] = []
        self._allocations[name] = 1.0 / len(self._strategies)
        logger.info(f"Registered strategy: {name}")

    def unregister_strategy(self, name: str):
        """Remove a strategy."""
        self._strategies.pop(name, None)
        self._trade_history.pop(name, None)
        self._allocations.pop(name, None)
        self._performance.pop(name, None)

    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        return_pct: float,
        regime: Optional[str] = None,
    ):
        """Record a completed trade for a strategy."""
        if strategy_name not in self._trade_history:
            self._trade_history[strategy_name] = []

        self._trade_history[strategy_name].append(
            {
                "pnl": pnl,
                "return_pct": return_pct,
                "regime": regime or self._current_regime,
                "timestamp": datetime.now(),
            }
        )

        # Update performance metrics
        self._update_performance(strategy_name)

    def set_regime(self, regime: str):
        """Set current market regime."""
        self._current_regime = regime

    def _update_performance(self, strategy_name: str):
        """Update performance metrics for a strategy."""
        trades = self._trade_history.get(strategy_name, [])

        if len(trades) < 5:
            return

        returns = [t["return_pct"] for t in trades]

        # Calculate metrics
        total_return = np.sum(returns)
        volatility = np.std(returns) if len(returns) > 1 else 0.01
        sharpe_ratio = np.mean(returns) / volatility * np.sqrt(252) if volatility > 0 else 0

        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns) if returns else 0

        gains = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = gains / losses if losses > 0 else 2.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Recent performance (last N trades)
        recent_n = min(10, len(returns))
        recent_performance = np.mean(returns[-recent_n:])

        # Performance by regime
        regime_performance = {}
        for regime in set(t["regime"] for t in trades if t["regime"]):
            regime_returns = [t["return_pct"] for t in trades if t["regime"] == regime]
            if regime_returns:
                regime_performance[regime] = np.mean(regime_returns)

        self._performance[strategy_name] = StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=len(trades),
            avg_trade_return=np.mean(returns),
            volatility=volatility,
            recent_performance=recent_performance,
            regime_performance=regime_performance,
        )

    def calculate_allocations(
        self,
        total_capital: float,
        method: Optional[AllocationMethod] = None,
    ) -> List[StrategyAllocation]:
        """
        Calculate strategy allocations.

        Args:
            total_capital: Total capital to allocate
            method: Allocation method (default: config method)

        Returns:
            List of StrategyAllocation objects
        """
        method = method or self.config.allocation_method
        allocations = []

        # Get strategies with sufficient performance data
        eligible = {
            name: perf
            for name, perf in self._performance.items()
            if perf.sharpe_ratio >= self.config.min_sharpe_for_allocation and perf.trade_count >= 10
        }

        if not eligible:
            # Equal weight fallback
            weight = 1.0 / len(self._strategies)
            for name in self._strategies:
                allocations.append(
                    StrategyAllocation(
                        strategy_name=name,
                        weight=weight,
                        capital_allocated=total_capital * weight,
                        reason="equal_weight_fallback",
                        constraints_applied=[],
                    )
                )
            return allocations

        # Calculate raw weights based on method
        if method == AllocationMethod.EQUAL_WEIGHT:
            raw_weights = {name: 1.0 for name in eligible}

        elif method == AllocationMethod.RISK_PARITY:
            raw_weights = self._risk_parity_weights(eligible)

        elif method == AllocationMethod.MOMENTUM:
            raw_weights = self._momentum_weights(eligible)

        elif method == AllocationMethod.MEAN_VARIANCE:
            raw_weights = self._mean_variance_weights(eligible)

        elif method == AllocationMethod.ADAPTIVE:
            raw_weights = self._adaptive_weights(eligible)

        else:
            raw_weights = {name: 1.0 for name in eligible}

        # Normalize and apply constraints
        total_weight = sum(raw_weights.values())
        normalized = {k: v / total_weight for k, v in raw_weights.items()}

        # Apply min/max constraints
        constrained = {}
        constraints_log = {}

        for name, weight in normalized.items():
            constraints = []

            if weight < self.config.min_allocation:
                weight = self.config.min_allocation
                constraints.append(f"min_allocation_{self.config.min_allocation}")
            elif weight > self.config.max_allocation:
                weight = self.config.max_allocation
                constraints.append(f"max_allocation_{self.config.max_allocation}")

            constrained[name] = weight
            constraints_log[name] = constraints

        # Re-normalize after constraints
        total = sum(constrained.values())
        final_weights = {k: v / total for k, v in constrained.items()}

        # Create allocation objects
        for name, weight in final_weights.items():
            perf = self._performance.get(name)
            reason = (
                f"{method.value}: sharpe={perf.sharpe_ratio:.2f}, recent={perf.recent_performance:.2%}"
                if perf
                else method.value
            )

            allocations.append(
                StrategyAllocation(
                    strategy_name=name,
                    weight=weight,
                    capital_allocated=total_capital * weight,
                    reason=reason,
                    constraints_applied=constraints_log.get(name, []),
                )
            )

        # Store allocations
        self._allocations = final_weights

        return allocations

    def _risk_parity_weights(self, eligible: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Calculate risk parity weights (equal risk contribution)."""
        weights = {}
        for name, perf in eligible.items():
            # Inverse volatility weighting
            if perf.volatility > 0:
                weights[name] = 1.0 / perf.volatility
            else:
                weights[name] = 1.0
        return weights

    def _momentum_weights(self, eligible: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Calculate momentum-based weights."""
        weights = {}
        for name, perf in eligible.items():
            # Weight by recent performance
            momentum_score = max(0, perf.recent_performance + 0.1)  # Offset to avoid negative
            weights[name] = momentum_score
        return weights

    def _mean_variance_weights(self, eligible: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """Calculate mean-variance optimized weights."""
        weights = {}
        for name, perf in eligible.items():
            # Sharpe ratio based allocation
            weights[name] = max(0.1, perf.sharpe_ratio)
        return weights

    def _adaptive_weights(self, eligible: Dict[str, StrategyPerformance]) -> Dict[str, float]:
        """
        Adaptive weights based on regime and multiple factors.

        Combines:
        - Sharpe ratio
        - Recent performance
        - Regime-specific performance
        - Win rate
        """
        weights = {}

        for name, perf in eligible.items():
            # Base score
            score = perf.score

            # Regime bonus
            if self._current_regime in perf.regime_performance:
                regime_perf = perf.regime_performance[self._current_regime]
                if regime_perf > 0:
                    score *= 1 + regime_perf

            # Recency bonus
            if perf.recent_performance > 0:
                score *= 1 + perf.recent_performance

            weights[name] = max(0.1, score)

        return weights

    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        hours_since = (datetime.now() - self._last_rebalance).total_seconds() / 3600

        if hours_since < self.config.rebalance_frequency_hours:
            return False

        # Check for significant drift
        # (Would need current positions to calculate actual drift)
        return True

    def combine_signals(
        self,
        strategy_signals: Dict[str, Dict[str, Any]],
    ) -> MetaSignal:
        """
        Combine signals from multiple strategies.

        Args:
            strategy_signals: Dict mapping strategy name to signal dict
                Each signal should have: action, confidence

        Returns:
            MetaSignal with combined decision
        """
        if not strategy_signals:
            return MetaSignal(
                action="FLAT",
                confidence=0,
                contributing_strategies=[],
                allocation=self._allocations,
                consensus_level=0,
                primary_strategy="",
                reasoning=["No strategy signals provided"],
            )

        # Collect votes weighted by allocation
        votes = {"LONG": 0, "SHORT": 0, "FLAT": 0}
        contributing = []
        reasoning = []

        for name, signal in strategy_signals.items():
            action = signal.get("action", "FLAT")
            confidence = signal.get("confidence", 0)
            weight = self._allocations.get(name, 0)

            if action in votes:
                weighted_vote = confidence * weight
                votes[action] += weighted_vote

                contributing.append(
                    {
                        "strategy": name,
                        "action": action,
                        "confidence": confidence,
                        "weight": weight,
                        "weighted_vote": weighted_vote,
                    }
                )

                reasoning.append(f"{name}: {action} ({confidence:.0%} conf, {weight:.0%} weight)")

        # Determine winner
        total_votes = sum(votes.values())
        if total_votes == 0:
            return MetaSignal(
                action="FLAT",
                confidence=0,
                contributing_strategies=contributing,
                allocation=self._allocations,
                consensus_level=0,
                primary_strategy="",
                reasoning=reasoning,
            )

        # Find winning action
        winning_action = max(votes, key=votes.get)
        winning_votes = votes[winning_action]
        consensus_level = winning_votes / total_votes

        # Calculate combined confidence
        combined_confidence = consensus_level * (winning_votes / len(strategy_signals))

        # Find primary strategy (highest weighted vote for winning action)
        primary = max(
            [c for c in contributing if c["action"] == winning_action],
            key=lambda x: x["weighted_vote"],
            default={"strategy": ""},
        )

        # Apply consensus threshold
        if consensus_level < self.config.consensus_threshold:
            winning_action = "FLAT"
            reasoning.append(
                f"Consensus too low ({consensus_level:.0%} < {self.config.consensus_threshold:.0%})"
            )

        # Check minimum strategies
        agreeing_strategies = sum(1 for c in contributing if c["action"] == winning_action)
        if agreeing_strategies < self.config.min_strategies_for_signal:
            winning_action = "FLAT"
            reasoning.append(
                f"Insufficient agreement ({agreeing_strategies} < {self.config.min_strategies_for_signal})"
            )

        return MetaSignal(
            action=winning_action,
            confidence=combined_confidence,
            contributing_strategies=contributing,
            allocation=self._allocations,
            consensus_level=consensus_level,
            primary_strategy=primary["strategy"],
            reasoning=reasoning,
        )

    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by performance."""
        rankings = []

        for name, perf in self._performance.items():
            rankings.append(
                {
                    "strategy": name,
                    "score": perf.score,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "win_rate": perf.win_rate,
                    "recent_performance": perf.recent_performance,
                    "allocation": self._allocations.get(name, 0),
                }
            )

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall meta strategy performance summary."""
        if not self._performance:
            return {"status": "insufficient_data"}

        # Portfolio-level metrics
        weighted_sharpe = sum(
            perf.sharpe_ratio * self._allocations.get(name, 0)
            for name, perf in self._performance.items()
        )

        weighted_win_rate = sum(
            perf.win_rate * self._allocations.get(name, 0)
            for name, perf in self._performance.items()
        )

        total_trades = sum(perf.trade_count for perf in self._performance.values())

        return {
            "strategy_count": len(self._strategies),
            "active_strategies": len(self._performance),
            "weighted_sharpe": round(weighted_sharpe, 4),
            "weighted_win_rate": round(weighted_win_rate, 4),
            "total_trades": total_trades,
            "current_regime": self._current_regime,
            "allocation_method": self.config.allocation_method.value,
            "last_rebalance": self._last_rebalance.isoformat(),
        }


def create_meta_strategy(config: Optional[MetaStrategyConfig] = None) -> MetaStrategy:
    """Factory function to create meta strategy."""
    return MetaStrategy(config=config)
