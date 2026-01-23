"""
Regime-Aware Strategy Selector.

Automatically selects optimal trading strategies based on
current market regime and historical performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy categories."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    DEFENSIVE = "defensive"
    MARKET_MAKING = "market_making"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""

    strategy_id: str
    name: str
    strategy_type: StrategyType
    suitable_regimes: Set[MarketRegime]
    unsuitable_regimes: Set[MarketRegime]
    min_volatility: float = 0.0
    max_volatility: float = 1.0
    position_sizing_factor: float = 1.0  # Multiplier for position size
    risk_factor: float = 1.0  # Risk adjustment factor
    priority: int = 0  # Higher = preferred when tied

    def is_suitable_for_regime(self, regime: MarketRegime) -> bool:
        """Check if strategy is suitable for given regime."""
        if regime in self.unsuitable_regimes:
            return False
        if self.suitable_regimes and regime not in self.suitable_regimes:
            return False
        return True

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "suitable_regimes": [r.value for r in self.suitable_regimes],
            "unsuitable_regimes": [r.value for r in self.unsuitable_regimes],
            "position_sizing_factor": self.position_sizing_factor,
            "risk_factor": self.risk_factor,
            "priority": self.priority,
        }


@dataclass
class StrategyPerformance:
    """Historical performance metrics for a strategy."""

    strategy_id: str
    regime: MarketRegime
    total_return: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    trade_count: int
    avg_holding_period: float  # hours
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Calculate performance score."""
        # Weighted score combining metrics
        sharpe_component = min(2, max(-2, self.sharpe_ratio)) / 2  # Normalize to -1,1
        win_component = self.win_rate - 0.5  # Center around 0.5
        pf_component = min(1, (self.profit_factor - 1) / 2) if self.profit_factor > 0 else -1
        dd_component = -min(1, self.max_drawdown / 0.3)  # Penalize >30% drawdown

        return (
            sharpe_component * 0.4 + win_component * 0.2 + pf_component * 0.25 + dd_component * 0.15
        )

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "regime": self.regime.value,
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "trade_count": self.trade_count,
            "score": round(self.score, 4),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SelectionResult:
    """Result of strategy selection."""

    selected_strategies: List[str]
    regime: MarketRegime
    confidence: float
    reasoning: List[str]
    position_scale: float  # Overall position size adjustment
    risk_scale: float  # Overall risk adjustment
    excluded_strategies: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "selected_strategies": self.selected_strategies,
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "position_scale": round(self.position_scale, 4),
            "risk_scale": round(self.risk_scale, 4),
            "excluded_strategies": self.excluded_strategies,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SelectorConfig:
    """Strategy selector configuration."""

    # Selection parameters
    max_concurrent_strategies: int = 3
    min_trade_count_for_scoring: int = 10
    performance_lookback_days: int = 90
    regime_transition_cooldown_minutes: int = 15

    # Position sizing
    base_position_scale: float = 1.0
    max_position_scale: float = 1.5
    min_position_scale: float = 0.3

    # Risk adjustments
    risk_off_scale: float = 0.5  # Scale during risk-off regimes
    crash_scale: float = 0.2  # Scale during crash


class RegimeStrategySelector:
    """
    Selects optimal strategies based on market regime.

    Features:
    - Regime-based strategy filtering
    - Performance-based ranking
    - Dynamic position sizing
    - Regime transition handling
    - Multi-strategy coordination
    """

    def __init__(self, config: Optional[SelectorConfig] = None):
        self.config = config or SelectorConfig()
        self._strategies: Dict[str, StrategyConfig] = {}
        self._performance: Dict[Tuple[str, MarketRegime], StrategyPerformance] = {}
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._last_regime_change: datetime = datetime.now()
        self._active_strategies: Set[str] = set()

    def register_strategy(self, config: StrategyConfig):
        """Register a strategy for selection."""
        self._strategies[config.strategy_id] = config
        logger.info(f"Registered strategy: {config.name} ({config.strategy_id})")

    def unregister_strategy(self, strategy_id: str):
        """Remove a strategy."""
        self._strategies.pop(strategy_id, None)

    def update_performance(
        self, strategy_id: str, regime: MarketRegime, performance: StrategyPerformance
    ):
        """Update performance data for a strategy."""
        key = (strategy_id, regime)
        self._performance[key] = performance
        logger.debug(f"Updated performance for {strategy_id} in {regime.value}")

    def select_strategies(
        self, regime: MarketRegime, volatility: float = 0.0, force_reselection: bool = False
    ) -> SelectionResult:
        """
        Select optimal strategies for current regime.

        Args:
            regime: Current market regime
            volatility: Current market volatility
            force_reselection: Force reselection even during cooldown

        Returns:
            SelectionResult with selected strategies
        """
        reasoning = []
        excluded = []

        # Check if regime changed
        regime_changed = regime != self._current_regime
        if regime_changed:
            time_since_change = (datetime.now() - self._last_regime_change).total_seconds() / 60
            if (
                time_since_change < self.config.regime_transition_cooldown_minutes
                and not force_reselection
            ):
                reasoning.append(f"In cooldown ({time_since_change:.1f}m since regime change)")
                # Keep current strategies during cooldown
                return SelectionResult(
                    selected_strategies=list(self._active_strategies),
                    regime=self._current_regime,
                    confidence=0.5,
                    reasoning=reasoning,
                    position_scale=self._calculate_position_scale(self._current_regime, volatility),
                    risk_scale=self._calculate_risk_scale(self._current_regime),
                    excluded_strategies=excluded,
                )

            self._current_regime = regime
            self._last_regime_change = datetime.now()
            reasoning.append(f"Regime changed to {regime.value}")

        # Filter suitable strategies
        candidates = []
        for strategy_id, config in self._strategies.items():
            if not config.is_suitable_for_regime(regime):
                excluded.append(strategy_id)
                continue

            if volatility < config.min_volatility or volatility > config.max_volatility:
                excluded.append(strategy_id)
                continue

            # Get performance score
            perf = self._performance.get((strategy_id, regime))
            if perf and perf.trade_count >= self.config.min_trade_count_for_scoring:
                score = perf.score
            else:
                # Default score based on strategy config
                score = config.priority * 0.1

            candidates.append((strategy_id, config, score))

        if not candidates:
            # Fallback to defensive strategies
            reasoning.append("No suitable strategies, using defensive mode")
            defensive = [
                sid
                for sid, cfg in self._strategies.items()
                if cfg.strategy_type == StrategyType.DEFENSIVE
            ]
            return SelectionResult(
                selected_strategies=defensive[:1] if defensive else [],
                regime=regime,
                confidence=0.3,
                reasoning=reasoning,
                position_scale=self.config.min_position_scale,
                risk_scale=self.config.risk_off_scale,
                excluded_strategies=excluded,
            )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Select top strategies
        selected = []
        for strategy_id, config, score in candidates[: self.config.max_concurrent_strategies]:
            selected.append(strategy_id)
            reasoning.append(f"Selected {config.name} (score={score:.3f})")

        # Update active strategies
        self._active_strategies = set(selected)

        # Calculate confidence based on performance data availability
        scored_count = sum(
            1
            for sid, _, score in candidates[: self.config.max_concurrent_strategies]
            if (sid, regime) in self._performance
        )
        confidence = 0.5 + (scored_count / self.config.max_concurrent_strategies) * 0.4

        return SelectionResult(
            selected_strategies=selected,
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
            position_scale=self._calculate_position_scale(regime, volatility),
            risk_scale=self._calculate_risk_scale(regime),
            excluded_strategies=excluded,
        )

    def _calculate_position_scale(self, regime: MarketRegime, volatility: float) -> float:
        """Calculate position size scale factor."""
        base = self.config.base_position_scale

        # Regime adjustments
        if regime == MarketRegime.CRASH:
            base *= self.config.crash_scale
        elif regime.is_risk_off:
            base *= self.config.risk_off_scale
        elif regime == MarketRegime.STRONG_BULL:
            base *= 1.2
        elif regime == MarketRegime.SIDEWAYS:
            base *= 0.8

        # Volatility adjustment
        if volatility > 0.03:  # High volatility
            base *= 0.7
        elif volatility > 0.05:
            base *= 0.5

        return max(self.config.min_position_scale, min(self.config.max_position_scale, base))

    def _calculate_risk_scale(self, regime: MarketRegime) -> float:
        """Calculate risk scale factor."""
        if regime == MarketRegime.CRASH:
            return self.config.crash_scale
        elif regime.is_risk_off:
            return self.config.risk_off_scale
        elif regime.is_bullish:
            return 1.0
        elif regime.is_bearish:
            return 0.7
        else:
            return 0.9

    def get_strategy_allocation(
        self, regime: MarketRegime, total_capital: float
    ) -> Dict[str, float]:
        """
        Get capital allocation across strategies.

        Args:
            regime: Current regime
            total_capital: Total available capital

        Returns:
            Dict mapping strategy_id to allocated capital
        """
        result = self.select_strategies(regime)

        if not result.selected_strategies:
            return {}

        # Calculate scores for selected strategies
        scores = []
        for strategy_id in result.selected_strategies:
            perf = self._performance.get((strategy_id, regime))
            if perf:
                scores.append((strategy_id, max(0.1, perf.score + 1)))  # Ensure positive
            else:
                config = self._strategies.get(strategy_id)
                scores.append((strategy_id, 1.0 + config.priority * 0.1 if config else 1.0))

        # Normalize scores to allocations
        total_score = sum(s[1] for s in scores)
        allocation = {}

        for strategy_id, score in scores:
            weight = score / total_score
            config = self._strategies.get(strategy_id)
            size_factor = config.position_sizing_factor if config else 1.0

            allocated = total_capital * weight * result.position_scale * size_factor
            allocation[strategy_id] = allocated

        return allocation

    def get_regime_strategy_matrix(self) -> pd.DataFrame:
        """Get matrix showing strategy suitability per regime."""
        data = []

        for strategy_id, config in self._strategies.items():
            row = {"strategy": config.name, "type": config.strategy_type.value}

            for regime in MarketRegime:
                if regime == MarketRegime.UNKNOWN:
                    continue

                perf = self._performance.get((strategy_id, regime))
                if perf:
                    row[regime.value] = f"{perf.score:.2f}"
                elif config.is_suitable_for_regime(regime):
                    row[regime.value] = "✓"
                else:
                    row[regime.value] = "✗"

            data.append(row)

        return pd.DataFrame(data)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current selector state."""
        return {
            "current_regime": self._current_regime.value,
            "active_strategies": list(self._active_strategies),
            "last_regime_change": self._last_regime_change.isoformat(),
            "registered_strategies": len(self._strategies),
            "performance_records": len(self._performance),
        }


# Pre-configured strategy templates


def create_default_strategies() -> List[StrategyConfig]:
    """Create default strategy configurations."""
    return [
        # Trend following strategies
        StrategyConfig(
            strategy_id="trend_ema",
            name="EMA Trend Following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={
                MarketRegime.BULL,
                MarketRegime.STRONG_BULL,
                MarketRegime.BEAR,
                MarketRegime.STRONG_BEAR,
            },
            unsuitable_regimes={MarketRegime.SIDEWAYS, MarketRegime.CRASH},
            priority=2,
        ),
        StrategyConfig(
            strategy_id="trend_macd",
            name="MACD Trend",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL, MarketRegime.STRONG_BULL, MarketRegime.BEAR},
            unsuitable_regimes={MarketRegime.CRASH, MarketRegime.HIGH_VOL},
            priority=1,
        ),
        # Mean reversion strategies
        StrategyConfig(
            strategy_id="mean_rev_bb",
            name="Bollinger Band Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            suitable_regimes={MarketRegime.SIDEWAYS},
            unsuitable_regimes={
                MarketRegime.STRONG_BULL,
                MarketRegime.STRONG_BEAR,
                MarketRegime.CRASH,
            },
            priority=2,
        ),
        StrategyConfig(
            strategy_id="mean_rev_rsi",
            name="RSI Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            suitable_regimes={MarketRegime.SIDEWAYS, MarketRegime.VOLATILE},
            unsuitable_regimes={MarketRegime.CRASH},
            priority=1,
        ),
        # Momentum strategies
        StrategyConfig(
            strategy_id="momentum_breakout",
            name="Momentum Breakout",
            strategy_type=StrategyType.MOMENTUM,
            suitable_regimes={MarketRegime.STRONG_BULL, MarketRegime.STRONG_BEAR},
            unsuitable_regimes={MarketRegime.SIDEWAYS, MarketRegime.CRASH},
            min_volatility=0.01,
            priority=1,
        ),
        # Volatility strategies
        StrategyConfig(
            strategy_id="vol_expansion",
            name="Volatility Expansion",
            strategy_type=StrategyType.VOLATILITY,
            suitable_regimes={MarketRegime.HIGH_VOL, MarketRegime.VOLATILE},
            unsuitable_regimes={MarketRegime.CRASH},
            min_volatility=0.02,
            position_sizing_factor=0.7,  # Smaller positions
            priority=1,
        ),
        # Defensive strategies
        StrategyConfig(
            strategy_id="defensive_cash",
            name="Defensive Cash",
            strategy_type=StrategyType.DEFENSIVE,
            suitable_regimes={MarketRegime.CRASH, MarketRegime.HIGH_VOL},
            unsuitable_regimes=set(),
            position_sizing_factor=0.3,
            risk_factor=0.5,
            priority=3,  # High priority in suitable regimes
        ),
    ]


def create_regime_strategy_selector(
    config: Optional[SelectorConfig] = None, include_defaults: bool = True
) -> RegimeStrategySelector:
    """
    Factory function to create regime strategy selector.

    Args:
        config: Selector configuration
        include_defaults: Whether to include default strategies
    """
    selector = RegimeStrategySelector(config=config)

    if include_defaults:
        for strategy_config in create_default_strategies():
            selector.register_strategy(strategy_config)

    return selector
