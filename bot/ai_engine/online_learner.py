"""
Online Learning System

Adapts trading strategies in real-time based on recent market behavior.

Features:
- Streaming updates from each trade
- Concept drift detection (market regime changes)
- Adaptive parameter adjustment
- Real-time performance tracking
- Automatic strategy switching
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import numpy as np

from .learning_db import LearningDatabase, TradeRecord, get_learning_db

logger = logging.getLogger(__name__)


@dataclass
class PerformanceWindow:
    """Rolling window of performance metrics."""
    window_size: int = 50
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=50))
    win_history: deque = field(default_factory=lambda: deque(maxlen=50))
    regime_history: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def win_rate(self) -> float:
        if not self.win_history:
            return 0.5
        return sum(self.win_history) / len(self.win_history)

    @property
    def avg_pnl(self) -> float:
        if not self.pnl_history:
            return 0.0
        return sum(self.pnl_history) / len(self.pnl_history)

    @property
    def sharpe(self) -> float:
        if len(self.pnl_history) < 10:
            return 0.0
        returns = list(self.pnl_history)
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    @property
    def current_regime(self) -> Optional[str]:
        if not self.regime_history:
            return None
        # Most common regime in recent history
        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return max(regime_counts, key=regime_counts.get)

    def add_trade(self, pnl_pct: float, is_win: bool, regime: str):
        self.pnl_history.append(pnl_pct)
        self.win_history.append(1 if is_win else 0)
        self.regime_history.append(regime)


@dataclass
class StrategyHealth:
    """Health status of a strategy."""
    strategy_id: str
    performance: PerformanceWindow = field(default_factory=PerformanceWindow)
    last_updated: Optional[datetime] = None
    consecutive_losses: int = 0
    is_degraded: bool = False
    degradation_reason: Optional[str] = None

    def update(self, pnl_pct: float, regime: str):
        """Update health with new trade."""
        is_win = pnl_pct > 0
        self.performance.add_trade(pnl_pct, is_win, regime)
        self.last_updated = datetime.now(timezone.utc)

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        # Check for degradation
        self._check_degradation()

    def _check_degradation(self):
        """Check if strategy is degraded."""
        self.is_degraded = False
        self.degradation_reason = None

        # Too many consecutive losses
        if self.consecutive_losses >= 5:
            self.is_degraded = True
            self.degradation_reason = f"Consecutive losses: {self.consecutive_losses}"

        # Win rate too low
        elif self.performance.win_rate < 0.35 and len(self.performance.win_history) >= 20:
            self.is_degraded = True
            self.degradation_reason = f"Low win rate: {self.performance.win_rate:.1%}"

        # Negative average P&L
        elif self.performance.avg_pnl < -1.0 and len(self.performance.pnl_history) >= 20:
            self.is_degraded = True
            self.degradation_reason = f"Negative avg P&L: {self.performance.avg_pnl:.2f}%"


class ConceptDriftDetector:
    """
    Detects when market behavior changes significantly.

    Uses statistical tests to detect distribution shifts.
    """

    def __init__(
        self,
        reference_window: int = 100,
        test_window: int = 20,
        threshold: float = 2.0,
    ):
        self.reference_window = reference_window
        self.test_window = test_window
        self.threshold = threshold
        self.reference_data: deque = deque(maxlen=reference_window)
        self.test_data: deque = deque(maxlen=test_window)

    def add_observation(self, value: float):
        """Add new observation."""
        self.reference_data.append(value)
        self.test_data.append(value)

    def detect_drift(self) -> Tuple[bool, float]:
        """
        Detect if recent data differs significantly from reference.

        Returns:
            Tuple of (drift_detected, drift_score)
        """
        if len(self.reference_data) < self.reference_window // 2:
            return False, 0.0

        if len(self.test_data) < self.test_window // 2:
            return False, 0.0

        ref_mean = np.mean(list(self.reference_data))
        ref_std = np.std(list(self.reference_data))

        if ref_std == 0:
            return False, 0.0

        test_mean = np.mean(list(self.test_data))

        # Z-score of test mean relative to reference distribution
        z_score = abs(test_mean - ref_mean) / (ref_std / np.sqrt(len(self.test_data)))

        drift_detected = z_score > self.threshold

        return drift_detected, z_score


class OnlineLearner:
    """
    Online learning system for real-time strategy adaptation.

    Features:
    1. Track performance of multiple strategies
    2. Detect when strategies degrade
    3. Adjust parameters in real-time
    4. Switch strategies based on regime
    5. Learn from each trade outcome
    """

    def __init__(
        self,
        db: LearningDatabase = None,
        adaptation_rate: float = 0.1,
        min_trades_to_adapt: int = 10,
    ):
        self.db = db or get_learning_db()
        self.adaptation_rate = adaptation_rate
        self.min_trades_to_adapt = min_trades_to_adapt

        # Strategy health tracking
        self.strategy_health: Dict[str, StrategyHealth] = {}

        # Drift detectors per indicator
        self.drift_detectors: Dict[str, ConceptDriftDetector] = {
            "volatility": ConceptDriftDetector(),
            "momentum": ConceptDriftDetector(),
            "volume": ConceptDriftDetector(),
        }

        # Parameter adjustments learned online
        self.learned_adjustments: Dict[str, Dict[str, float]] = {}

        # Trade outcome patterns
        self.outcome_patterns: Dict[str, List[Dict]] = {}

    def on_trade_complete(
        self,
        symbol: str,
        strategy_id: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        regime: str,
        indicators_at_entry: Dict[str, float],
        hold_duration_mins: int,
    ):
        """
        Called when a trade completes. Updates learning models.

        Args:
            symbol: Trading symbol
            strategy_id: ID of strategy that made the trade
            entry_price: Entry price
            exit_price: Exit price
            pnl_pct: P&L percentage
            regime: Market regime during trade
            indicators_at_entry: Indicator values at entry
            hold_duration_mins: How long position was held
        """
        # Update strategy health
        if strategy_id not in self.strategy_health:
            self.strategy_health[strategy_id] = StrategyHealth(strategy_id=strategy_id)

        self.strategy_health[strategy_id].update(pnl_pct, regime)

        # Update drift detectors
        if 'atr_ratio' in indicators_at_entry:
            self.drift_detectors['volatility'].add_observation(
                indicators_at_entry['atr_ratio']
            )
        if 'momentum_5' in indicators_at_entry:
            self.drift_detectors['momentum'].add_observation(
                indicators_at_entry['momentum_5']
            )
        if 'volume_ratio' in indicators_at_entry:
            self.drift_detectors['volume'].add_observation(
                indicators_at_entry['volume_ratio']
            )

        # Record outcome pattern for learning
        outcome = "WIN" if pnl_pct > 0 else "LOSS"
        pattern_key = f"{strategy_id}_{regime}"
        if pattern_key not in self.outcome_patterns:
            self.outcome_patterns[pattern_key] = []

        self.outcome_patterns[pattern_key].append({
            "indicators": indicators_at_entry,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
            "hold_duration": hold_duration_mins,
        })

        # Trim patterns to recent history
        if len(self.outcome_patterns[pattern_key]) > 200:
            self.outcome_patterns[pattern_key] = self.outcome_patterns[pattern_key][-200:]

        # Record to database
        try:
            trade = TradeRecord(
                id=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                action="LONG" if exit_price > entry_price else "SHORT",
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=1.0,  # Normalized
                pnl=pnl_pct,
                pnl_pct=pnl_pct,
                hold_duration_mins=hold_duration_mins,
                regime=regime,
                strategy_id=strategy_id,
                indicators=indicators_at_entry,
                outcome=outcome,
            )
            self.db.record_trade(trade)
        except Exception as e:
            logger.warning(f"Failed to record trade: {e}")

        # Learn parameter adjustments
        self._learn_adjustments(strategy_id, regime, indicators_at_entry, pnl_pct)

        logger.debug(
            f"Trade recorded: {strategy_id} in {regime} regime, "
            f"P&L={pnl_pct:+.2f}%, outcome={outcome}"
        )

    def _learn_adjustments(
        self,
        strategy_id: str,
        regime: str,
        indicators: Dict[str, float],
        pnl_pct: float,
    ):
        """Learn parameter adjustments from trade outcome."""
        key = f"{strategy_id}_{regime}"
        if key not in self.learned_adjustments:
            self.learned_adjustments[key] = {}

        # Simple online learning: adjust based on outcome
        # If winning in certain conditions, reinforce those conditions
        # If losing, adjust away from those conditions

        adjustment_factor = self.adaptation_rate * (pnl_pct / 10.0)  # Scale P&L impact

        for indicator, value in indicators.items():
            if indicator not in self.learned_adjustments[key]:
                self.learned_adjustments[key][indicator] = 0.0

            # Positive adjustment if winning, negative if losing
            self.learned_adjustments[key][indicator] += adjustment_factor * np.sign(value)

            # Clip to reasonable range
            self.learned_adjustments[key][indicator] = np.clip(
                self.learned_adjustments[key][indicator], -1.0, 1.0
            )

    def get_parameter_adjustments(
        self,
        strategy_id: str,
        regime: str,
        base_params: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Get learned parameter adjustments for a strategy/regime.

        Returns adjusted parameters based on online learning.
        """
        key = f"{strategy_id}_{regime}"
        adjustments = self.learned_adjustments.get(key, {})

        if not adjustments:
            return base_params

        adjusted = base_params.copy()

        # Apply learned adjustments
        for param, base_value in base_params.items():
            if param in adjustments:
                # Adjust by learned factor (max ±20%)
                factor = 1.0 + adjustments[param] * 0.2
                adjusted[param] = base_value * factor

        return adjusted

    def should_trade(
        self,
        strategy_id: str,
        regime: str,
        indicators: Dict[str, float],
    ) -> Tuple[bool, float, str]:
        """
        Check if strategy should trade based on online learning.

        Returns:
            Tuple of (should_trade, confidence_adjustment, reason)
        """
        # Check strategy health
        if strategy_id in self.strategy_health:
            health = self.strategy_health[strategy_id]

            if health.is_degraded:
                return False, 0.0, f"Strategy degraded: {health.degradation_reason}"

        # Check for concept drift
        for name, detector in self.drift_detectors.items():
            drift_detected, drift_score = detector.detect_drift()
            if drift_detected:
                logger.warning(f"Concept drift detected in {name}: score={drift_score:.2f}")
                # Reduce confidence during drift
                return True, -0.2, f"Drift in {name}"

        # Check outcome patterns
        pattern_key = f"{strategy_id}_{regime}"
        if pattern_key in self.outcome_patterns:
            patterns = self.outcome_patterns[pattern_key]

            if len(patterns) >= self.min_trades_to_adapt:
                # Find similar past trades
                similar_outcomes = self._find_similar_outcomes(patterns, indicators)

                if similar_outcomes:
                    win_rate = sum(1 for o in similar_outcomes if o["outcome"] == "WIN") / len(similar_outcomes)

                    if win_rate < 0.3:
                        return False, 0.0, f"Low historical win rate in similar conditions: {win_rate:.1%}"

                    if win_rate > 0.6:
                        # Boost confidence
                        return True, 0.1, f"High historical win rate: {win_rate:.1%}"

        return True, 0.0, "No adjustment"

    def _find_similar_outcomes(
        self,
        patterns: List[Dict],
        current_indicators: Dict[str, float],
        max_results: int = 20,
    ) -> List[Dict]:
        """Find past trades with similar indicator values."""
        if not patterns:
            return []

        similarities = []
        for pattern in patterns:
            past_indicators = pattern.get("indicators", {})

            # Calculate similarity (inverse of distance)
            distance = 0.0
            matched_keys = 0

            for key in current_indicators:
                if key in past_indicators:
                    diff = abs(current_indicators[key] - past_indicators[key])
                    # Normalize by typical range
                    normalized_diff = diff / (abs(past_indicators[key]) + 1e-6)
                    distance += normalized_diff
                    matched_keys += 1

            if matched_keys > 0:
                avg_distance = distance / matched_keys
                similarity = 1.0 / (1.0 + avg_distance)
                similarities.append((similarity, pattern))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in similarities[:max_results]]

    def get_strategy_recommendations(
        self,
        current_regime: str,
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations based on current conditions.

        Returns list of recommended strategies with confidence.
        """
        recommendations = []

        for strategy_id, health in self.strategy_health.items():
            if health.is_degraded:
                continue

            # Check if strategy performs well in current regime
            regime_performance = self._get_regime_performance(strategy_id, current_regime)

            if regime_performance:
                recommendations.append({
                    "strategy_id": strategy_id,
                    "regime": current_regime,
                    "win_rate": regime_performance["win_rate"],
                    "avg_pnl": regime_performance["avg_pnl"],
                    "sharpe": regime_performance["sharpe"],
                    "trades": regime_performance["trades"],
                    "is_healthy": not health.is_degraded,
                })

        # Sort by Sharpe ratio
        recommendations.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

        return recommendations

    def _get_regime_performance(
        self,
        strategy_id: str,
        regime: str,
    ) -> Optional[Dict[str, float]]:
        """Get strategy performance in specific regime."""
        pattern_key = f"{strategy_id}_{regime}"
        patterns = self.outcome_patterns.get(pattern_key, [])

        if len(patterns) < 5:
            return None

        wins = sum(1 for p in patterns if p["outcome"] == "WIN")
        pnls = [p["pnl_pct"] for p in patterns]

        return {
            "win_rate": wins / len(patterns),
            "avg_pnl": np.mean(pnls),
            "sharpe": np.mean(pnls) / (np.std(pnls) + 1e-6) * np.sqrt(252) if len(pnls) > 1 else 0,
            "trades": len(patterns),
        }

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        drift_status = {}
        for name, detector in self.drift_detectors.items():
            drift_detected, score = detector.detect_drift()
            drift_status[name] = {
                "drift_detected": drift_detected,
                "score": round(score, 2),
            }

        strategy_status = {}
        for strategy_id, health in self.strategy_health.items():
            strategy_status[strategy_id] = {
                "win_rate": round(health.performance.win_rate, 3),
                "avg_pnl": round(health.performance.avg_pnl, 3),
                "sharpe": round(health.performance.sharpe, 3),
                "consecutive_losses": health.consecutive_losses,
                "is_degraded": health.is_degraded,
                "degradation_reason": health.degradation_reason,
                "trades_tracked": len(health.performance.pnl_history),
            }

        return {
            "drift_detection": drift_status,
            "strategy_health": strategy_status,
            "patterns_tracked": len(self.outcome_patterns),
            "adjustments_learned": len(self.learned_adjustments),
        }


# Global instance
_learner: Optional[OnlineLearner] = None


def get_online_learner() -> OnlineLearner:
    """Get or create global online learner."""
    global _learner
    if _learner is None:
        _learner = OnlineLearner()
    return _learner
