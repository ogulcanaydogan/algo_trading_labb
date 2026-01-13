"""
Regime Transition Matrix Module.

Visualizes regime change probabilities and tracks
historical regime transitions.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Information about a regime state."""
    name: str
    description: str
    color: str
    avg_duration_hours: float = 0.0
    frequency: float = 0.0  # Percentage of time in this regime
    avg_return: float = 0.0
    avg_volatility: float = 0.0


@dataclass
class Transition:
    """A single regime transition."""
    from_regime: str
    to_regime: str
    timestamp: datetime
    price_at_transition: float
    confidence: float


@dataclass
class TransitionMatrix:
    """Complete transition matrix with probabilities."""
    matrix: Dict[str, Dict[str, float]]
    regimes: List[str]
    total_transitions: int
    transition_counts: Dict[str, Dict[str, int]]


class RegimeTransitionAnalyzer:
    """
    Analyzes market regime transitions.

    Calculates transition probabilities, tracks historical
    transitions, and provides visualization data.
    """

    REGIME_COLORS = {
        "BULL": "#4CAF50",
        "BEAR": "#F44336",
        "VOLATILE": "#FF9800",
        "ACCUMULATION": "#2196F3",
        "DISTRIBUTION": "#9C27B0",
        "NEUTRAL": "#9E9E9E",
        "TRENDING_UP": "#66BB6A",
        "TRENDING_DOWN": "#EF5350",
        "RANGING": "#78909C",
        "UNKNOWN": "#616161",
    }

    REGIME_DESCRIPTIONS = {
        "BULL": "Strong upward trend with high momentum",
        "BEAR": "Strong downward trend with negative momentum",
        "VOLATILE": "High volatility with rapid price swings",
        "ACCUMULATION": "Low volatility accumulation phase",
        "DISTRIBUTION": "Distribution phase before potential reversal",
        "NEUTRAL": "No clear trend or direction",
        "TRENDING_UP": "Moderate upward trend",
        "TRENDING_DOWN": "Moderate downward trend",
        "RANGING": "Sideways price action within range",
        "UNKNOWN": "Unable to classify regime",
    }

    def __init__(self):
        self._transitions: List[Transition] = []
        self._regime_history: List[Tuple[datetime, str, float]] = []  # (timestamp, regime, price)
        self._regime_states: Dict[str, RegimeState] = {}

    def load_regime_history(
        self,
        regimes: List[str],
        timestamps: Optional[List[datetime]] = None,
        prices: Optional[List[float]] = None,
    ) -> None:
        """
        Load historical regime classifications.

        Args:
            regimes: List of regime labels
            timestamps: Optional list of timestamps
            prices: Optional list of prices
        """
        if not regimes:
            return

        if timestamps is None:
            timestamps = [
                datetime.now() - timedelta(hours=len(regimes) - i)
                for i in range(len(regimes))
            ]

        if prices is None:
            prices = [0.0] * len(regimes)

        self._regime_history = list(zip(timestamps, regimes, prices))
        self._detect_transitions()
        self._calculate_regime_states()

    def load_from_json(self, json_path: str) -> None:
        """Load regime history from JSON file."""
        try:
            path = Path(json_path)
            if not path.exists():
                logger.warning(f"Regime file not found: {path}")
                return

            with open(path) as f:
                data = json.load(f)

            if isinstance(data, list):
                # List of regime entries
                regimes = []
                timestamps = []
                prices = []

                for entry in data:
                    regimes.append(entry.get("regime", "UNKNOWN"))
                    ts = entry.get("timestamp")
                    if ts:
                        timestamps.append(datetime.fromisoformat(ts))
                    prices.append(entry.get("price", 0))

                self.load_regime_history(regimes, timestamps if timestamps else None, prices)

            elif isinstance(data, dict):
                regimes = data.get("regimes", [])
                timestamps = [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
                prices = data.get("prices", [])
                self.load_regime_history(regimes, timestamps if timestamps else None, prices)

        except Exception as e:
            logger.error(f"Error loading regime history: {e}")

    def _detect_transitions(self) -> None:
        """Detect regime transitions from history."""
        self._transitions = []

        for i in range(1, len(self._regime_history)):
            prev_ts, prev_regime, prev_price = self._regime_history[i - 1]
            curr_ts, curr_regime, curr_price = self._regime_history[i]

            if prev_regime != curr_regime:
                self._transitions.append(Transition(
                    from_regime=prev_regime,
                    to_regime=curr_regime,
                    timestamp=curr_ts,
                    price_at_transition=curr_price,
                    confidence=1.0,  # Can be adjusted based on classification confidence
                ))

    def _calculate_regime_states(self) -> None:
        """Calculate statistics for each regime."""
        if not self._regime_history:
            return

        regime_durations: Dict[str, List[float]] = defaultdict(list)
        regime_returns: Dict[str, List[float]] = defaultdict(list)
        regime_counts: Dict[str, int] = defaultdict(int)

        current_regime = None
        regime_start = None

        for i, (ts, regime, price) in enumerate(self._regime_history):
            if regime != current_regime:
                if current_regime is not None and regime_start is not None:
                    duration = (ts - regime_start).total_seconds() / 3600
                    regime_durations[current_regime].append(duration)

                current_regime = regime
                regime_start = ts

            regime_counts[regime] += 1

            # Calculate return if we have previous price
            if i > 0:
                prev_price = self._regime_history[i - 1][2]
                if prev_price > 0:
                    ret = (price - prev_price) / prev_price
                    regime_returns[regime].append(ret)

        total_observations = len(self._regime_history)

        for regime in set(r for _, r, _ in self._regime_history):
            durations = regime_durations.get(regime, [])
            returns = regime_returns.get(regime, [])

            self._regime_states[regime] = RegimeState(
                name=regime,
                description=self.REGIME_DESCRIPTIONS.get(regime, "Unknown regime"),
                color=self.REGIME_COLORS.get(regime, "#616161"),
                avg_duration_hours=float(np.mean(durations)) if durations else 0,
                frequency=regime_counts[regime] / total_observations * 100 if total_observations > 0 else 0,
                avg_return=float(np.mean(returns)) * 100 if returns else 0,
                avg_volatility=float(np.std(returns)) * 100 if len(returns) > 1 else 0,
            )

    def calculate_transition_matrix(self) -> TransitionMatrix:
        """
        Calculate the regime transition probability matrix.

        Returns:
            TransitionMatrix with probabilities and counts
        """
        # Get unique regimes
        regimes = sorted(set(t.from_regime for t in self._transitions) |
                        set(t.to_regime for t in self._transitions))

        if not regimes:
            regimes = list(self.REGIME_COLORS.keys())[:6]

        # Count transitions
        counts: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in regimes} for r in regimes}

        for transition in self._transitions:
            if transition.from_regime in counts:
                counts[transition.from_regime][transition.to_regime] += 1

        # Calculate probabilities
        matrix: Dict[str, Dict[str, float]] = {}

        for from_regime in regimes:
            total_from = sum(counts[from_regime].values())
            matrix[from_regime] = {}

            for to_regime in regimes:
                if total_from > 0:
                    matrix[from_regime][to_regime] = round(
                        counts[from_regime][to_regime] / total_from, 3
                    )
                else:
                    # Default probability (stay in same regime)
                    matrix[from_regime][to_regime] = 1.0 if from_regime == to_regime else 0.0

        return TransitionMatrix(
            matrix=matrix,
            regimes=regimes,
            total_transitions=len(self._transitions),
            transition_counts=counts,
        )

    def get_current_regime(self) -> Optional[str]:
        """Get the most recent regime classification."""
        if self._regime_history:
            return self._regime_history[-1][1]
        return None

    def get_transition_history(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent transition history."""
        return [
            {
                "from": t.from_regime,
                "to": t.to_regime,
                "timestamp": t.timestamp.isoformat(),
                "price": t.price_at_transition,
                "from_color": self.REGIME_COLORS.get(t.from_regime, "#616161"),
                "to_color": self.REGIME_COLORS.get(t.to_regime, "#616161"),
            }
            for t in self._transitions[-limit:]
        ]

    def predict_next_regime(
        self,
        current_regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Predict probabilities for next regime.

        Args:
            current_regime: Override current regime

        Returns:
            Dictionary of regime probabilities
        """
        if current_regime is None:
            current_regime = self.get_current_regime()

        if current_regime is None:
            return {}

        matrix = self.calculate_transition_matrix()

        if current_regime in matrix.matrix:
            return dict(matrix.matrix[current_regime])
        else:
            return {}

    def get_regime_timeline(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get regime timeline for visualization."""
        timeline = []

        for ts, regime, price in self._regime_history[-limit:]:
            timeline.append({
                "timestamp": ts.isoformat(),
                "regime": regime,
                "price": price,
                "color": self.REGIME_COLORS.get(regime, "#616161"),
            })

        return timeline

    def get_regime_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Get distribution of regimes."""
        return {
            name: {
                "name": state.name,
                "description": state.description,
                "color": state.color,
                "frequency_pct": round(state.frequency, 1),
                "avg_duration_hours": round(state.avg_duration_hours, 1),
                "avg_return_pct": round(state.avg_return, 2),
                "avg_volatility_pct": round(state.avg_volatility, 2),
            }
            for name, state in self._regime_states.items()
        }

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        matrix = self.calculate_transition_matrix()

        return {
            "current_regime": self.get_current_regime(),
            "next_regime_probabilities": self.predict_next_regime(),
            "transition_matrix": {
                "regimes": matrix.regimes,
                "probabilities": matrix.matrix,
                "counts": matrix.transition_counts,
                "total_transitions": matrix.total_transitions,
            },
            "regime_distribution": self.get_regime_distribution(),
            "recent_transitions": self.get_transition_history(20),
            "timeline": self.get_regime_timeline(100),
            "regime_info": {
                regime: {
                    "color": self.REGIME_COLORS.get(regime, "#616161"),
                    "description": self.REGIME_DESCRIPTIONS.get(regime, "Unknown"),
                }
                for regime in matrix.regimes
            },
        }
