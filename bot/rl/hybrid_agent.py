"""
Hybrid ML+RL Trading Agent.

Combines traditional ML predictions with RL policy for robust trading decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .policy_network import TradingPolicyNetwork, PolicyConfig

# Try to import PyTorch
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None


@dataclass
class HybridConfig:
    """Configuration for hybrid ML+RL agent."""
    ml_weight: float = 0.6  # Weight for ML predictions
    rl_weight: float = 0.4  # Weight for RL policy
    adaptive_weights: bool = True  # Adjust weights based on performance
    ml_confidence_threshold: float = 0.7  # High confidence threshold
    rl_override_threshold: float = 0.85  # RL can override if very confident
    weight_adaptation_rate: float = 0.01


class HybridMLRLAgent:
    """
    Hybrid Agent combining ML predictions with RL policy.

    Features:
    - Weighted combination of ML and RL signals
    - Adaptive weight adjustment based on performance
    - High-confidence override capability
    - Ensemble decision making

    Usage:
        agent = HybridMLRLAgent(ml_model, rl_policy)
        action, confidence = agent.predict(state)
    """

    def __init__(
        self,
        ml_model: Any,
        rl_policy: Optional[TradingPolicyNetwork] = None,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize hybrid agent.

        Args:
            ml_model: Traditional ML model with predict_proba method
            rl_policy: RL policy network
            config: Agent configuration
        """
        self.ml_model = ml_model
        self.rl_policy = rl_policy
        self.config = config or HybridConfig()

        # Performance tracking for adaptive weights
        self._ml_performance: float = 0.5
        self._rl_performance: float = 0.5
        self._ml_weight = self.config.ml_weight
        self._rl_weight = self.config.rl_weight

        # Recent predictions for tracking
        self._recent_predictions: list = []

    def predict(
        self,
        state: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """
        Get trading decision combining ML and RL.

        Args:
            state: Full state vector (for RL)
            features: Feature vector (for ML, if different from state)

        Returns:
            Tuple of (action_string, confidence)
        """
        features = features if features is not None else state

        # Get ML predictions
        ml_probs = self._get_ml_probs(features)

        # Get RL predictions
        rl_probs = self._get_rl_probs(state)

        # Combine predictions
        combined_probs = self._combine_predictions(ml_probs, rl_probs)

        # Get action and confidence
        action_idx = np.argmax(combined_probs)
        confidence = combined_probs[action_idx]

        # Check for high-confidence overrides
        action_idx, confidence = self._check_overrides(
            action_idx, confidence, ml_probs, rl_probs
        )

        # Convert to action string
        action = ["SHORT", "FLAT", "LONG"][action_idx]

        return action, float(confidence)

    def _get_ml_probs(self, features: np.ndarray) -> np.ndarray:
        """Get ML model prediction probabilities."""
        if self.ml_model is None:
            return np.array([0.33, 0.34, 0.33])

        try:
            if hasattr(self.ml_model, "predict_proba"):
                # Ensure 2D input
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                probs = self.ml_model.predict_proba(features)[0]
                # Ensure 3 classes
                if len(probs) != 3:
                    return np.array([0.33, 0.34, 0.33])
                return probs
            else:
                # Binary or regression model
                pred = self.ml_model.predict(features.reshape(1, -1))[0]
                # Convert to 3-class probabilities
                if pred > 0.5:
                    return np.array([0.1, 0.2, 0.7])
                elif pred < -0.5:
                    return np.array([0.7, 0.2, 0.1])
                else:
                    return np.array([0.2, 0.6, 0.2])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _get_rl_probs(self, state: np.ndarray) -> np.ndarray:
        """Get RL policy prediction probabilities."""
        if self.rl_policy is None:
            return np.array([0.33, 0.34, 0.33])

        try:
            if HAS_TORCH:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits, _, _ = self.rl_policy.forward(state_tensor)
                    probs = F.softmax(logits, dim=-1).numpy()[0]
                return probs
            else:
                # Fallback for non-PyTorch
                return np.array([0.33, 0.34, 0.33])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _combine_predictions(
        self,
        ml_probs: np.ndarray,
        rl_probs: np.ndarray,
    ) -> np.ndarray:
        """Combine ML and RL predictions."""
        # Weighted average
        combined = (
            self._ml_weight * ml_probs +
            self._rl_weight * rl_probs
        )

        # Normalize
        combined = combined / combined.sum()

        return combined

    def _check_overrides(
        self,
        action_idx: int,
        confidence: float,
        ml_probs: np.ndarray,
        rl_probs: np.ndarray,
    ) -> Tuple[int, float]:
        """Check for high-confidence override conditions."""
        ml_max = ml_probs.max()
        rl_max = rl_probs.max()

        # ML high-confidence override
        if ml_max > self.config.ml_confidence_threshold:
            ml_action = ml_probs.argmax()
            if ml_action != action_idx:
                # Override to ML action if ML is very confident
                action_idx = ml_action
                confidence = ml_max * 0.95

        # RL high-confidence override
        if rl_max > self.config.rl_override_threshold:
            rl_action = rl_probs.argmax()
            if rl_action != action_idx:
                # RL can override only if extremely confident
                action_idx = rl_action
                confidence = rl_max * 0.9

        return action_idx, confidence

    def update_weights(
        self,
        ml_correct: bool,
        rl_correct: bool,
    ) -> None:
        """
        Update adaptive weights based on prediction outcomes.

        Args:
            ml_correct: Whether ML prediction was correct
            rl_correct: Whether RL prediction was correct
        """
        if not self.config.adaptive_weights:
            return

        # Update performance estimates with exponential moving average
        alpha = self.config.weight_adaptation_rate
        self._ml_performance = (1 - alpha) * self._ml_performance + alpha * int(ml_correct)
        self._rl_performance = (1 - alpha) * self._rl_performance + alpha * int(rl_correct)

        # Recalculate weights based on relative performance
        total_perf = self._ml_performance + self._rl_performance
        if total_perf > 0:
            self._ml_weight = self._ml_performance / total_perf
            self._rl_weight = self._rl_performance / total_perf

            # Clamp weights
            self._ml_weight = max(0.3, min(0.8, self._ml_weight))
            self._rl_weight = 1.0 - self._ml_weight

    def record_outcome(
        self,
        prediction: str,
        actual_outcome: str,
        ml_prediction: str,
        rl_prediction: str,
    ) -> None:
        """
        Record prediction outcome for weight adaptation.

        Args:
            prediction: The hybrid prediction made
            actual_outcome: The actual market outcome
            ml_prediction: What ML alone would have predicted
            rl_prediction: What RL alone would have predicted
        """
        # Determine correctness (simplified)
        outcome_correct = prediction == actual_outcome
        ml_correct = ml_prediction == actual_outcome
        rl_correct = rl_prediction == actual_outcome

        # Update weights
        self.update_weights(ml_correct, rl_correct)

        # Track for analysis
        self._recent_predictions.append({
            "prediction": prediction,
            "actual": actual_outcome,
            "ml_prediction": ml_prediction,
            "rl_prediction": rl_prediction,
            "hybrid_correct": outcome_correct,
            "ml_correct": ml_correct,
            "rl_correct": rl_correct,
            "ml_weight": self._ml_weight,
            "rl_weight": self._rl_weight,
        })

        # Keep only recent history
        if len(self._recent_predictions) > 1000:
            self._recent_predictions = self._recent_predictions[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and performance metrics."""
        if not self._recent_predictions:
            return {
                "ml_weight": self._ml_weight,
                "rl_weight": self._rl_weight,
                "ml_performance": self._ml_performance,
                "rl_performance": self._rl_performance,
                "predictions_recorded": 0,
            }

        recent = self._recent_predictions[-100:]

        hybrid_accuracy = sum(1 for p in recent if p["hybrid_correct"]) / len(recent)
        ml_accuracy = sum(1 for p in recent if p["ml_correct"]) / len(recent)
        rl_accuracy = sum(1 for p in recent if p["rl_correct"]) / len(recent)

        return {
            "ml_weight": self._ml_weight,
            "rl_weight": self._rl_weight,
            "ml_performance": self._ml_performance,
            "rl_performance": self._rl_performance,
            "predictions_recorded": len(self._recent_predictions),
            "recent_accuracy": {
                "hybrid": hybrid_accuracy,
                "ml": ml_accuracy,
                "rl": rl_accuracy,
            },
        }

    def set_ml_model(self, model: Any) -> None:
        """Update the ML model."""
        self.ml_model = model

    def set_rl_policy(self, policy: TradingPolicyNetwork) -> None:
        """Update the RL policy."""
        self.rl_policy = policy

    def save(self, path: str) -> None:
        """Save agent state."""
        import json
        state = {
            "config": {
                "ml_weight": self.config.ml_weight,
                "rl_weight": self.config.rl_weight,
                "adaptive_weights": self.config.adaptive_weights,
            },
            "current_ml_weight": self._ml_weight,
            "current_rl_weight": self._rl_weight,
            "ml_performance": self._ml_performance,
            "rl_performance": self._rl_performance,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        ml_model: Any,
        rl_policy: Optional[TradingPolicyNetwork] = None,
    ) -> "HybridMLRLAgent":
        """Load agent from file."""
        import json
        with open(path, "r") as f:
            state = json.load(f)

        config = HybridConfig(**state["config"])
        agent = cls(ml_model, rl_policy, config)

        agent._ml_weight = state["current_ml_weight"]
        agent._rl_weight = state["current_rl_weight"]
        agent._ml_performance = state["ml_performance"]
        agent._rl_performance = state["rl_performance"]

        return agent
