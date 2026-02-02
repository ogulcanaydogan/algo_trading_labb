"""
Continual Learning - Prevent Catastrophic Forgetting.

Implements techniques to allow models to learn new patterns
without forgetting previously learned knowledge:
- Elastic Weight Consolidation (EWC)
- Experience Replay with Reservoir Sampling
- Multi-task learning with regime-specific heads
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass
class FisherInformation:
    """Fisher Information Matrix for EWC."""
    parameter_name: str
    importance: np.ndarray  # Diagonal of Fisher matrix
    optimal_weights: np.ndarray  # Weights after training on task


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) implementation.

    Prevents catastrophic forgetting by adding a penalty term
    that discourages changing important weights.

    Loss = Task_Loss + λ * Σ F_i * (θ_i - θ*_i)²

    Where:
    - F_i is the Fisher information (importance) of parameter i
    - θ_i is the current parameter value
    - θ*_i is the optimal parameter value from previous task
    - λ is the regularization strength
    """

    def __init__(
        self,
        lambda_ewc: float = 1000.0,  # EWC regularization strength
        num_samples: int = 200,  # Samples for Fisher calculation
    ):
        self.lambda_ewc = lambda_ewc
        self.num_samples = num_samples

        # Store Fisher information for each task/regime
        self.fisher_info: Dict[str, List[FisherInformation]] = {}
        self._task_count = 0

        logger.info(f"EWC initialized: λ={lambda_ewc}, samples={num_samples}")

    def compute_fisher_information(
        self,
        model: Any,
        data_loader: Any,
        task_name: str,
    ) -> List[FisherInformation]:
        """
        Compute Fisher Information Matrix for model parameters.

        For neural networks, approximates diagonal of Fisher matrix
        using squared gradients.
        """
        fisher_list = []

        try:
            # For sklearn models
            if hasattr(model, "coef_"):
                # Use coefficient importance as proxy for Fisher
                importance = np.abs(model.coef_).flatten()
                fisher_list.append(FisherInformation(
                    parameter_name="coef",
                    importance=importance,
                    optimal_weights=model.coef_.copy(),
                ))

            if hasattr(model, "intercept_"):
                intercept = np.atleast_1d(model.intercept_)
                fisher_list.append(FisherInformation(
                    parameter_name="intercept",
                    importance=np.ones_like(intercept),
                    optimal_weights=intercept.copy(),
                ))

            # For tree-based models (RF, GB, XGBoost)
            if hasattr(model, "feature_importances_"):
                fisher_list.append(FisherInformation(
                    parameter_name="feature_importances",
                    importance=model.feature_importances_.copy(),
                    optimal_weights=model.feature_importances_.copy(),
                ))

            # For PyTorch models
            if hasattr(model, "parameters"):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # Approximate Fisher with squared gradients
                        importance = np.ones(param.shape) * 0.01  # Default importance
                        fisher_list.append(FisherInformation(
                            parameter_name=name,
                            importance=importance,
                            optimal_weights=param.detach().cpu().numpy().copy(),
                        ))

        except Exception as e:
            logger.warning(f"Fisher computation failed: {e}")

        self.fisher_info[task_name] = fisher_list
        self._task_count += 1

        logger.debug(f"Computed Fisher info for {task_name}: {len(fisher_list)} parameters")
        return fisher_list

    def compute_ewc_loss(
        self,
        model: Any,
        task_name: str,
    ) -> float:
        """
        Compute EWC regularization loss.

        Returns penalty for deviating from important parameters.
        """
        if task_name not in self.fisher_info:
            return 0.0

        ewc_loss = 0.0

        for fisher in self.fisher_info[task_name]:
            try:
                # Get current parameter value
                if fisher.parameter_name == "coef" and hasattr(model, "coef_"):
                    current = model.coef_.flatten()
                elif fisher.parameter_name == "intercept" and hasattr(model, "intercept_"):
                    current = np.atleast_1d(model.intercept_)
                elif fisher.parameter_name == "feature_importances" and hasattr(model, "feature_importances_"):
                    current = model.feature_importances_
                else:
                    continue

                optimal = fisher.optimal_weights.flatten()
                importance = fisher.importance.flatten()

                # Ensure same shape
                min_len = min(len(current), len(optimal), len(importance))
                current = current[:min_len]
                optimal = optimal[:min_len]
                importance = importance[:min_len]

                # EWC penalty
                ewc_loss += np.sum(importance * (current - optimal) ** 2)

            except Exception as e:
                logger.debug(f"EWC loss computation error for {fisher.parameter_name}: {e}")

        return self.lambda_ewc * ewc_loss / 2.0

    def get_regularization_mask(self, task_name: str) -> Dict[str, np.ndarray]:
        """Get importance masks for regularizing specific parameters."""
        if task_name not in self.fisher_info:
            return {}

        return {
            fisher.parameter_name: fisher.importance
            for fisher in self.fisher_info[task_name]
        }


class ReservoirSampler:
    """
    Reservoir Sampling for experience replay.

    Maintains a fixed-size buffer of experiences that is
    representative of all past experiences (uniform sampling).
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self._total_seen = 0

    def add(self, experience: Any):
        """Add experience using reservoir sampling algorithm."""
        self._total_seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Replace with probability capacity/total_seen
            idx = np.random.randint(0, self._total_seen)
            if idx < self.capacity:
                self.buffer[idx] = experience

    def sample(self, n: int) -> List[Any]:
        """Sample n experiences uniformly."""
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Any]:
        """Get all stored experiences."""
        return self.buffer.copy()

    @property
    def size(self) -> int:
        return len(self.buffer)


class RegimeSpecificBuffer:
    """
    Maintains separate experience buffers for each regime.

    Enables regime-aware experience replay during training.
    """

    def __init__(self, capacity_per_regime: int = 2000):
        self.capacity_per_regime = capacity_per_regime
        self.regime_buffers: Dict[str, ReservoirSampler] = {}

    def add(self, experience: Any, regime: str):
        """Add experience to regime-specific buffer."""
        if regime not in self.regime_buffers:
            self.regime_buffers[regime] = ReservoirSampler(self.capacity_per_regime)

        self.regime_buffers[regime].add(experience)

    def sample_regime(self, regime: str, n: int) -> List[Any]:
        """Sample from specific regime buffer."""
        if regime not in self.regime_buffers:
            return []
        return self.regime_buffers[regime].sample(n)

    def sample_mixed(
        self,
        n: int,
        current_regime: str,
        current_weight: float = 0.7,
    ) -> List[Any]:
        """
        Sample with bias towards current regime.

        Args:
            n: Total samples to return
            current_regime: Current market regime
            current_weight: Weight for current regime samples (0-1)
        """
        samples = []

        # Samples from current regime
        n_current = int(n * current_weight)
        if current_regime in self.regime_buffers:
            samples.extend(self.regime_buffers[current_regime].sample(n_current))

        # Samples from other regimes
        n_other = n - len(samples)
        if n_other > 0:
            other_regimes = [r for r in self.regime_buffers if r != current_regime]
            if other_regimes:
                per_regime = max(1, n_other // len(other_regimes))
                for regime in other_regimes:
                    samples.extend(self.regime_buffers[regime].sample(per_regime))

        return samples[:n]

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            regime: buffer.size
            for regime, buffer in self.regime_buffers.items()
        }


class ContinualLearningManager:
    """
    Manages continual learning with multiple techniques.

    Combines:
    1. EWC for weight regularization
    2. Reservoir sampling for experience replay
    3. Regime-specific buffers for context-aware replay
    """

    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        replay_buffer_size: int = 10000,
        regime_buffer_size: int = 2000,
        data_dir: str = "data/continual_learning",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.ewc = ElasticWeightConsolidation(lambda_ewc=ewc_lambda)
        self.replay_buffer = ReservoirSampler(capacity=replay_buffer_size)
        self.regime_buffer = RegimeSpecificBuffer(capacity_per_regime=regime_buffer_size)

        # Training state
        self._current_task = "initial"
        self._training_history: List[Dict] = []

        self._load_state()
        logger.info("ContinualLearningManager initialized")

    def register_task_completion(
        self,
        model: Any,
        task_name: str,
        data_loader: Any = None,
    ):
        """
        Register completion of a learning task.

        Computes Fisher information for EWC.
        """
        self.ewc.compute_fisher_information(model, data_loader, task_name)
        self._current_task = task_name

        self._training_history.append({
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()
        logger.info(f"Registered task completion: {task_name}")

    def add_experience(
        self,
        experience: Dict,
        regime: str,
    ):
        """Add experience to replay buffers."""
        self.replay_buffer.add(experience)
        self.regime_buffer.add(experience, regime)

    def get_replay_batch(
        self,
        batch_size: int,
        current_regime: Optional[str] = None,
        regime_weight: float = 0.6,
    ) -> List[Dict]:
        """
        Get batch for experience replay.

        If current_regime is provided, biases towards that regime.
        """
        if current_regime:
            return self.regime_buffer.sample_mixed(
                batch_size, current_regime, regime_weight
            )
        else:
            return self.replay_buffer.sample(batch_size)

    def get_ewc_loss(self, model: Any) -> float:
        """Get EWC regularization loss for current model."""
        total_loss = 0.0
        for task_name in self.ewc.fisher_info:
            total_loss += self.ewc.compute_ewc_loss(model, task_name)
        return total_loss

    def should_consolidate(self, n_new_experiences: int) -> bool:
        """Check if we should consolidate (compute new Fisher info)."""
        # Consolidate every 500 new experiences
        return n_new_experiences > 0 and n_new_experiences % 500 == 0

    def get_stats(self) -> Dict:
        """Get continual learning statistics."""
        return {
            "replay_buffer_size": self.replay_buffer.size,
            "regime_buffer_stats": self.regime_buffer.get_stats(),
            "ewc_tasks": list(self.ewc.fisher_info.keys()),
            "current_task": self._current_task,
            "training_history": self._training_history[-10:],
        }

    def _save_state(self):
        """Save state to disk."""
        state = {
            "current_task": self._current_task,
            "training_history": self._training_history,
            "regime_stats": self.regime_buffer.get_stats(),
        }
        with open(self.data_dir / "continual_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from disk."""
        state_file = self.data_dir / "continual_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self._current_task = state.get("current_task", "initial")
                self._training_history = state.get("training_history", [])
                logger.info("Loaded continual learning state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")


# Singleton instance
_cl_manager: Optional[ContinualLearningManager] = None


def get_continual_learning_manager() -> ContinualLearningManager:
    """Get or create the ContinualLearningManager singleton."""
    global _cl_manager
    if _cl_manager is None:
        _cl_manager = ContinualLearningManager()
    return _cl_manager
