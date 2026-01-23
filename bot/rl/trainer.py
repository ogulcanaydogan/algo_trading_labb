"""
PPO Trainer for RL Trading Agent.

Implements Proximal Policy Optimization for training trading policies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .environment import TradingEnvironment
from .policy_network import TradingPolicyNetwork, PolicyConfig

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2  # PPO clip ratio
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01  # Exploration bonus

    # Training schedule
    n_steps: int = 2048  # Steps per rollout
    n_epochs: int = 10  # PPO epochs per update
    batch_size: int = 64
    n_updates: int = 100  # Total updates

    # Optimization
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01  # Early stopping

    # Evaluation
    eval_freq: int = 10  # Evaluate every N updates
    n_eval_episodes: int = 5


@dataclass
class TrainingResults:
    """Results from training."""

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    eval_returns: List[float] = field(default_factory=list)
    best_eval_return: float = -np.inf
    total_timesteps: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_rewards": self.episode_rewards[-100:],
            "episode_lengths": self.episode_lengths[-100:],
            "policy_losses": self.policy_losses[-100:],
            "value_losses": self.value_losses[-100:],
            "entropy_losses": self.entropy_losses[-100:],
            "eval_returns": self.eval_returns,
            "best_eval_return": self.best_eval_return,
            "total_timesteps": self.total_timesteps,
            "training_time_seconds": self.training_time_seconds,
        }


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, buffer_size: int, state_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim

        # Buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute advantages and returns for completed path."""
        path_slice = slice(self.path_start_idx, self.ptr)

        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # Append last value for bootstrapping
        values_extended = np.append(values, last_value)

        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        # Returns = advantages + values
        returns = advantages + values

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns

        self.path_start_idx = self.ptr

    def get(self) -> Tuple[np.ndarray, ...]:
        """Get all data from buffer."""
        data = (
            self.states[: self.ptr],
            self.actions[: self.ptr],
            self.log_probs[: self.ptr],
            self.advantages[: self.ptr],
            self.returns[: self.ptr],
        )

        # Normalize advantages
        adv = data[3]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return (data[0], data[1], data[2], adv, data[4])

    def reset(self) -> None:
        """Reset buffer."""
        self.ptr = 0
        self.path_start_idx = 0


class PPOTrainer:
    """
    Proximal Policy Optimization Trainer.

    Implements PPO algorithm for training trading policies:
    - Clipped objective for stable updates
    - GAE for advantage estimation
    - Entropy bonus for exploration
    - Early stopping on KL divergence

    Usage:
        trainer = PPOTrainer(policy, env)
        results = trainer.train(n_updates=100)
    """

    def __init__(
        self,
        policy: TradingPolicyNetwork,
        env: TradingEnvironment,
        config: Optional[TrainingConfig] = None,
        save_dir: str = "data/rl_training",
    ):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy network to train
            env: Training environment
            config: Training configuration
            save_dir: Directory for saving checkpoints
        """
        self.policy = policy
        self.env = env
        self.config = config or TrainingConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer if PyTorch available
        if HAS_TORCH:
            self.optimizer = optim.Adam(
                policy.parameters(),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = None

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            state_dim=env.observation_space.shape[0],
        )

        # Results tracking
        self.results = TrainingResults()

    def train(self, n_updates: Optional[int] = None) -> TrainingResults:
        """
        Train the policy.

        Args:
            n_updates: Number of updates to perform

        Returns:
            TrainingResults with metrics
        """
        import time

        start_time = time.time()

        n_updates = n_updates or self.config.n_updates

        logger.info(f"Starting PPO training for {n_updates} updates")

        for update in range(n_updates):
            # Collect rollouts
            self._collect_rollouts()

            # Update policy
            policy_loss, value_loss, entropy_loss = self._update_policy()

            self.results.policy_losses.append(policy_loss)
            self.results.value_losses.append(value_loss)
            self.results.entropy_losses.append(entropy_loss)

            # Evaluation
            if (update + 1) % self.config.eval_freq == 0:
                eval_return = self._evaluate()
                self.results.eval_returns.append(eval_return)

                if eval_return > self.results.best_eval_return:
                    self.results.best_eval_return = eval_return
                    self._save_checkpoint("best")

                logger.info(
                    f"Update {update + 1}/{n_updates}: "
                    f"Policy Loss={policy_loss:.4f}, "
                    f"Value Loss={value_loss:.4f}, "
                    f"Eval Return={eval_return:.4f}"
                )

        self.results.training_time_seconds = time.time() - start_time
        self._save_checkpoint("final")
        self._save_results()

        return self.results

    def _collect_rollouts(self) -> None:
        """Collect rollouts from environment."""
        self.buffer.reset()

        state = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(self.config.n_steps):
            # Get action from policy
            if HAS_TORCH:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, value = self.policy.get_action(state_tensor)
            else:
                action, log_prob, value = self.policy.get_action(state)

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store in buffer
            self.buffer.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                # Finish path with zero value for terminal state
                self.buffer.finish_path(
                    last_value=0,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )

                self.results.episode_rewards.append(episode_reward)
                self.results.episode_lengths.append(episode_length)
                self.results.total_timesteps += episode_length

                # Reset
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0

        # Bootstrap if episode not done
        if not done:
            if HAS_TORCH:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, _, last_value = self.policy.get_action(state_tensor)
            else:
                _, _, last_value = self.policy.get_action(state)

            self.buffer.finish_path(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

    def _update_policy(self) -> Tuple[float, float, float]:
        """Update policy using PPO."""
        if not HAS_TORCH:
            return 0.0, 0.0, 0.0

        # Get data from buffer
        states, actions, old_log_probs, advantages, returns = self.buffer.get()

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_batches = 0

        for _ in range(self.config.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (clipped PPO objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy loss (negative for maximization)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_batches += 1

                # KL divergence early stopping
                if self.config.target_kl is not None:
                    approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
                    if abs(approx_kl) > self.config.target_kl:
                        break

        return (
            total_policy_loss / max(n_batches, 1),
            total_value_loss / max(n_batches, 1),
            total_entropy_loss / max(n_batches, 1),
        )

    def _evaluate(self) -> float:
        """Evaluate current policy."""
        total_return = 0.0

        for _ in range(self.config.n_eval_episodes):
            state = self.env.reset()
            episode_return = 0.0
            done = False

            while not done:
                if HAS_TORCH:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action, _, _ = self.policy.get_action(state_tensor, deterministic=True)
                else:
                    action, _, _ = self.policy.get_action(state, deterministic=True)

                state, reward, done, _ = self.env.step(action)
                episode_return += reward

            total_return += episode_return

        return total_return / self.config.n_eval_episodes

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.save_dir / f"policy_{name}.pt"
        self.policy.save(str(path))
        logger.info(f"Saved checkpoint to {path}")

    def _save_results(self) -> None:
        """Save training results."""
        path = self.save_dir / "training_results.json"
        with open(path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)
