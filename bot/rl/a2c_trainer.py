"""
A2C (Advantage Actor-Critic) Trainer for RL Trading Agent.

Alternative to PPO with synchronous updates for faster training.
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
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


@dataclass
class A2CConfig:
    """Configuration for A2C training."""
    # Learning parameters
    learning_rate: float = 7e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01  # Exploration bonus
    max_grad_norm: float = 0.5

    # Training schedule
    n_steps: int = 5  # Steps per update (A2C uses smaller values than PPO)
    n_envs: int = 1   # Number of parallel environments
    total_timesteps: int = 100000

    # Optimization
    use_rms_prop: bool = True  # Original A2C uses RMSprop
    rms_alpha: float = 0.99
    rms_epsilon: float = 1e-5

    # Evaluation
    eval_freq: int = 1000
    n_eval_episodes: int = 5

    # Learning rate schedule
    lr_schedule: str = "linear"  # "linear", "constant", "cosine"


@dataclass
class A2CResults:
    """Results from A2C training."""
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


class A2CTrainer:
    """
    Advantage Actor-Critic (A2C) Trainer.

    Synchronous version of A3C algorithm. Simpler than PPO but
    can be faster for simple environments.

    Key differences from PPO:
    - No clipping (uses vanilla policy gradient)
    - Smaller batch sizes (n_steps)
    - Synchronous updates (vs async in A3C)
    - Often uses RMSprop optimizer

    Usage:
        trainer = A2CTrainer(policy, env)
        results = trainer.train(total_timesteps=100000)
    """

    def __init__(
        self,
        policy: TradingPolicyNetwork,
        env: TradingEnvironment,
        config: Optional[A2CConfig] = None,
        save_dir: str = "data/a2c_training"
    ):
        """
        Initialize A2C trainer.

        Args:
            policy: Policy network to train
            env: Training environment
            config: Training configuration
            save_dir: Directory for saving checkpoints
        """
        self.policy = policy
        self.env = env
        self.config = config or A2CConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        if HAS_TORCH:
            if self.config.use_rms_prop:
                self.optimizer = optim.RMSprop(
                    policy.parameters(),
                    lr=self.config.learning_rate,
                    alpha=self.config.rms_alpha,
                    eps=self.config.rms_epsilon
                )
            else:
                self.optimizer = optim.Adam(
                    policy.parameters(),
                    lr=self.config.learning_rate
                )
        else:
            self.optimizer = None

        # Results tracking
        self.results = A2CResults()

        # Current episode tracking
        self._episode_reward = 0.0
        self._episode_length = 0

    def train(self, total_timesteps: Optional[int] = None) -> A2CResults:
        """
        Train the policy using A2C.

        Args:
            total_timesteps: Total environment steps to train

        Returns:
            A2CResults with training metrics
        """
        import time
        start_time = time.time()

        total_timesteps = total_timesteps or self.config.total_timesteps
        n_updates = total_timesteps // self.config.n_steps

        logger.info(f"Starting A2C training for {total_timesteps} timesteps")

        # Initialize environment
        state = self.env.reset()
        self._episode_reward = 0.0
        self._episode_length = 0

        for update in range(n_updates):
            # Adjust learning rate
            if self.config.lr_schedule == "linear":
                frac = 1.0 - update / n_updates
                lr = self.config.learning_rate * frac
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            # Collect n_steps of experience
            states, actions, rewards, dones, values, log_probs, state = self._collect_rollout(state)

            # Compute returns and advantages
            returns, advantages = self._compute_returns(
                rewards, values, dones, state
            )

            # Update policy
            policy_loss, value_loss, entropy_loss = self._update(
                states, actions, returns, advantages, log_probs
            )

            # Record losses
            self.results.policy_losses.append(policy_loss)
            self.results.value_losses.append(value_loss)
            self.results.entropy_losses.append(entropy_loss)
            self.results.total_timesteps = (update + 1) * self.config.n_steps

            # Evaluation
            if (update + 1) * self.config.n_steps % self.config.eval_freq < self.config.n_steps:
                eval_return = self._evaluate()
                self.results.eval_returns.append(eval_return)

                if eval_return > self.results.best_eval_return:
                    self.results.best_eval_return = eval_return
                    self._save_checkpoint("best")

                logger.info(
                    f"Step {self.results.total_timesteps}/{total_timesteps}: "
                    f"Policy Loss={policy_loss:.4f}, "
                    f"Value Loss={value_loss:.4f}, "
                    f"Eval Return={eval_return:.4f}"
                )

        self.results.training_time_seconds = time.time() - start_time
        self._save_checkpoint("final")
        self._save_results()

        return self.results

    def _collect_rollout(
        self,
        state: np.ndarray
    ) -> Tuple[List, List, List, List, List, List, np.ndarray]:
        """
        Collect n_steps of experience.

        Returns:
            Tuple of (states, actions, rewards, dones, values, log_probs, final_state)
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        for _ in range(self.config.n_steps):
            # Get action from policy
            if HAS_TORCH:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, value = self.policy.get_action(state_tensor)
            else:
                action = self.env.action_space.sample()
                log_prob = 0.0
                value = 0.0

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)

            # Track episode
            self._episode_reward += reward
            self._episode_length += 1

            if done:
                self.results.episode_rewards.append(self._episode_reward)
                self.results.episode_lengths.append(self._episode_length)
                self._episode_reward = 0.0
                self._episode_length = 0
                state = self.env.reset()
            else:
                state = next_state

        return states, actions, rewards, dones, values, log_probs, state

    def _compute_returns(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            last_state: Final state for bootstrapping

        Returns:
            Tuple of (returns, advantages)
        """
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)

        # Bootstrap value for last state
        if HAS_TORCH:
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0)
            with torch.no_grad():
                _, last_value, _ = self.policy.forward(state_tensor)
                last_value = last_value.item()
        else:
            last_value = 0.0

        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values

        return returns, advantages

    def _update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        returns: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: List[float]
    ) -> Tuple[float, float, float]:
        """
        Update policy with collected experience.

        Returns:
            Tuple of (policy_loss, value_loss, entropy_loss)
        """
        if not HAS_TORCH:
            return 0.0, 0.0, 0.0

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        new_log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

        # Policy loss (vanilla policy gradient with advantage)
        policy_loss = -(new_log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (negative for maximization)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

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
                    action = 1  # FLAT

                state, reward, done, _ = self.env.step(action)
                episode_return += reward

            total_return += episode_return

        return total_return / self.config.n_eval_episodes

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.save_dir / f"a2c_policy_{name}.pt"
        self.policy.save(str(path))
        logger.info(f"Saved A2C checkpoint to {path}")

    def _save_results(self) -> None:
        """Save training results."""
        path = self.save_dir / "a2c_training_results.json"
        with open(path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)


class MultiEnvA2CTrainer:
    """
    A2C trainer with multiple parallel environments.

    Improves sample efficiency by collecting experience
    from multiple environments simultaneously.
    """

    def __init__(
        self,
        policy: TradingPolicyNetwork,
        env_fn: callable,
        n_envs: int = 4,
        config: Optional[A2CConfig] = None,
        save_dir: str = "data/a2c_multi_training"
    ):
        """
        Initialize multi-env A2C trainer.

        Args:
            policy: Policy network to train
            env_fn: Function that creates a new environment
            n_envs: Number of parallel environments
            config: Training configuration
            save_dir: Directory for saving checkpoints
        """
        self.policy = policy
        self.n_envs = n_envs
        self.config = config or A2CConfig()
        self.config.n_envs = n_envs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create environments
        self.envs = [env_fn() for _ in range(n_envs)]

        # Setup optimizer
        if HAS_TORCH:
            if self.config.use_rms_prop:
                self.optimizer = optim.RMSprop(
                    policy.parameters(),
                    lr=self.config.learning_rate,
                    alpha=self.config.rms_alpha,
                    eps=self.config.rms_epsilon
                )
            else:
                self.optimizer = optim.Adam(
                    policy.parameters(),
                    lr=self.config.learning_rate
                )
        else:
            self.optimizer = None

        self.results = A2CResults()

    def train(self, total_timesteps: Optional[int] = None) -> A2CResults:
        """Train with multiple environments."""
        import time
        start_time = time.time()

        total_timesteps = total_timesteps or self.config.total_timesteps
        n_updates = total_timesteps // (self.config.n_steps * self.n_envs)

        logger.info(
            f"Starting multi-env A2C training: "
            f"{total_timesteps} timesteps, {self.n_envs} envs"
        )

        # Initialize states
        states = [env.reset() for env in self.envs]
        episode_rewards = [0.0] * self.n_envs
        episode_lengths = [0] * self.n_envs

        for update in range(n_updates):
            # Learning rate schedule
            if self.config.lr_schedule == "linear":
                frac = 1.0 - update / n_updates
                lr = self.config.learning_rate * frac
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            # Collect rollout from all envs
            all_states = []
            all_actions = []
            all_rewards = []
            all_dones = []
            all_values = []
            all_log_probs = []

            for step in range(self.config.n_steps):
                step_states = []
                step_actions = []
                step_rewards = []
                step_dones = []
                step_values = []
                step_log_probs = []

                for i, (env, state) in enumerate(zip(self.envs, states)):
                    # Get action
                    if HAS_TORCH:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action, log_prob, value = self.policy.get_action(state_tensor)
                    else:
                        action = env.action_space.sample()
                        log_prob = 0.0
                        value = 0.0

                    # Step env
                    next_state, reward, done, _ = env.step(action)

                    step_states.append(state)
                    step_actions.append(action)
                    step_rewards.append(reward)
                    step_dones.append(done)
                    step_values.append(value)
                    step_log_probs.append(log_prob)

                    episode_rewards[i] += reward
                    episode_lengths[i] += 1

                    if done:
                        self.results.episode_rewards.append(episode_rewards[i])
                        self.results.episode_lengths.append(episode_lengths[i])
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        states[i] = env.reset()
                    else:
                        states[i] = next_state

                all_states.append(step_states)
                all_actions.append(step_actions)
                all_rewards.append(step_rewards)
                all_dones.append(step_dones)
                all_values.append(step_values)
                all_log_probs.append(step_log_probs)

            # Compute returns and advantages for all envs
            returns, advantages = self._compute_multi_env_returns(
                all_rewards, all_values, all_dones, states
            )

            # Flatten all data
            flat_states = np.array(all_states).reshape(-1, all_states[0][0].shape[0])
            flat_actions = np.array(all_actions).flatten()
            flat_returns = returns.flatten()
            flat_advantages = advantages.flatten()

            # Update
            policy_loss, value_loss, entropy_loss = self._update(
                flat_states, flat_actions, flat_returns, flat_advantages
            )

            self.results.policy_losses.append(policy_loss)
            self.results.value_losses.append(value_loss)
            self.results.entropy_losses.append(entropy_loss)
            self.results.total_timesteps = (update + 1) * self.config.n_steps * self.n_envs

            # Log progress
            if (update + 1) % 100 == 0:
                avg_reward = np.mean(self.results.episode_rewards[-100:]) if self.results.episode_rewards else 0
                logger.info(
                    f"Update {update + 1}/{n_updates}: "
                    f"Avg Reward={avg_reward:.2f}, "
                    f"Policy Loss={policy_loss:.4f}"
                )

        self.results.training_time_seconds = time.time() - start_time
        self._save_checkpoint("final")
        self._save_results()

        return self.results

    def _compute_multi_env_returns(
        self,
        all_rewards: List[List[float]],
        all_values: List[List[float]],
        all_dones: List[List[bool]],
        last_states: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns for all environments."""
        all_rewards = np.array(all_rewards)  # (n_steps, n_envs)
        all_values = np.array(all_values)
        all_dones = np.array(all_dones)

        # Bootstrap values
        last_values = []
        for state in last_states:
            if HAS_TORCH:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    _, value, _ = self.policy.forward(state_tensor)
                    last_values.append(value.item())
            else:
                last_values.append(0.0)
        last_values = np.array(last_values)

        # GAE for each env
        advantages = np.zeros_like(all_rewards)

        for env_idx in range(self.n_envs):
            last_gae = 0
            for t in reversed(range(self.config.n_steps)):
                if t == self.config.n_steps - 1:
                    next_value = last_values[env_idx]
                    next_non_terminal = 1.0 - all_dones[t, env_idx]
                else:
                    next_value = all_values[t + 1, env_idx]
                    next_non_terminal = 1.0 - all_dones[t, env_idx]

                delta = (
                    all_rewards[t, env_idx] +
                    self.config.gamma * next_value * next_non_terminal -
                    all_values[t, env_idx]
                )
                advantages[t, env_idx] = last_gae = (
                    delta +
                    self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
                )

        returns = advantages + all_values
        return returns, advantages

    def _update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray
    ) -> Tuple[float, float, float]:
        """Update policy."""
        if not HAS_TORCH:
            return 0.0, 0.0, 0.0

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward
        log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def _save_checkpoint(self, name: str) -> None:
        """Save checkpoint."""
        path = self.save_dir / f"a2c_multi_policy_{name}.pt"
        self.policy.save(str(path))

    def _save_results(self) -> None:
        """Save results."""
        path = self.save_dir / "a2c_multi_training_results.json"
        with open(path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)
