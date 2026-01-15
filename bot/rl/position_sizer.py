"""
RL-based Position Sizing Module.

Uses reinforcement learning to dynamically determine optimal position sizes
based on market conditions, portfolio state, and risk metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


@dataclass
class PositionSizerConfig:
    """Configuration for RL position sizer."""
    # State dimensions
    market_feature_dim: int = 20  # Technical indicators
    portfolio_feature_dim: int = 8  # Portfolio state
    regime_feature_dim: int = 5   # Market regime

    # Action space
    min_position_size: float = 0.0  # Minimum position (% of max)
    max_position_size: float = 1.0  # Maximum position
    position_granularity: int = 11  # Discrete levels (0, 0.1, 0.2, ..., 1.0)

    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = True

    # Training
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    buffer_size: int = 100000
    batch_size: int = 64

    # Risk constraints
    max_drawdown_threshold: float = 0.10
    position_limit_multiplier: float = 1.0
    volatility_scaling: bool = True


@dataclass
class PositionSizerState:
    """State representation for position sizer."""
    market_features: np.ndarray  # Technical indicators
    portfolio_state: np.ndarray  # Cash, positions, PnL
    regime_features: np.ndarray  # Market regime indicators
    signal_strength: float       # Strategy signal (-1 to 1)
    signal_confidence: float     # Confidence (0 to 1)
    volatility: float           # Current volatility
    drawdown: float             # Current drawdown from peak

    def to_tensor(self) -> np.ndarray:
        """Convert to flat tensor."""
        return np.concatenate([
            self.market_features,
            self.portfolio_state,
            self.regime_features,
            np.array([
                self.signal_strength,
                self.signal_confidence,
                self.volatility,
                self.drawdown
            ])
        ]).astype(np.float32)


class ReplayBuffer:
    """Experience replay buffer for position sizer training."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        return self.size


if HAS_TORCH:
    class PositionSizerNetwork(nn.Module):
        """
        Neural network for position sizing.

        Uses a continuous action space to output position size as a percentage.
        Implements SAC-style actor-critic with entropy regularization.
        """

        def __init__(self, config: PositionSizerConfig):
            super().__init__()
            self.config = config

            # Input dimension
            self.state_dim = (
                config.market_feature_dim +
                config.portfolio_feature_dim +
                config.regime_feature_dim +
                4  # signal_strength, confidence, volatility, drawdown
            )

            # Shared encoder
            self.encoder = self._build_encoder()

            # Actor (policy) network - outputs mean and std for position size
            self.actor_mean = nn.Linear(config.hidden_dim, 1)
            self.actor_log_std = nn.Linear(config.hidden_dim, 1)

            # Critic (Q-value) networks - twin Q for stability
            self.q1 = self._build_critic()
            self.q2 = self._build_critic()

            # Target networks
            self.q1_target = self._build_critic()
            self.q2_target = self._build_critic()

            # Copy weights to targets
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

            # Entropy coefficient (learnable)
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.target_entropy = -1.0  # Target entropy for 1D action

            self._init_weights()

        def _build_encoder(self) -> nn.Module:
            """Build shared state encoder."""
            layers = []
            in_dim = self.state_dim

            for i in range(self.config.num_layers):
                out_dim = self.config.hidden_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ])
                in_dim = out_dim

            if self.config.use_attention:
                # Self-attention layer
                layers.append(SelfAttention(self.config.hidden_dim))

            return nn.Sequential(*layers)

        def _build_critic(self) -> nn.Module:
            """Build Q-value network."""
            return nn.Sequential(
                nn.Linear(self.state_dim + 1, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 1)
            )

        def _init_weights(self) -> None:
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def encode(self, state: torch.Tensor) -> torch.Tensor:
            """Encode state through shared encoder."""
            return self.encoder(state)

        def get_action(
            self,
            state: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get position size action from policy.

            Returns:
                Tuple of (action, log_prob)
            """
            encoded = self.encode(state)

            mean = self.actor_mean(encoded)
            log_std = self.actor_log_std(encoded).clamp(-20, 2)
            std = log_std.exp()

            if deterministic:
                action = torch.sigmoid(mean)  # Map to [0, 1]
                log_prob = torch.zeros_like(action)
            else:
                # Sample from Gaussian and apply sigmoid
                normal = torch.distributions.Normal(mean, std)
                x = normal.rsample()
                action = torch.sigmoid(x)

                # Log prob with correction for sigmoid transformation
                log_prob = normal.log_prob(x) - torch.log(action * (1 - action) + 1e-6)

            return action.squeeze(-1), log_prob.squeeze(-1)

        def get_q_values(
            self,
            state: torch.Tensor,
            action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Get Q-values for state-action pair."""
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            sa = torch.cat([state, action], dim=-1)

            q1 = self.q1(sa).squeeze(-1)
            q2 = self.q2(sa).squeeze(-1)

            return q1, q2

        def get_target_q_values(
            self,
            state: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:
            """Get target Q-values (min of twin Q)."""
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            sa = torch.cat([state, action], dim=-1)

            q1 = self.q1_target(sa).squeeze(-1)
            q2 = self.q2_target(sa).squeeze(-1)

            return torch.min(q1, q2)

        def soft_update_targets(self, tau: float) -> None:
            """Soft update target networks."""
            for param, target_param in zip(
                self.q1.parameters(), self.q1_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            for param, target_param in zip(
                self.q2.parameters(), self.q2_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        @property
        def alpha(self) -> torch.Tensor:
            """Get entropy coefficient."""
            return self.log_alpha.exp()


    class SelfAttention(nn.Module):
        """Self-attention layer for feature importance."""

        def __init__(self, dim: int):
            super().__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.scale = dim ** 0.5

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # For 1D input, just apply learned transformation
            if x.dim() == 2:
                return self.value(x)

            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

            attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn = F.softmax(attn, dim=-1)

            return torch.matmul(attn, v)


class RLPositionSizer:
    """
    RL-based Position Sizer.

    Uses SAC (Soft Actor-Critic) to learn optimal position sizes
    based on market conditions and risk constraints.

    Features:
    - Continuous action space for fine-grained sizing
    - Risk-adjusted rewards
    - Volatility scaling
    - Drawdown protection
    - Online and batch learning modes

    Usage:
        sizer = RLPositionSizer()

        # Get position size
        state = PositionSizerState(...)
        position_size = sizer.get_position_size(state)

        # Update with feedback
        sizer.update(state, position_size, reward, next_state, done)
    """

    def __init__(
        self,
        config: Optional[PositionSizerConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize RL position sizer.

        Args:
            config: Sizer configuration
            model_path: Path to load pre-trained model
        """
        self.config = config or PositionSizerConfig()

        # State dimension
        self.state_dim = (
            self.config.market_feature_dim +
            self.config.portfolio_feature_dim +
            self.config.regime_feature_dim +
            4  # Additional features
        )

        # Initialize network
        if HAS_TORCH:
            self.network = PositionSizerNetwork(self.config)

            # Optimizers
            self.actor_optimizer = optim.Adam(
                list(self.network.encoder.parameters()) +
                list(self.network.actor_mean.parameters()) +
                list(self.network.actor_log_std.parameters()),
                lr=self.config.learning_rate
            )

            self.critic_optimizer = optim.Adam(
                list(self.network.q1.parameters()) +
                list(self.network.q2.parameters()),
                lr=self.config.learning_rate
            )

            self.alpha_optimizer = optim.Adam(
                [self.network.log_alpha],
                lr=self.config.learning_rate
            )

            # Load pre-trained if available
            if model_path and Path(model_path).exists():
                self.load(model_path)
        else:
            self.network = None

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.config.buffer_size,
            self.state_dim
        )

        # Training state
        self.training_steps = 0
        self.episode_rewards: List[float] = []

        # Performance tracking
        self._position_history: List[Dict[str, Any]] = []
        self._reward_history: List[float] = []

    def get_position_size(
        self,
        state: PositionSizerState,
        deterministic: bool = False
    ) -> float:
        """
        Get optimal position size for given state.

        Args:
            state: Current market/portfolio state
            deterministic: If True, return mean action (no exploration)

        Returns:
            Position size as fraction of maximum (0.0 to 1.0)
        """
        if self.network is None:
            # Fallback: simple volatility-based sizing
            return self._fallback_sizing(state)

        state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.network.get_action(state_tensor, deterministic)
            position_size = action.item()

        # Apply risk constraints
        position_size = self._apply_risk_constraints(position_size, state)

        # Record for analysis
        self._position_history.append({
            "timestamp": datetime.now().isoformat(),
            "signal_strength": state.signal_strength,
            "signal_confidence": state.signal_confidence,
            "volatility": state.volatility,
            "drawdown": state.drawdown,
            "raw_size": action.item() if HAS_TORCH else position_size,
            "final_size": position_size,
        })

        # Keep limited history
        if len(self._position_history) > 10000:
            self._position_history = self._position_history[-5000:]

        return position_size

    def _fallback_sizing(self, state: PositionSizerState) -> float:
        """Simple volatility-based sizing when RL not available."""
        # Base size from signal strength
        base_size = abs(state.signal_strength) * state.signal_confidence

        # Volatility scaling
        if self.config.volatility_scaling and state.volatility > 0:
            target_vol = 0.02  # 2% daily target volatility
            vol_scale = target_vol / state.volatility
            vol_scale = np.clip(vol_scale, 0.5, 2.0)
            base_size *= vol_scale

        # Drawdown protection
        if state.drawdown > self.config.max_drawdown_threshold:
            reduction = 1 - (state.drawdown / self.config.max_drawdown_threshold)
            base_size *= max(0.1, reduction)

        return np.clip(base_size, 0.0, 1.0)

    def _apply_risk_constraints(
        self,
        position_size: float,
        state: PositionSizerState
    ) -> float:
        """Apply risk constraints to position size."""
        # Drawdown constraint
        if state.drawdown > self.config.max_drawdown_threshold:
            max_allowed = 1.0 - (state.drawdown / self.config.max_drawdown_threshold)
            position_size = min(position_size, max(0.1, max_allowed))

        # Volatility constraint
        if self.config.volatility_scaling and state.volatility > 0:
            target_vol = 0.02
            if state.volatility > target_vol * 2:
                # High volatility - reduce position
                vol_adjustment = target_vol / state.volatility
                position_size *= vol_adjustment

        # Position limit
        position_size *= self.config.position_limit_multiplier

        return np.clip(position_size, self.config.min_position_size, self.config.max_position_size)

    def update(
        self,
        state: PositionSizerState,
        action: float,
        reward: float,
        next_state: PositionSizerState,
        done: bool
    ) -> Dict[str, float]:
        """
        Update position sizer with experience.

        Args:
            state: State when action was taken
            action: Position size chosen
            reward: Reward received (risk-adjusted PnL)
            next_state: Resulting state
            done: Episode terminal flag

        Returns:
            Training metrics
        """
        # Add to buffer
        self.buffer.add(
            state.to_tensor(),
            action,
            reward,
            next_state.to_tensor(),
            done
        )

        self._reward_history.append(reward)

        # Train if enough samples
        if len(self.buffer) >= self.config.batch_size:
            return self._train_step()

        return {}

    def _train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if not HAS_TORCH or self.network is None:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute target Q-value
        with torch.no_grad():
            next_actions, next_log_probs = self.network.get_action(next_states)
            target_q = self.network.get_target_q_values(next_states, next_actions)
            target_q = rewards + self.config.gamma * (1 - dones) * (
                target_q - self.network.alpha * next_log_probs
            )

        # Update critics
        q1, q2 = self.network.get_q_values(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.network.get_action(states)
        q1_new, q2_new = self.network.get_q_values(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.network.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy coefficient
        alpha_loss = -(self.network.log_alpha * (
            log_probs.detach() + self.network.target_entropy
        )).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets
        self.network.soft_update_targets(self.config.tau)

        self.training_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.network.alpha.item(),
        }

    def calculate_reward(
        self,
        pnl: float,
        position_size: float,
        volatility: float,
        drawdown: float,
        holding_time: int
    ) -> float:
        """
        Calculate risk-adjusted reward for position.

        Reward components:
        - PnL (main objective)
        - Risk penalty (volatility, drawdown)
        - Efficiency bonus (good sizing)
        - Holding cost (discourage excessive trading)
        """
        reward = 0.0

        # PnL component (scaled)
        reward += pnl * 100

        # Risk penalty
        if drawdown > self.config.max_drawdown_threshold:
            reward -= (drawdown - self.config.max_drawdown_threshold) * 50

        # Volatility penalty for large positions in high vol
        if volatility > 0.03 and position_size > 0.5:
            reward -= (volatility - 0.03) * position_size * 20

        # Efficiency bonus: good sizing for the volatility
        optimal_size = 0.02 / max(volatility, 0.001)
        size_efficiency = 1 - abs(position_size - optimal_size)
        if size_efficiency > 0.5 and pnl > 0:
            reward += size_efficiency * 5

        # Holding cost (small penalty per step)
        reward -= holding_time * 0.001

        return reward

    def get_status(self) -> Dict[str, Any]:
        """Get sizer status and statistics."""
        recent_positions = self._position_history[-100:]
        recent_rewards = self._reward_history[-100:]

        avg_position = 0.0
        avg_reward = 0.0

        if recent_positions:
            avg_position = np.mean([p["final_size"] for p in recent_positions])

        if recent_rewards:
            avg_reward = np.mean(recent_rewards)

        return {
            "training_steps": self.training_steps,
            "buffer_size": len(self.buffer),
            "avg_position_size": float(avg_position),
            "avg_reward": float(avg_reward),
            "total_decisions": len(self._position_history),
            "has_torch": HAS_TORCH,
            "alpha": self.network.alpha.item() if HAS_TORCH and self.network else 0.0,
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_TORCH and self.network:
            torch.save({
                "network_state": self.network.state_dict(),
                "config": self.config,
                "training_steps": self.training_steps,
            }, path)
            logger.info(f"Saved position sizer to {path}")
        else:
            # Save config only
            with open(path, "w") as f:
                json.dump({
                    "training_steps": self.training_steps,
                    "config": self.config.__dict__,
                }, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from file."""
        if HAS_TORCH and self.network:
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint["network_state"])
            self.training_steps = checkpoint.get("training_steps", 0)
            logger.info(f"Loaded position sizer from {path}")


class DiscretePositionSizer:
    """
    Discrete action space position sizer.

    Simpler alternative using DQN-style discrete actions.
    Position sizes: [0%, 10%, 20%, ..., 100%]
    """

    def __init__(
        self,
        config: Optional[PositionSizerConfig] = None,
        model_path: Optional[str] = None
    ):
        self.config = config or PositionSizerConfig()
        self.n_actions = self.config.position_granularity

        # Map action index to position size
        self.action_to_size = {
            i: i / (self.n_actions - 1) for i in range(self.n_actions)
        }

        self.state_dim = (
            self.config.market_feature_dim +
            self.config.portfolio_feature_dim +
            self.config.regime_feature_dim +
            4
        )

        # Q-network
        if HAS_TORCH:
            self.q_network = self._build_q_network()
            self.target_network = self._build_q_network()
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.config.learning_rate
            )

            if model_path and Path(model_path).exists():
                self.load(model_path)
        else:
            self.q_network = None
            self.target_network = None

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Buffer
        self.buffer = ReplayBuffer(self.config.buffer_size, self.state_dim)
        self.training_steps = 0

    def _build_q_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.n_actions)
        )

    def get_position_size(
        self,
        state: PositionSizerState,
        deterministic: bool = False
    ) -> float:
        """Get position size."""
        if not HAS_TORCH or self.q_network is None:
            return self._fallback_sizing(state)

        state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0)

        # Epsilon-greedy
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=-1).item()

        position_size = self.action_to_size[action]

        # Apply risk constraints
        return self._apply_risk_constraints(position_size, state)

    def _fallback_sizing(self, state: PositionSizerState) -> float:
        """Fallback sizing without RL."""
        base_size = abs(state.signal_strength) * state.signal_confidence
        return np.clip(base_size, 0.0, 1.0)

    def _apply_risk_constraints(
        self,
        position_size: float,
        state: PositionSizerState
    ) -> float:
        """Apply risk constraints."""
        if state.drawdown > self.config.max_drawdown_threshold:
            max_allowed = 1.0 - (state.drawdown / self.config.max_drawdown_threshold)
            position_size = min(position_size, max(0.1, max_allowed))

        return np.clip(position_size, 0.0, 1.0)

    def update(
        self,
        state: PositionSizerState,
        action: float,
        reward: float,
        next_state: PositionSizerState,
        done: bool
    ) -> Dict[str, float]:
        """Update with experience."""
        self.buffer.add(
            state.to_tensor(),
            action,
            reward,
            next_state.to_tensor(),
            done
        )

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.buffer) >= self.config.batch_size:
            return self._train_step()

        return {}

    def _train_step(self) -> Dict[str, float]:
        """Train step with DQN."""
        if not HAS_TORCH or self.q_network is None:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(
            [int(a * (self.n_actions - 1)) for a in actions]
        )
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=-1)[0]
            target_q = rewards + self.config.gamma * (1 - dones) * next_q

        # Loss
        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        return {
            "training_steps": self.training_steps,
            "buffer_size": len(self.buffer),
            "epsilon": self.epsilon,
            "n_actions": self.n_actions,
        }

    def save(self, path: str) -> None:
        """Save model."""
        if HAS_TORCH and self.q_network:
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            }, path)

    def load(self, path: str) -> None:
        """Load model."""
        if HAS_TORCH and self.q_network:
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.epsilon = checkpoint.get("epsilon", 0.05)
            self.training_steps = checkpoint.get("training_steps", 0)
