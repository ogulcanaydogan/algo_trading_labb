"""
Reinforcement Learning Trading Agent

Uses Deep Q-Learning (DQN) to learn optimal trading actions.

Features:
- State representation from market data
- Action space: BUY, SELL, HOLD
- Reward based on P&L and risk-adjusted returns
- Experience replay for stable learning
- Target network for stability
- Can learn without predefined rules
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

from .learning_db import LearningDatabase, get_learning_db

logger = logging.getLogger(__name__)

# Try to import torch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using simplified RL agent.")


# Action space
class Action:
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class State:
    """Market state representation for RL agent."""
    # Price features
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    price_vs_ema20: float
    price_vs_ema50: float

    # Momentum features
    rsi: float
    rsi_change: float
    macd_hist: float
    momentum_5: float

    # Volatility features
    atr_ratio: float
    bb_position: float
    volatility_ratio: float

    # Trend features
    adx: float
    trend_direction: float  # +1 up, -1 down, 0 sideways

    # Volume features
    volume_ratio: float

    # Position features
    current_position: float  # -1 short, 0 flat, 1 long
    position_pnl: float
    position_duration: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network."""
        return np.array([
            self.price_change_1h,
            self.price_change_4h,
            self.price_change_24h,
            self.price_vs_ema20,
            self.price_vs_ema50,
            self.rsi / 100.0,  # Normalize
            self.rsi_change / 100.0,
            self.macd_hist,
            self.momentum_5 / 100.0,
            self.atr_ratio,
            self.bb_position,
            self.volatility_ratio,
            self.adx / 100.0,
            self.trend_direction,
            self.volume_ratio,
            self.current_position,
            self.position_pnl / 100.0,
            min(self.position_duration / 100.0, 1.0),
        ], dtype=np.float32)

    @classmethod
    def from_market_data(
        cls,
        indicators: Dict[str, float],
        position: float = 0,
        position_pnl: float = 0,
        position_duration: int = 0,
    ) -> 'State':
        """Create state from indicator dict."""
        return cls(
            price_change_1h=indicators.get('price_change_1h', 0),
            price_change_4h=indicators.get('price_change_4h', 0),
            price_change_24h=indicators.get('price_change_24h', 0),
            price_vs_ema20=indicators.get('price_vs_ema20', 0),
            price_vs_ema50=indicators.get('price_vs_ema50', 0),
            rsi=indicators.get('rsi', 50),
            rsi_change=indicators.get('rsi_change', 0),
            macd_hist=indicators.get('macd_hist', 0),
            momentum_5=indicators.get('momentum_5', 0),
            atr_ratio=indicators.get('atr_ratio', 1),
            bb_position=indicators.get('bb_position', 0.5),
            volatility_ratio=indicators.get('volatility_ratio', 1),
            adx=indicators.get('adx', 25),
            trend_direction=indicators.get('trend_direction', 0),
            volume_ratio=indicators.get('volume_ratio', 1),
            current_position=position,
            position_pnl=position_pnl,
            position_duration=position_duration,
        )


if TORCH_AVAILABLE:
    class DQNetwork(nn.Module):
        """Deep Q-Network for action value estimation."""

        def __init__(
            self,
            state_size: int = 18,
            action_size: int = 3,
            hidden_sizes: List[int] = [128, 64, 32],
        ):
            super().__init__()

            layers = []
            prev_size = state_size

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, action_size))

            self.network = nn.Sequential(*layers)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class RLTradingAgent:
    """
    Deep Q-Learning agent for trading.

    Learns optimal actions through:
    1. Observing market state
    2. Taking actions (BUY/SELL/HOLD)
    3. Receiving rewards (P&L, risk-adjusted)
    4. Updating Q-values via neural network
    """

    def __init__(
        self,
        state_size: int = 18,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,  # Discount factor
        epsilon: float = 1.0,  # Exploration rate
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        db: LearningDatabase = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.db = db or get_learning_db()

        self.memory = ReplayBuffer()
        self.training_steps = 0

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQNetwork(state_size, action_size).to(self.device)
            self.target_net = DQNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            # Simplified linear model without PyTorch
            self.weights = np.random.randn(state_size, action_size) * 0.01
            self.bias = np.zeros(action_size)
            self.learning_rate = learning_rate

    def select_action(
        self,
        state: State,
        training: bool = True,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current market state
            training: If True, use exploration; if False, exploit only

        Returns:
            Action (0=HOLD, 1=BUY, 2=SELL)
        """
        state_array = state.to_array()

        # Exploration (random action)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation (best action from model)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            # Linear model prediction
            q_values = np.dot(state_array, self.weights) + self.bias
            return np.argmax(q_values)

    def calculate_reward(
        self,
        action: int,
        pnl_pct: float,
        position_before: float,
        position_after: float,
        volatility: float = 1.0,
    ) -> float:
        """
        Calculate reward for an action.

        Rewards:
        - Positive P&L = positive reward
        - Risk-adjusted returns (higher reward for same return in volatile market)
        - Penalty for excessive trading
        - Bonus for holding winners
        """
        reward = 0.0

        # Base reward from P&L
        reward += pnl_pct * 10  # Scale P&L

        # Risk-adjusted bonus
        if volatility > 0:
            risk_adjusted = pnl_pct / volatility
            reward += risk_adjusted * 5

        # Trading cost penalty
        if position_before != position_after:
            reward -= 0.1  # Small transaction cost

        # Holding bonus for profitable positions
        if position_before != 0 and pnl_pct > 0 and action == Action.HOLD:
            reward += 0.5  # Reward for holding winners

        # Penalty for holding losers too long
        if position_before != 0 and pnl_pct < -2 and action == Action.HOLD:
            reward -= 0.5  # Penalty for not cutting losses

        return reward

    def remember(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        symbol: str = "unknown",
    ):
        """Store experience in replay buffer and database."""
        state_array = state.to_array()
        next_state_array = next_state.to_array()

        self.memory.push(state_array, action, reward, next_state_array, done)

        # Also save to database for persistent learning
        try:
            self.db.save_rl_experience(
                state=state_array.tolist(),
                action=action,
                reward=reward,
                next_state=next_state_array.tolist(),
                done=done,
                symbol=symbol,
            )
        except Exception as e:
            logger.warning(f"Failed to save RL experience: {e}")

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        if TORCH_AVAILABLE:
            return self._train_step_torch(states, actions, rewards, next_states, dones)
        else:
            return self._train_step_numpy(states, actions, rewards, next_states, dones)

    def _train_step_torch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Training step using PyTorch."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def _train_step_numpy(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Simplified training without PyTorch."""
        # Simple gradient descent on linear model
        total_loss = 0.0

        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            # Current Q value
            current_q = np.dot(state, self.weights) + self.bias
            current_q_action = current_q[action]

            # Target Q value
            if done:
                target = reward
            else:
                next_q = np.dot(next_state, self.weights) + self.bias
                target = reward + self.gamma * np.max(next_q)

            # Update weights
            error = target - current_q_action
            total_loss += error ** 2

            # Gradient update
            self.weights[:, action] += self.learning_rate * error * state
            self.bias[action] += self.learning_rate * error

        self.training_steps += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / len(states)

    def load_experiences_from_db(self, symbol: str = None, limit: int = 10000):
        """Load past experiences from database."""
        experiences = self.db.get_rl_experiences(symbol, limit)
        for state, action, reward, next_state, done in experiences:
            self.memory.push(
                np.array(state, dtype=np.float32),
                action,
                reward,
                np.array(next_state, dtype=np.float32),
                done,
            )
        logger.info(f"Loaded {len(experiences)} experiences from database")

    def save_model(self, path: Path):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if TORCH_AVAILABLE:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
            }, path)
        else:
            np.savez(
                path,
                weights=self.weights,
                bias=self.bias,
                epsilon=self.epsilon,
                training_steps=self.training_steps,
            )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model weights."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return

        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
        else:
            data = np.load(path)
            self.weights = data['weights']
            self.bias = data['bias']
            self.epsilon = float(data['epsilon'])
            self.training_steps = int(data['training_steps'])

        logger.info(f"Model loaded from {path}")

    def get_action_probabilities(self, state: State) -> Dict[str, float]:
        """Get probability distribution over actions."""
        state_array = state.to_array()

        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        else:
            q_values = np.dot(state_array, self.weights) + self.bias

        # Softmax to get probabilities
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        return {
            "hold": float(probs[0]),
            "buy": float(probs[1]),
            "sell": float(probs[2]),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "training_steps": self.training_steps,
            "epsilon": round(self.epsilon, 4),
            "memory_size": len(self.memory),
            "torch_available": TORCH_AVAILABLE,
            "device": str(self.device) if TORCH_AVAILABLE else "cpu",
        }


# Global instance
_agent: Optional[RLTradingAgent] = None


def get_rl_agent() -> RLTradingAgent:
    """Get or create global RL agent."""
    global _agent
    if _agent is None:
        _agent = RLTradingAgent()
    return _agent
