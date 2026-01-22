"""
Reinforcement Learning Optimizer for Trading

Uses RL to learn optimal:
1. Position sizing
2. Entry/exit timing
3. Risk management
4. Portfolio allocation
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Current trading state for RL agent."""
    price: float
    position: float  # -1 to 1 (short to long)
    unrealized_pnl: float
    realized_pnl: float
    cash: float
    features: np.ndarray
    time_in_position: int
    max_drawdown: float
    win_rate: float
    volatility: float


@dataclass
class TradingAction:
    """Action taken by RL agent."""
    position_delta: float  # Change in position (-1 to 1)
    stop_loss: float
    take_profit: float
    confidence: float


@dataclass
class Experience:
    """Single experience for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for RL training."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for Trading.

    Uses a simple neural network approximation with numpy
    (no PyTorch dependency for lightweight deployment).
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 5,  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Simple 2-layer network weights
        self.hidden_size = 64
        self.W1 = np.random.randn(state_dim, self.hidden_size) * 0.01
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.b2 = np.zeros(self.hidden_size)
        self.W3 = np.random.randn(self.hidden_size, action_dim) * 0.01
        self.b3 = np.zeros(action_dim)

        # Target network (for stability)
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()

        self.replay_buffer = ReplayBuffer()
        self.update_counter = 0
        self.target_update_freq = 100

        # Action mapping
        self.action_map = {
            0: ('STRONG_BUY', 1.0),
            1: ('BUY', 0.5),
            2: ('HOLD', 0.0),
            3: ('SELL', -0.5),
            4: ('STRONG_SELL', -1.0)
        }

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through network."""
        if use_target:
            W1, b1 = self.target_W1, self.target_b1
            W2, b2 = self.target_W2, self.target_b2
            W3, b3 = self.target_W3, self.target_b3
        else:
            W1, b1 = self.W1, self.b1
            W2, b2 = self.W2, self.b2
            W3, b3 = self.W3, self.b3

        h1 = self._relu(np.dot(state, W1) + b1)
        h2 = self._relu(np.dot(h1, W2) + b2)
        q_values = np.dot(h2, W3) + b3

        return q_values

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self._forward(state)
        return int(np.argmax(q_values))

    def get_action_details(self, action: int) -> Tuple[str, float]:
        """Get action name and position delta."""
        return self.action_map[action]

    def train_step(self, batch_size: int = 32) -> float:
        """Perform single training step."""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Compute target Q-values
        current_q = self._forward(states)
        next_q = self._forward(next_states, use_target=True)

        target_q = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Simple gradient descent update
        loss = self._update_weights(states, target_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()

        return loss

    def _update_weights(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Update network weights using gradient descent."""
        # Forward pass with gradient tracking
        h1 = self._relu(np.dot(states, self.W1) + self.b1)
        h2 = self._relu(np.dot(h1, self.W2) + self.b2)
        output = np.dot(h2, self.W3) + self.b3

        # Compute loss
        loss = np.mean((output - targets) ** 2)

        # Backpropagation (simplified)
        d_output = 2 * (output - targets) / len(states)

        # Layer 3 gradients
        dW3 = np.dot(h2.T, d_output)
        db3 = np.sum(d_output, axis=0)

        # Layer 2 gradients
        d_h2 = np.dot(d_output, self.W3.T) * (h2 > 0)
        dW2 = np.dot(h1.T, d_h2)
        db2 = np.sum(d_h2, axis=0)

        # Layer 1 gradients
        d_h1 = np.dot(d_h2, self.W2.T) * (h1 > 0)
        dW1 = np.dot(states.T, d_h1)
        db1 = np.sum(d_h1, axis=0)

        # Update weights
        self.W3 -= self.learning_rate * np.clip(dW3, -1, 1)
        self.b3 -= self.learning_rate * np.clip(db3, -1, 1)
        self.W2 -= self.learning_rate * np.clip(dW2, -1, 1)
        self.b2 -= self.learning_rate * np.clip(db2, -1, 1)
        self.W1 -= self.learning_rate * np.clip(dW1, -1, 1)
        self.b1 -= self.learning_rate * np.clip(db1, -1, 1)

        return loss

    def _update_target_network(self):
        """Copy weights to target network."""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(Experience(state, action, reward, next_state, done))


class TradingEnvironment:
    """
    Trading environment for RL training.

    Simulates trading with realistic constraints.
    """

    def __init__(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        self.prices = prices.values
        self.features = features.values
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0.0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.entry_price = 0.0
        self.trades = []
        self.max_portfolio_value = self.initial_capital

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Technical features
        tech_features = self.features[self.current_step]

        # Position features
        position_features = np.array([
            self.position,
            self.portfolio_value / self.initial_capital - 1,  # Return
            (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value,  # Drawdown
        ])

        # Price features
        current_price = self.prices[self.current_step]
        if self.current_step > 0:
            price_change = (current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        else:
            price_change = 0

        price_features = np.array([price_change])

        return np.concatenate([tech_features, position_features, price_features])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute trading action.

        Args:
            action: 0=STRONG_BUY, 1=BUY, 2=HOLD, 3=SELL, 4=STRONG_SELL

        Returns:
            next_state, reward, done, info
        """
        action_map = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}
        target_position = action_map[action] * self.max_position

        current_price = self.prices[self.current_step]
        position_change = target_position - self.position

        # Calculate transaction cost
        cost = abs(position_change) * current_price * self.transaction_cost

        # Execute trade
        if position_change != 0:
            self.cash -= position_change * current_price + cost
            self.position = target_position

            if position_change > 0 and self.entry_price == 0:
                self.entry_price = current_price

            self.trades.append({
                'step': self.current_step,
                'action': action,
                'position_change': position_change,
                'price': current_price,
                'cost': cost
            })

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        if not done:
            next_price = self.prices[self.current_step]

            # Calculate portfolio value
            self.portfolio_value = self.cash + self.position * next_price
            self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

            # Calculate reward
            reward = self._calculate_reward(current_price, next_price)
        else:
            # Final reward: total return
            reward = (self.portfolio_value - self.initial_capital) / self.initial_capital

        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'trades': len(self.trades)
        }

        next_state = self._get_state() if not done else np.zeros(self.features.shape[1] + 4)

        return next_state, reward, done, info

    def _calculate_reward(self, current_price: float, next_price: float) -> float:
        """
        Calculate reward with multiple components.

        Rewards:
        - Profit from correct direction
        - Penalty for wrong direction
        - Penalty for excessive trading
        - Bonus for holding winners
        """
        price_change = (next_price - current_price) / current_price

        # Directional reward
        directional_reward = self.position * price_change * 100

        # Transaction penalty (encourage holding)
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step - 1:
            transaction_penalty = -0.01
        else:
            transaction_penalty = 0

        # Drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -drawdown * 0.1

        reward = directional_reward + transaction_penalty + drawdown_penalty

        return reward


class RLTradingOptimizer:
    """
    Main RL optimizer that trains and uses DQN agent.
    """

    def __init__(
        self,
        state_dim: int = 20,
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ):
        self.agent = DQNAgent(
            state_dim=state_dim + 4,  # Features + position info
            learning_rate=learning_rate,
            gamma=gamma
        )
        self.trained = False
        self.training_history = []

    def train(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        episodes: int = 100,
        initial_capital: float = 10000
    ) -> Dict[str, List]:
        """
        Train the RL agent.

        Args:
            prices: Price series
            features: Feature DataFrame
            episodes: Number of training episodes
            initial_capital: Starting capital

        Returns:
            Training history
        """
        logger.info(f"Training RL agent for {episodes} episodes...")

        env = TradingEnvironment(prices, features, initial_capital)
        history = {'episode': [], 'reward': [], 'portfolio_value': [], 'trades': []}

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                self.agent.store_experience(state, action, reward, next_state, done)
                loss = self.agent.train_step()

                state = next_state
                total_reward += reward

            history['episode'].append(episode)
            history['reward'].append(total_reward)
            history['portfolio_value'].append(info['portfolio_value'])
            history['trades'].append(info['trades'])

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(history['reward'][-10:])
                avg_pv = np.mean(history['portfolio_value'][-10:])
                logger.info(f"Episode {episode + 1}: Avg Reward={avg_reward:.2f}, "
                          f"Avg PV=${avg_pv:.2f}, Epsilon={self.agent.epsilon:.3f}")

        self.trained = True
        self.training_history = history

        logger.info(f"Training complete. Final portfolio: ${history['portfolio_value'][-1]:.2f}")

        return history

    def get_action(self, state: np.ndarray) -> Tuple[str, float, float]:
        """
        Get optimal action for current state.

        Returns:
            (action_name, position_delta, confidence)
        """
        action = self.agent.select_action(state, training=False)
        action_name, position_delta = self.agent.get_action_details(action)

        # Calculate confidence from Q-values
        q_values = self.agent._forward(state)
        confidence = np.exp(q_values[action]) / np.sum(np.exp(q_values))

        return action_name, position_delta, confidence

    def evaluate(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent on test data.

        Returns:
            Evaluation metrics
        """
        env = TradingEnvironment(prices, features, initial_capital)
        state = env.reset()
        done = False
        actions_taken = []

        while not done:
            action = self.agent.select_action(state, training=False)
            actions_taken.append(action)
            next_state, _, done, info = env.step(action)
            state = next_state

        final_value = info['portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate metrics
        buy_hold_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

        return {
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'num_trades': info['trades'],
            'action_distribution': dict(pd.Series(actions_taken).value_counts())
        }


# Global singleton
_rl_optimizer: Optional[RLTradingOptimizer] = None


def get_rl_optimizer(state_dim: int = 20) -> RLTradingOptimizer:
    """Get or create RL optimizer."""
    global _rl_optimizer
    if _rl_optimizer is None:
        _rl_optimizer = RLTradingOptimizer(state_dim=state_dim)
    return _rl_optimizer
