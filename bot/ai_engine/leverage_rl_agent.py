"""
Leverage-Aware Reinforcement Learning Agent

Extended RL agent that learns:
1. When to go LONG vs SHORT
2. Optimal leverage for each trade (1x-20x)
3. Position sizing with margin management
4. Liquidation risk awareness

This enables the AI to profit in both bull and bear markets
while optimally using leverage.
"""

from __future__ import annotations

import json
import logging
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

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using simplified leverage RL agent.")


class LeverageAction:
    """Extended action space for leverage trading."""
    HOLD = 0           # Do nothing
    LONG_1X = 1        # Open/hold long at 1x
    LONG_3X = 2        # Open/hold long at 3x
    LONG_5X = 3        # Open/hold long at 5x
    LONG_10X = 4       # Open/hold long at 10x
    SHORT_1X = 5       # Open/hold short at 1x
    SHORT_3X = 6       # Open/hold short at 3x
    SHORT_5X = 7       # Open/hold short at 5x
    SHORT_10X = 8      # Open/hold short at 10x
    CLOSE = 9          # Close current position
    REDUCE_HALF = 10   # Reduce position by 50%

    # Action descriptions
    DESCRIPTIONS = {
        0: "HOLD",
        1: "LONG_1X",
        2: "LONG_3X",
        3: "LONG_5X",
        4: "LONG_10X",
        5: "SHORT_1X",
        6: "SHORT_3X",
        7: "SHORT_5X",
        8: "SHORT_10X",
        9: "CLOSE",
        10: "REDUCE_HALF",
    }

    @staticmethod
    def get_leverage(action: int) -> float:
        """Get leverage multiplier for action."""
        leverage_map = {
            1: 1.0, 5: 1.0,
            2: 3.0, 6: 3.0,
            3: 5.0, 7: 5.0,
            4: 10.0, 8: 10.0,
        }
        return leverage_map.get(action, 1.0)

    @staticmethod
    def is_long(action: int) -> bool:
        """Check if action is a long position."""
        return action in [1, 2, 3, 4]

    @staticmethod
    def is_short(action: int) -> bool:
        """Check if action is a short position."""
        return action in [5, 6, 7, 8]


@dataclass
class LeverageState:
    """Extended state for leverage-aware trading."""
    # Price features
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    price_vs_ema20: float
    price_vs_ema50: float
    price_vs_vwap: float

    # Momentum features
    rsi: float
    rsi_change: float
    macd_hist: float
    macd_signal_cross: float  # +1 bullish cross, -1 bearish cross, 0 none
    momentum_5: float
    momentum_20: float

    # Volatility features
    atr_ratio: float
    bb_position: float
    bb_width: float
    volatility_ratio: float
    high_volatility: float  # 1 if volatility > 2x normal

    # Trend features
    adx: float
    trend_direction: float  # +1 up, -1 down, 0 sideways
    trend_strength: float   # 0-1 how strong is the trend
    ema_alignment: float    # +1 all aligned bullish, -1 all aligned bearish

    # Volume features
    volume_ratio: float
    buy_volume_ratio: float  # Ratio of buy to sell volume

    # Market structure
    funding_rate: float      # For crypto perpetuals
    open_interest_change: float
    long_short_ratio: float  # Longs vs shorts in market

    # Position features
    current_position: float  # -1 to +1 (short to long)
    current_leverage: float  # 1x to 10x
    position_pnl: float
    position_duration: float
    unrealized_pnl: float
    margin_ratio: float      # How close to liquidation (0-1)

    # Risk features
    drawdown_current: float  # Current drawdown from peak
    consecutive_losses: int
    win_rate_recent: float   # Last 10 trades win rate

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network."""
        return np.array([
            # Price (6)
            self.price_change_1h,
            self.price_change_4h,
            self.price_change_24h,
            self.price_vs_ema20,
            self.price_vs_ema50,
            self.price_vs_vwap,
            # Momentum (6)
            self.rsi / 100.0,
            self.rsi_change / 100.0,
            self.macd_hist,
            self.macd_signal_cross,
            self.momentum_5 / 100.0,
            self.momentum_20 / 100.0,
            # Volatility (5)
            self.atr_ratio,
            self.bb_position,
            self.bb_width,
            self.volatility_ratio,
            self.high_volatility,
            # Trend (4)
            self.adx / 100.0,
            self.trend_direction,
            self.trend_strength,
            self.ema_alignment,
            # Volume (2)
            self.volume_ratio,
            self.buy_volume_ratio,
            # Market structure (3)
            self.funding_rate * 100,  # Amplify small values
            self.open_interest_change,
            self.long_short_ratio,
            # Position (6)
            self.current_position,
            self.current_leverage / 10.0,  # Normalize to 0-1
            self.position_pnl / 100.0,
            min(self.position_duration / 100.0, 1.0),
            self.unrealized_pnl / 100.0,
            self.margin_ratio,
            # Risk (3)
            self.drawdown_current / 100.0,
            min(self.consecutive_losses / 5.0, 1.0),
            self.win_rate_recent,
        ], dtype=np.float32)

    @classmethod
    def from_market_data(
        cls,
        indicators: Dict[str, float],
        position_info: Dict[str, float] = None,
        risk_info: Dict[str, float] = None,
    ) -> 'LeverageState':
        """Create state from indicator dicts."""
        pos = position_info or {}
        risk = risk_info or {}

        return cls(
            # Price
            price_change_1h=indicators.get('price_change_1h', 0),
            price_change_4h=indicators.get('price_change_4h', 0),
            price_change_24h=indicators.get('price_change_24h', 0),
            price_vs_ema20=indicators.get('price_vs_ema20', 0),
            price_vs_ema50=indicators.get('price_vs_ema50', 0),
            price_vs_vwap=indicators.get('price_vs_vwap', 0),
            # Momentum
            rsi=indicators.get('rsi', 50),
            rsi_change=indicators.get('rsi_change', 0),
            macd_hist=indicators.get('macd_hist', 0),
            macd_signal_cross=indicators.get('macd_signal_cross', 0),
            momentum_5=indicators.get('momentum_5', 0),
            momentum_20=indicators.get('momentum_20', 0),
            # Volatility
            atr_ratio=indicators.get('atr_ratio', 1),
            bb_position=indicators.get('bb_position', 0.5),
            bb_width=indicators.get('bb_width', 0.02),
            volatility_ratio=indicators.get('volatility_ratio', 1),
            high_volatility=1.0 if indicators.get('volatility_ratio', 1) > 2 else 0.0,
            # Trend
            adx=indicators.get('adx', 25),
            trend_direction=indicators.get('trend_direction', 0),
            trend_strength=indicators.get('trend_strength', 0.5),
            ema_alignment=indicators.get('ema_alignment', 0),
            # Volume
            volume_ratio=indicators.get('volume_ratio', 1),
            buy_volume_ratio=indicators.get('buy_volume_ratio', 0.5),
            # Market structure
            funding_rate=indicators.get('funding_rate', 0),
            open_interest_change=indicators.get('open_interest_change', 0),
            long_short_ratio=indicators.get('long_short_ratio', 1),
            # Position
            current_position=pos.get('position', 0),
            current_leverage=pos.get('leverage', 1),
            position_pnl=pos.get('pnl', 0),
            position_duration=pos.get('duration', 0),
            unrealized_pnl=pos.get('unrealized_pnl', 0),
            margin_ratio=pos.get('margin_ratio', 0),
            # Risk
            drawdown_current=risk.get('drawdown', 0),
            consecutive_losses=risk.get('consecutive_losses', 0),
            win_rate_recent=risk.get('win_rate_recent', 0.5),
        )


if TORCH_AVAILABLE:
    class LeverageDQNetwork(nn.Module):
        """Deep Q-Network for leverage trading decisions."""

        def __init__(
            self,
            state_size: int = 35,
            action_size: int = 11,
            hidden_sizes: List[int] = [256, 128, 64, 32],
        ):
            super().__init__()

            # Dueling DQN architecture for better value estimation
            self.feature_layers = nn.Sequential(
                nn.Linear(state_size, hidden_sizes[0]),
                nn.LayerNorm(hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.LayerNorm(hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], 1),
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], action_size),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            features = self.feature_layers(state)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine using dueling architecture
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
            return q_values


class LeverageReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 200000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        leverage_info: Dict[str, float] = None,
    ):
        """Add experience with priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        experience = (state, action, reward, next_state, done, leverage_info or {})

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample with prioritization."""
        if self.size < batch_size:
            batch_size = self.size

        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, leverage_infos = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            leverage_infos,
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after learning."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant for stability

    def __len__(self):
        return self.size


class LeverageRLAgent:
    """
    Leverage-aware Deep Q-Learning agent.

    Features:
    1. Extended action space with leverage options
    2. Short selling capability
    3. Risk-aware reward function
    4. Liquidation prevention
    5. Prioritized experience replay
    6. Dueling DQN architecture
    """

    def __init__(
        self,
        state_size: int = 35,
        action_size: int = 11,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 128,
        tau: float = 0.005,  # Soft update parameter
        db: LearningDatabase = None,
        max_leverage: float = 10.0,
        liquidation_threshold: float = 0.9,  # Warn at 90% margin used
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.db = db or get_learning_db()
        self.max_leverage = max_leverage
        self.liquidation_threshold = liquidation_threshold

        self.memory = LeverageReplayBuffer()
        self.training_steps = 0
        self.total_reward = 0

        # Performance tracking
        self.leverage_usage_stats = {i: 0 for i in range(action_size)}
        self.reward_by_leverage = {1: [], 3: [], 5: [], 10: []}
        self.short_vs_long_pnl = {'long': [], 'short': []}

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = LeverageDQNetwork(state_size, action_size).to(self.device)
            self.target_net = LeverageDQNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
            )
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10000, gamma=0.9
            )
        else:
            # Simplified model without PyTorch
            self.weights = np.random.randn(state_size, action_size) * 0.01
            self.bias = np.zeros(action_size)
            self.learning_rate = learning_rate

    def select_action(
        self,
        state: LeverageState,
        training: bool = True,
        force_safe: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action with leverage consideration.

        Args:
            state: Current market state
            training: If True, use exploration
            force_safe: If True, avoid high leverage actions

        Returns:
            Tuple of (action, info_dict)
        """
        state_array = state.to_array()

        # Safety check: if close to liquidation, force close
        if state.margin_ratio > self.liquidation_threshold:
            logger.warning(f"Margin ratio {state.margin_ratio:.1%} > threshold, forcing close")
            return LeverageAction.CLOSE, {"reason": "liquidation_risk"}

        # Safety check: avoid high leverage in high volatility
        if force_safe or (state.high_volatility > 0 and state.margin_ratio > 0.5):
            valid_actions = [0, 1, 5, 9, 10]  # Only low leverage or close
        else:
            valid_actions = list(range(self.action_size))

        # Exploration
        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
            return action, {"exploration": True}

        # Exploitation
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        else:
            q_values = np.dot(state_array, self.weights) + self.bias

        # Mask invalid actions with very negative values
        masked_q = q_values.copy()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q[i] = -1e9

        action = int(np.argmax(masked_q))
        confidence = float(np.exp(q_values[action]) / np.exp(q_values).sum())

        return action, {
            "q_values": q_values.tolist(),
            "confidence": confidence,
            "exploration": False,
        }

    def calculate_reward(
        self,
        action: int,
        pnl_pct: float,
        position_before: float,
        position_after: float,
        leverage_used: float,
        volatility: float = 1.0,
        is_short: bool = False,
        margin_ratio: float = 0.0,
        liquidation_distance: float = 100.0,
        trade_duration: int = 0,
    ) -> float:
        """
        Calculate reward considering leverage and shorting.

        Rewards:
        - P&L scaled by leverage risk
        - Bonus for profitable shorts in bear market
        - Penalty for high leverage in volatile markets
        - Penalty for approaching liquidation
        - Bonus for appropriate leverage selection
        """
        reward = 0.0

        # Base P&L reward (normalized by leverage risk)
        # Higher leverage = lower reward per PnL (risk adjusted)
        leverage_risk_factor = 1.0 / math.sqrt(leverage_used)
        reward += pnl_pct * 10 * leverage_risk_factor

        # Risk-adjusted return (Sharpe-like)
        if volatility > 0:
            risk_adjusted = pnl_pct / (volatility * leverage_used)
            reward += risk_adjusted * 5

        # Appropriate leverage bonus
        # Low volatility + high leverage = OK
        # High volatility + low leverage = OK
        volatility_leverage_match = 1.0 - abs(volatility - 1.0 / leverage_used)
        if volatility_leverage_match > 0.5:
            reward += 0.5  # Bonus for appropriate leverage

        # Short selling bonuses/penalties
        if is_short:
            if pnl_pct > 0:
                reward += 1.0  # Bonus for profitable short
                self.short_vs_long_pnl['short'].append(pnl_pct)
            elif pnl_pct < -5:
                reward -= 2.0  # Big penalty for shorts that blow up
        else:
            if pnl_pct > 0:
                self.short_vs_long_pnl['long'].append(pnl_pct)

        # Liquidation risk penalty
        if margin_ratio > 0.7:
            reward -= (margin_ratio - 0.7) * 10  # Progressive penalty
        if liquidation_distance < 5:  # Within 5% of liquidation
            reward -= (5 - liquidation_distance) * 2

        # Position duration considerations
        if trade_duration > 100 and pnl_pct < 0:
            reward -= 0.5  # Penalty for holding losers too long
        if trade_duration < 5 and pnl_pct < 0:
            reward -= 0.3  # Penalty for quick losses (over-trading)

        # Transaction cost
        if position_before != position_after:
            reward -= 0.1 * leverage_used  # Higher leverage = higher cost

        # Track leverage usage
        self.leverage_usage_stats[action] += 1
        lev = LeverageAction.get_leverage(action)
        if lev in self.reward_by_leverage:
            self.reward_by_leverage[lev].append(reward)

        return reward

    def remember(
        self,
        state: LeverageState,
        action: int,
        reward: float,
        next_state: LeverageState,
        done: bool,
        leverage_info: Dict[str, float] = None,
        symbol: str = "unknown",
    ):
        """Store experience with leverage info."""
        state_array = state.to_array()
        next_state_array = next_state.to_array()

        self.memory.push(state_array, action, reward, next_state_array, done, leverage_info)
        self.total_reward += reward

        # Save to database
        try:
            self.db.save_rl_experience(
                state=state_array.tolist(),
                action=action,
                reward=reward,
                next_state=next_state_array.tolist(),
                done=done,
                symbol=symbol,
                metadata=leverage_info,
            )
        except Exception as e:
            logger.warning(f"Failed to save leverage RL experience: {e}")

    def train_step(self, beta: float = 0.4) -> Optional[float]:
        """
        Train with prioritized experience replay.

        Args:
            beta: Importance sampling correction

        Returns:
            Loss value if training occurred
        """
        if len(self.memory) < self.batch_size:
            return None

        result = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones, _, indices, weights = result

        if TORCH_AVAILABLE:
            loss = self._train_step_torch(
                states, actions, rewards, next_states, dones, indices, weights
            )
        else:
            loss = self._train_step_numpy(
                states, actions, rewards, next_states, dones
            )

        self.training_steps += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def _train_step_torch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """PyTorch training with prioritized replay."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Double DQN: use policy net to select action, target net to evaluate
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Select best action using policy network
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Weighted loss
        td_errors = (current_q.squeeze() - target_q).abs().detach().cpu().numpy()
        loss = (weights * nn.MSELoss(reduction='none')(current_q.squeeze(), target_q)).mean()

        # Update priorities
        self.memory.update_priorities(indices, td_errors)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Soft update target network
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

        return loss.item()

    def _train_step_numpy(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Simplified numpy training."""
        total_loss = 0.0

        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            current_q = np.dot(state, self.weights) + self.bias
            current_q_action = current_q[action]

            if done:
                target = reward
            else:
                next_q = np.dot(next_state, self.weights) + self.bias
                target = reward + self.gamma * np.max(next_q)

            error = target - current_q_action
            total_loss += error ** 2

            self.weights[:, action] += self.learning_rate * error * state
            self.bias[action] += self.learning_rate * error

        return total_loss / len(states)

    def get_action_analysis(self, state: LeverageState) -> Dict[str, Any]:
        """Get detailed analysis of action selection."""
        state_array = state.to_array()

        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        else:
            q_values = np.dot(state_array, self.weights) + self.bias

        # Softmax probabilities
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()

        best_action = int(np.argmax(q_values))
        best_leverage = LeverageAction.get_leverage(best_action)

        return {
            "best_action": best_action,
            "action_name": LeverageAction.DESCRIPTIONS.get(best_action, "UNKNOWN"),
            "best_leverage": best_leverage,
            "is_short": LeverageAction.is_short(best_action),
            "is_long": LeverageAction.is_long(best_action),
            "q_values": {
                LeverageAction.DESCRIPTIONS[i]: round(q_values[i], 4)
                for i in range(len(q_values))
            },
            "probabilities": {
                LeverageAction.DESCRIPTIONS[i]: round(probs[i], 4)
                for i in range(len(probs))
            },
            "confidence": float(probs[best_action]),
            "market_conditions": {
                "volatility": state.volatility_ratio,
                "trend_direction": state.trend_direction,
                "margin_ratio": state.margin_ratio,
            },
            "recommendation": self._get_recommendation(state, best_action, probs),
        }

    def _get_recommendation(
        self,
        state: LeverageState,
        best_action: int,
        probs: np.ndarray,
    ) -> str:
        """Generate human-readable recommendation."""
        confidence = probs[best_action]
        leverage = LeverageAction.get_leverage(best_action)

        if best_action == LeverageAction.HOLD:
            return "Market conditions uncertain. Hold current position."

        if best_action == LeverageAction.CLOSE:
            return "Risk/reward unfavorable. Close position to protect capital."

        if best_action == LeverageAction.REDUCE_HALF:
            return "Elevated risk detected. Reduce position size by 50%."

        direction = "SHORT" if LeverageAction.is_short(best_action) else "LONG"
        conf_level = "HIGH" if confidence > 0.6 else "MODERATE" if confidence > 0.4 else "LOW"

        risk_warning = ""
        if leverage >= 5 and state.volatility_ratio > 1.5:
            risk_warning = " WARNING: High leverage in volatile market."
        if state.margin_ratio > 0.5:
            risk_warning += " CAUTION: Margin usage elevated."

        return f"{direction} at {leverage}x leverage ({conf_level} confidence).{risk_warning}"

    def get_optimal_leverage(
        self,
        state: LeverageState,
        direction: str = "long",
    ) -> Tuple[float, float]:
        """
        Get optimal leverage for a given direction.

        Returns:
            Tuple of (optimal_leverage, confidence)
        """
        analysis = self.get_action_analysis(state)

        if direction == "long":
            relevant_actions = [1, 2, 3, 4]  # LONG_1X to LONG_10X
        else:
            relevant_actions = [5, 6, 7, 8]  # SHORT_1X to SHORT_10X

        best_action = None
        best_q = float('-inf')

        for action in relevant_actions:
            q = analysis['q_values'][LeverageAction.DESCRIPTIONS[action]]
            if q > best_q:
                best_q = q
                best_action = action

        if best_action is None:
            return 1.0, 0.0

        optimal_leverage = LeverageAction.get_leverage(best_action)
        confidence = analysis['probabilities'][LeverageAction.DESCRIPTIONS[best_action]]

        return optimal_leverage, confidence

    def get_leverage_performance_stats(self) -> Dict[str, Any]:
        """Get statistics about leverage usage and performance."""
        stats = {
            "action_usage": {
                LeverageAction.DESCRIPTIONS[k]: v
                for k, v in self.leverage_usage_stats.items()
            },
            "reward_by_leverage": {},
            "short_vs_long": {},
        }

        for lev, rewards in self.reward_by_leverage.items():
            if rewards:
                stats["reward_by_leverage"][f"{lev}x"] = {
                    "count": len(rewards),
                    "mean_reward": round(np.mean(rewards), 4),
                    "std_reward": round(np.std(rewards), 4),
                }

        for direction, pnls in self.short_vs_long_pnl.items():
            if pnls:
                stats["short_vs_long"][direction] = {
                    "count": len(pnls),
                    "mean_pnl": round(np.mean(pnls), 4),
                    "total_pnl": round(sum(pnls), 4),
                }

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "training_steps": self.training_steps,
            "epsilon": round(self.epsilon, 4),
            "memory_size": len(self.memory),
            "total_reward": round(self.total_reward, 2),
            "torch_available": TORCH_AVAILABLE,
            "device": str(self.device) if TORCH_AVAILABLE else "cpu",
            "leverage_stats": self.get_leverage_performance_stats(),
        }

    def save_model(self, path: Path):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if TORCH_AVAILABLE:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps,
                'total_reward': self.total_reward,
                'leverage_usage_stats': self.leverage_usage_stats,
            }, path)
        else:
            np.savez(
                path,
                weights=self.weights,
                bias=self.bias,
                epsilon=self.epsilon,
                training_steps=self.training_steps,
                total_reward=self.total_reward,
            )

        logger.info(f"Leverage RL model saved to {path}")

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
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            self.total_reward = checkpoint.get('total_reward', 0)
            self.leverage_usage_stats = checkpoint.get(
                'leverage_usage_stats',
                {i: 0 for i in range(self.action_size)}
            )
        else:
            data = np.load(path, allow_pickle=True)
            self.weights = data['weights']
            self.bias = data['bias']
            self.epsilon = float(data['epsilon'])
            self.training_steps = int(data['training_steps'])

        logger.info(f"Leverage RL model loaded from {path}")


# Global instance
_leverage_agent: Optional[LeverageRLAgent] = None


def get_leverage_rl_agent() -> LeverageRLAgent:
    """Get or create global leverage-aware RL agent."""
    global _leverage_agent
    if _leverage_agent is None:
        _leverage_agent = LeverageRLAgent()
    return _leverage_agent
