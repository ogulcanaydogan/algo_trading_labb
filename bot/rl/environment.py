"""
Trading Environment for Reinforcement Learning.

Gym-compatible environment for training RL trading agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""

    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # Max position as fraction of balance
    trading_fee: float = 0.001  # 0.1% fee
    slippage: float = 0.0005  # 0.05% slippage
    lookback_window: int = 60  # Number of past bars in state
    episode_length: int = 1000  # Bars per episode
    reward_scaling: float = 100.0  # Scale rewards for training
    use_risk_adjusted_reward: bool = True
    max_drawdown_penalty: float = 0.5
    inactivity_penalty: float = 0.001  # Penalty for not trading


class TradingEnvironment:
    """
    Gym-compatible Trading Environment.

    State space:
    - Market features (OHLCV-derived indicators)
    - Portfolio state (cash, position, PnL)
    - Time features (position duration, etc.)

    Action space:
    - 0: SHORT (sell/short)
    - 1: FLAT (close position / do nothing)
    - 2: LONG (buy/long)

    Rewards:
    - Sparse: PnL at episode end
    - Shaped: Immediate rewards based on position PnL

    Usage:
        env = TradingEnvironment(data=ohlcv_df)
        state = env.reset()

        done = False
        while not done:
            action = agent.predict(state)
            state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        config: Optional[TradingEnvConfig] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            data: OHLCV DataFrame with features
            config: Environment configuration
        """
        self.config = config or TradingEnvConfig()
        self._data = data
        self._features: Optional[np.ndarray] = None

        if data is not None:
            self._prepare_data(data)

        # State
        self._current_step = 0
        self._balance = self.config.initial_balance
        self._position = 0.0  # -1 to 1 (short to long)
        self._position_entry_price = 0.0
        self._position_entry_step = 0
        self._unrealized_pnl = 0.0
        self._realized_pnl = 0.0
        self._max_balance = self.config.initial_balance
        self._trade_count = 0

        # History for analysis
        self._balance_history: List[float] = []
        self._action_history: List[int] = []
        self._reward_history: List[float] = []

        # Spaces (gym-compatible)
        self.action_space = DiscreteSpace(3)  # SHORT, FLAT, LONG
        self.observation_space = BoxSpace(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_state_dim(),),
        )

    def _prepare_data(self, data: pd.DataFrame) -> None:
        """Prepare data for the environment."""
        self._data = data.copy()

        # Extract features
        feature_cols = [
            col for col in data.columns if col not in ["open", "high", "low", "close", "volume"]
        ]

        if feature_cols:
            self._features = data[feature_cols].values
        else:
            # Use basic OHLCV as features
            self._features = data[["open", "high", "low", "close", "volume"]].values

        # Store prices
        self._prices = data["close"].values
        self._highs = data["high"].values
        self._lows = data["low"].values

    def set_data(self, data: pd.DataFrame) -> None:
        """Set market data for the environment."""
        self._prepare_data(data)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.

        Returns:
            Initial state observation
        """
        # Reset to random starting point (with enough lookback)
        if self._features is not None:
            max_start = len(self._features) - self.config.episode_length - 1
            if max_start > self.config.lookback_window:
                self._current_step = np.random.randint(self.config.lookback_window, max_start)
            else:
                self._current_step = self.config.lookback_window
        else:
            self._current_step = self.config.lookback_window

        # Reset portfolio
        self._balance = self.config.initial_balance
        self._position = 0.0
        self._position_entry_price = 0.0
        self._position_entry_step = 0
        self._unrealized_pnl = 0.0
        self._realized_pnl = 0.0
        self._max_balance = self.config.initial_balance
        self._trade_count = 0

        # Clear history
        self._balance_history = [self._balance]
        self._action_history = []
        self._reward_history = []

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: 0=SHORT, 1=FLAT, 2=LONG

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Store previous state for reward calculation
        prev_balance = self._balance + self._unrealized_pnl

        # Execute action
        self._execute_action(action)

        # Move to next step
        self._current_step += 1

        # Update unrealized PnL
        self._update_unrealized_pnl()

        # Calculate reward
        current_balance = self._balance + self._unrealized_pnl
        reward = self._calculate_reward(prev_balance, current_balance, action)

        # Update max balance for drawdown
        self._max_balance = max(self._max_balance, current_balance)

        # Record history
        self._balance_history.append(current_balance)
        self._action_history.append(action)
        self._reward_history.append(reward)

        # Check if done
        done = self._is_done()

        # Get new state
        state = self._get_state()

        # Info dict
        info = {
            "balance": self._balance,
            "position": self._position,
            "unrealized_pnl": self._unrealized_pnl,
            "realized_pnl": self._realized_pnl,
            "total_pnl": self._balance + self._unrealized_pnl - self.config.initial_balance,
            "trade_count": self._trade_count,
            "step": self._current_step,
            "current_price": self._get_current_price(),
        }

        return state, reward, done, info

    def _execute_action(self, action: int) -> None:
        """Execute trading action."""
        target_position = {0: -1.0, 1: 0.0, 2: 1.0}[action]

        if target_position != self._position:
            # Position change
            self._close_position()

            if target_position != 0:
                self._open_position(target_position)

    def _close_position(self) -> None:
        """Close current position."""
        if self._position == 0:
            return

        current_price = self._get_current_price()

        # Calculate PnL
        if self._position > 0:  # Long
            pnl = (
                (current_price - self._position_entry_price)
                * abs(self._position)
                * self._balance
                / self._position_entry_price
            )
        else:  # Short
            pnl = (
                (self._position_entry_price - current_price)
                * abs(self._position)
                * self._balance
                / self._position_entry_price
            )

        # Apply fees and slippage
        fees = (
            abs(self._position) * self._balance * (self.config.trading_fee + self.config.slippage)
        )
        pnl -= fees

        # Update balance
        self._balance += pnl
        self._realized_pnl += pnl
        self._unrealized_pnl = 0
        self._trade_count += 1

        # Reset position
        self._position = 0
        self._position_entry_price = 0
        self._position_entry_step = 0

    def _open_position(self, position: float) -> None:
        """Open new position."""
        current_price = self._get_current_price()

        # Apply entry fees
        fees = abs(position) * self._balance * self.config.trading_fee
        self._balance -= fees

        # Set position
        self._position = position * self.config.max_position_size
        self._position_entry_price = current_price * (1 + self.config.slippage * np.sign(position))
        self._position_entry_step = self._current_step

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL based on current price."""
        if self._position == 0:
            self._unrealized_pnl = 0
            return

        current_price = self._get_current_price()
        position_value = abs(self._position) * self._balance

        if self._position > 0:  # Long
            price_change = (current_price - self._position_entry_price) / self._position_entry_price
        else:  # Short
            price_change = (self._position_entry_price - current_price) / self._position_entry_price

        self._unrealized_pnl = position_value * price_change

    def _calculate_reward(
        self,
        prev_balance: float,
        current_balance: float,
        action: int,
    ) -> float:
        """Calculate reward for the step."""
        # Base reward: portfolio return
        reward = (current_balance - prev_balance) / self.config.initial_balance

        if self.config.use_risk_adjusted_reward:
            # Drawdown penalty
            drawdown = (self._max_balance - current_balance) / self._max_balance
            if drawdown > 0.05:  # Penalize drawdowns > 5%
                reward -= drawdown * self.config.max_drawdown_penalty

            # Inactivity penalty (encourage trading when appropriate)
            if action == 1 and self._position == 0:
                reward -= self.config.inactivity_penalty

        return reward * self.config.reward_scaling

    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        if self._features is None:
            return np.zeros(self._get_state_dim())

        # Market features (lookback window)
        start_idx = max(0, self._current_step - self.config.lookback_window)
        end_idx = self._current_step

        market_features = self._features[start_idx:end_idx]

        # Pad if needed
        if len(market_features) < self.config.lookback_window:
            padding = np.zeros(
                (self.config.lookback_window - len(market_features), market_features.shape[1])
            )
            market_features = np.vstack([padding, market_features])

        # Flatten market features
        market_flat = market_features.flatten()

        # Portfolio state
        current_price = self._get_current_price()
        portfolio_features = np.array(
            [
                self._balance / self.config.initial_balance,  # Normalized balance
                self._position,  # Current position
                self._unrealized_pnl / self.config.initial_balance,  # Normalized unrealized PnL
                self._realized_pnl / self.config.initial_balance,  # Normalized realized PnL
                (self._current_step - self._position_entry_step) / 100
                if self._position != 0
                else 0,  # Position duration
                (current_price - self._position_entry_price) / self._position_entry_price
                if self._position_entry_price > 0
                else 0,  # Position return
            ]
        )

        # Combine
        state = np.concatenate([market_flat, portfolio_features])

        # Handle NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        return state.astype(np.float32)

    def _get_state_dim(self) -> int:
        """Get state dimension."""
        if self._features is not None:
            market_dim = self.config.lookback_window * self._features.shape[1]
        else:
            market_dim = self.config.lookback_window * 5  # OHLCV

        portfolio_dim = 6  # Balance, position, unrealized, realized, duration, return

        return market_dim + portfolio_dim

    def _get_current_price(self) -> float:
        """Get current close price."""
        if self._prices is not None and self._current_step < len(self._prices):
            return float(self._prices[self._current_step])
        return 1.0

    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Episode length reached
        if self._features is not None:
            if self._current_step >= len(self._features) - 1:
                return True
            if len(self._action_history) >= self.config.episode_length:
                return True

        # Bankrupt
        if self._balance + self._unrealized_pnl <= 0:
            return True

        return False

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        final_balance = self._balance + self._unrealized_pnl
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance

        # Calculate metrics
        returns = np.diff(self._balance_history) / np.array(self._balance_history[:-1])
        returns = returns[~np.isnan(returns)]

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized

        max_drawdown = 0.0
        peak = self.config.initial_balance
        for balance in self._balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "trade_count": self._trade_count,
            "total_reward": sum(self._reward_history),
            "final_balance": final_balance,
        }

    def render(self, mode: str = "human") -> None:
        """Render environment state (for debugging)."""
        print(f"Step: {self._current_step}")
        print(f"Balance: ${self._balance:.2f}")
        print(f"Position: {self._position:.2f}")
        print(f"Unrealized PnL: ${self._unrealized_pnl:.2f}")
        print(
            f"Total PnL: ${self._balance + self._unrealized_pnl - self.config.initial_balance:.2f}"
        )


# Simple space classes (gym-compatible interface without full gym dependency)
class DiscreteSpace:
    """Discrete action space."""

    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(0, self.n)

    def contains(self, x: int) -> bool:
        return 0 <= x < self.n


class BoxSpace:
    """Continuous observation space."""

    def __init__(
        self,
        low: float,
        high: float,
        shape: Tuple[int, ...],
    ):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

    def contains(self, x: np.ndarray) -> bool:
        return x.shape == self.shape
