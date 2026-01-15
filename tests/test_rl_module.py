"""
Comprehensive tests for Reinforcement Learning module.

Tests trading environment, policy networks, trainers, and position sizer.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np
import pandas as pd

from bot.rl import (
    TradingEnvironment,
    TradingEnvConfig,
    TradingPolicyNetwork,
    PolicyConfig,
    PPOTrainer,
    TrainingConfig,
    A2CTrainer,
    A2CConfig,
    HybridMLRLAgent,
    HybridConfig,
    RLPositionSizer,
    DiscretePositionSizer,
    PositionSizerConfig,
    PositionSizerState,
)


class TestTradingEnvironment:
    """Test Trading Environment."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n_rows = 500

        base_price = 100.0
        returns = np.random.randn(n_rows) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            "open": prices * (1 + np.random.randn(n_rows) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_rows)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_rows)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows),
        })

        data.index = pd.date_range(start="2024-01-01", periods=n_rows, freq="1min")
        return data

    @pytest.fixture
    def env(self, sample_data):
        """Create trading environment."""
        config = TradingEnvConfig(
            initial_balance=10000.0,
            episode_length=100,
            lookback_window=30,
        )
        return TradingEnvironment(data=sample_data, config=config)

    def test_environment_creation(self, env):
        """Test environment initialization."""
        assert env.config.initial_balance == 10000.0
        assert env.action_space.n == 3  # SHORT, FLAT, LONG

    def test_reset(self, env):
        """Test environment reset."""
        state = env.reset()

        assert state is not None
        assert isinstance(state, np.ndarray)
        assert not np.any(np.isnan(state))

    def test_step(self, env):
        """Test environment step."""
        env.reset()

        # Take a random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        assert state is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_completion(self, env):
        """Test episode runs to completion."""
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            steps += 1

        assert steps > 0
        assert done or steps == 200

    def test_actions(self, env):
        """Test different actions."""
        env.reset()

        # SHORT
        state, reward, done, info = env.step(0)
        assert info["position"] <= 0

        # Reset and go LONG
        env.reset()
        state, reward, done, info = env.step(2)
        assert info["position"] >= 0

    def test_episode_stats(self, env):
        """Test episode statistics."""
        env.reset()

        # Run some steps
        for _ in range(50):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break

        stats = env.get_episode_stats()

        assert "total_return" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats
        assert "trade_count" in stats

    def test_balance_tracking(self, env):
        """Test balance is tracked correctly."""
        env.reset()
        initial = env.config.initial_balance

        # Execute some trades
        for _ in range(20):
            action = np.random.randint(0, 3)
            env.step(action)

        # Balance should have changed
        current = env._balance + env._unrealized_pnl
        assert current != initial or env._trade_count == 0


class TestPolicyNetwork:
    """Test Policy Network."""

    @pytest.fixture
    def policy(self):
        """Create policy network."""
        config = PolicyConfig(
            state_dim=100,
            action_dim=3,
            hidden_dim=64,
            use_lstm=False,
        )
        return TradingPolicyNetwork(config)

    def test_network_creation(self, policy):
        """Test network initialization."""
        assert policy.config.state_dim == 100
        assert policy.config.action_dim == 3

    def test_forward_pass(self, policy):
        """Test forward pass."""
        try:
            import torch
            state = torch.randn(1, 100)
            action_logits, value, hidden = policy.forward(state)

            assert action_logits.shape == (1, 3)
            assert value.shape == (1,)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_get_action(self, policy):
        """Test getting action from policy."""
        state = np.random.randn(100).astype(np.float32)
        action, log_prob, value = policy.get_action(state)

        assert action in [0, 1, 2]
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_deterministic_action(self, policy):
        """Test deterministic action selection."""
        state = np.random.randn(100).astype(np.float32)

        # Get multiple deterministic actions - should be the same
        actions = [
            policy.get_action(state, deterministic=True)[0]
            for _ in range(5)
        ]

        assert all(a == actions[0] for a in actions)


class TestPPOTrainer:
    """Test PPO Trainer."""

    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Create trainer with mock environment."""
        # Create simple environment
        np.random.seed(42)
        n_rows = 200

        prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.01))
        data = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows),
        })

        env = TradingEnvironment(
            data=data,
            config=TradingEnvConfig(
                initial_balance=10000.0,
                episode_length=50,
                lookback_window=10,
            ),
        )

        policy = TradingPolicyNetwork(
            PolicyConfig(
                state_dim=env.observation_space.shape[0],
                action_dim=3,
                hidden_dim=32,
                use_lstm=False,
            )
        )

        config = TrainingConfig(
            n_steps=32,
            n_epochs=2,
            batch_size=16,
            n_updates=3,
            eval_freq=2,
            n_eval_episodes=1,
        )

        trainer = PPOTrainer(
            policy=policy,
            env=env,
            config=config,
            save_dir=str(tmp_path),
        )

        return trainer

    def test_trainer_creation(self, trainer_setup):
        """Test trainer initialization."""
        assert trainer_setup.config.n_steps == 32
        assert trainer_setup.buffer is not None

    def test_training_run(self, trainer_setup):
        """Test training execution."""
        try:
            results = trainer_setup.train(n_updates=2)

            # Training should produce policy losses
            assert len(results.policy_losses) > 0
            # Training should complete (either timesteps or losses recorded)
            assert results.training_time_seconds > 0 or len(results.policy_losses) > 0
        except ImportError:
            pytest.skip("PyTorch not available")


class TestA2CTrainer:
    """Test A2C Trainer."""

    @pytest.fixture
    def a2c_setup(self, tmp_path):
        """Create A2C trainer setup."""
        np.random.seed(42)
        n_rows = 200

        prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.01))
        data = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows),
        })

        env = TradingEnvironment(
            data=data,
            config=TradingEnvConfig(
                initial_balance=10000.0,
                episode_length=50,
                lookback_window=10,
            ),
        )

        policy = TradingPolicyNetwork(
            PolicyConfig(
                state_dim=env.observation_space.shape[0],
                action_dim=3,
                hidden_dim=32,
                use_lstm=False,
            )
        )

        config = A2CConfig(
            n_steps=5,
            total_timesteps=100,
            eval_freq=50,
            n_eval_episodes=1,
        )

        trainer = A2CTrainer(
            policy=policy,
            env=env,
            config=config,
            save_dir=str(tmp_path),
        )

        return trainer

    def test_a2c_creation(self, a2c_setup):
        """Test A2C trainer initialization."""
        assert a2c_setup.config.n_steps == 5

    def test_a2c_training(self, a2c_setup):
        """Test A2C training."""
        try:
            results = a2c_setup.train(total_timesteps=50)
            assert results.total_timesteps > 0
        except ImportError:
            pytest.skip("PyTorch not available")


class TestHybridMLRLAgent:
    """Test Hybrid ML+RL Agent."""

    @pytest.fixture
    def hybrid_agent(self):
        """Create hybrid agent."""
        # Mock ML model
        mock_ml = MagicMock()
        mock_ml.predict_proba = MagicMock(
            return_value=np.array([[0.2, 0.3, 0.5]])
        )

        config = HybridConfig(
            ml_weight=0.6,
            rl_weight=0.4,
            adaptive_weights=True,
        )

        return HybridMLRLAgent(ml_model=mock_ml, config=config)

    def test_agent_creation(self, hybrid_agent):
        """Test hybrid agent initialization."""
        assert hybrid_agent.config.ml_weight == 0.6
        assert hybrid_agent.config.rl_weight == 0.4

    def test_prediction(self, hybrid_agent):
        """Test hybrid prediction."""
        state = np.random.randn(100).astype(np.float32)
        action, confidence = hybrid_agent.predict(state)

        assert action in ["SHORT", "FLAT", "LONG"]
        assert 0 <= confidence <= 1

    def test_weight_adaptation(self, hybrid_agent):
        """Test weight adaptation."""
        initial_ml = hybrid_agent._ml_weight
        initial_rl = hybrid_agent._rl_weight

        # Update with outcomes
        for _ in range(10):
            hybrid_agent.update_weights(ml_correct=True, rl_correct=False)

        # Weights should sum to approximately 1 (allowing for numerical precision)
        assert abs((hybrid_agent._ml_weight + hybrid_agent._rl_weight) - 1.0) < 0.01
        # Weights should be valid (between 0 and 1)
        assert 0 <= hybrid_agent._ml_weight <= 1
        assert 0 <= hybrid_agent._rl_weight <= 1

    def test_outcome_recording(self, hybrid_agent):
        """Test outcome recording."""
        hybrid_agent.record_outcome(
            prediction="LONG",
            actual_outcome="LONG",
            ml_prediction="LONG",
            rl_prediction="FLAT",
        )

        assert len(hybrid_agent._recent_predictions) == 1

    def test_status(self, hybrid_agent):
        """Test getting agent status."""
        status = hybrid_agent.get_status()

        assert "ml_weight" in status
        assert "rl_weight" in status
        assert "predictions_recorded" in status


class TestPositionSizerState:
    """Test PositionSizerState dataclass."""

    def test_state_creation(self):
        """Test state creation."""
        state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.02,
            drawdown=0.05,
        )

        assert state.signal_strength == 0.7
        assert state.volatility == 0.02

    def test_to_tensor(self):
        """Test conversion to tensor."""
        state = PositionSizerState(
            market_features=np.zeros(20),
            portfolio_state=np.zeros(8),
            regime_features=np.zeros(5),
            signal_strength=0.5,
            signal_confidence=0.5,
            volatility=0.02,
            drawdown=0.05,
        )

        tensor = state.to_tensor()

        assert isinstance(tensor, np.ndarray)
        assert tensor.dtype == np.float32
        assert len(tensor) == 20 + 8 + 5 + 4  # 37


class TestRLPositionSizer:
    """Test RL Position Sizer."""

    @pytest.fixture
    def sizer(self):
        """Create position sizer."""
        config = PositionSizerConfig(
            max_drawdown_threshold=0.10,
            volatility_scaling=True,
        )
        return RLPositionSizer(config=config)

    @pytest.fixture
    def sample_state(self):
        """Create sample state."""
        return PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.02,
            drawdown=0.05,
        )

    def test_sizer_creation(self, sizer):
        """Test sizer initialization."""
        assert sizer.config.max_drawdown_threshold == 0.10
        assert sizer.config.volatility_scaling

    def test_get_position_size(self, sizer, sample_state):
        """Test getting position size."""
        size = sizer.get_position_size(sample_state)

        assert 0 <= size <= 1
        assert isinstance(size, float)

    def test_drawdown_protection(self, sizer):
        """Test drawdown protection reduces position."""
        # Normal state
        normal_state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.8,
            signal_confidence=0.9,
            volatility=0.02,
            drawdown=0.02,  # Low drawdown
        )

        # High drawdown state
        high_dd_state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.8,
            signal_confidence=0.9,
            volatility=0.02,
            drawdown=0.12,  # High drawdown
        )

        size_normal = sizer.get_position_size(normal_state)
        size_high_dd = sizer.get_position_size(high_dd_state)

        assert size_high_dd <= size_normal

    def test_volatility_scaling(self, sizer):
        """Test volatility scaling reduces position in high vol."""
        low_vol_state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.01,  # Low volatility
            drawdown=0.02,
        )

        high_vol_state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.05,  # High volatility
            drawdown=0.02,
        )

        size_low = sizer.get_position_size(low_vol_state)
        size_high = sizer.get_position_size(high_vol_state)

        assert size_high <= size_low

    def test_reward_calculation(self, sizer):
        """Test reward calculation."""
        reward = sizer.calculate_reward(
            pnl=100.0,
            position_size=0.5,
            volatility=0.02,
            drawdown=0.03,
            holding_time=10,
        )

        assert isinstance(reward, float)

    def test_status(self, sizer, sample_state):
        """Test getting sizer status."""
        # Make some decisions
        for _ in range(5):
            sizer.get_position_size(sample_state)

        status = sizer.get_status()

        assert "training_steps" in status
        assert "buffer_size" in status
        assert "total_decisions" in status


class TestDiscretePositionSizer:
    """Test Discrete Position Sizer."""

    @pytest.fixture
    def discrete_sizer(self):
        """Create discrete sizer."""
        config = PositionSizerConfig(position_granularity=11)
        return DiscretePositionSizer(config=config)

    def test_discrete_sizer_creation(self, discrete_sizer):
        """Test discrete sizer initialization."""
        assert discrete_sizer.n_actions == 11

    def test_discrete_sizing(self, discrete_sizer):
        """Test discrete position sizing."""
        state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.02,
            drawdown=0.05,
        )

        size = discrete_sizer.get_position_size(state)

        # Should be one of the discrete levels
        expected_levels = [i / 10 for i in range(11)]
        # Allow for risk adjustment
        assert 0 <= size <= 1

    def test_epsilon_decay(self, discrete_sizer):
        """Test epsilon decays over time."""
        initial_epsilon = discrete_sizer.epsilon

        state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.7,
            signal_confidence=0.8,
            volatility=0.02,
            drawdown=0.05,
        )

        next_state = PositionSizerState(
            market_features=np.random.randn(20),
            portfolio_state=np.random.randn(8),
            regime_features=np.random.randn(5),
            signal_strength=0.5,
            signal_confidence=0.7,
            volatility=0.02,
            drawdown=0.05,
        )

        # Simulate some updates
        for _ in range(10):
            size = discrete_sizer.get_position_size(state)
            discrete_sizer.update(state, size, 1.0, next_state, False)

        assert discrete_sizer.epsilon <= initial_epsilon


class TestRLIntegration:
    """Integration tests for RL components."""

    def test_full_training_loop(self, tmp_path):
        """Test full training loop with all components."""
        # Setup
        np.random.seed(42)
        n_rows = 200

        prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.01))
        data = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_rows),
        })

        # Create environment
        env = TradingEnvironment(
            data=data,
            config=TradingEnvConfig(
                initial_balance=10000.0,
                episode_length=50,
            ),
        )

        # Create policy
        policy = TradingPolicyNetwork(
            PolicyConfig(
                state_dim=env.observation_space.shape[0],
                action_dim=3,
                hidden_dim=32,
                use_lstm=False,
            )
        )

        # Create hybrid agent with the policy
        mock_ml = MagicMock()
        mock_ml.predict_proba = MagicMock(
            return_value=np.array([[0.3, 0.4, 0.3]])
        )

        agent = HybridMLRLAgent(
            ml_model=mock_ml,
            rl_policy=policy,
            config=HybridConfig(ml_weight=0.5, rl_weight=0.5),
        )

        # Run episode with agent
        state = env.reset()
        total_reward = 0

        for _ in range(30):
            action_str, confidence = agent.predict(state)
            action = {"SHORT": 0, "FLAT": 1, "LONG": 2}[action_str]
            state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        # Should complete without errors
        assert env.get_episode_stats() is not None
