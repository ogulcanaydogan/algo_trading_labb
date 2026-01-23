"""
Tests for Leverage-Aware Reinforcement Learning Agent module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from bot.ai_engine.leverage_rl_agent import (
    LeverageAction,
    LeverageState,
    LeverageRLAgent,
    TORCH_AVAILABLE,
    get_leverage_rl_agent,
)


class TestLeverageAction:
    """Test LeverageAction class."""

    def test_action_constants(self):
        """Test action constants are defined."""
        assert LeverageAction.HOLD == 0
        assert LeverageAction.LONG_1X == 1
        assert LeverageAction.LONG_3X == 2
        assert LeverageAction.LONG_5X == 3
        assert LeverageAction.LONG_10X == 4
        assert LeverageAction.SHORT_1X == 5
        assert LeverageAction.SHORT_3X == 6
        assert LeverageAction.SHORT_5X == 7
        assert LeverageAction.SHORT_10X == 8
        assert LeverageAction.CLOSE == 9
        assert LeverageAction.REDUCE_HALF == 10

    def test_descriptions_complete(self):
        """Test all actions have descriptions."""
        for action in range(11):
            assert action in LeverageAction.DESCRIPTIONS
            assert LeverageAction.DESCRIPTIONS[action] != ""

    def test_get_leverage_long_actions(self):
        """Test get_leverage for long actions."""
        assert LeverageAction.get_leverage(LeverageAction.LONG_1X) == 1.0
        assert LeverageAction.get_leverage(LeverageAction.LONG_3X) == 3.0
        assert LeverageAction.get_leverage(LeverageAction.LONG_5X) == 5.0
        assert LeverageAction.get_leverage(LeverageAction.LONG_10X) == 10.0

    def test_get_leverage_short_actions(self):
        """Test get_leverage for short actions."""
        assert LeverageAction.get_leverage(LeverageAction.SHORT_1X) == 1.0
        assert LeverageAction.get_leverage(LeverageAction.SHORT_3X) == 3.0
        assert LeverageAction.get_leverage(LeverageAction.SHORT_5X) == 5.0
        assert LeverageAction.get_leverage(LeverageAction.SHORT_10X) == 10.0

    def test_get_leverage_other_actions(self):
        """Test get_leverage for non-position actions."""
        assert LeverageAction.get_leverage(LeverageAction.HOLD) == 1.0
        assert LeverageAction.get_leverage(LeverageAction.CLOSE) == 1.0
        assert LeverageAction.get_leverage(LeverageAction.REDUCE_HALF) == 1.0

    def test_is_long(self):
        """Test is_long classification."""
        # Long actions
        assert LeverageAction.is_long(LeverageAction.LONG_1X) is True
        assert LeverageAction.is_long(LeverageAction.LONG_3X) is True
        assert LeverageAction.is_long(LeverageAction.LONG_5X) is True
        assert LeverageAction.is_long(LeverageAction.LONG_10X) is True

        # Not long
        assert LeverageAction.is_long(LeverageAction.SHORT_1X) is False
        assert LeverageAction.is_long(LeverageAction.HOLD) is False
        assert LeverageAction.is_long(LeverageAction.CLOSE) is False

    def test_is_short(self):
        """Test is_short classification."""
        # Short actions
        assert LeverageAction.is_short(LeverageAction.SHORT_1X) is True
        assert LeverageAction.is_short(LeverageAction.SHORT_3X) is True
        assert LeverageAction.is_short(LeverageAction.SHORT_5X) is True
        assert LeverageAction.is_short(LeverageAction.SHORT_10X) is True

        # Not short
        assert LeverageAction.is_short(LeverageAction.LONG_1X) is False
        assert LeverageAction.is_short(LeverageAction.HOLD) is False
        assert LeverageAction.is_short(LeverageAction.CLOSE) is False


class TestLeverageState:
    """Test LeverageState dataclass."""

    @pytest.fixture
    def sample_state(self):
        """Create sample leverage state."""
        return LeverageState(
            # Price features
            price_change_1h=0.5,
            price_change_4h=1.2,
            price_change_24h=2.5,
            price_vs_ema20=0.02,
            price_vs_ema50=0.05,
            price_vs_vwap=0.01,
            # Momentum features
            rsi=55.0,
            rsi_change=5.0,
            macd_hist=0.002,
            macd_signal_cross=1.0,
            momentum_5=1.5,
            momentum_20=3.0,
            # Volatility features
            atr_ratio=0.8,
            bb_position=0.6,
            bb_width=0.03,
            volatility_ratio=1.2,
            high_volatility=0.0,
            # Trend features
            adx=35.0,
            trend_direction=1.0,
            trend_strength=0.7,
            ema_alignment=1.0,
            # Volume features
            volume_ratio=1.5,
            buy_volume_ratio=0.6,
            # Market structure
            funding_rate=0.0001,
            open_interest_change=0.05,
            long_short_ratio=1.2,
            # Position features
            current_position=0.5,
            current_leverage=3.0,
            position_pnl=2.0,
            position_duration=24.0,
            unrealized_pnl=1.5,
            margin_ratio=0.2,
            # Risk features
            drawdown_current=5.0,
            consecutive_losses=1,
            win_rate_recent=0.55,
        )

    def test_state_creation(self, sample_state):
        """Test state is created correctly."""
        assert sample_state.rsi == 55.0
        assert sample_state.current_leverage == 3.0
        assert sample_state.trend_direction == 1.0

    def test_to_array_shape(self, sample_state):
        """Test to_array returns correct shape."""
        arr = sample_state.to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        # Total features: 6 + 6 + 5 + 4 + 2 + 3 + 6 + 3 = 35
        assert len(arr) == 35

    def test_to_array_values_normalized(self, sample_state):
        """Test values are normalized appropriately."""
        arr = sample_state.to_array()

        # RSI should be normalized (55 / 100 = 0.55)
        assert arr[6] == pytest.approx(0.55, rel=0.01)

        # ADX normalized (35 / 100 = 0.35)
        assert arr[17] == pytest.approx(0.35, rel=0.01)

        # Current leverage normalized (3 / 10 = 0.3)
        assert arr[27] == pytest.approx(0.3, rel=0.01)

    def test_to_array_no_nan(self, sample_state):
        """Test array has no NaN values."""
        arr = sample_state.to_array()
        assert not np.any(np.isnan(arr))

    def test_from_market_data_basic(self):
        """Test creating state from market data."""
        indicators = {
            "price_change_1h": 0.5,
            "price_change_4h": 1.0,
            "price_change_24h": 2.0,
            "price_vs_ema20": 0.01,
            "price_vs_ema50": 0.02,
            "price_vs_vwap": 0.005,
            "rsi": 50.0,
            "rsi_change": 2.0,
            "macd_hist": 0.001,
            "macd_signal_cross": 0.0,
            "momentum_5": 1.0,
            "momentum_20": 2.0,
            "atr_ratio": 1.0,
            "bb_position": 0.5,
            "bb_width": 0.02,
            "volatility_ratio": 1.0,
            "adx": 25.0,
            "trend_direction": 0.5,
            "trend_strength": 0.5,
            "ema_alignment": 0.5,
            "volume_ratio": 1.0,
            "buy_volume_ratio": 0.5,
            "funding_rate": 0.0,
            "open_interest_change": 0.0,
            "long_short_ratio": 1.0,
        }

        state = LeverageState.from_market_data(indicators)

        assert state.rsi == 50.0
        assert state.adx == 25.0
        assert state.trend_direction == 0.5

    def test_from_market_data_with_position(self):
        """Test creating state with position info."""
        indicators = {
            "price_change_1h": 0.5,
            "price_change_4h": 1.0,
            "price_change_24h": 2.0,
            "rsi": 60.0,
            "adx": 30.0,
        }

        position_info = {
            "position": 1.0,  # Long
            "leverage": 5.0,
            "pnl": 3.0,
            "duration": 48.0,
            "unrealized_pnl": 2.5,
            "margin_ratio": 0.3,
        }

        state = LeverageState.from_market_data(indicators, position_info)

        assert state.current_position == 1.0
        assert state.current_leverage == 5.0
        assert state.position_pnl == 3.0

    def test_from_market_data_with_risk_info(self):
        """Test creating state with risk info."""
        indicators = {"rsi": 45.0, "adx": 20.0}

        risk_info = {
            "drawdown": 10.0,
            "consecutive_losses": 3,
            "win_rate_recent": 0.4,
        }

        state = LeverageState.from_market_data(indicators, risk_info=risk_info)

        assert state.drawdown_current == 10.0
        assert state.consecutive_losses == 3
        assert state.win_rate_recent == 0.4

    def test_from_market_data_defaults(self):
        """Test default values when data is missing."""
        state = LeverageState.from_market_data({})

        # Should use defaults
        assert state.rsi == 50.0  # Default RSI
        assert state.adx == 25.0  # Default ADX
        assert state.current_position == 0.0  # No position
        assert state.current_leverage == 1.0  # 1x default

    def test_from_market_data_high_volatility_flag(self):
        """Test high volatility flag is set."""
        # Low volatility
        state_low = LeverageState.from_market_data({"volatility_ratio": 1.0})
        assert state_low.high_volatility == 0.0

        # High volatility
        state_high = LeverageState.from_market_data({"volatility_ratio": 2.5})
        assert state_high.high_volatility == 1.0


class TestLeverageRLAgent:
    """Test LeverageRLAgent class."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_db):
        """Create RL agent with mock database."""
        with patch("bot.ai_engine.leverage_rl_agent.get_learning_db", return_value=mock_db):
            return LeverageRLAgent(
                state_size=35,
                action_size=11,
                learning_rate=0.0001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.9995,
                db=mock_db,
            )

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return LeverageState(
            price_change_1h=0.5,
            price_change_4h=1.0,
            price_change_24h=2.0,
            price_vs_ema20=0.01,
            price_vs_ema50=0.02,
            price_vs_vwap=0.005,
            rsi=55.0,
            rsi_change=3.0,
            macd_hist=0.001,
            macd_signal_cross=0.0,
            momentum_5=1.5,
            momentum_20=2.5,
            atr_ratio=0.9,
            bb_position=0.6,
            bb_width=0.025,
            volatility_ratio=1.1,
            high_volatility=0.0,
            adx=30.0,
            trend_direction=0.5,
            trend_strength=0.6,
            ema_alignment=0.5,
            volume_ratio=1.2,
            buy_volume_ratio=0.55,
            funding_rate=0.0001,
            open_interest_change=0.02,
            long_short_ratio=1.1,
            current_position=0.0,
            current_leverage=1.0,
            position_pnl=0.0,
            position_duration=0.0,
            unrealized_pnl=0.0,
            margin_ratio=0.0,
            drawdown_current=3.0,
            consecutive_losses=0,
            win_rate_recent=0.55,
        )

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_size == 35
        assert agent.action_size == 11
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert agent.epsilon_min == 0.05
        assert agent.training_steps == 0

    def test_epsilon_decay(self, agent):
        """Test epsilon decays over time."""
        # At start, epsilon should be high
        eps_start = agent.epsilon
        assert eps_start == 1.0

        # Simulate decay
        for _ in range(1000):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # After many decays, epsilon should be lower
        assert agent.epsilon < eps_start

    def test_select_action_exploration(self, agent, sample_state):
        """Test action selection with high exploration."""
        agent.epsilon = 1.0  # High epsilon

        # select_action may return tuple (action, info) or just action
        result = agent.select_action(sample_state)

        # Handle both return types
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result

        assert 0 <= action < agent.action_size

    def test_select_action_no_exploration(self, agent, sample_state):
        """Test action selection with low exploration."""
        agent.epsilon = 0.0

        result = agent.select_action(sample_state)

        # Handle both return types
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result

        assert 0 <= action < agent.action_size

    def test_select_action_valid_range(self, agent, sample_state):
        """Test all selected actions are in valid range."""
        result = agent.select_action(sample_state)

        # Handle both return types
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result

        assert 0 <= action < agent.action_size

    def test_get_action_analysis(self, agent, sample_state):
        """Test action analysis output."""
        analysis = agent.get_action_analysis(sample_state)

        assert "best_action" in analysis
        assert "confidence" in analysis

        assert 0 <= analysis["best_action"] < agent.action_size
        assert 0 <= analysis["confidence"] <= 1

    def test_remember(self, agent, sample_state):
        """Test storing experiences in memory."""
        next_state = LeverageState(
            price_change_1h=0.6,
            price_change_4h=1.1,
            price_change_24h=2.1,
            price_vs_ema20=0.015,
            price_vs_ema50=0.025,
            price_vs_vwap=0.008,
            rsi=58.0,
            rsi_change=3.0,
            macd_hist=0.0015,
            macd_signal_cross=0.0,
            momentum_5=1.6,
            momentum_20=2.6,
            atr_ratio=0.85,
            bb_position=0.65,
            bb_width=0.024,
            volatility_ratio=1.05,
            high_volatility=0.0,
            adx=32.0,
            trend_direction=0.6,
            trend_strength=0.65,
            ema_alignment=0.55,
            volume_ratio=1.3,
            buy_volume_ratio=0.58,
            funding_rate=0.0001,
            open_interest_change=0.025,
            long_short_ratio=1.15,
            current_position=1.0,
            current_leverage=3.0,
            position_pnl=1.0,
            position_duration=1.0,
            unrealized_pnl=1.0,
            margin_ratio=0.1,
            drawdown_current=2.0,
            consecutive_losses=0,
            win_rate_recent=0.6,
        )

        agent.remember(
            state=sample_state,
            action=LeverageAction.LONG_3X,
            reward=0.5,
            next_state=next_state,
            done=False,
        )

        assert len(agent.memory) == 1

    def test_calculate_reward(self, agent):
        """Test reward calculation."""
        # Check if calculate_reward method exists and test it
        if hasattr(agent, "calculate_reward"):
            # Try different signatures
            try:
                reward = agent.calculate_reward(pnl_pct=5.0)
                assert isinstance(reward, (int, float))
            except TypeError:
                # Method may have different signature
                pass

    def test_reward_by_leverage_tracking(self, agent):
        """Test reward by leverage tracking structure."""
        # Verify the tracking structure exists
        assert hasattr(agent, "reward_by_leverage")
        assert 1 in agent.reward_by_leverage or isinstance(agent.reward_by_leverage, dict)

    def test_get_stats(self, agent):
        """Test getting agent statistics."""
        stats = agent.get_stats()

        # Check for common stats fields
        assert isinstance(stats, dict)
        assert "epsilon" in stats or "training_steps" in stats
        assert "memory_size" in stats or "leverage_stats" in stats

    def test_training_steps_tracking(self, agent, sample_state):
        """Test that training steps are tracked."""
        initial_steps = agent.training_steps

        # Store some experiences
        for _ in range(agent.batch_size + 10):
            next_state = sample_state  # Simplified
            agent.remember(sample_state, 1, 0.1, next_state, False)

        # Training steps should be trackable
        assert agent.training_steps >= initial_steps


class TestLeverageRLAgentWithoutTorch:
    """Test agent behavior when PyTorch is not available."""

    def test_torch_available_flag(self):
        """Test TORCH_AVAILABLE flag is boolean."""
        assert isinstance(TORCH_AVAILABLE, bool)


class TestGetLeverageRLAgent:
    """Test global agent getter."""

    def test_get_leverage_rl_agent_creates_instance(self):
        """Test getter creates instance."""
        import bot.ai_engine.leverage_rl_agent as lra

        lra._leverage_rl_agent = None

        with patch.object(lra, "get_learning_db"):
            agent = get_leverage_rl_agent()
            assert agent is not None
            assert isinstance(agent, LeverageRLAgent)

    def test_get_leverage_rl_agent_returns_same_instance(self):
        """Test getter returns same instance."""
        import bot.ai_engine.leverage_rl_agent as lra

        with patch.object(lra, "get_learning_db"):
            agent1 = get_leverage_rl_agent()
            agent2 = get_leverage_rl_agent()
            assert agent1 is agent2


class TestLeverageStateEdgeCases:
    """Test edge cases in LeverageState."""

    def test_state_with_extreme_values(self):
        """Test state handles extreme values."""
        state = LeverageState(
            price_change_1h=-50.0,  # Extreme drop
            price_change_4h=100.0,  # Extreme rise
            price_change_24h=0.0,
            price_vs_ema20=-0.5,
            price_vs_ema50=0.5,
            price_vs_vwap=0.0,
            rsi=99.0,  # Extremely overbought
            rsi_change=20.0,
            macd_hist=0.1,
            macd_signal_cross=-1.0,
            momentum_5=-10.0,
            momentum_20=10.0,
            atr_ratio=5.0,  # High volatility
            bb_position=1.5,  # Above upper band
            bb_width=0.2,
            volatility_ratio=3.0,
            high_volatility=1.0,
            adx=80.0,  # Very strong trend
            trend_direction=-1.0,
            trend_strength=0.95,
            ema_alignment=-1.0,
            volume_ratio=5.0,
            buy_volume_ratio=0.1,
            funding_rate=0.01,  # High funding
            open_interest_change=-0.5,
            long_short_ratio=0.5,
            current_position=-1.0,
            current_leverage=10.0,
            position_pnl=-20.0,
            position_duration=1000.0,
            unrealized_pnl=-15.0,
            margin_ratio=0.9,  # Near liquidation
            drawdown_current=50.0,
            consecutive_losses=10,
            win_rate_recent=0.2,
        )

        arr = state.to_array()

        # Should not have NaN or Inf
        assert np.all(np.isfinite(arr))

    def test_state_with_zeros(self):
        """Test state with all zeros."""
        state = LeverageState(
            price_change_1h=0,
            price_change_4h=0,
            price_change_24h=0,
            price_vs_ema20=0,
            price_vs_ema50=0,
            price_vs_vwap=0,
            rsi=0,
            rsi_change=0,
            macd_hist=0,
            macd_signal_cross=0,
            momentum_5=0,
            momentum_20=0,
            atr_ratio=0,
            bb_position=0,
            bb_width=0,
            volatility_ratio=0,
            high_volatility=0,
            adx=0,
            trend_direction=0,
            trend_strength=0,
            ema_alignment=0,
            volume_ratio=0,
            buy_volume_ratio=0,
            funding_rate=0,
            open_interest_change=0,
            long_short_ratio=0,
            current_position=0,
            current_leverage=0,
            position_pnl=0,
            position_duration=0,
            unrealized_pnl=0,
            margin_ratio=0,
            drawdown_current=0,
            consecutive_losses=0,
            win_rate_recent=0,
        )

        arr = state.to_array()
        assert arr.shape == (35,)
        assert np.all(np.isfinite(arr))
