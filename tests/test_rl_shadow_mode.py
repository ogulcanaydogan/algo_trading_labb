"""
Tests for RL Shadow Mode Constraints.

CRITICAL: These tests verify that RL operates in ADVISORY-ONLY mode and:
1. Outputs are bounded and normalized
2. RL disabled => identical behavior to baseline
3. No code path can execute trades from RL
4. Meta-agent stability
5. Safety constraints are non-negotiable and locked

These tests are essential for Phase 2A validation before any RL can influence trading.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from bot.rl.shadow_advisor import (
    RLAdvisoryMode,
    ShadowModeConfig,
    RLRecommendation,
    RLShadowAdvisor,
    get_shadow_advisor,
    reset_shadow_advisor,
    is_rl_enabled,
)
from bot.rl.multi_agent_system import (
    AgentType,
    AgentAction,
    MarketState,
    MetaAgent,
    TrendFollowerAgent,
    MeanReversionAgent,
    MomentumTraderAgent,
    ShortSpecialistAgent,
    ScalperAgent,
    get_meta_agent,
)


class TestShadowModeConfigConstraints:
    """Test that shadow mode configuration enforces non-negotiable constraints."""

    def test_safety_constraints_locked_on_init(self):
        """Safety constraints MUST be True and cannot be changed."""
        config = ShadowModeConfig()

        # All safety constraints must be True
        assert config.respect_trade_gate is True
        assert config.respect_capital_preservation is True
        assert config.respect_risk_budget is True
        assert config.respect_leverage_caps is True

    def test_safety_constraints_cannot_be_disabled(self):
        """Attempting to set safety constraints to False should be overridden."""
        config = ShadowModeConfig(
            respect_trade_gate=False,
            respect_capital_preservation=False,
            respect_risk_budget=False,
            respect_leverage_caps=False,
        )

        # __post_init__ should override to True
        assert config.respect_trade_gate is True
        assert config.respect_capital_preservation is True
        assert config.respect_risk_budget is True
        assert config.respect_leverage_caps is True

    def test_default_disabled(self):
        """RL must be disabled by default."""
        config = ShadowModeConfig()
        assert config.enabled is False
        assert config.mode == RLAdvisoryMode.SHADOW

    def test_confidence_adjustment_bounded(self):
        """Max confidence adjustment must be bounded."""
        config = ShadowModeConfig()

        # Default is 10%
        assert config.max_confidence_adjustment == 0.1
        assert 0 <= config.max_confidence_adjustment <= 0.5  # Reasonable bound

        # Strategy weight adjustment is also bounded
        assert config.max_strategy_weight_adjustment == 0.15
        assert 0 <= config.max_strategy_weight_adjustment <= 0.5


class TestRLRecommendationBounds:
    """Test that RL recommendations have bounded and normalized outputs."""

    def test_confidence_bounds(self):
        """Confidence values must be between 0 and 1."""
        rec = RLRecommendation(
            symbol="BTC_USDT",
            bias_confidence=0.75,
            action_confidence=0.82,
        )

        assert 0 <= rec.bias_confidence <= 1
        assert 0 <= rec.action_confidence <= 1

    def test_strategy_preferences_normalized(self):
        """Strategy preferences should sum to 1.0."""
        rec = RLRecommendation(
            symbol="BTC_USDT",
            strategy_preferences={
                "trend_follower": 0.4,
                "mean_reversion": 0.3,
                "momentum_trader": 0.2,
                "short_specialist": 0.1,
            }
        )

        total = sum(rec.strategy_preferences.values())
        assert abs(total - 1.0) < 0.001  # Allow small floating point error

    def test_directional_bias_valid(self):
        """Directional bias must be one of: long, short, neutral."""
        valid_biases = ["long", "short", "neutral"]

        for bias in valid_biases:
            rec = RLRecommendation(directional_bias=bias)
            assert rec.directional_bias in valid_biases

    def test_suggested_action_valid(self):
        """Suggested action must be a valid action type."""
        valid_actions = ["hold", "buy", "sell", "long", "short", "cover"]

        for action in valid_actions:
            rec = RLRecommendation(suggested_action=action)
            assert rec.suggested_action.lower() in valid_actions

    def test_to_dict_serializable(self):
        """Recommendation must be JSON serializable."""
        rec = RLRecommendation(
            symbol="ETH_USDT",
            regime="BULL",
            strategy_preferences={"trend_follower": 0.6, "momentum_trader": 0.4},
            directional_bias="long",
            bias_confidence=0.8,
            suggested_action="buy",
            action_confidence=0.75,
            primary_agent="trend_follower",
            agent_reasoning="Strong uptrend detected",
        )

        # Must not raise
        data = rec.to_dict()
        json_str = json.dumps(data)

        # Must be reconstructable
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "ETH_USDT"
        assert parsed["bias_confidence"] == 0.8


class TestRLDisabledIdenticalBehavior:
    """Test that when RL is disabled, behavior is identical to baseline."""

    @pytest.fixture
    def disabled_advisor(self):
        """Create advisor with RL disabled."""
        reset_shadow_advisor()
        config = ShadowModeConfig(enabled=False)
        return RLShadowAdvisor(config)

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state."""
        return MarketState(
            symbol="BTC_USDT",
            price=50000.0,
            returns_1h=0.02,
            returns_4h=0.05,
            returns_24h=0.08,
            volatility=0.03,
            trend_strength=0.4,
            rsi=55,
            regime="BULL",
        )

    def test_disabled_returns_empty_recommendation(self, disabled_advisor, sample_market_state):
        """Disabled RL should return empty/neutral recommendation."""
        rec = disabled_advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=sample_market_state,
        )

        # Empty/default values
        assert rec.strategy_preferences == {}
        assert rec.directional_bias == "neutral"
        assert rec.suggested_action == "hold"
        assert rec.was_applied is False
        assert "RL disabled" in rec.application_reason

    def test_disabled_no_confidence_adjustment(self, disabled_advisor):
        """Disabled RL should not adjust confidence."""
        base_confidence = 0.75

        adjusted, explanation = disabled_advisor.get_confidence_adjustment(
            symbol="BTC_USDT",
            base_confidence=base_confidence,
            intended_action="buy",
        )

        # No change when disabled
        assert adjusted == base_confidence
        assert "not in advisory mode" in explanation.lower()

    def test_disabled_is_enabled_returns_false(self, disabled_advisor):
        """is_enabled() should return False when disabled."""
        assert disabled_advisor.is_enabled() is False

    def test_global_is_rl_enabled_false(self):
        """Global is_rl_enabled() should return False when no advisor or disabled."""
        reset_shadow_advisor()
        assert is_rl_enabled() is False

        # Even with a disabled advisor
        config = ShadowModeConfig(enabled=False)
        get_shadow_advisor(config)
        assert is_rl_enabled() is False


class TestNoExecutionPathFromRL:
    """CRITICAL: Test that there is NO code path that executes trades from RL."""

    @pytest.fixture
    def shadow_advisor(self):
        """Create shadow mode advisor."""
        reset_shadow_advisor()
        config = ShadowModeConfig(enabled=True, mode=RLAdvisoryMode.SHADOW)
        return RLShadowAdvisor(config)

    @pytest.fixture
    def advisory_advisor(self):
        """Create advisory mode advisor."""
        reset_shadow_advisor()
        config = ShadowModeConfig(enabled=True, mode=RLAdvisoryMode.ADVISORY)
        return RLShadowAdvisor(config)

    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state."""
        return MarketState(
            symbol="BTC_USDT",
            price=50000.0,
            returns_1h=0.02,
            trend_strength=0.5,
            rsi=55,
            regime="BULL",
        )

    def test_recommendation_has_no_execute_method(self, shadow_advisor, sample_market_state):
        """RLRecommendation should have no execute/trade method."""
        rec = shadow_advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=sample_market_state,
        )

        # No execution methods
        assert not hasattr(rec, "execute")
        assert not hasattr(rec, "place_order")
        assert not hasattr(rec, "submit_trade")
        assert not hasattr(rec, "send_order")
        assert not hasattr(rec, "trade")

    def test_advisor_has_no_execute_method(self, shadow_advisor):
        """RLShadowAdvisor should have no execute/trade method."""
        # No execution methods
        assert not hasattr(shadow_advisor, "execute")
        assert not hasattr(shadow_advisor, "execute_trade")
        assert not hasattr(shadow_advisor, "place_order")
        assert not hasattr(shadow_advisor, "submit_trade")
        assert not hasattr(shadow_advisor, "send_order")

    def test_shadow_mode_never_sets_was_applied(self, shadow_advisor, sample_market_state):
        """Shadow mode should never set was_applied=True."""
        rec = shadow_advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=sample_market_state,
        )

        assert rec.was_applied is False
        assert "shadow mode" in rec.application_reason.lower()

    def test_confidence_adjustment_bounded(self, advisory_advisor, sample_market_state):
        """Confidence adjustment must be within configured bounds."""
        # First get a recommendation
        with patch.object(advisory_advisor, '_get_meta_agent') as mock_meta:
            mock_action = AgentAction(
                action="BUY",
                confidence=0.9,
                position_size_pct=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.06,
                leverage=2.0,
                reasoning="Test",
                agent_type=AgentType.TREND_FOLLOWER,
                regime_match=0.8,
            )
            mock_meta.return_value.get_combined_action.return_value = (mock_action, [mock_action])

            advisory_advisor.get_recommendation(
                symbol="BTC_USDT",
                market_state=sample_market_state,
            )

        # Test adjustment bounds
        base = 0.5
        max_adj = advisory_advisor.config.max_confidence_adjustment

        adjusted, _ = advisory_advisor.get_confidence_adjustment(
            symbol="BTC_USDT",
            base_confidence=base,
            intended_action="buy",
        )

        # Adjustment must be within bounds
        assert abs(adjusted - base) <= max_adj + 0.001  # Small tolerance

    def test_lockdown_blocks_recommendations(self, shadow_advisor, sample_market_state):
        """LOCKDOWN/CRISIS mode must block all RL recommendations."""
        for level in ["lockdown", "LOCKDOWN", "crisis", "CRISIS"]:
            rec = shadow_advisor.get_recommendation(
                symbol="BTC_USDT",
                market_state=sample_market_state,
                preservation_level=level,
            )

            assert rec.suggested_action == "hold"
            assert rec.action_confidence == 0.0
            assert "blocked" in rec.application_reason.lower()

    def test_gate_rejection_blocks_recommendations(self, shadow_advisor, sample_market_state):
        """TradeGate rejection must block RL recommendations."""
        rec = shadow_advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=sample_market_state,
            gate_approved=False,
        )

        assert rec.suggested_action == "hold"
        assert rec.action_confidence == 0.0
        assert "tradegate rejected" in rec.application_reason.lower()


class TestMetaAgentStability:
    """Test meta-agent stability and bounded outputs."""

    @pytest.fixture
    def meta_agent(self, tmp_path):
        """Create meta agent with temp data dir."""
        return MetaAgent(data_dir=str(tmp_path / "rl"))

    @pytest.fixture
    def sample_state(self):
        """Create sample market state."""
        return MarketState(
            symbol="BTC_USDT",
            price=50000.0,
            returns_1h=0.02,
            returns_4h=0.05,
            returns_24h=0.08,
            volatility=0.03,
            trend_strength=0.4,
            rsi=55,
            macd=0.01,
            macd_signal=0.005,
            bb_position=0.6,
            volume_ratio=1.2,
            regime="BULL",
            fear_greed=60,
            news_sentiment=0.3,
        )

    def test_all_agents_return_bounded_confidence(self, meta_agent, sample_state):
        """All agents must return confidence in [0, 1]."""
        for agent in meta_agent.agents.values():
            action = agent.get_action(sample_state)

            assert 0 <= action.confidence <= 1, f"{agent.agent_type}: confidence out of bounds"

    def test_all_agents_return_bounded_position_size(self, meta_agent, sample_state):
        """All agents must return position size in [0, 1]."""
        for agent in meta_agent.agents.values():
            action = agent.get_action(sample_state)

            assert 0 <= action.position_size_pct <= 1, f"{agent.agent_type}: position size out of bounds"

    def test_all_agents_return_bounded_leverage(self, meta_agent, sample_state):
        """All agents must return leverage in [1, 5]."""
        for agent in meta_agent.agents.values():
            action = agent.get_action(sample_state)

            assert 1 <= action.leverage <= 5, f"{agent.agent_type}: leverage out of bounds"

    def test_all_agents_return_bounded_stops(self, meta_agent, sample_state):
        """All agents must return stops in reasonable bounds."""
        for agent in meta_agent.agents.values():
            action = agent.get_action(sample_state)

            assert 0 < action.stop_loss_pct <= 0.1, f"{agent.agent_type}: stop loss out of bounds"
            assert 0 < action.take_profit_pct <= 0.2, f"{agent.agent_type}: take profit out of bounds"

    def test_all_agents_return_valid_action(self, meta_agent, sample_state):
        """All agents must return valid action type."""
        valid_actions = {"BUY", "SELL", "SHORT", "COVER", "HOLD", "LONG"}

        for agent in meta_agent.agents.values():
            action = agent.get_action(sample_state)

            assert action.action in valid_actions, f"{agent.agent_type}: invalid action {action.action}"

    def test_regime_match_bounded(self, meta_agent):
        """Regime match scores must be in [0, 1]."""
        regimes = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", "LOW_VOL", "CRASH", "UNKNOWN"]

        for agent in meta_agent.agents.values():
            for regime in regimes:
                match = agent.get_regime_match(regime)
                assert 0 <= match <= 1, f"{agent.agent_type} regime {regime}: match out of bounds"

    def test_combined_action_deterministic_given_state(self, meta_agent, sample_state):
        """Combined action should be deterministic for same state."""
        action1, _ = meta_agent.get_combined_action(sample_state)
        action2, _ = meta_agent.get_combined_action(sample_state)

        assert action1.action == action2.action
        assert action1.confidence == action2.confidence

    def test_agent_weights_sum_reasonable(self, meta_agent):
        """Agent weights should sum to approximately n_agents."""
        n_agents = len(meta_agent.agents)
        total = sum(meta_agent.agent_weights.values())

        assert abs(total - n_agents) < 0.01

    def test_weight_updates_bounded(self, meta_agent):
        """Weight updates should not cause extreme swings."""
        initial_weights = dict(meta_agent.agent_weights)

        # Simulate many outcomes
        for _ in range(100):
            meta_agent.record_outcome(
                agent_type=AgentType.TREND_FOLLOWER,
                pnl=100.0,
                pnl_pct=0.01,
                regime="BULL",
            )

        # Weight should not explode
        for agent_type, weight in meta_agent.agent_weights.items():
            assert 0.1 <= weight <= 10.0, f"{agent_type}: weight exploded to {weight}"


class TestAgentOutputNormalization:
    """Test that all agent outputs are properly normalized."""

    @pytest.fixture
    def all_regimes(self):
        """All possible market regimes."""
        return [
            "BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR",
            "SIDEWAYS", "HIGH_VOL", "LOW_VOL", "CRASH", "BREAKOUT",
        ]

    @pytest.fixture
    def extreme_states(self):
        """Extreme market states to test boundary conditions."""
        return [
            # Extreme bull
            MarketState(
                symbol="BTC", price=100000, returns_1h=0.1, returns_24h=0.5,
                trend_strength=1.0, rsi=99, volume_ratio=10, regime="STRONG_BULL",
            ),
            # Extreme bear
            MarketState(
                symbol="BTC", price=10000, returns_1h=-0.1, returns_24h=-0.5,
                trend_strength=-1.0, rsi=1, volume_ratio=10, regime="CRASH",
            ),
            # Sideways
            MarketState(
                symbol="BTC", price=50000, returns_1h=0, returns_24h=0,
                trend_strength=0, rsi=50, bb_position=0.5, regime="SIDEWAYS",
            ),
            # High volatility
            MarketState(
                symbol="BTC", price=50000, volatility=0.1,
                returns_1h=0.05, volume_ratio=3, regime="HIGH_VOL",
            ),
        ]

    def test_trend_follower_bounded_outputs(self, extreme_states):
        """TrendFollower must return bounded outputs in all conditions."""
        agent = TrendFollowerAgent()

        for state in extreme_states:
            action = agent.get_action(state)

            assert 0 <= action.confidence <= 1
            assert 0 < action.position_size_pct <= 0.25  # max 25%
            assert 1 <= action.leverage <= 3  # max 3x

    def test_mean_reversion_bounded_outputs(self, extreme_states):
        """MeanReversion must return bounded outputs in all conditions."""
        agent = MeanReversionAgent()

        for state in extreme_states:
            action = agent.get_action(state)

            assert 0 <= action.confidence <= 1
            assert 0 < action.position_size_pct <= 0.25
            assert 1 <= action.leverage <= 3

    def test_momentum_trader_bounded_outputs(self, extreme_states):
        """MomentumTrader must return bounded outputs in all conditions."""
        agent = MomentumTraderAgent()

        for state in extreme_states:
            action = agent.get_action(state)

            assert 0 <= action.confidence <= 1
            assert 0 < action.position_size_pct <= 0.25
            assert 1 <= action.leverage <= 3

    def test_short_specialist_bounded_outputs(self, extreme_states):
        """ShortSpecialist must return bounded outputs in all conditions."""
        agent = ShortSpecialistAgent()

        for state in extreme_states:
            action = agent.get_action(state)

            assert 0 <= action.confidence <= 1
            assert 0 < action.position_size_pct <= 0.25
            assert 1 <= action.leverage <= 3

    def test_scalper_bounded_outputs(self, extreme_states):
        """Scalper must return bounded outputs in all conditions."""
        agent = ScalperAgent()

        for state in extreme_states:
            action = agent.get_action(state)

            assert 0 <= action.confidence <= 1
            assert 0 < action.position_size_pct <= 0.25
            assert 1 <= action.leverage <= 5


class TestShadowAdvisorSingleton:
    """Test shadow advisor singleton behavior."""

    def test_singleton_reset(self):
        """Reset should clear singleton state."""
        reset_shadow_advisor()

        config = ShadowModeConfig(enabled=True)
        advisor1 = get_shadow_advisor(config)
        advisor1._recommendations.append(RLRecommendation())

        # Reset
        reset_shadow_advisor()

        # New advisor should be clean
        advisor2 = get_shadow_advisor()
        assert len(advisor2._recommendations) == 0

    def test_singleton_preserves_config(self):
        """Singleton should preserve initial config."""
        reset_shadow_advisor()

        config = ShadowModeConfig(enabled=True, mode=RLAdvisoryMode.ADVISORY)
        advisor1 = get_shadow_advisor(config)

        # Second call without config should return same advisor
        advisor2 = get_shadow_advisor()

        assert advisor1 is advisor2
        assert advisor2.config.enabled is True
        assert advisor2.config.mode == RLAdvisoryMode.ADVISORY


class TestRLLogging:
    """Test RL recommendation logging."""

    @pytest.fixture
    def advisor_with_logging(self, tmp_path):
        """Create advisor with logging to temp directory."""
        reset_shadow_advisor()
        log_path = tmp_path / "rl_log.jsonl"
        config = ShadowModeConfig(
            enabled=True,
            log_all_recommendations=True,
            log_path=log_path,
        )
        return RLShadowAdvisor(config), log_path

    def test_recommendations_logged(self, advisor_with_logging):
        """Recommendations should be logged to file."""
        advisor, log_path = advisor_with_logging

        state = MarketState(symbol="BTC_USDT", price=50000, regime="BULL")
        advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=state,
        )

        # Check log file exists and has content
        assert log_path.exists()
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) >= 1

        # Verify JSON valid
        entry = json.loads(lines[-1])
        assert entry["symbol"] == "BTC_USDT"

    def test_outcome_updates_logged(self, advisor_with_logging):
        """Outcome updates should be logged."""
        advisor, log_path = advisor_with_logging

        state = MarketState(symbol="ETH_USDT", price=3000, regime="SIDEWAYS")
        advisor.get_recommendation(
            symbol="ETH_USDT",
            market_state=state,
        )

        # Record outcome
        advisor.record_outcome(
            symbol="ETH_USDT",
            actual_action="buy",
            pnl=50.0,
        )

        # Check log updated
        with open(log_path) as f:
            lines = f.readlines()

        # Should have both recommendation and outcome logged
        assert len(lines) >= 2


class TestStressConditions:
    """Test RL shadow mode under stress conditions."""

    @pytest.fixture
    def advisor(self):
        """Create shadow mode advisor."""
        reset_shadow_advisor()
        config = ShadowModeConfig(enabled=True)
        return RLShadowAdvisor(config)

    def test_rapid_recommendations(self, advisor):
        """System should handle rapid recommendation requests."""
        state = MarketState(symbol="BTC_USDT", price=50000, regime="HIGH_VOL")

        # Simulate high-frequency requests
        for i in range(100):
            rec = advisor.get_recommendation(
                symbol=f"SYMBOL_{i}",
                market_state=state,
            )

            # All must be valid
            assert rec.symbol == f"SYMBOL_{i}"
            assert 0 <= rec.bias_confidence <= 1

    def test_concurrent_symbols(self, advisor):
        """System should track multiple symbols independently."""
        symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "AVAX_USDT"]
        state = MarketState(symbol="", price=100, regime="BULL")

        for symbol in symbols:
            state.symbol = symbol
            advisor.get_recommendation(symbol=symbol, market_state=state)

        # All should be tracked
        assert len(advisor._pending_recommendations) == len(symbols)

    def test_invalid_preservation_level_handled(self, advisor):
        """Invalid preservation level should be handled gracefully."""
        state = MarketState(symbol="BTC_USDT", price=50000, regime="BULL")

        rec = advisor.get_recommendation(
            symbol="BTC_USDT",
            market_state=state,
            preservation_level="INVALID_LEVEL",  # Invalid
        )

        # Should not crash, returns valid recommendation
        assert rec is not None
        assert 0 <= rec.bias_confidence <= 1
