"""
Tests for AI Integration Layer.

Tests the AIIntegrationManager, learning modes, decision aggregation, and training.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from bot.ai_integration import (
    AIComponentType,
    LearningMode,
    AIComponentConfig,
    LearningEvent,
    AIDecision,
    AIIntegrationManager,
    get_ai_integration,
    create_ai_integration,
)


class TestAIComponentType:
    """Test AIComponentType enum."""

    def test_all_component_types_defined(self):
        """Test all expected component types exist."""
        assert AIComponentType.PARAMETER_OPTIMIZER
        assert AIComponentType.STRATEGY_EVOLVER
        assert AIComponentType.RL_AGENT
        assert AIComponentType.LEVERAGE_RL
        assert AIComponentType.ONLINE_LEARNER
        assert AIComponentType.META_ALLOCATOR
        assert AIComponentType.SUPERVISED_MODEL

    def test_component_type_values(self):
        """Test component type values are strings."""
        assert AIComponentType.RL_AGENT.value == "rl_agent"
        assert AIComponentType.META_ALLOCATOR.value == "meta_allocator"


class TestLearningMode:
    """Test LearningMode enum."""

    def test_all_learning_modes_defined(self):
        """Test all learning modes exist."""
        assert LearningMode.DISABLED
        assert LearningMode.SHADOW
        assert LearningMode.VALIDATED
        assert LearningMode.LIVE

    def test_learning_mode_values(self):
        """Test learning mode values."""
        assert LearningMode.SHADOW.value == "shadow"
        assert LearningMode.VALIDATED.value == "validated"


class TestAIComponentConfig:
    """Test AIComponentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AIComponentConfig(component_type=AIComponentType.RL_AGENT)

        assert config.enabled is True
        assert config.learning_mode == LearningMode.SHADOW
        assert config.train_interval_hours == 24.0
        assert config.min_samples_to_train == 100
        assert config.require_walk_forward is True
        assert config.min_validation_score == 60.0
        assert config.min_shadow_days == 7

    def test_custom_config(self):
        """Test custom configuration."""
        config = AIComponentConfig(
            component_type=AIComponentType.META_ALLOCATOR,
            enabled=True,
            learning_mode=LearningMode.VALIDATED,
            train_interval_hours=12.0,
            min_samples_to_train=50,
        )

        assert config.train_interval_hours == 12.0
        assert config.min_samples_to_train == 50
        assert config.learning_mode == LearningMode.VALIDATED


class TestLearningEvent:
    """Test LearningEvent dataclass."""

    def test_event_creation(self):
        """Test learning event creation."""
        event = LearningEvent(
            component_type=AIComponentType.RL_AGENT,
            event_type="train_started",
            details={"num_samples": 100},
        )

        assert event.component_type == AIComponentType.RL_AGENT
        assert event.event_type == "train_started"
        assert event.success is True
        assert isinstance(event.timestamp, datetime)

    def test_event_to_dict(self):
        """Test event serialization."""
        event = LearningEvent(
            component_type=AIComponentType.META_ALLOCATOR,
            event_type="train_completed",
            success=True,
        )

        data = event.to_dict()

        assert data["component_type"] == "meta_allocator"
        assert data["event_type"] == "train_completed"
        assert data["success"] is True
        assert "timestamp" in data


class TestAIDecision:
    """Test AIDecision dataclass."""

    def test_decision_creation(self):
        """Test AI decision creation."""
        decision = AIDecision(
            component_type=AIComponentType.RL_AGENT,
            action="BUY",
            confidence=0.85,
            reasoning="Strong momentum signal",
        )

        assert decision.action == "BUY"
        assert decision.confidence == 0.85
        assert decision.validated is False

    def test_decision_to_dict(self):
        """Test decision serialization."""
        decision = AIDecision(
            component_type=AIComponentType.SUPERVISED_MODEL,
            action="SELL",
            confidence=0.7,
            reasoning="Overbought conditions",
            validated=True,
            validation_score=75.0,
        )

        data = decision.to_dict()

        assert data["action"] == "SELL"
        assert data["confidence"] == 0.7
        assert data["validated"] is True
        assert data["validation_score"] == 75.0


class TestAIIntegrationManager:
    """Test AIIntegrationManager class."""

    @pytest.fixture
    def manager(self):
        """Create AI integration manager."""
        return AIIntegrationManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert len(manager.component_configs) > 0
        assert len(manager.decision_weights) > 0
        assert manager.learning_events == []

    def test_default_configs_created(self, manager):
        """Test default configs are created for all components."""
        for component_type in AIComponentType:
            assert component_type in manager.component_configs

    def test_decision_weights_sum_to_one(self, manager):
        """Test decision weights roughly sum to 1."""
        total = sum(manager.decision_weights.values())
        assert 0.95 <= total <= 1.05

    def test_register_component(self, manager):
        """Test registering a component."""
        mock_component = MagicMock()

        manager.register_component(
            AIComponentType.RL_AGENT,
            mock_component,
        )

        assert AIComponentType.RL_AGENT in manager._components
        assert manager._components[AIComponentType.RL_AGENT] == mock_component

    def test_set_learning_mode(self, manager):
        """Test setting learning mode."""
        manager.set_learning_mode(
            AIComponentType.RL_AGENT,
            LearningMode.VALIDATED
        )

        config = manager.component_configs[AIComponentType.RL_AGENT]
        assert config.learning_mode == LearningMode.VALIDATED

    def test_get_decision_no_component(self, manager):
        """Test getting decision when component not registered."""
        decision = manager.get_decision(
            AIComponentType.RL_AGENT,
            {"symbol": "BTC/USDT", "price": 50000}
        )

        assert decision is None

    def test_get_decision_with_component(self, manager):
        """Test getting decision from registered component."""
        mock_component = MagicMock()
        mock_component.predict.return_value = {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Test",
        }

        manager.register_component(AIComponentType.RL_AGENT, mock_component)

        decision = manager.get_decision(
            AIComponentType.RL_AGENT,
            {"symbol": "BTC/USDT"}
        )

        assert decision is not None
        assert decision.action == "BUY"
        assert decision.confidence == 0.8

    def test_aggregate_decisions_empty(self, manager):
        """Test aggregating decisions with no components."""
        result = manager.aggregate_decisions({"symbol": "BTC/USDT"})

        assert result["consensus_action"] == "HOLD"
        assert result["consensus_confidence"] == 0.0
        assert len(result["decisions"]) == 0

    def test_aggregate_decisions_with_components(self, manager):
        """Test aggregating decisions from multiple components."""
        # Register components with different decisions
        mock_rl = MagicMock()
        mock_rl.predict.return_value = {"action": "BUY", "confidence": 0.8}

        mock_supervised = MagicMock()
        mock_supervised.predict.return_value = {"action": "BUY", "confidence": 0.7}

        manager.register_component(AIComponentType.RL_AGENT, mock_rl)
        manager.register_component(AIComponentType.SUPERVISED_MODEL, mock_supervised)

        # Set to validated mode so decisions are included
        manager.set_learning_mode(AIComponentType.RL_AGENT, LearningMode.VALIDATED)
        manager.set_learning_mode(AIComponentType.SUPERVISED_MODEL, LearningMode.VALIDATED)

        result = manager.aggregate_decisions({"symbol": "BTC/USDT"})

        assert result["consensus_action"] == "BUY"
        assert len(result["decisions"]) == 2

    def test_get_status(self, manager):
        """Test getting manager status."""
        status = manager.get_status()

        assert "components" in status
        assert "learning_events_recent" in status
        assert len(status["components"]) > 0


class TestAIIntegrationManagerAsync:
    """Test async methods of AIIntegrationManager."""

    @pytest.fixture
    def manager(self):
        """Create AI integration manager."""
        return AIIntegrationManager()

    @pytest.mark.asyncio
    async def test_check_and_train_disabled(self, manager):
        """Test training skipped when disabled."""
        manager.component_configs[AIComponentType.RL_AGENT].learning_mode = LearningMode.DISABLED

        result = await manager.check_and_train(
            AIComponentType.RL_AGENT,
            [{"data": 1}] * 100,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_check_and_train_insufficient_samples(self, manager):
        """Test training skipped with insufficient samples."""
        result = await manager.check_and_train(
            AIComponentType.RL_AGENT,
            [{"data": 1}] * 10,  # Less than min_samples_to_train
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_check_and_train_success(self, manager):
        """Test successful training."""
        mock_component = MagicMock()
        mock_component.train = AsyncMock(return_value={"accuracy": 0.7})

        manager.register_component(AIComponentType.RL_AGENT, mock_component)
        manager.component_configs[AIComponentType.RL_AGENT].min_samples_to_train = 10

        result = await manager.check_and_train(
            AIComponentType.RL_AGENT,
            [{"data": i} for i in range(20)],
            force=True,
        )

        assert result is not None
        assert result.event_type == "train_completed"


class TestFactoryFunctions:
    """Test factory functions."""

    def test_get_ai_integration(self):
        """Test getting global AI integration."""
        manager = get_ai_integration()
        assert manager is not None
        assert isinstance(manager, AIIntegrationManager)

    def test_create_ai_integration(self):
        """Test creating AI integration with dependencies."""
        mock_risk = MagicMock()
        mock_ledger = MagicMock()

        manager = create_ai_integration(
            risk_guardian=mock_risk,
            trade_ledger=mock_ledger,
        )

        assert manager.risk_guardian == mock_risk
        assert manager.trade_ledger == mock_ledger
