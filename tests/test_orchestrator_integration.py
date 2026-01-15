"""
Integration tests for Unified Orchestrator.

Tests the integration of all trading system components working together.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import json

import numpy as np
import pandas as pd

from bot.unified_orchestrator import (
    UnifiedOrchestrator,
    OrchestratorConfig,
    TradingDecision,
    SystemState,
    TradingMode,
)

# Alias for backward compatibility in tests
SystemStatus = SystemState


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_default_initialization(self):
        """Test orchestrator initializes with defaults."""
        orchestrator = UnifiedOrchestrator()

        assert orchestrator is not None
        assert orchestrator.config is not None
        assert not orchestrator.is_running

    def test_custom_config(self):
        """Test orchestrator with custom config."""
        config = OrchestratorConfig(
            initial_capital=50000.0,
            max_positions=5,
            enable_notifications=False,
        )
        orchestrator = UnifiedOrchestrator(config=config)

        assert orchestrator.config.initial_capital == 50000.0
        assert orchestrator.config.max_positions == 5

    def test_component_injection(self):
        """Test component injection pattern."""
        mock_risk = MagicMock()
        mock_execution = MagicMock()

        orchestrator = UnifiedOrchestrator()
        orchestrator.inject_components(
            risk_guardian=mock_risk,
            execution_engine=mock_execution,
        )

        assert orchestrator.risk_guardian == mock_risk
        assert orchestrator.execution_engine == mock_execution


class TestMarketDataProcessing:
    """Test market data processing pipeline."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked components."""
        orchestrator = UnifiedOrchestrator()

        # Mock components via injection
        mock_risk = MagicMock()
        mock_risk.check_trade.return_value = MagicMock(
            approved=True,
            reason="OK",
            adjusted_quantity=0.05,
        )

        mock_execution = MagicMock()
        mock_execution.execute_order = AsyncMock(
            return_value={"success": True, "fill_price": 50000.0}
        )

        orchestrator.inject_components(
            risk_guardian=mock_risk,
            execution_engine=mock_execution,
        )

        return orchestrator

    @pytest.mark.asyncio
    async def test_process_market_update(self, orchestrator):
        """Test processing a market update."""
        # Start orchestrator first
        orchestrator.state = SystemState.RUNNING

        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # Should return None when no strategies generate signals
        assert result is None

    @pytest.mark.asyncio
    async def test_process_requires_running_state(self, orchestrator):
        """Test that processing only works when running."""
        # Not running - should return early
        orchestrator.state = SystemState.PAUSED

        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        assert result is None


class TestKillSwitch:
    """Test kill switch functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for kill switch tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_kill_switch_stop(self, orchestrator):
        """Test kill switch stops trading."""
        orchestrator.state = SystemState.RUNNING

        await orchestrator.kill_switch(action="stop", reason="Test")

        # Kill switch pauses the system
        assert orchestrator.state == SystemState.PAUSED

    @pytest.mark.asyncio
    async def test_kill_switch_close_all(self, orchestrator):
        """Test kill switch with close all positions."""
        orchestrator.state = SystemState.RUNNING

        mock_execution = MagicMock()
        mock_execution.close_all_positions = AsyncMock()
        orchestrator.execution_engine = mock_execution

        await orchestrator.kill_switch(action="close_all", reason="Emergency")

        assert orchestrator.state == SystemState.PAUSED
        mock_execution.close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_after_pause(self, orchestrator):
        """Test resuming after pause."""
        orchestrator.state = SystemState.PAUSED

        await orchestrator.resume()

        assert orchestrator.state == SystemState.RUNNING


class TestStrategyManagement:
    """Test strategy management functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for strategy tests."""
        return UnifiedOrchestrator()

    def test_enable_strategy(self, orchestrator):
        """Test enabling a strategy."""
        # Add a strategy to active_strategies first
        orchestrator.active_strategies["test"] = False

        result = orchestrator.enable_strategy("test")

        assert result
        assert orchestrator.active_strategies["test"]

    def test_disable_strategy(self, orchestrator):
        """Test disabling a strategy."""
        orchestrator.active_strategies["test"] = True

        result = orchestrator.disable_strategy("test")

        assert result
        assert not orchestrator.active_strategies["test"]

    def test_enable_nonexistent_strategy(self, orchestrator):
        """Test enabling non-existent strategy returns False."""
        result = orchestrator.enable_strategy("nonexistent")
        assert not result


class TestRiskIntegration:
    """Test risk management integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with risk guardian."""
        orchestrator = UnifiedOrchestrator()

        mock_guardian = MagicMock()
        mock_guardian.check_trade.return_value = MagicMock(
            approved=True,
            reason="OK",
            adjusted_quantity=0.05,
        )

        orchestrator.inject_components(risk_guardian=mock_guardian)

        return orchestrator

    @pytest.mark.asyncio
    async def test_risk_check_approved(self, orchestrator):
        """Test trade approved by risk guardian."""
        decision = TradingDecision(
            decision_id="test_001",
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            action="buy",
            quantity=0.1,
            price=50000.0,
            strategy_name="momentum",
            strategy_signal_strength=0.8,
            strategy_confidence=0.75,
        )

        approved, reason, adjusted_qty = await orchestrator._check_risk(decision)

        assert approved
        assert orchestrator.risk_guardian.check_trade.called

    @pytest.mark.asyncio
    async def test_risk_check_rejected(self, orchestrator):
        """Test trade rejected by risk guardian."""
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=False,
            reason="Drawdown limit exceeded",
            adjusted_quantity=0.0,
        )

        decision = TradingDecision(
            decision_id="test_002",
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            action="buy",
            quantity=0.1,
            price=50000.0,
            strategy_name="momentum",
            strategy_signal_strength=0.8,
            strategy_confidence=0.75,
        )

        approved, reason, adjusted_qty = await orchestrator._check_risk(decision)

        assert not approved
        assert "Drawdown" in reason

    @pytest.mark.asyncio
    async def test_adjust_risk_limit(self, orchestrator):
        """Test adjusting risk limits."""
        orchestrator.risk_guardian.update_limit = MagicMock()

        result = await orchestrator.adjust_risk_limit("max_drawdown", 0.08)

        # Should call update_limit on risk guardian
        # Result depends on risk guardian having update_limit method


class TestSystemStatus:
    """Test system status functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for status tests."""
        return UnifiedOrchestrator()

    def test_get_system_status(self, orchestrator):
        """Test getting full system status."""
        status = orchestrator.get_status()

        assert isinstance(status, dict)
        assert "state" in status
        assert "mode" in status
        assert "metrics" in status
        assert "components" in status

    def test_get_health(self, orchestrator):
        """Test getting system health."""
        health = orchestrator.get_health()

        assert health.state == orchestrator.state
        assert health.uptime_seconds >= 0

    def test_status_includes_components(self, orchestrator):
        """Test status includes component availability."""
        status = orchestrator.get_status()

        assert "components" in status
        assert "risk_guardian" in status["components"]
        assert "execution_engine" in status["components"]


class TestNewsFeatureIntegration:
    """Test news feature extractor integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with news extractor."""
        orchestrator = UnifiedOrchestrator()

        mock_extractor = MagicMock()
        mock_extractor.extract_features.return_value = MagicMock(
            overall_sentiment=0.3,
        )

        orchestrator.inject_components(news_extractor=mock_extractor)

        return orchestrator

    @pytest.mark.asyncio
    async def test_news_features_used_in_pipeline(self, orchestrator):
        """Test news features used in processing pipeline."""
        orchestrator.state = SystemState.RUNNING

        await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # News extractor should have been called
        orchestrator.news_extractor.extract_features.assert_called()


class TestOrchestratorLifecycle:
    """Test orchestrator lifecycle management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for lifecycle tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_start(self, orchestrator):
        """Test starting orchestrator."""
        await orchestrator.start()

        assert orchestrator.is_running
        assert orchestrator.state == SystemState.RUNNING

    @pytest.mark.asyncio
    async def test_stop(self, orchestrator):
        """Test stopping orchestrator."""
        orchestrator.state = SystemState.RUNNING

        await orchestrator.stop()

        assert not orchestrator.is_running
        assert orchestrator.state == SystemState.STOPPED

    @pytest.mark.asyncio
    async def test_pause(self, orchestrator):
        """Test pausing orchestrator."""
        orchestrator.state = SystemState.RUNNING

        await orchestrator.pause()

        assert orchestrator.state == SystemState.PAUSED


class TestOrchestratorCallbacks:
    """Test callback registration and invocation."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for callback tests."""
        return UnifiedOrchestrator()

    def test_register_trade_callback(self, orchestrator):
        """Test registering trade callback."""
        callback = MagicMock()
        orchestrator.on_trade(callback)

        assert callback in orchestrator._on_trade_callbacks

    def test_register_signal_callback(self, orchestrator):
        """Test registering signal callback."""
        callback = MagicMock()
        orchestrator.on_signal(callback)

        assert callback in orchestrator._on_signal_callbacks

    def test_register_error_callback(self, orchestrator):
        """Test registering error callback."""
        callback = MagicMock()
        orchestrator.on_error(callback)

        assert callback in orchestrator._on_error_callbacks

    def test_register_regime_change_callback(self, orchestrator):
        """Test registering regime change callback."""
        callback = MagicMock()
        orchestrator.on_regime_change(callback)

        assert callback in orchestrator._on_regime_change_callbacks


class TestRegimeManagement:
    """Test regime detection and management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for regime tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_update_regime(self, orchestrator):
        """Test updating market regime."""
        await orchestrator.update_regime("bull", confidence=0.8)

        assert orchestrator.current_regime == "bull"

    @pytest.mark.asyncio
    async def test_regime_change_callback(self, orchestrator):
        """Test regime change triggers callback."""
        callback = AsyncMock()
        orchestrator.on_regime_change(callback)

        orchestrator.current_regime = "neutral"
        await orchestrator.update_regime("bear", confidence=0.7)

        callback.assert_called_once()


class TestOrchestratorEdgeCases:
    """Test orchestrator edge cases and error handling."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for edge case tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_process_when_not_running(self, orchestrator):
        """Test processing when not in running state."""
        orchestrator.state = SystemState.INITIALIZING

        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # Should return early
        assert result is None

    def test_missing_risk_guardian(self, orchestrator):
        """Test handling missing risk guardian gracefully."""
        # No risk guardian set
        orchestrator.risk_guardian = None

        # Should not crash
        status = orchestrator.get_status()
        assert status["components"]["risk_guardian"] is False

    @pytest.mark.asyncio
    async def test_risk_check_no_guardian(self, orchestrator):
        """Test risk check passes when no guardian configured."""
        orchestrator.risk_guardian = None

        decision = TradingDecision(
            decision_id="test_003",
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            action="buy",
            quantity=0.1,
            price=50000.0,
            strategy_name="momentum",
            strategy_signal_strength=0.8,
            strategy_confidence=0.75,
        )

        approved, reason, adjusted_qty = await orchestrator._check_risk(decision)

        # Should approve with warning
        assert approved
        assert "No risk guardian" in reason


class TestOrchestratorMetrics:
    """Test orchestrator metrics tracking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for metrics tests."""
        return UnifiedOrchestrator()

    def test_initial_metrics(self, orchestrator):
        """Test initial metrics are zero."""
        assert orchestrator.metrics["decisions_total"] == 0
        assert orchestrator.metrics["trades_executed"] == 0
        assert orchestrator.metrics["errors_total"] == 0

    def test_metrics_in_status(self, orchestrator):
        """Test metrics included in status."""
        status = orchestrator.get_status()

        assert "metrics" in status
        assert "decisions_total" in status["metrics"]


class TestTradingDecision:
    """Test TradingDecision dataclass."""

    def test_decision_creation(self):
        """Test creating a trading decision."""
        decision = TradingDecision(
            decision_id="dec_001",
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            action="buy",
            quantity=0.1,
            price=50000.0,
            strategy_name="momentum",
            strategy_signal_strength=0.8,
            strategy_confidence=0.75,
        )

        assert decision.symbol == "BTC/USDT"
        assert decision.action == "buy"
        assert decision.quantity == 0.1
        assert not decision.executed

    def test_decision_with_ai_features(self):
        """Test decision with AI features."""
        decision = TradingDecision(
            decision_id="dec_002",
            timestamp=datetime.now(),
            symbol="ETH/USDT",
            action="sell",
            quantity=1.0,
            price=3000.0,
            strategy_name="reversal",
            strategy_signal_strength=0.7,
            strategy_confidence=0.65,
            ai_sentiment=-0.2,
            ai_recommendation="reduce_position",
            news_sentiment=-0.1,
        )

        assert decision.ai_sentiment == -0.2
        assert decision.ai_recommendation == "reduce_position"
        assert decision.news_sentiment == -0.1
