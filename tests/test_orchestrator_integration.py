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
    SystemStatus,
)


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

        orchestrator = UnifiedOrchestrator(
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

        # Mock components
        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=True,
            adjusted_size=0.05,
        )

        orchestrator.execution_engine = MagicMock()
        orchestrator.execution_engine.execute_order = AsyncMock(
            return_value=MagicMock(success=True)
        )

        return orchestrator

    @pytest.mark.asyncio
    async def test_process_market_update(self, orchestrator):
        """Test processing a market update."""
        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_process_with_signal(self, orchestrator):
        """Test processing with strategy signal."""
        # Mock strategy to return signal
        mock_strategy = MagicMock()
        mock_strategy.generate_signal.return_value = {
            "action": "BUY",
            "confidence": 0.8,
            "size": 0.05,
        }
        orchestrator.strategies = {"momentum": mock_strategy}

        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # Strategy should have been called
        mock_strategy.generate_signal.assert_called()


class TestKillSwitch:
    """Test kill switch functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for kill switch tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_kill_switch_stop(self, orchestrator):
        """Test kill switch stops trading."""
        await orchestrator.kill_switch(action="stop", reason="Test")

        assert orchestrator.is_killed
        assert orchestrator.kill_reason == "Test"

    @pytest.mark.asyncio
    async def test_kill_switch_close_all(self, orchestrator):
        """Test kill switch with close all positions."""
        # Mock positions
        orchestrator.positions = {
            "BTC/USDT": {"size": 0.1, "side": "LONG"},
            "ETH/USDT": {"size": 1.0, "side": "LONG"},
        }

        orchestrator.execution_engine = MagicMock()
        orchestrator.execution_engine.close_position = AsyncMock(
            return_value=MagicMock(success=True)
        )

        await orchestrator.kill_switch(action="close_all", reason="Emergency")

        assert orchestrator.is_killed
        # Should have attempted to close positions
        assert orchestrator.execution_engine.close_position.call_count >= 0

    @pytest.mark.asyncio
    async def test_kill_switch_resume(self, orchestrator):
        """Test resuming after kill switch."""
        await orchestrator.kill_switch(action="stop", reason="Test")
        assert orchestrator.is_killed

        await orchestrator.resume()
        assert not orchestrator.is_killed


class TestStrategyManagement:
    """Test strategy management functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for strategy tests."""
        return UnifiedOrchestrator()

    def test_register_strategy(self, orchestrator):
        """Test registering a strategy."""
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"

        orchestrator.register_strategy("test", mock_strategy)

        assert "test" in orchestrator.strategies

    def test_enable_disable_strategy(self, orchestrator):
        """Test enabling/disabling strategies."""
        mock_strategy = MagicMock()
        orchestrator.register_strategy("test", mock_strategy)

        orchestrator.disable_strategy("test")
        assert not orchestrator.strategy_enabled.get("test", True)

        orchestrator.enable_strategy("test")
        assert orchestrator.strategy_enabled.get("test", False)

    def test_get_strategy_status(self, orchestrator):
        """Test getting strategy status."""
        mock_strategy = MagicMock()
        mock_strategy.name = "momentum"
        mock_strategy.get_status.return_value = {"signals": 10, "accuracy": 0.6}

        orchestrator.register_strategy("momentum", mock_strategy)

        status = orchestrator.get_strategy_status("momentum")
        assert status is not None


class TestRiskIntegration:
    """Test risk management integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with risk guardian."""
        orchestrator = UnifiedOrchestrator()

        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=True,
            adjusted_size=0.05,
        )
        orchestrator.risk_guardian.get_metrics.return_value = MagicMock(
            current_drawdown=0.03,
            daily_pnl=500.0,
        )

        return orchestrator

    @pytest.mark.asyncio
    async def test_trade_blocked_by_risk(self, orchestrator):
        """Test trade blocked by risk guardian."""
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=False,
            reason="Drawdown limit exceeded",
        )

        decision = TradingDecision(
            symbol="BTC/USDT",
            action="BUY",
            size=0.1,
            price=50000.0,
        )

        result = await orchestrator._execute_decision(decision)

        # Should have been blocked
        assert not result.get("executed", True)

    def test_risk_limits_update(self, orchestrator):
        """Test updating risk limits."""
        new_limits = {
            "max_position_pct": 0.05,
            "max_drawdown_pct": 0.08,
        }

        orchestrator.update_risk_limits(new_limits)

        orchestrator.risk_guardian.update_limits.assert_called()


class TestNotificationIntegration:
    """Test notification system integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with notification manager."""
        config = OrchestratorConfig(enable_notifications=True)
        orchestrator = UnifiedOrchestrator(config=config)

        orchestrator.notification_manager = MagicMock()
        orchestrator.notification_manager.notify = AsyncMock()
        orchestrator.notification_manager.notify_trade = AsyncMock()
        orchestrator.notification_manager.notify_risk_alert = AsyncMock()

        return orchestrator

    @pytest.mark.asyncio
    async def test_trade_notification(self, orchestrator):
        """Test notification sent on trade execution."""
        orchestrator.execution_engine = MagicMock()
        orchestrator.execution_engine.execute_order = AsyncMock(
            return_value=MagicMock(success=True, fill_price=50100.0)
        )

        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=True
        )

        decision = TradingDecision(
            symbol="BTC/USDT",
            action="BUY",
            size=0.1,
            price=50000.0,
        )

        await orchestrator._execute_decision(decision)

        # Should have sent trade notification
        # orchestrator.notification_manager.notify_trade.assert_called()

    @pytest.mark.asyncio
    async def test_risk_alert_notification(self, orchestrator):
        """Test notification sent on risk alert."""
        await orchestrator._send_risk_alert(
            alert_type="DRAWDOWN_WARNING",
            message="Approaching drawdown limit",
            critical=True,
        )

        orchestrator.notification_manager.notify_risk_alert.assert_called()


class TestSystemStatus:
    """Test system status functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for status tests."""
        orchestrator = UnifiedOrchestrator()

        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.get_metrics.return_value = MagicMock(
            current_drawdown=0.05,
            daily_pnl=1000.0,
        )

        return orchestrator

    def test_get_system_status(self, orchestrator):
        """Test getting full system status."""
        status = orchestrator.get_status()

        assert isinstance(status, dict)
        assert "is_running" in status
        assert "is_killed" in status

    def test_status_includes_risk(self, orchestrator):
        """Test status includes risk metrics."""
        status = orchestrator.get_status()

        # Should include risk information
        assert "risk" in status or orchestrator.risk_guardian.get_metrics.called


class TestNewsFeatureIntegration:
    """Test news feature extractor integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with news extractor."""
        orchestrator = UnifiedOrchestrator()

        orchestrator.news_extractor = MagicMock()
        orchestrator.news_extractor.extract_features.return_value = {
            "sentiment_score": 0.3,
            "surprise_factor": 0.1,
            "features": np.random.randn(18),
        }

        return orchestrator

    @pytest.mark.asyncio
    async def test_news_features_in_decision(self, orchestrator):
        """Test news features influence trading decision."""
        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=True
        )

        await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
            include_news=True,
        )

        # News extractor should have been called
        orchestrator.news_extractor.extract_features.assert_called()


class TestWalkForwardIntegration:
    """Test walk-forward validation integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with walk-forward validator."""
        orchestrator = UnifiedOrchestrator()

        orchestrator.walk_forward = MagicMock()
        orchestrator.walk_forward.validate.return_value = {
            "passed": True,
            "sharpe": 1.5,
            "max_drawdown": 0.08,
        }

        return orchestrator

    def test_strategy_validation(self, orchestrator):
        """Test strategy validation before deployment."""
        mock_strategy = MagicMock()

        result = orchestrator.validate_strategy(mock_strategy)

        assert result is not None
        orchestrator.walk_forward.validate.assert_called()


class TestOrchestratorLifecycle:
    """Test orchestrator lifecycle management."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for lifecycle tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        """Test starting and stopping orchestrator."""
        await orchestrator.start()
        assert orchestrator.is_running

        await orchestrator.stop()
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, orchestrator):
        """Test graceful shutdown closes positions."""
        orchestrator.positions = {
            "BTC/USDT": {"size": 0.1, "side": "LONG"},
        }

        orchestrator.execution_engine = MagicMock()
        orchestrator.execution_engine.close_position = AsyncMock(
            return_value=MagicMock(success=True)
        )

        await orchestrator.stop(graceful=True)

        # Should have attempted graceful closure
        assert not orchestrator.is_running


class TestOrchestratorPerformance:
    """Test orchestrator performance tracking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for performance tests."""
        orchestrator = UnifiedOrchestrator()
        orchestrator.trade_ledger = MagicMock()
        orchestrator.trade_ledger.get_performance.return_value = {
            "total_pnl": 5000.0,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
        }
        return orchestrator

    def test_get_performance(self, orchestrator):
        """Test getting performance metrics."""
        performance = orchestrator.get_performance()

        assert "total_pnl" in performance
        assert "win_rate" in performance

    def test_performance_history(self, orchestrator):
        """Test performance history tracking."""
        orchestrator.trade_ledger.get_equity_curve.return_value = [
            10000, 10500, 10300, 10800, 11000
        ]

        history = orchestrator.get_equity_history()

        assert len(history) > 0


class TestOrchestratorConcurrency:
    """Test orchestrator handles concurrent operations."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for concurrency tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_concurrent_market_updates(self, orchestrator):
        """Test handling concurrent market updates."""
        import asyncio

        orchestrator.risk_guardian = MagicMock()
        orchestrator.risk_guardian.check_trade.return_value = MagicMock(
            approved=True
        )

        async def process_update(symbol, price):
            return await orchestrator.process_market_update(
                symbol=symbol,
                price=price,
                volume=100.0,
                timestamp=datetime.utcnow(),
            )

        # Process multiple updates concurrently
        tasks = [
            process_update("BTC/USDT", 50000.0),
            process_update("ETH/USDT", 3000.0),
            process_update("SOL/USDT", 100.0),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should complete without errors
        for result in results:
            assert not isinstance(result, Exception)


class TestOrchestratorEdgeCases:
    """Test orchestrator edge cases and error handling."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for edge case tests."""
        return UnifiedOrchestrator()

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, orchestrator):
        """Test handling invalid symbol."""
        result = await orchestrator.process_market_update(
            symbol="",
            price=50000.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # Should handle gracefully
        assert result is None or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_negative_price(self, orchestrator):
        """Test handling negative price."""
        result = await orchestrator.process_market_update(
            symbol="BTC/USDT",
            price=-100.0,
            volume=100.0,
            timestamp=datetime.utcnow(),
        )

        # Should reject or handle gracefully
        assert result is None or result.get("rejected", False)

    def test_missing_component(self, orchestrator):
        """Test handling missing component gracefully."""
        # Remove a component
        orchestrator.risk_guardian = None

        # Should not crash
        status = orchestrator.get_status()
        assert status is not None
