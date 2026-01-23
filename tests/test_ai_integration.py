"""
Tests for AI Integration Module.

Tests the AIOrchestrator and EnhancedSignal classes.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from bot.ai_integration import (
    EnhancedSignal,
    AIOrchestrator,
    get_ai_orchestrator,
)


class TestEnhancedSignal:
    """Test EnhancedSignal dataclass."""

    def test_enhanced_signal_creation(self):
        """Test creating an EnhancedSignal."""
        signal = EnhancedSignal(
            symbol="BTC_USDT",
            timestamp=datetime.now(),
            action="BUY",
            confidence=0.8,
            position_size_pct=0.5,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop=True,
            dca_enabled=False,
            regime="bull",
            trend="uptrend",
            volatility="medium",
            sentiment_score=0.3,
            fear_greed=45.0,
            reason="ML: BUY (80%)",
            ml_action="BUY",
            rl_action="BUY",
            intelligence_bias="bullish",
            ml_confidence=0.8,
            rl_confidence=0.75,
            sentiment_confidence=0.6,
        )
        assert signal.symbol == "BTC_USDT"
        assert signal.action == "BUY"
        assert signal.confidence == 0.8

    def test_enhanced_signal_to_dict(self):
        """Test EnhancedSignal.to_dict() method."""
        ts = datetime.now()
        signal = EnhancedSignal(
            symbol="ETH_USDT",
            timestamp=ts,
            action="SELL",
            confidence=0.7,
            position_size_pct=0.3,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            trailing_stop=False,
            dca_enabled=True,
            regime="bear",
            trend="downtrend",
            volatility="high",
            sentiment_score=-0.2,
            fear_greed=30.0,
            reason="ML: SELL (70%)",
            ml_action="SELL",
            rl_action="HOLD",
            intelligence_bias="bearish",
            ml_confidence=0.7,
            rl_confidence=0.5,
            sentiment_confidence=0.55,
        )
        result = signal.to_dict()
        assert isinstance(result, dict)
        assert result["symbol"] == "ETH_USDT"
        assert result["action"] == "SELL"
        assert result["timestamp"] == ts.isoformat()


class TestAIOrchestrator:
    """Test AIOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test AIOrchestrator initialization."""
        orchestrator = AIOrchestrator(enable_rl=True, enable_intelligence=True)
        assert orchestrator.enable_rl is True
        assert orchestrator.enable_intelligence is True
        assert orchestrator.weights == {"ml": 0.4, "rl": 0.3, "sentiment": 0.15, "regime": 0.15}

    def test_orchestrator_initialization_disabled(self):
        """Test AIOrchestrator with features disabled."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)
        assert orchestrator.enable_rl is False
        assert orchestrator.enable_intelligence is False

    def test_combine_signals_buy(self):
        """Test signal combination resulting in BUY."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)
        action, confidence = orchestrator._combine_signals(
            ml_action="BUY",
            ml_confidence=0.8,
            rl_action="BUY",
            rl_confidence=0.75,
            sentiment_score=0.3,
            fear_greed=50,
            regime="bull",
        )
        assert action == "BUY"
        assert confidence > 0.5

    def test_combine_signals_sell(self):
        """Test signal combination resulting in SELL."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)
        action, confidence = orchestrator._combine_signals(
            ml_action="SELL",
            ml_confidence=0.8,
            rl_action="SELL",
            rl_confidence=0.75,
            sentiment_score=-0.3,
            fear_greed=80,
            regime="bear",
        )
        assert action == "SELL"
        assert confidence > 0.5

    def test_combine_signals_hold(self):
        """Test signal combination resulting in HOLD."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)
        action, confidence = orchestrator._combine_signals(
            ml_action="HOLD",
            ml_confidence=0.5,
            rl_action="HOLD",
            rl_confidence=0.5,
            sentiment_score=0.0,
            fear_greed=50,
            regime="neutral",
        )
        assert action == "HOLD"

    def test_combine_signals_conflict_penalty(self):
        """Test that conflicting signals reduce confidence."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)

        # Agreeing signals
        _, conf_agree = orchestrator._combine_signals(
            ml_action="BUY",
            ml_confidence=0.7,
            rl_action="BUY",
            rl_confidence=0.7,
            sentiment_score=0.0,
            fear_greed=50,
            regime="neutral",
        )

        # Conflicting signals
        _, conf_conflict = orchestrator._combine_signals(
            ml_action="BUY",
            ml_confidence=0.7,
            rl_action="SELL",
            rl_confidence=0.7,
            sentiment_score=0.0,
            fear_greed=50,
            regime="neutral",
        )

        # Conflicting signals should have lower confidence
        assert conf_conflict < conf_agree

    def test_combine_signals_crash_regime(self):
        """Test that crash regime reduces BUY confidence."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)

        # Normal regime
        _, conf_normal = orchestrator._combine_signals(
            ml_action="BUY",
            ml_confidence=0.8,
            rl_action="BUY",
            rl_confidence=0.8,
            sentiment_score=0.3,
            fear_greed=50,
            regime="bull",
        )

        # Crash regime
        _, conf_crash = orchestrator._combine_signals(
            ml_action="BUY",
            ml_confidence=0.8,
            rl_action="BUY",
            rl_confidence=0.8,
            sentiment_score=0.3,
            fear_greed=50,
            regime="crash",
        )

        # Crash regime should reduce BUY confidence
        assert conf_crash < conf_normal

    def test_get_system_status(self):
        """Test get_system_status method."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)
        status = orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert "weights" in status
        assert "ai_brain_active" in status
        assert "data_intelligence_active" in status
        assert "continuous_learner_active" in status

    @pytest.mark.asyncio
    async def test_generate_enhanced_signal(self):
        """Test generate_enhanced_signal method."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)

        ml_signal = {
            "action": "BUY",
            "confidence": 0.75,
            "signal_meta": {
                "regime": "bull",
                "trend": "uptrend",
                "volatility": "medium",
                "regime_strategy": {
                    "position_size_multiplier": 0.8,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.04,
                },
            },
        }
        market_data = {"price": 50000, "volume": 1000}

        signal = await orchestrator.generate_enhanced_signal("BTC_USDT", ml_signal, market_data)

        assert isinstance(signal, EnhancedSignal)
        assert signal.symbol == "BTC_USDT"
        assert signal.ml_action == "BUY"
        assert signal.ml_confidence == 0.75

    def test_record_trade_outcome(self):
        """Test record_trade_outcome method doesn't raise."""
        orchestrator = AIOrchestrator(enable_rl=False, enable_intelligence=False)

        # Should not raise even without AI brain/learner
        orchestrator.record_trade_outcome(
            symbol="BTC_USDT",
            action="BUY",
            entry_price=50000,
            exit_price=51000,
            pnl=100,
            pnl_pct=0.02,
            hold_time_minutes=60,
            was_stopped=False,
            was_target_hit=True,
            features=None,
        )


class TestGetAIOrchestrator:
    """Test get_ai_orchestrator factory function."""

    def test_get_ai_orchestrator_singleton(self):
        """Test that get_ai_orchestrator returns singleton."""
        # Reset the global
        import bot.ai_integration as ai_module

        ai_module._ai_orchestrator = None

        orch1 = get_ai_orchestrator(enable_rl=False, enable_intelligence=False)
        orch2 = get_ai_orchestrator()

        assert orch1 is orch2

    def test_get_ai_orchestrator_creates_instance(self):
        """Test that get_ai_orchestrator creates an instance."""
        import bot.ai_integration as ai_module

        ai_module._ai_orchestrator = None

        orchestrator = get_ai_orchestrator(enable_rl=False, enable_intelligence=False)

        assert isinstance(orchestrator, AIOrchestrator)
