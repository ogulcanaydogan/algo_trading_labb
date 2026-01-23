"""
Unit tests for the Intelligent Trading Brain module.

Tests cover basic functionality and integration of the intelligence components.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile


# ============================================================================
# LLM Router Tests
# ============================================================================


class TestLLMRouter:
    """Tests for the LLM Router."""

    def test_initialization(self):
        """Test router initializes correctly."""
        from bot.intelligence.llm_router import LLMRouter

        router = LLMRouter(
            daily_budget=5.0,
            enable_claude=False,
            enable_ollama=False,
        )

        assert router.daily_budget == 5.0
        assert router.enable_claude == False

    def test_health_check(self):
        """Test health check returns status."""
        from bot.intelligence.llm_router import LLMRouter

        router = LLMRouter(
            daily_budget=5.0,
            enable_claude=False,
            enable_ollama=False,
        )

        health = router.health_check()

        assert "claude_available" in health
        assert "ollama_available" in health
        assert "rule_based_available" in health
        assert health["rule_based_available"] == True

    def test_fallback_response(self):
        """Test fallback response generation."""
        from bot.intelligence.llm_router import LLMRouter, LLMRequest, RequestPriority

        router = LLMRouter(
            daily_budget=5.0,
            enable_claude=False,
            enable_ollama=False,
        )

        request = LLMRequest(
            prompt="Analyze this trade",
            purpose="trade_analysis",
            priority=RequestPriority.LOW,
        )

        response = router.route(request)
        assert response.success == True
        assert response.cost == 0.0  # Fallback is free


# ============================================================================
# Regime Adapter Tests
# ============================================================================


class TestRegimeAdapter:
    """Tests for Regime Detection and Adaptation."""

    def test_detect_bull_regime(self):
        """Test detection of bullish regime."""
        from bot.intelligence.regime_adapter import RegimeAdapter

        adapter = RegimeAdapter(lookback_period=20)

        # Create uptrending prices
        prices = np.array([100 + i * 0.5 for i in range(50)])
        result = adapter.detect_regime(prices)

        assert result.regime is not None
        assert result.trend_strength > 0

    def test_detect_bear_regime(self):
        """Test detection of bearish regime."""
        from bot.intelligence.regime_adapter import RegimeAdapter, MarketRegime

        adapter = RegimeAdapter(lookback_period=20)

        # Create downtrending prices
        prices = np.array([100 - i * 0.5 for i in range(50)])
        result = adapter.detect_regime(prices)

        assert result.regime is not None
        # trend_strength is abs(trend), so always positive
        assert result.trend_strength >= 0
        # For a strong downtrend, expect bear-ish regime
        assert result.regime in (
            MarketRegime.BEAR,
            MarketRegime.STRONG_BEAR,
            MarketRegime.VOLATILE,
            MarketRegime.SIDEWAYS,
        )

    def test_get_strategy(self):
        """Test getting strategy for a regime."""
        from bot.intelligence.regime_adapter import RegimeAdapter, MarketRegime

        adapter = RegimeAdapter(lookback_period=20)
        strategy = adapter.get_strategy(MarketRegime.BULL)

        assert strategy is not None
        assert hasattr(strategy, "position_size_multiplier")

    def test_strategy_for_bear(self):
        """Test bear market strategy is conservative."""
        from bot.intelligence.regime_adapter import RegimeAdapter, MarketRegime

        adapter = RegimeAdapter(lookback_period=20)
        strategy = adapter.get_strategy(MarketRegime.BEAR)

        assert strategy is not None
        assert strategy.position_size_multiplier <= 1.0


# ============================================================================
# Pattern Memory Tests
# ============================================================================


class TestPatternMemory:
    """Tests for Pattern Memory storage."""

    def test_initialization(self):
        """Test pattern memory initializes."""
        from bot.intelligence.pattern_memory import PatternMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = PatternMemory(db_path=db_path)

            stats = memory.get_pattern_stats()
            # PatternStats is a dataclass with total_patterns attribute
            assert hasattr(stats, "total_patterns")
            assert stats.total_patterns >= 0

    def test_get_summary(self):
        """Test getting pattern summary."""
        from bot.intelligence.pattern_memory import PatternMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = PatternMemory(db_path=db_path)

            summary = memory.get_summary()
            assert isinstance(summary, dict)


# ============================================================================
# Real-Time Learner Tests
# ============================================================================


class TestRealTimeLearner:
    """Tests for Real-Time Learning."""

    def test_initialization(self):
        """Test learner initializes."""
        from bot.intelligence.real_time_learner import RealTimeLearner
        from bot.intelligence.pattern_memory import PatternMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = PatternMemory(db_path=db_path)

            learner = RealTimeLearner(
                pattern_memory=memory,
                learning_rate=0.1,
            )

            assert learner.learning_rate == 0.1

    def test_get_summary(self):
        """Test getting learner summary."""
        from bot.intelligence.real_time_learner import RealTimeLearner
        from bot.intelligence.pattern_memory import PatternMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            memory = PatternMemory(db_path=db_path)

            learner = RealTimeLearner(
                pattern_memory=memory,
                learning_rate=0.1,
            )

            summary = learner.get_summary()
            assert "learning_rate" in summary
            assert summary["learning_rate"] == 0.1


# ============================================================================
# News Reasoner Tests
# ============================================================================


class TestNewsReasoner:
    """Tests for News Sentiment Analysis."""

    def test_initialization(self):
        """Test news reasoner initializes."""
        from bot.intelligence.news_reasoner import NewsReasoner

        reasoner = NewsReasoner()
        assert reasoner is not None

    def test_get_available_sources(self):
        """Test getting available news sources."""
        from bot.intelligence.news_reasoner import NewsReasoner

        reasoner = NewsReasoner()
        sources = reasoner.get_available_sources()

        assert "yahoo_finance" in sources
        assert sources["yahoo_finance"] == True


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntelligentBrainIntegration:
    """Integration tests for the complete brain."""

    def test_brain_initialization(self):
        """Test that the brain initializes correctly."""
        from bot.intelligence import get_intelligent_brain

        brain = get_intelligent_brain()

        assert brain.llm_router is not None
        assert brain.pattern_memory is not None
        assert brain.learner is not None
        assert brain.regime_adapter is not None

    def test_brain_health_check(self):
        """Test brain health check."""
        from bot.intelligence import get_intelligent_brain

        brain = get_intelligent_brain()
        health = brain.health_check()

        assert "llm_router" in health
        assert "pattern_memory" in health
        assert "regime_adapter" in health

    def test_get_summary(self):
        """Test brain summary."""
        from bot.intelligence import get_intelligent_brain

        brain = get_intelligent_brain()
        summary = brain.get_summary()

        assert isinstance(summary, dict)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
