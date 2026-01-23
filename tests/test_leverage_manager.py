"""
Tests for AI Leverage Manager module.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from bot.ai_engine.leverage_manager import (
    LeverageDecision,
    MarginStatus,
    AILeverageManager,
    get_leverage_manager,
)


class TestLeverageDecision:
    """Test LeverageDecision dataclass."""

    def test_basic_creation(self):
        """Test creating leverage decision."""
        decision = LeverageDecision(
            recommended_leverage=5.0,
            direction="long",
            position_size_pct=0.1,
            confidence=0.75,
            reasoning="Strong uptrend detected",
            risk_metrics={"max_loss_pct": 2.5},
            warnings=["High volatility"],
        )

        assert decision.recommended_leverage == 5.0
        assert decision.direction == "long"
        assert decision.position_size_pct == 0.1
        assert decision.confidence == 0.75
        assert "uptrend" in decision.reasoning
        assert decision.risk_metrics["max_loss_pct"] == 2.5
        assert len(decision.warnings) == 1

    def test_short_decision(self):
        """Test short decision."""
        decision = LeverageDecision(
            recommended_leverage=3.0,
            direction="short",
            position_size_pct=0.08,
            confidence=0.65,
            reasoning="Bearish reversal pattern",
            risk_metrics={},
            warnings=[],
        )

        assert decision.direction == "short"
        assert decision.recommended_leverage == 3.0

    def test_close_decision(self):
        """Test close position decision."""
        decision = LeverageDecision(
            recommended_leverage=0.0,
            direction="close",
            position_size_pct=0.0,
            confidence=0.8,
            reasoning="Exit signal triggered",
            risk_metrics={},
            warnings=[],
        )

        assert decision.direction == "close"
        assert decision.position_size_pct == 0.0


class TestMarginStatus:
    """Test MarginStatus dataclass."""

    def test_basic_creation(self):
        """Test creating margin status."""
        status = MarginStatus(
            total_margin=10000.0,
            used_margin=2000.0,
            available_margin=8000.0,
            margin_ratio=0.2,
            liquidation_price=45000.0,
            distance_to_liquidation=15.5,
            unrealized_pnl=150.0,
        )

        assert status.total_margin == 10000.0
        assert status.used_margin == 2000.0
        assert status.available_margin == 8000.0
        assert status.margin_ratio == 0.2
        assert status.liquidation_price == 45000.0
        assert status.distance_to_liquidation == 15.5
        assert status.unrealized_pnl == 150.0

    def test_no_position(self):
        """Test margin status with no position."""
        status = MarginStatus(
            total_margin=5000.0,
            used_margin=0.0,
            available_margin=5000.0,
            margin_ratio=0.0,
            liquidation_price=None,
            distance_to_liquidation=100.0,
            unrealized_pnl=0.0,
        )

        assert status.used_margin == 0.0
        assert status.liquidation_price is None
        assert status.distance_to_liquidation == 100.0


class TestAILeverageManager:
    """Test AILeverageManager class."""

    @pytest.fixture
    def mock_rl_agent(self):
        """Create mock RL agent."""
        agent = MagicMock()
        agent.get_action_analysis.return_value = {
            "best_action": MagicMock(),
            "confidence": 0.65,
            "q_values": [0.1, 0.2, 0.3],
        }
        return agent

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = MagicMock()
        return db

    @pytest.fixture
    def manager(self, mock_rl_agent, mock_db):
        """Create leverage manager with mocked dependencies."""
        return AILeverageManager(
            rl_agent=mock_rl_agent,
            db=mock_db,
            max_leverage=10.0,
            min_leverage=1.0,
            base_position_size=0.1,
            max_position_size=0.25,
        )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.max_leverage == 10.0
        assert manager.min_leverage == 1.0
        assert manager.base_position_size == 0.1
        assert manager.max_position_size == 0.25
        assert manager.leverage_history == []
        assert len(manager.performance_by_leverage) == 5

    def test_adjust_leverage_high_volatility(self, manager):
        """Test leverage adjustment with high volatility."""
        adjusted = manager._adjust_leverage(
            base_leverage=5.0,
            volatility=2.0,  # High volatility
            trend_strength=0.5,
            adx=25,
            confidence=0.5,
            regime="neutral",
            margin_ratio=0.0,
        )

        # High volatility should reduce leverage (may round to nearest common value)
        assert adjusted <= 5.0

    def test_adjust_leverage_low_volatility(self, manager):
        """Test leverage adjustment with low volatility."""
        adjusted = manager._adjust_leverage(
            base_leverage=3.0,
            volatility=0.5,  # Low volatility
            trend_strength=0.5,
            adx=25,
            confidence=0.5,
            regime="neutral",
            margin_ratio=0.0,
        )

        # Low volatility can increase leverage slightly
        # Result depends on other factors too
        assert adjusted >= 1.0

    def test_adjust_leverage_strong_trend(self, manager):
        """Test leverage adjustment with strong trend."""
        adjusted_strong = manager._adjust_leverage(
            base_leverage=3.0,
            volatility=1.0,
            trend_strength=0.8,  # Strong trend
            adx=45,  # High ADX
            confidence=0.7,
            regime="bull",
            margin_ratio=0.0,
        )

        adjusted_weak = manager._adjust_leverage(
            base_leverage=3.0,
            volatility=1.0,
            trend_strength=0.2,  # Weak trend
            adx=15,  # Low ADX
            confidence=0.3,
            regime="neutral",
            margin_ratio=0.0,
        )

        # Strong trend should result in higher leverage than weak
        assert adjusted_strong > adjusted_weak

    def test_adjust_leverage_regime_effect(self, manager):
        """Test regime affects leverage."""
        bull_leverage = manager._adjust_leverage(
            base_leverage=5.0,
            volatility=1.0,
            trend_strength=0.5,
            adx=30,
            confidence=0.6,
            regime="strong_bull",
            margin_ratio=0.0,
        )

        bear_leverage = manager._adjust_leverage(
            base_leverage=5.0,
            volatility=1.0,
            trend_strength=0.5,
            adx=30,
            confidence=0.6,
            regime="strong_bear",
            margin_ratio=0.0,
        )

        # Bull regime should allow higher leverage than bear
        assert bull_leverage >= bear_leverage

    def test_adjust_leverage_high_margin(self, manager):
        """Test high margin ratio reduces leverage."""
        normal = manager._adjust_leverage(
            base_leverage=5.0,
            volatility=1.0,
            trend_strength=0.5,
            adx=30,
            confidence=0.5,
            regime="neutral",
            margin_ratio=0.0,
        )

        high_margin = manager._adjust_leverage(
            base_leverage=5.0,
            volatility=1.0,
            trend_strength=0.5,
            adx=30,
            confidence=0.5,
            regime="neutral",
            margin_ratio=0.7,  # 70% margin used
        )

        # High margin should reduce leverage
        assert high_margin <= normal

    def test_adjust_leverage_clamped(self, manager):
        """Test leverage is clamped to limits."""
        # Try to get very high leverage
        adjusted = manager._adjust_leverage(
            base_leverage=15.0,  # Above max
            volatility=0.5,
            trend_strength=0.9,
            adx=50,
            confidence=0.9,
            regime="strong_bull",
            margin_ratio=0.0,
        )

        assert adjusted <= manager.max_leverage
        assert adjusted >= manager.min_leverage

    def test_calculate_position_size_kelly(self, manager):
        """Test position size calculation uses Kelly-like approach."""
        size_high_conf = manager._calculate_position_size(
            leverage=3.0,
            confidence=0.8,  # High confidence
            volatility=1.0,
            risk_budget=0.02,
            win_rate=0.6,
        )

        size_low_conf = manager._calculate_position_size(
            leverage=3.0,
            confidence=0.4,  # Low confidence
            volatility=1.0,
            risk_budget=0.02,
            win_rate=0.4,
        )

        # Higher confidence and win rate should lead to larger position
        assert size_high_conf > size_low_conf

    def test_calculate_position_size_volatility_adjusted(self, manager):
        """Test position size adjusted for volatility."""
        size_low_vol = manager._calculate_position_size(
            leverage=3.0,
            confidence=0.6,
            volatility=0.5,  # Low volatility
            risk_budget=0.02,
            win_rate=0.55,
        )

        size_high_vol = manager._calculate_position_size(
            leverage=3.0,
            confidence=0.6,
            volatility=2.0,  # High volatility
            risk_budget=0.02,
            win_rate=0.55,
        )

        # Lower volatility should allow larger position
        assert size_low_vol >= size_high_vol

    def test_calculate_position_size_clamped(self, manager):
        """Test position size is clamped to valid range."""
        size = manager._calculate_position_size(
            leverage=1.0,
            confidence=0.99,
            volatility=0.3,
            risk_budget=0.5,
            win_rate=0.8,
        )

        assert size >= 0.02  # Minimum
        assert size <= manager.max_position_size

    def test_calculate_risk_metrics(self, manager):
        """Test risk metrics calculation."""
        metrics = manager._calculate_risk_metrics(
            leverage=5.0,
            position_size=0.1,
            volatility=1.0,
            account_balance=10000,
            direction="long",
        )

        assert "position_value" in metrics
        assert "daily_volatility_pct" in metrics
        assert "max_loss_pct" in metrics
        assert "max_loss_usd" in metrics
        assert "liquidation_distance_pct" in metrics
        assert "effective_leverage" in metrics
        assert "risk_reward_ratio" in metrics

        # Check calculations
        assert metrics["position_value"] == 5000  # 10000 * 0.1 * 5
        assert metrics["effective_leverage"] == 0.5  # 5 * 0.1

    def test_build_reasoning_long(self, manager):
        """Test reasoning for long position."""
        reasoning = manager._build_reasoning(
            direction="long",
            leverage=3.0,
            confidence=0.7,
            indicators={"trend_direction": 0.5, "rsi": 35},
            analysis={"best_action": "LONG_3X"},
        )

        assert "LONG" in reasoning
        assert "3" in reasoning  # 3.0x leverage mentioned
        assert "high" in reasoning.lower()  # High confidence

    def test_build_reasoning_short(self, manager):
        """Test reasoning for short position."""
        reasoning = manager._build_reasoning(
            direction="short",
            leverage=2.0,
            confidence=0.6,
            indicators={"trend_direction": -0.5, "rsi": 75},
            analysis={"best_action": "SHORT_2X"},
        )

        assert "SHORT" in reasoning
        assert "2" in reasoning  # 2.0x leverage mentioned
        assert "overbought" in reasoning.lower()

    def test_build_reasoning_close(self, manager):
        """Test reasoning for close position."""
        reasoning = manager._build_reasoning(
            direction="close",
            leverage=0,
            confidence=0.5,
            indicators={},
            analysis={},
        )

        assert "Close" in reasoning

    def test_build_reasoning_high_volatility(self, manager):
        """Test reasoning includes volatility note."""
        reasoning = manager._build_reasoning(
            direction="long",
            leverage=2.0,
            confidence=0.5,
            indicators={"volatility_ratio": 2.0},  # High volatility
            analysis={},
        )

        assert "volatility" in reasoning.lower()

    def test_get_margin_status_basic(self, manager):
        """Test margin status calculation."""
        status = manager.get_margin_status(
            account_info={
                "total_margin": 10000,
                "used_margin": 2000,
            }
        )

        assert status.total_margin == 10000
        assert status.used_margin == 2000
        assert status.available_margin == 8000
        assert status.margin_ratio == 0.2
        assert status.liquidation_price is None
        assert status.distance_to_liquidation == 100.0

    def test_get_margin_status_with_position(self, manager):
        """Test margin status with active position."""
        status = manager.get_margin_status(
            account_info={
                "total_margin": 10000,
                "used_margin": 5000,
            },
            position_info={
                "entry_price": 50000,
                "leverage": 5,
                "side": "long",
                "current_price": 52000,
                "unrealized_pnl": 400,
            },
        )

        assert status.margin_ratio == 0.5
        assert status.liquidation_price is not None
        assert status.distance_to_liquidation > 0
        assert status.unrealized_pnl == 400

    def test_get_margin_status_short_position(self, manager):
        """Test margin status with short position."""
        status = manager.get_margin_status(
            account_info={
                "total_margin": 10000,
                "used_margin": 3000,
            },
            position_info={
                "entry_price": 50000,
                "leverage": 3,
                "side": "short",
                "current_price": 48000,
                "unrealized_pnl": 200,
            },
        )

        assert status.liquidation_price is not None
        # For short, liquidation is above entry
        assert status.liquidation_price > 50000

    def test_should_reduce_leverage_high_margin(self, manager):
        """Test reduce leverage check with high margin."""
        status = MarginStatus(
            total_margin=10000,
            used_margin=8000,  # 80% used
            available_margin=2000,
            margin_ratio=0.8,
            liquidation_price=None,
            distance_to_liquidation=50,
            unrealized_pnl=0,
        )

        should, reason = manager.should_reduce_leverage(
            margin_status=status,
            volatility=1.0,
            consecutive_losses=0,
        )

        assert should is True
        assert "margin" in reason.lower()

    def test_should_reduce_leverage_close_to_liquidation(self, manager):
        """Test reduce leverage when close to liquidation."""
        status = MarginStatus(
            total_margin=10000,
            used_margin=5000,
            available_margin=5000,
            margin_ratio=0.5,
            liquidation_price=45000,
            distance_to_liquidation=5.0,  # Only 5% away
            unrealized_pnl=-500,
        )

        should, reason = manager.should_reduce_leverage(
            margin_status=status,
            volatility=1.0,
            consecutive_losses=0,
        )

        assert should is True
        assert "liquidation" in reason.lower()

    def test_should_reduce_leverage_extreme_volatility(self, manager):
        """Test reduce leverage in extreme volatility."""
        status = MarginStatus(
            total_margin=10000,
            used_margin=2000,
            available_margin=8000,
            margin_ratio=0.2,
            liquidation_price=None,
            distance_to_liquidation=100,
            unrealized_pnl=0,
        )

        should, reason = manager.should_reduce_leverage(
            margin_status=status,
            volatility=2.5,  # 2.5x normal
            consecutive_losses=0,
        )

        assert should is True
        assert "volatility" in reason.lower()

    def test_should_reduce_leverage_losing_streak(self, manager):
        """Test reduce leverage on losing streak."""
        status = MarginStatus(
            total_margin=10000,
            used_margin=2000,
            available_margin=8000,
            margin_ratio=0.2,
            liquidation_price=None,
            distance_to_liquidation=100,
            unrealized_pnl=0,
        )

        should, reason = manager.should_reduce_leverage(
            margin_status=status,
            volatility=1.0,
            consecutive_losses=4,
        )

        assert should is True
        assert "losses" in reason.lower()

    def test_should_reduce_leverage_healthy(self, manager):
        """Test no reduce needed when healthy."""
        status = MarginStatus(
            total_margin=10000,
            used_margin=2000,
            available_margin=8000,
            margin_ratio=0.2,
            liquidation_price=None,
            distance_to_liquidation=100,
            unrealized_pnl=100,
        )

        should, reason = manager.should_reduce_leverage(
            margin_status=status,
            volatility=1.0,
            consecutive_losses=1,
        )

        assert should is False
        assert "healthy" in reason.lower()

    def test_record_trade_result(self, manager):
        """Test recording trade results."""
        manager.record_trade_result(
            leverage=3.0,
            pnl_pct=2.5,
            direction="long",
            hold_duration=60,
            market_conditions={"symbol": "BTC/USDT", "regime": "bull"},
        )

        assert len(manager.leverage_history) == 1
        assert manager.leverage_history[0]["leverage"] == 3.0
        assert manager.leverage_history[0]["pnl_pct"] == 2.5
        assert 3.0 in manager.performance_by_leverage
        assert 2.5 in manager.performance_by_leverage[3.0]

    def test_get_leverage_performance_summary_empty(self, manager):
        """Test summary with no trades."""
        summary = manager.get_leverage_performance_summary()

        assert "by_leverage" in summary
        assert "total_trades" in summary
        assert summary["total_trades"] == 0

    def test_get_leverage_performance_summary_with_trades(self, manager):
        """Test summary with trades."""
        # Record some trades
        for pnl in [2.0, -1.0, 1.5, 3.0, -0.5]:
            manager.record_trade_result(
                leverage=3.0,
                pnl_pct=pnl,
                direction="long",
                hold_duration=60,
                market_conditions={},
            )

        summary = manager.get_leverage_performance_summary()

        # Key format is "{leverage}x" - could be "3x" or "3.0x" depending on implementation
        leverage_key = "3.0x" if "3.0x" in summary["by_leverage"] else "3x"
        assert leverage_key in summary["by_leverage"]
        stats = summary["by_leverage"][leverage_key]
        assert stats["trades"] == 5
        assert stats["total_pnl"] == 5.0  # 2 - 1 + 1.5 + 3 - 0.5
        assert stats["win_rate"] == 0.6  # 3 wins out of 5


class TestGetLeverageManager:
    """Test global manager getter."""

    def test_get_leverage_manager_creates_instance(self):
        """Test getter creates instance."""
        # Reset global instance first
        import bot.ai_engine.leverage_manager as lm

        lm._leverage_manager = None

        with patch.object(lm, "get_leverage_rl_agent"), patch.object(lm, "get_learning_db"):
            manager = get_leverage_manager()
            assert manager is not None
            assert isinstance(manager, AILeverageManager)

    def test_get_leverage_manager_returns_same_instance(self):
        """Test getter returns same instance."""
        import bot.ai_engine.leverage_manager as lm

        with patch.object(lm, "get_leverage_rl_agent"), patch.object(lm, "get_learning_db"):
            manager1 = get_leverage_manager()
            manager2 = get_leverage_manager()
            assert manager1 is manager2
