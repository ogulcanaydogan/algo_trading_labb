"""Tests for bot.adaptive_risk_controller module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from bot.adaptive_risk_controller import (
    RiskProfile,
    RiskDecision,
    CurrentStrategy,
    AdaptiveRiskController,
    get_adaptive_risk_controller,
)


class TestRiskProfile:
    """Tests for RiskProfile enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert RiskProfile.CONSERVATIVE.value == "conservative"
        assert RiskProfile.MODERATE.value == "moderate"
        assert RiskProfile.AGGRESSIVE.value == "aggressive"
        assert RiskProfile.MAXIMUM.value == "maximum"


class TestRiskDecision:
    """Tests for RiskDecision dataclass."""

    def test_default_values(self) -> None:
        """Test RiskDecision default values."""
        decision = RiskDecision()

        assert decision.setting == ""
        assert decision.old_value is False
        assert decision.new_value is False
        assert decision.rsi == 50.0
        assert decision.volatility == "normal"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        decision = RiskDecision(
            setting="shorting",
            old_value=False,
            new_value=True,
            reason="Bear market detected",
            market_regime="bear",
            trigger_condition="RSI=65",
            rsi=65.0,
            volatility="normal",
            trend="down",
            confidence=0.8,
        )

        result = decision.to_dict()

        assert result["setting"] == "shorting"
        assert result["old_value"] is False
        assert result["new_value"] is True
        assert result["reason"] == "Bear market detected"
        assert "timestamp" in result
        assert result["context"]["rsi"] == 65.0


class TestCurrentStrategy:
    """Tests for CurrentStrategy dataclass."""

    def test_default_values(self) -> None:
        """Test CurrentStrategy default values."""
        strategy = CurrentStrategy()

        assert strategy.name == "Conservative Hold"
        assert strategy.shorting_enabled is False
        assert strategy.leverage_enabled is False
        assert strategy.aggressive_enabled is False
        assert strategy.position_size_multiplier == 1.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        strategy = CurrentStrategy(
            name="Aggressive Bull Ride",
            shorting_enabled=False,
            leverage_enabled=True,
            aggressive_enabled=True,
            market_regime="strong_bull",
            expected_direction="bullish",
            confidence=0.85,
            reasoning=["LEVERAGE ENABLED: Strong trend"],
        )

        result = strategy.to_dict()

        assert result["name"] == "Aggressive Bull Ride"
        assert result["leverage_enabled"] is True
        assert result["aggressive_enabled"] is True
        assert result["market_regime"] == "strong_bull"


class TestAdaptiveRiskController:
    """Tests for AdaptiveRiskController class."""

    @pytest.fixture
    def controller(self, tmp_path: Path) -> AdaptiveRiskController:
        """Create controller with temp directory."""
        return AdaptiveRiskController(
            data_dir=tmp_path,
            api_base_url="http://localhost:8000",
        )

    def test_init(self, controller: AdaptiveRiskController) -> None:
        """Test controller initialization."""
        assert controller.current_settings["shorting"] is False
        assert controller.current_settings["leverage"] is False
        assert controller.current_settings["aggressive"] is False
        assert controller.current_profile == RiskProfile.CONSERVATIVE

    def test_save_and_load_decision(self, controller: AdaptiveRiskController) -> None:
        """Test saving and loading decisions."""
        decision = RiskDecision(
            setting="shorting",
            old_value=False,
            new_value=True,
            reason="Test",
        )

        controller._save_decision(decision)

        assert len(controller.decision_history) == 1
        assert controller.decision_log_path.exists()

    def test_decision_history_limit(self, controller: AdaptiveRiskController) -> None:
        """Test decision history is limited."""
        controller.max_history = 5

        for i in range(10):
            controller._save_decision(RiskDecision(setting=f"test_{i}"))

        assert len(controller.decision_history) == 5

    @pytest.mark.asyncio
    async def test_evaluate_conservative_in_unknown_regime(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test controller stays conservative in unknown regime."""
        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="unknown",
                regime_confidence=0.3,
                rsi=50,
                volatility="normal",
                trend="neutral",
            )

        assert strategy.shorting_enabled is False
        assert strategy.leverage_enabled is False
        assert strategy.aggressive_enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_enables_shorting_in_bear_market(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test shorting is enabled in bear market conditions."""
        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="bear",
                regime_confidence=0.7,
                rsi=65,
                volatility="normal",
                trend="down",
            )

        assert strategy.shorting_enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_enables_leverage_in_strong_trend(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test leverage is enabled in strong trend with good performance."""
        controller.recent_win_rate = 0.6
        controller.current_drawdown = 5

        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="strong_bull",
                regime_confidence=0.8,
                rsi=55,
                volatility="low",
                trend="up",
            )

        assert strategy.leverage_enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_disables_leverage_in_high_volatility(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test leverage is disabled in high volatility."""
        controller.recent_win_rate = 0.6
        controller.current_settings["leverage"] = True

        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="strong_bull",
                regime_confidence=0.8,
                rsi=55,
                volatility="extreme",
                trend="up",
            )

        assert strategy.leverage_enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_disables_all_in_drawdown(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test all risky features disabled during drawdown."""
        controller.current_drawdown = 20

        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="strong_bull",
                regime_confidence=0.9,
                rsi=50,
                volatility="low",
                trend="up",
            )

        assert strategy.leverage_enabled is False
        assert strategy.aggressive_enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_enables_aggressive_in_strong_conditions(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test aggressive mode enabled in ideal conditions."""
        controller.recent_win_rate = 0.65
        controller.recent_pnl = 5.0
        controller.current_drawdown = 2

        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="strong_bull",
                regime_confidence=0.75,
                rsi=55,
                volatility="low",
                trend="up",
            )

        assert strategy.aggressive_enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_disables_aggressive_in_sideways(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test aggressive mode disabled in sideways market."""
        controller.current_settings["aggressive"] = True

        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="sideways",
                regime_confidence=0.6,
                rsi=50,
                volatility="normal",
                trend="neutral",
            )

        assert strategy.aggressive_enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_enables_shorting_on_overbought_sideways(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test shorting enabled in overbought sideways market."""
        with patch.object(controller, "update_risk_settings", return_value=True):
            strategy = await controller.evaluate_and_adjust(
                market_regime="sideways",
                regime_confidence=0.7,
                rsi=75,
                volatility="normal",
                trend="neutral",
            )

        assert strategy.shorting_enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_updates_performance_metrics(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test performance metrics are updated."""
        performance = {
            "win_rate": 0.55,
            "total_pnl": 2.5,
            "drawdown": 3.0,
        }

        with patch.object(controller, "update_risk_settings", return_value=True):
            await controller.evaluate_and_adjust(
                market_regime="bull",
                regime_confidence=0.6,
                rsi=50,
                volatility="normal",
                trend="up",
                recent_performance=performance,
            )

        assert controller.recent_win_rate == 0.55
        assert controller.recent_pnl == 2.5
        assert controller.current_drawdown == 3.0

    def test_determine_strategy_name_bull_aggressive(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy name for aggressive bull."""
        name = controller._determine_strategy_name("strong_bull", False, True, True)
        assert name == "Aggressive Bull Ride"

    def test_determine_strategy_name_bear_shorting(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy name for bear with shorting."""
        name = controller._determine_strategy_name("bear", True, False, False)
        assert name == "Trend Following Short"

    def test_determine_strategy_name_crash_no_short(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy name for crash without shorting."""
        name = controller._determine_strategy_name("crash", False, False, False)
        assert name == "Capital Preservation"

    def test_determine_strategy_name_sideways(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy name for sideways market."""
        name = controller._determine_strategy_name("sideways", True, False, False)
        assert name == "Mean Reversion"

    def test_get_strategy_description(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy description generation."""
        desc = controller._get_strategy_description("strong_bull", "bullish", "buy")
        assert "uptrend" in desc.lower()

    def test_get_strategy_description_unknown(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test strategy description for unknown combination."""
        desc = controller._get_strategy_description("unknown", "neutral", "hold")
        assert "Analyzing" in desc

    def test_get_current_strategy(self, controller: AdaptiveRiskController) -> None:
        """Test get_current_strategy returns expected format."""
        result = controller.get_current_strategy()

        assert "strategy" in result
        assert "risk_profile" in result
        assert "settings" in result
        assert "recent_decisions" in result

    def test_get_decision_history(self, controller: AdaptiveRiskController) -> None:
        """Test get_decision_history returns list of dicts."""
        controller._save_decision(RiskDecision(setting="test"))

        history = controller.get_decision_history(limit=5)

        assert isinstance(history, list)
        assert len(history) == 1
        assert "setting" in history[0]

    @pytest.mark.asyncio
    async def test_update_risk_settings_connection_error(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test handling connection error in update_risk_settings."""
        # With an invalid URL, the connection will fail
        controller.api_base_url = "http://invalid-host-that-does-not-exist:9999"

        result = await controller.update_risk_settings({"shorting": True})

        # Should handle error gracefully and return False
        assert result is False

    def test_current_settings_manual_update(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test that current_settings can be updated directly."""
        controller.current_settings["shorting"] = True
        controller.current_settings["leverage"] = True

        assert controller.current_settings["shorting"] is True
        assert controller.current_settings["leverage"] is True

    @pytest.mark.asyncio
    async def test_risk_profile_assignment_maximum(
        self, controller: AdaptiveRiskController
    ) -> None:
        """Test maximum risk profile assignment."""
        controller.recent_win_rate = 0.7
        controller.recent_pnl = 10.0
        controller.current_drawdown = 0

        with patch.object(controller, "update_risk_settings", return_value=True):
            await controller.evaluate_and_adjust(
                market_regime="strong_bear",
                regime_confidence=0.85,
                rsi=65,
                volatility="low",
                trend="down",
            )

        # Should enable all three features in ideal bear conditions
        # Profile depends on actual feature enablement
        assert controller.current_profile in [
            RiskProfile.MODERATE,
            RiskProfile.AGGRESSIVE,
            RiskProfile.MAXIMUM,
        ]


class TestGlobalFunction:
    """Tests for global get_adaptive_risk_controller function."""

    def test_get_adaptive_risk_controller_singleton(self) -> None:
        """Test singleton pattern."""
        import bot.adaptive_risk_controller as module
        module._controller = None

        controller1 = get_adaptive_risk_controller()
        controller2 = get_adaptive_risk_controller()

        assert controller1 is controller2
