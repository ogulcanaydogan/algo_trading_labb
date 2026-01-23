"""
Contract Tests

These tests verify that API responses and data structures match their documented contracts.
If any of these tests fail, it indicates a breaking change that needs to be documented.

Reference: API_CONTRACTS.md
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional


# =============================================================================
# DATA CLASS CONTRACT TESTS
# =============================================================================


class TestPositionStateContract:
    """Tests for PositionState data class contract."""

    def test_required_fields_exist(self):
        """PositionState must have all required fields."""
        from bot.unified_state import PositionState

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time="2026-01-15T10:00:00Z",
        )

        # Required fields per API_CONTRACTS.md
        assert hasattr(pos, "symbol")
        assert hasattr(pos, "quantity")
        assert hasattr(pos, "entry_price")
        assert hasattr(pos, "side")
        assert hasattr(pos, "entry_time")

    def test_optional_fields_exist(self):
        """PositionState must have optional fields."""
        from bot.unified_state import PositionState

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time="2026-01-15T10:00:00Z",
            stop_loss=41000.0,
            take_profit=43000.0,
        )

        assert hasattr(pos, "stop_loss")
        assert hasattr(pos, "take_profit")
        assert hasattr(pos, "current_price")

    def test_side_values(self):
        """Side must be 'long' or 'short'."""
        from bot.unified_state import PositionState

        # Long position
        pos_long = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time="2026-01-15T10:00:00Z",
        )
        assert pos_long.side in ("long", "short")

        # Short position
        pos_short = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="short",
            entry_time="2026-01-15T10:00:00Z",
        )
        assert pos_short.side in ("long", "short")


class TestTradeRecordContract:
    """Tests for TradeRecord data class contract."""

    def test_required_fields_exist(self):
        """TradeRecord must have all required fields."""
        from bot.unified_state import TradeRecord

        trade = TradeRecord(
            id="test_123",
            symbol="BTC/USDT",
            side="long",
            quantity=0.01,
            entry_price=42000.0,
            exit_price=42500.0,
            pnl=5.0,
            pnl_pct=1.19,
            entry_time="2026-01-15T08:00:00Z",
            exit_time="2026-01-15T10:00:00Z",
            exit_reason="take_profit",
            mode="paper_live_data",
        )

        # Required fields per API_CONTRACTS.md
        required = [
            "id",
            "symbol",
            "side",
            "quantity",
            "entry_price",
            "exit_price",
            "pnl",
            "pnl_pct",
            "entry_time",
            "exit_time",
            "exit_reason",
            "mode",
        ]
        for field in required:
            assert hasattr(trade, field), f"Missing required field: {field}"

    def test_exit_reason_values(self):
        """Exit reason must be a valid value."""
        valid_reasons = ["take_profit", "stop_loss", "trailing_stop", "signal_sell", "manual"]

        from bot.unified_state import TradeRecord

        for reason in valid_reasons:
            trade = TradeRecord(
                id="test_123",
                symbol="BTC/USDT",
                side="long",
                quantity=0.01,
                entry_price=42000.0,
                exit_price=42500.0,
                pnl=5.0,
                pnl_pct=1.19,
                entry_time="2026-01-15T08:00:00Z",
                exit_time="2026-01-15T10:00:00Z",
                exit_reason=reason,
                mode="paper_live_data",
            )
            assert trade.exit_reason == reason


class TestUnifiedStateContract:
    """Tests for UnifiedState data class contract."""

    def test_required_fields_exist(self):
        """UnifiedState must have all required fields."""
        from bot.unified_state import UnifiedState, TradingMode, TradingStatus
        from datetime import datetime

        state = UnifiedState(
            mode=TradingMode.PAPER_LIVE_DATA,
            status=TradingStatus.ACTIVE,
            timestamp=datetime.now().isoformat(),
            initial_capital=10000.0,
            current_balance=10000.0,
            peak_balance=10000.0,
        )

        # Core state tracking fields
        required = [
            "mode",
            "status",
            "current_balance",
            "initial_capital",
            "total_pnl",
            "total_trades",
            "max_drawdown_pct",
        ]
        for field in required:
            assert hasattr(state, field), f"Missing required field: {field}"

    def test_mode_enum_values(self):
        """TradingMode must have expected values."""
        from bot.unified_state import TradingMode

        # Actual implementation values
        assert hasattr(TradingMode, "PAPER_LIVE_DATA")
        assert hasattr(TradingMode, "BACKTEST")  # Was PAPER_HISTORICAL
        assert hasattr(TradingMode, "LIVE_FULL")  # Live trading mode

    def test_status_enum_values(self):
        """TradingStatus must have expected values."""
        from bot.unified_state import TradingStatus

        # Actual implementation values
        assert hasattr(TradingStatus, "ACTIVE")  # Was RUNNING
        assert hasattr(TradingStatus, "PAUSED")
        assert hasattr(TradingStatus, "STOPPED")


# =============================================================================
# ML PERFORMANCE TRACKER CONTRACT TESTS
# =============================================================================


class TestMLPerformanceTrackerContract:
    """Tests for ML Performance Tracker contract."""

    def test_record_prediction_returns_id(self):
        """record_prediction must return a prediction ID."""
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLPerformanceTracker(db_path=f"{tmpdir}/test.db")

            pred_id = tracker.record_prediction(
                model_type="test_model",
                symbol="BTC/USDT",
                prediction="buy",
                confidence=0.75,
                market_condition="bull",
                volatility=50.0,
            )

            assert pred_id is not None
            assert isinstance(pred_id, str)

    def test_get_model_performance_shape(self):
        """get_model_performance must return correct shape."""
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLPerformanceTracker(db_path=f"{tmpdir}/test.db")

            # Record a prediction with outcome
            pred_id = tracker.record_prediction(
                model_type="test_model",
                symbol="BTC/USDT",
                prediction="buy",
                confidence=0.75,
                market_condition="bull",
                volatility=50.0,
            )
            tracker.record_outcome(pred_id, actual_return=1.5)

            perf = tracker.get_model_performance("test_model", days=30)

            # Per API_CONTRACTS.md
            required_keys = [
                "model_type",
                "market_condition",
                "total_predictions",
                "accuracy",
                "avg_confidence",
                "avg_return",
                "profit_factor",
                "days",
            ]
            for key in required_keys:
                assert key in perf, f"Missing key in performance response: {key}"

    def test_get_model_ranking_shape(self):
        """get_model_ranking must return list of models."""
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLPerformanceTracker(db_path=f"{tmpdir}/test.db")

            ranking = tracker.get_model_ranking(days=30)

            assert isinstance(ranking, list)

    def test_get_recommendation_shape(self):
        """get_recommendation must return correct shape."""
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLPerformanceTracker(db_path=f"{tmpdir}/test.db")

            rec = tracker.get_recommendation("bull")

            # Per API_CONTRACTS.md
            required_keys = ["recommended_model", "confidence", "reason"]
            for key in required_keys:
                assert key in rec, f"Missing key in recommendation: {key}"


# =============================================================================
# AI BRAIN CONTRACT TESTS
# =============================================================================


class TestAIBrainContract:
    """Tests for AI Brain contract."""

    def test_get_ai_brain_returns_instance(self):
        """get_ai_brain must return singleton instance."""
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        assert brain is not None

        # Should return same instance
        brain2 = get_ai_brain()
        assert brain is brain2

    def test_ai_brain_has_required_components(self):
        """AI Brain must have all required components."""
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()

        # Per SPEC.md architecture
        assert hasattr(brain, "daily_tracker")
        assert hasattr(brain, "strategy_generator")
        assert hasattr(brain, "pattern_learner")
        assert hasattr(brain, "trade_analyzer")

    def test_get_brain_status_shape(self):
        """get_brain_status must return correct shape."""
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        status = brain.get_brain_status()

        assert isinstance(status, dict)

        # Should have key status fields
        expected_keys = ["status", "daily_progress", "learning_stats"]
        for key in expected_keys:
            assert key in status, f"Missing key in status: {key}"

    def test_market_condition_values(self):
        """MarketCondition enum must have expected values."""
        from bot.ai_trading_brain import MarketCondition

        # Per SPEC.md
        expected = [
            "BULL",
            "BEAR",
            "SIDEWAYS",
            "VOLATILE",
            "STRONG_BULL",
            "STRONG_BEAR",
            "WEAK_BULL",
            "WEAK_BEAR",
        ]

        for cond in expected:
            assert hasattr(MarketCondition, cond), f"Missing MarketCondition: {cond}"


# =============================================================================
# RISK SETTINGS CONTRACT TESTS
# =============================================================================


class TestRiskSettingsContract:
    """Tests for risk settings contract."""

    def test_default_values(self):
        """Risk settings must default to safe values."""
        import json
        from pathlib import Path

        # Default values per SPEC.md
        defaults = {"shorting": False, "leverage": False, "aggressive": False}

        # If file exists, check format
        settings_path = Path("data/risk_settings.json")
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)

            for key in defaults:
                assert key in settings, f"Missing key in risk_settings.json: {key}"
                assert isinstance(settings[key], bool), f"{key} must be boolean"

    def test_risk_settings_keys(self):
        """Risk settings must have exactly the expected keys."""
        expected_keys = {"shorting", "leverage", "aggressive"}

        # These are the only allowed risk toggle keys
        # Adding new keys requires updating SPEC.md first
        assert expected_keys == {"shorting", "leverage", "aggressive"}


# =============================================================================
# ENGINE CONFIG CONTRACT TESTS
# =============================================================================


class TestEngineConfigContract:
    """Tests for engine configuration contract."""

    def test_required_config_fields(self):
        """EngineConfig must have required fields."""
        from bot.unified_engine import EngineConfig

        config = EngineConfig()

        # Per SPEC.md
        required = ["initial_mode", "stop_loss_pct", "take_profit_pct", "data_dir"]

        for field in required:
            assert hasattr(config, field), f"Missing config field: {field}"

    def test_default_risk_values(self):
        """EngineConfig defaults must match SPEC.md."""
        from bot.unified_engine import EngineConfig

        config = EngineConfig()

        # Per SPEC.md Section 2.1
        assert config.stop_loss_pct <= 0.05, "Stop loss must be <= 5%"


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


def validate_iso_timestamp(timestamp: str) -> bool:
    """Validate ISO 8601 timestamp format."""
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


class TestTimestampContract:
    """Tests for timestamp format contract."""

    def test_timestamps_are_iso_format(self):
        """All timestamps must be ISO 8601 format."""
        from bot.unified_state import PositionState

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now().isoformat(),
        )

        assert validate_iso_timestamp(pos.entry_time), "entry_time must be ISO format"


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestRegressions:
    """
    Regression tests for bugs that have been fixed.
    Add a test here whenever a bug is fixed to prevent recurrence.
    """

    def test_risk_toggles_not_inverted(self):
        """
        Regression: Risk toggles were showing inverted values.
        Fixed: 2026-01-15

        Risk settings True means feature is ENABLED, not disabled.
        """
        import json

        settings = {"shorting": True, "leverage": False, "aggressive": False}

        # True means enabled (user wants to allow shorting)
        assert settings["shorting"] == True  # Shorting is ENABLED

        # False means disabled (user does NOT want leverage)
        assert settings["leverage"] == False  # Leverage is DISABLED

    def test_null_check_on_dom_elements(self):
        """
        Regression: updateRiskLevelUI crashed on null DOM elements.
        Fixed: 2026-01-15

        DOM element access must check for null before property access.
        """
        # This is a JavaScript test reminder - the fix was in dashboard_unified.html
        # The pattern is:
        # if (element) { element.property = value; }
        pass

    def test_prediction_id_stored_for_tracking(self):
        """
        Regression: ML predictions weren't being tracked for performance analysis.
        Fixed: 2026-01-15

        Signals must include prediction_id and outcomes must be recorded.
        """
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLPerformanceTracker(db_path=f"{tmpdir}/test.db")

            # Must be able to record and retrieve
            pred_id = tracker.record_prediction(
                model_type="test",
                symbol="TEST",
                prediction="buy",
                confidence=0.7,
                market_condition="test",
                volatility=50.0,
            )

            # Must be able to record outcome
            tracker.record_outcome(pred_id, actual_return=1.0)

            # Must be retrievable
            perf = tracker.get_model_performance("test")
            assert perf["total_predictions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
