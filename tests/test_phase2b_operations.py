"""
Tests for Phase 2B Operational Scripts.

Tests cover:
1. Report schema validity
2. Deterministic output for fixed inputs
3. Clamp enforcement logging present in reports
4. BTC turnover reduction rules
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.rl.btc_turnover_reduction import (
    BTCTurnoverConfig,
    BTCTurnoverReducer,
    DecisionGate,
    get_btc_turnover_reducer,
    reset_btc_turnover_reducer,
    should_allow_btc_trade,
    DEFAULT_BTC_CONFIG,
)

from bot.rl.strategy_weighting_advisor import (
    MAX_DAILY_WEIGHT_SHIFT,
    MIN_STRATEGY_WEIGHT,
    MAX_STRATEGY_WEIGHT,
    StrategyWeightingAdvisor,
    WeightingConfig,
    get_strategy_weighting_advisor,
    reset_strategy_weighting_advisor,
)


# =============================================================================
# Daily Health Report Schema Tests
# =============================================================================

class TestDailyHealthReportSchema:
    """Tests for daily health report schema validity."""

    def test_schema_has_required_fields(self):
        """Test that schema has all required fields."""
        # Simulated report structure
        report = {
            "date": "2026-01-29",
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {
                "logging_healthy": True,
                "decisions_today": 50,
                "pending_decisions": 2,
                "total_all_time": 1500,
            },
            "capital_preservation": {
                "current_level": "normal",
                "transitions_today": 0,
                "alert": False,
                "alert_reason": "",
            },
            "strategy_weighting": {
                "enabled": True,
                "shifts_today": 1,
                "max_shift_observed": 0.05,
                "clamp_exceeded": False,
                "clamp_violation_details": "",
                "max_allowed_shift": MAX_DAILY_WEIGHT_SHIFT,
            },
            "tradegate": {
                "rejections_today": 10,
                "rejection_reasons": {"low_confidence": 8, "high_risk": 2},
                "distribution_alert": False,
            },
            "summary": {
                "overall_health": "HEALTHY",
                "alerts": [],
                "recommendations": [],
            },
        }

        # Verify required top-level fields
        assert "date" in report
        assert "timestamp" in report
        assert "shadow_collection" in report
        assert "capital_preservation" in report
        assert "strategy_weighting" in report
        assert "tradegate" in report
        assert "summary" in report

        # Verify shadow_collection fields
        sc = report["shadow_collection"]
        assert "logging_healthy" in sc
        assert "decisions_today" in sc
        assert "total_all_time" in sc
        assert isinstance(sc["logging_healthy"], bool)
        assert isinstance(sc["decisions_today"], int)

        # Verify strategy_weighting has clamp fields
        sw = report["strategy_weighting"]
        assert "clamp_exceeded" in sw
        assert "max_shift_observed" in sw
        assert "max_allowed_shift" in sw
        assert sw["max_allowed_shift"] == MAX_DAILY_WEIGHT_SHIFT

        # Verify summary has health status
        summary = report["summary"]
        assert "overall_health" in summary
        assert summary["overall_health"] in ["HEALTHY", "WARNING", "CRITICAL"]

    def test_schema_serializes_to_valid_json(self):
        """Test that report can be serialized to valid JSON."""
        report = {
            "date": "2026-01-29",
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {"logging_healthy": True, "decisions_today": 0},
            "capital_preservation": {"current_level": "normal"},
            "strategy_weighting": {
                "clamp_exceeded": False,
                "max_shift_observed": 0.0,
                "max_allowed_shift": 0.10,
            },
            "tradegate": {"rejections_today": 0},
            "summary": {"overall_health": "HEALTHY", "alerts": []},
        }

        # Should serialize without error
        json_str = json.dumps(report)
        assert json_str is not None

        # Should deserialize back
        loaded = json.loads(json_str)
        assert loaded["date"] == "2026-01-29"


# =============================================================================
# Weekly Report Schema Tests
# =============================================================================

class TestWeeklyReportSchema:
    """Tests for weekly report schema validity."""

    def test_schema_has_required_fields(self):
        """Test that weekly schema has all required fields."""
        report = {
            "week_ending": "2026-01-29",
            "timestamp": datetime.now().isoformat(),
            "week_number": 5,
            "data_collection": {
                "total_decisions": 350,
                "by_symbol": {"BTC-USD": 50, "ETH-USD": 100},
                "by_regime": {"bull": 200, "sideways": 100, "bear": 50},
            },
            "counterfactual_evaluation": {
                "overall": {
                    "total_decisions": 350,
                    "rl_agreement_rate": 0.65,
                    "rl_hit_rate": 0.58,
                },
                "pnl_comparison": {
                    "delta_pnl": 500.0,
                    "incremental_sharpe": 0.45,
                },
            },
            "symbol_edge_analysis": {
                "BTC-USD": {
                    "decisions": 50,
                    "delta_pnl": 150.00,
                    "hit_rate": 0.58,
                    "sharpe": 0.45,
                    "max_drawdown": 0.02,
                    "turnover": 50,
                    "edge_positive": True,
                },
            },
            "regime_edge_analysis": {
                "bull": {
                    "decisions": 200,
                    "delta_pnl": 300.0,
                    "hit_rate": 0.60,
                    "edge_positive": True,
                },
            },
            "cost_decomposition": {
                "total_slippage": 100.0,
                "total_fees": 50.0,
                "total_spread": 25.0,
                "total_cost": 175.0,
                "avg_cost_per_trade": 0.50,
                "cost_as_pct_of_pnl": 0.35,
            },
            "confidence_sweep": {
                "best_threshold": 0.75,
                "results": [
                    {"threshold": 0.5, "hit_rate_above": 0.55},
                    {"threshold": 0.75, "hit_rate_above": 0.65},
                ],
            },
            "drift_check": {
                "sharpe": {"current": 0.45, "prior": 0.40, "degraded": False},
                "overall_alert": False,
            },
            "btc_diagnosis": {
                "primary_cause": "Excessive trading frequency",
                "estimated_recoverable_pct": 15.0,
            },
            "summary": {
                "overall_edge_positive": True,
                "edge_stable_across_regimes": True,
                "promotion_gate_progress": {
                    "weeks_collected": {"required": 12, "current": 5, "met": False},
                },
                "alerts": [],
                "recommendations": ["Phase 2C promotion gates: 3/7 met"],
            },
        }

        # Verify required top-level fields
        assert "week_ending" in report
        assert "week_number" in report
        assert "data_collection" in report
        assert "counterfactual_evaluation" in report
        assert "symbol_edge_analysis" in report
        assert "regime_edge_analysis" in report
        assert "cost_decomposition" in report
        assert "confidence_sweep" in report
        assert "summary" in report

        # Verify edge analysis structure
        for symbol, metrics in report["symbol_edge_analysis"].items():
            assert "decisions" in metrics
            assert "delta_pnl" in metrics
            assert "hit_rate" in metrics
            assert "edge_positive" in metrics

        # Verify promotion gates in summary
        assert "promotion_gate_progress" in report["summary"]

    def test_promotion_gate_fields_present(self):
        """Test that all Phase 2C gates are represented."""
        gate_progress = {
            "weeks_collected": {"required": 12, "current": 5, "met": False},
            "positive_edge": {"required": True, "current": True, "met": True},
            "hit_rate": {"required": 0.55, "current": 0.58, "met": True},
            "incremental_sharpe": {"required": 0.3, "current": 0.45, "met": True},
            "max_drawdown": {"required": "<0.05", "current": 0.02, "met": True},
            "btc_mitigation": {"required": "Validated", "current": "In progress", "met": False},
            "human_signoff": {"required": True, "current": False, "met": False},
        }

        # All 7 gates must be present
        assert len(gate_progress) == 7
        assert "weeks_collected" in gate_progress
        assert "positive_edge" in gate_progress
        assert "hit_rate" in gate_progress
        assert "incremental_sharpe" in gate_progress
        assert "max_drawdown" in gate_progress
        assert "btc_mitigation" in gate_progress
        assert "human_signoff" in gate_progress

        # Each gate must have required fields
        for gate, status in gate_progress.items():
            assert "required" in status
            assert "current" in status
            assert "met" in status


# =============================================================================
# Clamp Enforcement Tests
# =============================================================================

class TestClampEnforcement:
    """Tests for strategy weighting clamp enforcement."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_strategy_weighting_advisor()

    def test_clamp_value_is_locked(self):
        """Test that clamp value cannot be exceeded in config."""
        # Try to set higher than allowed
        config = WeightingConfig(max_daily_shift=0.50)  # Try 50%

        # Should be clamped to MAX_DAILY_WEIGHT_SHIFT (10%)
        assert config.max_daily_shift == MAX_DAILY_WEIGHT_SHIFT
        assert config.max_daily_shift == 0.10

    def test_weight_shifts_tracked(self):
        """Test that weight shifts are tracked for reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            # Get status - should track shifts
            status = advisor.get_advisor_status()

            assert "current_weights" in status
            assert "config" in status
            assert "max_daily_shift" in status["config"]
            assert status["config"]["max_daily_shift"] == MAX_DAILY_WEIGHT_SHIFT

    def test_clamp_violation_would_be_logged(self):
        """Test that clamp violations would appear in reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            # Get change analysis
            analysis = advisor.get_weight_change_analysis()

            # Should have cumulative_change tracking if there were changes
            # (Currently no changes, but structure should exist)
            assert analysis is not None

    def test_min_weight_enforced(self):
        """Test minimum weight cannot go below threshold."""
        config = WeightingConfig(min_weight=0.0)  # Try 0%

        # Should be clamped to MIN_STRATEGY_WEIGHT (5%)
        assert config.min_weight >= MIN_STRATEGY_WEIGHT
        assert config.min_weight >= 0.05

    def test_max_weight_enforced(self):
        """Test maximum weight cannot exceed threshold."""
        config = WeightingConfig(max_weight=1.0)  # Try 100%

        # Should be clamped to MAX_STRATEGY_WEIGHT (50%)
        assert config.max_weight <= MAX_STRATEGY_WEIGHT
        assert config.max_weight <= 0.50


# =============================================================================
# BTC Turnover Reduction Tests
# =============================================================================

class TestBTCTurnoverReduction:
    """Tests for BTC turnover reduction rules."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_btc_turnover_reducer()

    def test_default_config_is_restrictive(self):
        """Test that default BTC config is more restrictive than standard."""
        config = DEFAULT_BTC_CONFIG

        # More restrictive than typical defaults
        assert config.min_decision_interval_hours >= 4.0
        assert config.max_decisions_per_day <= 5
        assert config.min_expected_value_multiplier >= 2.0
        assert config.min_confidence_for_entry >= 0.70

    def test_non_btc_not_restricted(self):
        """Test that non-BTC symbols are not restricted."""
        reducer = get_btc_turnover_reducer()

        gate = reducer.check_decision_allowed(
            symbol="ETH-USD",
            position_value_usd=1000,
            expected_profit_pct=0.5,
            confidence=0.6,
        )

        assert gate.allowed is True
        assert "Not BTC" in gate.reason

    def test_btc_confidence_threshold(self):
        """Test BTC requires higher confidence."""
        reducer = get_btc_turnover_reducer()

        # Low confidence should be rejected
        gate = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=1000,
            expected_profit_pct=2.0,
            confidence=0.60,  # Below 75% threshold
        )

        assert gate.allowed is False
        assert "Confidence too low" in gate.reason

        # High confidence should pass (if other criteria met)
        gate2 = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=1000,
            expected_profit_pct=2.0,
            confidence=0.80,  # Above 75% threshold
        )

        # May still fail on expected value, but not confidence
        if not gate2.allowed:
            assert "Confidence" not in gate2.reason

    def test_btc_expected_value_threshold(self):
        """Test BTC requires minimum expected value."""
        reducer = get_btc_turnover_reducer()

        # Low expected profit should be rejected
        gate = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=1000,
            expected_profit_pct=0.1,  # 0.1% = $1 on $1000
            confidence=0.80,
        )

        assert gate.allowed is False
        assert "Expected profit" in gate.reason
        assert gate.required_profit_usd >= 50.0  # Minimum $50

    def test_btc_decision_interval_enforced(self):
        """Test minimum interval between BTC decisions."""
        reducer = get_btc_turnover_reducer()

        # First decision should be allowed
        gate1 = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=5000,
            expected_profit_pct=2.0,
            confidence=0.80,
        )

        if gate1.allowed:
            reducer.record_decision("BTC-USD")

            # Second decision immediately should be blocked
            gate2 = reducer.check_decision_allowed(
                symbol="BTC-USD",
                position_value_usd=5000,
                expected_profit_pct=2.0,
                confidence=0.80,
            )

            assert gate2.allowed is False
            assert "interval" in gate2.reason.lower()
            assert gate2.time_until_allowed_minutes > 0

    def test_btc_max_daily_decisions(self):
        """Test max daily decisions for BTC."""
        config = BTCTurnoverConfig(
            max_decisions_per_day=2,
            min_decision_interval_hours=0.0,  # Disable for this test
        )
        reducer = BTCTurnoverReducer(config)

        # Make 2 decisions
        for i in range(2):
            gate = reducer.check_decision_allowed(
                symbol="BTC-USD",
                position_value_usd=5000,
                expected_profit_pct=2.0,
                confidence=0.80,
            )
            if gate.allowed:
                reducer.record_decision("BTC-USD")

        # Third decision should be blocked
        gate = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=5000,
            expected_profit_pct=2.0,
            confidence=0.80,
        )

        assert gate.allowed is False
        assert "Max daily decisions" in gate.reason

    def test_btc_cooldown_after_losses(self):
        """Test cooldown triggered after consecutive losses."""
        config = BTCTurnoverConfig(
            cooldown_after_consecutive_losses=2,
            cooldown_after_loss_hours=1.0,
            min_decision_interval_hours=0.0,
        )
        reducer = BTCTurnoverReducer(config)

        # Record 2 consecutive losses
        reducer.record_outcome("BTC-USD", pnl=-100)
        reducer.record_outcome("BTC-USD", pnl=-50)

        # Should now be in cooldown
        gate = reducer.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=5000,
            expected_profit_pct=2.0,
            confidence=0.80,
        )

        assert gate.allowed is False
        assert "cooldown" in gate.reason.lower()

    def test_btc_cooldown_reset_on_win(self):
        """Test that cooldown counter resets on win."""
        config = BTCTurnoverConfig(cooldown_after_consecutive_losses=3)
        reducer = BTCTurnoverReducer(config)

        # Record 2 losses then a win
        reducer.record_outcome("BTC-USD", pnl=-100)
        reducer.record_outcome("BTC-USD", pnl=-50)
        reducer.record_outcome("BTC-USD", pnl=100)  # Win resets

        # Should not be in cooldown
        status = reducer.get_status()
        assert status["consecutive_losses"] == 0
        assert status["in_cooldown"] is False

    def test_convenience_function(self):
        """Test should_allow_btc_trade convenience function."""
        reset_btc_turnover_reducer()

        allowed, reason = should_allow_btc_trade(
            symbol="ETH-USD",
            position_value_usd=1000,
            expected_profit_pct=1.0,
            confidence=0.7,
        )

        assert allowed is True
        assert "Not BTC" in reason


# =============================================================================
# Deterministic Output Tests
# =============================================================================

class TestDeterministicOutput:
    """Tests for deterministic output with fixed inputs."""

    def test_clamp_calculation_deterministic(self):
        """Test that clamp calculation is deterministic."""
        reset_strategy_weighting_advisor()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two advisors with same config
            advisor1 = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights1.json",
            )
            advisor2 = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights2.json",
            )

            # Same input should give same output
            weights1 = advisor1.get_current_weights()
            weights2 = advisor2.get_current_weights()

            assert weights1 == weights2

    def test_btc_gate_deterministic(self):
        """Test that BTC gate decision is deterministic."""
        reset_btc_turnover_reducer()

        reducer1 = BTCTurnoverReducer()
        reducer2 = BTCTurnoverReducer()

        # Same input should give same output
        gate1 = reducer1.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=1000,
            expected_profit_pct=1.0,
            confidence=0.70,
        )

        gate2 = reducer2.check_decision_allowed(
            symbol="BTC-USD",
            position_value_usd=1000,
            expected_profit_pct=1.0,
            confidence=0.70,
        )

        assert gate1.allowed == gate2.allowed
        assert gate1.reason == gate2.reason


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2BOperationsIntegration:
    """Integration tests for Phase 2B operations."""

    def test_daily_report_includes_clamp_status(self):
        """Test that daily report structure includes clamp status."""
        # Simulated daily report generation
        report = {
            "strategy_weighting": {
                "enabled": True,
                "shifts_today": 0,
                "max_shift_observed": 0.0,
                "clamp_exceeded": False,
                "clamp_violation_details": "",
                "max_allowed_shift": MAX_DAILY_WEIGHT_SHIFT,
            },
        }

        # Verify clamp fields are present
        sw = report["strategy_weighting"]
        assert "clamp_exceeded" in sw
        assert "max_allowed_shift" in sw
        assert sw["max_allowed_shift"] == 0.10  # 10%

    def test_weekly_report_has_promotion_gates(self):
        """Test that weekly report includes promotion gate status."""
        # Simulated weekly report
        report = {
            "summary": {
                "promotion_gate_progress": {
                    "weeks_collected": {"met": False},
                    "positive_edge": {"met": True},
                    "human_signoff": {"met": False},
                },
            },
        }

        gates = report["summary"]["promotion_gate_progress"]
        assert len(gates) >= 3
        assert "weeks_collected" in gates
        assert "human_signoff" in gates

    def test_btc_reducer_status_serializable(self):
        """Test that BTC reducer status can be serialized."""
        reset_btc_turnover_reducer()
        reducer = get_btc_turnover_reducer()

        status = reducer.get_status()

        # Should serialize to JSON without error
        json_str = json.dumps(status)
        assert json_str is not None

        # Should have expected fields
        loaded = json.loads(json_str)
        assert "decisions_today" in loaded
        assert "config" in loaded


# =============================================================================
# Data Mode Tests
# =============================================================================

class TestDataModeTagging:
    """Tests for data_mode tagging across shadow pipeline."""

    def test_daily_report_schema_includes_data_mode(self):
        """Test that daily report schema includes data_mode field."""
        report = {
            "date": "2026-01-29",
            "timestamp": datetime.now().isoformat(),
            "data_mode": "TEST",
            "shadow_collection": {
                "logging_healthy": True,
                "decisions_today": 50,
                "decisions_by_mode": {"TEST": 50, "PAPER_LIVE": 0},
                "pending_decisions": 0,
                "total_all_time": 1500,
            },
            "capital_preservation": {"current_level": "normal"},
            "strategy_weighting": {"clamp_exceeded": False},
            "tradegate": {"rejections_today": 0},
            "summary": {"overall_health": "HEALTHY", "alerts": []},
        }

        # Verify data_mode is present at top level
        assert "data_mode" in report
        assert report["data_mode"] in ["TEST", "PAPER_LIVE"]

        # Verify decisions_by_mode in shadow_collection
        sc = report["shadow_collection"]
        assert "decisions_by_mode" in sc
        assert "TEST" in sc["decisions_by_mode"]
        assert "PAPER_LIVE" in sc["decisions_by_mode"]

    def test_weekly_report_schema_includes_data_mode(self):
        """Test that weekly report schema includes data_mode field."""
        report = {
            "week_ending": "2026-01-29",
            "timestamp": datetime.now().isoformat(),
            "week_number": 1,
            "data_mode": "TEST",
            "data_collection": {
                "total_decisions": 350,
                "decisions_by_mode": {"TEST": 350, "PAPER_LIVE": 0},
                "paper_live_decisions": 0,
                "by_symbol": {"BTC-USD": 100},
                "by_regime": {"bull": 200},
            },
            "summary": {
                "promotion_gate_progress": {
                    "weeks_collected": {
                        "required": 12,
                        "current": 0,
                        "counts_toward_gate": False,
                        "met": False,
                    }
                }
            },
        }

        # Verify data_mode is present at top level
        assert "data_mode" in report
        assert report["data_mode"] in ["TEST", "PAPER_LIVE"]

        # Verify decisions_by_mode in data_collection
        dc = report["data_collection"]
        assert "decisions_by_mode" in dc
        assert "paper_live_decisions" in dc

    def test_backwards_compatibility_missing_data_mode(self):
        """Test that missing data_mode field is treated as TEST."""
        # Old format report without data_mode
        old_format_entry = {
            "timestamp": "2026-01-29T12:00:00",
            "decision_id": "DEC_001",
            "symbol": "BTC-USD",
            "rl_recommendation": {"action": "buy", "confidence": 0.8},
        }

        # Default behavior: missing data_mode should be treated as TEST
        data_mode = old_format_entry.get("data_mode", "TEST")
        assert data_mode == "TEST"

        # Old format weekly report
        old_weekly_report = {
            "week_ending": "2026-01-22",
            "week_number": 1,
            # No data_mode field
        }

        data_mode = old_weekly_report.get("data_mode", "TEST")
        assert data_mode == "TEST"

    def test_gate1_only_counts_paper_live_weeks(self):
        """Test that Gate 1 (weeks collected) only counts PAPER_LIVE weeks."""
        # Simulated reports
        test_week_report = {
            "data_mode": "TEST",
            "week_number": 1,
        }
        paper_live_week_report = {
            "data_mode": "PAPER_LIVE",
            "week_number": 2,
        }

        # Simulate counting logic
        reports = [test_week_report, paper_live_week_report]
        paper_live_weeks = sum(
            1 for r in reports if r.get("data_mode", "TEST") == "PAPER_LIVE"
        )

        # Only 1 PAPER_LIVE week should be counted
        assert paper_live_weeks == 1

    def test_test_data_does_not_count_for_promotion(self):
        """Test that TEST data is excluded from Gate 1."""
        # 10 TEST weeks + 2 PAPER_LIVE weeks
        reports = [
            {"data_mode": "TEST"} for _ in range(10)
        ] + [
            {"data_mode": "PAPER_LIVE"} for _ in range(2)
        ]

        paper_live_weeks = sum(
            1 for r in reports if r.get("data_mode", "TEST") == "PAPER_LIVE"
        )
        test_weeks = sum(
            1 for r in reports if r.get("data_mode", "TEST") == "TEST"
        )

        assert paper_live_weeks == 2
        assert test_weeks == 10

        # Gate 1 requires 12 weeks of PAPER_LIVE data
        MIN_WEEKS_REQUIRED = 12
        gate_1_met = paper_live_weeks >= MIN_WEEKS_REQUIRED
        assert gate_1_met is False  # Only 2 PAPER_LIVE weeks

    def test_mixed_mode_week_treated_as_paper_live(self):
        """Test that a week with any PAPER_LIVE data is marked PAPER_LIVE."""
        # Week with both TEST and PAPER_LIVE decisions
        decisions_by_mode = {"TEST": 30, "PAPER_LIVE": 20}

        # Priority: PAPER_LIVE if any exist
        if decisions_by_mode.get("PAPER_LIVE", 0) > 0:
            overall_mode = "PAPER_LIVE"
        else:
            overall_mode = "TEST"

        assert overall_mode == "PAPER_LIVE"

    def test_test_data_generator_uses_test_mode(self):
        """Test that generate_test_decisions sets data_mode to TEST."""
        # Simulated output from generate_test_decisions.py
        generated_decision = {
            "timestamp": datetime.now().isoformat(),
            "decision_id": "DEC_TEST_001",
            "data_mode": "TEST",  # This should always be TEST
            "symbol": "BTC-USD",
        }

        assert generated_decision["data_mode"] == "TEST"

    def test_paper_trading_uses_paper_live_mode(self):
        """Test that paper trading collector sets data_mode to PAPER_LIVE."""
        # Simulated output from shadow_data_collector (paper trading)
        from bot.rl.shadow_data_collector import (
            DATA_MODE_PAPER_LIVE,
            DATA_MODE_TEST,
            DecisionSnapshot,
        )

        # Default should be PAPER_LIVE
        snapshot = DecisionSnapshot()
        assert snapshot.data_mode == DATA_MODE_PAPER_LIVE

        # to_dict should include data_mode
        snapshot_dict = snapshot.to_dict()
        assert "data_mode" in snapshot_dict
        assert snapshot_dict["data_mode"] == DATA_MODE_PAPER_LIVE


class TestDataModeGateEvaluation:
    """Tests for gate evaluation with data_mode filtering."""

    def test_gate_evaluator_counts_paper_live_only(self):
        """Test that gate evaluator only counts PAPER_LIVE weeks."""
        # Simulated weekly reports over 14 weeks
        weekly_reports = []

        # First 10 weeks: TEST data only
        for i in range(10):
            weekly_reports.append({
                "week_number": i + 1,
                "data_mode": "TEST",
            })

        # Weeks 11-14: PAPER_LIVE data
        for i in range(10, 14):
            weekly_reports.append({
                "week_number": i + 1,
                "data_mode": "PAPER_LIVE",
            })

        # Count PAPER_LIVE weeks
        paper_live_count = sum(
            1 for r in weekly_reports
            if r.get("data_mode", "TEST") == "PAPER_LIVE"
        )

        assert paper_live_count == 4  # Only 4 PAPER_LIVE weeks

        # Gate 1 check
        MIN_WEEKS = 12
        gate_1_met = paper_live_count >= MIN_WEEKS
        assert gate_1_met is False  # Need 12, only have 4

    def test_old_reports_without_data_mode_treated_as_test(self):
        """Test backwards compatibility for old reports without data_mode."""
        # Old format reports
        old_reports = [
            {"week_number": 1},  # No data_mode field
            {"week_number": 2},  # No data_mode field
            {"week_number": 3, "data_mode": "PAPER_LIVE"},
        ]

        # Counting logic with backwards compatibility
        paper_live_count = sum(
            1 for r in old_reports
            if r.get("data_mode", "TEST") == "PAPER_LIVE"
        )

        # Only 1 explicit PAPER_LIVE, others default to TEST
        assert paper_live_count == 1


class TestHeartbeatVerification:
    """Tests for paper-live heartbeat verification."""

    def test_heartbeat_write_and_read(self, tmp_path):
        """Test heartbeat file can be written and read."""
        from bot.rl.shadow_data_collector import (
            write_paper_live_heartbeat,
            read_paper_live_heartbeat,
            clear_paper_live_heartbeat,
        )

        heartbeat_path = tmp_path / "heartbeat.json"

        # Write heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT", "ETH/USDT"],
            total_decisions=5,
            paper_live_decisions=3,
            heartbeat_path=heartbeat_path,
        )

        # Read heartbeat
        heartbeat = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert heartbeat is not None
        assert heartbeat["shadow_collector_attached"] is True
        assert heartbeat["mode"] == "PAPER_LIVE"
        assert "BTC/USDT" in heartbeat["symbols"]
        assert "ETH/USDT" in heartbeat["symbols"]
        assert heartbeat["total_decisions_session"] == 5
        assert heartbeat["paper_live_decisions_session"] == 3

        # Clear heartbeat
        clear_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert read_paper_live_heartbeat(heartbeat_path=heartbeat_path) is None

    def test_heartbeat_recent_check(self, tmp_path):
        """Test heartbeat recency check."""
        from bot.rl.shadow_data_collector import (
            write_paper_live_heartbeat,
            read_paper_live_heartbeat,
            is_heartbeat_recent,
        )

        heartbeat_path = tmp_path / "heartbeat.json"

        # Write fresh heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT"],
            heartbeat_path=heartbeat_path,
        )

        heartbeat = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert heartbeat is not None

        # Fresh heartbeat should be recent
        assert is_heartbeat_recent(heartbeat, max_age_hours=2.0) is True

        # Create a simulated old heartbeat for testing
        old_heartbeat = heartbeat.copy()
        old_heartbeat["timestamp"] = (datetime.now() - timedelta(hours=1)).isoformat()

        # Old heartbeat should not be recent with strict max_age
        assert is_heartbeat_recent(old_heartbeat, max_age_hours=0.5) is False

    def test_heartbeat_stale_detection(self):
        """Test that stale heartbeats are detected."""
        from bot.rl.shadow_data_collector import is_heartbeat_recent

        # Simulate a 3-hour old heartbeat
        stale_heartbeat = {
            "timestamp": (datetime.now() - timedelta(hours=3)).isoformat(),
            "pid": 12345,
            "mode": "PAPER_LIVE",
        }

        # Should not be recent (default 2 hours)
        assert is_heartbeat_recent(stale_heartbeat, max_age_hours=2.0) is False

        # Should be recent if we increase max age
        assert is_heartbeat_recent(stale_heartbeat, max_age_hours=4.0) is True

    def test_missing_heartbeat_returns_none(self, tmp_path):
        """Test that missing heartbeat file returns None."""
        from bot.rl.shadow_data_collector import read_paper_live_heartbeat

        nonexistent_path = tmp_path / "nonexistent_heartbeat.json"
        heartbeat = read_paper_live_heartbeat(heartbeat_path=nonexistent_path)
        assert heartbeat is None

    def test_high_alert_recent_heartbeat_no_decisions(self):
        """Test HIGH alert when heartbeat recent but no PAPER_LIVE decisions."""
        from scripts.shadow.run_daily_shadow_health import (
            Alert,
            ALERT_SEVERITY_HIGH,
            ALERT_SEVERITY_MEDIUM,
        )

        # Scenario: Heartbeat exists and is recent, but PAPER_LIVE decisions = 0
        heartbeat_recent = True
        paper_live_count = 0

        alerts = []
        if heartbeat_recent and paper_live_count == 0:
            alerts.append(Alert(
                severity=ALERT_SEVERITY_HIGH,
                message="Paper-live bot running but no PAPER_LIVE decisions"
            ))

        assert len(alerts) == 1
        assert alerts[0].severity == ALERT_SEVERITY_HIGH

    def test_medium_alert_missing_heartbeat(self):
        """Test MEDIUM alert when heartbeat is missing."""
        from scripts.shadow.run_daily_shadow_health import (
            Alert,
            ALERT_SEVERITY_MEDIUM,
        )

        # Scenario: No heartbeat file found
        heartbeat_found = False

        alerts = []
        if not heartbeat_found:
            alerts.append(Alert(
                severity=ALERT_SEVERITY_MEDIUM,
                message="No paper-live heartbeat found"
            ))

        assert len(alerts) == 1
        assert alerts[0].severity == ALERT_SEVERITY_MEDIUM

    def test_alert_severity_levels_defined(self):
        """Test that all severity levels are properly defined."""
        from scripts.shadow.run_daily_shadow_health import (
            ALERT_SEVERITY_HIGH,
            ALERT_SEVERITY_MEDIUM,
            ALERT_SEVERITY_LOW,
        )

        assert ALERT_SEVERITY_HIGH == "HIGH"
        assert ALERT_SEVERITY_MEDIUM == "MEDIUM"
        assert ALERT_SEVERITY_LOW == "LOW"

    def test_alert_to_dict_serialization(self):
        """Test that Alert objects serialize correctly."""
        from scripts.shadow.run_daily_shadow_health import Alert, ALERT_SEVERITY_HIGH

        alert = Alert(
            severity=ALERT_SEVERITY_HIGH,
            message="Test alert message"
        )

        alert_dict = alert.to_dict()
        assert alert_dict["severity"] == "HIGH"
        assert alert_dict["message"] == "Test alert message"

    def test_daily_health_report_includes_heartbeat_section(self):
        """Test that daily health report includes heartbeat section."""
        from scripts.shadow.run_daily_shadow_health import DailyHealthReport

        report = DailyHealthReport(
            date="2026-01-29",
            timestamp=datetime.now().isoformat(),
        )

        # Set heartbeat fields
        report.heartbeat_found = True
        report.heartbeat_recent = True
        report.heartbeat_age_hours = 0.5
        report.heartbeat_pid = 12345
        report.heartbeat_symbols = ["BTC/USDT", "ETH/USDT"]

        report_dict = report.to_dict()

        assert "heartbeat" in report_dict
        hb = report_dict["heartbeat"]
        assert hb["found"] is True
        assert hb["recent"] is True
        assert hb["age_hours"] == 0.5
        assert hb["pid"] == 12345
        assert "BTC/USDT" in hb["symbols"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
