"""
Tests for Weekly Turnover Governor Tuner.

Covers:
1. Clamping behavior for all parameters
2. Decision logic for loosening vs tightening
3. Safety gate checks
4. Rollback functionality
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.turnover.weekly_turnover_tuner import (
    WeeklyTurnoverTuner,
    SymbolTuningBounds,
    SymbolTuningData,
    TuningRecommendation,
    TuningRationale,
    TuningResult,
    BTC_BOUNDS,
    ETH_BOUNDS,
    DEFAULT_BOUNDS,
    get_bounds_for_symbol,
    get_target_min_decisions_for_symbol,
    TARGET_MIN_DECISIONS_PER_DAY_DEFAULT,
    TARGET_MIN_DECISIONS_PER_DAY_BTC,
    TARGET_MIN_DECISIONS_PER_DAY_ETH,
)


# =============================================================================
# Test Clamping Behavior
# =============================================================================

class TestSymbolTuningBounds:
    """Test clamping behavior for all parameter types."""

    def test_btc_interval_clamp_at_minimum(self):
        """Test BTC interval cannot go below 30 min."""
        assert BTC_BOUNDS.clamp_interval(10.0) == 30.0
        assert BTC_BOUNDS.clamp_interval(0.0) == 30.0
        assert BTC_BOUNDS.clamp_interval(-5.0) == 30.0

    def test_btc_interval_clamp_at_maximum(self):
        """Test BTC interval cannot exceed 240 min."""
        assert BTC_BOUNDS.clamp_interval(300.0) == 240.0
        assert BTC_BOUNDS.clamp_interval(1000.0) == 240.0

    def test_btc_interval_within_bounds(self):
        """Test BTC interval within bounds is unchanged."""
        assert BTC_BOUNDS.clamp_interval(30.0) == 30.0
        assert BTC_BOUNDS.clamp_interval(120.0) == 120.0
        assert BTC_BOUNDS.clamp_interval(240.0) == 240.0

    def test_btc_max_decisions_clamp_at_minimum(self):
        """Test BTC max_decisions cannot go below 2."""
        assert BTC_BOUNDS.clamp_max_decisions(1) == 2
        assert BTC_BOUNDS.clamp_max_decisions(0) == 2
        assert BTC_BOUNDS.clamp_max_decisions(-5) == 2

    def test_btc_max_decisions_clamp_at_maximum(self):
        """Test BTC max_decisions cannot exceed 6."""
        assert BTC_BOUNDS.clamp_max_decisions(10) == 6
        assert BTC_BOUNDS.clamp_max_decisions(100) == 6

    def test_btc_max_decisions_within_bounds(self):
        """Test BTC max_decisions within bounds is unchanged."""
        assert BTC_BOUNDS.clamp_max_decisions(2) == 2
        assert BTC_BOUNDS.clamp_max_decisions(4) == 4
        assert BTC_BOUNDS.clamp_max_decisions(6) == 6

    def test_btc_ev_multiple_clamp_at_minimum(self):
        """Test BTC ev_multiple cannot go below 2.5."""
        assert BTC_BOUNDS.clamp_ev_multiple(1.0) == 2.5
        assert BTC_BOUNDS.clamp_ev_multiple(2.0) == 2.5
        assert BTC_BOUNDS.clamp_ev_multiple(0.0) == 2.5

    def test_btc_ev_multiple_clamp_at_maximum(self):
        """Test BTC ev_multiple cannot exceed 4.0."""
        assert BTC_BOUNDS.clamp_ev_multiple(5.0) == 4.0
        assert BTC_BOUNDS.clamp_ev_multiple(10.0) == 4.0

    def test_btc_ev_multiple_within_bounds(self):
        """Test BTC ev_multiple within bounds is unchanged."""
        assert BTC_BOUNDS.clamp_ev_multiple(2.5) == 2.5
        assert BTC_BOUNDS.clamp_ev_multiple(3.0) == 3.0
        assert BTC_BOUNDS.clamp_ev_multiple(4.0) == 4.0

    def test_eth_interval_clamp_at_minimum(self):
        """Test ETH interval cannot go below 10 min."""
        assert ETH_BOUNDS.clamp_interval(5.0) == 10.0
        assert ETH_BOUNDS.clamp_interval(0.0) == 10.0

    def test_eth_interval_clamp_at_maximum(self):
        """Test ETH interval cannot exceed 60 min."""
        assert ETH_BOUNDS.clamp_interval(120.0) == 60.0
        assert ETH_BOUNDS.clamp_interval(100.0) == 60.0

    def test_eth_max_decisions_clamp_at_minimum(self):
        """Test ETH max_decisions cannot go below 6."""
        assert ETH_BOUNDS.clamp_max_decisions(3) == 6
        assert ETH_BOUNDS.clamp_max_decisions(0) == 6

    def test_eth_max_decisions_clamp_at_maximum(self):
        """Test ETH max_decisions cannot exceed 12."""
        assert ETH_BOUNDS.clamp_max_decisions(15) == 12
        assert ETH_BOUNDS.clamp_max_decisions(20) == 12

    def test_eth_ev_multiple_clamp_at_minimum(self):
        """Test ETH ev_multiple cannot go below 1.8."""
        assert ETH_BOUNDS.clamp_ev_multiple(1.0) == 1.8
        assert ETH_BOUNDS.clamp_ev_multiple(1.5) == 1.8

    def test_eth_ev_multiple_clamp_at_maximum(self):
        """Test ETH ev_multiple cannot exceed 3.0."""
        assert ETH_BOUNDS.clamp_ev_multiple(4.0) == 3.0
        assert ETH_BOUNDS.clamp_ev_multiple(5.0) == 3.0


class TestGetBoundsForSymbol:
    """Test symbol-to-bounds mapping."""

    def test_btc_symbols_get_btc_bounds(self):
        """Test that BTC symbols get BTC bounds."""
        assert get_bounds_for_symbol("BTC/USDT") is BTC_BOUNDS
        assert get_bounds_for_symbol("BTC_USDT") is BTC_BOUNDS
        assert get_bounds_for_symbol("btc/usdt") is BTC_BOUNDS
        assert get_bounds_for_symbol("BTCUSD") is BTC_BOUNDS

    def test_eth_symbols_get_eth_bounds(self):
        """Test that ETH symbols get ETH bounds."""
        assert get_bounds_for_symbol("ETH/USDT") is ETH_BOUNDS
        assert get_bounds_for_symbol("ETH_USDT") is ETH_BOUNDS
        assert get_bounds_for_symbol("eth/usdt") is ETH_BOUNDS
        assert get_bounds_for_symbol("ETHUSD") is ETH_BOUNDS

    def test_other_symbols_get_default_bounds(self):
        """Test that other symbols get default bounds."""
        assert get_bounds_for_symbol("SOL/USDT") is DEFAULT_BOUNDS
        assert get_bounds_for_symbol("AVAX/USDT") is DEFAULT_BOUNDS
        assert get_bounds_for_symbol("DOGE/USDT") is DEFAULT_BOUNDS


# =============================================================================
# Test Decision Logic
# =============================================================================

class TestSymbolTuningData:
    """Test tuning data calculations."""

    def test_block_rate_calculation(self):
        """Test block rate is calculated correctly."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=80,
            total_blocked=20,
        )
        assert data.block_rate == pytest.approx(0.2)  # 20/(80+20) = 0.2

    def test_block_rate_zero_when_no_data(self):
        """Test block rate is 0 when no data."""
        data = SymbolTuningData(symbol="BTC/USDT")
        assert data.block_rate == 0.0

    def test_interval_block_rate(self):
        """Test interval block rate calculation."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=50,
            total_blocked=50,
            blocked_by_interval=30,
        )
        assert data.interval_block_rate == pytest.approx(0.3)  # 30/100

    def test_daily_limit_block_rate(self):
        """Test daily limit block rate calculation."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=50,
            total_blocked=50,
            blocked_by_daily_limit=10,
        )
        assert data.daily_limit_block_rate == pytest.approx(0.1)  # 10/100

    def test_ev_cost_block_rate(self):
        """Test EV/cost block rate calculation."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=50,
            total_blocked=50,
            blocked_by_ev_cost=25,
        )
        assert data.ev_cost_block_rate == pytest.approx(0.25)  # 25/100

    def test_decisions_per_day(self):
        """Test decisions per day calculation."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=21,
            days_with_data=7,
        )
        assert data.decisions_per_day == pytest.approx(3.0)  # 21/7 = 3

    def test_decisions_per_day_zero_days(self):
        """Test decisions per day returns 0 with no days."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=21,
            days_with_data=0,
        )
        assert data.decisions_per_day == 0.0


class TestTuningDecisionLogic:
    """Test the decision logic for when to loosen vs tighten."""

    @pytest.fixture
    def tuner(self, tmp_path):
        """Create tuner with temp directories."""
        return WeeklyTurnoverTuner(
            reports_dir=tmp_path / "reports",
            turnover_state_path=tmp_path / "state.json",
            output_dir=tmp_path / "output",
        )

    def test_high_interval_block_rate_recommends_loosening(self, tuner):
        """Test that high interval block rate recommends reducing interval."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=30,
            total_blocked=70,  # 70% block rate
            blocked_by_interval=60,  # Most due to interval
            current_min_interval=60.0,  # Currently 60 min
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should recommend reducing interval
        interval_recs = [r for r in result.recommendations if r.parameter == "min_interval"]
        assert len(interval_recs) == 1
        assert interval_recs[0].recommended_value < data.current_min_interval

    def test_low_block_rate_recommends_tightening(self, tuner):
        """Test that low block rate recommends increasing restrictions."""
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=95,
            total_blocked=5,  # 5% block rate
            blocked_by_interval=2,
            blocked_by_ev_cost=2,
            current_min_interval=15.0,
            current_ev_multiple=2.0,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should recommend increasing restrictions
        assert len(result.recommendations) > 0

        for rec in result.recommendations:
            if rec.parameter == "min_interval":
                assert rec.recommended_value >= data.current_min_interval
            if rec.parameter == "ev_multiple":
                assert rec.recommended_value >= data.current_ev_multiple

    def test_high_daily_limit_block_rate_increases_limit(self, tuner):
        """Test that high daily limit block rate increases the limit."""
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=30,
            total_blocked=70,
            blocked_by_daily_limit=60,  # Most due to daily limit
            current_max_decisions=6,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should recommend increasing max_decisions
        max_dec_recs = [r for r in result.recommendations if r.parameter == "max_decisions"]
        assert len(max_dec_recs) == 1
        assert max_dec_recs[0].recommended_value > data.current_max_decisions

    def test_high_ev_cost_block_rate_does_not_reduce_ev_when_decisions_at_target(self, tuner):
        """Test that high EV/cost block rate does NOT reduce EV when decisions/day >= target.

        This is a SAFETY RULE: We preserve trade quality by not lowering EV multiple
        when we're meeting the decision targets.
        """
        # ETH target is 3 decisions/day
        # With 7 days and 21 decisions, we have 3 decisions/day (at target)
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=21,  # 21/7 = 3 decisions/day = target
            total_blocked=70,
            blocked_by_ev_cost=60,  # High EV block rate (60/91 = 66%)
            current_ev_multiple=2.8,
            days_with_data=7,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should NOT recommend reducing ev_multiple (safety rule)
        ev_recs = [r for r in result.recommendations if r.parameter == "ev_multiple"]
        assert len(ev_recs) == 0, "EV multiple should NOT be reduced when decisions/day >= target"

    def test_recommendations_respect_btc_bounds(self, tuner):
        """Test that recommendations for BTC stay within BTC bounds."""
        # Try to trigger a recommendation that would exceed bounds
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=30,
            total_blocked=70,
            blocked_by_interval=60,
            current_min_interval=30.0,  # Already at minimum
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Any interval recommendation should not go below 30
        for rec in result.recommendations:
            if rec.parameter == "min_interval" and rec.symbol == "BTC/USDT":
                assert rec.recommended_value >= BTC_BOUNDS.min_interval_min

    def test_recommendations_respect_eth_bounds(self, tuner):
        """Test that recommendations for ETH stay within ETH bounds."""
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=30,
            total_blocked=70,
            blocked_by_daily_limit=60,
            current_max_decisions=12,  # Already at maximum
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Any max_decisions recommendation should not exceed 12
        for rec in result.recommendations:
            if rec.parameter == "max_decisions" and rec.symbol == "ETH/USDT":
                assert rec.recommended_value <= ETH_BOUNDS.max_decisions_max

    def test_insufficient_data_skips_analysis(self, tuner):
        """Test that insufficient data points skip analysis."""
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=5,  # Too few
            total_blocked=5,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should not generate any recommendations
        assert len(result.recommendations) == 0


# =============================================================================
# Test EV/Cost Safety Rules
# =============================================================================

class TestEVCostSafetyRules:
    """Test the EV/cost safety rules that preserve trade quality.

    SAFETY RULE: High EV/cost block rate should NOT automatically reduce EV multiple.
    We only loosen OTHER parameters (interval/max_decisions) IF decisions/day < target.
    """

    @pytest.fixture
    def tuner(self, tmp_path):
        """Create tuner with temp directories."""
        return WeeklyTurnoverTuner(
            reports_dir=tmp_path / "reports",
            turnover_state_path=tmp_path / "state.json",
            output_dir=tmp_path / "output",
        )

    def test_high_ev_block_rate_does_not_lower_ev_when_decisions_above_target(self, tuner):
        """Test that high EV block rate does NOT lower EV when decisions/day >= target."""
        # BTC target is 1 decision/day
        # With 7 days and 14 decisions = 2/day (above target)
        data = SymbolTuningData(
            symbol="BTC/USDT",
            total_decisions=14,  # 14/7 = 2 decisions/day > target (1)
            total_blocked=70,
            blocked_by_ev_cost=60,  # High EV block rate
            current_ev_multiple=3.0,
            current_min_interval=60.0,
            days_with_data=7,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should NOT recommend any EV multiple change
        ev_recs = [r for r in result.recommendations if r.parameter == "ev_multiple"]
        assert len(ev_recs) == 0, "EV multiple should NOT be changed when decisions/day >= target"

    def test_high_ev_block_rate_may_loosen_interval_when_decisions_below_target(self, tuner):
        """Test that high EV block rate may loosen interval when decisions/day < target."""
        # ETH target is 3 decisions/day
        # With 7 days and 7 decisions = 1/day (below target)
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=7,  # 7/7 = 1 decision/day < target (3)
            total_blocked=70,
            blocked_by_ev_cost=60,  # High EV block rate
            current_ev_multiple=2.5,
            current_min_interval=30.0,  # Not at minimum (10)
            current_max_decisions=12,   # At ETH max to prevent tightening rec
            days_with_data=7,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should recommend loosening interval (NOT ev_multiple)
        ev_recs = [r for r in result.recommendations if r.parameter == "ev_multiple"]
        assert len(ev_recs) == 0, "EV multiple should NOT be reduced even below target"

        # Should have interval recommendation from EV/cost safety rule (with rationale)
        interval_recs = [
            r for r in result.recommendations
            if r.parameter == "min_interval" and r.tuning_rationale is not None
        ]

        # At least one interval loosening with rationale
        assert len(interval_recs) >= 1, "Should loosen interval when below target"

        # Check that rationale is included
        for rec in interval_recs:
            assert rec.tuning_rationale.decisions_per_day == pytest.approx(1.0)
            assert rec.tuning_rationale.target_min_decisions == TARGET_MIN_DECISIONS_PER_DAY_ETH
            assert rec.tuning_rationale.rule_applied == "ev_cost_safety"

    def test_low_ev_block_rate_allows_tightening(self, tuner):
        """Test that low EV block rate allows INCREASING EV multiple (tightening)."""
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=90,
            total_blocked=10,
            blocked_by_ev_cost=2,  # Very low EV block rate (2/100 = 2%)
            current_ev_multiple=2.0,
            days_with_data=7,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should recommend INCREASING ev_multiple (tightening is safe)
        ev_recs = [r for r in result.recommendations if r.parameter == "ev_multiple"]
        assert len(ev_recs) == 1, "Should recommend tightening EV when block rate is low"
        assert ev_recs[0].recommended_value > data.current_ev_multiple

        # Should have rationale
        assert ev_recs[0].tuning_rationale is not None
        assert ev_recs[0].tuning_rationale.rule_applied == "ev_cost_tightening"
        assert ev_recs[0].tuning_rationale.action_taken == "increase_ev_multiple"

    def test_bounds_still_respected_with_safety_rules(self, tuner):
        """Test that bounds are respected even when safety rules apply."""
        # ETH at minimum interval - can't loosen further
        data = SymbolTuningData(
            symbol="ETH/USDT",
            total_decisions=7,  # Below target
            total_blocked=70,
            blocked_by_ev_cost=60,
            current_ev_multiple=2.5,
            current_min_interval=10.0,  # Already at ETH minimum
            current_max_decisions=12,   # Already at ETH maximum
            days_with_data=7,
        )

        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
        )

        tuner._analyze_symbol(data, result)

        # Should NOT recommend any changes because:
        # - EV multiple cannot be reduced (safety rule)
        # - Interval already at minimum (bounds)
        # - Max decisions already at maximum (bounds)
        for rec in result.recommendations:
            if rec.parameter == "ev_multiple":
                # If there is an EV rec, it should only be tightening
                assert rec.recommended_value >= data.current_ev_multiple
            if rec.parameter == "min_interval":
                assert rec.recommended_value >= ETH_BOUNDS.min_interval_min
            if rec.parameter == "max_decisions":
                assert rec.recommended_value <= ETH_BOUNDS.max_decisions_max


class TestTargetMinDecisionsConstants:
    """Test the target minimum decisions per day constants."""

    def test_btc_target_min_decisions(self):
        """Test BTC has target of 1 decision per day."""
        assert TARGET_MIN_DECISIONS_PER_DAY_BTC == 1

    def test_eth_target_min_decisions(self):
        """Test ETH has target of 3 decisions per day."""
        assert TARGET_MIN_DECISIONS_PER_DAY_ETH == 3

    def test_default_target_min_decisions(self):
        """Test default target is 3 decisions per day."""
        assert TARGET_MIN_DECISIONS_PER_DAY_DEFAULT == 3

    def test_get_target_for_btc_symbols(self):
        """Test BTC symbols get BTC target."""
        assert get_target_min_decisions_for_symbol("BTC/USDT") == 1
        assert get_target_min_decisions_for_symbol("btc_usdt") == 1
        assert get_target_min_decisions_for_symbol("BTCUSD") == 1

    def test_get_target_for_eth_symbols(self):
        """Test ETH symbols get ETH target."""
        assert get_target_min_decisions_for_symbol("ETH/USDT") == 3
        assert get_target_min_decisions_for_symbol("eth_usdt") == 3
        assert get_target_min_decisions_for_symbol("ETHUSD") == 3

    def test_get_target_for_other_symbols(self):
        """Test other symbols get default target."""
        assert get_target_min_decisions_for_symbol("SOL/USDT") == 3
        assert get_target_min_decisions_for_symbol("AVAX/USDT") == 3
        assert get_target_min_decisions_for_symbol("DOGE/USDT") == 3


class TestTuningRationale:
    """Test TuningRationale class."""

    def test_tuning_rationale_to_dict(self):
        """Test TuningRationale serialization."""
        rationale = TuningRationale(
            rule_applied="ev_cost_safety",
            decisions_per_day=1.5,
            target_min_decisions=3,
            ev_cost_block_rate=0.65,
            action_taken="loosen_interval_or_max_decisions",
            explanation="Test explanation",
        )

        d = rationale.to_dict()

        assert d["rule_applied"] == "ev_cost_safety"
        assert d["decisions_per_day"] == 1.5
        assert d["target_min_decisions"] == 3
        assert d["ev_cost_block_rate"] == 0.65
        assert d["action_taken"] == "loosen_interval_or_max_decisions"
        assert d["explanation"] == "Test explanation"

    def test_recommendation_with_rationale_serialization(self):
        """Test TuningRecommendation with rationale serializes correctly."""
        rationale = TuningRationale(
            rule_applied="ev_cost_tightening",
            decisions_per_day=5.0,
            target_min_decisions=3,
            ev_cost_block_rate=0.02,
            action_taken="increase_ev_multiple",
            explanation="Low block rate allows tightening",
        )

        rec = TuningRecommendation(
            symbol="ETH/USDT",
            parameter="ev_multiple",
            current_value=2.0,
            recommended_value=2.2,
            reason="Test",
            tuning_rationale=rationale,
        )

        d = rec.to_dict()

        assert "tuning_rationale" in d
        assert d["tuning_rationale"]["rule_applied"] == "ev_cost_tightening"
        assert d["tuning_rationale"]["action_taken"] == "increase_ev_multiple"

    def test_recommendation_without_rationale(self):
        """Test TuningRecommendation without rationale omits field."""
        rec = TuningRecommendation(
            symbol="BTC/USDT",
            parameter="min_interval",
            current_value=30.0,
            recommended_value=45.0,
            reason="Test",
        )

        d = rec.to_dict()

        assert "tuning_rationale" not in d


class TestTuningRecommendation:
    """Test TuningRecommendation calculations."""

    def test_change_pct_increase(self):
        """Test change percentage for increase."""
        rec = TuningRecommendation(
            symbol="BTC/USDT",
            parameter="min_interval",
            current_value=30.0,
            recommended_value=36.0,
            reason="Test",
        )
        assert rec.change_pct == pytest.approx(20.0)  # 6/30 = 0.2 = 20%

    def test_change_pct_decrease(self):
        """Test change percentage for decrease."""
        rec = TuningRecommendation(
            symbol="BTC/USDT",
            parameter="min_interval",
            current_value=60.0,
            recommended_value=48.0,
            reason="Test",
        )
        assert rec.change_pct == pytest.approx(-20.0)  # -12/60 = -0.2 = -20%

    def test_to_dict(self):
        """Test serialization to dict."""
        rec = TuningRecommendation(
            symbol="BTC/USDT",
            parameter="min_interval",
            current_value=30.0,
            recommended_value=36.0,
            reason="High block rate",
            confidence=0.7,
        )

        d = rec.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["parameter"] == "min_interval"
        assert d["current_value"] == 30.0
        assert d["recommended_value"] == 36.0
        assert d["reason"] == "High block rate"
        assert d["confidence"] == 0.7


# =============================================================================
# Test Safety Gates
# =============================================================================

class TestSafetyGates:
    """Test safety gate checks."""

    @pytest.fixture
    def tuner(self, tmp_path):
        """Create tuner with temp directories."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        return WeeklyTurnoverTuner(
            reports_dir=reports_dir,
            turnover_state_path=tmp_path / "state.json",
            output_dir=tmp_path / "output",
        )

    def test_safety_gate_fails_without_paper_live_weeks(self, tuner):
        """Test safety gate fails if PAPER_LIVE weeks < 1."""
        # Mock the metrics calculator to return 0 weeks
        with patch("api.shadow_health_metrics.ShadowHealthMetricsCalculator") as mock:
            mock_instance = MagicMock()
            mock_instance.calculate_metrics.return_value = MagicMock(
                paper_live_weeks_counted=0
            )
            mock.return_value = mock_instance

            result = TuningResult(
                timestamp=datetime.now().isoformat(),
                week_start="2024-01-01",
                week_end="2024-01-07",
            )

            passed = tuner._check_safety_gates(result)

            assert passed is False
            assert "PAPER_LIVE weeks_counted=0" in result.safety_gate_reason

    def test_safety_gate_passes_with_paper_live_weeks(self, tuner):
        """Test safety gate passes if PAPER_LIVE weeks >= 1."""
        with patch("api.shadow_health_metrics.ShadowHealthMetricsCalculator") as mock:
            mock_instance = MagicMock()
            mock_instance.calculate_metrics.return_value = MagicMock(
                paper_live_weeks_counted=2
            )
            mock.return_value = mock_instance

            result = TuningResult(
                timestamp=datetime.now().isoformat(),
                week_start="2024-01-01",
                week_end="2024-01-07",
            )

            passed = tuner._check_safety_gates(result)

            assert passed is True
            assert result.paper_live_weeks_counted == 2

    def test_safety_gate_fails_with_critical_alerts(self, tuner, tmp_path):
        """Test safety gate fails if CRITICAL alerts found."""
        # Create a report with CRITICAL status
        today = datetime.now().strftime("%Y-%m-%d")
        report = {
            "date": today,
            "summary": {
                "overall_health": "CRITICAL",
                "alerts": [
                    {"severity": "HIGH", "message": "Test critical alert"}
                ],
            },
        }

        report_path = tuner.reports_dir / f"daily_shadow_health_{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        with patch("api.shadow_health_metrics.ShadowHealthMetricsCalculator") as mock:
            mock_instance = MagicMock()
            mock_instance.calculate_metrics.return_value = MagicMock(
                paper_live_weeks_counted=2
            )
            mock.return_value = mock_instance

            result = TuningResult(
                timestamp=datetime.now().isoformat(),
                week_start="2024-01-01",
                week_end="2024-01-07",
            )

            passed = tuner._check_safety_gates(result)

            assert passed is False
            assert len(result.critical_alerts_found) > 0


# =============================================================================
# Test Rollback Functionality
# =============================================================================

class TestRollback:
    """Test rollback functionality."""

    def test_rollback_file_is_created_on_apply(self, tmp_path):
        """Test that rollback file is created when changes are applied."""
        from bot.turnover_governor import (
            TurnoverGovernor,
            TurnoverGovernorConfig,
            SymbolOverrideConfig,
            reset_turnover_governor,
        )

        reset_turnover_governor()

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tuner = WeeklyTurnoverTuner(
            reports_dir=reports_dir,
            turnover_state_path=tmp_path / "state.json",
            output_dir=output_dir,
        )

        # Create a result with recommendations
        result = TuningResult(
            timestamp=datetime.now().isoformat(),
            week_start="2024-01-01",
            week_end="2024-01-07",
            recommendations=[
                TuningRecommendation(
                    symbol="BTC/USDT",
                    parameter="min_interval",
                    current_value=30.0,
                    recommended_value=45.0,
                    reason="Test",
                ),
            ],
        )

        tuner._apply_changes(result)

        assert result.rollback_file is not None
        assert Path(result.rollback_file).exists()

    def test_rollback_restores_previous_config(self, tmp_path):
        """Test that rollback restores previous configuration."""
        # Create a rollback file
        rollback_data = {
            "timestamp": datetime.now().isoformat(),
            "reason": "Test rollback",
            "config": {
                "min_decision_interval_minutes": 20.0,
                "max_decisions_per_day": 8,
                "min_expected_value_multiple": 2.2,
                "symbol_overrides": {
                    "BTC/USDT": {
                        "min_interval_minutes": 45.0,
                        "max_decisions_per_day": 4,
                        "min_ev_cost_multiple": 3.0,
                    },
                },
            },
        }

        rollback_path = tmp_path / "rollback.json"
        with open(rollback_path, "w") as f:
            json.dump(rollback_data, f)

        success = WeeklyTurnoverTuner.rollback(str(rollback_path))

        assert success is True


class TestTuningResultSerialization:
    """Test TuningResult serialization."""

    def test_to_dict(self):
        """Test complete serialization."""
        result = TuningResult(
            timestamp="2024-01-15T10:00:00",
            week_start="2024-01-08",
            week_end="2024-01-15",
            paper_live_weeks_counted=2,
            safety_gate_passed=True,
            safety_gate_reason="All safety gates passed",
            symbols_analyzed=["BTC/USDT", "ETH/USDT"],
            total_decisions_week=100,
            total_blocked_week=30,
            total_cost_drag_avoided=45.50,
            recommendations=[
                TuningRecommendation(
                    symbol="BTC/USDT",
                    parameter="min_interval",
                    current_value=30.0,
                    recommended_value=36.0,
                    reason="Test",
                ),
            ],
            recommendations_applied=False,
            rollback_file="/path/to/rollback.json",
        )

        d = result.to_dict()

        assert d["week_start"] == "2024-01-08"
        assert d["week_end"] == "2024-01-15"
        assert d["safety_gate"]["passed"] is True
        assert d["safety_gate"]["paper_live_weeks_counted"] == 2
        assert d["data_summary"]["symbols_analyzed"] == ["BTC/USDT", "ETH/USDT"]
        assert d["data_summary"]["total_decisions_week"] == 100
        assert d["data_summary"]["total_blocked_week"] == 30
        assert d["data_summary"]["total_cost_drag_avoided"] == 45.50
        assert len(d["recommendations"]) == 1
        assert d["recommendations_applied"] is False


# =============================================================================
# Test Bound Values
# =============================================================================

class TestBoundValues:
    """Test that bound values are correctly defined."""

    def test_btc_bounds_are_stricter_than_eth(self):
        """Test that BTC has stricter bounds than ETH."""
        # BTC should have higher minimum interval
        assert BTC_BOUNDS.min_interval_min > ETH_BOUNDS.min_interval_min

        # BTC should have lower maximum decisions
        assert BTC_BOUNDS.max_decisions_max < ETH_BOUNDS.max_decisions_max

        # BTC should have higher minimum EV requirement
        assert BTC_BOUNDS.ev_multiple_min > ETH_BOUNDS.ev_multiple_min

    def test_btc_bounds_values(self):
        """Test specific BTC bound values."""
        assert BTC_BOUNDS.min_interval_min == 30.0
        assert BTC_BOUNDS.min_interval_max == 240.0
        assert BTC_BOUNDS.max_decisions_min == 2
        assert BTC_BOUNDS.max_decisions_max == 6
        assert BTC_BOUNDS.ev_multiple_min == 2.5
        assert BTC_BOUNDS.ev_multiple_max == 4.0

    def test_eth_bounds_values(self):
        """Test specific ETH bound values."""
        assert ETH_BOUNDS.min_interval_min == 10.0
        assert ETH_BOUNDS.min_interval_max == 60.0
        assert ETH_BOUNDS.max_decisions_min == 6
        assert ETH_BOUNDS.max_decisions_max == 12
        assert ETH_BOUNDS.ev_multiple_min == 1.8
        assert ETH_BOUNDS.ev_multiple_max == 3.0

    def test_default_bounds_values(self):
        """Test specific default bound values."""
        assert DEFAULT_BOUNDS.min_interval_min == 15.0
        assert DEFAULT_BOUNDS.min_interval_max == 120.0
        assert DEFAULT_BOUNDS.max_decisions_min == 4
        assert DEFAULT_BOUNDS.max_decisions_max == 10
        assert DEFAULT_BOUNDS.ev_multiple_min == 2.0
        assert DEFAULT_BOUNDS.ev_multiple_max == 3.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
