"""
Comprehensive tests for Readiness Calculator.

Tests each decision path:
- NO_GO: kill switch, capital preservation critical, shadow critical
- CONDITIONAL: heartbeat not recent, insufficient streak
- GO: all systems operational

Also tests schema validation and component aggregation.
"""

import pytest
from unittest.mock import patch, MagicMock

from api.readiness_calculator import (
    calculate_readiness,
    calculate_live_rollout_readiness,
    get_readiness,
    ShadowStatus,
    LiveStatus,
    TurnoverStatus,
    CapitalPreservationStatus,
    DailyReportsStatus,
    ExecutionRealismStatus,
    ReadinessResult,
    READINESS_GO,
    READINESS_CONDITIONAL,
    READINESS_NO_GO,
    MIN_STREAK_FOR_GO,
    LIVE_ROLLOUT_MIN_STREAK,
    LIVE_ROLLOUT_MIN_WEEKS,
    TURNOVER_BLOCK_RATE_MIN,
    TURNOVER_BLOCK_RATE_MAX,
)


# =============================================================================
# Test Fixtures - Component Statuses
# =============================================================================


@pytest.fixture
def healthy_shadow():
    """Shadow status with all healthy metrics."""
    return ShadowStatus(
        paper_live_decisions_today=15,
        paper_live_decisions_7d=85,
        paper_live_days_streak=10,  # > 7 minimum
        paper_live_weeks_counted=3,
        heartbeat_recent=1,
        overall_health="HEALTHY",
    )


@pytest.fixture
def healthy_live():
    """Live guardrails status with no issues."""
    return LiveStatus(
        live_mode_enabled=False,
        kill_switch_active=False,
        kill_switch_reason=None,
        daily_trades_remaining=3,
        overall_status="SAFE",
    )


@pytest.fixture
def healthy_turnover():
    """Turnover governor in healthy state."""
    return TurnoverStatus(
        enabled=True,
        symbols_configured=2,
        total_blocks_today=5,
        per_symbol_config={},
    )


@pytest.fixture
def healthy_capital_preservation():
    """Capital preservation in normal state."""
    return CapitalPreservationStatus(
        current_level="NORMAL",
        last_escalation=None,
        restrictions_active=False,
    )


@pytest.fixture
def healthy_daily_reports():
    """Daily reports status for live rollout."""
    return DailyReportsStatus(
        latest_report_timestamp="2026-01-29T10:00:00Z",
        latest_report_age_hours=2.5,
        reports_last_24h=True,
        critical_alerts_14d=0,
    )


@pytest.fixture
def healthy_execution_realism():
    """Execution realism status with no drift."""
    return ExecutionRealismStatus(
        drift_detected=False,
        slippage_7d_avg=0.0015,
        slippage_prior_7d_avg=0.0012,
        slippage_worsened=False,
        available=True,
    )


@pytest.fixture
def live_rollout_healthy_shadow():
    """Shadow status meeting live rollout requirements (14+ day streak)."""
    return ShadowStatus(
        paper_live_decisions_today=15,
        paper_live_decisions_7d=105,
        paper_live_days_streak=14,  # Exactly at live rollout minimum
        paper_live_weeks_counted=2,  # Exactly at live rollout minimum
        heartbeat_recent=1,
        overall_health="HEALTHY",
    )


@pytest.fixture
def live_rollout_healthy_turnover():
    """Turnover status in healthy band for live rollout."""
    return TurnoverStatus(
        enabled=True,
        symbols_configured=2,
        total_blocks_today=10,  # 33% block rate (in healthy 5-70% band)
        total_decisions_today=20,
        block_rate_pct=33.3,
        per_symbol_config={},
    )


# =============================================================================
# Test NO_GO Decision Paths
# =============================================================================


class TestNoGoDecisionPaths:
    """Test all conditions that result in NO_GO status."""

    def test_kill_switch_active_returns_no_go(
        self, healthy_shadow, healthy_turnover, healthy_capital_preservation
    ):
        """Test: kill_switch_active => NO_GO"""
        live = LiveStatus(
            live_mode_enabled=True,
            kill_switch_active=True,
            kill_switch_reason="Emergency market conditions",
            daily_trades_remaining=3,
            overall_status="BLOCKED",
        )

        result = calculate_readiness(
            shadow=healthy_shadow,
            live=live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_NO_GO
        assert any("Kill switch" in r for r in result.reasons)
        assert any("Emergency market conditions" in r for r in result.reasons)
        assert any("deactivate" in a.lower() for a in result.recommended_next_actions)

    def test_capital_preservation_lockdown_returns_no_go(
        self, healthy_shadow, healthy_live, healthy_turnover
    ):
        """Test: capital preservation in LOCKDOWN => NO_GO"""
        capital = CapitalPreservationStatus(
            current_level="LOCKDOWN",
            last_escalation="2026-01-29T10:00:00Z",
            restrictions_active=True,
        )

        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=capital,
        )

        assert result.overall_readiness == READINESS_NO_GO
        assert any("LOCKDOWN" in r for r in result.reasons)
        assert any("NORMAL" in a for a in result.recommended_next_actions)

    def test_capital_preservation_crisis_returns_no_go(
        self, healthy_shadow, healthy_live, healthy_turnover
    ):
        """Test: capital preservation in CRISIS => NO_GO"""
        capital = CapitalPreservationStatus(
            current_level="CRISIS",
            last_escalation="2026-01-29T08:00:00Z",
            restrictions_active=True,
        )

        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=capital,
        )

        assert result.overall_readiness == READINESS_NO_GO
        assert any("CRISIS" in r for r in result.reasons)

    def test_shadow_health_critical_returns_no_go(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: shadow overall_health == CRITICAL => NO_GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=0,
            paper_live_decisions_7d=0,
            paper_live_days_streak=0,
            paper_live_weeks_counted=0,
            heartbeat_recent=0,
            overall_health="CRITICAL",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_NO_GO
        assert any("CRITICAL" in r for r in result.reasons)
        assert any("shadow" in a.lower() for a in result.recommended_next_actions)

    def test_multiple_no_go_conditions_all_reported(
        self, healthy_turnover
    ):
        """Test: Multiple NO_GO conditions are all reported."""
        shadow = ShadowStatus(
            overall_health="CRITICAL",
            heartbeat_recent=0,
        )
        live = LiveStatus(
            kill_switch_active=True,
            kill_switch_reason="Test",
        )
        capital = CapitalPreservationStatus(
            current_level="LOCKDOWN",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=live,
            turnover=healthy_turnover,
            capital_preservation=capital,
        )

        assert result.overall_readiness == READINESS_NO_GO
        # All three issues should be in reasons
        assert any("Kill switch" in r for r in result.reasons)
        assert any("LOCKDOWN" in r for r in result.reasons)
        assert any("CRITICAL" in r for r in result.reasons)


# =============================================================================
# Test CONDITIONAL Decision Paths
# =============================================================================


class TestConditionalDecisionPaths:
    """Test all conditions that result in CONDITIONAL status."""

    def test_heartbeat_not_recent_returns_conditional(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: heartbeat_recent == 0 => CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=10,
            paper_live_decisions_7d=50,
            paper_live_days_streak=10,  # Good streak
            paper_live_weeks_counted=2,
            heartbeat_recent=0,  # Not recent
            overall_health="WARNING",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_CONDITIONAL
        assert any("heartbeat" in r.lower() for r in result.reasons)
        assert any("shadow collector" in a.lower() for a in result.recommended_next_actions)

    def test_insufficient_streak_returns_conditional(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: paper_live_days_streak < 7 => CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=45,
            paper_live_days_streak=3,  # Below 7
            paper_live_weeks_counted=0,
            heartbeat_recent=1,  # Recent heartbeat
            overall_health="HEALTHY",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_CONDITIONAL
        assert any("streak" in r.lower() for r in result.reasons)
        assert any(str(MIN_STREAK_FOR_GO) in r for r in result.reasons)
        # Should recommend continuing trading
        assert any("continue" in a.lower() for a in result.recommended_next_actions)

    def test_streak_exactly_at_boundary_returns_conditional(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: streak exactly at 6 (one below minimum) => CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=90,
            paper_live_days_streak=6,  # Exactly one below minimum (7)
            paper_live_weeks_counted=1,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_CONDITIONAL
        assert any("streak" in r.lower() for r in result.reasons)

    def test_heartbeat_and_streak_both_conditional(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: Both heartbeat and streak issues => CONDITIONAL with both reasons."""
        shadow = ShadowStatus(
            paper_live_decisions_today=5,
            paper_live_decisions_7d=20,
            paper_live_days_streak=2,  # Below minimum
            paper_live_weeks_counted=0,
            heartbeat_recent=0,  # Not recent
            overall_health="WARNING",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_CONDITIONAL
        # Both issues should be mentioned
        assert any("heartbeat" in r.lower() for r in result.reasons)
        assert any("streak" in r.lower() for r in result.reasons)


# =============================================================================
# Test GO Decision Path
# =============================================================================


class TestGoDecisionPath:
    """Test conditions that result in GO status."""

    def test_all_healthy_returns_go(
        self, healthy_shadow, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: All systems healthy => GO"""
        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO
        # Should have positive message
        assert any("operational" in r.lower() or "ready" in a.lower()
                   for r in result.reasons
                   for a in result.recommended_next_actions)

    def test_streak_exactly_at_minimum_returns_go(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: streak exactly at minimum (7) => GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=7,  # Exactly at minimum
            paper_live_weeks_counted=1,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO

    def test_streak_above_minimum_returns_go(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: streak above minimum => GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=20,
            paper_live_decisions_7d=140,
            paper_live_days_streak=14,  # Well above minimum
            paper_live_weeks_counted=2,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO

    def test_warning_health_with_good_metrics_returns_go(
        self, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test: WARNING health but good metrics => GO (with note)."""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=10,
            paper_live_weeks_counted=2,
            heartbeat_recent=1,
            overall_health="WARNING",  # Warning but not blocking
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO
        # Warning should be noted in reasons
        assert any("WARNING" in r for r in result.reasons)


# =============================================================================
# Test Schema Validation
# =============================================================================


class TestSchemaValidation:
    """Test that output matches expected schema."""

    def test_result_has_all_required_fields(
        self, healthy_shadow, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test that result contains all required fields."""
        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert hasattr(result, "overall_readiness")
        assert hasattr(result, "reasons")
        assert hasattr(result, "recommended_next_actions")
        assert hasattr(result, "shadow")
        assert hasattr(result, "live")
        assert hasattr(result, "turnover")
        assert hasattr(result, "capital_preservation")

    def test_to_dict_has_correct_structure(
        self, healthy_shadow, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test that to_dict produces correct structure."""
        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        d = result.to_dict()

        assert "overall_readiness" in d
        assert d["overall_readiness"] in ("GO", "CONDITIONAL", "NO_GO")
        assert "reasons" in d
        assert isinstance(d["reasons"], list)
        assert "recommended_next_actions" in d
        assert isinstance(d["recommended_next_actions"], list)
        assert "components" in d
        assert "shadow" in d["components"]
        assert "live" in d["components"]
        assert "turnover" in d["components"]
        assert "capital_preservation" in d["components"]

    def test_shadow_component_has_required_fields(self, healthy_shadow):
        """Test shadow component has all required fields."""
        d = healthy_shadow.to_dict()

        assert "paper_live_decisions_today" in d
        assert "paper_live_decisions_7d" in d
        assert "paper_live_days_streak" in d
        assert "heartbeat_recent" in d
        assert "overall_health" in d

    def test_live_component_has_required_fields(self, healthy_live):
        """Test live component has all required fields."""
        d = healthy_live.to_dict()

        assert "live_mode_enabled" in d
        assert "kill_switch_active" in d
        assert "daily_trades_remaining" in d
        assert "overall_status" in d

    def test_turnover_component_has_required_fields(self, healthy_turnover):
        """Test turnover component has all required fields."""
        d = healthy_turnover.to_dict()

        assert "enabled" in d
        assert "symbols_configured" in d
        assert "total_blocks_today" in d

    def test_capital_preservation_component_has_required_fields(
        self, healthy_capital_preservation
    ):
        """Test capital preservation component has all required fields."""
        d = healthy_capital_preservation.to_dict()

        assert "current_level" in d
        assert "last_escalation" in d
        assert "restrictions_active" in d


# =============================================================================
# Test Component Data Fetchers
# =============================================================================


class TestComponentFetchers:
    """Test the data fetching functions with mocks."""

    def test_get_shadow_status_handles_exception(self):
        """Test shadow status returns default on exception."""
        from api.readiness_calculator import get_shadow_status

        with patch("api.shadow_health_metrics.get_shadow_health_metrics", side_effect=Exception("Test error")):
            # Should not raise, should return default
            status = get_shadow_status()
            assert status.overall_health == "UNAVAILABLE"

    def test_get_live_status_returns_defaults(self):
        """Test live status returns reasonable defaults."""
        from api.readiness_calculator import get_live_status

        # Just test that it returns valid data
        status = get_live_status()
        assert isinstance(status.live_mode_enabled, bool)
        assert isinstance(status.kill_switch_active, bool)

    def test_get_turnover_status_returns_defaults(self):
        """Test turnover status returns valid data."""
        from api.readiness_calculator import get_turnover_status

        status = get_turnover_status()
        assert isinstance(status.enabled, bool)
        assert isinstance(status.symbols_configured, int)

    def test_get_capital_preservation_returns_defaults(self):
        """Test capital preservation returns valid data."""
        from api.readiness_calculator import get_capital_preservation_status

        status = get_capital_preservation_status()
        assert isinstance(status.current_level, str)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_none_inputs_uses_defaults(self):
        """Test that None inputs trigger fetching."""
        # This will use the actual fetchers, which may return defaults
        result = calculate_readiness()

        assert result.overall_readiness in (READINESS_GO, READINESS_CONDITIONAL, READINESS_NO_GO)
        assert isinstance(result.reasons, list)
        assert isinstance(result.recommended_next_actions, list)

    def test_empty_reasons_for_go_status(
        self, healthy_shadow, healthy_live, healthy_turnover, healthy_capital_preservation
    ):
        """Test that GO status has appropriate reasons."""
        result = calculate_readiness(
            shadow=healthy_shadow,
            live=healthy_live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO
        # Should have at least one reason
        assert len(result.reasons) > 0

    def test_capital_preservation_normal_variations(
        self, healthy_shadow, healthy_live, healthy_turnover
    ):
        """Test various NORMAL level representations."""
        for level in ["NORMAL", "normal", "Normal"]:
            capital = CapitalPreservationStatus(
                current_level=level,
                restrictions_active=False,
            )

            result = calculate_readiness(
                shadow=healthy_shadow,
                live=healthy_live,
                turnover=healthy_turnover,
                capital_preservation=capital,
            )

            # Should not be NO_GO due to capital preservation
            assert result.overall_readiness == READINESS_GO

    def test_live_mode_enabled_noted_in_reasons(
        self, healthy_shadow, healthy_turnover, healthy_capital_preservation
    ):
        """Test that enabled live mode is noted."""
        live = LiveStatus(
            live_mode_enabled=True,
            kill_switch_active=False,
            daily_trades_remaining=3,
            overall_status="LIVE_ACTIVE",
        )

        result = calculate_readiness(
            shadow=healthy_shadow,
            live=live,
            turnover=healthy_turnover,
            capital_preservation=healthy_capital_preservation,
        )

        assert result.overall_readiness == READINESS_GO
        assert any("ENABLED" in r or "enabled" in r.lower() for r in result.reasons)


# =============================================================================
# Test Integration with API Endpoint
# =============================================================================


class TestAPIEndpointIntegration:
    """Test the API endpoint returns correct schema."""

    def test_endpoint_returns_valid_response(self):
        """Test that the API endpoint can be called."""
        from api.api import readiness_check

        response = readiness_check()

        assert response.overall_readiness in ("GO", "CONDITIONAL", "NO_GO")
        assert isinstance(response.reasons, list)
        assert isinstance(response.recommended_next_actions, list)
        assert isinstance(response.components, dict)

    def test_endpoint_components_have_expected_keys(self):
        """Test that components have expected keys."""
        from api.api import readiness_check

        response = readiness_check()

        assert "shadow" in response.components
        assert "live" in response.components
        assert "turnover" in response.components
        assert "capital_preservation" in response.components


# =============================================================================
# Test Priority of Decisions
# =============================================================================


class TestDecisionPriority:
    """Test that NO_GO takes priority over CONDITIONAL."""

    def test_no_go_takes_priority_over_conditional(self):
        """Test that NO_GO conditions override CONDITIONAL conditions."""
        shadow = ShadowStatus(
            paper_live_decisions_today=0,
            paper_live_decisions_7d=0,
            paper_live_days_streak=0,  # Would be CONDITIONAL
            heartbeat_recent=0,  # Would be CONDITIONAL
            overall_health="CRITICAL",  # NO_GO
        )
        live = LiveStatus(
            kill_switch_active=False,
        )
        capital = CapitalPreservationStatus(
            current_level="NORMAL",
        )
        turnover = TurnoverStatus()

        result = calculate_readiness(
            shadow=shadow,
            live=live,
            turnover=turnover,
            capital_preservation=capital,
        )

        # Should be NO_GO due to CRITICAL, not CONDITIONAL
        assert result.overall_readiness == READINESS_NO_GO

    def test_kill_switch_takes_priority_over_all(self):
        """Test that kill switch is checked first."""
        shadow = ShadowStatus(
            overall_health="HEALTHY",
            heartbeat_recent=1,
            paper_live_days_streak=10,
        )
        live = LiveStatus(
            kill_switch_active=True,
            kill_switch_reason="Test",
        )
        capital = CapitalPreservationStatus(
            current_level="NORMAL",
        )
        turnover = TurnoverStatus()

        result = calculate_readiness(
            shadow=shadow,
            live=live,
            turnover=turnover,
            capital_preservation=capital,
        )

        assert result.overall_readiness == READINESS_NO_GO
        # Kill switch should be the first/primary reason
        assert "Kill switch" in result.reasons[0]


# =============================================================================
# Test Live Rollout NO_GO Decision Paths
# =============================================================================


class TestLiveRolloutNoGoDecisionPaths:
    """Test all conditions that result in live rollout NO_GO status."""

    def test_kill_switch_active_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: kill_switch_active => live_rollout NO_GO"""
        live = LiveStatus(
            live_mode_enabled=False,
            kill_switch_active=True,
            kill_switch_reason="Emergency market conditions",
            daily_trades_remaining=3,
            overall_status="BLOCKED",
            symbol_allowlist=["ETH/USDT"],
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("Kill switch" in r for r in result.reasons)
        assert any("Emergency market conditions" in r for r in result.reasons)
        assert any("deactivate" in a.lower() for a in result.next_actions)

    def test_capital_preservation_lockdown_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: capital preservation in LOCKDOWN => live_rollout NO_GO"""
        capital = CapitalPreservationStatus(
            current_level="LOCKDOWN",
            last_escalation="2026-01-29T10:00:00Z",
            restrictions_active=True,
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=capital,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("LOCKDOWN" in r for r in result.reasons)
        assert any("NORMAL" in a for a in result.next_actions)

    def test_capital_preservation_crisis_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: capital preservation in CRISIS => live_rollout NO_GO"""
        capital = CapitalPreservationStatus(
            current_level="CRISIS",
            last_escalation="2026-01-29T08:00:00Z",
            restrictions_active=True,
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=capital,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("CRISIS" in r for r in result.reasons)

    def test_shadow_health_critical_returns_live_rollout_no_go(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: shadow overall_health == CRITICAL => live_rollout NO_GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=0,
            paper_live_decisions_7d=0,
            paper_live_days_streak=0,
            paper_live_weeks_counted=0,
            heartbeat_recent=0,
            overall_health="CRITICAL",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("CRITICAL" in r for r in result.reasons)
        assert any("shadow" in a.lower() for a in result.next_actions)

    def test_live_mode_enabled_empty_allowlist_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: live_mode_enabled with empty allowlist => live_rollout NO_GO (misconfiguration)"""
        live = LiveStatus(
            live_mode_enabled=True,  # Enabled
            kill_switch_active=False,
            kill_switch_reason=None,
            daily_trades_remaining=3,
            overall_status="SAFE",
            symbol_allowlist=[],  # Empty allowlist - misconfiguration!
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("allowlist" in r.lower() for r in result.reasons)
        assert any("misconfiguration" in r.lower() for r in result.reasons)
        assert any("LIVE_SYMBOL_ALLOWLIST" in a or "LIVE_MODE" in a for a in result.next_actions)

    def test_daily_reports_missing_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_execution_realism,
    ):
        """Test: daily health reports missing in last 24h => live_rollout NO_GO"""
        daily_reports = DailyReportsStatus(
            latest_report_timestamp="2026-01-27T10:00:00Z",  # 48+ hours ago
            latest_report_age_hours=48.0,
            reports_last_24h=False,  # Missing!
            critical_alerts_14d=0,
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("24h" in r for r in result.reasons)
        assert any("report" in r.lower() for r in result.reasons)
        assert any("cron" in a.lower() or "report" in a.lower() for a in result.next_actions)

    def test_daily_reports_never_generated_returns_live_rollout_no_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_execution_realism,
    ):
        """Test: no daily reports ever generated => live_rollout NO_GO"""
        daily_reports = DailyReportsStatus(
            latest_report_timestamp=None,
            latest_report_age_hours=999.0,
            reports_last_24h=False,
            critical_alerts_14d=0,
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_NO_GO
        assert any("never" in r.lower() or "24h" in r for r in result.reasons)


# =============================================================================
# Test Live Rollout CONDITIONAL Decision Paths
# =============================================================================


class TestLiveRolloutConditionalDecisionPaths:
    """Test all conditions that result in live rollout CONDITIONAL status."""

    def test_streak_below_14_returns_live_rollout_conditional(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: paper_live_days_streak < 14 => live_rollout CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=10,  # Below 14 day minimum
            paper_live_weeks_counted=2,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("streak" in r.lower() for r in result.reasons)
        assert any(str(LIVE_ROLLOUT_MIN_STREAK) in r for r in result.reasons)
        assert any("4 more" in a for a in result.next_actions)  # 14 - 10 = 4

    def test_weeks_below_2_returns_live_rollout_conditional(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: paper_live_weeks_counted < 2 => live_rollout CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=14,  # Good streak
            paper_live_weeks_counted=1,  # Below 2 week minimum
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("weeks" in r.lower() for r in result.reasons)
        assert any(str(LIVE_ROLLOUT_MIN_WEEKS) in r for r in result.reasons)

    def test_heartbeat_not_recent_returns_live_rollout_conditional(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: heartbeat_recent == 0 => live_rollout CONDITIONAL"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=14,
            paper_live_weeks_counted=2,
            heartbeat_recent=0,  # Not recent
            overall_health="WARNING",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("heartbeat" in r.lower() for r in result.reasons)
        assert any("shadow collector" in a.lower() for a in result.next_actions)

    def test_turnover_block_rate_above_70_returns_live_rollout_conditional(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: turnover block rate > 70% => live_rollout CONDITIONAL (over-throttling)"""
        turnover = TurnoverStatus(
            enabled=True,
            symbols_configured=2,
            total_blocks_today=80,  # 80% block rate
            total_decisions_today=20,
            block_rate_pct=80.0,  # Above 70% max
            per_symbol_config={},
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("80.0%" in r or "block rate" in r.lower() for r in result.reasons)
        assert any("over-throttling" in r.lower() for r in result.reasons)
        assert any("too strict" in a.lower() for a in result.next_actions)

    def test_turnover_block_rate_below_5_returns_live_rollout_conditional(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: turnover block rate < 5% => live_rollout CONDITIONAL (too loose)"""
        turnover = TurnoverStatus(
            enabled=True,
            symbols_configured=2,
            total_blocks_today=1,  # 2% block rate
            total_decisions_today=49,
            block_rate_pct=2.0,  # Below 5% min
            per_symbol_config={},
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("2.0%" in r or "block rate" in r.lower() for r in result.reasons)
        assert any("too loose" in r.lower() for r in result.reasons)
        assert any("tightening" in a.lower() for a in result.next_actions)

    def test_execution_realism_drift_returns_live_rollout_conditional(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
    ):
        """Test: execution realism drift detected => live_rollout CONDITIONAL"""
        execution_realism = ExecutionRealismStatus(
            drift_detected=True,
            slippage_7d_avg=0.0025,  # Significantly worse
            slippage_prior_7d_avg=0.0012,
            slippage_worsened=True,
            available=True,
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("drift" in r.lower() or "realism" in r.lower() for r in result.reasons)
        assert any("slippage" in r.lower() for r in result.reasons)
        assert any("investigate" in a.lower() for a in result.next_actions)

    def test_critical_alerts_14d_returns_live_rollout_conditional(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_execution_realism,
    ):
        """Test: CRITICAL alerts in last 14 days => live_rollout CONDITIONAL"""
        daily_reports = DailyReportsStatus(
            latest_report_timestamp="2026-01-29T10:00:00Z",
            latest_report_age_hours=2.5,
            reports_last_24h=True,
            critical_alerts_14d=2,  # Had 2 critical alerts
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        assert any("CRITICAL" in r for r in result.reasons)
        assert any("14 days" in r for r in result.reasons)
        assert any("resolve" in a.lower() for a in result.next_actions)

    def test_multiple_conditional_issues_all_reported(
        self,
        healthy_live,
        healthy_capital_preservation,
        healthy_daily_reports,
    ):
        """Test: Multiple CONDITIONAL issues are all reported."""
        shadow = ShadowStatus(
            paper_live_decisions_today=10,
            paper_live_decisions_7d=50,
            paper_live_days_streak=10,  # Below 14
            paper_live_weeks_counted=1,  # Below 2
            heartbeat_recent=0,  # Not recent
            overall_health="WARNING",
        )
        turnover = TurnoverStatus(
            enabled=True,
            symbols_configured=2,
            total_blocks_today=85,
            total_decisions_today=15,
            block_rate_pct=85.0,  # Above 70%
            per_symbol_config={},
        )
        execution_realism = ExecutionRealismStatus(
            drift_detected=True,
            slippage_7d_avg=0.003,
            slippage_prior_7d_avg=0.001,
            slippage_worsened=True,
            available=True,
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=execution_realism,
        )

        assert result.readiness == READINESS_CONDITIONAL
        # All issues should be in reasons
        assert any("streak" in r.lower() for r in result.reasons)
        assert any("weeks" in r.lower() for r in result.reasons)
        assert any("heartbeat" in r.lower() for r in result.reasons)
        assert any("block rate" in r.lower() for r in result.reasons)
        assert any("drift" in r.lower() or "realism" in r.lower() for r in result.reasons)


# =============================================================================
# Test Live Rollout GO Decision Path
# =============================================================================


class TestLiveRolloutGoDecisionPath:
    """Test conditions that result in live rollout GO status."""

    def test_all_criteria_met_returns_live_rollout_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: All live rollout criteria met => GO"""
        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO
        assert any("criteria met" in r.lower() for r in result.reasons)
        assert any("April 1st" in a or "ready" in a.lower() for a in result.next_actions)

    def test_streak_exactly_at_14_returns_live_rollout_go(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: streak exactly at 14 days => GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=105,
            paper_live_days_streak=14,  # Exactly at minimum
            paper_live_weeks_counted=2,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO

    def test_streak_above_14_returns_live_rollout_go(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: streak above 14 days => GO"""
        shadow = ShadowStatus(
            paper_live_decisions_today=20,
            paper_live_decisions_7d=140,
            paper_live_days_streak=21,  # Well above minimum
            paper_live_weeks_counted=3,
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO

    def test_turnover_block_rate_at_5_percent_returns_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: turnover block rate at exactly 5% => GO (at lower boundary)"""
        turnover = TurnoverStatus(
            enabled=True,
            symbols_configured=2,
            total_blocks_today=5,
            total_decisions_today=95,
            block_rate_pct=5.0,  # Exactly at lower boundary
            per_symbol_config={},
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO

    def test_turnover_block_rate_at_70_percent_returns_go(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: turnover block rate at exactly 70% => GO (at upper boundary)"""
        turnover = TurnoverStatus(
            enabled=True,
            symbols_configured=2,
            total_blocks_today=70,
            total_decisions_today=30,
            block_rate_pct=70.0,  # Exactly at upper boundary
            per_symbol_config={},
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO

    def test_live_mode_enabled_with_valid_allowlist_returns_go(
        self,
        live_rollout_healthy_shadow,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: live_mode_enabled with valid allowlist => GO (not misconfigured)"""
        live = LiveStatus(
            live_mode_enabled=True,  # Enabled
            kill_switch_active=False,
            kill_switch_reason=None,
            daily_trades_remaining=3,
            overall_status="LIVE_ACTIVE",
            symbol_allowlist=["ETH/USDT", "BTC/USDT"],  # Valid non-empty allowlist
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert result.readiness == READINESS_GO

    def test_execution_realism_unavailable_does_not_block(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
    ):
        """Test: execution realism unavailable does not cause CONDITIONAL."""
        execution_realism = ExecutionRealismStatus(
            drift_detected=False,
            slippage_7d_avg=0.0,
            slippage_prior_7d_avg=0.0,
            slippage_worsened=False,
            available=False,  # Data not available - should not block
        )

        result = calculate_live_rollout_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=execution_realism,
        )

        assert result.readiness == READINESS_GO


# =============================================================================
# Test Live Rollout Integration with Main Readiness
# =============================================================================


class TestLiveRolloutIntegration:
    """Test that live rollout is properly integrated with main readiness."""

    def test_calculate_readiness_includes_live_rollout(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test that calculate_readiness includes live_rollout result."""
        result = calculate_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        assert hasattr(result, "live_rollout")
        assert result.live_rollout.readiness in (READINESS_GO, READINESS_CONDITIONAL, READINESS_NO_GO)
        assert isinstance(result.live_rollout.reasons, list)
        assert isinstance(result.live_rollout.next_actions, list)

    def test_to_dict_includes_live_rollout_fields(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test that to_dict includes live_rollout fields."""
        result = calculate_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        d = result.to_dict()

        assert "live_rollout_readiness" in d
        assert d["live_rollout_readiness"] in ("GO", "CONDITIONAL", "NO_GO")
        assert "live_rollout_reasons" in d
        assert isinstance(d["live_rollout_reasons"], list)
        assert "live_rollout_next_actions" in d
        assert isinstance(d["live_rollout_next_actions"], list)

    def test_components_include_new_statuses(
        self,
        live_rollout_healthy_shadow,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test that components include daily_reports and execution_realism."""
        result = calculate_readiness(
            shadow=live_rollout_healthy_shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        d = result.to_dict()

        assert "daily_reports" in d["components"]
        assert "execution_realism" in d["components"]

    def test_api_endpoint_returns_live_rollout_fields(self):
        """Test that API endpoint returns live_rollout fields."""
        from api.api import readiness_check

        response = readiness_check()

        assert hasattr(response, "live_rollout_readiness")
        assert response.live_rollout_readiness in ("GO", "CONDITIONAL", "NO_GO")
        assert hasattr(response, "live_rollout_reasons")
        assert isinstance(response.live_rollout_reasons, list)
        assert hasattr(response, "live_rollout_next_actions")
        assert isinstance(response.live_rollout_next_actions, list)

    def test_overall_go_but_live_rollout_conditional(
        self,
        healthy_live,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_daily_reports,
        healthy_execution_realism,
    ):
        """Test: overall GO but live_rollout CONDITIONAL (stricter requirements)."""
        # Shadow has 7 day streak (enough for general GO) but not 14 (needed for live rollout)
        shadow = ShadowStatus(
            paper_live_decisions_today=15,
            paper_live_decisions_7d=100,
            paper_live_days_streak=7,  # Exactly at general minimum
            paper_live_weeks_counted=1,  # Below live rollout minimum
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_readiness(
            shadow=shadow,
            live=healthy_live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=healthy_daily_reports,
            execution_realism=healthy_execution_realism,
        )

        # General readiness should be GO (7 day streak is enough)
        assert result.overall_readiness == READINESS_GO
        # Live rollout should be CONDITIONAL (needs 14 days and 2 weeks)
        assert result.live_rollout.readiness == READINESS_CONDITIONAL
        assert any("14" in r for r in result.live_rollout.reasons)


# =============================================================================
# Test Live Rollout Priority of Decisions
# =============================================================================


class TestLiveRolloutDecisionPriority:
    """Test that live rollout NO_GO takes priority over CONDITIONAL."""

    def test_live_rollout_no_go_returns_early(
        self,
        live_rollout_healthy_shadow,
        live_rollout_healthy_turnover,
        healthy_capital_preservation,
        healthy_execution_realism,
    ):
        """Test that NO_GO returns early without checking CONDITIONAL conditions."""
        # Kill switch (NO_GO) + missing reports (NO_GO) + low streak (CONDITIONAL)
        live = LiveStatus(
            live_mode_enabled=False,
            kill_switch_active=True,
            kill_switch_reason="Test",
            symbol_allowlist=["ETH/USDT"],
        )
        daily_reports = DailyReportsStatus(
            latest_report_age_hours=48.0,
            reports_last_24h=False,  # Also NO_GO
            critical_alerts_14d=0,
        )
        shadow = ShadowStatus(
            paper_live_days_streak=10,  # Would be CONDITIONAL
            paper_live_weeks_counted=1,  # Would be CONDITIONAL
            heartbeat_recent=1,
            overall_health="HEALTHY",
        )

        result = calculate_live_rollout_readiness(
            shadow=shadow,
            live=live,
            turnover=live_rollout_healthy_turnover,
            capital_preservation=healthy_capital_preservation,
            daily_reports=daily_reports,
            execution_realism=healthy_execution_realism,
        )

        # Should be NO_GO
        assert result.readiness == READINESS_NO_GO
        # NO_GO reasons should be present
        assert any("Kill switch" in r for r in result.reasons)
        # CONDITIONAL reasons should NOT be present (early return)
        assert not any("streak" in r.lower() and "14" in r for r in result.reasons)
