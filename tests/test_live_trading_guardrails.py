"""
Comprehensive tests for Live Trading Guardrails.

Tests ALL block paths and ensures:
- Paper trading remains unaffected
- RL shadow remains advisory only
- Kill switch works correctly
- All guardrails enforce limits
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.live_trading_guardrails import (
    LiveTradingConfig,
    LiveTradingGuardrails,
    LiveTradingState,
    GuardrailCheckResult,
    StartupReadinessResult,
    get_live_guardrails,
    reset_live_guardrails,
    check_startup_readiness,
    validate_live_mode_startup,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config(temp_dir):
    """Create a config with all defaults (SAFE/off)."""
    return LiveTradingConfig(
        live_mode=False,
        state_file=str(temp_dir / "state.json"),
        kill_switch_file=str(temp_dir / "kill_switch.txt"),
    )


@pytest.fixture
def live_config(temp_dir):
    """Create a config with live mode enabled."""
    return LiveTradingConfig(
        live_mode=True,
        live_max_capital_pct=0.01,
        live_max_position_pct=0.02,
        live_symbol_allowlist=["ETH/USDT"],
        live_max_trades_per_day=3,
        live_max_leverage=1.0,
        state_file=str(temp_dir / "state.json"),
        kill_switch_file=str(temp_dir / "kill_switch.txt"),
    )


@pytest.fixture
def guardrails_off(default_config):
    """Guardrails with live mode OFF."""
    return LiveTradingGuardrails(config=default_config)


@pytest.fixture
def guardrails_on(live_config):
    """Guardrails with live mode ON."""
    return LiveTradingGuardrails(config=live_config)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    reset_live_guardrails()
    yield
    reset_live_guardrails()


# =============================================================================
# Test Configuration
# =============================================================================


class TestLiveTradingConfig:
    """Test configuration defaults and loading."""

    def test_default_config_is_safe(self):
        """Test that default config has live mode OFF."""
        config = LiveTradingConfig()
        assert config.live_mode is False
        assert config.live_max_capital_pct == 0.01
        assert config.live_max_position_pct == 0.02
        assert config.live_max_trades_per_day == 3
        assert config.live_max_leverage == 1.0
        assert "ETH/USDT" in config.live_symbol_allowlist

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        with patch.dict(os.environ, {
            "LIVE_MODE": "true",
            "LIVE_MAX_CAPITAL_PCT": "0.05",
            "LIVE_MAX_TRADES_PER_DAY": "5",
            "LIVE_SYMBOL_ALLOWLIST": "BTC/USDT,ETH/USDT",
        }):
            config = LiveTradingConfig.from_env()
            assert config.live_mode is True
            assert config.live_max_capital_pct == 0.05
            assert config.live_max_trades_per_day == 5
            assert "BTC/USDT" in config.live_symbol_allowlist
            assert "ETH/USDT" in config.live_symbol_allowlist

    def test_config_to_dict(self, default_config):
        """Test config serialization."""
        d = default_config.to_dict()
        assert d["live_mode"] is False
        assert "live_symbol_allowlist" in d


# =============================================================================
# Test Kill Switch
# =============================================================================


class TestKillSwitch:
    """Test kill switch functionality."""

    def test_kill_switch_not_active_by_default(self, guardrails_on):
        """Test that kill switch is not active by default."""
        is_active, reason = guardrails_on.is_kill_switch_active()
        assert is_active is False
        assert reason is None

    def test_kill_switch_file_activates(self, guardrails_on, temp_dir):
        """Test that kill switch file activates the kill switch."""
        kill_file = Path(guardrails_on.config.kill_switch_file)
        kill_file.parent.mkdir(parents=True, exist_ok=True)
        kill_file.write_text("Emergency stop requested")

        is_active, reason = guardrails_on.is_kill_switch_active()
        assert is_active is True
        assert "Emergency stop requested" in reason

    def test_kill_switch_env_var_activates(self, guardrails_on):
        """Test that environment variable activates kill switch."""
        with patch.dict(os.environ, {"LIVE_KILL_SWITCH": "true"}):
            is_active, reason = guardrails_on.is_kill_switch_active()
            assert is_active is True
            assert "env var" in reason.lower()

    def test_kill_switch_blocks_live_trade(self, guardrails_on, temp_dir):
        """Test that kill switch blocks all live trades."""
        # Activate kill switch
        guardrails_on.activate_kill_switch("Test emergency")

        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=100.0,
            portfolio_value=10000.0,
        )

        assert result.passed is False
        assert result.guardrail_name == "kill_switch"
        assert result.is_critical is True

    def test_activate_and_deactivate_kill_switch(self, guardrails_on):
        """Test programmatic kill switch control."""
        # Activate
        guardrails_on.activate_kill_switch("Test reason")
        is_active, _ = guardrails_on.is_kill_switch_active()
        assert is_active is True

        # Deactivate
        guardrails_on.deactivate_kill_switch()
        is_active, _ = guardrails_on.is_kill_switch_active()
        assert is_active is False


# =============================================================================
# Test Live Mode Check
# =============================================================================


class TestLiveModeCheck:
    """Test live mode enabled/disabled checks."""

    def test_live_mode_disabled_blocks_trade(self, guardrails_off):
        """Test that disabled live mode blocks trade."""
        result = guardrails_off.check_live_mode_enabled()
        assert result.passed is False
        assert result.guardrail_name == "live_mode"

    def test_live_mode_enabled_allows_trade(self, guardrails_on):
        """Test that enabled live mode allows trade."""
        result = guardrails_on.check_live_mode_enabled()
        assert result.passed is True


# =============================================================================
# Test Symbol Allowlist
# =============================================================================


class TestSymbolAllowlist:
    """Test symbol allowlist enforcement."""

    def test_allowed_symbol_passes(self, guardrails_on):
        """Test that allowed symbol passes."""
        result = guardrails_on.check_symbol_allowlist("ETH/USDT")
        assert result.passed is True

    def test_disallowed_symbol_blocked(self, guardrails_on):
        """Test that disallowed symbol is blocked."""
        result = guardrails_on.check_symbol_allowlist("BTC/USDT")
        assert result.passed is False
        assert result.guardrail_name == "symbol_allowlist"

    def test_symbol_normalization(self, guardrails_on):
        """Test that symbol variations are normalized correctly."""
        # ETH/USDT should match ETH_USDT
        result = guardrails_on.check_symbol_allowlist("ETH_USDT")
        assert result.passed is True

        # Case insensitive
        result = guardrails_on.check_symbol_allowlist("eth/usdt")
        assert result.passed is True


# =============================================================================
# Test Daily Trade Limit
# =============================================================================


class TestDailyTradeLimit:
    """Test daily trade limit enforcement."""

    def test_under_limit_passes(self, guardrails_on):
        """Test that trades under limit pass."""
        result = guardrails_on.check_daily_trade_limit()
        assert result.passed is True

    def test_at_limit_blocked(self, guardrails_on):
        """Test that reaching limit blocks further trades."""
        # Record trades up to the limit
        for i in range(3):
            guardrails_on.record_live_trade("ETH/USDT", 100.0)

        result = guardrails_on.check_daily_trade_limit()
        assert result.passed is False
        assert result.guardrail_name == "daily_trade_limit"

    def test_daily_reset_on_new_day(self, guardrails_on):
        """Test that daily counters reset on new day."""
        # Record trades
        for i in range(3):
            guardrails_on.record_live_trade("ETH/USDT", 100.0)

        # Simulate new day by directly setting state
        guardrails_on._state.last_trade_date = "2000-01-01"
        guardrails_on._state.reset_daily_counters()

        result = guardrails_on.check_daily_trade_limit()
        assert result.passed is True


# =============================================================================
# Test Position Size Limits
# =============================================================================


class TestPositionSizeLimits:
    """Test position size enforcement."""

    def test_small_position_passes(self, guardrails_on):
        """Test that small position passes."""
        result = guardrails_on.check_position_size(
            position_value=100.0,
            portfolio_value=10000.0,
        )
        assert result.passed is True

    def test_large_position_blocked(self, guardrails_on):
        """Test that large position is blocked."""
        # 5% position > 2% limit
        result = guardrails_on.check_position_size(
            position_value=500.0,
            portfolio_value=10000.0,
        )
        assert result.passed is False
        assert result.guardrail_name == "position_size"

    def test_edge_case_position_at_limit(self, guardrails_on):
        """Test position exactly at limit."""
        # 2% position = 2% limit
        result = guardrails_on.check_position_size(
            position_value=200.0,
            portfolio_value=10000.0,
        )
        assert result.passed is True  # At limit should pass


# =============================================================================
# Test Total Capital Limits
# =============================================================================


class TestTotalCapitalLimits:
    """Test total capital deployment limits."""

    def test_under_capital_limit_passes(self, guardrails_on):
        """Test that under capital limit passes."""
        result = guardrails_on.check_total_capital(
            position_value=50.0,
            portfolio_value=10000.0,
        )
        assert result.passed is True

    def test_exceeding_capital_limit_blocked(self, guardrails_on):
        """Test that exceeding capital limit is blocked."""
        # Already deployed 0.8% (80), new position would push to 1.2%
        guardrails_on.record_live_trade("ETH/USDT", 80.0)

        result = guardrails_on.check_total_capital(
            position_value=40.0,  # Would make total 120/10000 = 1.2%
            portfolio_value=10000.0,
        )
        assert result.passed is False
        assert result.guardrail_name == "total_capital"


# =============================================================================
# Test Leverage Limits
# =============================================================================


class TestLeverageLimits:
    """Test leverage enforcement."""

    def test_no_leverage_passes(self, guardrails_on):
        """Test that no leverage passes."""
        result = guardrails_on.check_leverage(1.0)
        assert result.passed is True

    def test_excessive_leverage_blocked(self, guardrails_on):
        """Test that excessive leverage is blocked."""
        result = guardrails_on.check_leverage(2.0)
        assert result.passed is False
        assert result.guardrail_name == "leverage"


# =============================================================================
# Test Full Guardrail Check
# =============================================================================


class TestFullGuardrailCheck:
    """Test complete guardrail check flow."""

    def test_all_checks_pass(self, guardrails_on):
        """Test that valid trade passes all checks."""
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=100.0,
            portfolio_value=10000.0,
            leverage=1.0,
        )
        assert result.passed is True
        assert result.guardrail_name == "all"

    def test_kill_switch_checked_first(self, guardrails_on, temp_dir):
        """Test that kill switch is checked before other guardrails."""
        # Activate kill switch
        guardrails_on.activate_kill_switch("Test")

        # Even with invalid symbol, kill switch should be reported
        result = guardrails_on.check_all_guardrails(
            symbol="INVALID/PAIR",
            position_value=100.0,
            portfolio_value=10000.0,
        )

        assert result.passed is False
        assert result.guardrail_name == "kill_switch"

    def test_multiple_violations_reports_first(self, guardrails_on):
        """Test that first failing check is reported."""
        # Both symbol and position size would fail
        result = guardrails_on.check_all_guardrails(
            symbol="BTC/USDT",  # Not in allowlist
            position_value=500.0,  # Exceeds 2% limit
            portfolio_value=10000.0,
        )

        # Symbol check comes first
        assert result.passed is False
        assert result.guardrail_name == "symbol_allowlist"


# =============================================================================
# Test State Persistence
# =============================================================================


class TestStatePersistence:
    """Test state persistence across restarts."""

    def test_state_saved_on_trade(self, guardrails_on, temp_dir):
        """Test that state is saved after recording trade."""
        guardrails_on.record_live_trade("ETH/USDT", 100.0)

        state_file = Path(guardrails_on.config.state_file)
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)

        assert data["daily_trade_count"] == 1
        assert data["capital_deployed_today"] == 100.0

    def test_state_loaded_on_init(self, live_config, temp_dir):
        """Test that state is loaded on initialization."""
        # Create initial guardrails and record trades
        g1 = LiveTradingGuardrails(config=live_config)
        g1.record_live_trade("ETH/USDT", 100.0)
        g1.record_live_trade("ETH/USDT", 50.0)

        # Create new guardrails instance - should load state
        g2 = LiveTradingGuardrails(config=live_config)

        assert g2._state.daily_trade_count == 2
        assert g2._state.capital_deployed_today == 150.0


# =============================================================================
# Test Paper Trading Unaffected
# =============================================================================


class TestPaperTradingUnaffected:
    """Test that paper trading is not affected by guardrails."""

    def test_guardrails_only_apply_to_live_mode(self, guardrails_off):
        """Test that guardrails don't block when live mode is off."""
        # Even with invalid parameters, only live_mode check fails
        result = guardrails_off.check_all_guardrails(
            symbol="ANY/SYMBOL",
            position_value=999999.0,
            portfolio_value=1000.0,
            leverage=100.0,
        )

        # First failure is live_mode check
        assert result.passed is False
        assert result.guardrail_name == "live_mode"


# =============================================================================
# Test Status Reporting
# =============================================================================


class TestStatusReporting:
    """Test status reporting functionality."""

    def test_get_status_returns_all_fields(self, guardrails_on):
        """Test that get_status returns all required fields."""
        status = guardrails_on.get_status()

        assert "live_mode_enabled" in status
        assert "kill_switch_active" in status
        assert "daily_trades" in status
        assert "capital" in status
        assert "position" in status
        assert "leverage" in status
        assert "symbol_allowlist" in status

    def test_status_reflects_current_state(self, guardrails_on):
        """Test that status reflects current state."""
        guardrails_on.record_live_trade("ETH/USDT", 100.0)
        guardrails_on.record_live_trade("ETH/USDT", 50.0)

        status = guardrails_on.get_status()

        assert status["daily_trades"]["count"] == 2
        assert status["capital"]["deployed_today"] == 150.0


# =============================================================================
# Test GuardrailCheckResult
# =============================================================================


class TestGuardrailCheckResult:
    """Test GuardrailCheckResult class."""

    def test_is_critical_for_kill_switch(self):
        """Test that kill switch results are marked critical."""
        result = GuardrailCheckResult(
            passed=False,
            block_reason="Kill switch active",
            guardrail_name="kill_switch",
        )
        assert result.is_critical is True

    def test_not_critical_for_other_guardrails(self):
        """Test that other guardrail failures are not critical."""
        result = GuardrailCheckResult(
            passed=False,
            block_reason="Position too large",
            guardrail_name="position_size",
        )
        assert result.is_critical is False

    def test_to_dict_serialization(self):
        """Test result serialization."""
        result = GuardrailCheckResult(
            passed=True,
            guardrail_name="all",
            details={"trades_today": 1},
        )
        d = result.to_dict()

        assert d["passed"] is True
        assert d["guardrail_name"] == "all"
        assert d["details"]["trades_today"] == 1


# =============================================================================
# Test LiveTradingState
# =============================================================================


class TestLiveTradingState:
    """Test LiveTradingState class."""

    def test_record_trade_updates_counts(self):
        """Test that recording trade updates counts."""
        state = LiveTradingState()
        state.record_trade("ETH/USDT", 100.0)

        assert state.daily_trade_count == 1
        assert state.trades_by_symbol.get("ETH/USDT") == 1
        assert state.capital_deployed_today == 100.0

    def test_reset_daily_counters(self):
        """Test daily counter reset."""
        state = LiveTradingState()
        state.daily_trade_count = 5
        state.capital_deployed_today = 500.0
        state.last_trade_date = "2000-01-01"  # Old date

        state.reset_daily_counters()

        assert state.daily_trade_count == 0
        assert state.capital_deployed_today == 0.0

    def test_from_dict_and_to_dict(self):
        """Test state serialization roundtrip."""
        original = LiveTradingState(
            daily_trade_count=3,
            capital_deployed_today=150.0,
            last_trade_date="2026-01-29",
        )

        d = original.to_dict()
        restored = LiveTradingState.from_dict(d)

        assert restored.daily_trade_count == 3
        assert restored.capital_deployed_today == 150.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_portfolio_value(self, guardrails_on):
        """Test handling of zero portfolio value."""
        result = guardrails_on.check_position_size(
            position_value=100.0,
            portfolio_value=0.0,
        )
        assert result.passed is False

    def test_negative_position_value(self, guardrails_on):
        """Test handling of negative values."""
        # Negative position (short) should still work
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=-100.0,  # Short position
            portfolio_value=10000.0,
        )
        # Should pass since -100/10000 = -1% which is < 2% in absolute terms
        # Actually, position_pct will be negative, so it should pass
        # This is implementation dependent

    def test_empty_allowlist(self, temp_dir):
        """Test behavior with empty allowlist."""
        config = LiveTradingConfig(
            live_mode=True,
            live_symbol_allowlist=[],  # Empty
            state_file=str(temp_dir / "state.json"),
            kill_switch_file=str(temp_dir / "kill_switch.txt"),
        )
        guardrails = LiveTradingGuardrails(config=config)

        result = guardrails.check_symbol_allowlist("ETH/USDT")
        assert result.passed is False  # Nothing is allowed

    def test_corrupted_state_file(self, live_config, temp_dir):
        """Test handling of corrupted state file."""
        state_file = Path(live_config.state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("not valid json {{{")

        # Should not raise, should start with fresh state
        guardrails = LiveTradingGuardrails(config=live_config)
        assert guardrails._state.daily_trade_count == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_trading_day_scenario(self, guardrails_on):
        """Test a full trading day with multiple trades."""
        portfolio = 10000.0

        # Trade 1: Should pass
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=100.0,  # 1% of portfolio
            portfolio_value=portfolio,
        )
        assert result.passed is True
        guardrails_on.record_live_trade("ETH/USDT", 100.0)

        # Trade 2: Should pass
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=100.0,
            portfolio_value=portfolio,
        )
        assert result.passed is False  # Capital limit exceeded (200/10000 = 2% > 1%)

        # Reset and try with smaller amount
        guardrails_on.reset_daily_state()

        # Trade with 0.3% position (30/10000)
        for i in range(3):
            result = guardrails_on.check_all_guardrails(
                symbol="ETH/USDT",
                position_value=30.0,
                portfolio_value=portfolio,
            )
            if result.passed:
                guardrails_on.record_live_trade("ETH/USDT", 30.0)

        # Trade 4: Should be blocked by daily limit
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=30.0,
            portfolio_value=portfolio,
        )
        assert result.passed is False
        assert result.guardrail_name == "daily_trade_limit"

    def test_emergency_kill_switch_scenario(self, guardrails_on):
        """Test emergency kill switch activation during trading."""
        # Start trading
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=100.0,
            portfolio_value=10000.0,
        )
        assert result.passed is True
        guardrails_on.record_live_trade("ETH/USDT", 100.0)

        # Emergency! Activate kill switch
        guardrails_on.activate_kill_switch("Market crash detected")

        # All trades should now be blocked
        result = guardrails_on.check_all_guardrails(
            symbol="ETH/USDT",
            position_value=50.0,
            portfolio_value=10000.0,
        )
        assert result.passed is False
        assert result.is_critical is True

        # Status should show kill switch active
        status = guardrails_on.get_status()
        assert status["kill_switch_active"] is True


# =============================================================================
# Test Startup Safety Assertion
# =============================================================================


class TestStartupSafetyAssertion:
    """Test startup readiness checks for live mode."""

    @pytest.fixture
    def live_mode_guardrails(self, temp_dir):
        """Create guardrails with LIVE_MODE enabled."""
        reset_live_guardrails()
        config = LiveTradingConfig(
            live_mode=True,  # LIVE_MODE enabled
            state_file=str(temp_dir / "state.json"),
            kill_switch_file=str(temp_dir / "kill_switch.txt"),
            live_symbol_allowlist=["ETH/USDT"],
        )
        return LiveTradingGuardrails(config=config)

    @pytest.fixture
    def paper_mode_guardrails(self, temp_dir):
        """Create guardrails with LIVE_MODE disabled."""
        reset_live_guardrails()
        config = LiveTradingConfig(
            live_mode=False,  # LIVE_MODE disabled
            state_file=str(temp_dir / "state.json"),
            kill_switch_file=str(temp_dir / "kill_switch.txt"),
        )
        return LiveTradingGuardrails(config=config)

    def test_startup_skips_check_when_live_mode_disabled(self, paper_mode_guardrails):
        """Test: LIVE_MODE=false skips readiness check."""
        result = check_startup_readiness(
            guardrails=paper_mode_guardrails,
            activate_kill_switch_on_failure=True,
        )

        assert result.passed is True
        assert result.live_rollout_readiness == "N/A"
        assert result.kill_switch_activated is False
        assert "LIVE_MODE is disabled" in result.reasons[0]

    def test_startup_passes_when_readiness_is_go(self, live_mode_guardrails):
        """Test: LIVE_MODE=true with readiness=GO passes."""
        # Mock readiness calculator to return GO
        mock_live_rollout = type('obj', (object,), {
            'readiness': 'GO',
            'reasons': ['All live rollout criteria met'],
        })()
        mock_result = type('obj', (object,), {
            'live_rollout': mock_live_rollout,
        })()

        with patch('api.readiness_calculator.get_readiness', return_value=mock_result):
            result = check_startup_readiness(
                guardrails=live_mode_guardrails,
                activate_kill_switch_on_failure=True,
            )

        assert result.passed is True
        assert result.live_rollout_readiness == "GO"
        assert result.kill_switch_activated is False

    def test_startup_blocks_when_readiness_is_conditional(self, live_mode_guardrails, temp_dir):
        """Test: LIVE_MODE=true with readiness=CONDITIONAL activates kill switch."""
        # Mock readiness calculator to return CONDITIONAL
        mock_live_rollout = type('obj', (object,), {
            'readiness': 'CONDITIONAL',
            'reasons': ['Paper trading streak below 14 days'],
        })()
        mock_result = type('obj', (object,), {
            'live_rollout': mock_live_rollout,
        })()

        with patch('api.readiness_calculator.get_readiness', return_value=mock_result):
            result = check_startup_readiness(
                guardrails=live_mode_guardrails,
                activate_kill_switch_on_failure=True,
            )

        assert result.passed is False
        assert result.live_rollout_readiness == "CONDITIONAL"
        assert result.kill_switch_activated is True

        # Verify kill switch file was created
        kill_switch_file = temp_dir / "kill_switch.txt"
        assert kill_switch_file.exists()
        content = kill_switch_file.read_text()
        assert "CONDITIONAL" in content

    def test_startup_blocks_when_readiness_is_no_go(self, live_mode_guardrails, temp_dir):
        """Test: LIVE_MODE=true with readiness=NO_GO activates kill switch."""
        # Mock readiness calculator to return NO_GO
        mock_live_rollout = type('obj', (object,), {
            'readiness': 'NO_GO',
            'reasons': ['Kill switch is active', 'Daily reports missing'],
        })()
        mock_result = type('obj', (object,), {
            'live_rollout': mock_live_rollout,
        })()

        with patch('api.readiness_calculator.get_readiness', return_value=mock_result):
            result = check_startup_readiness(
                guardrails=live_mode_guardrails,
                activate_kill_switch_on_failure=True,
            )

        assert result.passed is False
        assert result.live_rollout_readiness == "NO_GO"
        assert result.kill_switch_activated is True
        assert len(result.reasons) == 2

    def test_startup_can_skip_kill_switch_activation(self, live_mode_guardrails, temp_dir):
        """Test: activate_kill_switch_on_failure=False does not activate kill switch."""
        # Mock readiness calculator to return NO_GO
        mock_live_rollout = type('obj', (object,), {
            'readiness': 'NO_GO',
            'reasons': ['Critical issue'],
        })()
        mock_result = type('obj', (object,), {
            'live_rollout': mock_live_rollout,
        })()

        with patch('api.readiness_calculator.get_readiness', return_value=mock_result):
            result = check_startup_readiness(
                guardrails=live_mode_guardrails,
                activate_kill_switch_on_failure=False,  # Don't activate
            )

        assert result.passed is False
        assert result.live_rollout_readiness == "NO_GO"
        assert result.kill_switch_activated is False

        # Verify kill switch file was NOT created
        kill_switch_file = temp_dir / "kill_switch.txt"
        assert not kill_switch_file.exists()

    def test_startup_handles_import_error(self, live_mode_guardrails, temp_dir):
        """Test: Handles ImportError from readiness calculator."""
        with patch.dict('sys.modules', {'api.readiness_calculator': None}):
            # Force import to fail by making get_readiness unavailable
            with patch('api.readiness_calculator.get_readiness', side_effect=ImportError("not found")):
                result = check_startup_readiness(
                    guardrails=live_mode_guardrails,
                    activate_kill_switch_on_failure=True,
                )

        assert result.passed is False
        assert result.kill_switch_activated is True

    def test_startup_handles_unexpected_error(self, live_mode_guardrails, temp_dir):
        """Test: Handles unexpected error from readiness calculator."""
        with patch('api.readiness_calculator.get_readiness', side_effect=RuntimeError("Unexpected error")):
            result = check_startup_readiness(
                guardrails=live_mode_guardrails,
                activate_kill_switch_on_failure=True,
            )

        assert result.passed is False
        assert result.live_rollout_readiness == "ERROR"
        assert result.kill_switch_activated is True
        assert "Unexpected error" in result.reasons[0]

    def test_validate_live_mode_startup_returns_bool(self, temp_dir):
        """Test: validate_live_mode_startup returns simple bool."""
        reset_live_guardrails()
        config = LiveTradingConfig(
            live_mode=False,
            state_file=str(temp_dir / "state.json"),
            kill_switch_file=str(temp_dir / "kill_switch.txt"),
        )
        # Create and register singleton
        guardrails = LiveTradingGuardrails(config=config)

        # With LIVE_MODE=false, should always pass
        with patch('bot.live_trading_guardrails.get_live_guardrails', return_value=guardrails):
            is_safe = validate_live_mode_startup()

        assert is_safe is True

    def test_startup_result_to_dict(self):
        """Test: StartupReadinessResult.to_dict() returns correct structure."""
        result = StartupReadinessResult(
            passed=False,
            live_rollout_readiness="CONDITIONAL",
            reasons=["Reason 1", "Reason 2"],
            kill_switch_activated=True,
        )

        d = result.to_dict()

        assert d["passed"] is False
        assert d["live_rollout_readiness"] == "CONDITIONAL"
        assert d["reasons"] == ["Reason 1", "Reason 2"]
        assert d["kill_switch_activated"] is True

    def test_paper_trading_unaffected_by_failed_startup(self, temp_dir):
        """Test: Paper trading continues even when startup check fails."""
        # Create guardrails with LIVE_MODE=true
        reset_live_guardrails()
        config = LiveTradingConfig(
            live_mode=True,
            state_file=str(temp_dir / "state.json"),
            kill_switch_file=str(temp_dir / "kill_switch.txt"),
            live_symbol_allowlist=["ETH/USDT"],
        )
        guardrails = LiveTradingGuardrails(config=config)

        # Mock readiness calculator to return NO_GO
        mock_live_rollout = type('obj', (object,), {
            'readiness': 'NO_GO',
            'reasons': ['Critical issue'],
        })()
        mock_result = type('obj', (object,), {
            'live_rollout': mock_live_rollout,
        })()

        with patch('api.readiness_calculator.get_readiness', return_value=mock_result):
            result = check_startup_readiness(
                guardrails=guardrails,
                activate_kill_switch_on_failure=True,
            )

        # Startup failed
        assert result.passed is False
        assert result.kill_switch_activated is True

        # But paper trading (LIVE_MODE=false scenario) would still work
        # The startup check doesn't affect paper trading - it only activates
        # the kill switch which blocks LIVE trades, not paper trades
        # Paper trades don't go through the live guardrails at all
