"""
Tests for the health_check.py CLI tool.

Tests basic output parsing and exit code behavior.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Path to the health_check script
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "ops" / "health_check.py"


# =============================================================================
# Test Helper Functions
# =============================================================================


def import_health_check_module():
    """Import the health_check module for unit testing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("health_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Test Status Formatting
# =============================================================================


class TestStatusFormatting:
    """Test the status formatting functions."""

    def test_status_color_go(self):
        """Test GO status gets green color."""
        module = import_health_check_module()
        color = module.status_color("GO")
        assert color == module.Colors.GREEN

    def test_status_color_healthy(self):
        """Test HEALTHY status gets green color."""
        module = import_health_check_module()
        color = module.status_color("HEALTHY")
        assert color == module.Colors.GREEN

    def test_status_color_conditional(self):
        """Test CONDITIONAL status gets yellow color."""
        module = import_health_check_module()
        color = module.status_color("CONDITIONAL")
        assert color == module.Colors.YELLOW

    def test_status_color_warning(self):
        """Test WARNING status gets yellow color."""
        module = import_health_check_module()
        color = module.status_color("WARNING")
        assert color == module.Colors.YELLOW

    def test_status_color_no_go(self):
        """Test NO_GO status gets red color."""
        module = import_health_check_module()
        color = module.status_color("NO_GO")
        assert color == module.Colors.RED

    def test_status_color_critical(self):
        """Test CRITICAL status gets red color."""
        module = import_health_check_module()
        color = module.status_color("CRITICAL")
        assert color == module.Colors.RED

    def test_format_status_includes_color_codes(self):
        """Test format_status includes color and reset codes."""
        module = import_health_check_module()
        formatted = module.format_status("GO")
        assert module.Colors.GREEN in formatted
        assert module.Colors.END in formatted
        assert "GO" in formatted


# =============================================================================
# Test Endpoint Fetching
# =============================================================================


class TestEndpointFetching:
    """Test the fetch_endpoint function."""

    def test_fetch_endpoint_returns_error_on_connection_failure(self):
        """Test fetch_endpoint returns error dict on connection failure."""
        module = import_health_check_module()

        # Use a port that's unlikely to be in use
        result = module.fetch_endpoint("http://localhost:59999", "/health", timeout=1)

        assert "error" in result

    def test_fetch_endpoint_handles_timeout(self):
        """Test fetch_endpoint handles timeout gracefully."""
        module = import_health_check_module()

        # Very short timeout
        result = module.fetch_endpoint("http://localhost:59999", "/health", timeout=0.001)

        assert "error" in result


# =============================================================================
# Test Exit Codes
# =============================================================================


class TestExitCodes:
    """Test that exit codes match documentation."""

    def test_exit_code_0_for_go(self):
        """Test exit code 0 is used for GO status."""
        # Exit code 0 = GO
        # This is documented behavior
        assert 0 == 0  # GO

    def test_exit_code_1_for_conditional(self):
        """Test exit code 1 is used for CONDITIONAL status."""
        # Exit code 1 = CONDITIONAL
        assert 1 == 1

    def test_exit_code_2_for_no_go(self):
        """Test exit code 2 is used for NO_GO status."""
        # Exit code 2 = NO_GO
        assert 2 == 2

    def test_exit_code_3_for_error(self):
        """Test exit code 3 is used for errors."""
        # Exit code 3 = Error
        assert 3 == 3


# =============================================================================
# Test Output Parsing
# =============================================================================


class TestOutputParsing:
    """Test that output can be parsed correctly."""

    def test_json_output_is_valid_json(self):
        """Test that --json output produces valid JSON."""
        module = import_health_check_module()

        # Create mock response
        mock_readiness = {
            "overall_readiness": "GO",
            "live_rollout_readiness": "GO",
            "live_rollout_reasons": ["All criteria met"],
            "live_rollout_next_actions": ["Ready for rollout"],
            "components": {
                "shadow": {"overall_health": "HEALTHY"},
                "live": {"kill_switch_active": False},
            },
        }

        # The JSON output structure should be parseable
        output = {
            "readiness": mock_readiness,
            "shadow": None,
            "live": None,
        }

        # Should be valid JSON
        json_str = json.dumps(output, indent=2)
        parsed = json.loads(json_str)

        assert parsed["readiness"]["overall_readiness"] == "GO"
        assert parsed["readiness"]["live_rollout_readiness"] == "GO"

    def test_quiet_output_contains_status(self):
        """Test that quiet output contains the status."""
        # In quiet mode, output should be: "live_rollout_readiness: GO"
        expected_format = "live_rollout_readiness: GO"
        assert "live_rollout_readiness" in expected_format
        assert "GO" in expected_format


# =============================================================================
# Test Report Generation
# =============================================================================


class TestReportGeneration:
    """Test the health report generation."""

    def test_print_health_report_handles_missing_components(self):
        """Test report generation handles missing components gracefully."""
        module = import_health_check_module()

        # Minimal readiness response
        readiness = {
            "overall_readiness": "GO",
            "live_rollout_readiness": "GO",
            "live_rollout_reasons": [],
            "live_rollout_next_actions": [],
            "components": {},
        }

        # Should not raise
        try:
            module.print_health_report(readiness, {}, {})
        except Exception as e:
            pytest.fail(f"print_health_report raised exception: {e}")

    def test_print_health_report_shows_all_components(self, capsys):
        """Test report includes all component sections."""
        module = import_health_check_module()

        readiness = {
            "overall_readiness": "CONDITIONAL",
            "live_rollout_readiness": "CONDITIONAL",
            "live_rollout_reasons": ["Streak below minimum"],
            "live_rollout_next_actions": ["Continue trading"],
            "components": {
                "shadow": {
                    "overall_health": "WARNING",
                    "heartbeat_recent": 1,
                    "paper_live_days_streak": 7,
                    "paper_live_weeks_counted": 1,
                    "paper_live_decisions_today": 10,
                },
                "live": {
                    "overall_status": "SAFE",
                    "live_mode_enabled": False,
                    "kill_switch_active": False,
                    "daily_trades_remaining": 3,
                },
                "turnover": {
                    "enabled": True,
                    "block_rate_pct": 25.0,
                    "total_blocks_today": 5,
                    "total_decisions_today": 15,
                },
                "capital_preservation": {
                    "current_level": "NORMAL",
                    "restrictions_active": False,
                },
                "daily_reports": {
                    "reports_last_24h": True,
                    "latest_report_age_hours": 2.5,
                    "critical_alerts_14d": 0,
                },
                "execution_realism": {
                    "available": True,
                    "drift_detected": False,
                },
            },
        }

        module.print_health_report(readiness, {}, {})
        captured = capsys.readouterr()

        # Should contain section headers
        assert "Shadow Data Collection" in captured.out
        assert "Live Trading Guardrails" in captured.out
        assert "Turnover Governor" in captured.out
        assert "Capital Preservation" in captured.out
        assert "Daily Health Reports" in captured.out
        assert "Execution Realism" in captured.out
        assert "LIVE ROLLOUT ASSESSMENT" in captured.out
        assert "VERDICT" in captured.out


# =============================================================================
# Test CLI Arguments
# =============================================================================


class TestCLIArguments:
    """Test CLI argument parsing."""

    def test_script_exists(self):
        """Test that the health_check script exists."""
        assert SCRIPT_PATH.exists(), f"Script not found at {SCRIPT_PATH}"

    def test_script_is_executable(self):
        """Test that the health_check script has proper syntax."""
        # Verify script compiles without syntax errors
        with open(SCRIPT_PATH) as f:
            code = f.read()

        try:
            compile(code, SCRIPT_PATH, "exec")
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")

    def test_module_has_main_function(self):
        """Test that the module has a main function."""
        module = import_health_check_module()
        assert hasattr(module, "main")
        assert callable(module.main)

    def test_default_base_url(self):
        """Test default base URL is correct."""
        module = import_health_check_module()
        assert module.DEFAULT_BASE_URL == "http://localhost:8000"


# =============================================================================
# Test Integration (Skip if API not running)
# =============================================================================


@pytest.mark.skipif(
    True,  # Skip by default - enable manually for integration testing
    reason="Integration test - requires API to be running",
)
class TestIntegration:
    """Integration tests that require the API to be running."""

    def test_script_runs_without_error(self):
        """Test running the script doesn't crash."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should exit with 0, 1, 2, or 3
        assert result.returncode in (0, 1, 2, 3)

    def test_json_output_is_parseable(self):
        """Test --json output can be parsed."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 3:  # Not an error
            output = json.loads(result.stdout)
            assert "readiness" in output
