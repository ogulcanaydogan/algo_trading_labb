#!/usr/bin/env python3
"""
Health Check CLI Tool for April 1st Micro-Live Readiness.

READ-ONLY utility that calls local health endpoints and prints a clean summary.
Exits non-zero if live_rollout_readiness == NO_GO (for automation usage).

Usage:
    python scripts/ops/health_check.py          # Full report
    python scripts/ops/health_check.py --quiet  # Quiet mode (exit code only)
    python scripts/ops/health_check.py --json   # JSON output

Exit codes:
    0 = live_rollout_readiness is GO
    1 = live_rollout_readiness is CONDITIONAL
    2 = live_rollout_readiness is NO_GO
    3 = Error fetching health status
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen


# Default API endpoint
DEFAULT_BASE_URL = "http://localhost:8000"

# ANSI colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"
    CYAN = "\033[96m"


def fetch_endpoint(base_url: str, endpoint: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Fetch JSON from an endpoint."""
    url = f"{base_url}{endpoint}"
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except URLError as e:
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"error": str(e)}


def status_color(status: str) -> str:
    """Get ANSI color for a status."""
    if status in ("GO", "HEALTHY", "SAFE"):
        return Colors.GREEN
    elif status in ("CONDITIONAL", "WARNING"):
        return Colors.YELLOW
    else:  # NO_GO, CRITICAL, BLOCKED
        return Colors.RED


def format_status(status: str) -> str:
    """Format status with color."""
    return f"{status_color(status)}{status}{Colors.END}"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")


def print_health_report(readiness: Dict[str, Any], shadow: Dict[str, Any], live: Dict[str, Any]):
    """Print a formatted health report."""
    print_section("SYSTEM HEALTH CHECK")

    # Overall status
    overall = readiness.get("overall_readiness", "UNKNOWN")
    live_rollout = readiness.get("live_rollout_readiness", "UNKNOWN")

    print(f"\n{Colors.BOLD}Overall Readiness:{Colors.END}      {format_status(overall)}")
    print(f"{Colors.BOLD}Live Rollout Readiness:{Colors.END} {format_status(live_rollout)}")

    # Components summary
    components = readiness.get("components", {})

    print_section("COMPONENT STATUS")

    # Shadow status
    shadow_comp = components.get("shadow", {})
    shadow_health = shadow_comp.get("overall_health", "UNKNOWN")
    print(f"\n{Colors.BOLD}Shadow Data Collection:{Colors.END}")
    print(f"  Health:           {format_status(shadow_health)}")
    print(f"  Heartbeat Recent: {'Yes' if shadow_comp.get('heartbeat_recent', 0) else 'No'}")
    print(f"  Days Streak:      {shadow_comp.get('paper_live_days_streak', 0)}")
    print(f"  Weeks Counted:    {shadow_comp.get('paper_live_weeks_counted', 0)}")
    print(f"  Decisions Today:  {shadow_comp.get('paper_live_decisions_today', 0)}")

    # Live guardrails status
    live_comp = components.get("live", {})
    live_status = live_comp.get("overall_status", "UNKNOWN")
    print(f"\n{Colors.BOLD}Live Trading Guardrails:{Colors.END}")
    print(f"  Status:           {format_status(live_status)}")
    print(f"  Live Mode:        {'ENABLED' if live_comp.get('live_mode_enabled') else 'DISABLED'}")
    print(f"  Kill Switch:      {'ACTIVE' if live_comp.get('kill_switch_active') else 'Inactive'}")
    print(f"  Trades Remaining: {live_comp.get('daily_trades_remaining', 'N/A')}")

    # Turnover status
    turnover_comp = components.get("turnover", {})
    print(f"\n{Colors.BOLD}Turnover Governor:{Colors.END}")
    print(f"  Enabled:          {'Yes' if turnover_comp.get('enabled') else 'No'}")
    print(f"  Block Rate:       {turnover_comp.get('block_rate_pct', 0):.1f}%")
    print(f"  Blocks Today:     {turnover_comp.get('total_blocks_today', 0)}")
    print(f"  Decisions Today:  {turnover_comp.get('total_decisions_today', 0)}")

    # Capital preservation status
    capital_comp = components.get("capital_preservation", {})
    capital_level = capital_comp.get("current_level", "UNKNOWN")
    print(f"\n{Colors.BOLD}Capital Preservation:{Colors.END}")
    print(f"  Level:            {format_status(capital_level)}")
    print(f"  Restrictions:     {'Yes' if capital_comp.get('restrictions_active') else 'No'}")

    # Daily reports status
    reports_comp = components.get("daily_reports", {})
    print(f"\n{Colors.BOLD}Daily Health Reports:{Colors.END}")
    print(f"  Last 24h:         {'Yes' if reports_comp.get('reports_last_24h') else 'No'}")
    print(f"  Report Age:       {reports_comp.get('latest_report_age_hours', 999):.1f} hours")
    print(f"  CRITICAL (14d):   {reports_comp.get('critical_alerts_14d', 0)}")

    # Execution realism status
    realism_comp = components.get("execution_realism", {})
    print(f"\n{Colors.BOLD}Execution Realism:{Colors.END}")
    print(f"  Available:        {'Yes' if realism_comp.get('available') else 'No'}")
    print(f"  Drift Detected:   {'Yes' if realism_comp.get('drift_detected') else 'No'}")

    # Live rollout reasons and actions
    print_section("LIVE ROLLOUT ASSESSMENT")

    reasons = readiness.get("live_rollout_reasons", [])
    if reasons:
        print(f"\n{Colors.BOLD}Reasons:{Colors.END}")
        for reason in reasons:
            status_char = "+" if live_rollout == "GO" else "-"
            color = status_color(live_rollout)
            print(f"  {color}{status_char}{Colors.END} {reason}")

    actions = readiness.get("live_rollout_next_actions", [])
    if actions:
        print(f"\n{Colors.BOLD}Recommended Actions:{Colors.END}")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

    # Final verdict
    print_section("VERDICT")
    if live_rollout == "GO":
        print(f"\n{Colors.GREEN}{Colors.BOLD}READY FOR LIVE TRADING{Colors.END}")
        print("  All live rollout criteria met.")
    elif live_rollout == "CONDITIONAL":
        print(f"\n{Colors.YELLOW}{Colors.BOLD}CONDITIONAL - FIX ISSUES BEFORE LIVE{Colors.END}")
        print("  Address the issues above before enabling LIVE_MODE.")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}NOT READY - DO NOT ENABLE LIVE MODE{Colors.END}")
        print("  Critical issues must be resolved first.")


def main():
    parser = argparse.ArgumentParser(
        description="Health check CLI for April 1st micro-live readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 = live_rollout_readiness is GO
  1 = live_rollout_readiness is CONDITIONAL
  2 = live_rollout_readiness is NO_GO
  3 = Error fetching health status

Examples:
  python scripts/ops/health_check.py
  python scripts/ops/health_check.py --quiet
  python scripts/ops/health_check.py --json
  python scripts/ops/health_check.py --base-url http://localhost:8080
        """,
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only print status and exit code",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON response",
    )
    parser.add_argument(
        "--base-url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )

    args = parser.parse_args()

    # Fetch all health endpoints
    readiness = fetch_endpoint(args.base_url, "/health/readiness", args.timeout)
    shadow = fetch_endpoint(args.base_url, "/health/shadow", args.timeout)
    live = fetch_endpoint(args.base_url, "/health/live", args.timeout)

    # Check for errors
    if "error" in readiness:
        if not args.quiet:
            print(f"{Colors.RED}Error fetching /health/readiness: {readiness['error']}{Colors.END}")
        sys.exit(3)

    # JSON output mode
    if args.json:
        output = {
            "readiness": readiness,
            "shadow": shadow if "error" not in shadow else None,
            "live": live if "error" not in live else None,
        }
        print(json.dumps(output, indent=2))
    elif not args.quiet:
        # Full report
        print_health_report(readiness, shadow, live)
    else:
        # Quiet mode - just print status
        live_rollout = readiness.get("live_rollout_readiness", "UNKNOWN")
        print(f"live_rollout_readiness: {live_rollout}")

    # Determine exit code based on live_rollout_readiness
    live_rollout = readiness.get("live_rollout_readiness", "UNKNOWN")
    if live_rollout == "GO":
        sys.exit(0)
    elif live_rollout == "CONDITIONAL":
        sys.exit(1)
    else:  # NO_GO or UNKNOWN
        sys.exit(2)


if __name__ == "__main__":
    main()
