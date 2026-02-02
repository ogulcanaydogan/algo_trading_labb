#!/bin/bash
# Phase 2B Shadow Data Collection - Automation Setup
#
# This script installs launchd agents for:
# - Daily shadow health check (23:55 every day)
# - Weekly shadow report (23:30 every Sunday)
#
# Usage:
#   ./setup_automation.sh install   # Install the agents
#   ./setup_automation.sh uninstall # Remove the agents
#   ./setup_automation.sh status    # Check agent status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_DIR="$SCRIPT_DIR/launchd"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

DAILY_PLIST="com.yapay.shadow.daily.plist"
WEEKLY_PLIST="com.yapay.shadow.weekly.plist"

install_agents() {
    echo "Installing Phase 2B shadow automation agents..."

    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "$LAUNCH_AGENTS_DIR"

    # Copy plist files
    cp "$PLIST_DIR/$DAILY_PLIST" "$LAUNCH_AGENTS_DIR/"
    cp "$PLIST_DIR/$WEEKLY_PLIST" "$LAUNCH_AGENTS_DIR/"

    # Load the agents
    launchctl load "$LAUNCH_AGENTS_DIR/$DAILY_PLIST"
    launchctl load "$LAUNCH_AGENTS_DIR/$WEEKLY_PLIST"

    echo "Done! Agents installed:"
    echo "  - Daily health check: 23:55 every day"
    echo "  - Weekly report: 23:30 every Sunday"
    echo ""
    echo "Logs will be written to:"
    echo "  - logs/daily_shadow_health.log"
    echo "  - logs/weekly_shadow_report.log"
}

uninstall_agents() {
    echo "Uninstalling Phase 2B shadow automation agents..."

    # Unload the agents (ignore errors if not loaded)
    launchctl unload "$LAUNCH_AGENTS_DIR/$DAILY_PLIST" 2>/dev/null
    launchctl unload "$LAUNCH_AGENTS_DIR/$WEEKLY_PLIST" 2>/dev/null

    # Remove plist files
    rm -f "$LAUNCH_AGENTS_DIR/$DAILY_PLIST"
    rm -f "$LAUNCH_AGENTS_DIR/$WEEKLY_PLIST"

    echo "Done! Agents removed."
}

check_status() {
    echo "Phase 2B Shadow Automation Status"
    echo "=================================="
    echo ""

    echo "Daily Health Check Agent:"
    if launchctl list | grep -q "com.yapay.shadow.daily"; then
        echo "  Status: INSTALLED and LOADED"
        launchctl list com.yapay.shadow.daily 2>/dev/null || echo "  (agent info not available)"
    elif [ -f "$LAUNCH_AGENTS_DIR/$DAILY_PLIST" ]; then
        echo "  Status: INSTALLED but NOT LOADED"
    else
        echo "  Status: NOT INSTALLED"
    fi
    echo ""

    echo "Weekly Report Agent:"
    if launchctl list | grep -q "com.yapay.shadow.weekly"; then
        echo "  Status: INSTALLED and LOADED"
        launchctl list com.yapay.shadow.weekly 2>/dev/null || echo "  (agent info not available)"
    elif [ -f "$LAUNCH_AGENTS_DIR/$WEEKLY_PLIST" ]; then
        echo "  Status: INSTALLED but NOT LOADED"
    else
        echo "  Status: NOT INSTALLED"
    fi
    echo ""

    # Check for recent log files
    echo "Recent log files:"
    if [ -f "logs/daily_shadow_health.log" ]; then
        echo "  - daily_shadow_health.log ($(stat -f %Sm logs/daily_shadow_health.log 2>/dev/null || echo 'unknown'))"
    fi
    if [ -f "logs/weekly_shadow_report.log" ]; then
        echo "  - weekly_shadow_report.log ($(stat -f %Sm logs/weekly_shadow_report.log 2>/dev/null || echo 'unknown'))"
    fi
}

# Main
case "$1" in
    install)
        install_agents
        ;;
    uninstall)
        uninstall_agents
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
