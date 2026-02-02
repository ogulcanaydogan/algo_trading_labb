#!/usr/bin/env bash
# linux_run_paper_live.sh - Start paper-live trading engine on Linux
# Works standalone or via systemd service
#
# Usage:
#   ./scripts/ops/linux_run_paper_live.sh                          # Default settings
#   PAPER_LIVE_TURNOVER_MIN_RATIO=2.5 ./scripts/ops/linux_run_paper_live.sh  # Custom overrides

set -euo pipefail

# === Resolve repo root ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
echo "Repo root: ${REPO_ROOT}"

# === Setup paths ===
LOGS_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

STDOUT_LOG="${LOGS_DIR}/paper_live_longrun.out.log"
STDERR_LOG="${LOGS_DIR}/paper_live_longrun.err.log"
PIDFILE="${LOGS_DIR}/paper_live.pid"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT_PY="${REPO_ROOT}/run_unified_trading.py"

# === Paper-live turnover overrides (defaults, can be overridden via env) ===
export PAPER_LIVE_TURNOVER_MIN_RATIO="${PAPER_LIVE_TURNOVER_MIN_RATIO:-2.0}"
export PAPER_LIVE_TURNOVER_MAX_DAILY="${PAPER_LIVE_TURNOVER_MAX_DAILY:-10}"

# === Validate venv python ===
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: Missing venv python: ${VENV_PYTHON}"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# === Check if already running ===
is_engine_running() {
    if [[ -f "${PIDFILE}" ]]; then
        local existing_pid
        existing_pid="$(head -n1 "${PIDFILE}" 2>/dev/null | tr -d '[:space:]')"
        if [[ "${existing_pid}" =~ ^[0-9]+$ ]]; then
            if ps -p "${existing_pid}" -o args= 2>/dev/null | grep -q "run_unified_trading.py"; then
                echo "${existing_pid}"
                return 0
            fi
        fi
    fi

    # Check heartbeat as fallback
    local heartbeat_file="${REPO_ROOT}/data/rl/paper_live_heartbeat.json"
    if [[ -f "${heartbeat_file}" ]]; then
        local hb_pid
        hb_pid="$(python3 -c "import json; print(json.load(open('${heartbeat_file}')).get('pid', ''))" 2>/dev/null || true)"
        if [[ "${hb_pid}" =~ ^[0-9]+$ ]]; then
            if ps -p "${hb_pid}" -o args= 2>/dev/null | grep -q "run_unified_trading.py"; then
                echo "${hb_pid}"
                return 0
            fi
        fi
    fi

    return 1
}

if running_pid="$(is_engine_running)"; then
    echo "Paper-live already running. PID: ${running_pid}"
    echo ""
    echo "Log files:"
    echo "  stdout: ${STDOUT_LOG}"
    echo "  stderr: ${STDERR_LOG}"
    echo ""
    echo "Helper commands:"
    echo "  tail -f ${STDOUT_LOG}"
    echo "  tail -f ${STDERR_LOG}"
    echo "  ${SCRIPT_DIR}/linux_stop_paper_live.sh"
    exit 0
fi

# === Clean up stale PID file ===
if [[ -f "${PIDFILE}" ]]; then
    rm -f "${PIDFILE}"
fi

# === Start the engine ===
echo "Starting paper-live engine..."
echo "  Turnover overrides: min_ratio=${PAPER_LIVE_TURNOVER_MIN_RATIO}, max_daily=${PAPER_LIVE_TURNOVER_MAX_DAILY}"

# Start in background, redirect stdout/stderr
nohup "${VENV_PYTHON}" "${SCRIPT_PY}" run \
    --mode paper_live_data \
    --interval 60 \
    --capital 10000 \
    >> "${STDOUT_LOG}" 2>> "${STDERR_LOG}" &

ENGINE_PID=$!

# Wait briefly for startup
sleep 3

# Verify engine is running
if ! ps -p "${ENGINE_PID}" > /dev/null 2>&1; then
    echo "ERROR: Engine process ${ENGINE_PID} died immediately after start."
    echo "Check error log: tail -50 ${STDERR_LOG}"
    exit 1
fi

# Verify it's running run_unified_trading.py
if ! ps -p "${ENGINE_PID}" -o args= 2>/dev/null | grep -q "run_unified_trading.py"; then
    echo "ERROR: Process ${ENGINE_PID} is not running run_unified_trading.py"
    exit 1
fi

echo ""
echo "=============================================="
echo "Paper-live engine started successfully"
echo "=============================================="
echo "PID: ${ENGINE_PID}"
echo "PID file: ${PIDFILE}"
echo ""
echo "Log files:"
echo "  stdout: ${STDOUT_LOG}"
echo "  stderr: ${STDERR_LOG}"
echo ""
echo "Verification commands:"
echo "  pgrep -af 'run_unified_trading.py'"
echo "  cat ${PIDFILE}; cat data/rl/paper_live_heartbeat.json | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pid\"])'"
echo ""
echo "Stop command:"
echo "  ${SCRIPT_DIR}/linux_stop_paper_live.sh"
