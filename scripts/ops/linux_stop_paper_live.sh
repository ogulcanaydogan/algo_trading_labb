#!/usr/bin/env bash
# linux_stop_paper_live.sh - Stop paper-live trading engine on Linux
#
# Usage:
#   ./scripts/ops/linux_stop_paper_live.sh

set -euo pipefail

# === Resolve repo root ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# === Setup paths ===
PIDFILE="${REPO_ROOT}/logs/paper_live.pid"
HEARTBEAT_FILE="${REPO_ROOT}/data/rl/paper_live_heartbeat.json"

echo "Stopping paper-live engine..."

stopped=0

# === Try to stop by PID file ===
if [[ -f "${PIDFILE}" ]]; then
    pid="$(head -n1 "${PIDFILE}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${pid}" =~ ^[0-9]+$ ]]; then
        if ps -p "${pid}" > /dev/null 2>&1; then
            echo "Found PID file with PID: ${pid}"
            if ps -p "${pid}" -o args= 2>/dev/null | grep -q "run_unified_trading.py"; then
                echo "Sending SIGTERM to ${pid}..."
                kill -TERM "${pid}" 2>/dev/null || true

                # Wait for graceful shutdown (max 10 seconds)
                for i in {1..10}; do
                    if ! ps -p "${pid}" > /dev/null 2>&1; then
                        echo "Process ${pid} stopped gracefully."
                        stopped=1
                        break
                    fi
                    sleep 1
                done

                # Force kill if still running
                if ps -p "${pid}" > /dev/null 2>&1; then
                    echo "Process ${pid} still running, sending SIGKILL..."
                    kill -KILL "${pid}" 2>/dev/null || true
                    sleep 1
                    stopped=1
                fi
            else
                echo "PID ${pid} is not running run_unified_trading.py (stale PID file)"
            fi
        else
            echo "PID ${pid} is not running (stale PID file)"
        fi
    fi
    # Remove PID file
    rm -f "${PIDFILE}"
    echo "PID file removed."
fi

# === Try to stop by heartbeat PID ===
if [[ ${stopped} -eq 0 ]] && [[ -f "${HEARTBEAT_FILE}" ]]; then
    hb_pid="$(python3 -c "import json; print(json.load(open('${HEARTBEAT_FILE}')).get('pid', ''))" 2>/dev/null || true)"
    if [[ "${hb_pid}" =~ ^[0-9]+$ ]]; then
        if ps -p "${hb_pid}" > /dev/null 2>&1; then
            echo "Found heartbeat with PID: ${hb_pid}"
            if ps -p "${hb_pid}" -o args= 2>/dev/null | grep -q "run_unified_trading.py"; then
                echo "Sending SIGTERM to ${hb_pid}..."
                kill -TERM "${hb_pid}" 2>/dev/null || true

                # Wait for graceful shutdown
                for i in {1..10}; do
                    if ! ps -p "${hb_pid}" > /dev/null 2>&1; then
                        echo "Process ${hb_pid} stopped gracefully."
                        stopped=1
                        break
                    fi
                    sleep 1
                done

                if ps -p "${hb_pid}" > /dev/null 2>&1; then
                    echo "Process ${hb_pid} still running, sending SIGKILL..."
                    kill -KILL "${hb_pid}" 2>/dev/null || true
                    stopped=1
                fi
            fi
        fi
    fi
fi

# === Fallback: find any running engine process ===
if [[ ${stopped} -eq 0 ]]; then
    orphan_pids="$(pgrep -f 'run_unified_trading.py' 2>/dev/null || true)"
    if [[ -n "${orphan_pids}" ]]; then
        echo "Found orphan process(es): ${orphan_pids}"
        for pid in ${orphan_pids}; do
            echo "Stopping orphan PID ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
        done
        sleep 2

        # Force kill any remaining
        orphan_pids="$(pgrep -f 'run_unified_trading.py' 2>/dev/null || true)"
        if [[ -n "${orphan_pids}" ]]; then
            for pid in ${orphan_pids}; do
                kill -KILL "${pid}" 2>/dev/null || true
            done
        fi
        stopped=1
    fi
fi

# === Final status ===
if [[ ${stopped} -eq 1 ]]; then
    echo ""
    echo "Paper-live engine stopped."
else
    echo ""
    echo "No running paper-live engine found."
fi

# === Verify no engine running ===
remaining="$(pgrep -af 'run_unified_trading.py' 2>/dev/null || true)"
if [[ -n "${remaining}" ]]; then
    echo ""
    echo "WARNING: Some engine processes may still be running:"
    echo "${remaining}"
fi
