#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICE="algo_trading_paper_live"
PIDFILE="${ROOT}/logs/paper_live.pid"
HEARTBEAT="${ROOT}/data/rl/paper_live_heartbeat.json"
OUTLOG="${ROOT}/logs/paper_live_longrun.out.log"
HEARTBEAT_MAX_AGE_SECONDS="${HEARTBEAT_MAX_AGE_SECONDS:-120}"

fail_count=0

fail() {
  echo "FAIL: $1"
  fail_count=$((fail_count + 1))
}

echo "== Service status =="
systemctl --user status "${SERVICE}" --no-pager || fail "service status failed"
if ! systemctl --user is-active --quiet "${SERVICE}"; then
  fail "service not active"
fi

echo
echo "== Process check =="
pgrep -af "run_unified_trading.py" || true
proc_count="$(pgrep -af "run_unified_trading.py" | grep -v "pgrep -af" | grep -v "linux_verify_paper_live.sh" | wc -l | tr -d ' ')"
echo "Process count: ${proc_count}"
if [ "${proc_count}" -ne 1 ]; then
  fail "expected exactly 1 run_unified_trading.py process"
fi

echo
echo "== PID parity =="
if [ ! -f "${PIDFILE}" ]; then
  fail "pidfile missing: ${PIDFILE}"
else
  pidfile_pid="$(cat "${PIDFILE}")"
  echo "pidfile: ${pidfile_pid}"
fi
if [ ! -f "${HEARTBEAT}" ]; then
  fail "heartbeat missing: ${HEARTBEAT}"
else
  hb_pid="$(cat "${HEARTBEAT}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["pid"])')"
  echo "heartbeat pid: ${hb_pid}"
  if [ "${pidfile_pid:-}" = "${hb_pid}" ]; then
    echo "PID parity: MATCH"
  else
    echo "PID parity: NO MATCH"
    fail "pidfile does not match heartbeat pid"
  fi
fi

echo
echo "== Heartbeat timestamp =="
if [ -f "${HEARTBEAT}" ]; then
  hb_ts="$(cat "${HEARTBEAT}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("timestamp", ""))')"
  echo "heartbeat timestamp: ${hb_ts}"
  if [ -z "${hb_ts}" ]; then
    fail "heartbeat timestamp missing"
  else
    hb_age="$(python3 - <<'PY'
import sys
from datetime import datetime, timezone

ts = sys.argv[1]
try:
    dt = datetime.fromisoformat(ts)
except ValueError:
    # Fallback for timestamps without microseconds/offset
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
age = (now - dt).total_seconds()
print(int(age))
PY
"${hb_ts}")"
    echo "heartbeat age (s): ${hb_age}"
    if [ "${hb_age}" -gt "${HEARTBEAT_MAX_AGE_SECONDS}" ]; then
      fail "heartbeat too old (${hb_age}s > ${HEARTBEAT_MAX_AGE_SECONDS}s)"
    fi
  fi
fi

echo
echo "== Turnover overrides =="
override_line="$(grep -n "Paper-live turnover overrides" "${OUTLOG}" | tail -1 || true)"
if [ -n "${override_line}" ]; then
  echo "${override_line}"
else
  fail "turnover override line not found in out log"
fi

if [ "${fail_count}" -ne 0 ]; then
  echo
  echo "Verification failed: ${fail_count} gate(s) failed."
  exit 1
fi

echo
echo "Verification OK."
