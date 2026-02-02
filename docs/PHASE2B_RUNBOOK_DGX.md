# Phase-2B DGX (spark) Runbook

Scope: paper-live engine on DGX "spark" using a systemd user service.

Service details
- Host: spark
- Repo: `~/work/algo_trading_lab`
- Service: `algo_trading_paper_live.service`
- Python: `~/work/algo_trading_lab/.venv/bin/python`

## Start/Stop/Restart/Status
```bash
systemctl --user start algo_trading_paper_live
systemctl --user stop algo_trading_paper_live
systemctl --user restart algo_trading_paper_live
systemctl --user status algo_trading_paper_live --no-pager
```

## Logs
```bash
# systemd journal (follow)
journalctl --user -u algo_trading_paper_live -f

# long-run stdout/stderr logs
tail -n 200 ~/work/algo_trading_lab/logs/paper_live_longrun.out.log
tail -n 200 ~/work/algo_trading_lab/logs/paper_live_longrun.err.log

# quick turnover override check
grep -n "Paper-live turnover overrides" ~/work/algo_trading_lab/logs/paper_live_longrun.out.log | tail -5
```

## PID parity check
```bash
cat ~/work/algo_trading_lab/logs/paper_live.pid
cat ~/work/algo_trading_lab/data/rl/paper_live_heartbeat.json | \
  python3 -c 'import sys,json; print(json.load(sys.stdin)["pid"])'
```

## One-shot verification
```bash
~/work/algo_trading_lab/scripts/ops/linux_verify_paper_live.sh
```

## Heartbeat freshness
```bash
stat ~/work/algo_trading_lab/data/rl/paper_live_heartbeat.json
cat ~/work/algo_trading_lab/data/rl/paper_live_heartbeat.json | head -50
```

## Reboot persistence test
1. Confirm linger enabled:
   ```bash
   loginctl show-user "$USER" -p Linger
   ```
2. Reboot the host (planned window).
3. After reboot, verify the service restarts automatically:
   ```bash
   systemctl --user status algo_trading_paper_live --no-pager
   pgrep -af "run_unified_trading.py"
   ~/work/algo_trading_lab/scripts/ops/linux_verify_paper_live.sh
   ```

## Turnover overrides (paper-live only)
Overrides are env-driven and must remain paper-live-only:
- `PAPER_LIVE_TURNOVER_MIN_RATIO`
- `PAPER_LIVE_TURNOVER_MAX_DAILY`

Safe change procedure:
1. Edit the systemd user service to update env vars.
   - File: `~/.config/systemd/user/algo_trading_paper_live.service`
   - Update/insert `Environment=` lines, or point to an `EnvironmentFile=`.
   - Or use a drop-in override:
     ```bash
     systemctl --user edit algo_trading_paper_live
     ```
     Then add:
     ```ini
     [Service]
     Environment=PAPER_LIVE_TURNOVER_MIN_RATIO=2.0
     Environment=PAPER_LIVE_TURNOVER_MAX_DAILY=10
     ```
2. Reload and restart:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart algo_trading_paper_live
   ```
3. Confirm in logs:
   ```bash
   grep -n "Paper-live turnover overrides" ~/work/algo_trading_lab/logs/paper_live_longrun.out.log | tail -5
   ```

Do not change live trading defaults.

## Troubleshooting
```bash
# Recent logs
journalctl --user -u algo_trading_paper_live -n 200 --no-pager
tail -n 200 ~/work/algo_trading_lab/logs/paper_live_longrun.out.log
tail -n 200 ~/work/algo_trading_lab/logs/paper_live_longrun.err.log

# Quick restart
systemctl --user restart algo_trading_paper_live
```
