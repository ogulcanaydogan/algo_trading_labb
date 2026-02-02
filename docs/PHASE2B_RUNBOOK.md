# Phase 2B Operations Runbook

## 3-Month Shadow Data Collection Program

**Start Date:** _____________
**Target End Date:** _____________ (12 weeks minimum)
**Current Week:** _____ / 12

---

## Overview

Phase 2B collects shadow data during paper trading to validate RL recommendations
through counterfactual analysis. **RL has NO execution authority** - it provides
advisory recommendations only.

### Goals

1. Collect 3+ months of shadow decision data
2. Validate RL edge through counterfactual evaluation
3. Address BTC turnover degradation
4. Meet all Phase 2C promotion gates

### Non-Negotiables

- RL CANNOT place orders
- RL CANNOT bypass TradeGate
- RL CANNOT change leverage caps
- Strategy weighting shifts CLAMPED at 10% per day
- Capital Preservation remains final authority

---

## Quick Reference

### File Locations

| Purpose | Path |
|---------|------|
| Shadow log | `data/rl/shadow_decisions.jsonl` |
| Heartbeat | `data/rl/paper_live_heartbeat.json` |
| Logs | `logs/` |

---

## Windows 24/7 runner (Tailscale host)

### Pull latest repo
```powershell
cd C:\path\to\algo_trading_lab
git pull
```

### Create/activate venv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start long-run paper-live
```powershell
powershell -ExecutionPolicy Bypass -File scripts\ops\windows_run_paper_live.ps1
```

### Check logs + artifacts
```powershell
Get-Content -Tail 50 logs\paper_live_longrun.out.log
Get-Content -Tail 50 logs\paper_live_longrun.err.log
Get-Item data\rl\shadow_decisions.jsonl
Get-Item data\rl\paper_live_heartbeat.json
Get-Content data\rl\paper_live_heartbeat.json
Get-Content -Tail 5 data\rl\shadow_decisions.jsonl
```

### Status check (single process + PID parity)
```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "run_unified_trading.py" } |
  Select ProcessId,ParentProcessId,CommandLine | Format-List

$pidfile = Get-Content logs\paper_live.pid
$hb = Get-Content data\rl\paper_live_heartbeat.json | ConvertFrom-Json
"pidfile=$pidfile heartbeatPid=$($hb.pid) last_heartbeat=$($hb.timestamp) decisions=$($hb.total_decisions_session)"
```

### Reboot persistence test (manual)
```powershell
shutdown /r /t 0
```
After reboot:
```powershell
schtasks /Query /TN "algo_trading_lab_paper_live" /V /FO LIST
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "run_unified_trading.py" } |
  Select ProcessId,ParentProcessId,CommandLine | Format-List
Get-Content logs\paper_live.pid
Get-Content data\rl\paper_live_heartbeat.json
```

### Stop long-run
```powershell
powershell -ExecutionPolicy Bypass -File scripts\ops\windows_stop_paper_live.ps1
```

### Turnover Overrides (Paper-Live Only)
Set environment variables to override turnover settings for PAPER_LIVE mode:
```powershell
$env:PAPER_LIVE_TURNOVER_MIN_RATIO = "2.5"
$env:PAPER_LIVE_TURNOVER_MAX_DAILY = "8"
powershell -ExecutionPolicy Bypass -File scripts\ops\windows_run_paper_live.ps1
```

Verify overrides are active:
```powershell
Get-Content -Tail 200 C:\work\algo_trading_lab\logs\paper_live_longrun.out.log | Select-String "Paper-live turnover overrides"
```

### Task Scheduler Setup (Reboot Persistence)

**Query existing task:**
```powershell
schtasks /Query /TN "algo_trading_lab_paper_live" /V /FO LIST
```

**Create task (run as Admin):**
```powershell
schtasks /Create /TN "algo_trading_lab_paper_live" /TR "powershell.exe -ExecutionPolicy Bypass -File C:\work\algo_trading_lab\scripts\ops\windows_run_paper_live.ps1" /SC ONSTART /RU SYSTEM /RL HIGHEST /F
```

**Delete task if needed:**
```powershell
schtasks /Delete /TN "algo_trading_lab_paper_live" /F
```

**Run task manually:**
```powershell
schtasks /Run /TN "algo_trading_lab_paper_live"
```

### Additional File Locations

| Purpose | Path |
|---------|------|
| Daily health reports | `data/reports/daily_shadow_health_*.json` |
| Weekly evidence reports | `data/reports/weekly_shadow_report_*.json` |
| Weekly summaries | `docs/reports/weekly_shadow_summary_*.md` |
| Promotion gates status | `data/phase2b_go_nogo_summary.json` |
| Capital preservation state | `data/capital_preservation_state.json` |
| Strategy weights | `data/rl/strategy_weights.json` |

### Scripts

| Script | Schedule | Purpose |
|--------|----------|---------|
| `scripts/shadow/run_daily_shadow_health.py` | Daily (cron) | Health check |
| `scripts/shadow/run_weekly_shadow_report.py --week N` | Weekly (manual) | Evidence report |

---

## Enable/Disable Shadow Collector

### Enable Shadow Mode

```python
from bot.rl.shadow_data_collector import (
    ShadowCollectorConfig,
    ShadowDataCollector,
    get_shadow_collector,
)

# Enable with default config
config = ShadowCollectorConfig(
    enabled=True,
    enable_rl_shadow=True,
    log_path=Path("data/rl/shadow_decisions.jsonl"),
)
collector = get_shadow_collector(config)

# Verify enabled
print(f"Shadow collector enabled: {collector.config.enabled}")
print(f"RL shadow enabled: {collector.config.enable_rl_shadow}")
```

### Disable Shadow Mode

```python
from bot.rl.shadow_data_collector import reset_shadow_collector

# Disable by resetting
reset_shadow_collector()

# Or configure as disabled
config = ShadowCollectorConfig(enabled=False)
```

### Verify in Paper Trading

Add to your paper trading startup:

```python
# In run_unified_trading.py or equivalent
from bot.rl.shadow_data_collector import get_shadow_collector

collector = get_shadow_collector()
assert collector.config.enabled, "Shadow collector must be enabled for Phase 2B"

# Log startup confirmation
logger.info(f"Phase 2B Shadow Mode: ACTIVE")
logger.info(f"Shadow log: {collector.config.log_path}")
```

---

## Weekly Checklist

### Before Monday

- [ ] Verify paper trading ran all week
- [ ] Check shadow log has new entries

### Monday Morning

Run weekly report:

```bash
cd /path/to/algo_trading_lab
python scripts/shadow/run_weekly_shadow_report.py --week N
```

Where N = current week number (1-12+)

### Review Report

1. **Check overall edge:**
   ```
   Overall Edge Positive: Yes/No
   Edge Stable Across Regimes: Yes/No
   ```

2. **Check per-symbol:**
   - BTC-USD edge (watch for degradation)
   - ETH-USD edge (baseline comparison)

3. **Check per-regime:**
   - Edge must be positive in majority of regimes
   - Flag if only profitable in one regime (overfitting)

4. **Check costs:**
   - Cost as % of PnL < 50%
   - If higher, investigate turnover

5. **Check drift:**
   - Compare vs prior week
   - Flag if Sharpe dropped > 0.2

6. **Update promotion gate progress:**
   - Log in `data/phase2b_gate_progress.json`

### End of Week

- [ ] Archive weekly report
- [ ] Update tracking spreadsheet
- [ ] File BTC diagnosis if needed
- [ ] Plan next week's monitoring

---

## Daily Health Check

### Automated (via cron)

```cron
# Run daily at 6 AM
0 6 * * * cd /path/to/algo_trading_lab && python scripts/shadow/run_daily_shadow_health.py >> logs/daily_health.log 2>&1
```

### Manual

```bash
python scripts/shadow/run_daily_shadow_health.py
```

### Interpret Results

| Status | Meaning | Action |
|--------|---------|--------|
| HEALTHY | All checks pass | Continue |
| WARNING | Minor issues | Investigate |
| CRITICAL | Clamp violation or major issue | Stop and fix |

### Critical Alerts

**Clamp Exceeded:**
```
CLAMP VIOLATION: Strategy X shifted Y%, exceeds max 10%
```
Action: Immediately investigate strategy weighting advisor. This should never happen.

**Lockdown Active:**
```
Capital preservation in LOCKDOWN
```
Action: Review losses, wait for recovery, do not force trades.

**High Rejection Rate:**
```
High TradeGate rejection rate: 80%+
```
Action: Review signal quality or TradeGate thresholds.

---

## Failure Modes

### Shadow Logging Stopped

**Symptoms:**
- Daily health shows 0 decisions
- Shadow log file not updating

**Diagnosis:**
```bash
# Check log file
ls -la data/rl/shadow_decisions.jsonl

# Check last entry
tail -1 data/rl/shadow_decisions.jsonl | python -m json.tool

# Check paper trading is running
ps aux | grep unified_trading
```

**Resolution:**
1. Verify paper trading process is running
2. Check for exceptions in trading logs
3. Verify shadow collector is enabled in config
4. Restart paper trading if needed

### Strategy Weight Clamp Violated

**Symptoms:**
- Daily health shows CRITICAL status
- Clamp exceeded alert in report

**Diagnosis:**
```bash
# Check weight state
cat data/rl/strategy_weights.json | python -m json.tool
```

**Resolution:**
1. **Immediate:** This should NOT happen - indicates bug
2. Reset weights to default:
   ```python
   from bot.rl.strategy_weighting_advisor import reset_strategy_weighting_advisor
   reset_strategy_weighting_advisor()
   ```
3. Review strategy_weighting_advisor.py for bugs
4. File incident report

### Capital Preservation Lockdown

**Symptoms:**
- Lockdown alert in daily health
- No new trades being executed

**Diagnosis:**
```bash
cat data/capital_preservation_state.json | python -m json.tool
```

**Resolution:**
1. **DO NOT** manually override lockdown
2. Wait for recovery criteria to be met
3. Review what triggered lockdown
4. This is the system working as intended

### Drift Detected

**Symptoms:**
- Weekly report shows degradation vs prior week
- Sharpe dropped significantly

**Diagnosis:**
1. Review regime distribution - did market conditions change?
2. Review cost decomposition - did costs increase?
3. Review BTC specifically - turnover issue?

**Resolution:**
1. Document drift in weekly notes
2. If persistent (3+ weeks), flag for review
3. Consider adjusting confidence thresholds
4. **DO NOT** reduce realism to hide drift

---

## BTC Turnover Reduction

BTC has shown edge collapse under realistic friction. The following measures are implemented:

### Current BTC-Specific Rules

| Parameter | Default | BTC Value |
|-----------|---------|-----------|
| Min decision interval | 1h | 4h |
| Max decisions/day | unlimited | 3 |
| Min expected value | 1x cost | 2x cost |
| Min confidence | 60% | 75% |
| Cooldown after 2 losses | 2h | 6h |

### Enable BTC Turnover Reducer

```python
from bot.rl.btc_turnover_reduction import (
    BTCTurnoverConfig,
    get_btc_turnover_reducer,
)

# Use default restrictive config
reducer = get_btc_turnover_reducer()

# Check if trade allowed
gate = reducer.check_decision_allowed(
    symbol="BTC-USD",
    position_value_usd=1000,
    expected_profit_pct=1.0,
    confidence=0.8,
)

if not gate.allowed:
    print(f"BTC trade blocked: {gate.reason}")
```

### Monitor BTC Turnover

Run BTC-specific diagnosis weekly:

```python
from bot.rl.btc_diagnosis import BTCDiagnosisTool

tool = BTCDiagnosisTool()
report = tool.analyze_from_backtest_results(btc_results)
print(f"Primary cause: {report.primary_cause}")
print(f"Recoverable: {report.estimated_recoverable_pct:.1f}%")
```

---

## Phase 2C Promotion Gates

### Hard Requirements (ALL must be true)

| # | Gate | Required | How to Verify |
|---|------|----------|---------------|
| 1 | Data collection | 12+ weeks | Check week counter |
| 2 | Positive edge | ΔPnL > 0 consistently | Weekly reports |
| 3 | Hit rate | ≥ 55% | Weekly reports |
| 4 | Incremental Sharpe | ≥ 0.3 | Weekly reports |
| 5 | Max drawdown | < 5% | Weekly reports |
| 6 | BTC mitigation | Validated | BTC diagnosis shows improvement |
| 7 | Human sign-off | Documented | Manual approval |

### Gate Check Process

1. **Weekly:** Update `data/phase2b_gate_progress.json`
2. **Monthly:** Review with stakeholders
3. **At week 12+:** Full gate review

### NO-GO Conditions

If ANY of the following are true, Phase 2C is NO-GO:

- Less than 12 weeks of data
- Edge is negative or inconsistent
- Hit rate < 55%
- Incremental Sharpe < 0.3
- Drawdown > 5%
- BTC degradation not addressed
- No human sign-off

### Promotion Checklist

Before requesting Phase 2C promotion:

- [ ] 12+ weeks of shadow data collected
- [ ] All weekly reports generated and archived
- [ ] Counterfactual edge positive in ≥10 of 12 weeks
- [ ] Edge stable across multiple regimes
- [ ] BTC diagnosis shows improvement trend
- [ ] No critical alerts in past 4 weeks
- [ ] Human review completed
- [ ] Sign-off documented

---

## Report Formats

### Daily Health Report Schema

```json
{
  "date": "YYYY-MM-DD",
  "timestamp": "ISO8601",
  "shadow_collection": {
    "logging_healthy": true,
    "decisions_today": 50,
    "pending_decisions": 2,
    "total_all_time": 1500
  },
  "capital_preservation": {
    "current_level": "normal",
    "transitions_today": 0,
    "alert": false
  },
  "strategy_weighting": {
    "enabled": true,
    "shifts_today": 1,
    "max_shift_observed": 0.05,
    "clamp_exceeded": false
  },
  "tradegate": {
    "rejections_today": 10,
    "rejection_reasons": {"low_confidence": 8, "high_risk": 2}
  },
  "summary": {
    "overall_health": "HEALTHY",
    "alerts": [],
    "recommendations": []
  }
}
```

### Weekly Report Schema

```json
{
  "week_ending": "YYYY-MM-DD",
  "week_number": 5,
  "data_collection": {
    "total_decisions": 350,
    "by_symbol": {"BTC-USD": 50, "ETH-USD": 100},
    "by_regime": {"bull": 200, "sideways": 100}
  },
  "counterfactual_evaluation": { ... },
  "symbol_edge_analysis": {
    "BTC-USD": {
      "decisions": 50,
      "delta_pnl": 150.00,
      "hit_rate": 0.58,
      "sharpe": 0.45,
      "edge_positive": true
    }
  },
  "regime_edge_analysis": { ... },
  "cost_decomposition": { ... },
  "confidence_sweep": {
    "best_threshold": 0.75,
    "results": [ ... ]
  },
  "drift_check": { ... },
  "btc_diagnosis": { ... },
  "summary": {
    "overall_edge_positive": true,
    "edge_stable_across_regimes": true,
    "promotion_gate_progress": { ... }
  }
}
```

---

## Contacts & Escalation

| Situation | Escalate To |
|-----------|-------------|
| CRITICAL health status | On-call engineer |
| Clamp violation | Risk team |
| Capital preservation lockdown | Risk team |
| 3+ weeks drift | Quant lead |
| Phase 2C readiness | All stakeholders |

---

## Appendix: Cron Setup

```cron
# Daily health check at 6 AM
0 6 * * * cd /path/to/algo_trading_lab && python scripts/shadow/run_daily_shadow_health.py >> logs/daily_health.log 2>&1

# Weekly report reminder (Mondays at 9 AM)
0 9 * * 1 echo "Run weekly shadow report: python scripts/shadow/run_weekly_shadow_report.py --week N" | mail -s "Phase 2B Weekly Report Due" team@example.com
```

---

---

## April 1st Micro-Live Readiness

### Overview

Before enabling `LIVE_MODE=true` for real trading, the system must pass all readiness
checks. Use the `/health/readiness` endpoint to verify system status.

**CRITICAL RULE:** Do NOT enable `LIVE_MODE` unless `live_rollout_readiness == GO`

### Using GET /health/readiness

```bash
# Quick check via CLI tool (recommended)
python scripts/ops/health_check.py

# Direct API call
curl http://localhost:8000/health/readiness | python -m json.tool
```

The response includes both general readiness and live rollout readiness:

```json
{
  "overall_readiness": "GO",
  "live_rollout_readiness": "GO",
  "live_rollout_reasons": ["All live rollout criteria met"],
  "live_rollout_next_actions": ["System ready for April 1st live rollout"],
  "components": { ... }
}
```

### Live Rollout Readiness Statuses

| Status | Meaning | Action |
|--------|---------|--------|
| **GO** | All criteria met | Safe to enable `LIVE_MODE=true` |
| **CONDITIONAL** | Minor issues | Fix issues before enabling live mode |
| **NO_GO** | Critical issues | Do NOT enable live mode under any circumstances |

### Criteria for Live Rollout GO

All of the following must be true:

| Check | Requirement |
|-------|-------------|
| Kill switch | Not active |
| Capital preservation | Not in LOCKDOWN/CRISIS |
| Shadow health | Not CRITICAL |
| Daily reports | Generated within last 24h |
| Paper trading streak | ≥ 14 consecutive days |
| Paper trading weeks | ≥ 2 weeks counted |
| Heartbeat | Recent (< 2 hours) |
| Turnover block rate | Between 5% and 70% |
| Execution realism | No significant drift |
| CRITICAL alerts | None in last 14 days |

### Troubleshooting Table

| Reason | Status | Action |
|--------|--------|--------|
| "Kill switch is active" | NO_GO | Run `rm data/live_kill_switch.txt` or unset `LIVE_KILL_SWITCH` env var. Investigate why it was activated. |
| "Capital preservation in LOCKDOWN/CRISIS mode" | NO_GO | Wait for automatic recovery. Do NOT manually override. Review what triggered escalation. |
| "Shadow data collection health is CRITICAL" | NO_GO | Check if paper trading is running. Verify shadow collector is enabled. Review `data/rl/shadow_decisions.jsonl`. |
| "Live mode enabled but symbol allowlist is empty" | NO_GO | Set `LIVE_SYMBOL_ALLOWLIST=ETH/USDT` or add symbols to config. |
| "Daily health reports missing in last 24h" | NO_GO | Run `python scripts/shadow/run_daily_shadow_health.py`. Verify cron job is configured. |
| "PAPER_LIVE streak (X days) below minimum (14 days)" | CONDITIONAL | Continue paper trading. Streak must reach 14 consecutive days. |
| "PAPER_LIVE weeks counted (X) below minimum (2)" | CONDITIONAL | Continue paper trading. Must have 2+ weeks of data. |
| "Shadow collector heartbeat not recent" | CONDITIONAL | Verify paper trading process is running. Check for exceptions in logs. |
| "Turnover block rate (X%) exceeds 70%" | CONDITIONAL | Review turnover governor settings. May be over-throttling. Consider relaxing thresholds. |
| "Turnover block rate (X%) below 5%" | CONDITIONAL | Review turnover governor settings. May be too loose. Consider tightening thresholds. |
| "Execution realism degradation detected" | CONDITIONAL | Investigate slippage model. Compare recent vs prior 7-day averages. |
| "X CRITICAL alert(s) in last 14 days" | CONDITIONAL | Review and resolve the alerts. Ensure root causes are addressed. |

### Automated Health Check Script

Use the CLI tool for daily checks:

```bash
# Full status report
python scripts/ops/health_check.py

# For automation (exits non-zero if NO_GO)
python scripts/ops/health_check.py --quiet
if [ $? -ne 0 ]; then
    echo "System NOT ready for live trading"
fi
```

### Startup Safety

The system automatically validates readiness when `LIVE_MODE=true`:

1. On startup, if `LIVE_MODE=true`, the system checks `live_rollout_readiness`
2. If readiness is NOT `GO`, the system:
   - Logs a CRITICAL error
   - Activates the kill switch automatically
   - Blocks all live trades (paper trading continues normally)
3. This ensures you cannot accidentally start live trading with an unhealthy system

### Pre-Launch Checklist

Before April 1st go-live:

- [ ] `python scripts/ops/health_check.py` returns all green
- [ ] `live_rollout_readiness == GO`
- [ ] 14+ day consecutive paper trading streak
- [ ] 2+ weeks of shadow data collected
- [ ] No CRITICAL alerts in past 14 days
- [ ] Turnover block rate in 5-70% healthy band
- [ ] Daily health reports running via cron
- [ ] Kill switch mechanism tested
- [ ] Telegram/alert notifications configured
- [ ] Sign-off from stakeholders documented

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-29 | 1.1 | Added April 1st micro-live readiness section |
| 2026-01-29 | 1.0 | Initial runbook |
