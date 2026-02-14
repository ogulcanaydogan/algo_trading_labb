# Model Drift Monitoring System

## Overview

The Model Drift Monitoring System tracks live ML model predictions, compares performance against backtest baselines, and generates alerts when accuracy degradation is detected.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Model Drift Monitoring System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │ PredictionLogger │───▶│  DriftMonitor    │                  │
│  │                  │    │                  │                  │
│  │ • log_prediction │    │ • detect_drift   │                  │
│  │ • resolve_pred   │    │ • set_baseline   │                  │
│  │ • get_resolved   │    │ • rolling_acc    │                  │
│  └──────────────────┘    └────────┬─────────┘                  │
│                                   │                             │
│                    ┌──────────────┴──────────────┐             │
│                    │                             │              │
│           ┌────────▼─────────┐    ┌─────────────▼──────────┐   │
│           │ AlertIntegration │    │  DashboardExporter     │   │
│           │                  │    │                        │   │
│           │ • WhatsApp queue │    │ • JSON export          │   │
│           │ • trade_alerts   │    │ • accuracy trends      │   │
│           └──────────────────┘    └────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Alert Thresholds

| Level | Condition | Action |
|-------|-----------|--------|
| **Warning** | Accuracy drops 5% from baseline | Monitor closely, prepare retraining |
| **Critical** | Accuracy drops 10% from baseline | Immediate retraining, reduce positions 50% |
| **Retrain** | Accuracy below 50% (random) | STOP trading, retrain immediately |

## Components

### 1. PredictionLogger
Records all model predictions with timestamps and resolves them when outcomes are known.

**Key Methods:**
- `log_prediction(model_name, symbol, prediction, confidence, entry_price, horizon_minutes)` → `pred_id`
- `resolve_prediction(pred_id, exit_price)` → `bool`
- `get_resolved_predictions(model_name, window)` → `List[Prediction]`

### 2. DriftMonitor
Compares rolling accuracy against backtest baselines using statistical tests.

**Key Methods:**
- `set_baseline(model_name, symbol, accuracy, ...)` → Set reference metrics
- `detect_drift(model_name, symbol)` → `DriftAlert` or `None`
- `check_all_models()` → `List[DriftAlert]`
- `get_model_status(model_name, symbol)` → `Dict`

### 3. DriftAlertIntegration
Writes drift alerts to the WhatsApp alert queue (`data/trade_alerts.json`).

### 4. DashboardExporter
Exports monitoring data as JSON for dashboard visualization.

**Output Files:**
- `data/ml_monitoring/dashboard/dashboard.json` - All model status
- `data/ml_monitoring/dashboard/accuracy_trends.json` - Accuracy over time
- `data/ml_monitoring/dashboard/{model}_{symbol}_status.json` - Per-model status

## Usage

### Setting Baselines

After training or backtesting a model, set its baseline metrics:

```python
from bot.ml.drift_monitor import get_monitoring_system

system = get_monitoring_system()

# Set baseline from backtest results
system.set_baseline(
    model_name="lstm_btc",
    symbol="BTC/USDT",
    accuracy=0.62,
    precision=0.60,
    recall=0.64,
    f1_score=0.62,
    total_trades=500,
    backtest_period_days=30,
)
```

### Logging Predictions

Log predictions as they happen:

```python
# When model makes a prediction
pred_id = system.log_prediction(
    model_name="lstm_btc",
    symbol="BTC/USDT",
    prediction=1,       # 1=up, -1=down, 0=hold
    confidence=0.75,
    entry_price=42000.0,
    horizon_minutes=60,  # How long until outcome check
)
```

### Resolving Predictions

When the prediction horizon passes:

```python
# Get current price and resolve
system.resolve_prediction(pred_id, exit_price=43500.0)
```

### Checking for Drift

Run periodically (e.g., every 15 minutes via cron):

```python
# Check all models
alerts = system.check_drift()

for alert in alerts:
    print(f"DRIFT: {alert.model_name} - {alert.drift_level}")
    print(f"  Accuracy: {alert.current_accuracy:.1%} (baseline: {alert.baseline_accuracy:.1%})")
    print(f"  Recommendation: {alert.recommendation}")
```

### Convenience Functions

```python
from bot.ml.drift_monitor import (
    set_model_baseline,
    log_model_prediction,
    resolve_model_prediction,
    check_model_drift,
)

# Quick usage
set_model_baseline("my_model", "ETH/USDT", accuracy=0.65)
pred_id = log_model_prediction("my_model", "ETH/USDT", 1, 0.7, 3000.0)
resolve_model_prediction(pred_id, 3100.0)
alerts = check_model_drift()
```

## Integration Points

### 1. WhatsApp Alerts
Drift alerts are automatically written to `data/trade_alerts.json`. Clawdbot polls this file and delivers alerts via WhatsApp.

Alert format example:
```
🚨 MODEL DRIFT ALERT

Model: lstm_btc
Symbol: BTC/USDT
Level: CRITICAL

📉 Accuracy Dropped:
• Current: 52.0%
• Baseline: 65.0%
• Drop: 13.0%

📊 Analysis:
• Window: last 50 predictions
• p-value: 0.0234
• ✓ statistically significant

💡 Recommendation:
Model accuracy dropped 13.0% from baseline. Schedule immediate retraining and reduce position sizes by 50%.
```

### 2. Dashboard JSON
Export data for monitoring dashboards:

```python
dashboard = system.export_dashboard()
# Returns: {
#   "summary": {"total_models": 5, "healthy": 3, "warning": 1, "critical": 1},
#   "models": {...},
#   "thresholds": {...}
# }
```

### 3. Auto-Retraining
Check if retraining is needed:

```python
status = system.get_model_status("lstm_btc", "BTC/USDT")
if status["health"] in ["critical", "unknown"]:
    trigger_retraining(status["model_name"])
```

## Configuration

Customize thresholds via `DriftThresholds`:

```python
from bot.ml.drift_monitor import DriftThresholds, ModelDriftMonitoringSystem

custom_thresholds = DriftThresholds(
    warning_drop=0.03,        # 3% drop for warning
    critical_drop=0.08,       # 8% drop for critical
    random_baseline=0.52,     # Below 52% triggers retrain
    min_predictions=50,       # Need 50+ predictions
    window_short=50,          # Short-term window
    window_medium=100,        # Medium-term window
    window_long=200,          # Long-term window
    check_interval_minutes=30,# Check every 30 min
)

system = ModelDriftMonitoringSystem(thresholds=custom_thresholds)
```

## Rolling Windows

The system tracks accuracy at three time scales:
- **Short (50)**: Fast detection of sudden degradation
- **Medium (100)**: Balanced view of recent performance
- **Long (200)**: Identifies gradual drift trends

## Statistical Significance

Drift detection uses a one-proportion z-test to compare current accuracy against baseline:
- H₀: current_accuracy ≥ baseline_accuracy
- H₁: current_accuracy < baseline_accuracy
- Default α = 0.05

Alerts include p-value and significance flag to help prioritize action.

## File Locations

| File | Purpose |
|------|---------|
| `data/ml_monitoring/prediction_log.json` | All logged predictions |
| `data/ml_monitoring/model_baselines.json` | Baseline metrics per model |
| `data/ml_monitoring/drift_alerts.json` | Alert history |
| `data/ml_monitoring/dashboard/dashboard.json` | Dashboard data |
| `data/trade_alerts.json` | WhatsApp alert queue |

## Best Practices

1. **Set baselines after backtesting** - Use out-of-sample backtest results
2. **Log all predictions** - Even ones not executed as trades
3. **Check drift regularly** - Every 15-30 minutes during trading hours
4. **Act on alerts** - Don't ignore critical alerts
5. **Review trends** - Weekly review of accuracy trends
6. **Update baselines** - After successful retraining

## Scheduled Tasks

Example cron job for drift checking:

```bash
# Check drift every 15 minutes during trading hours
*/15 * * * * cd /path/to/algo_trading_lab && python -c "from bot.ml.drift_monitor import check_model_drift; check_model_drift()"
```

Or via Clawdbot heartbeat in `HEARTBEAT.md`:
```markdown
- [ ] Check ML model drift (every 2h)
```
