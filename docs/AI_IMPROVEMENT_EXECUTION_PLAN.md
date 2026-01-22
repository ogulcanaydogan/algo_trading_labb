# AI Improvement Execution Plan

## Objective
Raise decision accuracy and win rate while ensuring the system continuously learns from mistakes, adapts to regime shifts, and enforces safer leverage/shorting behavior.

## Success Metrics
- ML accuracy (out-of-sample): ≥ 80%
- Live/paper win rate: ≥ 60%
- Max drawdown: ≤ 12%
- Drift response: automatic threshold tightening within 1 window

## Prioritized Workstreams

### 1) Confidence Calibration + Unified Gating
- [x] Calibrate prediction confidence and use calibrated values for thresholds
- [x] Apply regime- and performance-adjusted thresholds to ML signals
- [x] Persist confidence metadata in trades for analysis

### 2) Learning Feedback in Decision Thresholds
- [x] Use OptimalActionTracker EV to suppress trades in low-EV states
- [x] Enforce daily risk budget gating (AI Brain) before order placement
- [x] Track signal accuracy and tighten thresholds when performance degrades

### 3) Regime/Leverage/Sizing Controls
- [x] Apply regime strategy multipliers to position size and stops
- [x] Require higher confidence for short signals in volatile/crash regimes
- [x] Block trades when regime is unknown or contradictory to action

### 4) Monitoring & Drift Response
- [x] Hook ML signal outcomes into ModelMonitor for calibration + drift alerts
- [x] Auto tighten thresholds on drift/high-loss periods

## Implementation Notes
- ML signal generation lives in [`bot/ml_signal_generator.py`](bot/ml_signal_generator.py:1)
- Execution and gating live in [`bot/unified_engine.py`](bot/unified_engine.py:1)
- State persistence in [`bot/unified_state.py`](bot/unified_state.py:1)
- Calibration/monitoring in [`bot/ml/model_monitor.py`](bot/ml/model_monitor.py:1)

## Rollout Plan
1. Enable calibration + performance-adjusted thresholds in paper mode
2. Validate results over 50–100 trades
3. If metrics pass, allow more aggressive shorting/leverage in controlled regimes

## Change Log
- 2026-01-21: Implemented calibration + drift-aware gating, learning feedback guards, regime-aware sizing/risk checks, and signal metadata persistence across [`bot/ml_signal_generator.py`](bot/ml_signal_generator.py:1), [`bot/ml/model_monitor.py`](bot/ml/model_monitor.py:1), [`bot/unified_engine.py`](bot/unified_engine.py:1), and [`bot/unified_state.py`](bot/unified_state.py:1).
- Planned changes will be documented in [`LOCAL_AI_IMPROVEMENTS.md`](LOCAL_AI_IMPROVEMENTS.md:1)
