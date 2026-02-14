# TSLA Paper Trading - V6 Improved Model

## Overview

This document describes the TSLA paper trading setup using the V6 improved model with **55.1% walk-forward accuracy**.

## Model Details

| Metric | Value |
|--------|-------|
| Model Type | VotingClassifier (XGBoost + RF + GB) |
| Walk-Forward Accuracy | 55.1% (9 valid folds) |
| Test Accuracy | 56.8% |
| Test AUC | 0.58 |
| High-Confidence Accuracy | 57.1% (319 samples) |
| Prediction Horizon | 8 hours |
| Confidence Threshold | 55% |
| Features | 35 technical indicators |

## Files

- `run_tsla_paper.py` - Main paper trading script
- `start_tsla_paper.ps1` - PowerShell startup script
- `bot/ml/v6_feature_extractor.py` - Feature engineering matching training
- `data/models/TSLA_*.pkl` - Model files

## Quick Start

### Check Status
```powershell
python run_tsla_paper.py --status
```

### Start Paper Trading
```powershell
# Using Python directly
python run_tsla_paper.py

# Or using PowerShell script
.\start_tsla_paper.ps1 start
```

### Stop Trading
```powershell
.\start_tsla_paper.ps1 stop
```

### Monitor Logs
```powershell
Get-Content data\tsla_paper\logs\tsla_paper_*.log -Wait -Tail 50
```

## Configuration

Edit `run_tsla_paper.py` to modify:

```python
TSLA_CONFIG = {
    "symbol": "TSLA",
    "model_version": "v6_improved",
    "walk_forward_accuracy": 0.551,
    "confidence_threshold": 0.55,    # Minimum confidence for signals
    "position_size_pct": 0.10,       # 10% of capital per trade
    "max_positions": 1,
    "stop_loss_pct": 0.015,          # 1.5% stop loss
    "take_profit_pct": 0.03,         # 3% take profit
    "trailing_stop_pct": 0.01,       # 1% trailing stop
    "loop_interval_seconds": 60,     # Check every minute
    "initial_capital": 10000,        # Starting capital
}
```

## Trading Logic

1. **Entry**: BUY signal with confidence > 55%
2. **Exit Conditions**:
   - Stop loss: -1.5%
   - Take profit: +3%
   - Signal reversal (SELL with high confidence)

## Risk Management

- Single position only (no pyramiding)
- 10% position size per trade
- 1.5% stop loss = max loss per trade ~0.15% of capital
- Paper trading only (no real money)

## Walk-Forward Results

| Fold | Accuracy | AUC | HC Accuracy | HC Samples |
|------|----------|-----|-------------|------------|
| 1 | 50.9% | 0.518 | 51.9% | 162 |
| 2 | 53.1% | 0.636 | 54.0% | 150 |
| 3 | 53.1% | 0.451 | 55.2% | 143 |
| 4 | **60.0%** | 0.573 | **59.2%** | 152 |
| 5 | 52.0% | 0.484 | 50.0% | 144 |
| 6 | 54.9% | 0.571 | 56.5% | 138 |
| 7 | 56.6% | 0.509 | 54.5% | 132 |
| 8 | 53.1% | 0.593 | 54.8% | 135 |
| 9 | **62.3%** | **0.666** | **63.4%** | 123 |
| **Avg** | **55.1%** | **0.556** | **55.5%** | - |

## Feature Importance (Top 10)

1. `vol_24h` - 24-hour volatility
2. `trend_strength` - ADX indicator
3. `adx` - Average Directional Index
4. `macd_signal` - MACD signal line
5. `volatility_regime` - Current vs historical volatility
6. `atr_14` - 14-period ATR
7. `vol_12h` - 12-hour volatility
8. `plus_di` - Positive directional indicator
9. `range_12h` - 12-hour price range
10. `atr_ratio` - ATR relative to price

## Notes

- Market hours: NYSE 9:30 AM - 4:00 PM EST
- Data source: yfinance (Yahoo Finance)
- Signals generated every 60 seconds during market hours
- State persisted to `data/tsla_paper/state.json`
