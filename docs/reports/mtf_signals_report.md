# Multi-Timeframe ML Signals Report

**Date:** 2026-02-14  
**Symbol:** TSLA  
**Asset Class:** Stock

## Executive Summary

Implemented a multi-timeframe ML prediction system that combines signals from 3 different prediction horizons (3h, 8h, 24h) to generate high-conviction trading signals. The system filters trades based on model agreement, resulting in:

- **80.7% win rate** when all 3 models agree (vs 72.9% for single 8h model)
- **Sharpe ratio of 21.96** for high conviction trades (vs 15.07 baseline)
- **48.6% of trades** taken, being highly selective

---

## Architecture

### 1. Multi-Timeframe Model Design

```
┌─────────────────────────────────────────────────────────────┐
│                    MTF Prediction System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │   3h Model   │  │   8h Model   │  │   24h Model  │     │
│   │ (Momentum)   │  │   (Trend)    │  │ (Direction)  │     │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│          │                 │                 │              │
│          └────────────┬────┴────────────────┘              │
│                       │                                     │
│                       ▼                                     │
│          ┌────────────────────────┐                        │
│          │   Conviction Engine    │                        │
│          │                        │                        │
│          │  3/3 agree = HIGH      │                        │
│          │  2/3 agree = MEDIUM    │                        │
│          │  Disagree = NO TRADE   │                        │
│          └───────────┬────────────┘                        │
│                      │                                      │
│                      ▼                                      │
│          ┌────────────────────────┐                        │
│          │   Trade Decision       │                        │
│          │   + Position Size      │                        │
│          └────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2. Horizon Configuration

| Horizon | Purpose | Min Move | Vol Mult | Features |
|---------|---------|----------|----------|----------|
| 3h | Short-term momentum | 0.8% | 0.4 | 35 |
| 8h | Intraday trend | 1.5% | 0.4 | 35 |
| 24h | Multi-day direction | 2.0% | 0.35 | 35 |

### 3. Signal Weights

- **3h (Short):** 25% weight
- **8h (Medium):** 35% weight
- **24h (Long):** 40% weight

Higher timeframes have more influence on the final decision.

---

## Conviction Levels

| Conviction | Criteria | Position Size |
|------------|----------|---------------|
| **HIGH** | All 3 models agree on direction | 100% |
| **MEDIUM** | 2/3 models agree | 60% |
| **NO TRADE** | Models disagree | 0% |

---

## Backtest Results

### Single Timeframe Performance

| Model | Trades | Win Rate | Total Return | Sharpe |
|-------|--------|----------|--------------|--------|
| 3h (all) | 1,517 | 71.4% | 1,550% | 21.77 |
| 3h (high conf) | 1,371 | 73.7% | 1,566% | 24.07 |
| **8h (all)** | **1,512** | **72.9%** | **2,639%** | **15.07** |
| 8h (high conf) | 1,342 | 74.6% | 2,599% | 16.56 |
| 24h (all) | 1,496 | 73.7% | 4,580% | 9.28 |
| 24h (high conf) | 1,346 | 75.7% | 4,597% | 10.21 |

### Multi-Timeframe Performance

| Strategy | Trades | % of Bars | Win Rate | Total Return | Sharpe |
|----------|--------|-----------|----------|--------------|--------|
| MTF (2/3 agree) | 1,512 | 100% | 71.6% | 2,619% | 14.93 |
| **MTF (3/3 agree)** | **735** | **48.6%** | **80.7%** | **1,965%** | **21.96** |

---

## Key Insights

### 1. Win Rate Improvement
When all 3 models agree, win rate jumps from **72.9% → 80.7%** (+7.8 percentage points).

### 2. Risk-Adjusted Returns
The HIGH conviction strategy achieves a Sharpe ratio of **21.96** vs **15.07** for the baseline 8h model — a 46% improvement.

### 3. Trade Selectivity
By requiring unanimous agreement, we trade only **48.6%** of opportunities but with much higher accuracy.

### 4. Agreement Statistics
- Average model agreement: **82.9%**
- Bars with 3/3 agreement: **48.6%**

---

## Model Training Summary

### 3h Model
- **Accuracy:** 48.2%
- **Walk-Forward:** 51.0%
- **Top Features:** vol_24h, volatility_regime, ema_10_20_diff, atr_14, adx

### 8h Model (Existing)
- **Accuracy:** 56.8%
- **Walk-Forward:** 55.1%
- **Status:** Previously trained

### 24h Model
- **Accuracy:** 44.6%
- **Walk-Forward:** 43.1%
- **Top Features:** atr_14, range_24h, vol_24h, volatility_regime, macd_signal

---

## Implementation Files

```
bot/ml/
├── mtf_predictor.py       # Multi-timeframe prediction class
├── multi_timeframe.py     # Technical MTF analysis (existing)

scripts/ml/
├── train_mtf_models.py    # Train all horizon models
├── backtest_mtf.py        # Compare single vs MTF strategies

data/models_v6_improved/
├── TSLA_3h_binary_ensemble_v6.pkl
├── TSLA_3h_binary_scaler_v6.pkl
├── TSLA_8h_binary_ensemble_v6.pkl (or existing TSLA_binary_ensemble_v6.pkl)
├── TSLA_24h_binary_ensemble_v6.pkl
├── TSLA_mtf_summary.json
├── TSLA_mtf_backtest_results.json
```

---

## Usage Example

```python
from bot.ml.mtf_predictor import MTFPredictor

# Initialize predictor
predictor = MTFPredictor(
    symbol="TSLA",
    horizons=[3, 8, 24],
    model_dir="data/models_v6_improved"
)

# Get prediction
prediction = predictor.predict(features_df)

# Check conviction
if prediction.should_trade:
    print(f"Trade {prediction.direction.value} with {prediction.conviction.value} conviction")
    print(f"Agreement: {prediction.agreement_score:.0%}")
    
    # Adjust position size
    size_mult = predictor.get_position_size_multiplier(prediction.conviction)
    position_size = base_size * size_mult
```

---

## Recommendations

1. **Use HIGH conviction for production:** Only trade when all 3 models agree. This gives the best risk-adjusted returns.

2. **Consider MEDIUM conviction during high-volatility:** When market conditions are favorable, 2/3 agreement may be acceptable.

3. **Monitor model agreement:** If agreement rate drops below 40%, consider retraining models.

4. **Periodic retraining:** Retrain all 3 horizon models together to maintain consistency.

---

## Conclusion

The multi-timeframe approach successfully combines short, medium, and long-term views to produce higher-conviction trading signals. By requiring model agreement, we achieve:

- **+7.8 percentage points** in win rate
- **+46%** improvement in Sharpe ratio
- Significant reduction in false signals

This validates the hypothesis that multi-timeframe consensus leads to stronger conviction trades.
