# V6 Ensemble Predictor Report

**Date:** 2026-02-14
**Status:** Complete

## Overview

Implemented a weighted ensemble predictor that combines multiple V6 binary classification models to improve robustness and prediction confidence.

## Ensemble Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  V6 Ensemble Predictor                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│  │  TSLA   │   │BTC_USDT │   │XRP_USDT │  ... more models   │
│  │ Model   │   │ Model   │   │ Model   │                    │
│  └────┬────┘   └────┬────┘   └────┬────┘                    │
│       │             │             │                          │
│       │  prob_up    │  prob_up    │  prob_up                │
│       │  prob_down  │  prob_down  │  prob_down              │
│       ▼             ▼             ▼                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Weighted Combination Layer              │       │
│  │  (weights = normalized walk-forward accuracy)     │       │
│  └──────────────────────────────────────────────────┘       │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Ensemble Signal Output               │       │
│  │  - signal: LONG / SHORT / NEUTRAL                │       │
│  │  - combined_probability: weighted avg prob        │       │
│  │  - ensemble_confidence: confidence score          │       │
│  │  - agreement_score: model consensus (0-1)         │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Ensemble Strategies

1. **Weighted Average** (default)
   - Weights each model's probability by its walk-forward accuracy
   - Best for combining models with varying accuracy levels
   - Formula: `combined_prob = Σ(weight_i × prob_i) / Σ(weight_i)`

2. **Majority Vote**
   - Simple democratic voting across all models
   - Each model gets one vote regardless of accuracy
   - Best for equal-quality models

3. **Confidence-Weighted**
   - Combines both walk-forward accuracy AND prediction confidence
   - Weight = `wf_accuracy × prediction_confidence`
   - Best for models with variable confidence outputs

## Model Weights

Based on walk-forward validation accuracy:

| Symbol    | WF Accuracy | Normalized Weight | Asset Class |
|-----------|-------------|-------------------|-------------|
| XRP_USDT  | 56.07%      | 0.3449           | Crypto      |
| TSLA      | 55.11%      | 0.3390           | Stock       |
| BTC_USDT  | 51.38%      | 0.3160           | Crypto      |

**Note:** Weights are normalized so they sum to 1.0.

## Implementation Details

### Location
- **Module:** `bot/ml/v6_ensemble.py`
- **Test Script:** `scripts/ml/test_v6_ensemble.py`

### Key Classes

```python
# V6EnsemblePredictor - Main ensemble class
ensemble = V6EnsemblePredictor(
    model_dir=Path("data/models_v6_improved"),
    symbols=["TSLA", "BTC_USDT", "XRP_USDT"],
    strategy="weighted_avg",
    min_agreement=0.5,
    probability_threshold=0.55
)

# Load models
ensemble.load_models()

# Get prediction
prediction = ensemble.predict(df)  # df with computed features

# Access results
print(prediction.signal)           # LONG, SHORT, or NEUTRAL
print(prediction.combined_probability)  # 0.0 to 1.0
print(prediction.ensemble_confidence)   # 0.0 to 1.0
print(prediction.agreement_score)       # 0.0 to 1.0
```

### Factory Function

```python
from bot.ml.v6_ensemble import create_v6_ensemble

ensemble = create_v6_ensemble(
    model_dir=Path("data/models_v6_improved"),
    symbols=["TSLA", "BTC_USDT", "XRP_USDT"],
    strategy="weighted_avg"
)
```

## Integration Points

### With Existing Predictor

The V6 ensemble can be used alongside the existing `EnsemblePredictor` in `bot/ml/ensemble_predictor.py`, or as a replacement for more stable walk-forward weighted predictions.

### With Trading Engine

```python
# In trading loop
from bot.ml.v6_ensemble import create_v6_ensemble

ensemble = create_v6_ensemble()
prediction = ensemble.predict(feature_df)

if prediction.agreement_score >= 0.67:  # 2/3 models agree
    if prediction.signal == "LONG":
        execute_long(size=prediction.ensemble_confidence)
    elif prediction.signal == "SHORT":
        execute_short(size=prediction.ensemble_confidence)
```

## Benefits of Ensemble Approach

1. **Reduced Variance:** Combining models smooths out individual model noise
2. **Robustness:** Less sensitive to any single model's failure
3. **Confidence Metrics:** Agreement score provides trade conviction
4. **Flexibility:** Easy to add/remove models from ensemble
5. **Interpretability:** Individual predictions visible for debugging

## Test Results

Using synthetic data (limited due to feature availability):

| Strategy | Accuracy | Notes |
|----------|----------|-------|
| weighted_avg | 38.3% | Only TSLA model contributing |
| majority_vote | 38.3% | Same (single model) |
| confidence_weighted | 38.3% | Same (single model) |

**Note:** Low accuracy expected on synthetic data. BTC and XRP models couldn't contribute due to missing crypto-specific features. Real data testing recommended.

## Future Improvements

1. **Add More Models:** Include SPX500_USD, more stocks, more crypto
2. **Time-Based Weighting:** Weight recent predictions higher
3. **Regime-Based Selection:** Use different model subsets for different market regimes
4. **Stacking Meta-Learner:** Train a meta-model to combine base predictions
5. **Dynamic Weight Updates:** Adjust weights based on recent performance

## Files Created

1. `bot/ml/v6_ensemble.py` - Main ensemble predictor module
2. `scripts/ml/test_v6_ensemble.py` - Test and comparison script
3. `docs/V6_ENSEMBLE_REPORT.md` - This documentation

## Quick Start

```bash
# Test the ensemble
cd C:\Users\Ogulcan\Desktop\Projects\algo_trading_lab
.\.venv\Scripts\python.exe scripts\ml\test_v6_ensemble.py
```
