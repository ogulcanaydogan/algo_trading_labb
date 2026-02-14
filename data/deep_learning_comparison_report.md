# Deep Learning vs V6 Ensemble Comparison Report

## Executive Summary

**Recommendation: STICK WITH V6 ENSEMBLE**

Deep learning models (LSTM, BiLSTM, Transformer) significantly underperformed compared to the V6 gradient boosting ensemble on TSLA stock prediction. V6 remains the better choice for this dataset.

## Results Summary

| Model | Test Acc | AUC-ROC | WF Acc | Stability (std) | Training Time |
|-------|----------|---------|--------|-----------------|---------------|
| **V6 Ensemble** | **56.85%** | **58.11%** | **55.11%** | **0.036 (stable)** | ~15s |
| LSTM v1 | 54.63% | 52.67% | 49.38% | 0.061 (unstable) | 57s |
| BiLSTM+Attention v2 | 51.13% | 55.73% | 46.59% | 0.085 (unstable) | 99s |
| Transformer v2 | 44.33% | 41.54% | 44.36% | 0.045 | 148s |

### Key Metrics Comparison (TSLA)

```
Metric              V6 Ensemble    LSTM v1     BiLSTM v2   Transformer v2
--------------------------------------------------------------------------
Test Accuracy       56.85%         54.63%      51.13%      44.33%
                                   (-2.2%)     (-5.7%)     (-12.5%)

Walk-Forward Acc    55.11%         49.38%      46.59%      44.36%
                                   (-5.7%)     (-8.5%)     (-10.8%)

Stability (std)     0.036          0.061       0.085       0.045
                    STABLE         UNSTABLE    VERY UNSTABLE   UNSTABLE
```

## Analysis

### Why V6 Outperforms Deep Learning

1. **Limited Data**: Only ~2,000 training samples - far too few for deep learning to learn meaningful patterns. V6 works better with small datasets.

2. **Low Signal-to-Noise Ratio**: Financial time series have very low predictive signal. Gradient boosting can still extract small signals while neural networks tend to overfit to noise.

3. **Non-Stationarity**: Financial markets change over time. Tree-based ensembles with proper regularization handle regime changes better than neural networks.

4. **Feature Engineering**: V6's hand-crafted features (RSI, MACD, momentum regimes) capture domain knowledge that neural networks would need much more data to discover.

5. **Walk-Forward Instability**: Deep learning models showed high variance across folds (std up to 0.085), indicating unreliable generalization.

### Deep Learning Model Architectures Tested

1. **LSTM v1** (67K params)
   - 2-layer LSTM, 64 hidden units
   - 24-step sequence
   - Binary classification head
   
2. **BiLSTM+Attention v2** (164K params)
   - Bidirectional LSTM
   - Self-attention mechanism
   - 48-step sequence
   - Feature selection (40 features)
   
3. **Transformer v2** (108K params)
   - 2-layer encoder
   - 4-head attention
   - 48-step sequence
   - Feature selection (40 features)

## When Deep Learning MIGHT Work

Deep learning could outperform V6 if:

1. **Much More Data**: 50,000+ samples (years of tick data)
2. **Higher Frequency**: Tick or minute data with clearer patterns
3. **Alternative Data**: Sentiment, order flow, or other non-price features
4. **Pre-trained Models**: Transfer learning from large financial models
5. **Careful Architecture**: Custom architectures for financial time series (e.g., Temporal Fusion Transformer)

## Conclusion

For the current TSLA dataset (~2,000 hourly samples), the V6 ensemble is clearly superior:

- **+5-11% better walk-forward accuracy**
- **Much more stable predictions**
- **3-10x faster training**
- **Simpler to maintain and deploy**

**Continue using V6 models for production trading.**

---

Generated: 2025-01-15
Dataset: TSLA (3,490 hourly bars, ~2,000 signal samples)
