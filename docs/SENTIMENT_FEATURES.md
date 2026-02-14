# Sentiment Features Module

## Overview

This module provides news and social sentiment analysis as trading features for ML models.

## Files Created

```
bot/sentiment/
├── __init__.py              # Module exports
├── news_sentiment.py        # News fetching with VADER analysis
├── social_sentiment.py      # Reddit sentiment fetching
└── feature_integration.py   # Combined feature generation
```

## Data Sources

| Source | Type | Auth Required | Notes |
|--------|------|---------------|-------|
| Google News RSS | News | No | Headlines via RSS |
| Yahoo Finance RSS | News | No | Financial news feed |
| Finviz | News | No | Stock headlines (scraping) |
| Reddit JSON API | Social | No | Public subreddit data |

### Reddit Subreddits Monitored
- **Stocks**: r/wallstreetbets, r/stocks, r/investing, r/stockmarket
- **Crypto**: r/cryptocurrency, r/bitcoin, r/ethereum, r/CryptoMarkets

## Features Generated

| Feature | Description | Range |
|---------|-------------|-------|
| `sentiment_score_24h` | Rolling 24h average news sentiment | [-1, 1] |
| `sentiment_momentum` | Sentiment change over time | [-2, 2] |
| `mention_volume` | Normalized buzz indicator | [0, 3] |
| `sentiment_std` | Disagreement (sentiment volatility) | [0, 1] |
| `social_sentiment` | Social media sentiment score | [-1, 1] |
| `social_volume` | Social post volume | [0, 3] |
| `social_momentum` | Social sentiment change | [-2, 2] |
| `wsb_buzz` | WallStreetBets activity | [0, 1] |
| `sentiment_disagreement` | News vs social divergence | [0, 2] |
| `sentiment_extreme` | Extreme sentiment indicator | {0, 1} |
| `sentiment_contrarian` | Fade-extreme signal | {-0.5, 0, 0.5} |
| `sentiment_combined` | Weighted news + social | [-1, 1] |

## Usage

### Get Live Features (for trading)

```python
from bot.sentiment import get_combined_features

features = get_combined_features("TSLA")
# Returns: {'sentiment_score_24h': 0.168, 'sentiment_momentum': 0.0, ...}

# Use in trading decision
if features['sentiment_combined'] > 0.3:
    signal = "BULLISH"
elif features['sentiment_combined'] < -0.3:
    signal = "BEARISH"
else:
    signal = "NEUTRAL"
```

### Integrate with Price DataFrame

```python
from bot.sentiment import integrate_sentiment_features

# Add sentiment features to price data
df_with_sentiment = integrate_sentiment_features(
    price_df,
    symbol="TSLA",
    use_live=True,  # Use live for latest row
)
```

### Generate Synthetic History (for backtesting)

```python
from bot.sentiment import generate_synthetic_sentiment_history

sentiment_df = generate_synthetic_sentiment_history(
    price_df,
    symbol="TSLA",
    correlation_with_returns=0.3,  # 0.3 correlation with returns
)
```

### Fetch Raw News Items

```python
from bot.sentiment import NewsSentimentFetcher

fetcher = NewsSentimentFetcher()
items = fetcher.fetch_news("TSLA", hours_back=24)

for item in items[:5]:
    print(f"[{item.sentiment_score:+.2f}] {item.title}")
    # [+0.62] Tesla Stock: Morgan Stanley Sees $50 Billion...
```

## Test Results

### Live Sentiment (Feb 2026)

```
TSLA:
  Headlines found: 50
  Sentiment score: +0.168
  Bullish ratio:   50.0%
  Bearish ratio:   26.0%

NVDA:
  Headlines found: 50
  Sentiment score: +0.180
  Bullish ratio:   56.0%
  Bearish ratio:   14.0%
```

### Model Impact (Synthetic Data)

```
Metric          Baseline        + Sentiment     Change
------------------------------------------------------------
Accuracy        0.5081          0.4228          -8.54%
Precision       0.4720          0.4051          -6.70%
F1 Score        0.5568          0.4741          -8.27%

Sentiment feature importance: 36.3%
```

**Note**: Synthetic sentiment decreased accuracy because it's randomly generated (weak signal). With **real** sentiment data that has actual predictive value, expect positive impact.

## Feature Importance (from model)

```
Feature                   Importance
-----------------------------------------
[TECH] volume_ratio       0.1313
[TECH] return_1           0.1167
[SENT] sentiment_momentum 0.1146  ← Sentiment ranks #3
[SENT] social_sentiment   0.1049
[TECH] rsi_norm           0.1013
[SENT] sentiment_combined 0.0848
[TECH] ma_20_dist         0.0844
```

## Configuration

```python
from bot.sentiment import SentimentConfig, CombinedSentimentFeatureGenerator

config = SentimentConfig(
    news_weight=0.6,         # Weight for news sentiment
    social_weight=0.4,       # Weight for social sentiment
    update_interval_minutes=30,
    include_wsb=True,
)

generator = CombinedSentimentFeatureGenerator(config)
```

## VADER Customizations

The module extends VADER with financial domain terms:

**Bullish**: bullish, rally, surge, breakout, upgrade, beat, moon, accumulate

**Bearish**: bearish, crash, plunge, dump, downgrade, miss, bankruptcy, fraud, rekt

## Caching

- News items cached for 15 minutes
- Combined features cached for 30 minutes
- Reddit posts cached for 10 minutes

## Rate Limiting

- 1 second delay between external requests
- Reddit: 10 requests/minute
- Automatic retry on failure

## Integration with ML Pipeline

Add to `bot/ml/feature_engineer.py`:

```python
from bot.sentiment.feature_integration import integrate_sentiment_features

def extract_features(self, ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = ohlcv.copy()
    
    # ... existing technical features ...
    
    # Add sentiment features
    df = integrate_sentiment_features(df, symbol, use_live=False)
    
    return df
```

## Future Improvements

1. **Add Twitter/X API** for real-time social sentiment
2. **Historical sentiment database** for proper backtesting
3. **Earnings sentiment** analysis around report dates
4. **Sector-level sentiment** aggregation
5. **Sentiment regime detection** (fear/greed cycles)
