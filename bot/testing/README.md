# A/B Testing Framework

A comprehensive framework for comparing trading strategies with statistical rigor.

## Overview

The A/B testing framework allows you to:
- Run multiple strategies **in parallel** with isolated paper portfolios
- Feed strategies the **same market data** for fair comparison
- Track detailed **performance metrics** (returns, Sharpe, drawdown, win rate)
- Perform **statistical analysis** (t-tests, confidence intervals, effect size)
- Generate **JSON dashboard** output for visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ABTest                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Unified Data Feed                    │  │
│  │            (Same prices for all strategies)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│       ┌───────────────────┼───────────────────┐            │
│       ▼                   ▼                   ▼            │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐        │
│  │ Strategy │        │ Strategy │        │ Strategy │        │
│  │    A     │        │    B     │        │    C     │        │
│  └────┬────┘        └────┬────┘        └────┬────┘        │
│       │                  │                  │              │
│       ▼                  ▼                  ▼              │
│  ┌─────────┐        ┌─────────┐        ┌─────────┐        │
│  │ Paper   │        │ Paper   │        │ Paper   │        │
│  │Portfolio│        │Portfolio│        │Portfolio│        │
│  │    A    │        │    B    │        │    C    │        │
│  └─────────┘        └─────────┘        └─────────┘        │
│                           │                                 │
│       └───────────────────┼───────────────────┘            │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Statistical Comparison                   │  │
│  │        (T-tests, Sharpe, Confidence Intervals)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 ABTestResult                          │  │
│  │              (JSON Dashboard Output)                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from bot.testing.ab_framework import ABTest, ABTestConfig
from bot.strategy_interface import EMACrossoverStrategy, MomentumStrategy

# 1. Configure the test
config = ABTestConfig(
    initial_balance=10000,
    trading_fee_pct=0.1,
    slippage_pct=0.05,
)

# 2. Create A/B test
ab_test = ABTest(config)

# 3. Register strategies to compare
ab_test.register_strategy("ema", EMACrossoverStrategy())
ab_test.register_strategy("momentum", MomentumStrategy())

# 4. Run the test
result = ab_test.run(market_data)  # market_data is List[Dict] of OHLCV+indicators

# 5. View results
result.print_summary()
result.save_dashboard("results.json")
```

### Using the Convenience Function

```python
from bot.testing.ab_framework import run_ab_test
from bot.strategy_interface import EMACrossoverStrategy, MomentumStrategy

result = run_ab_test(
    strategies={
        "ema": EMACrossoverStrategy(),
        "momentum": MomentumStrategy(),
    },
    market_data=bars,
    output_path="ab_results.json"
)
```

## Adding Custom Strategies

Strategies must implement the `Strategy` interface from `bot.strategy_interface`:

```python
from bot.strategy_interface import Strategy, StrategySignal, StrategyAction, MarketState

class MyCustomStrategy(Strategy):
    @property
    def name(self) -> str:
        return "my_custom_strategy"
    
    @property
    def version(self) -> str:
        return "1.0"
    
    @property
    def suitable_regimes(self) -> list:
        return ["trending", "volatile"]
    
    def predict(self, state: MarketState) -> StrategySignal:
        # Access indicators
        rsi = state.indicators.get("rsi", 50)
        ema_fast = state.indicators.get("ema_fast", 0)
        ema_slow = state.indicators.get("ema_slow", 0)
        
        # Generate signal
        if ema_fast > ema_slow and rsi < 70:
            return StrategySignal(
                action=StrategyAction.BUY,
                direction="long",
                confidence=0.7,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                position_size_pct=5.0,
                reasoning="Bullish crossover with RSI confirmation"
            )
        elif ema_fast < ema_slow and rsi > 30:
            return StrategySignal(
                action=StrategyAction.SELL,
                direction="short",
                confidence=0.7,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                position_size_pct=5.0,
                reasoning="Bearish crossover with RSI confirmation"
            )
        else:
            return StrategySignal(
                action=StrategyAction.HOLD,
                direction="flat",
                confidence=0.0,
                reasoning="No clear signal"
            )

# Register it
ab_test.register_strategy("my_strategy", MyCustomStrategy())
```

## Market Data Format

Market data should be a list of dictionaries with OHLCV and indicators:

```python
market_data = [
    {
        "timestamp": "2024-01-01T00:00:00",
        "symbol": "BTC/USDT",
        "open": 50000.0,
        "high": 50500.0,
        "low": 49500.0,
        "close": 50200.0,
        "volume": 1000.0,
        # Indicators
        "ema_fast": 50100.0,  # or "ema_12"
        "ema_slow": 49900.0,  # or "ema_26"
        "rsi": 55.0,          # or "rsi_14"
        "macd": 200.0,
        "macd_signal": 180.0,
        "macd_hist": 20.0,
        "atr": 500.0,         # or "atr_14"
        "adx": 28.0,          # or "adx_14"
        "bb_upper": 51000.0,
        "bb_middle": 50000.0,
        "bb_lower": 49000.0,
    },
    # ... more bars
]
```

## Configuration Options

```python
config = ABTestConfig(
    # Portfolio settings
    initial_balance=10000.0,      # Starting capital per strategy
    trading_fee_pct=0.1,          # Trading fee (0.1% = 10 bps)
    slippage_pct=0.05,            # Slippage (0.05%)
    
    # Test settings
    symbols=["BTC/USDT"],         # Symbols to test
    test_duration_bars=None,      # None = use all data
    warmup_bars=50,               # Skip first N bars for indicator warmup
    
    # Risk settings
    max_position_pct=100.0,       # Max position size
    max_drawdown_stop=25.0,       # Stop test if drawdown exceeds this
    
    # Statistical settings
    confidence_level=0.95,        # For confidence intervals (95%)
    min_trades_for_significance=30,
    
    # Output settings
    save_trade_history=True,
    save_equity_curve=True,
)
```

## Statistical Analysis

### Metrics Compared

For each strategy pair, the framework computes:

| Metric | Description |
|--------|-------------|
| **T-statistic** | Welch's t-test comparing mean returns |
| **P-value** | Statistical significance (< 0.05 = significant) |
| **Cohen's d** | Effect size (negligible/small/medium/large) |
| **95% CI** | Confidence interval on return difference |
| **Sharpe Ratio** | Risk-adjusted returns comparison |

### Interpretation

```json
{
  "t_test": {
    "t_statistic": 2.45,
    "p_value": 0.015,
    "is_significant": true     // p < 0.05
  },
  "effect_size": {
    "cohens_d": 0.35,
    "interpretation": "small"  // |d| < 0.2: negligible, < 0.5: small, < 0.8: medium, else large
  },
  "conclusion": {
    "winner": "strategy_a",
    "confidence": "high"       // high: significant + 30+ trades, medium: p<0.1, low: otherwise
  }
}
```

## Dashboard Output

The framework generates a JSON dashboard:

```json
{
  "summary": {
    "test_id": "abc123",
    "strategies_tested": ["ema", "momentum", "mean_reversion"],
    "bars_processed": 1000,
    "overall_winner": "ema",
    "confidence": "high"
  },
  "performance_comparison": {
    "ema": {
      "total_return_pct": 5.23,
      "win_rate": 0.55,
      "sharpe_ratio": 1.42,
      "max_drawdown_pct": 8.5,
      "total_trades": 45,
      "profit_factor": 1.65
    }
  },
  "head_to_head": [
    {
      "matchup": "ema vs momentum",
      "winner": "ema",
      "p_value": 0.023,
      "significant": true,
      "effect_size": "medium",
      "return_diff": 2.1
    }
  ],
  "rankings": {
    "by_return": ["ema", "mean_reversion", "momentum"],
    "by_sharpe": ["ema", "mean_reversion", "momentum"],
    "by_win_rate": ["ema", "momentum", "mean_reversion"]
  }
}
```

## Example Test Cases

### V6 Model vs V6 + Sentiment

```python
from bot.ml.v6_model import V6Strategy
from bot.ml.v6_sentiment import V6SentimentStrategy

ab_test.register_strategy("v6_base", V6Strategy())
ab_test.register_strategy("v6_sentiment", V6SentimentStrategy())
```

### Single vs Multi-Timeframe

```python
from bot.strategies import SingleTimeframeStrategy, MultiTimeframeStrategy

ab_test.register_strategy("single_tf", SingleTimeframeStrategy(timeframe="1h"))
ab_test.register_strategy("multi_tf", MultiTimeframeStrategy(
    timeframes=["15m", "1h", "4h"]
))
```

### Different Parameter Sets

```python
from bot.strategy_interface import EMACrossoverStrategy

# Conservative
ab_test.register_strategy("ema_conservative", EMACrossoverStrategy({
    "stop_loss_pct": 1.0,
    "take_profit_pct": 2.0,
}))

# Aggressive
ab_test.register_strategy("ema_aggressive", EMACrossoverStrategy({
    "stop_loss_pct": 3.0,
    "take_profit_pct": 6.0,
}))
```

## Run Example

```bash
# Quick comparison (2 strategies, 1000 bars)
python scripts/run_ab_test.py --quick

# Full test (4 strategies, 2000 bars)
python scripts/run_ab_test.py

# Custom settings
python scripts/run_ab_test.py --symbol ETH/USDT --bars 5000
```

## Files

```
bot/testing/
├── __init__.py           # Package exports
├── ab_framework.py       # Main framework
└── README.md             # This file

scripts/
└── run_ab_test.py        # Example test script

ab_test_results/          # Output directory
├── dashboard_*.json      # Dashboard outputs
└── full_results_*.json   # Full results with trades
```

## Best Practices

1. **Use sufficient data** - At least 1000 bars for meaningful statistics
2. **Check trade counts** - Strategies need 30+ trades for statistical significance
3. **Compare like-for-like** - Test strategies in their suitable market regimes
4. **Watch for overfitting** - Use out-of-sample data for final comparison
5. **Consider transaction costs** - Include realistic fees and slippage
6. **Multiple metrics** - Don't rely on just one metric (Sharpe + drawdown + win rate)
