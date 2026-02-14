# Real-Time Streaming Documentation

## Overview

The streaming module (`bot/data/stream_manager.py`) provides real-time market data via WebSocket connections with event-driven bar aggregation.

## Features

- **Multi-Exchange Support**
  - Binance WebSocket (crypto: BTC, XRP, ETH, SOL, etc.)
  - Alpaca WebSocket (stocks: TSLA, AAPL, GOOGL, etc.)

- **Bar Aggregation**
  - Tick-to-OHLCV bar conversion
  - Multiple timeframes (1m, 5m, 15m)
  - VWAP calculation
  - Buy/sell volume tracking

- **Event-Driven Architecture**
  - `TICK` events for every trade
  - `BAR_COMPLETE` events when bars close
  - `PREDICTION` and `SIGNAL` hooks for ML integration

- **Reliability**
  - Automatic reconnection with exponential backoff
  - Health monitoring
  - Connection status tracking

## Quick Start

```python
import asyncio
from bot.data import (
    StreamManager,
    StreamManagerConfig,
    create_default_trading_stream,
)

# Simple usage
async def main():
    manager = create_default_trading_stream(
        crypto_symbols=["BTC/USDT", "XRP/USDT"],
        stock_symbols=["TSLA"],  # Requires ALPACA_API_KEY
    )
    
    # Register callbacks
    manager.on_tick(lambda e: print(f"Tick: {e.data.symbol} @ {e.data.price}"))
    manager.on_bar_complete(lambda b: print(f"Bar: {b.symbol} OHLC"))
    
    await manager.start()
    await asyncio.sleep(60)  # Run for 60 seconds
    await manager.stop()

asyncio.run(main())
```

## Configuration

```python
from bot.data import StreamManagerConfig

config = StreamManagerConfig(
    # Bar aggregation
    default_timeframe="1m",
    supported_timeframes=["1m", "5m", "15m"],
    
    # Connection
    reconnect_delay=1.0,
    max_reconnect_delay=60.0,
    reconnect_attempts=10,
    
    # Buffers
    tick_buffer_size=10000,
    bar_history_size=500,
    
    # Health
    health_check_interval=10.0,
    stale_threshold_seconds=30.0,
)
```

## Integration with Trading Engine

```python
from bot.data import StreamingTradingEngine, StreamManager

async def run_trading():
    manager = StreamManager()
    manager.add_crypto_symbols(["BTC/USDT"])
    
    engine = StreamingTradingEngine(
        stream_manager=manager,
        predictor=my_ml_predictor,  # Optional ML model
        executor=my_order_executor,  # Optional execution
        signal_threshold=0.6,
    )
    
    await engine.start()
    
    # Engine automatically:
    # 1. Receives ticks → aggregates to bars
    # 2. On bar close → calculates features
    # 3. Generates predictions from features
    # 4. Converts predictions to signals
    # 5. Executes signals via executor
```

## Accessing Data

```python
# Get bar history as DataFrame (for features)
df = manager.get_bar_dataframe("BTC/USDT", timeframe="1m", limit=100)

# Get current building bar
current = manager.get_current_bar("BTC/USDT", timeframe="1m")

# Get health metrics
health = manager.get_health()
metrics = manager.get_metrics()
```

## Event Types

| Event | Trigger | Data |
|-------|---------|------|
| `TICK` | Every trade | `Tick` object (symbol, price, volume, side) |
| `BAR_COMPLETE` | Bar closes | `Bar` object (OHLCV, VWAP, etc.) |
| `PREDICTION` | ML prediction | Custom prediction data |
| `SIGNAL` | Trading signal | Signal direction, confidence |
| `RECONNECT` | Stream reconnects | Reconnect count |
| `HEALTH` | Health check | Stream health status |

## Environment Variables

For Alpaca stock streaming:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

## Latency Characteristics

| Source | Expected Latency |
|--------|-----------------|
| Binance trade stream | 10-50ms |
| Binance ticker stream | 100-500ms |
| Alpaca IEX stream | 50-200ms |
| Bar aggregation | <1ms |
| Full pipeline (tick→signal) | 50-100ms |

## Architecture Flow

```
Stream → Tick → BarAggregator → Bar Complete Event
                                      ↓
                              Feature Calculation
                                      ↓
                               ML Prediction
                                      ↓
                              Signal Generation
                                      ↓
                              Order Execution
```

## Files

- `bot/data/stream_manager.py` - Main streaming module
- `bot/data/websocket_stream.py` - Legacy WebSocket (still supported)
- `scripts/demo/test_streaming.py` - Test/demo script
