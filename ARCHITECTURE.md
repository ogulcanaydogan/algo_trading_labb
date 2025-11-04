# Algo Trading Lab – Architecture Documentation

## System Overview

A fully automated, AI-driven, multi-asset trading research and execution platform that combines technical analysis, machine learning predictions, and macro/news sentiment to generate trading signals and optimize strategies continuously.

### Key Capabilities

- **Multi-Asset Support**: Cryptocurrencies (via ccxt/Binance), equities, commodities, forex (via yfinance)
- **Automated Strategy Optimization**: Random-search optimizer with composite objectives (Sharpe, PnL, win rate, drawdown penalty)
- **AI-Enhanced Decision Making**: ML-based prediction layer blending technical features and macro sentiment
- **Macro/News Integration**: RSS feed ingestion with VADER sentiment → structured macro events → trading bias
- **Hot-Reload Configuration**: Per-asset strategy configs updated by optimizer and consumed by bot without restart
- **RESTful API**: Status, signals, equity curve, strategy overview, AI predictions, macro insights, Q&A
- **Backtesting Engine**: Trade lifecycle simulation with stop-loss/take-profit, position sizing, comprehensive metrics
- **Live/Dry-Run Trading**: Paper mode, testnet, or live execution with risk management

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
│  • CLI (run_backtest.py, run_live_trading.py)                      │
│  • Dashboard (http://localhost:8000/dashboard)                      │
│  • REST API (http://localhost:8000/docs)                            │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION SERVICES                            │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐                   │
│  │    Bot     │  │    API     │  │  Optimizer  │                   │
│  │  (bot.py)  │  │  (api.py)  │  │   Service   │                   │
│  └────────────┘  └────────────┘  └─────────────┘                   │
│         │              │                  │                          │
│         └──────────────┴──────────────────┘                          │
│                        │                                             │
└────────────────────────┼─────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       CORE ENGINES                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Strategy   │  │   AI Layer   │  │    Macro     │             │
│  │  (strategy)  │  │    (ai)      │  │   Sentiment  │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Backtester  │  │  Optimizer   │  │   Trading    │             │
│  │(backtesting) │  │ (optimizer)  │  │  Manager     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Market     │  │     State    │  │   Config     │             │
│  │     Data     │  │     Store    │  │   Loader     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│         │                 │                  │                       │
└─────────┼─────────────────┼──────────────────┼───────────────────────┘
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SYSTEMS                                  │
│  • Binance/CCXT (crypto)                                            │
│  • yfinance (equities/commodities/forex)                            │
│  • RSS feeds (news/macro)                                           │
│  • Filesystem (configs, state, signals, equity)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Strategy Optimization Flow

```
run_portfolio_optimize.py
    │
    ├─> Load portfolio.json (BTC/USDT, NVDA, GC=F, ...)
    │
    ├─> For each asset:
    │   ├─> Fetch OHLCV (YFinance/CCXT/Paper)
    │   ├─> Run random_search_optimize (bot/optimizer.py)
    │   │   ├─> Sample random parameter sets
    │   │   ├─> Run backtest for each trial
    │   │   ├─> Score: Sharpe, PnL%, win rate, MDD penalty
    │   │   └─> Return best parameters
    │   ├─> Write data/portfolio/<asset>/strategy_config.json
    │   └─> Append to portfolio_recommendations.json
    │
    └─> Save aggregate recommendations
```

### 2. Trading Bot Flow

```
bot/bot.py (main loop)
    │
    ├─> Load/reload strategy config (hot-reload every N loops)
    │   ├─> Try data/portfolio/<symbol>/strategy_config.json
    │   └─> Fallback to data/strategy_config.json
    │
    ├─> Fetch latest OHLCV
    │   ├─> YFinanceMarketDataClient (equities/commodities)
    │   ├─> ExchangeClient (crypto, testnet/live)
    │   └─> PaperExchangeClient (synthetic)
    │
    ├─> Compute indicators (EMA fast/slow, RSI)
    │
    ├─> Generate base signal (strategy.py)
    │   ├─> EMA crossover logic
    │   ├─> RSI overbought/oversold
    │   └─> Fallback RSI extreme signals
    │
    ├─> Enrich with AI prediction (ai.py)
    │   ├─> Extract features (EMA gap, momentum, RSI distance, volatility)
    │   ├─> Compute score distributions (long/short/flat)
    │   ├─> Blend with macro bias
    │   └─> Return recommended action + confidence
    │
    ├─> Assess macro sentiment (macro.py)
    │   ├─> Load events from MACRO_EVENTS_PATH (refresh every 300s)
    │   ├─> Filter relevant events for symbol
    │   ├─> Compute weighted bias score
    │   └─> Extract drivers, rate outlook, political risk
    │
    ├─> Make final decision (combine base signal + AI + macro)
    │
    ├─> Execute trade (if signal and position logic allows)
    │   ├─> Calculate position size (risk_per_trade_pct)
    │   ├─> Set stop-loss and take-profit
    │   ├─> Place order (dry-run / testnet / live)
    │   └─> Update state
    │
    ├─> Record state, signals, equity to JSON (state.py)
    │
    └─> Sleep and repeat
```

### 3. Macro/News Flow

```
tools/ingest_news_to_macro_events.py
    │
    ├─> Load RSS feed URLs from feeds.yml
    │
    ├─> For each feed:
    │   ├─> Parse entries (feedparser)
    │   ├─> Extract title, published timestamp, source
    │   ├─> Run VADER sentiment analysis
    │   ├─> Classify category (macro/politics/crypto/general)
    │   ├─> Infer impact (high/medium/low) from sentiment score
    │   └─> Map to affected assets
    │
    ├─> Write data/macro_events.news.json
    │
    └─> MacroSentimentEngine.refresh_if_needed()
        │
        ├─> Load baseline events (hardcoded defaults)
        ├─> Merge with events from MACRO_EVENTS_PATH
        ├─> For target symbol:
        │   ├─> Filter relevant events
        │   ├─> Compute weighted bias score
        │   └─> Extract drivers and outlooks
        └─> Return MacroInsight
```

---

## Component Details

### bot/strategy.py

**Purpose**: Core technical analysis and signal generation.

**Key Functions**:
- `compute_indicators(ohlcv, config)`: Adds EMA fast/slow, RSI columns
- `generate_signal(enriched_df, config)`: Produces LONG/SHORT/FLAT decision with confidence
- `calculate_position_size(balance, risk_pct, price, stop_loss_pct)`: Risk-based sizing

**Logic**:
1. LONG: EMA fast > slow AND RSI < overbought
2. SHORT: EMA fast < slow AND RSI > oversold
3. Fallback LONG: RSI < oversold
4. Fallback SHORT: RSI > overbought

**Config**: StrategyConfig (symbol, timeframe, ema_fast, ema_slow, rsi_period, rsi_overbought, rsi_oversold, risk_per_trade_pct, stop_loss_pct, take_profit_pct)

### bot/ai.py

**Purpose**: ML-inspired prediction layer blending technical features and macro.

**Key Functions**:
- `predict(features, macro)`: Returns PredictionSnapshot (action, confidence, probabilities, expected move)
- `_compute_feature_scores()`: Score distributions based on EMA gap, momentum, RSI, volatility
- `_blend_with_macro()`: Adjusts scores with macro bias and confidence

**Features**:
- ema_gap_pct: (fast - slow) / slow
- momentum_pct: (close - close_5_bars_ago) / close_5_bars_ago
- rsi_distance_from_mid: RSI - 50
- volatility_pct: rolling_std(close, 20) / close

**Scoring**: Exponential decay and thresholds applied to features; macro bias amplified with multiplier.

### bot/macro.py

**Purpose**: Fuse macro and geopolitical events into a trading bias.

**Key Classes**:
- `MacroEvent`: Structured event (title, category, sentiment, impact, bias, assets, interest_rate_expectation)
- `MacroInsight`: Aggregated output (bias_score, confidence, summary, drivers, outlooks, events)
- `MacroSentimentEngine`: Load/refresh events, assess symbol-specific bias

**Weighting**:
- Impact: low=0.8, medium=1.0, high=1.5, critical=2.0
- Sentiment: bullish=+0.6, bearish=-0.6, hawkish=-0.5, dovish=+0.5, neutral=0.0

**Refresh**: Reloads events from disk every MACRO_REFRESH_SECONDS (default 300).

### bot/optimizer.py

**Purpose**: Random-search parameter optimization over backtests.

**Key Functions**:
- `random_search_optimize(ohlcv, base_config, n_trials, ...)`: Samples random parameter sets, runs backtests, scores each, returns best
- `_objective_from_result(result, objective, mdd_weight)`: Composite scoring (sharpe, pnl, win_rate) with MDD penalty
- `_sample_params()`: Random draws from predefined ranges

**Parameter Ranges**:
- ema_fast: 8–30
- ema_slow: 30–100
- rsi_period: 7–30
- rsi_overbought: 65–85
- rsi_oversold: 15–35
- risk_per_trade_pct: 0.3–2.0
- stop_loss_pct: 0.002–0.03
- take_profit_pct: 0.004–0.06

**Objectives**: sharpe (default), pnl, composite (weighted blend + MDD penalty).

**Constraints**: min_trades (default 5) to filter out trivial parameter sets.

### bot/backtesting.py

**Purpose**: Simulate trades on historical OHLCV with realistic order fills and risk management.

**Key Classes**:
- `Trade`: Single trade record (direction, entry/exit price/timestamp, pnl, pnl_pct, exit_reason)
- `BacktestResult`: Summary (total_pnl, total_pnl_pct, trades, win_rate, avg_win, avg_loss, profit_factor, max_drawdown, sharpe_ratio, equity_curve)
- `Backtester`: Engine for running strategy over historical data

**Flow**:
1. Iterate OHLCV bars
2. Compute indicators
3. Generate signal
4. Open position if signal and no existing position
5. Check stop-loss / take-profit on every bar
6. Close position if hit or end of data
7. Track equity curve
8. Calculate metrics

### bot/market_data.py

**Purpose**: Unified interface for fetching OHLCV from multiple sources.

**Key Classes**:
- `YFinanceMarketDataClient`: Equities, commodities, forex, indices via yfinance
  - `fetch_ohlcv(symbol, timeframe, limit)`: Maps timeframe to yfinance interval, selects period, downloads, normalizes
  - `_normalize_frame()`: Flattens MultiIndex columns, renames to open/high/low/close/volume, ensures DatetimeIndex
  - `_select_period()`: Heuristic for intraday coverage (6.5 trading hours/day + 25% buffer)
- `sanitize_symbol_for_fs(symbol)`: Convert symbols to filesystem-safe names

### bot/exchange.py

**Purpose**: Real exchange connectivity and paper mode for testing.

**Key Classes**:
- `ExchangeClient`: Wrapper around ccxt for live/testnet trading
  - `fetch_ohlcv(symbol, timeframe, limit)`: Fetch from exchange
  - `fetch_balance()`, `create_market_order()`, `fetch_order()`, `cancel_order()`: Order management
- `PaperExchangeClient`: Synthetic data generator for backtesting without external API

### bot/state.py

**Purpose**: Persistent JSON-backed state store for bot and API.

**Key Classes**:
- `BotState`: Current position, balance, PnL, last signal, indicators, AI/macro fields
- `SignalEvent`: Historical signal record
- `EquityPoint`: Time-series equity value
- `StateStore`: Thread-safe read/write to state.json, signals.json, equity.json

**Methods**:
- `update_state(**kwargs)`: Update fields and flush to disk
- `record_signal(signal)`, `record_equity(point)`: Append and persist
- `get_state_dict()`, `get_signals(limit)`, `get_equity_curve()`: Read for API

### api/api.py

**Purpose**: RESTful API for monitoring and querying.

**Endpoints**:
- `GET /status`: Current bot state (position, balance, indicators, AI/macro)
- `GET /signals?limit=N`: Recent signals
- `GET /equity`: Equity curve
- `GET /strategy?symbol=X`: Strategy config (per-asset or global)
- `GET /portfolio/strategies`: List all per-asset strategies
- `GET /ai/prediction`: Latest AI recommendation
- `POST /ai/question`: Natural language Q&A
- `GET /macro/insights`: Macro bias, drivers, events
- `GET /dashboard`: HTML dashboard

**Config Loading**:
1. Try data/portfolio/<safe_symbol>/strategy_config.json
2. Fallback to data/strategy_config.json
3. Merge overrides with base StrategyConfig

### bot/portfolio.py

**Purpose**: Multi-asset portfolio runner and allocation.

**Key Classes**:
- `PortfolioAsset`: Asset definition (symbol, asset_type, allocation, timeframe, lookback)
- `PortfolioConfig`: Portfolio-wide settings (assets list, default timeframe/lookback, data_dir, macro_events_path)
- `PortfolioRunner`: Orchestrates multiple bot instances (one per asset) with shared macro engine

**Allocation Strategies**:
- Equal weight (default)
- Risk parity (via allocation field)
- Sharpe-weighted (from optimizer recommendations)

### bot/config_loader.py

**Purpose**: Load and merge strategy config overrides from JSON.

**Key Functions**:
- `load_overrides(path)`: Read JSON config file if exists
- `merge_config(base, overrides)`: Create new StrategyConfig with overridden fields

---

## Configuration Reference

### Environment Variables (.env)

```properties
# Runtime mode
PAPER_MODE=true                         # Use synthetic data if true
BINANCE_TESTNET_ENABLED=false           # Use Binance testnet if true

# Market selection
SYMBOL=BTC/USDT                         # Primary trading symbol
TIMEFRAME=1h                            # OHLCV timeframe
LOOKBACK=250                            # Bars to fetch per cycle

# Risk & position
STARTING_BALANCE=10000                  # Initial balance (paper/backtest)
RISK_PER_TRADE_PCT=0.5                  # % of balance to risk per trade
STOP_LOSS_PCT=0.004                     # Stop-loss distance (0.4%)
TAKE_PROFIT_PCT=0.008                   # Take-profit distance (0.8%)

# Auto optimizer settings
AUTO_OPTIMIZE_INTERVAL_MINUTES=180      # How often to re-optimize
AUTO_OPTIMIZE_TRIALS=60                 # Random trials per optimization
AUTO_OPTIMIZE_OBJECTIVE=sharpe          # sharpe|pnl|composite
AUTO_OPTIMIZE_MDD_WEIGHT=0.5            # Drawdown penalty weight
AUTO_OPTIMIZE_MIN_TRADES=5              # Min trades to accept trial
OPTIMIZE_LOOKBACK=2000                  # Bars for optimization backtest

# Macro/news events
MACRO_EVENTS_PATH=data/macro_events.news.json  # Path to macro events JSON
MACRO_REFRESH_SECONDS=300               # How often to reload events

# Data directory
DATA_DIR=./data                         # Root for state/config files

# API keys
BINANCE_TESTNET_API_KEY=                # Testnet key
BINANCE_TESTNET_API_SECRET=             # Testnet secret
EXCHANGE_API_KEY=                       # Live key (use with caution)
EXCHANGE_API_SECRET=                    # Live secret
```

### portfolio.json

```json
{
  "assets": [
    {
      "symbol": "BTC/USDT",
      "asset_type": "crypto",
      "allocation": 0.4,
      "timeframe": "15m",
      "lookback": 500
    },
    {
      "symbol": "NVDA",
      "asset_type": "equity",
      "allocation": 0.3,
      "timeframe": "1h",
      "lookback": 500
    },
    {
      "symbol": "GC=F",
      "asset_type": "commodity",
      "allocation": 0.3,
      "timeframe": "1h",
      "lookback": 500
    }
  ],
  "default_timeframe": "1h",
  "default_lookback": 500,
  "data_dir": "data/portfolio"
}
```

### strategy_config.json (per-asset or global)

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "15m",
  "ema_fast": 18,
  "ema_slow": 56,
  "rsi_period": 10,
  "rsi_overbought": 79.43,
  "rsi_oversold": 33.63,
  "risk_per_trade_pct": 1.11,
  "stop_loss_pct": 0.00679,
  "take_profit_pct": 0.04858
}
```

### macro_events.json

```json
[
  {
    "title": "Trump announces tariff review",
    "category": "politics",
    "sentiment": "bearish",
    "impact": "high",
    "actor": "Donald Trump",
    "summary": "Renewed tariff threats raise volatility.",
    "assets": {
      "BTC/USDT": -0.2,
      "ETH/USDT": -0.15,
      "*": -0.05
    }
  },
  {
    "title": "FOMC statement",
    "category": "central_bank",
    "sentiment": "dovish",
    "impact": "medium",
    "interest_rate_expectation": "Fed signals cautious path.",
    "summary": "Disinflation progress noted."
  }
]
```

---

## Deployment

### Docker Compose Services

```yaml
services:
  bot:
    build: .
    command: python -m bot.bot
    env_file: .env
    volumes:
      - ./:/app
      - state_data:/app/data

  api:
    build: .
    command: uvicorn api.api:app --host 0.0.0.0 --port 8000
    env_file: .env
    volumes:
      - ./:/app
      - state_data:/app/data
    ports:
      - "8000:8000"

  optimizer:
    build: .
    command: python run_auto_optimize.py
    env_file: .env
    volumes:
      - ./:/app
      - state_data:/app/data

volumes:
  state_data:
```

### Starting Services

```bash
# Bring all services up in background
docker compose up -d

# Check logs
docker compose logs -f bot
docker compose logs -f api
docker compose logs -f optimizer

# Stop services
docker compose down
```

### Running Portfolio Optimizer (One-Shot)

```bash
# Inside optimizer container
docker compose exec -T optimizer python run_portfolio_optimize.py

# Output: data/portfolio/<asset>/strategy_config.json
#         data/portfolio/portfolio_recommendations.json
```

### News Ingestion (Periodic)

```bash
# Inside optimizer container
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml \
  --out data/macro_events.news.json \
  --symbols "BTC/USDT,ETH/USDT,NVDA,GC=F"

# Set MACRO_EVENTS_PATH in .env and restart services
docker compose restart bot api
```

---

## Operational Procedures

### 1. Initial Setup

```bash
# Clone repository
git clone <repo_url>
cd algo_trading_lab

# Copy sample configs
cp data/portfolio.sample.json data/portfolio.json
# Edit portfolio.json to add/remove assets

# Configure .env
# - Set PAPER_MODE=true for safe testing
# - Add API keys only if using testnet/live
# - Set MACRO_EVENTS_PATH=data/macro_events.news.json if using news

# Build and start services
docker compose up -d

# Check services are running
docker compose ps
```

### 2. Backtesting a Strategy

```bash
# Interactive mode
python run_backtest.py

# Follow prompts:
# - Symbol: BTC/USDT
# - Timeframe: 1h
# - Lookback: 1000
# - Initial balance: 10000
# - Parameters: use defaults or optimized values
# - Data source: 1 (Binance Testnet) or 2 (Paper)
# - Save results: y/n
```

### 3. Running Live/Dry-Run Trading

```bash
# Interactive mode
python run_live_trading.py

# Choose mode:
# - 1: DRY RUN (logs only, no orders)
# - 2: TESTNET (Binance testnet, requires keys)
# - 3: LIVE (REAL MONEY, extreme caution)

# Configure:
# - Symbol, timeframe, loop interval
# - Risk, stop-loss, take-profit percentages

# Monitor:
# - Watch terminal for iteration logs
# - Check API: curl http://localhost:8000/status
# - Ctrl+C to stop (optionally close open position)
```

### 4. Monitoring via API

```bash
# Bot status
curl -sS http://localhost:8000/status | jq

# Recent signals
curl -sS http://localhost:8000/signals?limit=10 | jq

# Equity curve
curl -sS http://localhost:8000/equity | jq

# Strategy config for a symbol
curl -sS -G --data-urlencode "symbol=NVDA" http://localhost:8000/strategy | jq

# All portfolio strategies
curl -sS http://localhost:8000/portfolio/strategies | jq

# AI prediction
curl -sS http://localhost:8000/ai/prediction | jq

# Macro insights
curl -sS http://localhost:8000/macro/insights | jq

# Ask a question
curl -sS -X POST http://localhost:8000/ai/question \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the current macro outlook for BTC?"}' | jq

# Dashboard (browser)
open http://localhost:8000/dashboard
```

### 5. Updating Strategy Parameters

**Manual override:**
```bash
# Edit data/strategy_config.json or data/portfolio/<asset>/strategy_config.json
# Bot will hot-reload on next cycle (no restart needed)
```

**Optimizer-driven:**
```bash
# Single-asset (BTC/USDT by default, continuous every 180m)
# Already running via optimizer service

# Portfolio-wide (one-shot)
docker compose exec -T optimizer python run_portfolio_optimize.py

# Check generated configs
docker compose exec -T optimizer ls -l data/portfolio/*/strategy_config.json
docker compose exec -T optimizer cat data/portfolio/portfolio_recommendations.json
```

### 6. Refreshing Macro/News Events

**Manual:**
```bash
# Edit data/macro_events.json or data/macro_events.news.json
# Engine reloads automatically every MACRO_REFRESH_SECONDS (300s default)
```

**Automated ingestion:**
```bash
# Run news ingester
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml \
  --out data/macro_events.news.json \
  --symbols "BTC/USDT,ETH/USDT,NVDA,GC=F"

# Schedule via cron (host):
# 0 */4 * * * cd /path/to/algo_trading_lab && docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py --feeds feeds.news.yml --out data/macro_events.news.json --symbols "BTC/USDT,ETH/USDT,NVDA,GC=F"
```

### 7. Adding a New Asset

```bash
# Edit data/portfolio.json
# Add new entry:
{
  "symbol": "AAPL",
  "asset_type": "equity",
  "allocation": 0.2,
  "timeframe": "1h",
  "lookback": 500
}

# Run portfolio optimizer to generate config
docker compose exec -T optimizer python run_portfolio_optimize.py

# Verify config created
docker compose exec -T optimizer ls -l data/portfolio/AAPL/strategy_config.json

# Check via API
curl -sS -G --data-urlencode "symbol=AAPL" http://localhost:8000/strategy | jq
```

### 8. Troubleshooting

**Bot not trading:**
- Check logs: `docker compose logs -f bot`
- Verify PAPER_MODE or API keys in .env
- Check signal confidence threshold (default >0.4)
- Inspect state: `curl http://localhost:8000/status | jq`

**Optimizer producing no valid results:**
- Increase AUTO_OPTIMIZE_TRIALS (e.g., 200)
- Reduce AUTO_OPTIMIZE_MIN_TRADES (e.g., 3)
- Check data availability (lookback vs. available history)
- Review parameter ranges in bot/optimizer.py

**API returning 404:**
- State file missing: run bot once to create data/state.json
- Check DATA_DIR in .env matches docker volume mount

**Macro bias not updating:**
- Verify MACRO_EVENTS_PATH points to valid JSON
- Check MACRO_REFRESH_SECONDS (default 300s)
- Validate JSON format (array of MacroEvent objects)

**yfinance data issues:**
- Increase lookback if intraday data insufficient
- Verify symbol format (e.g., GC=F for gold futures)
- Check _select_period logic in bot/market_data.py

---

## Performance & Optimization

### Backtesting Performance

- **500 bars, 60 trials**: ~30–60 seconds (varies by asset and trial complexity)
- **2000 bars, 200 trials**: ~3–10 minutes
- **Parallelization**: Not implemented; trials run serially
- **Optimization**: Use smaller lookback for faster iteration; increase for robustness

### Live Trading Latency

- **Bot loop interval**: 60s default (configurable)
- **Market data fetch**: 100–500ms (exchange/yfinance dependent)
- **Indicator computation**: <10ms (pandas vectorized)
- **AI prediction**: <5ms (scoring logic)
- **Macro assessment**: <5ms (cached events, filtered on-demand)
- **State write**: <10ms (JSON flush to disk)

**Total cycle time**: ~1–2 seconds (plus sleep interval)

### API Response Times

- **GET /status**: ~5–10ms
- **GET /signals**: ~10–20ms (limited by result set size)
- **GET /equity**: ~10–50ms (scales with curve length)
- **GET /strategy**: ~5–10ms
- **GET /ai/prediction**: ~5–10ms
- **POST /ai/question**: ~50–200ms (LLM-style prompt assembly)
- **GET /macro/insights**: ~10–20ms

### Scaling Considerations

- **Multi-asset**: PortfolioRunner spawns threads (one per asset); scales linearly
- **High-frequency**: Not designed for sub-minute intervals; optimize loop/sleep logic
- **Database**: Currently JSON-backed; migrate to PostgreSQL/TimescaleDB for production
- **Caching**: State/signals cached in memory (StateStore); refresh on load()
- **Horizontal scaling**: Stateless API can scale behind load balancer; bot is stateful (one per symbol)

---

## Testing

### Unit Tests

Not currently implemented. Suggested structure:

```
tests/
  test_strategy.py       # compute_indicators, generate_signal
  test_ai.py             # feature extraction, scoring
  test_macro.py          # event parsing, bias calculation
  test_optimizer.py      # parameter sampling, objective scoring
  test_backtesting.py    # trade lifecycle, metrics
  test_market_data.py    # yfinance normalization, period selection
```

### Integration Tests

```bash
# Backtest with known data → validate metrics
python run_backtest.py < test_input.txt

# Dry-run bot for N iterations → check state consistency
# (requires manual setup or test fixture)

# API smoke tests
curl -f http://localhost:8000/status
curl -f http://localhost:8000/signals
curl -f http://localhost:8000/strategy
```

### End-to-End Test

```bash
# 1. Start services
docker compose up -d

# 2. Run portfolio optimizer
docker compose exec -T optimizer python run_portfolio_optimize.py

# 3. Verify configs generated
docker compose exec -T optimizer ls data/portfolio/*/strategy_config.json

# 4. Check API reflects configs
curl -sS http://localhost:8000/portfolio/strategies | jq -r '.[].symbol'

# 5. Run news ingestion
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml --out data/macro_events.news.json \
  --symbols "BTC/USDT,NVDA,GC=F"

# 6. Restart services to load macro events
docker compose restart bot api

# 7. Verify macro insights
curl -sS http://localhost:8000/macro/insights | jq '.summary'

# 8. Run dry-run bot for 3 iterations
python run_live_trading.py
# Choose: 1 (DRY RUN), BTC/USDT, 1h, interval=60
# Ctrl+C after 3 iterations

# 9. Check state updated
curl -sS http://localhost:8000/status | jq '.position,.balance'

# Success if all steps complete without errors
```

---

## Security & Compliance

### API Key Management

- **Never commit** .env or keys to version control
- Use environment variables or secrets manager (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly
- Use testnet keys for development; separate live keys per environment

### Risk Management

- **Position sizing**: Capped by risk_per_trade_pct (default 0.5%)
- **Stop-loss**: Mandatory on every position
- **Take-profit**: Configured per strategy
- **Balance checks**: Prevent over-allocation
- **Dry-run mode**: Test strategies before live execution

### Data Privacy

- State/signals/equity files contain PII (balances, positions)
- Restrict file permissions: `chmod 600 data/*.json`
- Use encrypted volumes in production
- Anonymize or redact before sharing logs/reports

### Regulatory Considerations

- **Not financial advice**: This is a research/educational platform
- **Compliance**: Check local regulations for automated trading
- **Reporting**: Maintain audit trail (signals, trades, state changes)
- **Broker rules**: Respect rate limits, order types, margin requirements

---

## Roadmap & Future Enhancements

### Phase 1: Core Stability (Current)
- ✅ Multi-asset support (crypto, equities, commodities)
- ✅ Random-search optimizer with composite objectives
- ✅ AI prediction layer with macro blending
- ✅ Macro/news ingestion pipeline
- ✅ RESTful API and dashboard
- ✅ Hot-reload configuration
- ✅ Docker Compose orchestration

### Phase 2: Advanced Optimization
- [ ] Bayesian optimization (Optuna/Hyperopt)
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Walk-forward analysis (train/test splits)
- [ ] Regime detection (bull/bear/sideways)
- [ ] Parameter stability analysis (sensitivity testing)

### Phase 3: ML Enhancements
- [ ] Replace scoring heuristics with trained models (XGBoost, LightGBM)
- [ ] Feature engineering pipeline (technical + alternative data)
- [ ] Online learning (incremental model updates)
- [ ] Ensemble predictions (combine multiple models)
- [ ] Explainability (SHAP values, feature importance)

### Phase 4: Portfolio Management
- [ ] Multi-asset position allocation (equal weight, risk parity, Sharpe-weighted)
- [ ] Rebalancing logic (threshold, periodic)
- [ ] Correlation-aware allocation (Markowitz, Black-Litterman)
- [ ] Risk budgeting (volatility targeting)
- [ ] Drawdown control (circuit breakers)

### Phase 5: Production Readiness
- [ ] PostgreSQL/TimescaleDB backend (replace JSON)
- [ ] Redis caching (state, market data)
- [ ] Prometheus metrics + Grafana dashboards
- [ ] Structured logging (ELK stack)
- [ ] Health checks and auto-restart
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Unit/integration test coverage >80%

### Phase 6: Advanced Features
- [ ] WebSocket real-time data feeds
- [ ] Multi-exchange routing (best execution)
- [ ] Order types (limit, stop-limit, trailing stop)
- [ ] Margin/leverage support
- [ ] Synthetic instruments (spreads, baskets)
- [ ] Alerts and notifications (Slack, email, SMS)

---

## Contributing

### Development Workflow

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test locally
4. Run linters: `ruff check .` (if configured)
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Style

- **Python**: PEP 8, type hints where feasible
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google style for public functions
- **Imports**: Group standard lib, third-party, local
- **Line length**: 100 characters (flexible)

### Testing Guidelines

- Add unit tests for new modules
- Update integration tests if API changes
- Run end-to-end test before submitting PR
- Document breaking changes in PR description

---

## License

See LICENSE file for details.

---

## Support & Contact

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check QUICKSTART.md and README.md for basics

---

## Acknowledgments

- **ccxt**: Unified exchange API
- **yfinance**: Yahoo Finance market data
- **ta**: Technical analysis library
- **FastAPI**: Modern web framework
- **pandas/numpy**: Data manipulation
- **feedparser/vaderSentiment**: News ingestion and sentiment

---

**Last Updated**: 2025-11-01  
**Version**: 0.1.0  
**Maintainer**: algo_trading_lab team
