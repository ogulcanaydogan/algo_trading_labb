w# Algo Trading Lab – Quick Reference

**Fully automated, AI-driven, multi-asset trading platform**

## 🎯 What's Inside

- Multi-asset support (crypto, stocks, commodities, forex)
- Automated strategy optimization (random search, composite objectives)
- AI prediction layer (technical + macro sentiment fusion)
- Macro/news integration (RSS → VADER → trading bias)
- Hot-reload configs (optimizer updates, bot consumes live)
- REST API + Dashboard (monitoring, insights, Q&A)
- Backtesting + Live trading (paper, testnet, live)

## ⚡ Quick Start

```bash
# 1. Setup
git clone <repo> && cd algo_trading_lab
cp .env.example .env
# Edit .env: PAPER_MODE=true for safety

# 2. Start services
docker compose up -d

# 3. Dashboard
open http://localhost:8000/dashboard

# 4. Backtest
python run_backtest.py

# 5. Live (dry-run)
python run_live_trading.py  # Choose mode 1

# 6. Optimize portfolio
docker compose exec -T optimizer python run_portfolio_optimize.py
```

## 🔍 Key Commands

**Monitor:**
```bash
# Status
curl http://localhost:8000/status | jq

# Signals
curl http://localhost:8000/signals | jq

# Portfolio strategies
curl http://localhost:8000/portfolio/strategies | jq

# Macro insights
curl http://localhost:8000/macro/insights | jq

# Ask AI
curl -X POST http://localhost:8000/ai/question \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the current outlook?"}' | jq
```

**Optimize:**
```bash
# Single asset (BTC/USDT, auto-runs every 180m)
# Already running via optimizer service

# Portfolio (one-shot)
docker compose exec -T optimizer python run_portfolio_optimize.py

# News ingestion
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml --out data/macro_events.news.json \
  --symbols "BTC/USDT,NVDA,GC=F"
```

**Logs:**
```bash
docker compose logs -f bot
docker compose logs -f api
docker compose logs -f optimizer
```

## 📁 Directory Structure

```
algo_trading_lab/
├── bot/                # Core engines
│   ├── ai.py          # ML prediction layer
│   ├── bot.py         # Main trading loop
│   ├── strategy.py    # EMA/RSI signals
│   ├── optimizer.py   # Random-search optimizer
│   ├── backtesting.py # Backtest engine
│   ├── macro.py       # Macro sentiment engine
│   ├── market_data.py # yfinance/ccxt/paper
│   └── portfolio.py   # Multi-asset runner
├── api/               # REST API
│   ├── api.py         # FastAPI app
│   └── schemas.py     # Response models
├── tools/             # Utilities
│   └── ingest_news_to_macro_events.py
├── data/              # State/configs
│   ├── portfolio.json
│   ├── portfolio/     # Per-asset configs
│   ├── macro_events.news.json
│   ├── state.json
│   └── strategy_config.json
├── run_backtest.py    # Interactive backtest
├── run_live_trading.py # Interactive live
├── run_portfolio_optimize.py # Portfolio optimizer
├── docker-compose.yml
├── Dockerfile
├── ARCHITECTURE.md    # Full system docs
└── README.md
```

## 🧩 Architecture (High-Level)

```
┌─────────────────────────────────────────────┐
│           USER INTERFACES                   │
│  Dashboard | API | CLI                      │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│         APPLICATION SERVICES                │
│  Bot | API | Optimizer (Docker Compose)     │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│            CORE ENGINES                     │
│  Strategy | AI | Macro | Backtester         │
│  Optimizer | Trading Manager                │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│             DATA LAYER                      │
│  Market Data | State Store | Config Loader  │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│         EXTERNAL SYSTEMS                    │
│  Binance/CCXT | yfinance | RSS | Filesystem│
└─────────────────────────────────────────────┘
```

## 🎨 Dashboard

**Live at**: http://localhost:8000/dashboard  
**Preview mode**: http://localhost:8000/dashboard/preview

**Sections:**
- Status cards (position, balance, PnL)
- Signal stream (recent decisions)
- Equity curve + risk metrics
- AI insights (prediction, probabilities, features)
- Decision playbook (strategy rules)
- Macro pulse (news, bias, drivers)
- Q&A assistant (ask questions)

## 🔐 Configuration

### .env
```properties
PAPER_MODE=true
SYMBOL=BTC/USDT
TIMEFRAME=1h
LOOKBACK=250
STARTING_BALANCE=10000
RISK_PER_TRADE_PCT=0.5
STOP_LOSS_PCT=0.004
TAKE_PROFIT_PCT=0.008
AUTO_OPTIMIZE_TRIALS=60
AUTO_OPTIMIZE_OBJECTIVE=sharpe
MACRO_EVENTS_PATH=data/macro_events.news.json
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
```

### portfolio.json
```json
{
  "assets": [
    {"symbol": "BTC/USDT", "asset_type": "crypto", "allocation": 0.4, "timeframe": "15m"},
    {"symbol": "NVDA", "asset_type": "equity", "allocation": 0.3, "timeframe": "1h"},
    {"symbol": "GC=F", "asset_type": "commodity", "allocation": 0.3, "timeframe": "1h"}
  ],
  "default_timeframe": "1h",
  "default_lookback": 500,
  "data_dir": "data/portfolio"
}
```

## 📈 Workflow Examples

**Backtest → Optimize → Live:**
```bash
# 1. Backtest with default params
python run_backtest.py

# 2. Optimize for better params
docker compose exec -T optimizer python run_portfolio_optimize.py

# 3. Dry-run with optimized params (hot-reload)
python run_live_trading.py  # Mode 1

# 4. Testnet with real orders
python run_live_trading.py  # Mode 2

# 5. Live (only if confident)
python run_live_trading.py  # Mode 3
```

**Add New Asset:**
```bash
# 1. Edit portfolio.json
# Add: {"symbol": "AAPL", "asset_type": "equity", ...}

# 2. Run optimizer
docker compose exec -T optimizer python run_portfolio_optimize.py

# 3. Verify config created
docker compose exec -T optimizer ls data/portfolio/AAPL/strategy_config.json

# 4. Check API
curl -sS -G --data-urlencode "symbol=AAPL" http://localhost:8000/strategy | jq
```

**Refresh Macro Events:**
```bash
# Run news ingester
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml --out data/macro_events.news.json \
  --symbols "BTC/USDT,NVDA,GC=F"

# Engine auto-refreshes every 300s (MACRO_REFRESH_SECONDS)
# Or restart services to force reload:
docker compose restart bot api
```

## 🛠 Troubleshooting

**Bot not trading:**
- Check logs: `docker compose logs -f bot`
- Verify PAPER_MODE or API keys in .env
- Ensure signal confidence > 0.4 (threshold)
- Check state: `curl http://localhost:8000/status | jq`

**Optimizer failing:**
- Increase AUTO_OPTIMIZE_TRIALS (e.g., 200)
- Reduce AUTO_OPTIMIZE_MIN_TRADES (e.g., 3)
- Check data availability (lookback vs. history)

**API 404:**
- State file missing: run bot once to create data/state.json
- Check DATA_DIR in .env matches volume mount

**Macro bias not updating:**
- Verify MACRO_EVENTS_PATH points to valid JSON
- Check MACRO_REFRESH_SECONDS (default 300)
- Validate JSON format (array of objects)

## 📚 Full Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system design, flows, components
- **[QUICKSTART.md](QUICKSTART.md)**: Detailed setup and usage
- **[docs/ui_walkthrough.md](docs/ui_walkthrough.md)**: Dashboard deep-dive

## 🎯 API Endpoints

```
GET  /status              # Bot state
GET  /signals?limit=N     # Recent signals
GET  /equity              # Equity curve
GET  /strategy?symbol=X   # Strategy config
GET  /portfolio/strategies # All strategies
GET  /ai/prediction       # AI recommendation
POST /ai/question         # Q&A
GET  /macro/insights      # Macro bias
GET  /dashboard           # Web UI
GET  /docs                # API docs
```

## 🚀 Performance

**Backtest:**
- 500 bars, 60 trials: ~30–60s
- 2000 bars, 200 trials: ~3–10m

**Live Loop:**
- Cycle time: ~1–2s (plus sleep interval)
- Market data fetch: 100–500ms
- Indicators: <10ms (vectorized)
- AI prediction: <5ms
- State write: <10ms

**API Response:**
- GET endpoints: 5–50ms
- POST /ai/question: 50–200ms

## 📦 Dependencies

```
ccxt>=4.0.0               # Exchange API
fastapi>=0.111.0          # Web framework
uvicorn[standard]>=0.30.0 # ASGI server
pydantic>=2.6.0           # Data validation
numpy>=1.24.0             # Numerical computing
pandas>=2.0.0             # Data manipulation
ta>=0.11.0                # Technical indicators
python-dotenv>=1.0.0      # Environment config
yfinance>=0.2.40          # Yahoo Finance data
PyYAML>=6.0.0             # YAML parsing
vaderSentiment>=3.3.2     # Sentiment analysis
feedparser>=6.0.11        # RSS feed parsing
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature/name`
6. Open Pull Request

## 📄 License

See LICENSE file.

## 🆘 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Docs**: ARCHITECTURE.md, QUICKSTART.md

---

**Version**: 0.1.0 | **Updated**: 2025-11-01 | **Maintainer**: algo_trading_lab team
