# 🚀 Algo Trading Lab – Complete Quick Start Guide

Welcome! This guide will get you from zero to trading in under 15 minutes.

---

## 📋 Prerequisites

- **Docker & Docker Compose** (recommended) OR Python 3.11+
- **Git** for cloning the repository
- **Basic terminal knowledge**
- **Optional**: Binance Testnet account for live testing

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Clone and Configure

```bash
# Clone repository
git clone https://github.com/ogulcanaydogan/algo_trading_lab.git
cd algo_trading_lab

# Copy environment template
cp .env.example .env

# Edit configuration (use your editor)
nano .env
```

### Step 2: Set Safe Defaults

For your first run, use these **safe settings** in `.env`:

```properties
# Safe paper trading mode (no real money)
PAPER_MODE=true
BINANCE_TESTNET_ENABLED=false

# Market & timeframe
SYMBOL=BTC/USDT
TIMEFRAME=1h
LOOKBACK=250

# Risk management
STARTING_BALANCE=10000
RISK_PER_TRADE_PCT=0.5
STOP_LOSS_PCT=0.4
TAKE_PROFIT_PCT=0.8

# Optimizer settings
AUTO_OPTIMIZE_TRIALS=60
AUTO_OPTIMIZE_OBJECTIVE=sharpe

# Macro events (we'll set this up later)
MACRO_EVENTS_PATH=data/macro_events.news.json
MACRO_REFRESH_SECONDS=300
```

Save and close (Ctrl+X, Y, Enter in nano).

### Step 3: Start Services

```bash
# Build and start all services
docker compose up -d --build

# Verify services are running
docker compose ps

# Expected output:
# NAME                         STATUS
# algo_trading_lab-bot-1       running
# algo_trading_lab-api-1       running
# algo_trading_lab-optimizer-1 running
```

### Step 4: Open Dashboard

```bash
# Open in browser (macOS/Linux)
open http://localhost:8000/dashboard

# Windows
start http://localhost:8000/dashboard

# Or navigate manually to:
# http://localhost:8000/dashboard
```

**🎉 Congratulations!** Your trading system is now running.

---

## 📊 Understanding the Dashboard

The dashboard provides a real-time view of your trading system:

### 1. **Status Cards** (Top Row)
- **Current Position**: LONG, SHORT, or FLAT
- **Balance**: Current portfolio value
- **Total PnL**: Profit/Loss since start
- **Last Signal**: Most recent trading decision

### 2. **Signal Stream** (Left Side)
Real-time feed of trading decisions:
```
2025-11-01 12:34:56 | FLAT | Confidence: 52% | RSI: 49.68
Reason: No clear crossover, waiting for signal
```

### 3. **Equity Curve** (Center)
Visual chart showing portfolio value over time:
- Green = Gains
- Red = Losses
- Hover for exact values

### 4. **AI Insights** (Right Top)
Machine learning predictions:
- **Action**: LONG/SHORT/FLAT
- **Confidence**: 0-100%
- **Probabilities**: Breakdown by action type

### 5. **Macro & News Pulse** (Right Middle)
Economic sentiment analysis:
- **Bias Score**: -1.0 (bearish) to +1.0 (bullish)
- **Confidence**: Reliability of analysis
- **Key Drivers**: Major market catalysts
- **Interest Rate Outlook**: Fed policy expectations

### 6. **Decision Playbook** (Bottom)
Strategy rules and current state:
- Technical indicators (EMA, RSI)
- Risk management rules
- Position sizing logic

---

## 🧪 Test the API

### Check Bot Status

```bash
curl -sS http://localhost:8000/status | jq

# Output:
# {
#   "timestamp": "2025-11-01T12:34:56Z",
#   "symbol": "BTC/USDT",
#   "position": "FLAT",
#   "balance": 10000.0,
#   "total_pnl": 0.0,
#   "last_signal": "FLAT"
# }
```

### View Recent Signals

```bash
curl -sS http://localhost:8000/signals?limit=5 | jq

# Output: Array of recent decisions with timestamps, confidence, reasons
```

### Get Strategy Config

```bash
curl -sS http://localhost:8000/strategy | jq

# Output: Current EMA, RSI, risk parameters
```

### Ask AI a Question

```bash
curl -sS -X POST http://localhost:8000/ai/question \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the current market condition?"}' \
  | jq -r '.answer'

# Output: Natural language analysis of current market state
```

### Full API Documentation

Visit: http://localhost:8000/docs

---

## 📈 Run Your First Backtest

### Interactive Mode

```bash
python run_backtest.py
```

Follow the prompts:

```
Symbol (default: BTC/USDT): [Enter]
Timeframe (default: 1h): [Enter]
Number of candles (default: 1000): 500
Starting balance (default: 10000): [Enter]

Strategy Parameters:
  EMA Fast (default: 12): [Enter]
  EMA Slow (default: 26): [Enter]
  RSI Period (default: 14): [Enter]
  Risk per trade % (default: 1.0): [Enter]
  Stop Loss % (default: 2.0): [Enter]
  Take Profit % (default: 4.0): [Enter]

Data Source:
1. Binance Testnet (real data)
2. Paper Exchange (synthetic data)
Choice (1/2): 2
```

### Review Results

```
============================================================
BACKTEST RESULTS
============================================================
Starting Balance: $10,000.00
Ending Balance: $10,450.00
Total P&L: $450.00 (4.50%)

Total Trades: 23
Winners: 15 | Losers: 8
Win Rate: 65.22%
Average Win: $85.30
Average Loss: $42.15
Profit Factor: 2.02
Max Drawdown: $180.00 (1.80%)
Sharpe Ratio: 1.12
============================================================
```

### Good Metrics Guide

- ✅ **Win Rate** > 55%
- ✅ **Profit Factor** > 1.5
- ✅ **Sharpe Ratio** > 1.0
- ✅ **Max Drawdown** < 10%

---

## 🤖 Optimize Strategy Parameters

### Automatic Optimization

The optimizer finds the best strategy parameters by testing many combinations:

```bash
# Run portfolio optimizer
docker compose exec -T optimizer python run_portfolio_optimize.py
```

### What It Does

1. **Loads Assets**: Reads `data/portfolio.json` (BTC/USDT, NVDA, GC=F)
2. **Fetches Data**: Downloads historical OHLCV data
3. **Random Search**: Tests 60 parameter combinations per asset
4. **Evaluates**: Computes Sharpe ratio, PnL%, win rate, drawdown
5. **Saves Best**: Writes optimal config to `data/portfolio/<asset>/strategy_config.json`

### Output Example

```
🔎 Optimizing BTC/USDT (crypto) tf=15m lookback=500
🔄 Running backtest... (500 bars)
Trial 1/60: sharpe=0.31 pnl%=4.2 win%=38.0 mdd%=3.5
Trial 2/60: sharpe=0.42 pnl%=6.8 win%=42.0 mdd%=2.8
...
Trial 60/60: sharpe=0.38 pnl%=5.5 win%=40.0 mdd%=2.9

✅ Best result: sharpe=0.449 pnl%=8.08 win%=40.0 mdd%=2.21
✅ Wrote data/portfolio/BTC_USDT/strategy_config.json

🔎 Optimizing NVDA (equity) tf=1h lookback=500
...
✅ Wrote data/portfolio/NVDA/strategy_config.json | Sharpe=0.439

🔎 Optimizing GC=F (commodity/gold) tf=1h lookback=500
...
✅ Wrote data/portfolio/GC_F/strategy_config.json | Sharpe=0.439

📊 Saved portfolio recommendations -> data/portfolio/portfolio_recommendations.json
```

### Verify Optimization

```bash
# List generated configs
docker compose exec optimizer ls -l data/portfolio/*/strategy_config.json

# View BTC config
docker compose exec optimizer cat data/portfolio/BTC_USDT/strategy_config.json

# Check via API
curl -sS http://localhost:8000/portfolio/strategies | jq
```

### Bot Auto-Reload

The bot automatically reloads configs (no restart needed!):

```bash
# Watch for reload messages
docker compose logs -f bot | grep "Strategy reloaded"

# Output:
# INFO: Strategy reloaded | ema_fast=18 ema_slow=56 rsi_period=10
```

---

## 📰 Enable Macro & News Integration

### Step 1: Create Feed Configuration

```bash
# Create RSS feed list inside container
docker compose exec -T optimizer sh -c 'cat > feeds.news.yml <<EOF
feeds:
  - https://feeds.reuters.com/reuters/topNews
  - https://feeds.a.dj.com/rss/RSSMarketsMain.xml
  - https://finance.yahoo.com/news/rssindex
  - https://www.cnbc.com/id/19746125/device/rss/rss.html
EOF'
```

### Step 2: Ingest News & Analyze Sentiment

```bash
# Run news ingester with VADER sentiment analysis
docker compose exec -T optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml \
  --out data/macro_events.news.json \
  --symbols "BTC/USDT,ETH/USDT,NVDA,GC=F"

# Output:
# Fetching https://feeds.reuters.com/reuters/topNews...
# Parsed 25 entries
# Fetching https://feeds.a.dj.com/rss/RSSMarketsMain.xml...
# Parsed 32 entries
# ...
# Wrote 89 events -> data/macro_events.news.json
```

### Step 3: Verify Events

```bash
# Check file created
docker compose exec optimizer ls -l data/macro_events.news.json

# Preview first event
docker compose exec optimizer head -n 20 data/macro_events.news.json

# Sample event:
# {
#   "title": "Trump vows fresh tariff review amid trade tensions",
#   "category": "politics",
#   "sentiment": "bearish",
#   "impact": "high",
#   "bias": -0.6,
#   "assets": ["BTC/USDT", "NVDA"],
#   "interest_rate_expectation": "neutral"
# }
```

### Step 4: Restart Services

```bash
# Restart to load macro events
docker compose restart bot api

# Wait for startup
sleep 5

# Check macro insights via API
curl -sS http://localhost:8000/macro/insights | jq
```

### Output Example

```json
{
  "symbol": "BTC/USDT",
  "bias_score": -0.54,
  "confidence": 0.78,
  "summary": "Macro bias is bearish (-0.54) based on 3 tracked catalysts...",
  "drivers": [
    "Trump vows fresh tariffs review (bearish, high impact)",
    "US payrolls surprise to upside (hawkish, high impact)",
    "Fed officials guide for data-dependent path (neutral, medium impact)"
  ],
  "interest_rate_outlook": "Neutral with hawkish tilt",
  "political_summary": "Trade policy uncertainty elevated",
  "tracked_event_count": 89,
  "last_refresh": "2025-11-01T12:34:56Z"
}
```

### Step 5: View in Dashboard

Refresh the dashboard to see the **Macro & News Pulse** section:
- Bias score with visual indicator
- Top drivers with impact ratings
- Interest rate expectations
- Political risk assessment

---

## 🎮 Try Live Trading (Dry-Run Mode)

### Option 1: Docker Bot (Automatic)

Already running! The Docker bot continuously trades using current configs.

```bash
# Watch bot logs
docker compose logs -f bot

# You'll see:
# INFO: decision=FLAT price=63845.23 macro_bias=-0.54 ai_conf=52%
```

### Option 2: Interactive Script (Manual)

```bash
# Stop Docker bot (optional)
docker compose stop bot

# Run interactive live trading
python run_live_trading.py
```

### Configure Live Trading

```
============================================================
LIVE TRADING MODE
============================================================
⚠️  CAUTION: This script executes real trades!
⚠️  Recommended: Start with DRY RUN mode.

Trading Mode:
1. DRY RUN (log only, no real orders)
2. TESTNET (Binance testnet, real orders with fake money)
3. LIVE (REAL EXCHANGE - CAUTION!)
Choice (1/2/3): 1

Symbol (default: BTC/USDT): [Enter]
Timeframe (default: 5m): 1h
Loop interval (seconds) (default: 60): [Enter]

Strategy Parameters:
  Risk per trade % (default: 1.0): 0.5
  Stop Loss % (default: 2.0): [Enter]
  Take Profit % (default: 4.0): [Enter]
```

### Monitor Loop

```
============================================================
ITERATION #1 - 2025-11-01 12:34:56
============================================================
📊 Fetching data...
   Current price: $63,845.23

📊 No open position

🔍 Analyzing signal...
   Decision: FLAT
   Confidence: 52.01%
   RSI: 49.68
   EMA Fast: $63,642.60
   EMA Slow: $63,226.88
   Reason: No strong trend, waiting for clear signal

⏸️  No action taken (no signal or low confidence)

⏳ Waiting 60 seconds for next iteration...
```

**Press Ctrl+C** to stop.

---

## 🔧 Common Tasks

### Add a New Asset

```bash
# Edit portfolio config
nano data/portfolio.json

# Add entry:
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
      "symbol": "AAPL",
      "asset_type": "equity",
      "allocation": 0.3,
      "timeframe": "1h",
      "lookback": 500
    }
    // Add more here
  ]
}

# Run optimizer
docker compose exec optimizer python run_portfolio_optimize.py

# Verify via API
curl -sS http://localhost:8000/portfolio/strategies | jq
```

### Adjust Risk Parameters

```bash
# Edit global strategy
nano data/strategy_config.json

# Or per-asset strategy
nano data/portfolio/BTC_USDT/strategy_config.json

# Change parameters:
{
  "risk_per_trade_pct": 1.5,  # Increase risk
  "stop_loss_pct": 1.0,       # Tighter stop
  "take_profit_pct": 3.0      # Higher target
}

# Bot auto-reloads within 30-60s (no restart!)
```

### Refresh Macro Events

```bash
# Re-run news ingester
docker compose exec optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml \
  --out data/macro_events.news.json \
  --symbols "BTC/USDT,ETH/USDT,NVDA,GC=F"

# Bot auto-reloads every 300s (no restart needed)
# Or force reload:
docker compose restart bot api
```

### Check Service Health

```bash
# Service status
docker compose ps

# Logs
docker compose logs -f bot
docker compose logs -f api
docker compose logs --tail=50 optimizer

# Resource usage
docker stats --no-stream

# Disk usage
du -sh data/
```

### Restart Services

```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart bot
docker compose restart api

# Stop all
docker compose down

# Start again
docker compose up -d
```

---

## 🐛 Troubleshooting

### Issue: "State not found" in API

**Cause**: Bot hasn't written state file yet.

**Fix**:
```bash
# Wait for bot to complete first loop
sleep 30

# Check state file exists
docker compose exec bot ls -l data/state.json

# Try API again
curl -sS http://localhost:8000/status | jq
```

### Issue: Dashboard shows empty charts

**Cause**: Not enough data points yet.

**Fix**:
```bash
# Use preview mode for sample data
open http://localhost:8000/dashboard/preview

# Or wait for bot to run several loops
docker compose logs -f bot

# Once you see multiple signals, refresh dashboard
```

### Issue: Optimizer producing "no valid results"

**Cause**: Too few trades meet minimum criteria.

**Fix**:
```bash
# Edit .env
nano .env

# Reduce minimum threshold
AUTO_OPTIMIZE_MIN_TRADES=3  # Down from 5

# Or increase trials
AUTO_OPTIMIZE_TRIALS=100    # Up from 60

# Restart
docker compose restart optimizer
```

### Issue: NVDA/equity data not loading

**Cause**: yfinance rate limiting or slow response.

**Fix**:
```bash
# Check optimizer logs
docker compose logs optimizer | grep NVDA

# If timeouts, increase period or reduce frequency
nano .env
# AUTO_OPTIMIZE_INTERVAL_MINUTES=360  # Less frequent

docker compose restart optimizer
```

### Issue: Permission denied on data/ files

**Cause**: Docker volume permissions.

**Fix (Linux/macOS)**:
```bash
sudo chown -R $USER:$USER data/

# Or run with your UID
# Edit docker-compose.yml:
# services:
#   bot:
#     user: "${UID}:${GID}"
```

### Issue: API returns 500 error

**Cause**: Service error or missing dependencies.

**Fix**:
```bash
# Check API logs
docker compose logs api

# Restart API
docker compose restart api

# If persistent, rebuild
docker compose up -d --build api
```

---

## 📚 Next Steps

### 1. Enable Testnet Trading

Move from synthetic data to real market data (with fake money):

```bash
# Sign up at https://testnet.binance.vision/
# Create API keys with TRADE, USER_DATA, USER_STREAM permissions

# Edit .env
nano .env

# Update:
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=your_key_here
BINANCE_TESTNET_API_SECRET=your_secret_here
PAPER_MODE=false

# Restart
docker compose restart bot

# Monitor
docker compose logs -f bot
```

### 2. Explore Advanced Features

- **Walk-forward optimization**: Split data into train/test periods
- **Multi-objective optimization**: Optimize for Sharpe AND low drawdown
- **Regime detection**: Adjust strategy based on market volatility
- **Custom indicators**: Add your own technical indicators to `bot/strategy.py`
- **Portfolio rebalancing**: Automate allocation adjustments

### 3. Production Deployment

For serious usage:

- **Database**: Replace JSON with PostgreSQL/TimescaleDB
- **Monitoring**: Add Prometheus metrics + Grafana dashboards
- **Secrets**: Use AWS Secrets Manager or HashiCorp Vault
- **CI/CD**: Set up GitHub Actions for testing/deployment
- **Health checks**: Implement liveness/readiness probes
- **Logging**: Centralized logging with ELK or CloudWatch
- **Backups**: Automated state/config backups

### 4. Learn the Internals

Deep dive into system architecture:

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system design (13,000+ lines)
- **[REFERENCE.md](REFERENCE.md)**: Quick command reference
- **API Docs**: http://localhost:8000/docs
- **Code walkthrough**: `docs/ui_walkthrough.md`

---

## 📊 Performance Metrics

### Backtest Performance
- **500 bars, 60 trials**: 30-60 seconds
- **1000 bars, 100 trials**: 60-120 seconds

### Live Trading Performance
- **Decision cycle**: 1-2 seconds per iteration
- **API response time**: 5-50ms
- **Dashboard load**: <1 second

### Resource Usage
- **Memory**: ~200-300 MB per service
- **CPU**: <5% idle, 20-40% during optimization
- **Disk**: ~10 MB for state/configs, ~50-100 MB for historical data cache

---

## ✅ Complete Health Check

Run this comprehensive check:

```bash
echo "=== Docker Services ===" && \
docker compose ps && \
echo -e "\n=== API Health ===" && \
curl -sS http://localhost:8000/status > /dev/null && echo "✅ API responding" || echo "❌ API down" && \
echo -e "\n=== Dashboard ===" && \
curl -sS http://localhost:8000/dashboard > /dev/null && echo "✅ Dashboard accessible" || echo "❌ Dashboard down" && \
echo -e "\n=== Portfolio Configs ===" && \
echo "$(docker compose exec -T optimizer ls data/portfolio/*/strategy_config.json 2>/dev/null | wc -l) configs found" && \
echo -e "\n=== Macro Events ===" && \
docker compose exec -T optimizer test -f data/macro_events.news.json && echo "✅ Macro events loaded" || echo "⚠️  No macro events" && \
echo -e "\n=== Recent Bot Activity ===" && \
docker compose logs --tail=3 bot
```

**Expected output:**
```
=== Docker Services ===
NAME                         STATUS
algo_trading_lab-bot-1       running
algo_trading_lab-api-1       running
algo_trading_lab-optimizer-1 running

=== API Health ===
✅ API responding

=== Dashboard ===
✅ Dashboard accessible

=== Portfolio Configs ===
3 configs found

=== Macro Events ===
✅ Macro events loaded

=== Recent Bot Activity ===
[Recent log lines showing bot decisions]
```

---

## 🎓 Understanding Metrics

### Backtest Metrics

| Metric | Good | Acceptable | Poor | Meaning |
|--------|------|------------|------|---------|
| **Win Rate** | >60% | 50-60% | <50% | % of profitable trades |
| **Profit Factor** | >2.0 | 1.5-2.0 | <1.5 | Total wins / Total losses |
| **Sharpe Ratio** | >1.5 | 1.0-1.5 | <1.0 | Risk-adjusted returns |
| **Max Drawdown** | <5% | 5-10% | >10% | Largest peak-to-trough loss |

### AI Confidence

- **>70%**: High confidence → Strong signal, likely to act
- **50-70%**: Moderate confidence → Proceed with caution
- **<50%**: Low confidence → Bot holds FLAT

### Macro Bias

- **>0.3**: Bullish → Positive for longs, avoid shorts
- **-0.3 to 0.3**: Neutral → No clear macro direction
- **<-0.3**: Bearish → Favorable for shorts, cautious on longs

---

## 🎉 You're Ready to Trade!

**What you've accomplished:**
- ✅ Started multi-service trading system
- ✅ Explored dashboard and API
- ✅ Ran backtests to validate strategies
- ✅ Optimized parameters automatically
- ✅ Integrated macro/news sentiment
- ✅ Tested dry-run live trading

**Recommended workflow:**
1. **Research**: Backtest strategies, iterate on parameters
2. **Optimize**: Run optimizer for best parameters
3. **Dry-run**: Test in dry-run mode (1-2 days)
4. **Testnet**: Test on Binance testnet (1 week minimum)
5. **Live**: Start with very small capital (if confident)

**⚠️  Important Reminders:**
- Always start with **PAPER_MODE=true** or **DRY RUN**
- Never trade with money you can't afford to lose
- This is a research/educational platform, NOT financial advice
- Past performance does NOT guarantee future results
- Always test thoroughly before going live
- Use stop-losses and proper risk management

---

## 📞 Support & Resources

- **Issues**: https://github.com/ogulcanaydogan/algo_trading_lab/issues
- **Full Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) (13K lines)
- **Quick Reference**: [REFERENCE.md](REFERENCE.md)
- **API Reference**: http://localhost:8000/docs
- **Turkish Guide**: [QUICKSTART.md](QUICKSTART.md)
- **Community**: GitHub Discussions

---

**Happy Trading! 🚀📈**

*Last Updated: 2025-11-01 | Version: 1.0.0*

---

## 📖 Appendix: Command Cheat Sheet

```bash
# === Service Management ===
docker compose up -d              # Start all services
docker compose down               # Stop all services
docker compose restart            # Restart all services
docker compose ps                 # Check service status
docker compose logs -f bot        # Watch bot logs
docker compose logs -f api        # Watch API logs

# === Optimization ===
docker compose exec optimizer python run_portfolio_optimize.py
docker compose exec optimizer ls data/portfolio/*/strategy_config.json

# === Backtesting ===
python run_backtest.py            # Interactive backtest
docker compose exec bot python run_backtest.py  # Inside container

# === Live Trading ===
python run_live_trading.py        # Interactive dry-run/testnet/live

# === Macro/News ===
docker compose exec optimizer python tools/ingest_news_to_macro_events.py \
  --feeds feeds.news.yml --out data/macro_events.news.json \
  --symbols "BTC/USDT,NVDA,GC=F"

# === API Testing ===
curl http://localhost:8000/status | jq
curl http://localhost:8000/signals?limit=10 | jq
curl http://localhost:8000/strategy | jq
curl http://localhost:8000/portfolio/strategies | jq
curl http://localhost:8000/macro/insights | jq

# === Dashboard ===
open http://localhost:8000/dashboard         # Main dashboard
open http://localhost:8000/dashboard/preview # Preview with sample data
open http://localhost:8000/docs              # API documentation

# === Monitoring ===
docker stats --no-stream          # Resource usage
du -sh data/                      # Disk usage
docker compose logs --tail=50 bot # Last 50 bot logs

# === Troubleshooting ===
docker compose restart bot        # Restart bot
docker compose up -d --build      # Rebuild and restart
docker compose exec bot ls -la data/  # Check data files
```

---

