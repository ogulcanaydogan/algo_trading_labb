# 🚀 Quick Start Guide - Paper Trading

## ✅ Prerequisites Check

You've already completed these:
- [x] Smoke tests passed (29/29)
- [x] Virtual environment activated
- [x] API keys configured (Finnhub, CryptoPanic, Binance, Kraken, Telegram, Anthropic)
- [x] All dependencies installed

## 📊 Current Status

- **Mode:** Paper Trading (stopped)
- **Balance:** $10,000 (virtual)
- **Symbols:** BTC, ETH, SOL, AVAX, XRP, ADA, DOT, MATIC, LINK, UNI
- **Progress to Testnet:** 24.3%

### Requirements to Graduate to Testnet:
- ⏳ Days in paper mode: 3/14 (need 11 more days)
- ⏳ Total trades: 0/100 (need 100 trades)
- ⏳ Win rate: 0%/45% (need 45%+ win rate)
- ✅ Max drawdown: 0%/12% (OK)
- ⏳ Profit factor: 0.0/1.0 (need 1.0+)

## 🎯 Step-by-Step Startup

### Step 1: Start Paper Trading Bot

**Option A - Quick Start Script (Recommended):**
```bash
./start_paper_trading.sh
```

**Option B - Manual Command:**
```bash
python3 run_unified_trading.py run --mode paper_live_data
```

### Step 2: Start API Dashboard (New Terminal)

Open a **new terminal window** and run:
```bash
cd /Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab
source .venv/bin/activate
python3 -m uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Open Dashboard

Open your browser:
- **Main Dashboard:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Alternative UI:** http://localhost:8000/redoc

## 📱 Monitoring Commands

Use these in a separate terminal while bot is running:

```bash
# Check current status
python3 run_unified_trading.py status

# Check progress to testnet
python3 run_unified_trading.py check-transition testnet

# Emergency stop (if needed)
python3 run_unified_trading.py emergency-stop
```

## 🔔 What Will Happen

1. **Bot starts** fetching live prices from Binance (paper mode, no real money)
2. **AI analyzes** market conditions using:
   - ML models (XGBoost, LSTM)
   - Technical indicators (EMA, RSI, MACD)
   - News sentiment (Finnhub + CryptoPanic)
   - Macro events
3. **Generates signals** when conditions are favorable
4. **Executes paper trades** (simulated, safe)
5. **Telegram notifications** sent for important events
6. **Dashboard updates** in real-time

## 📊 Dashboard Features

- **Live Equity Curve:** See your portfolio value over time
- **Position Tracker:** Monitor open positions
- **Trade History:** Review past trades
- **AI Insights:** See what the AI is thinking
- **Risk Metrics:** Track win rate, profit factor, drawdown
- **News Sentiment:** See market sentiment scores
- **Multi-Market View:** Compare crypto/stock/commodity performance

## ⚠️ Safety Features Active

- ✅ Paper mode only (no real money at risk)
- ✅ 2% daily loss limit (auto-stops if hit)
- ✅ $50 max position size (when you go live later)
- ✅ Stop-loss on every trade
- ✅ Emergency stop available anytime

## 🎓 Learning Path

### Week 1-2: Paper Trading
- Let bot run for 14 days
- Need 100+ trades with 45%+ win rate
- Learn from trade decisions

### Week 3: Testnet (If Qualified)
- Graduate to Binance testnet
- Test with fake money on real exchange
- Verify order execution works

### Week 4+: Live Trading (If Testnet Successful)
- Start with small capital ($50-200)
- Gradually increase as confidence grows
- Always monitor performance

## 🐛 Troubleshooting

### Bot won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Restart
./start_paper_trading.sh
```

### No trades being made
- This is normal if market conditions aren't favorable
- AI waits for high-confidence signals
- Check dashboard for signal history

### API errors
- Check `.env` file has correct API keys
- Verify internet connection
- Check API rate limits (should be OK with current optimization)

## 📞 Getting Help

If something goes wrong:
1. Check `data/unified_trading/logs/` for error messages
2. Review last trades in dashboard
3. Check API health: http://localhost:8000/health
4. Review smoke test: `python3 tests/smoke_test.py`

## 🎉 You're Ready!

Run this command to start:
```bash
./start_paper_trading.sh
```

Then open another terminal and start the dashboard:
```bash
python3 -m uvicorn api.api:app --reload
```

**Dashboard URL:** http://localhost:8000

---

*Last updated: January 16, 2026*
