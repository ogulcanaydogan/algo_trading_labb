# 🚀 QUICK START: Resume Live Trading

## Current Status
- ✅ Mode: **live_limited** (real Binance capital: $10,625)
- ✅ Positions: **0 open** (emergency closed)
- ✅ Balance: **$10,625.05** (+6.26% profit)
- ✅ Safety Controls: **ENFORCED** (5% position, 2% daily loss)
- ✅ System: **READY TO TRADE**

---

## Step 1: Start Trading Engine

```bash
# Option A: Basic start (BTC/USDT default)
python run_unified_trading.py run --mode live_limited --confirm

# Option B: Specific symbol
python run_unified_trading.py run --mode live_limited --confirm --symbol ETH/USDT

# Option C: Watch logs while running
tail -f data/unified_trading/logs/live_limited_*.log &
python run_unified_trading.py run --mode live_limited --confirm
```

**Expected Output:**
```
✅ Trading engine initialized
✅ Mode: live_limited
✅ Capital: $10,625.05
✅ Safety controller: ACTIVE
✅ Entering main trading loop...
```

---

## Step 2: Open Dashboard

```bash
# Ensure API server is running
curl http://localhost:8000/health

# Then open in browser
http://localhost:8000/dashboard

# Or preview mode (doesn't require trading engine)
http://localhost:8000/dashboard/preview
```

**What to Monitor:**
- Position size as % of balance (should be < 5%)
- Daily P&L (should stay > -2%)
- Trade execution count
- Latest signal timestamp

---

## Step 3: Set Up Alerts (Optional)

```bash
# Enable Telegram notifications
export TELEGRAM_BOT_TOKEN=your_bot_token
export TELEGRAM_CHAT_ID=your_chat_id

# Engine will now send alerts for:
# - New positions opened
# - Position size violations (should not happen now)
# - Daily loss limit hits
# - Emergency stops
# - Major profitable trades
```

To get Telegram token:
1. Message @BotFather on Telegram
2. Create new bot, get token
3. Message your bot and get chat ID using: `curl https://api.telegram.org/bot<token>/getUpdates`

---

## Step 4: Monitor Compliance

### Check Position Size
```bash
curl http://localhost:8000/api/unified/status | jq '.positions | .[] | {symbol, size_pct: (.entry_price * .quantity / 10625 * 100)}'
```
Expected: All positions < 5%

### Check Daily Loss
```bash
curl http://localhost:8000/api/unified/status | jq '{daily_pnl, max_daily_loss: 212.50}'
```
Expected: daily_pnl > -212.50

### Check Trades
```bash
curl http://localhost:8000/api/unified/status | jq '{total_trades, winning_trades, daily_trades}'
```

---

## 🛑 Emergency Stop (If Needed)

```bash
# Immediate halt
curl -X POST http://localhost:8000/api/unified/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason":"Manual emergency halt"}'

# Verify it's active
curl http://localhost:8000/api/unified/status | jq '.safety.emergency_stop_active'
# Should return: true

# Clear when issue is resolved
curl -X POST http://localhost:8000/api/unified/clear-stop \
  -H "Content-Type: application/json" \
  -d '{"approver":"admin"}'

# Verify cleared
curl http://localhost:8000/api/unified/status | jq '.safety.emergency_stop_active'
# Should return: false
```

---

## 📊 Key Metrics to Monitor

| Metric | Good Range | Action if Violated |
|--------|-----------|-------------------|
| Position Size | < 5% of balance | Auto-rejected by safety controller |
| Daily Loss | > -2% of balance | Trading auto-paused |
| Win Rate | > 50% | Adjust strategy if < 40% |
| API Latency | < 500ms | Restart if > 2000ms |
| Open Positions | 0-3 | Max 3 allowed |
| Consecutive Losses | < 5 | Auto-pause after 5 |

---

## 🔍 Troubleshooting

### Engine Won't Start
```bash
# Check if another instance is running
ps aux | grep run_unified

# Kill any stray processes
pkill -f run_unified

# Try again
python run_unified_trading.py run --mode live_limited --confirm
```

### API Not Responding
```bash
# Check if API server is running
curl http://localhost:8000/health

# If not, restart it
pkill -f "uvicorn api.api:app"
sleep 2
source .venv/bin/activate
python -m uvicorn api.api:app --host 0.0.0.0 --port 8000 &
```

### Position Too Large
```bash
# This should NOT happen (safety controller enforces 5%)
# If it does, immediately emergency stop:
curl -X POST http://localhost:8000/api/unified/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason":"Oversized position detected"}'

# Manually close via API
curl -X POST http://localhost:8000/api/unified/close-position \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","reason":"Manual closure - safety violation"}'
```

### Lost Trades or Balance Inconsistency
```bash
# Check state file integrity
cat data/unified_trading/state.json | jq '.balance, .total_pnl, .positions | length'

# Compare with API
curl http://localhost:8000/api/unified/status | jq '.balance, .total_pnl, .open_positions'

# If inconsistent, stop trading and investigate logs
tail -50 data/unified_trading/logs/live_limited_*.log
```

---

## 📝 Best Practices

1. **Monitor First 30 Minutes**
   - Watch dashboard for first trades
   - Verify position sizes are reasonable
   - Check P&L is updating correctly

2. **Set Phone Reminders**
   - Check dashboard at market open/close
   - Review daily P&L before market close
   - Check for emergency stops

3. **Daily Review**
   - Win rate trend
   - Average position size
   - API latency patterns
   - Telegram alerts received

4. **Weekly Review**
   - Total P&L and Sharpe ratio
   - Win/loss distribution
   - Position concentration
   - Capital efficiency

---

## 🎯 Success Criteria

✅ Trading is successful when:
- Win rate stays > 50%
- Daily P&L stays > -2%
- Position sizes stay < 5%
- No emergency stops triggered
- API latency stays < 500ms
- Trades execute within 1 second

---

## ⚡ Quick Commands

```bash
# Start trading
python run_unified_trading.py run --mode live_limited --confirm

# Check status
curl http://localhost:8000/api/unified/status | jq '.'

# Emergency stop
curl -X POST http://localhost:8000/api/unified/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason":"Emergency halt"}'

# View dashboard
open http://localhost:8000/dashboard

# View logs
tail -f data/unified_trading/logs/live_limited_*.log

# Check P&L
curl http://localhost:8000/api/unified/status | jq '.total_pnl, .daily_pnl'
```

---

**Ready?** Run: `python run_unified_trading.py run --mode live_limited --confirm`

**Questions?** Check `RECOVERY_REPORT.md` for full technical details.
