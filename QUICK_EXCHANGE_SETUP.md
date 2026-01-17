# 🚀 Quick Reference: Real Money Trading Setup

## Summary: Where Each Asset Trades

```
YOUR TRADING SYSTEM
│
├─ 🪙 CRYPTO (BTC, ETH, SOL, etc.)
│  └─→ BINANCE (binance.com)
│      • API Keys needed ✅
│      • Setup: 10 minutes
│      • Testnet available: Yes ✅
│      • Minimum: $10
│
├─ 📈 STOCKS (AAPL, MSFT, GOOGL, etc.)
│  └─→ ALPACA (alpaca.markets)  ← RECOMMENDED FOR STOCKS
│      • API Keys needed ✅
│      • Setup: 5 minutes
│      • Paper trading: Yes ✅
│      • Minimum: $100
│
│  OR Interactive Brokers (for all assets in one place)
│      • More complex but unified
│      • Minimum: $10,000
│
└─ 💎 COMMODITIES (Gold, Oil, Silver, Natural Gas)
   └─→ Interactive Brokers (interactivebrokers.com)  ← RECOMMENDED
       • API Keys needed ✅
       • Setup: 15 minutes
       • Testnet: Limited
       • Minimum: $10,000
   
   OR OANDA (oanda.com)  ← ALTERNATIVE
       • Easier to set up
       • No stocks
       • Minimum: $50
```

---

## 🎯 FASTEST PATH: Just Crypto

**If you just want to trade crypto on Binance:**

1. Sign up: https://binance.com
2. Go to Account → API
3. Create key (enable spot trading only)
4. Edit `.env`:
   ```bash
   EXCHANGE_API_KEY=your_key_here
   EXCHANGE_API_SECRET=your_secret_here
   EXCHANGE_SANDBOX=false
   ```
5. Run: `python run_unified_trading.py --mode live_limited --confirm`

**Time: 15 minutes** ⏱️

---

## 📊 COMPLETE PATH: All Three Asset Classes

**For crypto + stocks + commodities:**

### Step 1: Crypto on Binance (15 min)
```bash
# Get keys from Binance
EXCHANGE_API_KEY=binance_key
EXCHANGE_API_SECRET=binance_secret
```

### Step 2: Stocks on Alpaca (10 min)
```bash
# Get keys from Alpaca
ALPACA_API_KEY=alpaca_key
ALPACA_API_SECRET=alpaca_secret
```

### Step 3: Commodities - Choose One:

**Option A: OANDA (easier, 15 min)**
```bash
OANDA_ACCOUNT_ID=your_id
OANDA_API_TOKEN=your_token
```

**Option B: Interactive Brokers (more features, 30 min)**
```bash
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

**Total setup time: ~1 hour** ⏱️

---

## 💡 MY RECOMMENDATION

**Start with Binance only:**
1. Easy to set up
2. Lowest risk (crypto volatility lower than stocks)
3. Fast order execution
4. Can test API integration first
5. Testnet available to practice

**After 2 weeks of successful trading, add Alpaca for stocks.**

**After 1 month of profits, add OANDA for commodities.**

---

## 🔒 SECURITY CHECKLIST

Before going live:

✅ **API Key Security:**
- [ ] Created NEW keys (not account owner)
- [ ] Enabled read-only + trading only (no withdrawal)
- [ ] Whitelist your IP address
- [ ] Set max order size limit in exchange
- [ ] Store keys in `.env` (NOT in code)

✅ **Position Safety:**
- [ ] Max position size: 5% of capital
- [ ] Max daily loss: 2% of capital
- [ ] Stop loss: Required on every trade
- [ ] Take profit: Set automatically

✅ **Monitoring:**
- [ ] Watch logs: `tail -f data/unified_trading/logs/*.log`
- [ ] Check positions: Visit dashboard every hour
- [ ] Email alerts enabled
- [ ] Phone ready for emergency stop

---

## ⚡ EMERGENCY STOP

**If something goes wrong:**

```bash
# Immediately pause all trading
curl -X POST http://localhost:8000/api/trading/pause-all

# Kill the engine
pkill -f run_unified_trading

# Close any open positions manually on exchange
# Then analyze logs
tail -100 data/unified_trading/logs/*.log
```

---

## 📈 TESTING PHASES

| Phase | Mode | Capital | Duration | Goal |
|-------|------|---------|----------|------|
| 1 | Paper | $10K virtual | 1 week | Verify signals |
| 2 | Testnet (Crypto) | $0 (test funds) | 1 week | Test API |
| 3 | Live Limited | $100-500 | 2-4 weeks | Real money |
| 4 | Live Full | $5,000+ | Forever | Production |

---

## 📞 NEXT STEPS

1. **Read full guide:** [REAL_EXCHANGE_INTEGRATION.md](./REAL_EXCHANGE_INTEGRATION.md)
2. **Get Binance API keys:** https://binance.com → Account → API
3. **Update `.env` file** with keys
4. **Test on testnet first:** 
   ```bash
   BINANCE_TESTNET_ENABLED=true python run_unified_trading.py run
   ```
5. **Start live_limited:** 
   ```bash
   python run_unified_trading.py --mode live_limited --confirm
   ```
6. **Monitor dashboard:** http://localhost:8000

---

**Questions about setup? Check the detailed guide! ☝️**
