# Quick Start Guide

## 1. Setup and Initial Test

```bash
# Clone or download the repository
cd algo_trading_lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

### Get Binance Testnet API Key
1. Go to https://testnet.binance.vision/
2. Login and create an API key (TRADE, USER_DATA, USER_STREAM permissions)
3. Copy the API Key and Secret Key
4. Open `.env` file and add the keys:

```bash
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=your_api_key
BINANCE_TESTNET_API_SECRET=your_secret_key
PAPER_MODE=false
```

### Test Connection
```bash
python test_binance_testnet.py
```

Successful output:
```
Client created successfully
Successfully fetched 10 candles
Account access successful
All tests passed! Testnet connection is working.
```

---

## 2. Test Strategy with Backtest

```bash
python scripts/backtest/run_backtest.py
```

**Example Inputs:**
- Symbol: `BTC/USDT`
- Timeframe: `1h`
- Number of candles: `1000`
- Starting balance: `10000`
- EMA Fast: `12`
- EMA Slow: `26`
- RSI Period: `14`
- Risk per trade %: `1.0`
- Stop Loss %: `2.0`
- Take Profit %: `4.0`
- Data source: `1` (Binance Testnet)

### Evaluate Results

**Good Results:**
- Win Rate > 55%
- Profit Factor > 1.5
- Max Drawdown < 15%
- Sharpe Ratio > 1.0

**Poor Results:**
- Win Rate < 45%
- Profit Factor < 1.0
- Max Drawdown > 30%

If results are poor, change parameters and test again.

---

## 3. Live Test with DRY RUN

```bash
python scripts/trading/run_live_trading.py
```

**Selections:**
- Trading Mode: `1` (DRY RUN)
- Symbol: `BTC/USDT`
- Timeframe: `5m`
- Loop interval: `60` (seconds)

In this mode:
- Uses real data
- You see signals
- Keeps logs
- Does NOT send real orders

**What to monitor:**
1. Are signals logical?
2. Are stop loss and take profit levels appropriate?
3. Is it trading too frequently?
4. Are RSI and EMA working correctly?

---

## 4. Real Order Test with TESTNET

```bash
python scripts/trading/run_live_trading.py
```

**Selections:**
- Trading Mode: `2` (TESTNET)
- Other settings same

In this mode:
- Sends real orders (with test money)
- Stop loss and take profit orders work
- Order cancellation and position closing are tested

**What to monitor:**
1. Are orders being sent correctly?
2. Is stop loss triggering?
3. Is take profit working?
4. Any error messages?

---

## 5. Parameter Optimization

Run backtests with different parameters and find the best combination:

| Parameter | Test Values |
|-----------|-------------|
| EMA Fast | 8, 12, 16 |
| EMA Slow | 21, 26, 34 |
| RSI Period | 7, 14, 21 |
| Risk % | 0.5, 1.0, 2.0 |
| Stop Loss % | 1.0, 2.0, 3.0 |
| Take Profit % | 2.0, 4.0, 6.0 |

**Example Test Matrix:**
```bash
# Test 1: Fast EMA
EMA Fast: 8, EMA Slow: 21 -> Run backtest

# Test 2: Standard EMA
EMA Fast: 12, EMA Slow: 26 -> Run backtest

# Test 3: Slow EMA
EMA Fast: 16, EMA Slow: 34 -> Run backtest

# Select the combination with best results
```

---

## 6. Moving to Production (Optional)

**WARNING**: You will be using real money!

### Start Small First
1. Create real API key on Binance
2. Update `.env` file:
```bash
BINANCE_TESTNET_ENABLED=false
EXCHANGE_API_KEY=real_api_key
EXCHANGE_API_SECRET=real_secret
```

3. Use **very small position** for first trade
4. Monitor for 1 week
5. If successful, gradually increase

---

## Metrics Table

| Metric | Good | Average | Poor |
|--------|------|---------|------|
| Win Rate | >60% | 50-60% | <50% |
| Profit Factor | >2.0 | 1.5-2.0 | <1.5 |
| Max Drawdown | <10% | 10-20% | >20% |
| Sharpe Ratio | >1.5 | 1.0-1.5 | <1.0 |

---

## Troubleshooting

### Error: "Import could not be resolved"
```bash
# Did you activate the virtual environment?
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Error: "API keys not found"
Make sure `.env` file is in the correct location (project root)

### Error: "Not enough data for indicator calculation"
Fetch more candle data (increase lookback value)

### No test money
Binance testnet provides automatic test money, but it can sometimes reset. Try creating a new account.

---

## Next Steps

1. **WebSocket Integration**: Use WebSocket for faster data
2. **Multi-Timeframe**: Get signals from different timeframes
3. **Machine Learning**: Integrate ML models
4. **Dashboard**: Add visual interface with Streamlit or Dash
5. **Alert System**: Telegram or email notifications

---

## Tips

- **Risk Management**: Don't risk more than 1-2% of total balance in a single trade
- **Be Patient**: Wait for good opportunities, don't enter every signal
- **Backtest is Important**: Don't move to live trading without backtest
- **Use Stop Loss**: Always set a stop loss
- **Keep Logs**: Record and analyze all trades
- **Continuously Improve**: Regularly review results

---

## Help

For questions:
1. Read README.md file
2. Check docstrings in code
3. Analyze backtest results
4. Test on testnet first

**Good luck!**
