# Live Trading Setup Guide

## 📋 Overview

This guide covers how to configure broker connections for live trading in the Algo Trading Lab system. The system supports multiple asset classes through different brokers:

| Asset Class | Broker | Markets |
|-------------|--------|---------|
| **Crypto** | Binance | BTC, ETH, SOL, AVAX, XRP, etc. |
| **Stocks** | Alpaca Markets | AAPL, MSFT, GOOGL, NVDA, TSLA, etc. |
| **Forex** | OANDA | EUR/USD, GBP/USD, USD/JPY, etc. |
| **Commodities** | OANDA | XAU/USD (Gold), XAG/USD (Silver), Oil |
| **Indices** | OANDA | SPX500, NAS100, US30, etc. |

---

## 🔐 Environment Variables Setup

All broker credentials are configured via environment variables or a `.env` file. **Never commit real API keys to version control.**

### Step 1: Create .env File

```bash
# Copy the example file
cp .env.example .env
```

### Step 2: Configure Your Credentials

Edit `.env` with your actual API keys:

```bash
# =============================================================================
# TRADING MODE
# =============================================================================
PAPER_MODE=true                    # Start with paper trading!
LIVE_MODE=false                    # Set to true only after validation

# =============================================================================
# BINANCE (CRYPTO)
# =============================================================================
# For Testnet/Demo (recommended to start):
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret

# For Live Trading (only after extensive testing):
EXCHANGE_API_KEY=your_live_binance_api_key
EXCHANGE_API_SECRET=your_live_binance_api_secret
EXCHANGE_SANDBOX=false
EXCHANGE_ID=binance

# =============================================================================
# ALPACA MARKETS (STOCKS)
# =============================================================================
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_PAPER_MODE=true             # Use paper trading first!

# =============================================================================
# OANDA (FOREX / COMMODITIES / INDICES)
# =============================================================================
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_account_id   # e.g., 001-004-1234567-001
OANDA_ENVIRONMENT=practice         # practice or live

# =============================================================================
# NOTIFICATIONS (Optional)
# =============================================================================
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🏦 Broker-Specific Setup

### 1. Binance (Cryptocurrency)

#### Getting API Keys

**Testnet (Demo)** - Recommended for testing:
1. Go to https://testnet.binance.vision/
2. Log in with GitHub
3. Click "Generate HMAC_SHA256 Key"
4. Copy the API Key and Secret Key

**Live Trading**:
1. Log in to https://www.binance.com
2. Go to Profile → API Management
3. Create a new API key
4. Enable "Spot & Margin Trading" permission
5. **Restrict IP access** for security
6. Copy API Key and Secret

#### Configuration

```bash
# Testnet (Demo)
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=xxx
BINANCE_TESTNET_API_SECRET=xxx

# Live (use with caution)
EXCHANGE_API_KEY=xxx
EXCHANGE_API_SECRET=xxx
EXCHANGE_SANDBOX=false
```

#### Supported Symbols
- BTC/USDT, ETH/USDT, SOL/USDT, AVAX/USDT
- XRP/USDT, ADA/USDT, DOT/USDT, MATIC/USDT
- LINK/USDT, UNI/USDT, and 100+ more

---

### 2. Alpaca Markets (Stocks)

#### Getting API Keys

**Paper Trading**:
1. Sign up at https://alpaca.markets
2. Go to Dashboard → Paper Trading
3. Click "View API Keys"
4. Generate or copy existing keys

**Live Trading**:
1. Complete account verification (requires ID, funding)
2. Go to Dashboard → Live Trading
3. Generate API keys
4. Enable trading permissions

#### Configuration

```bash
ALPACA_API_KEY=PKxxx...
ALPACA_API_SECRET=xxx...
ALPACA_PAPER_MODE=true    # Set to false for live trading
```

#### Supported Symbols
- US Stocks: AAPL, MSFT, GOOGL, NVDA, AMZN, TSLA, META, etc.
- Commission-free trading
- Fractional shares supported

---

### 3. OANDA (Forex, Commodities, Indices)

#### Getting API Keys

**Practice Account**:
1. Sign up at https://www.oanda.com
2. Create a practice account (demo)
3. Go to "Manage API Access" in account settings
4. Generate an API token
5. Note your Account ID (format: 001-004-xxxxxxx-001)

**Live Account**:
1. Complete account verification and funding
2. Generate live API token
3. Use your live account ID

#### Configuration

```bash
OANDA_API_KEY=xxx-xxx-xxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxx
OANDA_ACCOUNT_ID=001-004-1234567-001
OANDA_ENVIRONMENT=practice   # or "live"
```

#### Supported Instruments

**Forex Majors:**
- EUR/USD, GBP/USD, USD/JPY, USD/CHF
- AUD/USD, NZD/USD, USD/CAD

**Commodities:**
- XAU/USD (Gold), XAG/USD (Silver)
- WTICO/USD (WTI Oil), BCO/USD (Brent Oil)
- NATGAS/USD (Natural Gas), XCU/USD (Copper)

**Indices:**
- SPX500/USD (S&P 500), NAS100/USD (Nasdaq 100)
- US30/USD (Dow Jones), UK100/GBP (FTSE 100)
- DE30/EUR (DAX), JP225/USD (Nikkei)

---

## ⚠️ Safety Features

The system includes multiple layers of safety controls:

### 1. Paper Trading Mode (Default)
```bash
PAPER_MODE=true
```
- Simulates all trades without real money
- Uses synthetic or live market data
- No risk to capital

### 2. Live Trading Guardrails
Located in `bot/live_trading_guardrails.py`:

```bash
# Enable live mode only after validation
LIVE_MODE=false                      # Master switch

# Position limits
LIVE_MAX_CAPITAL_PCT=0.01            # Max 1% of portfolio per day
LIVE_MAX_POSITION_PCT=0.02           # Max 2% per position

# Symbol restrictions (start with ONE symbol)
LIVE_SYMBOL_ALLOWLIST=ETH/USDT       # Only allowed symbols

# Trade limits
LIVE_MAX_TRADES_PER_DAY=3            # Max 3 trades per day
LIVE_MAX_LEVERAGE=1.0                # No leverage
```

### 3. Kill Switch
Create `data/live_kill_switch.txt` to immediately halt all live trading:
```bash
echo "Emergency stop - market conditions" > data/live_kill_switch.txt
```

Or via environment:
```bash
LIVE_KILL_SWITCH=true
```

### 4. Daily Loss Limits (in config.yaml)
```yaml
unified_trading:
  live_limited:
    capital_limit: 100              # Max $100 capital
    max_position_usd: 20            # Max $20 per position
    max_daily_loss_usd: 2           # Stop at $2 loss
    max_daily_loss_pct: 0.02        # Or 2% loss
    max_trades_per_day: 10          # Max trades
    max_open_positions: 3           # Max concurrent positions
```

---

## 🚀 Transitioning to Live Trading

### Phase 1: Paper Trading (2+ weeks)
1. Run paper trading with real market data
2. Verify strategies are profitable
3. Track metrics: win rate, Sharpe ratio, max drawdown
4. **Required**: Win rate > 45%, Max drawdown < 12%

### Phase 2: Testnet Trading (2+ weeks)
1. Enable Binance testnet
2. Trade with simulated funds on real exchange
3. Verify order execution works correctly
4. Test error handling and edge cases

### Phase 3: Live Limited (1+ month)
1. Set `LIVE_MODE=true` with strict guardrails
2. Trade with minimal capital ($20-100)
3. Single symbol only (e.g., ETH/USDT)
4. Max 3 trades per day
5. **Required**: Win rate > 45%, Profit factor > 1.0

### Phase 4: Live Full
1. Increase capital limits gradually
2. Add more symbols to allowlist
3. Continue monitoring and adjusting

---

## 🔍 Verification Commands

### Check Broker Connections

```python
# Run in Python to test connections
import asyncio
from bot.alpaca_adapter import create_alpaca_adapter
from bot.oanda_adapter import create_oanda_adapter

async def test_connections():
    # Test Alpaca
    alpaca = create_alpaca_adapter(is_paper=True)
    if alpaca:
        success = await alpaca.initialize()
        print(f"Alpaca: {'✓ Connected' if success else '✗ Failed'}")
        balance = await alpaca.get_balance()
        print(f"  Balance: ${balance.total:.2f}")
    
    # Test OANDA
    oanda = create_oanda_adapter(environment="practice")
    if oanda:
        success = await oanda.initialize()
        print(f"OANDA: {'✓ Connected' if success else '✗ Failed'}")
        balance = await oanda.get_balance()
        print(f"  Balance: {balance.currency} {balance.total:.2f}")

asyncio.run(test_connections())
```

### Check Live Trading Status

```python
from bot.live_trading_guardrails import get_live_guardrails

guardrails = get_live_guardrails()
status = guardrails.get_status()

print("Live Trading Status:")
print(f"  Live Mode: {status['live_mode_enabled']}")
print(f"  Kill Switch: {status['kill_switch_active']}")
print(f"  Trades Today: {status['daily_trades']['count']}/{status['daily_trades']['limit']}")
print(f"  Allowed Symbols: {status['symbol_allowlist']}")
```

---

## 📁 Configuration Files Summary

| File | Purpose |
|------|---------|
| `.env` | API keys and secrets (git-ignored) |
| `.env.example` | Template for .env file |
| `config/config.yaml` | Trading parameters, risk limits |
| `bot/config.py` | Configuration loading and validation |
| `data/live_trading_state.json` | Persistent trade tracking |
| `data/live_kill_switch.txt` | Emergency stop file |

---

## ❌ Common Issues

### "Alpaca credentials not found"
- Ensure `ALPACA_API_KEY` and `ALPACA_API_SECRET` are set
- Check for typos in variable names

### "OANDA initialization failed"
- Verify `OANDA_ACCOUNT_ID` format (e.g., `001-004-1234567-001`)
- Check if API key matches environment (practice vs live)
- Ensure account has trading permissions

### "Kill switch active"
- Remove `data/live_kill_switch.txt` file
- Or unset `LIVE_KILL_SWITCH` environment variable
- Verify system readiness before deactivating

### "Daily trade limit reached"
- Wait for next day (UTC reset)
- Or increase `LIVE_MAX_TRADES_PER_DAY` in config

---

## 🛡️ Security Best Practices

1. **Never commit `.env` to git** - It's in `.gitignore` by default
2. **Use IP restrictions** on Binance API keys
3. **Enable 2FA** on all broker accounts
4. **Use read-only keys** for monitoring scripts
5. **Rotate API keys** periodically
6. **Start with testnet/paper** before live trading
7. **Set withdrawal whitelist** on exchanges
8. **Use separate API keys** for different systems

---

## 📞 Broker Support Links

- **Binance**: https://www.binance.com/en/support
- **Alpaca**: https://alpaca.markets/learn
- **OANDA**: https://www.oanda.com/help/

---

*Last updated: 2026-01-31*
*For questions, check the existing docs in `docs/` or the README.md*
