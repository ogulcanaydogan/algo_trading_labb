# Real Exchange Integration Guide

## Current Setup

Your system is configured for **3 asset classes** across different exchanges:

| Asset Class | Current Config | Real Exchange | Status |
|-------------|----------------|---------------|--------|
| **Crypto** | Paper Trading | Binance Spot | ✅ Ready |
| **Commodities** | Paper Trading | Interactive Brokers (IB) | ⏳ Needs Setup |
| **Stocks** | Paper Trading | Alpaca or IB | ⏳ Needs Setup |

---

## 1️⃣ CRYPTO - Binance (Ready to Go)

### Current Symbols
```yaml
BTC/USDT, ETH/USDT, SOL/USDT, AVAX/USDT, XRP/USDT, 
ADA/USDT, DOT/USDT, MATIC/USDT, LINK/USDT, UNI/USDT
```

### Setup Steps

#### Step 1: Get Binance API Keys
1. Go to https://www.binance.com/
2. Sign up or log in
3. Go to **Account** → **API Management**
4. Click **"Create API"** → **"System Generated"**
5. Label: `algo_trading_lab`
6. Set restrictions:
   - ✅ Enable Spot Trading
   - ✅ Enable Reading
   - ✅ Enable Orders
   - ❌ Disable Withdrawal
   - Whitelist IP: Your server IP
7. Copy **API Key** and **Secret Key**

#### Step 2: Add to Environment

**Edit `.env` file:**
```bash
cp .env.example .env
```

**Add these lines:**
```bash
# Binance Live Trading (Real Money)
EXCHANGE_ID=binance
EXCHANGE_API_KEY=your_actual_api_key_here
EXCHANGE_API_SECRET=your_actual_api_secret_here
EXCHANGE_SANDBOX=false

# Or use Binance Testnet first (Recommended for testing!)
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=your_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret
```

#### Step 3: Switch Trading Mode

**Option A: Live Limited (Recommended for starting)**
```bash
# Small capital, limited risk
python run_unified_trading.py --mode live_limited --confirm
```

**Option B: Full Live Trading**
```bash
# Full capital, unlimited trading
python run_unified_trading.py --mode live_full --confirm
```

#### Step 4: Monitor Trading
```bash
# Watch logs
tail -f data/unified_trading/logs/*.log

# Check positions
curl http://localhost:8000/api/trading/control-panel | jq '.markets'

# View dashboard
open http://localhost:8000
```

---

## 2️⃣ COMMODITIES - Interactive Brokers or OANDA

### Symbols to Trade
```yaml
XAU/USD (Gold)
XAG/USD (Silver)
USOIL/USD (Crude Oil)
NATGAS/USD (Natural Gas)
```

### Option A: Interactive Brokers (Recommended)

#### Advantages:
- ✅ All asset classes (stocks, commodities, forex, crypto)
- ✅ Lowest commissions
- ✅ Professional-grade platform
- ✅ Global liquidity

#### Setup Steps:

1. **Create IB Account**
   - Go to https://www.interactivebrokers.com/
   - Create account with $10,000+ minimum
   - Complete verification (3-5 business days)

2. **Enable API Trading**
   - Log in to TWS (Trader Workstation) or Gateway
   - Account → Settings → API → Enable ActiveX and Socket Clients
   - Set Master Client ID and User ID

3. **Install IB Client Portal API**
   ```bash
   pip install ibapi
   ```

4. **Add to .env**
   ```bash
   IB_HOST=127.0.0.1
   IB_PORT=7497
   IB_CLIENT_ID=1
   IB_MASTER_ID=your_account_number
   ```

5. **Update execution adapter**
   - Modify `bot/execution_adapter.py` to add `IBExecutionAdapter` class
   - Map commodity symbols to IB contract specifications

### Option B: OANDA (Alternative)

#### Advantages:
- ✅ Forex + Commodities
- ✅ Easier setup
- ✅ Free API access
- ❌ No stocks

#### Setup Steps:

1. **Create OANDA Account**
   - Go to https://www.oanda.com/
   - Create trading account
   - Fund with $50+ minimum

2. **Get API Token**
   - Account Settings → Developer → API Access
   - Generate Personal Access Token

3. **Add to .env**
   ```bash
   OANDA_ACCOUNT_ID=your_account_id
   OANDA_API_TOKEN=your_api_token
   OANDA_ENVIRONMENT=live
   ```

4. **Update execution adapter**
   - Modify `bot/execution_adapter.py` to add `OANDAExecutionAdapter` class

---

## 3️⃣ STOCKS - Alpaca or Interactive Brokers

### Symbols to Trade
```yaml
AAPL, MSFT, GOOGL, NVDA, AMZN, TSLA, META, JPM, V, MA
```

### Option A: Alpaca (Easiest for Stocks)

#### Advantages:
- ✅ Easiest to set up
- ✅ Fractional shares
- ✅ Paper trading included
- ✅ Low minimum ($100)
- ❌ US stocks only

#### Setup Steps:

1. **Create Alpaca Account**
   - Go to https://alpaca.markets/
   - Sign up with Google/GitHub
   - Complete KYC (identity verification)
   - Fund with $100+ (or use paper trading)

2. **Get API Keys**
   - Dashboard → API Keys
   - Copy Key ID and Secret Key

3. **Add to .env**
   ```bash
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_api_secret
   ALPACA_PAPER_TRADING=false  # Set to false for live trading
   ```

4. **Install Alpaca SDK**
   ```bash
   pip install alpaca-trade-api
   ```

5. **Update execution adapter**
   - Modify `bot/execution_adapter.py` to add `AlpacaExecutionAdapter` class

### Option B: Interactive Brokers (For All Assets)

#### Setup:
Same as Commodities section above - IB covers stocks, commodities, forex, and crypto all in one account.

---

## 🔧 Implementation Steps

### 1. Modify Execution Adapter

**Location:** `bot/execution_adapter.py`

Add new adapter classes:

```python
class AlpacaExecutionAdapter(ExecutionAdapter):
    """Alpaca stocks execution adapter."""
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = False):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self._client = None
    
    async def connect(self) -> bool:
        from alpaca_trade_api import REST
        self._client = REST(
            base_url='https://api.alpaca.markets',
            key_id=self.api_key,
            secret_key=self.api_secret,
            api_version='v2'
        )
        return True
    
    async def get_current_price(self, symbol: str) -> float:
        bar = self._client.get_latest_bar(symbol)
        return bar.c  # Close price
    
    # ... implement other required methods
```

### 2. Update Routing Logic

**Location:** `bot/unified_engine.py` or `api/unified_trading_api.py`

Map symbols to correct adapters:

```python
SYMBOL_TO_EXCHANGE = {
    # Crypto → Binance
    "BTC/USDT": "binance",
    "ETH/USDT": "binance",
    
    # Commodities → Interactive Brokers
    "XAU/USD": "interactive_brokers",
    "XAG/USD": "interactive_brokers",
    "USOIL/USD": "interactive_brokers",
    
    # Stocks → Alpaca
    "AAPL": "alpaca",
    "MSFT": "alpaca",
    "GOOGL": "alpaca",
}
```

### 3. Create Exchange Factory

```python
async def create_execution_adapter(symbol: str, mode: str):
    """Create appropriate execution adapter for symbol and mode."""
    
    if mode == "paper_live_data":
        return PaperExecutionAdapter()
    
    exchange = SYMBOL_TO_EXCHANGE.get(symbol)
    
    if exchange == "binance":
        return BinanceExecutionAdapter(
            api_key=os.getenv("EXCHANGE_API_KEY"),
            api_secret=os.getenv("EXCHANGE_API_SECRET"),
            testnet=os.getenv("BINANCE_TESTNET_ENABLED", "false") == "true"
        )
    elif exchange == "alpaca":
        return AlpacaExecutionAdapter(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_API_SECRET"),
            paper=os.getenv("ALPACA_PAPER_TRADING", "true") == "true"
        )
    elif exchange == "interactive_brokers":
        return IBExecutionAdapter(
            host=os.getenv("IB_HOST"),
            port=int(os.getenv("IB_PORT", 7497))
        )
    else:
        return PaperExecutionAdapter()  # Fallback
```

---

## 🚀 Testing Progression

### Phase 1: Paper Trading (Current) ✅
```bash
python run_unified_trading.py  # Uses paper trading for all assets
```
- ✅ Test strategies without risk
- ✅ Verify signals and logic
- ✅ Optimize parameters

### Phase 2: Crypto Live (Testnet)
```bash
# Binance Testnet - fake money, real API
BINANCE_TESTNET_ENABLED=true python run_unified_trading.py --mode testnet
```
- ✅ Test Binance API integration
- ✅ Verify order execution
- ✅ Test real exchange connectivity

### Phase 3: Crypto Live (Limited)
```bash
# Real money but small amounts
python run_unified_trading.py --mode live_limited --confirm
```
- ✅ Real trading with $100 limit
- ✅ Test full production setup
- ✅ Monitor safety controls

### Phase 4: Full Multi-Asset (Progressive)
```bash
# All assets live with risk controls
python run_unified_trading.py --mode live_full --confirm
```
- ✅ Crypto on Binance
- ✅ Stocks on Alpaca
- ✅ Commodities on IB or OANDA

---

## ⚠️ Risk Management Checklist

Before going LIVE with real money:

- [ ] Created separate API keys for trading (not account owner keys)
- [ ] Enabled IP whitelisting on exchange
- [ ] Set API key to **read + trade** only (no withdraw)
- [ ] Tested on testnet/paper for 1+ week
- [ ] Set position size limits in `config.yaml`
- [ ] Set daily loss limits in `config.yaml`
- [ ] Monitored logs for 24 hours
- [ ] Have emergency stop plan ready
- [ ] Start with 10-20% of intended capital
- [ ] Set email alerts for large positions
- [ ] Have backup connectivity (mobile hotspot tested)

---

## 📊 Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Trading Engine                    │
│              (bot/unified_engine.py)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │   Symbol Router             │
                │  (Maps symbols → exchanges) │
                └─────────────────────────────┘
                   ↙        ↓         ↘
        ┌──────────────┐ ┌────────┐ ┌──────────────┐
        │    Crypto    │ │ Stocks │ │ Commodities  │
        │   (Binance)  │ │(Alpaca)│ │ (IB/OANDA)   │
        └──────────────┘ └────────┘ └──────────────┘
```

---

## 🎯 Recommended Approach

**Start Simple, Scale Gradually:**

1. **Week 1:** Keep all paper trading ✅
2. **Week 2:** Enable Binance testnet for crypto
3. **Week 3:** Switch crypto to live_limited mode ($100 capital)
4. **Week 4:** Add Alpaca for stocks in paper mode
5. **Week 5:** Test Alpaca on live_limited
6. **Week 6:** Add commodities (IB or OANDA) in paper mode
7. **Week 7+:** Gradually increase capital and position sizes

---

## 📞 Support Resources

- **Binance API:** https://binance-docs.github.io/apidocs/
- **Alpaca API:** https://alpaca.markets/docs/
- **Interactive Brokers:** https://www.interactivebrokers.com/en/index.php?f=5041
- **OANDA:** https://developer.oanda.com/

---

## Questions?

Check the existing adapters in `bot/execution_adapter.py` for reference implementations!
