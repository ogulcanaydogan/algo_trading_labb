# Binance Demo Trading Setup Instructions

## Important Change (November 2024)

Binance has **discontinued** the old Spot testnet system (`testnet.binance.vision`) and migrated to a new **Demo Trading** system.

### What Changed:

- ❌ **Old System**: `https://testnet.binance.vision` (API keys no longer work)
- ✅ **New System**: `https://demo-api.binance.com` (Demo Trading)

## How to Get New API Keys

### Step 1: Access Demo Trading
Visit: **https://demo.binance.com/en/my/settings/api-management**

### Step 2: Create New API Key
1. Click "Create API" button
2. Enter a label/description (e.g., "Algo Trading Bot")
3. Complete security verification
4. **Save both keys immediately** (Secret key shown only once)

### Step 3: Configure Permissions
Enable these permissions for the API key:
- ✅ **Enable Spot & Margin Trading**
- ✅ **Enable Reading** (default)
- ⚠️ **Do NOT enable withdrawals** (not needed for trading bot)

### Step 4: Update Your .env File
Copy the keys and update these lines in `.env`:

```bash
BINANCE_TESTNET_ENABLED=true
BINANCE_TESTNET_API_KEY=your_new_demo_api_key_here
BINANCE_TESTNET_API_SECRET=your_new_demo_secret_key_here
```

### Step 5: Test Connection
Run the test script:

```bash
source .venv/bin/activate
python test_binance_testnet.py
```

You should see:
```
✅ Client created successfully
✅ Successfully fetched 100 candles
```

## Key Differences: Demo Trading vs Old Testnet

| Feature | Old Testnet | New Demo Trading |
|---------|-------------|------------------|
| **Spot Trading** | ✅ Works | ✅ Works (production-like) |
| **Futures Trading** | ✅ Works | ✅ Works (new system) |
| **API Endpoint** | testnet.binance.vision | demo-api.binance.com |
| **Account Balance** | Fake USDT | Simulated funds |
| **Market Data** | Real-time | Real-time (same as prod) |

## Troubleshooting

### Error: "Invalid Api-Key ID" (-2008)

**Cause**: Using old testnet keys instead of new Demo Trading keys

**Solution**:
1. Go to https://demo.binance.com/en/my/settings/api-management
2. Create **new** API keys (old ones won't work)
3. Update `.env` with new keys

### Error: "IP address not whitelisted"

**Solution**: In Demo Trading API settings, ensure IP whitelist is either:
- Empty (unrestricted) - recommended for development
- Contains your current IP address

### Keys Still Not Working?

1. **Wait 5 minutes** - New keys may take time to activate
2. **Check permissions** - Ensure "Spot & Margin Trading" is enabled
3. **Regenerate keys** - Delete old key and create a fresh one
4. **Verify .env file** - No extra spaces, quotes, or special characters

## Official Documentation

- **Demo Trading Homepage**: https://demo.binance.com/
- **API Management**: https://demo.binance.com/en/my/settings/api-management
- **Binance FAQ**: https://www.binance.com/en/support/faq/detail/9be58f73e5e14338809e3b705b9687dd
- **Futures API Docs**: https://developers.binance.com/docs/derivatives/

## Alternative: Use Paper Mode (No API Keys Required)

If you don't need real testnet connectivity, you can use **Paper Mode** with synthetic data:

```bash
# In .env file:
PAPER_MODE=true
BINANCE_TESTNET_ENABLED=false
```

Then run:
```bash
python demo_dry_run_trading.py
```

This generates realistic market data without needing any exchange API keys.
