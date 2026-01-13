# Binance Testnet Integration Guide

This guide explains the role of Binance test environment (https://testnet.binance.vision) in the Algo Trading Lab project and how it connects to different layers of the bot. The goal is to safely validate all workflows, including high-frequency strategies, before moving to real money.

## 1. Why Binance Testnet?
- **Realistic order flow:** Testnet mimics the real API surface in Spot and Futures markets. This allows the ccxt-based client in `bot/exchange.py` to be used with the same code in both paper mode and testnet mode.
- **Experimentation without risk:** API limits, order rejection reasons, and latency behavior are close to the real environment; asset prices are simulated. This allows testing strategy loops, risk controls, and AI decision logic without affecting the live environment.
- **HFT proving ground:** WebSocket streams (ccxt.pro or python-binance) and order response times can be measured on testnet, providing data for future low-latency optimizations.

## 2. Environment Variables
Add the following variables to your `.env` file to configure testnet keys:

```ini
BINANCE_TESTNET_API_KEY=xxx
BINANCE_TESTNET_SECRET=yyy
# Set to true if using perpetual/futures testnet instead of spot.
BINANCE_TESTNET_USE_FUTURES=true
# Paper mode is disabled so orders are routed to testnet.
PAPER_MODE=false
```

Additionally, you can configure leverage and position mode settings through the ccxt client for Futures. Since spot testnet has no leverage, align risk calculations with parameters like `RISK_PER_TRADE_PCT` in `.env`.

## 3. Role in exchange.py
The `bot/exchange.py` file contains two main clients:
1. **PaperExchangeClient:** Default mock client that writes orders to local state.
2. **CcxtExchangeClient:** Connects to real or testnet APIs using ccxt.

To use testnet, select `CcxtExchangeClient` and provide testnet URLs to the ccxt session like `{'options': {'defaultType': 'future'}, 'urls': {'api': binance.urls['apiTest']}}`. ccxt does this automatically with `set_sandbox_mode(True)`. Example configuration code:

```python
from ccxt import binance

client = binance({
    "apiKey": os.environ["BINANCE_TESTNET_API_KEY"],
    "secret": os.environ["BINANCE_TESTNET_SECRET"],
    "enableRateLimit": True,
})
client.set_sandbox_mode(True)
```

`bot.bot.TradingBot` receives this client during initialization and routes order submissions to testnet. Paper/testnet distinction can be made through `PAPER_MODE` and `BINANCE_TESTNET_*` flags.

## 4. Data Feed and HFT Preparation
- **WebSocket:** If targeting HFT, choose WebSocket over REST. ccxt's `watch_trades`, `watch_order_book` methods work with testnet URLs. Make the bot loop `asyncio`-based to use WebSocket events.
- **Order Book Analytics:** With testnet data, measure spread, depth, and imbalance to extend signal rules in `bot/strategy.py`.
- **Latency Measurement:** Log order submission and fill times in `bot/state.py` to measure your performance budget before going to production.

## 5. Risk Management
Risk rules (max position size, stop-loss, take-profit) should be applied even on testnet. Position sizing formulas in `bot/strategy.py` and `bot/state.py` reference testnet prices. To account for leverage effects on Futures testnet:

- Keep `LEVERAGE` variable in `.env` and adapt position size as `balance * risk_pct * leverage / price`.
- Use `markPrice` field from ccxt `fetch_positions` output for unrealized PnL reporting.

## 6. Impact on Dashboard and API Layer
- `/status`, `/signals`, `/equity` endpoints will show state derived from testnet data.
- **Macro & News Pulse** and **AI Insights** sections on the dashboard provide narratives matching real-time testnet results; allowing you to observe policy/Fed news alongside order outcomes.
- To indicate testnet mode to users, add `"exchange": "binance-testnet"` field to `state.json` and use color coding on the dashboard.

## 7. Production Transition Checklist
1. Validate position open/close scenarios, stop-loss/TP triggers on testnet.
2. Log API rate limit violations and error codes; add automatic retry/backoff as needed.
3. Template paper ↔️ testnet ↔️ real mode transitions with `Makefile` or CLI commands.
4. For security, store prod keys in tools like AWS Secrets Manager instead of `.env`.
5. Before going to production, add an approval step for leverage, risk parameters, and asset list via management panel or config file.

These steps will help you use Binance test environment as a bridge within Algo Trading Lab, maturing both strategy interfaces and high-frequency execution infrastructure before production.
