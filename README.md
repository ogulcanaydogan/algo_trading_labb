# Algo Trading Lab

Algo Trading Lab is a modular trading bot framework designed for multi-asset signal generation, risk management, and support for both paper trading and live trading.

## Features
- Python-based bot loop (EMA crossover + RSI confirmation) with JSON-based state storage.
- Synthetic data generator for paper trading mode; ready for real exchange integration via ccxt.
- FastAPI service with `/status`, `/signals`, `/equity`, `/strategy` endpoints and built-in web dashboard.
- AI layer with `/ai/prediction` (forecasting) and `/ai/question` (Q&A) endpoints, plus AI Insights section on the dashboard.
- Macro engine scoring political events (e.g., Trump decisions) and Fed rate expectations; `/macro/insights` endpoint and **Macro & News Pulse** dashboard panel track recent catalysts.
- Multi-market portfolio via `/portfolio/playbook` endpoint and **Multi-Market Portfolio Playbook** dashboard panel; covers crypto (BTC, ETH), commodities (gold, silver, oil), and mega-cap stocks (AAPL, MSFT) with short/medium/long-term strategy breakdowns and macro summaries.
- **Strategy research tool** evaluating EMA/RSI ranges, RSI thresholds, and macro bias together for code-free grid-search experiments.
- *New:* Market data layer supporting non-crypto assets (stocks, indices, gold, commodities) via optional `yfinance`.
- *New:* Portfolio-level multi-asset runner with separate risk parameters and data folders per instrument for concurrent bot loops.
- Docker + docker-compose containerization for 24/7 operation.
- Decoupled strategy and state layers for future self-supervised learning model integration.

## Directory Structure
```
algo_trading_lab/
├── bot/
│   ├── ai.py           # Heuristic AI predictor and Q&A engine
│   ├── bot.py          # Main loop and risk management
│   ├── market_data.py  # ccxt/yfinance/paper data providers
│   ├── exchange.py     # ccxt wrapper + paper-exchange mock
│   ├── research.py     # EMA/RSI parameter search with macro-aware grid search
│   ├── state.py        # JSON-based state/signals/equity storage
│   ├── strategy.py     # EMA/RSI strategy and position sizing calculations
│   ├── portfolio.py    # Multi-asset portfolio runner
│   ├── backtesting.py  # Backtest engine
│   └── trading.py      # Live trade manager
├── api/
│   ├── api.py          # FastAPI application
│   └── schemas.py      # Pydantic response schemas
├── scripts/            # All runnable scripts (organized by category)
│   ├── trading/        # Trading bots (run_multi_market.py, etc.)
│   ├── ml/             # ML training scripts
│   ├── backtest/       # Backtesting scripts
│   ├── tools/          # Utilities (optimizer, telegram setup)
│   └── demo/           # Demo and smoke tests
├── data/               # State files, sample OHLCV, and macro event sets
├── docs/               # Documentation
├── tests/              # Test files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Getting Started

### Binance Spot Testnet Setup
1. Go to https://testnet.binance.vision/ and create an API key
2. Copy the API Key and Secret Key
3. Edit the `.env` file:
   ```bash
   cp .env.example .env
   ```
4. Update testnet credentials in `.env`:
   ```bash
   BINANCE_TESTNET_ENABLED=true
   BINANCE_TESTNET_API_KEY=your_api_key_here
   BINANCE_TESTNET_API_SECRET=your_secret_key_here
   PAPER_MODE=false  # Set to false to use testnet
   ```

### Test Connection
Test your Binance testnet connection:
```bash
python test_binance_testnet.py
```

## Strategy Testing and Trading Decisions

### 1. Backtest (Historical Data Testing)
Test your strategy with historical data:

```bash
python scripts/backtest/run_backtest.py
```

With this script you can:
- Test your strategy on historical data
- View metrics like win rate, profit factor, max drawdown
- Experiment with different parameters
- Save results to a JSON file

**Example Output:**
```
============================================================
BACKTEST RESULTS
============================================================
Starting Balance: $10,000.00
Ending Balance: $11,250.00
Total P&L: $1,250.00 (12.50%)

Total Trades: 45
Winners: 28 | Losers: 17
Win Rate: 62.22%
Average Win: $120.50
Average Loss: $65.30
Profit Factor: 1.85
Max Drawdown: $450.00 (4.50%)
Sharpe Ratio: 1.42
============================================================
```

### 2. Live Trading (Testnet or Real)
Run your strategy live:

```bash
python scripts/trading/run_live_trading.py
```

**3 Mode Options:**
1. **DRY RUN**: Only logs, doesn't send real orders (safe testing)
2. **TESTNET**: Sends real orders on Binance testnet (test money)
3. **LIVE**: Trades on REAL EXCHANGE (CAUTION!)

**Recommended Workflow:**
```
1. Test strategy with backtest
   └─> Continue if win rate > 55% and Profit Factor > 1.5

2. Test with live data in DRY RUN mode (1-2 days)
   └─> Check if signals make sense

3. Test with real orders in TESTNET mode (1 week)
   └─> Verify order execution, stop loss, take profit work

4. Move to LIVE with small capital
   └─> Validate risk management

5. Full capital production
```

### 3. Portfolio Bot (Multi-Asset)
Use the portfolio runner to track non-crypto assets (stocks, ETFs, gold, indices) in the same loop.

1. Copy the example configuration:
   ```bash
   cp data/portfolio.sample.json data/portfolio.json
   ```
2. Add desired symbols to the `assets` list. The `asset_type` field accepts values like `crypto`, `equity`, `commodity`, `forex`. For Yahoo Finance data, enter the relevant ticker in `data_symbol` (`GC=F`, `^GSPC`, `AAPL`, etc.).
3. Set total capital (`portfolio_capital`) and each asset's allocation (`allocation_pct`). Empty allocations split remaining percentage equally.
4. Start the bot:
   ```bash
   python scripts/trading/run_portfolio.py --config data/portfolio.json
   ```

> **Note:** `pip install yfinance` is required to fetch stock/commodity data. The macro sensitivity engine reports catalysts separately for each asset if `macro_symbol` is defined.

### Environment Variables
1. Edit environment variables:
   ```bash
   cp .env.example .env
   # Update values in .env as needed
   ```
   - For Binance Futures or Spot testnet keys, add `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_SECRET`, and `BINANCE_TESTNET_USE_FUTURES=true/false` to `.env`. See [docs/binance_testnet_guide.md](docs/binance_testnet_guide.md) for detailed integration steps.
2. Start containers:
   ```bash
   docker-compose up --build
   ```
3. FastAPI interface runs at `http://localhost:8000/docs` by default.
4. Access the management dashboard at `http://localhost:8000`.
   - Even if the bot isn't running, view a live preview via `/dashboard/preview` (or `?demo=1` parameter).
   - The **AI Insights** section shows model-recommended actions, probability distributions, and explanations from `/ai/prediction` and `/ai/question` endpoints.
   - The **Decision Playbook** section summarizes when the bot goes LONG/SHORT and how it manages risk based on `/strategy` endpoint data.

## Dashboard Overview

![Dashboard preview](docs/dashboard_preview.svg)

The dashboard is organized as a single-page interface with the following blocks:
- **Top status bar:** Color-coded cards showing selected symbol, position, entry price, unrealized PnL, and bot operation mode.
- **Signal Stream:** Recent signals, order summaries, and brief AI prediction explanations flow chronologically on the right side.
- **Equity & Risk:** Equity curve, daily PnL bar chart, and risk parameters displayed side by side in the center. Preview mode shows sample data; live mode shows actual values from state files.
- **AI Insights:** AI action, probabilities, explanatory features (EMA gap, momentum, etc.), and brief narrative box.
- **Decision Playbook:** Explains EMA/RSI thresholds, stop-loss/take-profit examples, and position sizing formula based on live strategy configuration.
- **Macro & News Pulse:** Lists catalysts like Trump/Fed, macro bias score, rate outlook, and political risk summaries.
- **Multi-Market Portfolio Playbook:** Shows short/medium/long-term performance summaries, macro bias scores, and strategy notes for BTC, ETH, XAU, XAG, oil, and mega-cap stocks side by side.
- **Assistant form:** Send questions to the `/ai/question` endpoint and see real-time responses; preview mode includes sample questions.

The `/dashboard/preview` route renders all components with sample data, allowing you to explore and customize the interface without starting the bot. For more detailed breakdown and both SVG and ASCII layout sketches, see [docs/ui_walkthrough.md](docs/ui_walkthrough.md).

## Running the Strategy Research Tool

Use the grid search tool in `bot/research.py` to try parameter sets sensitive to Trump/Fed news. By default it generates synthetic `PaperExchangeClient` data; you can also use your own CSV.

```bash
# Synthetic data with default parameter range, 500 candles
python -m bot.research --symbol BTC/USDT --timeframe 1m --lookback 500

# Your own CSV file (timestamp,open,high,low,close,volume) with macro events
python -m bot.research \
  --csv data/sample_ohlcv.csv \
  --macro-events data/macro_events.sample.json \
  --ema-fast 5,8,12,16 \
  --ema-slow 21,26,32,40 \
  --rsi-overbought 60,65,70 \
  --rsi-oversold 20,25,30
```

When complete, the best combinations scoring Sharpe, total return, win rate, and macro bias together are listed. Transfer the `EMA`, `RSI` values from the output to `.env` or the dashboard's **Decision Playbook** panel to apply to the live bot. See `data/sample_ohlcv.csv` for an example CSV.

## What Can I Improve?
The following areas can be easily extended initially:
1. **Visual theme and brand identity:** Tailwind-inspired utility classes exist in `api/dashboard.html`; modify CSS in `<style>` blocks or add an external CSS file for your color palette.
2. **Chart libraries:** Currently using lightweight SVG charts. Add Highcharts, Plotly, or TradingView widgets for more detailed charts.
3. **Multi-instrument support:** Expand the dashboard symbol selector to display signals/equity for multiple assets simultaneously.
4. **Notifications and alerts:** Send browser notifications for new signals or critical macro events via WebSocket/Server-Sent Events.
5. **User management:** Add an auth layer on FastAPI to password-protect the dashboard.

## Long-Term Vision and AI Roadmap

For more detailed recommendations on strategy research, macro awareness, local SSL training processes, and high-frequency execution, see [docs/product_vision_and_ai_roadmap.md](docs/product_vision_and_ai_roadmap.md). This document consolidates advanced ideas including multi-market coverage, news-feed catalyst evaluation, self-supervised learning pipeline, reinforcement-based policy optimization, and operator interface improvements.

## High Frequency Trading (HFT) Roadmap
Important technical developments when approaching HFT:
1. **Low-latency data streaming:** Use Binance WebSocket (ccxt.pro or python-binance) instead of REST calls for millisecond-level price updates.
2. **Asynchronous bot loop:** Make data fetching, signal calculation, and order submission `asyncio`-based in `bot/bot.py` for concurrency across multiple assets.
3. **Order book monitoring:** Read level-2 order book data instead of just OHLCV to generate microstructure signals (spread, imbalance).
4. **Risk guardrails:** Errors grow fast in HFT; add automatic circuit breakers for latency, failed order counts, or consecutive loss limits.
5. **Performance measurement:** Track average latency, fill rate, slippage, and PnL distribution with Prometheus metrics; add real-time charts to Grafana or custom dashboard.
6. **Backtest & simulation:** Simulate HFT strategy scenarios on second/minute data with vectorbt/backtrader and compare with real environment.

These roadmap steps can be gradually integrated into the existing architecture to transform UI insights into a millisecond-scale decision support system.

## AI-Powered Prediction and Q&A
- **AI Prediction (`GET /ai/prediction`)**: Returns the AI assessment from the last loop. Response includes recommended action (`LONG`/`SHORT`/`FLAT`), confidence score, long/short/flat probabilities, expected move percentage, and a quick summary of main features used.
- **AI Question (`POST /ai/question`)**: Ask strategy-related questions with a JSON body like `{ "question": "When should I buy?" }`. The engine responds using current state and AI prediction.
- Test the same Q&A experience from the browser using the dashboard form; preview mode simulates sample responses.
- Adding keywords like `macro`, `Trump`, `Fed`, `rates` to your questions prompts the AI engine to include insights from the macro module in its response.

## Multi-Market Portfolio Playbook
- **Endpoint (`GET /portfolio/playbook`)**: Collects crypto/commodity symbols like BTC, ETH, XAU, XAG, USOIL and mega-cap stocks like AAPL, MSFT, AMZN, GOOG, TSLA, NVDA from the bot's state file. Calculates expected return, Sharpe, win rate, trade count, and macro bias values for short (1m), medium (15m), and long (1h) horizons per asset.
- **Starting balance scenario**: Requires no request body; references `STARTING_BALANCE` from bot configuration and returns starting/ending balance info for each horizon.
- **Macro narratives**: Macro summaries and driver lists from events like Trump tariffs or Fed rate path are included in the same response; dashboard cards show brief notes, rate/policy warnings, and "best / most challenged horizon" headings.
- **Long/short-term ideas**: The **Multi-Market Portfolio Playbook** dashboard panel uses JSON from this endpoint to visually show which asset stands out at which horizon, which is under pressure, and how the macro environment affects risk appetite. The preview route renders the same panel with sample data.

## Macro & News Awareness
- The bot loop evaluates macro/political event lists each round using `MacroSentimentEngine` in `bot/macro.py`. Default includes sample events like Trump tariffs and Fed meeting guidance; extend with your own events via a `data/macro_events.json` file.
- Use a JSON list to add custom events. Example structure in `data/macro_events.sample.json`:
  ```json
  [
    {
      "title": "Trump announces new tariff schedule",
      "category": "politics",
      "sentiment": "bearish",
      "impact": "high",
      "actor": "Donald Trump",
      "summary": "Tariff threats raise volatility across risk assets.",
      "assets": { "BTC/USDT": -0.2, "ETH/USDT": -0.15 }
    },
    {
      "title": "FOMC statement",
      "category": "central_bank",
      "sentiment": "dovish",
      "impact": "medium",
      "interest_rate_expectation": "Fed signals a cautious path with one cut pencilled in for Q4."
    }
  ]
  ```
- Save the file as `macro_events.json` under `DATA_DIR` accessed by the bot and point to it with `MACRO_EVENTS_PATH=data/macro_events.json` in `.env`. Change the refresh interval (default 300 sec) with `MACRO_REFRESH_SECONDS`.
- The `/macro/insights` endpoint and **Macro & News Pulse** dashboard panel present the summary macro bias score, confidence level, rate expectations, and recent catalyst list as JSON or visuals. These signals are added as weights to AI predictions, allowing news flow to strengthen or weaken LONG/SHORT decisions.

## Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)  # or use dotenv
python -m bot.bot  # starts bot loop
uvicorn api.api:app --reload
```

## Example State Output
```json
{
  "timestamp": "2025-10-28T16:32:00Z",
  "symbol": "BTC/USDT",
  "position": "LONG",
  "entry_price": 67321.5,
  "unrealized_pnl_pct": 0.42,
  "last_signal": "LONG",
  "confidence": 0.66,
  "last_signal_reason": "AI reinforcement kept the position aligned with bullish crossover.",
  "technical_signal": "LONG",
  "technical_confidence": 0.62,
  "technical_reason": "EMA fast crossed above EMA slow with RSI support.",
  "ai_override_active": false,
  "rsi": 54.2,
  "ema_fast": 67310.1,
  "ema_slow": 67190.7,
  "risk_per_trade_pct": 0.5,
  "ai_action": "LONG",
  "ai_confidence": 0.72,
  "ai_probability_long": 0.72,
  "ai_probability_short": 0.18,
  "ai_probability_flat": 0.1,
  "ai_expected_move_pct": 0.64,
  "ai_summary": "Model leans upside with 72.0% confidence driven by EMA spread 0.48% and momentum 0.35%. Expected move: 0.64%",
  "ai_features": {
    "ema_gap_pct": 0.48,
    "momentum_pct": 0.35,
    "rsi_distance_from_mid": 8.5,
    "volatility_pct": 0.62
  },
  "macro_bias": -0.18,
  "macro_confidence": 0.58,
  "macro_summary": "Macro bias is bearish (-0.18) based on 3 tracked catalysts. Fed watch: Fed likely to keep rates unchanged but watch core inflation prints. Political risk: Donald Trump: Potential tariff escalation keeps risk assets cautious.",
  "macro_drivers": [
    "Trump vows fresh tariffs review (bearish, high impact)",
    "US payrolls surprise to upside (hawkish, high impact)"
  ],
  "macro_interest_rate_outlook": "Fed likely to keep rates unchanged but watch core inflation prints.",
  "macro_political_risk": "Donald Trump: Potential tariff escalation keeps risk assets cautious.",
  "macro_events": [
    {
      "title": "Trump vows fresh tariffs review",
      "category": "politics",
      "sentiment": "bearish",
      "impact": "high",
      "actor": "Donald Trump"
    },
    {
      "title": "Fed officials guide for data-dependent path",
      "category": "central_bank",
      "impact": "medium",
      "interest_rate_expectation": "Fed likely to keep rates unchanged but watch core inflation prints."
    }
  ]
}
```

## Notes
- `requirements.txt` contains base dependencies. PyTorch and PyTorch Lightning must be installed separately for SSL/ML integration (whl files vary by platform).
- **Testnet Usage**: Set `BINANCE_TESTNET_ENABLED=true` and `PAPER_MODE=false` in `.env` to use Binance Spot Testnet.
- **Production Usage**: For real exchange trading, update `PAPER_MODE=false`, `BINANCE_TESTNET_ENABLED=false`, and `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET` fields in `.env`.
- For multi-instrument support, add new services derived from the same image in `docker-compose` or extend the bot loop to accept parameters.

## HFT Recommendations
- Use Binance Futures Testnet (more realistic): https://testnet.binancefuture.com
- Listen to order book and trade streams via WebSocket instead of REST API
- Run your server in a region close to Binance for latency optimization
- Rate limits and order matching should be tested

## Backend and Frontend Gaps
To quickly see where the project is still incomplete, check [`docs/backend_frontend_gaps.md`](docs/backend_frontend_gaps.md). This document presents concrete items to be completed on both server side (exchange integration, risk management, deployment) and interface side (component architecture, real-time data flow, accessibility) as a checklist.
