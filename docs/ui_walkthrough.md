# Dashboard Walkthrough

This walkthrough shows exactly what appears on the `/dashboard/preview` route so you can picture the interface before running the stack.

![Dashboard mockup](dashboard_preview.svg)

## Hero & Status Strip
- **Header** – Dark gradient background with "Algo Trading Lab" title and environment badge.
- **Bot Status Cards** – Responsive grid showing symbol, mode (paper/live), balance, PnL, last signal, and next candle countdown. Each card uses neon cyan highlights on a deep-navy glassmorphism panel.

```
┌──────────────┬──────────────┬──────────────┐
│ Symbol       │ Mode         │ Account Value│
├──────────────┼──────────────┼──────────────┤
│ BTC/USDT     │ Paper        │ $25,000.00   │
└──────────────┴──────────────┴──────────────┘
```

## Decision Playbook
- **Strategy Snapshots** – Cards showing EMA fast/slow, RSI band, risk per trade, and conviction threshold.
- **Rule Narratives** – Collapsible list summarising when the bot buys, scales out, or stands aside.

## AI Insights
- **Probability Cards** – Bright cyan gauges for long/short odds plus expected move.
- **Narrative Block** – Paragraph explaining why the AI favours one direction (e.g., "Momentum aligned with macro tailwinds").
- **Q&A Pane** – Small chat bubble showing the most recent question/answer pair from `/ai/question`.

## Macro & News Pulse
- **Macro Bias Cards** – Display macro score, confidence, and leading catalysts (e.g., "Fed path dovish") with amber highlights when caution is warranted.
- **Events Timeline** – A vertical list that surfaces Trump/Fed events, impact score, and effective window.

## Multi-Market Portfolio Playbook
- **Dual Columns** – Left column covers crypto & commodities (BTC, ETH, gold, silver, oil); right column tracks mega-cap equities (AAPL, MSFT, AMZN, GOOG, TSLA, NVDA).
- **Horizon Rows** – Each card lists short (1m), medium (15m), and long (1h) horizons with signed return %, Sharpe, win rate, trade count, and macro bias so you can compare execution windows at a glance.
- **Macro & Notes** – Compact text blocks summarise rate outlook, political risk, and key takeaways (e.g., "best horizon" vs. "most pressured horizon") so discretionary traders can layer human judgement on top.

## Recent Signals & Equity
- **Signals Table** – Glowing rows with timestamp, decision, price, reason tags.
- **Equity Sparkline** – Gradient SVG chart with tooltips to visualise account curve.
- **Position Journal** – Text chips summarising current position, entry price, stop, and take-profit.

## Assistant Form (Optional Input)
- A right-aligned form lets you send quick questions to the AI assistant. In preview mode it auto-fills an example query so you can see how responses render.

## Colour & Typography
- Dark slate background (`#020617`) with cyan/teal accents (`#38bdf8`, `#0ea5e9`).
- Uses the Inter typeface, heavy weights for key numbers, and thin uppercase captions for labels.

## How to View It Yourself
1. `uvicorn api.api:app --port 8000`
2. Visit `http://localhost:8000/dashboard/preview`
3. Resize the window—the layout is responsive down to tablet widths.

The preview route hydrates with deterministic sample data, so what you see locally matches the walkthrough above even when the trading bot is offline.
