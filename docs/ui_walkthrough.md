# Dashboard Walkthrough

This walkthrough shows exactly what appears on the `/dashboard/preview` route so you can picture the interface before running the stack.

![Dashboard mockup](dashboard_preview.svg)

## Hero & Status Strip
- **Header** – Dark gradient background with "Algo Trading Lab" title and environment badge.
- **Bot Status Cards** – Responsive grid showing symbol, mode (paper/live), balance, PnL, son sinyal ve bu sinyalin teknik mi yoksa AI override’ı mı olduğunu belirten açıklayıcı satırlar. Her kart neon cyan vurgularla cam etkili panellerde sunulur.

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

## Live Action Analytics
- **Confidence Trace** – Line chart that overlays teknik sinyal güveni, yürütülen karar güveni ve AI olasılığını 20 döngü boyunca karşılaştırır.
- **AI Move vs Unrealized PnL** – Bars show expected move direction/magnitude while a dashed marker izler mevcut unrealized PnL’i; aynı panel AI ve teknik güven çizgilerini birlikte verir.
- **Commodity Bias & Confidence** – Horizontal heatmap, playbook’taki emtiaların makro bias’ını (sağ = long, sol = short) ve güven seviyesini (opaklık) görselleştirir.

## Macro & News Pulse
- **Macro Bias Cards** – Display macro score, confidence, and leading catalysts (e.g., "Fed path dovish") with amber highlights when caution is warranted.
- **Events Timeline** – A vertical list that surfaces Trump/Fed events, impact score, derived bias ve yenileme zamanını gösterir; bot döngüsü her çalıştığında yeni şablonlar refresh edilir, bu yüzden tek bir manşet tekrar etmez.

## Multi-Market Portfolio Playbook
- **Dual Columns** – Left column covers crypto & commodities (BTC, ETH, gold, silver, oil); right column tracks mega-cap equities (AAPL, MSFT, AMZN, GOOG, TSLA, NVDA).
- **Horizon Rows** – Each card lists short (1m), medium (15m), and long (1h) horizons with signed return %, Sharpe, win rate, trade count, and macro bias so you can compare execution windows at a glance.
- **Macro & Notes** – Compact text blocks summarise rate outlook, political risk, and key takeaways (e.g., "best horizon" vs. "most pressured horizon") so discretionary traders can layer human judgement on top.

## Recent Signals & Equity
- **Signals Table** – Glowing rows with timestamp, executed decision, teknik baz çizgisi, AI hareketi ve yürütme sebebi ayrı sütunlarda gösterilir.
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
