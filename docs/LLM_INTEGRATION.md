# LLM Integration Guide

## Overview

This system uses **local Mistral-7B LLM** to support the trading strategy development process.

---

## Installation

### 1. Ollama Installation

```bash
# macOS
brew install ollama

# Start service
brew services start ollama

# Download Mistral model
ollama pull mistral
```

### 2. Python Dependencies

```bash
pip install requests pyyaml feedparser vaderSentiment
```

### 3. Verification

```bash
# Test LLM client
python scripts/tools/llm_client.py

# Output:
# LLM service is running!
# Response: ...
```

---

## Usage

### A) LLM in Jupyter Notebook

```python
# Import LLM client
from scripts.tools.llm_client import LLMClient

# Initialize LLM
llm = LLMClient(model="mistral")

# Ask question
answer = llm.ask("How does EMA crossover strategy work for Bitcoin?")
print(answer)
```

**Notebook:** `notebooks/strategy_research.ipynb` - Section 15

### B) News Analysis (with LLM)

```bash
# Detailed analysis with LLM
python scripts/tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.llm.json \
  --symbols "BTC/USDT,NVDA,GC=F" \
  --use-llm

# Simple analysis with VADER (without LLM)
python scripts/tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.basic.json \
  --symbols "BTC/USDT,NVDA,GC=F"
```

### C) Bot Integration

```python
# Use LLM in bot/macro.py
from scripts.tools.llm_client import get_llm_client

llm = get_llm_client()
analysis = llm.analyze_news(news_items, symbol="BTC/USDT")

# Create MacroEvent
event = MacroEvent(
    title=f"LLM Analysis for {symbol}",
    sentiment=analysis['sentiment'],
    bias=analysis['bias_score'],
    impact=analysis['impact'],
    ...
)
```

---

## LLM Functions

### 1. `ask(prompt, system_prompt, temperature)`

General purpose Q&A.

```python
answer = llm.ask(
    "How should stop-loss be set when volatility is high?",
    temperature=0.7
)
```

### 2. `analyze_news(news_items, symbol)`

Analyze news, return sentiment + impact.

```python
analysis = llm.analyze_news(
    news_items=[...],
    symbol="BTC/USDT"
)
# Returns:
# {
#   "sentiment": "bullish",
#   "impact": "high",
#   "bias_score": 0.65,
#   "confidence": 0.82,
#   "summary": "...",
#   "catalysts": ["...", "..."]
# }
```

### 3. `suggest_strategy(symbol, performance, market_conditions)`

Strategy suggestion based on backtest results.

```python
suggestion = llm.suggest_strategy(
    symbol="BTC/USDT",
    historical_performance={
        "sharpe_ratio": 0.8,
        "win_rate": 55.0,
        "max_drawdown_pct": 12.5
    },
    market_conditions={
        "volatility": "high",
        "trend": "bullish",
        "rsi": 65
    }
)
```

### 4. `optimize_parameters(symbol, current_params, performance_history)`

Interpret grid search results, suggest optimization.

```python
advice = llm.optimize_parameters(
    symbol="NVDA",
    current_params={"ema_fast": 12, "ema_slow": 26},
    performance_history=[...]  # Top 10 combinations
)
```

### 5. `explain_trade(trade_data, market_context)`

Explain a trade (why opened, why closed).

```python
explanation = llm.explain_trade(
    trade_data={
        "side": "LONG",
        "entry_price": 30500,
        "exit_price": 31200,
        "pnl_pct": 2.3,
        "exit_reason": "Take Profit"
    },
    market_context={
        "ema_fast": 30450,
        "ema_slow": 30200,
        "rsi": 68
    }
)
```

---

## Example Prompts

### Strategy Development

```
"Suggest a mean reversion strategy for BTC/USDT using RSI 14 and EMA 50.
Stop-loss 2%, take-profit 4%. Write the Python code."
```

### Parameter Optimization

```
"Write grid search code testing EMA fast from 10-30, EMA slow from 30-100.
Find the highest Sharpe ratio."
```

### Risk Management

```
"Write a dynamic risk management function that automatically reduces
risk_per_trade when volatility increases and increases it when it decreases.
Use ATR."
```

### News Analysis

```
"How do these news affect BTC/USDT?
- Fed will continue raising rates
- Bitcoin ETF approvals approaching
Bullish or bearish? Why?"
```

---

## Configuration

### Model Selection

```python
# Mistral (default)
llm = LLMClient(model="mistral")

# Alternative models
llm = LLMClient(model="phi4")
llm = LLMClient(model="llama3.1")
```

### Temperature Setting

- **0.0-0.3**: Deterministic, consistent (metric calculation, classification)
- **0.4-0.7**: Balanced (general purpose, strategy suggestions)
- **0.8-1.0**: Creative (brainstorming, new ideas)

```python
# For consistent analysis
answer = llm.ask(prompt, temperature=0.3)

# For creative suggestions
answer = llm.ask(prompt, temperature=0.9)
```

---

## Important Notes

### What LLM DOES?

- **Generates ideas** - Strategy suggestions, parameter ranges
- **Writes code** - Prototype functions, algorithm skeletons
- **Performs analysis** - Backtest results, news sentiment
- **Explains** - Trade logic, technical indicator interpretations

### What LLM DOESN'T DO?

- **Real-time buy/sell decisions** - Your code does this
- **Real market data generation** - Use ccxt/yfinance
- **Guaranteed profits** - Just a tool, final decision is yours

### Safety

- **Always validate** LLM output
- Backtest/forward test is **mandatory**
- **Testnet/paper trading** before real money
- Risk limits (max drawdown, max exposure) at **code level**

---

## Performance

### M2 Pro (16GB RAM)

| Model | Parameters | Inference Speed | RAM Usage |
|-------|------------|-----------------|-----------|
| Mistral-7B | 7B | ~1-2 seconds | 8-10 GB |
| Phi-4-mini | 3.8B | ~0.5-1 second | 4-6 GB |
| Llama 3.1 | 8B | ~2-3 seconds | 10-12 GB |

### Optimization Tips

1. **Batch processing** - Combine multiple questions in single prompt
2. **Use cache** - Save results for repeated analyses
3. **Set timeout** - 120s timeout for long prompts
4. **Lower temperature** - 0.3 for classification, 0.8 for brainstorming

---

## Fine-Tuning (Future)

### Data Collection

```python
# Save trade logs
{
  "timestamp": "2025-11-01T12:00:00Z",
  "symbol": "BTC/USDT",
  "decision": "LONG",
  "entry_price": 30500,
  "indicators": {"rsi": 32, "ema_gap": 1.2},
  "outcome": "win",
  "pnl_pct": 2.3
}
```

### Fine-Tuning Process (3-6 months later)

1. **Collect data** - At least 500-1000 trade logs
2. **Format data** - Convert to instruction-tuning format
3. **Fine-tune with LoRA** - 2-4 hours on M2 Pro
4. **Evaluate** - Compare base model vs fine-tuned
5. **Deploy** - Use personalized model

**Tools:**
- `llama.cpp` - Inference + fine-tuning
- `Axolotl` - Fine-tuning framework
- `Unsloth` - macOS optimized

---

## Troubleshooting

### LLM not responding

```bash
# Check Ollama service
brew services list | grep ollama

# Start service
brew services start ollama

# Manual start
ollama serve
```

### JSON parse error

LLM sometimes returns text outside JSON. `llm_client.py` auto-cleans:

```python
# Extract from markdown code block
if "```json" in response:
    response = response.split("```json")[1].split("```")[0]
```

### Timeout error

```python
# Increase timeout
llm = LLMClient()
llm.timeout = 180  # 3 minutes
```

### Model not found

```bash
# List available models
ollama list

# Download model
ollama pull mistral
```

---

## Resources

- **Ollama Docs**: https://ollama.ai/docs
- **Mistral AI**: https://mistral.ai/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Notebook**: `notebooks/strategy_research.ipynb`
- **LLM Client**: `scripts/tools/llm_client.py`
- **News Ingestor**: `scripts/tools/ingest_news_llm.py`

---

## Success Stories

### Example 1: Parameter Optimization

```
User: "I want to optimize EMA parameters"
LLM: "Test EMA fast from 8-20, slow from 30-80..."
→ After grid search: Sharpe 0.5 → 1.2 (+140%)
```

### Example 2: Risk Management

```
User: "My losses increase in high volatility"
LLM: "Use ATR-based dynamic stop-loss. If ATR > ma(ATR)..."
→ Max drawdown: 18% → 9% (-50%)
```

### Example 3: News Analysis

```
187 news → LLM analysis → 3 macro events
Sentiment: Bearish bias -0.54
→ Bot using SHORT bias gained +8.2% in 5 days
```

---

**Happy Trading!**

*Last Updated: 2025-11-01*
