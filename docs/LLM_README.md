# LLM Trading Assistant - Quick Start

## Completed Setup

- [x] Ollama installed
- [x] Mistral-7B model downloaded
- [x] LLM client (`scripts/tools/llm_client.py`) created
- [x] Notebook integration complete
- [x] News analysis system ready

---

## Quick Start

### 1. Test the LLM

```bash
python scripts/tools/llm_client.py
```

**Expected Output:**
```
LLM service is running!
Response: [Response from Mistral]
```

### 2. Open Notebook

```bash
jupyter notebook notebooks/strategy_research.ipynb
```

**Go to Section 15** → LLM integration

### 3. Run News Analysis

```bash
# Detailed analysis with LLM
python scripts/tools/ingest_news_llm.py \
  --feeds data/feeds.news.yml \
  --out data/macro_events.llm.json \
  --symbols "BTC/USDT,NVDA,GC=F" \
  --use-llm
```

---

## LLM Functions

| Function | Description | Notebook Cell |
|----------|-------------|---------------|
| `ask()` | General Q&A | 15-D |
| `analyze_news()` | News sentiment analysis | - |
| `suggest_strategy()` | Strategy improvement suggestion | 15-A |
| `optimize_parameters()` | Parameter optimization analysis | 15-B |
| `explain_trade()` | Trade explanation | 15-C |

---

## Usage Examples

### Strategy Analysis

```python
from scripts.tools.llm_client import LLMClient

llm = LLMClient()

# Analyze backtest results
suggestion = llm.suggest_strategy(
    symbol="BTC/USDT",
    historical_performance={
        "sharpe_ratio": 0.8,
        "win_rate": 55.0
    },
    market_conditions={
        "volatility": "high",
        "trend": "bullish"
    }
)

print(suggestion)
```

### News Analysis

```python
analysis = llm.analyze_news(
    news_items=[{"title": "Fed raised rates"}, ...],
    symbol="BTC/USDT"
)

print(f"Sentiment: {analysis['sentiment']}")
print(f"Bias: {analysis['bias_score']}")
```

### Free Question

```python
answer = llm.ask("How should stop-loss be set when volatility is high?")
print(answer)
```

---

## File Structure

```
algo_trading_lab/
├── scripts/tools/
│   ├── llm_client.py              # LLM API client
│   └── ingest_news_llm.py         # LLM-powered news analysis
├── notebooks/
│   └── strategy_research.ipynb    # Section 15: LLM integration
├── data/
│   ├── feeds.news.yml             # RSS feed sources
│   ├── macro_events.llm.json      # LLM analysis results
│   └── macro_events.basic.json    # VADER analysis results
└── docs/
    └── LLM_INTEGRATION.md         # Detailed documentation
```

---

## Configuration

### Change Model

```python
# Mistral (default, recommended)
llm = LLMClient(model="mistral")

# Smaller, faster
llm = LLMClient(model="phi4")
```

### Temperature Setting

```python
# Deterministic (classification, metrics)
llm.ask(prompt, temperature=0.3)

# Balanced (general use)
llm.ask(prompt, temperature=0.7)

# Creative (brainstorming)
llm.ask(prompt, temperature=0.9)
```

---

## Next Steps

### What You Can Do Now

1. **Run notebook** → Have LLM analyze backtest results
2. **News analysis** → Evaluate daily news with LLM
3. **Strategy development** → Parameter optimization with LLM

### Future (3-6 months later)

4. **Collect data** → Record trade logs (500-1000 trades)
5. **Fine-tune** → Customize model with your own data (LoRA)
6. **Deploy** → Personalized strategist model

---

## Important Reminders

### LLM CAN:

- Generate ideas
- Write code
- Perform analysis
- Provide explanations

### LLM CANNOT:

- Make real-time buy/sell decisions
- Generate real market data
- Guarantee profits

### Safety:

- Always backtest/forward test
- Validate with paper trading
- Risk limits at code level
- LLM is just an advisor, decision is yours

---

## Having Issues?

### LLM not responding?

```bash
brew services restart ollama
```

### Model not found?

```bash
ollama list
ollama pull mistral
```

### Getting errors

Check detailed documentation:
```bash
cat docs/LLM_INTEGRATION.md
```

---

## Resources

- **Detailed Docs**: `docs/LLM_INTEGRATION.md`
- **LLM Client**: `scripts/tools/llm_client.py`
- **Notebook**: `notebooks/strategy_research.ipynb`
- **Architecture**: `ARCHITECTURE.md`

---

**You're ready! Start developing LLM-powered trading strategies!**

```bash
# First step:
jupyter notebook notebooks/strategy_research.ipynb
```
