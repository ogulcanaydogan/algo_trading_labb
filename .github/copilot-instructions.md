# Copilot Instructions for Algo Trading Lab

## System Architecture

This is an **AI-powered algorithmic trading system** targeting 1% daily gains with strict risk controls. The system supports progressive mode transitions: `paper_live_data` → `testnet` → `live_limited` → `live_full`.

### Core Components (DO NOT REFACTOR)

- **UnifiedTradingEngine** (`bot/unified_engine.py`): Central orchestrator managing all trading operations across modes
- **UnifiedState** (`bot/unified_state.py`): State persistence layer using SQLite for trades/equity + JSON for live state
- **SafetyController** (`bot/safety_controller.py`): Enforces risk limits (2% daily loss, 5% position size max)
- **AITradingBrain** (`bot/ai_trading_brain.py`): Self-learning pattern recognition and strategy generation

### Module Dependencies

```
UnifiedTradingEngine
  ├─> SafetyController (risk checks)
  ├─> ExecutionAdapter (ccxt/paper exchange)
  ├─> MLSignalGenerator (XGBoost/LSTM predictions)
  ├─> AITradingBrain (pattern learning & strategy selection)
  └─> UnifiedStateStore (persistence)
```

**Import Rule**: Parent modules can import children. Children MUST NOT import parents (prevents circular dependencies).

## Critical Developer Workflows

### Running the System

```bash
# Paper trading (default, safe)
python run_unified_trading.py

# Live trading (requires --confirm flag)
python run_unified_trading.py --mode live_limited --confirm

# Check status
python run_unified_trading.py status

# Check if ready for mode transition
python run_unified_trading.py check-transition testnet
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test types
pytest tests/ -m unit
pytest tests/ -m integration

# Run with coverage
pytest --cov=bot --cov=api --cov-report=html

# Smoke test (validates system integrity)
python tests/smoke_test.py
```

### Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv/bin/activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Optional ML/LLM features
pip install -e ".[ml,llm,dev]"

# Setup environment
cp .env.example .env
# Edit .env with API keys

# Start API server (for dashboard)
uvicorn api.api:app --reload
```

### Docker Deployment

```bash
# Start all services
docker-compose up --build

# Start specific service
docker-compose up api

# View logs
docker-compose logs -f orchestrator
```

## Configuration Management

### Priority Order
1. Environment variables (`.env`)
2. `config.yaml` values
3. Code defaults

### Key Config Files

- **`.env`**: Secrets (API keys, Telegram tokens) - NEVER commit
- **`config.yaml`**: System-wide settings (symbols, risk params, trading hours)
- **`data/risk_settings.json`**: Runtime toggles (shorting, leverage, aggressive mode)
- **`data/unified_trading/state.json`**: Live trading state
- **`data/unified_trading/control.json`**: Trading controls (pause/resume)

## Project-Specific Conventions

### State Persistence Pattern

```python
# Always use UnifiedStateStore for state operations
from bot.unified_state import UnifiedStateStore

store = UnifiedStateStore()
state = store.initialize(mode, capital, resume=True)  # resume=True loads existing
store.save_state(state)  # Saves JSON + inserts to SQLite
```

### Trading Mode Checks

```python
# Check if trading allowed before execution
if not engine.safety_controller.can_trade(state):
    logger.warning("Trading blocked by safety controller")
    return

# Mode-specific limits
if state.mode.is_live:
    # Extra validation for live trading
    pass
```

### Signal Generation Pattern

All signal generators must return `(signal: str, confidence: float, reason: str)`:

```python
class MySignalGenerator:
    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float, str]:
        # signal: "LONG", "SHORT", "FLAT"
        # confidence: 0.0 to 1.0
        # reason: human-readable explanation
        return "LONG", 0.75, "EMA crossover + RSI confirmation"
```

### Error Handling Pattern

```python
# Trading operations must handle failures gracefully
try:
    result = await adapter.execute_order(order)
except ExchangeConnectionError:
    logger.error("Exchange unavailable, will retry next loop")
    return  # Don't crash, continue to next iteration
except InsufficientBalanceError:
    logger.warning("Insufficient balance")
    return
```

### Notification Pattern

```python
from bot.notifications import NotificationManager, Alert, AlertLevel, AlertType

nm = NotificationManager()
if nm.has_channels():
    alert = Alert(
        level=AlertLevel.WARNING,
        alert_type=AlertType.TRADE,
        title="Stop Loss Hit",
        message=f"Position closed: {symbol} at ${price}",
    )
    nm.send_alert(alert)
```

## Risk Controls (STRICT)

### Feature Flags (`data/risk_settings.json`)

```json
{
  "shorting": false,      // Default: OFF (long only)
  "leverage": false,      // Default: OFF (no margin)
  "aggressive": false     // Default: OFF (conservative strategies)
}
```

These act as **kill switches** - even if AI recommends using leverage, it won't execute if flag is OFF.

### Auto-Pause Triggers

- Daily loss ≥ 2% → Auto-pause + Telegram alert
- Daily profit ≥ 1% AND 5+ trades → Optional auto-pause
- 5 consecutive losses → Auto-pause
- 3 API errors in 5 minutes → Auto-pause

### Position Limits

- Max position size: 5% of portfolio (SafetyController enforces)
- Max open positions: 3 concurrent (engine limit)
- Stop loss: REQUIRED on every position (no exceptions)
- Min trade interval: 1 hour (prevents overtrading)

## API Contract Rules

**NEVER change response shapes without versioning.** All endpoints must maintain backward compatibility.

Example of a protected shape:
```python
# DO NOT modify this structure
{
  "mode": "paper_live_data",
  "status": "running",
  "current_balance": 10250.50,
  "positions": [...]  # Array must always exist, can be empty
}
```

If you need to add fields, append them (don't restructure). If breaking changes are needed, create `/v2/` endpoints.

## AI/ML Integration Points

### Training AI Brain

```bash
# Train pattern learner on historical data
python scripts/ml/train_ai_brain.py --symbol BTC/USDT --days 90

# Train deep learning models
python scripts/ml/train_dl_models.py
```

### Model Performance Tracking

Every ML prediction is recorded with `prediction_id`. When position closes, outcome is saved to `ml_performance.db`:

```python
# Prediction recorded during signal generation
prediction_id = ml_signal_generator.record_prediction(...)

# Outcome recorded when position closes
outcome_tracker.record_outcome(prediction_id, actual_return)
```

Query model performance via `/api/ml/model-performance` to identify which models work best per market condition.

### Strategy Backtesting

AI-generated strategies must pass backtest before activation:

```python
brain = get_ai_brain()
strategy = brain.generate_strategy(market_data)
brain.activate_strategy(strategy, require_backtest=True)  # Validates on 30d history
```

## Integration with External Services

### Telegram Notifications

Set in `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

Alerts sent on: trade execution, daily target achieved, loss limit hit, auto-pause triggered.

### Ollama (Local LLM for explanations)

```bash
OLLAMA_HOST=http://localhost:11434
AI_MODEL=qwen2.5:7b
AI_TRADING_ENABLED=true
```

Used for trade explanation generation (optional, costs no money).

### Exchange Integration (ccxt)

Supports: Binance (spot/futures), Kraken, paper trading (synthetic data).

```python
# Execution adapter abstracts exchange differences
from bot.execution_adapter import create_execution_adapter

adapter = create_execution_adapter(mode, config)
balance = await adapter.get_balance()
```

## Data Flow

### Trading Loop (60s default interval)

```
1. Safety Check → 2. Fetch Market Data → 3. Generate Signals (ML + Technical + AI) 
→ 4. Execute Trades (if signals strong + safety pass) → 5. Monitor Positions 
→ 6. Update State (JSON + SQLite) → 7. Send Alerts
```

### Multi-Timeframe Analysis

Signals are confirmed against higher timeframes (4h, 1d) before execution. Counter-trend trades get confidence penalty.

## Common Pitfalls

1. **Don't disable safety checks** - SafetyController is the last line of defense
2. **Don't modify core state structures** - Breaking state format corrupts history
3. **Don't hardcode secrets** - Always use environment variables
4. **Don't skip backtesting** - Always validate strategies before live use
5. **Don't trust AI blindly** - Track prediction accuracy via `ml_performance.db`

## Debugging Tips

```bash
# Check logs
tail -f data/unified_trading/logs/paper_live_data_*.log

# Inspect state
cat data/unified_trading/state.json | jq

# Check database
sqlite3 data/unified_trading/portfolio.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;"

# API health
curl http://localhost:8000/health
```

## Where to Find Things

- **Entry point**: `run_unified_trading.py`
- **Core logic**: `bot/unified_engine.py`
- **Strategy logic**: `bot/ai_trading_brain.py`
- **Risk enforcement**: `bot/safety_controller.py`
- **API endpoints**: `api/api.py`, `api/unified_trading_api.py`
- **Documentation**: `SPEC.md` (source of truth), `DECISIONS.md` (architecture log), `API_CONTRACTS.md`
- **Examples**: `scripts/trading/` (runnable scenarios)

## References

- [SPEC.md](../SPEC.md): Architecture source of truth
- [DECISIONS.md](../DECISIONS.md): Why decisions were made
- [API_CONTRACTS.md](../API_CONTRACTS.md): API response contracts
- [README.md](../README.md): User-facing documentation
