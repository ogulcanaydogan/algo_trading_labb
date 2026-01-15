# Algo Trading Lab - Specification Document

> **This is the single source of truth for the project. Any changes must keep it consistent.**
> **If changes require updating SPEC.md, update it first before implementing.**

---

## 1. Project Goal

Build an AI-powered algorithmic trading system that achieves **1% daily gains** with strict risk management (max 2% daily loss). The system operates in paper trading mode with the capability to transition to live trading.

### Non-Goals (What Must NOT Be Built)
- No high-frequency trading (HFT) - minimum 1 hour between trades
- No uncapped leverage - max 2x leverage when enabled
- No trading without stop-loss - every position must have defined exit
- No "gambling" features - all decisions must be data-driven
- No direct exchange integration without paper testing first

---

## 2. Core Rules (MUST Follow)

### 2.1 Risk Management Rules
| Rule | Value | Enforcement |
|------|-------|-------------|
| Daily Loss Limit | -2% | Auto-pause trading |
| Daily Profit Target | +1% | Optional auto-pause |
| Max Position Size | 5% of portfolio | SafetyController |
| Stop Loss Required | Yes, always | Order validation |
| Max Open Positions | 3 concurrent | Engine limit |

### 2.2 API Rules
- Response shape must **never change** unless versioned
- All endpoints must return JSON with consistent structure
- Error responses must include `{"error": "message"}` format
- Success responses must not include `error` key

### 2.3 Code Rules
- No refactoring unless explicitly requested
- Keep diff minimal - one feature per change
- All new features must have corresponding tests
- Configuration via environment variables or JSON files
- No hardcoded secrets - use `.env` files

---

## 3. Folder Structure

```
algo_trading_lab/
├── api/                          # FastAPI REST endpoints
│   ├── api.py                    # Main API application
│   ├── schemas.py                # Pydantic request/response models
│   ├── security.py               # Auth and rate limiting
│   ├── unified_trading_api.py    # Unified trading endpoints
│   └── dashboard_unified.html    # Web dashboard
│
├── bot/                          # Core trading logic
│   ├── unified_engine.py         # Main trading engine (CRITICAL)
│   ├── ai_trading_brain.py       # AI decision making
│   ├── unified_state.py          # State management
│   ├── safety_controller.py      # Risk controls
│   ├── trade_alerts.py           # Telegram/Discord alerts
│   ├── ml_performance_tracker.py # ML model tracking
│   │
│   ├── ml/                       # Machine learning models
│   │   ├── predictor.py          # ML predictions
│   │   ├── feature_engineer.py   # Feature generation
│   │   └── lstm_model.py         # Deep learning
│   │
│   ├── rl/                       # Reinforcement learning
│   │   ├── ppo_trainer.py        # PPO algorithm
│   │   └── position_sizer.py     # RL position sizing
│   │
│   ├── llm/                      # LLM integrations
│   │   ├── advisor.py            # AI advisor
│   │   └── prompts.py            # Prompt templates
│   │
│   └── strategies/               # Trading strategies
│       └── base_strategy.py      # Strategy interface
│
├── scripts/                      # Executable scripts
│   ├── trading/                  # Trading runners
│   │   ├── run_live_paper_trading.py
│   │   ├── run_stock_trading.py
│   │   └── run_commodity_trading.py
│   │
│   └── ml/                       # ML training scripts
│       ├── train_ai_brain.py     # Pattern learner training
│       └── train_dl_models.py    # Deep learning training
│
├── data/                         # Persistent data (gitignored)
│   ├── live_paper_trading/       # Paper trading state
│   ├── models/                   # Trained ML models
│   └── cache/                    # Market data cache
│
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest fixtures
│   ├── test_*.py                 # Test modules
│   └── smoke_test.py             # System smoke test
│
├── SPEC.md                       # This file - source of truth
├── DECISIONS.md                  # Architectural decisions log
├── API_CONTRACTS.md              # API documentation
└── requirements.txt              # Python dependencies
```

---

## 4. Module Boundaries

### 4.1 Core Modules (DO NOT MODIFY without approval)

| Module | Purpose | Exports |
|--------|---------|---------|
| `unified_engine.py` | Main trading loop | `UnifiedTradingEngine`, `EngineConfig` |
| `unified_state.py` | State persistence | `UnifiedState`, `PositionState`, `TradeRecord` |
| `safety_controller.py` | Risk enforcement | `SafetyController` |

### 4.2 Feature Modules (Can extend, not refactor)

| Module | Purpose | Extension Points |
|--------|---------|------------------|
| `ai_trading_brain.py` | AI decisions | Add new analyzers |
| `ml_signal_generator.py` | ML signals | Add new model types |
| `trade_alerts.py` | Notifications | Add new alert types |

### 4.3 Module Import Rules

```python
# ALLOWED: Import from parent to child
from bot.unified_engine import UnifiedTradingEngine
from bot.ml.predictor import MLPredictor

# NOT ALLOWED: Circular imports
# bot/ml/predictor.py should NOT import from bot/unified_engine.py

# ALLOWED: Import interfaces/types
from bot.unified_state import PositionState  # Data class, OK

# NOT ALLOWED: Deep internal imports
# from bot.unified_engine import _private_function  # Never do this
```

---

## 5. State Model

### 5.1 Trading State (`UnifiedState`)

```python
@dataclass
class UnifiedState:
    mode: TradingMode          # PAPER_LIVE_DATA | PAPER_HISTORICAL | LIVE
    status: TradingStatus      # RUNNING | PAUSED | STOPPED
    current_balance: float     # Current portfolio value
    initial_balance: float     # Starting balance
    positions: Dict[str, PositionState]  # Open positions
    last_update: str           # ISO timestamp
```

### 5.2 Position State (`PositionState`)

```python
@dataclass
class PositionState:
    symbol: str
    quantity: float
    entry_price: float
    side: str                  # "long" | "short"
    entry_time: str            # ISO timestamp
    stop_loss: Optional[float]
    take_profit: Optional[float]
    current_price: Optional[float]
```

### 5.3 Database Schema

**SQLite: `portfolio.db`**
```sql
-- trades table
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_reason TEXT,
    mode TEXT NOT NULL
);

-- equity_history table
CREATE TABLE equity_history (
    timestamp TEXT PRIMARY KEY,
    balance REAL NOT NULL,
    pnl_pct REAL
);
```

**SQLite: `ml_performance.db`**
```sql
-- predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    market_condition TEXT NOT NULL,
    actual_return REAL,
    was_correct INTEGER
);
```

---

## 6. Configuration

### 6.1 Environment Variables

```bash
# Required
API_KEY=your_api_key              # For API authentication
DATA_DIR=./data                   # Data storage directory

# Optional - Trading
INITIAL_BALANCE=10000             # Starting balance
STOP_LOSS_PCT=0.02                # 2% stop loss
TAKE_PROFIT_PCT=0.03              # 3% take profit

# Optional - Alerts
TELEGRAM_BOT_TOKEN=xxx            # Telegram bot token
TELEGRAM_CHAT_ID=xxx              # Telegram chat ID

# Optional - ML
MODEL_DIR=./data/models           # ML model directory
```

### 6.2 JSON Configuration Files

**`data/*/state.json`** - Trading state
```json
{
  "mode": "paper_live_data",
  "status": "running",
  "current_balance": 10000.0,
  "positions": {}
}
```

**`data/*/control.json`** - Trading controls
```json
{
  "trading_enabled": true,
  "max_positions": 3,
  "risk_per_trade": 0.02
}
```

**`data/risk_settings.json`** - Risk toggles
```json
{
  "shorting": false,
  "leverage": false,
  "aggressive": false
}
```

---

## 7. Event Flow

### 7.1 Trading Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING LOOP (every 60s)                  │
├─────────────────────────────────────────────────────────────┤
│  1. Check Safety (daily limits, mode validation)            │
│  2. Fetch Market Data (prices, indicators)                  │
│  3. Generate Signals (ML + Technical + AI Brain)            │
│  4. Execute Trades (if signal + safety pass)                │
│  5. Monitor Positions (stop loss, take profit, trailing)    │
│  6. Update State (persist to DB)                            │
│  7. Send Alerts (if configured)                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Signal Generation Flow

```
Market Data
    │
    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ ML Predictor  │ +  │ Technical     │ +  │ AI Brain      │
│ (XGBoost/LSTM)│    │ (RSI/MACD/BB) │    │ (Pattern)     │
└───────────────┘    └───────────────┘    └───────────────┘
    │                      │                    │
    └──────────────────────┴────────────────────┘
                           │
                           ▼
                    Signal Aggregation
                           │
                           ▼
                    ┌─────────────┐
                    │ MTF Filter  │ (Multi-timeframe)
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Safety Check│
                    └─────────────┘
                           │
                           ▼
                     FINAL SIGNAL
```

---

## 8. Constraints

### 8.1 Runtime Constraints
- Python 3.10+
- SQLite for persistence (no external DB required)
- Maximum 100 API requests/minute (rate limited)
- Memory limit: 2GB recommended

### 8.2 Trading Constraints
- Minimum trade interval: 1 hour
- Maximum leverage: 2x
- Maximum position size: 5% of portfolio
- Required stop loss on all trades

### 8.3 Version Constraints
- FastAPI >= 0.100.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- torch >= 2.0.0 (optional, for deep learning)

---

## 9. Change Protocol

### Before Making Changes

1. **Read this SPEC.md** - Understand current architecture
2. **Check DECISIONS.md** - Review past decisions
3. **Plan the change** - List files to modify
4. **Verify compatibility** - Ensure no breaking changes

### When Making Changes

```markdown
## Change Checklist
- [ ] Does this change align with project goals?
- [ ] Does this maintain backward compatibility?
- [ ] Are all API response shapes preserved?
- [ ] Is the change isolated (one feature)?
- [ ] Are tests added/updated?
- [ ] Is DECISIONS.md updated if architectural?
```

### After Making Changes

1. Run smoke tests: `python tests/smoke_test.py`
2. Run unit tests: `pytest tests/`
3. Verify API contracts: `curl http://localhost:8000/health`
4. Update documentation if needed

---

## 10. Feature Flags

All risky features must be toggleable:

| Flag | Default | Description |
|------|---------|-------------|
| `shorting` | `false` | Enable short selling |
| `leverage` | `false` | Enable leveraged trading |
| `aggressive` | `false` | Enable aggressive strategies |
| `auto_pause` | `true` | Auto-pause on limits |
| `telegram_alerts` | `false` | Send Telegram notifications |

Flags are stored in `data/risk_settings.json` and controlled via dashboard toggles.

---

*Last Updated: 2026-01-15*
*Version: 1.0.0*
