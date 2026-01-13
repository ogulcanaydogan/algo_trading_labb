# Multi-AI Trading Engine Architecture

## Overview

The AI Engine is a comprehensive system that continuously improves trading strategies through multiple AI approaches working together.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           AI ORCHESTRATOR                                      ║
║                    Coordinates All AI Systems                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                ║
║  │   PARAMETER     │  │    STRATEGY     │  │  REINFORCEMENT  │                ║
║  │   OPTIMIZER     │  │    EVOLVER      │  │   LEARNING      │                ║
║  │                 │  │                 │  │    AGENT        │                ║
║  │ Finds optimal   │  │ Discovers new   │  │ Learns optimal  │                ║
║  │ indicator       │  │ strategies via  │  │ actions from    │                ║
║  │ parameters      │  │ genetic algo    │  │ experience      │                ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                ║
║           │                    │                    │                          ║
║           └────────────────────┼────────────────────┘                          ║
║                                │                                               ║
║                                ▼                                               ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                ║
║  │    ONLINE       │  │     META        │  │      LLM        │                ║
║  │   LEARNER       │  │   ALLOCATOR     │  │    ADVISOR      │                ║
║  │                 │  │                 │  │                 │                ║
║  │ Adapts in       │  │ Allocates       │  │ Provides        │                ║
║  │ real-time       │  │ capital across  │  │ context-aware   │                ║
║  │                 │  │ strategies      │  │ reasoning       │                ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                ║
║           │                    │                    │                          ║
║           └────────────────────┼────────────────────┘                          ║
║                                │                                               ║
║                                ▼                                               ║
║                    ┌───────────────────────┐                                  ║
║                    │    UNIFIED DECISION   │                                  ║
║                    │    BUY/SELL/HOLD      │                                  ║
║                    │    + Confidence       │                                  ║
║                    │    + Position Size    │                                  ║
║                    │    + Risk Params      │                                  ║
║                    └───────────────────────┘                                  ║
║                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## AI Systems

### 1. Parameter Optimizer

**Purpose**: Finds optimal indicator parameters for each market regime.

**How it works**:
- Uses Bayesian-like optimization to search parameter space
- Tests each parameter set via backtesting
- Learns which parameters work best in bull/bear/sideways markets
- Caches best parameters per symbol/regime

**Parameters Optimized**:
```
EMA periods (fast, slow)
RSI period, overbought, oversold thresholds
ADX period and threshold
ATR multipliers for stop loss and take profit
MACD parameters (fast, slow, signal)
Bollinger Band period and standard deviation
```

**Usage**:
```python
from bot.ai_engine import get_parameter_optimizer

optimizer = get_parameter_optimizer()
best_params, score = await optimizer.optimize(
    config=OptimizationConfig(symbol="BTC/USDT", regime="bull", n_trials=50),
    backtest_fn=my_backtest_function,
)
```

---

### 2. Strategy Evolver (Genetic Algorithm)

**Purpose**: Discovers entirely new trading strategies without human input.

**How it works**:
```
Generation 0: Random strategies (population of 50)
        │
        ▼
    Evaluate fitness via backtesting
        │
        ▼
    Select top performers (elitism)
        │
        ▼
    Crossover: Combine genes from two parents
        │
        ▼
    Mutation: Randomly modify genes
        │
        ▼
Generation N: Evolved strategies with better fitness
```

**Strategy Encoding**:
```
StrategyChromosome:
├── entry_long_genes: [RSI < 30, EMA_cross_above, Volume > 1.5x]
├── entry_short_genes: [RSI > 70, EMA_cross_below]
├── exit_genes: [RSI > 60, Hold > 50 bars]
├── filter_genes: [ADX > 25, Trend = up]
└── risk_params: {position_size: 15%, stop_loss: 2 ATR}
```

**Usage**:
```python
from bot.ai_engine import get_strategy_evolver

evolver = get_strategy_evolver()
best_strategy = await evolver.run_evolution(
    n_generations=20,
    fitness_fn=my_fitness_function,  # Returns Sharpe ratio
)
```

---

### 3. Reinforcement Learning Agent

**Purpose**: Learns optimal trading actions through trial and error.

**Architecture**:
```
State (18 features)              Action Space
├── Price changes (1h, 4h, 24h)  ├── HOLD (0)
├── EMA distances               ├── BUY (1)
├── RSI, MACD, ADX              └── SELL (2)
├── Volatility features
├── Volume ratio
└── Position status

        ▼
┌─────────────────────────┐
│    Deep Q-Network       │
│    (128 → 64 → 32 → 3)  │
└─────────────────────────┘
        ▼
    Q-Values → Best Action
```

**Reward Function**:
```python
reward = pnl_pct * 10                    # Base P&L reward
reward += risk_adjusted_return * 5       # Sharpe-like bonus
reward -= 0.1 if traded else 0           # Transaction cost
reward += 0.5 if holding_winner else 0   # Patience reward
reward -= 0.5 if holding_loser else 0    # Cut losses penalty
```

**Usage**:
```python
from bot.ai_engine import get_rl_agent, State

agent = get_rl_agent()
state = State.from_market_data(indicators, position=0)
action = agent.select_action(state, training=False)
# action: 0=HOLD, 1=BUY, 2=SELL
```

---

### 4. Online Learner

**Purpose**: Adapts strategies in real-time based on recent performance.

**Features**:

**a) Strategy Health Monitoring**:
```
StrategyHealth:
├── Rolling win rate (last 50 trades)
├── Consecutive losses counter
├── Degradation detection
│   ├── 5+ consecutive losses → DEGRADED
│   ├── Win rate < 35% → DEGRADED
│   └── Avg P&L < -1% → DEGRADED
```

**b) Concept Drift Detection**:
```
Detects when market behavior changes:
- Monitors volatility, momentum, volume
- Uses statistical tests (Z-score)
- Triggers parameter re-optimization
```

**c) Outcome Pattern Learning**:
```
Records: {indicators_at_entry, outcome, pnl}
Learns: Which conditions lead to wins/losses
Adjusts: Parameters based on patterns
```

**Usage**:
```python
from bot.ai_engine import get_online_learner

learner = get_online_learner()

# After each trade
learner.on_trade_complete(
    symbol="BTC/USDT",
    strategy_id="ema_cross",
    entry_price=95000,
    exit_price=96000,
    pnl_pct=1.05,
    regime="bull",
    indicators_at_entry=indicators,
    hold_duration_mins=120,
)

# Before trading
should_trade, adjustment, reason = learner.should_trade(
    strategy_id="ema_cross",
    regime="bull",
    indicators=current_indicators,
)
```

---

### 5. Meta-Allocator

**Purpose**: Manages multiple strategies like a "fund of funds".

**Allocation Logic**:
```
Score each strategy:
├── Sharpe ratio × 2.0
├── Win rate bonus (if > 55%)
├── Drawdown penalty × 3.0
├── Profit factor bonus
├── Regime-specific performance × 1.5
└── Online learner health check

Constraints:
├── Min allocation: 5%
├── Max allocation: 40%
└── Risk budget: 2% daily
```

**Diversification**:
```
Uses Herfindahl-Hirschman Index (HHI):
- Score 1.0 = Perfectly diversified
- Score 0.0 = Concentrated in one strategy
```

**Usage**:
```python
from bot.ai_engine import get_meta_allocator

allocator = get_meta_allocator()

# Register strategies
allocator.register_strategy("ema_cross", {"sharpe_ratio": 1.5, "win_rate": 0.58})
allocator.register_strategy("momentum", {"sharpe_ratio": 1.2, "win_rate": 0.55})

# Get allocation
plan = allocator.calculate_allocation(
    current_regime="bull",
    total_capital=10000,
    risk_budget=0.02,
)
# plan.allocations: [{strategy_id, weight, reason}, ...]
```

---

### 6. LLM Advisor (Ollama)

**Purpose**: Provides human-like reasoning and context awareness.

**When LLM Overrides**:
- 24h price drop despite bullish signals
- High volatility warning
- Regime transition detected
- Unusual market conditions

**Input to LLM**:
```
Symbol: BTC/USDT
Price: $95,393
24H Change: -2.3%
Strategy Signal: LONG
Regime: strong_bull
Portfolio P&L: +7.67%
```

**Output from LLM**:
```json
{
  "action": "HOLD",
  "confidence": 0.65,
  "reasoning": "Despite bullish regime, 24h decline
               suggests waiting for confirmation",
  "risk_level": "medium",
  "warnings": ["Potential trend reversal"]
}
```

---

## Decision Combination

The AI Orchestrator combines all signals:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL WEIGHTING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Technical Signal (30%)                                        │
│   └── From EMA, RSI, MACD, ADX analysis                        │
│                                                                 │
│   RL Agent (25%)                                                │
│   └── BUY/SELL/HOLD probabilities from DQN                     │
│                                                                 │
│   Evolved Strategy (20%)                                        │
│   └── Best matching strategy from evolution                     │
│                                                                 │
│   Online Learning (15%)                                         │
│   └── Confidence adjustment based on recent performance        │
│                                                                 │
│   LLM Advisor (10%)                                             │
│   └── Can override to HOLD if risky                            │
│                                                                 │
│   FINAL = Weighted combination → Action + Confidence            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Learning Database

All learning data is persisted in SQLite:

```
ai_learning.db
├── trades: Complete trade history with outcomes
├── strategy_performance: Performance metrics per strategy
├── optimization_results: Best parameters per symbol/regime
├── evolved_strategies: Genetic algorithm discoveries
├── rl_experiences: Replay buffer for DQN training
└── regime_transitions: Market regime change patterns
```

---

## Usage Example

```python
from bot.ai_engine import get_ai_orchestrator, initialize_ai_orchestrator
from bot.ai_trading_advisor import get_advisor

# Initialize with LLM advisor
llm_advisor = get_advisor()
orchestrator = await initialize_ai_orchestrator(
    llm_advisor=llm_advisor,
    enable_rl=True,
    enable_evolution=True,
    enable_online_learning=True,
    enable_llm=True,
)

# Get AI decision
decision = await orchestrator.get_decision(
    symbol="BTC/USDT",
    indicators=current_indicators,
    current_price=95393,
    regime="bull",
    position=0,
    portfolio_value=10766,
    base_signal="LONG",
    base_confidence=0.49,
)

print(f"Action: {decision.action}")
print(f"Confidence: {decision.confidence:.0%}")
print(f"Position Size: {decision.position_size_pct:.1f}%")
print(f"Reasoning: {decision.reasoning}")

# After trade completes
await orchestrator.on_trade_complete(
    symbol="BTC/USDT",
    entry_price=95393,
    exit_price=96500,
    pnl_pct=1.16,
    indicators_at_entry=entry_indicators,
    regime="bull",
    hold_duration_mins=240,
)
```

---

## Background Tasks

### Run Parameter Optimization
```python
best_params, score = await orchestrator.run_optimization(
    symbol="BTC/USDT",
    backtest_fn=my_backtest,
    regime="bull",
    n_trials=100,
)
```

### Run Strategy Evolution
```python
best_strategy = await orchestrator.run_evolution(
    fitness_fn=lambda s: backtest_strategy(s),  # Returns Sharpe
    n_generations=50,
)
```

---

## 7. Leverage RL Agent (NEW)

**Purpose**: Learns optimal leverage and shorting strategies for maximum profit in all markets.

**Extended Action Space**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVERAGE ACTION SPACE (11 actions)                │
├─────────────────────────────────────────────────────────────────────┤
│  HOLD        │ Do nothing, maintain current position                 │
│  LONG_1X     │ Long position at 1x leverage (spot equivalent)        │
│  LONG_3X     │ Long position at 3x leverage                          │
│  LONG_5X     │ Long position at 5x leverage                          │
│  LONG_10X    │ Long position at 10x leverage (high risk)             │
│  SHORT_1X    │ Short position at 1x leverage                         │
│  SHORT_3X    │ Short position at 3x leverage                         │
│  SHORT_5X    │ Short position at 5x leverage                         │
│  SHORT_10X   │ Short position at 10x leverage (high risk)            │
│  CLOSE       │ Close current position entirely                       │
│  REDUCE_HALF │ Reduce position by 50%                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Extended State (35 features)**:
```
Price Features (6):        Momentum Features (6):
├── price_change_1h        ├── rsi
├── price_change_4h        ├── rsi_change
├── price_change_24h       ├── macd_hist
├── price_vs_ema20         ├── macd_signal_cross
├── price_vs_ema50         ├── momentum_5
└── price_vs_vwap          └── momentum_20

Volatility Features (5):   Trend Features (4):
├── atr_ratio              ├── adx
├── bb_position            ├── trend_direction
├── bb_width               ├── trend_strength
├── volatility_ratio       └── ema_alignment
└── high_volatility

Market Structure (3):      Position Features (6):
├── funding_rate           ├── current_position
├── open_interest_change   ├── current_leverage
└── long_short_ratio       ├── position_pnl
                           ├── position_duration
Risk Features (3):         ├── unrealized_pnl
├── drawdown_current       └── margin_ratio
├── consecutive_losses
└── win_rate_recent
```

**Reward Function for Leverage Trading**:
```python
reward = 0.0

# Base P&L reward (risk-adjusted by leverage)
leverage_risk_factor = 1.0 / sqrt(leverage_used)
reward += pnl_pct * 10 * leverage_risk_factor

# Sharpe-like adjustment
risk_adjusted = pnl_pct / (volatility * leverage_used)
reward += risk_adjusted * 5

# Appropriate leverage selection bonus
if low_volatility + high_leverage: reward += 0.5
if high_volatility + low_leverage: reward += 0.5

# Short selling bonus (profit in bear markets)
if is_short and pnl > 0: reward += 1.0
if is_short and pnl < -5%: reward -= 2.0  # Penalty for blown shorts

# Liquidation risk penalty
if margin_ratio > 70%: reward -= (margin_ratio - 0.7) * 10
if within 5% of liquidation: reward -= distance_penalty

# Transaction costs (higher for leverage)
reward -= 0.1 * leverage_used
```

**Usage**:
```python
from bot.ai_engine import get_leverage_rl_agent, LeverageState

# Get the agent
agent = get_leverage_rl_agent()

# Create state from market data
state = LeverageState.from_market_data(
    indicators=indicators,
    position_info={'position': 0, 'leverage': 1, 'margin_ratio': 0.2},
    risk_info={'consecutive_losses': 0, 'win_rate_recent': 0.55},
)

# Get action and analysis
action, info = agent.select_action(state, training=False)
analysis = agent.get_action_analysis(state)

print(f"Action: {analysis['action_name']}")
print(f"Leverage: {analysis['best_leverage']}x")
print(f"Direction: {'SHORT' if analysis['is_short'] else 'LONG'}")
print(f"Confidence: {analysis['confidence']:.0%}")
print(f"Recommendation: {analysis['recommendation']}")
```

---

## 8. AI Leverage Manager

**Purpose**: Integrates leverage RL with risk management and position sizing.

**Features**:
1. **Kelly Criterion Position Sizing** - Optimal bet sizing based on edge
2. **Volatility-Adjusted Leverage** - Reduces leverage in volatile markets
3. **Liquidation Protection** - Automatic position reduction near liquidation
4. **Performance Tracking** - Learns which leverage works best

**Usage**:
```python
from bot.ai_engine import get_leverage_manager

manager = get_leverage_manager()

# Get optimal leverage decision
decision = manager.calculate_optimal_leverage(
    indicators=current_indicators,
    current_position={'leverage': 3, 'margin_ratio': 0.4},
    risk_budget=0.02,  # Max 2% portfolio risk
    account_balance=10000,
    regime='bull',
)

print(f"Direction: {decision.direction}")
print(f"Leverage: {decision.recommended_leverage}x")
print(f"Position Size: {decision.position_size_pct:.1%}")
print(f"Confidence: {decision.confidence:.0%}")
print(f"Reasoning: {decision.reasoning}")
print(f"Max Loss: ${decision.risk_metrics['max_loss_usd']:.2f}")

for warning in decision.warnings:
    print(f"⚠️ {warning}")
```

**Leverage Selection Flow**:
```
Market Data → State Creation → RL Agent Analysis
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Base Leverage (RL)    │
                        │   e.g., 5x from DQN     │
                        └─────────────────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              Volatility       Trend          Confidence
               Adjust          Adjust           Adjust
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │   Adjusted Leverage     │
                        │   e.g., 3x (reduced)    │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Position Sizing       │
                        │   Kelly + Risk Budget   │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Final Decision        │
                        │   SHORT 3x @ 8% size    │
                        └─────────────────────────┘
```

---

## Files

```
bot/ai_engine/
├── __init__.py           # Package exports
├── orchestrator.py       # Main coordinator
├── parameter_optimizer.py# Bayesian optimization
├── strategy_evolver.py   # Genetic algorithm
├── rl_agent.py          # Deep Q-Learning (basic)
├── leverage_rl_agent.py # Leverage-aware DQN (NEW)
├── leverage_manager.py  # Leverage optimization (NEW)
├── online_learner.py    # Real-time adaptation
├── meta_allocator.py    # Capital allocation
└── learning_db.py       # SQLite persistence
```

---

## Requirements

```
# Core (always available)
numpy>=1.24.0
scipy>=1.11.0

# Optional (for full DQN)
torch>=2.0.0  # Falls back to numpy if not available
```

---

## Summary

The Multi-AI Trading Engine provides:

| Component | Function | Improvement Area |
|-----------|----------|-----------------|
| Parameter Optimizer | Find best params | Indicator tuning |
| Strategy Evolver | Discover strategies | Strategy creation |
| RL Agent | Learn actions | Buy/sell timing |
| **Leverage RL Agent** | **Learn leverage + shorting** | **Risk-adjusted returns** |
| **Leverage Manager** | **Optimal position sizing** | **Capital efficiency** |
| Online Learner | Adapt real-time | Market changes |
| Meta-Allocator | Manage capital | Portfolio optimization |
| LLM Advisor | Reasoning | Risk management |

All systems continuously learn and improve trading performance across all market conditions.

---

## Shorting & Leverage Capabilities

### How the System Learns to Short

1. **State Awareness**: The agent receives 35 features including trend direction, momentum, and market structure
2. **Action Space**: 4 short actions (1x, 3x, 5x, 10x) allow the agent to profit in bear markets
3. **Reward Shaping**: Positive rewards for profitable shorts, penalties for blown positions
4. **Experience Replay**: Learns from both successful and failed short trades
5. **Regime Adaptation**: Different leverage preferences in bull vs bear markets

### How Leverage is Optimized

1. **Base Selection**: RL agent proposes leverage based on learned patterns
2. **Volatility Adjustment**: Reduces leverage when volatility spikes
3. **Trend Confirmation**: Higher leverage allowed in strong trends
4. **Margin Protection**: Automatic reduction near liquidation threshold
5. **Performance Tracking**: Learns which leverage levels work best for each condition

### Training the System

```python
# The system learns through experience
for trade in historical_trades:
    # Create state from market conditions at entry
    state = LeverageState.from_market_data(trade.entry_indicators)

    # Agent selected this action
    action = trade.action  # e.g., SHORT_5X

    # Calculate reward from outcome
    reward = agent.calculate_reward(
        action=action,
        pnl_pct=trade.pnl_pct,
        leverage_used=5.0,
        volatility=trade.volatility,
        is_short=True,
        margin_ratio=trade.max_margin_ratio,
    )

    # Store and learn
    agent.remember(state, action, reward, next_state, done)
    loss = agent.train_step()
```

The more trades executed, the better the system becomes at:
- Identifying shorting opportunities
- Selecting appropriate leverage
- Managing risk in leveraged positions
- Avoiding liquidation
