# Algo Trading Lab - System Architecture

## Overview

The Algo Trading Lab is a modular, multi-market algorithmic trading framework with **comprehensive AI integration** that continuously learns and improves.

**Key Features:**

- Multi-Asset Support: Crypto (Binance), Stocks/ETFs (yfinance), Commodities (yfinance)
- Multiple Trading Modes: Paper trading, testnet, and live trading
- **Multi-AI Engine**: 8 AI systems working together for continuous improvement
- **Leverage & Shorting**: AI learns optimal leverage (1-10x) and shorts bear markets
- Risk Management: Position sizing, dynamic risk engines, trailing stops
- Orchestration: Multi-market subprocess management with health monitoring

---

## System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                              ALGO TRADING LAB                                      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐ ║
║  │                         MULTI-AI ENGINE                                       │ ║
║  │                    (bot/ai_engine/orchestrator.py)                            │ ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │ ║
║  │  │  PARAMETER   │ │   STRATEGY   │ │     RL       │ │   ONLINE     │         │ ║
║  │  │  OPTIMIZER   │ │   EVOLVER    │ │    AGENT     │ │   LEARNER    │         │ ║
║  │  │  (Bayesian)  │ │  (Genetic)   │ │    (DQN)     │ │  (Adaptive)  │         │ ║
║  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘         │ ║
║  │         └────────────────┴────────────────┴────────────────┘                  │ ║
║  │                                   │                                           │ ║
║  │  ┌──────────────┐         ┌──────▼───────┐         ┌──────────────┐          │ ║
║  │  │    META      │         │     AI       │         │     LLM      │          │ ║
║  │  │  ALLOCATOR   │◄───────►│ ORCHESTRATOR │◄───────►│   ADVISOR    │          │ ║
║  │  │  (Capital)   │         │  (Combines)  │         │   (Ollama)   │          │ ║
║  │  └──────────────┘         └──────┬───────┘         └──────────────┘          │ ║
║  └───────────────────────────────────┼───────────────────────────────────────────┘ ║
║                                      │                                             ║
║  ┌───────────────────────────────────▼───────────────────────────────────────────┐ ║
║  │                      MULTI-MARKET ORCHESTRATOR                                 │ ║
║  │                   (scripts/trading/run_multi_market.py)                        │ ║
║  │                                                                                │ ║
║  │   ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐                 │ ║
║  │   │ CRYPTO BOT  │     │ COMMODITY BOT   │     │  STOCK BOT  │                 │ ║
║  │   │ BTC,ETH,SOL │     │ Gold,Oil,Silver │     │ AAPL,MSFT   │                 │ ║
║  │   └──────┬──────┘     └────────┬────────┘     └──────┬──────┘                 │ ║
║  │          └─────────────────────┼─────────────────────┘                         │ ║
║  │                                ↓                                               │ ║
║  │              ┌─────────────────────────────────┐                              │ ║
║  │              │   COMBINED PORTFOLIO STATUS     │                              │ ║
║  │              │   Total Value | P&L | Health    │                              │ ║
║  │              └─────────────────────────────────┘                              │ ║
║  └────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Multi-AI Engine

The core of the system is an **8-component AI Engine** that continuously learns and improves:

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                              AI ENGINE COMPONENTS                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  1. PARAMETER OPTIMIZER (bot/ai_engine/parameter_optimizer.py)                    ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Uses Bayesian-like optimization                                        │   ║
║     │ • Finds optimal: EMA periods, RSI thresholds, ADX levels, ATR multipliers│   ║
║     │ • Different parameters for BULL vs BEAR vs SIDEWAYS markets              │   ║
║     │ • Learns which settings work best per symbol/regime                      │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  2. STRATEGY EVOLVER (bot/ai_engine/strategy_evolver.py)                          ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Genetic Algorithm for strategy discovery                               │   ║
║     │ • Creates population of random strategies                                │   ║
║     │ • Evaluates fitness via backtesting (Sharpe ratio)                       │   ║
║     │ • Breeds top performers, mutates to explore                              │   ║
║     │ • DISCOVERS NEW STRATEGIES WITHOUT HUMAN INPUT                           │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  3. REINFORCEMENT LEARNING AGENT (bot/ai_engine/rl_agent.py)                      ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Deep Q-Network (DQN) with experience replay                            │   ║
║     │ • State: 18 features (price, indicators, position)                       │   ║
║     │ • Actions: BUY, SELL, HOLD                                               │   ║
║     │ • Reward: P&L + risk adjustment + trading costs                          │   ║
║     │ • LEARNS OPTIMAL TIMING FROM EVERY TRADE                                 │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  4. ONLINE LEARNER (bot/ai_engine/online_learner.py)                              ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Real-time strategy adaptation                                          │   ║
║     │ • Tracks strategy health (win rate, consecutive losses)                  │   ║
║     │ • Detects concept drift (market regime changes)                          │   ║
║     │ • Blocks degraded strategies automatically                               │   ║
║     │ • ADAPTS TO CHANGING MARKET CONDITIONS                                   │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  5. META-ALLOCATOR (bot/ai_engine/meta_allocator.py)                              ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Manages multiple strategies like a fund                                │   ║
║     │ • Allocates capital based on performance                                 │   ║
║     │ • Enforces diversification (min 5%, max 40% per strategy)                │   ║
║     │ • Risk budgeting (max 2% daily risk)                                     │   ║
║     │ • OPTIMIZES CAPITAL ACROSS WINNING STRATEGIES                            │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  6. LLM ADVISOR (bot/ai_trading_advisor.py)                                       ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Connects to Ollama (qwen2.5:7b)                                        │   ║
║     │ • Provides human-like reasoning                                          │   ║
║     │ • Context-aware risk warnings                                            │   ║
║     │ • Can override risky trades to HOLD                                      │   ║
║     │ • EXPLAINS WHY DECISIONS ARE MADE                                        │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  7. LEVERAGE RL AGENT (bot/ai_engine/leverage_rl_agent.py)         ★ NEW         ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Extended DQN with 11 actions (LONG/SHORT at 1x-10x leverage)          │   ║
║     │ • 35-feature state: price, momentum, volatility, margin, funding rate   │   ║
║     │ • Learns when to SHORT in bear markets                                  │   ║
║     │ • Learns optimal leverage for each market condition                     │   ║
║     │ • Liquidation-aware reward function                                     │   ║
║     │ • PROFITS IN BOTH BULL AND BEAR MARKETS                                 │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
║  8. AI LEVERAGE MANAGER (bot/ai_engine/leverage_manager.py)        ★ NEW         ║
║     ┌─────────────────────────────────────────────────────────────────────────┐   ║
║     │ • Kelly Criterion position sizing for optimal bet sizes                 │   ║
║     │ • Volatility-adjusted leverage (reduces in high vol)                    │   ║
║     │ • Liquidation protection (auto-reduces near danger zone)                │   ║
║     │ • Performance tracking per leverage level                               │   ║
║     │ • MAXIMIZES RISK-ADJUSTED RETURNS WITH LEVERAGE                         │   ║
║     └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Complete Data Flow

```
     ┌─────────────────────────────────────────────────────────────────┐
     │                     1. MARKET DATA                              │
     ├─────────────────────────────────────────────────────────────────┤
     │  BINANCE (ccxt)        YAHOO FINANCE         PAPER EXCHANGE     │
     │  └─ Crypto OHLCV       └─ Stocks/Commodities └─ Simulated       │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                  2. TECHNICAL INDICATORS                        │
     ├─────────────────────────────────────────────────────────────────┤
     │  EMA(12,26) │ RSI(14) │ ADX(14) │ MACD │ ATR │ Bollinger Bands  │
     │  Parameters can be AI-optimized per symbol/regime               │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌────────────────────────┐  ┌──────────────────┐
│  3A. SIGNAL GEN     │  │  3B. REGIME DETECTOR   │  │  3C. AI ENGINE   │
│  (bot/strategy.py)  │  │  (bot/regime/)         │  │  (6 AI Systems)  │
├─────────────────────┤  ├────────────────────────┤  ├──────────────────┤
│ Technical rules     │  │ Market classification  │  │ Combined AI:     │
│ EMA, RSI, MACD      │  │ BULL/BEAR/SIDEWAYS     │  │ • RL Agent (25%) │
│                     │  │ CRASH/HIGH_VOL         │  │ • Evolved (20%)  │
│ OUTPUT:             │  │                        │  │ • Online (15%)   │
│  signal: LONG       │  │ OUTPUT:                │  │ • LLM (10%)      │
│  confidence: 0.49   │  │  regime: strong_bull   │  │                  │
│  stop_loss: $X      │  │  confidence: 0.85      │  │ OUTPUT:          │
│                     │  │                        │  │  action: BUY     │
│                     │  │                        │  │  conf: 0.72      │
└─────────┬───────────┘  └───────────┬────────────┘  └────────┬─────────┘
          │ (30% weight)             │                        │
          └──────────────────────────┼────────────────────────┘
                                     │
                                     ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    4. AI DECISION COMBINATION                   │
     ├─────────────────────────────────────────────────────────────────┤
     │                                                                 │
     │   Technical Signal (30%)  ──┐                                   │
     │   RL Agent (25%)         ───┼──►  WEIGHTED COMBINATION          │
     │   Evolved Strategy (20%) ───┤                                   │
     │   Online Learning (15%)  ───┤     Final: BUY                    │
     │   LLM Advisor (10%)      ───┘     Confidence: 72%               │
     │                                   Position Size: 15%            │
     │                                                                 │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    5. RISK MANAGEMENT                           │
     ├─────────────────────────────────────────────────────────────────┤
     │  Position Size: AI-optimized (confidence × regime factor)       │
     │  Stop Loss: ATR × optimized multiplier                          │
     │  Take Profit: ATR × optimized multiplier                        │
     │  Daily Limit Check                                              │
     │  Max Position: 25% cap                                          │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    6. EXECUTION LAYER                           │
     ├─────────────────────────────────────────────────────────────────┤
     │    PAPER MODE         │    TESTNET        │    LIVE MODE        │
     │    (Simulated)        │    (Binance Test) │    (Real Money)     │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    7. LEARNING FEEDBACK LOOP                    │
     ├─────────────────────────────────────────────────────────────────┤
     │                                                                 │
     │   Trade Outcome (P&L, Duration, Regime)                         │
     │          │                                                      │
     │          ├──► RL Agent: Update Q-values, train DQN              │
     │          ├──► Online Learner: Update strategy health            │
     │          ├──► Meta-Allocator: Adjust strategy weights           │
     │          └──► Learning DB: Store for future optimization        │
     │                                                                 │
     │   EVERY TRADE MAKES THE SYSTEM SMARTER                          │
     │                                                                 │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    8. STATE PERSISTENCE                         │
     ├─────────────────────────────────────────────────────────────────┤
     │  data/state.json            → Current positions, P&L            │
     │  data/signals.json          → Signal history                    │
     │  data/equity_history.json   → Portfolio value over time         │
     │  data/ai_learning.db        → All AI learning data (SQLite)     │
     │  data/ai_orchestrator_state.json → AI system state              │
     └────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
     ┌─────────────────────────────────────────────────────────────────┐
     │                    9. DASHBOARD & API                           │
     ├─────────────────────────────────────────────────────────────────┤
     │  • Real-time portfolio value                                    │
     │  • Equity curve chart                                           │
     │  • Position details per market                                  │
     │  • AI decision explanations                                     │
     │  • Learning progress metrics                                    │
     └─────────────────────────────────────────────────────────────────┘
```

---

## AI Decision Weighting

The AI Orchestrator combines signals from all systems:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     AI SIGNAL COMBINATION                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  TECHNICAL SIGNAL (30%)                                              │ │
│   │  └── From EMA crossover, RSI, MACD, ADX confirmation                │ │
│   │      Example: LONG with 49% confidence                              │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  RL AGENT (25%)                                                      │ │
│   │  └── Deep Q-Network trained on trade outcomes                       │ │
│   │      Example: BUY with 65% probability                              │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  EVOLVED STRATEGY (20%)                                              │ │
│   │  └── Best matching strategy from genetic evolution                  │ │
│   │      Example: Strategy_Gen45_3 matches current conditions           │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  ONLINE LEARNING (15%)                                               │ │
│   │  └── Real-time adjustment based on recent performance               │ │
│   │      Example: +10% confidence boost (strategy healthy)              │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  LLM ADVISOR (10%)                                                   │ │
│   │  └── Context-aware reasoning, can override to HOLD                  │ │
│   │      Example: No override, agrees with signal                       │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   ═══════════════════════════════════════════════════════════════════════ │
│                                                                            │
│   FINAL DECISION: BUY                                                      │
│   CONFIDENCE: 72%                                                          │
│   POSITION SIZE: 15% (confidence × regime factor)                          │
│   REASONING: "Technical LONG + RL BUY + Evolved match + Healthy strategy" │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Continuous Learning Loop

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CONTINUOUS IMPROVEMENT CYCLE                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐    │
│    │  TRADE   │ ───► │ OUTCOME  │ ───► │  LEARN   │ ───► │ IMPROVE  │    │
│    │          │      │          │      │          │      │          │    │
│    │ BUY BTC  │      │ +1.6% P&L│      │ Update   │      │ Better   │    │
│    │ @$95,000 │      │ 4h hold  │      │ all AI   │      │ next     │    │
│    │          │      │ Bull mkt │      │ systems  │      │ decision │    │
│    └──────────┘      └──────────┘      └──────────┘      └──────────┘    │
│         │                                                      │          │
│         └──────────────────────────────────────────────────────┘          │
│                                                                            │
│   What gets updated after each trade:                                      │
│                                                                            │
│   1. RL Agent                                                              │
│      • Receives reward: +1.6% × 10 = +16 base reward                      │
│      • Updates Q-values for (state, action) pair                          │
│      • Trains neural network on experience replay                          │
│                                                                            │
│   2. Online Learner                                                        │
│      • Adds to rolling performance window                                  │
│      • Updates strategy health (win rate, avg P&L)                        │
│      • Checks for concept drift                                            │
│                                                                            │
│   3. Meta-Allocator                                                        │
│      • Updates strategy performance metrics                                │
│      • May increase allocation to winning strategy                         │
│      • Rebalances if drift exceeds threshold                              │
│                                                                            │
│   4. Learning Database                                                     │
│      • Stores trade record with all indicators                            │
│      • Used for future parameter optimization                              │
│      • Used for strategy evolution fitness                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## What Each AI System Learns

| AI System | Input | Learns | Output |
| --------- | ----- | ------ | ------ |
| **Parameter Optimizer** | Backtests across param ranges | Best EMA/RSI/ADX settings per regime | Optimized parameters |
| **Strategy Evolver** | Random strategy genes + fitness | New trading rules | Evolved strategies |
| **RL Agent** | State + Action + Reward | Optimal buy/sell timing | Action probabilities |
| **Online Learner** | Recent trade outcomes | Which conditions predict wins | Confidence adjustments |
| **Meta-Allocator** | Strategy performance | Capital distribution | Allocation weights |
| **LLM Advisor** | Market context | Risk assessment | Override decisions |
| **Leverage RL Agent** | 35 features + leverage outcome | Optimal leverage & short timing | Leverage decisions |
| **Leverage Manager** | Volatility + performance | Kelly sizing & risk limits | Position size + leverage |

---

## Strategy Evolution (Genetic Algorithm)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     GENETIC ALGORITHM PROCESS                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Generation 0: Random Population (50 strategies)                          │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ Strategy 1: RSI<30 AND EMA_cross → BUY      Fitness: 0.8            │ │
│   │ Strategy 2: MACD>0 AND ADX>25 → BUY         Fitness: 1.2            │ │
│   │ Strategy 3: BB_lower AND Volume>2x → BUY    Fitness: 0.5            │ │
│   │ ... (50 random strategies)                                          │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                           │                                                │
│                           ▼ Evaluate fitness via backtest                  │
│                                                                            │
│   Selection: Top 5 strategies (elitism)                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ Strategy 2: Fitness 1.2 ★                                           │ │
│   │ Strategy 7: Fitness 1.1                                             │ │
│   │ Strategy 15: Fitness 1.0                                            │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                           │                                                │
│                           ▼ Crossover + Mutation                           │
│                                                                            │
│   Generation N: Evolved Population                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ Strategy_Gen20_1: RSI<28 AND MACD>0.1 AND ADX>22   Fitness: 1.8 ★★  │ │
│   │ Strategy_Gen20_2: Combined best traits              Fitness: 1.6    │ │
│   │ (Better than any human-designed strategy)                           │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   Strategy Gene Encoding:                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ entry_long_genes:  [{rsi < 30, weight: 1.2}, {ema_cross, weight: 1.0}]│
│   │ entry_short_genes: [{rsi > 70, weight: 1.1}, {macd < 0, weight: 0.8}]│
│   │ exit_genes:        [{rsi > 60, weight: 1.0}, {hold > 50 bars}]      │ │
│   │ filter_genes:      [{adx > 25, weight: 1.3}]                        │ │
│   │ risk_params:       {position_size: 15%, stop_loss: 2 ATR}           │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Reinforcement Learning Agent

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     DEEP Q-LEARNING AGENT                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   STATE (18 features):                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ Price:     change_1h, change_4h, change_24h, vs_ema20, vs_ema50    │ │
│   │ Momentum:  rsi, rsi_change, macd_hist, momentum_5                   │ │
│   │ Volatility: atr_ratio, bb_position, volatility_ratio                │ │
│   │ Trend:     adx, trend_direction                                     │ │
│   │ Volume:    volume_ratio                                             │ │
│   │ Position:  current_position, position_pnl, position_duration        │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                           │                                                │
│                           ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                    DEEP Q-NETWORK                                    │ │
│   │                                                                      │ │
│   │   Input (18) → [128 neurons] → [64] → [32] → Output (3)            │ │
│   │                    ReLU         ReLU   ReLU                         │ │
│   │                                                                      │ │
│   │   Output: Q-values for each action                                  │ │
│   │   Q(HOLD) = 0.45                                                    │ │
│   │   Q(BUY)  = 0.72 ★ (selected)                                       │ │
│   │   Q(SELL) = 0.31                                                    │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                           │                                                │
│                           ▼                                                │
│   REWARD FUNCTION:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │ reward = pnl_pct × 10                    # Base P&L reward          │ │
│   │ reward += risk_adjusted_return × 5       # Sharpe-like bonus        │ │
│   │ reward -= 0.1 if traded                  # Transaction cost         │ │
│   │ reward += 0.5 if holding winner          # Patience reward          │ │
│   │ reward -= 0.5 if holding loser too long  # Cut losses penalty       │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│   LEARNING:                                                                │
│   • Experience Replay: Stores 100,000 (state, action, reward, next_state) │
│   • Target Network: Updated every 100 steps for stability                 │
│   • Epsilon-Greedy: Starts at 1.0, decays to 0.01 (exploration → exploit) │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Learning Database Schema

All learning data is persisted in SQLite:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     data/ai_learning.db                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   trades                          │ Complete trade history                 │
│   ├── timestamp, symbol, action   │ for learning                          │
│   ├── entry_price, exit_price     │                                       │
│   ├── pnl, pnl_pct               │                                        │
│   ├── regime, strategy_id         │                                       │
│   ├── indicators (JSON)           │                                       │
│   └── outcome (WIN/LOSS)          │                                       │
│                                                                            │
│   strategy_performance            │ Per-strategy metrics                   │
│   ├── strategy_id, regime         │                                       │
│   ├── win_rate, avg_pnl           │                                       │
│   ├── sharpe_ratio, max_drawdown  │                                       │
│   └── profit_factor               │                                       │
│                                                                            │
│   optimization_results            │ Parameter optimization                 │
│   ├── symbol, regime              │ results                               │
│   ├── parameters (JSON)           │                                       │
│   └── sharpe_ratio, win_rate      │                                       │
│                                                                            │
│   evolved_strategies              │ Genetic algorithm                      │
│   ├── generation, fitness         │ discoveries                           │
│   ├── genes (JSON)                │                                       │
│   └── regime, is_active           │                                       │
│                                                                            │
│   rl_experiences                  │ DQN replay buffer                      │
│   ├── state, action, reward       │                                       │
│   └── next_state, done            │                                       │
│                                                                            │
│   regime_transitions              │ Market regime                          │
│   ├── from_regime, to_regime      │ change patterns                       │
│   └── indicators_at_transition    │                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Signal Generation (bot/strategy.py)

### Technical Indicators Used

| Indicator | Default Parameters | AI-Optimized Range | Purpose |
|-----------|-------------------|-------------------|---------|
| EMA | Fast=12, Slow=26 | Fast: 5-20, Slow: 15-50 | Trend direction |
| RSI | Period=14, OB=70, OS=30 | Period: 7-21, OB: 65-80, OS: 20-35 | Momentum |
| ADX | Period=14, Threshold=20 | Period: 10-20, Threshold: 15-30 | Trend strength |
| MACD | 12, 26, 9 | Fast: 8-15, Slow: 20-30, Signal: 7-12 | Confirmation |
| ATR | Period=14 | Period: 10-20 | Volatility & stops |
| Bollinger | Period=20, Std=2 | Period: 15-25, Std: 1.5-2.5 | Range detection |

### Signal Logic

```
LONG Signal IF:
  • EMA Crossover (Fast > Slow) OR
  • Bullish Divergence (RSI hidden bullish)
  • PLUS Confirmations:
    - RSI not overbought (<70, AI-adjustable)
    - ADX confirms trend (>20, AI-adjustable)
    - Volume above MA
    - MACD histogram positive
    - +DI > -DI (directional indicator)

Parameters are continuously optimized by AI per symbol/regime.
```

---

## Regime Detection (bot/regime/)

The regime detector classifies market conditions:

| Regime | Detection Criteria | AI Adjustment |
|--------|-------------------|---------------|
| **BULL** | ADX > 25 + Price > MA | Wider stops, full position |
| **BEAR** | ADX > 25 + Price < MA | Faster signals, tighter stops |
| **CRASH** | Drawdown > 10%, Return < -5% | 50% position, emergency stops |
| **SIDEWAYS** | ADX < 20, tight Bollinger | Mean reversion, RSI-focused |
| **HIGH_VOL** | Volatility > 90th percentile | Reduced size, tight stops |

---

## Position Management

Once a position is opened, the system manages it with multiple mechanisms:

### 1. Stop Loss (ATR-Based)
- Default: Entry Price ± (ATR × 2.0)
- AI-optimized range: 1.0 - 3.0 ATR

### 2. Trailing Stop
- **Activation**: At +1% profit
- **Trail distance**: 0.5% or ATR × 1.5

### 3. Break-Even Stop
- **Activation**: At +0.5% profit
- **Moves to**: Entry + 0.05% buffer

### 4. Partial Profit Taking
| Target | Action |
|--------|--------|
| 1R (+2%) | Sell 25% of position |
| 2R (+4%) | Sell another 25% |
| 3R (+6%) | Sell another 25% |
| Remainder | Runs with trailing stop |

---

## Risk Management

### Position Sizing (AI-Enhanced)

| Method | Formula | When Used |
|--------|---------|-----------|
| **CONFIDENCE_SCALED** | Base × Confidence^1.5 | Default for AI |
| **REGIME_ADJUSTED** | Size × Regime_Factor | All trades |
| **KELLY_CRITERION** | f* = (p×b - q) / b | High-confidence |

### Regime Multipliers

| Regime | Position Size | Stop Loss |
|--------|---------------|-----------|
| BULL | 120% | Wide (2.5 ATR) |
| BEAR | 80% | Tight (1.5 ATR) |
| CRASH | 30% | Emergency (1 ATR) |
| SIDEWAYS | 70% | Medium (2 ATR) |
| HIGH_VOL | 50% | Tight (1.5 ATR) |

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ai/status` | GET | AI engine status (all 6 systems) |
| `/api/ai/advice` | POST | Get AI trading recommendation |
| `/api/ai/decision` | POST | Get full AI orchestrator decision |
| `/api/ai/learning` | GET | Learning progress and statistics |
| `/api/status` | GET | Portfolio summary |
| `/api/unified/readiness-check` | GET | System health check |
| `/dashboard` | GET | Web dashboard |

---

## Key Files

```
algo_trading_lab/
├── bot/
│   ├── bot.py                    # Main trading loop
│   ├── trading.py                # Position & order management
│   ├── strategy.py               # Technical signal generation
│   ├── ai.py                     # Rule-based AI enhancement
│   ├── ai_trading_advisor.py     # LLM advisor (Ollama)
│   ├── regime/                   # Regime detection & management
│   ├── ml/                       # ML models & training
│   │
│   └── ai_engine/                # Multi-AI Engine (8 systems)
│       ├── __init__.py           # Package exports
│       ├── orchestrator.py       # Master AI coordinator
│       ├── parameter_optimizer.py# Bayesian optimization
│       ├── strategy_evolver.py   # Genetic algorithm
│       ├── rl_agent.py           # Deep Q-Learning (basic)
│       ├── leverage_rl_agent.py  # ★ NEW: Leverage-aware DQN
│       ├── leverage_manager.py   # ★ NEW: Leverage optimization
│       ├── online_learner.py     # Real-time adaptation
│       ├── meta_allocator.py     # Capital allocation
│       └── learning_db.py        # SQLite persistence
│
├── scripts/trading/
│   ├── run_multi_market.py       # Multi-market orchestrator
│   ├── run_live_paper_trading.py # Crypto bot
│   ├── run_commodity_trading.py  # Commodity bot
│   └── run_stock_trading.py      # Stock bot
│
├── api/
│   ├── api.py                    # FastAPI endpoints
│   └── dashboard_unified.html    # Web interface
│
├── data/
│   ├── state.json                # Current bot state
│   ├── signals.json              # Signal history
│   ├── equity_history.json       # Equity curve
│   ├── ai_learning.db            # ★ NEW: AI learning database
│   └── ai_orchestrator_state.json# ★ NEW: AI system state
│
└── docs/
    ├── TRADING_SYSTEM_ARCHITECTURE.md  # This file
    └── AI_ENGINE_ARCHITECTURE.md       # Detailed AI docs
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama LLM server |
| `AI_TRADING_ENABLED` | `true` | Enable AI advisor |
| `AI_MODEL` | `qwen2.5:7b` | LLM model to use |
| `AI_TIMEOUT` | `30` | LLM request timeout (seconds) |
| `DATA_DIR` | `./data` | Data storage directory |
| `BINANCE_API_KEY` | - | Binance API key |
| `BINANCE_SECRET` | - | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | - | Telegram notifications |
| `TELEGRAM_CHAT_ID` | - | Telegram chat ID |

---

## Summary

The Algo Trading Lab is now an **8-component AI-powered decision engine** with full leverage and shorting capabilities:

| Layer | Component | Function |
| ----- | --------- | -------- |
| 1 | Technical Analysis | Rule-based signals from indicators |
| 2 | Regime Detection | Market condition classification |
| 3 | Parameter Optimizer | Finds optimal indicator settings |
| 4 | Strategy Evolver | Discovers new trading strategies |
| 5 | RL Agent | Learns optimal timing from experience |
| 6 | Online Learner | Adapts in real-time to market changes |
| 7 | Meta-Allocator | Optimizes capital across strategies |
| 8 | LLM Advisor | Provides reasoning and risk warnings |
| 9 | **Leverage RL Agent** | **Learns optimal leverage & shorting** |
| 10 | **Leverage Manager** | **Kelly sizing & liquidation protection** |

**The system continuously improves through:**

- Learning from every trade outcome
- Discovering new strategies via genetic evolution
- Optimizing parameters per market regime
- Adapting in real-time to changing conditions
- Allocating capital to best-performing approaches
- **Learning optimal leverage for each market condition**
- **Profiting in bear markets through shorting**

**All AI systems work together**, combining their insights into a unified decision that is more robust than any single approach.

---

## Leverage & Shorting Capabilities (NEW)

### Extended Action Space

The Leverage RL Agent has 11 possible actions:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVERAGE ACTION SPACE                            │
├─────────────────────────────────────────────────────────────────────┤
│  HOLD        │ Do nothing, maintain current position                │
│  LONG_1X     │ Long position at 1x leverage (spot equivalent)       │
│  LONG_3X     │ Long position at 3x leverage                         │
│  LONG_5X     │ Long position at 5x leverage                         │
│  LONG_10X    │ Long position at 10x leverage (high risk)            │
│  SHORT_1X    │ Short position at 1x leverage                        │
│  SHORT_3X    │ Short position at 3x leverage                        │
│  SHORT_5X    │ Short position at 5x leverage                        │
│  SHORT_10X   │ Short position at 10x leverage (high risk)           │
│  CLOSE       │ Close current position entirely                      │
│  REDUCE_HALF │ Reduce position by 50%                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Extended State (35 Features)

```text
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

### Leverage Reward Function

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

### Leverage Selection Flow

```text
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

### How the System Learns

1. **Every Trade Teaches**: After each trade, the system records:
   - Entry conditions (35 features)
   - Leverage used
   - Direction (long/short)
   - Outcome (P&L, duration, max adverse excursion)

2. **Reward Shaping**: The reward function is designed to:
   - Reward profitable shorts in bear markets
   - Penalize blown positions heavily
   - Encourage appropriate leverage for conditions
   - Protect from liquidation

3. **Continuous Improvement**: The more trades executed, the better the system becomes at:
   - Identifying shorting opportunities
   - Selecting appropriate leverage
   - Managing risk in leveraged positions
   - Avoiding liquidation
