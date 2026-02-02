# RL Shadow Mode Architecture

## Overview

This document describes the Phase 2A RL Shadow Mode implementation, which provides **advisory-only** RL outputs with **zero execution authority**.

## Non-Negotiable Safety Constraints

These constraints are **locked in code** and cannot be disabled:

| Constraint | Status | Enforcement |
|------------|--------|-------------|
| RL cannot place orders | LOCKED | No `execute`/`place_order` methods exist |
| RL cannot bypass TradeGate | LOCKED | `gate_approved=False` blocks recommendations |
| RL cannot override RiskBudgetEngine | LOCKED | `respect_risk_budget = True` (locked in `__post_init__`) |
| RL cannot override Capital Preservation | LOCKED | `respect_capital_preservation = True` (locked) |
| RL cannot adjust leverage caps | LOCKED | `respect_leverage_caps = True` (locked) |
| RL cannot trade during LOCKDOWN/CRISIS | LOCKED | Preservation level check in `get_recommendation()` |
| RL disabled by default | ENFORCED | `ShadowModeConfig(enabled=False)` default |

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL Shadow Mode Layer                         │
│                    (ADVISORY ONLY - NO EXECUTION)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │  ShadowModeConfig    │    │  RLRecommendation    │               │
│  │  ─────────────────   │    │  ─────────────────   │               │
│  │  • enabled: False    │    │  • strategy_prefs    │               │
│  │  • mode: SHADOW      │    │  • directional_bias  │               │
│  │  • max_adj: 0.1      │    │  • bias_confidence   │               │
│  │  • LOCKED:           │    │  • suggested_action  │               │
│  │    - trade_gate      │    │  • action_confidence │               │
│  │    - capital_pres    │    │  • agent_reasoning   │               │
│  │    - risk_budget     │    │  • was_applied       │               │
│  │    - leverage_caps   │    │  • actual_outcome    │               │
│  └──────────────────────┘    └──────────────────────┘               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    RLShadowAdvisor                            │   │
│  │  ─────────────────────────────────────────────────────────   │   │
│  │  • get_recommendation() → RLRecommendation                   │   │
│  │  • get_confidence_adjustment() → (float, str)                │   │
│  │  • record_outcome() → void                                   │   │
│  │                                                               │   │
│  │  SAFETY CHECKS (in get_recommendation):                      │   │
│  │  1. if not enabled → return empty recommendation             │   │
│  │  2. if LOCKDOWN/CRISIS → return hold with confidence=0       │   │
│  │  3. if gate_rejected → return hold with confidence=0         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    MetaAgent                                  │   │
│  │  ─────────────────────────────────────────────────────────   │   │
│  │  Specialized Agents (ADVISORY OUTPUT ONLY):                  │   │
│  │  • TrendFollower   → bull/bear trends                        │   │
│  │  • MeanReversion   → sideways markets                        │   │
│  │  • MomentumTrader  → breakouts                               │   │
│  │  • ShortSpecialist → bear/crash markets                      │   │
│  │  • Scalper         → quick scalping                          │   │
│  │                                                               │   │
│  │  Output: AgentAction with bounded values:                    │   │
│  │  • confidence: [0, 1]                                        │   │
│  │  • position_size_pct: [0, 0.25]                              │   │
│  │  • leverage: [1, 5]                                          │   │
│  │  • stop_loss_pct: [0, 0.1]                                   │   │
│  │  • take_profit_pct: [0, 0.2]                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ ADVISORY OUTPUT ONLY
                               │ (No execution path exists)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Existing Safety Stack                           │
│                    (RL CANNOT BYPASS THESE)                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ TradeGate   │   │ Capital     │   │ Risk Budget │               │
│  │             │   │ Preservation│   │ Engine      │               │
│  │ Gate must   │   │ LOCKDOWN    │   │ Position    │               │
│  │ approve     │   │ blocks all  │   │ limits      │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Execution Layer                                 │
│                   (RL HAS NO ACCESS HERE)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ExecutionAdapter → Exchange API                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Operating Modes

### 1. DISABLED (Default)
- RL generates no recommendations
- System behaves identically to pre-RL baseline
- `is_rl_enabled()` returns `False`

### 2. SHADOW
- RL generates recommendations
- Recommendations are **logged only**
- `was_applied = False` always
- No influence on trading decisions
- Used for counterfactual analysis

### 3. ADVISORY
- RL generates recommendations
- Recommendations can adjust confidence **up to ±10%**
- Cannot override TradeGate rejections
- Cannot trade in LOCKDOWN/CRISIS
- All adjustments bounded by `max_confidence_adjustment`

## Data Flow

```
Market State
    │
    ▼
RLShadowAdvisor.get_recommendation()
    │
    ├──► Safety Checks (LOCKDOWN? gate_approved?)
    │
    ├──► MetaAgent.get_combined_action()
    │         │
    │         ├──► TrendFollower.get_action()
    │         ├──► MeanReversion.get_action()
    │         ├──► MomentumTrader.get_action()
    │         ├──► ShortSpecialist.get_action()
    │         └──► Scalper.get_action()
    │
    ├──► Build RLRecommendation
    │
    ├──► Log to data/rl/shadow_log.jsonl
    │
    └──► Store in LearningDatabase (rl_recommendations table)
    │
    ▼
RLRecommendation (ADVISORY OUTPUT)
    │
    ├──► strategy_preferences: {"trend_follower": 0.4, ...}
    ├──► directional_bias: "long" | "short" | "neutral"
    ├──► bias_confidence: 0.0 - 1.0
    ├──► suggested_action: "hold" | "buy" | "sell" | ...
    └──► action_confidence: 0.0 - 1.0
```

## Counterfactual Analysis

RL recommendations are stored with actual outcomes for analysis:

```sql
SELECT
    suggested_action,
    actual_action,
    action_confidence,
    actual_pnl,
    CASE WHEN suggested_action = actual_action THEN 'followed' ELSE 'ignored' END as followed
FROM rl_recommendations
WHERE actual_pnl IS NOT NULL;
```

### Analysis Queries

**Agreement Rate:**
```sql
SELECT
    COUNT(CASE WHEN suggested_action = actual_action THEN 1 END) * 1.0 / COUNT(*) as agreement_rate
FROM rl_recommendations
WHERE actual_pnl IS NOT NULL;
```

**Counterfactual P&L (if RL had been followed):**
```sql
SELECT
    AVG(CASE WHEN suggested_action = actual_action THEN actual_pnl ELSE -actual_pnl END) as counterfactual_avg_pnl
FROM rl_recommendations
WHERE actual_pnl IS NOT NULL;
```

## Test Coverage

40 tests verify:

1. **Safety constraints locked** - Cannot disable safety overrides
2. **Default disabled** - RL off by default
3. **Outputs bounded** - All values in valid ranges
4. **No execution path** - No `execute()`, `place_order()`, etc.
5. **LOCKDOWN blocks** - Returns hold with confidence=0
6. **TradeGate respected** - Rejected gate blocks recommendations
7. **Meta-agent stability** - Weights normalized, outputs bounded
8. **Logging works** - Recommendations logged to JSONL
9. **Singleton behavior** - Proper state management

## Phase 2 Gating Rules

| Phase | Status | Requirements |
|-------|--------|--------------|
| 2A (Shadow Mode) | ALLOWED | RL advisory only, no execution |
| 2B (Advisory Influence) | BLOCKED | Requires B1, B2, B3 complete |
| 2C (Any Execution) | BLOCKED | Requires 2B + extended validation |

### Requirements for Phase 2B

- [ ] B1: Extended backtest (6+ months, realism ON)
- [ ] B2: Size-aware slippage model
- [ ] B3: Capital Preservation monitoring checklist

### Requirements for Phase 2C

- [ ] 2B complete
- [ ] 3+ months shadow mode data
- [ ] Counterfactual analysis shows positive edge
- [ ] Human review of RL decisions

## File Locations

| Component | Path |
|-----------|------|
| Shadow Advisor | `bot/rl/shadow_advisor.py` |
| Multi-Agent System | `bot/rl/multi_agent_system.py` |
| Reward Shaping | `bot/rl/reward_shaping.py` |
| Shadow Log | `data/rl/shadow_log.jsonl` |
| Meta Agent State | `data/rl/meta_agent_state.json` |
| Tests | `tests/test_rl_shadow_mode.py` |

## Safety Audit Checklist

Before enabling any RL influence:

- [ ] All 40 shadow mode tests pass
- [ ] No `execute`/`order` methods exist in RL code
- [ ] Safety constraints locked in `__post_init__`
- [ ] LOCKDOWN/CRISIS blocks verified
- [ ] TradeGate rejection blocks verified
- [ ] Confidence adjustments bounded
- [ ] All agent outputs bounded
- [ ] Logging captures all recommendations
- [ ] Counterfactual analysis implemented
