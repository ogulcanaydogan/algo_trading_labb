# Regime-Aware Strategy + Risk Engine - Technical Design Document

## 1. Overview

This document describes a modular system that:
1. **Detects market regimes** (Bull, Bear, Crash, Sideways, HighVol)
2. **Selects strategies** appropriate for each regime
3. **Enforces strict risk controls** before any order execution
4. **Provides an AI/ML upgrade path** with gated deployment

**Primary Objective**: Maximize risk-adjusted return subject to hard drawdown/tail-risk limits.

**Safety Principle**: Risk engine can block trades regardless of strategy/ML output.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRADING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Market Data  │───▶│   Regime     │───▶│  Strategy    │                   │
│  │   Adapter    │    │  Detector    │    │  Selector    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         │                   ▼                   ▼                            │
│         │           ┌──────────────┐    ┌──────────────┐                    │
│         │           │   Regime     │    │   Trade      │                    │
│         │           │   State      │    │   Signals    │                    │
│         │           └──────────────┘    └──────────────┘                    │
│         │                   │                   │                            │
│         │                   └─────────┬─────────┘                            │
│         │                             ▼                                      │
│         │                   ┌──────────────────────┐                        │
│         │                   │     RISK ENGINE      │◀── VETO POWER          │
│         │                   │  (Hard Constraints)  │                        │
│         │                   └──────────────────────┘                        │
│         │                             │                                      │
│         │                    PASS ────┼──── BLOCK                           │
│         │                             ▼                                      │
│         │                   ┌──────────────────────┐                        │
│         └──────────────────▶│  Execution Adapter   │                        │
│                             │  (Paper/Live Mode)   │                        │
│                             └──────────────────────┘                        │
│                                       │                                      │
│                                       ▼                                      │
│                             ┌──────────────────────┐                        │
│                             │   Decision Logger    │                        │
│                             │  (Audit Trail)       │                        │
│                             └──────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modules and Responsibilities

### 3.1 Regime Detector (`regime_detector.py`)

**Responsibility**: Analyze market data and classify current market regime.

**Input**: OHLCV data (required), optional: order book spread, funding rates, VIX proxy

**Output**: `RegimeState` with:
- `regime`: Enum (BULL, BEAR, CRASH, SIDEWAYS, HIGH_VOL)
- `confidence`: float [0.0, 1.0]
- `indicators`: dict of computed features
- `timestamp`: datetime

**Features Computed**:
| Feature | Description | Default Config |
|---------|-------------|----------------|
| `trend_strength` | ADX or MA slope | ADX > 25 = trending |
| `trend_direction` | Price vs MA | Above = bullish |
| `volatility` | ATR / price ratio | Normalized 0-1 |
| `realized_vol` | Rolling std of returns | 20-period |
| `drawdown` | Peak-to-trough | Rolling 50 bars |
| `range_bound` | Bollinger bandwidth | < 0.04 = ranging |

**Regime Classification Rules (v1)**:
```
CRASH:     drawdown > 10% AND volatility_spike > 2x AND return_short < -5%
HIGH_VOL:  volatility > 90th percentile (rolling)
BULL:      trend_direction > 0 AND trend_strength > threshold AND NOT crash
BEAR:      trend_direction < 0 AND trend_strength > threshold AND NOT crash
SIDEWAYS:  trend_strength < threshold AND NOT high_vol
```

### 3.2 Strategy Selector (`strategy_selector.py`)

**Responsibility**: Select and configure strategy based on regime.

**Strategy Mapping (v1)**:
| Regime | Strategy | Risk Adjustment |
|--------|----------|-----------------|
| BULL | Trend-following LONG | Normal size |
| BEAR | Trend-following SHORT or FLAT | Reduced size |
| CRASH | FLAT or bounded SHORT | Minimal size, tight stops |
| SIDEWAYS | Mean reversion | Reduced size |
| HIGH_VOL | FLAT or reduced activity | 50% size cap |

### 3.3 Risk Engine (`regime_risk_engine.py`)

**Responsibility**: Enforce hard constraints before any order. **Has absolute veto power.**

**Hard Limits (all configurable)**:
| Limit | Description | Default |
|-------|-------------|---------|
| `max_leverage_by_regime` | Per-regime leverage caps | Bull: 3x, Bear: 2x, Crash: 1x |
| `max_position_pct` | Max % of equity per position | 20% |
| `max_portfolio_heat` | Total risk exposure | 6% |
| `max_daily_loss_pct` | Daily loss circuit breaker | 3% |
| `max_drawdown_pct` | Equity curve drawdown limit | 10% |
| `max_consecutive_losses` | Loss streak cooldown | 5 |
| `min_time_between_trades` | Anti-overtrading | 60 seconds |
| `volatility_circuit_breaker` | ATR spike threshold | 3x normal |
| `spread_circuit_breaker` | Max bid-ask spread | 0.5% |

**Position Sizing Formula**:
```python
risk_amount = equity * risk_per_trade_pct
stop_distance = entry_price * atr_multiplier * atr / entry_price
position_size = risk_amount / stop_distance
position_size = min(position_size, max_position_by_regime)
position_size = min(position_size, available_margin / leverage_cap)
```

### 3.4 Execution Adapter Interface

**Required Methods** (already exists in your system):
```python
class ExecutionAdapterInterface(Protocol):
    async def get_latest_price(self, symbol: str) -> float: ...
    async def get_candles(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame: ...
    async def get_positions(self) -> List[Position]: ...
    async def get_balance(self) -> Balance: ...
    async def place_order(self, order: Order) -> OrderResult: ...
    async def cancel_order(self, order_id: str, symbol: str) -> bool: ...
    async def get_open_orders(self) -> List[Order]: ...
```

### 3.5 Decision Logger (`decision_logger.py`)

**Responsibility**: Audit trail for all decisions.

**Logged Events**:
- Regime transitions with reasons
- Trade signals generated
- Risk checks: PASS/BLOCK with specific constraint that triggered
- Orders placed/filled/rejected
- Kill switch activations

---

## 4. Data Flow

```
1. Market Data Tick/Bar arrives
         │
         ▼
2. Regime Detector computes features
         │
         ▼
3. Regime classified with confidence
         │
         ▼
4. Strategy Selector picks strategy for regime
         │
         ▼
5. Strategy generates signal (if any)
         │
         ▼
6. Risk Engine pre-trade checks:
   ├── Check kill switch status
   ├── Check circuit breakers
   ├── Check position limits
   ├── Check leverage limits
   ├── Check drawdown limits
   ├── Check daily loss limits
   ├── Calculate position size
   └── Validate stop placement
         │
         ├── BLOCK → Log reason, no order
         │
         └── PASS → Continue
                │
                ▼
7. Execution Adapter places order
         │
         ▼
8. Decision Logger records everything
         │
         ▼
9. Post-trade risk update (P&L, positions)
```

---

## 5. Configuration Format

```yaml
# regime_config.yaml

regime_detection:
  lookback_bars: 100
  timeframe: "1h"

  trend:
    ma_fast: 20
    ma_slow: 50
    adx_period: 14
    adx_threshold: 25

  volatility:
    atr_period: 14
    vol_lookback: 20
    high_vol_percentile: 90
    spike_multiplier: 2.0

  crash:
    drawdown_threshold: 0.10  # 10%
    return_threshold: -0.05   # -5% short-term return
    window: 24  # hours

  sideways:
    bb_period: 20
    bb_bandwidth_threshold: 0.04

risk_engine:
  # Per-regime limits
  regime_limits:
    BULL:
      max_leverage: 3.0
      max_position_pct: 0.25
      risk_per_trade_pct: 0.02
    BEAR:
      max_leverage: 2.0
      max_position_pct: 0.15
      risk_per_trade_pct: 0.015
    CRASH:
      max_leverage: 1.0
      max_position_pct: 0.05
      risk_per_trade_pct: 0.005
    SIDEWAYS:
      max_leverage: 2.0
      max_position_pct: 0.10
      risk_per_trade_pct: 0.01
    HIGH_VOL:
      max_leverage: 1.5
      max_position_pct: 0.10
      risk_per_trade_pct: 0.01

  # Hard limits (cannot be overridden by regime)
  hard_limits:
    max_daily_loss_pct: 0.03
    max_drawdown_pct: 0.10
    max_consecutive_losses: 5
    max_portfolio_heat: 0.06
    min_time_between_trades_sec: 60

  # Circuit breakers
  circuit_breakers:
    volatility_spike_multiplier: 3.0
    max_spread_pct: 0.005
    liquidity_min_volume: 1000

  # Kill switch conditions
  kill_switch:
    auto_trigger_drawdown: 0.15
    auto_trigger_daily_loss: 0.05
    cooldown_after_trigger_hours: 24

strategy:
  bull:
    type: "trend_following"
    direction: "long"
    entry: "ma_cross"
    ma_fast: 10
    ma_slow: 30

  bear:
    type: "trend_following"
    direction: "short"
    entry: "ma_cross"
    ma_fast: 10
    ma_slow: 30
    fallback: "flat"  # Go flat if confidence < threshold

  crash:
    type: "defensive"
    direction: "flat"
    allow_short: false

  sideways:
    type: "mean_reversion"
    bb_period: 20
    bb_std: 2.0

  high_vol:
    type: "reduced_activity"
    size_multiplier: 0.5

execution:
  slippage_bps: 5
  commission_bps: 10
  use_synthetic_stops: true

backtest:
  initial_capital: 10000
  walk_forward_splits: 5
  train_pct: 0.70
```

---

## 6. Observability / Metrics

### Metrics to Track:
```python
# Regime metrics
regime_current: Gauge  # Current regime (encoded)
regime_confidence: Gauge
regime_duration_seconds: Counter
regime_transitions_total: Counter

# Risk metrics
daily_pnl_pct: Gauge
drawdown_current_pct: Gauge
leverage_current: Gauge
portfolio_heat: Gauge
position_count: Gauge

# Trading metrics
trades_total: Counter (labels: regime, direction, outcome)
trades_blocked_total: Counter (labels: reason)
win_rate_rolling: Gauge
sharpe_rolling: Gauge

# Circuit breaker metrics
circuit_breaker_triggers: Counter (labels: type)
kill_switch_active: Gauge
```

### Log Format:
```json
{
  "timestamp": "2024-01-13T12:00:00Z",
  "event": "TRADE_BLOCKED",
  "symbol": "BTC/USDT",
  "regime": "HIGH_VOL",
  "signal": {"direction": "LONG", "confidence": 0.72},
  "block_reason": "volatility_circuit_breaker",
  "details": {
    "current_atr": 0.045,
    "normal_atr": 0.015,
    "spike_ratio": 3.0,
    "threshold": 3.0
  }
}
```

---

## 7. Failure Modes and Safety Controls

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Data feed failure | Stale price check | Block new trades, hold positions |
| Exchange API error | HTTP errors, timeouts | Retry with backoff, then block |
| Extreme volatility | ATR spike detection | Activate circuit breaker |
| Rapid drawdown | Equity curve monitor | Trigger kill switch |
| Strategy malfunction | Signal rate anomaly | Fall back to FLAT |
| ML model drift | Confidence degradation | Revert to rules-based |

### Kill Switch Behavior:
1. **Soft Kill**: Block new trades, keep existing positions
2. **Hard Kill**: Block new trades AND close all positions at market
3. **Manual Override**: Dashboard button or API call

---

## 8. Testing Strategy

### Unit Tests:
- Regime detection on known patterns (bull run, crash, ranging)
- Risk engine constraint checks (each limit individually)
- Position sizing calculations
- Circuit breaker triggers

### Integration Tests:
- End-to-end signal → risk check → execution flow
- Kill switch activation and recovery
- Regime transition handling

### Backtest Validation:
- Walk-forward out-of-sample testing
- Compare risk-adjusted metrics vs buy-and-hold
- Verify drawdown limits are respected

---

## 9. Production Readiness Checklist

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Paper trading for 2+ weeks with positive results
- [ ] Kill switch tested manually
- [ ] Circuit breakers tested with simulated spikes
- [ ] Logging verified and queryable
- [ ] Alerts configured for critical events
- [ ] Runbook documented for common issues
- [ ] Rollback procedure tested
- [ ] Capital limits set conservatively for initial live

---

## 10. AI/ML Upgrade Path

This section describes how to incrementally upgrade from rules-based to ML-enhanced regime detection and strategy selection while maintaining safety guarantees.

### 10.1 Upgrade Principles

1. **Shadow Mode First**: New models run in shadow mode before affecting trades
2. **Gated Rollout**: Gradual exposure (1% → 10% → 50% → 100%)
3. **Safety Preserved**: Risk engine ALWAYS has veto power regardless of ML output
4. **Fallback Ready**: Can instantly revert to rules-based on degradation

### 10.2 ML Model Types

#### 10.2.1 Regime Classification Model

**Purpose**: Replace rules-based regime detection with learned classification.

**Model Architecture Options**:
| Architecture | Pros | Cons |
|--------------|------|------|
| XGBoost Classifier | Fast, interpretable feature importance | Limited sequence modeling |
| LSTM/GRU | Captures temporal patterns | Slower, harder to interpret |
| Transformer | Best for complex patterns | Requires more data |
| Hidden Markov Model | Natural for regime states | Limited feature capacity |

**Recommended Starting Point**: XGBoost with engineered features, then upgrade to LSTM if needed.

**Input Features**:
```python
@dataclass
class MLFeatureSet:
    # Price-based (20 features)
    returns_1h: float  # Log returns at various horizons
    returns_4h: float
    returns_24h: float
    returns_7d: float

    # Technical indicators (30 features)
    sma_20: float
    sma_50: float
    sma_200: float
    rsi_14: float
    macd_signal: float
    bb_position: float  # 0-1, where in Bollinger band
    adx: float
    atr_ratio: float  # ATR / price

    # Volatility features (15 features)
    realized_vol_1d: float
    realized_vol_7d: float
    vol_of_vol: float
    vol_regime_change: float  # Rate of change of vol

    # Volume features (10 features)
    volume_sma_ratio: float
    volume_spike: float

    # Cross-market (optional, 10 features)
    btc_correlation: float  # For alts
    sp500_correlation: float
    dxy_direction: float

    # Time features (5 features)
    hour_of_day_sin: float
    hour_of_day_cos: float
    day_of_week_sin: float
    day_of_week_cos: float
    is_weekend: bool
```

**Output**:
```python
@dataclass
class MLRegimePrediction:
    regime: MarketRegime
    probabilities: Dict[MarketRegime, float]  # Probability per class
    confidence: float  # Max probability
    uncertainty: float  # Entropy of distribution

    # Calibration data
    calibrated_confidence: float  # After isotonic regression
```

#### 10.2.2 Signal Confidence Model

**Purpose**: Enhance strategy signals with ML-based confidence scoring.

**Architecture**: Gradient Boosting (LightGBM/CatBoost) for fast inference.

**Input Features**:
```python
@dataclass
class SignalFeatures:
    # Strategy signal context
    regime: MarketRegime
    strategy_signal: SignalDirection
    strategy_confidence: float

    # Market context at signal time
    spread_pct: float
    order_book_imbalance: float
    volume_24h: float

    # Historical performance
    strategy_win_rate_30d: float
    strategy_sharpe_30d: float
    regime_specific_win_rate: float

    # Time since events
    hours_since_regime_change: float
    hours_since_last_trade: float

    # Correlation features
    signal_agreement_count: int  # How many strategies agree
```

**Output**:
```python
@dataclass
class MLSignalConfidence:
    adjusted_confidence: float  # [0, 1]
    expected_rr: float  # Expected risk/reward
    recommended_size_mult: float  # 0.5 - 1.5
```

### 10.3 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML TRAINING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DATA COLLECTION                                                          │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│     │  Historical  │───▶│   Feature    │───▶│   Label      │               │
│     │    OHLCV     │    │  Engineering │    │  Generation  │               │
│     └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                         │
│                                                    ▼                         │
│  2. TRAINING                                                                 │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│     │ Walk-Forward │───▶│    Train     │───▶│   Validate   │               │
│     │    Split     │    │    Model     │    │  Out-of-Sample│              │
│     └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                         │
│                                                    ▼                         │
│  3. EVALUATION                                                               │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│     │  Backtest    │───▶│   Compare    │───▶│   Gate       │               │
│     │  Simulation  │    │  vs Baseline │    │   Decision   │               │
│     └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                    │                         │
│                              PASS ────────────────┼──── FAIL                │
│                                │                  │      │                   │
│                                ▼                  │      ▼                   │
│  4. DEPLOYMENT                │            REJECT & INVESTIGATE             │
│     ┌──────────────┐          │                                             │
│     │  Shadow      │◀─────────┘                                             │
│     │  Mode        │                                                         │
│     └──────────────┘                                                        │
│            │                                                                 │
│            ▼                                                                 │
│     ┌──────────────┐                                                        │
│     │  Gradual     │                                                        │
│     │  Rollout     │                                                        │
│     └──────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Gating Rules

**ML Model can only be promoted if ALL conditions are met**:

```python
@dataclass
class MLGatingCriteria:
    """Criteria for promoting ML model from shadow to live."""

    # Minimum data requirements
    min_shadow_days: int = 30
    min_predictions: int = 1000

    # Performance requirements (vs rules-based baseline)
    min_accuracy_improvement: float = 0.05  # 5% better accuracy
    min_sharpe_improvement: float = 0.10    # 10% better Sharpe
    max_drawdown_increase: float = 0.00     # Cannot increase max DD

    # Calibration requirements
    max_calibration_error: float = 0.05     # Brier score component

    # Stability requirements
    max_prediction_flip_rate: float = 0.10  # Can't change mind too often
    min_confidence_correlation: float = 0.50  # Confidence should correlate with accuracy

    # Safety requirements
    max_crash_misclassification_rate: float = 0.05  # MUST detect crashes

def check_gating_criteria(
    shadow_results: ShadowResults,
    baseline_results: BaselineResults,
    criteria: MLGatingCriteria
) -> GatingDecision:
    """Check if ML model meets promotion criteria."""

    decisions = []

    # Check each criterion
    if shadow_results.days < criteria.min_shadow_days:
        decisions.append(GatingCheck("min_days", False, f"{shadow_results.days} < {criteria.min_shadow_days}"))

    accuracy_improvement = shadow_results.accuracy - baseline_results.accuracy
    if accuracy_improvement < criteria.min_accuracy_improvement:
        decisions.append(GatingCheck("accuracy", False, f"Improvement {accuracy_improvement:.2%} < {criteria.min_accuracy_improvement:.2%}"))

    sharpe_improvement = (shadow_results.sharpe - baseline_results.sharpe) / baseline_results.sharpe
    if sharpe_improvement < criteria.min_sharpe_improvement:
        decisions.append(GatingCheck("sharpe", False, f"Improvement {sharpe_improvement:.2%} < {criteria.min_sharpe_improvement:.2%}"))

    # CRITICAL: Crash detection
    crash_miss_rate = shadow_results.crash_misclassifications / shadow_results.total_crashes
    if crash_miss_rate > criteria.max_crash_misclassification_rate:
        decisions.append(GatingCheck("crash_detection", False,
            f"CRITICAL: Missed {crash_miss_rate:.1%} of crashes", severity="CRITICAL"))

    passed = all(d.passed for d in decisions)
    return GatingDecision(
        passed=passed,
        checks=decisions,
        recommendation="PROMOTE" if passed else "REJECT"
    )
```

### 10.5 Shadow Mode Implementation

```python
class ShadowModeRunner:
    """Run ML model in shadow mode alongside rules-based."""

    def __init__(self, rules_detector: RegimeDetector, ml_detector: MLRegimeDetector):
        self.rules_detector = rules_detector
        self.ml_detector = ml_detector
        self.comparison_log: List[Dict] = []

    def detect(self, df: pd.DataFrame, symbol: str, timeframe: str) -> RegimeState:
        """Run both detectors and compare, but return rules-based result."""

        # Rules-based (always used for actual trading)
        rules_result = self.rules_detector.detect(df, symbol, timeframe)

        # ML (shadow, not used for trading)
        try:
            ml_result = self.ml_detector.detect(df, symbol, timeframe)

            # Log comparison
            self.comparison_log.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "rules_regime": rules_result.regime.value,
                "rules_confidence": rules_result.confidence,
                "ml_regime": ml_result.regime.value,
                "ml_confidence": ml_result.confidence,
                "agreement": rules_result.regime == ml_result.regime,
            })

            # Alert on significant disagreement
            if rules_result.regime != ml_result.regime:
                if ml_result.confidence > 0.8:
                    logger.warning(
                        f"Shadow ML disagrees: rules={rules_result.regime.value}, "
                        f"ml={ml_result.regime.value} (conf={ml_result.confidence:.2f})"
                    )
        except Exception as e:
            logger.error(f"Shadow ML model error: {e}")

        # Always return rules-based
        return rules_result

    def get_shadow_report(self) -> Dict:
        """Generate report on shadow mode performance."""
        if not self.comparison_log:
            return {"status": "no_data"}

        agreement_rate = sum(1 for c in self.comparison_log if c["agreement"]) / len(self.comparison_log)

        return {
            "total_comparisons": len(self.comparison_log),
            "agreement_rate": agreement_rate,
            "days_in_shadow": (datetime.now() - datetime.fromisoformat(self.comparison_log[0]["timestamp"])).days,
            "regime_distribution": self._compute_regime_distribution(),
        }
```

### 10.6 Gradual Rollout Strategy

```python
class GradualRollout:
    """Gradually increase ML model exposure."""

    STAGES = [
        {"name": "shadow", "ml_weight": 0.0, "min_days": 30},
        {"name": "canary", "ml_weight": 0.1, "min_days": 7},
        {"name": "partial", "ml_weight": 0.5, "min_days": 14},
        {"name": "full", "ml_weight": 1.0, "min_days": None},
    ]

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.current_stage_idx = 0
        self.stage_start_time = datetime.now()
        self.stage_metrics: List[Dict] = []

    def get_effective_regime(
        self,
        rules_result: RegimeState,
        ml_result: RegimeState,
    ) -> RegimeState:
        """Get effective regime based on current rollout stage."""

        stage = self.STAGES[self.current_stage_idx]
        ml_weight = stage["ml_weight"]

        if ml_weight == 0.0:
            return rules_result
        elif ml_weight == 1.0:
            return ml_result
        else:
            # Weighted decision (use ML only if high confidence)
            if ml_result.confidence > 0.8 and random.random() < ml_weight:
                return ml_result
            return rules_result

    def check_promotion(self, metrics: StageMetrics) -> bool:
        """Check if ready to promote to next stage."""

        stage = self.STAGES[self.current_stage_idx]

        # Check minimum time
        days_in_stage = (datetime.now() - self.stage_start_time).days
        if days_in_stage < stage["min_days"]:
            return False

        # Check performance criteria
        if metrics.sharpe_ratio < self.config.min_sharpe:
            return False
        if metrics.max_drawdown > self.config.max_drawdown:
            return False

        return True

    def promote(self) -> bool:
        """Promote to next rollout stage."""
        if self.current_stage_idx >= len(self.STAGES) - 1:
            return False

        self.current_stage_idx += 1
        self.stage_start_time = datetime.now()

        logger.info(f"Promoted to stage: {self.STAGES[self.current_stage_idx]['name']}")
        return True

    def rollback(self, reason: str) -> None:
        """Rollback to previous stage."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1

        logger.warning(f"Rolled back to stage {self.STAGES[self.current_stage_idx]['name']}: {reason}")
```

### 10.7 Monitoring and Auto-Rollback

```python
class MLModelMonitor:
    """Monitor ML model in production and trigger rollback if needed."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.prediction_history: List[Dict] = []
        self.performance_window: List[Dict] = []

    def record_prediction(self, prediction: RegimePrediction, actual: MarketRegime):
        """Record prediction for monitoring."""
        self.prediction_history.append({
            "timestamp": datetime.now(),
            "predicted": prediction.regime,
            "actual": actual,
            "confidence": prediction.confidence,
            "correct": prediction.regime == actual,
        })

        # Check for anomalies
        self._check_anomalies()

    def _check_anomalies(self) -> None:
        """Check for anomalies that require rollback."""

        recent = self.prediction_history[-100:]  # Last 100 predictions

        # Check 1: Accuracy degradation
        accuracy = sum(1 for p in recent if p["correct"]) / len(recent)
        if accuracy < self.config.min_accuracy_threshold:
            self._trigger_rollback(f"Accuracy degraded to {accuracy:.2%}")

        # Check 2: Confidence calibration
        high_conf_correct = sum(1 for p in recent if p["confidence"] > 0.8 and p["correct"])
        high_conf_total = sum(1 for p in recent if p["confidence"] > 0.8)
        if high_conf_total > 10:
            calibration = high_conf_correct / high_conf_total
            if calibration < 0.7:  # High confidence should be correct 70%+
                self._trigger_rollback(f"Calibration error: {calibration:.2%}")

        # Check 3: Crash detection failure (CRITICAL)
        crash_missed = sum(
            1 for p in recent
            if p["actual"] == MarketRegime.CRASH and p["predicted"] != MarketRegime.CRASH
        )
        crash_total = sum(1 for p in recent if p["actual"] == MarketRegime.CRASH)
        if crash_total > 0 and crash_missed / crash_total > 0.1:
            self._trigger_rollback(f"CRITICAL: Missed {crash_missed}/{crash_total} crashes")

    def _trigger_rollback(self, reason: str) -> None:
        """Trigger automatic rollback to rules-based."""
        logger.critical(f"AUTO-ROLLBACK TRIGGERED: {reason}")

        # Notify
        try:
            from bot.notifications import NotificationManager
            notifier = NotificationManager()
            notifier.send_critical(f"ML Model Rollback: {reason}")
        except Exception:
            pass

        # Trigger rollback
        raise MLModelRollbackRequired(reason)
```

### 10.8 Data Schema for ML Training

```python
# training_data.py

@dataclass
class TrainingExample:
    """Single training example for regime classification."""

    timestamp: datetime
    symbol: str

    # Features (see MLFeatureSet above)
    features: Dict[str, float]

    # Labels
    regime_1h_ahead: MarketRegime  # What regime was 1h later
    regime_4h_ahead: MarketRegime  # What regime was 4h later
    regime_24h_ahead: MarketRegime  # What regime was 24h later

    # Forward returns (for strategy evaluation)
    return_1h: float
    return_4h: float
    return_24h: float
    max_drawdown_24h: float

    # Meta
    data_quality_score: float  # 0-1, based on gaps, outliers

def generate_training_dataset(
    ohlcv_data: pd.DataFrame,
    min_quality: float = 0.9
) -> List[TrainingExample]:
    """Generate training dataset from historical OHLCV data."""

    examples = []

    for i in range(LOOKBACK, len(ohlcv_data) - FORWARD_WINDOW):
        window = ohlcv_data.iloc[i-LOOKBACK:i+1]
        forward = ohlcv_data.iloc[i:i+FORWARD_WINDOW]

        # Compute features
        features = compute_ml_features(window)

        # Compute labels (true regime based on forward price action)
        labels = compute_regime_labels(forward)

        # Quality check
        quality = compute_data_quality(window)
        if quality < min_quality:
            continue

        examples.append(TrainingExample(
            timestamp=ohlcv_data.index[i],
            symbol=symbol,
            features=features,
            **labels,
            data_quality_score=quality
        ))

    return examples
```

### 10.9 Upgrade Checklist

**Before Starting ML Upgrade**:
- [ ] Baseline rules-based system stable in production for 30+ days
- [ ] Comprehensive logging and metrics in place
- [ ] Shadow mode infrastructure ready
- [ ] Rollback mechanism tested

**Phase 1: Shadow Mode**:
- [ ] ML model trained on historical data
- [ ] Walk-forward validation shows improvement
- [ ] Shadow mode running for 30+ days
- [ ] Agreement rate with rules-based > 70%
- [ ] CRITICAL: Crash detection rate > 95%

**Phase 2: Canary (10% exposure)**:
- [ ] Gating criteria passed
- [ ] 7+ days at 10% with no degradation
- [ ] Auto-rollback mechanism working
- [ ] No increase in max drawdown

**Phase 3: Partial (50% exposure)**:
- [ ] 14+ days at canary with positive results
- [ ] Sharpe ratio improvement confirmed
- [ ] No critical alerts triggered
- [ ] Team review and approval

**Phase 4: Full Rollout**:
- [ ] 30+ days at partial with sustained improvement
- [ ] Executive approval for full rollout
- [ ] Incident response plan in place
- [ ] 24/7 monitoring enabled for first week
