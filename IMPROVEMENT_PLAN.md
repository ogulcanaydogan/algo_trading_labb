# Action Plan: 60%+ Win Rate & 80%+ ML Accuracy & Live Trading All Assets

## Current Status (Jan 18, 2026)

### Metrics
- **Win Rate**: 55% (Target: >60%)
- **ML Accuracy**: 67% avg (Target: >80%)
- **Paper Trades**: 100/200 (need 100 more for testnet)
- **Drawdown**: 100% (warning threshold: 12%)

### Asset Support
✅ **Crypto**: Binance testnet/live ready  
❌ **Commodities**: Yahoo Finance data only (no live broker)  
❌ **Stocks**: Yahoo Finance data only (no live broker)

## Phase 1: Improve ML Accuracy (67% → 80%+)

### Completed
✅ Trained improved models with enhanced features  
✅ Random Forest & Gradient Boosting on 730 days history  
✅ Results: BTC 71.4%, ETH 65.4% (avg 67%)

### Next Steps
1. **Add More Features** (Target: +5% accuracy)
   - Order flow imbalance indicators
   - Multi-timeframe trend alignment
   - Volatility regime classification
   - Market microstructure features

2. **Better Labels** (Target: +3% accuracy)
   - Use forward volatility-adjusted returns
   - Filter low-confidence training samples
   - Add future drawdown as auxiliary target

3. **Ensemble Methods** (Target: +5% accuracy)
   - Combine RF + GB + existing XGBoost
   - Voting classifier with weighted confidence
   - Meta-learner stacking

4. **Walk-Forward Optimization**
   - Retrain models monthly with rolling window
   - Validate on out-of-sample periods
   - Track real-time accuracy in ml_performance.db

**Command to retrain with all improvements:**
```bash
python scripts/ml/improved_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT AVAX/USDT \
  --days 730 \
  --ensemble \
  --walk-forward
```

## Phase 2: Improve Win Rate (55% → 60%+)

### Strategy 1: Raise Confidence Threshold
- Current: 0.45 (allows many marginal signals)
- Target: 0.60 (filter weak signals)
- Expected impact: +3% win rate, -30% trade frequency

**Edit bot/ml_signal_generator.py:**
```python
self.confidence_threshold = 0.60  # was 0.45
```

### Strategy 2: Enable Strict MTF Filter
- Require 4h and 1d timeframes to confirm 1h signals
- Block counter-trend trades

**Edit config.yaml:**
```yaml
ml:
  mtf_strict_mode: true
  mtf_min_timeframes: 3  # 1h + 4h + 1d all must agree
```

### Strategy 3: Tighter Stop Loss & Take Profit
- Current: 2% SL, 4% TP (2:1 R:R)
- Target: 1.5% SL, 3% TP with trailing stop

**Edit config.yaml:**
```yaml
trading:
  stop_loss_pct: 0.015  # 1.5%
  take_profit_pct: 0.03  # 3%
  trailing_stop: true
  trailing_stop_pct: 0.01  # lock profits after 1% gain
```

### Strategy 4: Regime Adaptation
- Bear market: Only high-confidence longs (>0.70), allow shorts
- Bull market: Lower threshold longs (0.55), no shorts
- Sideways: Flat bias, require 0.65+ confidence

**Currently implemented in bot/ai_trading_brain.py**

### Expected Results
- Confidence 0.45→0.60: +3% win rate
- MTF filter: +2% win rate  
- Tighter R:R: +1% win rate
- **Total: 55% → 61% win rate**

## Phase 3: Live Broker Integration (Stocks & Commodities)

### Option A: Alpaca (Easiest for Stocks)
**Features:**
- Commission-free stock trading
- Paper trading sandbox
- Real-time market data
- REST + WebSocket API

**Integration Steps:**
1. Sign up at alpaca.markets
2. Get API keys (paper + live)
3. Create `bot/alpaca_adapter.py`:
```python
class AlpacaAdapter(ExecutionAdapter):
    def __init__(self, api_key, secret, is_paper=True):
        from alpaca_trade_api import REST
        base_url = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"
        self.client = REST(api_key, secret, base_url)
    
    async def execute_order(self, order):
        return self.client.submit_order(
            symbol=order.symbol,
            qty=order.quantity,
            side=order.side.lower(),
            type="market",
            time_in_force="gtc"
        )
```

4. Update `bot/execution_adapter.py` to route stocks via Alpaca

**Timeline**: 2-3 days implementation + 1 week paper testing

### Option B: Interactive Brokers (All Assets)
**Features:**
- Stocks, commodities, forex, options, futures
- Lower fees for high volume
- TWS/Gateway API

**Integration Steps:**
1. Open IBKR account
2. Enable API access in Account Management
3. Install ib_insync: `pip install ib_insync`
4. Create `bot/ibkr_adapter.py`
5. Paper test on IB Paper Trading account

**Timeline**: 1 week implementation + 2 weeks paper testing

### Recommended Approach
Start with **Alpaca for stocks** (quick win), then add **IBKR for commodities** if needed.

## Phase 4: Accumulate Paper Trades (100/200 → 200/200)

### Current Blocking Issue
- Need 100 more trades to unlock testnet
- At current rate: ~5-10 trades/day = 10-20 days

### Acceleration Options

**Option 1: Run 24/7 with lower loop_interval**
```yaml
# config.yaml
trading:
  loop_interval: 60  # was 180, reduces to 1 min intervals
```
Expected: 20-30 trades/day = unlock in 3-5 days

**Option 2: Add More Symbols**
```yaml
crypto:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
    - AVAX/USDT
    - XRP/USDT  # New
    - ADA/USDT  # New
    - MATIC/USDT  # New
```
Expected: 3x more trade opportunities = unlock in 5-7 days

**Option 3: Use Backtest to Simulate**
Run backtests on historical data to validate strategy parameters, doesn't count toward gate but proves strategy works:
```bash
python scripts/backtest/run_backtest.py --days 90
```

## Immediate Actions (Next 24 Hours)

1. ✅ **Train improved ML models** (completed - 67% accuracy)
2. 🔄 **Raise confidence threshold to 0.60** (in progress)
3. 🔄 **Start paper bot with optimized config** (in progress)
4. ⏳ **Set up Alpaca paper account for stocks**
5. ⏳ **Monitor win rate improvement over 50 trades**

## Weekly Milestones

### Week 1 (Current)
- Day 1: Retrain models, optimize config ✅
- Day 2-3: Run paper bot 24/7, accumulate 50 trades
- Day 4-5: Analyze results, tune if needed
- Day 6-7: Complete 200 trades, check win rate

### Week 2
- Day 1-2: Set up Alpaca integration
- Day 3-5: Paper test stock trading
- Day 6-7: Add ensemble ML models for 80% accuracy

### Week 3
- Move BTC/ETH to testnet (if win rate >60%)
- Continue paper trading stocks via Alpaca
- Add commodity broker research

### Week 4
- Full live deployment if all metrics pass:
  - ✅ 200+ paper trades completed
  - ✅ Win rate >60%
  - ✅ ML accuracy >80%
  - ✅ Max drawdown <12%
  - ✅ All safety checks pass

## Risk Controls (Never Disable)

### Hard Limits
- Max 5% position size (enforced by SafetyController)
- Daily loss limit: 2% → auto-pause
- Max 3 concurrent positions
- Stop loss REQUIRED on all positions

### Feature Flags (data/risk_settings.json)
```json
{
  "shorting": false,      // Start with long-only
  "leverage": false,      // No margin in phase 1
  "aggressive": false     // Conservative strategies only
}
```

Enable shorting only after:
- 500+ trades with >60% win rate
- Proven bear market strategy
- Testnet validation complete

## Success Criteria

### ML Accuracy: 80%+
- Ensemble voting: RF + GB + XGBoost
- Walk-forward validation >78% on 5 folds
- Real-time tracking shows >80% over 100 predictions

### Win Rate: 60%+
- Confidence threshold 0.60+ filters weak signals
- MTF confirmation reduces false entries
- Trailing stops lock in profits
- Tracked over minimum 100 trades

### Live Trading All Assets
- ✅ Crypto: Binance testnet → live path ready
- 🔄 Stocks: Alpaca integration (Week 2)
- ⏳ Commodities: IBKR or futures broker (Week 3-4)

## Monitoring Dashboard

Access at: http://localhost:8000

### Key Metrics to Watch
1. **Real-time win rate** (target: >60%)
2. **ML prediction accuracy** (target: >80%)
3. **Daily P&L** (target: +1%, stop: -2%)
4. **Trade count** (target: 200/200 for testnet unlock)
5. **Drawdown** (target: <12%)

## Contact & Support

- Dashboard: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Logs: `data/unified_trading/logs/`
- State: `data/unified_trading/state.json`

---

**Last Updated**: Jan 18, 2026  
**Next Review**: Jan 25, 2026 (after 1 week of paper trading)
