import json

import pandas as pd
import pytest

from bot.ml_signal_generator import MLSignalGenerator
from bot.rl.shadow_data_collector import ShadowCollectorConfig, ShadowDataCollector
from bot.rl.multi_agent_system import MarketState


def _sample_df(rows: int = 60) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h")
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_gate_trace_populated_on_confidence_reject(tmp_path, monkeypatch):
    generator = MLSignalGenerator(use_ensemble=False, use_scalping=False, use_mtf_filter=False)
    generator.use_mtf_filter = False
    generator._initialized = True
    generator.models["BTC/USDT"] = object()

    def fake_fetch_prices(*_args, **_kwargs):
        df = _sample_df()
        generator._last_fetch_source = "ccxt"
        generator._last_fetch_timeframe = "1h"
        generator._last_fetch_rows = len(df)
        generator._last_fetch_first_ts = df.index[0].isoformat()
        generator._last_fetch_last_ts = df.index[-1].isoformat()
        return df

    monkeypatch.setattr(generator, "_fetch_prices", fake_fetch_prices)

    async def fake_ml_signal(*_args, **_kwargs):
        generator._set_block_reason("ml", "ml_confidence_below_threshold")
        return None

    monkeypatch.setattr(generator, "_ml_signal", fake_ml_signal)

    signal = await generator.generate_signal("BTC/USDT", 100.0)
    assert signal is None
    gate_trace = generator.get_last_gate_trace("BTC/USDT")
    assert gate_trace is not None
    assert gate_trace["stage"] == "ml"
    assert gate_trace["reason"] == "ml_confidence_below_threshold"

    log_path = tmp_path / "shadow_decisions.jsonl"
    collector = ShadowDataCollector(
        ShadowCollectorConfig(enabled=True, log_path=log_path, enable_rl_shadow=False)
    )

    market_state = MarketState(
        symbol="BTC/USDT",
        price=100.0,
        volatility=0.0,
        regime="unknown",
        fear_greed=50.0,
        news_sentiment=0.0,
        rsi=50.0,
        trend_strength=0.0,
    )

    collector.record_decision_point(
        symbol="BTC/USDT",
        market_state=market_state,
        gate_approved=False,
        gate_score=0.0,
        gate_rejection_reason="ml_confidence_below_threshold",
        preservation_level="normal",
        actual_action="hold",
        actual_confidence=0.0,
        strategy_used="signal_blocked",
        gate_trace=gate_trace,
    )

    with open(log_path, "r") as f:
        payload = json.loads(f.readline())
    assert payload["gate_trace"]["stage"] == "ml"
    assert payload["gate_trace"]["reason"] == "ml_confidence_below_threshold"


@pytest.mark.asyncio
async def test_gate_trace_populated_on_pass(tmp_path, monkeypatch):
    generator = MLSignalGenerator(use_ensemble=False, use_scalping=False)
    generator._initialized = True
    generator._last_features_count = 120

    def fake_fetch_prices(*_args, **_kwargs):
        df = _sample_df()
        generator._last_fetch_source = "ccxt"
        generator._last_fetch_timeframe = "1h"
        generator._last_fetch_rows = len(df)
        generator._last_fetch_first_ts = df.index[0].isoformat()
        generator._last_fetch_last_ts = df.index[-1].isoformat()
        return df

    monkeypatch.setattr(generator, "_fetch_prices", fake_fetch_prices)

    async def fake_technical_signal(*_args, **_kwargs):
        return {
            "action": "BUY",
            "confidence": 0.8,
            "threshold_used": 0.6,
            "trend": "up",
            "model_type": "technical",
        }

    monkeypatch.setattr(generator, "_technical_signal", fake_technical_signal)
    monkeypatch.setattr(generator, "_apply_mtf_filter", lambda signal, _df: signal)

    signal = await generator.generate_signal("ETH/USDT", 200.0)
    assert signal is not None
    gate_trace = signal.get("gate_trace")
    assert gate_trace is not None
    assert gate_trace["stage"] == "passed"
    assert gate_trace["action"] == "BUY"
    assert gate_trace["confidence"] == 0.8
    assert gate_trace["features_count"] == 120

    log_path = tmp_path / "shadow_decisions.jsonl"
    collector = ShadowDataCollector(
        ShadowCollectorConfig(enabled=True, log_path=log_path, enable_rl_shadow=False)
    )
    market_state = MarketState(
        symbol="ETH/USDT",
        price=200.0,
        volatility=0.0,
        regime="unknown",
        fear_greed=50.0,
        news_sentiment=0.0,
        rsi=50.0,
        trend_strength=0.0,
    )
    collector.record_decision_point(
        symbol="ETH/USDT",
        market_state=market_state,
        gate_approved=True,
        gate_score=0.8,
        gate_rejection_reason="",
        preservation_level="normal",
        actual_action="BUY",
        actual_confidence=0.8,
        strategy_used="technical",
        gate_trace=gate_trace,
    )

    with open(log_path, "r") as f:
        payload = json.loads(f.readline())
    assert payload["gate_trace"]["stage"] == "passed"
    assert payload["gate_trace"]["action"] == "BUY"
