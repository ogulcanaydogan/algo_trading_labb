import pytest

from api import unified_trading_api as unified_api


@pytest.mark.asyncio
async def test_execution_telemetry_aggregates(monkeypatch):
    logs = [
        {
            "symbol": "BTC/USDT",
            "slippage_pct": 0.1,
            "execution_time_ms": 150,
            "commission": 0.5,
        },
        {
            "symbol": "BTC/USDT",
            "slippage_pct": 0.05,
            "execution_time_ms": 170,
            "commission": 0.45,
        },
        {
            "symbol": "ETH/USDT",
            "slippage_pct": 0.2,
            "execution_time_ms": 210,
            "commission": 0.2,
        },
    ]

    monkeypatch.setattr(unified_api, "_load_execution_log", lambda limit=None: logs)

    telemetry = await unified_api.get_execution_telemetry()

    assert telemetry.total_trades == 3
    assert telemetry.avg_slippage_pct == pytest.approx((0.1 + 0.05 + 0.2) / 3)
    assert telemetry.avg_execution_time_ms == pytest.approx((150 + 170 + 210) / 3)
    assert telemetry.total_commission == pytest.approx(1.15)
    assert telemetry.worst_slippage_pct == pytest.approx(0.2)
    assert len(telemetry.symbol_breakdown) == 2
    btc_entry = next(e for e in telemetry.symbol_breakdown if e.symbol == "BTC/USDT")
    assert btc_entry.trades == 2
    assert btc_entry.avg_slippage_pct == pytest.approx((0.1 + 0.05) / 2)
