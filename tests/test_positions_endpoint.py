import json
from pathlib import Path

import pytest

from api import unified_trading_api as unified_api


@pytest.mark.asyncio
async def test_detailed_positions_endpoint(monkeypatch, tmp_path):
    """Detailed positions endpoint should return market value/pct details."""
    state_path = tmp_path / "state.json"
    state_payload = {
        "mode": "paper_live_data",
        "status": "active",
        "timestamp": "2025-01-01T00:00:00Z",
        "initial_capital": 1000.0,
        "current_balance": 1000.0,
        "positions": {
            "BTC/USDT": {
                "side": "LONG",
                "quantity": 0.01,
                "entry_price": 42000.0,
                "current_price": 43000.0,
                "entry_time": "2025-01-01T00:00:00Z",
                "unrealized_pnl": 100.0,
            },
            "ETH/USDT": {
                "side": "LONG",
                "quantity": 0.5,
                "entry_price": 190.0,
                "current_price": 200.0,
                "entry_time": "2025-01-01T00:00:00Z",
            },
        },
    }
    state_path.write_text(json.dumps(state_payload))

    monkeypatch.setattr(unified_api, "_get_state_path", lambda: state_path)

    response = await unified_api.get_detailed_positions()

    assert response.total_market_value == pytest.approx(530.0)
    assert response.total_unrealized_pnl == pytest.approx(105.0)
    assert len(response.positions) == 2

    btc = next(p for p in response.positions if p.symbol == "BTC/USDT")
    assert btc.unrealized_pnl == pytest.approx(100.0)
    assert btc.pct_of_portfolio == pytest.approx(430.0 / (1000.0 + 530.0))

    eth = next(p for p in response.positions if p.symbol == "ETH/USDT")
    assert eth.unrealized_pnl == pytest.approx(5.0)
    assert eth.entry_price == 190.0
    assert eth.market_value == pytest.approx(100.0)
