import json
from pathlib import Path

import bot.data_store as data_store
from api.unified_trading_api import _load_execution_log


def test_load_execution_log_rebuilds_from_datastore(monkeypatch, tmp_path):
    """Rebuilds execution_log.json from DataStore trades when missing."""
    monkeypatch.chdir(tmp_path)

    calls = {"count": 0}

    class FakeDataStore:
        def get_trades(self, limit=5000):
            calls["count"] += 1
            return [
                {
                    "trade_id": "t1",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "symbol": "BTC/USDT",
                    "action": "BUY",
                    "quantity": 0.1,
                    "price": 42000,
                    "metadata": {
                        "slippage_pct": 0.05,
                        "fees_paid": 1.2,
                        "execution_mode": "paper",
                        "status": "done",
                        "execution_time_ms": 120,
                    },
                },
                {
                    "trade_id": "t2",
                    "timestamp": "2024-01-01T01:00:00Z",
                    "symbol": "ETH/USDT",
                    "action": "SELL",
                    "quantity": 1.5,
                    "price": 3200,
                    "metadata": {},
                },
            ]

    monkeypatch.setattr(data_store, "DataStore", FakeDataStore)

    result = _load_execution_log(limit=10)

    log_path = Path("data/unified_trading/execution_log.json")
    assert log_path.exists(), "execution_log.json should be rebuilt"
    assert calls["count"] == 1, "DataStore.get_trades should be called once"

    assert len(result) == 2
    assert result[0]["order_id"] == "t1"
    assert result[0]["symbol"] == "BTC/USDT"
    assert result[0]["slippage_pct"] == 0.05
    assert result[0]["commission"] == 1.2
    assert result[0]["status"] == "done"
    assert result[0]["mode"] == "paper"
    assert result[1]["order_id"] == "t2"
    assert result[1]["symbol"] == "ETH/USDT"
    assert result[1]["commission"] == 0

    with open(log_path) as f:
        persisted = json.load(f)

    assert len(persisted) == 2
    assert persisted[0]["order_id"] == "t1"
    assert persisted[1]["order_id"] == "t2"
