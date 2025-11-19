from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import api as api_module
from api.schemas import PortfolioControlUpdateRequest
from bot.state import BotState


def test_load_portfolio_states_include_config_placeholders(tmp_path, monkeypatch):
    portfolio_dir = tmp_path / "portfolio"
    xau_dir = portfolio_dir / "xauusd"
    xau_dir.mkdir(parents=True)

    xau_state = BotState(
        symbol="XAU/USD",
        position="LONG",
        entry_price=1950.0,
        position_size=1.25,
        balance=10_200.0,
        initial_balance=None,
        risk_per_trade_pct=0.4,
    )
    (xau_dir / "state.json").write_text(
        json.dumps(xau_state.to_dict()),
        encoding="utf-8",
    )

    config_payload = {
        "portfolio_capital": 20_000,
        "default_timeframe": "1h",
        "default_loop_interval_seconds": 120,
        "default_paper_mode": True,
        "default_risk_per_trade_pct": 0.5,
        "default_stop_loss_pct": 0.015,
        "default_take_profit_pct": 0.03,
        "assets": [
            {
                "symbol": "XAU/USD",
                "asset_type": "commodity",
                "allocation_pct": 60,
                "starting_balance": 9_000,
                "paper_mode": False,
                "timeframe": "4h",
                "loop_interval_seconds": 90,
                "stop_loss_pct": 0.018,
                "take_profit_pct": 0.036,
            },
            {
                "symbol": "XAG/USD",
                "asset_type": "commodity",
                "allocation_pct": 40,
            },
        ],
    }
    config_path = tmp_path / "portfolio.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setattr(api_module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(api_module, "PORTFOLIO_CONFIG_PATH", config_path)

    statuses = api_module.load_portfolio_states()
    assert {status.symbol for status in statuses} == {"XAU/USD", "XAG/USD"}

    xau = next(status for status in statuses if status.symbol == "XAU/USD")
    xag = next(status for status in statuses if status.symbol == "XAG/USD")

    assert not xau.is_placeholder
    assert xau.asset_type == "commodity"
    assert xau.timeframe == "4h"
    assert xau.paper_mode is False
    assert xau.loop_interval_seconds == 90
    assert xau.balance == pytest.approx(10_200.0)
    assert xau.initial_balance == pytest.approx(9_000.0)
    assert xau.stop_loss_pct == pytest.approx(0.018)
    assert xau.take_profit_pct == pytest.approx(0.036)
    assert not xau.is_paused
    assert xau.pause_reason is None

    assert xag.is_placeholder
    assert xag.balance == pytest.approx(8_000.0)
    assert xag.initial_balance == pytest.approx(8_000.0)
    assert xag.asset_type == "commodity"
    assert xag.timeframe == "1h"
    assert xag.paper_mode is True
    assert xag.loop_interval_seconds == 120
    assert xag.risk_per_trade_pct == pytest.approx(0.5)
    assert xag.stop_loss_pct == pytest.approx(0.015)
    assert xag.take_profit_pct == pytest.approx(0.03)
    assert not xag.is_paused

    statuses_filtered = api_module.load_portfolio_states("XAG/USD")
    assert len(statuses_filtered) == 1
    assert statuses_filtered[0].symbol == "XAG/USD"


def test_portfolio_control_endpoint_updates_state(tmp_path, monkeypatch):
    portfolio_dir = tmp_path / "portfolio"
    config_payload = {
        "portfolio_capital": 10_000,
        "default_timeframe": "1h",
        "default_loop_interval_seconds": 60,
        "default_paper_mode": True,
        "default_risk_per_trade_pct": 0.5,
        "assets": [
            {
                "symbol": "XAU/USD",
                "asset_type": "commodity",
                "allocation_pct": 100,
            }
        ],
    }
    config_path = tmp_path / "portfolio.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setattr(api_module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(api_module, "PORTFOLIO_CONFIG_PATH", config_path)

    request = PortfolioControlUpdateRequest(
        symbol="XAU/USD",
        paused=True,
        reason="Macro volatility",
    )
    response = api_module.update_portfolio_control(request)
    assert response.paused is True
    assert response.symbol == "XAU/USD"

    control_file = portfolio_dir / "XAU_USD" / "control.json"
    assert control_file.exists()

    statuses = api_module.load_portfolio_states("XAU/USD")
    assert len(statuses) == 1
    status = statuses[0]
    assert status.is_paused is True
    assert status.pause_reason == "Macro volatility"


def test_repo_default_portfolio_config_exposed(monkeypatch, tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "data" / "portfolio.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected_symbols = {asset["symbol"] for asset in payload["assets"]}

    monkeypatch.setattr(api_module, "STATE_DIR", tmp_path)
    monkeypatch.setattr(api_module, "PORTFOLIO_CONFIG_PATH", config_path)

    statuses = api_module.load_portfolio_states()
    assert {status.symbol for status in statuses} == expected_symbols
    assert all(status.is_placeholder for status in statuses)

    for status in statuses:
        assert status.initial_balance == pytest.approx(10_000.0)
        assert status.balance == pytest.approx(10_000.0)
