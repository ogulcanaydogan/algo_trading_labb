import pytest

from bot.adaptive_risk_controller import AdaptiveRiskController


@pytest.mark.asyncio
async def test_enable_shorting_in_bear(monkeypatch, tmp_path):
    controller = AdaptiveRiskController(data_dir=tmp_path, api_base_url="http://test")

    async def fake_update(settings):
        controller.current_settings.update(settings)
        return True

    monkeypatch.setattr(controller, "update_risk_settings", fake_update)

    strategy = await controller.evaluate_and_adjust(
        market_regime="bear",
        regime_confidence=0.8,
        rsi=65,
        volatility="normal",
        trend="down",
        recent_performance={"win_rate": 0.6, "total_pnl": 1.0, "drawdown": 2.0},
    )

    assert controller.current_settings["shorting"] is True
    assert strategy.expected_direction == "bearish"
    assert strategy.suggested_action in {"sell", "hold"}


@pytest.mark.asyncio
async def test_disable_leverage_on_high_vol(monkeypatch, tmp_path):
    controller = AdaptiveRiskController(data_dir=tmp_path, api_base_url="http://test")
    controller.current_settings["leverage"] = True

    async def fake_update(settings):
        controller.current_settings.update(settings)
        return True

    monkeypatch.setattr(controller, "update_risk_settings", fake_update)

    await controller.evaluate_and_adjust(
        market_regime="strong_bull",
        regime_confidence=0.9,
        rsi=55,
        volatility="high",
        trend="up",
        recent_performance={"win_rate": 0.75, "total_pnl": 3.0, "drawdown": 0.0},
    )

    assert controller.current_settings["leverage"] is False


@pytest.mark.asyncio
async def test_aggressive_disabled_in_crash(monkeypatch, tmp_path):
    controller = AdaptiveRiskController(data_dir=tmp_path, api_base_url="http://test")
    controller.current_settings["aggressive"] = True

    async def fake_update(settings):
        controller.current_settings.update(settings)
        return True

    monkeypatch.setattr(controller, "update_risk_settings", fake_update)

    await controller.evaluate_and_adjust(
        market_regime="crash",
        regime_confidence=0.8,
        rsi=40,
        volatility="extreme",
        trend="down",
        recent_performance={"win_rate": 0.4, "total_pnl": -5.0, "drawdown": 12.0},
    )

    assert controller.current_settings["aggressive"] is False
