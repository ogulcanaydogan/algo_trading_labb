from bot.safety_controller import SafetyController, SafetyLimits


def test_safety_controller_basic_init():
    """Test basic initialization."""
    limits = SafetyLimits()
    controller = SafetyController(limits=limits)
    controller.update_balance(1000.0)

    assert controller._current_balance == 1000.0
    assert controller.limits.max_position_size_usd > 0


def test_safety_controller_update_positions():
    """Test that positions can be tracked."""
    limits = SafetyLimits()
    controller = SafetyController(limits=limits)
    controller.update_balance(1000.0)

    # Update positions
    positions = {"BTC/USDT": 50.0, "ETH/USDT": 25.0}
    controller.update_positions(positions)

    assert controller._open_positions == positions
