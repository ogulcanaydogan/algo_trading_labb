from bot.unified_engine import EngineConfig, UnifiedTradingEngine
from bot.trading_mode import TradingMode


def test_engine_config_basic():
    """Test basic engine config creation."""
    cfg = EngineConfig(
        initial_mode=TradingMode.PAPER_LIVE_DATA,
        initial_capital=5000.0,
    )

    assert cfg.initial_mode == TradingMode.PAPER_LIVE_DATA
    assert cfg.initial_capital == 5000.0


def test_engine_creation():
    """Test that engine can be instantiated."""
    cfg = EngineConfig(
        initial_mode=TradingMode.PAPER_LIVE_DATA,
        initial_capital=5000.0,
        use_ml_signals=False,
        use_ai_brain=False,
    )

    engine = UnifiedTradingEngine(cfg)
    assert engine.config == cfg
