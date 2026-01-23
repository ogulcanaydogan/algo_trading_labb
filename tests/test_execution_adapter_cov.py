from bot.execution_adapter import PaperExecutionAdapter, create_execution_adapter
from bot.trading_mode import TradingMode


def test_paper_adapter_initializes():
    """Test that PaperExecutionAdapter initializes correctly."""
    adapter = PaperExecutionAdapter(initial_balance=1000.0, commission_rate=0.001)

    assert adapter._balance == 1000.0
    assert adapter.commission_rate == 0.001
    assert adapter._positions == {}


def test_paper_adapter_set_price():
    """Test that paper adapter can set simulated prices."""
    adapter = PaperExecutionAdapter(initial_balance=1000.0)

    adapter.set_price("BTC/USDT", 50000.0)
    price = adapter._prices["BTC/USDT"]

    assert price == 50000.0


def test_factory_returns_paper_adapter():
    """Test that factory creates correct adapter type."""
    paper = create_execution_adapter(TradingMode.PAPER_LIVE_DATA.value)
    assert isinstance(paper, PaperExecutionAdapter)
