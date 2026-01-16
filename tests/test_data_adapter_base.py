"""
Tests for data adapter base class.
"""

import pytest
from datetime import datetime

import pandas as pd

from bot.data.adapters.base import DataAdapter
from bot.data.models import (
    DataSource,
    MarketType,
    NormalizedOHLCV,
    NormalizedQuote,
)


class ConcreteAdapter(DataAdapter):
    """Concrete implementation for testing."""

    name = "test"
    data_source = DataSource.YAHOO
    supported_markets = [MarketType.STOCK]
    supports_realtime = True
    supports_historical = True
    priority = 1

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=250, start_time=None, end_time=None):
        """Mock implementation."""
        return []

    def fetch_quote(self, symbol):
        """Mock implementation."""
        return NormalizedQuote(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.1,
        )

    def map_symbol(self, standard_symbol):
        """Mock implementation."""
        return standard_symbol.replace("/", "-")


class TestDataAdapterBase:
    """Test DataAdapter base class methods."""

    @pytest.fixture
    def adapter(self):
        """Create concrete adapter instance."""
        return ConcreteAdapter()

    def test_adapter_properties(self, adapter):
        """Test adapter class properties."""
        assert adapter.name == "test"
        assert adapter.data_source == DataSource.YAHOO
        assert MarketType.STOCK in adapter.supported_markets
        assert adapter.supports_realtime is True
        assert adapter.supports_historical is True
        assert adapter.priority == 1

    def test_map_symbol(self, adapter):
        """Test symbol mapping."""
        result = adapter.map_symbol("BTC/USDT")
        assert result == "BTC-USDT"

    def test_map_timeframe_default(self, adapter):
        """Test default timeframe mapping returns as-is."""
        assert adapter.map_timeframe("1h") == "1h"
        assert adapter.map_timeframe("4h") == "4h"
        assert adapter.map_timeframe("1d") == "1d"

    def test_to_dataframe_empty(self, adapter):
        """Test to_dataframe with empty list."""
        df = adapter.to_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 0

    def test_to_dataframe_with_data(self, adapter):
        """Test to_dataframe with data."""
        data = [
            NormalizedOHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000,
                timeframe="1h",
                market_type=MarketType.STOCK,
                data_source=DataSource.YAHOO,
            ),
            NormalizedOHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 11, 0, 0),
                open=151.0,
                high=153.0,
                low=150.0,
                close=152.0,
                volume=1100000,
                timeframe="1h",
                market_type=MarketType.STOCK,
                data_source=DataSource.YAHOO,
            ),
        ]

        df = adapter.to_dataframe(data)

        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.iloc[0]["open"] == 150.0
        assert df.iloc[1]["close"] == 152.0

    def test_to_dataframe_sorted_by_timestamp(self, adapter):
        """Test to_dataframe sorts by timestamp."""
        # Create data in reverse order
        data = [
            NormalizedOHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 12, 0, 0),  # Later
                open=152.0,
                high=154.0,
                low=151.0,
                close=153.0,
                volume=1200000,
                timeframe="1h",
                market_type=MarketType.STOCK,
                data_source=DataSource.YAHOO,
            ),
            NormalizedOHLCV(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),  # Earlier
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000000,
                timeframe="1h",
                market_type=MarketType.STOCK,
                data_source=DataSource.YAHOO,
            ),
        ]

        df = adapter.to_dataframe(data)

        # Should be sorted ascending by timestamp
        assert df.iloc[0]["open"] == 150.0  # Earlier timestamp first
        assert df.iloc[1]["open"] == 152.0  # Later timestamp second

    def test_health_check_default(self, adapter):
        """Test default health check returns True."""
        assert adapter.health_check() is True

    def test_get_rate_limit_info(self, adapter):
        """Test rate limit info."""
        info = adapter.get_rate_limit_info()

        assert "requests_per_minute" in info
        assert "requests_per_day" in info
        assert info["requests_per_minute"] == 60

    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)

        assert "ConcreteAdapter" in repr_str
        assert "test" in repr_str
        assert "yahoo" in repr_str

    def test_fetch_quote(self, adapter):
        """Test fetch_quote returns NormalizedQuote."""
        quote = adapter.fetch_quote("AAPL")

        assert isinstance(quote, NormalizedQuote)
        assert quote.symbol == "AAPL"
        assert quote.bid == 100.0
        assert quote.ask == 100.1

    def test_fetch_ohlcv(self, adapter):
        """Test fetch_ohlcv returns empty list (mock)."""
        data = adapter.fetch_ohlcv("AAPL", timeframe="1h", limit=100)

        assert isinstance(data, list)
        assert len(data) == 0


class TestAdapterWithCustomTimeframe:
    """Test adapter with custom timeframe mapping."""

    def test_custom_timeframe_mapping(self):
        """Test adapter can override timeframe mapping."""

        class CustomAdapter(ConcreteAdapter):
            def map_timeframe(self, timeframe):
                mapping = {"1h": "60", "4h": "240", "1d": "D"}
                return mapping.get(timeframe, timeframe)

        adapter = CustomAdapter()
        assert adapter.map_timeframe("1h") == "60"
        assert adapter.map_timeframe("4h") == "240"
        assert adapter.map_timeframe("1d") == "D"
        assert adapter.map_timeframe("5m") == "5m"  # No mapping, returns as-is
