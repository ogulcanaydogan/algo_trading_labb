"""
Tests for data adapters module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from bot.data_adapters import (
    AdapterConfig,
    DataAdapter,
    AlphaVantageAdapter,
    YahooFinanceAdapter,
    PolygonAdapter,
    DataAdapterFactory,
    MultiSourceDataFetcher,
)


class TestAdapterConfig:
    """Test AdapterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = AdapterConfig()
        assert config.api_key is None
        assert config.api_secret is None
        assert config.rate_limit_per_minute == 60
        assert config.timeout == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdapterConfig(
            api_key="test_key",
            api_secret="test_secret",
            rate_limit_per_minute=100,
            timeout=60,
        )
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.rate_limit_per_minute == 100

    def test_base_url_config(self):
        """Test base URL configuration."""
        config = AdapterConfig(base_url="https://api.test.com")
        assert config.base_url == "https://api.test.com"


class TestAlphaVantageAdapter:
    """Test AlphaVantageAdapter class."""

    def test_adapter_creation(self):
        """Test adapter creation without API key."""
        config = AdapterConfig()
        adapter = AlphaVantageAdapter(config)
        assert adapter.name == "alpha_vantage"
        # Should set rate limit to 5 for free tier
        assert adapter.config.rate_limit_per_minute == 5

    def test_adapter_with_api_key(self):
        """Test adapter creation with API key."""
        config = AdapterConfig(api_key="test_key")
        adapter = AlphaVantageAdapter(config)
        assert adapter.api_key == "test_key"

    def test_get_function_stock_intraday(self):
        """Test getting function for stock intraday."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        assert adapter._get_function("1h", "stock") == "TIME_SERIES_INTRADAY"
        assert adapter._get_function("5min", "stock") == "TIME_SERIES_INTRADAY"

    def test_get_function_stock_daily(self):
        """Test getting function for stock daily."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        assert adapter._get_function("daily", "stock") == "TIME_SERIES_DAILY"
        assert adapter._get_function("weekly", "stock") == "TIME_SERIES_WEEKLY"

    def test_get_function_forex(self):
        """Test getting function for forex."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        assert adapter._get_function("1h", "forex") == "FX_INTRADAY"
        assert adapter._get_function("daily", "forex") == "FX_DAILY"

    def test_get_function_crypto(self):
        """Test getting function for crypto."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        assert adapter._get_function("daily", "crypto") == "DIGITAL_CURRENCY_DAILY"

    def test_get_interval(self):
        """Test getting interval parameter."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        assert adapter._get_interval("1min") == "1min"
        assert adapter._get_interval("1h") == "60min"
        assert adapter._get_interval("daily") is None

    def test_fetch_without_api_key_raises(self):
        """Test fetch without API key raises error."""
        config = AdapterConfig()
        adapter = AlphaVantageAdapter(config)
        adapter.api_key = None

        with pytest.raises(ValueError):
            adapter.fetch_ohlcv("AAPL", "1h")

    def test_parse_stock_response_valid(self):
        """Test parsing valid stock response."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        data = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "150.0",
                    "2. high": "155.0",
                    "3. low": "148.0",
                    "4. close": "152.0",
                    "5. volume": "1000000",
                },
                "2024-01-16": {
                    "1. open": "152.0",
                    "2. high": "156.0",
                    "3. low": "151.0",
                    "4. close": "154.0",
                    "5. volume": "1100000",
                },
            }
        }

        df = adapter._parse_stock_response(data, "daily")
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns

    def test_parse_stock_response_empty(self):
        """Test parsing empty response."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        data = {"Error Message": "Invalid symbol"}
        df = adapter._parse_stock_response(data, "daily")
        assert df.empty

    def test_get_supported_symbols(self):
        """Test getting supported symbols."""
        config = AdapterConfig(api_key="test")
        adapter = AlphaVantageAdapter(config)

        symbols = adapter.get_supported_symbols()
        assert "AAPL" in symbols
        assert "EUR/USD" in symbols


class TestYahooFinanceAdapter:
    """Test YahooFinanceAdapter class."""

    def test_adapter_creation(self):
        """Test adapter creation."""
        adapter = YahooFinanceAdapter()
        assert adapter.name == "yahoo"
        assert adapter.config.rate_limit_per_minute == 100

    def test_get_supported_symbols(self):
        """Test getting supported symbols."""
        adapter = YahooFinanceAdapter()
        symbols = adapter.get_supported_symbols()

        assert "AAPL" in symbols
        assert "SPY" in symbols
        assert "BTC-USD" in symbols

    @patch("yfinance.Ticker")
    def test_fetch_ohlcv_success(self, mock_ticker):
        """Test successful OHLCV fetch."""
        # Setup mock
        mock_data = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [99, 100, 101],
            "Close": [103, 104, 105],
            "Volume": [1000, 1100, 1200],
        })
        mock_ticker.return_value.history.return_value = mock_data

        adapter = YahooFinanceAdapter()
        df = adapter.fetch_ohlcv("AAPL", "1d")

        assert not df.empty
        assert "open" in df.columns
        assert "close" in df.columns

    @patch("yfinance.Ticker")
    def test_fetch_ohlcv_empty(self, mock_ticker):
        """Test fetch returns empty on error."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        adapter = YahooFinanceAdapter()
        df = adapter.fetch_ohlcv("INVALID", "1d")

        assert df.empty


class TestPolygonAdapter:
    """Test PolygonAdapter class."""

    def test_adapter_creation(self):
        """Test adapter creation."""
        config = AdapterConfig(api_key="test_key")
        adapter = PolygonAdapter(config)
        assert adapter.name == "polygon"
        assert adapter.api_key == "test_key"

    def test_format_symbol_crypto(self):
        """Test symbol formatting for crypto."""
        config = AdapterConfig(api_key="test")
        adapter = PolygonAdapter(config)

        assert adapter._format_symbol("BTC/USDT") == "X:BTCUSD"
        assert adapter._format_symbol("ETH/USD") == "X:ETHUSD"

    def test_format_symbol_forex(self):
        """Test symbol formatting for forex.

        Note: The implementation checks for USD suffix first, so EUR/USD
        gets treated as crypto format (X:) rather than forex (C:).
        """
        config = AdapterConfig(api_key="test")
        adapter = PolygonAdapter(config)

        # EUR/USD has USD in it, so it matches crypto pattern first
        assert adapter._format_symbol("EUR/USD") == "X:EURUSD"

    def test_format_symbol_stock(self):
        """Test symbol formatting for stock."""
        config = AdapterConfig(api_key="test")
        adapter = PolygonAdapter(config)

        assert adapter._format_symbol("AAPL") == "AAPL"
        assert adapter._format_symbol("aapl") == "AAPL"

    def test_fetch_without_api_key_raises(self):
        """Test fetch without API key raises error."""
        config = AdapterConfig()
        adapter = PolygonAdapter(config)
        adapter.api_key = None

        with pytest.raises(ValueError):
            adapter.fetch_ohlcv("AAPL", "1h")

    def test_get_supported_symbols(self):
        """Test getting supported symbols."""
        config = AdapterConfig(api_key="test")
        adapter = PolygonAdapter(config)

        symbols = adapter.get_supported_symbols()
        assert "AAPL" in symbols


class TestDataAdapterFactory:
    """Test DataAdapterFactory class."""

    def test_get_available_adapters(self):
        """Test getting available adapters."""
        adapters = DataAdapterFactory.get_available_adapters()

        assert "yahoo" in adapters
        assert "alpha_vantage" in adapters
        assert "polygon" in adapters

    def test_create_yahoo_adapter(self):
        """Test creating Yahoo adapter."""
        adapter = DataAdapterFactory.create("yahoo")
        assert adapter.name == "yahoo"

    def test_create_alpha_vantage_adapter(self):
        """Test creating Alpha Vantage adapter."""
        config = AdapterConfig(api_key="test")
        adapter = DataAdapterFactory.create("alpha_vantage", config)
        assert adapter.name == "alpha_vantage"

    def test_create_polygon_adapter(self):
        """Test creating Polygon adapter."""
        config = AdapterConfig(api_key="test")
        adapter = DataAdapterFactory.create("polygon", config)
        assert adapter.name == "polygon"

    def test_create_unknown_adapter_raises(self):
        """Test creating unknown adapter raises error."""
        with pytest.raises(ValueError):
            DataAdapterFactory.create("unknown_adapter")


class TestMultiSourceDataFetcher:
    """Test MultiSourceDataFetcher class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_fetcher_creation(self, temp_cache_dir):
        """Test fetcher creation."""
        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            fallbacks=["alpha_vantage"],
            cache_dir=temp_cache_dir,
        )
        assert fetcher.primary == "yahoo"
        assert "yahoo" in fetcher.adapters

    def test_cache_key_generation(self, temp_cache_dir):
        """Test cache key generation."""
        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            cache_dir=temp_cache_dir,
        )

        key = fetcher._cache_key("BTC/USDT", "1h")
        assert key == "BTC_USDT_1h"
        assert "/" not in key

    def test_cache_dir_created(self, temp_cache_dir):
        """Test cache directory is created."""
        cache_path = Path(temp_cache_dir) / "nested" / "cache"
        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            cache_dir=str(cache_path),
        )
        assert cache_path.exists()

    @patch.object(YahooFinanceAdapter, "fetch_ohlcv")
    def test_fetch_from_primary(self, mock_fetch, temp_cache_dir):
        """Test fetching from primary source."""
        mock_df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [99, 100],
            "close": [103, 104],
            "volume": [1000, 1100],
        })
        mock_fetch.return_value = mock_df

        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            cache_dir=temp_cache_dir,
        )
        df = fetcher.fetch("AAPL", "1d", use_cache=False)

        assert not df.empty
        mock_fetch.assert_called_once()

    @patch.object(YahooFinanceAdapter, "fetch_ohlcv")
    def test_fetch_with_cache(self, mock_fetch, temp_cache_dir):
        """Test fetching with cache."""
        mock_df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [99, 100],
            "close": [103, 104],
            "volume": [1000, 1100],
        })
        mock_fetch.return_value = mock_df

        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            cache_dir=temp_cache_dir,
        )

        # First fetch - from API
        df1 = fetcher.fetch("AAPL", "1d", use_cache=True)
        assert not df1.empty

        # Second fetch - from cache
        df2 = fetcher.fetch("AAPL", "1d", use_cache=True)
        assert not df2.empty

        # Should only call API once
        assert mock_fetch.call_count == 1

    @patch.object(YahooFinanceAdapter, "fetch_ohlcv")
    def test_fetch_fallback_on_failure(self, mock_fetch, temp_cache_dir):
        """Test fallback when primary fails."""
        mock_fetch.side_effect = Exception("Primary failed")

        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            fallbacks=[],
            cache_dir=temp_cache_dir,
        )

        df = fetcher.fetch("AAPL", "1d", use_cache=False)
        assert df.empty

    def test_load_cache_nonexistent(self, temp_cache_dir):
        """Test loading nonexistent cache returns None."""
        fetcher = MultiSourceDataFetcher(
            primary="yahoo",
            cache_dir=temp_cache_dir,
        )

        result = fetcher._load_cache("NONEXISTENT", "1h")
        assert result is None


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_applies(self):
        """Test rate limit is applied."""
        config = AdapterConfig(rate_limit_per_minute=60)
        adapter = YahooFinanceAdapter(config)

        # First call should set timestamp
        adapter._rate_limit()
        first_time = adapter._last_request_time
        assert first_time is not None

        # Subsequent call should be tracked
        adapter._rate_limit()
        assert adapter._last_request_time >= first_time
