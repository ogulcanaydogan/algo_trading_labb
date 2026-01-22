"""
Tests for data models module.
"""

import pytest
from datetime import datetime

from bot.data.models import (
    MarketType,
    DataSource,
    NormalizedOHLCV,
    NormalizedQuote,
    SymbolInfo,
    get_symbol_info,
    get_symbols_by_market,
    ALL_SYMBOLS,
    CRYPTO_SYMBOLS,
    STOCK_SYMBOLS,
    COMMODITY_SYMBOLS,
)


class TestMarketType:
    """Test MarketType enum."""

    def test_market_type_values(self):
        """Test market type enum values."""
        assert MarketType.CRYPTO.value == "crypto"
        assert MarketType.STOCK.value == "stock"
        assert MarketType.COMMODITY.value == "commodity"
        assert MarketType.FOREX.value == "forex"
        assert MarketType.INDEX.value == "index"

    def test_market_type_uniqueness(self):
        """Test all market types are unique."""
        values = [m.value for m in MarketType]
        assert len(values) == len(set(values))


class TestDataSource:
    """Test DataSource enum."""

    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.CCXT.value == "ccxt"
        assert DataSource.YAHOO.value == "yahoo"
        assert DataSource.POLYGON.value == "polygon"
        assert DataSource.IEX.value == "iex"
        assert DataSource.ALPHA_VANTAGE.value == "alpha_vantage"
        assert DataSource.OANDA.value == "oanda"
        assert DataSource.FINNHUB.value == "finnhub"


class TestNormalizedOHLCV:
    """Test NormalizedOHLCV dataclass."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        return NormalizedOHLCV(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            open=42000.0,
            high=43000.0,
            low=41500.0,
            close=42800.0,
            volume=1000000.0,
            timeframe="1h",
            market_type=MarketType.CRYPTO,
            data_source=DataSource.CCXT,
            exchange="binance",
        )

    def test_basic_creation(self, sample_ohlcv):
        """Test basic OHLCV creation."""
        assert sample_ohlcv.symbol == "BTC/USDT"
        assert sample_ohlcv.open == 42000.0
        assert sample_ohlcv.high == 43000.0
        assert sample_ohlcv.low == 41500.0
        assert sample_ohlcv.close == 42800.0
        assert sample_ohlcv.volume == 1000000.0
        assert sample_ohlcv.timeframe == "1h"

    def test_optional_fields(self, sample_ohlcv):
        """Test optional fields default to None."""
        assert sample_ohlcv.adjusted_close is None
        assert sample_ohlcv.vwap is None
        assert sample_ohlcv.trade_count is None
        assert sample_ohlcv.latency_ms is None
        assert sample_ohlcv.raw_data is None

    def test_to_dict(self, sample_ohlcv):
        """Test conversion to dict."""
        d = sample_ohlcv.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["open"] == 42000.0
        assert d["high"] == 43000.0
        assert d["low"] == 41500.0
        assert d["close"] == 42800.0
        assert d["volume"] == 1000000.0
        assert d["timeframe"] == "1h"
        assert d["market_type"] == "crypto"
        assert d["data_source"] == "ccxt"
        assert d["exchange"] == "binance"
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)

    def test_from_dict(self, sample_ohlcv):
        """Test creation from dict."""
        d = sample_ohlcv.to_dict()
        restored = NormalizedOHLCV.from_dict(d)

        assert restored.symbol == sample_ohlcv.symbol
        assert restored.open == sample_ohlcv.open
        assert restored.close == sample_ohlcv.close
        assert restored.market_type == sample_ohlcv.market_type

    def test_from_dict_with_string_enums(self):
        """Test from_dict handles string enum values."""
        d = {
            "symbol": "ETH/USDT",
            "timestamp": "2024-01-15T10:00:00",
            "open": 2500.0,
            "high": 2600.0,
            "low": 2400.0,
            "close": 2550.0,
            "volume": 50000.0,
            "timeframe": "1h",
            "market_type": "crypto",  # String instead of enum
            "data_source": "ccxt",  # String instead of enum
        }

        ohlcv = NormalizedOHLCV.from_dict(d)

        assert ohlcv.market_type == MarketType.CRYPTO
        assert ohlcv.data_source == DataSource.CCXT

    def test_with_optional_fields(self):
        """Test OHLCV with optional fields filled."""
        ohlcv = NormalizedOHLCV(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=148.0,
            close=153.0,
            volume=10000000.0,
            timeframe="1d",
            market_type=MarketType.STOCK,
            data_source=DataSource.YAHOO,
            adjusted_close=152.5,
            vwap=151.0,
            trade_count=50000,
            latency_ms=25.5,
        )

        assert ohlcv.adjusted_close == 152.5
        assert ohlcv.vwap == 151.0
        assert ohlcv.trade_count == 50000
        assert ohlcv.latency_ms == 25.5


class TestNormalizedQuote:
    """Test NormalizedQuote dataclass."""

    @pytest.fixture
    def sample_quote(self):
        """Create sample quote."""
        return NormalizedQuote(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            bid=42000.0,
            ask=42010.0,
            bid_size=1.5,
            ask_size=2.0,
            last_price=42005.0,
            last_size=0.5,
            market_type=MarketType.CRYPTO,
            data_source=DataSource.CCXT,
            exchange="binance",
        )

    def test_basic_creation(self, sample_quote):
        """Test basic quote creation."""
        assert sample_quote.symbol == "BTC/USDT"
        assert sample_quote.bid == 42000.0
        assert sample_quote.ask == 42010.0

    def test_mid_price(self, sample_quote):
        """Test mid price calculation."""
        expected_mid = (42000.0 + 42010.0) / 2
        assert sample_quote.mid_price == expected_mid

    def test_spread(self, sample_quote):
        """Test spread calculation."""
        expected_spread = 42010.0 - 42000.0
        assert sample_quote.spread == expected_spread

    def test_spread_pct(self, sample_quote):
        """Test spread percentage calculation."""
        mid = (42000.0 + 42010.0) / 2
        expected_spread_pct = (10.0 / mid) * 100
        assert sample_quote.spread_pct == pytest.approx(expected_spread_pct)

    def test_spread_pct_zero_mid(self):
        """Test spread percentage with zero mid price."""
        quote = NormalizedQuote(
            symbol="TEST",
            timestamp=datetime.now(),
            bid=0.0,
            ask=0.0,
        )
        assert quote.spread_pct == 0.0

    def test_to_dict(self, sample_quote):
        """Test conversion to dict."""
        d = sample_quote.to_dict()

        assert d["symbol"] == "BTC/USDT"
        assert d["bid"] == 42000.0
        assert d["ask"] == 42010.0
        assert "mid_price" in d
        assert "spread" in d
        assert "spread_pct" in d
        assert d["market_type"] == "crypto"


class TestSymbolInfo:
    """Test SymbolInfo dataclass."""

    @pytest.fixture
    def sample_symbol(self):
        """Create sample symbol info."""
        return SymbolInfo(
            standard_symbol="BTC/USDT",
            market_type=MarketType.CRYPTO,
            exchange="binance",
            currency="USDT",
            provider_mappings={
                "ccxt": "BTC/USDT",
                "yahoo": "BTC-USD",
            },
            min_tick_size=0.01,
            lot_size=0.001,
        )

    def test_basic_creation(self, sample_symbol):
        """Test basic symbol info creation."""
        assert sample_symbol.standard_symbol == "BTC/USDT"
        assert sample_symbol.market_type == MarketType.CRYPTO
        assert sample_symbol.exchange == "binance"
        assert sample_symbol.currency == "USDT"
        assert sample_symbol.is_active is True

    def test_get_provider_symbol(self, sample_symbol):
        """Test getting provider-specific symbol."""
        assert sample_symbol.get_provider_symbol("ccxt") == "BTC/USDT"
        assert sample_symbol.get_provider_symbol("yahoo") == "BTC-USD"
        assert sample_symbol.get_provider_symbol("unknown") is None

    def test_is_market_open_crypto(self, sample_symbol):
        """Test crypto is always open."""
        assert sample_symbol.is_market_open() is True

    def test_is_market_open_stock(self):
        """Test stock market hours (defaults to open when no hours specified)."""
        stock = SymbolInfo(
            standard_symbol="AAPL",
            market_type=MarketType.STOCK,
            exchange="nasdaq",
            currency="USD",
            provider_mappings={"yahoo": "AAPL"},
        )
        # No trading hours specified, defaults to open
        assert stock.is_market_open() is True

    def test_is_market_open_with_hours(self):
        """Test market hours check with hours specified."""
        import pytz
        from datetime import datetime

        stock = SymbolInfo(
            standard_symbol="AAPL",
            market_type=MarketType.STOCK,
            exchange="nasdaq",
            currency="USD",
            provider_mappings={"yahoo": "AAPL"},
            trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        )

        # Test with specific times for deterministic results
        eastern = pytz.timezone('US/Eastern')

        # Monday at 10:00 AM Eastern - should be open
        market_open_time = eastern.localize(datetime(2024, 1, 15, 10, 0))
        assert stock.is_market_open(market_open_time) is True

        # Saturday at 10:00 AM Eastern - should be closed (weekend)
        weekend_time = eastern.localize(datetime(2024, 1, 13, 10, 0))
        assert stock.is_market_open(weekend_time) is False

        # Monday at 6:00 PM Eastern - should be closed (after hours)
        after_hours_time = eastern.localize(datetime(2024, 1, 15, 18, 0))
        assert stock.is_market_open(after_hours_time) is False

        # Monday at 8:00 AM Eastern - should be closed (before hours)
        before_hours_time = eastern.localize(datetime(2024, 1, 15, 8, 0))
        assert stock.is_market_open(before_hours_time) is False

    def test_to_dict(self, sample_symbol):
        """Test conversion to dict."""
        d = sample_symbol.to_dict()

        assert d["standard_symbol"] == "BTC/USDT"
        assert d["market_type"] == "crypto"
        assert d["exchange"] == "binance"
        assert d["currency"] == "USDT"
        assert d["provider_mappings"]["ccxt"] == "BTC/USDT"
        assert d["is_active"] is True


class TestSymbolConstants:
    """Test symbol constant dictionaries."""

    def test_crypto_symbols_exist(self):
        """Test crypto symbols are defined."""
        assert "BTC/USDT" in CRYPTO_SYMBOLS
        assert "ETH/USDT" in CRYPTO_SYMBOLS
        assert "SOL/USDT" in CRYPTO_SYMBOLS

    def test_stock_symbols_exist(self):
        """Test stock symbols are defined."""
        assert "AAPL" in STOCK_SYMBOLS
        assert "MSFT" in STOCK_SYMBOLS
        assert "GOOGL" in STOCK_SYMBOLS

    def test_commodity_symbols_exist(self):
        """Test commodity symbols are defined."""
        assert "XAU/USD" in COMMODITY_SYMBOLS
        assert "XAG/USD" in COMMODITY_SYMBOLS
        assert "USOIL/USD" in COMMODITY_SYMBOLS

    def test_all_symbols_combined(self):
        """Test ALL_SYMBOLS combines all symbol dicts."""
        assert "BTC/USDT" in ALL_SYMBOLS  # Crypto
        assert "AAPL" in ALL_SYMBOLS  # Stock
        assert "XAU/USD" in ALL_SYMBOLS  # Commodity

    def test_crypto_symbols_have_correct_type(self):
        """Test crypto symbols have correct market type."""
        for symbol, info in CRYPTO_SYMBOLS.items():
            assert info.market_type == MarketType.CRYPTO

    def test_stock_symbols_have_correct_type(self):
        """Test stock symbols have correct market type."""
        for symbol, info in STOCK_SYMBOLS.items():
            assert info.market_type == MarketType.STOCK

    def test_commodity_symbols_have_correct_type(self):
        """Test commodity symbols have correct market type."""
        for symbol, info in COMMODITY_SYMBOLS.items():
            assert info.market_type == MarketType.COMMODITY


class TestSymbolLookupFunctions:
    """Test symbol lookup helper functions."""

    def test_get_symbol_info_exists(self):
        """Test getting existing symbol info."""
        info = get_symbol_info("BTC/USDT")

        assert info is not None
        assert info.standard_symbol == "BTC/USDT"
        assert info.market_type == MarketType.CRYPTO

    def test_get_symbol_info_not_exists(self):
        """Test getting non-existent symbol info."""
        info = get_symbol_info("FAKE/SYMBOL")

        assert info is None

    def test_get_symbols_by_market_crypto(self):
        """Test getting crypto symbols by market type."""
        crypto = get_symbols_by_market(MarketType.CRYPTO)

        assert len(crypto) > 0
        assert "BTC/USDT" in crypto
        for info in crypto.values():
            assert info.market_type == MarketType.CRYPTO

    def test_get_symbols_by_market_stock(self):
        """Test getting stock symbols by market type."""
        stocks = get_symbols_by_market(MarketType.STOCK)

        assert len(stocks) > 0
        assert "AAPL" in stocks
        for info in stocks.values():
            assert info.market_type == MarketType.STOCK

    def test_get_symbols_by_market_commodity(self):
        """Test getting commodity symbols by market type."""
        commodities = get_symbols_by_market(MarketType.COMMODITY)

        assert len(commodities) > 0
        assert "XAU/USD" in commodities
        for info in commodities.values():
            assert info.market_type == MarketType.COMMODITY


class TestSymbolMetadata:
    """Test symbol metadata."""

    def test_stock_trading_hours(self):
        """Test stock symbols have trading hours."""
        aapl = get_symbol_info("AAPL")

        assert aapl.trading_hours is not None
        assert "open" in aapl.trading_hours
        assert "close" in aapl.trading_hours
        assert "timezone" in aapl.trading_hours

    def test_commodity_metadata(self):
        """Test commodity symbols have metadata."""
        gold = get_symbol_info("XAU/USD")

        assert gold.metadata is not None
        assert "name" in gold.metadata
        assert gold.metadata["name"] == "Gold"

    def test_symbol_provider_mappings(self):
        """Test symbols have provider mappings."""
        btc = get_symbol_info("BTC/USDT")

        assert "ccxt" in btc.provider_mappings
        assert "yahoo" in btc.provider_mappings
        assert btc.provider_mappings["yahoo"] == "BTC-USD"

    def test_stock_has_multiple_providers(self):
        """Test stocks have multiple provider mappings."""
        aapl = get_symbol_info("AAPL")

        assert "yahoo" in aapl.provider_mappings
        # Some stocks have polygon too
        if "polygon" in aapl.provider_mappings:
            assert aapl.provider_mappings["polygon"] == "AAPL"
