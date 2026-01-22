"""
Normalized data models for multi-market trading platform.

Provides unified data structures across all data providers (CCXT, Yahoo, Polygon).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketType(Enum):
    """Market type classification."""
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    FOREX = "forex"
    INDEX = "index"


class DataSource(Enum):
    """Data provider source."""
    CCXT = "ccxt"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    IEX = "iex"
    ALPHA_VANTAGE = "alpha_vantage"
    OANDA = "oanda"
    FINNHUB = "finnhub"


@dataclass
class NormalizedOHLCV:
    """
    Unified OHLCV data format across all providers.

    This is the standard format used throughout the trading platform
    regardless of the original data source.
    """
    symbol: str                    # Standard symbol format (e.g., "BTC/USDT", "AAPL", "XAU/USD")
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str                 # "1m", "5m", "15m", "1h", "4h", "1d"
    market_type: MarketType
    data_source: DataSource
    exchange: Optional[str] = None
    currency: str = "USD"
    adjusted_close: Optional[float] = None  # For stocks (split/dividend adjusted)
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    latency_ms: Optional[float] = None      # Data latency for monitoring
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe,
            "market_type": self.market_type.value,
            "data_source": self.data_source.value,
            "exchange": self.exchange,
            "currency": self.currency,
            "adjusted_close": self.adjusted_close,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedOHLCV":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            timeframe=data["timeframe"],
            market_type=MarketType(data["market_type"]) if isinstance(data["market_type"], str) else data["market_type"],
            data_source=DataSource(data["data_source"]) if isinstance(data["data_source"], str) else data["data_source"],
            exchange=data.get("exchange"),
            currency=data.get("currency", "USD"),
            adjusted_close=data.get("adjusted_close"),
            vwap=data.get("vwap"),
            trade_count=data.get("trade_count"),
            latency_ms=data.get("latency_ms"),
        )


@dataclass
class NormalizedQuote:
    """
    Real-time quote data.

    Used for current price snapshots and streaming updates.
    """
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    market_type: MarketType = MarketType.CRYPTO
    data_source: DataSource = DataSource.CCXT
    exchange: Optional[str] = None

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last_price": self.last_price,
            "last_size": self.last_size,
            "market_type": self.market_type.value,
            "data_source": self.data_source.value,
            "exchange": self.exchange,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
        }


@dataclass
class SymbolInfo:
    """
    Symbol metadata and provider mappings.

    Maps standard symbols to provider-specific formats and stores
    trading metadata like hours, tick sizes, etc.
    """
    standard_symbol: str              # Our internal format (e.g., "BTC/USDT", "AAPL", "XAU/USD")
    market_type: MarketType
    exchange: str                     # "binance", "nyse", "nasdaq", "lse", "cme"
    currency: str
    provider_mappings: Dict[str, str] # {provider_name: provider_symbol}
    trading_hours: Optional[Dict[str, str]] = None  # {"open": "09:30", "close": "16:00", "timezone": "US/Eastern"}
    min_tick_size: Optional[float] = None
    lot_size: Optional[float] = None
    margin_requirement: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def get_provider_symbol(self, provider: str) -> Optional[str]:
        """Get symbol format for a specific provider."""
        return self.provider_mappings.get(provider)

    def is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open (for stocks).

        Args:
            current_time: Time to check (defaults to now in UTC)

        Returns:
            True if market is open, False otherwise
        """
        if self.market_type == MarketType.CRYPTO:
            return True  # Crypto is 24/7

        if not self.trading_hours:
            return True  # Assume open if no hours specified

        try:
            import pytz

            # Get current time in market timezone
            tz_name = self.trading_hours.get("timezone", "US/Eastern")
            market_tz = pytz.timezone(tz_name)

            if current_time is None:
                current_time = datetime.now(pytz.UTC)
            elif current_time.tzinfo is None:
                current_time = pytz.UTC.localize(current_time)

            local_time = current_time.astimezone(market_tz)

            # Check if weekend (most stock markets closed Sat/Sun)
            if local_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

            # Parse trading hours
            open_str = self.trading_hours.get("open", "09:30")
            close_str = self.trading_hours.get("close", "16:00")

            open_hour, open_min = map(int, open_str.split(":"))
            close_hour, close_min = map(int, close_str.split(":"))

            # Create time objects for comparison
            market_open = local_time.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
            market_close = local_time.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)

            return market_open <= local_time <= market_close

        except Exception:
            # On any error, assume market is open to avoid blocking trades
            return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard_symbol": self.standard_symbol,
            "market_type": self.market_type.value,
            "exchange": self.exchange,
            "currency": self.currency,
            "provider_mappings": self.provider_mappings,
            "trading_hours": self.trading_hours,
            "min_tick_size": self.min_tick_size,
            "lot_size": self.lot_size,
            "margin_requirement": self.margin_requirement,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }


# Symbol mapping constants for common assets
CRYPTO_SYMBOLS = {
    "BTC/USDT": SymbolInfo(
        standard_symbol="BTC/USDT",
        market_type=MarketType.CRYPTO,
        exchange="binance",
        currency="USDT",
        provider_mappings={"ccxt": "BTC/USDT", "yahoo": "BTC-USD"},
    ),
    "ETH/USDT": SymbolInfo(
        standard_symbol="ETH/USDT",
        market_type=MarketType.CRYPTO,
        exchange="binance",
        currency="USDT",
        provider_mappings={"ccxt": "ETH/USDT", "yahoo": "ETH-USD"},
    ),
    "SOL/USDT": SymbolInfo(
        standard_symbol="SOL/USDT",
        market_type=MarketType.CRYPTO,
        exchange="binance",
        currency="USDT",
        provider_mappings={"ccxt": "SOL/USDT", "yahoo": "SOL-USD"},
    ),
    "AVAX/USDT": SymbolInfo(
        standard_symbol="AVAX/USDT",
        market_type=MarketType.CRYPTO,
        exchange="binance",
        currency="USDT",
        provider_mappings={"ccxt": "AVAX/USDT", "yahoo": "AVAX-USD"},
    ),
}

COMMODITY_SYMBOLS = {
    "XAU/USD": SymbolInfo(
        standard_symbol="XAU/USD",
        market_type=MarketType.COMMODITY,
        exchange="cme",
        currency="USD",
        provider_mappings={"yahoo": "GC=F", "oanda": "XAU_USD"},
        metadata={"name": "Gold", "unit": "oz"},
    ),
    "XAG/USD": SymbolInfo(
        standard_symbol="XAG/USD",
        market_type=MarketType.COMMODITY,
        exchange="cme",
        currency="USD",
        provider_mappings={"yahoo": "SI=F", "oanda": "XAG_USD"},
        metadata={"name": "Silver", "unit": "oz"},
    ),
    "USOIL/USD": SymbolInfo(
        standard_symbol="USOIL/USD",
        market_type=MarketType.COMMODITY,
        exchange="cme",
        currency="USD",
        provider_mappings={"yahoo": "CL=F", "oanda": "WTICO_USD"},
        metadata={"name": "Crude Oil WTI", "unit": "barrel"},
    ),
    "UKOIL/USD": SymbolInfo(
        standard_symbol="UKOIL/USD",
        market_type=MarketType.COMMODITY,
        exchange="ice",
        currency="USD",
        provider_mappings={"yahoo": "BZ=F", "oanda": "BCO_USD"},
        metadata={"name": "Crude Oil Brent", "unit": "barrel"},
    ),
    "NATGAS/USD": SymbolInfo(
        standard_symbol="NATGAS/USD",
        market_type=MarketType.COMMODITY,
        exchange="cme",
        currency="USD",
        provider_mappings={"yahoo": "NG=F"},
        metadata={"name": "Natural Gas", "unit": "MMBtu"},
    ),
    "COPPER/USD": SymbolInfo(
        standard_symbol="COPPER/USD",
        market_type=MarketType.COMMODITY,
        exchange="cme",
        currency="USD",
        provider_mappings={"yahoo": "HG=F"},
        metadata={"name": "Copper", "unit": "lb"},
    ),
}

STOCK_SYMBOLS = {
    "AAPL": SymbolInfo(
        standard_symbol="AAPL",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "AAPL", "polygon": "AAPL"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Apple Inc.", "sector": "Technology"},
    ),
    "MSFT": SymbolInfo(
        standard_symbol="MSFT",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "MSFT", "polygon": "MSFT"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Microsoft Corporation", "sector": "Technology"},
    ),
    "GOOGL": SymbolInfo(
        standard_symbol="GOOGL",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "GOOGL", "polygon": "GOOGL"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Alphabet Inc.", "sector": "Technology"},
    ),
    "AMZN": SymbolInfo(
        standard_symbol="AMZN",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "AMZN", "polygon": "AMZN"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    ),
    "NVDA": SymbolInfo(
        standard_symbol="NVDA",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "NVDA", "polygon": "NVDA"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "NVIDIA Corporation", "sector": "Technology"},
    ),
    "TSLA": SymbolInfo(
        standard_symbol="TSLA",
        market_type=MarketType.STOCK,
        exchange="nasdaq",
        currency="USD",
        provider_mappings={"yahoo": "TSLA", "polygon": "TSLA"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    ),
    # LSE stocks
    "VOD.L": SymbolInfo(
        standard_symbol="VOD.L",
        market_type=MarketType.STOCK,
        exchange="lse",
        currency="GBP",
        provider_mappings={"yahoo": "VOD.L"},
        trading_hours={"open": "08:00", "close": "16:30", "timezone": "Europe/London"},
        metadata={"name": "Vodafone Group Plc", "sector": "Telecommunications"},
    ),
    "HSBA.L": SymbolInfo(
        standard_symbol="HSBA.L",
        market_type=MarketType.STOCK,
        exchange="lse",
        currency="GBP",
        provider_mappings={"yahoo": "HSBA.L"},
        trading_hours={"open": "08:00", "close": "16:30", "timezone": "Europe/London"},
        metadata={"name": "HSBC Holdings", "sector": "Financial Services"},
    ),
    "BP.L": SymbolInfo(
        standard_symbol="BP.L",
        market_type=MarketType.STOCK,
        exchange="lse",
        currency="GBP",
        provider_mappings={"yahoo": "BP.L"},
        trading_hours={"open": "08:00", "close": "16:30", "timezone": "Europe/London"},
        metadata={"name": "BP Plc", "sector": "Energy"},
    ),
    # NYSE stocks
    "JPM": SymbolInfo(
        standard_symbol="JPM",
        market_type=MarketType.STOCK,
        exchange="nyse",
        currency="USD",
        provider_mappings={"yahoo": "JPM", "polygon": "JPM"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    ),
    "V": SymbolInfo(
        standard_symbol="V",
        market_type=MarketType.STOCK,
        exchange="nyse",
        currency="USD",
        provider_mappings={"yahoo": "V", "polygon": "V"},
        trading_hours={"open": "09:30", "close": "16:00", "timezone": "US/Eastern"},
        metadata={"name": "Visa Inc.", "sector": "Financial Services"},
    ),
}

# Combine all symbols
ALL_SYMBOLS = {**CRYPTO_SYMBOLS, **COMMODITY_SYMBOLS, **STOCK_SYMBOLS}


def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Get symbol info by standard symbol."""
    return ALL_SYMBOLS.get(symbol)


def get_symbols_by_market(market_type: MarketType) -> Dict[str, SymbolInfo]:
    """Get all symbols for a market type."""
    return {
        k: v for k, v in ALL_SYMBOLS.items()
        if v.market_type == market_type
    }
