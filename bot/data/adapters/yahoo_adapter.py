"""
Yahoo Finance data adapter.

Provides market data for stocks, commodities, and crypto via yfinance.
Free tier with 15-minute delay for most assets.

Features:
- Rate limiting to avoid 429 errors
- Multi-level caching (memory + disk)
- Automatic retry with exponential backoff
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from ..models import (
    ALL_SYMBOLS,
    COMMODITY_SYMBOLS,
    CRYPTO_SYMBOLS,
    STOCK_SYMBOLS,
    DataSource,
    MarketType,
    NormalizedOHLCV,
    NormalizedQuote,
    SymbolInfo,
)
from .base import DataAdapter

# Import rate limiter and cache utilities
try:
    from ...rate_limiter import (
        get_yahoo_rate_limiter,
        MultiLevelCache,
        with_retry,
    )

    HAS_RATE_LIMITER = True
except ImportError:
    HAS_RATE_LIMITER = False

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(os.getenv("DATA_DIR", "./data")) / "cache"


class YahooAdapter(DataAdapter):
    """
    Yahoo Finance adapter for stocks, commodities, and crypto backup.

    Features:
    - Free tier (no API key required)
    - 15-minute delay for real-time data
    - Good historical data coverage
    - Supports US/LSE stocks, commodities futures, and crypto
    """

    name = "yahoo"
    data_source = DataSource.YAHOO
    supported_markets = [
        MarketType.STOCK,
        MarketType.COMMODITY,
        MarketType.CRYPTO,
        MarketType.INDEX,
    ]
    supports_realtime = False  # 15-minute delay
    supports_historical = True
    priority = 5  # Lower priority than real-time providers

    # Symbol mapping from our standard format to Yahoo format
    SYMBOL_MAP: Dict[str, str] = {
        # Crypto
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "AVAX/USDT": "AVAX-USD",
        "BNB/USDT": "BNB-USD",
        "XRP/USDT": "XRP-USD",
        "ADA/USDT": "ADA-USD",
        "DOGE/USDT": "DOGE-USD",
        # Commodities
        "XAU/USD": "GC=F",  # Gold futures
        "XAG/USD": "SI=F",  # Silver futures
        "USOIL/USD": "CL=F",  # WTI Crude Oil futures
        "UKOIL/USD": "BZ=F",  # Brent Crude Oil futures
        "NATGAS/USD": "NG=F",  # Natural Gas futures
        "COPPER/USD": "HG=F",  # Copper futures
        "PLATINUM/USD": "PL=F",  # Platinum futures
        "PALLADIUM/USD": "PA=F",  # Palladium futures
        # Stocks - map from internal format to Yahoo format
        "AAPL/USD": "AAPL",  # Apple Inc.
        "MSFT/USD": "MSFT",  # Microsoft Corporation
        "GOOGL/USD": "GOOGL",  # Alphabet Inc. (Google)
        "GOOG/USD": "GOOG",  # Alphabet Class C
        "AMZN/USD": "AMZN",  # Amazon.com Inc.
        "NVDA/USD": "NVDA",  # NVIDIA Corporation
        "META/USD": "META",  # Meta Platforms Inc.
        "TSLA/USD": "TSLA",  # Tesla Inc.
        "JPM/USD": "JPM",  # JPMorgan Chase
        "V/USD": "V",  # Visa Inc.
        "JNJ/USD": "JNJ",  # Johnson & Johnson
        "UNH/USD": "UNH",  # UnitedHealth Group
        "HD/USD": "HD",  # Home Depot
        "PG/USD": "PG",  # Procter & Gamble
        "MA/USD": "MA",  # Mastercard
        "BAC/USD": "BAC",  # Bank of America
        "XOM/USD": "XOM",  # Exxon Mobil
        "CVX/USD": "CVX",  # Chevron
        "KO/USD": "KO",  # Coca-Cola
        "PEP/USD": "PEP",  # PepsiCo
        "ABBV/USD": "ABBV",  # AbbVie
        "MRK/USD": "MRK",  # Merck
        "WMT/USD": "WMT",  # Walmart
        "COST/USD": "COST",  # Costco
        "DIS/USD": "DIS",  # Disney
        "NFLX/USD": "NFLX",  # Netflix
        "AMD/USD": "AMD",  # AMD
        "INTC/USD": "INTC",  # Intel
    }

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",  # Not directly supported, will use 1h
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
    }

    # Period mapping for historical data
    PERIOD_MAP = {
        "1m": "7d",  # Max 7 days for 1m
        "2m": "60d",  # Max 60 days for 2m
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",  # Max 2 years for 1h
        "4h": "730d",
        "1d": "max",
        "1w": "max",
        "1M": "max",
    }

    def __init__(self, use_cache: bool = True, use_rate_limiter: bool = True):
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        self.use_cache = use_cache
        self.use_rate_limiter = use_rate_limiter and HAS_RATE_LIMITER

        # Initialize rate limiter
        if self.use_rate_limiter:
            self._rate_limiter = get_yahoo_rate_limiter()
        else:
            self._rate_limiter = None

        # Initialize cache
        if self.use_cache and HAS_RATE_LIMITER:
            self._cache = MultiLevelCache(
                cache_dir=CACHE_DIR,
                memory_max_size=200,
                memory_ttl=60.0,  # 1 minute in memory
                disk_ttl=300.0,  # 5 minutes on disk (quotes change frequently)
            )
            self._ohlcv_cache = MultiLevelCache(
                cache_dir=CACHE_DIR,
                memory_max_size=100,
                memory_ttl=300.0,  # 5 minutes in memory for OHLCV
                disk_ttl=3600.0,  # 1 hour on disk for OHLCV
            )
        else:
            self._cache = None
            self._ohlcv_cache = None

        logger.info(
            f"YahooAdapter initialized (rate_limiter={self.use_rate_limiter}, cache={self.use_cache})"
        )

    def map_symbol(self, standard_symbol: str) -> str:
        """Convert standard symbol to Yahoo format."""
        # Check explicit mapping first
        if standard_symbol in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[standard_symbol]

        # Check our symbol registry
        symbol_info = ALL_SYMBOLS.get(standard_symbol)
        if symbol_info:
            yahoo_symbol = symbol_info.provider_mappings.get("yahoo")
            if yahoo_symbol:
                return yahoo_symbol

        # For stocks, symbol usually maps directly
        return standard_symbol

    def map_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Yahoo format."""
        return self.TIMEFRAME_MAP.get(timeframe, timeframe)

    def _get_market_type(self, symbol: str) -> MarketType:
        """Determine market type from symbol."""
        # Check our registries
        if (
            symbol in CRYPTO_SYMBOLS
            or symbol.endswith("-USD")
            and "/" in self.SYMBOL_MAP.get(symbol, "")
        ):
            return MarketType.CRYPTO
        if symbol in COMMODITY_SYMBOLS or symbol in [
            "GC=F",
            "SI=F",
            "CL=F",
            "BZ=F",
            "NG=F",
            "HG=F",
        ]:
            return MarketType.COMMODITY
        if symbol in STOCK_SYMBOLS:
            return MarketType.STOCK

        # Infer from Yahoo symbol format
        yahoo_symbol = self.map_symbol(symbol)
        if yahoo_symbol.endswith("=F"):
            return MarketType.COMMODITY
        if yahoo_symbol.endswith("-USD"):
            return MarketType.CRYPTO
        return MarketType.STOCK

    def _acquire_rate_limit(self) -> bool:
        """Acquire rate limit permission."""
        if self._rate_limiter:
            return self._rate_limiter.acquire(block=True, timeout=120.0)
        return True

    def _report_success(self) -> None:
        """Report successful API call."""
        if self._rate_limiter:
            self._rate_limiter.report_success()

    def _report_error(self, error: Exception) -> None:
        """Report API error for backoff calculation."""
        if self._rate_limiter:
            error_str = str(error).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str
            self._rate_limiter.report_error(is_rate_limit=is_rate_limit)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 250,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[NormalizedOHLCV]:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Standard symbol (e.g., "AAPL", "XAU/USD")
            timeframe: Candle timeframe
            limit: Number of candles (used to calculate period if no dates)
            start_time: Start datetime (optional)
            end_time: End datetime (optional)

        Returns:
            List of NormalizedOHLCV objects
        """
        yahoo_symbol = self.map_symbol(symbol)
        yahoo_timeframe = self.map_timeframe(timeframe)
        market_type = self._get_market_type(symbol)

        # Generate cache key
        cache_key = f"ohlcv:{yahoo_symbol}:{timeframe}:{limit}"

        # Check cache first
        if self._ohlcv_cache:
            cached_data = self._ohlcv_cache.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for OHLCV {symbol}")
                # Reconstruct NormalizedOHLCV objects from cached data
                result = []
                for item in cached_data:
                    ohlcv = NormalizedOHLCV(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        open=item["open"],
                        high=item["high"],
                        low=item["low"],
                        close=item["close"],
                        volume=item["volume"],
                        timeframe=timeframe,
                        market_type=market_type,
                        data_source=self.data_source,
                        exchange=self._get_exchange(symbol),
                    )
                    result.append(ohlcv)
                return result

        # Acquire rate limit
        if not self._acquire_rate_limit():
            logger.error(f"Rate limit blocked for {symbol}")
            return []

        try:
            ticker = yf.Ticker(yahoo_symbol)

            if start_time and end_time:
                # Use date range
                df = ticker.history(
                    start=start_time,
                    end=end_time,
                    interval=yahoo_timeframe,
                )
            else:
                # Use period
                period = self.PERIOD_MAP.get(timeframe, "60d")
                df = ticker.history(period=period, interval=yahoo_timeframe)

            self._report_success()

            if df.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return []

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Limit results
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Convert to normalized format
            result = []
            cache_data = []
            for idx, row in df.iterrows():
                # Handle timezone-aware index
                ts = idx
                if hasattr(ts, "tz_localize"):
                    ts = ts.tz_localize(None)
                if not ts.tzinfo:
                    ts = ts.replace(tzinfo=timezone.utc)

                timestamp = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

                ohlcv = NormalizedOHLCV(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    timeframe=timeframe,
                    market_type=market_type,
                    data_source=self.data_source,
                    exchange=self._get_exchange(symbol),
                    adjusted_close=float(row.get("adj close", row["close"]))
                    if "adj close" in row
                    else None,
                )
                result.append(ohlcv)

                # Prepare for caching
                cache_data.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row.get("volume", 0)),
                    }
                )

            # Store in cache
            if self._ohlcv_cache and cache_data:
                self._ohlcv_cache.set(cache_key, cache_data)

            logger.debug(f"Fetched {len(result)} candles for {symbol} ({yahoo_symbol})")
            return result

        except Exception as e:
            self._report_error(e)
            logger.error(f"Error fetching {yahoo_symbol}: {e}")
            return []

    def fetch_quote(self, symbol: str) -> NormalizedQuote:
        """
        Fetch current quote from Yahoo Finance.

        Note: Yahoo provides 15-minute delayed quotes for free tier.
        Uses caching and rate limiting to avoid 429 errors.
        """
        yahoo_symbol = self.map_symbol(symbol)
        market_type = self._get_market_type(symbol)

        # Generate cache key
        cache_key = f"quote:{yahoo_symbol}"

        # Check cache first (short TTL for quotes)
        if self._cache:
            cached_quote = self._cache.get(cache_key)
            if cached_quote:
                logger.debug(f"Cache hit for quote {symbol}")
                return NormalizedQuote(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(cached_quote["timestamp"]),
                    bid=cached_quote["bid"],
                    ask=cached_quote["ask"],
                    last_price=cached_quote["last_price"],
                    market_type=market_type,
                    data_source=self.data_source,
                    exchange=self._get_exchange(symbol),
                )

        # Acquire rate limit
        if not self._acquire_rate_limit():
            logger.error(f"Rate limit blocked for quote {symbol}")
            raise ValueError(f"Rate limited: Could not fetch quote for {symbol}")

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            self._report_success()

            # Get current price data
            last_price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
            bid = info.get("bid", last_price * 0.9999)
            ask = info.get("ask", last_price * 1.0001)

            # Handle None values
            if bid is None or bid == 0:
                bid = last_price * 0.9999
            if ask is None or ask == 0:
                ask = last_price * 1.0001

            timestamp = datetime.now(timezone.utc)

            # Cache the quote (short TTL)
            if self._cache and last_price:
                self._cache.set(
                    cache_key,
                    {
                        "timestamp": timestamp.isoformat(),
                        "bid": float(bid),
                        "ask": float(ask),
                        "last_price": float(last_price) if last_price else None,
                    },
                    memory_ttl=30.0,  # 30 seconds in memory
                    disk_ttl=60.0,  # 1 minute on disk
                )

            return NormalizedQuote(
                symbol=symbol,
                timestamp=timestamp,
                bid=float(bid),
                ask=float(ask),
                bid_size=info.get("bidSize"),
                ask_size=info.get("askSize"),
                last_price=float(last_price) if last_price else None,
                last_size=info.get("regularMarketVolume"),
                market_type=market_type,
                data_source=self.data_source,
                exchange=self._get_exchange(symbol),
            )

        except Exception as e:
            self._report_error(e)
            logger.error(f"Error fetching quote for {yahoo_symbol}: {e}")

            # Return a basic quote with last known price from history
            # but still respect rate limiting
            try:
                if self._acquire_rate_limit():
                    ticker = yf.Ticker(yahoo_symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    self._report_success()
                    if not hist.empty:
                        last = hist["Close"].iloc[-1]
                        return NormalizedQuote(
                            symbol=symbol,
                            timestamp=datetime.now(timezone.utc),
                            bid=float(last * 0.9999),
                            ask=float(last * 1.0001),
                            last_price=float(last),
                            market_type=market_type,
                            data_source=self.data_source,
                        )
            except Exception as fallback_error:
                self._report_error(fallback_error)

            raise ValueError(f"Could not fetch quote for {symbol}")

    def fetch_ohlcv_dataframe(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 250,
    ) -> pd.DataFrame:
        """
        Convenience method to fetch OHLCV as DataFrame directly.

        Returns DataFrame with columns: open, high, low, close, volume
        """
        data = self.fetch_ohlcv(symbol, timeframe, limit)
        return self.to_dataframe(data)

    def _get_exchange(self, symbol: str) -> Optional[str]:
        """Get exchange for symbol."""
        symbol_info = ALL_SYMBOLS.get(symbol)
        if symbol_info:
            return symbol_info.exchange
        return None

    def health_check(self) -> bool:
        """Check if Yahoo Finance is responding."""
        # Use cached check if available
        cache_key = "health_check:yahoo"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if not self._acquire_rate_limit():
            logger.warning("Health check blocked by rate limiter")
            return False

        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            result = "regularMarketPrice" in info or "currentPrice" in info
            self._report_success()

            # Cache the result
            if self._cache:
                self._cache.set(cache_key, result, memory_ttl=60.0, disk_ttl=120.0)

            return result
        except Exception as e:
            self._report_error(e)
            logger.error(f"Yahoo Finance health check failed: {e}")
            return False

    def get_rate_limit_info(self) -> dict:
        """Yahoo Finance rate limits (unofficial) and current status."""
        info = {
            "requests_per_minute": 20,  # Our conservative limit
            "requests_per_hour": 500,
            "requests_per_day": 2000,
        }

        if self._rate_limiter:
            info["status"] = self._rate_limiter.get_status()

        return info

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "quote_cache": None,
            "ohlcv_cache": None,
        }

        if self._cache:
            stats["quote_cache"] = self._cache.get_stats()
        if self._ohlcv_cache:
            stats["ohlcv_cache"] = self._ohlcv_cache.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self._cache:
            self._cache.clear()
        if self._ohlcv_cache:
            self._ohlcv_cache.clear()
        logger.info("Yahoo adapter caches cleared")


# Convenience function to create Yahoo adapter
def create_yahoo_adapter() -> YahooAdapter:
    """Create and return a Yahoo Finance adapter."""
    return YahooAdapter()
