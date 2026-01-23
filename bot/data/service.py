"""
Unified Market Data Service.

Provides a single interface for accessing market data across all providers
with automatic caching, provider selection, and failover support.
"""

import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Type

import pandas as pd

from .adapters.base import DataAdapter
from .adapters.yahoo_adapter import YahooAdapter
from .adapters import HAS_CCXT

if HAS_CCXT:
    from .adapters.ccxt_adapter import CCXTAdapter
from .cache import CacheManager, make_ohlcv_key, make_quote_key
from .models import (
    ALL_SYMBOLS,
    DataSource,
    MarketType,
    NormalizedOHLCV,
    NormalizedQuote,
    SymbolInfo,
    get_symbol_info,
    get_symbols_by_market,
)

# Import core utilities for validation and rate limiting
from bot.core import (
    validate_ohlcv,
    RateLimiter,
    retry_async,
    get_logger,
    log_operation,
    metrics,
)

logger = get_logger(__name__)


class MarketDataService:
    """
    Unified market data service with automatic provider selection,
    caching, and failover support.

    Features:
    - Multi-provider support (Yahoo, CCXT, Polygon, etc.)
    - Automatic provider selection based on market type
    - Multi-level caching (memory + SQLite)
    - Automatic failover on provider errors
    - Rate limiting awareness
    """

    def __init__(
        self,
        data_dir: str = "data",
        enable_cache: bool = True,
    ):
        """
        Initialize MarketDataService.

        Args:
            data_dir: Directory for cache and data storage
            enable_cache: Whether to enable caching
        """
        self.data_dir = data_dir
        self.enable_cache = enable_cache

        # Initialize cache
        self.cache = CacheManager(data_dir) if enable_cache else None

        # Provider health status (must be before _init_adapters)
        self._provider_health: Dict[str, bool] = {}

        # Initialize adapters
        self.adapters: Dict[str, DataAdapter] = {}
        self._init_adapters()

        logger.info(f"MarketDataService initialized with adapters: {list(self.adapters.keys())}")

    def _init_adapters(self):
        """Initialize available data adapters."""
        # Yahoo Finance (always available, free)
        try:
            self.adapters["yahoo"] = YahooAdapter()
            self._provider_health["yahoo"] = True
            logger.info("Yahoo Finance adapter initialized")
        except ImportError as e:
            logger.warning(f"Yahoo adapter not available: {e}")

        # CCXT for real-time crypto data
        if HAS_CCXT:
            try:
                self.adapters["ccxt"] = CCXTAdapter(exchange_id="binance")
                self._provider_health["ccxt"] = True
                logger.info("CCXT adapter initialized (binance)")
            except Exception as e:
                logger.warning(f"CCXT adapter not available: {e}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 250,
        prefer_realtime: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with automatic provider selection and caching.

        Args:
            symbol: Standard symbol format (e.g., "AAPL", "BTC/USDT", "XAU/USD")
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            prefer_realtime: If True, prefer real-time providers over delayed
            use_cache: Whether to use cache

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = make_ohlcv_key(symbol, timeframe, limit)

        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol} {timeframe}")
                return cached

        # Get symbol info and select adapter
        symbol_info = get_symbol_info(symbol)
        adapter = self._select_adapter(symbol_info, prefer_realtime)

        if adapter is None:
            raise ValueError(f"No adapter available for {symbol}")

        # Fetch data with metrics tracking
        try:
            with log_operation(f"fetch_ohlcv_{symbol}") as op_metrics:
                data = adapter.fetch_ohlcv(symbol, timeframe, limit)
                df = adapter.to_dataframe(data)

            # Validate data quality
            validation = validate_ohlcv(df)
            if not validation.is_valid:
                logger.warning(f"Data validation failed for {symbol}: {validation.errors}")
                metrics.increment("data_validation_failures")
            elif validation.warnings:
                logger.debug(f"Data warnings for {symbol}: {validation.warnings}")

            # Track metrics
            metrics.increment("ohlcv_fetches")
            metrics.timing("ohlcv_fetch_ms", op_metrics.duration_ms)

            # Cache result
            if use_cache and self.cache and not df.empty:
                data_type = f"ohlcv_{timeframe}"
                self.cache.set(cache_key, df, data_type)

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} from {adapter.name}: {e}")
            metrics.increment("ohlcv_fetch_errors")
            # Try fallback
            return self._fetch_with_fallback(symbol, timeframe, limit, adapter.name, e)

    def fetch_quote(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> NormalizedQuote:
        """
        Fetch current quote for a symbol.

        Args:
            symbol: Standard symbol format
            use_cache: Whether to use cache (short TTL for quotes)

        Returns:
            NormalizedQuote object
        """
        cache_key = make_quote_key(symbol)

        # Check cache (very short TTL for quotes)
        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return NormalizedQuote(**cached) if isinstance(cached, dict) else cached

        # Get symbol info and select adapter
        symbol_info = get_symbol_info(symbol)
        adapter = self._select_adapter(symbol_info, prefer_realtime=True)

        if adapter is None:
            raise ValueError(f"No adapter available for {symbol}")

        # Fetch quote
        quote = adapter.fetch_quote(symbol)

        # Cache result
        if use_cache and self.cache:
            self.cache.set(cache_key, quote.to_dict(), "quote")

        return quote

    def fetch_multiple_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 250,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of standard symbols
            timeframe: Candle timeframe
            limit: Number of candles

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, limit)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results

    def fetch_market_data(
        self,
        market_type: MarketType,
        timeframe: str = "1h",
        limit: int = 250,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all symbols in a market type.

        Args:
            market_type: MarketType enum (CRYPTO, STOCK, COMMODITY)
            timeframe: Candle timeframe
            limit: Number of candles

        Returns:
            Dict mapping symbol to DataFrame
        """
        symbols_dict = get_symbols_by_market(market_type)
        symbols = list(symbols_dict.keys())
        return self.fetch_multiple_ohlcv(symbols, timeframe, limit)

    def get_current_prices(
        self,
        symbols: List[str],
    ) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of standard symbols

        Returns:
            Dict mapping symbol to current price
        """
        prices = {}
        for symbol in symbols:
            try:
                quote = self.fetch_quote(symbol)
                prices[symbol] = quote.last_price or quote.mid_price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        return prices

    def _select_adapter(
        self,
        symbol_info: Optional[SymbolInfo],
        prefer_realtime: bool,
    ) -> Optional[DataAdapter]:
        """
        Select best adapter for symbol based on market type and preferences.

        Args:
            symbol_info: SymbolInfo for the symbol (can be None)
            prefer_realtime: Whether to prefer real-time providers

        Returns:
            Best available DataAdapter or None
        """
        if not self.adapters:
            return None

        # Determine market type
        market_type = symbol_info.market_type if symbol_info else MarketType.STOCK

        # Filter adapters that support this market
        compatible = [
            (name, adapter)
            for name, adapter in self.adapters.items()
            if market_type in adapter.supported_markets and self._provider_health.get(name, True)
        ]

        if not compatible:
            # Fall back to any available adapter
            compatible = [
                (name, adapter)
                for name, adapter in self.adapters.items()
                if self._provider_health.get(name, True)
            ]

        if not compatible:
            return None

        # Sort by preference
        compatible.sort(
            key=lambda x: (
                not (x[1].supports_realtime and prefer_realtime),
                -x[1].priority,
            )
        )

        return compatible[0][1]

    def _fetch_with_fallback(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        failed_adapter: str,
        error: Exception,
    ) -> pd.DataFrame:
        """
        Try fetching from fallback adapters after primary fails.
        """
        # Mark failed adapter as unhealthy temporarily
        self._provider_health[failed_adapter] = False

        # Try other adapters
        for name, adapter in self.adapters.items():
            if name == failed_adapter:
                continue
            if not self._provider_health.get(name, True):
                continue

            try:
                logger.info(f"Trying fallback adapter: {name}")
                data = adapter.fetch_ohlcv(symbol, timeframe, limit)
                df = adapter.to_dataframe(data)
                if not df.empty:
                    # Restore original adapter health after some time
                    # (in production, use a background task)
                    self._provider_health[failed_adapter] = True
                    return df
            except Exception as e:
                logger.warning(f"Fallback {name} also failed: {e}")

        # All adapters failed
        self._provider_health[failed_adapter] = True  # Restore for next try
        raise ValueError(f"All adapters failed for {symbol}: {error}")

    def get_adapter(self, name: str) -> Optional[DataAdapter]:
        """Get a specific adapter by name."""
        return self.adapters.get(name)

    def add_adapter(self, name: str, adapter: DataAdapter) -> None:
        """Add a new adapter."""
        self.adapters[name] = adapter
        self._provider_health[name] = True
        logger.info(f"Added adapter: {name}")

    def remove_adapter(self, name: str) -> None:
        """Remove an adapter."""
        if name in self.adapters:
            del self.adapters[name]
            del self._provider_health[name]
            logger.info(f"Removed adapter: {name}")

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all adapters.

        Returns:
            Dict mapping adapter name to health status
        """
        results = {}
        for name, adapter in self.adapters.items():
            try:
                results[name] = adapter.health_check()
                self._provider_health[name] = results[name]
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
                self._provider_health[name] = False
        return results

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_supported_symbols(self, market_type: Optional[MarketType] = None) -> List[str]:
        """
        Get list of supported symbols.

        Args:
            market_type: Filter by market type (optional)

        Returns:
            List of standard symbol strings
        """
        if market_type:
            symbols_dict = get_symbols_by_market(market_type)
            return list(symbols_dict.keys())
        return list(ALL_SYMBOLS.keys())


# Singleton instance for convenience with thread-safe initialization
_default_service: Optional[MarketDataService] = None
_service_lock = threading.Lock()


def get_market_data_service(data_dir: str = "data") -> MarketDataService:
    """
    Get or create the default MarketDataService instance.

    Thread-safe using double-check locking pattern.

    Args:
        data_dir: Directory for cache and data storage

    Returns:
        MarketDataService instance
    """
    global _default_service
    # First check without lock (fast path)
    if _default_service is None:
        with _service_lock:
            # Second check with lock (thread-safe)
            if _default_service is None:
                _default_service = MarketDataService(data_dir=data_dir)
    return _default_service


def reset_market_data_service() -> None:
    """Reset the default service instance. Thread-safe."""
    global _default_service
    with _service_lock:
        _default_service = None
