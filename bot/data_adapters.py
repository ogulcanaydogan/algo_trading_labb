"""
Data Adapters for Multiple Market Data Providers.

Provides a unified interface for fetching market data from various sources:
- Binance (crypto)
- Alpha Vantage (stocks, forex, crypto)
- Yahoo Finance (stocks, ETFs)
- Coinbase (crypto)
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for data adapter."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout: int = 30


class DataAdapter(ABC):
    """Abstract base class for data adapters."""

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self._last_request_time: Optional[float] = None

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""
        pass

    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        if self._last_request_time:
            min_interval = 60.0 / self.config.rate_limit_per_minute
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()


class AlphaVantageAdapter(DataAdapter):
    """
    Alpha Vantage data adapter.

    Supports stocks, forex, and crypto data.
    Free tier: 5 calls/minute, 500 calls/day.
    Premium: Higher limits available.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.api_key = config.api_key if config else os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY env var.")
        # Free tier limit
        self.config.rate_limit_per_minute = 5

    @property
    def name(self) -> str:
        return "alpha_vantage"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., "AAPL") or forex pair (e.g., "EUR/USD")
            timeframe: "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
            start_date: Start date (Alpha Vantage returns full history)
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        self._rate_limit()

        # Determine data type and function
        if "/" in symbol:
            # Forex pair
            return self._fetch_forex(symbol, timeframe)
        elif "USDT" in symbol or "BTC" in symbol:
            # Crypto
            return self._fetch_crypto(symbol, timeframe)
        else:
            # Stock
            return self._fetch_stock(symbol, timeframe)

    def _fetch_stock(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch stock data."""
        function = self._get_function(timeframe, "stock")
        interval = self._get_interval(timeframe)

        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        if interval:
            params["interval"] = interval

        response = requests.get(self.BASE_URL, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()

        return self._parse_stock_response(data, timeframe)

    def _fetch_forex(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch forex data."""
        from_currency, to_currency = symbol.split("/")
        function = self._get_function(timeframe, "forex")
        interval = self._get_interval(timeframe)

        params = {
            "function": function,
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        if interval:
            params["interval"] = interval

        response = requests.get(self.BASE_URL, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()

        return self._parse_forex_response(data, timeframe)

    def _fetch_crypto(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch crypto data."""
        # Parse symbol (e.g., BTC/USDT -> BTC, USDT)
        if "/" in symbol:
            crypto, market = symbol.split("/")
        else:
            crypto = symbol.replace("USDT", "")
            market = "USD"

        function = self._get_function(timeframe, "crypto")

        params = {
            "function": function,
            "symbol": crypto,
            "market": market,
            "apikey": self.api_key,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()

        return self._parse_crypto_response(data, timeframe)

    def _get_function(self, timeframe: str, data_type: str) -> str:
        """Get Alpha Vantage function name."""
        if data_type == "stock":
            if timeframe in ["1min", "5min", "15min", "30min", "60min", "1h"]:
                return "TIME_SERIES_INTRADAY"
            elif timeframe == "daily":
                return "TIME_SERIES_DAILY"
            elif timeframe == "weekly":
                return "TIME_SERIES_WEEKLY"
            else:
                return "TIME_SERIES_MONTHLY"

        elif data_type == "forex":
            if timeframe in ["1min", "5min", "15min", "30min", "60min", "1h"]:
                return "FX_INTRADAY"
            elif timeframe == "daily":
                return "FX_DAILY"
            elif timeframe == "weekly":
                return "FX_WEEKLY"
            else:
                return "FX_MONTHLY"

        elif data_type == "crypto":
            if timeframe == "daily":
                return "DIGITAL_CURRENCY_DAILY"
            elif timeframe == "weekly":
                return "DIGITAL_CURRENCY_WEEKLY"
            else:
                return "DIGITAL_CURRENCY_MONTHLY"

        return "TIME_SERIES_DAILY"

    def _get_interval(self, timeframe: str) -> Optional[str]:
        """Get interval parameter for intraday data."""
        interval_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "1h": "60min",
        }
        return interval_map.get(timeframe)

    def _parse_stock_response(self, data: Dict, timeframe: str) -> pd.DataFrame:
        """Parse stock API response."""
        # Find the time series key
        ts_key = None
        for key in data.keys():
            if "Time Series" in key:
                ts_key = key
                break

        if not ts_key or ts_key not in data:
            logger.error(f"Invalid response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
            return pd.DataFrame()

        ts_data = data[ts_key]
        records = []

        for timestamp, values in ts_data.items():
            records.append({
                "timestamp": pd.Timestamp(timestamp),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": float(values.get("5. volume", 0)),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _parse_forex_response(self, data: Dict, timeframe: str) -> pd.DataFrame:
        """Parse forex API response."""
        ts_key = None
        for key in data.keys():
            if "Time Series" in key:
                ts_key = key
                break

        if not ts_key or ts_key not in data:
            logger.error(f"Invalid response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
            return pd.DataFrame()

        ts_data = data[ts_key]
        records = []

        for timestamp, values in ts_data.items():
            records.append({
                "timestamp": pd.Timestamp(timestamp),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": 0,  # Forex doesn't have volume in Alpha Vantage
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _parse_crypto_response(self, data: Dict, timeframe: str) -> pd.DataFrame:
        """Parse crypto API response."""
        ts_key = None
        for key in data.keys():
            if "Time Series" in key:
                ts_key = key
                break

        if not ts_key or ts_key not in data:
            logger.error(f"Invalid response: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
            return pd.DataFrame()

        ts_data = data[ts_key]
        records = []

        for timestamp, values in ts_data.items():
            # Crypto has different key format
            records.append({
                "timestamp": pd.Timestamp(timestamp),
                "open": float(values.get("1a. open (USD)", values.get("1. open", 0))),
                "high": float(values.get("2a. high (USD)", values.get("2. high", 0))),
                "low": float(values.get("3a. low (USD)", values.get("3. low", 0))),
                "close": float(values.get("4a. close (USD)", values.get("4. close", 0))),
                "volume": float(values.get("5. volume", 0)),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.set_index("timestamp", inplace=True)
        return df

    def get_supported_symbols(self) -> List[str]:
        """Alpha Vantage supports most symbols - return common ones."""
        return [
            # Major stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            # Major forex pairs
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
            # Major crypto
            "BTC", "ETH", "SOL",
        ]


class BinanceAdapter(DataAdapter):
    """
    Binance data adapter using CCXT.

    Supports spot and futures crypto markets.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        import ccxt
        self.exchange = ccxt.binance({
            "apiKey": config.api_key if config else os.getenv("BINANCE_API_KEY"),
            "secret": config.api_secret if config else os.getenv("BINANCE_SECRET_KEY"),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.config.rate_limit_per_minute = 1200

    @property
    def name(self) -> str:
        return "binance"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Binance."""
        self._rate_limit()

        since = int(start_date.timestamp() * 1000) if start_date else None
        limit = 1000

        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        if end_date:
            df = df[df.index <= end_date]

        return df

    def get_supported_symbols(self) -> List[str]:
        """Get Binance supported symbols."""
        try:
            markets = self.exchange.load_markets()
            return [s for s in markets.keys() if "USDT" in s][:50]  # Top 50 USDT pairs
        except Exception as e:
            logger.error(f"Failed to load Binance markets: {e}")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


class YahooFinanceAdapter(DataAdapter):
    """
    Yahoo Finance data adapter.

    Free, no API key required. Supports stocks, ETFs, indices.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.config.rate_limit_per_minute = 100

    @property
    def name(self) -> str:
        return "yahoo"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance."""
        import yfinance as yf

        self._rate_limit()

        # Convert symbol format for Yahoo
        yf_symbol = symbol.replace("/", "-")
        if "USDT" in symbol:
            yf_symbol = symbol.replace("/USDT", "-USD")

        # Map timeframe
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "1h",  # Yahoo doesn't support 4h
            "1d": "1d", "1w": "1wk", "1M": "1mo",
            "daily": "1d", "weekly": "1wk", "monthly": "1mo",
        }
        interval = interval_map.get(timeframe, "1h")

        # Calculate period
        if start_date:
            period = None
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d") if end_date else datetime.now().strftime("%Y-%m-%d")
        else:
            period = "1y"
            start = None
            end = None

        try:
            ticker = yf.Ticker(yf_symbol)
            if period:
                df = ticker.history(period=period, interval=interval)
            else:
                df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                return pd.DataFrame()

            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]

            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()

    def get_supported_symbols(self) -> List[str]:
        """Return common symbols supported by Yahoo."""
        return [
            # US Stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
            # ETFs
            "SPY", "QQQ", "IWM", "DIA",
            # Indices
            "^GSPC", "^DJI", "^IXIC",
            # Crypto
            "BTC-USD", "ETH-USD",
        ]


class DataAdapterFactory:
    """Factory for creating data adapters."""

    _adapters: Dict[str, type] = {
        "alpha_vantage": AlphaVantageAdapter,
        "binance": BinanceAdapter,
        "yahoo": YahooFinanceAdapter,
    }

    @classmethod
    def create(cls, adapter_name: str, config: Optional[AdapterConfig] = None) -> DataAdapter:
        """
        Create a data adapter by name.

        Args:
            adapter_name: Name of the adapter (alpha_vantage, binance, yahoo)
            config: Optional configuration

        Returns:
            DataAdapter instance
        """
        if adapter_name not in cls._adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}. Available: {list(cls._adapters.keys())}")

        return cls._adapters[adapter_name](config)

    @classmethod
    def get_available_adapters(cls) -> List[str]:
        """Get list of available adapter names."""
        return list(cls._adapters.keys())


class MultiSourceDataFetcher:
    """
    Fetches data from multiple sources with fallback.

    Tries primary source first, falls back to secondary if failed.
    """

    def __init__(
        self,
        primary: str = "binance",
        fallbacks: Optional[List[str]] = None,
        cache_dir: str = "data/cache",
    ):
        self.adapters: Dict[str, DataAdapter] = {}
        self.primary = primary
        self.fallbacks = fallbacks or ["yahoo"]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize adapters
        for adapter_name in [primary] + self.fallbacks:
            try:
                self.adapters[adapter_name] = DataAdapterFactory.create(adapter_name)
            except Exception as e:
                logger.warning(f"Could not initialize {adapter_name}: {e}")

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data with fallback support.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache:
            cached = self._load_cache(symbol, timeframe)
            if cached is not None:
                logger.debug(f"Using cached data for {symbol}")
                return cached

        # Try primary adapter
        sources_to_try = [self.primary] + self.fallbacks

        for source in sources_to_try:
            if source not in self.adapters:
                continue

            try:
                logger.info(f"Fetching {symbol} from {source}")
                df = self.adapters[source].fetch_ohlcv(
                    symbol, timeframe, start_date, end_date
                )

                if df is not None and not df.empty:
                    if use_cache:
                        self._save_cache(symbol, timeframe, df)
                    return df

            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")
                continue

        logger.error(f"All sources failed for {symbol}")
        return pd.DataFrame()

    def _cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key."""
        return f"{symbol.replace('/', '_')}_{timeframe}"

    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh."""
        cache_file = self.cache_dir / f"{self._cache_key(symbol, timeframe)}.parquet"

        if not cache_file.exists():
            return None

        # Check age (cache for 1 hour for intraday, 1 day for daily)
        max_age = 3600 if "h" in timeframe or "m" in timeframe else 86400
        age = time.time() - cache_file.stat().st_mtime

        if age > max_age:
            return None

        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{self._cache_key(symbol, timeframe)}.parquet"
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


# CLI interface
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Data Adapter CLI")
    parser.add_argument("--adapter", default="yahoo", help="Adapter to use")
    parser.add_argument("--symbol", default="AAPL", help="Symbol to fetch")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")
    parser.add_argument("--days", type=int, default=30, help="Days of history")

    args = parser.parse_args()

    adapter = DataAdapterFactory.create(args.adapter)
    start = datetime.now() - timedelta(days=args.days)

    df = adapter.fetch_ohlcv(args.symbol, args.timeframe, start)
    print(f"\nFetched {len(df)} records for {args.symbol}")
    print(df.tail(10))
