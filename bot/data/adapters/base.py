"""
Base data adapter interface for market data providers.

All provider-specific adapters must implement this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd

from ..models import DataSource, MarketType, NormalizedOHLCV, NormalizedQuote


class DataAdapter(ABC):
    """
    Abstract base class for all data provider adapters.

    Provides a unified interface for fetching market data from
    different providers (Yahoo, CCXT, Polygon, etc.).
    """

    # Must be set by subclasses
    name: str
    data_source: DataSource
    supported_markets: List[MarketType]
    supports_realtime: bool = False
    supports_historical: bool = True
    priority: int = 0  # Higher = preferred when multiple adapters available

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 250,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[NormalizedOHLCV]:
        """
        Fetch OHLCV candles for a symbol.

        Args:
            symbol: Provider-specific symbol format
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            start_time: Start of date range (optional)
            end_time: End of date range (optional)

        Returns:
            List of NormalizedOHLCV objects
        """
        pass

    @abstractmethod
    def fetch_quote(self, symbol: str) -> NormalizedQuote:
        """
        Fetch current quote for a symbol.

        Args:
            symbol: Provider-specific symbol format

        Returns:
            NormalizedQuote object
        """
        pass

    @abstractmethod
    def map_symbol(self, standard_symbol: str) -> str:
        """
        Convert standard symbol to provider-specific format.

        Args:
            standard_symbol: Our internal symbol format (e.g., "BTC/USDT")

        Returns:
            Provider-specific symbol (e.g., "BTCUSDT" or "BTC-USD")
        """
        pass

    def map_timeframe(self, timeframe: str) -> str:
        """
        Convert standard timeframe to provider-specific format.

        Default implementation returns as-is. Override if needed.
        """
        return timeframe

    def to_dataframe(self, data: List[NormalizedOHLCV]) -> pd.DataFrame:
        """
        Convert list of NormalizedOHLCV to pandas DataFrame.

        Returns DataFrame with columns: open, high, low, close, volume
        Index is timestamp.
        """
        if not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            [
                {
                    "timestamp": d.timestamp,
                    "open": d.open,
                    "high": d.high,
                    "low": d.low,
                    "close": d.close,
                    "volume": d.volume,
                }
                for d in data
            ]
        )
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    def health_check(self) -> bool:
        """
        Check if the provider is healthy and responding.

        Default implementation returns True. Override for actual check.
        """
        return True

    def get_rate_limit_info(self) -> dict:
        """
        Get rate limit information for this provider.

        Returns dict with keys: requests_per_minute, requests_per_day, etc.
        """
        return {
            "requests_per_minute": 60,
            "requests_per_day": None,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} source={self.data_source.value}>"
