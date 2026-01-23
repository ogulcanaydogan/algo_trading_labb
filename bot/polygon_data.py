"""
Polygon.io Data Provider.

Provides real-time and historical market data from Polygon.io
for stocks, forex, and crypto.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PolygonDataProvider:
    """
    Data provider using Polygon.io API.

    Features:
    - Real-time stock/crypto/forex data
    - Historical OHLCV data
    - News and sentiment data
    - Market status
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = requests.Session()

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make API request."""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polygon API error: {e}")
            return None

    # ==================== Stocks ====================

    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time stock quote.

        Args:
            symbol: Stock ticker (e.g., "AAPL")

        Returns:
            Quote data dict
        """
        data = self._request(f"/v2/last/trade/{symbol}")
        if data and data.get("status") == "OK":
            result = data.get("results", {})
            return {
                "symbol": symbol,
                "price": result.get("p"),
                "size": result.get("s"),
                "timestamp": result.get("t"),
                "exchange": result.get("x"),
            }
        return None

    def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "hour",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock bars.

        Args:
            symbol: Stock ticker
            timeframe: "minute", "hour", "day", "week", "month"
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum bars to return

        Returns:
            DataFrame with OHLCV data
        """
        # Default to last 30 days
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        multiplier = 1
        timespan = timeframe

        data = self._request(
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            params={"limit": limit, "sort": "asc"},
        )

        if data and data.get("status") == "OK" and data.get("results"):
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "transactions",
                }
            )
            df = df.set_index("timestamp")
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        return None

    def get_stock_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock snapshot with current price and today's stats."""
        data = self._request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        if data and data.get("status") == "OK":
            ticker = data.get("ticker", {})
            return {
                "symbol": symbol,
                "price": ticker.get("lastTrade", {}).get("p"),
                "today_open": ticker.get("day", {}).get("o"),
                "today_high": ticker.get("day", {}).get("h"),
                "today_low": ticker.get("day", {}).get("l"),
                "today_close": ticker.get("day", {}).get("c"),
                "today_volume": ticker.get("day", {}).get("v"),
                "today_vwap": ticker.get("day", {}).get("vw"),
                "prev_close": ticker.get("prevDay", {}).get("c"),
                "change": ticker.get("todaysChange"),
                "change_pct": ticker.get("todaysChangePerc"),
            }
        return None

    # ==================== Crypto ====================

    def get_crypto_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time crypto quote.

        Args:
            symbol: Crypto pair (e.g., "BTC-USD")

        Returns:
            Quote data dict
        """
        # Convert symbol format
        pair = symbol.replace("/", "-").replace("USDT", "USD")

        data = self._request(f"/v1/last/crypto/{pair}")
        if data and data.get("status") == "success":
            result = data.get("last", {})
            return {
                "symbol": symbol,
                "price": result.get("price"),
                "size": result.get("size"),
                "timestamp": result.get("timestamp"),
                "exchange": result.get("exchange"),
            }
        return None

    def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "hour",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical crypto bars.

        Args:
            symbol: Crypto pair (e.g., "BTC/USDT" or "BTC-USD")
            timeframe: "minute", "hour", "day", "week", "month"
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum bars to return

        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol format (BTC/USDT -> X:BTCUSD)
        pair = symbol.replace("/", "").replace("USDT", "USD")
        ticker = f"X:{pair}"

        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        multiplier = 1
        timespan = timeframe

        data = self._request(
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            params={"limit": limit, "sort": "asc"},
        )

        if data and data.get("status") == "OK" and data.get("results"):
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            df = df.set_index("timestamp")
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        return None

    # ==================== News ====================

    def get_ticker_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get news articles for a ticker.

        Args:
            symbol: Ticker symbol
            limit: Maximum articles to return

        Returns:
            List of news articles
        """
        data = self._request(
            "/v2/reference/news",
            params={"ticker": symbol, "limit": limit, "sort": "published_utc"},
        )

        if data and data.get("status") == "OK":
            articles = []
            for article in data.get("results", []):
                articles.append(
                    {
                        "title": article.get("title"),
                        "author": article.get("author"),
                        "published": article.get("published_utc"),
                        "url": article.get("article_url"),
                        "description": article.get("description"),
                        "tickers": article.get("tickers", []),
                        "sentiment": article.get("insights", []),
                    }
                )
            return articles
        return []

    # ==================== Market Status ====================

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        data = self._request("/v1/marketstatus/now")
        if data:
            return {
                "market": data.get("market"),
                "early_hours": data.get("earlyHours"),
                "after_hours": data.get("afterHours"),
                "server_time": data.get("serverTime"),
                "exchanges": data.get("exchanges", {}),
                "currencies": data.get("currencies", {}),
            }
        return {}

    # ==================== Ticker Details ====================

    def get_ticker_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a ticker."""
        data = self._request(f"/v3/reference/tickers/{symbol}")
        if data and data.get("status") == "OK":
            result = data.get("results", {})
            return {
                "symbol": result.get("ticker"),
                "name": result.get("name"),
                "market": result.get("market"),
                "type": result.get("type"),
                "currency": result.get("currency_name"),
                "exchange": result.get("primary_exchange"),
                "market_cap": result.get("market_cap"),
                "shares_outstanding": result.get("share_class_shares_outstanding"),
                "description": result.get("description"),
                "homepage": result.get("homepage_url"),
            }
        return None

    # ==================== Aggregates ====================

    def get_previous_close(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get previous day's close data."""
        data = self._request(f"/v2/aggs/ticker/{symbol}/prev")
        if data and data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            return {
                "symbol": symbol,
                "open": result.get("o"),
                "high": result.get("h"),
                "low": result.get("l"),
                "close": result.get("c"),
                "volume": result.get("v"),
                "vwap": result.get("vw"),
                "timestamp": result.get("t"),
            }
        return None


class PolygonDataAdapter:
    """
    Adapter to use Polygon.io data with the trading bot.

    Provides a unified interface compatible with the existing
    data fetching infrastructure.
    """

    def __init__(self, api_key: str):
        self.provider = PolygonDataProvider(api_key)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = 60  # seconds

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "AAPL", "BTC/USDT")
            timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
            limit: Number of bars

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"

        # Check cache
        if cache_key in self._cache:
            cache_age = (datetime.now() - self._cache_time[cache_key]).total_seconds()
            if cache_age < self._cache_ttl:
                return self._cache[cache_key]

        # Map timeframe
        tf_map = {
            "1m": "minute",
            "5m": "minute",
            "15m": "minute",
            "1h": "hour",
            "4h": "hour",
            "1d": "day",
        }
        polygon_tf = tf_map.get(timeframe, "hour")

        # Determine asset type and fetch
        if "/" in symbol or symbol.endswith("USD"):
            # Crypto
            df = self.provider.get_crypto_bars(symbol, timeframe=polygon_tf, limit=limit)
        else:
            # Stock
            df = self.provider.get_stock_bars(symbol, timeframe=polygon_tf, limit=limit)

        if df is not None:
            self._cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now()

        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if "/" in symbol or symbol.endswith("USD"):
            quote = self.provider.get_crypto_quote(symbol)
        else:
            quote = self.provider.get_stock_quote(symbol)

        if quote:
            return quote.get("price")
        return None

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for a symbol."""
        if "/" in symbol:
            # Crypto
            quote = self.provider.get_crypto_quote(symbol)
            bars = self.provider.get_crypto_bars(symbol, timeframe="day", limit=2)
        else:
            # Stock
            quote = self.provider.get_stock_snapshot(symbol)
            bars = self.provider.get_stock_bars(symbol, timeframe="day", limit=2)

        if quote is None:
            return None

        result = {
            "symbol": symbol,
            "price": quote.get("price"),
            "timestamp": datetime.now().isoformat(),
        }

        if bars is not None and len(bars) >= 2:
            result["prev_close"] = bars["close"].iloc[-2]
            result["change"] = result["price"] - result["prev_close"]
            result["change_pct"] = (result["change"] / result["prev_close"]) * 100

        return result


def create_polygon_provider(api_key: str) -> PolygonDataProvider:
    """Factory function to create Polygon data provider."""
    return PolygonDataProvider(api_key)


def create_polygon_adapter(api_key: str) -> PolygonDataAdapter:
    """Factory function to create Polygon data adapter."""
    return PolygonDataAdapter(api_key)
