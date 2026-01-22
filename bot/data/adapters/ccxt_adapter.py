"""
CCXT data adapter for cryptocurrency exchanges.

Provides real-time market data for crypto assets via the CCXT library.
Supports 100+ exchanges with unified API.

Features:
- Real-time and historical OHLCV data
- Order book and ticker data
- Rate limiting per exchange
- Automatic exchange selection
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

from ..models import (
    ALL_SYMBOLS,
    CRYPTO_SYMBOLS,
    DataSource,
    MarketType,
    NormalizedOHLCV,
    NormalizedQuote,
)
from .base import DataAdapter

logger = logging.getLogger(__name__)


class CCXTAdapter(DataAdapter):
    """
    CCXT adapter for cryptocurrency exchanges.

    Features:
    - Real-time data from 100+ exchanges
    - Unified symbol format handling
    - Built-in rate limiting
    - Automatic exchange failover
    """

    name = "ccxt"
    data_source = DataSource.CCXT
    supported_markets = [MarketType.CRYPTO]
    supports_realtime = True
    supports_historical = True
    priority = 10  # High priority for crypto

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1M",
    }

    # Default exchanges to use (in priority order)
    DEFAULT_EXCHANGES = ["binance", "bybit", "okx", "kraken", "coinbase"]

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: bool = False,
    ):
        """
        Initialize CCXT adapter.

        Args:
            exchange_id: Exchange to use (default: binance)
            api_key: API key for authenticated endpoints (optional)
            secret: API secret (optional)
            sandbox: Use testnet/sandbox mode
        """
        if ccxt is None:
            raise ImportError("ccxt not installed. Run: pip install ccxt")

        self.exchange_id = exchange_id
        self.sandbox = sandbox

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "enableRateLimit": True,
            "timeout": 30000,
        }

        if api_key and secret:
            config["apiKey"] = api_key
            config["secret"] = secret

        self.exchange: ccxt.Exchange = exchange_class(config)

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        # Load markets
        try:
            self.exchange.load_markets()
            logger.info(
                f"CCXTAdapter initialized: {exchange_id} "
                f"(sandbox={sandbox}, markets={len(self.exchange.markets)})"
            )
        except Exception as e:
            logger.warning(f"Failed to load markets for {exchange_id}: {e}")

    def map_symbol(self, standard_symbol: str) -> str:
        """
        Convert standard symbol to CCXT format.

        Standard format: BTC/USDT
        CCXT format: BTC/USDT (usually same)
        """
        # Check our symbol registry for explicit mapping
        symbol_info = ALL_SYMBOLS.get(standard_symbol)
        if symbol_info:
            ccxt_symbol = symbol_info.provider_mappings.get("ccxt")
            if ccxt_symbol:
                return ccxt_symbol

        # CCXT uses same format as our standard for crypto
        return standard_symbol

    def map_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to CCXT format."""
        return self.TIMEFRAME_MAP.get(timeframe, timeframe)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 250,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[NormalizedOHLCV]:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe
            limit: Number of candles
            start_time: Start datetime (optional)
            end_time: End datetime (optional)

        Returns:
            List of NormalizedOHLCV objects
        """
        ccxt_symbol = self.map_symbol(symbol)
        ccxt_timeframe = self.map_timeframe(timeframe)

        try:
            # Convert start_time to milliseconds if provided
            since = None
            if start_time:
                since = int(start_time.timestamp() * 1000)

            # Fetch OHLCV data
            ohlcv_data = self.exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=limit,
            )

            if not ohlcv_data:
                logger.warning(f"No data returned for {ccxt_symbol}")
                return []

            # Convert to normalized format
            result = []
            for candle in ohlcv_data:
                timestamp_ms, open_p, high_p, low_p, close_p, volume = candle

                # Filter by end_time if provided
                if end_time:
                    candle_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    if candle_time > end_time:
                        continue

                ohlcv = NormalizedOHLCV(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc),
                    open=float(open_p),
                    high=float(high_p),
                    low=float(low_p),
                    close=float(close_p),
                    volume=float(volume) if volume else 0.0,
                    timeframe=timeframe,
                    market_type=MarketType.CRYPTO,
                    data_source=self.data_source,
                    exchange=self.exchange_id,
                )
                result.append(ohlcv)

            logger.debug(f"Fetched {len(result)} candles for {symbol} from {self.exchange_id}")
            return result

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            return []
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []

    def fetch_quote(self, symbol: str) -> NormalizedQuote:
        """
        Fetch current quote/ticker for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            NormalizedQuote object
        """
        ccxt_symbol = self.map_symbol(symbol)

        try:
            ticker = self.exchange.fetch_ticker(ccxt_symbol)

            return NormalizedQuote(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid=float(ticker.get("bid", 0) or 0),
                ask=float(ticker.get("ask", 0) or 0),
                bid_size=float(ticker.get("bidVolume", 0) or 0) if ticker.get("bidVolume") else None,
                ask_size=float(ticker.get("askVolume", 0) or 0) if ticker.get("askVolume") else None,
                last_price=float(ticker.get("last", 0) or 0),
                last_size=float(ticker.get("quoteVolume", 0) or 0) if ticker.get("quoteVolume") else None,
                market_type=MarketType.CRYPTO,
                data_source=self.data_source,
                exchange=self.exchange_id,
            )

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching quote for {symbol}: {e}")
            raise ValueError(f"Network error: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching quote for {symbol}: {e}")
            raise ValueError(f"Exchange error: {e}")
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise ValueError(f"Could not fetch quote for {symbol}: {e}")

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict:
        """
        Fetch order book for a symbol.

        Args:
            symbol: Trading pair
            limit: Depth of order book

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        ccxt_symbol = self.map_symbol(symbol)

        try:
            order_book = self.exchange.fetch_order_book(ccxt_symbol, limit=limit)
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bids": order_book.get("bids", []),
                "asks": order_book.get("asks", []),
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {"symbol": symbol, "bids": [], "asks": []}

    def fetch_trades(
        self,
        symbol: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch recent trades for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of trades
            since: Start time (optional)

        Returns:
            List of trade dictionaries
        """
        ccxt_symbol = self.map_symbol(symbol)

        try:
            since_ms = int(since.timestamp() * 1000) if since else None
            trades = self.exchange.fetch_trades(ccxt_symbol, since=since_ms, limit=limit)

            return [
                {
                    "id": t.get("id"),
                    "timestamp": t.get("timestamp"),
                    "symbol": symbol,
                    "side": t.get("side"),
                    "price": t.get("price"),
                    "amount": t.get("amount"),
                    "cost": t.get("cost"),
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return []

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs on the exchange."""
        try:
            return list(self.exchange.markets.keys())
        except Exception:
            return []

    def health_check(self) -> bool:
        """Check if exchange is responding."""
        try:
            self.exchange.fetch_time()
            return True
        except Exception as e:
            logger.error(f"CCXT health check failed for {self.exchange_id}: {e}")
            return False

    def get_rate_limit_info(self) -> dict:
        """Get exchange rate limit information."""
        return {
            "exchange": self.exchange_id,
            "rate_limit": self.exchange.rateLimit,
            "requests_per_second": 1000 / self.exchange.rateLimit if self.exchange.rateLimit else None,
        }


def create_ccxt_adapter(
    exchange_id: str = "binance",
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    sandbox: bool = False,
) -> CCXTAdapter:
    """Create and return a CCXT adapter."""
    return CCXTAdapter(
        exchange_id=exchange_id,
        api_key=api_key,
        secret=secret,
        sandbox=sandbox,
    )
