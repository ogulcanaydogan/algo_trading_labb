"""
Kraken Exchange Adapter.

Provides trading functionality for Kraken exchange
as an alternative to Binance.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KrakenConfig:
    """Configuration for Kraken adapter."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = False  # Kraken doesn't have a sandbox, use paper trading
    rate_limit: bool = True

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("KRAKEN_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("KRAKEN_API_SECRET")


class KrakenAdapter:
    """
    Kraken exchange adapter using CCXT.

    Features:
    - Spot trading
    - Historical data
    - Balance and position management
    - Order management
    """

    # Symbol mapping (Kraken uses different symbols)
    SYMBOL_MAP = {
        "BTC/USDT": "BTC/USDT",
        "ETH/USDT": "ETH/USDT",
        "SOL/USDT": "SOL/USDT",
        "AVAX/USDT": "AVAX/USDT",
        "ADA/USDT": "ADA/USDT",
        "XRP/USDT": "XRP/USDT",
        "DOGE/USDT": "DOGE/USDT",
        "BTC/USD": "XBT/USD",
        "ETH/USD": "ETH/USD",
    }

    def __init__(self, config: Optional[KrakenConfig] = None):
        self.config = config or KrakenConfig()
        self._exchange: Optional[ccxt.kraken] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Kraken exchange."""
        try:
            self._exchange = ccxt.kraken(
                {
                    "apiKey": self.config.api_key,
                    "secret": self.config.api_secret,
                    "enableRateLimit": self.config.rate_limit,
                    "options": {
                        "defaultType": "spot",
                    },
                }
            )

            # Test connection
            self._exchange.fetch_time()
            self._connected = True
            logger.info("Connected to Kraken exchange")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from exchange."""
        self._exchange = None
        self._connected = False
        logger.info("Disconnected from Kraken")

    def _map_symbol(self, symbol: str) -> str:
        """Map symbol to Kraken format."""
        return self.SYMBOL_MAP.get(symbol, symbol)

    # ==================== Market Data ====================

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for a symbol."""
        if not self._exchange:
            return None

        try:
            kraken_symbol = self._map_symbol(symbol)
            ticker = self._exchange.fetch_ticker(kraken_symbol)
            return {
                "symbol": symbol,
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "last": ticker.get("last"),
                "high": ticker.get("high"),
                "low": ticker.get("low"),
                "volume": ticker.get("baseVolume"),
                "change_pct": ticker.get("percentage"),
                "timestamp": ticker.get("timestamp"),
            }
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        ticker = self.get_ticker(symbol)
        if ticker:
            return ticker.get("last")
        return None

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
            limit: Number of candles
            since: Start timestamp (ms)

        Returns:
            DataFrame with OHLCV data
        """
        if not self._exchange:
            return None

        try:
            kraken_symbol = self._map_symbol(symbol)
            ohlcv = self._exchange.fetch_ohlcv(
                kraken_symbol,
                timeframe=timeframe,
                limit=limit,
                since=since,
            )

            if not ohlcv:
                return None

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            return df

        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return None

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Get order book for a symbol."""
        if not self._exchange:
            return None

        try:
            kraken_symbol = self._map_symbol(symbol)
            book = self._exchange.fetch_order_book(kraken_symbol, limit)
            return {
                "symbol": symbol,
                "bids": book.get("bids", [])[:limit],
                "asks": book.get("asks", [])[:limit],
                "timestamp": book.get("timestamp"),
            }
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None

    # ==================== Account ====================

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance."""
        if not self._exchange or not self.config.api_key:
            return None

        try:
            balance = self._exchange.fetch_balance()

            # Extract relevant balances
            result = {
                "total": {},
                "free": {},
                "used": {},
            }

            for currency in ["USDT", "USD", "BTC", "ETH", "SOL", "XBT"]:
                if currency in balance:
                    result["total"][currency] = balance[currency].get("total", 0)
                    result["free"][currency] = balance[currency].get("free", 0)
                    result["used"][currency] = balance[currency].get("used", 0)

            return result

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions (based on balance)."""
        balance = self.get_balance()
        if not balance:
            return []

        positions = []
        for currency, amount in balance["total"].items():
            if currency not in ("USDT", "USD") and amount > 0:
                # Get current price
                symbol = f"{currency}/USDT"
                price = self.get_current_price(symbol)
                if price:
                    positions.append(
                        {
                            "symbol": symbol,
                            "quantity": amount,
                            "current_price": price,
                            "value": amount * price,
                        }
                    )

        return positions

    # ==================== Trading ====================

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            Order result
        """
        if not self._exchange or not self.config.api_key:
            logger.error("Cannot place order: not connected or no API key")
            return None

        try:
            kraken_symbol = self._map_symbol(symbol)

            order = self._exchange.create_market_order(
                symbol=kraken_symbol,
                side=side,
                amount=quantity,
            )

            return {
                "id": order.get("id"),
                "symbol": symbol,
                "side": side,
                "type": "market",
                "quantity": quantity,
                "filled": order.get("filled"),
                "average_price": order.get("average"),
                "cost": order.get("cost"),
                "status": order.get("status"),
                "timestamp": order.get("timestamp"),
            }

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price

        Returns:
            Order result
        """
        if not self._exchange or not self.config.api_key:
            return None

        try:
            kraken_symbol = self._map_symbol(symbol)

            order = self._exchange.create_limit_order(
                symbol=kraken_symbol,
                side=side,
                amount=quantity,
                price=price,
            )

            return {
                "id": order.get("id"),
                "symbol": symbol,
                "side": side,
                "type": "limit",
                "quantity": quantity,
                "price": price,
                "filled": order.get("filled"),
                "status": order.get("status"),
                "timestamp": order.get("timestamp"),
            }

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if not self._exchange or not self.config.api_key:
            return False

        try:
            kraken_symbol = self._map_symbol(symbol)
            self._exchange.cancel_order(order_id, kraken_symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        if not self._exchange or not self.config.api_key:
            return []

        try:
            kraken_symbol = self._map_symbol(symbol) if symbol else None
            orders = self._exchange.fetch_open_orders(kraken_symbol)

            return [
                {
                    "id": o.get("id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "quantity": o.get("amount"),
                    "price": o.get("price"),
                    "filled": o.get("filled"),
                    "status": o.get("status"),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_trades(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        if not self._exchange or not self.config.api_key:
            return []

        try:
            kraken_symbol = self._map_symbol(symbol)
            trades = self._exchange.fetch_my_trades(kraken_symbol, since=since, limit=limit)

            return [
                {
                    "id": t.get("id"),
                    "symbol": symbol,
                    "side": t.get("side"),
                    "quantity": t.get("amount"),
                    "price": t.get("price"),
                    "cost": t.get("cost"),
                    "fee": t.get("fee", {}).get("cost"),
                    "timestamp": t.get("timestamp"),
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    # ==================== Utility ====================

    def get_trading_fees(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get trading fees for a symbol."""
        if not self._exchange:
            return None

        try:
            # Kraken's default fees
            return {
                "maker": 0.0016,  # 0.16%
                "taker": 0.0026,  # 0.26%
            }
        except Exception as e:
            logger.error(f"Failed to get fees: {e}")
            return None

    def get_markets(self) -> List[str]:
        """Get available trading pairs."""
        if not self._exchange:
            return []

        try:
            self._exchange.load_markets()
            return list(self._exchange.markets.keys())
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []


def create_kraken_adapter(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> KrakenAdapter:
    """Factory function to create Kraken adapter."""
    config = KrakenConfig(
        api_key=api_key,
        api_secret=api_secret,
    )
    adapter = KrakenAdapter(config)
    adapter.connect()
    return adapter
