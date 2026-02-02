"""
OANDA Adapter - Forex & Commodities Trading Integration.

Integrates with OANDA v20 API for trading:
- Forex pairs (EUR/USD, GBP/USD, etc.)
- Commodities (Gold, Silver, Oil)
- Indices (SPX500, NAS100, UK100)
"""

import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from bot.execution_adapter import (
    ExecutionAdapter,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    OrderResult,
    Balance,
)

logger = logging.getLogger(__name__)


class OANDAAdapter(ExecutionAdapter):
    """
    OANDA v20 execution adapter for forex and commodities trading.

    Features:
    - Forex, metals, indices, commodities trading
    - Practice and live account support
    - Real-time streaming prices
    - REST API integration
    """

    # OANDA instrument mapping
    INSTRUMENT_MAP = {
        # Commodities
        "XAU/USD": "XAU_USD",  # Gold
        "XAG/USD": "XAG_USD",  # Silver
        "WTICO/USD": "WTICO_USD",  # WTI Crude Oil
        "BCO/USD": "BCO_USD",  # Brent Crude Oil
        "NATGAS/USD": "NATGAS_USD",  # Natural Gas
        "XCU/USD": "XCU_USD",  # Copper
        "WHEAT/USD": "WHEAT_USD",  # Wheat
        "CORN/USD": "CORN_USD",  # Corn
        "SOYBN/USD": "SOYBN_USD",  # Soybeans
        "SUGAR/USD": "SUGAR_USD",  # Sugar
        # Gold in other currencies
        "XAU/GBP": "XAU_GBP",
        "XAU/EUR": "XAU_EUR",
        # Major Forex
        "EUR/USD": "EUR_USD",
        "GBP/USD": "GBP_USD",
        "USD/JPY": "USD_JPY",
        "USD/CHF": "USD_CHF",
        "AUD/USD": "AUD_USD",
        "NZD/USD": "NZD_USD",
        "USD/CAD": "USD_CAD",
        # Forex Crosses
        "EUR/GBP": "EUR_GBP",
        "EUR/JPY": "EUR_JPY",
        "GBP/JPY": "GBP_JPY",
        "EUR/CHF": "EUR_CHF",
        # Indices
        "SPX500/USD": "SPX500_USD",
        "NAS100/USD": "NAS100_USD",
        "US30/USD": "US30_USD",
        "UK100/GBP": "UK100_GBP",
        "DE30/EUR": "DE30_EUR",
        "JP225/USD": "JP225_USD",
    }

    def __init__(
        self,
        api_key: str,
        account_id: str,
        environment: str = "practice",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment

        # Set base URL based on environment
        if base_url:
            self.base_url = base_url
        elif environment == "practice":
            self.base_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"

        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

        self._account_currency = "GBP"  # Will be updated on initialize

        logger.info(f"OANDA adapter initialized: {environment.upper()} - Account: {account_id}")

        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to OANDA."""
        success = await self.initialize()
        self._connected = success
        return success

    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        logger.info("OANDA adapter disconnected")

    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        price = await self.get_market_price(symbol)
        return price if price else 0.0

    async def place_order(self, order: Order) -> OrderResult:
        """Place an order and return result."""
        success = await self.execute_order(order)
        return OrderResult(
            success=success,
            order_id=f"OANDA_{int(datetime.now().timestamp())}",
            status=OrderStatus.FILLED if success else OrderStatus.FAILED,
            filled_quantity=order.quantity if success else 0.0,
            average_price=0.0,  # Will be updated by fill
            error_message=None if success else "Order failed",
        )

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return await self.get_positions()

    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol to OANDA instrument format."""
        # Check mapping first
        if symbol in self.INSTRUMENT_MAP:
            return self.INSTRUMENT_MAP[symbol]

        # Default: replace / with _
        return symbol.replace("/", "_")

    def _convert_from_oanda(self, instrument: str) -> str:
        """Convert OANDA instrument back to standard symbol."""
        # Reverse lookup
        for standard, oanda in self.INSTRUMENT_MAP.items():
            if oanda == instrument:
                return standard

        # Default: replace _ with /
        return instrument.replace("_", "/")

    async def initialize(self) -> bool:
        """Initialize connection and verify credentials."""
        try:
            account = self._get_account()
            self._account_currency = account.get("currency", "GBP")
            balance = float(account.get("balance", 0))

            logger.info(f"OANDA account verified: {self.account_id}")
            logger.info(f"Balance: {self._account_currency} {balance:.2f}")
            logger.info(f"NAV: {self._account_currency} {float(account.get('NAV', 0)):.2f}")
            return True
        except Exception as e:
            logger.error(f"OANDA initialization failed: {e}")
            return False

    def _get_account(self) -> Dict:
        """Get account information."""
        response = self._session.get(f"{self.base_url}/v3/accounts/{self.account_id}/summary")
        response.raise_for_status()
        return response.json().get("account", {})

    async def get_balance(self) -> Balance:
        """Get account balance."""
        try:
            account = self._get_account()
            return Balance(
                total=float(account.get("NAV", 0)),
                available=float(account.get("marginAvailable", 0)),
                in_positions=float(account.get("marginUsed", 0)),
                unrealized_pnl=float(account.get("unrealizedPL", 0)),
                currency=account.get("currency", "GBP"),
            )
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Balance(total=0, available=0, in_positions=0)

    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        try:
            response = self._session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
            )
            response.raise_for_status()
            positions_data = response.json().get("positions", [])

            positions = []
            for pos in positions_data:
                instrument = pos["instrument"]
                symbol = self._convert_from_oanda(instrument)

                # OANDA separates long and short
                long_units = float(pos.get("long", {}).get("units", 0))
                short_units = float(pos.get("short", {}).get("units", 0))

                if long_units > 0:
                    long_data = pos["long"]
                    position = Position(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=long_units,
                        entry_price=float(long_data.get("averagePrice", 0)),
                        current_price=float(
                            long_data.get("averagePrice", 0)
                        ),  # Will update with live price
                        unrealized_pnl=float(long_data.get("unrealizedPL", 0)),
                        unrealized_pnl_pct=0,  # Calculate separately
                    )
                    positions.append(position)

                if abs(short_units) > 0:
                    short_data = pos["short"]
                    position = Position(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(short_units),
                        entry_price=float(short_data.get("averagePrice", 0)),
                        current_price=float(short_data.get("averagePrice", 0)),
                        unrealized_pnl=float(short_data.get("unrealizedPL", 0)),
                        unrealized_pnl_pct=0,
                    )
                    positions.append(position)

            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    # Instruments that require integer units
    INTEGER_UNIT_INSTRUMENTS = {
        "SPX500_USD", "NAS100_USD", "US30_USD", "UK100_GBP",
        "DE30_EUR", "JP225_USD", "XAU_USD", "XAG_USD",
    }

    async def execute_order(self, order: Order) -> bool:
        """Execute a market order."""
        try:
            instrument = self._convert_symbol(order.symbol)

            # OANDA uses negative units for sell orders
            units = order.quantity if order.side == OrderSide.BUY else -order.quantity

            # Some instruments (indices, metals) require integer units
            if instrument in self.INTEGER_UNIT_INSTRUMENTS:
                units = int(round(units))
                if units == 0:
                    units = 1 if order.side == OrderSide.BUY else -1
                    logger.warning(f"Adjusted units for {instrument} to minimum: {units}")

            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(int(units)) if units == int(units) else str(units),
                    "type": "MARKET",
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT",
                }
            }

            # Add stop loss
            if order.stop_loss:
                order_data["order"]["stopLossOnFill"] = {"price": str(round(order.stop_loss, 5))}

            # Add take profit
            if order.take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(round(order.take_profit, 5))
                }

            response = self._session.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders", json=order_data
            )
            response.raise_for_status()

            result = response.json()

            # Check if order was filled
            if "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                logger.info(
                    f"OANDA order filled: {fill['id']} - "
                    f"{order.side.value} {order.quantity} {instrument} @ {fill.get('price', 'market')}"
                )
                return True
            elif "orderCancelTransaction" in result:
                cancel = result["orderCancelTransaction"]
                logger.warning(f"OANDA order cancelled: {cancel.get('reason', 'Unknown')}")
                return False
            else:
                logger.info(f"OANDA order submitted: {result}")
                return True

        except requests.exceptions.HTTPError as e:
            error_body = {}
            error_text = ""
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except Exception:
                    error_text = e.response.text[:500] if e.response.text else "empty"
            logger.error(
                f"OANDA order failed: {e} | "
                f"instrument={instrument} units={units} | "
                f"body={error_body or error_text}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            return False

    async def cancel_order(self, order_id: str, symbol: str = "") -> bool:
        """Cancel an open order."""
        try:
            response = self._session.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            )
            response.raise_for_status()
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str = "") -> OrderStatus:
        """Get order status."""
        try:
            response = self._session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}"
            )
            response.raise_for_status()
            order_data = response.json().get("order", {})

            status_map = {
                "PENDING": OrderStatus.PENDING,
                "FILLED": OrderStatus.FILLED,
                "TRIGGERED": OrderStatus.OPEN,
                "CANCELLED": OrderStatus.CANCELLED,
            }

            return status_map.get(order_data.get("state"), OrderStatus.OPEN)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return OrderStatus.FAILED

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price (mid price)."""
        try:
            instrument = self._convert_symbol(symbol)

            response = self._session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                params={"instruments": instrument},
            )
            response.raise_for_status()

            prices = response.json().get("prices", [])
            if prices:
                bid = float(prices[0].get("bids", [{}])[0].get("price", 0))
                ask = float(prices[0].get("asks", [{}])[0].get("price", 0))
                return (bid + ask) / 2

            return None
        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
            return None

    async def close_position(self, symbol: str) -> bool:
        """Close all positions for a symbol."""
        try:
            instrument = self._convert_symbol(symbol)

            # Close all units (longUnits=ALL or shortUnits=ALL)
            response = self._session.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close",
                json={"longUnits": "ALL", "shortUnits": "ALL"},
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Position closed: {symbol} - {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    async def get_historical_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> Optional[list]:
        """Get historical OHLCV data."""
        try:
            instrument = self._convert_symbol(symbol)

            # Map timeframe to OANDA granularity
            timeframe_map = {
                "1m": "M1",
                "5m": "M5",
                "15m": "M15",
                "30m": "M30",
                "1h": "H1",
                "4h": "H4",
                "1d": "D",
                "1w": "W",
            }
            granularity = timeframe_map.get(timeframe, "H1")

            response = self._session.get(
                f"{self.base_url}/v3/instruments/{instrument}/candles",
                params={
                    "granularity": granularity,
                    "count": limit,
                    "price": "M",  # Mid prices
                },
            )
            response.raise_for_status()
            data = response.json()

            # Convert to standard OHLCV format
            bars = []
            for candle in data.get("candles", []):
                if candle.get("complete", False):
                    mid = candle.get("mid", {})
                    bars.append(
                        {
                            "timestamp": candle["time"],
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": float(candle.get("volume", 0)),
                        }
                    )

            return bars
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None

    async def get_tradeable_instruments(self) -> List[Dict]:
        """Get list of tradeable instruments."""
        try:
            response = self._session.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            )
            response.raise_for_status()

            instruments = []
            for inst in response.json().get("instruments", []):
                instruments.append(
                    {
                        "symbol": self._convert_from_oanda(inst["name"]),
                        "oanda_name": inst["name"],
                        "type": inst.get("type", "UNKNOWN"),
                        "display_name": inst.get("displayName", inst["name"]),
                        "pip_location": inst.get("pipLocation", -4),
                        "min_units": float(inst.get("minimumTradeSize", 1)),
                        "margin_rate": float(inst.get("marginRate", 0.05)),
                    }
                )

            return instruments
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return []

    async def get_account_summary(self) -> Dict:
        """Get detailed account summary."""
        try:
            account = self._get_account()
            return {
                "id": account.get("id"),
                "currency": account.get("currency"),
                "balance": float(account.get("balance", 0)),
                "nav": float(account.get("NAV", 0)),
                "unrealized_pl": float(account.get("unrealizedPL", 0)),
                "realized_pl": float(account.get("pl", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "open_trade_count": int(account.get("openTradeCount", 0)),
                "open_position_count": int(account.get("openPositionCount", 0)),
                "pending_order_count": int(account.get("pendingOrderCount", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}


def create_oanda_adapter(environment: str = None) -> Optional[OANDAAdapter]:
    """
    Factory function to create OANDA adapter from environment variables.

    Required env vars:
    - OANDA_API_KEY
    - OANDA_ACCOUNT_ID
    - OANDA_ENVIRONMENT (practice/live)
    """
    api_key = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    env = environment or os.getenv("OANDA_ENVIRONMENT", "practice")

    if not api_key or not account_id:
        logger.error("OANDA credentials not found in environment")
        return None

    return OANDAAdapter(api_key=api_key, account_id=account_id, environment=env)


# Commodity symbols for easy reference
COMMODITY_SYMBOLS = [
    "XAU/USD",  # Gold
    "XAG/USD",  # Silver
    "WTICO/USD",  # WTI Crude Oil
    "BCO/USD",  # Brent Crude Oil
    "NATGAS/USD",  # Natural Gas
    "XCU/USD",  # Copper
]

FOREX_MAJORS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "NZD/USD",
    "USD/CAD",
]

INDEX_SYMBOLS = [
    "SPX500/USD",  # S&P 500
    "NAS100/USD",  # NASDAQ 100
    "US30/USD",  # Dow Jones
    "UK100/GBP",  # FTSE 100
    "DE30/EUR",  # DAX
]
