"""
Broker Router - Multi-Asset Execution Layer.

Routes orders to the appropriate broker based on asset type:
- Crypto -> Binance (via execution_adapter)
- Stocks -> Alpaca
- Commodities -> IBKR or futures broker
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from bot.execution_adapter import ExecutionAdapter, Order, Position
from bot.alpaca_adapter import AlpacaAdapter, create_alpaca_adapter
from bot.oanda_adapter import OANDAAdapter, create_oanda_adapter

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Asset type classification."""

    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    FOREX = "forex"
    INDEX = "index"


class BrokerRouter:
    """
    Routes orders to appropriate brokers based on asset type.

    Asset Type Detection:
    - /USDT, /USD, /BUSD -> Crypto
    - AAPL, MSFT, GOOGL -> Stocks
    - XAU/USD, XAGUSD, USOIL -> Commodities
    - EUR/USD, GBP/USD -> Forex
    """

    def __init__(
        self,
        crypto_adapter: Optional[ExecutionAdapter] = None,
        stock_adapter: Optional[AlpacaAdapter] = None,
        commodity_adapter: Optional[ExecutionAdapter] = None,
        forex_adapter: Optional[ExecutionAdapter] = None,
        index_adapter: Optional[ExecutionAdapter] = None,
    ):
        self.adapters: Dict[AssetType, ExecutionAdapter] = {}

        if crypto_adapter:
            self.adapters[AssetType.CRYPTO] = crypto_adapter
        if stock_adapter:
            self.adapters[AssetType.STOCK] = stock_adapter
        if commodity_adapter:
            self.adapters[AssetType.COMMODITY] = commodity_adapter
        if forex_adapter:
            self.adapters[AssetType.FOREX] = forex_adapter
        if index_adapter:
            self.adapters[AssetType.INDEX] = index_adapter

        logger.info(f"Broker router initialized with {len(self.adapters)} adapters")

    def detect_asset_type(self, symbol: str) -> AssetType:
        """Detect asset type from symbol."""
        symbol_upper = symbol.upper()

        # Index patterns (check first to avoid misclassification)
        index_patterns = [
            "SPX500",
            "NAS100",
            "US30",
            "UK100",
            "DE30",
            "JP225",
            "^GSPC",
            "^DJI",
            "^IXIC",
            "SPX",
            "DJX",
        ]
        if any(index in symbol_upper for index in index_patterns):
            return AssetType.INDEX

        # Commodity patterns (precious metals, oil, gas)
        commodity_patterns = [
            "XAU",
            "XAG",
            "WTICO",
            "BCO",
            "NATGAS",
            "XCU",
            "WHEAT",
            "CORN",
            "SOYBN",
            "SUGAR",
            "GOLD",
            "SILVER",
            "OIL",
            "GAS",
            "CL=",
            "GC=",
            "SI=",
            "NG=",
        ]
        if any(name in symbol_upper for name in commodity_patterns):
            return AssetType.COMMODITY

        # Forex patterns (major and cross currency pairs)
        forex_currencies = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        if "/" in symbol:
            parts = symbol_upper.split("/")
            # Check if both parts are forex currencies or USD
            if len(parts) == 2:
                base, quote = parts
                if (
                    base in forex_currencies or quote in forex_currencies
                ) and "USDT" not in symbol_upper:
                    # Make sure it's not crypto
                    crypto_bases = [
                        "BTC",
                        "ETH",
                        "SOL",
                        "AVAX",
                        "XRP",
                        "ADA",
                        "MATIC",
                        "DOT",
                        "LINK",
                        "ATOM",
                    ]
                    if base not in crypto_bases:
                        return AssetType.FOREX

        # Crypto patterns
        if any(suffix in symbol_upper for suffix in ["/USDT", "/BUSD", "/BTC", "/ETH"]):
            return AssetType.CRYPTO

        # Stock patterns (US equities)
        stock_symbols = [
            "AAPL",
            "NVDA",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "AMD",
            "NFLX",
            "UBER",
            "DIS",
            "BA",
            "JPM",
            "V",
            "MA",
            "PYPL",
            "INTC",
            "CRM",
            "ORCL",
            "CSCO",
        ]
        if "/USD" in symbol_upper:
            base = symbol_upper.split("/")[0]
            if base in stock_symbols:
                return AssetType.STOCK

        # Crypto with /USD (but not forex or stocks)
        if "/USD" in symbol_upper:
            crypto_bases = [
                "BTC",
                "ETH",
                "SOL",
                "AVAX",
                "XRP",
                "ADA",
                "MATIC",
                "DOT",
                "LINK",
                "ATOM",
                "DOGE",
                "SHIB",
            ]
            for crypto in crypto_bases:
                if symbol_upper.startswith(crypto):
                    return AssetType.CRYPTO

        # Default to stock if it's a simple ticker (1-5 letters, no slash)
        if "/" not in symbol and symbol.replace(".", "").isalpha() and 1 <= len(symbol) <= 5:
            return AssetType.STOCK

        # Fallback
        logger.warning(f"Could not detect asset type for {symbol}, defaulting to CRYPTO")
        return AssetType.CRYPTO

    def get_adapter(self, symbol: str) -> Optional[ExecutionAdapter]:
        """Get the appropriate adapter for a symbol."""
        asset_type = self.detect_asset_type(symbol)
        adapter = self.adapters.get(asset_type)

        if not adapter:
            logger.warning(f"No adapter for {asset_type.value} (symbol: {symbol})")
            return None

        return adapter

    async def initialize_all(self) -> Dict[AssetType, bool]:
        """Initialize all adapters."""
        results = {}
        for asset_type, adapter in self.adapters.items():
            try:
                # Try connect() first (standard ExecutionAdapter interface)
                # then fall back to initialize() for adapters that use that name
                if hasattr(adapter, "connect"):
                    success = await adapter.connect()
                elif hasattr(adapter, "initialize"):
                    success = await adapter.initialize()
                else:
                    success = True  # No init needed
                results[asset_type] = success
                logger.info(f"{asset_type.value} adapter: {'✓' if success else '✗'}")
            except Exception as e:
                logger.error(f"Failed to initialize {asset_type.value}: {e}")
                results[asset_type] = False
        return results

    async def get_balance(self, asset_type: Optional[AssetType] = None) -> Dict[AssetType, float]:
        """Get balance from all adapters or specific asset type."""
        balances = {}

        if asset_type:
            adapter = self.adapters.get(asset_type)
            if adapter:
                balance = await adapter.get_balance()
                balances[asset_type] = float(balance)
        else:
            for asset_type, adapter in self.adapters.items():
                try:
                    balance = await adapter.get_balance()
                    balances[asset_type] = float(balance)
                except Exception as e:
                    logger.error(f"Failed to get {asset_type.value} balance: {e}")
                    balances[asset_type] = 0.0

        return balances

    async def get_all_positions(self) -> Dict[AssetType, List[Position]]:
        """Get positions from all adapters."""
        all_positions = {}

        for asset_type, adapter in self.adapters.items():
            try:
                positions = await adapter.get_positions()
                if positions:
                    all_positions[asset_type] = positions
                    logger.info(f"{asset_type.value}: {len(positions)} positions")
            except Exception as e:
                logger.error(f"Failed to get {asset_type.value} positions: {e}")

        return all_positions

    async def execute_order(self, order: Order) -> bool:
        """Route order to appropriate adapter."""
        adapter = self.get_adapter(order.symbol)

        if not adapter:
            logger.error(f"No adapter available for {order.symbol}")
            return False

        asset_type = self.detect_asset_type(order.symbol)
        logger.info(
            f"Routing {order.side.value} order for {order.symbol} to {asset_type.value} adapter"
        )

        return await adapter.execute_order(order)

    async def close_position(self, symbol: str) -> bool:
        """Close position via appropriate adapter."""
        adapter = self.get_adapter(symbol)

        if not adapter:
            logger.error(f"No adapter available for {symbol}")
            return False

        return await adapter.close_position(symbol)

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get market price via appropriate adapter."""
        adapter = self.get_adapter(symbol)

        if not adapter:
            return None

        # Use get_current_price which is the standard ExecutionAdapter method
        if hasattr(adapter, "get_current_price"):
            return await adapter.get_current_price(symbol)
        elif hasattr(adapter, "get_market_price"):
            return await adapter.get_market_price(symbol)
        return None


class MultiAssetAdapter(ExecutionAdapter):
    """
    ExecutionAdapter implementation that wraps BrokerRouter.

    This allows the unified engine to use multi-broker routing transparently
    while maintaining the standard ExecutionAdapter interface.
    """

    def __init__(self, router: "BrokerRouter", mode: str = "paper"):
        self._router = router
        self._mode = mode
        self._connected = False

    async def connect(self) -> bool:
        """Connect all adapters in the router."""
        results = await self._router.initialize_all()
        self._connected = any(results.values())
        return self._connected

    async def disconnect(self) -> None:
        """Disconnect all adapters."""
        for adapter in self._router.adapters.values():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting adapter: {e}")
        self._connected = False

    async def get_balance(self):
        """Get total balance across all adapters."""
        from bot.execution_adapter import Balance

        total = 0.0
        available = 0.0
        in_positions = 0.0
        unrealized_pnl = 0.0

        for asset_type, adapter in self._router.adapters.items():
            try:
                balance = await adapter.get_balance()
                if balance is None:
                    continue
                if hasattr(balance, "total"):
                    # Safely extract values with None handling
                    b_total = getattr(balance, "total", None)
                    b_avail = getattr(balance, "available", None)
                    b_in_pos = getattr(balance, "in_positions", None)
                    b_unrealized = getattr(balance, "unrealized_pnl", None)

                    if b_total is not None:
                        total += float(b_total)
                    if b_avail is not None:
                        available += float(b_avail)
                    if b_in_pos is not None:
                        in_positions += float(b_in_pos)
                    if b_unrealized is not None:
                        unrealized_pnl += float(b_unrealized)
                elif balance is not None:
                    # Simple float balance
                    try:
                        val = float(balance)
                        total += val
                        available += val
                    except (TypeError, ValueError):
                        pass
            except Exception as e:
                logger.warning(f"Error getting {asset_type.value} balance: {e}")

        return Balance(
            total=total,
            available=available,
            in_positions=in_positions,
            unrealized_pnl=unrealized_pnl,
            currency="USD",
        )

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return await self._router.get_market_price(symbol)

    async def place_order(self, order: Order):
        """Place an order via the appropriate adapter."""
        from bot.execution_adapter import OrderResult, OrderStatus

        adapter = self._router.get_adapter(order.symbol)
        if not adapter:
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.FAILED,
                filled_quantity=0.0,
                average_price=0.0,
                error_message=f"No adapter for {order.symbol}",
            )

        return await adapter.place_order(order)

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        adapter = self._router.get_adapter(symbol)
        if not adapter:
            return None
        return await adapter.get_position(symbol)

    async def get_all_positions(self) -> List[Position]:
        """Get all positions from all adapters."""
        all_positions = []
        for adapter in self._router.adapters.values():
            try:
                positions = await adapter.get_all_positions()
                if positions:
                    all_positions.extend(positions)
            except Exception as e:
                logger.warning(f"Error getting positions: {e}")
        return all_positions

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order via the appropriate adapter."""
        adapter = self._router.get_adapter(symbol)
        if not adapter:
            return False
        return await adapter.cancel_order(order_id, symbol)

    async def get_order_status(self, order_id: str, symbol: str):
        """Get order status via the appropriate adapter."""
        from bot.execution_adapter import OrderStatus

        adapter = self._router.get_adapter(symbol)
        if not adapter:
            return OrderStatus.PENDING
        return await adapter.get_order_status(order_id, symbol)


def create_broker_router(mode: str = "paper", config: Optional[Dict] = None) -> BrokerRouter:
    """
    Factory function to create broker router with configured adapters.

    Args:
        mode: Trading mode (paper, testnet, live)
        config: Configuration dict with broker settings

    Returns:
        Configured BrokerRouter instance
    """
    from bot.execution_adapter import create_execution_adapter

    # Create crypto adapter (Binance)
    initial_balance = config.get("initial_balance", 30000.0) if config else 30000.0
    crypto_adapter = create_execution_adapter(mode, initial_balance=initial_balance)

    # Create stock adapter (Alpaca) - paper mode by default
    # Always try to create if credentials exist
    stock_adapter = None
    try:
        import os

        if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"):
            is_paper = mode in ["paper", "paper_live_data"] or (
                config and config.get("stocks", {}).get("alpaca", {}).get("paper_mode", True)
            )
            stock_adapter = create_alpaca_adapter(is_paper=is_paper)
            logger.info(f"Alpaca adapter created ({'paper' if is_paper else 'live'} mode)")
    except Exception as e:
        logger.warning(f"Alpaca adapter not available: {e}")

    # Create OANDA adapter for commodities and forex
    oanda_adapter = None
    try:
        import os

        if os.getenv("OANDA_API_KEY") and os.getenv("OANDA_ACCOUNT_ID"):
            # Use environment setting from .env - don't override based on trading mode
            # OANDA API key type must match environment (live key -> live env)
            env = os.getenv("OANDA_ENVIRONMENT", "live")
            oanda_adapter = create_oanda_adapter(environment=env)
            logger.info(f"OANDA adapter created ({env} mode)")
    except Exception as e:
        logger.warning(f"OANDA adapter not available: {e}")

    # Use OANDA for commodities, forex, and indices
    commodity_adapter = oanda_adapter
    forex_adapter = oanda_adapter
    index_adapter = oanda_adapter

    router = BrokerRouter(
        crypto_adapter=crypto_adapter,
        stock_adapter=stock_adapter,
        commodity_adapter=commodity_adapter,
        forex_adapter=forex_adapter,
        index_adapter=index_adapter,
    )

    return router


def create_multi_asset_adapter(
    mode: str = "paper", config: Optional[Dict] = None
) -> MultiAssetAdapter:
    """
    Factory function to create a MultiAssetAdapter.

    This is a drop-in replacement for create_execution_adapter
    that supports multi-broker routing for different asset classes.

    Args:
        mode: Trading mode (paper, testnet, live)
        config: Configuration dict with broker settings

    Returns:
        MultiAssetAdapter that routes to appropriate brokers
    """
    router = create_broker_router(mode, config)
    return MultiAssetAdapter(router, mode)
