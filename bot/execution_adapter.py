"""
Execution Adapter Module.

Provides abstract interface and implementations for order execution across
different trading modes (paper, testnet, live).
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"  # Good Till Cancelled

    # Metadata
    signal_confidence: Optional[float] = None
    signal_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OrderResult:
    """Result of an order execution."""

    success: bool
    order_id: str
    status: OrderStatus
    filled_quantity: float
    average_price: float
    commission: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # For paper trading simulation
    simulated: bool = False
    slippage_applied: float = 0.0


@dataclass
class Position:
    """Position representation."""

    symbol: str
    quantity: float
    entry_price: float
    side: str  # "long" or "short"
    unrealized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)

    @property
    def value(self) -> float:
        """Current position value."""
        return abs(self.quantity) * self.entry_price


@dataclass
class Balance:
    """Account balance representation."""

    total: float
    available: float
    in_positions: float
    unrealized_pnl: float = 0.0
    currency: str = "USDT"


class ExecutionAdapter(ABC):
    """Abstract base class for order execution."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange/service."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get current order status."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    async def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_balance(self) -> Balance:
        """Get account balance."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


class PaperExecutionAdapter(ExecutionAdapter):
    """Paper trading execution - simulates orders without real execution."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.0005,  # 0.05% slippage
        config: Optional[Dict] = None,
    ):
        super().__init__(config)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # State
        self._balance = initial_balance
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_results: Dict[str, OrderResult] = {}
        self._prices: Dict[str, float] = {}
        self._order_counter = 0

    async def connect(self) -> bool:
        """Paper trading doesn't need real connection."""
        self._connected = True
        logger.info("Paper execution adapter connected")
        return True

    async def disconnect(self) -> None:
        """Disconnect paper adapter."""
        self._connected = False
        logger.info("Paper execution adapter disconnected")

    def set_price(self, symbol: str, price: float) -> None:
        """Set simulated price for a symbol."""
        self._prices[symbol] = price

    async def get_current_price(self, symbol: str) -> float:
        """Get simulated current price."""
        if symbol in self._prices:
            return self._prices[symbol]
        # Default price for testing
        return 50000.0 if "BTC" in symbol else 3000.0

    async def place_order(self, order: Order) -> OrderResult:
        """Simulate order placement."""
        self._order_counter += 1
        order_id = f"PAPER_{int(time.time())}_{self._order_counter}"

        current_price = await self.get_current_price(order.symbol)

        # Apply slippage
        slippage = current_price * self.slippage_rate
        if order.side == OrderSide.BUY:
            fill_price = current_price + slippage
        else:
            fill_price = current_price - slippage

        # Calculate commission
        commission = order.quantity * fill_price * self.commission_rate

        # Check if we have enough balance
        if order.side == OrderSide.BUY:
            required = order.quantity * fill_price + commission
            if required > self._balance:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_price=0,
                    error_message="Insufficient balance",
                    simulated=True,
                )

        # Execute the simulated order
        if order.side == OrderSide.BUY:
            self._balance -= order.quantity * fill_price + commission
            # Update or create position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                # Average up
                total_qty = pos.quantity + order.quantity
                pos.entry_price = (
                    pos.quantity * pos.entry_price + order.quantity * fill_price
                ) / total_qty
                pos.quantity = total_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    side="long",
                )
        else:
            # Sell - close or reduce position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                realized_pnl = order.quantity * (fill_price - pos.entry_price)
                self._balance += order.quantity * fill_price - commission + realized_pnl
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self._positions[order.symbol]
            else:
                # Short selling (simplified)
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=-order.quantity,
                    entry_price=fill_price,
                    side="short",
                )

        result = OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=fill_price,
            commission=commission,
            simulated=True,
            slippage_applied=slippage,
        )

        self._orders[order_id] = order
        self._order_results[order_id] = result

        logger.info(
            f"Paper order executed: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {fill_price:.2f} (slippage: {slippage:.2f})"
        )

        return result

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order (not applicable for instant fills)."""
        return True

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status."""
        if order_id in self._order_results:
            return self._order_results[order_id].status
        return OrderStatus.PENDING

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        pos = self._positions.get(symbol)
        if pos:
            current_price = await self.get_current_price(symbol)
            if pos.side == "long":
                pos.unrealized_pnl = pos.quantity * (current_price - pos.entry_price)
            else:
                pos.unrealized_pnl = abs(pos.quantity) * (
                    pos.entry_price - current_price
                )
        return pos

    async def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        positions = []
        for symbol in self._positions:
            pos = await self.get_position(symbol)
            if pos:
                positions.append(pos)
        return positions

    async def get_balance(self) -> Balance:
        """Get current balance."""
        positions = await self.get_all_positions()
        in_positions = sum(p.value for p in positions)
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)

        return Balance(
            total=self._balance + in_positions + unrealized_pnl,
            available=self._balance,
            in_positions=in_positions,
            unrealized_pnl=unrealized_pnl,
        )


class TestnetExecutionAdapter(ExecutionAdapter):
    """Testnet execution - real orders on exchange testnet."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: Optional[Dict] = None,
    ):
        super().__init__(config)
        self.api_key = api_key
        self.api_secret = api_secret
        self._exchange = None

    async def connect(self) -> bool:
        """Connect to testnet exchange."""
        try:
            import ccxt.async_support as ccxt

            self._exchange = ccxt.binance(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "sandbox": True,  # Enable testnet
                    "options": {"defaultType": "future"},
                }
            )
            # Test connection
            await self._exchange.load_markets()
            self._connected = True
            logger.info("Testnet execution adapter connected to Binance testnet")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to testnet: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._exchange:
            await self._exchange.close()
        self._connected = False
        logger.info("Testnet execution adapter disconnected")

    async def get_current_price(self, symbol: str) -> float:
        """Get current market price from testnet."""
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")
        ticker = await self._exchange.fetch_ticker(symbol)
        return ticker["last"]

    async def place_order(self, order: Order) -> OrderResult:
        """Place order on testnet."""
        if not self._exchange:
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_price=0,
                error_message="Not connected to exchange",
            )

        try:
            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_market_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self._exchange.create_limit_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                    price=order.price,
                )
            else:
                return OrderResult(
                    success=False,
                    order_id="",
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_price=0,
                    error_message=f"Unsupported order type: {order.order_type}",
                )

            return OrderResult(
                success=True,
                order_id=result["id"],
                status=OrderStatus.FILLED
                if result["status"] == "closed"
                else OrderStatus.OPEN,
                filled_quantity=result.get("filled", 0),
                average_price=result.get("average", result.get("price", 0)),
                commission=result.get("fee", {}).get("cost", 0),
            )

        except Exception as e:
            logger.error(f"Testnet order failed: {e}")
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_price=0,
                error_message=str(e),
            )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on testnet."""
        if not self._exchange:
            return False
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status from testnet."""
        if not self._exchange:
            return OrderStatus.PENDING
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            status_map = {
                "open": OrderStatus.OPEN,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
            }
            return status_map.get(order["status"], OrderStatus.PENDING)
        except Exception:
            return OrderStatus.PENDING

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position from testnet."""
        if not self._exchange:
            return None
        try:
            positions = await self._exchange.fetch_positions([symbol])
            for pos in positions:
                if pos["symbol"] == symbol and float(pos["contracts"]) != 0:
                    return Position(
                        symbol=symbol,
                        quantity=float(pos["contracts"]),
                        entry_price=float(pos["entryPrice"]),
                        side="long" if pos["side"] == "long" else "short",
                        unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    )
            return None
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    async def get_all_positions(self) -> List[Position]:
        """Get all positions from testnet."""
        if not self._exchange:
            return []
        try:
            positions = await self._exchange.fetch_positions()
            result = []
            for pos in positions:
                if float(pos["contracts"]) != 0:
                    result.append(
                        Position(
                            symbol=pos["symbol"],
                            quantity=float(pos["contracts"]),
                            entry_price=float(pos["entryPrice"]),
                            side="long" if pos["side"] == "long" else "short",
                            unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                        )
                    )
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_balance(self) -> Balance:
        """Get balance from testnet."""
        if not self._exchange:
            return Balance(total=0, available=0, in_positions=0)
        try:
            balance = await self._exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return Balance(
                total=float(usdt.get("total", 0)),
                available=float(usdt.get("free", 0)),
                in_positions=float(usdt.get("used", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Balance(total=0, available=0, in_positions=0)


class LiveExecutionAdapter(ExecutionAdapter):
    """Live execution - real orders on production exchange."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        safety_controller: Any = None,  # Will be injected
        config: Optional[Dict] = None,
    ):
        super().__init__(config)
        self.api_key = api_key
        self.api_secret = api_secret
        self.safety_controller = safety_controller
        self._exchange = None

    async def connect(self) -> bool:
        """Connect to live exchange."""
        try:
            import ccxt.async_support as ccxt

            self._exchange = ccxt.binance(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "sandbox": False,  # Production
                    "options": {"defaultType": "spot"},
                }
            )
            await self._exchange.load_markets()
            self._connected = True
            logger.warning("LIVE execution adapter connected - REAL MONEY AT RISK")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to live exchange: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._exchange:
            await self._exchange.close()
        self._connected = False
        logger.info("Live execution adapter disconnected")

    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        if not self._exchange:
            raise RuntimeError("Not connected to exchange")
        ticker = await self._exchange.fetch_ticker(symbol)
        return ticker["last"]

    async def place_order(self, order: Order) -> OrderResult:
        """Place live order with safety checks."""
        if not self._exchange:
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_price=0,
                error_message="Not connected to exchange",
            )

        # Safety check before placing live order
        if self.safety_controller:
            is_safe, reason = self.safety_controller.pre_trade_check(order)
            if not is_safe:
                logger.warning(f"Order blocked by safety controller: {reason}")
                return OrderResult(
                    success=False,
                    order_id="",
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_price=0,
                    error_message=f"Safety check failed: {reason}",
                )

        try:
            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_market_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self._exchange.create_limit_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.quantity,
                    price=order.price,
                )
            else:
                return OrderResult(
                    success=False,
                    order_id="",
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    average_price=0,
                    error_message=f"Unsupported order type: {order.order_type}",
                )

            order_result = OrderResult(
                success=True,
                order_id=result["id"],
                status=OrderStatus.FILLED
                if result["status"] == "closed"
                else OrderStatus.OPEN,
                filled_quantity=result.get("filled", 0),
                average_price=result.get("average", result.get("price", 0)),
                commission=result.get("fee", {}).get("cost", 0),
            )

            # Post-trade safety check
            if self.safety_controller:
                self.safety_controller.post_trade_check(order_result)

            logger.info(
                f"LIVE order executed: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order_result.average_price}"
            )

            return order_result

        except Exception as e:
            logger.error(f"Live order failed: {e}")
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                average_price=0,
                error_message=str(e),
            )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel live order."""
        if not self._exchange:
            return False
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Get order status."""
        if not self._exchange:
            return OrderStatus.PENDING
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            status_map = {
                "open": OrderStatus.OPEN,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
            }
            return status_map.get(order["status"], OrderStatus.PENDING)
        except Exception:
            return OrderStatus.PENDING

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position (for spot, calculate from balance)."""
        if not self._exchange:
            return None
        try:
            balance = await self._exchange.fetch_balance()
            base_currency = symbol.split("/")[0]  # e.g., BTC from BTC/USDT
            if base_currency in balance and float(balance[base_currency]["total"]) > 0:
                return Position(
                    symbol=symbol,
                    quantity=float(balance[base_currency]["total"]),
                    entry_price=0,  # Would need trade history
                    side="long",
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None

    async def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        if not self._exchange:
            return []
        try:
            balance = await self._exchange.fetch_balance()
            positions = []
            for currency, amounts in balance.items():
                if currency not in ["USDT", "USD", "info", "free", "used", "total"]:
                    if float(amounts.get("total", 0)) > 0.0001:
                        positions.append(
                            Position(
                                symbol=f"{currency}/USDT",
                                quantity=float(amounts["total"]),
                                entry_price=0,
                                side="long",
                            )
                        )
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_balance(self) -> Balance:
        """Get live balance."""
        if not self._exchange:
            return Balance(total=0, available=0, in_positions=0)
        try:
            balance = await self._exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return Balance(
                total=float(usdt.get("total", 0)),
                available=float(usdt.get("free", 0)),
                in_positions=float(usdt.get("used", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return Balance(total=0, available=0, in_positions=0)


def create_execution_adapter(
    mode: str,
    initial_balance: float = 10000.0,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    safety_controller: Any = None,
) -> ExecutionAdapter:
    """Factory function to create appropriate execution adapter."""
    from bot.trading_mode import TradingMode

    mode_enum = TradingMode(mode) if isinstance(mode, str) else mode

    if mode_enum in (
        TradingMode.BACKTEST,
        TradingMode.PAPER_SYNTHETIC,
        TradingMode.PAPER_LIVE_DATA,
    ):
        return PaperExecutionAdapter(initial_balance=initial_balance)

    elif mode_enum == TradingMode.TESTNET:
        if not api_key or not api_secret:
            logger.warning("No API keys for testnet, falling back to paper")
            return PaperExecutionAdapter(initial_balance=initial_balance)
        return TestnetExecutionAdapter(api_key=api_key, api_secret=api_secret)

    elif mode_enum in (TradingMode.LIVE_LIMITED, TradingMode.LIVE_FULL):
        if not api_key or not api_secret:
            raise ValueError("API keys required for live trading")
        return LiveExecutionAdapter(
            api_key=api_key,
            api_secret=api_secret,
            safety_controller=safety_controller,
        )

    else:
        return PaperExecutionAdapter(initial_balance=initial_balance)
