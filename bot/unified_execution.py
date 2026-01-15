"""
Unified Execution Engine

Provides identical execution behavior across:
- Backtest mode (historical simulation)
- Paper mode (live data, simulated fills)
- Live mode (real exchange execution)

Key features:
- Same fill model, fees, slippage across all modes
- Order management with partial fills, retries, idempotency
- Risk Guardian integration for pre-trade checks
- Detailed execution logging for learning
"""

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class FeeStructure:
    """Fee structure for execution"""
    maker_fee_pct: float = 0.02  # 0.02% maker fee
    taker_fee_pct: float = 0.04  # 0.04% taker fee
    funding_rate_pct: float = 0.01  # 0.01% per 8 hours (for perpetuals)
    min_fee: float = 0.0

    def calculate_fee(self, notional_value: float, is_maker: bool = False) -> float:
        """Calculate fee for a trade"""
        fee_pct = self.maker_fee_pct if is_maker else self.taker_fee_pct
        fee = notional_value * (fee_pct / 100)
        return max(fee, self.min_fee)


@dataclass
class SlippageModel:
    """Slippage model for realistic fills"""
    base_slippage_pct: float = 0.02  # Base slippage 0.02%
    volume_impact_factor: float = 0.1  # Additional slippage per 1% of ADV
    volatility_factor: float = 0.5  # Multiplier for high volatility
    spread_pct: float = 0.01  # Bid-ask spread

    def calculate_slippage(
        self,
        order_size: float,
        avg_daily_volume: float,
        current_volatility: float,
        baseline_volatility: float = 0.02,
        is_buy: bool = True
    ) -> float:
        """
        Calculate expected slippage for an order.

        Returns slippage as percentage (e.g., 0.05 = 0.05%)
        """
        # Volume impact
        volume_pct = (order_size / avg_daily_volume) * 100 if avg_daily_volume > 0 else 0
        volume_slippage = volume_pct * self.volume_impact_factor

        # Volatility impact
        vol_ratio = current_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
        vol_slippage = (vol_ratio - 1) * self.volatility_factor * self.base_slippage_pct if vol_ratio > 1 else 0

        # Spread cost (half spread)
        spread_cost = self.spread_pct / 2

        # Total slippage
        total = self.base_slippage_pct + volume_slippage + vol_slippage + spread_cost

        return total


@dataclass
class Order:
    """Order object"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    leverage: float = 1.0
    reduce_only: bool = False
    time_in_force: str = "GTC"  # GTC, IOC, FOK

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Metadata
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    # Idempotency
    idempotency_key: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED
        ]

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill"""
        return self.quantity - self.filled_quantity

    @property
    def fill_pct(self) -> float:
        """Get fill percentage"""
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0


@dataclass
class Fill:
    """Fill/execution object"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Execution details
    is_maker: bool = False
    slippage_pct: float = 0.0
    expected_price: float = 0.0

    @property
    def notional_value(self) -> float:
        """Get notional value of fill"""
        return self.quantity * self.price


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    leverage: float = 1.0

    # PnL tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees_paid: float = 0.0

    # Risk metrics
    liquidation_price: float = 0.0
    margin_used: float = 0.0

    # Timestamps
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def notional_value(self) -> float:
        """Get position notional value"""
        return abs(self.quantity) * self.current_price

    @property
    def margin_ratio(self) -> float:
        """Get margin utilization ratio"""
        if self.margin_used <= 0:
            return 0.0
        return self.notional_value / self.margin_used

    def update_pnl(self, current_price: float):
        """Update unrealized PnL"""
        self.current_price = current_price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.quantity)
        else:
            self.unrealized_pnl = 0.0
        self.last_updated = datetime.now()


@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order: Order
    fills: List[Fill] = field(default_factory=list)
    error_message: Optional[str] = None

    # Execution metrics
    execution_time_ms: float = 0.0
    retries: int = 0

    @property
    def total_filled(self) -> float:
        """Total quantity filled"""
        return sum(f.quantity for f in self.fills)

    @property
    def average_price(self) -> float:
        """Average fill price"""
        if not self.fills:
            return 0.0
        total_value = sum(f.quantity * f.price for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_value / total_qty if total_qty > 0 else 0.0

    @property
    def total_fees(self) -> float:
        """Total fees paid"""
        return sum(f.fee for f in self.fills)


class ExecutionEngine(ABC):
    """
    Abstract base class for execution engines.
    All modes (backtest, paper, live) implement this interface.
    """

    def __init__(
        self,
        mode: ExecutionMode,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
        risk_guardian: Optional[Any] = None
    ):
        self.mode = mode
        self.fee_structure = fee_structure or FeeStructure()
        self.slippage_model = slippage_model or SlippageModel()
        self.risk_guardian = risk_guardian

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, Position] = {}

        # Idempotency tracking
        self._processed_keys: Dict[str, str] = {}  # idempotency_key -> order_id

        # Execution stats
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_volume = 0.0
        self.total_fees = 0.0

        # Callbacks
        self._on_fill_callbacks: List[Callable] = []
        self._on_order_update_callbacks: List[Callable] = []

    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"{self.mode.value}_{uuid.uuid4().hex[:12]}"

    def generate_fill_id(self) -> str:
        """Generate unique fill ID"""
        return f"fill_{uuid.uuid4().hex[:12]}"

    def generate_idempotency_key(self, symbol: str, side: OrderSide, quantity: float, timestamp: float) -> str:
        """Generate idempotency key for an order"""
        data = f"{symbol}:{side.value}:{quantity}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def on_fill(self, callback: Callable):
        """Register fill callback"""
        self._on_fill_callbacks.append(callback)

    def on_order_update(self, callback: Callable):
        """Register order update callback"""
        self._on_order_update_callbacks.append(callback)

    def _notify_fill(self, fill: Fill):
        """Notify fill callbacks"""
        for callback in self._on_fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _notify_order_update(self, order: Order):
        """Notify order update callbacks"""
        for callback in self._on_order_update_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order update callback error: {e}")

    def _check_risk_guardian(self, order: Order, current_equity: float) -> Tuple[bool, str]:
        """Check order against Risk Guardian"""
        if self.risk_guardian is None:
            return True, ""

        try:
            # Import here to avoid circular dependency
            from bot.risk_guardian import TradeRequest

            # Create trade request
            request = TradeRequest(
                symbol=order.symbol,
                side="long" if order.side == OrderSide.BUY else "short",
                size_pct=0.0,  # Will be calculated from order
                leverage=order.leverage,
                current_equity=current_equity
            )

            # Check with Risk Guardian
            result = self.risk_guardian.check_trade(request)

            if not result.approved:
                return False, f"Risk Guardian rejected: {[v.value for v in result.veto_reasons]}"

            return True, ""
        except ImportError:
            logger.warning("Risk Guardian not available")
            return True, ""
        except Exception as e:
            logger.error(f"Risk Guardian check failed: {e}")
            return True, ""  # Fail open for now

    @abstractmethod
    async def execute_order(self, order: Order, market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute an order - implemented by subclasses"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        """Get current position for symbol"""
        pass

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        leverage: float = 1.0,
        reduce_only: bool = False,
        idempotency_key: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None,
        current_equity: float = 0.0,
        tags: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Submit an order for execution.

        This is the main entry point for order submission across all modes.
        """
        # Check idempotency
        if idempotency_key:
            if idempotency_key in self._processed_keys:
                existing_order_id = self._processed_keys[idempotency_key]
                existing_order = self.orders.get(existing_order_id)
                if existing_order:
                    logger.info(f"Idempotent order already processed: {existing_order_id}")
                    return ExecutionResult(
                        success=True,
                        order=existing_order,
                        fills=[f for f in self.fills if f.order_id == existing_order_id]
                    )

        # Create order
        order = Order(
            order_id=self.generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            leverage=leverage,
            reduce_only=reduce_only,
            idempotency_key=idempotency_key,
            tags=tags or {}
        )

        # Check Risk Guardian
        if current_equity > 0:
            approved, rejection_reason = self._check_risk_guardian(order, current_equity)
            if not approved:
                order.status = OrderStatus.REJECTED
                self.orders[order.order_id] = order
                self.failed_orders += 1
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=rejection_reason
                )

        # Track order
        self.orders[order.order_id] = order
        self.total_orders += 1

        # Track idempotency
        if idempotency_key:
            self._processed_keys[idempotency_key] = order.order_id

        # Execute
        start_time = time.time()
        result = await self.execute_order(order, market_data or {})
        result.execution_time_ms = (time.time() - start_time) * 1000

        # Update stats
        if result.success:
            self.successful_orders += 1
            self.total_volume += result.total_filled * result.average_price
            self.total_fees += result.total_fees
        else:
            self.failed_orders += 1

        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "mode": self.mode.value,
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": self.successful_orders / self.total_orders if self.total_orders > 0 else 0,
            "total_volume": self.total_volume,
            "total_fees": self.total_fees,
            "open_orders": len([o for o in self.orders.values() if not o.is_complete]),
            "total_fills": len(self.fills)
        }


class BacktestExecutionEngine(ExecutionEngine):
    """
    Backtest execution engine.

    Simulates order execution using historical data with realistic
    fill models, fees, and slippage.
    """

    def __init__(
        self,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
        risk_guardian: Optional[Any] = None,
        fill_probability: float = 1.0,  # For limit orders
        partial_fill_enabled: bool = False
    ):
        super().__init__(
            mode=ExecutionMode.BACKTEST,
            fee_structure=fee_structure,
            slippage_model=slippage_model,
            risk_guardian=risk_guardian
        )
        self.fill_probability = fill_probability
        self.partial_fill_enabled = partial_fill_enabled
        self.current_timestamp: Optional[datetime] = None

    def set_timestamp(self, timestamp: datetime):
        """Set current backtest timestamp"""
        self.current_timestamp = timestamp

    async def execute_order(self, order: Order, market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute order in backtest mode"""
        order.submitted_at = self.current_timestamp or datetime.now()
        order.status = OrderStatus.SUBMITTED

        # Get market data
        current_price = market_data.get("close", market_data.get("price", 0))
        high_price = market_data.get("high", current_price)
        low_price = market_data.get("low", current_price)
        volume = market_data.get("volume", 1000000)
        volatility = market_data.get("volatility", 0.02)

        if current_price <= 0:
            order.status = OrderStatus.FAILED
            return ExecutionResult(
                success=False,
                order=order,
                error_message="Invalid market data: no price available"
            )

        fills = []

        # Handle different order types
        if order.order_type == OrderType.MARKET:
            # Market orders always fill (in backtest)
            fill_price = self._calculate_fill_price(
                order, current_price, volume, volatility
            )
            fill = self._create_fill(order, fill_price, order.quantity)
            fills.append(fill)

        elif order.order_type == OrderType.LIMIT:
            # Check if limit price would have been hit
            if order.side == OrderSide.BUY and order.price >= low_price:
                fill_price = min(order.price, current_price)
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            elif order.side == OrderSide.SELL and order.price <= high_price:
                fill_price = max(order.price, current_price)
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            else:
                # Order not filled this bar
                order.status = OrderStatus.SUBMITTED
                return ExecutionResult(success=True, order=order, fills=[])

        elif order.order_type == OrderType.STOP_LOSS:
            # Check if stop price was triggered
            triggered = False
            if order.side == OrderSide.SELL and low_price <= order.stop_price:
                triggered = True
            elif order.side == OrderSide.BUY and high_price >= order.stop_price:
                triggered = True

            if triggered:
                fill_price = self._calculate_fill_price(
                    order, order.stop_price, volume, volatility
                )
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            else:
                order.status = OrderStatus.SUBMITTED
                return ExecutionResult(success=True, order=order, fills=[])

        # Update order with fills
        if fills:
            order.filled_quantity = sum(f.quantity for f in fills)
            order.average_fill_price = sum(f.quantity * f.price for f in fills) / order.filled_quantity
            order.total_fees = sum(f.fee for f in fills)
            order.filled_at = self.current_timestamp or datetime.now()
            order.status = OrderStatus.FILLED

            # Store fills
            self.fills.extend(fills)

            # Update position
            self._update_position(order, fills)

            # Notify callbacks
            for fill in fills:
                self._notify_fill(fill)
            self._notify_order_update(order)

        return ExecutionResult(success=True, order=order, fills=fills)

    def _calculate_fill_price(
        self,
        order: Order,
        base_price: float,
        volume: float,
        volatility: float
    ) -> float:
        """Calculate fill price with slippage"""
        slippage_pct = self.slippage_model.calculate_slippage(
            order_size=order.quantity * base_price,
            avg_daily_volume=volume * base_price,
            current_volatility=volatility,
            is_buy=order.side == OrderSide.BUY
        )

        # Apply slippage (adverse direction)
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage_pct / 100)
        else:
            fill_price = base_price * (1 - slippage_pct / 100)

        return fill_price

    def _create_fill(self, order: Order, price: float, quantity: float) -> Fill:
        """Create a fill object"""
        fee = self.fee_structure.calculate_fee(
            notional_value=quantity * price,
            is_maker=order.order_type == OrderType.LIMIT
        )

        return Fill(
            fill_id=self.generate_fill_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp=self.current_timestamp or datetime.now(),
            is_maker=order.order_type == OrderType.LIMIT,
            expected_price=order.price or price
        )

    def _update_position(self, order: Order, fills: List[Fill]):
        """Update position based on fills"""
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]

        for fill in fills:
            if order.side == OrderSide.BUY:
                if pos.side == PositionSide.SHORT:
                    # Closing short position
                    close_qty = min(fill.quantity, abs(pos.quantity))
                    pnl = (pos.entry_price - fill.price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity += close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        # Opening new long
                        pos.side = PositionSide.LONG
                        pos.quantity = remaining
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    # Adding to long position
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    else:
                        # Average entry price
                        total_cost = pos.entry_price * pos.quantity + fill.price * fill.quantity
                        pos.entry_price = total_cost / (pos.quantity + fill.quantity)
                    pos.quantity += fill.quantity
                    pos.side = PositionSide.LONG
            else:  # SELL
                if pos.side == PositionSide.LONG:
                    # Closing long position
                    close_qty = min(fill.quantity, pos.quantity)
                    pnl = (fill.price - pos.entry_price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity -= close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        # Opening new short
                        pos.side = PositionSide.SHORT
                        pos.quantity = -remaining
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    # Adding to short position
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    else:
                        # Average entry price
                        total_cost = pos.entry_price * abs(pos.quantity) + fill.price * fill.quantity
                        pos.entry_price = total_cost / (abs(pos.quantity) + fill.quantity)
                    pos.quantity -= fill.quantity
                    pos.side = PositionSide.SHORT

            pos.total_fees_paid += fill.fee
            pos.leverage = order.leverage

        pos.last_updated = datetime.now()

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in backtest"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if not order.is_complete:
                order.status = OrderStatus.CANCELLED
                self._notify_order_update(order)
                return True
        return False

    def get_position(self, symbol: str) -> Position:
        """Get position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]


class PaperExecutionEngine(ExecutionEngine):
    """
    Paper trading execution engine.

    Uses live market data but simulates fills with the same model
    as backtest for consistency.
    """

    def __init__(
        self,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
        risk_guardian: Optional[Any] = None,
        latency_ms: float = 50.0  # Simulated latency
    ):
        super().__init__(
            mode=ExecutionMode.PAPER,
            fee_structure=fee_structure,
            slippage_model=slippage_model,
            risk_guardian=risk_guardian
        )
        self.latency_ms = latency_ms

        # Pending orders (for limit/stop orders)
        self.pending_orders: Dict[str, Order] = {}

    async def execute_order(self, order: Order, market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute order in paper mode"""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED

        current_price = market_data.get("price", market_data.get("close", 0))
        bid = market_data.get("bid", current_price * 0.9999)
        ask = market_data.get("ask", current_price * 1.0001)
        volume = market_data.get("volume", 1000000)
        volatility = market_data.get("volatility", 0.02)

        if current_price <= 0:
            order.status = OrderStatus.FAILED
            return ExecutionResult(
                success=False,
                order=order,
                error_message="No market data available"
            )

        fills = []

        if order.order_type == OrderType.MARKET:
            # Market order fills immediately
            base_price = ask if order.side == OrderSide.BUY else bid
            fill_price = self._calculate_fill_price(order, base_price, volume, volatility)
            fill = self._create_fill(order, fill_price, order.quantity)
            fills.append(fill)

        elif order.order_type == OrderType.LIMIT:
            # Check if limit is immediately fillable
            if order.side == OrderSide.BUY and order.price >= ask:
                fill_price = min(order.price, ask)
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            elif order.side == OrderSide.SELL and order.price <= bid:
                fill_price = max(order.price, bid)
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            else:
                # Add to pending orders
                self.pending_orders[order.order_id] = order
                return ExecutionResult(success=True, order=order, fills=[])

        elif order.order_type == OrderType.STOP_LOSS:
            # Check if stop is immediately triggered
            triggered = False
            if order.side == OrderSide.SELL and current_price <= order.stop_price:
                triggered = True
            elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                triggered = True

            if triggered:
                fill_price = self._calculate_fill_price(order, current_price, volume, volatility)
                fill = self._create_fill(order, fill_price, order.quantity)
                fills.append(fill)
            else:
                self.pending_orders[order.order_id] = order
                return ExecutionResult(success=True, order=order, fills=[])

        # Process fills
        if fills:
            order.filled_quantity = sum(f.quantity for f in fills)
            order.average_fill_price = sum(f.quantity * f.price for f in fills) / order.filled_quantity
            order.total_fees = sum(f.fee for f in fills)
            order.filled_at = datetime.now()
            order.status = OrderStatus.FILLED

            self.fills.extend(fills)
            self._update_position(order, fills)

            for fill in fills:
                self._notify_fill(fill)
            self._notify_order_update(order)

        return ExecutionResult(success=True, order=order, fills=fills)

    def _calculate_fill_price(self, order: Order, base_price: float, volume: float, volatility: float) -> float:
        """Calculate fill price with slippage"""
        slippage_pct = self.slippage_model.calculate_slippage(
            order_size=order.quantity * base_price,
            avg_daily_volume=volume * base_price,
            current_volatility=volatility,
            is_buy=order.side == OrderSide.BUY
        )

        if order.side == OrderSide.BUY:
            return base_price * (1 + slippage_pct / 100)
        else:
            return base_price * (1 - slippage_pct / 100)

    def _create_fill(self, order: Order, price: float, quantity: float) -> Fill:
        """Create a fill object"""
        fee = self.fee_structure.calculate_fee(
            notional_value=quantity * price,
            is_maker=order.order_type == OrderType.LIMIT
        )

        return Fill(
            fill_id=self.generate_fill_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=fee,
            timestamp=datetime.now(),
            is_maker=order.order_type == OrderType.LIMIT
        )

    def _update_position(self, order: Order, fills: List[Fill]):
        """Update position - same logic as backtest"""
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]

        for fill in fills:
            if order.side == OrderSide.BUY:
                if pos.side == PositionSide.SHORT:
                    close_qty = min(fill.quantity, abs(pos.quantity))
                    pnl = (pos.entry_price - fill.price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity += close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        pos.side = PositionSide.LONG
                        pos.quantity = remaining
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    else:
                        total_cost = pos.entry_price * pos.quantity + fill.price * fill.quantity
                        pos.entry_price = total_cost / (pos.quantity + fill.quantity)
                    pos.quantity += fill.quantity
                    pos.side = PositionSide.LONG
            else:
                if pos.side == PositionSide.LONG:
                    close_qty = min(fill.quantity, pos.quantity)
                    pnl = (fill.price - pos.entry_price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity -= close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        pos.side = PositionSide.SHORT
                        pos.quantity = -remaining
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                        pos.opened_at = fill.timestamp
                    else:
                        total_cost = pos.entry_price * abs(pos.quantity) + fill.price * fill.quantity
                        pos.entry_price = total_cost / (abs(pos.quantity) + fill.quantity)
                    pos.quantity -= fill.quantity
                    pos.side = PositionSide.SHORT

            pos.total_fees_paid += fill.fee
            pos.leverage = order.leverage

    async def check_pending_orders(self, market_data: Dict[str, Any]) -> List[ExecutionResult]:
        """Check and fill pending orders against current market data"""
        results = []
        filled_orders = []

        current_price = market_data.get("price", 0)
        bid = market_data.get("bid", current_price * 0.9999)
        ask = market_data.get("ask", current_price * 1.0001)

        for order_id, order in self.pending_orders.items():
            fills = []

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.price >= ask:
                    fill_price = min(order.price, ask)
                    fill = self._create_fill(order, fill_price, order.quantity)
                    fills.append(fill)
                elif order.side == OrderSide.SELL and order.price <= bid:
                    fill_price = max(order.price, bid)
                    fill = self._create_fill(order, fill_price, order.quantity)
                    fills.append(fill)

            elif order.order_type == OrderType.STOP_LOSS:
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    fill = self._create_fill(order, current_price, order.quantity)
                    fills.append(fill)
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    fill = self._create_fill(order, current_price, order.quantity)
                    fills.append(fill)

            if fills:
                order.filled_quantity = sum(f.quantity for f in fills)
                order.average_fill_price = sum(f.quantity * f.price for f in fills) / order.filled_quantity
                order.total_fees = sum(f.fee for f in fills)
                order.filled_at = datetime.now()
                order.status = OrderStatus.FILLED

                self.fills.extend(fills)
                self._update_position(order, fills)

                for fill in fills:
                    self._notify_fill(fill)
                self._notify_order_update(order)

                filled_orders.append(order_id)
                results.append(ExecutionResult(success=True, order=order, fills=fills))

        # Remove filled orders from pending
        for order_id in filled_orders:
            del self.pending_orders[order_id]

        return results

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            self._notify_order_update(order)
            return True
        elif order_id in self.orders:
            order = self.orders[order_id]
            if not order.is_complete:
                order.status = OrderStatus.CANCELLED
                self._notify_order_update(order)
                return True
        return False

    def get_position(self, symbol: str) -> Position:
        """Get position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]


class LiveExecutionEngine(ExecutionEngine):
    """
    Live execution engine.

    Executes real orders on exchanges with retry logic,
    partial fill handling, and comprehensive error handling.
    """

    def __init__(
        self,
        exchange_client: Any,  # Exchange client (ccxt or custom)
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
        risk_guardian: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay_ms: float = 1000.0
    ):
        super().__init__(
            mode=ExecutionMode.LIVE,
            fee_structure=fee_structure,
            slippage_model=slippage_model,
            risk_guardian=risk_guardian
        )
        self.exchange_client = exchange_client
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

    async def execute_order(self, order: Order, market_data: Dict[str, Any]) -> ExecutionResult:
        """Execute order on live exchange"""
        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Prepare order parameters
                params = self._prepare_order_params(order)

                # Submit to exchange
                if order.order_type == OrderType.MARKET:
                    result = await self._execute_market_order(order, params)
                elif order.order_type == OrderType.LIMIT:
                    result = await self._execute_limit_order(order, params)
                elif order.order_type == OrderType.STOP_LOSS:
                    result = await self._execute_stop_order(order, params)
                else:
                    raise ValueError(f"Unsupported order type: {order.order_type}")

                # Process result
                if result.get("status") == "closed" or result.get("filled", 0) > 0:
                    fills = self._parse_exchange_fills(order, result)

                    order.exchange_order_id = result.get("id")
                    order.filled_quantity = result.get("filled", 0)
                    order.average_fill_price = result.get("average", result.get("price", 0))
                    order.total_fees = result.get("fee", {}).get("cost", 0)
                    order.status = OrderStatus.FILLED if order.filled_quantity >= order.quantity else OrderStatus.PARTIALLY_FILLED
                    order.filled_at = datetime.now()

                    self.fills.extend(fills)
                    self._update_position(order, fills)

                    for fill in fills:
                        self._notify_fill(fill)
                    self._notify_order_update(order)

                    return ExecutionResult(
                        success=True,
                        order=order,
                        fills=fills,
                        retries=retries
                    )
                else:
                    # Order submitted but not filled yet
                    order.exchange_order_id = result.get("id")
                    return ExecutionResult(success=True, order=order, fills=[], retries=retries)

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order execution attempt {retries + 1} failed: {e}")
                retries += 1

                if retries <= self.max_retries:
                    await asyncio.sleep(self.retry_delay_ms / 1000)

        # All retries exhausted
        order.status = OrderStatus.FAILED
        return ExecutionResult(
            success=False,
            order=order,
            error_message=f"Execution failed after {retries} retries: {last_error}",
            retries=retries
        )

    def _prepare_order_params(self, order: Order) -> Dict[str, Any]:
        """Prepare order parameters for exchange"""
        params = {
            "symbol": order.symbol,
            "type": order.order_type.value.lower(),
            "side": order.side.value.lower(),
            "amount": order.quantity,
        }

        if order.price:
            params["price"] = order.price

        if order.stop_price:
            params["stopPrice"] = order.stop_price

        if order.leverage > 1:
            params["leverage"] = order.leverage

        if order.reduce_only:
            params["reduceOnly"] = True

        return params

    async def _execute_market_order(self, order: Order, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market order on exchange"""
        if hasattr(self.exchange_client, "create_market_order"):
            return await self.exchange_client.create_market_order(
                params["symbol"],
                params["side"],
                params["amount"]
            )
        else:
            return await self.exchange_client.create_order(**params)

    async def _execute_limit_order(self, order: Order, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute limit order on exchange"""
        if hasattr(self.exchange_client, "create_limit_order"):
            return await self.exchange_client.create_limit_order(
                params["symbol"],
                params["side"],
                params["amount"],
                params["price"]
            )
        else:
            return await self.exchange_client.create_order(**params)

    async def _execute_stop_order(self, order: Order, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stop order on exchange"""
        params["type"] = "stop_market"
        return await self.exchange_client.create_order(**params)

    def _parse_exchange_fills(self, order: Order, result: Dict[str, Any]) -> List[Fill]:
        """Parse exchange result into fills"""
        fills = []

        trades = result.get("trades", [])
        if trades:
            for trade in trades:
                fill = Fill(
                    fill_id=self.generate_fill_id(),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=trade.get("amount", 0),
                    price=trade.get("price", 0),
                    fee=trade.get("fee", {}).get("cost", 0),
                    timestamp=datetime.fromisoformat(trade.get("datetime", datetime.now().isoformat())),
                    is_maker=trade.get("takerOrMaker") == "maker"
                )
                fills.append(fill)
        else:
            # Single fill from result
            fill = Fill(
                fill_id=self.generate_fill_id(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=result.get("filled", order.quantity),
                price=result.get("average", result.get("price", 0)),
                fee=result.get("fee", {}).get("cost", 0),
                timestamp=datetime.now()
            )
            fills.append(fill)

        return fills

    def _update_position(self, order: Order, fills: List[Fill]):
        """Update position from fills"""
        # Implementation same as paper mode
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]

        for fill in fills:
            if order.side == OrderSide.BUY:
                if pos.side == PositionSide.SHORT:
                    close_qty = min(fill.quantity, abs(pos.quantity))
                    pnl = (pos.entry_price - fill.price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity += close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        pos.side = PositionSide.LONG
                        pos.quantity = remaining
                        pos.entry_price = fill.price
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                    else:
                        total_cost = pos.entry_price * pos.quantity + fill.price * fill.quantity
                        pos.entry_price = total_cost / (pos.quantity + fill.quantity)
                    pos.quantity += fill.quantity
                    pos.side = PositionSide.LONG
            else:
                if pos.side == PositionSide.LONG:
                    close_qty = min(fill.quantity, pos.quantity)
                    pnl = (fill.price - pos.entry_price) * close_qty
                    pos.realized_pnl += pnl - fill.fee
                    pos.quantity -= close_qty

                    remaining = fill.quantity - close_qty
                    if remaining > 0:
                        pos.side = PositionSide.SHORT
                        pos.quantity = -remaining
                        pos.entry_price = fill.price
                    elif pos.quantity == 0:
                        pos.side = PositionSide.FLAT
                else:
                    if pos.quantity == 0:
                        pos.entry_price = fill.price
                    else:
                        total_cost = pos.entry_price * abs(pos.quantity) + fill.price * fill.quantity
                        pos.entry_price = total_cost / (abs(pos.quantity) + fill.quantity)
                    pos.quantity -= fill.quantity
                    pos.side = PositionSide.SHORT

            pos.total_fees_paid += fill.fee
            pos.leverage = order.leverage

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on exchange"""
        order = self.orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            if order.exchange_order_id:
                await self.exchange_client.cancel_order(order.exchange_order_id, order.symbol)
            order.status = OrderStatus.CANCELLED
            self._notify_order_update(order)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_position(self, symbol: str) -> Position:
        """Get position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    async def sync_position(self, symbol: str) -> Position:
        """Sync position with exchange"""
        try:
            positions = await self.exchange_client.fetch_positions([symbol])
            if positions:
                exchange_pos = positions[0]
                pos = self.get_position(symbol)
                pos.quantity = exchange_pos.get("contracts", 0)
                pos.entry_price = exchange_pos.get("entryPrice", 0)
                pos.unrealized_pnl = exchange_pos.get("unrealizedPnl", 0)
                pos.liquidation_price = exchange_pos.get("liquidationPrice", 0)
                pos.margin_used = exchange_pos.get("initialMargin", 0)
                pos.leverage = exchange_pos.get("leverage", 1)

                if pos.quantity > 0:
                    pos.side = PositionSide.LONG
                elif pos.quantity < 0:
                    pos.side = PositionSide.SHORT
                else:
                    pos.side = PositionSide.FLAT

                return pos
        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

        return self.get_position(symbol)


def create_execution_engine(
    mode: ExecutionMode,
    exchange_client: Optional[Any] = None,
    fee_structure: Optional[FeeStructure] = None,
    slippage_model: Optional[SlippageModel] = None,
    risk_guardian: Optional[Any] = None
) -> ExecutionEngine:
    """
    Factory function to create execution engine.

    Ensures consistent configuration across all modes.
    """
    # Use same fee structure for all modes
    fees = fee_structure or FeeStructure(
        maker_fee_pct=0.02,
        taker_fee_pct=0.04,
        funding_rate_pct=0.01
    )

    # Use same slippage model for all modes
    slippage = slippage_model or SlippageModel(
        base_slippage_pct=0.02,
        volume_impact_factor=0.1,
        volatility_factor=0.5,
        spread_pct=0.01
    )

    if mode == ExecutionMode.BACKTEST:
        return BacktestExecutionEngine(
            fee_structure=fees,
            slippage_model=slippage,
            risk_guardian=risk_guardian
        )
    elif mode == ExecutionMode.PAPER:
        return PaperExecutionEngine(
            fee_structure=fees,
            slippage_model=slippage,
            risk_guardian=risk_guardian
        )
    elif mode == ExecutionMode.LIVE:
        if exchange_client is None:
            raise ValueError("Live mode requires exchange_client")
        return LiveExecutionEngine(
            exchange_client=exchange_client,
            fee_structure=fees,
            slippage_model=slippage,
            risk_guardian=risk_guardian
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


# Global execution engine instance
_execution_engine: Optional[ExecutionEngine] = None


def get_execution_engine() -> Optional[ExecutionEngine]:
    """Get global execution engine instance"""
    return _execution_engine


def set_execution_engine(engine: ExecutionEngine):
    """Set global execution engine instance"""
    global _execution_engine
    _execution_engine = engine


__all__ = [
    # Enums
    "ExecutionMode",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSide",
    # Models
    "FeeStructure",
    "SlippageModel",
    "Order",
    "Fill",
    "Position",
    "ExecutionResult",
    # Engines
    "ExecutionEngine",
    "BacktestExecutionEngine",
    "PaperExecutionEngine",
    "LiveExecutionEngine",
    # Factory
    "create_execution_engine",
    "get_execution_engine",
    "set_execution_engine",
]
