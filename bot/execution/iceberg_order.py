"""
Iceberg Orders - Hide large order size with slice execution.

Executes large orders in smaller visible chunks to minimize
market impact and avoid detection.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class IcebergState(Enum):
    """Iceberg order states."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class IcebergSlice:
    """Single slice of an iceberg order."""
    slice_id: str
    quantity: float
    price: float
    status: str  # "pending", "filled", "cancelled"
    fill_price: Optional[float] = None
    fill_quantity: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    order_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_quantity": self.fill_quantity,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "order_id": self.order_id,
        }


@dataclass
class IcebergOrder:
    """Iceberg order with hidden size."""
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    total_quantity: float
    visible_quantity: float  # Size shown per slice
    limit_price: Optional[float]
    state: IcebergState
    slices: List[IcebergSlice] = field(default_factory=list)
    filled_quantity: float = 0
    avg_fill_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    variance_pct: float = 0.1  # Random variance in slice size
    min_interval_ms: int = 500  # Minimum time between slices
    max_interval_ms: int = 2000  # Maximum time between slices

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": self.total_quantity,
            "visible_quantity": self.visible_quantity,
            "limit_price": self.limit_price,
            "state": self.state.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": round(self.avg_fill_price, 6),
            "fill_pct": round(self.filled_quantity / self.total_quantity * 100, 2),
            "slices_completed": sum(1 for s in self.slices if s.status == "filled"),
            "slices_total": len(self.slices),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def remaining_quantity(self) -> float:
        return self.total_quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.filled_quantity >= self.total_quantity * 0.99  # 99% tolerance


@dataclass
class IcebergConfig:
    """Iceberg order configuration."""
    # Slice sizing
    default_visible_pct: float = 0.05  # 5% visible by default
    min_slice_value: float = 10.0  # Minimum slice value in quote currency
    max_slices: int = 100

    # Timing
    default_min_interval_ms: int = 500
    default_max_interval_ms: int = 3000
    randomize_timing: bool = True

    # Price adjustment
    price_tolerance_pct: float = 0.001  # 0.1% price tolerance
    chase_price: bool = True  # Adjust price to chase market

    # Safety
    max_retries: int = 3
    timeout_minutes: int = 60


class IcebergExecutor:
    """
    Execute iceberg orders with hidden size.

    Features:
    - Slice large orders into smaller visible chunks
    - Randomize slice sizes and timing
    - Chase market price
    - Track execution quality
    """

    def __init__(
        self,
        config: Optional[IcebergConfig] = None,
        order_callback: Optional[Callable] = None,
    ):
        """
        Initialize iceberg executor.

        Args:
            config: Execution configuration
            order_callback: Async function to submit orders
                Should accept (symbol, side, quantity, price) and return order_id
        """
        self.config = config or IcebergConfig()
        self._order_callback = order_callback
        self._active_orders: Dict[str, IcebergOrder] = {}
        self._order_history: List[IcebergOrder] = []

    def set_order_callback(self, callback: Callable):
        """Set the order submission callback."""
        self._order_callback = callback

    def create_iceberg(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        visible_quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        variance_pct: float = 0.1,
        min_interval_ms: Optional[int] = None,
        max_interval_ms: Optional[int] = None,
    ) -> IcebergOrder:
        """
        Create an iceberg order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            total_quantity: Total order quantity
            visible_quantity: Visible quantity per slice (default: 5% of total)
            limit_price: Limit price (None for market)
            variance_pct: Random variance in slice size (0-1)
            min_interval_ms: Minimum milliseconds between slices
            max_interval_ms: Maximum milliseconds between slices

        Returns:
            Created IcebergOrder
        """
        order_id = f"iceberg_{uuid.uuid4().hex[:12]}"

        if visible_quantity is None:
            visible_quantity = total_quantity * self.config.default_visible_pct

        # Ensure minimum slice size
        if limit_price and visible_quantity * limit_price < self.config.min_slice_value:
            visible_quantity = self.config.min_slice_value / limit_price

        order = IcebergOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            visible_quantity=visible_quantity,
            limit_price=limit_price,
            state=IcebergState.PENDING,
            variance_pct=variance_pct,
            min_interval_ms=min_interval_ms or self.config.default_min_interval_ms,
            max_interval_ms=max_interval_ms or self.config.default_max_interval_ms,
        )

        # Generate slices
        self._generate_slices(order)

        self._active_orders[order_id] = order
        logger.info(
            f"Created iceberg order {order_id}: {side} {total_quantity} {symbol} "
            f"in {len(order.slices)} slices"
        )

        return order

    def _generate_slices(self, order: IcebergOrder):
        """Generate slice schedule for iceberg order."""
        remaining = order.total_quantity
        slice_num = 0

        while remaining > 0 and slice_num < self.config.max_slices:
            # Calculate slice size with variance
            base_size = min(order.visible_quantity, remaining)

            if order.variance_pct > 0 and base_size < remaining:
                variance = random.uniform(1 - order.variance_pct, 1 + order.variance_pct)
                slice_size = base_size * variance
            else:
                slice_size = base_size

            # Ensure we don't exceed remaining
            slice_size = min(slice_size, remaining)

            slice_id = f"{order.order_id}_s{slice_num}"

            order.slices.append(IcebergSlice(
                slice_id=slice_id,
                quantity=slice_size,
                price=order.limit_price or 0,
                status="pending",
            ))

            remaining -= slice_size
            slice_num += 1

    async def execute(
        self,
        order: IcebergOrder,
        get_current_price: Optional[Callable] = None,
    ) -> IcebergOrder:
        """
        Execute iceberg order.

        Args:
            order: IcebergOrder to execute
            get_current_price: Optional async function to get current market price

        Returns:
            Completed IcebergOrder
        """
        if not self._order_callback:
            raise ValueError("Order callback not set")

        order.state = IcebergState.ACTIVE
        logger.info(f"Starting iceberg execution: {order.order_id}")

        try:
            for i, slice_order in enumerate(order.slices):
                if order.state != IcebergState.ACTIVE:
                    break

                # Get current price for price adjustment
                current_price = order.limit_price
                if get_current_price and self.config.chase_price:
                    try:
                        current_price = await get_current_price(order.symbol)
                        if order.side == "buy":
                            slice_order.price = min(order.limit_price or current_price, current_price)
                        else:
                            slice_order.price = max(order.limit_price or current_price, current_price)
                    except Exception:
                        pass

                # Submit slice
                success = await self._execute_slice(order, slice_order)

                if success:
                    logger.debug(
                        f"Slice {i+1}/{len(order.slices)} filled: "
                        f"{slice_order.fill_quantity} @ {slice_order.fill_price}"
                    )

                # Random delay between slices
                if i < len(order.slices) - 1 and order.state == IcebergState.ACTIVE:
                    delay_ms = random.randint(order.min_interval_ms, order.max_interval_ms)
                    await asyncio.sleep(delay_ms / 1000)

            # Mark completion
            if order.is_complete:
                order.state = IcebergState.COMPLETED
            else:
                order.state = IcebergState.FAILED

            order.completed_at = datetime.now()

        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
            order.state = IcebergState.FAILED

        self._order_history.append(order)
        self._active_orders.pop(order.order_id, None)

        return order

    async def _execute_slice(
        self,
        order: IcebergOrder,
        slice_order: IcebergSlice,
        retry: int = 0,
    ) -> bool:
        """Execute a single slice."""
        try:
            slice_order.submitted_at = datetime.now()

            # Submit order through callback
            result = await self._order_callback(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_order.quantity,
                price=slice_order.price,
            )

            if result:
                # Assume filled (in production, wait for fill confirmation)
                slice_order.status = "filled"
                slice_order.filled_at = datetime.now()
                slice_order.fill_quantity = slice_order.quantity
                slice_order.fill_price = slice_order.price
                slice_order.order_id = result.get("order_id") if isinstance(result, dict) else str(result)

                # Update order totals
                order.filled_quantity += slice_order.fill_quantity

                # Update average price
                if order.filled_quantity > 0:
                    total_value = sum(
                        s.fill_quantity * s.fill_price
                        for s in order.slices
                        if s.status == "filled" and s.fill_quantity and s.fill_price
                    )
                    order.avg_fill_price = total_value / order.filled_quantity

                return True

            else:
                slice_order.status = "failed"
                if retry < self.config.max_retries:
                    await asyncio.sleep(1)
                    return await self._execute_slice(order, slice_order, retry + 1)
                return False

        except Exception as e:
            logger.error(f"Slice execution error: {e}")
            slice_order.status = "failed"
            return False

    def cancel(self, order_id: str) -> bool:
        """Cancel an active iceberg order."""
        order = self._active_orders.get(order_id)
        if not order:
            return False

        order.state = IcebergState.CANCELLED
        order.completed_at = datetime.now()

        # Cancel pending slices
        for slice_order in order.slices:
            if slice_order.status == "pending":
                slice_order.status = "cancelled"

        self._order_history.append(order)
        self._active_orders.pop(order_id, None)

        logger.info(f"Cancelled iceberg order: {order_id}")
        return True

    def pause(self, order_id: str) -> bool:
        """Pause an active iceberg order."""
        order = self._active_orders.get(order_id)
        if order and order.state == IcebergState.ACTIVE:
            order.state = IcebergState.PAUSED
            return True
        return False

    def resume(self, order_id: str) -> bool:
        """Resume a paused iceberg order."""
        order = self._active_orders.get(order_id)
        if order and order.state == IcebergState.PAUSED:
            order.state = IcebergState.ACTIVE
            return True
        return False

    def get_order(self, order_id: str) -> Optional[IcebergOrder]:
        """Get order by ID."""
        return self._active_orders.get(order_id)

    def get_active_orders(self) -> List[IcebergOrder]:
        """Get all active orders."""
        return list(self._active_orders.values())

    def get_execution_stats(self, order: IcebergOrder) -> Dict[str, Any]:
        """Get execution statistics for an order."""
        filled_slices = [s for s in order.slices if s.status == "filled"]

        if not filled_slices:
            return {"status": "no_fills"}

        fill_prices = [s.fill_price for s in filled_slices if s.fill_price]
        fill_times = [
            (s.filled_at - s.submitted_at).total_seconds()
            for s in filled_slices
            if s.filled_at and s.submitted_at
        ]

        # Calculate execution duration
        duration = None
        if order.completed_at:
            duration = (order.completed_at - order.created_at).total_seconds()

        # Price improvement vs limit
        price_improvement = 0
        if order.limit_price and order.avg_fill_price:
            if order.side == "buy":
                price_improvement = (order.limit_price - order.avg_fill_price) / order.limit_price
            else:
                price_improvement = (order.avg_fill_price - order.limit_price) / order.limit_price

        return {
            "order_id": order.order_id,
            "state": order.state.value,
            "fill_pct": order.filled_quantity / order.total_quantity,
            "avg_fill_price": order.avg_fill_price,
            "price_improvement_pct": price_improvement,
            "slices_filled": len(filled_slices),
            "slices_total": len(order.slices),
            "avg_slice_time_seconds": sum(fill_times) / len(fill_times) if fill_times else 0,
            "total_duration_seconds": duration,
            "price_range": {
                "min": min(fill_prices) if fill_prices else 0,
                "max": max(fill_prices) if fill_prices else 0,
            },
        }


def create_iceberg_executor(
    config: Optional[IcebergConfig] = None,
    order_callback: Optional[Callable] = None,
) -> IcebergExecutor:
    """Factory function to create iceberg executor."""
    return IcebergExecutor(config=config, order_callback=order_callback)
