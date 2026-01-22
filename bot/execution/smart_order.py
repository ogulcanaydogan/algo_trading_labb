"""
Smart Order Execution - TWAP/VWAP algorithms.

Reduces market impact by splitting large orders into smaller pieces
executed over time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Literal
import random

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Order execution algorithm types."""
    MARKET = "market"  # Single market order
    TWAP = "twap"      # Time-Weighted Average Price
    VWAP = "vwap"      # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Hidden quantity orders
    POV = "pov"        # Percentage of Volume


@dataclass
class SliceOrder:
    """A single slice of a parent order."""
    slice_id: int
    quantity: float
    target_time: datetime
    executed: bool = False
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    executed_time: Optional[datetime] = None


@dataclass
class SmartOrderResult:
    """Result of smart order execution."""
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    algorithm: ExecutionAlgorithm
    total_quantity: float
    executed_quantity: float
    avg_price: float
    vwap: float
    num_slices: int
    slices_executed: int
    start_time: datetime
    end_time: datetime
    market_vwap: Optional[float] = None
    slippage_vs_vwap_bps: Optional[float] = None

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.total_quantity == 0:
            return 0.0
        return self.executed_quantity / self.total_quantity

    @property
    def duration_seconds(self) -> float:
        """Total execution duration."""
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "algorithm": self.algorithm.value,
            "total_quantity": self.total_quantity,
            "executed_quantity": self.executed_quantity,
            "avg_price": round(self.avg_price, 8),
            "vwap": round(self.vwap, 8),
            "num_slices": self.num_slices,
            "slices_executed": self.slices_executed,
            "fill_rate": round(self.fill_rate, 4),
            "duration_seconds": round(self.duration_seconds, 1),
            "market_vwap": round(self.market_vwap, 8) if self.market_vwap else None,
            "slippage_vs_vwap_bps": round(self.slippage_vs_vwap_bps, 2) if self.slippage_vs_vwap_bps else None,
        }


@dataclass
class TWAPConfig:
    """Configuration for TWAP execution."""
    duration_minutes: int = 30
    num_slices: int = 10
    randomize_timing: bool = True  # Add randomness to avoid detection
    randomize_size: bool = True    # Vary slice sizes
    min_slice_pct: float = 0.05    # Minimum slice as % of total
    max_slice_pct: float = 0.20    # Maximum slice as % of total


@dataclass
class VWAPConfig:
    """Configuration for VWAP execution."""
    duration_minutes: int = 60
    volume_profile: Optional[List[float]] = None  # Historical volume distribution
    participation_rate: float = 0.1  # Max % of market volume to capture
    min_slices: int = 5
    max_slices: int = 20


class SmartOrderExecutor:
    """
    Execute large orders using smart algorithms to minimize market impact.

    Algorithms:
    - TWAP: Split order evenly over time
    - VWAP: Match historical volume pattern
    - Iceberg: Hide true order size
    - POV: Maintain percentage of market volume
    """

    def __init__(
        self,
        execute_fn: Callable,  # Function to execute single orders
        get_price_fn: Callable,  # Function to get current price
        get_volume_fn: Optional[Callable] = None,  # Function to get recent volume
    ):
        """
        Initialize smart order executor.

        Args:
            execute_fn: Async function(symbol, side, quantity, price) -> executed_price
            get_price_fn: Function(symbol) -> current_price
            get_volume_fn: Function(symbol, minutes) -> volume (optional, for VWAP)
        """
        self.execute_fn = execute_fn
        self.get_price_fn = get_price_fn
        self.get_volume_fn = get_volume_fn
        self._active_orders: Dict[str, List[SliceOrder]] = {}

    def generate_twap_schedule(
        self,
        total_quantity: float,
        config: TWAPConfig,
    ) -> List[SliceOrder]:
        """
        Generate TWAP execution schedule.

        Splits order into time-weighted slices with optional randomization.
        """
        slices = []
        now = datetime.now()
        interval = timedelta(minutes=config.duration_minutes / config.num_slices)

        # Generate slice sizes
        if config.randomize_size:
            # Random sizes within bounds
            sizes = []
            remaining = total_quantity
            for i in range(config.num_slices - 1):
                min_size = total_quantity * config.min_slice_pct
                max_size = min(total_quantity * config.max_slice_pct, remaining * 0.5)
                size = random.uniform(min_size, max_size)
                sizes.append(size)
                remaining -= size
            sizes.append(remaining)  # Last slice gets remainder
            random.shuffle(sizes)
        else:
            # Equal sizes
            sizes = [total_quantity / config.num_slices] * config.num_slices

        # Generate schedule
        for i, size in enumerate(sizes):
            target_time = now + interval * i
            if config.randomize_timing:
                # Add random offset (-30% to +30% of interval)
                offset = random.uniform(-0.3, 0.3) * interval.total_seconds()
                target_time += timedelta(seconds=offset)

            slices.append(SliceOrder(
                slice_id=i,
                quantity=size,
                target_time=target_time,
            ))

        # Sort by time
        slices.sort(key=lambda x: x.target_time)
        return slices

    def generate_vwap_schedule(
        self,
        total_quantity: float,
        config: VWAPConfig,
        historical_volume_profile: Optional[List[float]] = None,
    ) -> List[SliceOrder]:
        """
        Generate VWAP execution schedule.

        Matches historical volume pattern to minimize market impact.
        """
        # Use provided profile or default uniform
        profile = historical_volume_profile or config.volume_profile
        if not profile:
            # Default: assume higher volume at market open/close
            # This is a simplified U-shaped curve
            profile = [0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.07, 0.08, 0.10, 0.15, 0.08]

        # Normalize profile
        profile = np.array(profile)
        profile = profile / profile.sum()

        # Determine number of slices
        num_slices = min(max(config.min_slices, len(profile)), config.max_slices)

        # Resample profile if needed
        if len(profile) != num_slices:
            indices = np.linspace(0, len(profile) - 1, num_slices)
            profile = np.interp(indices, range(len(profile)), profile)
            profile = profile / profile.sum()

        # Generate slices
        slices = []
        now = datetime.now()
        interval = timedelta(minutes=config.duration_minutes / num_slices)

        for i, pct in enumerate(profile):
            slices.append(SliceOrder(
                slice_id=i,
                quantity=total_quantity * pct,
                target_time=now + interval * i,
            ))

        return slices

    async def execute_twap(
        self,
        order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        config: Optional[TWAPConfig] = None,
    ) -> SmartOrderResult:
        """
        Execute order using TWAP algorithm.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            total_quantity: Total quantity to execute
            config: TWAP configuration

        Returns:
            SmartOrderResult with execution details
        """
        config = config or TWAPConfig()
        slices = self.generate_twap_schedule(total_quantity, config)
        return await self._execute_slices(
            order_id, symbol, side, total_quantity,
            slices, ExecutionAlgorithm.TWAP
        )

    async def execute_vwap(
        self,
        order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        config: Optional[VWAPConfig] = None,
        volume_profile: Optional[List[float]] = None,
    ) -> SmartOrderResult:
        """
        Execute order using VWAP algorithm.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            total_quantity: Total quantity to execute
            config: VWAP configuration
            volume_profile: Historical volume distribution

        Returns:
            SmartOrderResult with execution details
        """
        config = config or VWAPConfig()
        slices = self.generate_vwap_schedule(total_quantity, config, volume_profile)
        return await self._execute_slices(
            order_id, symbol, side, total_quantity,
            slices, ExecutionAlgorithm.VWAP
        )

    async def _execute_slices(
        self,
        order_id: str,
        symbol: str,
        side: Literal["buy", "sell"],
        total_quantity: float,
        slices: List[SliceOrder],
        algorithm: ExecutionAlgorithm,
    ) -> SmartOrderResult:
        """Execute order slices according to schedule."""
        self._active_orders[order_id] = slices
        start_time = datetime.now()
        executed_trades = []

        logger.info(
            f"Starting {algorithm.value} execution: {order_id} "
            f"{side} {total_quantity} {symbol} in {len(slices)} slices"
        )

        try:
            for slice_order in slices:
                # Wait until target time
                now = datetime.now()
                if slice_order.target_time > now:
                    wait_seconds = (slice_order.target_time - now).total_seconds()
                    await asyncio.sleep(wait_seconds)

                # Execute slice
                try:
                    current_price = self.get_price_fn(symbol)
                    executed_price = await self.execute_fn(
                        symbol, side, slice_order.quantity, current_price
                    )

                    slice_order.executed = True
                    slice_order.executed_price = executed_price
                    slice_order.executed_quantity = slice_order.quantity
                    slice_order.executed_time = datetime.now()

                    executed_trades.append({
                        "quantity": slice_order.quantity,
                        "price": executed_price,
                    })

                    logger.debug(
                        f"Slice {slice_order.slice_id} executed: "
                        f"{slice_order.quantity} @ {executed_price}"
                    )

                except Exception as e:
                    logger.error(f"Slice {slice_order.slice_id} failed: {e}")
                    # Continue with remaining slices

        finally:
            del self._active_orders[order_id]

        # Calculate results
        end_time = datetime.now()
        executed_quantity = sum(t["quantity"] for t in executed_trades)
        total_cost = sum(t["quantity"] * t["price"] for t in executed_trades)
        avg_price = total_cost / executed_quantity if executed_quantity > 0 else 0
        vwap = avg_price  # For our execution, VWAP = avg price

        # Calculate slippage vs market VWAP if available
        market_vwap = None
        slippage_bps = None
        if self.get_volume_fn and executed_quantity > 0:
            try:
                # This would require actual market VWAP calculation
                # For now, use start price as proxy
                start_price = self.get_price_fn(symbol)
                market_vwap = start_price
                slippage_bps = (avg_price - market_vwap) / market_vwap * 10000
                if side == "sell":
                    slippage_bps = -slippage_bps
            except Exception:
                pass

        return SmartOrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            algorithm=algorithm,
            total_quantity=total_quantity,
            executed_quantity=executed_quantity,
            avg_price=avg_price,
            vwap=vwap,
            num_slices=len(slices),
            slices_executed=len(executed_trades),
            start_time=start_time,
            end_time=end_time,
            market_vwap=market_vwap,
            slippage_vs_vwap_bps=slippage_bps,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active smart order."""
        if order_id in self._active_orders:
            # Mark remaining slices as cancelled
            for slice_order in self._active_orders[order_id]:
                if not slice_order.executed:
                    slice_order.executed = True  # Mark to skip
            logger.info(f"Smart order {order_id} cancelled")
            return True
        return False

    def get_active_orders(self) -> List[str]:
        """Get list of active smart order IDs."""
        return list(self._active_orders.keys())


def create_smart_executor(
    execute_fn: Callable,
    get_price_fn: Callable,
    get_volume_fn: Optional[Callable] = None,
) -> SmartOrderExecutor:
    """Factory function to create smart order executor."""
    return SmartOrderExecutor(
        execute_fn=execute_fn,
        get_price_fn=get_price_fn,
        get_volume_fn=get_volume_fn,
    )
