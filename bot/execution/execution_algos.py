"""
Execution Algorithms - VWAP, TWAP, POV, Implementation Shortfall.

Advanced execution algorithms for minimizing market impact
and achieving best execution across trading venues.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import random

logger = logging.getLogger(__name__)


class AlgoStatus(Enum):
    """Execution algorithm status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class UrgencyLevel(Enum):
    """Order urgency level."""
    LOW = "low"  # Minimize impact, longer duration
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # Faster execution, accept more impact
    CRITICAL = "critical"  # Immediate execution


@dataclass
class AlgoOrder:
    """Order for execution algorithm."""
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    total_quantity: float
    limit_price: Optional[float] = None
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_fill_rate: float = 0.0  # Minimum fill rate required
    max_participation: float = 0.25  # Max % of volume to take

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now()


@dataclass
class AlgoSlice:
    """A slice/child order from execution algorithm."""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    target_price: Optional[float]
    actual_price: Optional[float] = None
    filled_quantity: float = 0.0
    status: str = "pending"
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    venue_id: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.filled_quantity >= self.quantity * 0.99

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "parent_order_id": self.parent_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "target_price": self.target_price,
            "actual_price": self.actual_price,
            "filled_quantity": self.filled_quantity,
            "fill_rate": round(self.fill_rate, 4),
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "venue_id": self.venue_id,
        }


@dataclass
class AlgoExecution:
    """Execution result from algorithm."""
    order_id: str
    algorithm: str
    symbol: str
    side: str
    total_quantity: float
    filled_quantity: float
    average_price: float
    vwap_benchmark: float
    arrival_price: float
    slippage_bps: float
    implementation_shortfall_bps: float
    participation_rate: float
    num_slices: int
    duration_seconds: float
    status: AlgoStatus
    slices: List[AlgoSlice] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "algorithm": self.algorithm,
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "fill_rate": round(self.fill_rate, 4),
            "average_price": round(self.average_price, 6),
            "vwap_benchmark": round(self.vwap_benchmark, 6),
            "arrival_price": round(self.arrival_price, 6),
            "slippage_bps": round(self.slippage_bps, 2),
            "implementation_shortfall_bps": round(self.implementation_shortfall_bps, 2),
            "participation_rate": round(self.participation_rate, 4),
            "num_slices": self.num_slices,
            "duration_seconds": round(self.duration_seconds, 2),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        interval_seconds: float = 1.0
    ):
        """
        Args:
            order_executor: Async function to execute orders
            market_data_provider: Async function to get market data
            interval_seconds: Interval between slices
        """
        self.order_executor = order_executor
        self.market_data_provider = market_data_provider
        self.interval_seconds = interval_seconds
        self._status = AlgoStatus.PENDING
        self._slices: List[AlgoSlice] = []
        self._cancel_requested = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        pass

    @abstractmethod
    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """
        Generate execution schedule.

        Returns list of (time, quantity) tuples.
        """
        pass

    async def execute(self, order: AlgoOrder) -> AlgoExecution:
        """
        Execute order using this algorithm.

        Args:
            order: Order to execute

        Returns:
            AlgoExecution with results
        """
        self._status = AlgoStatus.RUNNING
        self._slices = []
        self._cancel_requested = False

        start_time = datetime.now()

        # Get initial market data
        market_data = await self.market_data_provider(order.symbol)
        arrival_price = market_data.get("mid_price", market_data.get("last_price", 0))

        # Generate schedule
        schedule = await self.generate_schedule(order, market_data)

        total_filled = 0.0
        total_cost = 0.0
        volume_traded = 0.0
        vwap_sum = 0.0
        slice_count = 0

        try:
            for target_time, target_qty in schedule:
                if self._cancel_requested:
                    self._status = AlgoStatus.CANCELLED
                    break

                # Wait until target time
                now = datetime.now()
                if target_time > now:
                    wait_seconds = (target_time - now).total_seconds()
                    await asyncio.sleep(wait_seconds)

                # Check if paused
                while self._status == AlgoStatus.PAUSED:
                    await asyncio.sleep(0.1)
                    if self._cancel_requested:
                        break

                if self._cancel_requested:
                    self._status = AlgoStatus.CANCELLED
                    break

                # Get current market data
                current_data = await self.market_data_provider(order.symbol)
                current_price = current_data.get("mid_price", current_data.get("last_price", 0))
                current_volume = current_data.get("volume", 0)

                # Adjust quantity based on available liquidity
                adjusted_qty = self._adjust_quantity(
                    target_qty, order, current_data
                )

                if adjusted_qty <= 0:
                    continue

                # Check limit price
                if order.limit_price:
                    if order.side == "buy" and current_price > order.limit_price:
                        continue
                    if order.side == "sell" and current_price < order.limit_price:
                        continue

                # Execute slice
                slice_id = f"{order.order_id}-{slice_count}"
                slice_order = AlgoSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=adjusted_qty,
                    target_price=current_price,
                    submitted_at=datetime.now(),
                )

                try:
                    result = await self.order_executor(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=adjusted_qty,
                        price=current_price,
                    )

                    fill_price = result.get("fill_price", current_price)
                    fill_qty = result.get("filled_quantity", adjusted_qty)

                    slice_order.actual_price = fill_price
                    slice_order.filled_quantity = fill_qty
                    slice_order.filled_at = datetime.now()
                    slice_order.status = "filled" if fill_qty > 0 else "rejected"

                    total_filled += fill_qty
                    total_cost += fill_qty * fill_price
                    volume_traded += current_volume
                    vwap_sum += current_price * current_volume

                except Exception as e:
                    logger.error(f"Slice execution failed: {e}")
                    slice_order.status = "failed"

                self._slices.append(slice_order)
                slice_count += 1

                # Check if order complete
                if total_filled >= order.total_quantity * 0.99:
                    break

            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            avg_price = total_cost / total_filled if total_filled > 0 else 0
            vwap_benchmark = vwap_sum / volume_traded if volume_traded > 0 else avg_price

            # Slippage vs VWAP
            if order.side == "buy":
                slippage_bps = (avg_price - vwap_benchmark) / vwap_benchmark * 10000 if vwap_benchmark > 0 else 0
            else:
                slippage_bps = (vwap_benchmark - avg_price) / vwap_benchmark * 10000 if vwap_benchmark > 0 else 0

            # Implementation shortfall vs arrival price
            if order.side == "buy":
                is_bps = (avg_price - arrival_price) / arrival_price * 10000 if arrival_price > 0 else 0
            else:
                is_bps = (arrival_price - avg_price) / arrival_price * 10000 if arrival_price > 0 else 0

            # Participation rate
            participation = total_filled / volume_traded if volume_traded > 0 else 0

            if self._status != AlgoStatus.CANCELLED:
                self._status = AlgoStatus.COMPLETED if total_filled > 0 else AlgoStatus.FAILED

            return AlgoExecution(
                order_id=order.order_id,
                algorithm=self.name,
                symbol=order.symbol,
                side=order.side,
                total_quantity=order.total_quantity,
                filled_quantity=total_filled,
                average_price=avg_price,
                vwap_benchmark=vwap_benchmark,
                arrival_price=arrival_price,
                slippage_bps=slippage_bps,
                implementation_shortfall_bps=is_bps,
                participation_rate=participation,
                num_slices=slice_count,
                duration_seconds=duration,
                status=self._status,
                slices=self._slices,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}")
            self._status = AlgoStatus.FAILED
            raise

    def _adjust_quantity(
        self,
        target_qty: float,
        order: AlgoOrder,
        market_data: Dict
    ) -> float:
        """Adjust quantity based on market conditions."""
        # Check available liquidity
        if order.side == "buy":
            available = market_data.get("ask_size", float('inf'))
        else:
            available = market_data.get("bid_size", float('inf'))

        # Don't take more than max participation of available
        max_qty = available * order.max_participation
        adjusted = min(target_qty, max_qty)

        # Ensure we don't overfill
        remaining = order.total_quantity - sum(s.filled_quantity for s in self._slices)
        adjusted = min(adjusted, remaining)

        return max(0, adjusted)

    def pause(self):
        """Pause execution."""
        if self._status == AlgoStatus.RUNNING:
            self._status = AlgoStatus.PAUSED

    def resume(self):
        """Resume execution."""
        if self._status == AlgoStatus.PAUSED:
            self._status = AlgoStatus.RUNNING

    def cancel(self):
        """Cancel execution."""
        self._cancel_requested = True


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) Algorithm.

    Executes order in equal slices over specified time period.
    Best for orders with no urgency and when minimizing market impact.
    """

    @property
    def name(self) -> str:
        return "TWAP"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """Generate evenly-spaced execution schedule."""
        start = order.start_time or datetime.now()
        end = order.end_time or (start + self._get_default_duration(order))

        total_duration = (end - start).total_seconds()
        num_slices = max(1, int(total_duration / self.interval_seconds))

        slice_qty = order.total_quantity / num_slices
        schedule = []

        for i in range(num_slices):
            slice_time = start + timedelta(seconds=i * self.interval_seconds)
            # Add small randomization to avoid predictability
            jitter = random.uniform(-0.1, 0.1) * self.interval_seconds
            slice_time += timedelta(seconds=jitter)
            schedule.append((slice_time, slice_qty))

        return schedule

    def _get_default_duration(self, order: AlgoOrder) -> timedelta:
        """Get default duration based on urgency."""
        if order.urgency == UrgencyLevel.CRITICAL:
            return timedelta(minutes=5)
        elif order.urgency == UrgencyLevel.HIGH:
            return timedelta(minutes=30)
        elif order.urgency == UrgencyLevel.MEDIUM:
            return timedelta(hours=2)
        else:
            return timedelta(hours=6)


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) Algorithm.

    Executes order proportionally to expected volume profile.
    Best for achieving execution close to market VWAP.
    """

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        volume_profile: Optional[List[float]] = None,
        interval_seconds: float = 60.0
    ):
        super().__init__(order_executor, market_data_provider, interval_seconds)
        # Default hourly volume profile (percentage of daily volume)
        self.volume_profile = volume_profile or [
            0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.06, 0.06
        ]

    @property
    def name(self) -> str:
        return "VWAP"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """Generate volume-weighted execution schedule."""
        start = order.start_time or datetime.now()
        end = order.end_time or (start + self._get_default_duration(order))

        total_duration = (end - start).total_seconds()
        num_intervals = max(1, int(total_duration / self.interval_seconds))

        # Get volume weights for our time period
        schedule = []
        remaining_qty = order.total_quantity

        for i in range(num_intervals):
            slice_time = start + timedelta(seconds=i * self.interval_seconds)

            # Determine volume weight based on time of day
            hour = slice_time.hour
            profile_idx = min(hour, len(self.volume_profile) - 1)
            weight = self.volume_profile[profile_idx]

            # Calculate slice quantity
            slice_qty = order.total_quantity * weight / sum(self.volume_profile)

            # Adjust to not exceed remaining
            slice_qty = min(slice_qty, remaining_qty)

            if slice_qty > 0:
                schedule.append((slice_time, slice_qty))
                remaining_qty -= slice_qty

        # Distribute any remaining evenly
        if remaining_qty > 0 and schedule:
            extra_per_slice = remaining_qty / len(schedule)
            schedule = [(t, q + extra_per_slice) for t, q in schedule]

        return schedule

    def _get_default_duration(self, order: AlgoOrder) -> timedelta:
        """Get default duration based on urgency."""
        if order.urgency == UrgencyLevel.CRITICAL:
            return timedelta(minutes=10)
        elif order.urgency == UrgencyLevel.HIGH:
            return timedelta(hours=1)
        elif order.urgency == UrgencyLevel.MEDIUM:
            return timedelta(hours=4)
        else:
            return timedelta(hours=8)


class POVAlgorithm(ExecutionAlgorithm):
    """
    Percentage of Volume (POV) Algorithm.

    Executes as a fixed percentage of market volume.
    Adapts to market activity - more volume = more execution.
    """

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        target_participation: float = 0.1,
        interval_seconds: float = 30.0
    ):
        super().__init__(order_executor, market_data_provider, interval_seconds)
        self.target_participation = target_participation  # 10% of volume

    @property
    def name(self) -> str:
        return "POV"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """
        POV doesn't pre-generate schedule - it adapts to real-time volume.
        Return placeholder schedule that will be adjusted during execution.
        """
        start = order.start_time or datetime.now()
        end = order.end_time or (start + self._get_default_duration(order))

        total_duration = (end - start).total_seconds()
        num_intervals = max(1, int(total_duration / self.interval_seconds))

        # Initial estimate based on expected volume
        expected_volume = market_data.get("avg_volume", order.total_quantity * 10)
        expected_slice_volume = expected_volume / num_intervals

        schedule = []
        for i in range(num_intervals):
            slice_time = start + timedelta(seconds=i * self.interval_seconds)
            # Estimate based on participation rate
            slice_qty = expected_slice_volume * self.target_participation
            schedule.append((slice_time, slice_qty))

        return schedule

    def _adjust_quantity(
        self,
        target_qty: float,
        order: AlgoOrder,
        market_data: Dict
    ) -> float:
        """Adjust based on actual volume."""
        # Get recent volume
        recent_volume = market_data.get("recent_volume", market_data.get("volume", 0))

        # Calculate target based on participation
        volume_based_qty = recent_volume * self.target_participation

        # Use volume-based quantity, capped by order limits
        adjusted = min(volume_based_qty, target_qty)
        adjusted = min(adjusted, market_data.get("bid_size" if order.side == "sell" else "ask_size", float('inf')) * order.max_participation)

        # Don't overfill
        remaining = order.total_quantity - sum(s.filled_quantity for s in self._slices)
        adjusted = min(adjusted, remaining)

        return max(0, adjusted)

    def _get_default_duration(self, order: AlgoOrder) -> timedelta:
        """POV runs until filled or cancelled."""
        if order.urgency == UrgencyLevel.CRITICAL:
            return timedelta(minutes=15)
        elif order.urgency == UrgencyLevel.HIGH:
            return timedelta(hours=2)
        else:
            return timedelta(hours=8)


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """
    Implementation Shortfall (IS) Algorithm.

    Minimizes the difference between decision price and execution price.
    Balances market impact vs timing risk based on volatility.
    """

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        risk_aversion: float = 0.5,
        interval_seconds: float = 30.0
    ):
        super().__init__(order_executor, market_data_provider, interval_seconds)
        self.risk_aversion = risk_aversion  # 0 = minimize impact, 1 = minimize timing risk

    @property
    def name(self) -> str:
        return "IS"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """
        Generate schedule that balances impact vs timing risk.

        Uses Almgren-Chriss framework principles:
        - Higher volatility -> trade faster (reduce timing risk)
        - Larger order -> trade slower (reduce impact)
        """
        start = order.start_time or datetime.now()

        # Get market parameters
        volatility = market_data.get("volatility", 0.02)  # Daily vol
        spread = market_data.get("spread", 0.001)
        avg_volume = market_data.get("avg_volume", order.total_quantity * 10)

        # Calculate optimal trading rate
        # Higher risk aversion = trade faster
        # Higher volatility = trade faster
        order_as_pct_volume = order.total_quantity / avg_volume if avg_volume > 0 else 0.1

        # Estimate optimal duration
        base_duration_hours = 4  # Default
        vol_adjustment = 1 - min(1, volatility * 20)  # High vol = shorter duration
        size_adjustment = min(2, order_as_pct_volume * 10)  # Larger order = longer
        risk_adjustment = 1 - self.risk_aversion * 0.5  # High risk aversion = shorter

        optimal_hours = base_duration_hours * vol_adjustment * size_adjustment * risk_adjustment
        optimal_hours = max(0.25, min(8, optimal_hours))  # Clamp to 15min - 8hrs

        # Adjust for urgency
        if order.urgency == UrgencyLevel.CRITICAL:
            optimal_hours = min(optimal_hours, 0.25)
        elif order.urgency == UrgencyLevel.HIGH:
            optimal_hours = min(optimal_hours, 1)

        end = order.end_time or (start + timedelta(hours=optimal_hours))
        total_duration = (end - start).total_seconds()
        num_intervals = max(1, int(total_duration / self.interval_seconds))

        # Generate front-loaded schedule (execute more at start to reduce timing risk)
        schedule = []
        remaining_qty = order.total_quantity

        for i in range(num_intervals):
            slice_time = start + timedelta(seconds=i * self.interval_seconds)

            # Front-loaded: more aggressive at start
            progress = i / num_intervals
            weight = (1 - progress) ** (1 - self.risk_aversion)

            slice_qty = order.total_quantity * weight / sum(
                (1 - j/num_intervals) ** (1 - self.risk_aversion)
                for j in range(num_intervals)
            )
            slice_qty = min(slice_qty, remaining_qty)

            if slice_qty > 0:
                schedule.append((slice_time, slice_qty))
                remaining_qty -= slice_qty

        return schedule

    def _get_default_duration(self, order: AlgoOrder) -> timedelta:
        """Duration is calculated dynamically based on market conditions."""
        return timedelta(hours=4)


class IcebergAlgorithm(ExecutionAlgorithm):
    """
    Iceberg Algorithm.

    Shows only a small "visible" portion of the order.
    Automatically replenishes when filled.
    """

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        visible_quantity: float = 0.1,  # 10% visible
        interval_seconds: float = 5.0
    ):
        super().__init__(order_executor, market_data_provider, interval_seconds)
        self.visible_quantity = visible_quantity

    @property
    def name(self) -> str:
        return "Iceberg"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """Generate schedule with fixed visible size."""
        start = order.start_time or datetime.now()

        visible_qty = order.total_quantity * self.visible_quantity
        num_slices = math.ceil(order.total_quantity / visible_qty)

        schedule = []
        remaining = order.total_quantity

        for i in range(num_slices):
            slice_time = start + timedelta(seconds=i * self.interval_seconds)
            slice_qty = min(visible_qty, remaining)

            if slice_qty > 0:
                schedule.append((slice_time, slice_qty))
                remaining -= slice_qty

        return schedule


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """
    Adaptive Algorithm.

    Dynamically adjusts strategy based on market conditions.
    Combines elements of TWAP, VWAP, and IS.
    """

    def __init__(
        self,
        order_executor: Callable,
        market_data_provider: Callable,
        interval_seconds: float = 30.0
    ):
        super().__init__(order_executor, market_data_provider, interval_seconds)
        self._twap = TWAPAlgorithm(order_executor, market_data_provider, interval_seconds)
        self._vwap = VWAPAlgorithm(order_executor, market_data_provider, interval_seconds=interval_seconds)
        self._is = ImplementationShortfallAlgorithm(order_executor, market_data_provider, interval_seconds=interval_seconds)

    @property
    def name(self) -> str:
        return "Adaptive"

    async def generate_schedule(
        self,
        order: AlgoOrder,
        market_data: Dict[str, Any]
    ) -> List[Tuple[datetime, float]]:
        """Generate schedule based on current market conditions."""
        volatility = market_data.get("volatility", 0.02)
        spread = market_data.get("spread_bps", 10)
        volume = market_data.get("volume", 0)

        # Choose strategy based on conditions
        if volatility > 0.03:
            # High volatility: use IS (minimize timing risk)
            logger.info("Adaptive: High volatility, using IS")
            return await self._is.generate_schedule(order, market_data)

        elif spread > 30:
            # Wide spread: use TWAP (patient execution)
            logger.info("Adaptive: Wide spread, using TWAP")
            return await self._twap.generate_schedule(order, market_data)

        else:
            # Normal conditions: use VWAP
            logger.info("Adaptive: Normal conditions, using VWAP")
            return await self._vwap.generate_schedule(order, market_data)


class AlgorithmFactory:
    """Factory for creating execution algorithms."""

    @staticmethod
    def create(
        algorithm_type: str,
        order_executor: Callable,
        market_data_provider: Callable,
        **kwargs
    ) -> ExecutionAlgorithm:
        """
        Create an execution algorithm.

        Args:
            algorithm_type: "twap", "vwap", "pov", "is", "iceberg", "adaptive"
            order_executor: Async function to execute orders
            market_data_provider: Async function to get market data
            **kwargs: Algorithm-specific parameters
        """
        algo_type = algorithm_type.lower()

        if algo_type == "twap":
            return TWAPAlgorithm(order_executor, market_data_provider, **kwargs)
        elif algo_type == "vwap":
            return VWAPAlgorithm(order_executor, market_data_provider, **kwargs)
        elif algo_type == "pov":
            return POVAlgorithm(order_executor, market_data_provider, **kwargs)
        elif algo_type == "is":
            return ImplementationShortfallAlgorithm(order_executor, market_data_provider, **kwargs)
        elif algo_type == "iceberg":
            return IcebergAlgorithm(order_executor, market_data_provider, **kwargs)
        elif algo_type == "adaptive":
            return AdaptiveAlgorithm(order_executor, market_data_provider, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")


def create_execution_algorithm(
    algorithm_type: str,
    order_executor: Callable,
    market_data_provider: Callable,
    **kwargs
) -> ExecutionAlgorithm:
    """Factory function to create execution algorithm."""
    return AlgorithmFactory.create(
        algorithm_type, order_executor, market_data_provider, **kwargs
    )
