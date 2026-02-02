"""
Execution Reality Layer.

Production-grade execution simulator shared by backtest/paper/live evaluation:
- Spread and slippage modeling
- Partial fill simulation
- Latency modeling
- Fee calculation (maker/taker/exchange-specific)
- Funding rate costs (perpetuals)
- Borrow costs (shorting)

Designed to make backtests realistic and evaluate paper vs live differences.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types with different fee structures."""
    BINANCE_SPOT = "binance_spot"
    BINANCE_FUTURES = "binance_futures"
    KRAKEN_SPOT = "kraken_spot"
    COINBASE_PRO = "coinbase_pro"
    ALPACA_STOCKS = "alpaca_stocks"
    OANDA_FOREX = "oanda_forex"
    GENERIC = "generic"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class FeeSchedule:
    """Fee schedule for an exchange."""
    maker_fee_pct: float  # Fee for limit orders that add liquidity
    taker_fee_pct: float  # Fee for market orders that take liquidity
    min_fee_usd: float = 0.0  # Minimum fee per trade
    funding_rate_8h: float = 0.0  # Perpetual funding rate (8h)
    borrow_rate_daily: float = 0.0  # Daily borrow rate for shorting


@dataclass
class SlippageModel:
    """
    Slippage model parameters.

    Uses a square-root market impact model (empirically validated):
    - Linear component for small orders
    - Square-root component for larger orders (captures non-linear impact)
    - Size tiers with graduated impact factors
    """
    base_slippage_bps: float = 2.0  # Base slippage in basis points
    volume_impact_factor: float = 0.001  # Additional slippage per % of daily volume
    volatility_factor: float = 0.5  # Multiplier for volatility impact
    market_order_penalty: float = 1.5  # Extra slippage for market orders
    low_liquidity_multiplier: float = 2.0  # Multiplier for low liquidity conditions

    # Size-aware slippage parameters (Phase 2A addition)
    size_aware_enabled: bool = True  # Enable square-root market impact
    sqrt_impact_coefficient: float = 0.01  # Coefficient for sqrt(size/volume) term
    size_tier_thresholds: tuple = (0.001, 0.005, 0.01, 0.05)  # % of daily volume
    size_tier_multipliers: tuple = (1.0, 1.3, 1.8, 2.5, 4.0)  # Multipliers per tier


@dataclass
class LatencyModel:
    """Latency model for execution timing."""
    base_latency_ms: float = 50.0  # Base network latency
    exchange_processing_ms: float = 10.0  # Exchange order processing
    variance_ms: float = 20.0  # Random variance
    spike_probability: float = 0.02  # Probability of latency spike
    spike_multiplier: float = 5.0  # Latency during spikes


@dataclass
class PartialFillModel:
    """Partial fill model parameters."""
    fill_rate_base: float = 0.95  # Base fill rate for limit orders
    volume_fill_factor: float = 0.1  # Reduction per % of daily volume
    spread_fill_factor: float = 0.05  # Reduction based on spread distance
    min_fill_rate: float = 0.3  # Minimum partial fill


@dataclass
class SimulatorConfig:
    """Configuration for the execution simulator."""
    exchange_type: ExchangeType = ExchangeType.BINANCE_FUTURES
    slippage_model: SlippageModel = field(default_factory=SlippageModel)
    latency_model: LatencyModel = field(default_factory=LatencyModel)
    partial_fill_model: PartialFillModel = field(default_factory=PartialFillModel)
    custom_fee_schedule: Optional[FeeSchedule] = None


@dataclass
class ExecutionResult:
    """Result of simulated execution."""
    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    requested_quantity: float
    filled_quantity: float
    requested_price: float
    execution_price: float  # Average fill price
    slippage_pct: float
    slippage_usd: float
    fees_usd: float
    fee_breakdown: Dict[str, float]
    latency_ms: float
    is_partial_fill: bool
    fill_rate: float
    timestamp: datetime
    simulation_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Total cost including slippage and fees."""
        return self.slippage_usd + self.fees_usd

    @property
    def effective_price(self) -> float:
        """Effective price including all costs (for buys)."""
        if self.side.upper() == "BUY":
            return self.execution_price + self.total_cost / max(self.filled_quantity, 0.001)
        else:
            return self.execution_price - self.total_cost / max(self.filled_quantity, 0.001)


# Predefined fee schedules by exchange
FEE_SCHEDULES = {
    ExchangeType.BINANCE_SPOT: FeeSchedule(
        maker_fee_pct=0.0010,  # 0.10%
        taker_fee_pct=0.0010,  # 0.10% (with BNB discount)
    ),
    ExchangeType.BINANCE_FUTURES: FeeSchedule(
        maker_fee_pct=0.0002,  # 0.02%
        taker_fee_pct=0.0004,  # 0.04%
        funding_rate_8h=0.0001,  # 0.01% typical
    ),
    ExchangeType.KRAKEN_SPOT: FeeSchedule(
        maker_fee_pct=0.0016,  # 0.16%
        taker_fee_pct=0.0026,  # 0.26%
    ),
    ExchangeType.COINBASE_PRO: FeeSchedule(
        maker_fee_pct=0.0040,  # 0.40%
        taker_fee_pct=0.0060,  # 0.60%
    ),
    ExchangeType.ALPACA_STOCKS: FeeSchedule(
        maker_fee_pct=0.0,  # Commission-free
        taker_fee_pct=0.0,
    ),
    ExchangeType.OANDA_FOREX: FeeSchedule(
        maker_fee_pct=0.0,  # Spread-based
        taker_fee_pct=0.0,
    ),
    ExchangeType.GENERIC: FeeSchedule(
        maker_fee_pct=0.0020,  # 0.20%
        taker_fee_pct=0.0040,  # 0.40%
    ),
}


class ExecutionSimulator:
    """
    Realistic execution simulator for backtesting and paper trading.

    Features:
    - Exchange-specific fee schedules
    - Volume-adjusted slippage
    - Volatility-based slippage
    - Partial fill simulation
    - Latency modeling
    - Funding rate calculation
    - Borrow cost estimation
    """

    def __init__(
        self,
        exchange_type: ExchangeType = ExchangeType.BINANCE_FUTURES,
        fee_schedule: Optional[FeeSchedule] = None,
        slippage_model: Optional[SlippageModel] = None,
        latency_model: Optional[LatencyModel] = None,
        partial_fill_model: Optional[PartialFillModel] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the execution simulator.

        Args:
            exchange_type: Type of exchange for fee structure
            fee_schedule: Custom fee schedule (overrides exchange default)
            slippage_model: Slippage model parameters
            latency_model: Latency model parameters
            partial_fill_model: Partial fill model
            random_seed: Seed for reproducibility
        """
        self.exchange_type = exchange_type
        self.fee_schedule = fee_schedule or FEE_SCHEDULES.get(
            exchange_type, FEE_SCHEDULES[ExchangeType.GENERIC]
        )
        self.slippage_model = slippage_model or SlippageModel()
        self.latency_model = latency_model or LatencyModel()
        self.partial_fill_model = partial_fill_model or PartialFillModel()

        if random_seed is not None:
            random.seed(random_seed)

        # Statistics tracking
        self._total_slippage_usd: float = 0.0
        self._total_fees_usd: float = 0.0
        self._execution_count: int = 0
        self._partial_fill_count: int = 0

        logger.info(f"ExecutionSimulator initialized for {exchange_type.value}")

    def simulate_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        volatility: float = 0.02,
        daily_volume: float = 1_000_000.0,
        spread_bps: float = 5.0,
        is_short: bool = False,
        holding_hours: float = 0.0,
    ) -> ExecutionResult:
        """
        Simulate order execution with realistic friction.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Current market price
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            volatility: Current volatility (annualized or realized)
            daily_volume: 24h volume in quote currency
            spread_bps: Current bid-ask spread in basis points
            is_short: Whether this is a short position
            holding_hours: Hours position will be held (for funding)

        Returns:
            ExecutionResult with all execution details
        """
        order_id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # Calculate latency
        latency_ms = self._simulate_latency()

        # Calculate slippage
        slippage_pct, slippage_details = self._calculate_slippage(
            order_type=order_type,
            price=price,
            quantity=quantity,
            volatility=volatility,
            daily_volume=daily_volume,
            spread_bps=spread_bps,
            side=side,
        )

        # Calculate execution price
        if side.upper() == "BUY":
            execution_price = price * (1 + slippage_pct)
        else:
            execution_price = price * (1 - slippage_pct)

        # Calculate partial fill
        fill_rate, filled_quantity = self._calculate_partial_fill(
            order_type=order_type,
            quantity=quantity,
            price=price,
            limit_price=limit_price,
            daily_volume=daily_volume,
            spread_bps=spread_bps,
        )

        # Calculate fees
        fees, fee_breakdown = self._calculate_fees(
            quantity=filled_quantity,
            price=execution_price,
            order_type=order_type,
            is_short=is_short,
            holding_hours=holding_hours,
        )

        # Calculate slippage in USD
        slippage_usd = abs(execution_price - price) * filled_quantity

        # Update statistics
        self._total_slippage_usd += slippage_usd
        self._total_fees_usd += fees
        self._execution_count += 1
        if fill_rate < 1.0:
            self._partial_fill_count += 1

        result = ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            requested_price=price,
            execution_price=execution_price,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd,
            fees_usd=fees,
            fee_breakdown=fee_breakdown,
            latency_ms=latency_ms,
            is_partial_fill=fill_rate < 1.0,
            fill_rate=fill_rate,
            timestamp=datetime.now(),
            simulation_details={
                "exchange": self.exchange_type.value,
                "volatility": volatility,
                "daily_volume": daily_volume,
                "spread_bps": spread_bps,
                **slippage_details,
            },
        )

        logger.debug(
            f"Simulated execution: {symbol} {side} {filled_quantity:.4f} @ {execution_price:.4f} "
            f"(slip={slippage_pct*100:.3f}%, fees=${fees:.4f}, latency={latency_ms:.0f}ms)"
        )

        return result

    def _simulate_latency(self) -> float:
        """Simulate network/exchange latency."""
        base = self.latency_model.base_latency_ms + self.latency_model.exchange_processing_ms
        variance = random.gauss(0, self.latency_model.variance_ms)

        # Occasional latency spike
        if random.random() < self.latency_model.spike_probability:
            base *= self.latency_model.spike_multiplier

        return max(5.0, base + variance)

    def _calculate_slippage(
        self,
        order_type: OrderType,
        price: float,
        quantity: float,
        volatility: float,
        daily_volume: float,
        spread_bps: float,
        side: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate slippage based on market conditions with size-aware impact.

        Uses square-root market impact model for larger orders:
        - Impact = sigma * sqrt(Q/V) where Q is order size and V is daily volume
        - This is the Kyle/Almgren-Chriss model used in practice

        Returns:
            Tuple of (slippage_pct, details_dict)
        """
        model = self.slippage_model

        # Base slippage from spread
        base_slip = spread_bps / 10000 / 2  # Half spread

        # Order size as fraction of daily volume
        order_value = quantity * price
        volume_pct = order_value / max(daily_volume, 1.0)

        # Linear volume impact (small orders)
        linear_slip = volume_pct * model.volume_impact_factor

        # Size-aware slippage using square-root market impact model
        size_impact = 0.0
        size_tier_multiplier = 1.0
        if model.size_aware_enabled:
            # Square-root impact: proportional to sqrt(order_size / daily_volume)
            # This captures the empirical observation that larger orders have
            # diminishing (but still increasing) impact
            size_impact = model.sqrt_impact_coefficient * math.sqrt(volume_pct)

            # Determine size tier multiplier
            # Larger orders face progressively higher impact due to:
            # - Depleting visible liquidity at best prices
            # - Moving through multiple price levels
            # - Adverse selection (informed trading signal)
            for i, threshold in enumerate(model.size_tier_thresholds):
                if volume_pct <= threshold:
                    size_tier_multiplier = model.size_tier_multipliers[i]
                    break
            else:
                # Beyond largest threshold
                size_tier_multiplier = model.size_tier_multipliers[-1]

        # Volatility impact (higher vol = worse fills)
        vol_slip = volatility * model.volatility_factor * model.base_slippage_bps / 10000

        # Order type adjustment
        type_multiplier = 1.0
        if order_type == OrderType.MARKET:
            type_multiplier = model.market_order_penalty
        elif order_type == OrderType.STOP_MARKET:
            type_multiplier = model.market_order_penalty * 1.2  # Extra slippage for stops

        # Random component (market microstructure noise)
        random_slip = random.gauss(0, model.base_slippage_bps / 20000)

        # Total slippage with size-aware components
        total_slip = (
            base_slip +
            (linear_slip + size_impact) * size_tier_multiplier +
            vol_slip +
            abs(random_slip)
        ) * type_multiplier

        # Low liquidity multiplier (if volume is very low)
        if daily_volume < 100_000:
            total_slip *= model.low_liquidity_multiplier

        # Cap at reasonable maximum
        total_slip = min(total_slip, 0.03)  # Max 3% slippage

        details = {
            "base_slippage": base_slip,
            "linear_volume_impact": linear_slip,
            "sqrt_size_impact": size_impact,
            "size_tier_multiplier": size_tier_multiplier,
            "volume_pct_of_daily": volume_pct,
            "volatility_impact": vol_slip,
            "random_component": random_slip,
            "type_multiplier": type_multiplier,
            "total_slippage_pct": total_slip,
            "size_aware_enabled": model.size_aware_enabled,
        }

        return total_slip, details

    def _calculate_partial_fill(
        self,
        order_type: OrderType,
        quantity: float,
        price: float,
        limit_price: Optional[float],
        daily_volume: float,
        spread_bps: float,
    ) -> Tuple[float, float]:
        """
        Calculate fill rate and filled quantity.

        Returns:
            Tuple of (fill_rate, filled_quantity)
        """
        model = self.partial_fill_model

        # Market orders always fill (in simulation)
        if order_type == OrderType.MARKET:
            return 1.0, quantity

        # Limit orders may partially fill
        fill_rate = model.fill_rate_base

        # Volume impact on fill rate
        order_value = quantity * price
        volume_pct = order_value / max(daily_volume, 1.0)
        fill_rate -= volume_pct * model.volume_fill_factor

        # Spread distance impact (for limit orders)
        if limit_price is not None:
            price_diff_pct = abs(limit_price - price) / price
            fill_rate -= price_diff_pct * model.spread_fill_factor * 100

        # Add randomness
        fill_rate += random.gauss(0, 0.05)

        # Ensure within bounds
        fill_rate = max(model.min_fill_rate, min(1.0, fill_rate))

        filled_quantity = quantity * fill_rate

        return fill_rate, filled_quantity

    def _calculate_fees(
        self,
        quantity: float,
        price: float,
        order_type: OrderType,
        is_short: bool,
        holding_hours: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate all fees including trading fees, funding, and borrow costs.

        Returns:
            Tuple of (total_fees, breakdown_dict)
        """
        schedule = self.fee_schedule
        notional = quantity * price

        # Trading fee (maker vs taker)
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            trading_fee = notional * schedule.maker_fee_pct
            fee_type = "maker"
        else:
            trading_fee = notional * schedule.taker_fee_pct
            fee_type = "taker"

        # Minimum fee
        trading_fee = max(trading_fee, schedule.min_fee_usd)

        breakdown = {
            "trading_fee": trading_fee,
            "fee_type": fee_type,
        }

        total_fees = trading_fee

        # Funding costs (for perpetuals)
        if schedule.funding_rate_8h > 0 and holding_hours > 0:
            funding_periods = holding_hours / 8
            funding_cost = notional * schedule.funding_rate_8h * funding_periods
            breakdown["funding_cost"] = funding_cost
            total_fees += funding_cost

        # Borrow costs (for shorts)
        if is_short and schedule.borrow_rate_daily > 0 and holding_hours > 0:
            days = holding_hours / 24
            borrow_cost = notional * schedule.borrow_rate_daily * days
            breakdown["borrow_cost"] = borrow_cost
            total_fees += borrow_cost

        return total_fees, breakdown

    def estimate_round_trip_cost(
        self,
        quantity: float,
        price: float,
        volatility: float = 0.02,
        daily_volume: float = 1_000_000.0,
        spread_bps: float = 5.0,
        holding_hours: float = 24.0,
        is_short: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate total round-trip costs (entry + exit).

        Useful for determining minimum required move for profitability.
        """
        # Simulate entry
        entry = self.simulate_execution(
            symbol="ESTIMATE",
            side="BUY",
            quantity=quantity,
            price=price,
            order_type=OrderType.MARKET,
            volatility=volatility,
            daily_volume=daily_volume,
            spread_bps=spread_bps,
            is_short=is_short,
            holding_hours=holding_hours,
        )

        # Simulate exit
        exit_result = self.simulate_execution(
            symbol="ESTIMATE",
            side="SELL",
            quantity=quantity,
            price=price,  # Same price to isolate costs
            order_type=OrderType.MARKET,
            volatility=volatility,
            daily_volume=daily_volume,
            spread_bps=spread_bps,
            is_short=is_short,
            holding_hours=0,
        )

        total_slippage = entry.slippage_usd + exit_result.slippage_usd
        total_fees = entry.fees_usd + exit_result.fees_usd
        total_cost = total_slippage + total_fees
        notional = quantity * price

        return {
            "entry_slippage_usd": entry.slippage_usd,
            "exit_slippage_usd": exit_result.slippage_usd,
            "total_slippage_usd": total_slippage,
            "entry_fees_usd": entry.fees_usd,
            "exit_fees_usd": exit_result.fees_usd,
            "total_fees_usd": total_fees,
            "total_cost_usd": total_cost,
            "cost_pct": total_cost / notional * 100,
            "min_move_for_profit_pct": total_cost / notional * 100,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulated simulation statistics."""
        return {
            "execution_count": self._execution_count,
            "partial_fill_count": self._partial_fill_count,
            "partial_fill_rate": self._partial_fill_count / max(self._execution_count, 1),
            "total_slippage_usd": self._total_slippage_usd,
            "total_fees_usd": self._total_fees_usd,
            "avg_slippage_usd": self._total_slippage_usd / max(self._execution_count, 1),
            "avg_fees_usd": self._total_fees_usd / max(self._execution_count, 1),
        }

    def reset_statistics(self):
        """Reset accumulated statistics."""
        self._total_slippage_usd = 0.0
        self._total_fees_usd = 0.0
        self._execution_count = 0
        self._partial_fill_count = 0


# Singleton for default simulator
_default_simulator: Optional[ExecutionSimulator] = None


def get_execution_simulator(
    exchange_type: ExchangeType = ExchangeType.BINANCE_FUTURES,
) -> ExecutionSimulator:
    """Get or create the default ExecutionSimulator."""
    global _default_simulator
    if _default_simulator is None or _default_simulator.exchange_type != exchange_type:
        _default_simulator = ExecutionSimulator(exchange_type=exchange_type)
    return _default_simulator


def reset_execution_simulator() -> None:
    """Reset the singleton instance (for testing)."""
    global _default_simulator
    _default_simulator = None
