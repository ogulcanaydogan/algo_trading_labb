"""
Multi-Exchange Trading Coordinator.

Enables trading across multiple exchanges with:
- Best price aggregation
- Smart order routing
- Cross-exchange arbitrage detection
- Position consolidation
- Failover support

Usage:
    from bot.execution.multi_exchange import MultiExchangeCoordinator

    coordinator = MultiExchangeCoordinator()
    coordinator.add_exchange("binance", binance_adapter)
    coordinator.add_exchange("kraken", kraken_adapter)

    # Get best price across exchanges
    best = await coordinator.get_best_price("BTC/USDT", "buy")

    # Execute with smart routing
    result = await coordinator.execute_order(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        routing="best_price",  # or "split", "primary"
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """Exchange connection status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


class RoutingStrategy(Enum):
    """Order routing strategies."""
    BEST_PRICE = "best_price"  # Route to exchange with best price
    SPLIT = "split"  # Split order across exchanges
    PRIMARY = "primary"  # Use primary exchange, failover if needed
    ROUND_ROBIN = "round_robin"  # Rotate between exchanges
    LOWEST_FEE = "lowest_fee"  # Route to lowest fee exchange


@dataclass
class ExchangeQuote:
    """Price quote from an exchange."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    latency_ms: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid) * 10000 if self.mid > 0 else 0


@dataclass
class AggregatedBook:
    """Aggregated order book across exchanges."""
    symbol: str
    bids: List[Tuple[str, float, float]]  # (exchange, price, size)
    asks: List[Tuple[str, float, float]]
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[Tuple[str, float, float]]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Tuple[str, float, float]]:
        return self.asks[0] if self.asks else None


@dataclass
class RouteDecision:
    """Decision on how to route an order."""
    primary_exchange: str
    secondary_exchanges: List[str]
    allocation: Dict[str, float]  # exchange -> quantity
    expected_price: float
    expected_fees: float
    reason: str


@dataclass
class ExecutionResult:
    """Result of a multi-exchange execution."""
    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    average_price: float
    total_fees: float

    exchange_fills: Dict[str, Dict]  # exchange -> {quantity, price, fees}
    started_at: datetime
    completed_at: datetime
    success: bool
    errors: List[str] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.requested_quantity if self.requested_quantity > 0 else 0

    @property
    def execution_time_ms(self) -> float:
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": round(self.average_price, 8),
            "total_fees": round(self.total_fees, 8),
            "fill_rate": round(self.fill_rate, 4),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "exchange_fills": self.exchange_fills,
            "success": self.success,
            "errors": self.errors,
        }


@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_bps: float
    max_quantity: float
    profit_estimate: float
    detected_at: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "spread_bps": round(self.spread_bps, 2),
            "max_quantity": self.max_quantity,
            "profit_estimate": round(self.profit_estimate, 2),
            "detected_at": self.detected_at.isoformat(),
        }


class ExchangeAdapter:
    """
    Base class for exchange adapters.

    Implementations should override these methods for each exchange.
    """

    def __init__(self, exchange_id: str, fee_rate: float = 0.001):
        self.exchange_id = exchange_id
        self.fee_rate = fee_rate
        self.is_connected = False
        self.last_error: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to exchange."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        raise NotImplementedError

    async def get_quote(self, symbol: str) -> ExchangeQuote:
        """Get current quote for symbol."""
        raise NotImplementedError

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get orderbook for symbol."""
        raise NotImplementedError

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict:
        """Place an order."""
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError

    async def get_balance(self, currency: str) -> float:
        """Get balance for currency."""
        raise NotImplementedError

    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        raise NotImplementedError


class MockExchangeAdapter(ExchangeAdapter):
    """Mock adapter for testing."""

    def __init__(
        self,
        exchange_id: str,
        base_price: float = 50000.0,
        spread_bps: float = 10.0,
        fee_rate: float = 0.001,
    ):
        super().__init__(exchange_id, fee_rate)
        self.base_price = base_price
        self.spread_bps = spread_bps
        self._balances: Dict[str, float] = {"USDT": 100000, "BTC": 10}
        self._order_id = 0

    async def connect(self) -> bool:
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        self.is_connected = False

    async def get_quote(self, symbol: str) -> ExchangeQuote:
        spread = self.base_price * self.spread_bps / 10000
        bid = self.base_price - spread / 2
        ask = self.base_price + spread / 2

        return ExchangeQuote(
            exchange=self.exchange_id,
            symbol=symbol,
            bid=bid,
            ask=ask,
            bid_size=10.0,
            ask_size=10.0,
            timestamp=datetime.now(),
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict:
        self._order_id += 1
        quote = await self.get_quote(symbol)
        fill_price = quote.ask if side == "buy" else quote.bid

        return {
            "order_id": f"{self.exchange_id}_{self._order_id}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "filled_quantity": quantity,
            "price": fill_price,
            "fees": quantity * fill_price * self.fee_rate,
            "status": "filled",
        }

    async def get_balance(self, currency: str) -> float:
        return self._balances.get(currency, 0.0)


class MultiExchangeCoordinator:
    """
    Coordinates trading across multiple exchanges.

    Features:
    - Best price aggregation
    - Smart order routing
    - Arbitrage detection
    - Position consolidation
    - Automatic failover
    """

    def __init__(
        self,
        primary_exchange: Optional[str] = None,
        min_arb_spread_bps: float = 10.0,
        max_slippage_bps: float = 50.0,
    ):
        """
        Initialize multi-exchange coordinator.

        Args:
            primary_exchange: Default exchange for PRIMARY routing
            min_arb_spread_bps: Minimum spread for arbitrage detection
            max_slippage_bps: Maximum allowed slippage
        """
        self.primary_exchange = primary_exchange
        self.min_arb_spread_bps = min_arb_spread_bps
        self.max_slippage_bps = max_slippage_bps

        self._exchanges: Dict[str, ExchangeAdapter] = {}
        self._status: Dict[str, ExchangeStatus] = {}
        self._latencies: Dict[str, List[float]] = {}
        self._order_counter = 0

    def add_exchange(
        self,
        name: str,
        adapter: ExchangeAdapter,
        is_primary: bool = False,
    ) -> None:
        """
        Add an exchange adapter.

        Args:
            name: Exchange name
            adapter: Exchange adapter instance
            is_primary: Set as primary exchange
        """
        self._exchanges[name] = adapter
        self._status[name] = ExchangeStatus.DISCONNECTED
        self._latencies[name] = []

        if is_primary or self.primary_exchange is None:
            self.primary_exchange = name

        logger.info(f"Added exchange: {name} (primary: {is_primary})")

    def remove_exchange(self, name: str) -> None:
        """Remove an exchange."""
        if name in self._exchanges:
            del self._exchanges[name]
            del self._status[name]
            del self._latencies[name]

            if self.primary_exchange == name:
                self.primary_exchange = next(iter(self._exchanges), None)

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all exchanges."""
        results = {}
        tasks = []

        for name, adapter in self._exchanges.items():
            tasks.append(self._connect_exchange(name, adapter))

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for name, outcome in zip(self._exchanges.keys(), outcomes):
            if isinstance(outcome, Exception):
                results[name] = False
                self._status[name] = ExchangeStatus.UNHEALTHY
            else:
                results[name] = outcome
                self._status[name] = ExchangeStatus.HEALTHY if outcome else ExchangeStatus.UNHEALTHY

        return results

    async def _connect_exchange(self, name: str, adapter: ExchangeAdapter) -> bool:
        """Connect to a single exchange with error handling."""
        try:
            result = await adapter.connect()
            return result
        except Exception as e:
            logger.error(f"Failed to connect to {name}: {e}")
            adapter.last_error = str(e)
            return False

    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges."""
        for name, adapter in self._exchanges.items():
            try:
                await adapter.disconnect()
                self._status[name] = ExchangeStatus.DISCONNECTED
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")

    async def get_best_price(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
    ) -> Optional[Tuple[str, float]]:
        """
        Get best price across all exchanges.

        Args:
            symbol: Trading symbol
            side: Order side

        Returns:
            Tuple of (exchange_name, price) or None
        """
        quotes = await self._get_all_quotes(symbol)
        if not quotes:
            return None

        if side == "buy":
            # Best buy = lowest ask
            best = min(quotes, key=lambda q: q.ask)
            return (best.exchange, best.ask)
        else:
            # Best sell = highest bid
            best = max(quotes, key=lambda q: q.bid)
            return (best.exchange, best.bid)

    async def get_aggregated_book(self, symbol: str) -> AggregatedBook:
        """
        Get aggregated orderbook across all exchanges.

        Args:
            symbol: Trading symbol

        Returns:
            AggregatedBook with sorted bids/asks
        """
        quotes = await self._get_all_quotes(symbol)

        bids = [(q.exchange, q.bid, q.bid_size) for q in quotes]
        asks = [(q.exchange, q.ask, q.ask_size) for q in quotes]

        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x[1], reverse=True)
        asks.sort(key=lambda x: x[1])

        return AggregatedBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
        )

    async def _get_all_quotes(self, symbol: str) -> List[ExchangeQuote]:
        """Get quotes from all healthy exchanges."""
        quotes = []
        tasks = []

        healthy_exchanges = [
            (name, adapter)
            for name, adapter in self._exchanges.items()
            if self._status.get(name) in (ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED)
        ]

        for name, adapter in healthy_exchanges:
            tasks.append(self._get_quote_with_timing(name, adapter, symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, ExchangeQuote):
                quotes.append(result)

        return quotes

    async def _get_quote_with_timing(
        self,
        name: str,
        adapter: ExchangeAdapter,
        symbol: str,
    ) -> ExchangeQuote:
        """Get quote with latency tracking."""
        start = time.perf_counter()
        try:
            quote = await adapter.get_quote(symbol)
            latency = (time.perf_counter() - start) * 1000
            quote.latency_ms = latency

            # Track latency
            self._latencies[name].append(latency)
            if len(self._latencies[name]) > 100:
                self._latencies[name] = self._latencies[name][-100:]

            return quote
        except Exception as e:
            logger.error(f"Error getting quote from {name}: {e}")
            self._status[name] = ExchangeStatus.DEGRADED
            raise

    def _decide_route(
        self,
        symbol: str,
        side: str,
        quantity: float,
        quotes: List[ExchangeQuote],
        strategy: RoutingStrategy,
    ) -> RouteDecision:
        """Decide how to route an order."""
        if not quotes:
            raise ValueError("No quotes available for routing")

        if strategy == RoutingStrategy.BEST_PRICE:
            # Route everything to best price
            if side == "buy":
                best = min(quotes, key=lambda q: q.ask)
                price = best.ask
            else:
                best = max(quotes, key=lambda q: q.bid)
                price = best.bid

            return RouteDecision(
                primary_exchange=best.exchange,
                secondary_exchanges=[],
                allocation={best.exchange: quantity},
                expected_price=price,
                expected_fees=quantity * price * self._exchanges[best.exchange].fee_rate,
                reason=f"Best {side} price on {best.exchange}",
            )

        elif strategy == RoutingStrategy.SPLIT:
            # Split across exchanges based on available liquidity
            allocation = {}
            total_liquidity = sum(
                q.ask_size if side == "buy" else q.bid_size
                for q in quotes
            )

            if total_liquidity == 0:
                # Fallback to equal split
                per_exchange = quantity / len(quotes)
                allocation = {q.exchange: per_exchange for q in quotes}
            else:
                remaining = quantity
                for q in quotes:
                    liquidity = q.ask_size if side == "buy" else q.bid_size
                    share = min(remaining, liquidity * 0.5)  # Take max 50% of liquidity
                    if share > 0:
                        allocation[q.exchange] = share
                        remaining -= share
                    if remaining <= 0:
                        break

            # Calculate weighted average price
            total_qty = sum(allocation.values())
            avg_price = 0
            total_fees = 0

            for exchange, qty in allocation.items():
                quote = next(q for q in quotes if q.exchange == exchange)
                price = quote.ask if side == "buy" else quote.bid
                avg_price += price * (qty / total_qty)
                total_fees += qty * price * self._exchanges[exchange].fee_rate

            return RouteDecision(
                primary_exchange=list(allocation.keys())[0],
                secondary_exchanges=list(allocation.keys())[1:],
                allocation=allocation,
                expected_price=avg_price,
                expected_fees=total_fees,
                reason=f"Split across {len(allocation)} exchanges",
            )

        elif strategy == RoutingStrategy.PRIMARY:
            # Use primary exchange
            primary = self.primary_exchange
            if primary not in [q.exchange for q in quotes]:
                # Failover to best available
                return self._decide_route(symbol, side, quantity, quotes, RoutingStrategy.BEST_PRICE)

            quote = next(q for q in quotes if q.exchange == primary)
            price = quote.ask if side == "buy" else quote.bid

            return RouteDecision(
                primary_exchange=primary,
                secondary_exchanges=[],
                allocation={primary: quantity},
                expected_price=price,
                expected_fees=quantity * price * self._exchanges[primary].fee_rate,
                reason=f"Primary exchange: {primary}",
            )

        elif strategy == RoutingStrategy.LOWEST_FEE:
            # Route to lowest fee exchange
            best = min(quotes, key=lambda q: self._exchanges[q.exchange].fee_rate)
            price = best.ask if side == "buy" else best.bid

            return RouteDecision(
                primary_exchange=best.exchange,
                secondary_exchanges=[],
                allocation={best.exchange: quantity},
                expected_price=price,
                expected_fees=quantity * price * self._exchanges[best.exchange].fee_rate,
                reason=f"Lowest fee: {best.exchange} ({self._exchanges[best.exchange].fee_rate:.4%})",
            )

        else:
            # Default to best price
            return self._decide_route(symbol, side, quantity, quotes, RoutingStrategy.BEST_PRICE)

    async def execute_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        routing: RoutingStrategy = RoutingStrategy.BEST_PRICE,
        limit_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute an order with smart routing.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Quantity to trade
            routing: Routing strategy
            limit_price: Optional limit price

        Returns:
            ExecutionResult with fill details
        """
        self._order_counter += 1
        order_id = f"multi_{self._order_counter}"
        started_at = datetime.now()
        errors = []

        # Get quotes from all exchanges
        quotes = await self._get_all_quotes(symbol)

        if not quotes:
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                exchange_fills={},
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                errors=["No quotes available from any exchange"],
            )

        # Decide routing
        try:
            route = self._decide_route(symbol, side, quantity, quotes, routing)
        except Exception as e:
            logger.error(f"Routing decision failed: {e}")
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                exchange_fills={},
                started_at=started_at,
                completed_at=datetime.now(),
                success=False,
                errors=[str(e)],
            )

        # Execute on each exchange
        exchange_fills = {}
        total_filled = 0
        total_cost = 0
        total_fees = 0

        tasks = []
        for exchange, qty in route.allocation.items():
            adapter = self._exchanges[exchange]
            tasks.append(self._execute_on_exchange(adapter, symbol, side, qty, limit_price))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (exchange, qty), result in zip(route.allocation.items(), results):
            if isinstance(result, Exception):
                errors.append(f"{exchange}: {str(result)}")
                continue

            if result.get("status") == "filled":
                filled_qty = result.get("filled_quantity", 0)
                fill_price = result.get("price", 0)
                fees = result.get("fees", 0)

                exchange_fills[exchange] = {
                    "quantity": filled_qty,
                    "price": fill_price,
                    "fees": fees,
                    "order_id": result.get("order_id"),
                }

                total_filled += filled_qty
                total_cost += filled_qty * fill_price
                total_fees += fees
            else:
                errors.append(f"{exchange}: Order not filled - {result.get('status')}")

        avg_price = total_cost / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=total_filled,
            average_price=avg_price,
            total_fees=total_fees,
            exchange_fills=exchange_fills,
            started_at=started_at,
            completed_at=datetime.now(),
            success=total_filled > 0,
            errors=errors,
        )

    async def _execute_on_exchange(
        self,
        adapter: ExchangeAdapter,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: Optional[float],
    ) -> Dict:
        """Execute order on a single exchange."""
        order_type = "limit" if limit_price else "market"
        return await adapter.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=limit_price,
            order_type=order_type,
        )

    async def detect_arbitrage(
        self,
        symbol: str,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Detect cross-exchange arbitrage opportunities.

        Args:
            symbol: Trading symbol

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        quotes = await self._get_all_quotes(symbol)

        if len(quotes) < 2:
            return None

        # Find best bid and best ask across exchanges
        best_bid_quote = max(quotes, key=lambda q: q.bid)
        best_ask_quote = min(quotes, key=lambda q: q.ask)

        # Check if arbitrage exists (bid > ask on different exchanges)
        if (
            best_bid_quote.exchange != best_ask_quote.exchange
            and best_bid_quote.bid > best_ask_quote.ask
        ):
            spread = best_bid_quote.bid - best_ask_quote.ask
            spread_bps = (spread / best_ask_quote.ask) * 10000

            if spread_bps < self.min_arb_spread_bps:
                return None

            # Calculate potential profit
            max_qty = min(best_bid_quote.bid_size, best_ask_quote.ask_size)

            # Account for fees
            buy_fees = max_qty * best_ask_quote.ask * self._exchanges[best_ask_quote.exchange].fee_rate
            sell_fees = max_qty * best_bid_quote.bid * self._exchanges[best_bid_quote.exchange].fee_rate
            profit = (spread * max_qty) - buy_fees - sell_fees

            if profit <= 0:
                return None

            return ArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=best_ask_quote.exchange,
                sell_exchange=best_bid_quote.exchange,
                buy_price=best_ask_quote.ask,
                sell_price=best_bid_quote.bid,
                spread_bps=spread_bps,
                max_quantity=max_qty,
                profit_estimate=profit,
                detected_at=datetime.now(),
            )

        return None

    async def get_consolidated_positions(self) -> Dict[str, Dict[str, float]]:
        """
        Get consolidated positions across all exchanges.

        Returns:
            Dict mapping symbol to {exchange: quantity}
        """
        positions = {}

        for name, adapter in self._exchanges.items():
            if self._status.get(name) not in (ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED):
                continue

            try:
                exchange_positions = await adapter.get_positions()
                for pos in exchange_positions:
                    symbol = pos.get("symbol", "")
                    qty = pos.get("quantity", 0)

                    if symbol not in positions:
                        positions[symbol] = {}
                    positions[symbol][name] = qty
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {e}")

        return positions

    async def get_consolidated_balance(self, currency: str) -> Dict[str, float]:
        """
        Get consolidated balance for a currency across exchanges.

        Args:
            currency: Currency symbol (e.g., "USDT", "BTC")

        Returns:
            Dict mapping exchange to balance
        """
        balances = {}

        for name, adapter in self._exchanges.items():
            if self._status.get(name) not in (ExchangeStatus.HEALTHY, ExchangeStatus.DEGRADED):
                continue

            try:
                balance = await adapter.get_balance(currency)
                balances[name] = balance
            except Exception as e:
                logger.error(f"Error getting {currency} balance from {name}: {e}")

        return balances

    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status of all exchanges."""
        status = {}
        for name in self._exchanges:
            latencies = self._latencies.get(name, [])
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            status[name] = {
                "status": self._status.get(name, ExchangeStatus.DISCONNECTED).value,
                "is_primary": name == self.primary_exchange,
                "avg_latency_ms": round(avg_latency, 2),
                "fee_rate": self._exchanges[name].fee_rate,
            }

        return status


def create_multi_exchange_coordinator(
    exchanges: Optional[Dict[str, ExchangeAdapter]] = None,
    primary: Optional[str] = None,
) -> MultiExchangeCoordinator:
    """Factory function to create multi-exchange coordinator."""
    coordinator = MultiExchangeCoordinator(primary_exchange=primary)

    if exchanges:
        for name, adapter in exchanges.items():
            coordinator.add_exchange(name, adapter, is_primary=(name == primary))

    return coordinator
