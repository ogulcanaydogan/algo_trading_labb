"""
Smart Order Router - Best execution across multiple venues.

Routes orders to optimal exchange/broker based on price, liquidity,
fees, and latency considerations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Types of trading venues."""

    EXCHANGE = "exchange"
    BROKER = "broker"
    DARK_POOL = "dark_pool"
    ECN = "ecn"


class RoutingStrategy(Enum):
    """Order routing strategies."""

    BEST_PRICE = "best_price"
    LOWEST_FEE = "lowest_fee"
    FASTEST = "fastest"
    BEST_LIQUIDITY = "best_liquidity"
    SMART = "smart"  # Balanced across all factors


@dataclass
class VenueQuote:
    """Quote from a trading venue."""

    venue_id: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2
        return (self.spread / mid) * 10000 if mid > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "venue_id": self.venue_id,
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "spread_bps": round(self.spread_bps, 2),
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Venue:
    """Trading venue configuration."""

    venue_id: str
    name: str
    venue_type: VenueType
    maker_fee: float  # Fee for providing liquidity
    taker_fee: float  # Fee for taking liquidity
    min_order_size: float
    max_order_size: float
    supported_symbols: List[str]
    avg_latency_ms: float
    is_active: bool = True
    priority: int = 0  # Higher = preferred
    fill_rate: float = 0.95  # Historical fill rate

    def to_dict(self) -> Dict:
        return {
            "venue_id": self.venue_id,
            "name": self.name,
            "venue_type": self.venue_type.value,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "min_order_size": self.min_order_size,
            "max_order_size": self.max_order_size,
            "avg_latency_ms": self.avg_latency_ms,
            "is_active": self.is_active,
            "priority": self.priority,
            "fill_rate": self.fill_rate,
        }


@dataclass
class RouteDecision:
    """Routing decision for an order."""

    venue_id: str
    venue_name: str
    expected_price: float
    expected_fee: float
    expected_total_cost: float
    liquidity_available: float
    confidence: float
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "expected_price": round(self.expected_price, 6),
            "expected_fee": round(self.expected_fee, 6),
            "expected_total_cost": round(self.expected_total_cost, 6),
            "liquidity_available": self.liquidity_available,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "alternatives": self.alternatives[:3],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RoutingConfig:
    """Smart router configuration."""

    # Strategy
    default_strategy: RoutingStrategy = RoutingStrategy.SMART

    # Weights for smart routing
    price_weight: float = 0.4
    fee_weight: float = 0.2
    liquidity_weight: float = 0.25
    latency_weight: float = 0.15

    # Thresholds
    max_spread_bps: float = 50  # Max acceptable spread
    min_liquidity_ratio: float = 1.5  # Min liquidity vs order size
    max_latency_ms: float = 500

    # Split orders
    enable_order_splitting: bool = True
    min_split_venues: int = 2
    max_split_venues: int = 5


class SmartOrderRouter:
    """
    Route orders to optimal execution venue.

    Features:
    - Multi-venue price comparison
    - Fee optimization
    - Liquidity analysis
    - Order splitting across venues
    - Latency-aware routing
    """

    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self._venues: Dict[str, Venue] = {}
        self._quote_callbacks: Dict[str, Callable] = {}
        self._order_callbacks: Dict[str, Callable] = {}
        self._quote_cache: Dict[str, Dict[str, VenueQuote]] = {}
        self._execution_history: List[Dict] = []

    def register_venue(
        self,
        venue: Venue,
        quote_callback: Optional[Callable] = None,
        order_callback: Optional[Callable] = None,
    ):
        """
        Register a trading venue.

        Args:
            venue: Venue configuration
            quote_callback: Async function to get quotes
            order_callback: Async function to submit orders
        """
        self._venues[venue.venue_id] = venue

        if quote_callback:
            self._quote_callbacks[venue.venue_id] = quote_callback
        if order_callback:
            self._order_callbacks[venue.venue_id] = order_callback

        logger.info(f"Registered venue: {venue.name} ({venue.venue_id})")

    def unregister_venue(self, venue_id: str):
        """Remove a venue."""
        self._venues.pop(venue_id, None)
        self._quote_callbacks.pop(venue_id, None)
        self._order_callbacks.pop(venue_id, None)

    async def get_quotes(self, symbol: str) -> Dict[str, VenueQuote]:
        """
        Get quotes from all venues for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict mapping venue_id to VenueQuote
        """
        quotes = {}

        # Fetch quotes in parallel
        tasks = []
        venue_ids = []

        for venue_id, venue in self._venues.items():
            if not venue.is_active:
                continue
            if symbol not in venue.supported_symbols and "*" not in venue.supported_symbols:
                continue
            if venue_id in self._quote_callbacks:
                tasks.append(self._fetch_quote(venue_id, symbol))
                venue_ids.append(venue_id)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for venue_id, result in zip(venue_ids, results):
                if isinstance(result, VenueQuote):
                    quotes[venue_id] = result
                elif isinstance(result, Exception):
                    logger.warning(f"Quote fetch error for {venue_id}: {result}")

        # Update cache
        self._quote_cache[symbol] = quotes

        return quotes

    async def _fetch_quote(self, venue_id: str, symbol: str) -> VenueQuote:
        """Fetch quote from a single venue."""
        callback = self._quote_callbacks[venue_id]
        start = datetime.now()

        result = await callback(symbol)

        latency = (datetime.now() - start).total_seconds() * 1000

        return VenueQuote(
            venue_id=venue_id,
            symbol=symbol,
            bid=result.get("bid", 0),
            ask=result.get("ask", 0),
            bid_size=result.get("bid_size", 0),
            ask_size=result.get("ask_size", 0),
            latency_ms=latency,
        )

    def route_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        strategy: Optional[RoutingStrategy] = None,
        quotes: Optional[Dict[str, VenueQuote]] = None,
    ) -> RouteDecision:
        """
        Determine optimal route for an order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            strategy: Routing strategy (default: config default)
            quotes: Pre-fetched quotes (optional)

        Returns:
            RouteDecision with optimal venue
        """
        strategy = strategy or self.config.default_strategy
        quotes = quotes or self._quote_cache.get(symbol, {})

        if not quotes:
            raise ValueError(f"No quotes available for {symbol}")

        # Score each venue
        venue_scores = []

        for venue_id, quote in quotes.items():
            venue = self._venues.get(venue_id)
            if not venue or not venue.is_active:
                continue

            score = self._score_venue(venue, quote, side, quantity, strategy)

            if score is not None:
                venue_scores.append((venue_id, score, quote))

        if not venue_scores:
            raise ValueError(f"No suitable venues for {symbol} {side} {quantity}")

        # Sort by score (descending)
        venue_scores.sort(key=lambda x: x[1]["total_score"], reverse=True)

        # Select best venue
        best_venue_id, best_score, best_quote = venue_scores[0]
        best_venue = self._venues[best_venue_id]

        # Calculate expected costs
        price = best_quote.ask if side == "buy" else best_quote.bid
        fee = quantity * price * best_venue.taker_fee
        total_cost = quantity * price + fee if side == "buy" else quantity * price - fee

        # Build alternatives
        alternatives = []
        for venue_id, score, quote in venue_scores[1:4]:
            venue = self._venues[venue_id]
            alt_price = quote.ask if side == "buy" else quote.bid
            alternatives.append(
                {
                    "venue_id": venue_id,
                    "venue_name": venue.name,
                    "price": alt_price,
                    "score": score["total_score"],
                }
            )

        return RouteDecision(
            venue_id=best_venue_id,
            venue_name=best_venue.name,
            expected_price=price,
            expected_fee=fee,
            expected_total_cost=total_cost,
            liquidity_available=best_quote.ask_size if side == "buy" else best_quote.bid_size,
            confidence=min(1.0, best_score["total_score"]),
            reasoning=best_score["reasoning"],
            alternatives=alternatives,
        )

    def _score_venue(
        self,
        venue: Venue,
        quote: VenueQuote,
        side: str,
        quantity: float,
        strategy: RoutingStrategy,
    ) -> Optional[Dict[str, Any]]:
        """Score a venue for routing."""
        reasoning = []

        # Check basic requirements
        if quantity < venue.min_order_size:
            return None
        if quantity > venue.max_order_size:
            return None
        if quote.spread_bps > self.config.max_spread_bps:
            return None

        # Get relevant price and size
        if side == "buy":
            price = quote.ask
            available = quote.ask_size
        else:
            price = quote.bid
            available = quote.bid_size

        if price <= 0:
            return None

        # Check liquidity
        liquidity_ratio = available / quantity if quantity > 0 else 0
        if liquidity_ratio < self.config.min_liquidity_ratio:
            reasoning.append(f"Low liquidity: {liquidity_ratio:.2f}x")

        # Calculate scores
        scores = {}

        # Price score (normalized, inverted for buy)
        # Better price = higher score
        prices_in_cache = [
            q.ask if side == "buy" else q.bid
            for q in self._quote_cache.get(quote.symbol, {}).values()
            if (q.ask if side == "buy" else q.bid) > 0
        ]

        if prices_in_cache:
            best_price = min(prices_in_cache) if side == "buy" else max(prices_in_cache)
            worst_price = max(prices_in_cache) if side == "buy" else min(prices_in_cache)

            if best_price != worst_price:
                if side == "buy":
                    scores["price"] = (worst_price - price) / (worst_price - best_price)
                else:
                    scores["price"] = (price - worst_price) / (best_price - worst_price)
            else:
                scores["price"] = 1.0
        else:
            scores["price"] = 0.5

        # Fee score (lower = better)
        fee_rate = venue.taker_fee
        max_fee = 0.003  # 0.3% as max reference
        scores["fee"] = max(0, 1 - fee_rate / max_fee)

        # Liquidity score
        scores["liquidity"] = min(1.0, liquidity_ratio / 3)  # Cap at 3x

        # Latency score
        if quote.latency_ms > 0:
            scores["latency"] = max(0, 1 - quote.latency_ms / self.config.max_latency_ms)
        else:
            scores["latency"] = 0.5

        # Venue priority bonus
        priority_bonus = venue.priority * 0.05

        # Fill rate bonus
        fill_bonus = (venue.fill_rate - 0.9) * 0.5  # Bonus for fill rate > 90%

        # Calculate total score based on strategy
        if strategy == RoutingStrategy.BEST_PRICE:
            total = scores["price"]
            reasoning.append(f"Price priority: {price:.6f}")

        elif strategy == RoutingStrategy.LOWEST_FEE:
            total = scores["fee"]
            reasoning.append(f"Fee priority: {venue.taker_fee:.4%}")

        elif strategy == RoutingStrategy.FASTEST:
            total = scores["latency"]
            reasoning.append(f"Latency priority: {quote.latency_ms:.0f}ms")

        elif strategy == RoutingStrategy.BEST_LIQUIDITY:
            total = scores["liquidity"]
            reasoning.append(f"Liquidity priority: {liquidity_ratio:.2f}x")

        else:  # SMART
            total = (
                scores["price"] * self.config.price_weight
                + scores["fee"] * self.config.fee_weight
                + scores["liquidity"] * self.config.liquidity_weight
                + scores["latency"] * self.config.latency_weight
            )
            reasoning.append(
                f"Balanced: price={scores['price']:.2f}, "
                f"fee={scores['fee']:.2f}, liq={scores['liquidity']:.2f}"
            )

        total += priority_bonus + fill_bonus

        return {
            "total_score": total,
            "component_scores": scores,
            "reasoning": reasoning,
        }

    def split_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        quotes: Optional[Dict[str, VenueQuote]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Split order across multiple venues for better execution.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Total order quantity
            quotes: Pre-fetched quotes

        Returns:
            List of (venue_id, quantity) tuples
        """
        if not self.config.enable_order_splitting:
            decision = self.route_order(symbol, side, quantity, quotes=quotes)
            return [(decision.venue_id, quantity)]

        quotes = quotes or self._quote_cache.get(symbol, {})

        # Get available liquidity at each venue
        venue_liquidity = []
        for venue_id, quote in quotes.items():
            venue = self._venues.get(venue_id)
            if not venue or not venue.is_active:
                continue

            available = quote.ask_size if side == "buy" else quote.bid_size
            price = quote.ask if side == "buy" else quote.bid

            if available > 0 and price > 0:
                venue_liquidity.append(
                    {
                        "venue_id": venue_id,
                        "available": available,
                        "price": price,
                        "fee": venue.taker_fee,
                    }
                )

        if not venue_liquidity:
            raise ValueError(f"No liquidity available for {symbol}")

        # Sort by effective price (price + fee)
        if side == "buy":
            venue_liquidity.sort(key=lambda x: x["price"] * (1 + x["fee"]))
        else:
            venue_liquidity.sort(key=lambda x: x["price"] * (1 - x["fee"]), reverse=True)

        # Allocate quantity
        splits = []
        remaining = quantity

        for venue in venue_liquidity[: self.config.max_split_venues]:
            if remaining <= 0:
                break

            # Take up to available liquidity
            take = min(remaining, venue["available"])
            splits.append((venue["venue_id"], take))
            remaining -= take

        # If still remaining, add to best venue
        if remaining > 0 and splits:
            venue_id, qty = splits[0]
            splits[0] = (venue_id, qty + remaining)

        return splits

    async def execute_routed_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        strategy: Optional[RoutingStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Route and execute an order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            strategy: Routing strategy

        Returns:
            Execution result
        """
        # Get fresh quotes
        quotes = await self.get_quotes(symbol)

        # Route order
        decision = self.route_order(symbol, side, quantity, strategy, quotes)

        # Execute
        callback = self._order_callbacks.get(decision.venue_id)
        if not callback:
            raise ValueError(f"No order callback for venue {decision.venue_id}")

        result = await callback(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=decision.expected_price,
        )

        # Record execution
        execution = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "venue_id": decision.venue_id,
            "expected_price": decision.expected_price,
            "actual_price": result.get("fill_price", decision.expected_price),
            "timestamp": datetime.now().isoformat(),
        }
        self._execution_history.append(execution)

        return {
            "decision": decision.to_dict(),
            "execution": result,
        }

    def get_venue_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all venues."""
        stats = []

        for venue_id, venue in self._venues.items():
            # Calculate execution stats
            venue_executions = [e for e in self._execution_history if e["venue_id"] == venue_id]

            if venue_executions:
                slippage = sum(
                    abs(e["actual_price"] - e["expected_price"]) / e["expected_price"]
                    for e in venue_executions
                ) / len(venue_executions)
            else:
                slippage = 0

            stats.append(
                {
                    "venue": venue.to_dict(),
                    "executions": len(venue_executions),
                    "avg_slippage_pct": slippage * 100,
                }
            )

        return stats


def create_smart_router(config: Optional[RoutingConfig] = None) -> SmartOrderRouter:
    """Factory function to create smart order router."""
    return SmartOrderRouter(config=config)
