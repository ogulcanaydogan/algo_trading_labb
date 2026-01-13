"""
Order Book Visualization Module.

Provides order book data processing and visualization
including market depth, liquidity analysis, and spread tracking.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    quantity: float
    total: float  # Cumulative quantity
    percentage: float  # Percentage of total book


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    best_bid: float
    best_ask: float
    spread: float
    spread_percent: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    imbalance: float  # Positive = more bids, negative = more asks


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    bid_depth_1pct: float  # Volume within 1% of best bid
    ask_depth_1pct: float
    bid_depth_5pct: float
    ask_depth_5pct: float
    weighted_bid_price: float
    weighted_ask_price: float
    market_impact_buy: float  # Estimated slippage for market buy
    market_impact_sell: float
    liquidity_score: float  # 0-100


class OrderBookAnalyzer:
    """
    Analyzes order book data for market depth and liquidity.

    Provides real-time order book visualization data and
    liquidity metrics for trading decisions.
    """

    def __init__(self, depth: int = 20):
        """
        Initialize order book analyzer.

        Args:
            depth: Number of price levels to track
        """
        self.depth = depth
        self._snapshots: List[OrderBookSnapshot] = []
        self._spread_history: List[Tuple[datetime, float]] = []

    async def fetch_orderbook(
        self,
        exchange,  # ccxt exchange instance
        symbol: str,
    ) -> Optional[OrderBookSnapshot]:
        """
        Fetch and process order book from exchange.

        Args:
            exchange: CCXT exchange instance
            symbol: Trading symbol

        Returns:
            OrderBookSnapshot or None on error
        """
        try:
            orderbook = await exchange.fetch_order_book(symbol, limit=self.depth)
            return self._process_orderbook(orderbook, symbol)

        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

    def process_raw_orderbook(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        symbol: str,
    ) -> OrderBookSnapshot:
        """
        Process raw order book data.

        Args:
            bids: List of [price, quantity] bids
            asks: List of [price, quantity] asks
            symbol: Trading symbol

        Returns:
            OrderBookSnapshot
        """
        return self._process_orderbook({"bids": bids, "asks": asks}, symbol)

    def _process_orderbook(
        self,
        raw_book: Dict[str, List],
        symbol: str,
    ) -> OrderBookSnapshot:
        """Process raw order book into structured snapshot."""
        timestamp = datetime.now()

        raw_bids = raw_book.get("bids", [])[:self.depth]
        raw_asks = raw_book.get("asks", [])[:self.depth]

        # Calculate totals
        total_bid_vol = sum(b[1] for b in raw_bids)
        total_ask_vol = sum(a[1] for a in raw_asks)

        # Process bids (sorted high to low)
        bids = []
        cumulative = 0
        for price, qty in raw_bids:
            cumulative += qty
            bids.append(OrderBookLevel(
                price=float(price),
                quantity=float(qty),
                total=cumulative,
                percentage=(qty / total_bid_vol * 100) if total_bid_vol > 0 else 0,
            ))

        # Process asks (sorted low to high)
        asks = []
        cumulative = 0
        for price, qty in raw_asks:
            cumulative += qty
            asks.append(OrderBookLevel(
                price=float(price),
                quantity=float(qty),
                total=cumulative,
                percentage=(qty / total_ask_vol * 100) if total_ask_vol > 0 else 0,
            ))

        # Best prices
        best_bid = float(raw_bids[0][0]) if raw_bids else 0
        best_ask = float(raw_asks[0][0]) if raw_asks else 0

        # Spread
        spread = best_ask - best_bid if best_bid > 0 else 0
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 else 0

        # Imbalance
        total_vol = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0

        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_percent=spread_pct,
            mid_price=mid_price,
            total_bid_volume=total_bid_vol,
            total_ask_volume=total_ask_vol,
            imbalance=imbalance,
        )

        self._snapshots.append(snapshot)
        self._spread_history.append((timestamp, spread_pct))

        # Keep only recent data
        if len(self._snapshots) > 1000:
            self._snapshots = self._snapshots[-500:]
        if len(self._spread_history) > 1000:
            self._spread_history = self._spread_history[-500:]

        return snapshot

    def calculate_liquidity_metrics(
        self,
        snapshot: OrderBookSnapshot,
        trade_size: float = 1.0,  # Size in base currency
    ) -> LiquidityMetrics:
        """
        Calculate liquidity metrics from order book snapshot.

        Args:
            snapshot: OrderBookSnapshot to analyze
            trade_size: Size of hypothetical trade for impact calculation

        Returns:
            LiquidityMetrics
        """
        # Depth within price ranges
        bid_depth_1pct = 0.0
        bid_depth_5pct = 0.0
        ask_depth_1pct = 0.0
        ask_depth_5pct = 0.0

        threshold_1pct = snapshot.mid_price * 0.01
        threshold_5pct = snapshot.mid_price * 0.05

        # Bid depth
        for level in snapshot.bids:
            distance = snapshot.best_bid - level.price
            if distance <= threshold_1pct:
                bid_depth_1pct += level.quantity
            if distance <= threshold_5pct:
                bid_depth_5pct += level.quantity

        # Ask depth
        for level in snapshot.asks:
            distance = level.price - snapshot.best_ask
            if distance <= threshold_1pct:
                ask_depth_1pct += level.quantity
            if distance <= threshold_5pct:
                ask_depth_5pct += level.quantity

        # Volume-weighted average prices
        weighted_bid = self._calculate_vwap(snapshot.bids, trade_size)
        weighted_ask = self._calculate_vwap(snapshot.asks, trade_size)

        # Market impact (slippage for trade_size)
        impact_buy = ((weighted_ask - snapshot.best_ask) / snapshot.best_ask * 100) if snapshot.best_ask > 0 else 0
        impact_sell = ((snapshot.best_bid - weighted_bid) / snapshot.best_bid * 100) if snapshot.best_bid > 0 else 0

        # Liquidity score (0-100)
        liquidity_score = self._calculate_liquidity_score(
            spread_pct=snapshot.spread_percent,
            depth_1pct=min(bid_depth_1pct, ask_depth_1pct),
            imbalance=abs(snapshot.imbalance),
        )

        return LiquidityMetrics(
            bid_depth_1pct=round(bid_depth_1pct, 4),
            ask_depth_1pct=round(ask_depth_1pct, 4),
            bid_depth_5pct=round(bid_depth_5pct, 4),
            ask_depth_5pct=round(ask_depth_5pct, 4),
            weighted_bid_price=round(weighted_bid, 8),
            weighted_ask_price=round(weighted_ask, 8),
            market_impact_buy=round(impact_buy, 4),
            market_impact_sell=round(impact_sell, 4),
            liquidity_score=round(liquidity_score, 1),
        )

    def _calculate_vwap(self, levels: List[OrderBookLevel], size: float) -> float:
        """Calculate volume-weighted average price for given size."""
        if not levels:
            return 0.0

        remaining = size
        total_cost = 0.0

        for level in levels:
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            remaining -= fill_qty

            if remaining <= 0:
                break

        filled = size - max(0, remaining)
        return total_cost / filled if filled > 0 else levels[0].price

    def _calculate_liquidity_score(
        self,
        spread_pct: float,
        depth_1pct: float,
        imbalance: float,
    ) -> float:
        """Calculate overall liquidity score."""
        # Spread component (lower is better)
        spread_score = max(0, 100 - spread_pct * 100)  # 1% spread = 0 score

        # Depth component
        depth_score = min(100, depth_1pct * 10)  # 10 units = full score

        # Imbalance component (closer to 0 is better)
        imbalance_score = (1 - abs(imbalance)) * 100

        # Weighted combination
        return spread_score * 0.4 + depth_score * 0.4 + imbalance_score * 0.2

    def get_depth_chart_data(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Get data formatted for depth chart visualization.

        Returns:
            Dictionary with bid/ask depth data for charting
        """
        bid_data = []
        ask_data = []

        # Bids (cumulative from best bid down)
        for level in snapshot.bids:
            bid_data.append({
                "price": level.price,
                "quantity": level.quantity,
                "total": level.total,
            })

        # Asks (cumulative from best ask up)
        for level in snapshot.asks:
            ask_data.append({
                "price": level.price,
                "quantity": level.quantity,
                "total": level.total,
            })

        return {
            "bids": bid_data,
            "asks": ask_data,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "spread_percent": round(snapshot.spread_percent, 4),
        }

    def get_heatmap_data(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Get data formatted for order book heatmap visualization.

        Returns:
            Dictionary with price levels and quantities for heatmap
        """
        levels = []

        # Add bids
        for level in reversed(snapshot.bids):
            levels.append({
                "price": level.price,
                "quantity": level.quantity,
                "side": "bid",
                "intensity": level.percentage / 100,
            })

        # Add asks
        for level in snapshot.asks:
            levels.append({
                "price": level.price,
                "quantity": level.quantity,
                "side": "ask",
                "intensity": level.percentage / 100,
            })

        return {
            "levels": levels,
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
        }

    def get_spread_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical spread data."""
        return [
            {"timestamp": ts.isoformat(), "spread_percent": round(spread, 4)}
            for ts, spread in self._spread_history[-limit:]
        ]

    def get_imbalance_signal(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Generate trading signal based on order book imbalance.

        Returns:
            Signal with direction and strength
        """
        imbalance = snapshot.imbalance

        if imbalance > 0.3:
            signal = "BUY"
            strength = min(1.0, imbalance)
        elif imbalance < -0.3:
            signal = "SELL"
            strength = min(1.0, abs(imbalance))
        else:
            signal = "NEUTRAL"
            strength = 0.0

        return {
            "signal": signal,
            "strength": round(strength, 2),
            "imbalance": round(imbalance, 3),
            "bid_volume": snapshot.total_bid_volume,
            "ask_volume": snapshot.total_ask_volume,
            "ratio": round(snapshot.total_bid_volume / snapshot.total_ask_volume, 2) if snapshot.total_ask_volume > 0 else 0,
        }

    def to_api_response(
        self,
        snapshot: OrderBookSnapshot,
        liquidity: Optional[LiquidityMetrics] = None,
    ) -> Dict[str, Any]:
        """Convert to API response format."""
        if liquidity is None:
            liquidity = self.calculate_liquidity_metrics(snapshot)

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "symbol": snapshot.symbol,
            "summary": {
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "mid_price": snapshot.mid_price,
                "spread": snapshot.spread,
                "spread_percent": round(snapshot.spread_percent, 4),
                "imbalance": round(snapshot.imbalance, 3),
            },
            "volume": {
                "total_bid": round(snapshot.total_bid_volume, 4),
                "total_ask": round(snapshot.total_ask_volume, 4),
            },
            "liquidity": {
                "bid_depth_1pct": liquidity.bid_depth_1pct,
                "ask_depth_1pct": liquidity.ask_depth_1pct,
                "bid_depth_5pct": liquidity.bid_depth_5pct,
                "ask_depth_5pct": liquidity.ask_depth_5pct,
                "market_impact_buy": liquidity.market_impact_buy,
                "market_impact_sell": liquidity.market_impact_sell,
                "liquidity_score": liquidity.liquidity_score,
            },
            "depth_chart": self.get_depth_chart_data(snapshot),
            "signal": self.get_imbalance_signal(snapshot),
            "spread_history": self.get_spread_history(50),
        }
