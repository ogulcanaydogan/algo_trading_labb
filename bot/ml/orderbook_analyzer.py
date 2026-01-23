"""
Order Book and Market Microstructure Analysis

Analyzes:
1. Order book imbalance
2. Bid-ask spread dynamics
3. Large order detection (whale watching)
4. Order flow toxicity (VPIN)
5. Market depth analysis
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot."""

    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    bid_depth: float
    ask_depth: float


@dataclass
class OrderFlowSignal:
    """Signal derived from order flow analysis."""

    timestamp: datetime
    imbalance: float  # -1 (sell pressure) to 1 (buy pressure)
    toxicity: float  # 0 (normal) to 1 (toxic flow)
    whale_activity: float  # 0 to 1
    spread_percentile: float  # 0 to 1
    depth_ratio: float  # bid_depth / ask_depth
    signal: str  # BUY, SELL, NEUTRAL
    confidence: float


class OrderBookAnalyzer:
    """
    Real-time order book analysis for trading signals.
    """

    def __init__(
        self,
        depth_levels: int = 20,
        history_size: int = 1000,
        whale_threshold: float = 0.1,  # 10% of total depth
    ):
        self.depth_levels = depth_levels
        self.history_size = history_size
        self.whale_threshold = whale_threshold

        # History buffers
        self.snapshots: deque = deque(maxlen=history_size)
        self.trades: deque = deque(maxlen=history_size * 10)
        self.imbalance_history: deque = deque(maxlen=history_size)
        self.spread_history: deque = deque(maxlen=history_size)

        # VPIN calculation
        self.volume_buckets: deque = deque(maxlen=50)
        self.bucket_size = 0  # Will be set based on average volume

    def process_orderbook(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: Optional[datetime] = None,
    ) -> OrderBookSnapshot:
        """
        Process new order book data.

        Args:
            bids: List of (price, size) tuples, sorted desc by price
            asks: List of (price, size) tuples, sorted asc by price
            timestamp: Optional timestamp

        Returns:
            OrderBookSnapshot
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate metrics
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0

        # Depth at N levels
        bid_depth = sum(size for _, size in bids[: self.depth_levels])
        ask_depth = sum(size for _, size in asks[: self.depth_levels])

        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            bids=bids[: self.depth_levels],
            asks=asks[: self.depth_levels],
            mid_price=mid_price,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

        self.snapshots.append(snapshot)
        self.spread_history.append(spread)

        return snapshot

    def calculate_imbalance(self, snapshot: Optional[OrderBookSnapshot] = None) -> float:
        """
        Calculate order book imbalance.

        Returns value from -1 (strong sell) to 1 (strong buy).
        """
        if snapshot is None:
            if not self.snapshots:
                return 0.0
            snapshot = self.snapshots[-1]

        total_depth = snapshot.bid_depth + snapshot.ask_depth
        if total_depth == 0:
            return 0.0

        # Simple imbalance
        imbalance = (snapshot.bid_depth - snapshot.ask_depth) / total_depth

        # Weighted imbalance (closer to mid-price = more weight)
        weighted_bid = 0
        weighted_ask = 0

        for i, (price, size) in enumerate(snapshot.bids):
            weight = 1 / (i + 1)
            weighted_bid += size * weight

        for i, (price, size) in enumerate(snapshot.asks):
            weight = 1 / (i + 1)
            weighted_ask += size * weight

        total_weighted = weighted_bid + weighted_ask
        if total_weighted > 0:
            weighted_imbalance = (weighted_bid - weighted_ask) / total_weighted
            # Combine simple and weighted
            imbalance = 0.5 * imbalance + 0.5 * weighted_imbalance

        self.imbalance_history.append(imbalance)
        return imbalance

    def detect_whale_orders(self, snapshot: Optional[OrderBookSnapshot] = None) -> Dict[str, Any]:
        """
        Detect large orders (whale activity).

        Returns dict with whale detection metrics.
        """
        if snapshot is None:
            if not self.snapshots:
                return {"detected": False, "side": None, "size_ratio": 0}
            snapshot = self.snapshots[-1]

        total_depth = snapshot.bid_depth + snapshot.ask_depth
        if total_depth == 0:
            return {"detected": False, "side": None, "size_ratio": 0}

        # Check for large orders on each side
        whale_bids = []
        whale_asks = []

        threshold_size = total_depth * self.whale_threshold

        for price, size in snapshot.bids:
            if size >= threshold_size:
                whale_bids.append((price, size))

        for price, size in snapshot.asks:
            if size >= threshold_size:
                whale_asks.append((price, size))

        # Determine dominant whale side
        whale_bid_total = sum(s for _, s in whale_bids)
        whale_ask_total = sum(s for _, s in whale_asks)

        if whale_bid_total > whale_ask_total and whale_bids:
            return {
                "detected": True,
                "side": "BUY",
                "size_ratio": whale_bid_total / total_depth,
                "orders": whale_bids,
            }
        elif whale_ask_total > whale_bid_total and whale_asks:
            return {
                "detected": True,
                "side": "SELL",
                "size_ratio": whale_ask_total / total_depth,
                "orders": whale_asks,
            }

        return {"detected": False, "side": None, "size_ratio": 0}

    def calculate_vpin(self, trades: List[Dict]) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        Higher VPIN indicates more toxic (informed) order flow.

        Args:
            trades: List of trade dicts with 'price', 'size', 'side'

        Returns:
            VPIN score (0 to 1)
        """
        if not trades:
            return 0.0

        # Classify trades as buy or sell
        buy_volume = 0
        sell_volume = 0

        for trade in trades:
            if trade.get("side") == "buy":
                buy_volume += trade["size"]
            else:
                sell_volume += trade["size"]

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0

        # VPIN = |Buy Volume - Sell Volume| / Total Volume
        vpin = abs(buy_volume - sell_volume) / total_volume

        return vpin

    def calculate_spread_percentile(self) -> float:
        """
        Calculate current spread as percentile of historical spreads.

        High percentile = unusually wide spread (uncertainty/volatility).
        """
        if len(self.spread_history) < 10:
            return 0.5

        current_spread = self.spread_history[-1]
        historical = list(self.spread_history)[:-1]

        percentile = sum(1 for s in historical if s < current_spread) / len(historical)
        return percentile

    def get_signal(self) -> OrderFlowSignal:
        """
        Generate trading signal from order flow analysis.

        Returns:
            OrderFlowSignal with all metrics
        """
        if not self.snapshots:
            return OrderFlowSignal(
                timestamp=datetime.now(),
                imbalance=0,
                toxicity=0,
                whale_activity=0,
                spread_percentile=0.5,
                depth_ratio=1.0,
                signal="NEUTRAL",
                confidence=0,
            )

        snapshot = self.snapshots[-1]

        # Calculate all metrics
        imbalance = self.calculate_imbalance(snapshot)
        whale_info = self.detect_whale_orders(snapshot)
        spread_pct = self.calculate_spread_percentile()

        # Depth ratio
        depth_ratio = snapshot.bid_depth / snapshot.ask_depth if snapshot.ask_depth > 0 else 1.0

        # VPIN from recent trades
        recent_trades = list(self.trades)[-100:] if self.trades else []
        toxicity = self.calculate_vpin(recent_trades) if recent_trades else 0

        # Generate signal
        signal = "NEUTRAL"
        confidence = 0.0

        # Strong imbalance signals
        if imbalance > 0.3:
            signal = "BUY"
            confidence = min(0.9, 0.5 + imbalance)
        elif imbalance < -0.3:
            signal = "SELL"
            confidence = min(0.9, 0.5 + abs(imbalance))

        # Whale activity can amplify or confirm signals
        if whale_info["detected"]:
            if whale_info["side"] == signal:
                confidence = min(0.95, confidence + 0.15)
            elif whale_info["side"] != signal and signal != "NEUTRAL":
                # Conflicting signals - reduce confidence
                confidence *= 0.7

        # High toxicity reduces confidence (informed traders may be front-running)
        if toxicity > 0.6:
            confidence *= 0.8

        # Wide spreads indicate uncertainty
        if spread_pct > 0.8:
            confidence *= 0.85

        return OrderFlowSignal(
            timestamp=datetime.now(),
            imbalance=imbalance,
            toxicity=toxicity,
            whale_activity=whale_info["size_ratio"] if whale_info["detected"] else 0,
            spread_percentile=spread_pct,
            depth_ratio=depth_ratio,
            signal=signal,
            confidence=confidence,
        )

    def get_features(self) -> Dict[str, float]:
        """
        Get order book features for ML model.

        Returns dict of normalized features.
        """
        if not self.snapshots:
            return {
                "ob_imbalance": 0,
                "ob_imbalance_ma": 0,
                "ob_spread_pct": 0.5,
                "ob_depth_ratio": 1.0,
                "ob_whale_activity": 0,
                "ob_toxicity": 0,
                "ob_bid_depth_change": 0,
                "ob_ask_depth_change": 0,
            }

        signal = self.get_signal()

        # Moving average of imbalance
        imbalance_ma = np.mean(list(self.imbalance_history)[-20:]) if self.imbalance_history else 0

        # Depth changes
        if len(self.snapshots) >= 2:
            prev = self.snapshots[-2]
            curr = self.snapshots[-1]
            bid_depth_change = (curr.bid_depth - prev.bid_depth) / (prev.bid_depth + 1e-10)
            ask_depth_change = (curr.ask_depth - prev.ask_depth) / (prev.ask_depth + 1e-10)
        else:
            bid_depth_change = 0
            ask_depth_change = 0

        return {
            "ob_imbalance": signal.imbalance,
            "ob_imbalance_ma": imbalance_ma,
            "ob_spread_pct": signal.spread_percentile,
            "ob_depth_ratio": np.clip(signal.depth_ratio, 0.1, 10) / 10,  # Normalize
            "ob_whale_activity": signal.whale_activity,
            "ob_toxicity": signal.toxicity,
            "ob_bid_depth_change": np.clip(bid_depth_change, -1, 1),
            "ob_ask_depth_change": np.clip(ask_depth_change, -1, 1),
        }


class OptionsFlowAnalyzer:
    """
    Analyzes options flow for directional signals.

    Options flow can predict underlying asset movements:
    - Large call buying = bullish
    - Large put buying = bearish
    - Put/Call ratio extremes = contrarian signals
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.flow_history: deque = deque(maxlen=history_size)
        self.put_call_history: deque = deque(maxlen=100)

    def process_options_trade(
        self,
        symbol: str,
        option_type: str,  # 'call' or 'put'
        strike: float,
        expiry: datetime,
        premium: float,
        size: int,
        side: str,  # 'buy' or 'sell'
        underlying_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Process an options trade.

        Returns analysis of the trade's significance.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate trade characteristics
        days_to_expiry = (expiry - timestamp).days
        moneyness = (
            underlying_price / strike if option_type == "call" else strike / underlying_price
        )

        # Determine if this is significant
        is_sweep = False  # Would need multi-exchange data
        is_unusual = premium * size > 100000  # $100k+ premium

        trade_info = {
            "timestamp": timestamp,
            "symbol": symbol,
            "type": option_type,
            "strike": strike,
            "expiry": expiry,
            "premium": premium,
            "size": size,
            "side": side,
            "underlying": underlying_price,
            "days_to_expiry": days_to_expiry,
            "moneyness": moneyness,
            "notional": premium * size * 100,  # Options are per 100 shares
            "is_unusual": is_unusual,
        }

        self.flow_history.append(trade_info)

        return trade_info

    def calculate_put_call_ratio(self, lookback_hours: int = 24) -> float:
        """
        Calculate put/call ratio from recent flow.

        Returns:
            P/C ratio (>1 = bearish sentiment, <1 = bullish)
        """
        if not self.flow_history:
            return 1.0

        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        call_volume = 0
        put_volume = 0

        for trade in self.flow_history:
            if trade["timestamp"] < cutoff:
                continue

            if trade["type"] == "call":
                call_volume += trade["size"]
            else:
                put_volume += trade["size"]

        if call_volume == 0:
            return 2.0  # Max bearish

        ratio = put_volume / call_volume
        self.put_call_history.append(ratio)

        return ratio

    def get_unusual_activity(self, min_notional: float = 100000) -> List[Dict]:
        """
        Get unusual options activity.

        Returns list of large/unusual trades.
        """
        unusual = []
        cutoff = datetime.now() - timedelta(hours=24)

        for trade in self.flow_history:
            if trade["timestamp"] < cutoff:
                continue

            if trade["notional"] >= min_notional:
                unusual.append(trade)

        # Sort by notional value
        unusual.sort(key=lambda x: x["notional"], reverse=True)

        return unusual[:20]

    def get_signal(self) -> Dict[str, Any]:
        """
        Generate options-based trading signal.

        Returns:
            Dict with signal and supporting metrics
        """
        pc_ratio = self.calculate_put_call_ratio()
        unusual = self.get_unusual_activity()

        # Analyze unusual activity sentiment
        call_unusual = sum(
            t["notional"] for t in unusual if t["type"] == "call" and t["side"] == "buy"
        )
        put_unusual = sum(
            t["notional"] for t in unusual if t["type"] == "put" and t["side"] == "buy"
        )

        # Generate signal
        signal = "NEUTRAL"
        confidence = 0.5

        # Extreme P/C ratio (contrarian)
        if pc_ratio > 1.5:
            # Very bearish sentiment - contrarian bullish
            signal = "BUY"
            confidence = 0.6
        elif pc_ratio < 0.5:
            # Very bullish sentiment - contrarian bearish
            signal = "SELL"
            confidence = 0.6

        # Unusual activity (follow smart money)
        if call_unusual > put_unusual * 2:
            if signal == "BUY":
                confidence += 0.15
            else:
                signal = "BUY"
                confidence = 0.65
        elif put_unusual > call_unusual * 2:
            if signal == "SELL":
                confidence += 0.15
            else:
                signal = "SELL"
                confidence = 0.65

        return {
            "signal": signal,
            "confidence": min(0.9, confidence),
            "put_call_ratio": pc_ratio,
            "call_unusual_notional": call_unusual,
            "put_unusual_notional": put_unusual,
            "unusual_trades": len(unusual),
        }

    def get_features(self) -> Dict[str, float]:
        """
        Get options flow features for ML model.
        """
        signal = self.get_signal()
        pc_ratio = signal["put_call_ratio"]

        # Normalize P/C ratio (typical range 0.5-2.0)
        pc_normalized = np.clip((pc_ratio - 1) / 1, -1, 1)

        # P/C ratio percentile
        if len(self.put_call_history) > 10:
            pc_percentile = sum(1 for r in self.put_call_history if r < pc_ratio) / len(
                self.put_call_history
            )
        else:
            pc_percentile = 0.5

        return {
            "opt_put_call_ratio": pc_normalized,
            "opt_pc_percentile": pc_percentile,
            "opt_unusual_call": np.log1p(signal["call_unusual_notional"]) / 20,  # Normalize
            "opt_unusual_put": np.log1p(signal["put_unusual_notional"]) / 20,
            "opt_flow_signal": 1
            if signal["signal"] == "BUY"
            else (-1 if signal["signal"] == "SELL" else 0),
        }


# Global instances
_orderbook_analyzers: Dict[str, OrderBookAnalyzer] = {}
_options_analyzer: Optional[OptionsFlowAnalyzer] = None


def get_orderbook_analyzer(symbol: str) -> OrderBookAnalyzer:
    """Get or create order book analyzer for symbol."""
    if symbol not in _orderbook_analyzers:
        _orderbook_analyzers[symbol] = OrderBookAnalyzer()
    return _orderbook_analyzers[symbol]


def get_options_analyzer() -> OptionsFlowAnalyzer:
    """Get or create options flow analyzer."""
    global _options_analyzer
    if _options_analyzer is None:
        _options_analyzer = OptionsFlowAnalyzer()
    return _options_analyzer
