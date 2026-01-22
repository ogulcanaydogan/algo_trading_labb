"""
Order Flow Strategy - Trading based on market microstructure.

Analyzes order flow imbalances, trade aggression, and volume
patterns to generate high-frequency trading signals.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    price: float
    quantity: float
    side: Literal["buy", "sell"]
    timestamp: datetime
    is_aggressive: bool = True  # Taker order

    def to_dict(self) -> Dict:
        return {
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
            "is_aggressive": self.is_aggressive,
        }


@dataclass
class OrderbookLevel:
    """Single orderbook level."""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot."""
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price == 0:
            return 0
        return (self.spread / self.mid_price) * 10000


@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics."""
    buy_volume: float
    sell_volume: float
    net_volume: float
    volume_imbalance: float  # -1 to 1
    trade_count: int
    avg_trade_size: float
    large_trade_ratio: float
    aggressive_ratio: float
    vwap: float
    momentum: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "buy_volume": round(self.buy_volume, 4),
            "sell_volume": round(self.sell_volume, 4),
            "net_volume": round(self.net_volume, 4),
            "volume_imbalance": round(self.volume_imbalance, 4),
            "trade_count": self.trade_count,
            "avg_trade_size": round(self.avg_trade_size, 4),
            "large_trade_ratio": round(self.large_trade_ratio, 4),
            "aggressive_ratio": round(self.aggressive_ratio, 4),
            "vwap": round(self.vwap, 4),
            "momentum": round(self.momentum, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderFlowSignal:
    """Order flow trading signal."""
    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    signal_type: str  # "imbalance", "absorption", "iceberg", "sweep"
    metrics: OrderFlowMetrics
    reasoning: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "signal_type": self.signal_type,
            "metrics": self.metrics.to_dict(),
            "reasoning": self.reasoning,
            "entry_price": round(self.entry_price, 4) if self.entry_price else None,
            "stop_loss": round(self.stop_loss, 4) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 4) if self.take_profit else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderFlowConfig:
    """Order flow strategy configuration."""
    # Trade buffer settings
    trade_buffer_size: int = 1000
    analysis_window_seconds: int = 60

    # Volume imbalance thresholds
    imbalance_threshold: float = 0.3  # 30% imbalance to signal
    strong_imbalance_threshold: float = 0.5

    # Large trade detection
    large_trade_multiplier: float = 5.0  # X times avg size

    # Absorption detection
    absorption_volume_ratio: float = 3.0
    absorption_price_tolerance: float = 0.001  # 0.1%

    # Signal confidence requirements
    min_trades_for_signal: int = 20
    min_confidence: float = 0.5

    # Risk management
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.5


class OrderFlowStrategy:
    """
    Order flow based trading strategy.

    Signals:
    1. Volume Imbalance - More buying/selling pressure
    2. Absorption - Large orders absorbing opposite flow
    3. Iceberg Detection - Hidden large orders
    4. Sweep Detection - Aggressive market taking
    """

    def __init__(self, config: Optional[OrderFlowConfig] = None):
        self.config = config or OrderFlowConfig()
        self._trade_buffer: Dict[str, Deque[Trade]] = {}
        self._orderbook_history: Dict[str, Deque[OrderbookSnapshot]] = {}
        self._atr: Dict[str, float] = {}

    def add_trade(self, symbol: str, trade: Trade):
        """Add trade to buffer."""
        if symbol not in self._trade_buffer:
            self._trade_buffer[symbol] = deque(maxlen=self.config.trade_buffer_size)
        self._trade_buffer[symbol].append(trade)

    def add_orderbook(self, symbol: str, orderbook: OrderbookSnapshot):
        """Add orderbook snapshot."""
        if symbol not in self._orderbook_history:
            self._orderbook_history[symbol] = deque(maxlen=100)
        self._orderbook_history[symbol].append(orderbook)

    def set_atr(self, symbol: str, atr: float):
        """Set ATR for position sizing."""
        self._atr[symbol] = atr

    def analyze_order_flow(
        self,
        symbol: str,
        window_seconds: Optional[int] = None,
    ) -> OrderFlowMetrics:
        """
        Analyze order flow for a symbol.

        Args:
            symbol: Trading symbol
            window_seconds: Analysis window (default: config value)

        Returns:
            OrderFlowMetrics with analysis results
        """
        window = window_seconds or self.config.analysis_window_seconds
        cutoff = datetime.now() - timedelta(seconds=window)

        trades = self._trade_buffer.get(symbol, deque())
        recent_trades = [t for t in trades if t.timestamp > cutoff]

        if not recent_trades:
            return OrderFlowMetrics(
                buy_volume=0,
                sell_volume=0,
                net_volume=0,
                volume_imbalance=0,
                trade_count=0,
                avg_trade_size=0,
                large_trade_ratio=0,
                aggressive_ratio=0,
                vwap=0,
                momentum=0,
            )

        # Calculate volumes
        buy_volume = sum(t.quantity for t in recent_trades if t.side == "buy")
        sell_volume = sum(t.quantity for t in recent_trades if t.side == "sell")
        total_volume = buy_volume + sell_volume
        net_volume = buy_volume - sell_volume

        # Volume imbalance (-1 to 1)
        if total_volume > 0:
            volume_imbalance = net_volume / total_volume
        else:
            volume_imbalance = 0

        # Trade statistics
        trade_count = len(recent_trades)
        quantities = [t.quantity for t in recent_trades]
        avg_trade_size = np.mean(quantities) if quantities else 0

        # Large trade detection
        large_threshold = avg_trade_size * self.config.large_trade_multiplier
        large_trades = [t for t in recent_trades if t.quantity > large_threshold]
        large_trade_ratio = len(large_trades) / trade_count if trade_count > 0 else 0

        # Aggressive trade ratio
        aggressive_trades = [t for t in recent_trades if t.is_aggressive]
        aggressive_ratio = len(aggressive_trades) / trade_count if trade_count > 0 else 0

        # VWAP calculation
        if total_volume > 0:
            vwap = sum(t.price * t.quantity for t in recent_trades) / total_volume
        else:
            vwap = recent_trades[-1].price if recent_trades else 0

        # Momentum (price change direction weighted by volume)
        if len(recent_trades) >= 2:
            price_changes = []
            for i in range(1, len(recent_trades)):
                change = recent_trades[i].price - recent_trades[i-1].price
                weight = recent_trades[i].quantity
                price_changes.append(change * weight)
            momentum = np.sum(price_changes) / total_volume if total_volume > 0 else 0
        else:
            momentum = 0

        return OrderFlowMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_volume=net_volume,
            volume_imbalance=volume_imbalance,
            trade_count=trade_count,
            avg_trade_size=avg_trade_size,
            large_trade_ratio=large_trade_ratio,
            aggressive_ratio=aggressive_ratio,
            vwap=vwap,
            momentum=momentum,
        )

    def detect_absorption(self, symbol: str) -> Optional[Dict]:
        """
        Detect absorption patterns (large orders absorbing flow).

        Returns:
            Dict with absorption details or None
        """
        orderbooks = self._orderbook_history.get(symbol, deque())

        if len(orderbooks) < 5:
            return None

        recent_books = list(orderbooks)[-10:]

        # Check bid absorption (price stable despite selling)
        bid_prices = [ob.best_bid for ob in recent_books]
        ask_prices = [ob.best_ask for ob in recent_books]

        # Get trade metrics
        metrics = self.analyze_order_flow(symbol)

        # Bid absorption: heavy selling but bid holds
        if metrics.volume_imbalance < -self.config.imbalance_threshold:
            price_stability = (max(bid_prices) - min(bid_prices)) / np.mean(bid_prices)
            if price_stability < self.config.absorption_price_tolerance:
                return {
                    "type": "bid_absorption",
                    "level": recent_books[-1].best_bid,
                    "sell_volume": metrics.sell_volume,
                    "stability": 1 - price_stability,
                }

        # Ask absorption: heavy buying but ask holds
        if metrics.volume_imbalance > self.config.imbalance_threshold:
            price_stability = (max(ask_prices) - min(ask_prices)) / np.mean(ask_prices)
            if price_stability < self.config.absorption_price_tolerance:
                return {
                    "type": "ask_absorption",
                    "level": recent_books[-1].best_ask,
                    "buy_volume": metrics.buy_volume,
                    "stability": 1 - price_stability,
                }

        return None

    def detect_sweep(self, symbol: str) -> Optional[Dict]:
        """
        Detect sweep patterns (aggressive clearing of multiple levels).

        Returns:
            Dict with sweep details or None
        """
        trades = self._trade_buffer.get(symbol, deque())
        orderbooks = self._orderbook_history.get(symbol, deque())

        if len(trades) < 10 or len(orderbooks) < 2:
            return None

        recent_trades = list(trades)[-20:]

        # Check for rapid price movement through levels
        buy_trades = [t for t in recent_trades if t.side == "buy"]
        sell_trades = [t for t in recent_trades if t.side == "sell"]

        # Buy sweep detection
        if len(buy_trades) >= 5:
            prices = [t.price for t in buy_trades[-10:]]
            if len(set(prices)) >= 3:  # Multiple price levels
                price_range = max(prices) - min(prices)
                time_span = (buy_trades[-1].timestamp - buy_trades[-10].timestamp).total_seconds()
                if time_span > 0 and price_range / min(prices) > 0.001:  # 0.1% sweep
                    total_volume = sum(t.quantity for t in buy_trades[-10:])
                    return {
                        "type": "buy_sweep",
                        "levels_swept": len(set(prices)),
                        "volume": total_volume,
                        "price_range": price_range,
                        "speed": time_span,
                    }

        # Sell sweep detection
        if len(sell_trades) >= 5:
            prices = [t.price for t in sell_trades[-10:]]
            if len(set(prices)) >= 3:
                price_range = max(prices) - min(prices)
                time_span = (sell_trades[-1].timestamp - sell_trades[-10].timestamp).total_seconds()
                if time_span > 0 and price_range / min(prices) > 0.001:
                    total_volume = sum(t.quantity for t in sell_trades[-10:])
                    return {
                        "type": "sell_sweep",
                        "levels_swept": len(set(prices)),
                        "volume": total_volume,
                        "price_range": price_range,
                        "speed": time_span,
                    }

        return None

    def generate_signal(self, symbol: str) -> OrderFlowSignal:
        """
        Generate trading signal from order flow analysis.

        Args:
            symbol: Trading symbol

        Returns:
            OrderFlowSignal with trading decision
        """
        metrics = self.analyze_order_flow(symbol)

        # Check minimum data requirements
        if metrics.trade_count < self.config.min_trades_for_signal:
            return OrderFlowSignal(
                action="FLAT",
                confidence=0,
                signal_type="insufficient_data",
                metrics=metrics,
                reasoning=f"Insufficient trades ({metrics.trade_count})",
                entry_price=None,
                stop_loss=None,
                take_profit=None,
            )

        # Check for special patterns
        absorption = self.detect_absorption(symbol)
        sweep = self.detect_sweep(symbol)

        action = "FLAT"
        confidence = 0
        signal_type = "imbalance"
        reasoning = ""

        # Absorption signal (contrarian)
        if absorption:
            if absorption["type"] == "bid_absorption":
                action = "LONG"
                confidence = min(0.8, absorption["stability"])
                signal_type = "absorption"
                reasoning = f"Bid absorption at {absorption['level']:.4f} with {absorption['sell_volume']:.2f} sell volume absorbed"
            elif absorption["type"] == "ask_absorption":
                action = "SHORT"
                confidence = min(0.8, absorption["stability"])
                signal_type = "absorption"
                reasoning = f"Ask absorption at {absorption['level']:.4f} with {absorption['buy_volume']:.2f} buy volume absorbed"

        # Sweep signal (momentum)
        elif sweep:
            if sweep["type"] == "buy_sweep":
                action = "LONG"
                confidence = min(0.7, sweep["levels_swept"] / 5)
                signal_type = "sweep"
                reasoning = f"Buy sweep through {sweep['levels_swept']} levels, {sweep['volume']:.2f} volume"
            elif sweep["type"] == "sell_sweep":
                action = "SHORT"
                confidence = min(0.7, sweep["levels_swept"] / 5)
                signal_type = "sweep"
                reasoning = f"Sell sweep through {sweep['levels_swept']} levels, {sweep['volume']:.2f} volume"

        # Volume imbalance signal
        elif abs(metrics.volume_imbalance) > self.config.imbalance_threshold:
            if metrics.volume_imbalance > self.config.strong_imbalance_threshold:
                action = "LONG"
                confidence = min(0.7, metrics.volume_imbalance)
                reasoning = f"Strong buy imbalance: {metrics.volume_imbalance:.1%}"
            elif metrics.volume_imbalance > self.config.imbalance_threshold:
                action = "LONG"
                confidence = min(0.5, metrics.volume_imbalance)
                reasoning = f"Buy imbalance: {metrics.volume_imbalance:.1%}"
            elif metrics.volume_imbalance < -self.config.strong_imbalance_threshold:
                action = "SHORT"
                confidence = min(0.7, abs(metrics.volume_imbalance))
                reasoning = f"Strong sell imbalance: {metrics.volume_imbalance:.1%}"
            elif metrics.volume_imbalance < -self.config.imbalance_threshold:
                action = "SHORT"
                confidence = min(0.5, abs(metrics.volume_imbalance))
                reasoning = f"Sell imbalance: {metrics.volume_imbalance:.1%}"
        else:
            reasoning = f"No significant signal (imbalance: {metrics.volume_imbalance:.1%})"

        # Calculate entry, stop, and target
        entry_price = metrics.vwap
        atr = self._atr.get(symbol, entry_price * 0.01)  # Default 1% ATR

        if action == "LONG":
            stop_loss = entry_price - atr * self.config.stop_loss_atr_multiplier
            take_profit = entry_price + atr * self.config.take_profit_atr_multiplier
        elif action == "SHORT":
            stop_loss = entry_price + atr * self.config.stop_loss_atr_multiplier
            take_profit = entry_price - atr * self.config.take_profit_atr_multiplier
        else:
            stop_loss = None
            take_profit = None

        # Apply minimum confidence filter
        if confidence < self.config.min_confidence:
            action = "FLAT"

        return OrderFlowSignal(
            action=action,
            confidence=confidence,
            signal_type=signal_type,
            metrics=metrics,
            reasoning=reasoning,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def get_orderbook_imbalance(self, symbol: str, levels: int = 5) -> float:
        """
        Calculate orderbook imbalance from bid/ask volumes.

        Args:
            symbol: Trading symbol
            levels: Number of levels to analyze

        Returns:
            Imbalance ratio (-1 to 1)
        """
        orderbooks = self._orderbook_history.get(symbol, deque())
        if not orderbooks:
            return 0

        latest = orderbooks[-1]

        bid_volume = sum(level.quantity for level in latest.bids[:levels])
        ask_volume = sum(level.quantity for level in latest.asks[:levels])
        total = bid_volume + ask_volume

        if total == 0:
            return 0

        return (bid_volume - ask_volume) / total

    def clear_buffers(self, symbol: Optional[str] = None):
        """Clear trade and orderbook buffers."""
        if symbol:
            self._trade_buffer.pop(symbol, None)
            self._orderbook_history.pop(symbol, None)
        else:
            self._trade_buffer.clear()
            self._orderbook_history.clear()


def create_order_flow_strategy(config: Optional[OrderFlowConfig] = None) -> OrderFlowStrategy:
    """Factory function to create order flow strategy."""
    return OrderFlowStrategy(config=config)
