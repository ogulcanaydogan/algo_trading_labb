"""
Real-time WebSocket Streaming - Sub-second market data.

Provides WebSocket connections to exchanges for real-time price updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    TICKER = "ticker"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    KLINE = "kline"
    LIQUIDATIONS = "liquidations"


@dataclass
class TickerUpdate:
    """Real-time ticker update."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeUpdate:
    """Real-time trade update."""
    symbol: str
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    trade_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderbookUpdate:
    """Real-time orderbook update."""
    symbol: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    best_bid: float
    best_ask: float
    spread: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "bids": self.bids[:10],
            "asks": self.asks[:10],
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StreamConfig:
    """WebSocket stream configuration."""
    # Connection settings
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: float = 30.0

    # Buffer settings
    buffer_size: int = 1000

    # Rate limiting
    max_messages_per_second: int = 100


class WebSocketStream(ABC):
    """Abstract base class for WebSocket streams."""

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._running = False
        self._subscriptions: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._message_queue: Queue = Queue(maxsize=self.config.buffer_size)
        self._last_update: Dict[str, datetime] = {}

    @abstractmethod
    async def connect(self):
        """Establish WebSocket connection."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close WebSocket connection."""
        pass

    @abstractmethod
    async def subscribe(self, symbol: str, stream_type: StreamType):
        """Subscribe to a data stream."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str, stream_type: StreamType):
        """Unsubscribe from a data stream."""
        pass

    def add_callback(self, stream_type: StreamType, callback: Callable):
        """Add callback for stream updates."""
        key = stream_type.value
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def remove_callback(self, stream_type: StreamType, callback: Callable):
        """Remove callback."""
        key = stream_type.value
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)

    def _emit(self, stream_type: StreamType, data: Any):
        """Emit data to callbacks."""
        key = stream_type.value
        for callback in self._callbacks.get(key, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class BinanceWebSocket(WebSocketStream):
    """
    Binance WebSocket stream implementation.

    Supports:
    - Ticker streams
    - Trade streams
    - Orderbook streams
    - Kline streams
    """

    WEBSOCKET_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_URL = "wss://fstream.binance.com/ws"

    def __init__(self, config: Optional[StreamConfig] = None, use_futures: bool = False):
        super().__init__(config)
        self._ws = None
        self._use_futures = use_futures
        self._stream_id = 1

    async def connect(self):
        """Connect to Binance WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required: pip install websockets")
            return

        url = self.FUTURES_URL if self._use_futures else self.WEBSOCKET_URL

        try:
            self._ws = await websockets.connect(url)
            self._running = True
            logger.info(f"Connected to Binance WebSocket: {url}")

            # Start message handler
            asyncio.create_task(self._message_handler())

        except Exception as e:
            logger.error(f"Failed to connect to Binance WebSocket: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from Binance WebSocket")

    async def subscribe(self, symbol: str, stream_type: StreamType):
        """Subscribe to Binance stream."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        symbol_lower = symbol.lower().replace("/", "").replace("_", "")

        stream_name = self._get_stream_name(symbol_lower, stream_type)

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": self._stream_id,
        }
        self._stream_id += 1

        await self._ws.send(json.dumps(subscribe_msg))
        self._subscriptions.add(stream_name)
        logger.info(f"Subscribed to {stream_name}")

    async def unsubscribe(self, symbol: str, stream_type: StreamType):
        """Unsubscribe from Binance stream."""
        if not self._ws:
            return

        symbol_lower = symbol.lower().replace("/", "").replace("_", "")
        stream_name = self._get_stream_name(symbol_lower, stream_type)

        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": [stream_name],
            "id": self._stream_id,
        }
        self._stream_id += 1

        await self._ws.send(json.dumps(unsubscribe_msg))
        self._subscriptions.discard(stream_name)
        logger.info(f"Unsubscribed from {stream_name}")

    def _get_stream_name(self, symbol: str, stream_type: StreamType) -> str:
        """Get Binance stream name."""
        if stream_type == StreamType.TICKER:
            return f"{symbol}@ticker"
        elif stream_type == StreamType.TRADES:
            return f"{symbol}@trade"
        elif stream_type == StreamType.ORDERBOOK:
            return f"{symbol}@depth20@100ms"
        elif stream_type == StreamType.KLINE:
            return f"{symbol}@kline_1m"
        elif stream_type == StreamType.LIQUIDATIONS:
            return f"{symbol}@forceOrder"
        return f"{symbol}@ticker"

    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.config.ping_interval
                )
                data = json.loads(message)
                self._process_message(data)

            except asyncio.TimeoutError:
                # Send ping
                if self._ws:
                    await self._ws.ping()

            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                if self._running:
                    await self._reconnect()

    def _process_message(self, data: Dict):
        """Process incoming message and emit to callbacks."""
        if "e" not in data:
            return

        event_type = data["e"]

        if event_type == "24hrTicker":
            ticker = TickerUpdate(
                symbol=data["s"],
                price=float(data["c"]),
                bid=float(data["b"]),
                ask=float(data["a"]),
                volume_24h=float(data["v"]),
                change_24h=float(data["P"]),
                timestamp=datetime.fromtimestamp(data["E"] / 1000),
            )
            self._emit(StreamType.TICKER, ticker)
            self._last_update[ticker.symbol] = ticker.timestamp

        elif event_type == "trade":
            trade = TradeUpdate(
                symbol=data["s"],
                price=float(data["p"]),
                quantity=float(data["q"]),
                side="sell" if data["m"] else "buy",
                trade_id=str(data["t"]),
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
            )
            self._emit(StreamType.TRADES, trade)

        elif event_type == "depthUpdate":
            orderbook = OrderbookUpdate(
                symbol=data["s"],
                bids=[[float(p), float(q)] for p, q in data.get("b", [])],
                asks=[[float(p), float(q)] for p, q in data.get("a", [])],
                best_bid=float(data["b"][0][0]) if data.get("b") else 0,
                best_ask=float(data["a"][0][0]) if data.get("a") else 0,
                spread=0,
                timestamp=datetime.fromtimestamp(data["E"] / 1000),
            )
            if orderbook.best_bid and orderbook.best_ask:
                orderbook.spread = orderbook.best_ask - orderbook.best_bid
            self._emit(StreamType.ORDERBOOK, orderbook)

    async def _reconnect(self):
        """Reconnect to WebSocket."""
        delay = self.config.reconnect_delay

        while self._running:
            try:
                logger.info(f"Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                await self.connect()

                # Resubscribe
                for stream in self._subscriptions.copy():
                    await self._ws.send(json.dumps({
                        "method": "SUBSCRIBE",
                        "params": [stream],
                        "id": self._stream_id,
                    }))
                    self._stream_id += 1

                return

            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                delay = min(delay * 2, self.config.max_reconnect_delay)


class StreamManager:
    """
    Manage multiple WebSocket streams.

    Features:
    - Multi-exchange support
    - Automatic reconnection
    - Callback management
    - Rate limiting
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._streams: Dict[str, WebSocketStream] = {}
        self._latest_tickers: Dict[str, TickerUpdate] = {}
        self._latest_orderbooks: Dict[str, OrderbookUpdate] = {}
        self._trade_buffer: Dict[str, List[TradeUpdate]] = {}

    def add_stream(self, name: str, stream: WebSocketStream):
        """Add a WebSocket stream."""
        self._streams[name] = stream

        # Register internal callbacks
        stream.add_callback(StreamType.TICKER, self._on_ticker)
        stream.add_callback(StreamType.ORDERBOOK, self._on_orderbook)
        stream.add_callback(StreamType.TRADES, self._on_trade)

    async def start_all(self):
        """Start all streams."""
        for name, stream in self._streams.items():
            try:
                await stream.connect()
                logger.info(f"Started stream: {name}")
            except Exception as e:
                logger.error(f"Failed to start stream {name}: {e}")

    async def stop_all(self):
        """Stop all streams."""
        for name, stream in self._streams.items():
            try:
                await stream.disconnect()
                logger.info(f"Stopped stream: {name}")
            except Exception as e:
                logger.error(f"Failed to stop stream {name}: {e}")

    def _on_ticker(self, ticker: TickerUpdate):
        """Handle ticker update."""
        self._latest_tickers[ticker.symbol] = ticker

    def _on_orderbook(self, orderbook: OrderbookUpdate):
        """Handle orderbook update."""
        self._latest_orderbooks[orderbook.symbol] = orderbook

    def _on_trade(self, trade: TradeUpdate):
        """Handle trade update."""
        if trade.symbol not in self._trade_buffer:
            self._trade_buffer[trade.symbol] = []
        self._trade_buffer[trade.symbol].append(trade)

        # Keep buffer limited
        if len(self._trade_buffer[trade.symbol]) > 1000:
            self._trade_buffer[trade.symbol] = self._trade_buffer[trade.symbol][-500:]

    def get_latest_ticker(self, symbol: str) -> Optional[TickerUpdate]:
        """Get latest ticker for symbol."""
        return self._latest_tickers.get(symbol)

    def get_latest_orderbook(self, symbol: str) -> Optional[OrderbookUpdate]:
        """Get latest orderbook for symbol."""
        return self._latest_orderbooks.get(symbol)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeUpdate]:
        """Get recent trades for symbol."""
        trades = self._trade_buffer.get(symbol, [])
        return trades[-limit:]

    def get_vwap(self, symbol: str, num_trades: int = 100) -> Optional[float]:
        """Calculate VWAP from recent trades."""
        trades = self.get_recent_trades(symbol, num_trades)
        if not trades:
            return None

        total_value = sum(t.price * t.quantity for t in trades)
        total_quantity = sum(t.quantity for t in trades)

        return total_value / total_quantity if total_quantity > 0 else None

    def get_trade_imbalance(self, symbol: str, num_trades: int = 100) -> float:
        """Calculate buy/sell imbalance from recent trades."""
        trades = self.get_recent_trades(symbol, num_trades)
        if not trades:
            return 0.0

        buy_volume = sum(t.quantity for t in trades if t.side == "buy")
        sell_volume = sum(t.quantity for t in trades if t.side == "sell")
        total = buy_volume + sell_volume

        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total


def create_binance_stream(
    config: Optional[StreamConfig] = None,
    use_futures: bool = False
) -> BinanceWebSocket:
    """Factory function to create Binance WebSocket stream."""
    return BinanceWebSocket(config=config, use_futures=use_futures)


def create_stream_manager(config: Optional[StreamConfig] = None) -> StreamManager:
    """Factory function to create stream manager."""
    return StreamManager(config=config)
