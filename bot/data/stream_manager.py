"""
Real-Time Stream Manager - Event-Driven Trading Architecture.

Features:
- Binance WebSocket for crypto (BTC, XRP, ETH)
- Alpaca WebSocket for stocks (TSLA)
- Tick aggregation to OHLCV bars
- Event-driven signal generation
- Automatic reconnection and backfill
- Health monitoring and metrics

Flow: Stream → Buffer → Bar Aggregation → Feature Calc → Prediction → Signal
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import threading
from queue import Queue, Empty
import pandas as pd
import numpy as np

from .websocket_stream import (
    StreamType,
    StreamConfig,
    TickerUpdate,
    TradeUpdate,
    OrderbookUpdate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class AssetClass(Enum):
    """Asset class for routing."""
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"


class StreamStatus(Enum):
    """Stream connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class EventType(Enum):
    """Types of streaming events."""
    TICK = "tick"
    BAR_COMPLETE = "bar_complete"
    PREDICTION = "prediction"
    SIGNAL = "signal"
    HEALTH = "health"
    RECONNECT = "reconnect"
    BACKFILL = "backfill"


@dataclass
class Tick:
    """A single price tick."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    side: Optional[str] = None  # "buy" or "sell"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
        }


@dataclass
class Bar:
    """OHLCV bar aggregated from ticks."""
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    open_time: datetime
    close_time: datetime
    vwap: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "tick_count": self.tick_count,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "vwap": self.vwap,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
        }
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series with datetime index."""
        return pd.Series({
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }, name=self.close_time)


@dataclass
class StreamEvent:
    """Generic streaming event."""
    event_type: EventType
    symbol: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "data": self.data if isinstance(self.data, dict) else str(self.data),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StreamHealth:
    """Health status of a stream."""
    stream_name: str
    status: StreamStatus
    last_message: Optional[datetime] = None
    reconnect_count: int = 0
    error_count: int = 0
    latency_ms: float = 0.0
    messages_per_second: float = 0.0


@dataclass
class StreamManagerConfig:
    """Configuration for StreamManager."""
    # Bar aggregation
    default_timeframe: str = "1m"  # 1m, 5m, 15m, 1h
    supported_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    
    # Connection settings
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_attempts: int = 10
    ping_interval: float = 30.0
    
    # Buffer settings
    tick_buffer_size: int = 10000
    bar_history_size: int = 500  # Bars to keep for feature calculation
    
    # Health monitoring
    health_check_interval: float = 10.0
    stale_threshold_seconds: float = 30.0
    
    # Backfill settings
    backfill_on_reconnect: bool = True
    backfill_bars: int = 100


# =============================================================================
# Bar Aggregator
# =============================================================================

class BarAggregator:
    """
    Aggregates ticks into OHLCV bars.
    
    Supports multiple timeframes and emits events when bars complete.
    """
    
    TIMEFRAME_SECONDS = {
        "1s": 1,
        "5s": 5,
        "10s": 10,
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str],
        on_bar_complete: Optional[Callable[[Bar], None]] = None,
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.on_bar_complete = on_bar_complete
        
        # Current building bars for each timeframe
        self._building: Dict[str, Dict] = {}
        
        # Completed bar history
        self._history: Dict[str, List[Bar]] = {tf: [] for tf in timeframes}
        self._max_history = 500
        
        # Initialize building bars
        for tf in timeframes:
            self._init_building_bar(tf)
    
    def _init_building_bar(self, timeframe: str, open_time: Optional[datetime] = None):
        """Initialize a new building bar."""
        if open_time is None:
            open_time = self._align_time(datetime.now(), timeframe)
        
        self._building[timeframe] = {
            "open_time": open_time,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": 0.0,
            "tick_count": 0,
            "value_sum": 0.0,  # For VWAP calculation
            "buy_volume": 0.0,
            "sell_volume": 0.0,
        }
    
    def _align_time(self, dt: datetime, timeframe: str) -> datetime:
        """Align datetime to timeframe boundary."""
        seconds = self.TIMEFRAME_SECONDS.get(timeframe, 60)
        timestamp = dt.timestamp()
        aligned = (timestamp // seconds) * seconds
        return datetime.fromtimestamp(aligned)
    
    def _get_bar_end_time(self, open_time: datetime, timeframe: str) -> datetime:
        """Get the closing time for a bar."""
        seconds = self.TIMEFRAME_SECONDS.get(timeframe, 60)
        return open_time + timedelta(seconds=seconds)
    
    def process_tick(self, tick: Tick) -> List[Bar]:
        """
        Process a tick and return any completed bars.
        
        Args:
            tick: Incoming tick data
            
        Returns:
            List of completed bars (may be empty)
        """
        completed_bars = []
        
        for timeframe in self.timeframes:
            bar_data = self._building[timeframe]
            bar_end = self._get_bar_end_time(bar_data["open_time"], timeframe)
            
            # Check if current bar should close
            if tick.timestamp >= bar_end:
                # Finalize current bar if it has data
                if bar_data["tick_count"] > 0:
                    bar = self._finalize_bar(timeframe)
                    completed_bars.append(bar)
                    self._history[timeframe].append(bar)
                    
                    # Trim history
                    if len(self._history[timeframe]) > self._max_history:
                        self._history[timeframe] = self._history[timeframe][-self._max_history:]
                
                # Start new bar aligned to timeframe
                aligned_time = self._align_time(tick.timestamp, timeframe)
                self._init_building_bar(timeframe, aligned_time)
                bar_data = self._building[timeframe]
            
            # Update building bar
            if bar_data["open"] is None:
                bar_data["open"] = tick.price
            bar_data["high"] = max(bar_data["high"] or tick.price, tick.price)
            bar_data["low"] = min(bar_data["low"] or tick.price, tick.price)
            bar_data["close"] = tick.price
            bar_data["volume"] += tick.volume
            bar_data["tick_count"] += 1
            bar_data["value_sum"] += tick.price * tick.volume
            
            if tick.side == "buy":
                bar_data["buy_volume"] += tick.volume
            elif tick.side == "sell":
                bar_data["sell_volume"] += tick.volume
        
        # Emit completed bars
        for bar in completed_bars:
            if self.on_bar_complete:
                self.on_bar_complete(bar)
        
        return completed_bars
    
    def _finalize_bar(self, timeframe: str) -> Bar:
        """Finalize and return a completed bar."""
        bar_data = self._building[timeframe]
        
        vwap = 0.0
        if bar_data["volume"] > 0:
            vwap = bar_data["value_sum"] / bar_data["volume"]
        
        return Bar(
            symbol=self.symbol,
            timeframe=timeframe,
            open=bar_data["open"],
            high=bar_data["high"],
            low=bar_data["low"],
            close=bar_data["close"],
            volume=bar_data["volume"],
            tick_count=bar_data["tick_count"],
            open_time=bar_data["open_time"],
            close_time=self._get_bar_end_time(bar_data["open_time"], timeframe),
            vwap=vwap,
            buy_volume=bar_data["buy_volume"],
            sell_volume=bar_data["sell_volume"],
        )
    
    def get_current_bar(self, timeframe: str) -> Optional[Dict]:
        """Get the current building bar."""
        return self._building.get(timeframe)
    
    def get_history(self, timeframe: str, limit: int = 100) -> List[Bar]:
        """Get historical completed bars."""
        return self._history.get(timeframe, [])[-limit:]
    
    def get_dataframe(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get history as DataFrame for feature calculation."""
        bars = self.get_history(timeframe, limit)
        if not bars:
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            data.append({
                "timestamp": bar.close_time,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "buy_volume": bar.buy_volume,
                "sell_volume": bar.sell_volume,
            })
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df


# =============================================================================
# Base Stream Handler
# =============================================================================

class BaseStreamHandler(ABC):
    """Abstract base class for stream handlers."""
    
    def __init__(
        self,
        symbols: List[str],
        config: StreamManagerConfig,
    ):
        self.symbols = symbols
        self.config = config
        self.status = StreamStatus.DISCONNECTED
        self._ws = None
        self._running = False
        self._callbacks: Dict[EventType, List[Callable]] = defaultdict(list)
        self._last_message: Dict[str, datetime] = {}
        self._reconnect_count = 0
        self._error_count = 0
        self._message_count = 0
        self._last_count_time = time.time()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name."""
        pass
    
    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """Asset class this handler supports."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to WebSocket."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from WebSocket."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols."""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: str):
        """Handle incoming message."""
        pass
    
    def add_callback(self, event_type: EventType, callback: Callable):
        """Add event callback."""
        self._callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: EventType, callback: Callable):
        """Remove event callback."""
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
    
    def emit(self, event: StreamEvent):
        """Emit event to all callbacks."""
        for callback in self._callbacks.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error in {self.name}: {e}")
    
    def get_health(self) -> StreamHealth:
        """Get health status."""
        current_time = time.time()
        elapsed = current_time - self._last_count_time
        mps = self._message_count / elapsed if elapsed > 0 else 0
        
        return StreamHealth(
            stream_name=self.name,
            status=self.status,
            last_message=max(self._last_message.values()) if self._last_message else None,
            reconnect_count=self._reconnect_count,
            error_count=self._error_count,
            messages_per_second=mps,
        )
    
    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff."""
        delay = self.config.reconnect_delay
        attempts = 0
        
        while self._running and attempts < self.config.reconnect_attempts:
            self.status = StreamStatus.RECONNECTING
            attempts += 1
            self._reconnect_count += 1
            
            logger.info(f"{self.name}: Reconnecting (attempt {attempts})...")
            
            try:
                await asyncio.sleep(delay)
                success = await self.connect()
                
                if success:
                    logger.info(f"{self.name}: Reconnected successfully")
                    await self.subscribe(self.symbols)
                    
                    # Emit reconnect event
                    self.emit(StreamEvent(
                        event_type=EventType.RECONNECT,
                        symbol="*",
                        data={"attempts": attempts},
                    ))
                    return
                    
            except Exception as e:
                logger.error(f"{self.name}: Reconnection failed: {e}")
                self._error_count += 1
            
            delay = min(delay * 2, self.config.max_reconnect_delay)
        
        self.status = StreamStatus.ERROR
        logger.error(f"{self.name}: Max reconnection attempts reached")


# =============================================================================
# Binance Stream Handler
# =============================================================================

class BinanceStreamHandler(BaseStreamHandler):
    """
    Binance WebSocket handler for crypto.
    
    Supports: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, etc.
    """
    
    SPOT_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_URL = "wss://fstream.binance.com/ws"
    
    def __init__(
        self,
        symbols: List[str],
        config: StreamManagerConfig,
        use_futures: bool = False,
    ):
        super().__init__(symbols, config)
        self.use_futures = use_futures
        self._stream_id = 1
    
    @property
    def name(self) -> str:
        return "binance_futures" if self.use_futures else "binance_spot"
    
    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CRYPTO
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Binance format (btcusdt)."""
        return symbol.lower().replace("/", "").replace("_", "").replace("-", "")
    
    def _denormalize_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol back to standard format."""
        # Common pairs
        for quote in ["usdt", "usd", "btc", "eth", "bnb"]:
            if binance_symbol.lower().endswith(quote):
                base = binance_symbol[:-len(quote)].upper()
                return f"{base}/{quote.upper()}"
        return binance_symbol.upper()
    
    async def connect(self) -> bool:
        """Connect to Binance WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required: pip install websockets")
            return False
        
        url = self.FUTURES_URL if self.use_futures else self.SPOT_URL
        
        try:
            self.status = StreamStatus.CONNECTING
            self._ws = await websockets.connect(
                url,
                ping_interval=self.config.ping_interval,
                ping_timeout=10,
            )
            self.status = StreamStatus.CONNECTED
            self._running = True
            logger.info(f"{self.name}: Connected to {url}")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Connection failed: {e}")
            self.status = StreamStatus.ERROR
            self._error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.status = StreamStatus.DISCONNECTED
        logger.info(f"{self.name}: Disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to trade streams for symbols."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        
        streams = []
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            streams.append(f"{normalized}@trade")
            streams.append(f"{normalized}@ticker")
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self._stream_id,
        }
        self._stream_id += 1
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(f"{self.name}: Subscribed to {streams}")
    
    async def run(self):
        """Main message loop."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.config.ping_interval + 10,
                )
                self._message_count += 1
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                logger.warning(f"{self.name}: No message received, checking connection...")
                
            except Exception as e:
                logger.error(f"{self.name}: Message loop error: {e}")
                self._error_count += 1
                if self._running:
                    await self._reconnect_loop()
                break
    
    async def _handle_message(self, message: str):
        """Handle incoming Binance message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        
        # Skip subscription confirmations
        if "result" in data or "id" in data:
            return
        
        event_type = data.get("e")
        
        if event_type == "trade":
            tick = Tick(
                symbol=self._denormalize_symbol(data["s"]),
                price=float(data["p"]),
                volume=float(data["q"]),
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                side="sell" if data["m"] else "buy",
            )
            self._last_message[tick.symbol] = tick.timestamp
            
            self.emit(StreamEvent(
                event_type=EventType.TICK,
                symbol=tick.symbol,
                data=tick,
            ))
            
        elif event_type == "24hrTicker":
            # Ticker updates (for monitoring)
            symbol = self._denormalize_symbol(data["s"])
            self._last_message[symbol] = datetime.now()


# =============================================================================
# Alpaca Stream Handler
# =============================================================================

class AlpacaStreamHandler(BaseStreamHandler):
    """
    Alpaca WebSocket handler for US stocks.
    
    Supports: TSLA, AAPL, GOOGL, MSFT, etc.
    """
    
    # Paper trading / IEX data
    PAPER_URL = "wss://stream.data.alpaca.markets/v2/iex"
    # SIP data (paid)
    SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"
    
    def __init__(
        self,
        symbols: List[str],
        config: StreamManagerConfig,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_sip: bool = False,
    ):
        super().__init__(symbols, config)
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY", "")
        self.use_sip = use_sip
    
    @property
    def name(self) -> str:
        return "alpaca_sip" if self.use_sip else "alpaca_iex"
    
    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.STOCK
    
    async def connect(self) -> bool:
        """Connect to Alpaca WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required: pip install websockets")
            return False
        
        if not self.api_key or not self.api_secret:
            logger.error(f"{self.name}: API credentials not configured")
            return False
        
        url = self.SIP_URL if self.use_sip else self.PAPER_URL
        
        try:
            self.status = StreamStatus.CONNECTING
            self._ws = await websockets.connect(url)
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret,
            }
            await self._ws.send(json.dumps(auth_msg))
            
            # Wait for auth response
            response = await asyncio.wait_for(self._ws.recv(), timeout=10)
            data = json.loads(response)
            
            if isinstance(data, list) and len(data) > 0:
                if data[0].get("T") == "error":
                    logger.error(f"{self.name}: Auth failed: {data[0].get('msg')}")
                    self.status = StreamStatus.ERROR
                    return False
            
            self.status = StreamStatus.CONNECTED
            self._running = True
            logger.info(f"{self.name}: Connected and authenticated")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Connection failed: {e}")
            self.status = StreamStatus.ERROR
            self._error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.status = StreamStatus.DISCONNECTED
        logger.info(f"{self.name}: Disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to trade streams for symbols."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        
        subscribe_msg = {
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
        }
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info(f"{self.name}: Subscribed to {symbols}")
    
    async def run(self):
        """Main message loop."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.config.ping_interval + 10,
                )
                self._message_count += 1
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # Alpaca doesn't require ping/pong for keepalive
                pass
                
            except Exception as e:
                logger.error(f"{self.name}: Message loop error: {e}")
                self._error_count += 1
                if self._running:
                    await self._reconnect_loop()
                break
    
    async def _handle_message(self, message: str):
        """Handle incoming Alpaca message."""
        try:
            data_list = json.loads(message)
        except json.JSONDecodeError:
            return
        
        if not isinstance(data_list, list):
            return
        
        for data in data_list:
            msg_type = data.get("T")
            
            if msg_type == "t":  # Trade
                tick = Tick(
                    symbol=data["S"],
                    price=float(data["p"]),
                    volume=float(data["s"]),
                    timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")).replace(tzinfo=None),
                    side=None,  # Alpaca doesn't provide trade direction
                )
                self._last_message[tick.symbol] = tick.timestamp
                
                self.emit(StreamEvent(
                    event_type=EventType.TICK,
                    symbol=tick.symbol,
                    data=tick,
                ))
                
            elif msg_type == "q":  # Quote
                symbol = data["S"]
                self._last_message[symbol] = datetime.now()


# =============================================================================
# Stream Manager
# =============================================================================

class StreamManager:
    """
    Central manager for all real-time data streams.
    
    Features:
    - Multi-exchange support (Binance, Alpaca)
    - Automatic bar aggregation
    - Event-driven callbacks
    - Health monitoring
    - Graceful reconnection
    
    Usage:
        manager = StreamManager()
        manager.add_crypto_symbols(["BTC/USDT", "XRP/USDT"])
        manager.add_stock_symbols(["TSLA"])
        manager.on_bar_complete(handle_bar)
        await manager.start()
    """
    
    def __init__(
        self,
        config: Optional[StreamManagerConfig] = None,
    ):
        self.config = config or StreamManagerConfig()
        
        # Stream handlers
        self._handlers: Dict[str, BaseStreamHandler] = {}
        
        # Bar aggregators per symbol
        self._aggregators: Dict[str, BarAggregator] = {}
        
        # Event callbacks
        self._callbacks: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Async tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Metrics
        self._bars_produced = 0
        self._ticks_processed = 0
        self._start_time: Optional[datetime] = None
    
    def add_crypto_symbols(
        self,
        symbols: List[str],
        use_futures: bool = False,
    ):
        """Add crypto symbols to stream via Binance."""
        handler_name = "binance_futures" if use_futures else "binance_spot"
        
        if handler_name not in self._handlers:
            self._handlers[handler_name] = BinanceStreamHandler(
                symbols=symbols,
                config=self.config,
                use_futures=use_futures,
            )
            # Register internal tick handler
            self._handlers[handler_name].add_callback(
                EventType.TICK,
                self._on_tick,
            )
        else:
            self._handlers[handler_name].symbols.extend(symbols)
        
        # Create aggregators for symbols
        for symbol in symbols:
            if symbol not in self._aggregators:
                self._aggregators[symbol] = BarAggregator(
                    symbol=symbol,
                    timeframes=self.config.supported_timeframes,
                    on_bar_complete=self._on_bar_complete,
                )
    
    def add_stock_symbols(
        self,
        symbols: List[str],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """Add stock symbols to stream via Alpaca."""
        handler_name = "alpaca_iex"
        
        if handler_name not in self._handlers:
            self._handlers[handler_name] = AlpacaStreamHandler(
                symbols=symbols,
                config=self.config,
                api_key=api_key,
                api_secret=api_secret,
            )
            self._handlers[handler_name].add_callback(
                EventType.TICK,
                self._on_tick,
            )
        else:
            self._handlers[handler_name].symbols.extend(symbols)
        
        for symbol in symbols:
            if symbol not in self._aggregators:
                self._aggregators[symbol] = BarAggregator(
                    symbol=symbol,
                    timeframes=self.config.supported_timeframes,
                    on_bar_complete=self._on_bar_complete,
                )
    
    def on_tick(self, callback: Callable[[StreamEvent], None]):
        """Register tick callback."""
        self._callbacks[EventType.TICK].append(callback)
    
    def on_bar_complete(self, callback: Callable[[Bar], None]):
        """Register bar complete callback."""
        self._callbacks[EventType.BAR_COMPLETE].append(callback)
    
    def on_prediction(self, callback: Callable[[StreamEvent], None]):
        """Register prediction callback."""
        self._callbacks[EventType.PREDICTION].append(callback)
    
    def on_signal(self, callback: Callable[[StreamEvent], None]):
        """Register signal callback."""
        self._callbacks[EventType.SIGNAL].append(callback)
    
    def _on_tick(self, event: StreamEvent):
        """Internal tick handler."""
        tick = event.data
        self._ticks_processed += 1
        
        # Route to aggregator
        if tick.symbol in self._aggregators:
            self._aggregators[tick.symbol].process_tick(tick)
        
        # Forward to user callbacks
        for callback in self._callbacks.get(EventType.TICK, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
    
    def _on_bar_complete(self, bar: Bar):
        """Internal bar complete handler."""
        self._bars_produced += 1
        
        event = StreamEvent(
            event_type=EventType.BAR_COMPLETE,
            symbol=bar.symbol,
            data=bar,
        )
        
        for callback in self._callbacks.get(EventType.BAR_COMPLETE, []):
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    async def start(self):
        """Start all streams."""
        if self._running:
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        logger.info(f"StreamManager starting with handlers: {list(self._handlers.keys())}")
        
        # Connect and subscribe all handlers
        for name, handler in self._handlers.items():
            try:
                success = await handler.connect()
                if success:
                    await handler.subscribe(handler.symbols)
                    task = asyncio.create_task(handler.run())
                    task.set_name(f"stream_{name}")
                    self._tasks.append(task)
                else:
                    logger.error(f"Failed to start handler: {name}")
            except Exception as e:
                logger.error(f"Error starting handler {name}: {e}")
        
        # Start health monitoring
        health_task = asyncio.create_task(self._health_monitor())
        health_task.set_name("health_monitor")
        self._tasks.append(health_task)
        
        logger.info("StreamManager started")
    
    async def stop(self):
        """Stop all streams."""
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        
        # Disconnect handlers
        for name, handler in self._handlers.items():
            try:
                await handler.disconnect()
            except Exception as e:
                logger.error(f"Error stopping handler {name}: {e}")
        
        logger.info("StreamManager stopped")
    
    async def _health_monitor(self):
        """Monitor stream health."""
        while self._running:
            await asyncio.sleep(self.config.health_check_interval)
            
            for name, handler in self._handlers.items():
                health = handler.get_health()
                
                # Check for stale streams
                if health.last_message:
                    staleness = (datetime.now() - health.last_message).total_seconds()
                    if staleness > self.config.stale_threshold_seconds:
                        logger.warning(
                            f"{name}: Stream stale ({staleness:.1f}s since last message)"
                        )
                
                logger.debug(
                    f"{name}: status={health.status.value}, "
                    f"msgs/s={health.messages_per_second:.1f}, "
                    f"reconnects={health.reconnect_count}"
                )
    
    def get_bar_dataframe(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get bar history as DataFrame for feature calculation."""
        if symbol not in self._aggregators:
            return pd.DataFrame()
        return self._aggregators[symbol].get_dataframe(timeframe, limit)
    
    def get_current_bar(
        self,
        symbol: str,
        timeframe: str = "1m",
    ) -> Optional[Dict]:
        """Get currently building bar."""
        if symbol not in self._aggregators:
            return None
        return self._aggregators[symbol].get_current_bar(timeframe)
    
    def get_health(self) -> Dict[str, StreamHealth]:
        """Get health status of all streams."""
        return {
            name: handler.get_health()
            for name, handler in self._handlers.items()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics."""
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            "uptime_seconds": uptime,
            "ticks_processed": self._ticks_processed,
            "bars_produced": self._bars_produced,
            "ticks_per_second": self._ticks_processed / uptime if uptime > 0 else 0,
            "bars_per_second": self._bars_produced / uptime if uptime > 0 else 0,
            "symbols_streaming": len(self._aggregators),
            "handlers_active": len(self._handlers),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_stream_manager(
    config: Optional[StreamManagerConfig] = None,
) -> StreamManager:
    """Create a StreamManager instance."""
    return StreamManager(config=config)


def create_crypto_stream(
    symbols: List[str],
    config: Optional[StreamManagerConfig] = None,
    use_futures: bool = False,
) -> StreamManager:
    """Create a StreamManager pre-configured for crypto."""
    manager = StreamManager(config=config)
    manager.add_crypto_symbols(symbols, use_futures=use_futures)
    return manager


def create_stock_stream(
    symbols: List[str],
    config: Optional[StreamManagerConfig] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> StreamManager:
    """Create a StreamManager pre-configured for stocks."""
    manager = StreamManager(config=config)
    manager.add_stock_symbols(symbols, api_key=api_key, api_secret=api_secret)
    return manager


# =============================================================================
# Event-Driven Trading Integration
# =============================================================================

class StreamingTradingEngine:
    """
    Event-driven trading engine integrated with StreamManager.
    
    Flow:
    1. Stream receives ticks
    2. Ticks aggregated to bars
    3. On bar complete: calculate features
    4. Generate prediction from ML model
    5. Convert prediction to trading signal
    6. Execute signal via execution layer
    
    Usage:
        engine = StreamingTradingEngine(
            stream_manager=stream_manager,
            predictor=ml_predictor,
            executor=smart_executor,
        )
        await engine.start()
    """
    
    def __init__(
        self,
        stream_manager: StreamManager,
        predictor: Optional[Any] = None,  # ML predictor
        executor: Optional[Any] = None,    # Order executor
        signal_threshold: float = 0.6,
    ):
        self.stream_manager = stream_manager
        self.predictor = predictor
        self.executor = executor
        self.signal_threshold = signal_threshold
        
        # State
        self._signals: Dict[str, Any] = {}
        self._predictions: Dict[str, float] = {}
        
        # Register callbacks
        self.stream_manager.on_bar_complete(self._on_bar_complete)
    
    async def start(self):
        """Start the streaming engine."""
        await self.stream_manager.start()
    
    async def stop(self):
        """Stop the streaming engine."""
        await self.stream_manager.stop()
    
    def _on_bar_complete(self, bar: Bar):
        """Handle completed bar - trigger prediction pipeline."""
        try:
            # Get feature data
            df = self.stream_manager.get_bar_dataframe(
                bar.symbol,
                bar.timeframe,
                limit=100,
            )
            
            if df.empty or len(df) < 20:
                # Not enough data for features
                return
            
            # Calculate features (basic example)
            features = self._calculate_features(df)
            
            # Generate prediction
            prediction = self._generate_prediction(bar.symbol, features)
            self._predictions[bar.symbol] = prediction
            
            # Generate signal
            signal = self._generate_signal(bar.symbol, prediction, bar)
            
            if signal:
                self._signals[bar.symbol] = signal
                logger.info(
                    f"Signal generated: {bar.symbol} {signal['direction']} "
                    f"@ {bar.close:.2f} (confidence: {prediction:.2f})"
                )
                
                # Execute if we have an executor
                if self.executor:
                    self._execute_signal(signal)
                    
        except Exception as e:
            logger.error(f"Error in prediction pipeline for {bar.symbol}: {e}")
    
    def _calculate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features from bar data."""
        close = df["close"]
        volume = df["volume"]
        
        # Basic features
        features = {
            "return_1": (close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0,
            "return_5": (close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0,
            "return_20": (close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0,
            "volatility": close.pct_change().std() * np.sqrt(252) if len(close) > 1 else 0,
            "volume_ratio": volume.iloc[-1] / volume.mean() if volume.mean() > 0 else 1,
        }
        
        # Simple moving averages
        if len(close) >= 20:
            sma_20 = close.rolling(20).mean().iloc[-1]
            features["sma_20_ratio"] = close.iloc[-1] / sma_20 if sma_20 > 0 else 1
        
        return features
    
    def _generate_prediction(
        self,
        symbol: str,
        features: Dict[str, float],
    ) -> float:
        """Generate prediction from features."""
        if self.predictor:
            # Use ML predictor
            try:
                return self.predictor.predict(symbol, features)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return 0.5
        
        # Fallback: simple momentum-based prediction
        return_1 = features.get("return_1", 0)
        return_5 = features.get("return_5", 0)
        sma_ratio = features.get("sma_20_ratio", 1)
        
        # Combine signals
        momentum_signal = 0.5 + (return_1 * 5 + return_5) * 0.5  # Normalize to 0-1
        trend_signal = 0.5 + (sma_ratio - 1) * 2  # Above SMA = bullish
        
        return (momentum_signal * 0.6 + trend_signal * 0.4)
    
    def _generate_signal(
        self,
        symbol: str,
        prediction: float,
        bar: Bar,
    ) -> Optional[Dict]:
        """Convert prediction to trading signal."""
        if prediction > self.signal_threshold:
            return {
                "symbol": symbol,
                "direction": "long",
                "price": bar.close,
                "confidence": prediction,
                "timestamp": bar.close_time,
            }
        elif prediction < (1 - self.signal_threshold):
            return {
                "symbol": symbol,
                "direction": "short",
                "price": bar.close,
                "confidence": 1 - prediction,
                "timestamp": bar.close_time,
            }
        return None
    
    def _execute_signal(self, signal: Dict):
        """Execute trading signal."""
        try:
            # Here you would integrate with your execution layer
            # e.g., self.executor.submit_order(signal)
            logger.info(f"Would execute: {signal}")
        except Exception as e:
            logger.error(f"Execution error: {e}")
    
    def get_latest_signals(self) -> Dict[str, Any]:
        """Get latest signals for all symbols."""
        return self._signals.copy()
    
    def get_latest_predictions(self) -> Dict[str, float]:
        """Get latest predictions for all symbols."""
        return self._predictions.copy()


# =============================================================================
# Convenience: Ready-to-use configurations
# =============================================================================

def create_default_trading_stream(
    crypto_symbols: Optional[List[str]] = None,
    stock_symbols: Optional[List[str]] = None,
) -> StreamManager:
    """
    Create a ready-to-use StreamManager with default configuration.
    
    Args:
        crypto_symbols: Crypto symbols to stream (default: BTC, XRP)
        stock_symbols: Stock symbols to stream (default: TSLA)
    
    Returns:
        Configured StreamManager
    """
    crypto_symbols = crypto_symbols or ["BTC/USDT", "XRP/USDT"]
    stock_symbols = stock_symbols or ["TSLA"]
    
    config = StreamManagerConfig(
        default_timeframe="1m",
        supported_timeframes=["1m", "5m", "15m"],
        reconnect_delay=1.0,
        max_reconnect_delay=60.0,
        tick_buffer_size=10000,
        bar_history_size=500,
        health_check_interval=10.0,
    )
    
    manager = StreamManager(config=config)
    
    if crypto_symbols:
        manager.add_crypto_symbols(crypto_symbols)
    
    if stock_symbols:
        manager.add_stock_symbols(stock_symbols)
    
    return manager
