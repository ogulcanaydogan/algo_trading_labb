"""
WebSocket Streaming Module for Real-Time Dashboard

Phase 7 of the engineering roadmap: Real-time streaming of trading data
to dashboard clients via WebSocket connections.

Features:
1. Multi-client connection management
2. Topic-based subscriptions (portfolio, trades, alerts, risk)
3. Rate-limited broadcasting
4. Heartbeat/keepalive
5. Reconnection handling
6. Message queuing for reliability
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


class StreamTopic(Enum):
    """Topics that clients can subscribe to."""

    PORTFOLIO = "portfolio"  # Portfolio state updates
    POSITIONS = "positions"  # Position changes
    TRADES = "trades"  # Trade executions
    ORDERS = "orders"  # Order status changes
    RISK = "risk"  # Risk metrics and alerts
    SIGNALS = "signals"  # Strategy signals
    REGIME = "regime"  # Market regime changes
    ALERTS = "alerts"  # System alerts
    PERFORMANCE = "performance"  # Performance metrics
    HEARTBEAT = "heartbeat"  # Connection keepalive
    SYSTEM = "system"  # System status


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class StreamMessage:
    """A message to be streamed to clients."""

    topic: StreamTopic
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_json(self) -> str:
        """Convert to JSON string for transmission."""
        return json.dumps(
            {
                "topic": self.topic.value,
                "data": self.data,
                "timestamp": self.timestamp.isoformat(),
                "message_id": self.message_id,
            }
        )


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""

    client_id: str
    websocket: Any  # WebSocket connection object
    subscriptions: Set[StreamTopic] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    is_authenticated: bool = False
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class WebSocketHub:
    """
    Central hub for managing WebSocket connections and message broadcasting.

    This class handles:
    - Client connection/disconnection
    - Subscription management
    - Message broadcasting
    - Rate limiting
    - Connection health monitoring
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        max_clients: int = 100,
        message_rate_limit: float = 10.0,  # messages per second per topic
        connection_timeout: float = 120.0,
    ):
        """
        Initialize the WebSocket hub.

        Args:
            heartbeat_interval: Seconds between heartbeat messages
            max_clients: Maximum concurrent connections
            message_rate_limit: Max messages per second per topic
            connection_timeout: Seconds before inactive connection is closed
        """
        self.heartbeat_interval = heartbeat_interval
        self.max_clients = max_clients
        self.message_rate_limit = message_rate_limit
        self.connection_timeout = connection_timeout

        # Connected clients
        self.clients: Dict[str, ClientConnection] = {}

        # Message history per topic (for new client catch-up)
        self.topic_history: Dict[StreamTopic, deque] = {
            topic: deque(maxlen=50) for topic in StreamTopic
        }

        # Rate limiting state
        self._last_broadcast_time: Dict[StreamTopic, float] = {}
        self._broadcast_lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_dropped": 0,
            "current_connections": 0,
        }

        # Callbacks
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_message_callbacks: List[Callable] = []

        # Running state
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            f"WebSocketHub initialized: max_clients={max_clients}, "
            f"rate_limit={message_rate_limit}/s"
        )

    async def start(self):
        """Start background tasks (heartbeat, cleanup)."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WebSocketHub started")

    async def stop(self):
        """Stop background tasks and close all connections."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for client_id in list(self.clients.keys()):
            await self.disconnect_client(client_id, reason="Server shutdown")

        logger.info("WebSocketHub stopped")

    async def connect_client(
        self,
        websocket: Any,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        initial_subscriptions: Optional[List[StreamTopic]] = None,
    ) -> Optional[str]:
        """
        Register a new client connection.

        Args:
            websocket: WebSocket connection object
            client_id: Optional client ID (generated if not provided)
            user_id: Optional authenticated user ID
            initial_subscriptions: Topics to subscribe to immediately

        Returns:
            Client ID if successful, None if rejected
        """
        if len(self.clients) >= self.max_clients:
            logger.warning("Max clients reached, rejecting connection")
            return None

        client_id = client_id or str(uuid.uuid4())[:12]

        connection = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            user_id=user_id,
            is_authenticated=user_id is not None,
        )

        # Add initial subscriptions
        if initial_subscriptions:
            for topic in initial_subscriptions:
                connection.subscriptions.add(topic)

        # Always subscribe to system and heartbeat
        connection.subscriptions.add(StreamTopic.SYSTEM)
        connection.subscriptions.add(StreamTopic.HEARTBEAT)

        self.clients[client_id] = connection
        self.stats["total_connections"] += 1
        self.stats["current_connections"] = len(self.clients)

        # Notify callbacks
        for callback in self._on_connect_callbacks:
            try:
                await callback(client_id, connection)
            except Exception as e:
                logger.error(f"Connect callback error: {e}")

        # Send welcome message
        await self._send_to_client(
            client_id,
            StreamMessage(
                topic=StreamTopic.SYSTEM,
                data={
                    "type": "connected",
                    "client_id": client_id,
                    "subscriptions": [t.value for t in connection.subscriptions],
                    "server_time": datetime.now().isoformat(),
                },
            ),
        )

        # Send recent history for subscribed topics
        await self._send_topic_history(client_id)

        logger.info(f"Client connected: {client_id}")
        return client_id

    async def disconnect_client(self, client_id: str, reason: str = ""):
        """
        Disconnect a client.

        Args:
            client_id: Client to disconnect
            reason: Reason for disconnection
        """
        if client_id not in self.clients:
            return

        connection = self.clients.pop(client_id)
        self.stats["current_connections"] = len(self.clients)

        # Notify callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                await callback(client_id, connection, reason)
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")

        # Try to send disconnect message
        try:
            if hasattr(connection.websocket, "close"):
                await connection.websocket.close()
        except Exception:
            pass

        logger.info(f"Client disconnected: {client_id} ({reason})")

    def subscribe(self, client_id: str, topics: List[StreamTopic]) -> bool:
        """
        Subscribe a client to topics.

        Args:
            client_id: Client ID
            topics: Topics to subscribe to

        Returns:
            True if successful
        """
        if client_id not in self.clients:
            return False

        connection = self.clients[client_id]
        for topic in topics:
            connection.subscriptions.add(topic)
            logger.debug(f"Client {client_id} subscribed to {topic.value}")

        return True

    def unsubscribe(self, client_id: str, topics: List[StreamTopic]) -> bool:
        """
        Unsubscribe a client from topics.

        Args:
            client_id: Client ID
            topics: Topics to unsubscribe from

        Returns:
            True if successful
        """
        if client_id not in self.clients:
            return False

        connection = self.clients[client_id]
        for topic in topics:
            # Don't allow unsubscribing from system/heartbeat
            if topic not in (StreamTopic.SYSTEM, StreamTopic.HEARTBEAT):
                connection.subscriptions.discard(topic)
                logger.debug(f"Client {client_id} unsubscribed from {topic.value}")

        return True

    async def broadcast(self, message: StreamMessage):
        """
        Broadcast a message to all subscribed clients.

        Args:
            message: Message to broadcast
        """
        async with self._broadcast_lock:
            # Rate limiting
            topic = message.topic
            now = time.time()
            last_time = self._last_broadcast_time.get(topic, 0)

            if now - last_time < 1.0 / self.message_rate_limit:
                # Rate limited - queue for later
                self.stats["total_messages_dropped"] += 1
                return

            self._last_broadcast_time[topic] = now

        # Store in history
        self.topic_history[topic].append(message)

        # Find subscribed clients
        subscribed_clients = [
            client_id for client_id, conn in self.clients.items() if topic in conn.subscriptions
        ]

        # Send to all subscribed clients
        tasks = [self._send_to_client(client_id, message) for client_id in subscribed_clients]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.stats["total_messages_sent"] += len(subscribed_clients)

    async def send_to_client(self, client_id: str, message: StreamMessage) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: Target client
            message: Message to send

        Returns:
            True if sent successfully
        """
        return await self._send_to_client(client_id, message)

    async def _send_to_client(self, client_id: str, message: StreamMessage) -> bool:
        """Internal method to send message to a client."""
        if client_id not in self.clients:
            return False

        connection = self.clients[client_id]

        try:
            if hasattr(connection.websocket, "send"):
                await connection.websocket.send(message.to_json())
            elif hasattr(connection.websocket, "send_text"):
                await connection.websocket.send_text(message.to_json())
            else:
                logger.warning(f"Unknown websocket type for client {client_id}")
                return False

            connection.update_activity()
            return True

        except Exception as e:
            logger.warning(f"Failed to send to client {client_id}: {e}")
            # Queue for retry or disconnect
            connection.message_queue.append(message)
            return False

    async def _send_topic_history(self, client_id: str):
        """Send recent history for subscribed topics to a new client."""
        if client_id not in self.clients:
            return

        connection = self.clients[client_id]

        for topic in connection.subscriptions:
            if topic in (StreamTopic.SYSTEM, StreamTopic.HEARTBEAT):
                continue

            history = list(self.topic_history.get(topic, []))
            if history:
                # Send last 5 messages from history
                for msg in history[-5:]:
                    await self._send_to_client(client_id, msg)
                    await asyncio.sleep(0.01)  # Small delay between messages

    async def _heartbeat_loop(self):
        """Background task to send heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                message = StreamMessage(
                    topic=StreamTopic.HEARTBEAT,
                    data={
                        "server_time": datetime.now().isoformat(),
                        "connections": len(self.clients),
                    },
                )

                await self.broadcast(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.now()
                stale_clients = []

                for client_id, conn in self.clients.items():
                    inactive_seconds = (now - conn.last_activity).total_seconds()
                    if inactive_seconds > self.connection_timeout:
                        stale_clients.append(client_id)

                for client_id in stale_clients:
                    await self.disconnect_client(client_id, reason="Timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def on_connect(self, callback: Callable):
        """Register a callback for client connections."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """Register a callback for client disconnections."""
        self._on_disconnect_callbacks.append(callback)

    def on_message(self, callback: Callable):
        """Register a callback for incoming messages."""
        self._on_message_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get hub statistics."""
        return {
            **self.stats,
            "topics": {topic.value: len(history) for topic, history in self.topic_history.items()},
        }


class TradingDataStreamer:
    """
    Streams trading data to connected clients via WebSocket.

    This class bridges the trading system with the WebSocket hub,
    formatting and sending various types of trading updates.
    """

    def __init__(self, hub: WebSocketHub):
        """
        Initialize the streamer.

        Args:
            hub: WebSocket hub for broadcasting
        """
        self.hub = hub
        self._last_portfolio_state: Optional[Dict] = None
        self._last_positions: Dict[str, Dict] = {}

        logger.info("TradingDataStreamer initialized")

    async def stream_portfolio_update(
        self,
        total_value: float,
        cash: float,
        positions_value: float,
        pnl: float,
        pnl_percent: float,
        drawdown: float,
        risk_level: str = "normal",
    ):
        """Stream portfolio state update."""
        data = {
            "total_value": total_value,
            "cash": cash,
            "positions_value": positions_value,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "drawdown": drawdown,
            "risk_level": risk_level,
        }

        # Only send if changed
        if data != self._last_portfolio_state:
            self._last_portfolio_state = data
            await self.hub.broadcast(StreamMessage(topic=StreamTopic.PORTFOLIO, data=data))

    async def stream_position_update(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        pnl_percent: float,
        side: str = "long",
    ):
        """Stream position update."""
        data = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "side": side,
        }

        # Only send if changed
        if self._last_positions.get(symbol) != data:
            self._last_positions[symbol] = data
            await self.hub.broadcast(StreamMessage(topic=StreamTopic.POSITIONS, data=data))

    async def stream_trade_execution(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
        pnl: Optional[float] = None,
    ):
        """Stream trade execution notification."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.TRADES,
                data={
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "strategy": strategy,
                    "pnl": pnl,
                    "executed_at": datetime.now().isoformat(),
                },
            )
        )

    async def stream_order_update(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str,
        fill_quantity: float = 0.0,
    ):
        """Stream order status update."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.ORDERS,
                data={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "status": status,
                    "fill_quantity": fill_quantity,
                    "updated_at": datetime.now().isoformat(),
                },
            )
        )

    async def stream_risk_alert(
        self,
        alert_type: str,
        level: AlertLevel,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
        action_taken: Optional[str] = None,
    ):
        """Stream risk alert."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.RISK,
                data={
                    "alert_type": alert_type,
                    "level": level.value,
                    "message": message,
                    "metrics": metrics or {},
                    "action_taken": action_taken,
                },
            )
        )

    async def stream_strategy_signal(
        self,
        strategy_name: str,
        symbol: str,
        signal: str,
        strength: float,
        confidence: float,
        reasons: Optional[List[str]] = None,
    ):
        """Stream strategy signal."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.SIGNALS,
                data={
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "signal": signal,
                    "strength": strength,
                    "confidence": confidence,
                    "reasons": reasons or [],
                },
            )
        )

    async def stream_regime_change(
        self,
        previous_regime: str,
        new_regime: str,
        confidence: float,
        indicators: Optional[Dict[str, float]] = None,
    ):
        """Stream market regime change."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.REGIME,
                data={
                    "previous_regime": previous_regime,
                    "new_regime": new_regime,
                    "confidence": confidence,
                    "indicators": indicators or {},
                    "changed_at": datetime.now().isoformat(),
                },
            )
        )

    async def stream_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel,
        category: str = "general",
        action_required: bool = False,
        dismiss_after: Optional[int] = None,
    ):
        """Stream general alert."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.ALERTS,
                data={
                    "title": title,
                    "message": message,
                    "level": level.value,
                    "category": category,
                    "action_required": action_required,
                    "dismiss_after": dismiss_after,
                },
            )
        )

    async def stream_performance_update(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        trade_count: int,
        period: str = "all_time",
    ):
        """Stream performance metrics update."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.PERFORMANCE,
                data={
                    "total_return": total_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "trade_count": trade_count,
                    "period": period,
                },
            )
        )

    async def stream_system_status(
        self,
        status: str,
        trading_enabled: bool,
        connected_exchanges: List[str],
        active_strategies: List[str],
        warnings: Optional[List[str]] = None,
    ):
        """Stream system status update."""
        await self.hub.broadcast(
            StreamMessage(
                topic=StreamTopic.SYSTEM,
                data={
                    "status": status,
                    "trading_enabled": trading_enabled,
                    "connected_exchanges": connected_exchanges,
                    "active_strategies": active_strategies,
                    "warnings": warnings or [],
                    "uptime_seconds": self._get_uptime(),
                },
            )
        )

    def _get_uptime(self) -> float:
        """Get system uptime (placeholder)."""
        return 0.0


class WebSocketMessageHandler:
    """
    Handles incoming WebSocket messages from clients.

    Processes commands like subscribe, unsubscribe, and various queries.
    """

    def __init__(self, hub: WebSocketHub, streamer: TradingDataStreamer):
        """
        Initialize the message handler.

        Args:
            hub: WebSocket hub
            streamer: Trading data streamer
        """
        self.hub = hub
        self.streamer = streamer

        # Command handlers
        self._handlers: Dict[str, Callable] = {
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "ping": self._handle_ping,
            "get_portfolio": self._handle_get_portfolio,
            "get_positions": self._handle_get_positions,
            "get_performance": self._handle_get_performance,
            "kill_switch": self._handle_kill_switch,
            "adjust_risk": self._handle_adjust_risk,
        }

        # External handlers for trading operations
        self._kill_switch_handler: Optional[Callable] = None
        self._risk_adjust_handler: Optional[Callable] = None
        self._portfolio_getter: Optional[Callable] = None
        self._positions_getter: Optional[Callable] = None

        logger.info("WebSocketMessageHandler initialized")

    def set_kill_switch_handler(self, handler: Callable):
        """Set handler for kill switch command."""
        self._kill_switch_handler = handler

    def set_risk_adjust_handler(self, handler: Callable):
        """Set handler for risk adjustment command."""
        self._risk_adjust_handler = handler

    def set_portfolio_getter(self, getter: Callable):
        """Set getter for portfolio data."""
        self._portfolio_getter = getter

    def set_positions_getter(self, getter: Callable):
        """Set getter for positions data."""
        self._positions_getter = getter

    async def handle_message(self, client_id: str, raw_message: str):
        """
        Handle an incoming message from a client.

        Args:
            client_id: Client that sent the message
            raw_message: Raw message string
        """
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_error(client_id, "Invalid JSON")
            return

        command = message.get("command", message.get("type"))
        if not command:
            await self._send_error(client_id, "Missing command")
            return

        handler = self._handlers.get(command)
        if not handler:
            await self._send_error(client_id, f"Unknown command: {command}")
            return

        try:
            await handler(client_id, message)
        except Exception as e:
            logger.error(f"Error handling {command}: {e}")
            await self._send_error(client_id, str(e))

    async def _send_error(self, client_id: str, error: str):
        """Send error message to client."""
        await self.hub.send_to_client(
            client_id,
            StreamMessage(topic=StreamTopic.SYSTEM, data={"type": "error", "error": error}),
        )

    async def _send_response(self, client_id: str, command: str, data: Dict):
        """Send command response to client."""
        await self.hub.send_to_client(
            client_id,
            StreamMessage(
                topic=StreamTopic.SYSTEM,
                data={"type": "response", "command": command, "data": data},
            ),
        )

    async def _handle_subscribe(self, client_id: str, message: Dict):
        """Handle subscribe command."""
        topics_raw = message.get("topics", [])
        topics = []

        for t in topics_raw:
            try:
                topics.append(StreamTopic(t))
            except ValueError:
                pass

        if topics:
            self.hub.subscribe(client_id, topics)
            await self._send_response(
                client_id, "subscribe", {"subscribed": [t.value for t in topics]}
            )

    async def _handle_unsubscribe(self, client_id: str, message: Dict):
        """Handle unsubscribe command."""
        topics_raw = message.get("topics", [])
        topics = []

        for t in topics_raw:
            try:
                topics.append(StreamTopic(t))
            except ValueError:
                pass

        if topics:
            self.hub.unsubscribe(client_id, topics)
            await self._send_response(
                client_id, "unsubscribe", {"unsubscribed": [t.value for t in topics]}
            )

    async def _handle_ping(self, client_id: str, message: Dict):
        """Handle ping command."""
        await self._send_response(client_id, "pong", {"server_time": datetime.now().isoformat()})

    async def _handle_get_portfolio(self, client_id: str, message: Dict):
        """Handle get_portfolio command."""
        if self._portfolio_getter:
            portfolio = await self._portfolio_getter()
            await self._send_response(client_id, "get_portfolio", portfolio)
        else:
            await self._send_error(client_id, "Portfolio getter not configured")

    async def _handle_get_positions(self, client_id: str, message: Dict):
        """Handle get_positions command."""
        if self._positions_getter:
            positions = await self._positions_getter()
            await self._send_response(client_id, "get_positions", {"positions": positions})
        else:
            await self._send_error(client_id, "Positions getter not configured")

    async def _handle_get_performance(self, client_id: str, message: Dict):
        """Handle get_performance command."""
        period = message.get("period", "all_time")
        # This would be connected to actual performance tracking
        await self._send_response(client_id, "get_performance", {"period": period, "data": {}})

    async def _handle_kill_switch(self, client_id: str, message: Dict):
        """Handle kill switch command (emergency stop)."""
        if not self._kill_switch_handler:
            await self._send_error(client_id, "Kill switch not configured")
            return

        # Verify authentication (should be required for this operation)
        if client_id in self.hub.clients:
            client = self.hub.clients[client_id]
            if not client.is_authenticated:
                await self._send_error(client_id, "Authentication required for kill switch")
                return

        action = message.get("action", "stop")  # stop, resume
        reason = message.get("reason", "Manual kill switch")

        try:
            result = await self._kill_switch_handler(action, reason)
            await self._send_response(
                client_id,
                "kill_switch",
                {"action": action, "success": result, "message": f"Kill switch {action} executed"},
            )

            # Broadcast alert to all clients
            await self.streamer.stream_alert(
                title="Kill Switch Activated" if action == "stop" else "Trading Resumed",
                message=reason,
                level=AlertLevel.CRITICAL if action == "stop" else AlertLevel.WARNING,
                category="kill_switch",
                action_required=False,
            )

        except Exception as e:
            await self._send_error(client_id, f"Kill switch failed: {e}")

    async def _handle_adjust_risk(self, client_id: str, message: Dict):
        """Handle risk adjustment command."""
        if not self._risk_adjust_handler:
            await self._send_error(client_id, "Risk adjustment not configured")
            return

        # Verify authentication
        if client_id in self.hub.clients:
            client = self.hub.clients[client_id]
            if not client.is_authenticated:
                await self._send_error(client_id, "Authentication required")
                return

        adjustment_type = message.get("type")  # position_size, max_drawdown, etc.
        new_value = message.get("value")

        if adjustment_type is None or new_value is None:
            await self._send_error(client_id, "Missing type or value")
            return

        try:
            result = await self._risk_adjust_handler(adjustment_type, new_value)
            await self._send_response(
                client_id,
                "adjust_risk",
                {"type": adjustment_type, "value": new_value, "success": result},
            )

        except Exception as e:
            await self._send_error(client_id, f"Risk adjustment failed: {e}")


# FastAPI/Starlette WebSocket integration helper
def create_websocket_endpoint(hub: WebSocketHub, handler: WebSocketMessageHandler):
    """
    Create a WebSocket endpoint handler for FastAPI/Starlette.

    Usage:
        app = FastAPI()
        hub = WebSocketHub()
        handler = WebSocketMessageHandler(hub, streamer)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await handle_websocket(websocket, hub, handler)
    """

    async def handle_websocket(websocket):
        """Handle a WebSocket connection."""
        # Accept connection
        client_id = await hub.connect_client(
            websocket,
            initial_subscriptions=[
                StreamTopic.PORTFOLIO,
                StreamTopic.POSITIONS,
                StreamTopic.ALERTS,
            ],
        )

        if not client_id:
            return

        try:
            while True:
                # Receive messages
                if hasattr(websocket, "receive_text"):
                    message = await websocket.receive_text()
                elif hasattr(websocket, "recv"):
                    message = await websocket.recv()
                else:
                    break

                await handler.handle_message(client_id, message)

        except Exception as e:
            logger.warning(f"WebSocket error for {client_id}: {e}")
        finally:
            await hub.disconnect_client(client_id, reason="Connection closed")

    return handle_websocket


if __name__ == "__main__":
    # Demo usage
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def demo():
        # Create hub and streamer
        hub = WebSocketHub(heartbeat_interval=10.0, max_clients=50)
        streamer = TradingDataStreamer(hub)
        handler = WebSocketMessageHandler(hub, streamer)

        await hub.start()

        print("=== WebSocket Streaming Demo ===")
        print(f"Hub stats: {hub.get_stats()}")

        # Simulate some trading updates
        await streamer.stream_portfolio_update(
            total_value=100000,
            cash=50000,
            positions_value=50000,
            pnl=1500,
            pnl_percent=1.5,
            drawdown=0.02,
        )

        await streamer.stream_trade_execution(
            trade_id="trade_001",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=45000,
            strategy="momentum",
        )

        await streamer.stream_risk_alert(
            alert_type="drawdown_warning",
            level=AlertLevel.WARNING,
            message="Drawdown approaching 5% threshold",
            metrics={"current_drawdown": 0.045, "threshold": 0.05},
        )

        await streamer.stream_regime_change(
            previous_regime="trending_bullish",
            new_regime="mean_reverting",
            confidence=0.85,
            indicators={"volatility": 0.02, "trend_strength": 0.3},
        )

        print(f"\nMessages in history:")
        for topic, history in hub.topic_history.items():
            if history:
                print(f"  {topic.value}: {len(history)} messages")

        print(f"\nFinal stats: {hub.get_stats()}")

        await hub.stop()

    asyncio.run(demo())
