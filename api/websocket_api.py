"""
WebSocket API for Real-Time Dashboard Updates.

Provides WebSocket connections for:
- Real-time price updates
- Trade notifications
- Position updates
- Equity changes
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Connection manager
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: asyncio.Task = None

    async def connect(self, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections."""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected
        self.active_connections -= disconnected

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


manager = ConnectionManager()


# Data directory
DATA_DIR = Path("data/unified_trading")


async def get_dashboard_data() -> Dict[str, Any]:
    """Get current dashboard data."""
    state_file = DATA_DIR / "state.json"
    equity_file = DATA_DIR / "equity.json"
    trades_file = DATA_DIR / "trades.json"

    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    # Recent equity points
    equity = []
    if equity_file.exists():
        with open(equity_file) as f:
            all_equity = json.load(f)
            equity = all_equity[-100:]  # Last 100 points

    # Recent trades
    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            all_trades = json.load(f)
            trades = all_trades[-20:]  # Last 20 trades

    return {
        "type": "dashboard",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "balance": state.get("balance", 0),
            "initial_capital": state.get("initial_capital", 10000),
            "total_pnl": state.get("balance", 0) - state.get("initial_capital", 10000),
            "positions": state.get("positions", {}),
            "open_positions_count": len(state.get("positions", {})),
            "equity_curve": equity,
            "recent_trades": trades,
            "mode": state.get("mode", "paper"),
            "running": state.get("running", False),
        }
    }


async def get_prices() -> Dict[str, float]:
    """Get current prices for tracked symbols."""
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
        prices = {}

        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                prices[symbol] = ticker['last']
            except (KeyError, TypeError) as e:
                logger.debug(f"Failed to get price for {symbol}: {e}")

        return prices
    except Exception as e:
        logger.warning(f"Failed to get prices: {e}")
        return {}


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    Sends updates every second with:
    - Current prices
    - Balance and P&L
    - Position updates
    - Recent trades
    """
    await manager.connect(websocket)

    try:
        # Send initial data
        initial_data = await get_dashboard_data()
        await manager.send_personal(websocket, initial_data)

        # Keep connection alive and send updates
        last_data_hash = ""

        while True:
            try:
                # Wait for message or timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0
                    )
                    # Handle client messages
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await manager.send_personal(websocket, {"type": "pong"})
                except asyncio.TimeoutError:
                    pass

                # Get current data
                dashboard_data = await get_dashboard_data()

                # Only send if changed (simple hash check)
                data_hash = str(hash(json.dumps(dashboard_data["data"], sort_keys=True)))
                if data_hash != last_data_hash:
                    await manager.send_personal(websocket, dashboard_data)
                    last_data_hash = data_hash

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WebSocket error: {e}")
                await asyncio.sleep(1)

    finally:
        manager.disconnect(websocket)


@router.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price updates.

    Sends price updates every 2 seconds.
    """
    await manager.connect(websocket)

    try:
        while True:
            try:
                prices = await get_prices()
                await manager.send_personal(websocket, {
                    "type": "prices",
                    "timestamp": datetime.now().isoformat(),
                    "prices": prices,
                })
                await asyncio.sleep(2)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Price WebSocket error: {e}")
                await asyncio.sleep(5)

    finally:
        manager.disconnect(websocket)


@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """
    WebSocket endpoint for trade notifications.

    Sends notifications when new trades occur.
    """
    await manager.connect(websocket)

    trades_file = DATA_DIR / "trades.json"
    last_trade_count = 0

    # Get initial count
    if trades_file.exists():
        with open(trades_file) as f:
            last_trade_count = len(json.load(f))

    try:
        while True:
            try:
                # Check for new trades
                if trades_file.exists():
                    with open(trades_file) as f:
                        trades = json.load(f)

                    if len(trades) > last_trade_count:
                        # New trades!
                        new_trades = trades[last_trade_count:]
                        last_trade_count = len(trades)

                        await manager.send_personal(websocket, {
                            "type": "new_trades",
                            "timestamp": datetime.now().isoformat(),
                            "trades": new_trades,
                        })

                await asyncio.sleep(1)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Trade WebSocket error: {e}")
                await asyncio.sleep(5)

    finally:
        manager.disconnect(websocket)


# Broadcast functions for external use
async def broadcast_trade(trade: Dict[str, Any]):
    """Broadcast new trade to all connected clients."""
    await manager.broadcast({
        "type": "trade",
        "timestamp": datetime.now().isoformat(),
        "trade": trade,
    })


async def broadcast_position_update(position: Dict[str, Any]):
    """Broadcast position update to all connected clients."""
    await manager.broadcast({
        "type": "position_update",
        "timestamp": datetime.now().isoformat(),
        "position": position,
    })


async def broadcast_alert(alert_type: str, message: str, data: Dict[str, Any] = None):
    """Broadcast alert to all connected clients."""
    await manager.broadcast({
        "type": "alert",
        "alert_type": alert_type,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data or {},
    })
