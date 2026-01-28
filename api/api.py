from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Load .env before any other imports that might need env vars
from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from bot.ai import FeatureSnapshot, PredictionSnapshot, QuestionAnsweringEngine
from bot.control import load_bot_control, update_bot_control
from bot.macro import MacroInsight
from bot.state import StateStore, create_state_store, load_bot_state_from_path
from bot.strategy import StrategyConfig
from bot.market_data import sanitize_symbol_for_fs
from bot.config_loader import load_overrides, merge_config

from .schemas import (
    AIAnswerResponse,
    AIQuestionRequest,
    AIPredictionResponse,
    BotStateResponse,
    EquityPointResponse,
    HealthCheckResponse,
    MacroEventResponse,
    MacroInsightResponse,
    PortfolioBotStatusResponse,
    PortfolioControlStateResponse,
    PortfolioControlUpdateRequest,
    PortfolioPlaybookResponse,
    SignalResponse,
    StrategyOverviewResponse,
)
from .security import check_rate_limit, verify_api_key
from .unified_trading_api import router as unified_trading_router
from .validation import validate_trading_request, APIRequestValidator

STATE_DIR = Path(os.getenv("DATA_DIR", "./data"))

# Monotonic version counter for dashboard snapshots
# Strictly increasing, survives within process lifetime
_snapshot_version_counter = int(time.time() * 1000)  # Start from current ms timestamp

# Initialize logger
logger = logging.getLogger(__name__)

# Simple response cache for market summaries (to avoid slow API calls)
_market_summary_cache: Dict[str, tuple] = {}  # {market_type: (timestamp, data)}
_CACHE_TTL = 10.0  # Cache for 10 seconds
PAPER_TRADING_LOG = STATE_DIR / "logs" / "paper_trading.log"
# All trading log sources for aggregated view
TRADING_LOG_SOURCES = {
    "crypto": STATE_DIR / "logs" / "paper_trading.log",
    "commodity": STATE_DIR / "logs" / "commodity_trading.log",
    "stock": STATE_DIR / "logs" / "stock_trading.log",
    "regime": STATE_DIR / "logs" / "regime_paper_trading.log",
}
PORTFOLIO_CONFIG_PATH = Path(
    os.getenv("PORTFOLIO_CONFIG_PATH", str(STATE_DIR / "portfolio.json"))
).expanduser()

API_DESCRIPTION = """
# Algo Trading Lab API

A comprehensive algorithmic trading platform API supporting:

## Features

- **Multi-Market Trading**: Crypto, Stocks, Commodities
- **ML-Based Predictions**: LSTM, Transformer, XGBoost models
- **Portfolio Optimization**: Risk parity, mean-variance, minimum volatility
- **Real-Time WebSocket**: Live updates for dashboards
- **Backtesting**: Historical strategy testing

## Authentication

Most endpoints require an API key passed via `X-API-Key` header.
Set your API key in the `.env` file:
```
API_KEY=your_secret_api_key
```

## Rate Limiting

API requests are rate limited to prevent abuse:
- 100 requests per minute per IP
- 1000 requests per hour per API key

## WebSocket

Connect to `/ws/updates` for real-time data updates.

## Quick Start

```bash
# Get bot status
curl -H "X-API-Key: your_key" http://localhost:8000/status

# Get ML prediction
curl -H "X-API-Key: your_key" http://localhost:8000/ml/prediction?symbol=BTC/USDT
```
"""

API_TAGS_METADATA = [
    {
        "name": "Status",
        "description": "Bot status, signals, and equity curve endpoints",
    },
    {
        "name": "Portfolio",
        "description": "Portfolio management, controls, and optimization",
    },
    {
        "name": "ML/AI",
        "description": "Machine learning predictions, regime detection, and training",
    },
    {
        "name": "Markets",
        "description": "Multi-market data (Crypto, Stocks, Commodities)",
    },
    {
        "name": "Dashboard",
        "description": "Web dashboard and visualization endpoints",
    },
    {
        "name": "Health",
        "description": "Health checks and monitoring",
    },
]


# =============================================================================
# WebSocket Connection Manager for Real-Time Updates
# =============================================================================


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        disconnected: Set[WebSocket] = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

    def get_connection_count(self) -> int:
        """Return the number of active connections."""
        return len(self.active_connections)


# Global connection manager instance
ws_manager = ConnectionManager()


def _get_ws_update_payload() -> Dict[str, Any]:
    """
    Build the WebSocket update payload with portfolio data from UNIFIED trading state.

    CRITICAL: This MUST use the EXACT same calculations as /api/dashboard/unified-state
    to ensure data consistency across all dashboard components.
    """
    # Symbol to market mapping - MUST match the unified endpoint
    symbol_to_market = {
        "BTC/USDT": "crypto", "ETH/USDT": "crypto", "SOL/USDT": "crypto",
        "XRP/USDT": "crypto", "ADA/USDT": "crypto", "AVAX/USDT": "crypto",
        "DOGE/USDT": "crypto", "DOT/USDT": "crypto", "LINK/USDT": "crypto",
        "MATIC/USDT": "crypto", "LTC/USDT": "crypto", "UNI/USDT": "crypto",
        "XAU/USD": "commodity", "XAG/USD": "commodity", "USOIL/USD": "commodity",
        "NATGAS/USD": "commodity", "WTICO/USD": "commodity",
        "AAPL": "stock", "MSFT": "stock", "TSLA": "stock", "NVDA": "stock",
        "GOOGL": "stock", "AMZN": "stock", "META": "stock",
        "AAPL/USD": "stock", "MSFT/USD": "stock", "GOOGL/USD": "stock", "NVDA/USD": "stock",
    }

    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "portfolio_update",
        "prices": {},
        "signals": {},
        "portfolio_value": 0,
        "positions": {},
        "markets": {},
    }

    # Load UNIFIED trading state (single source of truth)
    unified_state_file = STATE_DIR / "unified_trading" / "state.json"

    if not unified_state_file.exists():
        result["markets"] = {
            "crypto": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
            "commodity": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
            "stock": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
        }
        return result

    try:
        with open(unified_state_file, "r") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        result["markets"] = {
            "crypto": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
            "commodity": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
            "stock": {"status": "not_running", "total_value": 10000, "pnl": 0, "pnl_pct": 0, "positions_count": 0},
        }
        return result

    # Extract unified state data
    initial_capital = float(state.get("initial_capital", 30000))
    positions = state.get("positions", {})

    # Calculate market-specific P&L from positions (SAME AS UNIFIED ENDPOINT)
    market_pnl = {"crypto": 0.0, "commodity": 0.0, "stock": 0.0}
    market_position_count = {"crypto": 0, "commodity": 0, "stock": 0}

    for symbol, pos_data in positions.items():
        market = symbol_to_market.get(symbol, "crypto")  # Default to crypto
        unrealized_pnl = float(pos_data.get("unrealized_pnl", 0.0))
        market_pnl[market] += unrealized_pnl
        market_position_count[market] += 1
        result["positions"][symbol] = {
            "value": pos_data.get("value", 0),
            "quantity": pos_data.get("quantity", pos_data.get("qty", 0)),
            "entry_price": pos_data.get("entry_price", 0),
            "unrealized_pnl": unrealized_pnl,
        }

    # Calculate per-market values (SAME AS UNIFIED ENDPOINT)
    capital_per_market = initial_capital / 3
    total_pnl = sum(market_pnl.values())

    # Portfolio totals
    total_value = sum(capital_per_market + market_pnl[m] for m in ["crypto", "commodity", "stock"])
    result["portfolio_value"] = total_value
    result["total_pnl"] = total_pnl
    result["total_pnl_pct"] = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

    # Market status
    is_active = state.get("status") == "active"

    # Build market data (SAME CALCULATION AS UNIFIED ENDPOINT)
    for market_id in ["crypto", "commodity", "stock"]:
        pnl = market_pnl[market_id]
        market_value = capital_per_market + pnl
        pnl_pct = (pnl / capital_per_market * 100) if capital_per_market > 0 else 0

        result["markets"][market_id] = {
            "total_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cash_balance": capital_per_market,
            "positions_count": market_position_count[market_id],
            "deep_learning_enabled": False if market_id != "crypto" else False,
            "dl_model_selection": None,
            "status": "running" if is_active else "stopped",
        }

    return result


async def _ws_broadcast_loop() -> None:
    """Background task that broadcasts updates every 3 seconds."""
    while ws_manager._running:
        if ws_manager.get_connection_count() > 0:
            try:
                payload = _get_ws_update_payload()
                await ws_manager.broadcast(payload)
            except Exception:
                pass  # Silently handle broadcast errors
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Start the WebSocket broadcast loop
    ws_manager._running = True
    ws_manager._broadcast_task = asyncio.create_task(_ws_broadcast_loop())

    yield

    # Shutdown: Stop the WebSocket broadcast loop
    ws_manager._running = False
    if ws_manager._broadcast_task:
        ws_manager._broadcast_task.cancel()
        try:
            await ws_manager._broadcast_task
        except asyncio.CancelledError:
            pass

    # Close all active connections
    for connection in list(ws_manager.active_connections):
        try:
            await connection.close()
        except Exception:
            pass


app = FastAPI(
    title="Algo Trading Lab API",
    version="0.3.0",
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Algo Trading Lab",
        "url": "https://github.com/ogulcanaydogan/algo_trading_lab",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# CORS middleware configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
# Ensure wildcard works properly
if "*" in CORS_ORIGINS:
    CORS_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include unified trading API router
app.include_router(unified_trading_router)

# Include advanced API router
try:
    from api.advanced_api import router as advanced_router

    app.include_router(advanced_router)
except ImportError as e:
    logger.warning(f"Advanced API not available: {e}")

# Include WebSocket router
try:
    from api.websocket_api import router as websocket_router

    app.include_router(websocket_router)
except ImportError as e:
    logger.warning(f"WebSocket API not available: {e}")

# Track application start time for health checks
_app_start_time = time.time()

_state_store: Optional[StateStore] = None


def get_store() -> StateStore:
    global _state_store
    if _state_store is None:
        _state_store = create_state_store(STATE_DIR)
    return _state_store


def _load_portfolio_config_payload() -> Dict[str, Any]:
    """Best-effort loader for the portfolio.json definition file."""

    if not PORTFOLIO_CONFIG_PATH.exists():
        return {}
    try:
        with PORTFOLIO_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _build_asset_index(config_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    asset_map: Dict[str, Dict[str, Any]] = {}
    for asset in config_payload.get("assets", []):
        symbol_value = str(asset.get("symbol") or "").strip()
        if not symbol_value:
            continue
        asset_map[symbol_value.upper()] = asset
    return asset_map


def _resolve_asset_data_dir(
    symbol_key: str,
    asset_map: Dict[str, Dict[str, Any]],
    portfolio_dir: Path,
) -> Path:
    asset = asset_map.get(symbol_key)
    if asset and asset.get("data_dir"):
        return Path(asset["data_dir"]).expanduser()
    symbol_value = asset.get("symbol") if asset else symbol_key
    safe = sanitize_symbol_for_fs(str(symbol_value))
    return portfolio_dir / safe


def _derive_asset_metadata(
    asset: Optional[Dict[str, Any]],
    *,
    default_timeframe: Optional[str],
    default_loop_interval: Optional[int],
    default_paper_mode: Optional[bool],
    default_stop_loss_pct: Optional[float],
    default_take_profit_pct: Optional[float],
) -> Dict[str, Any]:
    """Return a dict of optional metadata for dashboard consumption."""

    if not asset:
        return {}

    allocation_value = asset.get("allocation_pct")
    loop_value = asset.get("loop_interval_seconds")
    if loop_value is not None:
        try:
            loop_value = int(loop_value)
        except (TypeError, ValueError):
            loop_value = None
    elif default_loop_interval is not None:
        loop_value = default_loop_interval

    paper_mode_value = asset.get("paper_mode")
    if paper_mode_value is None:
        paper_mode_value = default_paper_mode
    if paper_mode_value is not None:
        paper_mode_value = bool(paper_mode_value)

    timeframe_value = asset.get("timeframe") or default_timeframe
    stop_loss_value = asset.get("stop_loss_pct")
    if stop_loss_value is None:
        stop_loss_value = default_stop_loss_pct
    take_profit_value = asset.get("take_profit_pct")
    if take_profit_value is None:
        take_profit_value = default_take_profit_pct

    return {
        "asset_type": asset.get("asset_type"),
        "timeframe": timeframe_value,
        "allocation_pct": float(allocation_value) if allocation_value is not None else None,
        "paper_mode": paper_mode_value,
        "loop_interval_seconds": loop_value,
        "stop_loss_pct": float(stop_loss_value) if stop_loss_value is not None else None,
        "take_profit_pct": float(take_profit_value) if take_profit_value is not None else None,
    }


def load_portfolio_states(
    symbol: Optional[str] = None,
) -> List[PortfolioBotStatusResponse]:
    """Load per-asset states from DATA_DIR/portfolio and config fallbacks."""

    portfolio_dir = STATE_DIR / "portfolio"
    symbol_filter = symbol.upper() if symbol else None

    config_payload = _load_portfolio_config_payload()
    asset_map = _build_asset_index(config_payload)
    default_risk_pct = config_payload.get("default_risk_per_trade_pct")
    default_risk_pct = float(default_risk_pct) if default_risk_pct is not None else None
    default_timeframe = config_payload.get("default_timeframe")
    default_loop_interval = config_payload.get("default_loop_interval_seconds")
    default_loop_interval = (
        int(default_loop_interval) if default_loop_interval is not None else None
    )
    default_paper_mode = config_payload.get("default_paper_mode")
    default_stop_loss_pct = config_payload.get("default_stop_loss_pct")
    default_stop_loss_pct = (
        float(default_stop_loss_pct) if default_stop_loss_pct is not None else None
    )
    default_take_profit_pct = config_payload.get("default_take_profit_pct")
    default_take_profit_pct = (
        float(default_take_profit_pct) if default_take_profit_pct is not None else None
    )
    portfolio_capital = float(config_payload.get("portfolio_capital", 0.0))

    results: Dict[str, PortfolioBotStatusResponse] = {}

    if portfolio_dir.exists():
        for item in sorted(portfolio_dir.iterdir()):
            if not item.is_dir():
                continue
            state_file = item / "state.json"
            state = load_bot_state_from_path(state_file)
            if state is None:
                continue
            symbol_value = state.symbol or ""
            symbol_key = symbol_value.upper()
            if symbol_filter and symbol_key != symbol_filter:
                continue
            try:
                relative_dir = item.relative_to(STATE_DIR)
            except ValueError:
                relative_dir = item

            metadata = _derive_asset_metadata(
                asset_map.get(symbol_key),
                default_timeframe=default_timeframe,
                default_loop_interval=default_loop_interval,
                default_paper_mode=default_paper_mode,
                default_stop_loss_pct=default_stop_loss_pct,
                default_take_profit_pct=default_take_profit_pct,
            )
            control = load_bot_control(item)
            results[symbol_key] = PortfolioBotStatusResponse(
                timestamp=state.timestamp,
                symbol=state.symbol,
                position=state.position,
                position_size=state.position_size,
                balance=state.balance,
                initial_balance=state.initial_balance or state.balance,
                entry_price=state.entry_price,
                unrealized_pnl_pct=state.unrealized_pnl_pct,
                last_signal=state.last_signal,
                confidence=state.confidence,
                risk_per_trade_pct=state.risk_per_trade_pct,
                ai_action=state.ai_action,
                ai_confidence=state.ai_confidence,
                data_directory=str(relative_dir),
                is_paused=control.paused,
                pause_reason=control.reason,
                pause_updated_at=control.updated_at,
                is_placeholder=False,
                **metadata,
            )

    for symbol_key, asset in asset_map.items():
        if symbol_filter and symbol_key != symbol_filter:
            continue
        if symbol_key in results:
            continue

        data_dir_path = _resolve_asset_data_dir(
            symbol_key,
            asset_map,
            portfolio_dir,
        )
        try:
            relative_dir = data_dir_path.relative_to(STATE_DIR)
        except ValueError:
            relative_dir = data_dir_path

        allocation_pct = asset.get("allocation_pct")
        starting_balance = asset.get("starting_balance")
        if starting_balance is None and allocation_pct is not None and portfolio_capital:
            starting_balance = (portfolio_capital * float(allocation_pct)) / 100.0
        balance_value = float(starting_balance) if starting_balance is not None else 0.0

        risk_value = asset.get("risk_per_trade_pct")
        if risk_value is None:
            risk_value = default_risk_pct
        risk_value = float(risk_value) if risk_value is not None else 0.0

        metadata = _derive_asset_metadata(
            asset,
            default_timeframe=default_timeframe,
            default_loop_interval=default_loop_interval,
            default_paper_mode=default_paper_mode,
            default_stop_loss_pct=default_stop_loss_pct,
            default_take_profit_pct=default_take_profit_pct,
        )
        control = load_bot_control(data_dir_path)

        results[symbol_key] = PortfolioBotStatusResponse(
            timestamp=datetime.now(timezone.utc),
            symbol=asset["symbol"],
            position="FLAT",
            position_size=0.0,
            balance=balance_value,
            initial_balance=balance_value,
            entry_price=None,
            unrealized_pnl_pct=0.0,
            last_signal=None,
            confidence=None,
            risk_per_trade_pct=risk_value,
            ai_action=None,
            ai_confidence=None,
            data_directory=str(relative_dir),
            is_paused=control.paused,
            pause_reason=control.reason,
            pause_updated_at=control.updated_at,
            is_placeholder=True,
            **metadata,
        )

    results_list = list(results.values())
    results_list.sort(key=lambda entry: (entry.symbol or ""))
    return results_list


@app.get("/portfolio/controls", response_model=List[PortfolioControlStateResponse])
def read_portfolio_controls(
    symbol: Optional[str] = Query(default=None),
) -> List[PortfolioControlStateResponse]:
    """Expose the current manual pause states for configured bots."""

    statuses = load_portfolio_states(symbol)
    return [
        PortfolioControlStateResponse(
            symbol=entry.symbol,
            paused=entry.is_paused,
            reason=entry.pause_reason,
            updated_at=entry.pause_updated_at,
        )
        for entry in statuses
    ]


@app.post("/portfolio/controls", response_model=PortfolioControlStateResponse)
def update_portfolio_control(
    payload: PortfolioControlUpdateRequest,
) -> PortfolioControlStateResponse:
    """Toggle manual pause/resume for a specific configured bot."""

    symbol_value = payload.symbol.strip()
    if not symbol_value:
        raise HTTPException(status_code=400, detail="Symbol is required.")
    symbol_key = symbol_value.upper()

    config_payload = _load_portfolio_config_payload()
    asset_map = _build_asset_index(config_payload)
    asset = asset_map.get(symbol_key)
    if asset is None:
        raise HTTPException(
            status_code=404,
            detail="Symbol is not part of the configured portfolio.",
        )

    portfolio_dir = STATE_DIR / "portfolio"
    data_dir = _resolve_asset_data_dir(symbol_key, asset_map, portfolio_dir)
    reason_value = payload.reason.strip() if payload.reason else None
    control = update_bot_control(
        data_dir,
        paused=payload.paused,
        reason=reason_value,
    )

    resolved_symbol = asset.get("symbol") or symbol_key
    return PortfolioControlStateResponse(
        symbol=resolved_symbol,
        paused=control.paused,
        reason=control.reason,
        updated_at=control.updated_at,
    )


@app.get(
    "/status",
    response_model=BotStateResponse,
    tags=["Status"],
    summary="Get bot status",
    description="Returns the current bot state including balance, positions, and P&L.",
)
def read_status(store: StateStore = Depends(get_store)) -> BotStateResponse:
    """
    Get the current status of the trading bot.

    Returns:
        - Current balance
        - Active positions
        - Daily P&L
        - Trading statistics
    """
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")
    return BotStateResponse.model_validate(payload)


@app.get("/signals", response_model=List[SignalResponse])
def read_signals(
    limit: int = Query(default=50, ge=1, le=500),
    store: StateStore = Depends(get_store),
) -> List[SignalResponse]:
    store.load()
    signals = store.get_signals(limit)
    return [SignalResponse.model_validate(item) for item in signals]


@app.get("/equity", response_model=List[EquityPointResponse])
def read_equity(store: StateStore = Depends(get_store)) -> List[EquityPointResponse]:
    """Return equity curve with real-time portfolio value appended."""
    store.load()
    curve = store.get_equity_curve()
    result = [EquityPointResponse.model_validate(point) for point in curve]

    # Add real-time portfolio value as the latest data point
    try:
        total_value = 0.0
        market_dirs = [
            STATE_DIR / "live_paper_trading" / "state.json",
            STATE_DIR / "commodity_trading" / "state.json",
            STATE_DIR / "stock_trading" / "state.json",
        ]
        for state_path in market_dirs:
            if state_path.exists():
                with open(state_path, "r") as f:
                    state = json.load(f)
                    total_value += state.get("total_value", state.get("balance", 0))

        if total_value > 0:
            now = datetime.now(timezone.utc)
            # Only add if it's different from the last point or enough time has passed
            should_add = True
            if result:
                last_point = result[-1]
                last_time = (
                    datetime.fromisoformat(last_point.timestamp.replace("Z", "+00:00"))
                    if isinstance(last_point.timestamp, str)
                    else last_point.timestamp
                )
                time_diff = (
                    (now - last_time).total_seconds()
                    if hasattr(last_time, "total_seconds") or isinstance(last_time, datetime)
                    else 60
                )
                if hasattr(time_diff, "__abs__"):
                    time_diff = abs(time_diff)
                should_add = time_diff >= 30 or abs(last_point.value - total_value) > 10

            if should_add:
                result.append(EquityPointResponse(timestamp=now.isoformat(), value=total_value))
    except Exception:
        pass  # If real-time data fails, just return the stored curve

    return result


@app.get("/equity/enhanced", response_model=List[EquityPointResponse])
def read_enhanced_equity() -> List[EquityPointResponse]:
    """Return equity curve data for the enhanced ML paper trading bot."""
    equity_file = STATE_DIR / "ml_paper_trading_enhanced" / "equity.json"
    if not equity_file.exists():
        return []
    try:
        with open(equity_file) as f:
            data = json.load(f)
        return [EquityPointResponse.model_validate(point) for point in data]
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/equity/aggressive", response_model=List[EquityPointResponse])
def read_aggressive_equity() -> List[EquityPointResponse]:
    """Return equity curve data for the aggressive ML paper trading bot."""
    equity_file = STATE_DIR / "ml_paper_trading_aggressive" / "equity.json"
    if not equity_file.exists():
        return []
    try:
        with open(equity_file) as f:
            data = json.load(f)
        return [EquityPointResponse.model_validate(point) for point in data]
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/strategy", response_model=StrategyOverviewResponse)
def read_strategy_overview(
    symbol: Optional[str] = Query(default=None),
) -> StrategyOverviewResponse:
    """Return the active strategy settings.

    If a symbol is provided, the API will try to load the per-asset overrides from
    DATA_DIR/portfolio/<safe_symbol>/strategy_config.json and fall back to the
    global DATA_DIR/strategy_config.json when not found.
    """
    config = StrategyConfig.from_env()

    # Try per-asset strategy first when query symbol is provided
    if symbol:
        safe = sanitize_symbol_for_fs(symbol)
        per_asset_path = STATE_DIR / "portfolio" / safe / "strategy_config.json"
        overrides = load_overrides(per_asset_path)
        if overrides:
            merged = merge_config(
                StrategyConfig(symbol=symbol, timeframe=config.timeframe), overrides
            )
            config = merged
        else:
            # fall back to global overrides
            overrides = load_overrides(STATE_DIR / "strategy_config.json")
            if overrides:
                config = merge_config(config, overrides)
    else:
        overrides = load_overrides(STATE_DIR / "strategy_config.json")
        if overrides:
            config = merge_config(config, overrides)
    decision_rules = [
        "Go LONG when the fast EMA crosses above the slow EMA and RSI stays below the overbought threshold.",
        "Go SHORT when the fast EMA crosses below the slow EMA and RSI stays above the oversold threshold.",
        "Fallback LONG when RSI dips below the oversold threshold even without a crossover.",
        "Fallback SHORT when RSI rises above the overbought threshold even without a crossover.",
    ]
    risk_notes = [
        "Risk per trade is capped by the configured percentage of the current balance.",
        "Stop-loss and take-profit levels are derived from the last entry price and the configured percentages.",
        "Position sizing scales inversely with stop distance to maintain consistent risk exposure.",
    ]

    return StrategyOverviewResponse(
        symbol=config.symbol,
        timeframe=config.timeframe,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        rsi_period=config.rsi_period,
        rsi_overbought=config.rsi_overbought,
        rsi_oversold=config.rsi_oversold,
        risk_per_trade_pct=config.risk_per_trade_pct,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        decision_rules=decision_rules,
        risk_management_notes=risk_notes,
    )


@app.get("/portfolio/strategies", response_model=List[StrategyOverviewResponse])
def list_portfolio_strategies() -> List[StrategyOverviewResponse]:
    """Enumerate per-asset strategies found under DATA_DIR/portfolio/*/strategy_config.json."""
    results: List[StrategyOverviewResponse] = []
    portfolio_dir = STATE_DIR / "portfolio"
    if not portfolio_dir.exists():
        return results
    for item in sorted(portfolio_dir.iterdir()):
        strategy_file = item / "strategy_config.json"
        if not strategy_file.exists():
            continue
        overrides = load_overrides(strategy_file)
        if not overrides:
            continue
        symbol = str(overrides.get("symbol") or item.name)
        timeframe = str(overrides.get("timeframe") or StrategyConfig.from_env().timeframe)
        cfg = merge_config(StrategyConfig(symbol=symbol, timeframe=timeframe), overrides)
        results.append(
            StrategyOverviewResponse(
                symbol=cfg.symbol,
                timeframe=cfg.timeframe,
                ema_fast=cfg.ema_fast,
                ema_slow=cfg.ema_slow,
                rsi_period=cfg.rsi_period,
                rsi_overbought=cfg.rsi_overbought,
                rsi_oversold=cfg.rsi_oversold,
                risk_per_trade_pct=cfg.risk_per_trade_pct,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
                decision_rules=[
                    (
                        "Go LONG when the fast EMA crosses above the slow EMA and "
                        "RSI stays below the overbought threshold."
                    ),
                    (
                        "Go SHORT when the fast EMA crosses below the slow EMA and "
                        "RSI stays above the oversold threshold."
                    ),
                    "Fallback LONG when RSI dips below the oversold threshold even without a crossover.",
                    "Fallback SHORT when RSI rises above the overbought threshold even without a crossover.",
                ],
                risk_management_notes=[
                    "Risk per trade is capped by the configured percentage of the current balance.",
                    (
                        "Stop-loss and take-profit levels are derived from the last entry price "
                        "and the configured percentages."
                    ),
                    "Position sizing scales inversely with stop distance to maintain consistent risk exposure.",
                ],
            )
        )
    return results


@app.get("/ai/prediction", response_model=AIPredictionResponse)
def read_ai_prediction(store: StateStore = Depends(get_store)) -> AIPredictionResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    features = payload.get("ai_features") or None

    return AIPredictionResponse(
        timestamp=payload["timestamp"],
        symbol=payload.get("symbol", ""),
        recommended_action=payload.get("ai_action"),
        confidence=payload.get("ai_confidence"),
        probability_long=payload.get("ai_probability_long"),
        probability_short=payload.get("ai_probability_short"),
        probability_flat=payload.get("ai_probability_flat"),
        expected_move_pct=payload.get("ai_expected_move_pct"),
        summary=payload.get("ai_summary"),
        features=features,
        macro_bias=payload.get("macro_bias"),
        macro_confidence=payload.get("macro_confidence"),
        macro_summary=payload.get("macro_summary"),
        macro_drivers=payload.get("macro_drivers") or [],
        macro_interest_rate_outlook=payload.get("macro_interest_rate_outlook"),
        macro_political_risk=payload.get("macro_political_risk"),
    )


@app.post("/ai/question", response_model=AIAnswerResponse)
def ask_ai(
    request: AIQuestionRequest,
    store: StateStore = Depends(get_store),
) -> AIAnswerResponse:
    store.load()
    config = StrategyConfig.from_env()
    engine = QuestionAnsweringEngine(config)

    state = store.state
    ai_features = getattr(state, "ai_features", {}) or {}
    ai_snapshot: Optional[PredictionSnapshot]
    if state.ai_action:
        ai_snapshot = PredictionSnapshot(
            recommended_action=state.ai_action,
            confidence=state.ai_confidence or 0.0,
            probability_long=state.ai_probability_long or 0.0,
            probability_short=state.ai_probability_short or 0.0,
            probability_flat=state.ai_probability_flat or 0.0,
            expected_move_pct=state.ai_expected_move_pct or 0.0,
            summary=state.ai_summary or "",
            features=FeatureSnapshot(
                ema_gap_pct=ai_features.get("ema_gap_pct", 0.0),
                momentum_pct=ai_features.get("momentum_pct", 0.0),
                rsi_distance_from_mid=ai_features.get("rsi_distance_from_mid", 0.0),
                volatility_pct=ai_features.get("volatility_pct", 0.0),
            ),
            macro_bias=state.macro_bias or 0.0,
            macro_confidence=state.macro_confidence or 0.0,
            macro_summary=state.macro_summary or "",
            macro_drivers=list(state.macro_drivers or []),
            macro_interest_rate_outlook=state.macro_interest_rate_outlook,
            macro_political_risk=state.macro_political_risk,
        )
    else:
        ai_snapshot = None

    macro_insight: Optional[MacroInsight] = None
    if state.macro_summary or state.macro_drivers or state.macro_bias is not None:
        macro_insight = MacroInsight(
            symbol=state.symbol,
            bias_score=state.macro_bias or 0.0,
            confidence=state.macro_confidence or 0.0,
            summary=state.macro_summary or "",
            drivers=list(state.macro_drivers or []),
            interest_rate_outlook=state.macro_interest_rate_outlook,
            political_risk=state.macro_political_risk,
            events=list(state.macro_events or []),
        )

    answer = engine.answer(
        request.question,
        state=state,
        ai_snapshot=ai_snapshot,
        macro_insight=macro_insight,
    )
    return AIAnswerResponse(question=request.question, answer=answer)


@app.get("/macro/insights", response_model=MacroInsightResponse)
def read_macro_insights(store: StateStore = Depends(get_store)) -> MacroInsightResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    events_payload: List[MacroEventResponse] = []
    for item in payload.get("macro_events") or []:
        if isinstance(item, dict) and item.get("title"):
            events_payload.append(MacroEventResponse.model_validate(item))

    return MacroInsightResponse(
        timestamp=payload["timestamp"],
        symbol=payload.get("symbol", ""),
        bias_score=payload.get("macro_bias"),
        confidence=payload.get("macro_confidence"),
        summary=payload.get("macro_summary"),
        drivers=payload.get("macro_drivers") or [],
        interest_rate_outlook=payload.get("macro_interest_rate_outlook"),
        political_risk=payload.get("macro_political_risk"),
        events=events_payload,
    )


@app.get("/portfolio/playbook", response_model=PortfolioPlaybookResponse)
def read_portfolio_playbook(
    store: StateStore = Depends(get_store),
) -> PortfolioPlaybookResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")

    playbook = payload.get("portfolio_playbook")
    if not playbook:
        # Return empty playbook instead of 404 to prevent dashboard errors
        from datetime import datetime, timezone

        return PortfolioPlaybookResponse(
            generated_at=datetime.now(timezone.utc),
            starting_balance=10000.0,
            commodities=[],
            equities=[],
            highlights=[],
        )

    return PortfolioPlaybookResponse.model_validate(playbook)


@app.get("/portfolio/status", response_model=List[PortfolioBotStatusResponse])
def read_portfolio_status(
    symbol: Optional[str] = Query(default=None),
    _auth: Optional[str] = Depends(verify_api_key),
    _rate_limit: None = Depends(check_rate_limit),
) -> List[PortfolioBotStatusResponse]:
    """Expose each running asset's status for the dashboard."""

    statuses = load_portfolio_states(symbol)
    if symbol and not statuses:
        raise HTTPException(status_code=404, detail=f"No state found for symbol {symbol}.")
    return statuses


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check",
    description="Health check endpoint for monitoring and orchestration systems.",
)
def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring and orchestration systems.

    Returns the current health status of the API and bot components.
    Does not require authentication to allow external health probes.

    Returns:
        - API status (healthy/unhealthy)
        - Uptime in seconds
        - Bot state freshness
        - Component status
    """
    now = datetime.now(timezone.utc)
    uptime = time.time() - _app_start_time
    stale_threshold = int(os.getenv("BOT_STALE_THRESHOLD_SECONDS", "300"))

    components: Dict[str, str] = {}
    bot_last_update: Optional[datetime] = None
    bot_stale = False

    # Check bot state - check all active trading state files
    active_state_files = [
        STATE_DIR / "unified_trading" / "state.json",  # Primary unified state
        STATE_DIR / "regime_trading" / "state.json",
        STATE_DIR / "live_paper_trading" / "state.json",
        STATE_DIR / "commodity_trading" / "state.json",
        STATE_DIR / "stock_trading" / "state.json",
        STATE_DIR / "production" / "state.json",
        STATE_DIR / "state.json",  # Legacy fallback
    ]

    try:
        most_recent_update: Optional[datetime] = None
        active_bots = 0

        for state_path in active_state_files:
            if state_path.exists():
                try:
                    import json

                    with open(state_path) as f:
                        data = json.load(f)
                    ts_str = data.get("timestamp")
                    if ts_str:
                        from dateutil.parser import parse

                        ts = parse(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if most_recent_update is None or ts > most_recent_update:
                            most_recent_update = ts
                        # Count as active if updated within threshold
                        age = (now - ts).total_seconds()
                        if age <= stale_threshold:
                            active_bots += 1
                except Exception:
                    continue

        bot_last_update = most_recent_update
        if bot_last_update:
            age_seconds = (now - bot_last_update).total_seconds()
            bot_stale = age_seconds > stale_threshold
            components["bot"] = "healthy" if not bot_stale else "stale"
            components["active_bots"] = str(active_bots)
        else:
            components["bot"] = "no_data"
    except Exception:
        components["bot"] = "error"
        bot_stale = True

    # Check state file accessibility
    try:
        state_files_exist = any(p.exists() for p in active_state_files)
        if state_files_exist:
            components["state_store"] = "healthy"
        else:
            components["state_store"] = "no_file"
    except Exception:
        components["state_store"] = "error"

    # Determine overall status (exclude info-only components like active_bots count)
    status_components = {k: v for k, v in components.items() if k != "active_bots"}
    if all(v == "healthy" for v in status_components.values()):
        status = "healthy"
    elif any(v == "error" for v in status_components.values()):
        status = "unhealthy"
    else:
        status = "degraded"

    return HealthCheckResponse(
        status=status,
        timestamp=now,
        uptime_seconds=round(uptime, 2),
        bot_last_update=bot_last_update,
        bot_stale=bot_stale,
        stale_threshold_seconds=stale_threshold,
        components=components,
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/preview", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the unified multi-market dashboard at root URL."""
    if not DASHBOARD_UNIFIED_TEMPLATE.exists():
        return HTMLResponse(
            content="<html><body><h1>Dashboard template missing.</h1></body></html>",
            status_code=200,
        )
    # Return with cache-control headers to prevent stale dashboard caching
    response = HTMLResponse(content=DASHBOARD_UNIFIED_TEMPLATE.read_text())
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/dashboard/v2")
@app.get("/live")
async def dashboard_v2():
    """Redirect legacy dashboard URLs to root."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/", status_code=302)


@app.get("/api/paper-trading-log")
async def get_paper_trading_log(
    lines: int = Query(default=50, description="Number of lines to return"),
    market: str = Query(default="all", description="Market filter: all, crypto, commodity, stock"),
) -> Dict[str, Any]:
    """Get recent paper trading log entries from all markets."""
    import re
    from datetime import datetime

    def parse_log_entry(line: str, market_source: str) -> dict | None:
        """Parse a single log line into structured entry."""
        line = line.strip()
        if not line:
            return None

        # Parse log format: "2026-01-11 18:20:29,022 | INFO | message"
        parts = line.split(" | ", 2)
        if len(parts) < 3:
            return None

        timestamp_str = parts[0].split(",")[0] if "," in parts[0] else parts[0]
        time_only = timestamp_str.split(" ")[-1] if " " in timestamp_str else timestamp_str
        level = parts[1].strip()
        message = parts[2].strip()

        # Parse full timestamp for sorting
        try:
            full_ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            full_ts = datetime.now()

        # Determine log type with more categories
        log_type = "info"
        icon = "📋"

        # Service interruptions and errors
        if level == "ERROR":
            log_type = "error"
            icon = "🔴"
        elif level == "WARNING":
            log_type = "warning"
            icon = "🟡"
            if "Rate limit" in message or "Too Many Requests" in message:
                log_type = "rate_limit"
                icon = "⏱️"
            elif "Failed" in message or "Error" in message:
                log_type = "error"
                icon = "⚠️"
            elif "skipping" in message.lower():
                log_type = "skip"
                icon = "⏭️"
        # Trading signals
        elif "BUY" in message.upper() or "LONG" in message.upper():
            log_type = "buy"
            icon = "🟢"
        elif "SELL" in message.upper() or "SHORT" in message.upper():
            log_type = "sell"
            icon = "🔴"
        # Position changes
        elif "position" in message.lower() or "rebalancing" in message.lower():
            log_type = "position"
            icon = "📊"
        # Portfolio updates
        elif "Portfolio:" in message or "Checkpoint" in message:
            log_type = "portfolio"
            icon = "💰"
        # Iterations
        elif "--- Iteration" in message:
            log_type = "iteration"
            icon = "🔄"
        # Price updates
        elif any(sym in message for sym in ["$", "BTC", "ETH", "Gold", "Silver", "AAPL", "MSFT"]):
            log_type = "price"
            icon = "📈"

        return {
            "time": time_only,
            "timestamp": timestamp_str,
            "full_ts": full_ts,
            "level": level,
            "message": message,
            "type": log_type,
            "icon": icon,
            "market": market_source,
        }

    try:
        all_entries = []
        sources_status = {}

        # Determine which log sources to read
        sources_to_read = (
            TRADING_LOG_SOURCES
            if market == "all"
            else {k: v for k, v in TRADING_LOG_SOURCES.items() if k == market}
        )

        for market_name, log_path in sources_to_read.items():
            if log_path.exists():
                sources_status[market_name] = "active"
                try:
                    with open(log_path, "r") as f:
                        # Read last N*2 lines to have enough after filtering
                        log_lines = f.readlines()[-(lines * 2) :]

                    for line in log_lines:
                        entry = parse_log_entry(line, market_name)
                        if entry:
                            all_entries.append(entry)
                except Exception as e:
                    sources_status[market_name] = f"error: {str(e)}"
            else:
                sources_status[market_name] = "not_found"

        # Sort by timestamp (newest first)
        all_entries.sort(key=lambda x: x["full_ts"], reverse=True)

        # Remove internal timestamp field and limit results
        for entry in all_entries:
            del entry["full_ts"]

        # Count by type for summary
        type_counts = {}
        for entry in all_entries[:lines]:
            t = entry["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "entries": all_entries[:lines],
            "sources": sources_status,
            "summary": type_counts,
            "total_entries": len(all_entries),
        }
    except Exception as e:
        return {"entries": [], "error": str(e)}


@app.post("/bot/stop")
async def stop_bot() -> Dict[str, str]:
    """Send stop signal to paper trading bot."""
    import subprocess

    try:
        # Find and kill the paper trading process
        result = subprocess.run(
            ["pkill", "-f", "run_live_paper_trading"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return {"status": "stopped", "message": "Paper trading bot stopped"}
        return {"status": "not_running", "message": "No paper trading bot found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/live-status")
async def get_live_trading_status() -> Dict[str, Any]:
    """Get current live paper trading status by parsing the trading log."""
    import subprocess
    import re

    result = {
        "running": False,
        "iteration": 0,
        "timestamp": None,
        "prices": {},
        "signals": {},
        "portfolio_value": 0,
        "cash_balance": 0,
        "positions": {},
        "initial_capital": 10000,
        "pnl": 0,
        "pnl_pct": 0,
    }

    # Check if bot is running
    try:
        ps_result = subprocess.run(
            ["pgrep", "-f", "run_live_paper_trading"],
            capture_output=True,
            text=True,
        )
        result["running"] = ps_result.returncode == 0
    except Exception:
        pass

    # Parse the trading log
    if not PAPER_TRADING_LOG.exists():
        return result

    try:
        with open(PAPER_TRADING_LOG, "r") as f:
            lines = f.readlines()[-100:]  # Last 100 lines

        # Find most recent iteration data
        current_iteration = 0
        current_timestamp = None
        prices = {}
        signals = {}
        portfolio_value = 10000
        cash_balance = 10000

        for line in lines:
            line = line.strip()

            # Parse iteration line
            iter_match = re.search(
                r"Iteration (\d+) \| (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line
            )
            if iter_match:
                current_iteration = int(iter_match.group(1))
                current_timestamp = iter_match.group(2)

            # Parse price lines: "BTC/USDT: $90,783.36 → (-0.00%)"
            price_match = re.search(r"([\w/]+): \$([0-9,.]+)", line)
            if price_match and "/" in price_match.group(1):
                symbol = price_match.group(1)
                price = float(price_match.group(2).replace(",", ""))
                prices[symbol] = price

            # Parse signal lines: "BTC/USDT: FLAT ⚪ | sideways | conf: 61%"
            signal_match = re.search(
                r"([\w/]+): (LONG|SHORT|FLAT) [🟢🔴⚪] \| (\w+) \| conf: (\d+)%", line
            )
            if signal_match:
                symbol = signal_match.group(1)
                action = signal_match.group(2)
                regime = signal_match.group(3)
                confidence = int(signal_match.group(4)) / 100
                signals[symbol] = {
                    "action": action,
                    "regime": regime,
                    "confidence": confidence,
                }

            # Parse portfolio line: "Portfolio: $10,000.31 | Cash: $4,138.34"
            portfolio_match = re.search(r"Portfolio: \$([0-9,.]+) \| Cash: \$([0-9,.]+)", line)
            if portfolio_match:
                portfolio_value = float(portfolio_match.group(1).replace(",", ""))
                cash_balance = float(portfolio_match.group(2).replace(",", ""))

        # Calculate positions from portfolio value and cash
        invested = portfolio_value - cash_balance
        positions = {}
        for symbol, sig in signals.items():
            if sig["action"] != "FLAT" and symbol in prices:
                # Estimate position based on invested capital
                positions[symbol] = {
                    "value": invested / len([s for s in signals.values() if s["action"] != "FLAT"])
                    if invested > 0
                    else 0,
                    "price": prices.get(symbol, 0),
                    "signal": sig["action"],
                    "regime": sig["regime"],
                }

        initial_capital = 10000  # From config
        pnl = portfolio_value - initial_capital
        pnl_pct = (pnl / initial_capital) * 100

        result.update(
            {
                "iteration": current_iteration,
                "timestamp": current_timestamp,
                "prices": prices,
                "signals": signals,
                "portfolio_value": round(portfolio_value, 2),
                "cash_balance": round(cash_balance, 2),
                "positions": positions,
                "initial_capital": initial_capital,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
        )

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# SMART ENGINE ENDPOINTS - ML, Regime Analysis, Strategy Selection
# =============================================================================

_smart_engine = None
_regime_classifier = None


def _get_smart_engine():
    """Lazy-load the smart trading engine."""
    global _smart_engine
    if _smart_engine is None:
        try:
            from bot.smart_engine import SmartTradingEngine, EngineConfig

            _smart_engine = SmartTradingEngine(
                config=EngineConfig(),
                data_dir=str(STATE_DIR),
            )
        except ImportError:
            return None
    return _smart_engine


def _get_regime_classifier():
    """Lazy-load the regime classifier."""
    global _regime_classifier
    if _regime_classifier is None:
        try:
            from bot.ml import MarketRegimeClassifier

            _regime_classifier = MarketRegimeClassifier()
        except ImportError:
            return None
    return _regime_classifier


@app.get("/ml/regime")
async def get_market_regime(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get current market regime analysis.

    Returns bull/bear/sideways/volatile classification with confidence.
    """
    classifier = _get_regime_classifier()
    if classifier is None:
        raise HTTPException(status_code=503, detail="ML module not available")

    # Try to load recent data
    try:
        import yfinance as yf

        yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="60d", interval="1h")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    analysis = classifier.classify(df)
    params = classifier.get_strategy_parameters(analysis.regime)

    return {
        "symbol": symbol,
        "regime": analysis.regime.value,
        "confidence": round(analysis.confidence, 4),
        "trend_strength": round(analysis.trend_strength, 4),
        "volatility": {
            "level": analysis.volatility_level,
            "percentile": round(analysis.volatility_percentile, 2),
        },
        "indicators": {
            "adx": round(analysis.adx_value, 2),
            "momentum": round(analysis.momentum_score, 4),
        },
        "levels": {
            "support": round(analysis.support_level, 2),
            "resistance": round(analysis.resistance_level, 2),
        },
        "recommended_strategy": analysis.recommended_strategy,
        "strategy_parameters": params,
        "reasoning": analysis.reasoning,
        "regime_duration_bars": analysis.regime_duration,
    }


@app.get("/ml/prediction")
async def get_ml_prediction(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get ML model prediction for the symbol.

    Returns LONG/SHORT/FLAT recommendation with probabilities.
    """
    engine = _get_smart_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Smart engine not available")

    if not engine.ml_predictor or not engine.ml_predictor.is_trained:
        raise HTTPException(
            status_code=503, detail="ML model not trained. POST to /ml/train first."
        )

    # Fetch data
    try:
        import yfinance as yf

        yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="60d", interval="1h")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    prediction = engine.ml_predictor.predict(df)

    return {
        "symbol": symbol,
        "prediction": prediction.to_dict(),
        "model_status": {
            "is_trained": engine.ml_predictor.is_trained,
            "model_type": engine.ml_predictor.model_type,
            "should_retrain": engine.should_retrain(),
        },
    }


@app.post("/ml/train")
async def train_ml_model(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    days: int = Query(default=365, description="Days of historical data"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Train or retrain the ML model on historical data.

    This may take a few minutes depending on data size.
    """
    engine = _get_smart_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Smart engine not available")

    # Fetch training data
    try:
        import yfinance as yf

        yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    # Train model
    try:
        metrics = engine.train_ml(df)
        return {
            "status": "success",
            "symbol": symbol,
            "data_points": len(df),
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/ml/decision")
async def get_smart_decision(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get a comprehensive trading decision from the smart engine.

    Combines regime analysis, ML prediction, and strategy selection.
    """
    engine = _get_smart_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Smart engine not available")

    # Fetch data
    try:
        import yfinance as yf

        yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="60d", interval="1h")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    decision = engine.analyze(df)

    return {
        "symbol": symbol,
        "current_price": round(float(df["close"].iloc[-1]), 2),
        "decision": decision.to_dict(),
        "explanation": engine.explain_decision(decision),
    }


@app.get("/ml/strategies")
async def list_strategies(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all available trading strategies."""
    engine = _get_smart_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Smart engine not available")

    strategies = engine.strategy_selector.get_available_strategies()
    strategy_info = []

    for name in strategies:
        strategy = engine.strategy_selector.get_strategy(name)
        if strategy:
            strategy_info.append(
                {
                    "name": name,
                    "description": strategy.description,
                    "suitable_regimes": strategy.suitable_regimes,
                    "indicators": strategy.get_required_indicators(),
                }
            )

    return {
        "strategies": strategy_info,
        "multi_strategy_enabled": engine.config.use_multi_strategy,
    }


@app.get("/ml/status")
async def get_ml_status(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get status of ML components."""
    engine = _get_smart_engine()
    if engine is None:
        return {
            "available": False,
            "error": "Smart engine not available. Install scikit-learn.",
        }

    return {
        "available": True,
        "status": engine.get_status(),
    }


@app.get("/ml/performance")
async def get_performance_analysis(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get performance analysis of recent trades."""
    store = get_store()
    store.load()

    signals = store.signals_history
    equity = store.equity_history

    if not signals:
        return {"error": "No trading history available"}

    engine = _get_smart_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Smart engine not available")

    # Convert signals to trade format
    trades = []
    for sig in signals:
        trades.append(
            {
                "pnl": sig.get("pnl", 0),
                "pnl_pct": sig.get("pnl_pct", 0),
                "direction": sig.get("decision", "FLAT"),
            }
        )

    report = engine.get_analysis_report(trades, equity)

    return {
        "report": report.to_dict(),
        "markdown": report.to_markdown(),
    }


# =============================================================================
# PORTFOLIO OPTIMIZER ENDPOINTS - Multi-Asset Portfolio Management
# =============================================================================

_multi_asset_engine = None


def _get_multi_asset_engine():
    """Lazy-load the multi-asset trading engine."""
    global _multi_asset_engine
    if _multi_asset_engine is None:
        try:
            from bot.multi_asset_engine import (
                MultiAssetTradingEngine,
                MultiAssetConfig,
                AssetConfig,
            )
            from bot.portfolio_optimizer import OptimizationMethod

            # Default crypto portfolio configuration
            config = MultiAssetConfig(
                assets=[
                    AssetConfig(symbol="BTC/USDT", max_weight=0.40, min_weight=0.10),
                    AssetConfig(symbol="ETH/USDT", max_weight=0.35, min_weight=0.10),
                    AssetConfig(symbol="SOL/USDT", max_weight=0.30, min_weight=0.05),
                ],
                optimization_method=OptimizationMethod.RISK_PARITY,
                rebalance_threshold=0.05,
                total_capital=10000.0,
            )

            _multi_asset_engine = MultiAssetTradingEngine(
                config=config,
                data_dir=str(STATE_DIR / "portfolio"),
            )
        except ImportError as e:
            print(f"Multi-asset engine import error: {e}")
            return None
    return _multi_asset_engine


@app.get("/portfolio/optimize")
async def optimize_portfolio(
    method: str = Query(default="risk_parity", description="Optimization method"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Run portfolio optimization with specified method.

    Methods: equal_weight, risk_parity, min_volatility, max_sharpe, max_diversification
    """
    try:
        from bot.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
        import pandas as pd
        import numpy as np

        # Map string to enum
        method_map = {
            "equal_weight": OptimizationMethod.EQUAL_WEIGHT,
            "risk_parity": OptimizationMethod.RISK_PARITY,
            "min_volatility": OptimizationMethod.MIN_VOLATILITY,
            "max_sharpe": OptimizationMethod.MAX_SHARPE,
            "max_diversification": OptimizationMethod.MAX_DIVERSIFICATION,
            "inverse_volatility": OptimizationMethod.INVERSE_VOLATILITY,
        }

        opt_method = method_map.get(method.lower())
        if opt_method is None:
            raise HTTPException(
                status_code=400, detail=f"Invalid method. Choose from: {list(method_map.keys())}"
            )

        # Generate sample returns for demonstration
        # In production, this would fetch real market data
        np.random.seed(42)
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "MATIC/USDT"]
        returns_data = np.random.randn(252, len(symbols)) * 0.03  # ~3% daily vol
        returns_df = pd.DataFrame(returns_data, columns=symbols)

        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            min_weight=0.05,
            max_weight=0.40,
        )

        result = optimizer.optimize(returns_df, opt_method)

        return {
            "method": method,
            "allocation": result.to_dict(),
            "correlation_analysis": optimizer.analyze_correlations(returns_df),
        }

    except ImportError:
        raise HTTPException(
            status_code=503, detail="Portfolio optimizer not available. Install scipy."
        )


@app.get("/portfolio/optimizer/status")
async def get_portfolio_optimizer_status(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get status of the multi-asset portfolio engine."""
    engine = _get_multi_asset_engine()
    if engine is None:
        return {
            "available": False,
            "error": "Multi-asset engine not available. Check imports.",
        }

    return {
        "available": True,
        "status": engine.get_portfolio_status(),
        "config": {
            "optimization_method": engine.config.optimization_method.value,
            "rebalance_threshold": engine.config.rebalance_threshold,
            "total_capital": engine.config.total_capital,
            "assets": [
                {
                    "symbol": a.symbol,
                    "min_weight": a.min_weight,
                    "max_weight": a.max_weight,
                    "enabled": a.enabled,
                }
                for a in engine.config.assets
            ],
        },
    }


@app.get("/portfolio/optimizer/correlations", tags=["Portfolio"])
async def get_portfolio_correlations() -> Dict[str, Any]:
    """Get correlation analysis for portfolio assets using live market data.

    This endpoint is public (no API key required) for dashboard access.
    """
    import pandas as pd

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    symbol_map = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "AVAX/USDT": "AVAX-USD",
    }

    try:
        import yfinance as yf

        # Fetch returns data for each symbol
        returns_dict = {}
        for symbol in symbols:
            yf_symbol = symbol_map.get(symbol, symbol.replace("/", "-").replace("USDT", "USD"))
            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="60d", interval="1d")
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    returns_dict[symbol] = df["close"].pct_change().dropna()
            except Exception:
                continue

        if len(returns_dict) < 2:
            return {
                "status": "no_data",
                "message": "Could not fetch enough market data for correlation analysis.",
            }

        # Align returns data
        min_len = min(len(r) for r in returns_dict.values())
        returns_df = pd.DataFrame({s: r.tail(min_len).values for s, r in returns_dict.items()})

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Build correlation pairs
        pairs = []
        processed = set()
        for s1 in corr_matrix.columns:
            for s2 in corr_matrix.columns:
                if s1 != s2 and (s2, s1) not in processed:
                    pairs.append(
                        {
                            "asset1": s1,
                            "asset2": s2,
                            "correlation": round(float(corr_matrix.loc[s1, s2]), 4),
                        }
                    )
                    processed.add((s1, s2))

        # Calculate average correlation
        correlations = [p["correlation"] for p in pairs]
        avg_corr = sum(correlations) / len(correlations) if correlations else 0

        # Find highest and lowest correlations
        sorted_pairs = sorted(pairs, key=lambda x: x["correlation"], reverse=True)

        return {
            "status": "success",
            "analysis": {
                "pairs": pairs,
                "average_correlation": round(avg_corr, 4),
                "highest_correlation": sorted_pairs[0] if sorted_pairs else None,
                "lowest_correlation": sorted_pairs[-1] if sorted_pairs else None,
                "matrix": {
                    s: {s2: round(float(corr_matrix.loc[s, s2]), 4) for s2 in corr_matrix.columns}
                    for s in corr_matrix.columns
                },
            },
        }

    except ImportError:
        return {
            "status": "error",
            "message": "yfinance not installed. Run: pip install yfinance",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error calculating correlations: {str(e)}",
        }


@app.post("/portfolio/optimizer/analyze")
async def analyze_portfolio_allocation(
    symbols: List[str] = Query(
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"], description="Symbols to include in portfolio"
    ),
    capital: float = Query(default=10000.0, description="Total capital"),
    method: str = Query(default="risk_parity", description="Optimization method"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Analyze portfolio allocation for given symbols.

    Fetches market data and runs optimization.
    """
    try:
        from bot.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
        from bot.multi_asset_engine import MultiAssetTradingEngine, MultiAssetConfig, AssetConfig
        import pandas as pd
        import numpy as np

        method_map = {
            "equal_weight": OptimizationMethod.EQUAL_WEIGHT,
            "risk_parity": OptimizationMethod.RISK_PARITY,
            "min_volatility": OptimizationMethod.MIN_VOLATILITY,
            "max_sharpe": OptimizationMethod.MAX_SHARPE,
            "max_diversification": OptimizationMethod.MAX_DIVERSIFICATION,
        }

        opt_method = method_map.get(method.lower(), OptimizationMethod.RISK_PARITY)

        # Try to fetch real data via yfinance
        returns_dict = {}
        prices = {}

        try:
            import yfinance as yf

            for symbol in symbols:
                yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="90d", interval="1d")
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    returns_dict[symbol] = df["close"].pct_change().dropna()
                    prices[symbol] = float(df["close"].iloc[-1])
        except Exception as e:
            # Fall back to synthetic data
            np.random.seed(hash(str(symbols)) % 2**32)
            for symbol in symbols:
                returns_dict[symbol] = pd.Series(np.random.randn(90) * 0.03)
                prices[symbol] = 100.0

        if not returns_dict:
            raise HTTPException(status_code=404, detail="No data available for symbols")

        # Align returns
        min_len = min(len(r) for r in returns_dict.values())
        returns_df = pd.DataFrame({s: r.tail(min_len).values for s, r in returns_dict.items()})

        # Run optimization
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            min_weight=0.05,
            max_weight=0.50,
        )

        result = optimizer.optimize(returns_df, opt_method)

        # Calculate target positions
        positions = {}
        for symbol, weight in result.weights.items():
            target_value = capital * weight
            price = prices.get(symbol, 100.0)
            positions[symbol] = {
                "weight": round(weight, 4),
                "target_value": round(target_value, 2),
                "target_quantity": round(target_value / price, 6),
                "price": round(price, 2),
            }

        return {
            "symbols": symbols,
            "capital": capital,
            "method": method,
            "allocation": result.to_dict(),
            "positions": positions,
            "prices": {s: round(p, 2) for s, p in prices.items()},
        }

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Required modules not available: {e}")


@app.get("/portfolio/optimizer/compare")
async def compare_optimization_methods(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Compare all optimization methods for current portfolio."""
    try:
        from bot.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
        import pandas as pd
        import numpy as np

        # Generate returns data
        np.random.seed(42)
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "MATIC/USDT"]

        # Simulate correlated returns
        n_assets = len(symbols)
        corr = np.array(
            [
                [1.0, 0.7, 0.5, 0.5, 0.6],
                [0.7, 1.0, 0.6, 0.5, 0.5],
                [0.5, 0.6, 1.0, 0.7, 0.6],
                [0.5, 0.5, 0.7, 1.0, 0.5],
                [0.6, 0.5, 0.6, 0.5, 1.0],
            ]
        )
        L = np.linalg.cholesky(corr)

        daily_vol = np.array([0.04, 0.05, 0.06, 0.07, 0.06])
        daily_mean = np.array([0.001, 0.0008, 0.0005, 0.0003, 0.0004])

        random_returns = np.random.randn(252, n_assets)
        correlated_returns = random_returns @ L.T
        returns_data = correlated_returns * daily_vol + daily_mean

        returns_df = pd.DataFrame(returns_data, columns=symbols)

        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            min_weight=0.05,
            max_weight=0.40,
        )

        methods = [
            OptimizationMethod.EQUAL_WEIGHT,
            OptimizationMethod.INVERSE_VOLATILITY,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MIN_VOLATILITY,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.MAX_DIVERSIFICATION,
        ]

        results = []
        for method in methods:
            result = optimizer.optimize(returns_df, method)
            results.append(
                {
                    "method": method.value,
                    "weights": {k: round(v, 4) for k, v in result.weights.items()},
                    "metrics": {
                        "expected_return": round(result.metrics.expected_return, 4),
                        "volatility": round(result.metrics.volatility, 4),
                        "sharpe_ratio": round(result.metrics.sharpe_ratio, 4),
                        "diversification_ratio": round(result.metrics.diversification_ratio, 4),
                        "effective_n": round(result.metrics.effective_n, 2),
                    },
                }
            )

        # Find best by Sharpe ratio
        best = max(results, key=lambda r: r["metrics"]["sharpe_ratio"])

        return {
            "symbols": symbols,
            "comparison": results,
            "best_method": best["method"],
            "best_sharpe": best["metrics"]["sharpe_ratio"],
            "correlation_matrix": returns_df.corr().to_dict(),
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not available")


# ============================================================================
# Multi-Market Endpoints
# ============================================================================

DASHBOARD_UNIFIED_TEMPLATE = Path(__file__).with_name("dashboard_unified.html")


@app.get("/api/markets/{market_type}/summary")
async def get_market_summary(
    market_type: str,
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get summary data for a specific market type.

    Args:
        market_type: One of "crypto", "commodity", "stock", or "all"

    Returns unified market data including prices, positions, signals.
    """
    # Skip cache - need fresh data from unified state
    # global _market_summary_cache
    # if market_type in _market_summary_cache:
    #     cached_time, cached_data = _market_summary_cache[market_type]
    #     if time.time() - cached_time < _CACHE_TTL:
    #         return cached_data

    try:
        from bot.data import MarketDataService, MarketType, get_symbols_by_market

        # Map string to enum
        market_map = {
            "crypto": MarketType.CRYPTO,
            "commodity": MarketType.COMMODITY,
            "stock": MarketType.STOCK,
        }

        if market_type not in market_map and market_type != "all":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid market_type. Must be one of: crypto, commodity, stock, all",
            )

        data_service = MarketDataService(data_dir=str(STATE_DIR))

        # Get symbols for this market
        if market_type == "all":
            from bot.data import ALL_SYMBOLS

            symbols = list(ALL_SYMBOLS.keys())
        else:
            symbols_dict = get_symbols_by_market(market_map[market_type])
            symbols = list(symbols_dict.keys())

        # Load unified state for all markets (ensures consistency)
        unified_state_file = STATE_DIR / "unified_trading" / "state.json"
        bot_state = {}
        bot_signals = {}
        ai_features = {}

        if unified_state_file.exists():
            try:
                with open(unified_state_file) as f:
                    unified_state = json.load(f)
                    bot_state = unified_state
                    bot_signals = unified_state.get("signals", {})
                    ai_features = unified_state.get("ai_features", {})
            except Exception:
                pass

        # For market-specific portfolio data, calculate from positions
        # Split total portfolio equally across 3 markets
        total_balance = bot_state.get("current_balance", 10000.0)
        initial_capital = bot_state.get("initial_capital", 10000.0)
        capital_per_market = total_balance / 3  # ✅ Use current balance, not initial capital

        # Get positions for this market
        all_positions = bot_state.get("positions", {})
        symbol_to_market_mapping = {
            # Crypto
            "BTC/USDT": "crypto",
            "ETH/USDT": "crypto",
            "SOL/USDT": "crypto",
            "XRP/USDT": "crypto",
            "ADA/USDT": "crypto",
            "AVAX/USDT": "crypto",
            "DOGE/USDT": "crypto",
            "DOT/USDT": "crypto",
            "LINK/USDT": "crypto",
            "MATIC/USDT": "crypto",
            # Commodities
            "XAU/USD": "commodity",
            "XAG/USD": "commodity",
            "USOIL/USD": "commodity",
            "NATGAS/USD": "commodity",
            "WTICO/USD": "commodity",
            # Stocks
            "AAPL": "stock",
            "MSFT": "stock",
            "GOOGL": "stock",
            "AMZN": "stock",
            "TSLA": "stock",
            "NVDA": "stock",
            "META": "stock",
            "AAPL/USD": "stock",
            "MSFT/USD": "stock",
            "GOOGL/USD": "stock",
            "NVDA/USD": "stock",
        }

        # Calculate market-specific P&L and position value from positions
        market_pnl = 0.0
        market_position_count = 0
        market_positions_value = 0.0
        for symbol, position in all_positions.items():
            if symbol_to_market_mapping.get(symbol) == market_type:
                market_pnl += position.get("unrealized_pnl", 0.0)
                market_position_count += 1
                value = position.get("value")
                if value is None:
                    qty = position.get("quantity", 0.0)
                    price = position.get("current_price", position.get("price", 0.0))
                    value = qty * price
                market_positions_value += float(value or 0.0)

        # Market balance = capital allocation + unrealized P&L
        market_balance = capital_per_market + market_pnl
        market_pnl_pct = (market_pnl / capital_per_market * 100) if capital_per_market > 0 else 0.0

        # Fast price fetching using yfinance batch download (bypasses rate limiter)
        import yfinance as yf
        from bot.data import ALL_SYMBOLS

        # Map symbols to Yahoo format
        yahoo_symbols = []
        symbol_mapping = {}  # yahoo_symbol -> our_symbol
        for symbol in symbols:
            symbol_info = ALL_SYMBOLS.get(symbol)
            if symbol_info and symbol_info.provider_mappings.get("yahoo"):
                yahoo_sym = symbol_info.provider_mappings["yahoo"]
            elif "/" in symbol:
                # Crypto format BTC/USDT -> BTC-USD
                yahoo_sym = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
            else:
                yahoo_sym = symbol
            yahoo_symbols.append(yahoo_sym)
            symbol_mapping[yahoo_sym] = symbol

        # Fetch all prices in one batch call
        try:
            tickers = yf.Tickers(" ".join(yahoo_symbols))
            assets = []
            for yahoo_sym, our_symbol in symbol_mapping.items():
                try:
                    ticker = tickers.tickers.get(yahoo_sym)
                    if ticker:
                        info = ticker.fast_info
                        price = info.last_price if hasattr(info, "last_price") else 0
                        if price and price > 0:
                            signal_data = bot_signals.get(our_symbol, {})
                            spread = price * 0.0002  # Estimate 0.02% spread
                            assets.append(
                                {
                                    "symbol": our_symbol,
                                    "price": price,
                                    "change_24h_pct": 0,
                                    "bid": price - spread,
                                    "ask": price + spread,
                                    "spread_pct": 0.02,
                                    "market_type": market_type
                                    if market_type != "all"
                                    else "unknown",
                                    "signal": signal_data.get("signal"),
                                    "regime": signal_data.get("regime"),
                                    "confidence": signal_data.get("confidence"),
                                }
                            )
                except Exception:
                    pass
        except Exception:
            # Fallback to empty list if batch fails
            assets = []

        # Extract AI features from bot state
        ai_features = bot_state.get("ai_features", {})

        result = {
            "market_type": market_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assets": assets,
            "market_stats": {
                "total_assets": len(assets),
                "available_data": len([a for a in assets if a.get("price")]),
            },
            # Use market-specific portfolio data
            "total_value": market_balance,
            "cash_balance": max(0.0, capital_per_market - market_positions_value),
            "initial_capital": capital_per_market,
            "pnl": market_pnl,
            "pnl_pct": market_pnl_pct,
            "positions_count": market_position_count,
            "positions": {
                k: v
                for k, v in all_positions.items()
                if symbol_to_market_mapping.get(k) == market_type
            },
            "bot_running": bool(bot_state),
            "deep_learning_enabled": bot_state.get("deep_learning_enabled", False),
            # AI Enhancement features
            "mode": bot_state.get("mode", "standard"),
            "model_type": bot_state.get("model_type", "unknown"),
            "ai_features": {
                "auto_retraining_enabled": ai_features.get("auto_retraining_enabled", False),
                "online_learning_enabled": ai_features.get("online_learning_enabled", False),
                "llm_enabled": ai_features.get("llm_enabled", False),
                "win_count": ai_features.get("win_count", 0),
                "loss_count": ai_features.get("loss_count", 0),
                "win_rate": ai_features.get("win_rate", 0.0),
            }
            if ai_features
            else None,
        }

        # Skip cache - need fresh data
        # _market_summary_cache[market_type] = (time.time(), result)

        return result

    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Data service not available: {e}")


@app.get("/api/markets/indexes")
async def get_market_indexes(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get relevant market indexes for context.

    Returns crypto dominance, commodity indexes, and equity benchmarks.
    """
    try:
        from bot.data import MarketDataService

        data_service = MarketDataService(data_dir=str(STATE_DIR))

        indexes = {
            "crypto": {},
            "commodity": {},
            "equity": {},
        }

        # Try to get BTC for crypto reference
        try:
            btc_quote = data_service.fetch_quote("BTC/USDT")
            indexes["crypto"]["btc_price"] = btc_quote.last_price or btc_quote.mid_price
        except Exception:
            pass

        # Try to get gold for commodity reference
        try:
            gold_quote = data_service.fetch_quote("XAU/USD")
            indexes["commodity"]["gold_price"] = gold_quote.last_price or gold_quote.mid_price
        except Exception:
            pass

        # Try to get SPY-like for equity reference (using AAPL as proxy)
        try:
            aapl_quote = data_service.fetch_quote("AAPL")
            indexes["equity"]["aapl_price"] = aapl_quote.last_price or aapl_quote.mid_price
        except Exception:
            pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "indexes": indexes,
        }

    except ImportError:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "indexes": {},
            "error": "Data service not available",
        }


# --- Unified Reset Endpoint ---
@app.post("/api/unified/reset")
async def reset_unified_portfolio(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """Reset unified trading state to a clean baseline.

    Sets initial capital and clears positions and P&L to remove accumulated noise.
    """
    try:
        unified_dir = STATE_DIR / "unified_trading"
        unified_dir.mkdir(parents=True, exist_ok=True)
        state_file = unified_dir / "state.json"

        baseline = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": 10000.0,
            "current_balance": 10000.0,
            "total_pnl": 0.0,
            "signals": {},
            "positions": {},
            "mode": "paper_live_data",
        }

        with open(state_file, "w") as f:
            json.dump(baseline, f)

        # Also clear in-memory cache if any
        global _market_summary_cache
        _market_summary_cache = {}

        return {"status": "reset", "state": baseline}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.get("/api/macro/indicators")
async def get_macro_indicators(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get macro market indicators: VIX, DXY, BTC Dominance.
    """
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vix": None,
        "dxy": None,
        "btc_dominance": None,
    }

    try:
        import yfinance as yf

        # VIX - CBOE Volatility Index
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            if not vix_data.empty:
                result["vix"] = float(vix_data["Close"].iloc[-1])
        except Exception:
            pass

        # DXY - US Dollar Index
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy.history(period="1d")
            if not dxy_data.empty:
                result["dxy"] = float(dxy_data["Close"].iloc[-1])
        except Exception:
            pass

        # BTC Dominance - placeholder (would need CoinGecko/CMC API for real data)
        result["btc_dominance"] = 52.0

    except ImportError:
        pass

    return result


@app.get("/api/bot/state/{market_type}")
async def get_bot_state(
    market_type: str,
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get real-time bot portfolio state from state files.

    Args:
        market_type: One of "crypto", "commodity", "stock", or "all"

    Returns the bot's current portfolio state including positions, signals, and P&L.
    """
    # Use lists to check multiple possible state files (newest first)
    state_dirs = {
        "crypto": [
            STATE_DIR / "production" / "state.json",  # Live/testnet trading
            STATE_DIR / "live_paper_trading" / "state.json",
        ],
        "commodity": [STATE_DIR / "commodity_trading" / "state.json"],
        "stock": [STATE_DIR / "stock_trading" / "state.json"],
    }

    def find_best_state(paths):
        """Find the newest state file from a list of paths."""
        best_state = None
        best_mtime = 0
        for path in paths:
            if path.exists():
                try:
                    mtime = path.stat().st_mtime
                    if mtime > best_mtime:
                        with open(path) as f:
                            best_state = json.load(f)
                            best_mtime = mtime
                except Exception:
                    pass
        return best_state

    if market_type == "all":
        # Return all bot states
        all_states = {}
        total_value = 0
        total_pnl = 0

        for mtype, state_paths in state_dirs.items():
            state = find_best_state(state_paths)
            if state:
                all_states[mtype] = state
                total_value += state.get("total_value", 0)
                total_pnl += state.get("pnl", 0)
            else:
                all_states[mtype] = {"status": "not_running"}

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_portfolio_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / 30000 * 100) if total_value > 0 else 0,
            "markets": all_states,
        }

    if market_type not in state_dirs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market_type. Must be one of: crypto, commodity, stock, all",
        )

    state = find_best_state(state_dirs[market_type])
    if not state:
        return {
            "market_type": market_type,
            "status": "not_running",
            "message": f"No state file found. The {market_type} bot may not be running.",
        }

    return {
        "market_type": market_type,
        "status": "running",
        **state,
    }


@app.get("/api/dashboard/aggregate")
async def get_dashboard_aggregate(
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Single endpoint returning all dashboard data to minimize requests.

    Combines portfolio status, equity, signals, and market data.
    """
    store = get_store()

    # Get basic status
    state = store.get_state_dict()
    equity = store.get_equity_curve()
    signals = store.get_signals()

    # Get portfolio status if available
    portfolio_status = []
    config_payload = _load_portfolio_config_payload()
    asset_map = _build_asset_index(config_payload)
    portfolio_dir_str = config_payload.get("portfolio_data_dir") or str(STATE_DIR / "portfolio")
    portfolio_dir = Path(portfolio_dir_str).expanduser()

    for symbol_key in asset_map.keys():
        try:
            asset_data_dir = _resolve_asset_data_dir(symbol_key, asset_map, portfolio_dir)
            bot_state = load_bot_state_from_path(asset_data_dir)
            if bot_state:
                portfolio_status.append(
                    {
                        "symbol": symbol_key,
                        "position": bot_state.position,
                        "entry_price": bot_state.entry_price,
                        "entry_time": bot_state.entry_time,
                        "balance": bot_state.balance,
                    }
                )
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": {
            "position": state.get("position", "FLAT") if state else "FLAT",
            "balance": state.get("balance", 0) if state else 0,
        },
        "equity": [{"timestamp": e.get("timestamp"), "value": e.get("value")} for e in equity[-50:]] if equity else [],
        "signals": [
            {"timestamp": s.get("timestamp"), "action": s.get("decision"), "confidence": s.get("confidence")}
            for s in signals[-20:]
        ] if signals else [],
        "portfolio_status": portfolio_status,
        "markets_available": ["crypto", "commodity", "stock"],
    }


@app.get("/dashboard/unified")
async def dashboard_unified(request: Request):
    """Redirect /dashboard/unified to root for backwards compatibility."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/", status_code=302)


# =============================================================================
# OHLCV Chart Data Endpoint
# =============================================================================

# Simple in-memory cache for OHLCV data to avoid rate limits
_ohlcv_cache: Dict[str, Dict[str, Any]] = {}
_OHLCV_CACHE_TTL_SECONDS = 60  # Cache for 1 minute


def _get_cached_ohlcv(symbol: str) -> Optional[Dict[str, Any]]:
    """Get cached OHLCV data if still valid."""
    if symbol not in _ohlcv_cache:
        return None
    cached = _ohlcv_cache[symbol]
    if time.time() - cached["timestamp"] > _OHLCV_CACHE_TTL_SECONDS:
        return None
    return cached["data"]


def _set_cached_ohlcv(symbol: str, data: Dict[str, Any]) -> None:
    """Cache OHLCV data."""
    _ohlcv_cache[symbol] = {
        "timestamp": time.time(),
        "data": data,
    }


@app.get("/api/charts/{symbol:path}/ohlcv")
async def get_chart_ohlcv(
    symbol: str,
    period: str = Query(default="30d", description="Data period (e.g., 7d, 30d, 90d)"),
    interval: str = Query(default="1h", description="Candle interval (e.g., 15m, 1h, 4h, 1d)"),
    limit: int = Query(default=100, ge=10, le=500, description="Number of candles to return"),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get OHLCV (candlestick) data for a symbol.

    Returns data formatted for TradingView lightweight-charts:
    - candles: Array of {time, open, high, low, close} objects
    - volume: Array of {time, value, color} objects
    - sma20: Array of {time, value} objects for 20-period SMA

    Args:
        symbol: Trading symbol (e.g., BTC/USDT, XAU/USD, AAPL)
        period: Historical data period
        interval: Candle interval
        limit: Maximum number of candles to return

    Data is cached for 60 seconds to avoid rate limits.
    """
    # Normalize symbol for cache key
    cache_key = f"{symbol}:{period}:{interval}"

    # Check cache first
    cached_data = _get_cached_ohlcv(cache_key)
    if cached_data:
        # Apply limit to cached data
        return {
            "symbol": symbol,
            "interval": interval,
            "cached": True,
            "candles": cached_data["candles"][-limit:],
            "volume": cached_data["volume"][-limit:],
            "sma20": cached_data["sma20"][-limit:],
        }

    try:
        import yfinance as yf
        import pandas as pd

        # Convert symbol to yfinance format
        yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")

        # Some symbols need special handling
        symbol_map = {
            "XAU-USD": "GC=F",  # Gold futures
            "XAG-USD": "SI=F",  # Silver futures
            "USOIL-USD": "CL=F",  # Crude oil futures
            "NATGAS-USD": "NG=F",  # Natural gas futures
        }
        yf_symbol = symbol_map.get(yf_symbol, yf_symbol)

        # Fetch data from yfinance
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for symbol {symbol}")

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Calculate 20-period SMA
        df["sma20"] = df["close"].rolling(window=20).mean()

        # Convert timestamps to Unix time (seconds)
        df = df.reset_index()
        if "date" in df.columns:
            df["time"] = df["date"].apply(lambda x: int(x.timestamp()))
        elif "datetime" in df.columns:
            df["time"] = df["datetime"].apply(lambda x: int(x.timestamp()))
        else:
            # Try the index
            df["time"] = pd.to_datetime(df.iloc[:, 0]).apply(lambda x: int(x.timestamp()))

        # Build candle data for TradingView
        candles = []
        for _, row in df.iterrows():
            candles.append(
                {
                    "time": int(row["time"]),
                    "open": round(float(row["open"]), 6),
                    "high": round(float(row["high"]), 6),
                    "low": round(float(row["low"]), 6),
                    "close": round(float(row["close"]), 6),
                }
            )

        # Build volume data
        volume = []
        for _, row in df.iterrows():
            color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            volume.append(
                {
                    "time": int(row["time"]),
                    "value": float(row["volume"]) if pd.notna(row["volume"]) else 0,
                    "color": color,
                }
            )

        # Build SMA data (skip NaN values)
        sma20 = []
        for _, row in df.iterrows():
            if pd.notna(row["sma20"]):
                sma20.append(
                    {
                        "time": int(row["time"]),
                        "value": round(float(row["sma20"]), 6),
                    }
                )

        # Cache the full result
        full_data = {
            "candles": candles,
            "volume": volume,
            "sma20": sma20,
        }
        _set_cached_ohlcv(cache_key, full_data)

        return {
            "symbol": symbol,
            "interval": interval,
            "cached": False,
            "candles": candles[-limit:],
            "volume": volume[-limit:],
            "sma20": sma20[-limit:],
        }

    except ImportError:
        raise HTTPException(
            status_code=503, detail="yfinance not installed. Run: pip install yfinance"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV data: {str(e)}")


# =============================================================================
# WebSocket Endpoint for Real-Time Updates
# =============================================================================


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time portfolio updates.

    Broadcasts portfolio data every 3 seconds including:
    - Current prices for all tracked assets
    - Trading signals with confidence levels
    - Portfolio value and P&L
    - Open positions across all markets

    Clients should handle reconnection on disconnect.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial data immediately upon connection
        initial_payload = _get_ws_update_payload()
        initial_payload["type"] = "initial"
        await websocket.send_json(initial_payload)

        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for any client message (ping/pong or commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0,  # 60 second timeout for client pings
                )
                # Handle ping messages
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send a keep-alive ping
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.disconnect(websocket)


@app.get("/ws/status")
async def websocket_status() -> Dict[str, Any]:
    """Get WebSocket connection status."""
    return {
        "active_connections": ws_manager.get_connection_count(),
        "broadcast_running": ws_manager._running,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

from fastapi.responses import PlainTextResponse


def _calculate_performance_metrics(equity_data: List[float]) -> Dict[str, float]:
    """Calculate portfolio performance metrics from equity curve."""
    if not equity_data or len(equity_data) < 2:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility": 0.0,
            "total_return": 0.0,
            "total_return_pct": 0.0,
        }

    import numpy as np

    values = np.array(equity_data, dtype=np.float64)
    returns = np.diff(values) / values[:-1]

    # Total return
    initial = values[0]
    final = values[-1]
    total_return = final - initial
    total_return_pct = (total_return / initial) * 100 if initial > 0 else 0

    # Volatility (annualized, assuming hourly data)
    volatility = np.std(returns) * np.sqrt(365 * 24)

    # Sharpe Ratio (annualized, risk-free rate = 0.02)
    risk_free_rate = 0.02 / (365 * 24)  # Hourly risk-free rate
    excess_returns = returns - risk_free_rate
    sharpe_ratio = 0.0
    if len(excess_returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(365 * 24)

    # Sortino Ratio (only considers downside volatility)
    downside_returns = returns[returns < 0]
    sortino_ratio = 0.0
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(365 * 24)

    # Max Drawdown
    peak = values[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in values:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = (dd / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    return {
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "sortino_ratio": round(float(sortino_ratio), 4),
        "max_drawdown": round(float(max_dd), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "volatility": round(float(volatility), 4),
        "total_return": round(float(total_return), 2),
        "total_return_pct": round(float(total_return_pct), 2),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.

    Exposes trading metrics in Prometheus text format for monitoring:
    - Portfolio value and P&L
    - Position counts and signal statistics
    - API health and performance
    - WebSocket connection counts

    Example scrape config for Prometheus:
    ```yaml
    scrape_configs:
      - job_name: 'trading_bot'
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: '/metrics'
    ```
    """
    lines = []
    lines.append("# HELP trading_bot_info Trading bot version and configuration")
    lines.append("# TYPE trading_bot_info gauge")
    lines.append('trading_bot_info{version="0.2.0"} 1')

    # API metrics
    uptime = time.time() - _app_start_time
    lines.append("")
    lines.append("# HELP trading_api_uptime_seconds API uptime in seconds")
    lines.append("# TYPE trading_api_uptime_seconds gauge")
    lines.append(f"trading_api_uptime_seconds {round(uptime, 2)}")

    lines.append("")
    lines.append("# HELP trading_websocket_connections Active WebSocket connections")
    lines.append("# TYPE trading_websocket_connections gauge")
    lines.append(f"trading_websocket_connections {ws_manager.get_connection_count()}")

    # Load bot states for each market (check production first for crypto)
    state_dirs = {
        "crypto": [
            STATE_DIR / "production" / "state.json",  # Live/testnet trading
            STATE_DIR / "live_paper_trading" / "state.json",
        ],
        "commodity": [STATE_DIR / "commodity_trading" / "state.json"],
        "stock": [STATE_DIR / "stock_trading" / "state.json"],
        "unified": [Path("data/unified_trading/state.json")],  # Unified engine
    }

    total_portfolio_value = 0.0
    total_pnl = 0.0
    total_positions = 0

    for market_type, state_files in state_dirs.items():
        # Find the first existing state file
        state_file = None
        for sf in state_files:
            if sf.exists():
                state_file = sf
                break
        if state_file:
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                # Handle both unified and legacy state formats
                portfolio_value = state_data.get("total_value", state_data.get("current_balance", 0))
                cash_balance = state_data.get("cash_balance", state_data.get("current_balance", 0))
                pnl = state_data.get("pnl", state_data.get("total_pnl", 0))
                initial_capital = state_data.get("initial_capital", 10000)
                pnl_pct = state_data.get("pnl_pct", pnl / initial_capital * 100 if initial_capital else 0)
                positions_count = state_data.get("positions_count", len(state_data.get("positions", {})))
                signals = state_data.get("signals", {})

                # Add unified trading specific metrics
                if market_type == "unified":
                    total_trades = state_data.get("total_trades", 0)
                    win_rate = state_data.get("win_rate", 0)
                    max_drawdown = state_data.get("max_drawdown_pct", 0)
                    mode = state_data.get("mode", "paper")
                    status = state_data.get("status", "stopped")
                    lines.append("")
                    lines.append("# HELP trading_unified_total_trades Total trades in unified engine")
                    lines.append("# TYPE trading_unified_total_trades counter")
                    lines.append(f"trading_unified_total_trades {total_trades}")
                    lines.append(f"trading_unified_win_rate {round(win_rate * 100, 2)}")
                    lines.append(f"trading_unified_max_drawdown_pct {round(max_drawdown * 100, 2)}")
                    lines.append(f'trading_unified_mode{{mode="{mode}"}} 1')
                    lines.append(f'trading_unified_status{{status="{status}"}} 1')

                total_portfolio_value += portfolio_value
                total_pnl += pnl
                total_positions += positions_count

                # Portfolio value
                lines.append("")
                lines.append(f"# HELP trading_portfolio_value_{market_type} Portfolio value in USD")
                lines.append(f"# TYPE trading_portfolio_value_{market_type} gauge")
                lines.append(f"trading_portfolio_value_{market_type} {round(portfolio_value, 2)}")

                # Cash balance
                lines.append(f"trading_cash_balance_{market_type} {round(cash_balance, 2)}")

                # P&L
                lines.append(f"trading_pnl_{market_type} {round(pnl, 2)}")
                lines.append(f"trading_pnl_pct_{market_type} {round(pnl_pct, 4)}")

                # Position count
                lines.append(f"trading_positions_count_{market_type} {positions_count}")

                # Signal counts
                long_count = sum(
                    1 for s in signals.values() if isinstance(s, dict) and s.get("signal") == "LONG"
                )
                short_count = sum(
                    1
                    for s in signals.values()
                    if isinstance(s, dict) and s.get("signal") == "SHORT"
                )
                flat_count = sum(
                    1 for s in signals.values() if isinstance(s, dict) and s.get("signal") == "FLAT"
                )

                lines.append(f"trading_signals_long_{market_type} {long_count}")
                lines.append(f"trading_signals_short_{market_type} {short_count}")
                lines.append(f"trading_signals_flat_{market_type} {flat_count}")

            except Exception:
                pass

    # Total aggregates
    lines.append("")
    lines.append("# HELP trading_total_portfolio_value Total portfolio value across all markets")
    lines.append("# TYPE trading_total_portfolio_value gauge")
    lines.append(f"trading_total_portfolio_value {round(total_portfolio_value, 2)}")

    lines.append("")
    lines.append("# HELP trading_total_pnl Total P&L across all markets")
    lines.append("# TYPE trading_total_pnl gauge")
    lines.append(f"trading_total_pnl {round(total_pnl, 2)}")

    lines.append("")
    lines.append("# HELP trading_total_positions Total open positions across all markets")
    lines.append("# TYPE trading_total_positions gauge")
    lines.append(f"trading_total_positions {total_positions}")

    # Try to load equity data and calculate performance metrics
    try:
        equity_file = STATE_DIR / "equity.json"
        if equity_file.exists():
            with open(equity_file, "r") as f:
                equity_data = json.load(f)
            if isinstance(equity_data, list) and equity_data:
                values = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                values = [v for v in values if isinstance(v, (int, float))]
                if values:
                    metrics = _calculate_performance_metrics(values)

                    lines.append("")
                    lines.append("# HELP trading_sharpe_ratio Annualized Sharpe ratio")
                    lines.append("# TYPE trading_sharpe_ratio gauge")
                    lines.append(f"trading_sharpe_ratio {metrics['sharpe_ratio']}")

                    lines.append("")
                    lines.append("# HELP trading_sortino_ratio Annualized Sortino ratio")
                    lines.append("# TYPE trading_sortino_ratio gauge")
                    lines.append(f"trading_sortino_ratio {metrics['sortino_ratio']}")

                    lines.append("")
                    lines.append("# HELP trading_max_drawdown_pct Maximum drawdown percentage")
                    lines.append("# TYPE trading_max_drawdown_pct gauge")
                    lines.append(f"trading_max_drawdown_pct {metrics['max_drawdown_pct']}")

                    lines.append("")
                    lines.append("# HELP trading_volatility Annualized volatility")
                    lines.append("# TYPE trading_volatility gauge")
                    lines.append(f"trading_volatility {metrics['volatility']}")
    except Exception:
        pass

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


@app.get("/api/portfolio/stats")
async def get_portfolio_stats(
    market_type: str = Query(
        default="all", description="Market type: crypto, commodity, stock, all"
    ),
    _: bool = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get detailed portfolio performance metrics.

    Returns Sharpe ratio, Sortino ratio, max drawdown, and other metrics.
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_type": market_type,
        "metrics": {},
        "markets": {},
    }

    # Get equity data for performance calculation
    # Check production state first for crypto
    state_dirs = {
        "crypto": [
            STATE_DIR / "production" / "state.json",  # Live/testnet trading
            STATE_DIR / "live_paper_trading" / "state.json",
        ],
        "commodity": [STATE_DIR / "commodity_trading" / "state.json"],
        "stock": [STATE_DIR / "stock_trading" / "state.json"],
    }

    all_equity_values = []

    for mtype, state_files in state_dirs.items():
        if market_type != "all" and mtype != market_type:
            continue

        # Find the newest state file for this market
        state_file = None
        for sf in state_files:
            if sf.exists():
                state_file = sf
                break

        if state_file and state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                total_value = state_data.get("total_value", 0)
                initial_capital = state_data.get("initial_capital", 10000)

                # Try to load market-specific equity data
                equity_file = state_file.parent / "equity.json"
                if equity_file.exists():
                    with open(equity_file, "r") as f:
                        equity_data = json.load(f)
                    values = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                    values = [v for v in values if isinstance(v, (int, float))]
                else:
                    # Use current value as single data point
                    values = [initial_capital, total_value] if total_value > 0 else []

                if values:
                    all_equity_values.extend(values)
                    metrics = _calculate_performance_metrics(values)
                    result["markets"][mtype] = {
                        "current_value": total_value,
                        "initial_capital": initial_capital,
                        **metrics,
                    }
            except Exception as e:
                result["markets"][mtype] = {"error": str(e)}

    # Calculate overall metrics
    if all_equity_values:
        result["metrics"] = _calculate_performance_metrics(all_equity_values)

    return result


# =============================================================================
# Walk-Forward Analysis Endpoints
# =============================================================================


@app.get("/api/walk-forward/results", tags=["ML/AI"])
async def get_walk_forward_results(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
) -> Dict[str, Any]:
    """Get walk-forward validation results."""
    try:
        from bot.walk_forward_ui import WalkForwardUIProvider

        provider = WalkForwardUIProvider()
        results = provider.get_latest_results(symbol)

        if results is None:
            return {"status": "no_results", "message": "No walk-forward results found"}

        return provider.to_api_response(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/walk-forward/history", tags=["ML/AI"])
async def get_walk_forward_history() -> Dict[str, Any]:
    """Get history of all walk-forward validation runs."""
    try:
        from bot.walk_forward_ui import WalkForwardUIProvider

        provider = WalkForwardUIProvider()
        results = provider.get_all_results()

        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trade Journal Export Endpoints
# =============================================================================


@app.get("/api/trade-history", tags=["Portfolio"])
async def get_trade_history(
    limit: int = Query(default=50, description="Number of trades to return"),
    market: str = Query(
        default="all", description="Filter by market: all, crypto, commodity, stock"
    ),
) -> Dict[str, Any]:
    """Get trade history from all sources."""
    import json
    from datetime import datetime

    all_trades = []

    # Trade history sources
    trade_sources = [
        STATE_DIR / "strategy_tracker" / "trade_history.json",
        STATE_DIR / "live_paper_trading" / "state.json",
        STATE_DIR / "regime_trading" / "state.json",
        STATE_DIR / "commodity_trading" / "state.json",
        STATE_DIR / "stock_trading" / "state.json",
    ]

    for source in trade_sources:
        if source.exists():
            try:
                with open(source, "r") as f:
                    data = json.load(f)

                # Handle different formats
                trades = []
                if isinstance(data, list):
                    trades = data
                elif isinstance(data, dict):
                    trades = data.get("trades", data.get("trade_history", []))

                for trade in trades:
                    if isinstance(trade, dict):
                        # Normalize trade format
                        normalized = {
                            "date": trade.get(
                                "exit_time", trade.get("entry_time", trade.get("timestamp", ""))
                            ),
                            "symbol": trade.get("symbol", ""),
                            "side": trade.get("side", "long"),
                            "entry": trade.get("entry_price", 0),
                            "exit": trade.get("exit_price", trade.get("entry_price", 0)),
                            "pnl": trade.get("pnl", 0),
                            "pnl_pct": trade.get("pnl_pct", 0),
                            "regime": trade.get("regime", ""),
                            "exit_reason": trade.get("exit_reason", ""),
                            "quantity": trade.get("quantity", 0),
                        }

                        # Determine market type from symbol
                        symbol = normalized["symbol"].upper()
                        if "USDT" in symbol or "BTC" in symbol or "ETH" in symbol:
                            normalized["market"] = "crypto"
                        elif any(
                            c in symbol for c in ["XAU", "XAG", "OIL", "GAS", "GOLD", "SILVER"]
                        ):
                            normalized["market"] = "commodity"
                        else:
                            normalized["market"] = "stock"

                        all_trades.append(normalized)
            except Exception as e:
                continue

    # Filter by market
    if market != "all":
        all_trades = [t for t in all_trades if t.get("market") == market]

    # Sort by date (newest first)
    all_trades.sort(key=lambda x: x.get("date", ""), reverse=True)

    # Calculate summary stats
    total_pnl = sum(t.get("pnl", 0) for t in all_trades)
    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in all_trades if t.get("pnl", 0) < 0)
    win_rate = (wins / len(all_trades) * 100) if all_trades else 0

    return {
        "trades": all_trades[:limit],
        "total_trades": len(all_trades),
        "summary": {
            "total_pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
        },
    }


# =============================================================================
# Data Storage & Backup Endpoints
# =============================================================================


@app.get("/api/storage/stats", tags=["Storage"])
async def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics and metadata."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        return store.get_storage_stats()
    except ImportError:
        return {"error": "Data store module not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/storage/trades", tags=["Storage"])
async def get_stored_trades(
    market: str = Query(default=None, description="Filter by market"),
    symbol: str = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=100, description="Number of trades to return"),
) -> Dict[str, Any]:
    """Get trades from persistent data store."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        trades = store.get_trades(market=market, symbol=symbol, limit=limit)
        summary = store.get_trade_summary()
        return {"trades": trades, "summary": summary}
    except ImportError:
        return {"trades": [], "error": "Data store module not available"}
    except Exception as e:
        return {"trades": [], "error": str(e)}


@app.get("/api/storage/snapshots", tags=["Storage"])
async def get_portfolio_snapshots(
    days: int = Query(default=30, description="Number of days of history"),
) -> Dict[str, Any]:
    """Get portfolio snapshots (equity curve data)."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        snapshots = store.get_equity_curve(days=days)
        return {"snapshots": snapshots, "count": len(snapshots)}
    except ImportError:
        return {"snapshots": [], "error": "Data store module not available"}
    except Exception as e:
        return {"snapshots": [], "error": str(e)}


@app.get("/api/storage/signals", tags=["Storage"])
async def get_signal_history(
    symbol: str = Query(default=None, description="Filter by symbol"),
    market: str = Query(default=None, description="Filter by market"),
    limit: int = Query(default=100, description="Number of signals to return"),
) -> Dict[str, Any]:
    """Get historical trading signals."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        signals = store.get_signals(symbol=symbol, market=market, limit=limit)
        return {"signals": signals, "count": len(signals)}
    except ImportError:
        return {"signals": [], "error": "Data store module not available"}
    except Exception as e:
        return {"signals": [], "error": str(e)}


@app.post("/api/storage/backup", tags=["Storage"])
async def create_backup() -> Dict[str, Any]:
    """Create a full backup of all trading data."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        backup_path = store.create_backup()
        return {
            "status": "success",
            "message": "Backup created successfully",
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat(),
        }
    except ImportError:
        return {"status": "error", "message": "Data store module not available"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/storage/export", tags=["Storage"])
async def export_all_data() -> Dict[str, Any]:
    """Export all data to a single JSON file for server migration."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        export_path = store.export_all()
        return {
            "status": "success",
            "message": "Data exported successfully",
            "export_path": export_path,
            "timestamp": datetime.now().isoformat(),
        }
    except ImportError:
        return {"status": "error", "message": "Data store module not available"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/storage/strategies", tags=["Storage"])
async def get_strategy_performance(
    strategy_id: str = Query(default=None, description="Strategy ID"),
) -> Dict[str, Any]:
    """Get strategy performance metrics."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        performance = store.get_strategy_performance(strategy_id=strategy_id)
        return {"strategies": performance}
    except ImportError:
        return {"strategies": {}, "error": "Data store module not available"}
    except Exception as e:
        return {"strategies": {}, "error": str(e)}


@app.get("/api/storage/models", tags=["Storage"])
async def get_registered_models(
    model_type: str = Query(default=None, description="Filter by model type"),
    symbol: str = Query(default=None, description="Filter by symbol"),
) -> Dict[str, Any]:
    """Get registered ML models."""
    try:
        from bot.data_store import get_data_store

        store = get_data_store()
        models = store.get_models(model_type=model_type, symbol=symbol)
        return {"models": models, "count": len(models)}
    except ImportError:
        return {"models": {}, "error": "Data store module not available"}
    except Exception as e:
        return {"models": {}, "error": str(e)}


@app.get("/api/ml/model-status", tags=["ML/AI"])
async def get_ml_model_status(
    market_type: str = Query(default="crypto", description="Market type: crypto, stock, commodity"),
) -> Dict[str, Any]:
    """
    Get comprehensive ML model status for all symbols.

    Shows:
    - Which symbols have trained models
    - Model types available (LSTM, Transformer, RandomForest, etc.)
    - Model accuracy/performance
    - Training date
    - Symbols without models
    """
    symbols_with_models: Dict[str, List[Dict]] = {}
    model_types_found = set()

    # Try registry first
    try:
        from bot.ml.registry.model_registry import ModelRegistry

        registry = ModelRegistry()
        all_models = registry.list_models(market_type=market_type, active_only=True)

        for model in all_models:
            symbol = model.symbol
            if symbol not in symbols_with_models:
                symbols_with_models[symbol] = []
            symbols_with_models[symbol].append(
                {
                    "model_type": model.model_type,
                    "accuracy": round(model.accuracy * 100, 1),
                    "val_accuracy": round(model.val_accuracy * 100, 1),
                    "created_at": model.created_at.isoformat(),
                    "is_active": model.is_active,
                    "source": "registry",
                }
            )
            model_types_found.add(model.model_type)
    except Exception as e:
        print(f"Registry not available: {e}")

    # Fallback: scan data/models directory directly
    models_dir = STATE_DIR / "models"
    if models_dir.exists():
        try:
            # Define which symbols belong to which market
            market_symbols = {
                "crypto": ["BTC", "ETH", "SOL", "AVAX", "XRP", "ADA", "DOT", "LINK"],
                "stock": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
                "commodity": ["XAU", "XAG", "USOIL", "NATGAS"],
            }
            target_symbols = market_symbols.get(market_type, [])

            for model_file in models_dir.glob("*_model.pkl"):
                model_name = model_file.stem.replace("_model", "")
                parts = model_name.split("_")

                if len(parts) >= 3:
                    base_symbol = parts[0]

                    # Check if this symbol belongs to requested market
                    if base_symbol not in target_symbols:
                        continue

                    # Reconstruct symbol
                    if parts[1] == "USDT":
                        symbol = f"{parts[0]}/USDT"
                    elif parts[1] == "USD":
                        symbol = f"{parts[0]}/USD"
                    else:
                        symbol = parts[0]

                    # Skip if already in registry results
                    if symbol in symbols_with_models:
                        # Check if this model type already exists
                        existing_types = [m["model_type"] for m in symbols_with_models[symbol]]
                        model_type = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
                        if model_type in existing_types:
                            continue

                    # Get model type from filename
                    model_type = "_".join(parts[2:]) if len(parts) > 2 else "unknown"

                    # Try to load metadata
                    meta_file = models_dir / f"{model_name}_meta.json"
                    accuracy = 55.0
                    created_at = datetime.now(timezone.utc)

                    if meta_file.exists():
                        try:
                            with open(meta_file) as f:
                                meta = json.load(f)
                                raw_accuracy = meta.get("metrics", {}).get("accuracy", 0.55)
                                # Handle both decimal (0.93) and percentage (93.1) formats
                                if raw_accuracy <= 1.0:
                                    accuracy = raw_accuracy * 100
                                else:
                                    accuracy = raw_accuracy
                                if "saved_at" in meta:
                                    created_at = datetime.fromisoformat(
                                        meta["saved_at"].replace("Z", "+00:00")
                                    )
                        except Exception as parse_err:
                            print(f"Error parsing meta file {meta_file}: {parse_err}")
                            pass

                    if symbol not in symbols_with_models:
                        symbols_with_models[symbol] = []

                    symbols_with_models[symbol].append(
                        {
                            "model_type": model_type,
                            "accuracy": round(accuracy, 1),
                            "val_accuracy": round(accuracy, 1),
                            "created_at": created_at.isoformat(),
                            "is_active": True,
                            "source": "data/models",
                        }
                    )
                    model_types_found.add(model_type)

        except Exception as e:
            print(f"Error scanning models directory: {e}")

    # Define expected symbols per market
    expected_symbols = {
        "crypto": [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOT/USDT",
            "LINK/USDT",
        ],
        "stock": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
        "commodity": ["XAU/USD", "XAG/USD", "USOIL/USD", "NATGAS/USD"],
    }

    expected = expected_symbols.get(market_type, [])
    symbols_without_models = [s for s in expected if s not in symbols_with_models]

    # Calculate summary stats
    total_models = sum(len(models) for models in symbols_with_models.values())
    avg_accuracy = 0
    if total_models > 0:
        all_accuracies = [
            m["val_accuracy"] for models in symbols_with_models.values() for m in models
        ]
        avg_accuracy = sum(all_accuracies) / len(all_accuracies)

    return {
        "market_type": market_type,
        "summary": {
            "total_models": total_models,
            "symbols_with_models": len(symbols_with_models),
            "symbols_without_models": len(symbols_without_models),
            "average_accuracy": round(avg_accuracy, 1),
            "coverage_pct": round(len(symbols_with_models) / max(len(expected), 1) * 100, 1),
        },
        "symbols": symbols_with_models,
        "missing_symbols": symbols_without_models,
        "model_types_available": list(model_types_found),
    }


@app.get("/api/ml/data-freshness", tags=["ML/AI"])
async def get_data_freshness_status() -> Dict[str, Any]:
    """
    Get data freshness status for all tracked symbols.

    Shows data age and whether each symbol is tradeable.
    """
    try:
        from bot.data_freshness import get_monitor

        monitor = get_monitor()
        return monitor.get_summary()
    except ImportError:
        return {"error": "Data freshness module not available", "symbols": {}}
    except Exception as e:
        return {"error": str(e), "symbols": {}}


@app.get("/api/trade-journal/export", tags=["Portfolio"])
async def export_trade_journal(
    format: str = Query(default="csv", description="Export format: csv, excel, json"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
) -> Dict[str, Any]:
    """Export trade journal in specified format."""
    try:
        from bot.trade_journal import TradeJournal

        journal = TradeJournal()

        # Try to load from database first, then JSON
        db_path = STATE_DIR / "trading.db"
        json_path = STATE_DIR / "live_paper_trading" / "state.json"

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        if db_path.exists():
            journal.load_trades_from_db(str(db_path), start, end, symbol)
        elif json_path.exists():
            journal.load_trades_from_json(str(json_path))

        if format == "csv":
            filepath = journal.export_to_csv()
        elif format == "excel":
            filepath = journal.export_to_excel()
        elif format == "json":
            filepath = journal.export_to_json()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {format}")

        return {
            "status": "success",
            "filepath": filepath,
            "format": format,
            "trades_exported": len(journal._trades),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trade-journal/summary", tags=["Portfolio"])
async def get_trade_journal_summary() -> Dict[str, Any]:
    """Get trade journal summary statistics."""
    try:
        from bot.trade_journal import TradeJournal

        journal = TradeJournal()
        json_path = STATE_DIR / "live_paper_trading" / "state.json"

        if json_path.exists():
            journal.load_trades_from_json(str(json_path))

        return {
            "summary": journal.calculate_summary(journal._trades),
            "by_strategy": journal.calculate_by_strategy(journal._trades),
            "daily_performance": journal.calculate_daily_performance(journal._trades),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Strategy Comparison Endpoints
# =============================================================================


@app.get("/api/strategy/comparison", tags=["ML/AI"])
async def get_strategy_comparison() -> Dict[str, Any]:
    """Get side-by-side strategy comparison."""
    try:
        from bot.strategy_comparison import StrategyComparator

        comparator = StrategyComparator()
        comparator.load_strategy_results(
            backtest_dir=str(STATE_DIR / "backtest_results"),
            live_dir=str(STATE_DIR / "live_paper_trading"),
        )

        if not comparator._strategies:
            return {"status": "no_data", "message": "No strategy results found"}

        comparison = comparator.compare()
        return comparator.to_api_response(comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategy/metrics", tags=["ML/AI"])
async def get_strategy_metrics() -> Dict[str, Any]:
    """Get detailed metrics for all strategies."""
    try:
        from bot.strategy_comparison import StrategyComparator

        comparator = StrategyComparator()
        comparator.load_strategy_results()

        return {"strategies": comparator.get_metrics_table()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Drawdown Analysis Endpoints
# =============================================================================


@app.get("/api/drawdown/analysis", tags=["Portfolio"])
async def get_drawdown_analysis() -> Dict[str, Any]:
    """Get comprehensive drawdown analysis."""
    try:
        from bot.drawdown_analysis import DrawdownAnalyzer

        analyzer = DrawdownAnalyzer()

        # Load equity curve
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"
        if equity_path.exists():
            analyzer.load_equity_from_json(str(equity_path))
        else:
            # Try portfolio equity
            portfolio_equity = STATE_DIR / "portfolio" / "equity.json"
            if portfolio_equity.exists():
                analyzer.load_equity_from_json(str(portfolio_equity))
            else:
                return {"status": "no_data", "message": "No equity data found"}

        analysis = analyzer.analyze()
        return analyzer.to_api_response(analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drawdown/top", tags=["Portfolio"])
async def get_top_drawdowns(
    n: int = Query(default=5, description="Number of top drawdowns to return"),
) -> Dict[str, Any]:
    """Get the top N largest drawdown periods."""
    try:
        from bot.drawdown_analysis import DrawdownAnalyzer

        analyzer = DrawdownAnalyzer()
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"

        if equity_path.exists():
            analyzer.load_equity_from_json(str(equity_path))
            return {"top_drawdowns": analyzer.get_top_drawdowns(n)}
        else:
            return {"status": "no_data", "top_drawdowns": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PnL Notifications Endpoints
# =============================================================================


@app.get("/api/notifications/status", tags=["Status"])
async def get_notification_status() -> Dict[str, Any]:
    """Get PnL notification status and configuration."""
    try:
        from bot.pnl_notifications import PnLNotificationManager

        manager = PnLNotificationManager()
        return manager.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notifications/update-equity", tags=["Status"])
async def update_notification_equity(
    equity: float = Query(..., description="Current equity value"),
) -> Dict[str, Any]:
    """Update equity for notification tracking."""
    try:
        from bot.pnl_notifications import PnLNotificationManager

        manager = PnLNotificationManager()
        notifications = manager.update_equity(equity)

        return {
            "notifications_triggered": len(notifications),
            "notifications": [
                {
                    "type": n.type.value,
                    "title": n.title,
                    "message": n.message,
                    "priority": n.priority.value,
                }
                for n in notifications
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Multi-Timeframe Analysis Endpoints
# =============================================================================


@app.get("/api/multi-timeframe/analysis", tags=["ML/AI"])
async def get_multi_timeframe_analysis(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
) -> Dict[str, Any]:
    """Get multi-timeframe analysis for a symbol."""
    try:
        from bot.multi_timeframe import MultiTimeframeAnalyzer
        import pandas as pd

        analyzer = MultiTimeframeAnalyzer()

        # Try to load data from market data cache
        data_path = STATE_DIR / "market_data" / f"{sanitize_symbol_for_fs(symbol)}_1h.json"

        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

            analyzer.load_data(symbol, df)
            result = analyzer.analyze(symbol)
            return analyzer.to_api_response(result)
        else:
            return {
                "status": "no_data",
                "message": f"No market data found for {symbol}",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Risk Metrics Endpoints
# =============================================================================


@app.get("/api/risk/metrics", tags=["Portfolio"])
async def get_risk_metrics() -> Dict[str, Any]:
    """Get comprehensive risk metrics for the portfolio."""
    try:
        from bot.risk_metrics import RiskMetricsCalculator

        calculator = RiskMetricsCalculator()

        # Load equity curve
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"
        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)

            if isinstance(equity_data, list):
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]
                calculator.load_equity_curve(equity)

        return calculator.to_api_response()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/breakdown", tags=["Portfolio"])
async def get_risk_breakdown() -> Dict[str, Any]:
    """Get categorized risk metrics breakdown."""
    try:
        from bot.risk_metrics import RiskMetricsCalculator

        calculator = RiskMetricsCalculator()
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"

        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)

            if isinstance(equity_data, list):
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]
                calculator.load_equity_curve(equity)

        return calculator.get_risk_breakdown()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/rolling", tags=["Portfolio"])
async def get_rolling_risk_metrics(
    window: int = Query(default=30, description="Rolling window size"),
) -> Dict[str, Any]:
    """Get rolling risk metrics over time."""
    try:
        from bot.risk_metrics import RiskMetricsCalculator

        calculator = RiskMetricsCalculator()
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"

        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)

            if isinstance(equity_data, list):
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]
                calculator.load_equity_curve(equity)

        return {"rolling_metrics": calculator.get_rolling_metrics(window)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Order Book Endpoints
# =============================================================================


@app.get("/api/orderbook/snapshot", tags=["Markets"])
async def get_orderbook_snapshot(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
) -> Dict[str, Any]:
    """Get order book snapshot with liquidity metrics."""
    try:
        from bot.orderbook import OrderBookAnalyzer

        analyzer = OrderBookAnalyzer()

        # Try to load cached orderbook data
        orderbook_path = (
            STATE_DIR / "market_data" / f"{sanitize_symbol_for_fs(symbol)}_orderbook.json"
        )

        if orderbook_path.exists():
            with open(orderbook_path) as f:
                data = json.load(f)

            snapshot = analyzer.process_raw_orderbook(
                bids=data.get("bids", []),
                asks=data.get("asks", []),
                symbol=symbol,
            )

            return analyzer.to_api_response(snapshot)
        else:
            return {
                "status": "no_data",
                "message": f"No orderbook data cached for {symbol}. Live fetching requires exchange connection.",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Calendar View Endpoints
# =============================================================================


@app.get("/api/calendar/performance", tags=["Portfolio"])
async def get_calendar_performance() -> Dict[str, Any]:
    """Get calendar heatmap data for daily performance."""
    try:
        from bot.calendar_view import CalendarViewGenerator

        generator = CalendarViewGenerator()

        # Load trades
        state_path = STATE_DIR / "live_paper_trading" / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)

            trades = data.get("trades", [])
            generator.load_trades(trades)

        calendar_data = generator.generate_calendar()
        return generator.to_api_response(calendar_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calendar/month", tags=["Portfolio"])
async def get_calendar_month(
    year: int = Query(..., description="Year"),
    month: int = Query(..., description="Month (1-12)"),
) -> Dict[str, Any]:
    """Get calendar grid for a specific month."""
    try:
        from bot.calendar_view import CalendarViewGenerator

        generator = CalendarViewGenerator()

        state_path = STATE_DIR / "live_paper_trading" / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)

            trades = data.get("trades", [])
            generator.load_trades(trades)

        return {"year": year, "month": month, "grid": generator.get_month_grid(year, month)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Rate Limit Monitor Endpoints
# =============================================================================


@app.get("/api/rate-limits/status", tags=["Health"])
async def get_rate_limit_status() -> Dict[str, Any]:
    """Get API rate limit status across all providers."""
    try:
        from bot.rate_limit_monitor import get_rate_limit_monitor

        monitor = get_rate_limit_monitor()
        return monitor.to_api_response()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rate-limits/errors", tags=["Health"])
async def get_rate_limit_errors(
    api_name: Optional[str] = Query(default=None, description="Filter by API name"),
    limit: int = Query(default=20, description="Number of errors to return"),
) -> Dict[str, Any]:
    """Get recent rate limit errors."""
    try:
        from bot.rate_limit_monitor import get_rate_limit_monitor

        monitor = get_rate_limit_monitor()
        return {"errors": monitor.get_recent_errors(api_name, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Emergency Control Endpoints
# =============================================================================

# Global safety controller instance
_safety_controller = None


def _get_safety_controller():
    """Get or create the safety controller instance."""
    global _safety_controller
    if _safety_controller is None:
        from bot.safety_controller import SafetyController

        _safety_controller = SafetyController()
    return _safety_controller


@app.get("/api/emergency/status", tags=["Status"])
async def get_emergency_status() -> Dict[str, Any]:
    """
    Get current trading status and emergency stop state.

    Returns:
        Trading status, emergency stop state, daily stats, and system health.
    """
    controller = _get_safety_controller()
    status = controller.get_status()
    allowed, reason = controller.is_trading_allowed()

    # Get orchestrator status if available
    orchestrator_running = False
    active_markets = []

    try:
        # Check for running markets by looking at state files
        markets = ["crypto", "commodity", "stock"]
        for market in markets:
            state_paths = [
                STATE_DIR / "live_paper_trading" / "state.json",
                STATE_DIR / f"{market}_trading" / "state.json",
            ]
            for path in state_paths:
                if path.exists():
                    import time

                    mtime = path.stat().st_mtime
                    if time.time() - mtime < 120:  # Updated in last 2 minutes
                        active_markets.append(market)
                        orchestrator_running = True
                        break
    except Exception:
        pass

    return {
        "trading_allowed": allowed,
        "trading_status_reason": reason,
        "emergency_stop_active": status["emergency_stop_active"],
        "emergency_stop_reason": status["emergency_stop_reason"],
        "safety_status": status["status"],
        "orchestrator_running": orchestrator_running,
        "active_markets": list(set(active_markets)),
        "daily_stats": status["daily_stats"],
        "limits": status["limits"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/emergency/stop", tags=["Status"])
async def trigger_emergency_stop(
    reason: str = Query(
        default="Manual emergency stop via dashboard", description="Reason for emergency stop"
    ),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Trigger emergency stop - immediately halts all trading.

    This will:
    - Stop all new trade entries
    - Keep existing positions (does not auto-close)
    - Require manual resume to continue trading

    Args:
        reason: Reason for triggering emergency stop

    Returns:
        Confirmation and updated status
    """
    controller = _get_safety_controller()
    controller.emergency_stop(reason)

    # Also pause all portfolio bots
    try:
        portfolio_path = STATE_DIR / "portfolio.json"
        if portfolio_path.exists():
            with open(portfolio_path, "r") as f:
                portfolio = json.load(f)

            symbols = portfolio.get("symbols", [])
            for symbol in symbols:
                from bot.market_data import sanitize_symbol_for_fs

                symbol_dir = STATE_DIR / sanitize_symbol_for_fs(symbol)
                update_bot_control(symbol_dir, paused=True, reason=f"Emergency stop: {reason}")
    except Exception as e:
        pass  # Continue even if portfolio pause fails

    # Send Telegram notification
    try:
        from bot.trade_alerts import TelegramTradeAlerts

        alerts = TelegramTradeAlerts()
        alerts.send_risk_alert(
            "EMERGENCY_STOP",
            f"Trading has been halted.\n\nReason: {reason}",
            severity="critical",
        )
    except Exception:
        pass

    return {
        "success": True,
        "message": "Emergency stop triggered",
        "reason": reason,
        "status": controller.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/emergency/resume", tags=["Status"])
async def resume_trading(
    approver: str = Query(default="dashboard_user", description="Who approved the resume"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Clear emergency stop and resume trading.

    This will:
    - Clear the emergency stop flag
    - Allow new trades to be placed
    - Resume all paused portfolio bots

    Args:
        approver: Who is approving the resume (for audit trail)

    Returns:
        Confirmation and updated status
    """
    controller = _get_safety_controller()

    # Check if emergency stop is active
    status = controller.get_status()
    if not status["emergency_stop_active"]:
        return {
            "success": True,
            "message": "No emergency stop was active",
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Clear emergency stop
    controller.clear_emergency_stop(approver)

    # Resume all portfolio bots
    try:
        portfolio_path = STATE_DIR / "portfolio.json"
        if portfolio_path.exists():
            with open(portfolio_path, "r") as f:
                portfolio = json.load(f)

            symbols = portfolio.get("symbols", [])
            for symbol in symbols:
                from bot.market_data import sanitize_symbol_for_fs

                symbol_dir = STATE_DIR / sanitize_symbol_for_fs(symbol)
                update_bot_control(symbol_dir, paused=False, reason=f"Resumed by {approver}")
    except Exception:
        pass

    # Send Telegram notification
    try:
        from bot.trade_alerts import TelegramTradeAlerts

        alerts = TelegramTradeAlerts()
        alerts._send_html(
            f"✅ <b>Trading Resumed</b>\n\n"
            f"Emergency stop has been cleared.\n"
            f"Approved by: {approver}\n\n"
            f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
    except Exception:
        pass

    return {
        "success": True,
        "message": "Trading resumed",
        "approved_by": approver,
        "status": controller.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/emergency/live-status", tags=["Status"])
async def get_live_trading_status() -> Dict[str, Any]:
    """
    Get real-time trading activity status.

    Returns:
        Whether bots are actively trading, last signal time, open positions.
    """
    result = {
        "is_trading": False,
        "last_activity": None,
        "active_bots": [],
        "open_positions": 0,
        "last_signal": None,
        "health": "unknown",
    }

    try:
        # Check main paper trading state
        state_path = STATE_DIR / "live_paper_trading" / "state.json"
        if state_path.exists():
            mtime = state_path.stat().st_mtime
            last_update = datetime.fromtimestamp(mtime)
            result["last_activity"] = last_update.isoformat()

            # If updated in last 5 minutes, bot is active
            if (datetime.now() - last_update).total_seconds() < 300:
                result["is_trading"] = True
                result["active_bots"].append("crypto")
                result["health"] = "healthy"
            else:
                result["health"] = "stale"

            with open(state_path, "r") as f:
                state = json.load(f)
                result["open_positions"] = len(state.get("open_positions", []))

        # Check signals
        signals_path = STATE_DIR / "signals.json"
        if signals_path.exists():
            with open(signals_path, "r") as f:
                signals = json.load(f)
                if isinstance(signals, list) and signals:
                    result["last_signal"] = signals[-1]

        # Check other markets
        for market in ["commodity", "stock"]:
            market_state = STATE_DIR / f"{market}_trading" / "state.json"
            if market_state.exists():
                mtime = market_state.stat().st_mtime
                if (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() < 300:
                    result["active_bots"].append(market)
                    result["is_trading"] = True

    except Exception as e:
        result["error"] = str(e)

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/api/trading/control-panel", tags=["Status"])
async def get_trading_control_panel() -> Dict[str, Any]:
    """
    Get trading control panel status from unified trading engine.
    """
    result = {
        "master": {
            "emergency_stop": False,
            "all_paused": False,
            "any_active": False,
        },
        "markets": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Check emergency stop status
    controller = _get_safety_controller()
    status = controller.get_status()
    result["master"]["emergency_stop"] = status["emergency_stop_active"]

    # Check unified trading control file for pause status
    unified_control_path = STATE_DIR / "unified_trading" / "control.json"
    if unified_control_path.exists():
        try:
            with open(unified_control_path, "r") as f:
                unified_control = json.load(f)
                result["master"]["all_paused"] = unified_control.get("paused", False)
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.debug(f"Failed to read unified control: {e}")

    # Map unified symbols to market categories
    symbol_to_market = {
        # Crypto
        "BTC/USDT": "crypto",
        "ETH/USDT": "crypto",
        "SOL/USDT": "crypto",
        "XRP/USDT": "crypto",
        "ADA/USDT": "crypto",
        "AVAX/USDT": "crypto",
        "DOGE/USDT": "crypto",
        "DOT/USDT": "crypto",
        "LINK/USDT": "crypto",
        "MATIC/USDT": "crypto",
        # Commodities
        "XAU/USD": "commodity",
        "XAG/USD": "commodity",
        "USOIL/USD": "commodity",
        "NATGAS/USD": "commodity",
        "WTICO/USD": "commodity",
        # Stocks
        "AAPL": "stock",
        "MSFT": "stock",
        "TSLA": "stock",
        "NVDA": "stock",
        "GOOGL": "stock",
        "AMZN": "stock",
        "META": "stock",
        "AAPL/USD": "stock",
        "MSFT/USD": "stock",
        "GOOGL/USD": "stock",
        "NVDA/USD": "stock",
    }

    # Initialize market summaries
    market_config = {
        "crypto": {"name": "Crypto", "emoji": "🪙"},
        "commodity": {"name": "Commodity", "emoji": "🛢️"},
        "stock": {"name": "Stock", "emoji": "📈"},
    }

    # Read from unified trading state
    unified_state_path = STATE_DIR / "unified_trading" / "state.json"

    try:
        if unified_state_path.exists():
            mtime = unified_state_path.stat().st_mtime
            last_update = datetime.fromtimestamp(mtime)
            age_seconds = (datetime.now() - last_update).total_seconds()

            with open(unified_state_path, "r") as f:
                unified_state = json.load(f)

            # Extract overall state
            current_balance = unified_state.get("current_balance", 10000.0)
            initial_capital = unified_state.get("initial_capital", 10000.0)
            total_pnl = current_balance - initial_capital
            total_pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

            positions = unified_state.get("positions", {})
            is_active = age_seconds < 120  # Updated in last 2 minutes
            has_positions = len(positions) > 0

            # Count positions by market and calculate per-market P&L
            market_positions = {"crypto": 0, "commodity": 0, "stock": 0}
            market_pnl = {"crypto": 0.0, "commodity": 0.0, "stock": 0.0}

            for symbol, position in positions.items():
                market = symbol_to_market.get(symbol, "crypto")  # Default to crypto for unknown symbols
                market_positions[market] += 1
                # Add unrealized P&L from this position to the market's total
                unrealized = position.get("unrealized_pnl", 0.0)
                market_pnl[market] += unrealized

            # Calculate per-market balance based on equal capital allocation
            capital_per_market = current_balance / 3  # ✅ Use current balance, not initial capital
            market_balance = {}
            for market_id in market_config.keys():
                market_balance[market_id] = capital_per_market + market_pnl[market_id]

            # Determine health
            if is_active:
                health = "healthy"
                result["master"]["any_active"] = True
            elif age_seconds < 600:
                health = "stale"
            else:
                health = "offline"

            # Build market status for each category
            for market_id, config in market_config.items():
                market_balance_val = market_balance.get(market_id, capital_per_market)
                market_pnl_val = market_pnl.get(market_id, 0.0)
                market_pnl_pct_val = (
                    (market_pnl_val / capital_per_market * 100) if capital_per_market > 0 else 0.0
                )

                market_status = {
                    "name": config["name"],
                    "emoji": config["emoji"],
                    "running": is_active,
                    "paused": result["master"]["all_paused"],
                    "actively_trading": has_positions and is_active,
                    "last_update": last_update.isoformat(),
                    "last_trade": None,
                    "open_positions": market_positions[market_id],
                    "current_signal": "FLAT",
                    "balance": market_balance_val,
                    "pnl": market_pnl_val,
                    "pnl_pct": market_pnl_pct_val,
                    "health": health,
                }

                result["markets"][market_id] = market_status
        else:
            # No state file found - return offline status
            for market_id, config in market_config.items():
                result["markets"][market_id] = {
                    "name": config["name"],
                    "emoji": config["emoji"],
                    "running": False,
                    "paused": result["master"]["all_paused"],
                    "actively_trading": False,
                    "last_update": None,
                    "last_trade": None,
                    "open_positions": 0,
                    "current_signal": "FLAT",
                    "balance": 0.0,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                    "health": "offline",
                }

    except Exception as e:
        logger.error(f"Error reading unified state: {e}")
        # Return offline status on error
        for market_id, config in market_config.items():
            result["markets"][market_id] = {
                "name": config["name"],
                "emoji": config["emoji"],
                "running": False,
                "paused": result["master"]["all_paused"],
                "actively_trading": False,
                "last_update": None,
                "last_trade": None,
                "open_positions": 0,
                "current_signal": "FLAT",
                "balance": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "health": "offline",
            }

    return result


@app.get("/api/dashboard/unified-state", tags=["Dashboard"])
async def get_dashboard_unified_state() -> Dict[str, Any]:
    """
    SINGLE SOURCE OF TRUTH for dashboard data.

    Returns ALL market data in ONE atomic read of the state file.
    This prevents data inconsistencies from multiple parallel API calls.

    CRITICAL GUARANTEES:
    1. All values are PRE-ROUNDED at source (no rounding in UI)
    2. total.total_value === crypto.total_value + commodity.total_value + stock.total_value
    3. version is STRICTLY INCREASING (monotonic counter, never resets within process)
    4. All values from same atomic snapshot
    5. Integer cents provided for bulletproof math (_cents suffix)
    """
    global _snapshot_version_counter
    from decimal import Decimal, ROUND_HALF_UP

    # Increment version counter (strictly increasing, even on restart)
    _snapshot_version_counter += 1
    snapshot_version = _snapshot_version_counter
    snapshot_ts_ms = int(time.time() * 1000)  # Server time in milliseconds

    def round_money(value: float, places: int = 2) -> float:
        """Deterministic rounding - done ONCE at source."""
        return float(Decimal(str(value)).quantize(
            Decimal(10) ** -places,
            rounding=ROUND_HALF_UP
        ))

    def to_cents(value: float) -> int:
        """Convert dollars to integer cents for bulletproof math."""
        return int(round(value * 100))

    # Symbol to market mapping - MUST match the trading engine
    symbol_to_market = {
        # Crypto
        "BTC/USDT": "crypto", "ETH/USDT": "crypto", "SOL/USDT": "crypto",
        "XRP/USDT": "crypto", "ADA/USDT": "crypto", "AVAX/USDT": "crypto",
        "DOGE/USDT": "crypto", "DOT/USDT": "crypto", "LINK/USDT": "crypto",
        "MATIC/USDT": "crypto", "LTC/USDT": "crypto", "UNI/USDT": "crypto",
        # Commodities
        "XAU/USD": "commodity", "XAG/USD": "commodity", "USOIL/USD": "commodity",
        "NATGAS/USD": "commodity", "WTICO/USD": "commodity",
        # Stocks
        "AAPL": "stock", "MSFT": "stock", "TSLA": "stock", "NVDA": "stock",
        "GOOGL": "stock", "AMZN": "stock", "META": "stock",
        "AAPL/USD": "stock", "MSFT/USD": "stock", "GOOGL/USD": "stock", "NVDA/USD": "stock",
    }

    # Initialize result with defaults - use monotonic version
    result = {
        "version": snapshot_version,  # Strictly increasing monotonic counter
        "snapshot_ts_ms": snapshot_ts_ms,  # Server time in milliseconds
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "unified_trading",
        "data_version": 3,  # API format version (bumped for cents support)
        "total": {
            "initial_capital": 30000.0,
            "initial_capital_cents": 3000000,
            "current_balance": 30000.0,
            "current_balance_cents": 3000000,
            "total_value": 30000.0,
            "total_value_cents": 3000000,
            "pnl": 0.0,
            "pnl_cents": 0,
            "pnl_pct": 0.0,
            "positions_count": 0,
        },
        "markets": {
            "crypto": {
                "initial_capital": 10000.0,
                "initial_capital_cents": 1000000,
                "total_value": 10000.0,
                "total_value_cents": 1000000,
                "pnl": 0.0,
                "pnl_cents": 0,
                "pnl_pct": 0.0,
                "positions_count": 0,
                "positions": {},
                "status": "offline",
            },
            "commodity": {
                "initial_capital": 10000.0,
                "initial_capital_cents": 1000000,
                "total_value": 10000.0,
                "total_value_cents": 1000000,
                "pnl": 0.0,
                "pnl_cents": 0,
                "pnl_pct": 0.0,
                "positions_count": 0,
                "positions": {},
                "status": "offline",
            },
            "stock": {
                "initial_capital": 10000.0,
                "initial_capital_cents": 1000000,
                "total_value": 10000.0,
                "total_value_cents": 1000000,
                "pnl": 0.0,
                "pnl_cents": 0,
                "pnl_pct": 0.0,
                "positions_count": 0,
                "positions": {},
                "status": "offline",
            },
        },
        "bot_status": {
            "running": False,
            "paused": False,
            "last_update": None,
            "health": "offline",
        },
    }

    # SINGLE atomic read of the state file
    unified_state_path = STATE_DIR / "unified_trading" / "state.json"

    try:
        if not unified_state_path.exists():
            return result

        # Read file ONCE
        mtime = unified_state_path.stat().st_mtime
        last_update = datetime.fromtimestamp(mtime)
        age_seconds = (datetime.now() - last_update).total_seconds()

        with open(unified_state_path, "r") as f:
            state = json.load(f)

        # Extract values from single read
        initial_capital = float(state.get("initial_capital", 30000.0))
        current_balance = float(state.get("current_balance", initial_capital))
        positions = state.get("positions", {})

        # Calculate market-specific data from positions
        market_pnl = {"crypto": 0.0, "commodity": 0.0, "stock": 0.0}
        market_positions = {"crypto": {}, "commodity": {}, "stock": {}}
        market_position_count = {"crypto": 0, "commodity": 0, "stock": 0}

        for symbol, position in positions.items():
            market = symbol_to_market.get(symbol, "crypto")  # Default to crypto
            unrealized_pnl = float(position.get("unrealized_pnl", 0.0))
            market_pnl[market] += unrealized_pnl
            market_positions[market][symbol] = position
            market_position_count[market] += 1

        # Calculate per-market values (equal allocation: total / 3)
        capital_per_market = initial_capital / 3
        total_pnl = sum(market_pnl.values())

        # Determine bot status
        is_running = age_seconds < 120
        is_paused = state.get("status") == "paused"
        health = "healthy" if is_running else ("stale" if age_seconds < 600 else "offline")

        # Build result
        result["bot_status"] = {
            "running": is_running,
            "paused": is_paused,
            "last_update": last_update.isoformat(),
            "health": health,
        }

        # STEP 1: Calculate and ROUND each market's values FIRST
        market_values = {}
        for market_id in ["crypto", "commodity", "stock"]:
            pnl = round_money(market_pnl[market_id])
            market_value = round_money(capital_per_market + market_pnl[market_id])
            pnl_pct = round_money((pnl / capital_per_market * 100) if capital_per_market > 0 else 0.0, 4)
            initial_cap = round_money(capital_per_market)

            market_values[market_id] = {
                "initial_capital": initial_cap,
                "initial_capital_cents": to_cents(initial_cap),
                "total_value": market_value,
                "total_value_cents": to_cents(market_value),
                "pnl": pnl,
                "pnl_cents": to_cents(pnl),
                "pnl_pct": pnl_pct,
                "positions_count": market_position_count[market_id],
                "positions": market_positions[market_id],
                "status": health,
            }
            result["markets"][market_id] = market_values[market_id]

        # STEP 2: DERIVE total from children using CENTS - GUARANTEES total === sum(children)
        # Using cents eliminates floating-point drift
        derived_total_cents = (
            market_values["crypto"]["total_value_cents"] +
            market_values["commodity"]["total_value_cents"] +
            market_values["stock"]["total_value_cents"]
        )
        derived_pnl_cents = (
            market_values["crypto"]["pnl_cents"] +
            market_values["commodity"]["pnl_cents"] +
            market_values["stock"]["pnl_cents"]
        )

        # Convert back to dollars for display
        derived_total_value = derived_total_cents / 100.0
        derived_total_pnl = derived_pnl_cents / 100.0

        result["total"] = {
            "initial_capital": round_money(initial_capital),
            "initial_capital_cents": to_cents(initial_capital),
            "current_balance": round_money(current_balance),
            "current_balance_cents": to_cents(current_balance),
            "total_value": derived_total_value,  # DERIVED from children (cents -> dollars)
            "total_value_cents": derived_total_cents,  # DERIVED from children
            "pnl": derived_total_pnl,  # DERIVED from children (cents -> dollars)
            "pnl_cents": derived_pnl_cents,  # DERIVED from children
            "pnl_pct": round_money((derived_total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0, 4),
            "positions_count": len(positions),
        }

    except Exception as e:
        logger.error(f"Error in unified-state endpoint: {e}")
        result["error"] = str(e)

    return result


def _get_market_dir(market_id: str) -> Path:
    """Get the appropriate market directory, preferring production for crypto."""
    if market_id == "crypto":
        prod_dir = STATE_DIR / "production"
        if prod_dir.exists() and (prod_dir / "state.json").exists():
            return prod_dir
        return STATE_DIR / "live_paper_trading"
    elif market_id == "commodity":
        return STATE_DIR / "commodity_trading"
    elif market_id == "stock":
        return STATE_DIR / "stock_trading"
    raise ValueError(f"Unknown market: {market_id}")


@app.post("/api/trading/market/{market_id}/pause", tags=["Status"])
async def pause_market(
    market_id: str,
    reason: str = Query(default="Manual pause via dashboard", description="Reason for pause"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Pause a specific market's trading."""
    valid_markets = ["crypto", "commodity", "stock"]
    if market_id not in valid_markets:
        raise HTTPException(status_code=400, detail=f"Unknown market: {market_id}")

    market_dir = _get_market_dir(market_id)
    market_dir.mkdir(parents=True, exist_ok=True)

    update_bot_control(market_dir, paused=True, reason=reason)

    return {
        "success": True,
        "market": market_id,
        "action": "paused",
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/trading/market/{market_id}/resume", tags=["Status"])
async def resume_market(
    market_id: str,
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Resume a specific market's trading."""
    valid_markets = ["crypto", "commodity", "stock"]
    if market_id not in valid_markets:
        raise HTTPException(status_code=400, detail=f"Unknown market: {market_id}")

    market_dir = _get_market_dir(market_id)

    # Check if emergency stop is active
    controller = _get_safety_controller()
    if controller.get_status()["emergency_stop_active"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot resume: Emergency stop is active. Clear emergency stop first.",
        )

    update_bot_control(market_dir, paused=False, reason="Resumed via dashboard")

    return {
        "success": True,
        "market": market_id,
        "action": "resumed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/trading/pause-all", tags=["Status"])
async def pause_all_markets(
    reason: str = Query(default="Manual pause all via dashboard", description="Reason for pause"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Pause all markets by setting unified control file."""
    try:
        unified_dir = STATE_DIR / "unified_trading"
        unified_dir.mkdir(parents=True, exist_ok=True)
        update_bot_control(unified_dir, paused=True, reason=reason)

        return {
            "success": True,
            "action": "pause_all",
            "message": "All markets paused",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/resume-all", tags=["Status"])
async def resume_all_markets(
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Resume all markets by updating unified control file."""
    # Check if emergency stop is active
    controller = _get_safety_controller()
    if controller.get_status()["emergency_stop_active"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot resume: Emergency stop is active. Clear emergency stop first.",
        )

    try:
        unified_dir = STATE_DIR / "unified_trading"
        unified_dir.mkdir(parents=True, exist_ok=True)
        update_bot_control(unified_dir, paused=False, reason="Resumed via dashboard")

        return {
            "success": True,
            "action": "resume_all",
            "message": "All markets resumed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/start-engine", tags=["Status"])
async def start_trading_engine(
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Start the unified trading engine in background."""
    import subprocess
    import sys

    try:
        # Check if engine is already running
        check_result = subprocess.run(
            ["pgrep", "-f", "run_unified_trading.py"],
            capture_output=True,
            text=True,
        )

        if check_result.returncode == 0:
            return {
                "success": True,
                "status": "already_running",
                "message": "Trading engine is already running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Start the unified trading engine
        venv_path = Path(__file__).parent.parent / ".venv" / "bin" / "python"
        if not venv_path.exists():
            venv_path = sys.executable

        script_path = Path(__file__).parent.parent / "run_unified_trading.py"

        subprocess.Popen(
            [str(venv_path), str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return {
            "success": True,
            "status": "started",
            "message": "Trading engine started in background",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Failed to start trading engine: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Risk Settings Endpoints
# =============================================================================

RISK_SETTINGS_FILE = STATE_DIR / "risk_settings.json"


def _load_risk_settings() -> Dict[str, bool]:
    """Load risk settings from file."""
    defaults = {
        "shorting": False,
        "leverage": False,
        "aggressive": False,
    }
    try:
        if RISK_SETTINGS_FILE.exists():
            with open(RISK_SETTINGS_FILE) as f:
                data = json.load(f)
                return {**defaults, **data}
    except Exception:
        pass
    return defaults


def _save_risk_settings(settings: Dict[str, bool]) -> None:
    """Save risk settings to file."""
    RISK_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RISK_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


@app.get("/api/trading/risk-settings", tags=["Risk"])
async def get_risk_settings(
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get current risk settings."""
    settings = _load_risk_settings()

    # Calculate risk level
    risk_count = sum(1 for v in settings.values() if v)
    if risk_count == 0:
        risk_level = "low"
    elif risk_count == 1:
        risk_level = "medium"
    elif risk_count == 2:
        risk_level = "high"
    else:
        risk_level = "extreme"

    return {
        **settings,
        "risk_level": risk_level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/trading/risk-settings", tags=["Risk"])
async def update_risk_settings(
    request: Request,
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Update risk settings. Accepts partial updates."""
    body = await request.json()

    # Load current settings
    settings = _load_risk_settings()

    # Update only provided fields
    allowed_keys = {"shorting", "leverage", "aggressive"}
    for key in allowed_keys:
        if key in body:
            settings[key] = bool(body[key])

    # Save settings
    _save_risk_settings(settings)

    # Apply settings to running systems
    await _apply_risk_settings(settings)

    # Calculate risk level
    risk_count = sum(1 for v in settings.values() if v)
    if risk_count == 0:
        risk_level = "low"
    elif risk_count == 1:
        risk_level = "medium"
    elif risk_count == 2:
        risk_level = "high"
    else:
        risk_level = "extreme"

    return {
        **settings,
        "risk_level": risk_level,
        "updated": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _apply_risk_settings(settings: Dict[str, bool]) -> None:
    """Apply risk settings to running trading systems."""
    # Update control files for each market
    for market_id in ["crypto", "commodity", "stock"]:
        try:
            market_dir = _get_market_dir(market_id)
            control_file = market_dir / "control.json"

            control = {}
            if control_file.exists():
                with open(control_file) as f:
                    control = json.load(f)

            # Update risk settings in control
            control["allow_shorting"] = settings.get("shorting", False)
            control["allow_leverage"] = settings.get("leverage", False)
            control["aggressive_mode"] = settings.get("aggressive", False)
            control["risk_settings_updated"] = datetime.now(timezone.utc).isoformat()

            with open(control_file, "w") as f:
                json.dump(control, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not apply risk settings to {market_id}: {e}")

    # Also update unified trading control file
    try:
        unified_control_file = STATE_DIR / "unified_trading" / "control.json"
        unified_control_file.parent.mkdir(parents=True, exist_ok=True)

        control = {}
        if unified_control_file.exists():
            with open(unified_control_file) as f:
                control = json.load(f)

        control["allow_shorting"] = settings.get("shorting", False)
        control["allow_leverage"] = settings.get("leverage", False)
        control["aggressive_mode"] = settings.get("aggressive", False)
        control["max_leverage"] = 3.0 if settings.get("leverage", False) else 1.0
        control["risk_settings_updated"] = datetime.now(timezone.utc).isoformat()

        with open(unified_control_file, "w") as f:
            json.dump(control, f, indent=2)

        print(
            f"Risk settings applied: shorting={settings.get('shorting')}, leverage={settings.get('leverage')}"
        )
    except Exception as e:
        print(f"Warning: Could not apply risk settings to unified trading: {e}")


# =============================================================================
# Monte Carlo Simulation Endpoints
# =============================================================================


@app.get("/api/monte-carlo/simulation", tags=["ML/AI"])
async def run_monte_carlo_simulation(
    simulations: int = Query(default=1000, description="Number of simulations"),
    horizon_days: int = Query(default=252, description="Simulation horizon in days"),
    initial_capital: float = Query(default=10000, description="Initial capital"),
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for portfolio."""
    try:
        from bot.monte_carlo import MonteCarloSimulator, SimulationConfig

        config = SimulationConfig(
            num_simulations=min(simulations, 5000),  # Cap at 5000
            time_horizon_days=horizon_days,
            initial_capital=initial_capital,
        )

        simulator = MonteCarloSimulator(config)

        # Load equity curve for historical returns
        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"
        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)

            if isinstance(equity_data, list):
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]

                if len(equity) < 30:
                    return {
                        "status": "insufficient_data",
                        "message": "Need at least 30 data points",
                    }

                simulator.load_equity_curve(equity)
                result = simulator.run_simulation()
                return simulator.to_api_response(result)

        return {"status": "no_data", "message": "No equity data found for simulation"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monte-carlo/stress-test", tags=["ML/AI"])
async def run_stress_test() -> Dict[str, Any]:
    """Run stress test scenarios on portfolio."""
    try:
        from bot.monte_carlo import MonteCarloSimulator

        simulator = MonteCarloSimulator()

        equity_path = STATE_DIR / "live_paper_trading" / "equity.json"
        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)

            if isinstance(equity_data, list):
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]

                if len(equity) >= 30:
                    simulator.load_equity_curve(equity)
                    return {"stress_test_results": simulator.run_stress_test()}

        return {"status": "no_data", "message": "Insufficient data for stress test"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Regime Transition Endpoints
# =============================================================================


@app.get("/api/regime/transitions", tags=["ML/AI"])
async def get_regime_transitions() -> Dict[str, Any]:
    """Get regime transition matrix and analysis."""
    try:
        from bot.regime_transitions import RegimeTransitionAnalyzer

        analyzer = RegimeTransitionAnalyzer()

        # Try to load regime history
        regime_path = STATE_DIR / "regime_history.json"
        if regime_path.exists():
            analyzer.load_from_json(str(regime_path))
        else:
            # Try to extract from state files
            state_path = STATE_DIR / "live_paper_trading" / "state.json"
            if state_path.exists():
                with open(state_path) as f:
                    data = json.load(f)

                regimes = data.get("regime_history", [])
                if isinstance(regimes, list) and regimes:
                    analyzer.load_regime_history(regimes)

        return analyzer.to_api_response()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regime/current", tags=["ML/AI"])
async def get_current_regime() -> Dict[str, Any]:
    """Get current market regime and next regime prediction."""
    try:
        from bot.regime_transitions import RegimeTransitionAnalyzer

        analyzer = RegimeTransitionAnalyzer()
        regime_path = STATE_DIR / "regime_history.json"

        if regime_path.exists():
            analyzer.load_from_json(str(regime_path))

        current = analyzer.get_current_regime()
        predictions = analyzer.predict_next_regime()

        return {
            "current_regime": current,
            "next_regime_probabilities": predictions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Factor Analysis Endpoints
# =============================================================================


@app.get("/api/factor/analysis", tags=["ML/AI"])
async def get_factor_analysis() -> Dict[str, Any]:
    """Get factor attribution analysis."""
    try:
        from bot.factor_analysis import FactorAnalyzer
        import numpy as np

        analyzer = FactorAnalyzer()

        # Load price data to calculate factors
        data_path = STATE_DIR / "market_data" / "BTC_USDT_1h.json"
        if data_path.exists():
            with open(data_path) as f:
                market_data = json.load(f)

            if isinstance(market_data, list) and len(market_data) > 50:
                prices = np.array(
                    [d.get("close", d) if isinstance(d, dict) else d for d in market_data]
                )
                volumes = np.array(
                    [d.get("volume", 0) if isinstance(d, dict) else 0 for d in market_data]
                )

                # Calculate factors from prices
                factors = analyzer.calculate_factors_from_prices(prices, volumes)

                if factors:
                    # Calculate returns
                    returns = list(np.diff(prices) / prices[:-1])
                    returns = returns[20:]  # Align with factors

                    analyzer.load_data(returns, factors)
                    result = analyzer.run_analysis()
                    return analyzer.to_api_response(result)

        return {"status": "no_data", "message": "Insufficient market data for factor analysis"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/factor/correlations", tags=["ML/AI"])
async def get_factor_correlations() -> Dict[str, Any]:
    """Get correlations between trading factors."""
    try:
        from bot.factor_analysis import FactorAnalyzer
        import numpy as np

        analyzer = FactorAnalyzer()

        data_path = STATE_DIR / "market_data" / "BTC_USDT_1h.json"
        if data_path.exists():
            with open(data_path) as f:
                market_data = json.load(f)

            if isinstance(market_data, list) and len(market_data) > 50:
                prices = np.array(
                    [d.get("close", d) if isinstance(d, dict) else d for d in market_data]
                )
                volumes = np.array(
                    [d.get("volume", 0) if isinstance(d, dict) else 0 for d in market_data]
                )

                factors = analyzer.calculate_factors_from_prices(prices, volumes)
                analyzer._factor_data = factors

                return {"correlations": analyzer.get_factor_correlations()}

        return {"status": "no_data", "correlations": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Report Generation Endpoints
# =============================================================================


@app.post("/api/reports/generate", tags=["Dashboard"])
async def generate_report(
    period: str = Query(default="weekly", description="Report period: daily, weekly, monthly"),
    include_charts: bool = Query(default=True),
    include_trades: bool = Query(default=True),
) -> Dict[str, Any]:
    """Generate a trading performance report."""
    try:
        from bot.report_generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            period=period,
            include_charts=include_charts,
            include_trades=include_trades,
        )

        generator = ReportGenerator(config)

        # Load data from various sources
        trades = []
        equity = []
        risk_metrics = {}

        # Load trades - check production first, then paper trading
        state_paths = [
            STATE_DIR / "production" / "state.json",
            STATE_DIR / "live_paper_trading" / "state.json",
        ]
        for state_path in state_paths:
            if state_path.exists():
                with open(state_path) as f:
                    data = json.load(f)
                trades = data.get("trades", [])
                break

        # Load equity - check production first, then paper trading
        equity_paths = [
            STATE_DIR / "production" / "equity.json",
            STATE_DIR / "live_paper_trading" / "equity.json",
        ]
        for equity_path in equity_paths:
            if equity_path.exists():
                with open(equity_path) as f:
                    equity_data = json.load(f)
                equity = [e.get("value", e) if isinstance(e, dict) else e for e in equity_data]
                equity = [e for e in equity if isinstance(e, (int, float))]
                break

        # Calculate risk metrics
        if equity and len(equity) > 10:
            from bot.risk_metrics import RiskMetricsCalculator

            calc = RiskMetricsCalculator()
            calc.load_equity_curve(equity)
            metrics = calc.calculate()
            risk_metrics = {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "profit_factor": metrics.profit_factor,
            }

        generator.load_data(
            trades=trades,
            equity_curve=equity,
            risk_metrics=risk_metrics,
        )

        filepath = generator.generate_report()

        return {
            "status": "success",
            "filepath": filepath,
            "metadata": generator.get_report_metadata(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/list", tags=["Dashboard"])
async def list_reports() -> Dict[str, Any]:
    """List available generated reports."""
    try:
        reports_dir = STATE_DIR / "reports"
        if not reports_dir.exists():
            return {"reports": []}

        reports = []
        for report_file in sorted(reports_dir.glob("*.html"), reverse=True):
            reports.append(
                {
                    "filename": report_file.name,
                    "path": str(report_file),
                    "size_kb": round(report_file.stat().st_size / 1024, 1),
                    "created": datetime.fromtimestamp(report_file.stat().st_ctime).isoformat(),
                }
            )

        return {"reports": reports[:20]}  # Return last 20 reports
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Daily Reporter Endpoints
# =============================================================================


@app.get("/api/daily-reports/status", tags=["Reports"])
async def get_daily_reporter_status() -> Dict[str, Any]:
    """Get status of the daily reporter scheduler."""
    try:
        from bot.regime.daily_reporter import get_daily_reporter

        reporter = get_daily_reporter()
        return {
            "status": "ok",
            **reporter.get_status(),
        }
    except ImportError:
        return {
            "status": "not_configured",
            "running": False,
            "schedules": [],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/daily-reports/send", tags=["Reports"])
async def send_daily_report(
    report_type: str = Query(default="daily", description="Report type: daily, weekly, monthly"),
) -> Dict[str, Any]:
    """Send a performance report immediately via Telegram."""
    try:
        from bot.regime.performance_tracker import get_performance_tracker

        tracker = get_performance_tracker()

        # Generate the report message
        message = tracker.generate_telegram_summary(report_type)

        if not message:
            return {
                "success": False,
                "error": "No data available for report",
            }

        # Try to send via Telegram
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if telegram_token and telegram_chat_id:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                    json={
                        "chat_id": telegram_chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                    },
                )
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Report sent to Telegram",
                        "preview": message[:500] + "..." if len(message) > 500 else message,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Telegram API error: {response.text}",
                        "preview": message[:500] + "..." if len(message) > 500 else message,
                    }
        else:
            return {
                "success": False,
                "error": "Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)",
                "preview": message[:500] + "..." if len(message) > 500 else message,
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/performance/summary", tags=["Reports"])
async def get_performance_summary(
    period: str = Query(default="daily", description="Period: daily, weekly, monthly, all"),
) -> Dict[str, Any]:
    """Get trading performance summary for the specified period."""
    try:
        from bot.regime.performance_tracker import get_performance_tracker

        tracker = get_performance_tracker()

        if period == "daily":
            summary = tracker.get_daily_summary()
        elif period == "all":
            paper_metrics = tracker.get_paper_metrics()
            summary = {
                "period": "all",
                **paper_metrics.to_dict(),
                "comparison": tracker.compare_performance(),
                "regime_performance": {
                    k: v.to_dict() for k, v in tracker.get_regime_performance("paper").items()
                },
            }
        else:
            days = 7 if period == "weekly" else 30
            paper_metrics = tracker.get_paper_metrics(days=days)
            summary = {
                "period": period,
                "days": days,
                **paper_metrics.to_dict(),
            }

        return {
            "status": "ok",
            "summary": summary,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Performance Attribution Endpoints
# =============================================================================


@app.get("/api/performance/attribution", tags=["Analytics"])
async def get_performance_attribution(
    days: int = Query(default=30, description="Number of days to analyze"),
) -> Dict[str, Any]:
    """Get performance attribution report with multi-factor analysis."""
    try:
        from bot.performance_attribution import PerformanceAttributor

        attributor = PerformanceAttributor()

        # Load trades from production/paper trading state
        state_paths = [
            STATE_DIR / "production" / "state.json",
            STATE_DIR / "live_paper_trading" / "state.json",
        ]

        for state_path in state_paths:
            if state_path.exists():
                with open(state_path) as f:
                    data = json.load(f)
                    trades = data.get("trades", [])

                    for trade in trades:
                        if isinstance(trade, dict) and trade.get("entry_price"):
                            attributor.log_trade(
                                trade_id=trade.get("trade_id", f"T{len(attributor.trades)}"),
                                symbol=trade.get("symbol", "unknown"),
                                action=trade.get("side", "BUY").upper(),
                                quantity=trade.get("quantity", 0),
                                entry_price=trade.get("entry_price", 0),
                                exit_price=trade.get("exit_price"),
                                pnl=trade.get("pnl", 0),
                                strategy=trade.get("strategy", "regime"),
                                model=trade.get("model", "ml"),
                                regime=trade.get("regime", "unknown"),
                                confidence=trade.get("confidence", 0.5),
                                entry_time=datetime.fromisoformat(trade["entry_time"])
                                if trade.get("entry_time")
                                else datetime.now(),
                                exit_time=datetime.fromisoformat(trade["exit_time"])
                                if trade.get("exit_time")
                                else None,
                            )
                break

        report = attributor.generate_report(days=days)
        return {
            "status": "ok",
            "report": report.to_dict(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/performance/daily", tags=["Analytics"])
async def get_daily_performance(
    date: Optional[str] = Query(default=None, description="Date in YYYY-MM-DD format"),
) -> Dict[str, Any]:
    """Get detailed daily performance breakdown."""
    try:
        from datetime import date as dateclass
        from bot.performance_attribution import PerformanceAttributor

        target_date = dateclass.fromisoformat(date) if date else dateclass.today()
        attributor = PerformanceAttributor()

        # Load trades from state
        state_paths = [
            STATE_DIR / "production" / "state.json",
            STATE_DIR / "live_paper_trading" / "state.json",
        ]

        for state_path in state_paths:
            if state_path.exists():
                with open(state_path) as f:
                    data = json.load(f)
                    trades = data.get("trades", [])

                    for trade in trades:
                        if isinstance(trade, dict) and trade.get("entry_price"):
                            entry_time = (
                                datetime.fromisoformat(trade["entry_time"])
                                if trade.get("entry_time")
                                else datetime.now()
                            )
                            attributor.log_trade(
                                trade_id=trade.get("trade_id", f"T{len(attributor.trades)}"),
                                symbol=trade.get("symbol", "unknown"),
                                action=trade.get("side", "BUY").upper(),
                                quantity=trade.get("quantity", 0),
                                entry_price=trade.get("entry_price", 0),
                                exit_price=trade.get("exit_price"),
                                pnl=trade.get("pnl", 0),
                                strategy=trade.get("strategy", "regime"),
                                regime=trade.get("regime", "unknown"),
                                entry_time=entry_time,
                            )
                break

        summary = attributor.get_daily_summary(target_date)
        return {
            "status": "ok",
            "summary": summary,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Data Fetcher API - Rate-Limited Training Data Collection
# =============================================================================

# Global data fetcher instance
_data_fetcher = None


def get_data_fetcher():
    """Get or create data fetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        from bot.data_fetcher import SmartDataFetcher

        _data_fetcher = SmartDataFetcher(provider="binance", data_dir="data/training")
    return _data_fetcher


@app.get("/api/data-fetcher/capacity", tags=["Data Fetcher"])
async def get_fetch_capacity(provider: str = "binance") -> Dict[str, Any]:
    """
    Get available API capacity for data fetching.

    Shows how many requests and records you can fetch within rate limits.
    """
    try:
        from bot.data_fetcher import SmartDataFetcher

        fetcher = SmartDataFetcher(provider=provider)
        return fetcher.get_available_capacity()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-fetcher/estimate", tags=["Data Fetcher"])
async def estimate_fetch_time(
    symbols: str = "BTC/USDT,ETH/USDT",
    timeframe: str = "1h",
    days: int = 365,
    provider: str = "binance",
) -> Dict[str, Any]:
    """
    Estimate time required to fetch historical data.

    Args:
        symbols: Comma-separated list of trading pairs
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        days: Number of days of history to fetch
        provider: Data provider (binance, coinbase, kraken, yahoo, polygon)
    """
    try:
        from bot.data_fetcher import SmartDataFetcher

        symbol_list = [s.strip() for s in symbols.split(",")]
        fetcher = SmartDataFetcher(provider=provider)
        return fetcher.estimate_fetch_time(symbol_list, timeframe, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-fetcher/stats", tags=["Data Fetcher"])
async def get_fetch_stats() -> Dict[str, Any]:
    """Get current data fetcher statistics and progress."""
    try:
        fetcher = get_data_fetcher()
        return fetcher.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-fetcher/start", tags=["Data Fetcher"])
async def start_data_fetch(
    symbols: str = "BTC/USDT,ETH/USDT,SOL/USDT",
    timeframe: str = "1h",
    days: int = 365,
    provider: str = "binance",
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """
    Start fetching historical data for ML training.

    This runs in the background and respects API rate limits.
    Use /api/data-fetcher/stats to check progress.

    Args:
        symbols: Comma-separated list of trading pairs
        timeframe: Candle timeframe
        days: Number of days of history
        provider: Data provider
    """
    try:
        from bot.data_fetcher import SmartDataFetcher
        from datetime import timedelta

        symbol_list = [s.strip() for s in symbols.split(",")]

        # Create new fetcher for this job
        fetcher = SmartDataFetcher(provider=provider, data_dir="data/training")

        # Estimate first
        estimate = fetcher.estimate_fetch_time(symbol_list, timeframe, days)

        # Define background task
        def fetch_task():
            start_date = datetime.now() - timedelta(days=days)
            fetcher.fetch_multiple_symbols(symbol_list, timeframe, start_date)

        if background_tasks:
            background_tasks.add_task(fetch_task)

        return {
            "status": "started",
            "message": f"Fetching {len(symbol_list)} symbols",
            "symbols": symbol_list,
            "timeframe": timeframe,
            "days": days,
            "estimate": estimate,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-fetcher/stop", tags=["Data Fetcher"])
async def stop_data_fetch() -> Dict[str, Any]:
    """Stop any ongoing data fetch operation."""
    try:
        fetcher = get_data_fetcher()
        fetcher.stop()
        return {"status": "stopped", "message": "Fetch operation stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-fetcher/datasets", tags=["Data Fetcher"])
async def list_training_datasets() -> Dict[str, Any]:
    """List available training datasets."""
    try:
        from pathlib import Path

        data_dir = Path("data/training")
        if not data_dir.exists():
            return {"datasets": [], "total_size_mb": 0}

        datasets = []
        total_size = 0

        for file in sorted(data_dir.glob("*.parquet")):
            if file.name == "combined_training.parquet":
                continue

            size = file.stat().st_size
            total_size += size

            # Parse symbol and timeframe from filename
            parts = file.stem.rsplit("_", 1)
            symbol = parts[0].replace("_", "/") if len(parts) >= 1 else file.stem
            timeframe = parts[1] if len(parts) == 2 else "unknown"

            # Get record count
            try:
                import pandas as pd

                df = pd.read_parquet(file)
                record_count = len(df)
                if len(df) > 0:
                    date_range = {
                        "start": pd.to_datetime(df["timestamp"].min(), unit="ms").isoformat(),
                        "end": pd.to_datetime(df["timestamp"].max(), unit="ms").isoformat(),
                    }
                else:
                    date_range = None
            except Exception:
                record_count = 0
                date_range = None

            datasets.append(
                {
                    "filename": file.name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "records": record_count,
                    "size_mb": round(size / 1024 / 1024, 2),
                    "date_range": date_range,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                }
            )

        return {
            "datasets": datasets,
            "count": len(datasets),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-fetcher/combine", tags=["Data Fetcher"])
async def combine_training_datasets() -> Dict[str, Any]:
    """Combine all fetched data into a single training dataset."""
    try:
        from bot.data_fetcher import create_training_dataset

        df = create_training_dataset()

        if len(df) == 0:
            return {"status": "no_data", "message": "No datasets to combine"}

        return {
            "status": "success",
            "records": len(df),
            "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
            "output_file": "data/training/combined_training.parquet",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-fetcher/rate-limits", tags=["Data Fetcher"])
async def get_provider_rate_limits() -> Dict[str, Any]:
    """Get rate limit configurations for all supported providers."""
    from bot.data_fetcher import RATE_LIMITS

    return {
        provider: {
            "requests_per_minute": config.requests_per_minute,
            "requests_per_hour": config.requests_per_hour,
            "requests_per_day": config.requests_per_day,
            "max_records_per_request": config.max_records_per_request,
            "safe_requests_per_minute": config.safe_requests_per_minute,
            "min_interval_seconds": config.min_interval_seconds,
        }
        for provider, config in RATE_LIMITS.items()
    }


# =============================================================================
# Strategy Tracker API - Monitor Strategy Performance
# =============================================================================

STRATEGY_DASHBOARD_TEMPLATE = Path(__file__).parent / "strategy_dashboard.html"


@app.get("/api/strategies/dashboard", tags=["ML/AI"])
async def get_strategy_dashboard() -> Dict[str, Any]:
    """
    Get strategy performance dashboard data.

    Returns a comprehensive view of all strategy performance metrics.
    """
    try:
        from bot.regime import get_tracker

        tracker = get_tracker()
        return tracker.get_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/list", tags=["ML/AI"])
async def list_strategies() -> Dict[str, Any]:
    """List all registered trading strategies with their performance."""
    try:
        from bot.regime import get_tracker

        tracker = get_tracker()
        strategies = tracker.get_all_strategies()
        return {
            "strategies": [s.to_dict() for s in strategies],
            "count": len(strategies),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/{strategy_id}", tags=["ML/AI"])
async def get_strategy_performance(strategy_id: str) -> Dict[str, Any]:
    """Get performance metrics for a specific strategy."""
    try:
        from bot.regime import get_tracker

        tracker = get_tracker()
        perf = tracker.get_performance(strategy_id)
        if perf is None:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        return perf.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/comparison/text", tags=["ML/AI"])
async def get_strategy_comparison_text() -> Dict[str, str]:
    """Get a text-formatted comparison of all strategies."""
    try:
        from bot.regime import get_tracker

        tracker = get_tracker()
        return {"comparison": tracker.get_strategy_comparison()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/strategies", response_class=HTMLResponse, tags=["Dashboard"])
async def strategy_dashboard() -> HTMLResponse:
    """Serve the strategy performance dashboard."""
    if not STRATEGY_DASHBOARD_TEMPLATE.exists():
        return HTMLResponse(
            content="<html><body><h1>Strategy dashboard template missing.</h1></body></html>",
            status_code=200,
        )
    return HTMLResponse(content=STRATEGY_DASHBOARD_TEMPLATE.read_text())


# =============================================================================
# Trading Mode Toggle (Paper <-> Live)
# =============================================================================

TRADING_MODE_FILE = STATE_DIR / "trading_mode.json"


def _get_trading_mode() -> Dict[str, Any]:
    """Get current trading mode configuration."""
    default = {
        "mode": "paper",
        "live_enabled": False,
        "exchange": "binance",
        "last_switch": None,
        "paper_capital": 30000.0,
        "live_capital": 0.0,
        "confirmation_required": True,
    }
    if TRADING_MODE_FILE.exists():
        try:
            with open(TRADING_MODE_FILE, "r") as f:
                stored = json.load(f)
                default.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return default


def _save_trading_mode(config: Dict[str, Any]) -> None:
    """Save trading mode configuration."""
    TRADING_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADING_MODE_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)


@app.get("/api/trading/mode", tags=["Trading"])
async def get_trading_mode() -> Dict[str, Any]:
    """
    Get the current trading mode (paper, testnet, or live).

    Returns:
        mode: 'paper', 'testnet', or 'live'
        live_enabled: Whether live/testnet trading is available
        exchange: Connected exchange for trading
        paper_capital: Current paper trading capital
        testnet_capital: Current testnet capital (if connected)
        live_capital: Current live trading capital (if connected)
    """
    config = _get_trading_mode()

    # Check if Binance API keys are configured
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    has_keys = bool(api_key and api_secret and len(api_key) > 10)

    # Get current portfolio values - check production state first
    paper_value = 0.0
    production_value = 0.0

    # Check production state first
    prod_state_path = STATE_DIR / "production" / "state.json"
    if prod_state_path.exists():
        try:
            with open(prod_state_path, "r") as f:
                state = json.load(f)
                production_value = state.get("total_value", state.get("balance", 0))
        except (json.JSONDecodeError, OSError):
            pass

    # Check paper trading states
    for market_dir in ["live_paper_trading", "commodity_trading", "stock_trading"]:
        state_path = STATE_DIR / market_dir / "state.json"
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                    paper_value += state.get("total_value", state.get("balance", 0))
            except (json.JSONDecodeError, OSError):
                pass

    # Get testnet/live balance if keys are configured
    testnet_value = 0.0
    live_value = 0.0
    if has_keys:
        try:
            import ccxt

            # Check testnet
            if config["mode"] == "testnet":
                exchange = ccxt.binance(
                    {
                        "apiKey": api_key,
                        "secret": api_secret,
                        "sandbox": True,
                        "options": {"defaultType": "spot"},
                    }
                )
                balance = exchange.fetch_balance()
                testnet_value = float(balance.get("total", {}).get("USDT", 0))
            elif config["mode"] == "live":
                exchange = ccxt.binance(
                    {
                        "apiKey": api_key,
                        "secret": api_secret,
                        "sandbox": False,
                        "options": {"defaultType": "spot"},
                    }
                )
                balance = exchange.fetch_balance()
                live_value = float(balance.get("total", {}).get("USDT", 0))
        except Exception:
            pass

    return {
        "mode": config["mode"],
        "live_enabled": has_keys,
        "testnet_enabled": has_keys,
        "exchange": config.get("exchange", "binance"),
        "paper_capital": paper_value,
        "production_capital": production_value,
        "testnet_capital": testnet_value,
        "live_capital": live_value,
        "last_switch": config.get("last_switch"),
        "confirmation_required": config.get("confirmation_required", True),
        "api_keys_configured": has_keys,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/trading/mode", tags=["Trading"])
async def set_trading_mode(
    mode: str = Query(..., description="Trading mode: 'paper', 'testnet', or 'live'"),
    confirm: bool = Query(False, description="Confirmation for switching to live"),
) -> Dict[str, Any]:
    """
    Switch between paper, testnet, and live trading modes.

    IMPORTANT: Switching to live mode requires confirmation and valid API keys.

    Args:
        mode: 'paper', 'testnet', or 'live'
        confirm: Must be True to switch to live trading

    Returns:
        Success message and updated configuration
    """
    if mode not in ["paper", "testnet", "live"]:
        raise HTTPException(status_code=400, detail="Mode must be 'paper', 'testnet', or 'live'")

    config = _get_trading_mode()

    # Testnet and live require API keys
    if mode in ["testnet", "live"]:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if not api_key or not api_secret or len(api_key) < 10:
            raise HTTPException(
                status_code=400,
                detail="Binance API keys not configured. Add BINANCE_API_KEY and BINANCE_API_SECRET to .env file.",
            )

        # Live mode requires additional confirmation
        if mode == "live" and not confirm:
            raise HTTPException(
                status_code=400,
                detail="Switching to live trading requires confirmation. Set confirm=true to proceed.",
            )

        # Verify API keys work
        try:
            import ccxt

            sandbox = mode == "testnet"
            exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "sandbox": sandbox,
                    "options": {"defaultType": "spot"},
                }
            )
            balance = exchange.fetch_balance()
            capital = float(balance.get("total", {}).get("USDT", 0))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to connect to Binance {'testnet' if mode == 'testnet' else 'mainnet'}: {str(e)}",
            )

        if mode == "live":
            config["live_capital"] = capital
        else:
            config["testnet_capital"] = capital

    # Update production config if it exists
    prod_config_path = STATE_DIR / "production_config.json"
    if prod_config_path.exists():
        try:
            with open(prod_config_path, "r") as f:
                prod_config = json.load(f)
            prod_config["mode"] = mode
            with open(prod_config_path, "w") as f:
                json.dump(prod_config, f, indent=2)
        except Exception:
            pass

    config["mode"] = mode
    config["last_switch"] = datetime.now(timezone.utc).isoformat()

    _save_trading_mode(config)

    return {
        "success": True,
        "message": f"Trading mode switched to {mode}",
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/ai/status", tags=["AI"])
async def get_ai_status() -> Dict[str, Any]:
    """
    Get AI trading advisor status and availability.
    """
    try:
        from bot.ai_trading_advisor import get_advisor

        advisor = get_advisor()
        available = await advisor.check_availability()

        return {
            "enabled": advisor.enabled,
            "available": available,
            "ollama_host": advisor.ollama_host,
            "model": advisor.model,
            "last_advice": advisor.get_all_advice(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "enabled": False,
            "available": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.post("/api/ai/advice", tags=["AI"])
async def get_ai_trading_advice(
    symbol: str = Query(..., description="Trading symbol"),
    current_price: float = Query(..., description="Current price"),
    price_change_24h: float = Query(0.0, description="24h price change %"),
    current_signal: str = Query("FLAT", description="Current strategy signal"),
    regime: str = Query("unknown", description="Market regime"),
    confidence: float = Query(0.5, description="Signal confidence 0-1"),
    portfolio_value: float = Query(10000.0, description="Total portfolio value"),
    position_value: float = Query(0.0, description="Current position value"),
    pnl_pct: float = Query(0.0, description="Current P&L %"),
    asset_type: str = Query("crypto", description="Asset type"),
) -> Dict[str, Any]:
    """
    Get AI trading advice for a specific asset.

    Returns AI-generated recommendation with confidence and reasoning.
    """
    try:
        from bot.ai_trading_advisor import get_ai_advice

        advice = await get_ai_advice(
            symbol=symbol,
            current_price=current_price,
            price_change_24h=price_change_24h,
            current_signal=current_signal,
            regime=regime,
            confidence=confidence,
            portfolio_value=portfolio_value,
            position_value=position_value,
            pnl_pct=pnl_pct,
            asset_type=asset_type,
        )
        return advice.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI advice failed: {str(e)}")


# =============================================================================
# Adaptive Risk Controller Endpoints
# =============================================================================


@app.get("/api/trading/current-strategy", tags=["Risk"])
async def get_current_strategy(
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get the current adaptive trading strategy.

    Returns:
    - Current strategy name and description
    - Risk settings (shorting, leverage, aggressive)
    - Market regime and expected direction
    - Reasoning for current settings
    - Recent risk decisions
    """
    try:
        from bot.adaptive_risk_controller import get_adaptive_risk_controller

        controller = get_adaptive_risk_controller()
        return controller.get_current_strategy()
    except Exception as e:
        # Return default strategy if controller not available
        return {
            "strategy": {
                "name": "Initializing...",
                "description": "Adaptive risk controller is starting up",
                "shorting_enabled": False,
                "leverage_enabled": False,
                "aggressive_enabled": False,
                "market_regime": "unknown",
                "expected_direction": "neutral",
                "confidence": 0.0,
                "suggested_action": "hold",
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "reasoning": ["System initializing..."],
            },
            "risk_profile": "conservative",
            "settings": {"shorting": False, "leverage": False, "aggressive": False},
            "recent_decisions": [],
        }


@app.post("/api/trading/evaluate-risk", tags=["Risk"])
async def evaluate_risk_settings(
    request: Request,
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Trigger a risk evaluation with current market conditions.

    Body should contain:
    - market_regime: str (bull, bear, sideways, crash, etc.)
    - regime_confidence: float (0-1)
    - rsi: float (0-100)
    - volatility: str (low, normal, high, extreme)
    - trend: str (up, down, neutral)
    - recent_performance: Optional[Dict] with win_rate, total_pnl, drawdown
    """
    try:
        from bot.adaptive_risk_controller import get_adaptive_risk_controller

        body = await request.json()

        controller = get_adaptive_risk_controller()
        strategy = await controller.evaluate_and_adjust(
            market_regime=body.get("market_regime", "unknown"),
            regime_confidence=body.get("regime_confidence", 0.5),
            rsi=body.get("rsi", 50.0),
            volatility=body.get("volatility", "normal"),
            trend=body.get("trend", "neutral"),
            recent_performance=body.get("recent_performance"),
        )

        return {
            "evaluated": True,
            "strategy": strategy.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk evaluation failed: {str(e)}")


@app.get("/api/trading/risk-decisions", tags=["Risk"])
async def get_risk_decisions(
    limit: int = Query(default=20, description="Number of decisions to return"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get recent risk adjustment decisions.

    Returns history of when and why risk settings were changed.
    """
    try:
        from bot.adaptive_risk_controller import get_adaptive_risk_controller

        controller = get_adaptive_risk_controller()
        decisions = controller.get_decision_history(limit=limit)

        return {
            "decisions": decisions,
            "count": len(decisions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"decisions": [], "count": 0, "error": str(e)}


# =============================================================================
# Optimal Action Tracker Endpoints
# =============================================================================


@app.get("/api/trading/optimal-action", tags=["ML/AI"])
async def get_optimal_action(
    regime: str = Query(default="unknown", description="Market regime"),
    rsi: float = Query(default=50.0, description="RSI value"),
    trend: str = Query(default="neutral", description="Trend direction"),
    volatility: str = Query(default="normal", description="Volatility regime"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get optimal action for given market state based on historical learning.

    Returns recommended action (buy, sell, hold) with expected value.
    """
    try:
        from bot.optimal_action_tracker import get_tracker, MarketState

        tracker = get_tracker()
        state = MarketState(
            regime=regime,
            rsi=rsi,
            trend_direction=trend,
            volatility_regime=volatility,
        )

        action, expected_value = tracker.get_optimal_action(state)

        return {
            "state_key": state.to_state_key(),
            "optimal_action": action.value,
            "expected_value": round(expected_value, 4),
            "based_on": "historical_data"
            if state.to_state_key() in tracker._q_table
            else "heuristic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "optimal_action": "hold",
            "expected_value": 0.0,
            "based_on": "error_fallback",
            "error": str(e),
        }


@app.get("/api/trading/action-stats", tags=["ML/AI"])
async def get_action_stats(
    state_key: Optional[str] = Query(default=None, description="Filter by state key"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get statistics for all actions or filtered by state.

    Returns win rate, average return, and count for each action type.
    """
    try:
        from bot.optimal_action_tracker import get_tracker

        tracker = get_tracker()
        stats = tracker.get_action_stats(state_key=state_key)

        return {
            "stats": stats,
            "state_key_filter": state_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"stats": {}, "error": str(e)}


@app.get("/api/trading/best-states", tags=["ML/AI"])
async def get_best_states(
    action: str = Query(default="buy", description="Action to analyze"),
    min_count: int = Query(default=5, description="Minimum sample count"),
    _auth: Optional[str] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get market states where a specific action performs best.

    Useful for understanding when to buy/sell based on historical data.
    """
    try:
        from bot.optimal_action_tracker import get_tracker

        tracker = get_tracker()
        best_states = tracker.get_best_states(action=action, min_count=min_count)

        return {
            "action": action,
            "best_states": best_states,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"action": action, "best_states": [], "error": str(e)}


@app.get("/api/trading/live/balance", tags=["Trading"])
async def get_live_balance() -> Dict[str, Any]:
    """
    Get live Binance account balance.

    Returns USDT balance and positions from connected Binance account.
    """
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        raise HTTPException(status_code=400, detail="Binance API keys not configured")

    try:
        import ccxt

        exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "sandbox": False,
            }
        )

        balance = exchange.fetch_balance()

        # Get non-zero balances
        positions = {}
        for asset, amount in balance.get("total", {}).items():
            if float(amount) > 0:
                positions[asset] = {
                    "total": float(amount),
                    "free": float(balance.get("free", {}).get(asset, 0)),
                    "used": float(balance.get("used", {}).get(asset, 0)),
                }

        return {
            "connected": True,
            "exchange": "binance",
            "usdt_balance": float(balance.get("total", {}).get("USDT", 0)),
            "positions": positions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch balance: {str(e)}")


# =============================================================================
# AI TRADING BRAIN ENDPOINTS
# =============================================================================


@app.get("/api/ai-brain/status", tags=["AI Brain"])
async def get_ai_brain_status() -> Dict[str, Any]:
    """
    Get AI Trading Brain status including daily target progress and learning stats.

    Returns comprehensive status of the AI brain including:
    - Daily progress toward 1% goal
    - Learning statistics
    - Active strategies
    - Best conditions for trading
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        return brain.get_brain_status()
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.get("/api/ai-brain/daily-target", tags=["AI Brain"])
async def get_daily_target_status() -> Dict[str, Any]:
    """
    Get daily target tracking status.

    Returns progress toward the 1% daily goal including:
    - Current progress
    - Trades taken today
    - Risk budget used
    - Recommendation for next action
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        return brain.daily_tracker.get_status()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai-brain/strategies", tags=["AI Brain"])
async def get_ai_strategies() -> Dict[str, Any]:
    """
    Get all AI-generated trading strategies.

    Returns list of strategies with their entry/exit conditions and performance.
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        return {
            "strategies": brain.get_all_strategies(),
            "active_count": len([s for s in brain.strategy_generator.strategies if s.is_active]),
            "total_count": len(brain.strategy_generator.strategies),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"strategies": [], "error": str(e)}


@app.post("/api/ai-brain/strategies/generate", tags=["AI Brain"])
async def generate_new_strategy() -> Dict[str, Any]:
    """
    Generate a new trading strategy based on learned patterns.

    The AI will analyze the best performing patterns and create
    a new strategy with entry/exit conditions.
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        strategy = brain.generate_new_strategy()

        if strategy:
            return {
                "success": True,
                "strategy": strategy,
                "message": f"Created strategy: {strategy['name']}",
            }
        return {"success": False, "message": "Not enough pattern data to generate strategy"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/ai-brain/strategies/{name}/activate", tags=["AI Brain"])
async def activate_strategy(name: str) -> Dict[str, Any]:
    """Activate a strategy for live trading."""
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        success = brain.activate_strategy(name)

        return {
            "success": success,
            "strategy": name,
            "action": "activated" if success else "not found",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/ai-brain/strategies/{name}/deactivate", tags=["AI Brain"])
async def deactivate_strategy(name: str) -> Dict[str, Any]:
    """Deactivate a strategy."""
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        success = brain.deactivate_strategy(name)

        return {
            "success": success,
            "strategy": name,
            "action": "deactivated" if success else "not found",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/ai-brain/patterns/best", tags=["AI Brain"])
async def get_best_patterns(
    action: str = Query("buy", description="Action type: buy, sell, or hold"),
) -> Dict[str, Any]:
    """
    Get best performing patterns for a specific action.

    Shows which market conditions historically lead to the best outcomes
    for the specified action.
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        patterns = brain.pattern_learner.get_best_conditions_for_action(action, min_samples=5)

        return {
            "action": action,
            "best_patterns": patterns,
            "total_patterns": len(brain.pattern_learner.profitable_patterns)
            + len(brain.pattern_learner.losing_patterns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"action": action, "best_patterns": [], "error": str(e)}


@app.get("/api/ai-brain/insights", tags=["AI Brain"])
async def get_trade_insights() -> Dict[str, Any]:
    """
    Get aggregated insights from trade analysis.

    Returns learned lessons including:
    - Best conditions to trade
    - Conditions to avoid
    - Optimal holding time
    - Entry/exit timing recommendations
    """
    try:
        from bot.ai_trading_brain import get_ai_brain

        brain = get_ai_brain()
        return brain.trade_analyzer.get_insights()
    except Exception as e:
        return {"error": str(e)}


from bot.core.circuit_breaker import circuit_breaker, CommonCircuitBreakers


@app.post("/api/ai-brain/record-trade", tags=["AI Brain"])
@circuit_breaker(CommonCircuitBreakers.ML_PREDICTION, failure_threshold=3, timeout=30)
async def record_trade_for_learning(request: Request) -> Dict[str, Any]:
    """
    Record a completed trade for AI learning.

    The AI will analyze the trade and extract lessons for future improvement.
    """
    try:
        from bot.ai_trading_brain import get_ai_brain, MarketSnapshot, MarketCondition

        brain = get_ai_brain()

        data = await request.json()

        # Validate input data
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON data format")
        
        required_fields = ["symbol", "action", "entry_price"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate symbol
        validated_symbol = APIRequestValidator.sanitize_string(data.get("symbol", ""))
        if not validated_symbol:
            raise ValueError("Symbol is required and cannot be empty")

        # Validate action
        action = data.get("action", "").lower().strip()
        if action not in ["buy", "sell", "long", "short"]:
            raise ValueError("Action must be one of: buy, sell, long, short")

        # Validate prices
        try:
            entry_price = float(data.get("entry_price", 0))
            exit_price = float(data.get("exit_price", 0))
            
            if entry_price <= 0:
                raise ValueError("Entry price must be positive")
            
            if exit_price < 0:
                raise ValueError("Exit price cannot be negative")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid price format: {e}")

        # Create a market snapshot from data
        snapshot = MarketSnapshot(
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
            symbol=validated_symbol,
            price=entry_price,
            trend_1h=data.get("trend", "neutral"),
            rsi=float(data.get("rsi", 50)),
            volatility_percentile=float(data.get("volatility", 50)),
            condition=MarketCondition(data.get("market_condition", "sideways")),
        )

        result = brain.record_trade_result(
            trade_id=APIRequestValidator.sanitize_string(data.get("trade_id", f"trade_{int(time.time())}")),
            symbol=validated_symbol,
            entry_snapshot=snapshot,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=float(data.get("position_size", 1.0)),
            price_history=data.get("price_history", []),
            holding_hours=float(data.get("holding_hours", 0)),
        )

        logger.info(f"Trade recorded for learning: {validated_symbol} {action}", extra={
            "trade_id": result.get("trade_id"),
            "symbol": validated_symbol,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
        })

        return result

    except ValueError as e:
        logger.warning(f"Validation error in record_trade_for_learning: {e}")
        return {"success": False, "error": f"Validation error: {e}"}
    except ImportError as e:
        logger.error(f"AI brain module not available: {e}")
        return {"success": False, "error": "AI learning module not available"}
    except AttributeError as e:
        logger.error(f"AI brain interface error: {e}")
        return {"success": False, "error": "AI brain interface error"}
    except Exception as e:
        logger.error(f"Unexpected error in record_trade_for_learning: {e}", exc_info=True)
        return {"success": False, "error": f"Internal server error: {e}"}


# =============================================================================
# ML MODEL PERFORMANCE TRACKING ENDPOINTS
# =============================================================================


@app.get("/api/ml/model-performance", tags=["ML/AI"])
async def get_ml_model_performance(
    model_type: Optional[str] = Query(default=None, description="Filter by model type"),
    market_condition: Optional[str] = Query(default=None, description="Filter by market condition"),
    days: int = Query(default=30, description="Number of days to analyze"),
) -> Dict[str, Any]:
    """
    Get ML model performance metrics.

    Shows accuracy, profit factor, and other metrics for ML models.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()

        if model_type:
            return tracker.get_model_performance(
                model_type=model_type, market_condition=market_condition, days=days
            )
        else:
            return tracker.get_summary(days=days)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml/model-ranking", tags=["ML/AI"])
async def get_ml_model_ranking(
    days: int = Query(default=30, description="Number of days to analyze"),
) -> Dict[str, Any]:
    """
    Get ranking of ML models by performance.

    Returns models sorted by average return.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()
        ranking = tracker.get_model_ranking(days=days)

        return {"period_days": days, "ranking": ranking, "total_models": len(ranking)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml/best-model", tags=["ML/AI"])
async def get_best_model_for_condition(
    market_condition: str = Query(
        ..., description="Market condition (bull, bear, sideways, volatile)"
    ),
    min_predictions: int = Query(default=10, description="Minimum predictions required"),
) -> Dict[str, Any]:
    """
    Get the best performing model for a specific market condition.

    Helps with dynamic model selection based on current market.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()
        best = tracker.get_best_model_for_condition(
            market_condition=market_condition, min_predictions=min_predictions
        )

        if best:
            return {
                "market_condition": market_condition,
                "recommended_model": best["model_type"],
                "accuracy": best["accuracy"],
                "total_predictions": best["total_predictions"],
                "avg_return": best.get("avg_return", 0),
            }
        else:
            return {
                "market_condition": market_condition,
                "recommended_model": None,
                "message": f"No models with at least {min_predictions} predictions for this condition",
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml/recommendation", tags=["ML/AI"])
async def get_model_recommendation(
    market_condition: str = Query(..., description="Current market condition"),
) -> Dict[str, Any]:
    """
    Get model recommendation for current market conditions.

    Returns the recommended model with confidence score.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()
        return tracker.get_recommendation(market_condition)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ml/performance-matrix", tags=["ML/AI"])
async def get_performance_matrix(
    days: int = Query(default=30, description="Number of days to analyze"),
) -> Dict[str, Any]:
    """
    Get performance matrix of models by market condition.

    Shows how each model performs across different conditions.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()
        matrix = tracker.get_condition_performance_matrix(days=days)

        return {"period_days": days, "matrix": matrix}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/ml/record-outcome", tags=["ML/AI"])
async def record_prediction_outcome(request: Request) -> Dict[str, Any]:
    """
    Record the actual outcome of a prediction.

    Called after a trade closes to track model accuracy.
    """
    try:
        from bot.ml_performance_tracker import get_ml_tracker

        tracker = get_ml_tracker()
        data = await request.json()

        prediction_id = data.get("prediction_id")
        actual_return = data.get("actual_return", 0)

        if not prediction_id:
            return {"error": "prediction_id is required"}

        tracker.record_outcome(prediction_id, actual_return)

        return {"success": True, "prediction_id": prediction_id, "actual_return": actual_return}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# SYSTEM LOG ENDPOINTS
# =============================================================================


@app.get("/api/system/status", tags=["System"])
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.

    Shows active bots, AI components, recent errors, and health status.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        return syslog.get_system_status()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/events", tags=["System"])
async def get_system_events(
    limit: int = Query(default=100, description="Number of events to return"),
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
    component: Optional[str] = Query(default=None, description="Filter by component"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    since: Optional[str] = Query(default=None, description="Events since timestamp"),
) -> Dict[str, Any]:
    """
    Get system events log.

    Returns recent events with optional filtering.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        events = syslog.get_recent_events(
            limit=limit, event_type=event_type, component=component, severity=severity, since=since
        )

        return {"total": len(events), "events": events}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/summary", tags=["System"])
async def get_system_summary(
    hours: int = Query(default=24, description="Hours to summarize"),
) -> Dict[str, Any]:
    """
    Get system activity summary.

    Shows event counts by type, severity, and component.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        return syslog.get_summary(hours=hours)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/active", tags=["System"])
async def get_active_components() -> Dict[str, Any]:
    """
    Get all active components (bots, AI, etc.).

    Shows what's currently running.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        return syslog.get_active_components()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/errors", tags=["System"])
async def get_recent_errors(
    limit: int = Query(default=20, description="Number of errors to return"),
    hours: int = Query(default=24, description="Look back hours"),
) -> Dict[str, Any]:
    """
    Get recent system errors.

    Quick way to check for problems.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        errors = syslog.get_recent_events(limit=limit, severity="error", since=since)

        return {"period_hours": hours, "total_errors": len(errors), "errors": errors}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/system/last-entry", tags=["System"])
async def get_last_entry() -> Dict[str, Any]:
    """
    Get the last log entry - quick health check.

    Use this to see if the system is alive and what happened last.
    """
    try:
        from bot.system_logger import get_system_logger

        syslog = get_system_logger()
        status = syslog.get_system_status()
        events = syslog.get_recent_events(limit=1)

        return {
            "timestamp": datetime.now().isoformat(),
            "active_bots": status["active_bots"],
            "active_ai": status["active_ai"],
            "system_status": status["status"],
            "last_event": events[0] if events else None,
            "bots": [b["id"] for b in status["bots"]],
            "ai_components": [a["id"] for a in status["ai_components"]],
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# AI INTELLIGENCE ENDPOINTS
# ============================================================================


@app.get("/api/ai/brain/status", tags=["AI"])
async def get_brain_status() -> Dict[str, Any]:
    """
    Get the Intelligent Trading Brain status with live regime detection.

    Shows AI components health, pattern memory stats, and learning status.
    Also fetches current price data to detect live market regime.
    """
    try:
        from bot.intelligence import get_intelligent_brain
        import numpy as np

        brain = get_intelligent_brain()
        health = brain.health_check()

        # Try to get live price data and detect regime
        try:
            import yfinance as yf

            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period="5d", interval="1h")
            if len(hist) >= 50:
                prices = hist["Close"].values
                regime_state = brain.regime_adapter.detect_regime(prices)
                health["regime_adapter"]["current_regime"] = regime_state.regime.value
                health["regime_adapter"]["confidence"] = round(regime_state.confidence * 100, 1)
                health["regime_adapter"]["volatility"] = round(regime_state.volatility * 100, 2)
                health["regime_adapter"]["trend_strength"] = round(
                    regime_state.trend_strength * 100, 2
                )
        except Exception as e:
            pass  # Keep default values if price fetch fails

        return health
    except ImportError:
        return {"error": "Intelligent Brain not available", "status": "not_installed"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/brain/summary", tags=["AI"])
async def get_brain_summary() -> Dict[str, Any]:
    """
    Get comprehensive brain summary including learning stats.
    """
    try:
        from bot.intelligence import get_intelligent_brain

        brain = get_intelligent_brain()
        return brain.get_summary()
    except ImportError:
        return {"error": "Intelligent Brain not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/patterns", tags=["AI"])
async def get_patterns(
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get learned trading patterns from memory.
    """
    try:
        from bot.intelligence import PatternMemory

        memory = PatternMemory()
        patterns = memory.get_similar_patterns(
            symbol=symbol,
            regime=regime,
            limit=limit,
        )

        return {
            "total": len(patterns),
            "patterns": [p.to_dict() for p in patterns[:limit]],
            "summary": memory.get_summary(),
        }
    except ImportError:
        return {"error": "Pattern Memory not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/news", tags=["AI"])
async def get_news_context(
    symbols: str = "BTC,ETH",
    hours: int = 24,
) -> Dict[str, Any]:
    """
    Get news context and sentiment for symbols.
    """
    try:
        from bot.intelligence import NewsReasoner

        symbol_list = [s.strip() for s in symbols.split(",")]
        reasoner = NewsReasoner()
        context = reasoner.get_news_context(symbol_list, hours_lookback=hours)

        return context.to_dict()
    except ImportError:
        return {"error": "News Reasoner not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ai/regime", tags=["AI"])
async def get_regime_status() -> Dict[str, Any]:
    """
    Get current market regime detection status.
    """
    try:
        from bot.intelligence import get_intelligent_brain

        brain = get_intelligent_brain()
        return brain.regime_adapter.get_summary()
    except ImportError:
        return {"error": "Regime Adapter not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/pnl/calendar", tags=["Performance"])
async def get_pnl_calendar(days: int = 30) -> Dict[str, Any]:
    """
    Get P&L calendar with daily 1% target tracking.

    Returns daily performance for the last N days with indicators
    for whether the 1% daily target was reached.
    """
    from datetime import timedelta
    import sqlite3

    calendar = []
    daily_target_pct = 1.0  # 1% daily target

    # Collect data from all portfolio databases
    db_paths = [
        STATE_DIR / "live_paper_trading" / "portfolio.db",
        STATE_DIR / "stock_trading" / "portfolio.db",
        STATE_DIR / "commodity_trading" / "portfolio.db",
        STATE_DIR / "regime_trading" / "portfolio.db",
    ]

    # Get trades from all sources
    all_trades = []
    for db_path in db_paths:
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT date(timestamp) as day, SUM(pnl), COUNT(*)
                    FROM trades
                    WHERE timestamp >= date('now', ?)
                    GROUP BY date(timestamp)
                    ORDER BY day DESC
                """,
                    (f"-{days} days",),
                )
                rows = cursor.fetchall()
                conn.close()
                for row in rows:
                    all_trades.append(
                        {
                            "date": row[0],
                            "pnl": row[1] or 0,
                            "trades": row[2] or 0,
                        }
                    )
            except (sqlite3.Error, sqlite3.OperationalError) as e:
                logger.debug(f"Failed to read trades from {db_path}: {e}")

    # Aggregate by date
    daily_data = {}
    for trade in all_trades:
        date = trade["date"]
        if date not in daily_data:
            daily_data[date] = {"pnl": 0, "trades": 0}
        daily_data[date]["pnl"] += trade["pnl"]
        daily_data[date]["trades"] += trade["trades"]

    # Build calendar with target indicators
    # Assume starting capital of 10000 for percentage calculation
    starting_capital = 10000.0

    today = datetime.now().date()
    stats = {"target_hit_days": 0, "total_days": 0, "total_pnl": 0, "best_day": 0, "worst_day": 0}

    for i in range(days):
        date = (today - timedelta(days=i)).isoformat()
        data = daily_data.get(date, {"pnl": 0, "trades": 0})
        pnl = data["pnl"]
        pnl_pct = (pnl / starting_capital) * 100 if starting_capital > 0 else 0
        target_hit = pnl_pct >= daily_target_pct

        if data["trades"] > 0:
            stats["total_days"] += 1
            stats["total_pnl"] += pnl
            if target_hit:
                stats["target_hit_days"] += 1
            if pnl > stats["best_day"]:
                stats["best_day"] = pnl
            if pnl < stats["worst_day"]:
                stats["worst_day"] = pnl

        calendar.append(
            {
                "date": date,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "trades": data["trades"],
                "target_hit": target_hit,
                "status": "hit"
                if target_hit
                else ("loss" if pnl < 0 else ("profit" if pnl > 0 else "neutral")),
            }
        )

    return {
        "calendar": calendar,
        "daily_target_pct": daily_target_pct,
        "stats": {
            "target_hit_days": stats["target_hit_days"],
            "trading_days": stats["total_days"],
            "hit_rate": round(stats["target_hit_days"] / max(1, stats["total_days"]) * 100, 1),
            "total_pnl": round(stats["total_pnl"], 2),
            "best_day": round(stats["best_day"], 2),
            "worst_day": round(stats["worst_day"], 2),
        },
        "today": {
            "date": today.isoformat(),
            "pnl": calendar[0]["pnl"] if calendar else 0,
            "pnl_pct": calendar[0]["pnl_pct"] if calendar else 0,
            "target_hit": calendar[0]["target_hit"] if calendar else False,
        },
    }
