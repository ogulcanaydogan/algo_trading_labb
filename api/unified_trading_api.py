"""
Unified Trading API Endpoints.

Provides REST API endpoints for the unified trading engine:
- Mode switching
- Status monitoring
- Transition validation
- Safety controls
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/unified",
    tags=["Unified Trading"],
    responses={
        401: {"description": "Unauthorized - Invalid or missing API key"},
        403: {"description": "Forbidden - Insufficient permissions"},
        500: {"description": "Internal server error"},
    },
)


# =============================================================================
# Pydantic Models
# =============================================================================


class TradingModeInfo(BaseModel):
    """Information about a trading mode and its configuration.

    Each mode has different risk characteristics and safety limits.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "paper_live_data",
                "description": "Paper trading with real market data",
                "is_paper": True,
                "is_live": False,
                "uses_real_data": True,
                "capital_limit": 10000.0,
                "max_position_usd": 1000.0,
                "max_daily_loss_usd": 200.0,
            }
        }
    )

    mode: str = Field(..., description="Mode identifier (e.g., paper_live_data)")
    description: str = Field(..., description="Human-readable mode description")
    is_paper: bool = Field(..., description="Whether this is paper trading")
    is_live: bool = Field(..., description="Whether this uses real money")
    uses_real_data: bool = Field(..., description="Whether this uses real market data")
    capital_limit: Optional[float] = Field(None, description="Maximum capital allowed")
    max_position_usd: Optional[float] = Field(None, description="Max position size in USD")
    max_daily_loss_usd: Optional[float] = Field(None, description="Max daily loss allowed")


class UnifiedStatusResponse(BaseModel):
    """Response model for unified trading status.

    Data Format Conventions:
    - win_rate: Decimal (0.0-1.0) - multiply by 100 for display
    - max_drawdown: Percentage (0-100) - display directly with %
    - total_pnl_pct: Percentage (0-100) - display directly with %
    - portfolio_value: USD - includes cash + positions at current prices

    Note: /api/unified/analytics uses percentage format for win_rate and rolling metrics
    because its dashboard display expects pre-formatted percentages.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "paper_live_data",
                "status": "active",
                "running": True,
                "balance": 9850.50,
                "initial_capital": 10000.0,
                "portfolio_value": 10250.75,
                "total_pnl": 250.75,
                "total_pnl_pct": 2.51,
                "total_trades": 15,
                "win_rate": 0.67,
                "max_drawdown": 3.2,
                "open_positions": 2,
                "positions": {
                    "BTC/USDT": {"quantity": 0.01, "entry_price": 42000.0},
                },
                "safety": {"emergency_stop_active": False},
                "daily_trades": 3,
                "daily_pnl": 125.50,
            }
        }
    )

    mode: str = Field(..., description="Current trading mode")
    status: str = Field(..., description="Engine status (active/stopped)")
    running: bool = Field(..., description="Whether engine is actively trading")
    balance: float = Field(..., description="Current cash balance in USD")
    initial_capital: float = Field(..., description="Starting capital")
    portfolio_value: float = Field(
        ..., description="Total value: cash + positions at current prices"
    )
    total_pnl: float = Field(..., description="Total profit/loss in USD")
    total_pnl_pct: float = Field(..., description="Total P&L as percentage (0-100)")
    total_trades: int = Field(..., description="Total number of completed trades")
    win_rate: float = Field(..., description="Win rate as decimal (0.0-1.0)")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage (0-100)")
    open_positions: int = Field(..., description="Number of currently open positions")
    positions: Dict[str, Any] = Field(..., description="Current position details")
    safety: Dict[str, Any] = Field(..., description="Safety controller status")
    daily_trades: int = Field(..., description="Trades executed today")
    daily_pnl: float = Field(..., description="P&L for today in USD")


class TransitionProgressResponse(BaseModel):
    """Response model for transition progress."""

    from_mode: str
    to_mode: str
    overall_progress: float
    allowed: bool
    requires_approval: bool
    progress_details: List[Dict[str, Any]]
    blocking_reasons: List[str]
    estimated_days_remaining: Optional[int]


class SafetyStatusResponse(BaseModel):
    """Response model for safety status."""

    status: str
    emergency_stop_active: bool
    emergency_stop_reason: Optional[str]
    daily_stats: Dict[str, Any]
    limits: Dict[str, Any]
    current_balance: float
    peak_balance: float
    open_positions: int


class SafetyLimitsResponse(BaseModel):
    """Response model for current safety limits and utilization."""

    max_position_size_usd: float
    max_position_size_pct: float
    max_daily_loss_usd: float
    max_daily_loss_pct: float
    max_trades_per_day: int
    max_open_positions: int
    daily_loss_used_pct: float
    trades_remaining: int
    positions_open: int
    emergency_stop_active: bool


class ModeSwitchRequest(BaseModel):
    """Request model for mode switching."""

    target_mode: str
    confirm: bool = False
    approver: Optional[str] = None


class EmergencyStopRequest(BaseModel):
    """Request model for emergency stop."""

    reason: str


class ClearStopRequest(BaseModel):
    """Request model for clearing emergency stop."""

    approver: str


class ExecutionSymbolTelemetry(BaseModel):
    symbol: str
    trades: int
    avg_slippage_pct: float
    avg_execution_time_ms: float
    total_commission: float


class ExecutionTelemetryResponse(BaseModel):
    total_trades: int
    avg_slippage_pct: float
    avg_execution_time_ms: float
    total_commission: float
    worst_slippage_pct: float
    symbol_breakdown: List[ExecutionSymbolTelemetry]


# =============================================================================
# Helper Functions
# =============================================================================


def _get_state_path() -> Path:
    """Get the unified trading state path."""
    return Path("data/unified_trading/state.json")


def _load_execution_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load execution log; rebuild from DataStore trades if missing/empty."""
    log_path = Path("data/unified_trading/execution_log.json")

    try:
        if log_path.exists():
            with open(log_path) as f:
                logs = json.load(f)
        else:
            logs = []
    except (OSError, json.JSONDecodeError):
        logs = []

    if logs:
        return logs[-limit:] if limit else logs

    # Rebuild from DataStore trades when execution log is absent
    try:
        from bot.data_store import DataStore

        ds = DataStore()
        trades = ds.get_trades(limit=5000)  # grab recent history
        rebuilt: List[Dict[str, Any]] = []

        for trade in trades:
            meta = trade.get("metadata", {}) or {}
            rebuilt.append(
                {
                    "order_id": trade.get("trade_id")
                    or meta.get("order_id")
                    or f"sync_{trade.get('timestamp', '')}",
                    "timestamp": trade.get("timestamp"),
                    "symbol": trade.get("symbol"),
                    "side": str(trade.get("action", "")).lower(),
                    "quantity": trade.get("quantity", 0),
                    "price": trade.get("price", 0),
                    "fill_price": trade.get("price", None),
                    "slippage_pct": meta.get("slippage_pct"),
                    "commission": meta.get("fees_paid") or trade.get("fees_paid") or 0,
                    "status": meta.get("status", "filled"),
                    "mode": meta.get("mode", meta.get("execution_mode", "sync")),
                    "execution_time_ms": meta.get("execution_time_ms"),
                }
            )

        if rebuilt:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(rebuilt, f, indent=2)
            return rebuilt[-limit:] if limit else rebuilt
    except Exception as e:  # pragma: no cover - safe fallback
        logger.warning(f"Could not rebuild execution log: {e}")

    return []


def _get_safety_path() -> Path:
    """Get the safety state path."""
    return Path("data/safety_state.json")


def _load_state() -> Optional[Dict[str, Any]]:
    """Load the current unified trading state."""
    state_path = _get_state_path()
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _load_safety_state() -> Optional[Dict[str, Any]]:
    """Load the current safety state."""
    safety_path = _get_safety_path()
    if safety_path.exists():
        try:
            with open(safety_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return None


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/modes",
    response_model=List[TradingModeInfo],
    summary="List available trading modes",
    description="""
    Returns all available trading modes with their configurations.

    **Available Modes:**
    - `backtest`: Historical backtesting with simulated trades
    - `paper_synthetic`: Paper trading with synthetic data
    - `paper_live_data`: Paper trading with real market data
    - `testnet`: Real orders on exchange testnet
    - `live_limited`: Live trading with strict capital limits ($100 max)
    - `live_full`: Full live trading with configurable limits

    Each mode has different safety limits and risk characteristics.
    """,
)
async def get_available_modes():
    """Get list of available trading modes with their configurations."""
    from bot.trading_mode import ModeConfig, TradingMode

    modes = []
    for mode in TradingMode:
        config = ModeConfig.get_default(mode)
        modes.append(
            TradingModeInfo(
                mode=mode.value,
                description=_get_mode_description(mode),
                is_paper=mode.is_paper,
                is_live=mode.is_live,
                uses_real_data=mode.uses_real_data,
                capital_limit=config.capital_limit,
                max_position_usd=config.max_position_usd
                if config.max_position_usd < float("inf")
                else None,
                max_daily_loss_usd=config.max_daily_loss_usd
                if config.max_daily_loss_usd < float("inf")
                else None,
            )
        )
    return modes


def _get_mode_description(mode) -> str:
    """Get human-readable description for a mode."""
    descriptions = {
        "backtest": "Historical backtesting with simulated trades",
        "paper_synthetic": "Paper trading with synthetic data",
        "paper_live_data": "Paper trading with real market data",
        "testnet": "Real orders on exchange testnet",
        "live_limited": "Live trading with strict capital limits ($100 max)",
        "live_full": "Full live trading with configurable limits",
    }
    return descriptions.get(mode.value, "Unknown mode")


@router.get(
    "/status",
    response_model=UnifiedStatusResponse,
    summary="Get trading engine status",
    description="""
    Returns the current state of the unified trading engine.

    **Includes:**
    - Current trading mode and status
    - Balance and portfolio value
    - P&L metrics (total and daily)
    - Open positions
    - Safety controller status
    - Win rate and drawdown statistics

    **Data Format Notes:**
    - `win_rate`: Decimal (0.0-1.0), multiply by 100 for percentage
    - `max_drawdown`: Already in percentage (0-100)
    - `total_pnl_pct`: Already in percentage (0-100)
    """,
    responses={
        404: {"description": "No trading state found - start trading first"},
    },
)
async def get_unified_status():
    """Get current unified trading engine status."""
    from datetime import datetime

    state = _load_state()
    safety = _load_safety_state()

    if not state:
        raise HTTPException(
            status_code=404,
            detail="No unified trading state found. Start trading first.",
        )

    # Check if it's a new day and reset daily stats if needed
    today = datetime.now().strftime("%Y-%m-%d")
    stored_date = state.get("daily_date", "")
    if today != stored_date:
        # New day - reset daily stats
        state["daily_date"] = today
        state["daily_trades"] = 0
        state["daily_pnl"] = 0.0
        state["daily_losses"] = 0
        # Save the reset state
        state_file = _get_state_path()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    # Calculate win rate as decimal (0.0-1.0) for consistency with /performance endpoint
    total_trades = state.get("total_trades", 0)
    winning_trades = state.get("winning_trades", 0)
    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

    # Calculate portfolio value: cash balance + positions value at current prices
    current_balance = state.get("current_balance", 0)
    positions_dict = state.get("positions", {})
    positions_value = 0.0

    # Filter and count only active positions (with quantity > 0)
    active_positions = {}
    for symbol, pos_data in positions_dict.items():
        if isinstance(pos_data, dict):
            qty = pos_data.get("quantity", 0) or 0
            if qty > 0:
                # Use current_price if available, fallback to entry_price
                current_price = pos_data.get("current_price") or pos_data.get("entry_price", 0) or 0
                positions_value += qty * current_price
                active_positions[symbol] = pos_data

    portfolio_value = current_balance + positions_value

    return UnifiedStatusResponse(
        mode=state.get("mode", "unknown"),
        status=state.get("status", "unknown"),
        running=state.get("status") == "active",
        balance=current_balance,
        initial_capital=state.get("initial_capital", 0),
        portfolio_value=portfolio_value,
        total_pnl=state.get("total_pnl", 0),
        total_pnl_pct=(
            state.get("total_pnl", 0) / state.get("initial_capital", 1) * 100
            if state.get("initial_capital", 0) > 0
            else 0
        ),
        total_trades=total_trades,
        win_rate=win_rate,  # Now decimal 0.0-1.0
        max_drawdown=state.get("max_drawdown_pct", 0) * 100,
        open_positions=len(active_positions),  # Only count active positions
        positions=state.get("positions", {}),
        safety=safety or {},
        daily_trades=state.get("daily_trades", 0),
        daily_pnl=state.get("daily_pnl", 0),
    )


@router.get("/transition-progress", response_model=TransitionProgressResponse)
async def get_transition_progress(
    target_mode: str = Query(..., description="Target mode to check progress for"),
):
    """Get progress towards transitioning to a target mode."""
    from bot.trading_mode import TradingMode
    from bot.transition_validator import create_transition_validator
    from bot.unified_state import UnifiedStateStore

    try:
        target = TradingMode(target_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target mode: {target_mode}",
        )

    state_store = UnifiedStateStore()
    try:
        state = state_store.initialize(
            mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=30000,
            resume=True,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No trading state found. Start trading first.",
        )

    current_mode = state.mode
    mode_state = state_store.get_mode_state()
    validator = create_transition_validator()

    progress = validator.get_transition_progress(current_mode, target, mode_state)

    return TransitionProgressResponse(**progress)


@router.get(
    "/safety",
    response_model=SafetyStatusResponse,
    summary="Get safety controller status",
    description="""
    Returns the current state of the safety controller.

    **Includes:**
    - Emergency stop status and reason
    - Daily trading statistics
    - Position limits and utilization
    - Current balance and peak balance
    """,
)
async def get_safety_status():
    """Get current safety controller status and limits."""
    from bot.safety_controller import SafetyController

    controller = SafetyController()
    status = controller.get_status()

    return SafetyStatusResponse(**status)


@router.get("/safety/limits", response_model=SafetyLimitsResponse)
async def get_safety_limits():
    """Get safety limits and current utilization for dashboards/monitoring."""
    from bot.safety_controller import SafetyController

    controller = SafetyController()
    status = controller.get_status()

    limits = status.get("limits", {})
    daily_loss = status.get("daily_stats", {}).get("total_loss", 0.0) or 0.0
    max_daily_loss_usd = float(limits.get("max_daily_loss_usd", 0.0) or 0.0)
    daily_loss_used_pct = (daily_loss / max_daily_loss_usd) * 100 if max_daily_loss_usd > 0 else 0.0

    trades_remaining = int(limits.get("trades_remaining", 0) or 0)
    positions_open = int(status.get("open_positions", 0) or 0)

    # Some limits may not be present in compact status; default to 0/False
    return SafetyLimitsResponse(
        max_position_size_usd=float(limits.get("max_position_size_usd", 0.0) or 0.0),
        max_position_size_pct=float(limits.get("max_position_size_pct", 0.0) or 0.0)
        if "max_position_size_pct" in limits
        else 0.05,
        max_daily_loss_usd=max_daily_loss_usd,
        max_daily_loss_pct=float(limits.get("max_daily_loss_pct", 0.0) or 0.0)
        if "max_daily_loss_pct" in limits
        else 0.02,
        max_trades_per_day=int(limits.get("max_trades_per_day", 0) or 0),
        max_open_positions=int(limits.get("max_open_positions", 0) or 0)
        if "max_open_positions" in limits
        else 3,
        daily_loss_used_pct=round(daily_loss_used_pct, 2),
        trades_remaining=trades_remaining,
        positions_open=positions_open,
        emergency_stop_active=bool(status.get("emergency_stop_active", False)),
    )


@router.post(
    "/switch-mode",
    summary="Switch trading mode",
    description="""
    Switch to a different trading mode.

    **Safety Requirements:**
    - Live modes require `confirm=true` in the request
    - Live modes require an `approver` name to be specified
    - Certain transitions may be blocked based on trading performance

    **Mode Progression:**
    - Paper modes can be switched freely
    - Testnet requires paper trading experience
    - Live limited requires successful testnet trading
    - Live full requires proven live limited performance
    """,
    responses={
        400: {"description": "Invalid mode or missing confirmation"},
        404: {"description": "No trading state found"},
    },
)
async def switch_mode(request: ModeSwitchRequest):
    """
    Switch trading mode with safety validations.

    Requires confirmation for live modes and approval for restricted transitions.
    """
    from bot.trading_mode import TradingMode
    from bot.transition_validator import create_transition_validator
    from bot.unified_state import UnifiedStateStore

    try:
        target = TradingMode(request.target_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target mode: {request.target_mode}",
        )

    # Safety check for live modes
    if target.is_live and not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required for live trading modes. Set confirm=true.",
        )

    if target.is_live and not request.approver:
        raise HTTPException(
            status_code=400,
            detail="Approver name required for live trading modes.",
        )

    # Load current state
    state_store = UnifiedStateStore()
    try:
        state = state_store.initialize(
            mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=30000,
            resume=True,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No trading state found. Start trading first.",
        )

    current_mode = state.mode
    mode_state = state_store.get_mode_state()
    validator = create_transition_validator()

    # Check if transition is allowed
    result = validator.can_transition(current_mode, target, mode_state)

    if not result.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Transition not allowed",
                "blocking_reasons": result.blocking_reasons,
            },
        )

    # Record mode change
    state_store.change_mode(target, reason="api_switch", approver=request.approver or "")

    return {
        "success": True,
        "message": f"Mode switched from {current_mode.value} to {target.value}",
        "new_mode": target.value,
        "approver": request.approver,
    }


@router.get("/execution-telemetry", response_model=ExecutionTelemetryResponse)
async def get_execution_telemetry():
    """High-signal telemetry derived from the execution log."""
    logs = _load_execution_log()
    if not logs:
        return ExecutionTelemetryResponse(
            total_trades=0,
            avg_slippage_pct=0,
            avg_execution_time_ms=0,
            total_commission=0,
            worst_slippage_pct=0,
            symbol_breakdown=[],
        )

    slippages = [l.get("slippage_pct") for l in logs if l.get("slippage_pct") is not None]
    exec_times = [
        l.get("execution_time_ms") for l in logs if l.get("execution_time_ms") is not None
    ]
    total_commission = sum((l.get("commission") or 0) for l in logs)
    symbol_map: Dict[str, Dict[str, Any]] = {}

    for log in logs:
        symbol = log.get("symbol", "unknown")
        entry = symbol_map.setdefault(
            symbol,
            {
                "trades": 0,
                "slippages": [],
                "exec_times": [],
                "commission": 0,
            },
        )
        slip = log.get("slippage_pct")
        if slip is not None:
            entry["slippages"].append(slip)
        exec_time = log.get("execution_time_ms")
        if exec_time is not None:
            entry["exec_times"].append(exec_time)
        entry["trades"] += 1
        entry["commission"] += log.get("commission", 0) or 0

    symbol_breakdown = []
    for symbol, entry in symbol_map.items():
        avg_slip = sum(entry["slippages"]) / len(entry["slippages"]) if entry["slippages"] else 0
        avg_exec = sum(entry["exec_times"]) / len(entry["exec_times"]) if entry["exec_times"] else 0
        symbol_breakdown.append(
            ExecutionSymbolTelemetry(
                symbol=symbol,
                trades=entry["trades"],
                avg_slippage_pct=avg_slip,
                avg_execution_time_ms=avg_exec,
                total_commission=entry["commission"],
            )
        )

    return ExecutionTelemetryResponse(
        total_trades=len(logs),
        avg_slippage_pct=sum(slippages) / len(slippages) if slippages else 0,
        avg_execution_time_ms=sum(exec_times) / len(exec_times) if exec_times else 0,
        total_commission=total_commission,
        worst_slippage_pct=max(slippages) if slippages else 0,
        symbol_breakdown=symbol_breakdown,
    )


@router.post("/emergency-stop")
async def trigger_emergency_stop(request: EmergencyStopRequest):
    """Trigger emergency stop - halts all trading immediately."""
    from bot.safety_controller import SafetyController

    controller = SafetyController()
    controller.emergency_stop(request.reason)

    return {
        "success": True,
        "message": "Emergency stop triggered",
        "reason": request.reason,
    }


@router.post("/clear-stop")
async def clear_emergency_stop(request: ClearStopRequest):
    """Clear emergency stop to resume trading."""
    from bot.safety_controller import SafetyController

    controller = SafetyController()
    success = controller.clear_emergency_stop(approver=request.approver)

    if success:
        return {
            "success": True,
            "message": "Emergency stop cleared",
            "approver": request.approver,
        }
    else:
        return {
            "success": False,
            "message": "No emergency stop was active",
        }


@router.get("/trades")
async def get_recent_trades(limit: int = Query(default=50, le=500)):
    """Get recent trade history."""
    trades_path = Path("data/unified_trading/trades.json")

    if not trades_path.exists():
        return []

    try:
        with open(trades_path) as f:
            trades = json.load(f)
        return trades[-limit:]
    except (OSError, json.JSONDecodeError):
        return []


@router.get("/equity")
async def get_equity_curve(limit: int = Query(default=500, le=5000)):
    """Get equity curve data."""
    equity_path = Path("data/unified_trading/equity.json")

    if not equity_path.exists():
        return []

    try:
        with open(equity_path) as f:
            equity = json.load(f)
        return equity[-limit:]
    except (OSError, json.JSONDecodeError):
        return []


@router.get("/mode-history")
async def get_mode_history():
    """Get history of mode transitions."""
    history_path = Path("data/unified_trading/mode_history.json")

    if not history_path.exists():
        return []

    try:
        with open(history_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


# =============================================================================
# Real Trading Readiness Endpoints
# =============================================================================


class ExchangeConnectionResponse(BaseModel):
    """Response for exchange connection test."""

    connected: bool
    exchange: str
    mode: str  # testnet or live
    balance_available: Optional[float]
    error: Optional[str]
    latency_ms: Optional[float]
    timestamp: str


class OrderExecutionLogEntry(BaseModel):
    """Order execution log entry."""

    order_id: str
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    fill_price: Optional[float]
    slippage_pct: Optional[float]
    commission: float
    status: str
    mode: str
    execution_time_ms: Optional[float]


class PendingOrderConfirmation(BaseModel):
    """Pending order awaiting confirmation."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    estimated_value: float
    signal_confidence: float
    signal_reason: str
    risk_assessment: Dict[str, Any]
    created_at: str
    expires_at: str


class PositionDetail(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    pct_of_portfolio: float
    entry_time: str
    stop_loss: Optional[float]
    take_profit: Optional[float]


class DetailedPositionsResponse(BaseModel):
    positions: List[PositionDetail]
    total_market_value: float
    total_unrealized_pnl: float


class ReadinessCheckResponse(BaseModel):
    """Real trading readiness check response."""

    ready: bool
    current_mode: str
    target_mode: str
    checks: List[Dict[str, Any]]
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


@router.get("/exchange/test-connection", response_model=ExchangeConnectionResponse)
async def test_exchange_connection(
    mode: str = Query(default="testnet", description="testnet or live"),
):
    """
    Test connection to exchange.

    Tests API connectivity, authentication, and retrieves balance.
    Use 'testnet' mode for safe testing before going live.
    """
    import os
    import time
    from datetime import datetime

    # Use testnet keys if mode is testnet, otherwise use live keys
    if mode == "testnet":
        api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    else:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        return ExchangeConnectionResponse(
            connected=False,
            exchange="binance",
            mode=mode,
            balance_available=None,
            error="API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET in .env",
            latency_ms=None,
            timestamp=datetime.now().isoformat(),
        )

    try:
        import ccxt

        start_time = time.time()

        exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "sandbox": mode == "testnet",
                "options": {"defaultType": "spot"},
            }
        )

        # Test connection by fetching balance
        balance = exchange.fetch_balance()
        latency = (time.time() - start_time) * 1000

        usdt_balance = balance.get("USDT", {}).get("free", 0)

        return ExchangeConnectionResponse(
            connected=True,
            exchange="binance",
            mode=mode,
            balance_available=float(usdt_balance),
            error=None,
            latency_ms=round(latency, 2),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return ExchangeConnectionResponse(
            connected=False,
            exchange="binance",
            mode=mode,
            balance_available=None,
            error=str(e),
            latency_ms=None,
            timestamp=datetime.now().isoformat(),
        )


@router.get("/execution-log", response_model=List[OrderExecutionLogEntry])
async def get_execution_log(limit: int = Query(default=50, le=500)):
    """
    Get order execution log with slippage tracking.

    Shows all executed orders with fill prices, slippage, and execution times.
    Critical for monitoring real trading performance.
    """
    return _load_execution_log(limit=limit)


@router.get("/pending-orders", response_model=List[PendingOrderConfirmation])
async def get_pending_orders():
    """
    Get orders pending confirmation.

    In live_limited mode, orders require manual confirmation before execution.
    This endpoint shows pending orders that need approval.
    """
    pending_path = Path("data/unified_trading/pending_orders.json")

    if not pending_path.exists():
        return []

    try:
        with open(pending_path) as f:
            orders = json.load(f)
        # Filter out expired orders
        from datetime import datetime

        now = datetime.now().isoformat()
        return [o for o in orders if o.get("expires_at", "") > now]
    except (OSError, json.JSONDecodeError):
        return []


@router.post("/confirm-order/{order_id}")
async def confirm_order(order_id: str, approve: bool = True):
    """
    Confirm or reject a pending order.

    Required for live trading when require_confirmation is enabled.
    """
    pending_path = Path("data/unified_trading/pending_orders.json")

    if not pending_path.exists():
        raise HTTPException(status_code=404, detail="No pending orders")

    try:
        with open(pending_path) as f:
            orders = json.load(f)

        # Find and remove the order
        found = None
        remaining = []
        for order in orders:
            if order.get("order_id") == order_id:
                found = order
            else:
                remaining.append(order)

        if not found:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        # Save remaining orders
        with open(pending_path, "w") as f:
            json.dump(remaining, f, indent=2)

        if approve:
            # Move to confirmed orders for execution
            confirmed_path = Path("data/unified_trading/confirmed_orders.json")
            confirmed = []
            if confirmed_path.exists():
                with open(confirmed_path) as f:
                    confirmed = json.load(f)
            confirmed.append(found)
            with open(confirmed_path, "w") as f:
                json.dump(confirmed, f, indent=2)

            return {"success": True, "message": f"Order {order_id} approved for execution"}
        else:
            return {"success": True, "message": f"Order {order_id} rejected"}

    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness-check", response_model=ReadinessCheckResponse)
async def check_trading_readiness(
    target_mode: str = Query(
        default=None, description="Target trading mode (if None, uses next mode in progression)"
    ),
):
    """
    Comprehensive readiness check for transitioning to live trading.

    Checks:
    - API key configuration
    - Exchange connectivity
    - Paper trading performance requirements
    - Safety limits configuration
    - Risk management setup
    - Days in current mode
    """
    import os
    from bot.trading_mode import TradingMode

    # Load current state to determine actual current mode
    state = _load_state()
    current_mode_str = state.get("mode", "paper_live_data") if state else "paper_live_data"
    current_mode = TradingMode(current_mode_str)

    # If no target_mode specified, get the next mode in progression
    if target_mode is None:
        progression = TradingMode.get_progression()
        try:
            current_idx = progression.index(current_mode)
            if current_idx < len(progression) - 1:
                target = progression[current_idx + 1]
            else:
                target = progression[-1]  # Already at highest mode
        except ValueError:
            target = TradingMode.TESTNET  # Default fallback
    else:
        try:
            target = TradingMode(target_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid target mode: {target_mode}")

    checks = []
    blocking = []
    warnings = []
    recommendations = []

    # 1. API Keys Check
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if api_key and api_secret:
        checks.append(
            {
                "name": "API Keys",
                "status": "pass",
                "message": "Binance API keys configured",
                "icon": "🔑",
            }
        )
    else:
        checks.append(
            {
                "name": "API Keys",
                "status": "fail",
                "message": "API keys not configured",
                "icon": "🔑",
            }
        )
        blocking.append("Configure BINANCE_API_KEY and BINANCE_API_SECRET in .env file")

    # 2. Paper Trading Performance Check
    state = _load_state()
    if state:
        total_trades = state.get("total_trades", 0)
        winning_trades = state.get("winning_trades", 0)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        max_drawdown = state.get("max_drawdown_pct", 0)

        # Check minimum trades
        min_trades_required = 100 if target == TradingMode.LIVE_LIMITED else 200
        if total_trades >= min_trades_required:
            checks.append(
                {
                    "name": "Trade Count",
                    "status": "pass",
                    "message": f"{total_trades} trades completed (min: {min_trades_required})",
                    "icon": "📊",
                }
            )
        else:
            checks.append(
                {
                    "name": "Trade Count",
                    "status": "fail",
                    "message": f"{total_trades}/{min_trades_required} trades",
                    "icon": "📊",
                }
            )
            blocking.append(f"Complete {min_trades_required - total_trades} more paper trades")

        # Check win rate
        min_win_rate = 0.45
        if win_rate >= min_win_rate:
            checks.append(
                {
                    "name": "Win Rate",
                    "status": "pass",
                    "message": f"{win_rate:.1%} (min: {min_win_rate:.0%})",
                    "icon": "🎯",
                }
            )
        else:
            checks.append(
                {
                    "name": "Win Rate",
                    "status": "fail",
                    "message": f"{win_rate:.1%} (need {min_win_rate:.0%})",
                    "icon": "🎯",
                }
            )
            blocking.append(f"Improve win rate to at least {min_win_rate:.0%}")

        # Check drawdown
        max_allowed_dd = 0.12
        if max_drawdown <= max_allowed_dd:
            checks.append(
                {
                    "name": "Max Drawdown",
                    "status": "pass",
                    "message": f"{max_drawdown:.1%} (max allowed: {max_allowed_dd:.0%})",
                    "icon": "📉",
                }
            )
        else:
            checks.append(
                {
                    "name": "Max Drawdown",
                    "status": "warn",
                    "message": f"{max_drawdown:.1%} exceeds {max_allowed_dd:.0%}",
                    "icon": "📉",
                }
            )
            warnings.append("High drawdown - consider adjusting risk parameters")

        # Check days in mode (NEW: This is critical for mode transitions)
        days_in_mode = state.get("days_in_mode", 0)

        # Get required days based on transition path
        required_days = 0
        if current_mode == TradingMode.PAPER_LIVE_DATA and target == TradingMode.TESTNET:
            required_days = 14
        elif current_mode == TradingMode.TESTNET and target == TradingMode.LIVE_LIMITED:
            required_days = 14
        elif current_mode == TradingMode.LIVE_LIMITED and target == TradingMode.LIVE_FULL:
            required_days = 30

        if required_days > 0:
            if days_in_mode >= required_days:
                checks.append(
                    {
                        "name": "Time in Mode",
                        "status": "pass",
                        "message": f"{days_in_mode} days in current mode (min: {required_days})",
                        "icon": "⏰",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "Time in Mode",
                        "status": "fail",
                        "message": f"{days_in_mode}/{required_days} days in current mode",
                        "icon": "⏰",
                    }
                )
                days_remaining = required_days - days_in_mode
                blocking.append(
                    f"Wait {days_remaining} more day(s) in current mode before transitioning"
                )
    else:
        checks.append(
            {
                "name": "Paper Trading",
                "status": "fail",
                "message": "No paper trading history found",
                "icon": "📊",
            }
        )
        blocking.append("Complete paper trading first before going live")

    # 3. Safety Configuration Check
    safety_state = _load_safety_state()
    if safety_state and not safety_state.get("emergency_stop_active"):
        checks.append(
            {
                "name": "Safety System",
                "status": "pass",
                "message": "Safety controller active, no emergency stops",
                "icon": "🛡️",
            }
        )
    else:
        if safety_state and safety_state.get("emergency_stop_active"):
            checks.append(
                {
                    "name": "Safety System",
                    "status": "fail",
                    "message": f"Emergency stop active: {safety_state.get('emergency_stop_reason')}",
                    "icon": "🛡️",
                }
            )
            blocking.append("Clear emergency stop before proceeding")
        else:
            checks.append(
                {
                    "name": "Safety System",
                    "status": "warn",
                    "message": "Safety state not initialized",
                    "icon": "🛡️",
                }
            )

    # 4. Telegram Notifications Check
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")

    if telegram_token and telegram_chat:
        checks.append(
            {
                "name": "Notifications",
                "status": "pass",
                "message": "Telegram notifications configured",
                "icon": "📱",
            }
        )
    else:
        checks.append(
            {
                "name": "Notifications",
                "status": "warn",
                "message": "Telegram not configured - recommended for live trading",
                "icon": "📱",
            }
        )
        recommendations.append("Configure Telegram for real-time trade alerts")

    # 5. Capital Limits Check
    if target == TradingMode.LIVE_LIMITED:
        checks.append(
            {
                "name": "Capital Limit",
                "status": "pass",
                "message": "$100 max capital enforced for live_limited mode",
                "icon": "💰",
            }
        )
        recommendations.append("Start with small amounts ($20-50) to validate execution")

    # Add general recommendations
    recommendations.extend(
        [
            "Test with Binance Testnet first before using real funds",
            "Monitor first few live trades closely for any issues",
            "Keep emergency stop accessible on your dashboard",
            "Review execution log after each trade for slippage",
        ]
    )

    ready = len(blocking) == 0

    return ReadinessCheckResponse(
        ready=ready,
        current_mode=current_mode_str,
        target_mode=target.value,
        checks=checks,
        blocking_issues=blocking,
        warnings=warnings,
        recommendations=recommendations,
    )


@router.get("/positions")
async def get_positions():
    """
    Get currently open positions.

    Returns list of all open positions with entry price, current price,
    unrealized P&L, and other position details.
    """
    state_path = Path("data/unified_trading/state.json")

    if not state_path.exists():
        return {"positions": [], "total_open": 0}

    try:
        with open(state_path) as f:
            state = json.load(f)

        positions = state.get("positions", {})
        position_list = []

        for symbol, pos_data in positions.items():
            if isinstance(pos_data, dict):
                position_list.append(
                    {
                        "symbol": symbol,
                        "side": pos_data.get("side", "LONG"),
                        "quantity": pos_data.get("quantity", 0),
                        "entry_price": pos_data.get("entry_price", 0),
                        "current_price": pos_data.get(
                            "current_price", pos_data.get("entry_price", 0)
                        ),
                        "unrealized_pnl": pos_data.get("unrealized_pnl", 0),
                        "unrealized_pnl_pct": pos_data.get("unrealized_pnl_pct", 0),
                        "entry_time": pos_data.get("entry_time", ""),
                        "stop_loss": pos_data.get("stop_loss", 0),
                        "take_profit": pos_data.get("take_profit", 0),
                    }
                )

        return {
            "positions": position_list,
            "total_open": len(position_list),
        }

    except (OSError, json.JSONDecodeError):
        return {"positions": [], "total_open": 0}


@router.get("/positions/detailed", response_model=DetailedPositionsResponse)
async def get_detailed_positions():
    """Return enriched position data for every open position."""
    state = _load_state()
    if not state:
        return DetailedPositionsResponse(positions=[], total_market_value=0, total_unrealized_pnl=0)

    current_balance = state.get("current_balance", 0) or 0
    raw_positions = state.get("positions", {}) or {}

    total_market_value = 0.0
    total_unrealized_pnl = 0.0
    entries = []

    for symbol, pos_data in raw_positions.items():
        if not isinstance(pos_data, dict):
            continue

        quantity = pos_data.get("quantity", 0) or 0
        entry_price = pos_data.get("entry_price") or 0
        current_price = pos_data.get("current_price") or entry_price or 0
        market_value = quantity * current_price
        unrealized_pnl = pos_data.get("unrealized_pnl")
        if unrealized_pnl is None:
            unrealized_pnl = (current_price - entry_price) * quantity

        unrealized_pct = 0.0
        if entry_price:
            unrealized_pct = ((current_price - entry_price) / entry_price) * 100

        total_market_value += market_value
        total_unrealized_pnl += unrealized_pnl

        entries.append(
            {
                "symbol": symbol,
                "side": pos_data.get("side", "LONG"),
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pct,
                "entry_time": pos_data.get("entry_time", ""),
                "stop_loss": pos_data.get("stop_loss"),
                "take_profit": pos_data.get("take_profit"),
            }
        )

    portfolio_value = (
        current_balance + total_market_value if current_balance + total_market_value else 0
    )
    detailed_positions = []
    for entry in entries:
        pct_of_portfolio = entry["market_value"] / portfolio_value if portfolio_value else 0
        detailed_positions.append(
            PositionDetail(
                symbol=entry["symbol"],
                side=entry["side"],
                quantity=entry["quantity"],
                entry_price=entry["entry_price"],
                current_price=entry["current_price"],
                market_value=entry["market_value"],
                unrealized_pnl=entry["unrealized_pnl"],
                unrealized_pnl_pct=entry["unrealized_pnl_pct"],
                pct_of_portfolio=pct_of_portfolio,
                entry_time=entry["entry_time"],
                stop_loss=entry["stop_loss"],
                take_profit=entry["take_profit"],
            )
        )

    return DetailedPositionsResponse(
        positions=detailed_positions,
        total_market_value=total_market_value,
        total_unrealized_pnl=total_unrealized_pnl,
    )


@router.get("/performance")
async def get_performance(days: int = Query(default=7, ge=1, le=365)):
    """
    Get performance metrics for the last N days.

    Returns daily P&L, win rate, trade count, and other metrics
    for the specified lookback period.
    """
    state_path = Path("data/unified_trading/state.json")
    equity_path = Path("data/unified_trading/equity.json")
    trades_path = Path("data/unified_trading/trades.json")

    try:
        # Get current state
        state_data = {}
        if state_path.exists():
            with open(state_path) as f:
                state_data = json.load(f)

        # Get recent trades
        recent_trades = []
        if trades_path.exists():
            with open(trades_path) as f:
                all_trades = json.load(f)
                # Get trades from last N days
                from datetime import datetime, timedelta

                cutoff_time = datetime.now() - timedelta(days=days)
                for trade in all_trades:
                    try:
                        trade_time = datetime.fromisoformat(trade.get("entry_time", ""))
                        if trade_time > cutoff_time:
                            recent_trades.append(trade)
                    except (ValueError, KeyError):
                        pass

        # Calculate metrics
        total_trades = len(recent_trades)
        winning_trades = len([t for t in recent_trades if (t.get("pnl", 0) or 0) > 0])
        losing_trades = len([t for t in recent_trades if (t.get("pnl", 0) or 0) < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(t.get("pnl", 0) or 0 for t in recent_trades)
        avg_win = (
            (
                sum(t.get("pnl", 0) or 0 for t in recent_trades if (t.get("pnl", 0) or 0) > 0)
                / winning_trades
            )
            if winning_trades > 0
            else 0
        )
        avg_loss = (
            (
                sum(t.get("pnl", 0) or 0 for t in recent_trades if (t.get("pnl", 0) or 0) < 0)
                / losing_trades
            )
            if losing_trades > 0
            else 0
        )

        return {
            "days": days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate / 100,  # Return as decimal (0.0 to 1.0)
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "current_balance": state_data.get("current_balance", 0),
            "initial_capital": state_data.get("initial_capital", 0),
        }

    except (OSError, json.JSONDecodeError) as e:
        return {
            "days": days,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "current_balance": 0,
            "initial_capital": 0,
            "error": str(e),
        }


@router.get("/analytics")
async def get_advanced_analytics(days: int = Query(default=30, ge=1, le=365)):
    """
    Get comprehensive trading analytics with professional metrics.

    Returns:
    - Expectancy (expected $ per trade)
    - Sharpe Ratio (risk-adjusted returns)
    - Sortino Ratio (downside risk-adjusted)
    - Calmar Ratio (return vs max drawdown)
    - Profit Factor (gross profit / gross loss)
    - Quality Score and Grade
    - By-symbol breakdown
    - Rolling metrics
    """
    from bot.analytics import calculate_all_metrics

    trades_path = Path("data/unified_trading/trades.json")
    equity_path = Path("data/unified_trading/equity.json")
    state_path = Path("data/unified_trading/state.json")

    # Also check paper trading paths
    alt_trades_paths = [
        Path("data/ml_paper_trading_enhanced/trades.json"),
        Path("data/ml_paper_trading/trades.json"),
    ]

    try:
        trades = []
        equity = []
        initial_capital = 30000.0

        # Load state for initial capital
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                initial_capital = state.get("initial_capital", 30000.0)

        # Load trades
        if trades_path.exists():
            with open(trades_path) as f:
                trades = json.load(f)

        # Try alternate paths if no trades found
        if not trades:
            for alt_path in alt_trades_paths:
                if alt_path.exists():
                    with open(alt_path) as f:
                        trades = json.load(f)
                    if trades:
                        break

        # Filter by days
        if trades and days < 365:
            from datetime import datetime, timedelta

            cutoff = datetime.now() - timedelta(days=days)
            filtered = []
            for t in trades:
                try:
                    ts = t.get("timestamp", t.get("entry_time", ""))
                    if ts:
                        trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if trade_time.replace(tzinfo=None) > cutoff:
                            filtered.append(t)
                except Exception:
                    filtered.append(t)  # Include if can't parse date
            trades = filtered

        # Load equity curve
        if equity_path.exists():
            with open(equity_path) as f:
                equity = json.load(f)

        # Calculate all metrics
        metrics = calculate_all_metrics(trades, equity, initial_capital)

        return {
            "days": days,
            "trades_analyzed": metrics.total_trades,
            # Core metrics
            "win_rate": round(metrics.win_rate * 100, 1),
            "profit_factor": round(metrics.profit_factor, 2),
            "expectancy": round(metrics.expectancy, 2),
            "expectancy_pct": round(metrics.expectancy_pct, 2),
            # Risk-adjusted returns
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "sortino_ratio": round(metrics.sortino_ratio, 2),
            "calmar_ratio": round(metrics.calmar_ratio, 2),
            # P&L
            "total_pnl": round(metrics.total_pnl, 2),
            "total_pnl_pct": round(metrics.total_pnl_pct, 2),
            "avg_win": round(metrics.avg_win, 2),
            "avg_loss": round(metrics.avg_loss, 2),
            "avg_trade": round(metrics.avg_trade, 2),
            # Drawdown
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
            "current_drawdown_pct": round(metrics.current_drawdown_pct, 2),
            # Streaks
            "max_win_streak": metrics.max_win_streak,
            "max_loss_streak": metrics.max_loss_streak,
            "current_streak": metrics.current_streak,
            "current_streak_type": metrics.current_streak_type,
            # R-Multiple
            "avg_r_multiple": round(metrics.avg_r_multiple, 2),
            "total_r": round(metrics.total_r, 2),
            # Rolling metrics
            "rolling_win_rate_10": round(metrics.rolling_win_rate_10 * 100, 1),
            "rolling_win_rate_20": round(metrics.rolling_win_rate_20 * 100, 1),
            # Quality
            "quality_score": round(metrics.quality_score, 1),
            "quality_grade": metrics.quality_grade,
            # By symbol
            "by_symbol": {
                sym: {
                    "trades": data["trades"],
                    "win_rate": round(data["win_rate"] * 100, 1),
                    "total_pnl": round(data["total_pnl"], 2),
                }
                for sym, data in metrics.by_symbol.items()
            },
            "calculated_at": metrics.calculated_at,
        }

    except Exception as e:
        return {
            "days": days,
            "trades_analyzed": 0,
            "error": str(e),
            "message": "Error calculating analytics",
        }


@router.get("/journal")
async def get_trade_journal():
    """
    Get trade journal insights and pattern analysis.

    Returns:
    - Best/worst trading hours
    - Best/worst trading days
    - Regime performance
    - Pattern insights
    - Recommendations for improvement
    """
    from bot.analytics import TradeJournal

    try:
        journal = TradeJournal()

        # Load trades and add to journal if not already there
        trades_paths = [
            Path("data/unified_trading/trades.json"),
            Path("data/ml_paper_trading_enhanced/trades.json"),
            Path("data/ml_paper_trading/trades.json"),
        ]

        for trades_path in trades_paths:
            if trades_path.exists():
                with open(trades_path) as f:
                    trades = json.load(f)

                for trade in trades:
                    if "pnl" in trade and trade.get("pnl") is not None:
                        # Check if trade not already in journal
                        ts = trade.get("timestamp", trade.get("entry_time", ""))
                        if not any(e.entry_time == ts for e in journal.entries):
                            journal.add_trade(trade)

        summary = journal.get_summary()

        return {
            "total_journal_entries": len(journal.entries),
            # Time analysis
            "best_trading_hours": summary.best_trading_hours,
            "worst_trading_hours": summary.worst_trading_hours,
            "best_trading_days": summary.best_trading_days,
            "worst_trading_days": summary.worst_trading_days,
            # Exit analysis
            "premature_exits": summary.premature_exits,
            "optimal_exits": summary.optimal_exits,
            "avg_profit_left_on_table": round(summary.avg_profit_left_on_table, 2),
            # Regime
            "best_regime": summary.best_regime,
            "worst_regime": summary.worst_regime,
            "regime_performance": summary.regime_performance,
            # Patterns
            "patterns": [
                {
                    "name": p.pattern_name,
                    "description": p.description,
                    "occurrences": p.occurrences,
                    "win_rate": round(p.win_rate * 100, 1),
                    "recommendation": p.recommendation,
                }
                for p in summary.patterns
            ],
            # Recommendations
            "top_lessons": summary.top_lessons,
            "recommendations": summary.recommendations,
            "calculated_at": summary.calculated_at,
        }

    except Exception as e:
        return {
            "total_journal_entries": 0,
            "error": str(e),
            "message": "Error generating journal summary",
        }


@router.get("/pnl-chart")
async def get_pnl_chart_data(limit: int = Query(default=100, le=1000)):
    """
    Get P&L chart data for visualization.

    Returns time-series data for:
    - Cumulative P&L
    - Per-trade P&L
    - Drawdown
    """
    trades_paths = [
        Path("data/unified_trading/trades.json"),
        Path("data/ml_paper_trading_enhanced/trades.json"),
    ]

    equity_path = Path("data/unified_trading/equity.json")

    try:
        trades = []
        for path in trades_paths:
            if path.exists():
                with open(path) as f:
                    trades = json.load(f)
                if trades:
                    break

        # Filter closed trades with P&L
        closed = [t for t in trades if "pnl" in t and t.get("pnl") is not None]
        closed = closed[-limit:]  # Last N trades

        # Build P&L series
        pnl_series = []
        cumulative_pnl = 0
        peak_pnl = 0

        for trade in closed:
            pnl = trade.get("pnl", 0) or 0
            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl

            pnl_series.append(
                {
                    "timestamp": trade.get("timestamp", trade.get("exit_time", "")),
                    "symbol": trade.get("symbol", "unknown"),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(trade.get("pnl_pct", 0) or 0, 2),
                    "cumulative_pnl": round(cumulative_pnl, 2),
                    "drawdown": round(drawdown, 2),
                }
            )

        # Get equity curve if available
        equity_curve = []
        if equity_path.exists():
            with open(equity_path) as f:
                equity_data = json.load(f)
                equity_curve = [
                    {
                        "timestamp": e.get("timestamp", ""),
                        "equity": round(e.get("total_equity", e.get("balance", 0)), 2),
                        "unrealized_pnl": round(e.get("unrealized_pnl", 0), 2),
                    }
                    for e in equity_data[-limit:]
                ]

        return {
            "trades_count": len(pnl_series),
            "pnl_series": pnl_series,
            "equity_curve": equity_curve,
            "summary": {
                "total_pnl": round(cumulative_pnl, 2),
                "max_drawdown": round(
                    peak_pnl - min((p["cumulative_pnl"] for p in pnl_series), default=0), 2
                )
                if pnl_series
                else 0,
                "winning_trades": len([p for p in pnl_series if p["pnl"] > 0]),
                "losing_trades": len([p for p in pnl_series if p["pnl"] < 0]),
            },
        }

    except Exception as e:
        return {
            "trades_count": 0,
            "pnl_series": [],
            "equity_curve": [],
            "error": str(e),
        }


@router.get("/daily-calendar")
async def get_daily_calendar(
    year: int = Query(default=None, description="Year (defaults to current)"),
    month: int = Query(default=None, ge=1, le=12, description="Month (defaults to current)"),
    daily_target_pct: float = Query(default=1.0, description="Daily profit target percentage"),
):
    """
    Get daily P&L calendar data with 1% (or custom) target tracking.

    Returns:
    - Calendar grid for the month
    - Each day shows: P&L, trades, target hit status
    - Monthly stats: days hit target, total P&L, best/worst day
    """
    from datetime import datetime, timedelta
    from collections import defaultdict

    now = datetime.now()
    year = year or now.year
    month = month or now.month

    trades_paths = [
        Path("data/unified_trading/trades.json"),
        Path("data/ml_paper_trading_enhanced/trades.json"),
    ]

    state_path = Path("data/unified_trading/state.json")

    try:
        # Load trades
        trades = []
        for path in trades_paths:
            if path.exists():
                with open(path) as f:
                    trades = json.load(f)
                if trades:
                    break

        # Load initial capital for percentage calculations
        initial_capital = 30000.0
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                initial_capital = state.get("initial_capital", 30000.0)

        # Group trades by day
        daily_data = defaultdict(
            lambda: {
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            }
        )

        for trade in trades:
            if "pnl" not in trade or trade.get("pnl") is None:
                continue

            try:
                ts = trade.get("timestamp", trade.get("exit_time", ""))
                if not ts:
                    continue
                trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                date_str = trade_date.isoformat()

                pnl = trade.get("pnl", 0) or 0
                pnl_pct = trade.get("pnl_pct", 0) or 0

                daily_data[date_str]["pnl"] += pnl
                daily_data[date_str]["pnl_pct"] += pnl_pct
                daily_data[date_str]["trades"] += 1
                if pnl > 0:
                    daily_data[date_str]["wins"] += 1
                elif pnl < 0:
                    daily_data[date_str]["losses"] += 1
            except Exception:
                continue

        # Build calendar grid for the month
        import calendar as cal

        first_day = datetime(year, month, 1)
        days_in_month = cal.monthrange(year, month)[1]
        first_weekday = first_day.weekday()  # Monday = 0

        calendar_days = []
        target_hit_days = 0
        profit_days = 0
        loss_days = 0
        best_day = {"date": "", "pnl_pct": float("-inf")}
        worst_day = {"date": "", "pnl_pct": float("inf")}
        monthly_pnl = 0.0
        monthly_pnl_pct = 0.0
        trading_days = 0

        for day in range(1, days_in_month + 1):
            date = datetime(year, month, day)
            date_str = date.strftime("%Y-%m-%d")
            weekday = date.weekday()

            day_data = daily_data.get(date_str, {})
            pnl = day_data.get("pnl", 0)
            pnl_pct = day_data.get("pnl_pct", 0)
            trades_count = day_data.get("trades", 0)
            wins = day_data.get("wins", 0)
            losses = day_data.get("losses", 0)

            target_hit = pnl_pct >= daily_target_pct
            is_today = date.date() == now.date()
            is_future = date.date() > now.date()
            is_weekend = weekday >= 5

            if trades_count > 0:
                trading_days += 1
                monthly_pnl += pnl
                monthly_pnl_pct += pnl_pct

                if target_hit:
                    target_hit_days += 1
                if pnl > 0:
                    profit_days += 1
                elif pnl < 0:
                    loss_days += 1

                if pnl_pct > best_day["pnl_pct"]:
                    best_day = {"date": date_str, "pnl_pct": pnl_pct, "pnl": pnl}
                if pnl_pct < worst_day["pnl_pct"]:
                    worst_day = {"date": date_str, "pnl_pct": pnl_pct, "pnl": pnl}

            # Determine status
            if is_future:
                status = "future"
            elif trades_count == 0:
                status = "no_trades"
            elif target_hit:
                status = "target_hit"
            elif pnl > 0:
                status = "profit"
            elif pnl < 0:
                status = "loss"
            else:
                status = "breakeven"

            calendar_days.append(
                {
                    "day": day,
                    "date": date_str,
                    "weekday": weekday,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "trades": trades_count,
                    "wins": wins,
                    "losses": losses,
                    "target_hit": target_hit,
                    "target_progress": round((pnl_pct / daily_target_pct) * 100, 1)
                    if daily_target_pct > 0
                    else 0,
                    "status": status,
                    "is_today": is_today,
                    "is_weekend": is_weekend,
                }
            )

        # Fix infinity values
        if best_day["pnl_pct"] == float("-inf"):
            best_day = {"date": "", "pnl_pct": 0, "pnl": 0}
        if worst_day["pnl_pct"] == float("inf"):
            worst_day = {"date": "", "pnl_pct": 0, "pnl": 0}

        return {
            "year": year,
            "month": month,
            "month_name": cal.month_name[month],
            "first_weekday": first_weekday,
            "days_in_month": days_in_month,
            "daily_target_pct": daily_target_pct,
            "calendar": calendar_days,
            "stats": {
                "trading_days": trading_days,
                "target_hit_days": target_hit_days,
                "hit_rate": round(
                    (target_hit_days / trading_days * 100) if trading_days > 0 else 0, 1
                ),
                "profit_days": profit_days,
                "loss_days": loss_days,
                "monthly_pnl": round(monthly_pnl, 2),
                "monthly_pnl_pct": round(monthly_pnl_pct, 2),
                "best_day": best_day,
                "worst_day": worst_day,
            },
            "today": {
                "date": now.strftime("%Y-%m-%d"),
                "pnl": daily_data.get(now.strftime("%Y-%m-%d"), {}).get("pnl", 0),
                "pnl_pct": daily_data.get(now.strftime("%Y-%m-%d"), {}).get("pnl_pct", 0),
                "trades": daily_data.get(now.strftime("%Y-%m-%d"), {}).get("trades", 0),
                "target_hit": daily_data.get(now.strftime("%Y-%m-%d"), {}).get("pnl_pct", 0)
                >= daily_target_pct,
            },
        }

    except Exception as e:
        return {
            "year": year,
            "month": month,
            "error": str(e),
            "calendar": [],
            "stats": {},
        }


@router.get("/slippage-analysis")
async def get_slippage_analysis():
    """
    Analyze historical slippage from execution log.

    Returns average slippage, worst slippage, and per-symbol breakdown.
    Critical for understanding true trading costs.
    """
    logs = _load_execution_log()

    if not logs:
        return {
            "total_trades": 0,
            "avg_slippage_pct": 0,
            "max_slippage_pct": 0,
            "total_slippage_cost": 0,
            "by_symbol": {},
            "message": "No execution data available yet",
        }

    try:
        slippages = [l.get("slippage_pct", 0) for l in logs if l.get("slippage_pct") is not None]
        by_symbol = {}

        for log in logs:
            symbol = log.get("symbol", "unknown")
            if symbol not in by_symbol:
                by_symbol[symbol] = {"trades": 0, "total_slippage": 0, "slippages": []}
            by_symbol[symbol]["trades"] += 1
            slip = log.get("slippage_pct", 0) or 0
            by_symbol[symbol]["total_slippage"] += slip
            by_symbol[symbol]["slippages"].append(slip)

        # Calculate averages per symbol
        for symbol in by_symbol:
            data = by_symbol[symbol]
            data["avg_slippage_pct"] = (
                data["total_slippage"] / data["trades"] if data["trades"] > 0 else 0
            )
            data["max_slippage_pct"] = max(data["slippages"]) if data["slippages"] else 0
            del data["slippages"]
            del data["total_slippage"]

        return {
            "total_trades": len(logs),
            "avg_slippage_pct": sum(slippages) / len(slippages) if slippages else 0,
            "max_slippage_pct": max(slippages) if slippages else 0,
            "total_slippage_cost": sum(l.get("slippage_cost", 0) or 0 for l in logs),
            "by_symbol": by_symbol,
        }

    except (OSError, json.JSONDecodeError):
        return {
            "total_trades": 0,
            "avg_slippage_pct": 0,
            "max_slippage_pct": 0,
            "total_slippage_cost": 0,
            "by_symbol": {},
        }


# =============================================================================
# Position Management Endpoints
# =============================================================================


class ClosePositionRequest(BaseModel):
    """Request to close a position."""

    symbol: str
    reason: Optional[str] = "Manual closure"


class ClosePositionResponse(BaseModel):
    """Response for position closure."""

    success: bool
    symbol: str
    closed_qty: float
    close_price: float
    pnl: float
    message: str


@router.post("/close-position", response_model=ClosePositionResponse)
async def close_position(request: ClosePositionRequest):
    """
    Manually close an open position.

    This endpoint allows manual liquidation of positions during emergency stops
    or for manual portfolio management.
    """
    import os
    from datetime import datetime

    state_path = _get_state_path()

    if not state_path.exists():
        raise HTTPException(status_code=404, detail="State file not found")

    try:
        # Load current state
        with open(state_path) as f:
            state = json.load(f)

        positions = state.get("positions", {})

        if request.symbol not in positions:
            raise HTTPException(
                status_code=404, detail=f"No open position found for {request.symbol}"
            )

        position = positions[request.symbol]
        qty = position.get("quantity", position.get("qty", 0))
        entry_price = position.get("entry_price", 0)
        side = position.get("side", "LONG").upper()

        if qty == 0:
            raise HTTPException(status_code=400, detail=f"Position has zero quantity")

        # Get current market price (use entry price as fallback for emergency closures)
        # In production, this should fetch from exchange
        close_price = entry_price  # Simplified for emergency closure

        # Calculate P&L
        if side == "LONG":
            pnl = (close_price - entry_price) * qty
        else:
            pnl = (entry_price - close_price) * qty

        # Update state
        del positions[request.symbol]
        state["positions"] = positions
        state["current_balance"] = state.get("current_balance", 0) + pnl
        state["total_pnl"] = state.get("total_pnl", 0) + pnl
        state["daily_pnl"] = state.get("daily_pnl", 0) + pnl
        state["timestamp"] = datetime.utcnow().isoformat()

        # Log closure
        closure_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": request.symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "close_price": close_price,
            "pnl": pnl,
            "reason": request.reason,
            "type": "manual_closure",
        }

        # Append to trade history
        if "trade_history" not in state:
            state["trade_history"] = []
        state["trade_history"].append(closure_log)

        # Save updated state
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        return ClosePositionResponse(
            success=True,
            symbol=request.symbol,
            closed_qty=qty,
            close_price=close_price,
            pnl=pnl,
            message=f"Position closed successfully: {side} {qty} @ ${close_price:.2f}, P&L: ${pnl:.2f}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing position: {str(e)}")


# =============================================================================
# Backtest Endpoints
# =============================================================================


class BacktestRequest(BaseModel):
    """Request model for running a backtest."""

    symbol: str = "BTC/USDT"
    days: int = 90
    model_type: str = "gradient_boosting"
    initial_balance: float = 10000.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    confidence_threshold: float = 0.6


class BacktestResponse(BaseModel):
    """Response model for backtest results."""

    success: bool
    symbol: str
    days: int
    model_type: str
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_trade: float
    expectancy: float
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    error: Optional[str] = None


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run an ML backtest on historical data.

    Tests the specified ML model on historical price data and returns
    comprehensive performance metrics including:
    - Win rate, profit factor, expectancy
    - Sharpe ratio, Sortino ratio
    - Max drawdown
    - Trade history and equity curve
    """
    try:
        from bot.ml_backtester import MLBacktester

        backtester = MLBacktester(
            model_type=request.model_type,
            initial_balance=request.initial_balance,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            confidence_threshold=request.confidence_threshold,
        )

        metrics = backtester.run(request.symbol, request.days)

        return BacktestResponse(
            success=True,
            symbol=request.symbol,
            days=request.days,
            model_type=request.model_type,
            initial_balance=metrics.initial_balance,
            final_balance=metrics.final_balance,
            total_return=metrics.total_return,
            total_return_pct=metrics.total_return_pct,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown_pct=metrics.max_drawdown_pct,
            avg_trade=metrics.avg_trade,
            expectancy=metrics.expectancy,
            trades=[t.to_dict() for t in metrics.trades[-50:]],  # Last 50 trades
            equity_curve=metrics.equity_curve[-200:],  # Last 200 points
        )

    except Exception as e:
        return BacktestResponse(
            success=False,
            symbol=request.symbol,
            days=request.days,
            model_type=request.model_type,
            initial_balance=request.initial_balance,
            final_balance=request.initial_balance,
            total_return=0,
            total_return_pct=0,
            total_trades=0,
            win_rate=0,
            profit_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown_pct=0,
            avg_trade=0,
            expectancy=0,
            trades=[],
            equity_curve=[],
            error=str(e),
        )


@router.get("/backtest/quick")
async def quick_backtest(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
    days: int = Query(default=30, description="Number of days to backtest"),
) -> Dict[str, Any]:
    """
    Run a quick backtest with default settings.

    Returns summary metrics without full trade history.
    """
    try:
        from bot.ml_backtester import MLBacktester

        backtester = MLBacktester(
            model_type="gradient_boosting",
            initial_balance=10000.0,
        )

        metrics = backtester.run(symbol, days)

        return {
            "success": True,
            "symbol": symbol,
            "days": days,
            "summary": {
                "initial_balance": metrics.initial_balance,
                "final_balance": round(metrics.final_balance, 2),
                "total_return_pct": round(metrics.total_return_pct, 2),
                "total_trades": metrics.total_trades,
                "win_rate": round(metrics.win_rate * 100, 1),
                "profit_factor": round(metrics.profit_factor, 2),
                "sharpe_ratio": round(metrics.sharpe_ratio, 2),
                "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
                "expectancy": round(metrics.expectancy, 2),
            },
        }

    except Exception as e:
        return {
            "success": False,
            "symbol": symbol,
            "days": days,
            "error": str(e),
        }


@router.get("/models/status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get status of all ML models including ensemble metrics.

    Returns model accuracies, weights, and health status.
    """
    try:
        from bot.ml.ensemble_predictor import create_ensemble_predictor
        from pathlib import Path
        import json

        model_dir = Path("data/models")
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]

        models_status = {}

        for symbol in symbols:
            symbol_clean = symbol.replace("/", "_")

            # Try to load ensemble
            predictor = create_ensemble_predictor(symbol)

            if predictor:
                models_status[symbol] = {
                    "loaded": True,
                    "ml_models": len(predictor.models),
                    "dl_models": len(predictor.dl_models),
                    "total_models": len(predictor.models) + len(predictor.dl_models),
                    "weights": {k: round(v, 4) for k, v in predictor.model_weights.items()},
                    "accuracies": {k: round(v, 4) for k, v in predictor.model_accuracies.items()},
                }
            else:
                models_status[symbol] = {"loaded": False, "ml_models": 0, "dl_models": 0}

            # Check for tuned models
            tuned_meta = model_dir / f"{symbol_clean}_tuned_meta.json"
            if tuned_meta.exists():
                with open(tuned_meta) as f:
                    tuned = json.load(f)
                    models_status[symbol]["tuned"] = True
                    models_status[symbol]["tuned_accuracy"] = tuned.get("results", {})

            # Check for feature selection
            feat_meta = model_dir / f"{symbol_clean}_selected_features.json"
            if feat_meta.exists():
                with open(feat_meta) as f:
                    feat = json.load(f)
                    models_status[symbol]["feature_selection"] = {
                        "n_features": len(feat.get("selected_features", [])),
                        "accuracy": feat.get("accuracy", 0),
                    }

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "models": models_status,
            "summary": {
                "total_symbols": len([s for s in models_status.values() if s.get("loaded")]),
                "total_models": sum(s.get("total_models", 0) for s in models_status.values()),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/models/predictions")
async def get_model_predictions(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
) -> Dict[str, Any]:
    """
    Get current model predictions for a symbol.

    Returns predictions from all models in the ensemble.
    """
    try:
        from bot.ml.ensemble_predictor import create_ensemble_predictor
        from bot.ml.feature_engineer import FeatureEngineer
        import numpy as np

        predictor = create_ensemble_predictor(symbol)

        if not predictor:
            return {"success": False, "error": f"No models loaded for {symbol}"}

        # Get current features (simplified - in production would use live data)
        # For now return model info without prediction
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "ensemble_info": {
                "ml_models": list(predictor.models.keys()),
                "dl_models": list(predictor.dl_models.keys()),
                "weights": predictor.model_weights,
                "voting_strategy": predictor.voting_strategy,
            },
            "note": "Live predictions require market data feed",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/models/onchain")
async def get_onchain_metrics(
    symbol: str = Query(default="BTC/USDT", description="Trading symbol"),
) -> Dict[str, Any]:
    """
    Get on-chain metrics for a symbol.

    Returns funding rates, open interest, and smart money signals.
    """
    try:
        from bot.ml.onchain_data import get_onchain_signal

        metrics = await get_onchain_signal(symbol)

        return {
            "success": True,
            "symbol": symbol,
            "metrics": metrics,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
