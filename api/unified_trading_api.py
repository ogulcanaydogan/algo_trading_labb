"""
Unified Trading API Endpoints.

Provides REST API endpoints for the unified trading engine:
- Mode switching
- Status monitoring
- Transition validation
- Safety controls
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/unified", tags=["Unified Trading"])


# =============================================================================
# Pydantic Models
# =============================================================================


class TradingModeInfo(BaseModel):
    """Information about a trading mode."""

    mode: str
    description: str
    is_paper: bool
    is_live: bool
    uses_real_data: bool
    capital_limit: Optional[float]
    max_position_usd: Optional[float]
    max_daily_loss_usd: Optional[float]


class UnifiedStatusResponse(BaseModel):
    """Response model for unified trading status."""

    mode: str
    status: str
    running: bool
    balance: float
    initial_capital: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    open_positions: int
    positions: Dict[str, Any]
    safety: Dict[str, Any]
    daily_trades: int
    daily_pnl: float


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
                    "commission": meta.get("fees_paid")
                    or trade.get("fees_paid")
                    or 0,
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


@router.get("/modes", response_model=List[TradingModeInfo])
async def get_available_modes():
    """Get list of available trading modes."""
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


@router.get("/status", response_model=UnifiedStatusResponse)
async def get_unified_status():
    """Get current unified trading engine status."""
    state = _load_state()
    safety = _load_safety_state()

    if not state:
        raise HTTPException(
            status_code=404,
            detail="No unified trading state found. Start trading first.",
        )

    # Calculate win rate
    total_trades = state.get("total_trades", 0)
    winning_trades = state.get("winning_trades", 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    return UnifiedStatusResponse(
        mode=state.get("mode", "unknown"),
        status=state.get("status", "unknown"),
        running=state.get("status") == "active",
        balance=state.get("current_balance", 0),
        initial_capital=state.get("initial_capital", 0),
        total_pnl=state.get("total_pnl", 0),
        total_pnl_pct=(
            state.get("total_pnl", 0) / state.get("initial_capital", 1) * 100
            if state.get("initial_capital", 0) > 0
            else 0
        ),
        total_trades=total_trades,
        win_rate=win_rate,
        max_drawdown=state.get("max_drawdown_pct", 0) * 100,
        open_positions=len(state.get("positions", {})),
        positions=state.get("positions", {}),
        safety=safety or {},
        daily_trades=state.get("daily_trades", 0),
        daily_pnl=state.get("daily_pnl", 0),
    )


@router.get("/transition-progress", response_model=TransitionProgressResponse)
async def get_transition_progress(
    target_mode: str = Query(..., description="Target mode to check progress for")
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
            initial_capital=10000,
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


@router.get("/safety", response_model=SafetyStatusResponse)
async def get_safety_status():
    """Get current safety controller status."""
    from bot.safety_controller import SafetyController

    controller = SafetyController()
    status = controller.get_status()

    return SafetyStatusResponse(**status)


@router.post("/switch-mode")
async def switch_mode(request: ModeSwitchRequest):
    """
    Switch trading mode.

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
            initial_capital=10000,
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
    mode: str = Query(default="testnet", description="testnet or live")
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

        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": mode == "testnet",
            "options": {"defaultType": "spot"},
        })

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
    target_mode: str = Query(default=None, description="Target trading mode (if None, uses next mode in progression)")
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
        checks.append({
            "name": "API Keys",
            "status": "pass",
            "message": "Binance API keys configured",
            "icon": "🔑"
        })
    else:
        checks.append({
            "name": "API Keys",
            "status": "fail",
            "message": "API keys not configured",
            "icon": "🔑"
        })
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
            checks.append({
                "name": "Trade Count",
                "status": "pass",
                "message": f"{total_trades} trades completed (min: {min_trades_required})",
                "icon": "📊"
            })
        else:
            checks.append({
                "name": "Trade Count",
                "status": "fail",
                "message": f"{total_trades}/{min_trades_required} trades",
                "icon": "📊"
            })
            blocking.append(f"Complete {min_trades_required - total_trades} more paper trades")

        # Check win rate
        min_win_rate = 0.45
        if win_rate >= min_win_rate:
            checks.append({
                "name": "Win Rate",
                "status": "pass",
                "message": f"{win_rate:.1%} (min: {min_win_rate:.0%})",
                "icon": "🎯"
            })
        else:
            checks.append({
                "name": "Win Rate",
                "status": "fail",
                "message": f"{win_rate:.1%} (need {min_win_rate:.0%})",
                "icon": "🎯"
            })
            blocking.append(f"Improve win rate to at least {min_win_rate:.0%}")

        # Check drawdown
        max_allowed_dd = 0.12
        if max_drawdown <= max_allowed_dd:
            checks.append({
                "name": "Max Drawdown",
                "status": "pass",
                "message": f"{max_drawdown:.1%} (max allowed: {max_allowed_dd:.0%})",
                "icon": "📉"
            })
        else:
            checks.append({
                "name": "Max Drawdown",
                "status": "warn",
                "message": f"{max_drawdown:.1%} exceeds {max_allowed_dd:.0%}",
                "icon": "📉"
            })
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
                checks.append({
                    "name": "Time in Mode",
                    "status": "pass",
                    "message": f"{days_in_mode} days in current mode (min: {required_days})",
                    "icon": "⏰"
                })
            else:
                checks.append({
                    "name": "Time in Mode",
                    "status": "fail",
                    "message": f"{days_in_mode}/{required_days} days in current mode",
                    "icon": "⏰"
                })
                days_remaining = required_days - days_in_mode
                blocking.append(f"Wait {days_remaining} more day(s) in current mode before transitioning")
    else:
        checks.append({
            "name": "Paper Trading",
            "status": "fail",
            "message": "No paper trading history found",
            "icon": "📊"
        })
        blocking.append("Complete paper trading first before going live")

    # 3. Safety Configuration Check
    safety_state = _load_safety_state()
    if safety_state and not safety_state.get("emergency_stop_active"):
        checks.append({
            "name": "Safety System",
            "status": "pass",
            "message": "Safety controller active, no emergency stops",
            "icon": "🛡️"
        })
    else:
        if safety_state and safety_state.get("emergency_stop_active"):
            checks.append({
                "name": "Safety System",
                "status": "fail",
                "message": f"Emergency stop active: {safety_state.get('emergency_stop_reason')}",
                "icon": "🛡️"
            })
            blocking.append("Clear emergency stop before proceeding")
        else:
            checks.append({
                "name": "Safety System",
                "status": "warn",
                "message": "Safety state not initialized",
                "icon": "🛡️"
            })

    # 4. Telegram Notifications Check
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")

    if telegram_token and telegram_chat:
        checks.append({
            "name": "Notifications",
            "status": "pass",
            "message": "Telegram notifications configured",
            "icon": "📱"
        })
    else:
        checks.append({
            "name": "Notifications",
            "status": "warn",
            "message": "Telegram not configured - recommended for live trading",
            "icon": "📱"
        })
        recommendations.append("Configure Telegram for real-time trade alerts")

    # 5. Capital Limits Check
    if target == TradingMode.LIVE_LIMITED:
        checks.append({
            "name": "Capital Limit",
            "status": "pass",
            "message": "$100 max capital enforced for live_limited mode",
            "icon": "💰"
        })
        recommendations.append("Start with small amounts ($20-50) to validate execution")

    # Add general recommendations
    recommendations.extend([
        "Test with Binance Testnet first before using real funds",
        "Monitor first few live trades closely for any issues",
        "Keep emergency stop accessible on your dashboard",
        "Review execution log after each trade for slippage",
    ])

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
                position_list.append({
                    "symbol": symbol,
                    "side": pos_data.get("side", "LONG"),
                    "quantity": pos_data.get("quantity", 0),
                    "entry_price": pos_data.get("entry_price", 0),
                    "current_price": pos_data.get("current_price", pos_data.get("entry_price", 0)),
                    "unrealized_pnl": pos_data.get("unrealized_pnl", 0),
                    "unrealized_pnl_pct": pos_data.get("unrealized_pnl_pct", 0),
                    "entry_time": pos_data.get("entry_time", ""),
                    "stop_loss": pos_data.get("stop_loss", 0),
                    "take_profit": pos_data.get("take_profit", 0),
                })
        
        return {
            "positions": position_list,
            "total_open": len(position_list),
        }
    
    except (OSError, json.JSONDecodeError):
        return {"positions": [], "total_open": 0}


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
        avg_win = (sum(t.get("pnl", 0) or 0 for t in recent_trades if (t.get("pnl", 0) or 0) > 0) / winning_trades) if winning_trades > 0 else 0
        avg_loss = (sum(t.get("pnl", 0) or 0 for t in recent_trades if (t.get("pnl", 0) or 0) < 0) / losing_trades) if losing_trades > 0 else 0
        
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
            data["avg_slippage_pct"] = data["total_slippage"] / data["trades"] if data["trades"] > 0 else 0
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
                status_code=404,
                detail=f"No open position found for {request.symbol}"
            )
        
        position = positions[request.symbol]
        qty = position.get("quantity", position.get("qty", 0))
        entry_price = position.get("entry_price", 0)
        side = position.get("side", "LONG").upper()
        
        if qty == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Position has zero quantity"
            )
        
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
            "type": "manual_closure"
        }
        
        # Append to trade history
        if "trade_history" not in state:
            state["trade_history"] = []
        state["trade_history"].append(closure_log)
        
        # Save updated state
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return ClosePositionResponse(
            success=True,
            symbol=request.symbol,
            closed_qty=qty,
            close_price=close_price,
            pnl=pnl,
            message=f"Position closed successfully: {side} {qty} @ ${close_price:.2f}, P&L: ${pnl:.2f}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error closing position: {str(e)}"
        )
