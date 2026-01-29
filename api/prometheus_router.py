"""
Prometheus Metrics API Router.

Provides /metrics endpoint for Prometheus scraping with:
- Trading metrics (positions, P&L, trades)
- API metrics (requests, latency)
- WebSocket metrics (connections, messages)
- ML model metrics (predictions, accuracy)
- Risk metrics (VaR, Sharpe ratio)

Usage:
    from api.prometheus_router import router as prometheus_router
    app.include_router(prometheus_router, tags=["metrics"])
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from bot.core.prometheus import (
    trading_registry,
    update_portfolio_metrics,
    portfolio_value,
    positions_count,
    circuit_breaker_status,
    exchange_connection_status,
    websocket_connections,
)

router = APIRouter()

# Data directory for reading trading state
DATA_DIR = Path("data/unified_trading")


def _read_trading_state() -> Dict[str, Any]:
    """Read current trading state from file."""
    state_file = DATA_DIR / "state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}


def _update_trading_metrics():
    """Update trading metrics from current state."""
    state = _read_trading_state()
    if not state:
        return

    # Overall portfolio
    initial = state.get("initial_capital", 30000)
    balance = state.get("current_balance", initial)
    positions = state.get("positions", {})
    total_pnl = state.get("total_pnl", 0)
    max_dd = state.get("max_drawdown_pct", 0)

    # Calculate position values
    crypto_value = 0
    stock_value = 0
    commodity_value = 0
    crypto_positions = []
    stock_positions = []

    for symbol, pos in positions.items():
        pos_value = abs(pos.get("quantity", 0) * pos.get("current_price", 0))
        if "/USDT" in symbol:
            crypto_value += pos_value
            crypto_positions.append(symbol)
        elif "/USD" in symbol:
            stock_value += pos_value
            stock_positions.append(symbol)
        else:
            commodity_value += pos_value

    # Update portfolio metrics
    total_value = balance
    pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

    update_portfolio_metrics(
        market="total",
        value=total_value,
        pnl=total_pnl,
        pnl_pct=pnl_pct,
        pos_count=len(positions),
        drawdown=max_dd * 100,
    )

    # Per-market metrics
    update_portfolio_metrics(
        market="crypto",
        value=crypto_value,
        pnl=sum(pos.get("unrealized_pnl", 0) for s, pos in positions.items() if "/USDT" in s),
        pnl_pct=0,
        pos_count=len(crypto_positions),
    )

    update_portfolio_metrics(
        market="stock",
        value=stock_value,
        pnl=sum(pos.get("unrealized_pnl", 0) for s, pos in positions.items() if "/USD" in s and "/USDT" not in s),
        pnl_pct=0,
        pos_count=len(stock_positions),
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics(request: Request):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format.

    Example output:
        # HELP trades_total Total number of trades executed
        # TYPE trades_total counter
        trades_total{symbol="BTC/USDT",side="buy"} 5
        trades_total{symbol="BTC/USDT",side="sell"} 3

        # HELP portfolio_value_dollars Total portfolio value in dollars
        # TYPE portfolio_value_dollars gauge
        portfolio_value_dollars{market="total"} 29526.64
    """
    # Update metrics from current state
    _update_trading_metrics()

    # Export all registered metrics
    metrics_output = trading_registry.export()

    # Add process metrics
    import os
    import psutil

    process = psutil.Process(os.getpid())
    metrics_output += "\n# HELP process_cpu_percent Process CPU usage percentage\n"
    metrics_output += "# TYPE process_cpu_percent gauge\n"
    metrics_output += f"process_cpu_percent {process.cpu_percent()}\n"

    metrics_output += "\n# HELP process_memory_bytes Process memory usage in bytes\n"
    metrics_output += "# TYPE process_memory_bytes gauge\n"
    metrics_output += f"process_memory_bytes {process.memory_info().rss}\n"

    metrics_output += "\n# HELP process_open_fds Number of open file descriptors\n"
    metrics_output += "# TYPE process_open_fds gauge\n"
    try:
        metrics_output += f"process_open_fds {process.num_fds()}\n"
    except AttributeError:
        # Windows doesn't have num_fds
        metrics_output += "process_open_fds 0\n"

    metrics_output += "\n# HELP process_threads Number of threads\n"
    metrics_output += "# TYPE process_threads gauge\n"
    metrics_output += f"process_threads {process.num_threads()}\n"

    metrics_output += "\n# HELP process_uptime_seconds Process uptime in seconds\n"
    metrics_output += "# TYPE process_uptime_seconds gauge\n"
    metrics_output += f"process_uptime_seconds {time.time() - process.create_time()}\n"

    return Response(
        content=metrics_output,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/json")
async def get_metrics_json():
    """
    Get metrics in JSON format for easier debugging.

    Returns:
        Dict with all metrics organized by category
    """
    _update_trading_metrics()
    state = _read_trading_state()

    import os
    import psutil
    process = psutil.Process(os.getpid())

    return {
        "timestamp": time.time(),
        "portfolio": {
            "initial_capital": state.get("initial_capital", 0),
            "current_balance": state.get("current_balance", 0),
            "total_pnl": state.get("total_pnl", 0),
            "positions_count": len(state.get("positions", {})),
            "max_drawdown_pct": state.get("max_drawdown_pct", 0),
        },
        "trading": {
            "total_trades": state.get("total_trades", 0),
            "winning_trades": state.get("winning_trades", 0),
            "losing_trades": state.get("losing_trades", 0),
            "win_rate": (
                state.get("winning_trades", 0) / state.get("total_trades", 1)
                if state.get("total_trades", 0) > 0
                else 0
            ),
        },
        "process": {
            "cpu_percent": process.cpu_percent(),
            "memory_bytes": process.memory_info().rss,
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "threads": process.num_threads(),
            "uptime_seconds": time.time() - process.create_time(),
        },
        "status": state.get("status", "unknown"),
        "mode": state.get("mode", "unknown"),
    }


@router.get("/metrics/health")
async def get_metrics_health():
    """
    Health check endpoint for monitoring.

    Returns simplified health status suitable for load balancers.
    """
    state = _read_trading_state()

    # Check if state is fresh (updated within last 5 minutes)
    from datetime import datetime
    timestamp_str = state.get("timestamp", "")
    is_fresh = False
    if timestamp_str:
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age_seconds = (datetime.now(ts.tzinfo) - ts).total_seconds()
            is_fresh = age_seconds < 300  # 5 minutes
        except (ValueError, TypeError):
            pass

    status = state.get("status", "unknown")
    healthy = status == "active" and is_fresh

    return {
        "healthy": healthy,
        "status": status,
        "mode": state.get("mode", "unknown"),
        "positions": len(state.get("positions", {})),
        "timestamp": timestamp_str,
    }
