"""
Multi-Asset Portfolio REST API endpoints.
Exposes portfolio management and monitoring capabilities.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime
from bot.portfolio_config import PortfolioLoader, RebalanceStrategy
from bot.portfolio_manager import PortfolioManager
from bot.multi_asset_signals import MultiAssetSignalAggregator

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# Global portfolio manager (would be initialized from app startup)
portfolio_manager: Optional[PortfolioManager] = None
signal_aggregator: Optional[MultiAssetSignalAggregator] = None


def init_portfolio(config_path: str):
    """Initialize portfolio manager from config file."""
    global portfolio_manager, signal_aggregator
    portfolio = PortfolioLoader.load(config_path)
    portfolio_manager = PortfolioManager(portfolio)
    signal_aggregator = MultiAssetSignalAggregator()


@router.get("/summary")
async def get_portfolio_summary():
    """Get current portfolio summary."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    return portfolio_manager.get_portfolio_summary()


@router.get("/allocations")
async def get_allocations():
    """Get current asset allocations."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    target = {a.symbol: a.allocation_pct for a in portfolio_manager.portfolio.assets}
    current = portfolio_manager.state.allocation_pcts

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "target_allocations": target,
        "current_allocations": current,
        "drifts": {
            symbol: current.get(symbol, 0) - target.get(symbol, 0) for symbol in target.keys()
        },
    }


@router.get("/rebalancing-status")
async def get_rebalancing_status():
    """Check if portfolio needs rebalancing."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    needs_rebalance, assets = portfolio_manager.needs_rebalancing()

    if needs_rebalance:
        trades = portfolio_manager.calculate_rebalancing_trades()
    else:
        trades = {}

    return {
        "needs_rebalancing": needs_rebalance,
        "reason": f"Strategy: {portfolio_manager.portfolio.rebalance_strategy.value}",
        "assets_affected": assets,
        "recommended_trades": trades,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/rebalance")
async def rebalance_portfolio():
    """Trigger portfolio rebalancing."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    trades = portfolio_manager.calculate_rebalancing_trades()
    portfolio_manager.record_rebalancing(trades)

    return {
        "status": "rebalancing_initiated",
        "trades": trades,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/positions")
async def get_positions(symbol: Optional[str] = None):
    """Get current positions."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    positions = portfolio_manager.state.positions

    if symbol:
        if symbol not in positions:
            raise HTTPException(status_code=404, detail=f"Position {symbol} not found")
        positions = {symbol: positions[symbol]}

    return {
        "positions": {
            sym: {
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "position_value": pos.position_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            }
            for sym, pos in positions.items()
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/diversification")
async def get_diversification():
    """Get portfolio diversification metrics."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    hhi = portfolio_manager.get_portfolio_diversification()

    return {
        "herfindahl_hirschman_index": hhi,
        "diversification_level": "excellent"
        if hhi < 1500
        else "good"
        if hhi < 2500
        else "fair"
        if hhi < 5000
        else "poor",
        "target_hhi": portfolio_manager.portfolio.diversification_target,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/signals")
async def get_portfolio_signals():
    """Get portfolio-level signals and sentiment."""
    if not signal_aggregator:
        raise HTTPException(status_code=503, detail="Signal aggregator not initialized")

    return signal_aggregator.get_signals_summary()


@router.get("/assets")
async def list_assets():
    """List all assets in portfolio."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    return {
        "assets": [
            {
                "symbol": a.symbol,
                "type": a.asset_type.value,
                "data_source": a.data_source,
                "target_allocation": a.allocation_pct,
                "risk_limit": a.risk_limit_pct,
                "active": a.active,
            }
            for a in portfolio_manager.portfolio.assets
        ],
        "total_assets": len(portfolio_manager.portfolio.assets),
    }


@router.get("/performance")
async def get_performance(period_days: int = Query(30, ge=1, le=365)):
    """Get portfolio performance metrics."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    state = portfolio_manager.state
    total_pnl = state.total_unrealized_pnl
    total_pnl_pct = (total_pnl / portfolio_manager.portfolio.total_capital) * 100

    return {
        "period_days": period_days,
        "total_return_pct": total_pnl_pct,
        "total_return_usd": total_pnl,
        "starting_capital": portfolio_manager.portfolio.total_capital,
        "current_equity": state.total_equity,
        "avg_return_pct_daily": total_pnl_pct / period_days,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/update-prices")
async def update_prices(prices: Dict[str, float]):
    """Update current market prices for all positions."""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio not initialized")

    portfolio_manager.update_prices(prices)

    return {
        "status": "prices_updated",
        "count": len(prices),
        "timestamp": datetime.utcnow().isoformat(),
    }
