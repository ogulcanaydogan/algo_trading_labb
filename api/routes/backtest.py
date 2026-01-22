"""
Backtesting API - Programmatic access to backtesting.

Provides endpoints for running backtests, analyzing results,
and comparing strategies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtesting"])


# Request/Response Models
class BacktestRequest(BaseModel):
    """Backtest request parameters."""
    symbol: str = Field(..., description="Trading symbol")
    strategy: str = Field(..., description="Strategy name")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000, description="Starting capital")
    position_size_pct: float = Field(default=0.1, description="Position size as % of capital")
    commission_pct: float = Field(default=0.001, description="Commission percentage")
    slippage_pct: float = Field(default=0.0005, description="Slippage percentage")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class BacktestResult(BaseModel):
    """Backtest result summary."""
    backtest_id: str
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    exposure_time: float
    status: str


class TradeRecord(BaseModel):
    """Individual trade record."""
    trade_id: int
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    holding_period: int
    exit_reason: str


class OptimizationRequest(BaseModel):
    """Strategy optimization request."""
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    param_grid: Dict[str, List[Any]] = Field(..., description="Parameter search grid")
    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    n_jobs: int = Field(default=1, description="Parallel jobs")


class WalkForwardRequest(BaseModel):
    """Walk-forward analysis request."""
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    train_period_days: int = Field(default=180, description="Training period")
    test_period_days: int = Field(default=30, description="Test period")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)


# In-memory storage for backtest results
_backtest_results: Dict[str, Dict] = {}
_backtest_counter = 0


class BacktestEngine:
    """Simple backtest engine for API."""

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.capital = initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        position_size_pct: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run backtest with given signals.

        Args:
            prices: OHLCV DataFrame
            signals: Series of signals (1=long, -1=short, 0=flat)
            position_size_pct: Position size as fraction of capital

        Returns:
            Backtest results dictionary
        """
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]

        position = 0
        entry_price = 0
        entry_date = None
        entry_idx = 0

        for i, (date, row) in enumerate(prices.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = row["close"]

            # Exit existing position
            if position != 0 and signal != position:
                # Calculate PnL
                exit_price = price * (1 - self.slippage_pct * np.sign(position))
                commission = abs(position) * exit_price * self.commission_pct

                if position > 0:
                    pnl = (exit_price - entry_price) * abs(position) - commission
                else:
                    pnl = (entry_price - exit_price) * abs(position) - commission

                self.capital += pnl

                self.trades.append({
                    "trade_id": len(self.trades) + 1,
                    "entry_date": entry_date.isoformat() if entry_date else "",
                    "exit_date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                    "side": "long" if position > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": abs(position),
                    "pnl": pnl,
                    "return_pct": pnl / (entry_price * abs(position)),
                    "holding_period": i - entry_idx,
                    "exit_reason": "signal_reversal" if signal != 0 else "signal_flat",
                })

                position = 0

            # Enter new position
            if signal != 0 and position == 0:
                position_value = self.capital * position_size_pct
                entry_price = price * (1 + self.slippage_pct * signal)
                position = signal * (position_value / entry_price)
                entry_date = date
                entry_idx = i

                # Deduct commission
                commission = abs(position) * entry_price * self.commission_pct
                self.capital -= commission

            # Update equity curve
            if position != 0:
                unrealized = (price - entry_price) * position
                self.equity_curve.append(self.capital + unrealized)
            else:
                self.equity_curve.append(self.capital)

        # Close any remaining position
        if position != 0:
            exit_price = prices["close"].iloc[-1]
            if position > 0:
                pnl = (exit_price - entry_price) * abs(position)
            else:
                pnl = (entry_price - exit_price) * abs(position)
            self.capital += pnl

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {
                "total_return": 0,
                "total_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "avg_trade_return": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "avg_holding_period": 0,
                "exposure_time": 0,
            }

        returns = [t["return_pct"] for t in self.trades]
        pnls = [t["pnl"] for t in self.trades]

        # Basic metrics
        total_return = self.capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        # Trade statistics
        avg_trade_return = np.mean(returns)
        best_trade = max(returns)
        worst_trade = min(returns)
        avg_holding_period = np.mean([t["holding_period"] for t in self.trades])

        # Exposure time
        total_bars = len(self.equity_curve)
        bars_in_trade = sum(t["holding_period"] for t in self.trades)
        exposure_time = bars_in_trade / total_bars if total_bars > 0 else 0

        return {
            "final_capital": self.capital,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_trade_return": avg_trade_return,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_holding_period": avg_holding_period,
            "exposure_time": exposure_time,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }


def generate_dummy_signals(prices: pd.DataFrame, strategy: str) -> pd.Series:
    """Generate dummy signals for testing (replace with real strategy)."""
    signals = pd.Series(0, index=prices.index)

    if strategy == "ema_crossover":
        ema_fast = prices["close"].ewm(span=12).mean()
        ema_slow = prices["close"].ewm(span=26).mean()
        signals[ema_fast > ema_slow] = 1
        signals[ema_fast < ema_slow] = -1

    elif strategy == "rsi":
        delta = prices["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals[rsi < 30] = 1
        signals[rsi > 70] = -1

    elif strategy == "random":
        np.random.seed(42)
        signals = pd.Series(np.random.choice([-1, 0, 1], size=len(prices)), index=prices.index)

    return signals


def generate_dummy_prices(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Generate dummy price data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    np.random.seed(hash(symbol) % 2**32)

    # Random walk with drift
    returns = np.random.normal(0.0002, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        "close": prices,
        "volume": np.random.uniform(1000000, 5000000, len(dates)),
    }, index=dates)

    return df


# API Endpoints

@router.post("/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest with specified parameters.

    Returns backtest results including performance metrics.
    """
    global _backtest_counter
    _backtest_counter += 1
    backtest_id = f"bt_{_backtest_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        # Get price data (replace with real data fetching)
        prices = generate_dummy_prices(
            request.symbol,
            request.start_date,
            request.end_date,
        )

        # Generate signals (replace with real strategy)
        signals = generate_dummy_signals(prices, request.strategy)

        # Run backtest
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission_pct=request.commission_pct,
            slippage_pct=request.slippage_pct,
        )

        results = engine.run(prices, signals, request.position_size_pct)

        # Store results
        _backtest_results[backtest_id] = {
            "request": request.model_dump(),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        return BacktestResult(
            backtest_id=backtest_id,
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            final_capital=results["final_capital"],
            total_return=results["total_return"],
            total_return_pct=results["total_return_pct"],
            sharpe_ratio=results["sharpe_ratio"],
            max_drawdown=results["max_drawdown"],
            win_rate=results["win_rate"],
            profit_factor=results["profit_factor"],
            total_trades=results["total_trades"],
            avg_trade_return=results["avg_trade_return"],
            best_trade=results["best_trade"],
            worst_trade=results["worst_trade"],
            avg_holding_period=results["avg_holding_period"],
            exposure_time=results["exposure_time"],
            status="completed",
        )

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """Get detailed results for a specific backtest."""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return _backtest_results[backtest_id]


@router.get("/{backtest_id}/trades", response_model=List[TradeRecord])
async def get_backtest_trades(backtest_id: str):
    """Get trade list for a backtest."""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    trades = _backtest_results[backtest_id]["results"].get("trades", [])
    return [TradeRecord(**t) for t in trades]


@router.get("/{backtest_id}/equity")
async def get_equity_curve(backtest_id: str):
    """Get equity curve for a backtest."""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    equity = _backtest_results[backtest_id]["results"].get("equity_curve", [])
    return {"equity_curve": equity}


@router.post("/compare")
async def compare_strategies(requests: List[BacktestRequest]):
    """
    Compare multiple strategies on the same data.

    Returns comparison metrics for all strategies.
    """
    results = []

    for req in requests:
        try:
            prices = generate_dummy_prices(req.symbol, req.start_date, req.end_date)
            signals = generate_dummy_signals(prices, req.strategy)

            engine = BacktestEngine(
                initial_capital=req.initial_capital,
                commission_pct=req.commission_pct,
                slippage_pct=req.slippage_pct,
            )

            result = engine.run(prices, signals, req.position_size_pct)

            results.append({
                "strategy": req.strategy,
                "symbol": req.symbol,
                "total_return_pct": result["total_return_pct"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
                "win_rate": result["win_rate"],
                "total_trades": result["total_trades"],
            })
        except Exception as e:
            results.append({
                "strategy": req.strategy,
                "symbol": req.symbol,
                "error": str(e),
            })

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)

    return {
        "comparison": results,
        "best_strategy": results[0]["strategy"] if results else None,
    }


@router.post("/walk-forward")
async def run_walk_forward(request: WalkForwardRequest):
    """
    Run walk-forward analysis.

    Tests strategy across multiple out-of-sample periods.
    """
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")

        results = []
        current_start = start

        while current_start + timedelta(days=request.train_period_days + request.test_period_days) <= end:
            train_end = current_start + timedelta(days=request.train_period_days)
            test_end = train_end + timedelta(days=request.test_period_days)

            # Get test period data
            prices = generate_dummy_prices(
                request.symbol,
                train_end.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )

            signals = generate_dummy_signals(prices, request.strategy)

            engine = BacktestEngine()
            result = engine.run(prices, signals)

            results.append({
                "period_start": train_end.strftime("%Y-%m-%d"),
                "period_end": test_end.strftime("%Y-%m-%d"),
                "return_pct": result["total_return_pct"],
                "sharpe_ratio": result["sharpe_ratio"],
                "max_drawdown": result["max_drawdown"],
                "trades": result["total_trades"],
            })

            current_start = train_end

        # Aggregate results
        if results:
            avg_return = np.mean([r["return_pct"] for r in results])
            avg_sharpe = np.mean([r["sharpe_ratio"] for r in results])
            consistency = sum(1 for r in results if r["return_pct"] > 0) / len(results)

            return {
                "periods": results,
                "summary": {
                    "num_periods": len(results),
                    "avg_return_pct": avg_return,
                    "avg_sharpe_ratio": avg_sharpe,
                    "consistency": consistency,
                    "total_trades": sum(r["trades"] for r in results),
                }
            }

        return {"periods": [], "summary": {"error": "Insufficient data"}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backtests(limit: int = 20):
    """List recent backtests."""
    items = list(_backtest_results.items())[-limit:]
    return [
        {
            "backtest_id": bt_id,
            "strategy": data["request"]["strategy"],
            "symbol": data["request"]["symbol"],
            "timestamp": data["timestamp"],
            "return_pct": data["results"]["total_return_pct"],
        }
        for bt_id, data in items
    ]


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest result."""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    del _backtest_results[backtest_id]
    return {"status": "deleted", "backtest_id": backtest_id}
