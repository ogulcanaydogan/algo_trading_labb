"""
Advanced API Endpoints.

Provides endpoints for:
- Kelly criterion position sizing
- Monte Carlo simulation
- Benchmark comparison
- Feature importance
- Performance reports
- Trailing stop management
- Ensemble predictions
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/advanced", tags=["Advanced"])

# Data directory
DATA_DIR = Path("data/unified_trading")


# Request/Response Models
class KellyRequest(BaseModel):
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive


class MonteCarloRequest(BaseModel):
    initial_balance: float = 10000
    num_simulations: int = 1000
    num_trades: int = 100


class BenchmarkRequest(BaseModel):
    benchmark_symbol: str = "BTC/USDT"
    days: int = 30


# Kelly Criterion
@router.get("/kelly")
async def get_kelly_criterion(
    risk_tolerance: str = Query("moderate", enum=["conservative", "moderate", "aggressive"]),
) -> Dict[str, Any]:
    """
    Calculate Kelly criterion optimal position sizing.
    """
    from bot.advanced_risk import AdvancedRiskManager
    import json

    # Load trades
    trades_file = DATA_DIR / "trades.json"
    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)

    manager = AdvancedRiskManager(risk_tolerance=risk_tolerance)
    kelly = manager.calculate_kelly(trades)

    return {
        "kelly": kelly.to_dict(),
        "trades_analyzed": len(trades),
        "recommendation": f"Use {kelly.recommended_fraction*100:.1f}% of capital per trade",
    }


# Drawdown Scaling
@router.get("/drawdown-scaling")
async def get_drawdown_scaling() -> Dict[str, Any]:
    """
    Get current position scaling based on drawdown.
    """
    from bot.advanced_risk import AdvancedRiskManager
    import json

    # Load state
    state_file = DATA_DIR / "state.json"
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    current_balance = state.get("balance", 10000)
    peak_balance = state.get("peak_balance", current_balance)

    manager = AdvancedRiskManager()
    scaling = manager.calculate_drawdown_scaling(current_balance, peak_balance)

    return {
        "scaling": scaling.to_dict(),
        "current_balance": current_balance,
        "peak_balance": peak_balance,
    }


# Monte Carlo
@router.post("/monte-carlo")
async def run_monte_carlo(request: MonteCarloRequest) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for risk analysis.
    """
    from bot.advanced_risk import AdvancedRiskManager
    import json

    # Load trades
    trades_file = DATA_DIR / "trades.json"
    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)

    if len(trades) < 10:
        raise HTTPException(
            status_code=400,
            detail="Need at least 10 trades for Monte Carlo simulation"
        )

    manager = AdvancedRiskManager()
    result = manager.run_monte_carlo(
        trades,
        initial_balance=request.initial_balance,
        num_simulations=request.num_simulations,
        num_trades=request.num_trades,
    )

    return {
        "monte_carlo": result.to_dict(),
        "interpretation": {
            "worst_case_5pct": f"5% chance of ending below ${result.percentile_5:,.2f}",
            "best_case_5pct": f"5% chance of ending above ${result.percentile_95:,.2f}",
            "median_outcome": f"50% chance of ending around ${result.median_final_balance:,.2f}",
            "ruin_probability": f"{result.probability_of_ruin*100:.1f}% chance of 50%+ drawdown",
        }
    }


# Benchmark Comparison
@router.get("/benchmark")
async def compare_to_benchmark(
    benchmark_symbol: str = Query("BTC/USDT"),
    days: int = Query(30, ge=7, le=365),
) -> Dict[str, Any]:
    """
    Compare strategy performance to buy-and-hold benchmark.
    """
    from bot.advanced_risk import AdvancedRiskManager
    import json

    # Load equity curve
    equity_file = DATA_DIR / "equity.json"
    equity_history = []
    if equity_file.exists():
        with open(equity_file) as f:
            equity_history = json.load(f)

    if len(equity_history) < 10:
        raise HTTPException(
            status_code=400,
            detail="Not enough equity history for benchmark comparison"
        )

    strategy_equity = [e["equity"] for e in equity_history[-days*24:]]  # Assuming hourly

    # Fetch benchmark prices
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        from datetime import timedelta
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(benchmark_symbol, "1h", since=since, limit=1000)

        benchmark_prices = [candle[4] for candle in ohlcv]  # Close prices

        # Normalize to same starting point
        if benchmark_prices:
            initial_price = benchmark_prices[0]
            initial_equity = strategy_equity[0] if strategy_equity else 10000
            benchmark_normalized = [p / initial_price * initial_equity for p in benchmark_prices]
        else:
            benchmark_normalized = strategy_equity  # Fallback

    except Exception as e:
        # Fallback - use strategy equity as benchmark
        benchmark_normalized = strategy_equity

    manager = AdvancedRiskManager()
    comparison = manager.compare_to_benchmark(strategy_equity, benchmark_normalized)

    return {
        "comparison": comparison.to_dict(),
        "benchmark_symbol": benchmark_symbol,
        "period_days": days,
        "verdict": "Outperforming" if comparison.alpha > 0 else "Underperforming",
    }


# Feature Importance
@router.get("/feature-importance/{symbol}")
async def get_feature_importance(
    symbol: str,
    model_type: str = Query("gradient_boosting"),
    top_n: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """
    Get feature importance from ML model.
    """
    from bot.advanced_risk import FeatureImportanceAnalyzer

    analyzer = FeatureImportanceAnalyzer()
    top_features = analyzer.get_top_features(symbol, top_n)

    if not top_features:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for {symbol}"
        )

    return {
        "symbol": symbol,
        "model_type": model_type,
        "top_features": [
            {"name": name, "importance": round(imp, 4)}
            for name, imp in top_features
        ],
        "total_features": len(top_features),
    }


# Performance Reports
@router.get("/reports/daily")
async def get_daily_report(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
) -> Dict[str, Any]:
    """
    Generate daily performance report.
    """
    from bot.performance_reports import PerformanceReporter

    reporter = PerformanceReporter()
    report = reporter.generate_daily_report(date)

    return report.to_dict()


@router.get("/reports/weekly")
async def get_weekly_report(
    week_end: Optional[str] = Query(None, description="Week end date in YYYY-MM-DD format"),
) -> Dict[str, Any]:
    """
    Generate weekly performance report.
    """
    from bot.performance_reports import PerformanceReporter

    reporter = PerformanceReporter()
    report = reporter.generate_weekly_report(week_end)

    return report.to_dict()


@router.post("/reports/email")
async def send_report_email(
    date: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Send daily report via email.
    """
    from bot.performance_reports import PerformanceReporter

    reporter = PerformanceReporter()
    report = reporter.generate_daily_report(date)
    success = reporter.send_email_report(report)

    return {
        "success": success,
        "date": report.date,
        "message": "Email sent successfully" if success else "Email sending failed or not configured",
    }


# Trailing Stops
@router.get("/trailing-stops")
async def get_trailing_stops() -> Dict[str, Any]:
    """
    Get all active trailing stops.
    """
    from bot.trailing_stop import TrailingStopManager

    manager = TrailingStopManager()

    # Load stops from state file
    stops_file = DATA_DIR / "trailing_stops.json"
    if stops_file.exists():
        import json
        with open(stops_file) as f:
            stops_data = json.load(f)
            manager.load_state(stops_data)

    return {
        "active_stops": manager.get_all_stops(),
        "count": len(manager.stops),
    }


@router.post("/trailing-stops/{symbol}")
async def add_trailing_stop(
    symbol: str,
    entry_price: float = Query(...),
    side: str = Query("long", enum=["long", "short"]),
) -> Dict[str, Any]:
    """
    Add trailing stop for a position.
    """
    from bot.trailing_stop import TrailingStopManager
    import json

    manager = TrailingStopManager()

    # Load existing stops
    stops_file = DATA_DIR / "trailing_stops.json"
    if stops_file.exists():
        with open(stops_file) as f:
            stops_data = json.load(f)
            manager.load_state(stops_data)

    # Add new stop
    state = manager.add_position(symbol, entry_price, side)

    # Save
    with open(stops_file, "w") as f:
        json.dump(manager.get_all_stops(), f, indent=2)

    return {
        "success": True,
        "stop": state.to_dict(),
    }


# Ensemble Predictions
@router.get("/ensemble/{symbol}")
async def get_ensemble_prediction(
    symbol: str,
) -> Dict[str, Any]:
    """
    Get ensemble prediction combining multiple models.
    """
    from bot.ml.ensemble_predictor import create_ensemble_predictor
    from pathlib import Path
    import numpy as np

    predictor = create_ensemble_predictor(
        symbol=symbol,
        model_dir=Path("data/models"),
        voting_strategy="performance",
    )

    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail=f"No models found for {symbol}"
        )

    # Get latest features (placeholder - in production would fetch live data)
    # This is a simplified example
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=50)

        # Simple feature extraction
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        features = np.array([[
            (closes[-1] - closes[-2]) / closes[-2],  # return_1
            (closes[-1] - closes[-6]) / closes[-6],  # return_5
            (closes[-1] - closes[-11]) / closes[-11],  # return_10
            (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 else 0,  # return_20
            closes[-1] / np.mean(closes[-5:]),  # price_sma5_ratio
            closes[-1] / np.mean(closes[-10:]),  # price_sma10_ratio
            closes[-1] / np.mean(closes[-20:]) if len(closes) >= 20 else 1,  # price_sma20_ratio
            np.std(np.diff(closes[-5:])/closes[-6:-1]),  # volatility_5
            np.std(np.diff(closes[-20:])/closes[-21:-1]) if len(closes) > 20 else 0,  # volatility_20
            volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1,  # volume_ratio
            50,  # RSI placeholder
            0,  # MACD hist placeholder
            0.5,  # BB position placeholder
            0.02,  # ATR ratio placeholder
        ]])

        prediction, confidence, details = predictor.predict(features)

        action_map = {1: "LONG", 0: "FLAT", -1: "SHORT"}

        return {
            "symbol": symbol,
            "prediction": action_map.get(prediction, "FLAT"),
            "confidence": round(confidence, 4),
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Correlation Analysis
@router.get("/correlations")
async def get_position_correlations() -> Dict[str, Any]:
    """
    Get correlation analysis for current positions.
    """
    try:
        import ccxt
        import numpy as np
        from datetime import timedelta

        exchange = ccxt.binance({'enableRateLimit': True})

        # Get current positions
        import json
        state_file = DATA_DIR / "state.json"
        state = {}
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

        positions = state.get("positions", {})
        symbols = list(positions.keys()) + ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        symbols = list(set(symbols))[:5]  # Limit to 5

        # Fetch price data
        since = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
        price_data = {}

        for symbol in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=720)
                returns = []
                for i in range(1, len(ohlcv)):
                    ret = (ohlcv[i][4] - ohlcv[i-1][4]) / ohlcv[i-1][4]
                    returns.append(ret)
                price_data[symbol] = returns
            except:
                continue

        # Calculate correlations
        correlations = {}
        for sym1 in price_data:
            correlations[sym1] = {}
            for sym2 in price_data:
                if len(price_data[sym1]) == len(price_data[sym2]) and len(price_data[sym1]) > 0:
                    corr = np.corrcoef(price_data[sym1], price_data[sym2])[0, 1]
                    correlations[sym1][sym2] = round(corr, 4)
                else:
                    correlations[sym1][sym2] = 0

        return {
            "correlations": correlations,
            "symbols": symbols,
            "period": "30 days",
        }

    except Exception as e:
        return {
            "error": str(e),
            "correlations": {},
        }


# Risk Summary
@router.get("/risk-summary")
async def get_risk_summary() -> Dict[str, Any]:
    """
    Get comprehensive risk summary combining all metrics.
    """
    import json

    # Load state
    state_file = DATA_DIR / "state.json"
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    # Load trades
    trades_file = DATA_DIR / "trades.json"
    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)

    from bot.advanced_risk import AdvancedRiskManager

    manager = AdvancedRiskManager()

    # Kelly
    kelly = manager.calculate_kelly(trades)

    # Drawdown
    current_balance = state.get("balance", 10000)
    peak_balance = state.get("peak_balance", current_balance)
    drawdown = manager.calculate_drawdown_scaling(current_balance, peak_balance)

    # Monte Carlo (quick version)
    monte_carlo = None
    if len(trades) >= 10:
        monte_carlo = manager.run_monte_carlo(
            trades,
            initial_balance=current_balance,
            num_simulations=100,
            num_trades=50,
        )

    return {
        "balance": current_balance,
        "peak_balance": peak_balance,
        "kelly": kelly.to_dict(),
        "drawdown": drawdown.to_dict(),
        "monte_carlo": monte_carlo.to_dict() if monte_carlo else None,
        "trades_analyzed": len(trades),
        "overall_risk_level": "LOW" if drawdown.scale_factor > 0.75 else "MEDIUM" if drawdown.scale_factor > 0.25 else "HIGH",
    }
