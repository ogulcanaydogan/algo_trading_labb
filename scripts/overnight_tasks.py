#!/usr/bin/env python3
"""
Overnight Tasks Script.

Runs monitoring, backtests, and ML training in background.
Results are saved to data/overnight_results/
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

# Setup logging
log_dir = Path("data/overnight_results")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "overnight.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def fetch_data(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Fetch historical data for a symbol."""
    yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


def run_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """Run backtest for a symbol."""
    from bot.backtesting import Backtester

    try:
        backtester = Backtester(
            initial_capital=10000,
            commission_rate=0.001,
        )

        results = backtester.run(
            df,
            stop_loss_pct=0.015,
            take_profit_pct=0.05,
        )

        return {
            "symbol": symbol,
            "status": "success",
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
        }
    except Exception as e:
        return {"symbol": symbol, "status": "error", "error": str(e)}


def run_strategy_comparison(symbol: str, df: pd.DataFrame) -> dict:
    """Compare different strategy parameters."""
    results = []

    # Test different stop loss / take profit combinations
    params_to_test = [
        {"sl": 0.01, "tp": 0.03},
        {"sl": 0.015, "tp": 0.05},
        {"sl": 0.02, "tp": 0.04},
        {"sl": 0.02, "tp": 0.06},
        {"sl": 0.025, "tp": 0.05},
    ]

    for params in params_to_test:
        try:
            from bot.backtesting import Backtester

            backtester = Backtester(initial_capital=10000)
            result = backtester.run(
                df,
                stop_loss_pct=params["sl"],
                take_profit_pct=params["tp"],
            )

            results.append({
                "stop_loss": params["sl"],
                "take_profit": params["tp"],
                "return": result.get("total_return", 0),
                "sharpe": result.get("sharpe_ratio", 0),
                "win_rate": result.get("win_rate", 0),
            })
        except:
            pass

    # Find best params
    if results:
        best = max(results, key=lambda x: x.get("return", 0))
        return {
            "symbol": symbol,
            "best_params": best,
            "all_results": results,
        }

    return {"symbol": symbol, "status": "no_results"}


def train_models(symbols: list) -> dict:
    """Train ML models for symbols."""
    results = {}

    for symbol in symbols:
        logger.info(f"Training model for {symbol}...")

        try:
            df = fetch_data(symbol)
            if df is None or len(df) < 200:
                results[symbol] = {"status": "insufficient_data"}
                continue

            from bot.ml.trainer import ModelTrainer

            trainer = ModelTrainer(model_type="gradient_boosting")
            metrics = trainer.train(df)
            trainer.save(f"data/models/{symbol.replace('/', '_')}_gradient_boosting")

            results[symbol] = {
                "status": "success",
                "accuracy": metrics.get("accuracy", 0),
            }

        except Exception as e:
            results[symbol] = {"status": "error", "error": str(e)}

    return results


def monitor_performance() -> dict:
    """Check current testnet performance."""
    try:
        import ccxt

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            return {"status": "no_credentials"}

        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
        })
        exchange.set_sandbox_mode(True)

        balance = exchange.fetch_balance()
        usdt = balance["USDT"]["total"]

        positions = []
        total_value = usdt

        for symbol in ["BTC", "ETH", "SOL", "AVAX", "ADA", "XRP", "DOGE"]:
            if symbol in balance and balance[symbol]["total"] > 0.0001:
                try:
                    ticker = exchange.fetch_ticker(f"{symbol}/USDT")
                    value = balance[symbol]["total"] * ticker["last"]
                    total_value += value
                    positions.append({
                        "symbol": symbol,
                        "quantity": balance[symbol]["total"],
                        "price": ticker["last"],
                        "value": value,
                    })
                except:
                    pass

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "usdt_balance": usdt,
            "total_value": total_value,
            "pnl_from_10k": total_value - 10000,
            "pnl_pct": ((total_value / 10000) - 1) * 100,
            "positions": positions,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    """Run all overnight tasks."""
    logger.info("=" * 60)
    logger.info("OVERNIGHT TASKS STARTED")
    logger.info("=" * 60)

    results = {
        "started_at": datetime.now().isoformat(),
        "tasks": {},
    }

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "ADA/USDT", "XRP/USDT", "DOGE/USDT"]

    # 1. Monitor current performance
    logger.info("\n[1/4] Monitoring performance...")
    results["tasks"]["monitoring"] = monitor_performance()

    # 2. Run backtests
    logger.info("\n[2/4] Running backtests...")
    backtest_results = {}
    for symbol in symbols:
        logger.info(f"  Backtesting {symbol}...")
        df = fetch_data(symbol)
        if df is not None and len(df) > 100:
            backtest_results[symbol] = run_backtest(symbol, df)
        else:
            backtest_results[symbol] = {"status": "insufficient_data"}
    results["tasks"]["backtests"] = backtest_results

    # 3. Strategy comparison
    logger.info("\n[3/4] Comparing strategies...")
    strategy_results = {}
    for symbol in symbols[:3]:  # Top 3 only
        logger.info(f"  Comparing strategies for {symbol}...")
        df = fetch_data(symbol)
        if df is not None:
            strategy_results[symbol] = run_strategy_comparison(symbol, df)
    results["tasks"]["strategy_comparison"] = strategy_results

    # 4. Auto-retraining check
    logger.info("\n[4/5] Checking for model retraining needs...")
    try:
        from bot.auto_retraining import AutoRetrainingSystem
        from bot.ml_performance_tracker import get_ml_tracker

        retrainer = AutoRetrainingSystem(
            models_dir=Path("data/models"),
            performance_tracker=get_ml_tracker(),
            check_interval_minutes=60,
            min_retrain_interval_hours=24,
        )

        # Check if any models need retraining
        retraining_results = {}
        for symbol in symbols[:3]:
            model_id = f"{symbol.replace('/', '_')}_gradient_boosting"
            check = retrainer.check_model_degradation(model_id)
            retraining_results[symbol] = {
                "needs_retraining": check is not None,
                "reason": check.value if check else None,
            }
            if check:
                logger.info(f"  {symbol}: Retraining recommended ({check.value})")
            else:
                logger.info(f"  {symbol}: Model performing well")

        results["tasks"]["auto_retraining"] = {
            "status": "success",
            "models_checked": len(retraining_results),
            "results": retraining_results,
        }
    except Exception as e:
        logger.warning(f"Auto-retraining check failed: {e}")
        results["tasks"]["auto_retraining"] = {"status": "error", "error": str(e)}

    # 5. Save results
    logger.info("\n[5/5] Saving results...")
    results["completed_at"] = datetime.now().isoformat()

    output_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("OVERNIGHT TASKS COMPLETED")
    logger.info("=" * 60)

    monitoring = results["tasks"].get("monitoring", {})
    if monitoring.get("status") == "success":
        logger.info(f"Portfolio Value: ${monitoring.get('total_value', 0):,.2f}")
        logger.info(f"P&L: ${monitoring.get('pnl_from_10k', 0):,.2f} ({monitoring.get('pnl_pct', 0):.2f}%)")

    logger.info(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
