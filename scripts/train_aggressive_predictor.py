#!/usr/bin/env python3
"""
Train Aggressive Profit Hunter

This script trains the aggressive ML predictor for high-frequency profit generation.
Target: 1%+ daily returns through frequent trading with leverage.

Usage:
    python scripts/train_aggressive_predictor.py --symbol BTC/USDT --timeframe 1h
    python scripts/train_aggressive_predictor.py --all  # Train all symbols
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from bot.ml import (
    AggressiveProfitHunter,
    AggressiveConfig,
    ProfitOptimizer,
    create_aggressive_predictor,
)
from bot.data.service import get_market_data_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
MODEL_DIR = Path("data/models")


def fetch_training_data(
    symbol: str,
    timeframe: str = "1h",
    days: int = 90,
) -> pd.DataFrame:
    """Fetch OHLCV data for training."""
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")

    service = get_market_data_service()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        # Use fetch_ohlcv method
        ohlcv = service.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=days * 24,  # Approximate bars for hourly data
        )

        if ohlcv is None or ohlcv.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        logger.info(f"Fetched {len(ohlcv)} bars for {symbol}")
        return ohlcv

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def train_aggressive_model(
    symbol: str,
    ohlcv: pd.DataFrame,
    max_leverage: float = 5.0,
) -> dict:
    """Train aggressive predictor for a symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Aggressive Predictor for {symbol}")
    logger.info(f"{'='*60}")

    # Create predictor with aggressive settings
    config = AggressiveConfig(
        short_horizon=1,
        medium_horizon=3,
        long_horizon=8,
        min_confidence_to_trade=0.55,
        high_confidence_threshold=0.70,
        extreme_confidence_threshold=0.80,
        base_leverage=2.0,
        max_leverage=max_leverage,
        base_position_pct=0.05,
        max_position_pct=0.25,
        enable_learning=True,
    )

    predictor = AggressiveProfitHunter(config=config, model_dir=str(MODEL_DIR))

    try:
        # Train
        metrics = predictor.train(ohlcv, symbol=symbol)

        # Save model
        safe_symbol = symbol.replace("/", "_")
        predictor.save(f"aggressive_{safe_symbol}")

        logger.info(f"\nTraining Results for {symbol}:")
        logger.info(f"  Ensemble Accuracy: {metrics['overall']['ensemble_accuracy']:.2%}")
        logger.info(f"  Best Model: {metrics['overall']['best_model']}")
        logger.info(f"  Features: {metrics['overall']['features']}")
        logger.info(f"  Train Samples: {metrics['overall']['train_samples']}")

        logger.info("\nIndividual Model Performance:")
        for model_name, model_metrics in metrics["models"].items():
            logger.info(
                f"  {model_name}: "
                f"train={model_metrics['train_accuracy']:.2%}, "
                f"test={model_metrics['test_accuracy']:.2%}, "
                f"profit_potential={model_metrics['profit_potential']:.2%}"
            )

        return metrics

    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def train_profit_optimizer(ohlcv: pd.DataFrame) -> None:
    """Train the profit optimizer."""
    logger.info("\nInitializing Profit Optimizer...")

    optimizer = ProfitOptimizer(model_dir=str(MODEL_DIR))

    # The optimizer learns from actual trades, but we can pre-warm with historical analysis
    # This will be used during live trading to optimize entry/exit timing

    optimizer.save("profit_optimizer")
    logger.info("Profit optimizer initialized and saved")


def backtest_model(
    symbol: str,
    ohlcv: pd.DataFrame,
    predictor: AggressiveProfitHunter,
) -> dict:
    """Run a quick backtest on the trained model."""
    logger.info(f"\nBacktesting {symbol}...")

    # Use last 20% of data for backtest
    test_start = int(len(ohlcv) * 0.8)
    test_data = ohlcv.iloc[test_start:].copy()

    if len(test_data) < 50:
        logger.warning("Insufficient data for backtest")
        return {}

    trades = []
    position = None
    equity = 10000.0
    peak_equity = equity

    for i in range(50, len(test_data) - 10):
        window = test_data.iloc[i-50:i+1]
        current_price = test_data.iloc[i]["close"]

        try:
            signal = predictor.predict(window)
        except Exception as e:
            continue

        # Manage position
        if position is None:
            # Entry logic
            if signal.action in ["LONG", "SHORT"] and signal.confidence >= 0.55:
                position = {
                    "direction": signal.action,
                    "entry_price": current_price,
                    "entry_bar": i,
                    "size_pct": signal.position_size_pct,
                    "leverage": min(signal.recommended_leverage, 3.0),  # Cap for backtest
                    "stop_loss": signal.stop_loss_pct,
                    "take_profit": signal.take_profit_pct,
                }
        else:
            # Exit logic
            bars_held = i - position["entry_bar"]

            if position["direction"] == "LONG":
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
            else:
                pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]

            # Apply leverage
            pnl_pct *= position["leverage"]

            should_exit = (
                pnl_pct >= position["take_profit"] or
                pnl_pct <= -position["stop_loss"] or
                bars_held >= signal.max_hold_bars or
                (signal.action != position["direction"] and signal.confidence >= 0.6)
            )

            if should_exit:
                # Calculate trade PnL
                trade_pnl = equity * position["size_pct"] * pnl_pct
                equity += trade_pnl
                peak_equity = max(peak_equity, equity)

                trades.append({
                    "direction": position["direction"],
                    "pnl_pct": pnl_pct,
                    "pnl_usd": trade_pnl,
                    "bars_held": bars_held,
                    "was_winner": pnl_pct > 0,
                })

                # Record for learning
                predictor.record_trade_outcome(
                    predicted_action=position["direction"],
                    actual_pnl_pct=pnl_pct,
                    confidence=signal.confidence,
                    features=window[predictor.feature_names].iloc[-1].values if predictor.feature_names else [],
                    regime=signal.regime,
                    holding_period=bars_held,
                )

                position = None

    # Calculate backtest metrics
    if not trades:
        return {"error": "No trades executed"}

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t["was_winner"])
    win_rate = winning_trades / total_trades

    total_pnl = sum(t["pnl_usd"] for t in trades)
    total_return = (equity - 10000) / 10000

    avg_win = sum(t["pnl_pct"] for t in trades if t["was_winner"]) / max(1, winning_trades)
    avg_loss = sum(t["pnl_pct"] for t in trades if not t["was_winner"]) / max(1, total_trades - winning_trades)

    max_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

    # Estimate daily return (assuming test period is ~20% of 90 days = 18 days)
    test_days = len(test_data) / 24  # Assuming 1h timeframe
    daily_return = total_return / max(1, test_days)

    results = {
        "total_trades": total_trades,
        "win_rate": f"{win_rate:.2%}",
        "total_return": f"{total_return:.2%}",
        "daily_return": f"{daily_return:.2%}",
        "final_equity": f"${equity:,.2f}",
        "max_drawdown": f"{max_drawdown:.2%}",
        "avg_win": f"{avg_win:.2%}",
        "avg_loss": f"{avg_loss:.2%}",
        "profit_factor": f"{abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades) + 1e-10)):.2f}",
    }

    logger.info(f"\nBacktest Results for {symbol}:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Aggressive Profit Hunter")
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., BTC/USDT)")
    parser.add_argument("--all", action="store_true", help="Train all default symbols")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=90, help="Days of data to fetch (default: 90)")
    parser.add_argument("--max-leverage", type=float, default=5.0, help="Max leverage (default: 5.0)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest after training")

    args = parser.parse_args()

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Determine symbols to train
    if args.all:
        symbols = DEFAULT_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = DEFAULT_SYMBOLS

    logger.info(f"Training Aggressive Predictor")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Max Leverage: {args.max_leverage}x")

    all_results = {}

    for symbol in symbols:
        # Fetch data
        ohlcv = fetch_training_data(symbol, args.timeframe, args.days)

        if ohlcv.empty:
            logger.warning(f"Skipping {symbol} - no data")
            continue

        # Train model
        metrics = train_aggressive_model(symbol, ohlcv, args.max_leverage)

        if "error" not in metrics:
            all_results[symbol] = metrics

            # Optionally run backtest
            if args.backtest:
                safe_symbol = symbol.replace("/", "_")
                predictor = create_aggressive_predictor(
                    max_leverage=args.max_leverage,
                    model_dir=str(MODEL_DIR),
                )
                if predictor.load(f"aggressive_{safe_symbol}"):
                    backtest_results = backtest_model(symbol, ohlcv, predictor)
                    all_results[symbol]["backtest"] = backtest_results

    # Train profit optimizer (uses first symbol's data)
    if symbols and all_results:
        first_symbol = symbols[0]
        ohlcv = fetch_training_data(first_symbol, args.timeframe, args.days)
        if not ohlcv.empty:
            train_profit_optimizer(ohlcv)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)

    for symbol, metrics in all_results.items():
        if "overall" in metrics:
            logger.info(f"\n{symbol}:")
            logger.info(f"  Accuracy: {metrics['overall']['ensemble_accuracy']:.2%}")
            if "backtest" in metrics and "win_rate" in metrics["backtest"]:
                logger.info(f"  Backtest Win Rate: {metrics['backtest']['win_rate']}")
                logger.info(f"  Backtest Daily Return: {metrics['backtest']['daily_return']}")

    logger.info("\n" + "="*60)
    logger.info("Training complete! Models saved to data/models/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
