#!/usr/bin/env python3
"""
Automated Model Retraining Runner.

Run as a scheduled job (e.g., cron) or as a background service.

Usage:
    # Check which models need retraining
    python run_model_retraining.py --check

    # Run retraining for all models that need it
    python run_model_retraining.py --retrain

    # Force retrain specific model
    python run_model_retraining.py --retrain --symbol BTC/USDT --model lstm

    # Run as daemon (continuous monitoring)
    python run_model_retraining.py --daemon --interval 3600
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.training import (
    RetrainingPipeline,
    RetrainingConfig,
    RetrainingTrigger,
    PerformanceMetrics,
)
from bot.ml.models.deep_learning.lstm import LSTMPredictor
from bot.ml.models.deep_learning.transformer import TransformerPredictor
from bot.ml.feature_engineer import FeatureEngineer
from bot.notifications import send_notification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/model_retraining.log"),
    ],
)
logger = logging.getLogger(__name__)


# Default symbols and models
DEFAULT_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
DEFAULT_COMMODITY_SYMBOLS = ["XAU/USD", "XAG/USD", "USOIL/USD", "NATGAS/USD"]
DEFAULT_MODEL_TYPES = ["lstm", "transformer"]


def get_training_data(
    symbol: str,
    market_type: str,
    lookback_days: int = 365,
) -> Optional[pd.DataFrame]:
    """Fetch historical data for training."""
    try:
        if market_type == "crypto":
            import ccxt
            exchange = ccxt.binance()
            since = int((datetime.now().timestamp() - lookback_days * 86400) * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=1000)
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
        else:
            import yfinance as yf
            # Convert symbol format
            yf_symbol = symbol.replace("/", "").replace("USD", "=F")
            if market_type == "stock":
                yf_symbol = symbol

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=f"{lookback_days}d", interval="1h")
            df.columns = [c.lower() for c in df.columns]
            if "adj close" in df.columns:
                df = df.drop(columns=["adj close"])
            df = df.rename(columns={"dividends": "div", "stock splits": "splits"})
            df = df[["open", "high", "low", "close", "volume"]]

        return df

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def train_model(
    model_type: str,
    training_data: pd.DataFrame,
    symbol: str,
    market_type: str,
) -> Optional[object]:
    """Train a model on the given data."""
    try:
        # Generate features
        fe = FeatureEngineer()
        features_df = fe.generate_features(training_data)

        # Create target (next period return sign)
        features_df["target"] = np.sign(features_df["close"].shift(-1) - features_df["close"])
        features_df = features_df.dropna()

        # Prepare X and y
        feature_cols = [c for c in features_df.columns if c not in ["target", "open", "high", "low", "close", "volume"]]
        X = features_df[feature_cols].values
        y = features_df["target"].values

        # Map target to 0, 1, 2
        y = np.where(y > 0, 2, np.where(y < 0, 0, 1))

        if model_type == "lstm":
            model = LSTMPredictor(
                input_size=len(feature_cols),
                hidden_size=64,
                num_layers=2,
                num_classes=3,
                sequence_length=60,
            )
        elif model_type == "transformer":
            model = TransformerPredictor(
                input_size=len(feature_cols),
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                num_classes=3,
                sequence_length=60,
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

        # Train
        model.fit(X, y, epochs=50, batch_size=32)

        return model

    except Exception as e:
        logger.error(f"Training failed for {symbol} {model_type}: {e}")
        return None


def validate_model(
    model: object,
    validation_data: pd.DataFrame,
) -> PerformanceMetrics:
    """Validate a model and return performance metrics."""
    try:
        fe = FeatureEngineer()
        features_df = fe.generate_features(validation_data)
        features_df["target"] = np.sign(features_df["close"].shift(-1) - features_df["close"])
        features_df = features_df.dropna()

        feature_cols = [c for c in features_df.columns if c not in ["target", "open", "high", "low", "close", "volume"]]
        X = features_df[feature_cols].values
        y_true = features_df["target"].values
        y_true = np.where(y_true > 0, 2, np.where(y_true < 0, 0, 1))

        # Get predictions
        predictions = []
        for i in range(len(X) - model.sequence_length):
            seq = X[i:i + model.sequence_length]
            pred = model.predict(seq.reshape(1, -1))
            predictions.append(pred.action)

        # Map predictions
        pred_map = {"LONG": 2, "SHORT": 0, "FLAT": 1}
        y_pred = np.array([pred_map.get(p, 1) for p in predictions])
        y_true_aligned = y_true[model.sequence_length:len(predictions) + model.sequence_length]

        # Calculate metrics
        accuracy = np.mean(y_pred == y_true_aligned)

        # Simulate trading for other metrics
        returns = features_df["close"].pct_change().dropna().values
        returns_aligned = returns[model.sequence_length:len(predictions) + model.sequence_length]

        strategy_returns = []
        for pred, ret in zip(predictions, returns_aligned):
            if pred == "LONG":
                strategy_returns.append(ret)
            elif pred == "SHORT":
                strategy_returns.append(-ret)
            else:
                strategy_returns.append(0)

        strategy_returns = np.array(strategy_returns)

        # Calculate metrics
        wins = strategy_returns > 0
        losses = strategy_returns < 0
        win_rate = np.sum(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0.5

        avg_win = np.mean(strategy_returns[wins]) if np.any(wins) else 0
        avg_loss = abs(np.mean(strategy_returns[losses])) if np.any(losses) else 1
        profit_factor = (avg_win * np.sum(wins)) / (avg_loss * np.sum(losses)) if np.sum(losses) > 0 else 1.0

        # Sharpe ratio (annualized for hourly data)
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(365 * 24)

        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,
            f1_score=accuracy,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return PerformanceMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=1.0,
        )


def check_models(pipeline: RetrainingPipeline) -> List[dict]:
    """Check which models need retraining."""
    all_symbols = (
        [(s, "crypto") for s in DEFAULT_CRYPTO_SYMBOLS] +
        [(s, "stock") for s in DEFAULT_STOCK_SYMBOLS] +
        [(s, "commodity") for s in DEFAULT_COMMODITY_SYMBOLS]
    )

    results = []

    for symbol, market_type in all_symbols:
        for model_type in DEFAULT_MODEL_TYPES:
            should, trigger, reason = pipeline.should_retrain(symbol, model_type)
            results.append({
                "symbol": symbol,
                "market_type": market_type,
                "model_type": model_type,
                "should_retrain": should,
                "trigger": trigger.value,
                "reason": reason,
            })

            if should:
                logger.info(f"  {symbol} {model_type}: NEEDS RETRAINING - {reason}")
            else:
                logger.debug(f"  {symbol} {model_type}: OK - {reason}")

    return results


def run_retraining(
    pipeline: RetrainingPipeline,
    symbol: Optional[str] = None,
    model_type: Optional[str] = None,
    market_type: Optional[str] = None,
    force: bool = False,
) -> int:
    """Run retraining for models that need it."""
    retrained_count = 0

    if symbol and model_type and market_type:
        # Single model
        models_to_check = [(symbol, market_type, model_type)]
    else:
        # All models
        models_to_check = []
        for s in DEFAULT_CRYPTO_SYMBOLS:
            for m in DEFAULT_MODEL_TYPES:
                models_to_check.append((s, "crypto", m))
        for s in DEFAULT_STOCK_SYMBOLS:
            for m in DEFAULT_MODEL_TYPES:
                models_to_check.append((s, "stock", m))
        for s in DEFAULT_COMMODITY_SYMBOLS:
            for m in DEFAULT_MODEL_TYPES:
                models_to_check.append((s, "commodity", m))

    for sym, mkt, mdl in models_to_check:
        should, trigger, reason = pipeline.should_retrain(sym, mdl)

        if not should and not force:
            continue

        if force:
            trigger = RetrainingTrigger.MANUAL
            reason = "Manual trigger"

        logger.info(f"\n{'='*60}")
        logger.info(f"Retraining {sym} {mdl}")
        logger.info(f"Trigger: {trigger.value} - {reason}")
        logger.info(f"{'='*60}")

        # Get training data
        data = get_training_data(sym, mkt)
        if data is None or len(data) < 500:
            logger.warning(f"Insufficient data for {sym}, skipping")
            continue

        # Run retraining
        result = pipeline.retrain(
            symbol=sym,
            model_type=mdl,
            training_data=data,
            trigger=trigger,
            train_func=lambda df: train_model(mdl, df, sym, mkt),
            validate_func=validate_model,
        )

        if result.success:
            retrained_count += 1
            logger.info(f"SUCCESS: {sym} {mdl}")
            if result.new_metrics:
                logger.info(f"  Accuracy: {result.new_metrics.accuracy:.3f}")
                logger.info(f"  Sharpe: {result.new_metrics.sharpe_ratio:.2f}")
                logger.info(f"  Win Rate: {result.new_metrics.win_rate:.2%}")
            if result.deployed:
                logger.info("  Model deployed!")
                try:
                    send_notification(
                        f"Model Retrained: {sym} {mdl}\n"
                        f"Accuracy: {result.new_metrics.accuracy:.3f}\n"
                        f"Deployed: Yes",
                        "model_retraining"
                    )
                except Exception:
                    pass
        else:
            logger.error(f"FAILED: {sym} {mdl} - {result.error_message}")

    return retrained_count


def run_daemon(pipeline: RetrainingPipeline, interval_seconds: int = 3600):
    """Run as a background daemon, checking periodically."""
    logger.info(f"Starting retraining daemon (interval: {interval_seconds}s)")

    while True:
        try:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Daemon check at {datetime.now()}")
            logger.info(f"{'#'*60}")

            # Check and retrain
            count = run_retraining(pipeline)
            logger.info(f"Retrained {count} models")

        except Exception as e:
            logger.error(f"Daemon error: {e}")

        logger.info(f"Next check in {interval_seconds} seconds")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automated Model Retraining")
    parser.add_argument("--check", action="store_true", help="Check which models need retraining")
    parser.add_argument("--retrain", action="store_true", help="Run retraining")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Daemon check interval (seconds)")
    parser.add_argument("--symbol", type=str, help="Specific symbol to retrain")
    parser.add_argument("--model", type=str, help="Specific model type to retrain")
    parser.add_argument("--market", type=str, choices=["crypto", "stock", "commodity"], help="Market type")
    parser.add_argument("--force", action="store_true", help="Force retraining even if not needed")
    args = parser.parse_args()

    # Ensure log directory exists
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    config = RetrainingConfig(
        min_accuracy=0.52,
        min_sharpe_ratio=0.3,
        max_model_age_days=30,
        drift_threshold=0.25,
    )
    pipeline = RetrainingPipeline(config)

    if args.check:
        logger.info("Checking model retraining status...")
        results = check_models(pipeline)

        needs_retraining = [r for r in results if r["should_retrain"]]
        logger.info(f"\n{len(needs_retraining)} models need retraining")

        if needs_retraining:
            for r in needs_retraining:
                print(f"  - {r['symbol']} {r['model_type']}: {r['reason']}")

    elif args.daemon:
        run_daemon(pipeline, args.interval)

    elif args.retrain:
        count = run_retraining(
            pipeline,
            symbol=args.symbol,
            model_type=args.model,
            market_type=args.market,
            force=args.force,
        )
        logger.info(f"\nRetraining complete. {count} models retrained.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
