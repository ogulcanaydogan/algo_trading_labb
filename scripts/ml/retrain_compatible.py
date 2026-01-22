#!/usr/bin/env python3
"""
Retrain ML models using MLPredictor for feature compatibility.

This script ensures models are trained with the same features
that MLSignalGenerator expects at runtime.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.ml.predictor import MLPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, days: int = 60) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol."""
    try:
        yf_symbol = symbol.replace("/", "-")
        if yf_symbol.endswith("-USDT"):
            yf_symbol = yf_symbol.replace("-USDT", "-USD")

        logger.info(f"Fetching {days}d of hourly data for {symbol} ({yf_symbol})")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")

        if df.empty:
            logger.error(f"No data returned for {symbol}")
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def train_symbol(
    symbol: str,
    model_type: str = "gradient_boosting",
    days: int = 60,
    model_dir: str = "data/models",
) -> bool:
    """Train model for a single symbol using MLPredictor."""

    # Fetch data
    df = fetch_ohlcv(symbol, days)
    if df is None or len(df) < 200:
        logger.error(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} candles)")
        return False

    # Create predictor
    predictor = MLPredictor(
        model_type=model_type,
        model_dir=model_dir,
    )

    # Train
    logger.info(f"Training {model_type} model for {symbol}...")
    metrics = predictor.train(df, test_size=0.2, validate=True)

    logger.info(f"Training Results for {symbol}:")
    logger.info(f"  Accuracy: {metrics.accuracy:.2%}")
    logger.info(f"  CV Score: {metrics.cross_val_mean:.2%} +/- {metrics.cross_val_std:.2%}")
    logger.info(f"  Train samples: {metrics.train_samples}")
    logger.info(f"  Test samples: {metrics.test_samples}")
    logger.info(f"  Features used: {len(predictor.feature_names)}")

    # Save model
    symbol_clean = symbol.replace("/", "_")
    model_name = f"{symbol_clean}_{model_type}"
    predictor.save(model_name)

    logger.info(f"Model saved: {model_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrain ML models using MLPredictor for feature compatibility"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["LINK/USDT", "DOT/USDT", "POL/USDT", "ATOM/USDT"],
        help="Symbols to train",
    )
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "random_forest", "xgboost"],
        default="gradient_boosting",
        help="Model type to train",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Days of historical data to use",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models",
        help="Directory to save models",
    )

    args = parser.parse_args()

    success_count = 0
    for symbol in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol}")
        logger.info(f"{'='*60}")

        if train_symbol(symbol, args.model_type, args.days, args.model_dir):
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete: {success_count}/{len(args.symbols)} symbols")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
