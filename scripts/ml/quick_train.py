#!/usr/bin/env python3
"""Quick ML model training for BTC and ETH."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from bot.ml.predictor import MLPredictor


def train_symbol(symbol: str, model_types: list = None):
    """Train models for a single symbol."""
    if model_types is None:
        model_types = ["random_forest", "gradient_boosting"]

    print("")
    print("=" * 50)
    print(f"Training models for {symbol}")
    print("=" * 50)

    # Fetch data
    yf_symbol = symbol.replace("/USDT", "-USD")
    logger.info(f"Fetching data for {yf_symbol}...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="180d", interval="1h")

    if df.empty:
        logger.error(f"No data for {symbol}")
        return None

    # Standardize
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"Got {len(df)} candles")

    results = {}

    for model_type in model_types:
        try:
            logger.info(f"Training {model_type}...")

            predictor = MLPredictor(
                model_type=model_type,
                model_dir="data/models",
            )

            metrics = predictor.train(df)

            model_name = f"{symbol.replace('/', '_')}_{model_type}"
            predictor.save(model_name)

            logger.info(f"  Accuracy: {metrics.accuracy:.2%}")
            logger.info(f"  Cross-val: {metrics.cross_val_mean:.2%} (+/- {metrics.cross_val_std:.2%})")
            logger.info(f"  Saved as: {model_name}")

            results[model_type] = {
                "accuracy": metrics.accuracy,
                "cross_val_mean": metrics.cross_val_mean,
                "model_name": model_name,
            }

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results


def main():
    # All crypto symbols to train
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

    # Create models directory
    Path("data/models").mkdir(parents=True, exist_ok=True)

    all_results = {}

    for symbol in symbols:
        results = train_symbol(symbol)
        if results:
            all_results[symbol] = results

    # Summary
    print("")
    print("=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for model_type, metrics in results.items():
            if "accuracy" in metrics:
                print(f"  {model_type}: {metrics['accuracy']:.2%}")
            elif "error" in metrics:
                print(f"  {model_type}: ERROR - {metrics['error']}")

    print("")
    print("=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
