#!/usr/bin/env python3
"""ML model training for commodity symbols (Gold, Silver, Oil)."""

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

# Commodity symbol to Yahoo Finance futures mapping
COMMODITY_MAPPING = {
    "XAU/USD": "GC=F",   # Gold futures
    "XAG/USD": "SI=F",   # Silver futures
    "USOIL/USD": "CL=F", # Crude Oil futures
}


def train_commodity(symbol: str, model_types: list = None):
    """Train models for a single commodity symbol."""
    if model_types is None:
        model_types = ["random_forest", "gradient_boosting"]

    print("")
    print("=" * 50)
    print(f"Training models for {symbol}")
    print("=" * 50)

    # Get Yahoo Finance symbol for this commodity
    yf_symbol = COMMODITY_MAPPING.get(symbol)
    if not yf_symbol:
        logger.error(f"No Yahoo Finance mapping for {symbol}")
        return None

    logger.info(f"Fetching data for {yf_symbol} (futures for {symbol})...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period="180d", interval="1h")

    if df.empty:
        logger.warning(f"No hourly data, trying daily data for {yf_symbol}...")
        df = ticker.history(period="2y", interval="1d")

    if df.empty:
        logger.error(f"No data for {symbol}")
        return None

    # Standardize
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"Got {len(df)} candles")

    if len(df) < 100:
        logger.warning(f"Limited data ({len(df)} candles), model may be less accurate")

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
            import traceback
            traceback.print_exc()
            results[model_type] = {"error": str(e)}

    return results


def main():
    # Commodity symbols to train
    symbols = ["XAU/USD", "XAG/USD", "USOIL/USD"]

    # Create models directory
    Path("data/models").mkdir(parents=True, exist_ok=True)

    all_results = {}

    for symbol in symbols:
        results = train_commodity(symbol)
        if results:
            all_results[symbol] = results

    # Summary
    print("")
    print("=" * 50)
    print("COMMODITY TRAINING SUMMARY")
    print("=" * 50)

    for symbol, results in all_results.items():
        yf_symbol = COMMODITY_MAPPING.get(symbol, "N/A")
        print(f"\n{symbol} ({yf_symbol}):")
        for model_type, metrics in results.items():
            if "accuracy" in metrics:
                print(f"  {model_type}: {metrics['accuracy']:.2%} (CV: {metrics['cross_val_mean']:.2%})")
            elif "error" in metrics:
                print(f"  {model_type}: ERROR - {metrics['error']}")

    print("")
    print("=" * 50)
    print("Commodity training complete!")
    print("Models saved to: data/models/")
    print("=" * 50)


if __name__ == "__main__":
    main()
