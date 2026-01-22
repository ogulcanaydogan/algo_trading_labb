#!/usr/bin/env python3
"""
Train ML models using OANDA data for forex, commodities, and indices.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from bot.oanda_adapter import create_oanda_adapter
from bot.ml.predictor import MLPredictor
from bot.ml.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_oanda_data(adapter, symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data from OANDA."""
    logger.info(f"Fetching {days} days of data for {symbol} from OANDA...")

    # OANDA limits to 5000 candles per request for H1
    # For 365 days of hourly data, we need ~8760 candles
    # So we'll fetch in chunks

    all_bars = []
    candles_per_request = 4000
    total_candles_needed = days * 24

    bars = await adapter.get_historical_data(
        symbol=symbol,
        timeframe="1h",
        limit=min(candles_per_request, total_candles_needed)
    )

    if bars:
        all_bars.extend(bars)
        logger.info(f"  Got {len(bars)} candles for {symbol}")

    if not all_bars:
        logger.warning(f"  No data returned for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    # Keep lowercase column names (expected by feature engineer)
    # df already has: open, high, low, close, volume

    logger.info(f"  Final dataset: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def train_model_for_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    model_types: list = ["random_forest"],
    save_dir: str = "data/models"
) -> dict:
    """Train ML models for a symbol."""
    results = {}
    symbol_clean = symbol.replace('/', '_')

    for model_type in model_types:
        logger.info(f"Training {model_type} for {symbol}...")

        try:
            predictor = MLPredictor(model_type=model_type, model_dir=save_dir)
            metrics = predictor.train(ohlcv, symbol=symbol)

            # Save with symbol name
            save_name = f"{symbol_clean}_{model_type}"
            predictor.save(name=save_name)
            logger.info(f"  Saved model as: {save_name}")

            results[model_type] = {
                'accuracy': metrics.accuracy,
                'cv_mean': metrics.cross_val_mean,
                'cv_std': metrics.cross_val_std
            }

            logger.info(f"  {model_type}: Accuracy={metrics.accuracy:.2%}, CV={metrics.cross_val_mean:.2%} +/- {metrics.cross_val_std:.2%}")

        except Exception as e:
            logger.error(f"  Failed to train {model_type}: {e}")
            results[model_type] = {'error': str(e)}

    return results


async def main():
    parser = argparse.ArgumentParser(description="Train ML models using OANDA data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["XAU/USD", "EUR/USD", "GBP/USD", "WTICO/USD", "SPX500/USD", "NAS100/USD"],
        help="Symbols to train"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest"],
        help="Model types to train"
    )
    parser.add_argument(
        "--save-dir",
        default="data/models",
        help="Directory to save models"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OANDA ML MODEL TRAINING")
    print("=" * 60)
    print(f"Symbols: {args.symbols}")
    print(f"Days: {args.days}")
    print(f"Models: {args.models}")
    print("=" * 60)

    # Create OANDA adapter
    adapter = create_oanda_adapter(environment='live')
    if not adapter:
        logger.error("Failed to create OANDA adapter")
        return

    # Connect
    success = await adapter.connect()
    if not success:
        logger.error("Failed to connect to OANDA")
        return

    # Train models for each symbol
    all_results = {}

    for symbol in args.symbols:
        print(f"\n{'=' * 40}")
        print(f"Processing {symbol}")
        print("=" * 40)

        # Fetch data
        ohlcv = await fetch_oanda_data(adapter, symbol, args.days)

        if ohlcv.empty or len(ohlcv) < 100:
            logger.warning(f"Skipping {symbol} - insufficient data ({len(ohlcv)} candles)")
            continue

        # Train models
        results = train_model_for_symbol(
            symbol=symbol,
            ohlcv=ohlcv,
            model_types=args.models,
            save_dir=args.save_dir
        )

        all_results[symbol] = results

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for model_type, metrics in results.items():
            if 'error' in metrics:
                print(f"  {model_type}: FAILED - {metrics['error']}")
            else:
                print(f"  {model_type}: {metrics['accuracy']:.2%} (CV: {metrics['cv_mean']:.2%})")

    print("\n" + "=" * 60)
    print("Training complete!")

    # Disconnect
    await adapter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
