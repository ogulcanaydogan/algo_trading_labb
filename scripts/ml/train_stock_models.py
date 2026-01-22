#!/usr/bin/env python3
"""
Train ML models using Alpaca data for US stocks.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from bot.ml.predictor import MLPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data for US stocks using yfinance."""
    logger.info(f"Fetching {days} days of hourly data for {symbol}...")

    try:
        # yfinance format doesn't need /USD suffix
        ticker_symbol = symbol.replace('/USD', '')
        ticker = yf.Ticker(ticker_symbol)

        # Fetch hourly data (yfinance limits to ~2 years for hourly)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=min(days, 730))  # Max 2 years for hourly

        df = ticker.history(
            start=start_date,
            end=end_date,
            interval="1h"
        )

        if df.empty:
            logger.warning(f"  No data returned for {symbol}, trying daily data...")
            # Fallback to daily data with more history
            df = ticker.history(period=f"{days}d", interval="1d")
            if df.empty:
                return pd.DataFrame()

        # Rename columns to lowercase (expected by feature engineer)
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"  Missing column {col} for {symbol}")
                return pd.DataFrame()

        df = df[required_cols]
        df = df.dropna()

        logger.info(f"  Got {len(df)} candles for {symbol} from {df.index[0]} to {df.index[-1]}")
        return df

    except Exception as e:
        logger.error(f"  Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()


def train_model_for_symbol(
    symbol: str,
    ohlcv: pd.DataFrame,
    model_types: list = ["random_forest"],
    save_dir: str = "data/models"
) -> dict:
    """Train ML models for a symbol."""
    results = {}
    symbol_clean = symbol.replace('/', '_')

    if len(ohlcv) < 500:
        logger.warning(f"Insufficient data for {symbol}: {len(ohlcv)} candles (need 500+)")
        return results

    for model_type in model_types:
        logger.info(f"Training {model_type} model...")

        try:
            predictor = MLPredictor(
                model_type=model_type,
                model_dir=save_dir
            )

            # Train using OHLCV data (predictor handles feature extraction internally)
            metrics = predictor.train(
                ohlcv=ohlcv,
                test_size=0.2,
                validate=True,
                report_data_quality=True,
                report_dir="data/reports",
                symbol=symbol
            )

            logger.info(f"Training complete. Accuracy: {metrics.accuracy:.4f}, CV: {metrics.cross_val_mean:.4f} +/- {metrics.cross_val_std:.4f}")

            # Save model with symbol name
            model_name = f"{symbol_clean}_{model_type}"
            predictor.save(name=model_name)
            logger.info(f"Model saved to {save_dir}/{model_name}")

            results[model_type] = {
                'accuracy': metrics.accuracy,
                'cv_mean': metrics.cross_val_mean,
                'cv_std': metrics.cross_val_std
            }

        except Exception as e:
            logger.error(f"Training failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {'error': str(e)}

    return results


async def main():
    parser = argparse.ArgumentParser(description="Train stock models using market data")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'AMD'],
        help='Stock symbols to train'
    )
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--model-types', nargs='+', default=['random_forest'], help='Model types to train')

    args = parser.parse_args()

    all_results = {}

    for symbol in args.symbols:
        print(f"\n{'='*40}")
        print(f"Processing {symbol}")
        print('='*40)

        # Fetch data
        ohlcv = fetch_stock_data(symbol, args.days)

        if ohlcv.empty:
            logger.error(f"No data available for {symbol}, skipping...")
            continue

        # Train models - use symbol with /USD suffix for consistency
        symbol_normalized = f"{symbol}/USD"
        results = train_model_for_symbol(
            symbol_normalized,
            ohlcv,
            model_types=args.model_types
        )

        if results:
            all_results[symbol] = results

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        for model_type, metrics in results.items():
            if 'error' in metrics:
                print(f"  {model_type}: ERROR - {metrics['error']}")
            else:
                print(f"  {model_type}: {metrics['accuracy']*100:.2f}% (CV: {metrics['cv_mean']*100:.2f}%)")

    print("\n" + "="*60)
    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
