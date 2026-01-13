#!/usr/bin/env python3
"""
Walk-Forward Validation Script for Deep Learning Models.

This script performs walk-forward validation on LSTM or Transformer models
to assess their robustness and prevent overfitting.

Usage:
    python tools/run_walk_forward.py --symbol BTC/USDT --model lstm
    python tools/run_walk_forward.py --symbol AAPL --model transformer --source yfinance
    python tools/run_walk_forward.py --symbol ETH/USDT --train-days 120 --test-days 20

Arguments:
    --symbol: Trading symbol (e.g., BTC/USDT, AAPL, SPY)
    --model: Model type to validate (lstm or transformer)
    --source: Data source (ccxt or yfinance)
    --timeframe: Data timeframe (e.g., 1h, 4h, 1d)
    --train-days: Training window size in days
    --test-days: Test window size in days
    --step-days: Step size between windows in days
    --min-samples: Minimum training samples required
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def fetch_data_ccxt(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch historical data using CCXT (for crypto).

    Args:
        symbol: Trading symbol (e.g., BTC/USDT)
        timeframe: Data timeframe
        days: Number of days of historical data

    Returns:
        OHLCV DataFrame with DatetimeIndex
    """
    try:
        from bot.exchange import ExchangeClient
    except ImportError:
        logger.error("ExchangeClient not available. Install ccxt.")
        raise

    logger.info(f"Fetching {days} days of {symbol} data from exchange...")

    # Calculate number of candles needed
    bars_per_day = _get_bars_per_day(timeframe)
    limit = int(days * bars_per_day)

    # Binance has a max limit per request
    max_per_request = 1000

    client = ExchangeClient(exchange_id="binance")
    all_data = []

    while limit > 0:
        fetch_limit = min(limit, max_per_request)
        try:
            df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
            if df.empty:
                break
            all_data.append(df)
            limit -= fetch_limit
        except Exception as e:
            logger.warning(f"Error fetching data: {e}")
            break

    if not all_data:
        raise ValueError(f"Could not fetch data for {symbol}")

    result = pd.concat(all_data)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    logger.info(f"Fetched {len(result)} candles from {result.index[0]} to {result.index[-1]}")
    return result


def fetch_data_yfinance(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch historical data using yfinance (for stocks/ETFs).

    Args:
        symbol: Trading symbol (e.g., AAPL, SPY)
        timeframe: Data timeframe
        days: Number of days of historical data

    Returns:
        OHLCV DataFrame with DatetimeIndex
    """
    try:
        from bot.market_data import YFinanceMarketDataClient
    except ImportError:
        logger.error("YFinanceMarketDataClient not available. Install yfinance.")
        raise

    logger.info(f"Fetching {days} days of {symbol} data from yfinance...")

    client = YFinanceMarketDataClient()
    bars_per_day = _get_bars_per_day(timeframe)
    limit = int(days * bars_per_day)

    df = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def _get_bars_per_day(timeframe: str) -> float:
    """Get number of bars per day for a given timeframe."""
    tf = timeframe.strip().lower()

    if tf.endswith("m"):
        minutes = int(tf[:-1])
        return (24 * 60) / minutes
    elif tf.endswith("h"):
        hours = int(tf[:-1])
        return 24 / hours
    elif tf.endswith("d"):
        days = int(tf[:-1])
        return 1 / days
    else:
        return 24  # Default to hourly


def run_validation(
    symbol: str,
    model_type: str,
    data_source: str,
    timeframe: str,
    train_days: int,
    test_days: int,
    step_days: int,
    min_samples: int,
    data_days: int,
    epochs: int,
    verbose: bool,
) -> None:
    """
    Run walk-forward validation.

    Args:
        symbol: Trading symbol
        model_type: Model type (lstm or transformer)
        data_source: Data source (ccxt or yfinance)
        timeframe: Data timeframe
        train_days: Training window in days
        test_days: Test window in days
        step_days: Step size in days
        min_samples: Minimum training samples
        data_days: Total days of data to fetch
        epochs: Number of training epochs per window
        verbose: Whether to print detailed output
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Model: {model_type.upper()}")
    print(f"Data Source: {data_source}")
    print(f"Timeframe: {timeframe}")
    print(f"Config: train={train_days}d, test={test_days}d, step={step_days}d")
    print("=" * 70 + "\n")

    # Fetch data
    if data_source == "yfinance":
        ohlcv = fetch_data_yfinance(symbol, timeframe, data_days)
    else:
        ohlcv = fetch_data_ccxt(symbol, timeframe, data_days)

    if len(ohlcv) < min_samples:
        logger.error(f"Insufficient data: {len(ohlcv)} < {min_samples} required")
        sys.exit(1)

    # Import validation module
    from bot.walk_forward import WalkForwardConfig, WalkForwardValidator

    # Create config
    config = WalkForwardConfig(
        train_window_days=train_days,
        test_window_days=test_days,
        step_days=step_days,
        min_train_samples=min_samples,
    )

    # Import model classes
    if model_type == "lstm":
        from bot.ml.models.deep_learning.lstm import LSTMConfig, LSTMModel
        model_class = LSTMModel
        model_config = LSTMConfig(epochs=epochs)
    elif model_type == "transformer":
        from bot.ml.models.deep_learning.transformer import TransformerConfig, TransformerModel
        model_class = TransformerModel
        model_config = TransformerConfig(epochs=epochs)
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)

    # Run validation
    validator = WalkForwardValidator(config=config)

    try:
        results = validator.run_validation(
            model_class=model_class,
            data=ohlcv,
            model_config=model_config,
            symbol=symbol,
            timeframe=timeframe,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print summary
    results.print_summary()

    # Print detailed window results
    if verbose:
        print("\n" + "=" * 70)
        print("DETAILED WINDOW RESULTS")
        print("=" * 70)

        for window in results.windows:
            status = "PROFITABLE" if window.is_profitable else "UNPROFITABLE"
            print(f"\nWindow {window.window_id + 1} [{status}]")
            print(f"  Period: {window.test_start} to {window.test_end}")
            print(f"  Samples: {window.test_samples}")
            print(f"  Accuracy: {window.test_metrics.accuracy:.4f}")
            print(f"  Sharpe Ratio: {window.test_metrics.sharpe_ratio:.4f}")
            print(f"  Profit Factor: {window.test_metrics.profit_factor:.4f}")
            print(f"  Win Rate: {window.test_metrics.win_rate:.4f}")
            print(f"  Training Time: {window.training_time_seconds:.1f}s")

    # Print recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if results.is_robust:
        print("\nThe model demonstrates ROBUST performance:")
        print("  - Consistent profitability across validation windows")
        print("  - Low overfitting (train/test gap is acceptable)")
        print("  - Stable performance metrics")
        print("\n  RECOMMENDATION: Model is suitable for paper trading evaluation.")
    else:
        issues = []
        if results.robustness_score < 0.6:
            issues.append("  - Low robustness: Inconsistent profitability across windows")
        if results.consistency_score < 0.5:
            issues.append("  - Low consistency: High variance in performance metrics")
        if results.overfitting_score > 0.4:
            issues.append("  - High overfitting: Large gap between training and test performance")

        print("\nThe model shows CONCERNS:")
        for issue in issues:
            print(issue)
        print("\n  RECOMMENDATION: Further tuning needed before deployment.")
        print("  Consider:")
        print("    - More regularization (dropout, weight decay)")
        print("    - Different architecture (layers, hidden size)")
        print("    - More training data")
        print("    - Different feature engineering")

    print("\n" + "=" * 70)
    print(f"Results saved to: data/walk_forward_results/")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation for deep learning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTC/USDT, AAPL)",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "transformer"],
        default="lstm",
        help="Model type to validate",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["ccxt", "yfinance"],
        default="ccxt",
        help="Data source (ccxt for crypto, yfinance for stocks)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Data timeframe (e.g., 1h, 4h, 1d)",
    )

    parser.add_argument(
        "--train-days",
        type=int,
        default=180,
        help="Training window size in days",
    )

    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Test window size in days",
    )

    parser.add_argument(
        "--step-days",
        type=int,
        default=30,
        help="Step size between windows in days",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Minimum training samples required",
    )

    parser.add_argument(
        "--data-days",
        type=int,
        default=730,
        help="Total days of historical data to fetch",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per window",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Run validation
    run_validation(
        symbol=args.symbol,
        model_type=args.model,
        data_source=args.source,
        timeframe=args.timeframe,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_samples=args.min_samples,
        data_days=args.data_days,
        epochs=args.epochs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
