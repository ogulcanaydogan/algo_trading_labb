#!/usr/bin/env python3
"""
Training Data Fetcher - Maximize data collection within API rate limits.

This script intelligently fetches historical market data while respecting
API rate limits. Perfect for building ML training datasets.

Usage:
    # Show available capacity
    python run_data_fetcher.py --capacity

    # Estimate fetch time
    python run_data_fetcher.py --estimate --symbols BTC/USDT,ETH/USDT --days 365

    # Fetch 1 year of hourly data for top cryptos
    python run_data_fetcher.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h --days 365

    # Continuous mode - keeps fetching and updating
    python run_data_fetcher.py --continuous --symbols BTC/USDT,ETH/USDT

    # Fetch from different provider
    python run_data_fetcher.py --provider coinbase --symbols BTC-USD,ETH-USD
"""

import argparse
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.data_fetcher import SmartDataFetcher, RATE_LIMITS, create_training_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default symbols for each provider
DEFAULT_SYMBOLS = {
    "binance": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT"
    ],
    "coinbase": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOT-USD"
    ],
    "kraken": [
        "XXBTZUSD", "XETHZUSD", "SOLUSD", "DOTUSD", "ADAUSD"
    ],
    "yahoo": [
        "BTC-USD", "ETH-USD", "GC=F", "SI=F", "CL=F"  # Crypto + Gold + Silver + Oil
    ],
}

# Global fetcher for signal handling
_fetcher = None


def signal_handler(signum, frame):
    """Handle CTRL+C gracefully."""
    print("\n\nReceived interrupt signal. Stopping fetch...")
    if _fetcher:
        _fetcher.stop()
    sys.exit(0)


def print_rate_limits():
    """Print rate limits for all providers."""
    print("\n" + "=" * 70)
    print("PROVIDER RATE LIMITS")
    print("=" * 70)

    for provider, config in RATE_LIMITS.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)
        print(f"  Requests/minute:  {config.requests_per_minute:,}")
        print(f"  Requests/hour:    {config.requests_per_hour:,}")
        print(f"  Requests/day:     {config.requests_per_day:,}")
        print(f"  Records/request:  {config.max_records_per_request:,}")
        print(f"  Min interval:     {config.min_interval_seconds:.2f}s")
        print(f"  Safe req/min:     {config.safe_requests_per_minute}")

    print("\n" + "=" * 70)


def print_capacity(fetcher: SmartDataFetcher):
    """Print current available capacity."""
    cap = fetcher.get_available_capacity()

    print("\n" + "=" * 70)
    print(f"AVAILABLE CAPACITY - {cap['provider'].upper()}")
    print("=" * 70)

    print("\nLimits:")
    print(f"  Per minute: {cap['limits']['per_minute']:,}")
    print(f"  Per hour:   {cap['limits']['per_hour']:,}")
    print(f"  Per day:    {cap['limits']['per_day']:,}")

    print("\nUsed:")
    print(f"  This minute: {cap['used']['this_minute']}")
    print(f"  This hour:   {cap['used']['this_hour']}")
    print(f"  Today:       {cap['used']['today']}")

    print("\nAvailable:")
    print(f"  This minute: {cap['available']['this_minute']}")
    print(f"  This hour:   {cap['available']['this_hour']}")
    print(f"  Today:       {cap['available']['today']}")

    print("\nEstimated Records Available:")
    print(f"  This minute: {cap['estimated_records_available']['this_minute']:,}")
    print(f"  This hour:   {cap['estimated_records_available']['this_hour']:,}")
    print(f"  Today:       {cap['estimated_records_available']['today']:,}")

    print("\n" + "=" * 70)


def print_estimate(fetcher: SmartDataFetcher, symbols: list, timeframe: str, days: int):
    """Print fetch time estimate."""
    est = fetcher.estimate_fetch_time(symbols, timeframe, days)

    print("\n" + "=" * 70)
    print("FETCH TIME ESTIMATE")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Timeframe:  {timeframe}")
    print(f"  Days:       {days}")

    print(f"\nData Volume:")
    print(f"  Total candles:     {est['total_candles']:,}")
    print(f"  Requests needed:   {est['requests_needed']:,}")

    print(f"\nEstimated Time:")
    print(f"  Seconds: {est['estimated_time']['seconds']:,}")
    print(f"  Minutes: {est['estimated_time']['minutes']:.1f}")
    print(f"  Hours:   {est['estimated_time']['hours']:.2f}")

    print("\n" + "=" * 70)

    return est


def print_progress(fetcher: SmartDataFetcher):
    """Print current fetch progress."""
    stats = fetcher.get_stats()

    print("\n" + "-" * 50)
    print("FETCH PROGRESS")
    print("-" * 50)

    print(f"Total requests:    {stats['total_requests']:,}")
    print(f"Total records:     {stats['total_records']:,}")
    print(f"Symbols completed: {stats['symbols_completed']}")
    print(f"Errors:            {stats['errors']}")
    print(f"Runtime:           {stats['runtime_minutes']:.1f} minutes")

    if stats['progress']:
        print("\nPer-symbol progress:")
        for key, p in stats['progress'].items():
            status = "DONE" if p['is_complete'] else f"{p['progress_pct']:.1f}%"
            print(f"  {p['symbol']} ({p['timeframe']}): {status} - {p['records']:,} records")


def run_fetch(args):
    """Run the data fetching process."""
    global _fetcher

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_SYMBOLS.get(args.provider, DEFAULT_SYMBOLS["binance"])[:5]

    # Parse timeframes
    timeframes = [t.strip() for t in args.timeframe.split(",")]

    # Create fetcher
    _fetcher = SmartDataFetcher(
        provider=args.provider,
        data_dir=args.output_dir,
        max_retries=args.max_retries,
    )

    # Show capacity
    print_capacity(_fetcher)

    # Show estimate
    for tf in timeframes:
        print_estimate(_fetcher, symbols, tf, args.days)

    # Confirm
    if not args.yes:
        response = input("\nProceed with fetch? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print("\n" + "=" * 70)
    print("STARTING DATA FETCH")
    print("=" * 70)
    print(f"Press CTRL+C to stop gracefully\n")

    start_date = datetime.now() - timedelta(days=args.days)

    if args.continuous:
        # Continuous mode
        def on_data(symbol, timeframe, df):
            logger.info(f"Fetched {len(df)} records for {symbol} ({timeframe})")

        try:
            _fetcher.schedule_continuous_fetch(symbols, timeframes, callback=on_data)
        except KeyboardInterrupt:
            pass
    else:
        # One-time fetch
        for tf in timeframes:
            logger.info(f"\nFetching timeframe: {tf}")
            _fetcher.fetch_multiple_symbols(symbols, tf, start_date)

    # Final stats
    print_progress(_fetcher)

    # Combine datasets if requested
    if args.combine:
        print("\nCombining datasets...")
        df = create_training_dataset(args.output_dir)
        print(f"Combined dataset: {len(df):,} records")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch maximum training data within API rate limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data selection
    parser.add_argument(
        "--symbols", "-s",
        help="Comma-separated trading pairs (e.g., BTC/USDT,ETH/USDT)"
    )
    parser.add_argument(
        "--timeframe", "-t",
        default="1h",
        help="Candle timeframe(s), comma-separated (default: 1h)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Days of history to fetch (default: 365)"
    )

    # Provider
    parser.add_argument(
        "--provider", "-p",
        default="binance",
        choices=list(RATE_LIMITS.keys()),
        help="Data provider (default: binance)"
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        default="data/training",
        help="Output directory (default: data/training)"
    )

    # Modes
    parser.add_argument(
        "--capacity",
        action="store_true",
        help="Show available API capacity and exit"
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate fetch time and exit"
    )
    parser.add_argument(
        "--rate-limits",
        action="store_true",
        help="Show rate limits for all providers"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously, updating data periodically"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all data into single training file after fetch"
    )

    # Options
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per symbol (default: 3)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Handle info modes
    if args.rate_limits:
        print_rate_limits()
        return

    if args.capacity:
        fetcher = SmartDataFetcher(provider=args.provider)
        print_capacity(fetcher)
        return

    if args.estimate:
        fetcher = SmartDataFetcher(provider=args.provider)
        symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SYMBOLS.get(args.provider, DEFAULT_SYMBOLS["binance"])[:5]
        print_estimate(fetcher, symbols, args.timeframe, args.days)
        return

    # Run fetch
    run_fetch(args)


if __name__ == "__main__":
    main()
