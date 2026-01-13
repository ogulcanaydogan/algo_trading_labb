"""
Smart Data Fetcher with Rate Limit Management.

Automatically fetches maximum data within API rate limits
for ML model training. Supports multiple providers with
intelligent scheduling and incremental storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import threading

import ccxt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported data providers."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO = "yahoo"
    POLYGON = "polygon"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a provider."""
    provider: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    weight_per_request: int = 1  # Some APIs use weighted limits
    max_records_per_request: int = 1000
    cooldown_seconds: float = 0.1  # Minimum delay between requests

    @property
    def safe_requests_per_minute(self) -> int:
        """Get safe limit (80% of actual to avoid hitting limits)."""
        return int(self.requests_per_minute * 0.8)

    @property
    def min_interval_seconds(self) -> float:
        """Minimum seconds between requests to stay within limits."""
        return max(60 / self.safe_requests_per_minute, self.cooldown_seconds)


# Provider rate limit configurations
RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "binance": RateLimitConfig(
        provider="binance",
        requests_per_minute=1200,  # Weight limit
        requests_per_hour=72000,
        requests_per_day=1728000,
        weight_per_request=1,
        max_records_per_request=1000,
        cooldown_seconds=0.05,
    ),
    "coinbase": RateLimitConfig(
        provider="coinbase",
        requests_per_minute=10,
        requests_per_hour=600,
        requests_per_day=10000,
        max_records_per_request=300,
        cooldown_seconds=6,
    ),
    "kraken": RateLimitConfig(
        provider="kraken",
        requests_per_minute=15,
        requests_per_hour=900,
        requests_per_day=21600,
        max_records_per_request=720,
        cooldown_seconds=4,
    ),
    "yahoo": RateLimitConfig(
        provider="yahoo",
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=48000,
        max_records_per_request=10000,  # Yahoo allows large date ranges
        cooldown_seconds=0.6,
    ),
    "polygon": RateLimitConfig(
        provider="polygon",
        requests_per_minute=5,  # Free tier
        requests_per_hour=300,
        requests_per_day=5000,
        max_records_per_request=50000,
        cooldown_seconds=12,
    ),
}


@dataclass
class FetchProgress:
    """Track fetch progress for a symbol."""
    symbol: str
    provider: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    current_date: datetime
    records_fetched: int = 0
    requests_made: int = 0
    last_fetch_time: Optional[datetime] = None
    is_complete: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        total_seconds = (self.end_date - self.start_date).total_seconds()
        done_seconds = (self.current_date - self.start_date).total_seconds()
        return min(100, (done_seconds / total_seconds * 100)) if total_seconds > 0 else 0


@dataclass
class FetchStats:
    """Overall fetching statistics."""
    total_requests: int = 0
    total_records: int = 0
    requests_today: int = 0
    requests_this_hour: int = 0
    requests_this_minute: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_request_time: Optional[datetime] = None
    errors: int = 0
    symbols_completed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_records": self.total_records,
            "requests_today": self.requests_today,
            "requests_this_hour": self.requests_this_hour,
            "requests_this_minute": self.requests_this_minute,
            "runtime_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "errors": self.errors,
            "symbols_completed": self.symbols_completed,
        }


class SmartDataFetcher:
    """
    Intelligent data fetcher that maximizes data collection
    within API rate limits.

    Features:
    - Automatic rate limit tracking
    - Incremental data storage
    - Resume capability
    - Multi-symbol parallel fetching
    - Progress tracking
    """

    def __init__(
        self,
        provider: str = "binance",
        data_dir: str = "data/training",
        max_retries: int = 3,
    ):
        self.provider = provider
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        self.rate_config = RATE_LIMITS.get(provider, RATE_LIMITS["binance"])
        self.stats = FetchStats()
        self.progress: Dict[str, FetchProgress] = {}

        self._request_times: List[datetime] = []
        self._lock = threading.Lock()
        self._stop_flag = False

        # Initialize exchange
        self._exchange = self._init_exchange()

        # Load progress if exists
        self._load_progress()

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange."""
        exchange_class = getattr(ccxt, self.provider, ccxt.binance)
        return exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    def _load_progress(self) -> None:
        """Load fetch progress from disk."""
        progress_file = self.data_dir / "fetch_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file) as f:
                    data = json.load(f)
                for key, p in data.items():
                    self.progress[key] = FetchProgress(
                        symbol=p["symbol"],
                        provider=p["provider"],
                        timeframe=p["timeframe"],
                        start_date=datetime.fromisoformat(p["start_date"]),
                        end_date=datetime.fromisoformat(p["end_date"]),
                        current_date=datetime.fromisoformat(p["current_date"]),
                        records_fetched=p["records_fetched"],
                        requests_made=p["requests_made"],
                        is_complete=p["is_complete"],
                    )
                logger.info(f"Loaded progress for {len(self.progress)} fetch jobs")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")

    def _save_progress(self) -> None:
        """Save fetch progress to disk."""
        progress_file = self.data_dir / "fetch_progress.json"
        data = {}
        for key, p in self.progress.items():
            data[key] = {
                "symbol": p.symbol,
                "provider": p.provider,
                "timeframe": p.timeframe,
                "start_date": p.start_date.isoformat(),
                "end_date": p.end_date.isoformat(),
                "current_date": p.current_date.isoformat(),
                "records_fetched": p.records_fetched,
                "requests_made": p.requests_made,
                "is_complete": p.is_complete,
            }
        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def _can_make_request(self) -> bool:
        """Check if we can make another request within rate limits."""
        now = datetime.now()

        with self._lock:
            # Clean old request times
            self._request_times = [
                t for t in self._request_times
                if (now - t).total_seconds() < 86400  # Keep last 24h
            ]

            # Count requests in different windows
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            requests_minute = sum(1 for t in self._request_times if t > minute_ago)
            requests_hour = sum(1 for t in self._request_times if t > hour_ago)
            requests_day = sum(1 for t in self._request_times if t > day_ago)

            # Update stats
            self.stats.requests_this_minute = requests_minute
            self.stats.requests_this_hour = requests_hour
            self.stats.requests_today = requests_day

            # Check limits
            if requests_minute >= self.rate_config.safe_requests_per_minute:
                return False
            if requests_hour >= int(self.rate_config.requests_per_hour * 0.8):
                return False
            if requests_day >= int(self.rate_config.requests_per_day * 0.8):
                return False

            return True

    def _record_request(self) -> None:
        """Record that a request was made."""
        with self._lock:
            now = datetime.now()
            self._request_times.append(now)
            self.stats.total_requests += 1
            self.stats.last_request_time = now

    def _wait_for_rate_limit(self) -> float:
        """Wait until we can make another request. Returns seconds waited."""
        waited = 0.0
        while not self._can_make_request() and not self._stop_flag:
            time.sleep(1)
            waited += 1
            if waited % 30 == 0:
                logger.info(f"Rate limit cooldown: waited {waited:.0f}s")
        return waited

    def get_available_capacity(self) -> Dict[str, Any]:
        """Get current available request capacity."""
        now = datetime.now()

        with self._lock:
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            requests_minute = sum(1 for t in self._request_times if t > minute_ago)
            requests_hour = sum(1 for t in self._request_times if t > hour_ago)
            requests_day = sum(1 for t in self._request_times if t > day_ago)

        return {
            "provider": self.provider,
            "limits": {
                "per_minute": self.rate_config.requests_per_minute,
                "per_hour": self.rate_config.requests_per_hour,
                "per_day": self.rate_config.requests_per_day,
            },
            "used": {
                "this_minute": requests_minute,
                "this_hour": requests_hour,
                "today": requests_day,
            },
            "available": {
                "this_minute": max(0, self.rate_config.safe_requests_per_minute - requests_minute),
                "this_hour": max(0, int(self.rate_config.requests_per_hour * 0.8) - requests_hour),
                "today": max(0, int(self.rate_config.requests_per_day * 0.8) - requests_day),
            },
            "records_per_request": self.rate_config.max_records_per_request,
            "estimated_records_available": {
                "this_minute": max(0, self.rate_config.safe_requests_per_minute - requests_minute) * self.rate_config.max_records_per_request,
                "this_hour": max(0, int(self.rate_config.requests_per_hour * 0.8) - requests_hour) * self.rate_config.max_records_per_request,
                "today": max(0, int(self.rate_config.requests_per_day * 0.8) - requests_day) * self.rate_config.max_records_per_request,
            },
        }

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_incrementally: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol within rate limits.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date (default: 2 years ago)
            end_date: End date (default: now)
            save_incrementally: Save data after each batch

        Returns:
            DataFrame with OHLCV data
        """
        # Set defaults
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years

        # Create progress key
        progress_key = f"{symbol}_{timeframe}"

        # Check for existing progress
        if progress_key in self.progress and not self.progress[progress_key].is_complete:
            progress = self.progress[progress_key]
            start_date = progress.current_date
            logger.info(f"Resuming fetch for {symbol} from {start_date}")
        else:
            progress = FetchProgress(
                symbol=symbol,
                provider=self.provider,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                current_date=start_date,
            )
            self.progress[progress_key] = progress

        # Calculate timeframe in milliseconds
        tf_minutes = self._timeframe_to_minutes(timeframe)
        tf_ms = tf_minutes * 60 * 1000

        all_data = []
        current_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        # Load existing data if any
        data_file = self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        if data_file.exists():
            try:
                existing_df = pd.read_parquet(data_file)
                all_data = existing_df.values.tolist()
                if len(existing_df) > 0:
                    last_ts = int(existing_df["timestamp"].max())
                    current_ts = max(current_ts, last_ts + tf_ms)
                    logger.info(f"Loaded {len(existing_df)} existing records")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")

        logger.info(f"Fetching {symbol} {timeframe} from {datetime.fromtimestamp(current_ts/1000)} to {end_date}")

        batch_count = 0
        while current_ts < end_ts and not self._stop_flag:
            # Wait for rate limit
            self._wait_for_rate_limit()

            if self._stop_flag:
                break

            # Fetch batch
            try:
                self._record_request()
                ohlcv = self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_ts,
                    limit=self.rate_config.max_records_per_request,
                )

                if not ohlcv:
                    logger.info(f"No more data for {symbol}")
                    break

                all_data.extend(ohlcv)
                progress.records_fetched += len(ohlcv)
                progress.requests_made += 1
                self.stats.total_records += len(ohlcv)

                # Update current timestamp
                last_ts = ohlcv[-1][0]
                current_ts = last_ts + tf_ms
                progress.current_date = datetime.fromtimestamp(current_ts / 1000)

                batch_count += 1

                # Log progress
                if batch_count % 10 == 0:
                    logger.info(
                        f"{symbol}: {progress.progress_pct:.1f}% complete, "
                        f"{progress.records_fetched} records, "
                        f"{progress.requests_made} requests"
                    )

                # Save incrementally
                if save_incrementally and batch_count % 50 == 0:
                    self._save_data(all_data, symbol, timeframe)
                    self._save_progress()

                # Respect minimum interval
                time.sleep(self.rate_config.min_interval_seconds)

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit hit, cooling down: {e}")
                progress.error_count += 1
                progress.last_error = str(e)
                time.sleep(60)  # Wait a minute

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                progress.error_count += 1
                progress.last_error = str(e)
                self.stats.errors += 1

                if progress.error_count >= self.max_retries:
                    logger.error(f"Max retries reached for {symbol}")
                    break

                time.sleep(5)

        # Mark complete if we reached end
        if current_ts >= end_ts:
            progress.is_complete = True
            self.stats.symbols_completed += 1

        # Final save
        df = self._save_data(all_data, symbol, timeframe)
        self._save_progress()

        logger.info(
            f"Completed {symbol}: {len(df)} total records, "
            f"{progress.requests_made} requests"
        )

        return df

    def _save_data(
        self,
        data: List,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Save OHLCV data to parquet file."""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Convert timestamp
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Save
        file_path = self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        df.to_parquet(file_path, index=False)

        logger.debug(f"Saved {len(df)} records to {file_path}")
        return df

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        multipliers = {
            "m": 1,
            "h": 60,
            "d": 1440,
            "w": 10080,
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 1)

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parallel: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            parallel: Whether to use threading (be careful with rate limits)

        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}

        for symbol in symbols:
            if self._stop_flag:
                break

            logger.info(f"Starting fetch for {symbol}")
            df = self.fetch_ohlcv(
                symbol, timeframe, start_date, end_date
            )
            results[symbol] = df

        return results

    def schedule_continuous_fetch(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1h", "4h", "1d"],
        callback: Optional[Callable[[str, str, pd.DataFrame], None]] = None,
    ) -> None:
        """
        Schedule continuous data fetching to maximize data collection.

        This will run indefinitely, fetching data for all symbols
        and timeframes while respecting rate limits.

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes to fetch
            callback: Optional callback when new data is fetched
        """
        logger.info(
            f"Starting continuous fetch for {len(symbols)} symbols, "
            f"{len(timeframes)} timeframes"
        )

        while not self._stop_flag:
            for tf in timeframes:
                for symbol in symbols:
                    if self._stop_flag:
                        break

                    progress_key = f"{symbol}_{tf}"

                    # Skip if complete and recent
                    if progress_key in self.progress:
                        p = self.progress[progress_key]
                        if p.is_complete:
                            # Update with recent data
                            p.end_date = datetime.now()
                            p.current_date = p.end_date - timedelta(hours=1)
                            p.is_complete = False

                    try:
                        df = self.fetch_ohlcv(symbol, tf)

                        if callback and len(df) > 0:
                            callback(symbol, tf, df)

                    except Exception as e:
                        logger.error(f"Error in continuous fetch: {e}")
                        time.sleep(10)

            # After completing all symbols, wait before next round
            if not self._stop_flag:
                logger.info("Completed fetch round, waiting 5 minutes...")
                time.sleep(300)

    def stop(self) -> None:
        """Stop any ongoing fetch operations."""
        self._stop_flag = True
        logger.info("Stop signal sent")

    def get_stats(self) -> Dict[str, Any]:
        """Get current fetch statistics."""
        return {
            **self.stats.to_dict(),
            "capacity": self.get_available_capacity(),
            "progress": {
                k: {
                    "symbol": p.symbol,
                    "timeframe": p.timeframe,
                    "progress_pct": round(p.progress_pct, 1),
                    "records": p.records_fetched,
                    "requests": p.requests_made,
                    "is_complete": p.is_complete,
                }
                for k, p in self.progress.items()
            },
        }

    def estimate_fetch_time(
        self,
        symbols: List[str],
        timeframe: str,
        days: int,
    ) -> Dict[str, Any]:
        """
        Estimate time to fetch historical data.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            Estimation details
        """
        tf_minutes = self._timeframe_to_minutes(timeframe)
        candles_per_day = 1440 / tf_minutes
        total_candles = int(candles_per_day * days * len(symbols))

        requests_needed = total_candles / self.rate_config.max_records_per_request

        # Time based on rate limits
        time_per_request = self.rate_config.min_interval_seconds
        total_seconds = requests_needed * time_per_request

        # Account for hourly/daily limits
        if requests_needed > self.rate_config.safe_requests_per_minute:
            # Will need to spread across minutes
            minutes_needed = requests_needed / self.rate_config.safe_requests_per_minute
            total_seconds = max(total_seconds, minutes_needed * 60)

        if requests_needed > self.rate_config.requests_per_hour * 0.8:
            # Will need to spread across hours
            hours_needed = requests_needed / (self.rate_config.requests_per_hour * 0.8)
            total_seconds = max(total_seconds, hours_needed * 3600)

        return {
            "symbols": len(symbols),
            "timeframe": timeframe,
            "days": days,
            "total_candles": total_candles,
            "requests_needed": int(requests_needed),
            "estimated_time": {
                "seconds": int(total_seconds),
                "minutes": round(total_seconds / 60, 1),
                "hours": round(total_seconds / 3600, 2),
            },
            "rate_limits": {
                "requests_per_minute": self.rate_config.safe_requests_per_minute,
                "records_per_request": self.rate_config.max_records_per_request,
            },
        }


def create_training_dataset(
    data_dir: str = "data/training",
    output_file: str = "data/training/combined_training.parquet",
) -> pd.DataFrame:
    """
    Combine all fetched data into a single training dataset.

    Args:
        data_dir: Directory containing fetched data
        output_file: Output file path

    Returns:
        Combined DataFrame
    """
    data_path = Path(data_dir)
    all_dfs = []

    for file in data_path.glob("*.parquet"):
        if file.name == "combined_training.parquet":
            continue

        try:
            df = pd.read_parquet(file)
            # Extract symbol and timeframe from filename
            parts = file.stem.rsplit("_", 1)
            if len(parts) == 2:
                df["symbol"] = parts[0].replace("_", "/")
                df["timeframe"] = parts[1]
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"])

    # Save combined
    combined.to_parquet(output_file, index=False)
    logger.info(f"Created training dataset: {len(combined)} records")

    return combined


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Data Fetcher")
    parser.add_argument("--provider", default="binance", help="Data provider")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--estimate", action="store_true", help="Only estimate time")
    parser.add_argument("--capacity", action="store_true", help="Show available capacity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    fetcher = SmartDataFetcher(provider=args.provider)

    if args.capacity:
        print(json.dumps(fetcher.get_available_capacity(), indent=2))
    elif args.estimate:
        est = fetcher.estimate_fetch_time(args.symbols, args.timeframe, args.days)
        print(json.dumps(est, indent=2))
    else:
        start = datetime.now() - timedelta(days=args.days)
        fetcher.fetch_multiple_symbols(args.symbols, args.timeframe, start)
        print(json.dumps(fetcher.get_stats(), indent=2))
