#!/usr/bin/env python3
"""
Enhanced data fetcher for V6 training - gets maximum historical data.
Uses ccxt for crypto (2+ years) and yfinance for stocks.
"""

import ccxt
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def fetch_crypto(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch crypto data from Binance via ccxt - up to 2+ years."""
    exchange = ccxt.binance()
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    
    print(f"  Fetching {symbol} from Binance ({days} days)...")
    batch_count = 0
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
        except Exception as e:
            print(f"    Rate limit hit, waiting... ({e})")
            time.sleep(5)
            continue
            
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"    ... {len(all_data)} bars fetched")
        
        if len(ohlcv) < 1000:
            break
        
        # Rate limit protection
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"    Got {len(df)} bars ({len(df)/24:.0f} days)")
    return df


def fetch_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock/index data from Yahoo Finance - hourly if available."""
    print(f"  Fetching {symbol} from Yahoo Finance ({period})...")
    
    ticker = yf.Ticker(symbol)
    
    # Try hourly first (max ~2 years for most)
    try:
        df = ticker.history(period=period, interval="1h")
        if not df.empty and len(df) > 500:
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df[['open', 'high', 'low', 'close', 'volume']]
            print(f"    Got {len(df)} hourly bars")
            return df
    except Exception as e:
        print(f"    Hourly failed: {e}")
    
    # Fallback to daily with more history
    try:
        df = ticker.history(period="5y", interval="1d")
        if not df.empty:
            # Resample daily to "fake hourly" by keeping same value
            # This gives more samples for training
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df[['open', 'high', 'low', 'close', 'volume']]
            print(f"    Got {len(df)} daily bars (will use as-is)")
            return df
    except Exception as e:
        print(f"    Daily failed: {e}")
    
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, symbol: str):
    """Save as parquet file."""
    sym = symbol.replace("/", "_").replace("-", "_").replace("^", "")
    path = OUTPUT_DIR / f"{sym}_extended.parquet"
    df.to_parquet(path)
    print(f"    Saved to {path}")


def main():
    print("\n" + "="*70)
    print("  ENHANCED TRAINING DATA FETCHER v2")
    print("  - Crypto: 2+ years of hourly data")
    print("  - Stocks: All tech giants with 8h horizon")
    print("="*70 + "\n")
    
    # Crypto assets - fetch 2+ years for stability
    crypto_symbols = [
        ("BTC/USDT", 730),   # 2 years
        ("XRP/USDT", 730),   # 2 years
        ("ETH/USDT", 730),   # 2 years
        ("SOL/USDT", 365),   # 1 year (newer)
    ]
    
    # Stock assets - for 8h horizon training
    stock_symbols = [
        ("AAPL", "AAPL"),
        ("MSFT", "MSFT"),
        ("GOOGL", "GOOGL"),
        ("TSLA", "TSLA"),
        ("AMZN", "AMZN"),
        ("^GSPC", "SPX500_USD"),     # S&P 500
        ("^NDX", "NAS100_USD"),      # NASDAQ 100
        ("^VIX", "VIX"),             # Volatility index (for SPX features)
    ]
    
    # Fetch crypto
    print("CRYPTO DATA:")
    print("-" * 50)
    for sym, days in crypto_symbols:
        try:
            df = fetch_crypto(sym, days=days)
            if not df.empty:
                save_parquet(df, sym)
        except Exception as e:
            print(f"    ERROR fetching {sym}: {e}")
    
    print("\nSTOCK/INDEX DATA:")
    print("-" * 50)
    # Fetch stocks/indices
    for yf_sym, save_name in stock_symbols:
        try:
            df = fetch_stock(yf_sym)
            if not df.empty:
                save_parquet(df, save_name)
        except Exception as e:
            print(f"    ERROR fetching {yf_sym}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  DATA SUMMARY")
    print("="*70)
    
    for f in OUTPUT_DIR.glob("*_extended.parquet"):
        df = pd.read_parquet(f)
        print(f"  {f.stem}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
    
    print("\n  Done! Ready for training.")


if __name__ == "__main__":
    main()
