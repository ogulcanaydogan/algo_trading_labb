#!/usr/bin/env python3
"""Quick data fetcher for training - uses ccxt for crypto and yfinance for stocks."""

import ccxt
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "training"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def fetch_crypto(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch crypto data from Binance via ccxt."""
    exchange = ccxt.binance()
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    
    print(f"  Fetching {symbol} from Binance...")
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"    Got {len(df)} bars")
    return df

def fetch_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock/index data from Yahoo Finance."""
    print(f"  Fetching {symbol} from Yahoo Finance...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1h")
    
    if df.empty:
        # Try daily data if hourly fails
        df = ticker.history(period="5y", interval="1d")
        print(f"    Using daily data (hourly not available)")
    
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'timestamp'
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"    Got {len(df)} bars")
    return df

def save_parquet(df: pd.DataFrame, symbol: str):
    """Save as parquet file."""
    sym = symbol.replace("/", "_").replace("-", "_")
    path = OUTPUT_DIR / f"{sym}_extended.parquet"
    df.to_parquet(path)
    print(f"    Saved to {path}")

def main():
    # Crypto assets
    crypto_symbols = ["XRP/USDT", "BTC/USDT", "ETH/USDT"]
    
    # Stock/Index assets  
    stock_symbols = [
        ("TSLA", "TSLA"),
        ("^GSPC", "SPX500_USD"),  # S&P 500 index
    ]
    
    print("\n" + "="*60)
    print("  FETCHING TRAINING DATA")
    print("="*60)
    
    # Fetch crypto
    for sym in crypto_symbols:
        try:
            df = fetch_crypto(sym, days=365)
            if not df.empty:
                save_parquet(df, sym)
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Fetch stocks/indices
    for yf_sym, save_name in stock_symbols:
        try:
            df = fetch_stock(yf_sym)
            if not df.empty:
                save_parquet(df, save_name)
        except Exception as e:
            print(f"    ERROR: {e}")
    
    print("\n" + "="*60)
    print("  Done! Files in:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()
