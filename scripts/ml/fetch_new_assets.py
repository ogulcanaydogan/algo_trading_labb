#!/usr/bin/env python3
"""
Fetch training data for new high-volume assets to increase trading signals.
Target: 1% daily returns through more trading opportunities.
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
        
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"    Got {len(df)} bars ({len(df)/24:.0f} days)")
    return df


def fetch_stock(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance - hourly."""
    print(f"  Fetching {symbol} from Yahoo Finance ({period})...")
    
    ticker = yf.Ticker(symbol)
    
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
    
    # Fallback to daily
    try:
        df = ticker.history(period="5y", interval="1d")
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'timestamp'
            df = df[['open', 'high', 'low', 'close', 'volume']]
            print(f"    Got {len(df)} daily bars")
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
    print("  FETCH NEW HIGH-VOLUME ASSETS")
    print("  Goal: More trading signals for 1% daily returns")
    print("="*70 + "\n")
    
    # NEW high-volume stocks for more signals
    new_stocks = [
        ("NVDA", "NVDA"),    # NVIDIA - AI/GPU leader, high volatility
        ("AMD", "AMD"),      # AMD - semiconductor, volatile
        ("META", "META"),    # Meta - social media, volatile
        ("COIN", "COIN"),    # Coinbase - crypto proxy, very volatile
        ("AVGO", "AVGO"),    # Broadcom - semiconductor
    ]
    
    # Additional crypto for 24/7 trading  
    new_crypto = [
        ("DOGE/USDT", 365),  # Meme coin, very volatile
        ("LINK/USDT", 365),  # Chainlink, oracle leader
        ("AVAX/USDT", 365),  # Avalanche
        ("MATIC/USDT", 365), # Polygon
    ]
    
    print("NEW STOCK DATA:")
    print("-" * 50)
    for yf_sym, save_name in new_stocks:
        try:
            df = fetch_stock(yf_sym)
            if not df.empty:
                save_parquet(df, save_name)
        except Exception as e:
            print(f"    ERROR fetching {yf_sym}: {e}")
        time.sleep(1)  # Rate limit protection
    
    print("\nNEW CRYPTO DATA:")
    print("-" * 50)
    for sym, days in new_crypto:
        try:
            df = fetch_crypto(sym, days=days)
            if not df.empty:
                save_parquet(df, sym)
        except Exception as e:
            print(f"    ERROR fetching {sym}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  DATA SUMMARY - ALL ASSETS")
    print("="*70)
    
    for f in sorted(OUTPUT_DIR.glob("*_extended.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.stem:<25}: {len(df):>6} bars, {str(df.index.min())[:10]} to {str(df.index.max())[:10]}")
    
    print("\n  Done! Ready for training new models.")


if __name__ == "__main__":
    main()
