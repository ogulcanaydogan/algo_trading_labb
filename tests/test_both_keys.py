#!/usr/bin/env python3
"""Test both sets of Binance testnet API keys"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bot.exchange import ExchangeClient

def check_keys(api_key, api_secret, label):
    """Check a set of API keys (manual script, not pytest)"""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    
    try:
        client = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=False,
            testnet=True,
        )
        
        print("✅ Client created")
        
        # Try to fetch candles
        print("🔄 Fetching BTC/USDT candles...")
        df = client.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=5)
        
        print(f"✅ SUCCESS! Fetched {len(df)} candles")
        print(f"   Latest price: ${float(df.iloc[-1]['close']):,.2f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("Testing Binance Testnet API Keys")
    
    # Keys from .env.example (old/working keys)
    old_key = "jun2rqxqGnKICW8QuP9e6WxOjNLPiL7ebYGuPzRH32dWYiflOWJI1mBA736ibOJe"
    old_secret = "VegnJQ4YR1iwqzJ3v651o8cTFDcVOT5CJNKTGW1XwJzoai2HozDIcksjzIJFnybF"
    
    # Your new keys
    new_key = "0V45GqUlAFTRqaO2slyCQLr1viuQSUsiEVjaRetXBYy9kkiTjOYCza0QvLQfKfKL"
    new_secret = "lJxxWySjopVUiYJfX02Ykr5fqOlIanRXkF29bHq2SeJ20l0nqwdqCBy3CBNkURMX"
    
    # Test old keys first
    old_works = check_keys(old_key, old_secret, "OLD KEYS (from .env.example)")

    # Test new keys
    new_works = check_keys(new_key, new_secret, "NEW KEYS (your fresh keys)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Old keys (.env.example): {'✅ WORKING' if old_works else '❌ FAILED'}")
    print(f"New keys (your keys):    {'✅ WORKING' if new_works else '❌ FAILED'}")
    
    if old_works and not new_works:
        print("\n💡 Recommendation: Use old keys (new keys need time to activate)")
        print("   I can update your .env file with the working keys.")
    elif new_works:
        print("\n✅ Your new keys are working! You're all set.")
    else:
        print("\n⚠️  Both sets failing - may be a temporary Binance testnet issue")

if __name__ == "__main__":
    main()
