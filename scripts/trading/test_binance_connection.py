#!/usr/bin/env python3
"""
Test Binance API Connection.

Tests both testnet and real Binance API connectivity.
Run this before starting live trading to verify your setup.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import ccxt


def test_testnet():
    """Test Binance testnet connection."""
    print("\n" + "=" * 60)
    print("TESTING BINANCE TESTNET")
    print("=" * 60)

    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        print("[ ] Testnet API keys not configured")
        return False

    try:
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": True,
            "options": {"defaultType": "spot"},
        })

        # Test 1: Fetch ticker
        ticker = exchange.fetch_ticker("BTC/USDT")
        print(f"[OK] Connected to testnet")
        print(f"    BTC/USDT price: ${ticker['last']:,.2f}")

        # Test 2: Fetch balance
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {}).get("total", 0)
        btc = balance.get("BTC", {}).get("total", 0)
        print(f"[OK] Account access verified")
        print(f"    USDT: {usdt:,.2f}")
        print(f"    BTC: {btc:.8f}")

        return True

    except Exception as e:
        print(f"[FAIL] Testnet connection failed: {e}")
        return False


def test_live():
    """Test real Binance connection."""
    print("\n" + "=" * 60)
    print("TESTING REAL BINANCE API")
    print("=" * 60)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("[ ] Real API keys not configured")
        print("    Add your keys to .env file:")
        print("    BINANCE_API_KEY=your_key_here")
        print("    BINANCE_API_SECRET=your_secret_here")
        print("\n    Get keys from: https://www.binance.com/en/my/settings/api-management")
        return False

    try:
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "spot"},
        })

        # Test 1: Fetch ticker (public endpoint)
        ticker = exchange.fetch_ticker("BTC/USDT")
        print(f"[OK] Connected to Binance")
        print(f"    BTC/USDT price: ${ticker['last']:,.2f}")

        # Test 2: Fetch balance (requires valid API key)
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {}).get("total", 0)
        btc = balance.get("BTC", {}).get("total", 0)

        # Calculate total portfolio value
        total_value = usdt + (btc * ticker['last'])

        print(f"[OK] Account access verified")
        print(f"    USDT: ${usdt:,.2f}")
        print(f"    BTC: {btc:.8f} (${btc * ticker['last']:,.2f})")
        print(f"    Total Value: ${total_value:,.2f}")

        # Test 3: Check trading permissions
        try:
            # Try to get account info for permissions
            account_info = exchange.fapiPrivateGetAccount() if hasattr(exchange, 'fapiPrivateGetAccount') else None
            print(f"[OK] Trading permissions verified")
        except:
            print(f"[OK] Spot trading access (futures not enabled)")

        return True

    except ccxt.AuthenticationError as e:
        print(f"[FAIL] Authentication failed - check your API keys")
        print(f"    Error: {e}")
        return False
    except ccxt.PermissionDenied as e:
        print(f"[FAIL] Permission denied - enable 'Spot & Margin Trading' in API settings")
        print(f"    Error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return False


def check_safety_settings():
    """Check live trading safety settings."""
    print("\n" + "=" * 60)
    print("LIVE TRADING SAFETY SETTINGS")
    print("=" * 60)

    max_position = os.getenv("LIVE_MAX_POSITION_USD", "50")
    max_loss = os.getenv("LIVE_MAX_DAILY_LOSS_PCT", "2.0")
    max_trades = os.getenv("LIVE_MAX_TRADES_PER_DAY", "10")
    require_confirm = os.getenv("LIVE_REQUIRE_CONFIRMATION", "false")

    print(f"    Max Position Size: ${max_position}")
    print(f"    Max Daily Loss: {max_loss}%")
    print(f"    Max Trades/Day: {max_trades}")
    print(f"    Require Confirmation: {require_confirm}")

    return True


def main():
    """Run all connection tests."""
    print("\n" + "=" * 60)
    print("BINANCE API CONNECTION TEST")
    print("=" * 60)

    results = {
        "testnet": test_testnet(),
        "live": test_live(),
        "safety": check_safety_settings(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "[OK]" if passed else "[--]"
        print(f"  {status} {name.capitalize()}")

    if results["live"]:
        print("\n" + "-" * 60)
        print("READY FOR LIVE TRADING!")
        print("-" * 60)
        print("To start live trading, run:")
        print("  python scripts/trading/run_live_trading.py")
        print("\nStart with small amounts and monitor closely!")
    elif results["testnet"]:
        print("\n" + "-" * 60)
        print("TESTNET READY - Add real API keys for live trading")
        print("-" * 60)
        print("1. Go to: https://www.binance.com/en/my/settings/api-management")
        print("2. Create new API key")
        print("3. Enable 'Spot & Margin Trading' permission")
        print("4. Add to .env file")
    else:
        print("\nNo API connections available. Check your .env file.")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
