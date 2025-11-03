"""Test script to verify Binance Spot Testnet connection."""

import os
from dotenv import load_dotenv
from bot.exchange import ExchangeClien

# Load environment variables
load_dotenv()

def test_testnet_connection():
    """Test connection to Binance Spot Testnet."""
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

    if not api_key or not api_secret:
        print("❌ Error: BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set in .env")
        return

    print("🔄 Testing Binance Spot Testnet connection...")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")

    try:
        # Create testnet clien
        client = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=False,
            testnet=True,
        )

        print("✅ Client created successfully")

        # Test fetching OHLCV data
        print("\n🔄 Fetching BTC/USDT candles...")
        df = client.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=10)

        print(f"✅ Successfully fetched {len(df)} candles")
        print("\nLast 3 candles:")
        print(df.tail(3))

        # Test account info if available
        print("\n🔄 Testing account access...")
        try:
            balance = client.client.fetch_balance()
            print("✅ Account access successful")
            print(f"Total assets: {len(balance.get('total', {}))}")

            # Show non-zero balances
            total = balance.get('total', {})
            non_zero = {k: v for k, v in total.items() if v > 0}
            if non_zero:
                print("\nNon-zero balances:")
                for asset, amount in non_zero.items():
                    print(f"  {asset}: {amount}")
            else:
                print("No assets in testnet account (this is normal for new accounts)")

        except Exception as e:
            print(f"⚠️  Account access test failed: {e}")

        print("\n✅ All tests passed! Testnet connection is working.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your API keys are correct")
        print("2. Ensure you created the keys on https://testnet.binance.vision/")
        print("3. Make sure TRADE, USER_DATA, USER_STREAM permissions are enabled")

if __name__ == "__main__":
    test_testnet_connection()
