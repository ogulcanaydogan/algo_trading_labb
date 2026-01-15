"""
Live Trading Script (Testnet or Real Exchange)

This script allows you to run your strategy live.
It's recommended to test with DRY RUN mode first.
"""

import os
import time

from dotenv import load_dotenv
from bot.exchange import ExchangeClient
from bot.strategy import (
    StrategyConfig,
    calculate_position_size,
    compute_indicators,
    generate_signal,
)
from bot.trading import TradingManager

# Load .env file
load_dotenv()


def run_live_trading():
    """Start live trading"""

    print("="*60)
    print("LIVE TRADING MODE")
    print("="*60)
    print("WARNING: This script executes real trades!")
    print("It's recommended to test with DRY RUN mode first.\n")

    # Mode selection
    print("Trading Mode:")
    print("1. DRY RUN (only log, no real orders)")
    print("2. TESTNET (Binance testnet, real orders)")
    print("3. LIVE (REAL EXCHANGE - CAUTION!)")

    mode = input("Your choice (1/2/3): ").strip()

    if mode not in ["1", "2", "3"]:
        print("Invalid selection!")
        return

    if mode == "3":
        confirm = input("WARNING: YOU WILL BE TRADING ON REAL EXCHANGE! Are you sure? (type YES): ").strip()
        if confirm != "YES":
            print("Operation cancelled.")
            return

    dry_run = mode == "1"
    use_testnet = mode == "2"

    # Configuration
    symbol = input("\nSymbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (default: 5m): ").strip() or "5m"
    loop_interval = int(input("Loop interval (seconds) (default: 60): ").strip() or "60")

    # Strategy parameters
    print("\nStrategy Parameters:")
    risk_pct = float(input("  Risk per trade % (default: 1.0): ").strip() or "1.0")
    stop_loss_pct = float(input("  Stop Loss % (default: 2.0): ").strip() or "2.0") / 100
    take_profit_pct = float(input("  Take Profit % (default: 4.0): ").strip() or "4.0") / 100

    config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        risk_per_trade_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    # Create exchange client
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            print("Testnet API keys not found in .env file!")
            return
        print("\nConnecting to Binance Testnet...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )
    else:
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv("EXCHANGE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("EXCHANGE_API_SECRET")
        if not api_key or not api_secret:
            print("API keys not found in .env file!")
            print("Add BINANCE_API_KEY and BINANCE_API_SECRET to your .env file")
            print("Get keys from: https://www.binance.com/en/my/settings/api-management")
            return
        print("\nConnecting to Binance...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
        )

    # Create trading manager
    trading_manager = TradingManager(
        exchange_client=exchange,
        symbol=symbol,
        dry_run=dry_run,
    )

    print("\nConnection successful!")
    print("Starting trading...")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'TESTNET' if use_testnet else 'LIVE'}")
    print(f"   Loop Interval: {loop_interval}s")
    print("\nTrading loop started... (Ctrl+C to stop)\n")

    iteration = 0

    try:
        while True:
            iteration += 1
            start_time = time.time()

            print("\n" + "=" * 60)
            print(f"ITERATION #{iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            try:
                # 1. Fetch data
                print("Fetching data...")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
                current_price = float(ohlcv.iloc[-1]["close"])
                print(f"   Current price: ${current_price:,.2f}")

                # 2. Check current position
                position_info = trading_manager.get_position_info()
                if position_info:
                    print("\nCurrent Position:")
                    print(f"   Direction: {position_info['direction']}")
                    print(f"   Entry: ${position_info['entry_price']:,.2f}")
                    print(f"   Current: ${position_info.get('current_price', current_price):,.2f}")
                    print(f"   P&L: ${position_info.get('unrealized_pnl', 0):,.2f} "
                          f"({position_info.get('unrealized_pnl_pct', 0):.2f}%)")
                    print(f"   Stop Loss: ${position_info['stop_loss']:,.2f}")
                    print(f"   Take Profit: ${position_info['take_profit']:,.2f}")

                    # Exit check
                    exit_reason = trading_manager.check_position_exit(current_price)
                    if exit_reason:
                        print(f"\nClosing position: {exit_reason}")
                        result = trading_manager.close_position(reason=exit_reason)
                        if result.success:
                            print(f"Position closed: {result.order_id}")
                        else:
                            print(f"Error: {result.error}")
                else:
                    print("\nNo open position")

                    # 3. Generate signal
                    print("\nAnalyzing signal...")
                    enriched = compute_indicators(ohlcv, config)
                    signal = generate_signal(enriched, config)

                    print(f"   Decision: {signal['decision']}")
                    print(f"   Confidence: {signal['confidence']:.2%}")
                    print(f"   RSI: {signal['rsi']:.2f}")
                    print(f"   EMA Fast: ${signal['ema_fast']:,.2f}")
                    print(f"   EMA Slow: ${signal['ema_slow']:,.2f}")
                    print(f"   Reason: {signal['reason']}")

                    # 4. Open position (if signal exists)
                    if signal["decision"] != "FLAT" and signal["confidence"] > 0.4:
                        print(f"\nOpening {signal['decision']} position...")

                        # Calculate position size
                        # Simple example: Get balance from testnet
                        try:
                            balance_info = exchange.client.fetch_balance()
                            usdt_balance = balance_info.get("USDT", {}).get("free", 10000)
                        except Exception:
                            usdt_balance = 10000  # Default

                        size = calculate_position_size(
                            usdt_balance,
                            config.risk_per_trade_pct,
                            current_price,
                            config.stop_loss_pct,
                        )

                        # Convert to BTC size
                        size_btc = size / current_price

                        # Calculate stop loss and take profit
                        if signal["decision"] == "LONG":
                            stop_loss = current_price * (1 - config.stop_loss_pct)
                            take_profit = current_price * (1 + config.take_profit_pct)
                        else:
                            stop_loss = current_price * (1 + config.stop_loss_pct)
                            take_profit = current_price * (1 - config.take_profit_pct)

                        result = trading_manager.open_position(
                            direction=signal["decision"],
                            size=size_btc,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            signal_info=signal,
                        )

                        if result.success:
                            print(f"Position opened: {result.order_id}")
                        else:
                            print(f"Error: {result.error}")
                    else:
                        print("\nNo trade (no signal or low confidence)")

            except Exception as e:
                print(f"\nLoop error: {e}")
                import traceback
                traceback.print_exc()

            # Wait
            elapsed = time.time() - start_time
            sleep_time = max(1, loop_interval - int(elapsed))
            print(f"\nWaiting {sleep_time} seconds for next iteration...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nTrading stopped!")

        # If open position exists, ask
        if trading_manager.current_position:
            close_pos = input("\nClose open position? (y/n): ").strip().lower()
            if close_pos == "y":
                result = trading_manager.close_position(reason="Manual stop")
                if result.success:
                    print("Position closed")
                else:
                    print(f"Error: {result.error}")


if __name__ == "__main__":
    try:
        run_live_trading()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
