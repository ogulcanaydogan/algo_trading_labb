"""
Backtest Runner Script

This script allows you to test your strategy with historical data.
"""

import os

from dotenv import load_dotenv
from bot.exchange import ExchangeClient, PaperExchangeClient
from bot.strategy import StrategyConfig
from bot.backtesting import Backtester, save_backtest_results

# Load .env file
load_dotenv()


def run_backtest():
    """Run backtest"""

    print("="*60)
    print("BACKTEST MODE")
    print("="*60)

    # Configuration
    symbol = input("Symbol (default: BTC/USDT): ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (default: 1h): ").strip() or "1h"
    lookback = int(input("How many candles back? (default: 1000): ").strip() or "1000")
    initial_balance = float(input("Starting balance ($) (default: 10000): ").strip() or "10000")

    # Strategy parameters
    print("\nStrategy Parameters:")
    ema_fast = int(input("  EMA Fast (default: 12): ").strip() or "12")
    ema_slow = int(input("  EMA Slow (default: 26): ").strip() or "26")
    rsi_period = int(input("  RSI Period (default: 14): ").strip() or "14")
    risk_pct = float(input("  Risk per trade % (default: 1.0): ").strip() or "1.0")
    stop_loss_pct = float(input("  Stop Loss % (default: 2.0): ").strip() or "2.0") / 100
    take_profit_pct = float(input("  Take Profit % (default: 4.0): ").strip() or "4.0") / 100

    # Create strategy config
    config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        risk_per_trade_pct=risk_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    # Select data source
    print("\nData Source:")
    print("1. Binance Testnet (real data)")
    print("2. Paper Exchange (synthetic data)")
    data_source = input("Your choice (1/2): ").strip()

    if data_source == "1":
        # Binance Testnet
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

        if not api_key or not api_secret:
            print("Testnet API keys not found in .env file!")
            return

        print("Fetching data from Binance Testnet...")
        exchange = ExchangeClient(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
    else:
        # Paper Exchange
        print("Generating synthetic data...")
        exchange = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
        ohlcv = exchange.fetch_ohlcv(limit=lookback)

    print(f"{len(ohlcv)} candles fetched")
    print(f"   Start: {ohlcv.index[0]}")
    print(f"   End: {ohlcv.index[-1]}")

    # Run backtest
    backtester = Backtester(
        strategy_config=config,
        initial_balance=initial_balance,
    )

    result = backtester.run(ohlcv)

    # Show results
    result.print_summary()

    # Show last 10 trades
    if result.trades:
        print("\nLast 10 Trades:")
        print("-" * 80)
        for trade in result.trades[-10:]:
            emoji = "+" if trade.pnl > 0 else "-"
            print(
                f"{emoji} {trade.direction:5s} | "
                f"Entry: ${trade.entry_price:8.2f} | "
                f"Exit: ${trade.exit_price:8.2f} | "
                f"P&L: ${trade.pnl:8.2f} ({trade.pnl_pct:6.2f}%) | "
                f"{trade.exit_reason}"
            )
        print("-" * 80)

    # Save results
    save_choice = input("\nSave results to JSON file? (y/n): ").strip().lower()
    if save_choice == "y":
        filename = input("Filename (default: backtest_results.json): ").strip() or "backtest_results.json"
        save_backtest_results(result, filename)

    print("\nBacktest completed!")


if __name__ == "__main__":
    try:
        run_backtest()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
