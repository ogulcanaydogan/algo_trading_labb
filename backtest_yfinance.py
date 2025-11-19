#!/usr/bin/env python3
"""
Quick backtest with Yahoo Finance historical data (no API keys needed)
"""
import yfinance as yf
import pandas as pd
from bot.strategy import StrategyConfig, compute_indicators, generate_signal
from bot.backtesting import Backtester

print("="*60)
print("BACKTEST WITH YAHOO FINANCE DATA")
print("="*60)

# Configuration
symbol_input = input("Symbol (BTC-USD, ETH-USD, default: BTC-USD): ").strip() or "BTC-USD"
period = input("Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, default: 1y): ").strip() or "1y"
interval = input("Interval (1h, 1d, 1wk, default: 1d): ").strip() or "1d"

print(f"\n📊 Fetching {symbol_input} data for {period} at {interval} intervals...")

# Fetch data from Yahoo Finance
ticker = yf.Ticker(symbol_input)
df = ticker.history(period=period, interval=interval)

if df.empty:
    print("❌ No data found. Try a different symbol.")
    exit(1)

# Convert to OHLCV format expected by backtester
ohlcv = pd.DataFrame({
    'open': df['Open'],
    'high': df['High'],
    'low': df['Low'],
    'close': df['Close'],
    'volume': df['Volume']
})

print(f"✅ Got {len(ohlcv)} candles")
print(f"   Period: {ohlcv.index[0]} to {ohlcv.index[-1]}")
print(f"   Current price: ${ohlcv['close'].iloc[-1]:,.2f}")

# Strategy configuration
config = StrategyConfig(
    symbol=symbol_input.replace("-", "/"),  # Convert BTC-USD to BTC/USDT format
    timeframe=interval,
    ema_fast=12,
    ema_slow=26,
    rsi_period=14,
    risk_per_trade_pct=1.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
)

# Run backtest
print("\n🚀 Running backtest...")
backtester = Backtester(
    strategy_config=config,
    initial_balance=10000.0,
)

result = backtester.run(ohlcv)

# Show results
result.print_summary()

# Show last 10 trades
if result.trades:
    print("\n📊 Last 10 Trades:")
    print("-" * 80)
    for trade in result.trades[-10:]:
        print(trade)

print("\n✅ Backtest complete!")
print(f"📈 Final Balance: ${result.final_balance:,.2f}")
total_return = ((result.final_balance - 10000) / 10000) * 100
print(f"💰 Total Return: {total_return:.2f}%")
print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"📉 Max Drawdown: {result.max_drawdown_pct:.2f}%")
