#!/usr/bin/env python3
"""
Paper Trading - Automated trading with fictional money
Runs continuously, generates real signals, tracks performance
"""
import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf

from bot.exchange import PaperExchangeClient, ExchangeClient
from bot.strategy import (
    StrategyConfig,
    compute_indicators,
    generate_signal,
    calculate_position_size,
)
from bot.trading import TradingManager
from bot.state import create_state_store, EquityPoint

load_dotenv()

print("="*60)
print("PAPER TRADING MODE - Testing with Fictional Money")
print("="*60)
print("✅ Can use real market data (Binance public or Yahoo Finance)")
print("✅ Generates real signals")
print("✅ Simulates real trades")
print("✅ Tracks P&L and performance")
print("❌ NO real money at risk!")
print("="*60)

# Configuration
symbol = input("\nSymbol (default: BTC/USDT): ").strip() or "BTC/USDT"
timeframe = input("Timeframe (default: 1h): ").strip() or "1h"
print("\nData Source:")
print("1. Binance (public, no keys)")
print("2. Yahoo Finance")
print("3. Synthetic (random walk)")
source_choice = input("Select (1/2/3, default 1): ").strip() or "1"
starting_balance = float(input("Starting balance ($, default: 10000): ").strip() or "10000")
loop_interval = int(input("Check for signals every X seconds (default: 60): ").strip() or "60")
max_loops = int(input("Max iterations (0 = infinite, default: 100): ").strip() or "100")

print(f"\n📊 Starting Paper Trading:")
print(f"   Symbol: {symbol}")
print(f"   Timeframe: {timeframe}")
print(f"   Starting Balance: ${starting_balance:,.2f}")
print(f"   Check interval: {loop_interval}s")
print(f"   Max iterations: {'Infinite' if max_loops == 0 else max_loops}")
source_map = {"1": "BINANCE", "2": "YAHOO", "3": "SYNTHETIC"}
data_source = source_map.get(source_choice, "BINANCE")
print(f"   Data Source: {data_source}")

# Strategy config
config = StrategyConfig(
    symbol=symbol,
    timeframe=timeframe,
    ema_fast=int(os.getenv("EMA_FAST", "12")),
    ema_slow=int(os.getenv("EMA_SLOW", "26")),
    rsi_period=int(os.getenv("RSI_PERIOD", "14")),
    risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "1.0")),
    stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.02")),
    take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.04")),
)

state_store = create_state_store(Path("./data"))

# Exchange / data fetch setup
binance_client: Optional[ExchangeClient] = None
paper_exchange: Optional[PaperExchangeClient] = None

if data_source == "BINANCE":
    binance_client = ExchangeClient(exchange_id="binance")  # public endpoints only
elif data_source == "SYNTHETIC":
    paper_exchange = PaperExchangeClient(symbol=symbol, timeframe=timeframe)

trading_manager = TradingManager(
    exchange_client=paper_exchange if paper_exchange else binance_client,
    symbol=symbol,
    dry_run=True,
    data_dir=Path("./data"),
)

# Initialize balance in state
state_store.update_state(balance=starting_balance)

print(f"\n🚀 Starting paper trading loop...")
print(f"💡 Press Ctrl+C to stop\n")

iteration = 0
try:
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Fetch latest data depending on source
        print("📊 Fetching market data...")
        if data_source == "BINANCE" and binance_client:
            ohlcv = binance_client.fetch_ohlcv(symbol, timeframe=timeframe, limit=250)
        elif data_source == "YAHOO":
            yahoo_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD").replace("/", "-")
            period = "7d"  # rolling short window for speed
            yf_interval = timeframe
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=yf_interval)
            if df.empty:
                print("❌ Yahoo Finance returned no data; skipping iteration")
                time.sleep(loop_interval)
                continue
            ohlcv = pd.DataFrame({
                "open": df["Open"],
                "high": df["High"],
                "low": df["Low"],
                "close": df["Close"],
                "volume": df["Volume"],
            })
        elif data_source == "SYNTHETIC" and paper_exchange:
            ohlcv = paper_exchange.fetch_ohlcv(limit=250)
        else:
            print("❌ Invalid data source configuration")
            break

        current_price = float(ohlcv["close"].iloc[-1])
        print(f"   Current {symbol} price: ${current_price:,.2f}")
        
        # Compute indicators
        print("🔍 Computing indicators...")
        enriched = compute_indicators(ohlcv, config)
        
        # Generate signal
        print("🎯 Generating signal...")
        signal = generate_signal(enriched, config)
        
        print(f"\n📋 Signal: {signal['decision']} ({signal['confidence']:.1f}% confidence)")
        print(f"   RSI: {signal['rsi']:.2f}")
        print(f"   Reason: {signal['reason']}")
        
        # Get current state
        # Access current state object
        state_store.load()  # refresh from disk
        state = state_store.state
        current_position = state.position
        current_balance = state.balance
        
        print(f"\n💰 Current Status:")
        print(f"   Balance: ${current_balance:,.2f}")
        print(f"   Position: {current_position}")
        if current_position != "FLAT" and state.entry_price:
            pnl_pct = ((current_price - state.entry_price) / state.entry_price) * 100
            if current_position == "SHORT":
                pnl_pct = -pnl_pct
            print(f"   Entry: ${state.entry_price:,.2f}")
            print(f"   Unrealized P&L: {pnl_pct:+.2f}%")
        
        # Trading logic
        if current_position == "FLAT":
            # No position, check if should enter
            if signal['decision'] in ['LONG', 'SHORT'] and float(signal.get('confidence', 0)) >= 30:
                print(f"\n🎯 Opening {signal['decision']} position...")
                position_size = calculate_position_size(
                    balance=current_balance,
                    risk_pct=config.risk_per_trade_pct,
                    price=current_price,
                    stop_loss_pct=config.stop_loss_pct,
                )
                # Compute protective levels
                if signal['decision'] == 'LONG':
                    stop_loss_price = current_price * (1 - config.stop_loss_pct)
                    take_profit_price = current_price * (1 + config.take_profit_pct)
                else:  # SHORT
                    stop_loss_price = current_price * (1 + config.stop_loss_pct)
                    take_profit_price = current_price * (1 - config.take_profit_pct)

                trading_manager.open_position(
                    direction=signal['decision'],
                    size=position_size,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    signal_info=signal,
                )
                print(f"✅ Position opened!")
            else:
                print(f"⏸️  No action: Confidence too low or signal is FLAT")
        
        else:
            # Have position, check exit conditions
            print(f"\n🔍 Checking exit conditions...")

            # Check for partial profit exits first
            partial_exit = trading_manager.check_partial_profit(current_price)
            if partial_exit:
                print(f"📊 Partial exit triggered at {partial_exit['r_target']}R ({partial_exit['current_r']:.2f}R profit)")
                result = trading_manager.execute_partial_exit(partial_exit)
                if result.success:
                    print(f"✅ Partial exit executed: {partial_exit['exit_pct']*100:.0f}% of position")
                    print(f"   Remaining: {partial_exit['remaining_size']:.6f}")

            # Check for full exit (stop loss / take profit / trailing stop)
            exit_reason = trading_manager.check_position_exit(current_price)
            if exit_reason:
                print(f"📉 Closing position: {exit_reason}")
                trading_manager.close_position(exit_reason)
                print(f"✅ Position closed!")
            else:
                # Show trailing stop info if active
                trailing_info = trading_manager.get_trailing_stop_info()
                if trailing_info and trailing_info.get("active"):
                    print(f"📈 Trailing stop active: Peak ${trailing_info['peak_price']:.2f}, Stop ${trailing_info['current_stop']:.2f}")
                print(f"✅ Holding position")
        
        # Summary
        state_store.load()
        updated_state = state_store.state
        # Record equity point
        state_store.record_equity(EquityPoint(timestamp=updated_state.timestamp, value=updated_state.balance))
        print(f"\n📊 Summary:")
        print(f"   Final Balance: ${updated_state.balance:,.2f}")
        total_pnl = updated_state.balance - starting_balance
        total_pnl_pct = (total_pnl / starting_balance) * 100
        print(f"   Total P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
        
        # Check if should stop
        if max_loops > 0 and iteration >= max_loops:
            print(f"\n🏁 Reached max iterations ({max_loops})")
            break
        
        # Wait before next iteration
        print(f"\n⏰ Waiting {loop_interval} seconds...")
        time.sleep(loop_interval)

except KeyboardInterrupt:
    print(f"\n\n🛑 Stopped by user")

# Final report
print(f"\n{'='*60}")
print(f"PAPER TRADING SESSION COMPLETE")
print(f"{'='*60}")
state_store.load()
final_state = state_store.state
print(f"Starting Balance: ${starting_balance:,.2f}")
print(f"Final Balance: ${final_state.balance:,.2f}")
final_pnl = final_state.balance - starting_balance
final_pnl_pct = (final_pnl / starting_balance) * 100
print(f"Total P&L: ${final_pnl:+,.2f} ({final_pnl_pct:+.2f}%)")
print(f"Total Iterations: {iteration}")
print(f"\n📊 View detailed results at: http://127.0.0.1:8000/dashboard")
print(f"   (Start API server: python api/api.py)")
print(f"\n✅ All trades saved in: ./data/signals.json")
