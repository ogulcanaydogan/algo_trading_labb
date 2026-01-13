#!/usr/bin/env python3
"""
Quick Paper Trading Demo - Shows 1 complete trading cycle
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from bot.exchange import PaperExchangeClient
from bot.strategy import StrategyConfig, compute_indicators, generate_signal, calculate_position_size
from bot.trading import TradingManager
from bot.state import create_state_store

load_dotenv()

print("="*60)
print("PAPER TRADING DEMO - One Complete Cycle")
print("="*60)
print("This demo will:")
print("1. Fetch real market data")
print("2. Generate a trading signal")
print("3. Open a position (if signal is strong)")
print("4. Show how P&L would be tracked")
print("5. Demonstrate exit conditions")
print("="*60)

# Configuration
STARTING_BALANCE = 10000.0
symbol = "BTC/USDT"
timeframe = "1h"

print(f"\n📊 Setup:")
print(f"   Symbol: {symbol}")
print(f"   Starting Balance: ${STARTING_BALANCE:,.2f}")
print(f"   Risk per trade: 1%")
print(f"   Stop Loss: 2%")
print(f"   Take Profit: 4%")

# Strategy config
config = StrategyConfig(
    symbol=symbol,
    timeframe=timeframe,
    ema_fast=12,
    ema_slow=26,
    rsi_period=14,
    risk_per_trade_pct=1.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
)

# Initialize
exchange = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
state_store = create_state_store(Path("./data"))
trading_manager = TradingManager(
    exchange_client=exchange,
    symbol=symbol,
    dry_run=True,
)

state_store.update_state(balance=STARTING_BALANCE)

print(f"\n{'='*60}")
print("STEP 1: Fetch Market Data")
print(f"{'='*60}")
ohlcv = exchange.fetch_ohlcv(limit=100)
current_price = float(ohlcv['close'].iloc[-1])
print(f"✅ Fetched {len(ohlcv)} candles")
print(f"   Current price: ${current_price:,.2f}")

print(f"\n{'='*60}")
print("STEP 2: Compute Technical Indicators")
print(f"{'='*60}")
enriched = compute_indicators(ohlcv, config)
print(f"✅ Computed:")
print(f"   EMA Fast (12): ${enriched['ema_fast'].iloc[-1]:,.2f}")
print(f"   EMA Slow (26): ${enriched['ema_slow'].iloc[-1]:,.2f}")
print(f"   RSI (14): {enriched['rsi'].iloc[-1]:.2f}")

print(f"\n{'='*60}")
print("STEP 3: Generate Trading Signal")
print(f"{'='*60}")
signal = generate_signal(enriched, config)
print(f"✅ Signal Generated:")
print(f"   Decision: {signal['decision']}")
print(f"   Confidence: {signal['confidence']:.1f}%")
print(f"   Reason: {signal['reason']}")

print(f"\n{'='*60}")
print("STEP 4: Trading Decision")
print(f"{'='*60}")

state = state_store.state

if signal['decision'] in ['LONG', 'SHORT'] and signal['confidence'] >= 30:
    print(f"✅ Signal is strong enough to trade!")
    
    # Calculate position size
    position_size = calculate_position_size(
        balance=state.balance,
        risk_pct=config.risk_per_trade_pct,
        price=current_price,
        stop_loss_pct=config.stop_loss_pct,
    )
    
    print(f"\n💰 Position Sizing:")
    print(f"   Balance: ${state.balance:,.2f}")
    print(f"   Risk amount: ${state.balance * 0.01:,.2f} (1%)")
    print(f"   Position size: ${position_size:,.2f}")
    
    # Open position
    print(f"\n🎯 Opening {signal['decision']} position...")
    trading_manager.open_position(
        direction=signal['decision'],
        price=current_price,
        size_usd=position_size,
        signal=signal,
    )
    
    # Check updated state
    updated_state = state_store.state
    print(f"\n📊 Position Opened:")
    print(f"   Direction: {updated_state.position}")
    print(f"   Entry Price: ${updated_state.entry_price:,.2f}")
    print(f"   Size: {updated_state.size:.6f} BTC")
    print(f"   Stop Loss: ${updated_state.stop_loss:,.2f} ({config.stop_loss_pct*100:.1f}%)")
    print(f"   Take Profit: ${updated_state.take_profit:,.2f} ({config.take_profit_pct*100:.1f}%)")
    
    # Simulate price movements
    print(f"\n{'='*60}")
    print("STEP 5: Simulating Price Movements")
    print(f"{'='*60}")
    
    # Scenario 1: Stop loss hit
    stop_loss_price = updated_state.stop_loss
    print(f"\n❌ Scenario A: Price drops to ${stop_loss_price:,.2f} (Stop Loss)")
    if updated_state.position == "LONG":
        loss = (stop_loss_price - updated_state.entry_price) * updated_state.size
    else:
        loss = (updated_state.entry_price - stop_loss_price) * updated_state.size
    loss_pct = (loss / position_size) * 100
    print(f"   P&L would be: ${loss:,.2f} ({loss_pct:.2f}%)")
    print(f"   Balance would be: ${state.balance + loss:,.2f}")
    
    # Scenario 2: Take profit hit
    take_profit_price = updated_state.take_profit
    print(f"\n✅ Scenario B: Price rises to ${take_profit_price:,.2f} (Take Profit)")
    if updated_state.position == "LONG":
        profit = (take_profit_price - updated_state.entry_price) * updated_state.size
    else:
        profit = (updated_state.entry_price - take_profit_price) * updated_state.size
    profit_pct = (profit / position_size) * 100
    print(f"   P&L would be: ${profit:,.2f} ({profit_pct:.2f}%)")
    print(f"   Balance would be: ${state.balance + profit:,.2f}")
    
else:
    print(f"⏸️  Signal not strong enough to trade")
    print(f"   Need: LONG/SHORT with confidence ≥ 30%")
    print(f"   Got: {signal['decision']} with {signal['confidence']:.1f}%")

print(f"\n{'='*60}")
print("DEMO COMPLETE")
print(f"{'='*60}")
print(f"\n💡 How Paper Trading Works:")
print(f"   1. Uses REAL market data (Yahoo Finance)")
print(f"   2. Generates REAL trading signals (EMA + RSI)")
print(f"   3. Simulates position opening/closing")
print(f"   4. Tracks P&L with fictional money")
print(f"   5. Updates dashboard with results")
print(f"\n📊 To run continuous paper trading:")
print(f"   python run_paper_trading.py")
print(f"\n📈 To view dashboard:")
print(f"   python api/api.py")
print(f"   Then open: http://127.0.0.1:8000/dashboard")
