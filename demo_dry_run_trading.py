#!/usr/bin/env python3
"""
Demo: Simulated Trading with Dashboard Integration

This script demonstrates how DRY RUN trades appear on the dashboard.
It will:
1. Start the API server
2. Generate a trading signal
3. Open a simulated position (DRY RUN)
4. Wait a bit
5. Close the position
6. Show the results on the dashboard
"""

import sys
import time
import subprocess
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.exchange import PaperExchangeClient
from bot.strategy import StrategyConfig, compute_indicators, generate_signal, calculate_position_size
from bot.trading import TradingManager
from bot.state import create_state_store

def main():
    print("=" * 60)
    print("SIMULATED TRADING DEMO (DRY RUN)")
    print("=" * 60)
    
    # Start API server in background
    print("\n🚀 Starting API server...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.api:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)
    
    try:
        # Initialize components
        print("📊 Initializing trading components...")
        symbol = "BTC/USDT"
        exchange = PaperExchangeClient(symbol=symbol, timeframe="1m")
        config = StrategyConfig(symbol=symbol)
        
        trading_manager = TradingManager(
            exchange_client=exchange,
            symbol=symbol,
            dry_run=True,  # DRY RUN mode
        )
        
        # Fetch data and generate signal
        print(f"\n📈 Fetching market data for {symbol}...")
        candles = exchange.fetch_ohlcv(limit=250)
        current_price = float(candles.iloc[-1]["close"])
        print(f"   Current price: ${current_price:,.2f}")
        
        print("\n🔍 Computing indicators...")
        enriched = compute_indicators(candles, config)
        
        print("🎯 Generating trading signal...")
        signal = generate_signal(enriched, config)
        
        print(f"\n📋 Signal Details:")
        print(f"   Decision: {signal['decision']}")
        print(f"   Confidence: {signal['confidence']:.2%}")
        print(f"   RSI: {signal['rsi']:.2f}")
        print(f"   EMA Fast: ${signal['ema_fast']:,.2f}")
        print(f"   EMA Slow: ${signal['ema_slow']:,.2f}")
        print(f"   Reason: {signal['reason']}")
        
        # Open position if signal is not FLAT
        if signal["decision"] != "FLAT" and signal["confidence"] > 0.3:
            print(f"\n🚀 Opening {signal['decision']} position (DRY RUN)...")
            
            # Calculate position size
            balance = 10000.0
            size_usd = calculate_position_size(
                balance, 
                config.risk_per_trade_pct, 
                current_price, 
                config.stop_loss_pct
            )
            size_btc = size_usd / current_price
            
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
                print(f"   ✅ Position opened: {result.order_id}")
                print(f"   Entry: ${result.price:,.2f}")
                print(f"   Size: {result.quantity:.6f} BTC")
                print(f"   Stop Loss: ${stop_loss:,.2f}")
                print(f"   Take Profit: ${take_profit:,.2f}")
                
                # Check dashboard
                print("\n🎨 Checking dashboard...")
                time.sleep(1)
                resp = requests.get("http://127.0.0.1:8000/status", timeout=5)
                if resp.status_code == 200:
                    state = resp.json()
                    print(f"   Dashboard shows:")
                    print(f"   - Position: {state.get('position')}")
                    print(f"   - Entry Price: ${state.get('entry_price'):,.2f}" if state.get('entry_price') else "   - Entry Price: None")
                    print(f"   - Size: {state.get('position_size'):.6f} BTC")
                    print(f"   - Last Signal: {state.get('last_signal')}")
                
                # Wait a bit and then close
                print("\n⏳ Waiting 3 seconds...")
                time.sleep(3)
                
                print(f"\n🚪 Closing position...")
                close_result = trading_manager.close_position(reason="Demo complete")
                
                if close_result.success:
                    print(f"   ✅ Position closed: {close_result.order_id}")
                    print(f"   Exit: ${close_result.price:,.2f}")
                    
                    # Check dashboard again
                    time.sleep(1)
                    resp = requests.get("http://127.0.0.1:8000/status", timeout=5)
                    if resp.status_code == 200:
                        state = resp.json()
                        print(f"\n   Dashboard updated:")
                        print(f"   - Position: {state.get('position')}")
                        print(f"   - Last Signal: {state.get('last_signal')}")
                else:
                    print(f"   ❌ Failed to close: {close_result.error}")
            else:
                print(f"   ❌ Failed to open position: {result.error}")
        else:
            print(f"\n⏸️  No trade: Signal is {signal['decision']} with confidence {signal['confidence']:.2%}")
            print("   (Need LONG/SHORT with confidence > 30%)")
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60)
        print(f"\n🌐 View dashboard at: http://127.0.0.1:8000/dashboard")
        print("   (API server will keep running for 30 seconds)")
        
        # Keep server running for a bit
        time.sleep(30)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Stopping API server...")
        api_proc.terminate()
        api_proc.wait()

if __name__ == "__main__":
    main()
