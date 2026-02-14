"""
Test Real-Time Streaming Module.

Demonstrates:
- Binance WebSocket connection (crypto)
- Alpaca WebSocket connection (stocks)  
- Tick to bar aggregation
- Event-driven callbacks
- Health monitoring

Usage:
    python scripts/demo/test_streaming.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.data.stream_manager import (
    StreamManager,
    StreamManagerConfig,
    Bar,
    StreamEvent,
    create_default_trading_stream,
)


# Counters for demonstration
tick_count = 0
bar_count = 0


def on_tick(event: StreamEvent):
    """Handle incoming ticks."""
    global tick_count
    tick_count += 1
    tick = event.data
    
    if tick_count % 10 == 0:  # Print every 10th tick
        print(f"  📊 Tick #{tick_count}: {tick.symbol} @ {tick.price:.2f} "
              f"({tick.side or 'n/a'}) vol={tick.volume:.4f}")


def on_bar_complete(bar: Bar):
    """Handle completed bars."""
    global bar_count
    bar_count += 1
    
    print(f"\n🕯️  BAR COMPLETE #{bar_count}: {bar.symbol} [{bar.timeframe}]")
    print(f"    Open: {bar.open:.2f} | High: {bar.high:.2f} | "
          f"Low: {bar.low:.2f} | Close: {bar.close:.2f}")
    print(f"    Volume: {bar.volume:.4f} | Ticks: {bar.tick_count} | VWAP: {bar.vwap:.2f}")
    print(f"    Buy Vol: {bar.buy_volume:.4f} | Sell Vol: {bar.sell_volume:.4f}")
    print(f"    Time: {bar.open_time} → {bar.close_time}")


async def test_crypto_only():
    """Test crypto streaming only (no API keys needed)."""
    print("\n" + "="*60)
    print("TEST: Crypto Streaming (Binance)")
    print("="*60)
    
    config = StreamManagerConfig(
        supported_timeframes=["1m"],  # Just 1 minute bars
        health_check_interval=5.0,
    )
    
    manager = StreamManager(config=config)
    
    # Add BTC and XRP
    manager.add_crypto_symbols(["BTC/USDT", "XRP/USDT"])
    
    # Register callbacks
    manager.on_tick(on_tick)
    manager.on_bar_complete(on_bar_complete)
    
    print("\nStarting crypto stream...")
    print("(Will run for 30 seconds then stop)\n")
    
    try:
        await manager.start()
        
        # Run for 30 seconds
        for i in range(30):
            await asyncio.sleep(1)
            
            if i % 10 == 0 and i > 0:
                metrics = manager.get_metrics()
                health = manager.get_health()
                
                print(f"\n📈 Stats @ {i}s:")
                print(f"   Ticks processed: {metrics['ticks_processed']}")
                print(f"   Bars produced: {metrics['bars_produced']}")
                print(f"   Ticks/sec: {metrics['ticks_per_second']:.1f}")
                
                for name, h in health.items():
                    print(f"   {name}: {h.status.value} "
                          f"(msgs/s: {h.messages_per_second:.1f})")
        
    finally:
        await manager.stop()
        print(f"\n✅ Final: {tick_count} ticks, {bar_count} bars")


async def test_with_stocks():
    """Test both crypto and stock streaming (requires Alpaca API keys)."""
    print("\n" + "="*60)
    print("TEST: Full Streaming (Crypto + Stocks)")
    print("="*60)
    
    # Check for Alpaca credentials
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not alpaca_key or not alpaca_secret:
        print("\n⚠️  ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
        print("   Stock streaming will be skipped")
        print("   Set environment variables to test stock streaming\n")
        return await test_crypto_only()
    
    # Create manager with both crypto and stocks
    manager = create_default_trading_stream(
        crypto_symbols=["BTC/USDT", "XRP/USDT"],
        stock_symbols=["TSLA"],
    )
    
    manager.on_tick(on_tick)
    manager.on_bar_complete(on_bar_complete)
    
    print("\nStarting combined stream...")
    print("(Will run for 60 seconds then stop)\n")
    
    try:
        await manager.start()
        
        for i in range(60):
            await asyncio.sleep(1)
            
            if i % 15 == 0 and i > 0:
                metrics = manager.get_metrics()
                print(f"\n📈 Stats @ {i}s: {metrics['ticks_processed']} ticks, "
                      f"{metrics['bars_produced']} bars")
        
    finally:
        await manager.stop()
        print(f"\n✅ Final: {tick_count} ticks, {bar_count} bars")


async def test_bar_aggregation():
    """Test bar aggregation with simulated ticks."""
    print("\n" + "="*60)
    print("TEST: Bar Aggregation (Simulated)")
    print("="*60)
    
    from bot.data.stream_manager import BarAggregator, Tick
    from datetime import timedelta
    
    bars_received = []
    
    def capture_bar(bar: Bar):
        bars_received.append(bar)
        print(f"  Bar: O={bar.open:.2f} H={bar.high:.2f} "
              f"L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:.4f}")
    
    # Create aggregator for 1-second bars (for fast testing)
    aggregator = BarAggregator(
        symbol="TEST/USD",
        timeframes=["1s", "5s"],
        on_bar_complete=capture_bar,
    )
    
    print("\nSimulating 20 ticks over 10 seconds...")
    
    base_time = datetime.now()
    base_price = 100.0
    
    for i in range(20):
        # Simulate price movement
        price_change = (i % 5 - 2) * 0.1  # Oscillate around base
        tick = Tick(
            symbol="TEST/USD",
            price=base_price + price_change,
            volume=0.1 + (i % 3) * 0.05,
            timestamp=base_time + timedelta(seconds=i * 0.5),
            side="buy" if i % 2 == 0 else "sell",
        )
        
        completed = aggregator.process_tick(tick)
        
        if completed:
            for bar in completed:
                print(f"  → {bar.timeframe} bar completed at {bar.close_time}")
    
    print(f"\n✅ Aggregation test: {len(bars_received)} bars produced")
    
    # Check 5-second bars in history
    df = aggregator.get_dataframe("5s", limit=10)
    if not df.empty:
        print(f"\n5-second bar DataFrame:\n{df}")


async def main():
    """Run all tests."""
    print("\n🚀 Real-Time Streaming Test Suite")
    print("=" * 60)
    
    # Test 1: Bar aggregation (no network needed)
    await test_bar_aggregation()
    
    # Test 2: Crypto streaming (Binance, no API key needed)
    print("\n\n" + "-"*60)
    response = input("Run live crypto streaming test? (y/n): ").strip().lower()
    if response == 'y':
        await test_crypto_only()
    
    # Test 3: Full streaming (requires Alpaca keys)
    print("\n\n" + "-"*60)
    response = input("Run full streaming test (crypto + stocks)? (y/n): ").strip().lower()
    if response == 'y':
        await test_with_stocks()
    
    print("\n\n✅ All tests completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
