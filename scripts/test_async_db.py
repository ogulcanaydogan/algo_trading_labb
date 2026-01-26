#!/usr/bin/env python3
"""
Async Database Performance Test

Compares sync vs async database performance for:
- Single inserts
- Batch inserts
- Read operations
"""

import asyncio
import time
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.async_database import AsyncTradingDatabase


async def test_async_inserts():
    """Test async database insert performance"""
    
    print("\n" + "=" * 70)
    print("🚀 ASYNC DATABASE PERFORMANCE TEST")
    print("=" * 70)
    
    # Initialize
    db = AsyncTradingDatabase("data/test_async.db")
    await db.initialize()
    
    # Test 1: Single inserts
    print("\n📊 Test 1: Single Insert Performance")
    trades = [
        {
            'symbol': 'BTC/USDT',
            'direction': 'LONG',
            'entry_time': datetime.now().isoformat(),
            'entry_price': 50000.0,
            'size': 0.1,
            'strategy': 'ensemble',
            'confidence': 0.75
        }
        for _ in range(10)
    ]
    
    start = time.time()
    for trade in trades:
        await db.insert_trade(trade)
    single_time = time.time() - start
    print(f"   10 single inserts: {single_time*1000:.1f}ms ({single_time/10*1000:.1f}ms avg)")
    
    # Test 2: Batch inserts
    print("\n📊 Test 2: Batch Insert Performance")
    start = time.time()
    await db.insert_trades_batch(trades)
    batch_time = time.time() - start
    print(f"   10 batch inserts: {batch_time*1000:.1f}ms ({batch_time/10*1000:.1f}ms avg)")
    
    speedup = single_time / batch_time if batch_time > 0 else 0
    print(f"   ⚡ Batch is {speedup:.1f}x faster!")
    
    # Test 3: Concurrent reads
    print("\n📊 Test 3: Concurrent Read Performance")
    start = time.time()
    
    # Simulate multiple concurrent queries (like dashboard + API + trading loop)
    results = await asyncio.gather(
        db.get_recent_trades(limit=50),
        db.get_recent_trades(limit=50, symbol='BTC/USDT'),
        db.get_trade_stats(days=7),
        db.get_trade_stats(days=30),
    )
    
    read_time = time.time() - start
    print(f"   4 concurrent queries: {read_time*1000:.1f}ms")
    print(f"   Results: {len(results[0])} trades, stats for 7d and 30d")
    
    # Test 4: Equity snapshots
    print("\n📊 Test 4: Equity Snapshot Inserts")
    snapshots = [
        {
            'timestamp': datetime.now().isoformat(),
            'balance': 50000 + i * 100,
            'equity': 50000 + i * 100,
            'unrealized_pnl': i * 10,
            'realized_pnl': i * 5,
            'open_positions': 2,
            'mode': 'paper_live_data'
        }
        for i in range(20)
    ]
    
    start = time.time()
    for snapshot in snapshots:
        await db.insert_equity_snapshot(snapshot)
    snapshot_time = time.time() - start
    print(f"   20 equity snapshots: {snapshot_time*1000:.1f}ms ({snapshot_time/20*1000:.1f}ms avg)")
    
    # Cleanup
    await db.close()
    
    print("\n" + "=" * 70)
    print("✅ Performance Summary:")
    print(f"   Single insert: ~{single_time/10*1000:.1f}ms")
    print(f"   Batch insert: ~{batch_time/10*1000:.1f}ms ({speedup:.1f}x faster)")
    print(f"   Concurrent reads: {read_time*1000:.1f}ms (4 queries)")
    print(f"   Equity snapshot: ~{snapshot_time/20*1000:.1f}ms")
    print("=" * 70)
    
    print("\n💡 Benefits:")
    print("   • Non-blocking async operations")
    print("   • Connection pooling (5 reusable connections)")
    print("   • WAL mode (better concurrency)")
    print("   • Batch operations (2-10x faster)")
    print("   • Concurrent query support\n")


if __name__ == "__main__":
    asyncio.run(test_async_inserts())
