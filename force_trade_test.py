#!/usr/bin/env python
"""Force a test trade to verify the trading system is working."""
import asyncio
import json
from pathlib import Path
from bot.unified_state import UnifiedStateStore
from bot.execution_adapter import create_execution_adapter
from bot.unified_state import TradingMode
from bot.config_loader import load_config

async def main():
    config = load_config()
    store = UnifiedStateStore()
    
    # Load current state
    state_file = Path("data/unified_trading/state.json")
    with open(state_file) as f:
        state = json.load(f)
    
    print("=" * 60)
    print("FORCING TEST TRADE")
    print("=" * 60)
    print(f"Current Balance: ${state['current_balance']:.2f}")
    print(f"Current Positions: {len(state['positions'])}")
    print()
    
    # Create execution adapter
    mode = TradingMode.PAPER_LIVE_DATA
    adapter = create_execution_adapter(mode, config, capital=state['current_balance'])
    
    # Try to place a test BUY order for BTC
    symbol = "BTC/USDT"
    current_price = 100000.0  # Synthetic price for paper trading
    position_size = state['current_balance'] * 0.02  # 2% of balance
    quantity = position_size / current_price
    
    print(f"Placing TEST BUY order:")
    print(f"  Symbol: {symbol}")
    print(f"  Price: ${current_price:,.2f}")
    print(f"  Quantity: {quantity:.6f}")
    print(f"  Position Size: ${position_size:.2f}")
    print()
    
    try:
        result = await adapter.execute_order(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=current_price
        )
        print("✅ Order executed successfully!")
        print(f"   Order ID: {result.order_id}")
        print(f"   Filled: {result.filled_qty} @ ${result.fill_price:.2f}")
        print(f"   Status: {result.status}")
        
        # Update state
        state['positions'][symbol] = {
            "symbol": symbol,
            "side": "LONG",
            "quantity": result.filled_qty,
            "entry_price": result.fill_price,
            "current_price": result.fill_price,
            "unrealized_pnl": 0.0,
            "value": result.filled_qty * result.fill_price,
            "stop_loss": result.fill_price * 0.98,  # 2% stop loss
            "take_profit": result.fill_price * 1.03,  # 3% take profit
        }
        state['daily_trades'] = state.get('daily_trades', 0) + 1
        state['total_trades'] += 1
        
        # Save state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print()
        print("✅ State updated successfully!")
        print(f"   Open Positions: {len(state['positions'])}")
        
    except Exception as e:
        print(f"❌ Order failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
