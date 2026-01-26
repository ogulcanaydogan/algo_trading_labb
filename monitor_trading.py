#!/usr/bin/env python3
"""
Real-time Paper Trading Dashboard
Shows progress toward 200-trade testnet unlock goal
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

def load_state():
    """Load current trading state"""
    state_file = Path("data/unified_trading/state.json")
    if not state_file.exists():
        return None
    
    with open(state_file) as f:
        return json.load(f)

def format_duration(started_at):
    """Format time since start"""
    started = datetime.fromisoformat(started_at)
    duration = datetime.now() - started
    days = duration.days
    hours = duration.seconds // 3600
    mins = (duration.seconds % 3600) // 60
    return f"{days}d {hours}h {mins}m"

def display_dashboard(state):
    """Display trading dashboard"""
    print("\n" + "=" * 70)
    print("📊 PAPER TRADING DASHBOARD - Testnet Unlock Progress")
    print("=" * 70)
    
    # Status
    print(f"\n🎯 Mode: {state['mode'].upper()}")
    print(f"⏱  Running: {format_duration(state['mode_started_at'])}")
    print(f"📅 Daily Date: {state['daily_date']}")
    
    # Balance & P&L
    balance = state['current_balance']
    initial = state['initial_capital']
    total_pnl = balance - initial
    total_pnl_pct = (total_pnl / initial) * 100
    peak = state['peak_balance']
    drawdown = state['max_drawdown_pct']
    
    print(f"\n💰 Balance: ${balance:,.2f}")
    print(f"💵 Initial: ${initial:,.2f}")
    print(f"📈 Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"🏔️  Peak: ${peak:,.2f}")
    print(f"📉 Max DD: {drawdown:.2f}%")
    
    # Trade Progress
    total_trades = state['total_trades']
    winning = state['winning_trades']
    losing = state['losing_trades']
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    
    progress_pct = (total_trades / 200) * 100
    remaining = max(0, 200 - total_trades)
    
    print(f"\n🎲 Total Trades: {total_trades} / 200 ({progress_pct:.1f}%)")
    print(f"✅ Winning: {winning} ({win_rate:.1f}%)")
    print(f"❌ Losing: {losing}")
    print(f"⏳ Remaining: {remaining} trades to testnet unlock")
    
    # Progress bar
    bar_width = 50
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n[{bar}] {progress_pct:.1f}%")
    
    # Positions
    positions = state.get('positions', {})
    print(f"\n📍 Open Positions: {len(positions)}")
    if positions:
        for symbol, pos in positions.items():
            side_emoji = "🟢" if pos['side'] == 'long' else "🔴"
            pnl = pos.get('unrealized_pnl', 0)
            conf = pos.get('signal_confidence', 0) * 100
            print(f"  {side_emoji} {symbol}: {pos['side'].upper()} @ ${pos['entry_price']:.2f}")
            print(f"     PnL: ${pnl:+.2f} | Confidence: {conf:.1f}% | {pos.get('signal_reason', 'N/A')}")
    
    # Daily Stats
    daily_pnl = state.get('daily_pnl', 0)
    daily_trades = state.get('daily_trades', 0)
    daily_pnl_pct = (daily_pnl / balance * 100) if balance > 0 else 0
    
    print(f"\n📆 Today's Stats:")
    print(f"   Trades: {daily_trades}")
    print(f"   P&L: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
    
    # Projections
    if state['days_in_mode'] > 0:
        avg_trades_per_day = total_trades / max(1, state['days_in_mode'])
        days_to_unlock = remaining / avg_trades_per_day if avg_trades_per_day > 0 else 999
        unlock_date = datetime.now() + timedelta(days=days_to_unlock)
        
        print(f"\n🔮 Projections:")
        print(f"   Avg trades/day: {avg_trades_per_day:.1f}")
        print(f"   Days to 200 trades: {days_to_unlock:.1f}")
        print(f"   Estimated unlock: {unlock_date.strftime('%b %d, %Y')}")
    
    # Goals
    print(f"\n🎯 Testnet Unlock Requirements:")
    check_trades = "✅" if total_trades >= 200 else "⏳"
    check_wr = "✅" if win_rate >= 60 else "⏳"
    check_dd = "✅" if drawdown < 5 else "⏳"
    
    print(f"   {check_trades} 200+ trades ({total_trades}/200)")
    print(f"   {check_wr} 60%+ win rate ({win_rate:.1f}%)")
    print(f"   {check_dd} <5% drawdown ({drawdown:.2f}%)")
    
    print("=" * 70)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Refresh: python monitor_trading.py")
    print("=" * 70 + "\n")

def main():
    """Main dashboard loop"""
    state = load_state()
    
    if not state:
        print("\n❌ No trading state found. Start the engine first:")
        print("   bash start_paper_trading.sh")
        return
    
    display_dashboard(state)

if __name__ == "__main__":
    main()
