#!/usr/bin/env python3
"""
Real-time Trading Dashboard - Terminal UI

Displays:
- System status
- Open positions
- P&L summary
- Recent trades
- Model performance
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_state():
    """Load trading state from file."""
    state_path = Path("data/unified_trading/state.json")
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def load_equity():
    """Load equity history."""
    equity_path = Path("data/unified_trading/equity.json")
    if equity_path.exists():
        with open(equity_path) as f:
            data = json.load(f)
            return data.get("history", [])[-20:]  # Last 20 entries
    return []


def format_pnl(value):
    """Format P&L with color indicator."""
    if value > 0:
        return f"+${value:,.2f} 📈"
    elif value < 0:
        return f"-${abs(value):,.2f} 📉"
    return f"${value:,.2f}"


def print_header():
    print("=" * 70)
    print("🤖 ALGO TRADING LAB - LIVE DASHBOARD")
    print(f"   Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_status(state):
    mode = state.get("mode", "unknown")
    balance = state.get("balance", 0)

    print(f"\n📊 MODE: {mode.upper()}")
    print(f"💰 BALANCE: ${balance:,.2f}")

    # Calculate P&L from positions
    positions = state.get("positions", {})
    total_pnl = 0

    print(f"\n📈 OPEN POSITIONS ({len(positions)}):")
    print("-" * 70)

    if positions:
        print(f"{'Symbol':<12} {'Side':<8} {'Entry':<12} {'Qty':<10} {'P&L':<12}")
        print("-" * 70)
        for symbol, pos in positions.items():
            side = pos.get("side", "long")
            entry = pos.get("entry_price", 0)
            qty = pos.get("quantity", 0)
            # Note: Current price not available, showing entry
            print(f"{symbol:<12} {side:<8} ${entry:<11,.2f} {qty:<10.4f}")
    else:
        print("   No open positions")


def print_recent_trades(state):
    trades = state.get("trade_log", [])[-10:]  # Last 10 trades

    print(f"\n📜 RECENT TRADES ({len(trades)}):")
    print("-" * 70)

    if trades:
        for trade in trades:
            ts = trade.get("timestamp", "")[:19]
            symbol = trade.get("symbol", "???")
            action = trade.get("action", "???")
            price = trade.get("price", 0)
            print(f"   {ts} | {symbol:<12} | {action:<6} | ${price:,.2f}")
    else:
        print("   No trades yet")


def print_model_status():
    """Show ML model status."""
    model_dir = Path("data/models")
    models = list(model_dir.glob("*_metadata.json"))

    print(f"\n🧠 ML MODELS ({len(models)}):")
    print("-" * 70)

    for meta_path in models[:8]:  # Show first 8
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                symbol = meta.get("symbol", meta_path.stem)
                n_features = meta.get("n_features", "?")
                rf_acc = meta.get("models", {}).get("random_forest", {}).get("cv_accuracy", 0)
                print(f"   {symbol:<12} | Features: {n_features} | RF Acc: {rf_acc:.1%}")
        except Exception:
            continue


def print_footer():
    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit | Refresh: 30s")
    print("=" * 70)


def main():
    """Main dashboard loop."""
    refresh_interval = 30  # seconds

    try:
        while True:
            clear_screen()
            state = load_state()

            print_header()
            print_status(state)
            print_recent_trades(state)
            print_model_status()
            print_footer()

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
