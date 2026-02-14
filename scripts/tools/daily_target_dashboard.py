#!/usr/bin/env python3
"""
Daily Target Dashboard - Track Progress Toward 1% Daily Returns

Features:
- Morning: Show target ($100 on $10k)
- Throughout day: Real-time progress
- Alert when goal reached or at risk
- End of day: Summary and next-day adjustments
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Default styling without rich
COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


class DailyTargetDashboard:
    """Dashboard for tracking daily return targets."""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        daily_target_pct: float = 0.01,
        daily_loss_limit_pct: float = 0.03,
        state_file: Optional[str] = None,
    ):
        self.initial_capital = initial_capital
        self.daily_target_pct = daily_target_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.state_file = state_file or str(PROJECT_ROOT / "data" / "daily_state.json")
        
        self.console = Console() if HAS_RICH else None
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from file or initialize new."""
        if Path(self.state_file).exists():
            with open(self.state_file) as f:
                state = json.load(f)
                # Check if it's a new day
                if state.get("date") != datetime.now().strftime("%Y-%m-%d"):
                    # Carry over balance, reset daily stats
                    return self._new_day_state(state.get("current_balance", self.initial_capital))
                return state
        return self._new_day_state(self.initial_capital)
    
    def _new_day_state(self, balance: float) -> Dict:
        """Create new day state."""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "start_balance": balance,
            "current_balance": balance,
            "daily_target_usd": balance * self.daily_target_pct,
            "daily_loss_limit_usd": balance * self.daily_loss_limit_pct,
            "trades": [],
            "pnl_usd": 0,
            "pnl_pct": 0,
            "trades_won": 0,
            "trades_lost": 0,
            "target_reached": False,
            "loss_limit_hit": False,
            "alerts": [],
        }
    
    def _save_state(self):
        """Save state to file."""
        Path(self.state_file).parent.mkdir(exist_ok=True, parents=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def add_trade(
        self,
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        entry_price: float,
        exit_price: float,
        position_size: float,
        position_pct: float,
        conviction: str,
        pnl_usd: float,
    ):
        """Record a completed trade."""
        trade = {
            "time": datetime.now().isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry": entry_price,
            "exit": exit_price,
            "position_size": position_size,
            "position_pct": position_pct,
            "conviction": conviction,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_usd / self.state["start_balance"],
        }
        
        self.state["trades"].append(trade)
        self.state["current_balance"] += pnl_usd
        self.state["pnl_usd"] += pnl_usd
        self.state["pnl_pct"] = self.state["pnl_usd"] / self.state["start_balance"]
        
        if pnl_usd > 0:
            self.state["trades_won"] += 1
        else:
            self.state["trades_lost"] += 1
        
        # Check alerts
        self._check_alerts()
        self._save_state()
        
        return trade
    
    def _check_alerts(self):
        """Check and generate alerts."""
        # Target reached
        if self.state["pnl_pct"] >= self.daily_target_pct and not self.state["target_reached"]:
            self.state["target_reached"] = True
            self.state["alerts"].append({
                "time": datetime.now().isoformat(),
                "type": "TARGET_REACHED",
                "message": f"🎯 Daily target reached! +{self.state['pnl_pct']:.2%}",
            })
        
        # Loss limit
        if self.state["pnl_pct"] <= -self.daily_loss_limit_pct and not self.state["loss_limit_hit"]:
            self.state["loss_limit_hit"] = True
            self.state["alerts"].append({
                "time": datetime.now().isoformat(),
                "type": "LOSS_LIMIT",
                "message": f"⚠️ Daily loss limit hit! {self.state['pnl_pct']:.2%}",
            })
        
        # Halfway there
        if (self.state["pnl_pct"] >= self.daily_target_pct / 2 and
            len([a for a in self.state["alerts"] if a["type"] == "HALFWAY"]) == 0):
            self.state["alerts"].append({
                "time": datetime.now().isoformat(),
                "type": "HALFWAY",
                "message": f"📈 Halfway to daily target! +{self.state['pnl_pct']:.2%}",
            })
    
    def get_progress(self) -> Dict:
        """Get current progress toward daily target."""
        pnl = self.state["pnl_pct"]
        target = self.daily_target_pct
        
        progress_pct = min(1.0, max(0, pnl / target)) if target > 0 else 0
        
        trades_today = len(self.state["trades"])
        win_rate = (self.state["trades_won"] / trades_today * 100) if trades_today > 0 else 0
        
        # Estimate trades needed
        if trades_today > 0:
            avg_pnl_per_trade = self.state["pnl_pct"] / trades_today
            remaining = target - pnl
            if avg_pnl_per_trade > 0:
                trades_needed = max(0, remaining / avg_pnl_per_trade)
            else:
                trades_needed = float("inf")
        else:
            trades_needed = 5  # Default estimate
        
        return {
            "date": self.state["date"],
            "start_balance": self.state["start_balance"],
            "current_balance": self.state["current_balance"],
            "pnl_usd": self.state["pnl_usd"],
            "pnl_pct": pnl,
            "target_usd": self.state["daily_target_usd"],
            "target_pct": target,
            "progress_pct": progress_pct,
            "trades_today": trades_today,
            "trades_won": self.state["trades_won"],
            "trades_lost": self.state["trades_lost"],
            "win_rate": win_rate,
            "trades_needed_estimate": trades_needed,
            "target_reached": self.state["target_reached"],
            "loss_limit_hit": self.state["loss_limit_hit"],
            "status": self._get_status(),
        }
    
    def _get_status(self) -> str:
        """Get current status."""
        pnl = self.state["pnl_pct"]
        target = self.daily_target_pct
        
        if self.state["loss_limit_hit"]:
            return "STOPPED"
        elif pnl >= target:
            return "GOAL_MET"
        elif pnl >= target * 0.75:
            return "ALMOST_THERE"
        elif pnl >= target * 0.5:
            return "ON_TRACK"
        elif pnl >= 0:
            return "IN_PROGRESS"
        elif pnl >= -self.daily_loss_limit_pct / 2:
            return "CAUTION"
        else:
            return "AT_RISK"
    
    def display_simple(self):
        """Display progress without rich."""
        p = self.get_progress()
        
        # Status color
        status_colors = {
            "GOAL_MET": COLORS["green"],
            "ALMOST_THERE": COLORS["green"],
            "ON_TRACK": COLORS["blue"],
            "IN_PROGRESS": COLORS["blue"],
            "CAUTION": COLORS["yellow"],
            "AT_RISK": COLORS["red"],
            "STOPPED": COLORS["red"],
        }
        color = status_colors.get(p["status"], COLORS["reset"])
        
        print("\n" + "="*60)
        print(f"  {COLORS['bold']}DAILY TARGET DASHBOARD{COLORS['reset']} - {p['date']}")
        print("="*60)
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * p["progress_pct"])
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\n  Progress: [{bar}] {p['progress_pct']*100:.1f}%")
        
        # Key metrics
        pnl_color = COLORS["green"] if p["pnl_usd"] >= 0 else COLORS["red"]
        print(f"\n  {COLORS['bold']}P&L:{COLORS['reset']} {pnl_color}${p['pnl_usd']:+.2f} ({p['pnl_pct']:+.2%}){COLORS['reset']}")
        print(f"  {COLORS['bold']}Target:{COLORS['reset']} ${p['target_usd']:.2f} ({p['target_pct']:.2%})")
        print(f"  {COLORS['bold']}Remaining:{COLORS['reset']} ${p['target_usd'] - p['pnl_usd']:.2f}")
        
        print(f"\n  {COLORS['bold']}Status:{COLORS['reset']} {color}{p['status']}{COLORS['reset']}")
        print(f"  {COLORS['bold']}Balance:{COLORS['reset']} ${p['current_balance']:,.2f}")
        
        # Trades
        print(f"\n  {COLORS['bold']}Trades:{COLORS['reset']} {p['trades_today']} "
              f"({p['trades_won']}W / {p['trades_lost']}L)")
        print(f"  {COLORS['bold']}Win Rate:{COLORS['reset']} {p['win_rate']:.1f}%")
        
        if p["trades_needed_estimate"] < float("inf"):
            print(f"  {COLORS['bold']}Est. Trades Needed:{COLORS['reset']} ~{p['trades_needed_estimate']:.1f}")
        
        # Recent trades
        if self.state["trades"]:
            print(f"\n  {COLORS['bold']}Recent Trades:{COLORS['reset']}")
            for t in self.state["trades"][-5:]:
                t_color = COLORS["green"] if t["pnl_usd"] > 0 else COLORS["red"]
                print(f"    {t['symbol']:<12} {t['direction']:<6} {t_color}${t['pnl_usd']:+.2f}{COLORS['reset']} ({t['conviction']})")
        
        # Alerts
        recent_alerts = [a for a in self.state["alerts"] if a["type"] in ("TARGET_REACHED", "LOSS_LIMIT")]
        if recent_alerts:
            print(f"\n  {COLORS['bold']}Alerts:{COLORS['reset']}")
            for a in recent_alerts:
                print(f"    {a['message']}")
        
        print("\n" + "="*60)
    
    def display_rich(self):
        """Display progress with rich formatting."""
        if not HAS_RICH:
            return self.display_simple()
        
        p = self.get_progress()
        
        # Status styling
        status_styles = {
            "GOAL_MET": "[bold green]",
            "ALMOST_THERE": "[green]",
            "ON_TRACK": "[blue]",
            "IN_PROGRESS": "[cyan]",
            "CAUTION": "[yellow]",
            "AT_RISK": "[red]",
            "STOPPED": "[bold red]",
        }
        style = status_styles.get(p["status"], "")
        
        # Build table
        table = Table(title=f"Daily Target Dashboard - {p['date']}", 
                      show_header=False, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        pnl_style = "[green]" if p["pnl_usd"] >= 0 else "[red]"
        
        table.add_row("Status", f"{style}{p['status']}[/]")
        table.add_row("Balance", f"${p['current_balance']:,.2f}")
        table.add_row("P&L", f"{pnl_style}${p['pnl_usd']:+.2f} ({p['pnl_pct']:+.2%})[/]")
        table.add_row("Target", f"${p['target_usd']:.2f} ({p['target_pct']:.2%})")
        table.add_row("Progress", f"{p['progress_pct']*100:.1f}%")
        table.add_row("", "")
        table.add_row("Trades", f"{p['trades_today']} ({p['trades_won']}W / {p['trades_lost']}L)")
        table.add_row("Win Rate", f"{p['win_rate']:.1f}%")
        
        if p["trades_needed_estimate"] < float("inf"):
            table.add_row("Est. Trades Needed", f"~{p['trades_needed_estimate']:.1f}")
        
        self.console.print(table)
        
        # Progress bar
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Daily Progress", 
                completed=p["progress_pct"]*100, 
                total=100
            )
        
        # Trades table
        if self.state["trades"]:
            trades_table = Table(title="Recent Trades", border_style="dim")
            trades_table.add_column("Time", style="dim")
            trades_table.add_column("Symbol")
            trades_table.add_column("Dir")
            trades_table.add_column("P&L", justify="right")
            trades_table.add_column("Conv")
            
            for t in self.state["trades"][-5:]:
                t_style = "[green]" if t["pnl_usd"] > 0 else "[red]"
                time_str = datetime.fromisoformat(t["time"]).strftime("%H:%M")
                trades_table.add_row(
                    time_str,
                    t["symbol"],
                    t["direction"],
                    f"{t_style}${t['pnl_usd']:+.2f}[/]",
                    t["conviction"],
                )
            
            self.console.print(trades_table)
    
    def display(self):
        """Display dashboard."""
        if HAS_RICH:
            self.display_rich()
        else:
            self.display_simple()
    
    def get_morning_summary(self) -> str:
        """Get morning summary string."""
        balance = self.state["start_balance"]
        target = self.state["daily_target_usd"]
        
        return f"""
🌅 DAILY TRADING TARGET

Date: {self.state['date']}
Starting Balance: ${balance:,.2f}
Daily Target: ${target:.2f} ({self.daily_target_pct:.1%})
Loss Limit: ${self.state['daily_loss_limit_usd']:.2f} ({self.daily_loss_limit_pct:.1%})

📊 To hit 1% today:
- With 80% win rate, 10% position: ~5 trades
- With 80% win rate, 15% position: ~3-4 trades
- With HIGH conviction (25%): ~2 trades

Good luck! 🚀
"""
    
    def get_eod_summary(self) -> str:
        """Get end of day summary."""
        p = self.get_progress()
        
        status_emoji = {
            "GOAL_MET": "🎯",
            "ALMOST_THERE": "📈",
            "ON_TRACK": "✅",
            "IN_PROGRESS": "⏳",
            "CAUTION": "⚠️",
            "AT_RISK": "🚨",
            "STOPPED": "🛑",
        }
        
        emoji = status_emoji.get(p["status"], "📊")
        result = "SUCCESS" if p["target_reached"] else "MISS"
        
        return f"""
{emoji} END OF DAY SUMMARY

Date: {p['date']}
Result: {result}

Balance: ${p['current_balance']:,.2f}
P&L: ${p['pnl_usd']:+.2f} ({p['pnl_pct']:+.2%})
Target: ${p['target_usd']:.2f} ({p['target_pct']:.2%})

Trades: {p['trades_today']} ({p['trades_won']}W / {p['trades_lost']}L)
Win Rate: {p['win_rate']:.1f}%

{"🎉 Target reached! Great job!" if p['target_reached'] else f"📉 Fell short by ${p['target_usd'] - p['pnl_usd']:.2f}"}

Tomorrow's target: ${p['current_balance'] * self.daily_target_pct:.2f}
"""


def main():
    parser = argparse.ArgumentParser(description="Daily Target Dashboard")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--target", type=float, default=0.01, help="Daily target (decimal)")
    parser.add_argument("--morning", action="store_true", help="Show morning summary")
    parser.add_argument("--eod", action="store_true", help="Show end of day summary")
    parser.add_argument("--watch", action="store_true", help="Watch mode (live updates)")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval (seconds)")
    args = parser.parse_args()
    
    dashboard = DailyTargetDashboard(
        initial_capital=args.balance,
        daily_target_pct=args.target,
    )
    
    if args.morning:
        print(dashboard.get_morning_summary())
    elif args.eod:
        print(dashboard.get_eod_summary())
    elif args.watch:
        print("Starting watch mode... Press Ctrl+C to exit")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                dashboard.display()
                print(f"\n  Refreshing in {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    else:
        dashboard.display()


if __name__ == "__main__":
    main()
