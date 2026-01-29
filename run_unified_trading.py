#!/usr/bin/env python3
"""
Unified Trading CLI - Paper to Live Trading with Safety Controls.

Usage:
    python run_unified_trading.py                              # Paper trading (default)
    python run_unified_trading.py --mode live_limited --confirm  # Live trading
    python run_unified_trading.py status                        # Check status
    python run_unified_trading.py check-transition testnet      # Check readiness
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from bot.trading_mode import TradingMode
from bot.unified_engine import EngineConfig, UnifiedTradingEngine
from bot.broker_router import create_multi_asset_adapter


def setup_logging(mode: str, data_dir: Path) -> None:
    """Setup logging."""
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode}_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


async def run_trading(args) -> None:
    """Run the trading engine."""
    mode = TradingMode(args.mode)

    if mode.is_live and not args.confirm:
        print("\nWARNING: Live trading requires --confirm flag")
        print(f"  python run_unified_trading.py --mode {mode.value} --confirm")
        return

    data_dir = Path("data/unified_trading")
    setup_logging(mode.value, data_dir)
    logger = logging.getLogger(__name__)

    # Determine symbols based on mode
    multi_asset = getattr(args, 'multi_asset', False)
    if multi_asset:
        # Multi-asset mode: crypto + forex + commodities + stocks - expanded for 1% daily target
        symbols = [
            # Crypto (Binance) - high volatility, 24/7
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            # Forex (OANDA) - best performers
            "EUR/USD", "GBP/USD",
            # Indices (OANDA) - major indices
            "SPX500/USD", "NAS100/USD",
            # Commodities (OANDA) - Gold, Silver, Oil
            "XAU/USD",    # Gold
            "XAG/USD",    # Silver
            "WTICO/USD",  # WTI Oil
            # US Stocks (Alpaca) - expanded list for more opportunities
            "AAPL/USD", "NVDA/USD", "MSFT/USD", "GOOGL/USD",
            "TSLA/USD", "AMZN/USD", "META/USD",  # Added high-volatility stocks
        ]
        logger.info(f"Multi-asset mode: {len(symbols)} symbols for maximum opportunities")
    else:
        symbols = args.symbols.split(",") if args.symbols else ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT"]

    config = EngineConfig(
        initial_mode=mode,
        initial_capital=args.capital,
        symbols=symbols,
        loop_interval_seconds=args.interval,
        data_dir=data_dir,
        multi_asset=multi_asset,
    )

    engine = UnifiedTradingEngine(config)

    def handle_signal(sig):
        logger.info(f"Received signal {sig}, stopping...")
        asyncio.create_task(engine.stop())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    logger.info("=" * 60)
    logger.info("UNIFIED TRADING ENGINE")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode.value}")
    logger.info(f"Capital: ${config.initial_capital:.2f}")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Interval: {config.loop_interval_seconds}s")
    logger.info("=" * 60)

    if not await engine.initialize(resume=not args.fresh):
        logger.error("Failed to initialize")
        return

    # Send Telegram notification
    try:
        from bot.notifications import NotificationManager, Alert, AlertLevel, AlertType
        nm = NotificationManager()
        if nm.has_channels():
            alert = Alert(
                level=AlertLevel.INFO,
                alert_type=AlertType.SYSTEM,
                title="Trading Engine Started",
                message=f"Mode: {mode.value}\nCapital: ${config.initial_capital}\nSymbols: {', '.join(config.symbols)}",
            )
            nm.send_alert(alert)
    except Exception:
        pass

    await engine.start()

    try:
        while engine._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    status = engine.get_status()
    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATUS")
    logger.info(f"Balance: ${status['balance']:.2f}")
    logger.info(f"P&L: ${status['total_pnl']:.2f} ({status['total_pnl_pct']:.2f}%)")
    logger.info(f"Trades: {status['total_trades']}")
    logger.info("=" * 60)


async def show_status(args) -> None:
    """Show current status."""
    import json
    state_file = Path("data/unified_trading/state.json")

    print("\n" + "=" * 60)
    print("TRADING ENGINE STATUS")
    print("=" * 60)

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"\nMode: {state['mode']}")
        print(f"Status: {state['status']}")
        print(f"Balance: ${state['current_balance']:.2f}")
        print(f"P&L: ${state['total_pnl']:.2f}")
        print(f"Trades: {state['total_trades']}")
        if state['total_trades'] > 0:
            wr = state['winning_trades'] / state['total_trades'] * 100
            print(f"Win Rate: {wr:.1f}%")
        if state.get('positions'):
            print(f"\nPositions ({len(state['positions'])}):")
            for sym, pos in state['positions'].items():
                print(f"  {sym}: {pos['side']} @ ${pos['entry_price']:.2f}")
    else:
        print("No state found. Start trading first.")
    print("=" * 60)


async def check_transition(args) -> None:
    """Check transition readiness."""
    from bot.trading_mode import TradingMode
    from bot.transition_validator import create_transition_validator
    from bot.unified_state import UnifiedStateStore

    target = TradingMode(args.target_mode)
    store = UnifiedStateStore()

    try:
        state = store.initialize(TradingMode.PAPER_LIVE_DATA, 10000, resume=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"No state found: {e}")
        return

    validator = create_transition_validator()
    progress = validator.get_transition_progress(state.mode, target, store.get_mode_state())

    print(f"\nTransition: {state.mode.value} -> {target.value}")
    print(f"Progress: {progress['overall_progress']:.1f}%")
    print(f"Allowed: {'Yes' if progress['allowed'] else 'No'}")
    for d in progress['progress_details']:
        status = "[OK]" if d['passed'] else "[X]"
        print(f"  {status} {d['requirement']}: {d['current']:.1f}/{d['required']:.1f}")


async def emergency_stop(args) -> None:
    """Trigger emergency stop."""
    from bot.safety_controller import SafetyController
    controller = SafetyController()
    controller.emergency_stop(args.reason or "Manual stop")
    print("EMERGENCY STOP TRIGGERED")


def main():
    parser = argparse.ArgumentParser(description="Unified Trading Engine")
    subparsers = parser.add_subparsers(dest="command")

    # Run
    run_p = subparsers.add_parser("run", help="Run trading")
    run_p.add_argument("--mode", default="paper_live_data")
    run_p.add_argument("--capital", type=float, default=30000)
    run_p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,ADA/USDT,AVAX/USDT", help="Trading symbols (comma-separated)")
    run_p.add_argument("--interval", type=int, default=30)  # Faster for more opportunities
    run_p.add_argument("--confirm", action="store_true")
    run_p.add_argument("--fresh", action="store_true")
    run_p.add_argument("--multi-asset", action="store_true", help="Enable multi-asset trading (crypto, forex, commodities)")

    # Status
    subparsers.add_parser("status", help="Show status")

    # Check transition
    check_p = subparsers.add_parser("check-transition", help="Check transition")
    check_p.add_argument("target_mode")

    # Emergency stop
    stop_p = subparsers.add_parser("emergency-stop", help="Emergency stop")
    stop_p.add_argument("--reason", default="Manual")

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"
        args.mode = "paper_live_data"
        args.capital = 30000
        args.symbols = "BTC/USDT,ETH/USDT,SOL/USDT"
        args.interval = 30  # Faster loop for more opportunities
        args.confirm = False
        args.fresh = False
        args.multi_asset = False

    if args.command == "run":
        asyncio.run(run_trading(args))
    elif args.command == "status":
        asyncio.run(show_status(args))
    elif args.command == "check-transition":
        asyncio.run(check_transition(args))
    elif args.command == "emergency-stop":
        asyncio.run(emergency_stop(args))


if __name__ == "__main__":
    main()
