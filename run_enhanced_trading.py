#!/usr/bin/env python3
"""
Enhanced Trading Engine Runner.

Run the enhanced trading system with integrated:
- Execution algorithms (TWAP, VWAP, POV, IS, Adaptive)
- Regime-aware strategy selection
- Dynamic position sizing
- Circuit breaker and risk controls
- Pre-trade stress testing

Usage:
    # Paper trading mode (default)
    python run_enhanced_trading.py

    # With specific symbols
    python run_enhanced_trading.py --symbols BTC/USDT ETH/USDT

    # With different execution algorithm
    python run_enhanced_trading.py --algo twap

    # Disable stress testing
    python run_enhanced_trading.py --no-stress-test
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.enhanced_trading_engine import (
    EnhancedTradingEngine,
    EnhancedTradingConfig,
)
from bot.execution import UrgencyLevel
from bot.regime import SimplePaperAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_trading.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "data/enhanced_trading_config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid config JSON: {e}")
        return {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Enhanced Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic paper trading
  python run_enhanced_trading.py

  # Trade specific symbols
  python run_enhanced_trading.py --symbols BTC/USDT ETH/USDT SOL/USDT

  # Use VWAP execution algorithm
  python run_enhanced_trading.py --algo vwap

  # Higher urgency execution
  python run_enhanced_trading.py --urgency high

  # Custom initial balance
  python run_enhanced_trading.py --balance 50000
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trading symbols (default: BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial paper trading balance (default: 10000)",
    )
    parser.add_argument(
        "--algo",
        choices=["twap", "vwap", "pov", "is", "iceberg", "adaptive"],
        default="adaptive",
        help="Execution algorithm (default: adaptive)",
    )
    parser.add_argument(
        "--urgency",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Execution urgency (default: medium)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Update interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-circuit-breaker",
        action="store_true",
        help="Disable correlation circuit breaker",
    )
    parser.add_argument(
        "--no-stress-test",
        action="store_true",
        help="Disable pre-trade stress testing",
    )
    parser.add_argument(
        "--config",
        default="data/enhanced_trading_config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    file_config = load_config(args.config)

    # Map urgency string to enum
    urgency_map = {
        "low": UrgencyLevel.LOW,
        "medium": UrgencyLevel.MEDIUM,
        "high": UrgencyLevel.HIGH,
        "critical": UrgencyLevel.CRITICAL,
    }

    # Create engine config
    config = EnhancedTradingConfig(
        symbols=args.symbols,
        default_execution_algo=args.algo,
        execution_urgency=urgency_map[args.urgency],
        update_interval_seconds=args.interval,
        enable_circuit_breaker=not args.no_circuit_breaker,
        enable_stress_testing=not args.no_stress_test,
        stress_test_before_trade=not args.no_stress_test,
    )

    # Apply file config overrides
    if "risk_controls" in file_config:
        rc = file_config["risk_controls"]
        if "position_sizing" in rc:
            ps = rc["position_sizing"]
            config.base_risk_per_trade = ps.get("base_risk_per_trade", config.base_risk_per_trade)
            config.max_position_size = ps.get("max_position_size", config.max_position_size)
            config.target_volatility = ps.get("target_volatility", config.target_volatility)
        if "stress_testing" in rc:
            st = rc["stress_testing"]
            config.max_stress_loss_pct = st.get("max_acceptable_loss_pct", config.max_stress_loss_pct)

    # Create paper trading adapter
    adapter = SimplePaperAdapter(initial_balance=args.balance, auto_fetch=True)

    # Create engine
    engine = EnhancedTradingEngine(config, execution_adapter=adapter)

    # Set up callbacks
    def on_trade(trade_info):
        logger.info(
            f"TRADE: {trade_info['side'].upper()} {trade_info['quantity']:.6f} "
            f"{trade_info['symbol']} @ ${trade_info['price']:,.2f} | "
            f"Algo: {trade_info.get('algorithm', 'N/A')} | "
            f"Slippage: {trade_info.get('slippage_bps', 0):.2f}bps"
        )

    def on_regime_change(old_regime, new_regime):
        logger.info(f"REGIME: {old_regime} -> {new_regime}")

    def on_circuit_breaker(status):
        logger.warning(f"CIRCUIT BREAKER: {status.state.value} - {status.trigger_reason}")

    engine.on_trade(on_trade)
    engine.on_regime_change(on_regime_change)
    engine.on_circuit_breaker(on_circuit_breaker)

    # Handle shutdown gracefully
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Print startup info
    print("\n" + "=" * 60)
    print("ENHANCED TRADING ENGINE")
    print("=" * 60)
    print(f"Symbols:           {', '.join(args.symbols)}")
    print(f"Initial Balance:   ${args.balance:,.2f}")
    print(f"Execution Algo:    {args.algo.upper()}")
    print(f"Urgency:           {args.urgency.upper()}")
    print(f"Update Interval:   {args.interval}s")
    print(f"Circuit Breaker:   {'ENABLED' if not args.no_circuit_breaker else 'DISABLED'}")
    print(f"Stress Testing:    {'ENABLED' if not args.no_stress_test else 'DISABLED'}")
    print("=" * 60 + "\n")

    # Start engine
    await engine.start()
    logger.info("Enhanced trading engine started")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Stop engine
    await engine.stop()

    # Print final status
    status = engine.get_status()
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    print(f"Trades Today:      {status['state']['trades_today']}")
    print(f"Daily P&L:         ${status['state']['daily_pnl']:,.2f}")
    print(f"Equity:            ${status['state']['equity']:,.2f}")
    print(f"Current Regime:    {status['state']['current_regime']}")
    print(f"Risk Level:        {status['position_risk_level']}")
    print("=" * 60 + "\n")

    logger.info("Enhanced trading engine stopped")


if __name__ == "__main__":
    asyncio.run(main())
