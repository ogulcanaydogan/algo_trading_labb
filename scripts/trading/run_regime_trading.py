#!/usr/bin/env python3
"""
Regime-Based Trading Runner.

This script demonstrates how to use the regime-based trading system:
1. RegimeTradingEngine - Main trading orchestrator
2. StrategyTracker - Performance monitoring

Usage:
    python scripts/trading/run_regime_trading.py --mode paper
    python scripts/trading/run_regime_trading.py --mode backtest --symbol BTC/USDT
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bot.regime import (
    MarketRegime,
    RegimeDetector,
    RegimePositionManager,
    RegimeRiskEngine,
    RegimeTradingEngine,
    SimplePaperAdapter,
    TradingConfig,
    TradingMode,
    StrategyTracker,
    StrategyStatus,
    get_tracker,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import data store for persistent recording
try:
    from bot.data_store import (
        get_data_store,
        record_trade as store_trade,
        record_snapshot,
        record_signal,
    )
    HAS_DATA_STORE = True
    logger.info("Data Store available for persistent recording")
except ImportError:
    HAS_DATA_STORE = False

# System Logger for centralized event tracking
try:
    from bot.system_logger import (
        log_bot_start, log_bot_stop, heartbeat, EventType
    )
    HAS_SYSTEM_LOGGER = True
    logger.info("System Logger available for event tracking")
except ImportError:
    HAS_SYSTEM_LOGGER = False


def run_backtest(symbol: str = "BTC/USDT", days: int = 365):
    """
    Run a backtest with regime-based position sizing.

    This demonstrates the core concept: stay invested, but adjust exposure
    based on detected market regime.
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    logger.info(f"Running regime backtest for {symbol} ({days} days)")

    # Map crypto symbols to yfinance format
    yf_symbol = symbol.replace("/", "-")
    if "USDT" in yf_symbol:
        yf_symbol = yf_symbol.replace("USDT", "USD")

    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)

    if data.empty:
        logger.error(f"No data fetched for {yf_symbol}")
        return

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    logger.info(f"Fetched {len(data)} bars")

    # Initialize components
    detector = RegimeDetector()
    position_manager = RegimePositionManager()
    tracker = get_tracker()

    # Register strategy
    strategy_id = f"regime_sizing_{symbol.replace('/', '_')}"
    tracker.register_strategy(
        strategy_id=strategy_id,
        strategy_name=f"Regime Sizing {symbol}",
        strategy_type="regime_sizing",
        status=StrategyStatus.TESTING,
    )

    # Backtest parameters
    initial_capital = 10000.0
    position_size = 1000.0  # Fixed position size for clear results

    # Tracking variables
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    current_regime = None
    current_trade_id = None

    equity_curve = [initial_capital]
    regime_history = []
    trades = []

    # Run backtest
    for i in range(100, len(data)):  # Need 100 bars for regime detection
        window = data.iloc[i-100:i+1].copy()
        current_bar = data.iloc[i]
        price = float(current_bar["Close"])

        # Detect regime
        regime_state = detector.detect(window)
        regime = regime_state.regime

        if current_regime != regime:
            logger.info(f"Regime change: {current_regime} -> {regime} (confidence: {regime_state.confidence:.2f})")
            current_regime = regime

        regime_history.append({
            "date": data.index[i],
            "regime": regime.value,
            "confidence": regime_state.confidence,
        })

        # Get position recommendation
        current_value = position * price if position > 0 else 0
        recommendation = position_manager.get_recommendation(
            regime_state=regime_state,
            current_equity=capital + current_value,
            current_position_value=current_value,
        )

        # Execute based on recommendation
        if recommendation.should_execute:
            if recommendation.action.value == "increase" and position == 0:
                # Enter position
                buy_qty = position_size / price
                position = buy_qty
                entry_price = price
                capital -= position_size
                current_trade_id = tracker.record_entry(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    side="long",
                    entry_price=price,
                    quantity=buy_qty,
                    regime=regime.value,
                )
                trades.append({
                    "date": data.index[i],
                    "action": "BUY",
                    "price": price,
                    "regime": regime.value,
                })

            elif recommendation.action.value in ["decrease", "close_all"] and position > 0:
                # Exit position
                sell_value = position * price
                pnl = sell_value - (position * entry_price)
                capital += sell_value
                if current_trade_id:
                    tracker.record_exit(
                        trade_id=current_trade_id,
                        exit_price=price,
                        exit_reason=f"regime_{regime.value}",
                        regime=regime.value,
                    )
                trades.append({
                    "date": data.index[i],
                    "action": "SELL",
                    "price": price,
                    "pnl": pnl,
                    "regime": regime.value,
                })
                position = 0
                entry_price = 0
                current_trade_id = None

        # Track equity
        current_equity = capital + (position * price if position > 0 else 0)
        equity_curve.append(current_equity)

    # Final exit if still in position
    if position > 0:
        final_price = float(data.iloc[-1]["Close"])
        sell_value = position * final_price
        pnl = sell_value - (position * entry_price)
        capital += sell_value
        if current_trade_id:
            tracker.record_exit(
                trade_id=current_trade_id,
                exit_price=final_price,
                exit_reason="backtest_end",
            )

    # Calculate metrics
    final_equity = capital
    total_return = (final_equity - initial_capital) / initial_capital
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_dd = np.max(drawdown)

    # Buy & hold comparison
    buy_hold_return = (float(data.iloc[-1]["Close"]) - float(data.iloc[100]["Close"])) / float(data.iloc[100]["Close"])

    # Print results
    print("\n" + "=" * 60)
    print("REGIME-BASED BACKTEST RESULTS")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {data.index[100].date()} to {data.index[-1].date()}")
    print(f"Total bars: {len(data) - 100}")
    print("-" * 60)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital:   ${final_equity:,.2f}")
    print(f"Total Return:    {total_return*100:.2f}%")
    print(f"Max Drawdown:    {max_dd*100:.2f}%")
    print("-" * 60)
    print(f"Buy & Hold:      {buy_hold_return*100:.2f}%")
    print(f"Alpha:           {(total_return - buy_hold_return)*100:.2f}%")
    print("-" * 60)
    print(f"Total Trades:    {len([t for t in trades if t['action'] == 'BUY'])}")

    # Regime distribution
    regime_counts = {}
    for r in regime_history:
        regime = r["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print("\nRegime Distribution:")
    for regime, count in sorted(regime_counts.items()):
        pct = count / len(regime_history) * 100
        print(f"  {regime:<10}: {count:>4} bars ({pct:.1f}%)")

    print("\n" + tracker.get_strategy_comparison())
    print("=" * 60)

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "buy_hold_return": buy_hold_return,
        "trades": len([t for t in trades if t["action"] == "BUY"]),
    }


async def run_paper_trading(symbols: list = None):
    """
    Run paper trading with the regime trading engine.

    Args:
        symbols: List of symbols to trade (e.g., ["BTC/USDT", "ETH/USDT", "SPY"])
    """
    if symbols is None:
        symbols = ["BTC/USDT"]

    logger.info(f"Starting paper trading for {', '.join(symbols)}")

    # Create trading config
    config = TradingConfig(
        symbols=symbols,
        mode=TradingMode.PAPER,
        update_interval_seconds=60,  # 1 minute updates
        state_file=Path("data/regime_trading/state.json"),
    )

    # Create paper trading adapter with initial balance
    initial_capital = 50000.0  # $50k for multi-asset trading (5 assets)
    adapter = SimplePaperAdapter(initial_balance=initial_capital)

    # Create and start engine
    engine = RegimeTradingEngine(config=config, execution_adapter=adapter)

    # Register each symbol with tracker
    tracker = get_tracker()
    for symbol in symbols:
        strategy_id = f"regime_engine_{symbol.replace('/', '_')}"
        tracker.register_strategy(
            strategy_id=strategy_id,
            strategy_name=f"Regime Engine {symbol}",
            strategy_type="regime_sizing",
            status=StrategyStatus.ACTIVE,
        )

    # Handle shutdown - use loop-based approach for background compatibility
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    # Only set signal handlers if running in foreground
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # Signal handling not available (e.g., in certain threading contexts)
        pass

    # Start engine
    await engine.start()
    logger.info("Engine started. Press Ctrl+C to stop.")

    # Register with system logger
    if HAS_SYSTEM_LOGGER:
        log_bot_start("regime_trading", "regime", {
            "initial_capital": initial_capital,
            "symbols": symbols,
            "mode": "paper"
        })

    iteration = 0
    try:
        # Run the trading loop - either until shutdown signal or indefinitely
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
            iteration += 1

            # Send heartbeat every 30 seconds
            if HAS_SYSTEM_LOGGER and iteration % 30 == 0:
                heartbeat("regime_trading")

            # Record snapshot every 60 seconds
            if HAS_DATA_STORE and iteration % 60 == 0:
                try:
                    status = engine.get_status()
                    state = status.get('state', {})
                    equity = state.get('equity', initial_capital)
                    positions = state.get('positions', {})

                    record_snapshot(
                        total_value=equity,
                        cash_balance=state.get('cash', equity),
                        positions=positions,
                        pnl=equity - initial_capital,
                        pnl_pct=(equity - initial_capital) / initial_capital * 100,
                        market_values={"regime": equity},
                    )
                    logger.debug(f"Recorded snapshot: ${equity:,.2f}")
                except Exception as e:
                    logger.warning(f"Failed to record snapshot: {e}")

            # Check if engine is still running
            if not engine.state.is_running:
                logger.info("Engine stopped externally")
                break
    except asyncio.CancelledError:
        logger.info("Task cancelled")
    finally:
        # Unregister from system logger
        if HAS_SYSTEM_LOGGER:
            log_bot_stop("regime_trading", "normal_shutdown")

        await engine.stop()
        logger.info("Engine stopped")

        # Print final status
        status = engine.get_status()
        state = status.get('state', {})
        print("\n" + "=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Current Regime: {status.get('current_regime', 'N/A')}")
        print(f"Portfolio Value: ${state.get('equity', initial_capital):,.2f}")
        print(f"Total P&L: ${state.get('total_pnl', 0):,.2f}")
        print(f"Total Trades: {state.get('total_trades', 0)}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Regime-Based Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_regime_trading.py --mode backtest --symbol BTC/USDT
  python run_regime_trading.py --mode paper --symbols BTC/USDT,ETH/USDT,SPY
  python run_regime_trading.py --mode backtest --symbol SPY --days 730

Available modes:
  backtest  - Run historical backtest (single symbol)
  paper     - Run paper trading (multiple symbols supported)
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["backtest", "paper"],
        default="backtest",
        help="Trading mode",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading symbol (for backtest mode)",
    )
    parser.add_argument(
        "--symbols",
        default="BTC/USDT,ETH/USDT,SOL/USDT,SPY,GLD",
        help="Comma-separated list of symbols for paper trading (e.g., BTC/USDT,ETH/USDT,SOL/USDT,SPY,GLD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history for backtest",
    )

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(symbol=args.symbol, days=args.days)
    elif args.mode == "paper":
        # Parse comma-separated symbols
        symbols = [s.strip() for s in args.symbols.split(",")]
        asyncio.run(run_paper_trading(symbols=symbols))


if __name__ == "__main__":
    main()
