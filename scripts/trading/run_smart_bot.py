#!/usr/bin/env python3
"""
Smart Bot Runner - AI-Enhanced Trading Bot.

This script runs the trading bot with full AI enhancements:
- Market regime detection for strategy selection
- ML-based predictions
- Multi-strategy voting
- Dynamic risk adjustment
- Optional LLM-powered analysis

Usage:
    python run_smart_bot.py [options]

Options:
    --symbol        Trading symbol (default: BTC/USDT)
    --timeframe     Candle timeframe (default: 1h)
    --paper         Run in paper trading mode
    --train         Train ML model before starting
    --interval      Loop interval in seconds (default: 60)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.strategy import StrategyConfig
from bot.state import StateStore, create_state_store
from bot.smart_engine import SmartTradingEngine, EngineConfig, TradingDecision

# Market data
try:
    from bot.market_data import YFinanceMarketDataClient
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from bot.exchange import ExchangeClient, PaperExchangeClient
    HAS_EXCHANGE = True
except ImportError:
    HAS_EXCHANGE = False


def create_market_client(symbol: str, paper_mode: bool):
    """Create appropriate market data client."""
    if paper_mode and HAS_EXCHANGE:
        print("Using Paper Exchange (synthetic data)")
        return PaperExchangeClient()

    if HAS_YFINANCE:
        print("Using YFinance for market data")
        return YFinanceMarketDataClient()

    if HAS_EXCHANGE:
        print("Using CCXT Exchange client")
        return ExchangeClient()

    raise RuntimeError("No market data client available")


def fetch_market_data(client, symbol: str, timeframe: str, lookback: int = 300):
    """Fetch OHLCV data from the market client."""
    try:
        if hasattr(client, 'fetch_ohlcv_df'):
            return client.fetch_ohlcv_df(symbol, timeframe, lookback)
        elif hasattr(client, 'fetch_ohlcv'):
            return client.fetch_ohlcv(symbol, timeframe, lookback)
        else:
            # YFinance client
            import yfinance as yf
            yf_symbol = symbol.replace("/", "-").replace("USDT", "USD")
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="60d", interval=timeframe)
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def run_smart_bot(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    paper_mode: bool = True,
    loop_interval: int = 60,
    train_first: bool = False,
    data_dir: str = "data",
):
    """
    Run the smart trading bot.

    Args:
        symbol: Trading symbol
        timeframe: Candle timeframe
        paper_mode: Use paper trading
        loop_interval: Seconds between iterations
        train_first: Train ML model before starting
        data_dir: Directory for state and models
    """
    print("="*60)
    print("SMART TRADING BOT")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"Loop Interval: {loop_interval}s")
    print("="*60)

    # Initialize components
    strategy_config = StrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
    )

    engine_config = EngineConfig(
        use_ml=True,
        use_multi_strategy=True,
        use_llm=True,
        min_confidence=0.4,
    )

    engine = SmartTradingEngine(
        config=engine_config,
        strategy_config=strategy_config,
        data_dir=data_dir,
    )

    state_store = create_state_store(Path(data_dir))
    market_client = create_market_client(symbol, paper_mode)

    # Train ML model if requested
    if train_first:
        print("\nTraining ML model...")
        df = fetch_market_data(market_client, symbol, timeframe, lookback=500)
        if df is not None and len(df) > 200:
            metrics = engine.train_ml(df)
            print(f"Training complete: Accuracy={metrics.get('accuracy', 0):.1%}")
        else:
            print("Not enough data for training, skipping...")

    # Load existing model if available
    if engine.ml_predictor:
        loaded = engine.ml_predictor.load("smart_engine")
        if loaded:
            print("Loaded existing ML model")

    print(f"\nStarting bot loop... (Ctrl+C to stop)")
    print("-"*60)

    iteration = 0
    last_decision: TradingDecision = None

    try:
        while True:
            iteration += 1
            loop_start = time.time()

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iteration {iteration}")

            # Fetch market data
            df = fetch_market_data(market_client, symbol, timeframe)

            if df is None or len(df) < 50:
                print("  Insufficient data, waiting...")
                time.sleep(loop_interval)
                continue

            current_price = float(df["close"].iloc[-1])
            print(f"  Current Price: ${current_price:,.2f}")

            # Get smart decision
            try:
                decision = engine.analyze(df)
                last_decision = decision

                print(f"  Regime: {decision.regime} ({decision.regime_confidence:.0%})")
                print(f"  Decision: {decision.action} ({decision.confidence:.0%})")
                print(f"  Source: {decision.source}")
                print(f"  Position Multiplier: {decision.position_size_multiplier:.2f}x")

                if decision.stop_loss and decision.take_profit:
                    print(f"  SL: ${decision.stop_loss:,.2f} | TP: ${decision.take_profit:,.2f}")

                # Update state
                state_store.load()
                state = state_store.state
                state.symbol = symbol
                state.timestamp = datetime.now()
                state.last_signal = decision.action
                state.confidence = decision.confidence
                state.ai_action = decision.action
                state.ai_confidence = decision.ml_probability

                # Record signal
                signal_record = {
                    "timestamp": datetime.now().isoformat(),
                    "decision": decision.action,
                    "confidence": decision.confidence,
                    "regime": decision.regime,
                    "price": current_price,
                    "source": decision.source,
                }
                state_store.add_signal(signal_record)

                # Record equity
                state_store.add_equity_point(
                    timestamp=datetime.now(),
                    balance=state.balance,
                )

                state_store.save()

            except Exception as e:
                print(f"  Error in analysis: {e}")

            # Check if should retrain
            if engine.should_retrain():
                print("  ML model should be retrained (>30 days old)")

            # Calculate sleep time
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - elapsed)

            print(f"  Loop took {elapsed:.1f}s, sleeping {sleep_time:.0f}s...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nShutting down...")

        # Print final summary
        if last_decision:
            print("\n" + "="*60)
            print("FINAL STATUS")
            print("="*60)
            print(f"Last Decision: {last_decision.action}")
            print(f"Market Regime: {last_decision.regime}")
            print(f"Total Iterations: {iteration}")

            # Get performance summary
            status = engine.get_status()
            print(f"\nML Model: {'Trained' if status['ml_status']['is_trained'] else 'Not trained'}")
            print(f"Strategies Available: {len(status['available_strategies'])}")

        print("\nBot stopped.")


def main():
    parser = argparse.ArgumentParser(description="Smart Trading Bot with AI")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode (USE WITH CAUTION)")
    parser.add_argument("--train", action="store_true", help="Train ML model before starting")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()

    load_dotenv()

    paper_mode = not args.live

    if args.live:
        print("\n" + "!"*60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute real trades with real money!")
        print("!"*60)
        confirm = input("Type 'CONFIRM' to continue: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    run_smart_bot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        paper_mode=paper_mode,
        loop_interval=args.interval,
        train_first=args.train,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
