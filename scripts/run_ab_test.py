#!/usr/bin/env python3
"""
Example A/B Test Script

Demonstrates how to use the A/B testing framework to compare strategies.

Usage:
    python scripts/run_ab_test.py
    python scripts/run_ab_test.py --symbol BTC/USDT --bars 1000
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.testing.ab_framework import (
    ABTest,
    ABTestConfig,
    SimpleMovingAverageStrategy,
    RSIStrategy,
    run_ab_test,
)
from bot.strategy_interface import (
    EMACrossoverStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    MomentumStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_bars: int = 1000,
    symbol: str = "BTC/USDT",
    start_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
) -> list[dict]:
    """
    Generate synthetic market data for testing.
    
    Creates OHLCV data with pre-computed indicators.
    """
    np.random.seed(42)  # Reproducible
    
    bars = []
    price = start_price
    timestamp = datetime.now() - timedelta(minutes=n_bars)
    
    # Price history for indicators
    closes = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(n_bars):
        # Generate OHLCV
        change = np.random.normal(trend, volatility)
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
        volume = np.random.uniform(100, 1000) * (1 + abs(change) * 10)
        
        closes.append(close_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
        
        # Compute indicators (need enough history)
        indicators = {}
        
        if len(closes) >= 26:
            # EMAs
            ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
            indicators["ema_fast"] = ema_12
            indicators["ema_slow"] = ema_26
            indicators["ema_12"] = ema_12
            indicators["ema_26"] = ema_26
            
            # RSI
            deltas = pd.Series(closes).diff()
            gains = deltas.where(deltas > 0, 0).rolling(14).mean()
            losses = (-deltas.where(deltas < 0, 0)).rolling(14).mean()
            rs = gains / losses.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi"] = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
            indicators["rsi_14"] = indicators["rsi"]
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = pd.Series([macd]).ewm(span=9).mean().iloc[-1]
            indicators["macd"] = macd
            indicators["macd_signal"] = macd_signal
            indicators["macd_hist"] = macd - macd_signal
            
            # Bollinger Bands
            sma_20 = pd.Series(closes).rolling(20).mean().iloc[-1]
            std_20 = pd.Series(closes).rolling(20).std().iloc[-1]
            indicators["bb_upper"] = sma_20 + 2 * std_20
            indicators["bb_middle"] = sma_20
            indicators["bb_lower"] = sma_20 - 2 * std_20
            
            # ATR (simplified)
            tr = [max(h - l, abs(h - closes[max(0, j-1)]), abs(l - closes[max(0, j-1)])) 
                  for j, (h, l) in enumerate(zip(highs[-14:], lows[-14:]))]
            indicators["atr"] = np.mean(tr) if tr else 0
            indicators["atr_14"] = indicators["atr"]
            
            # ADX (simplified approximation)
            indicators["adx"] = 25 + np.random.uniform(-10, 10)
            indicators["adx_14"] = indicators["adx"]
            
            # High/Low channels
            indicators["high_20"] = max(highs[-20:]) if len(highs) >= 20 else high_price
            indicators["low_20"] = min(lows[-20:]) if len(lows) >= 20 else low_price
        
        bar = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            **indicators,
        }
        
        bars.append(bar)
        price = close_price
        timestamp += timedelta(minutes=1)
    
    return bars


def run_example_test():
    """Run an example A/B test with built-in strategies"""
    
    logger.info("=" * 60)
    logger.info("A/B TESTING FRAMEWORK - EXAMPLE")
    logger.info("=" * 60)
    
    # Generate test data
    logger.info("Generating synthetic market data...")
    market_data = generate_synthetic_data(
        n_bars=2000,
        symbol="BTC/USDT",
        start_price=50000,
        volatility=0.015,
        trend=0.0001,
    )
    logger.info(f"Generated {len(market_data)} bars")
    
    # Configure test
    config = ABTestConfig(
        initial_balance=10000.0,
        trading_fee_pct=0.1,
        slippage_pct=0.05,
        symbols=["BTC/USDT"],
        warmup_bars=50,
        max_drawdown_stop=30.0,
        confidence_level=0.95,
        save_trade_history=True,
        save_equity_curve=True,
    )
    
    # Create test
    ab_test = ABTest(config)
    
    # Register strategies to compare
    logger.info("Registering strategies...")
    
    # Strategy A: EMA Crossover (trend following)
    ab_test.register_strategy("ema_crossover", EMACrossoverStrategy({
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
    }))
    
    # Strategy B: Momentum (MACD + ADX)
    ab_test.register_strategy("momentum", MomentumStrategy({
        "stop_loss_pct": 2.5,
        "take_profit_pct": 5.0,
    }))
    
    # Strategy C: Mean Reversion (Bollinger + RSI)
    ab_test.register_strategy("mean_reversion", MeanReversionStrategy({
        "stop_loss_pct": 1.5,
        "take_profit_pct": 2.0,
    }))
    
    # Strategy D: Breakout
    ab_test.register_strategy("breakout", BreakoutStrategy({
        "position_size_pct": 5.0,
    }))
    
    logger.info(f"Registered {len(ab_test.strategies)} strategies")
    
    # Run test with progress
    def progress_callback(current, total):
        if current % 500 == 0:
            pct = (current / total) * 100
            logger.info(f"Progress: {current}/{total} bars ({pct:.1f}%)")
    
    logger.info("Running A/B test...")
    result = ab_test.run(market_data, progress_callback=progress_callback)
    
    # Print results
    result.print_summary()
    
    # Save results
    output_dir = Path("ab_test_results")
    output_dir.mkdir(exist_ok=True)
    
    dashboard_path = output_dir / f"dashboard_{result.test_id}.json"
    result.save_dashboard(dashboard_path)
    logger.info(f"Dashboard saved to: {dashboard_path}")
    
    full_path = output_dir / f"full_results_{result.test_id}.json"
    result.save_full_results(full_path)
    logger.info(f"Full results saved to: {full_path}")
    
    # Print dashboard JSON
    print("\n" + "=" * 60)
    print("DASHBOARD JSON:")
    print("=" * 60)
    print(json.dumps(result.to_dashboard(), indent=2))
    
    return result


def run_quick_comparison():
    """Quick comparison using the convenience function"""
    
    logger.info("Running quick A/B comparison...")
    
    # Generate data
    market_data = generate_synthetic_data(n_bars=1000)
    
    # Run comparison
    result = run_ab_test(
        strategies={
            "simple_sma": SimpleMovingAverageStrategy(),
            "rsi_extremes": RSIStrategy(),
        },
        market_data=market_data,
        config=ABTestConfig(initial_balance=10000),
        output_path="ab_test_results/quick_comparison.json",
    )
    
    result.print_summary()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run A/B test on trading strategies")
    parser.add_argument("--quick", action="store_true", help="Run quick comparison")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--bars", type=int, default=2000, help="Number of bars")
    args = parser.parse_args()
    
    if args.quick:
        run_quick_comparison()
    else:
        run_example_test()


if __name__ == "__main__":
    main()
