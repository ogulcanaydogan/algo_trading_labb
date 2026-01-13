#!/usr/bin/env python3
"""
Strategy Backtest Runner.

Tests all strategies against historical data to compare performance.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from bot.strategies import (
    EMACrossoverStrategy,
    BollingerBandStrategy,
    BollingerBandConfig,
    MACDDivergenceStrategy,
    RSIMeanReversionStrategy,
    StochasticDivergenceStrategy,
    KeltnerChannelStrategy,
    GridTradingStrategy,
)
from bot.strategies.base import StrategySignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data for backtesting."""
    try:
        import yfinance as yf

        # Convert symbol for Yahoo
        if "/" in symbol:
            yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
        else:
            yf_symbol = symbol

        logger.info(f"Fetching {days} days of data for {symbol}")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        return df.dropna()

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def backtest_strategy(
    strategy,
    ohlcv: pd.DataFrame,
    initial_capital: float = 10000,
    position_size_pct: float = 0.1,
    commission_pct: float = 0.001,
) -> Dict:
    """
    Run backtest on a strategy.

    Args:
        strategy: Strategy instance
        ohlcv: OHLCV DataFrame
        initial_capital: Starting capital
        position_size_pct: Position size as percentage of capital
        commission_pct: Commission per trade

    Returns:
        Dictionary with backtest results
    """
    results = {
        "strategy": strategy.name,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "win_rate": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pnl": 0.0,
        "signals": [],
    }

    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    position = None
    trade_pnls = []
    gross_profit = 0.0
    gross_loss = 0.0

    # Need minimum data for indicators
    min_rows = 200
    if len(ohlcv) < min_rows:
        logger.warning(f"Insufficient data for {strategy.name}")
        return results

    # Iterate through data with rolling window
    for i in range(min_rows, len(ohlcv)):
        # Get window of data
        window = ohlcv.iloc[:i+1].copy()
        current_price = window["close"].iloc[-1]
        timestamp = window.index[-1]

        try:
            signal = strategy.generate_signal(window)
        except Exception as e:
            continue

        # Process signal
        if position is None:
            # No position - look for entry
            if signal.decision == "LONG" and signal.confidence > 0.5:
                position = {
                    "side": "long",
                    "entry_price": current_price,
                    "size": (capital * position_size_pct) / current_price,
                    "entry_time": timestamp,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                }
                results["signals"].append({
                    "time": str(timestamp),
                    "type": "entry_long",
                    "price": current_price,
                    "confidence": signal.confidence,
                })

            elif signal.decision == "SHORT" and signal.confidence > 0.5:
                position = {
                    "side": "short",
                    "entry_price": current_price,
                    "size": (capital * position_size_pct) / current_price,
                    "entry_time": timestamp,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                }
                results["signals"].append({
                    "time": str(timestamp),
                    "type": "entry_short",
                    "price": current_price,
                    "confidence": signal.confidence,
                })

        else:
            # Have position - check for exit
            should_exit = False
            exit_reason = ""

            # Check stop loss
            if position["stop_loss"]:
                if position["side"] == "long" and current_price <= position["stop_loss"]:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif position["side"] == "short" and current_price >= position["stop_loss"]:
                    should_exit = True
                    exit_reason = "stop_loss"

            # Check take profit
            if position["take_profit"]:
                if position["side"] == "long" and current_price >= position["take_profit"]:
                    should_exit = True
                    exit_reason = "take_profit"
                elif position["side"] == "short" and current_price <= position["take_profit"]:
                    should_exit = True
                    exit_reason = "take_profit"

            # Check signal reversal
            if signal.decision == "FLAT":
                should_exit = True
                exit_reason = "signal_flat"
            elif signal.decision == "LONG" and position["side"] == "short":
                should_exit = True
                exit_reason = "signal_reversal"
            elif signal.decision == "SHORT" and position["side"] == "long":
                should_exit = True
                exit_reason = "signal_reversal"

            if should_exit:
                # Calculate PnL
                if position["side"] == "long":
                    pnl = (current_price - position["entry_price"]) * position["size"]
                else:  # short
                    pnl = (position["entry_price"] - current_price) * position["size"]

                # Deduct commission
                commission = (position["entry_price"] + current_price) * position["size"] * commission_pct
                pnl -= commission

                # Update stats
                results["trades"] += 1
                trade_pnls.append(pnl)

                if pnl > 0:
                    results["wins"] += 1
                    gross_profit += pnl
                else:
                    results["losses"] += 1
                    gross_loss += abs(pnl)

                capital += pnl

                results["signals"].append({
                    "time": str(timestamp),
                    "type": f"exit_{exit_reason}",
                    "price": current_price,
                    "pnl": round(pnl, 2),
                })

                # Track drawdown
                if capital > peak_capital:
                    peak_capital = capital
                drawdown = (peak_capital - capital) / peak_capital
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                position = None

    # Calculate final metrics
    results["total_pnl"] = capital - initial_capital
    results["total_return_pct"] = ((capital - initial_capital) / initial_capital) * 100
    results["max_drawdown_pct"] = max_drawdown * 100

    if results["trades"] > 0:
        results["win_rate"] = results["wins"] / results["trades"]
        results["avg_trade_pnl"] = np.mean(trade_pnls) if trade_pnls else 0

        # Sharpe ratio (annualized, assuming hourly data)
        if len(trade_pnls) > 1:
            returns = np.array(trade_pnls) / initial_capital
            results["sharpe_ratio"] = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0

        # Profit factor
        if gross_loss > 0:
            results["profit_factor"] = gross_profit / gross_loss
        elif gross_profit > 0:
            results["profit_factor"] = float("inf")

    return results


def run_all_strategies(
    symbol: str,
    ohlcv: pd.DataFrame,
    initial_capital: float = 10000,
) -> Dict[str, Dict]:
    """Run backtest for all strategies."""
    strategies = {
        "EMA Crossover": EMACrossoverStrategy(),
        "Bollinger Mean Reversion": BollingerBandStrategy(BollingerBandConfig(mode="mean_reversion")),
        "Bollinger Breakout": BollingerBandStrategy(BollingerBandConfig(mode="breakout")),
        "MACD Divergence": MACDDivergenceStrategy(),
        "RSI Mean Reversion": RSIMeanReversionStrategy(),
        "Stochastic Divergence": StochasticDivergenceStrategy(),
        "Keltner Channel": KeltnerChannelStrategy(),
        "Grid Trading": GridTradingStrategy(),
    }

    results = {}

    for name, strategy in strategies.items():
        logger.info(f"Testing {name}...")
        try:
            result = backtest_strategy(strategy, ohlcv, initial_capital)
            results[name] = result
        except Exception as e:
            logger.error(f"Failed to backtest {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def print_results(results: Dict[str, Dict], symbol: str) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS - {symbol}")
    print("=" * 80)

    # Sort by total return
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("total_return_pct", -999),
        reverse=True,
    )

    print(f"\n{'Strategy':<25} {'Return %':>10} {'Win Rate':>10} {'Trades':>8} {'Sharpe':>8} {'Max DD %':>10} {'PF':>8}")
    print("-" * 80)

    for name, result in sorted_results:
        if "error" in result:
            print(f"{name:<25} {'ERROR':>10}")
            continue

        print(
            f"{name:<25} "
            f"{result['total_return_pct']:>9.2f}% "
            f"{result['win_rate']*100:>9.1f}% "
            f"{result['trades']:>8} "
            f"{result['sharpe_ratio']:>8.2f} "
            f"{result['max_drawdown_pct']:>9.2f}% "
            f"{result['profit_factor']:>8.2f}"
        )

    print("=" * 80)

    # Best strategy
    if sorted_results:
        best = sorted_results[0]
        if "error" not in best[1]:
            print(f"\nBest Strategy: {best[0]}")
            print(f"  Return: {best[1]['total_return_pct']:.2f}%")
            print(f"  Win Rate: {best[1]['win_rate']*100:.1f}%")
            print(f"  Trades: {best[1]['trades']}")


def main():
    parser = argparse.ArgumentParser(description="Strategy Backtest Runner")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol to test")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols to test")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--output", help="Output CSV file for results")

    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]
    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Backtesting {symbol}")
        logger.info(f"{'='*40}")

        ohlcv = fetch_data(symbol, args.days)
        if ohlcv is None or len(ohlcv) < 200:
            logger.warning(f"Skipping {symbol} - insufficient data")
            continue

        logger.info(f"Got {len(ohlcv)} candles")

        results = run_all_strategies(symbol, ohlcv, args.capital)
        all_results[symbol] = results
        print_results(results, symbol)

    # Save to CSV if requested
    if args.output:
        rows = []
        for symbol, results in all_results.items():
            for strategy, metrics in results.items():
                if "error" not in metrics:
                    rows.append({
                        "symbol": symbol,
                        "strategy": strategy,
                        "return_pct": metrics["total_return_pct"],
                        "win_rate": metrics["win_rate"],
                        "trades": metrics["trades"],
                        "sharpe": metrics["sharpe_ratio"],
                        "max_drawdown_pct": metrics["max_drawdown_pct"],
                        "profit_factor": metrics["profit_factor"],
                    })

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
