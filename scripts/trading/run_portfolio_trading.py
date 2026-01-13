#!/usr/bin/env python3
"""
Multi-Asset Portfolio Trading Runner.

Demonstrates the portfolio optimizer integration with smart trading.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load unified configuration
from bot.config import load_config, get_data_dir

app_config = load_config()

from bot.multi_asset_engine import (
    MultiAssetTradingEngine,
    MultiAssetConfig,
    AssetConfig,
)
from bot.portfolio_optimizer import OptimizationMethod


def generate_sample_ohlcv(
    symbol: str,
    days: int = 90,
    base_price: float = 100.0,
    volatility: float = 0.03,
    trend: float = 0.0005,
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(hash(symbol) % 2**32)

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="1h")

    # Generate price with trend and noise
    returns = np.random.normal(trend, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Add intraday variation
    high_factor = 1 + np.abs(np.random.normal(0, volatility/2, len(dates)))
    low_factor = 1 - np.abs(np.random.normal(0, volatility/2, len(dates)))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, volatility/4, len(dates))),
        "high": prices * high_factor,
        "low": prices * low_factor,
        "close": prices,
        "volume": np.random.uniform(1000, 10000, len(dates)),
    }, index=dates)

    # Ensure high >= close >= low and high >= open >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def run_portfolio_demo():
    """Run portfolio trading demonstration."""
    print("=" * 70)
    print("MULTI-ASSET PORTFOLIO TRADING DEMO")
    print("=" * 70)

    # Build asset configs from unified config
    crypto_assets = [
        AssetConfig(
            symbol=symbol,
            max_weight=app_config.crypto.max_weight,
            min_weight=app_config.crypto.min_weight,
        )
        for symbol in app_config.crypto.symbols
    ]

    # Map optimization method string to enum
    opt_method_map = {
        "equal_weight": OptimizationMethod.EQUAL_WEIGHT,
        "risk_parity": OptimizationMethod.RISK_PARITY,
        "min_volatility": OptimizationMethod.MIN_VOLATILITY,
        "max_sharpe": OptimizationMethod.MAX_SHARPE,
        "max_diversification": OptimizationMethod.MAX_DIVERSIFICATION,
        "inverse_volatility": OptimizationMethod.INVERSE_VOLATILITY,
    }
    opt_method = opt_method_map.get(
        app_config.portfolio.optimization_method,
        OptimizationMethod.RISK_PARITY
    )

    # Configure portfolio using unified config
    config = MultiAssetConfig(
        assets=crypto_assets,
        optimization_method=opt_method,
        rebalance_threshold=app_config.trading.rebalance_threshold,
        rebalance_frequency=app_config.portfolio.rebalance_frequency,
        total_capital=app_config.trading.initial_capital,
        use_correlation_filter=app_config.portfolio.use_correlation_filter,
        max_correlation=app_config.portfolio.max_correlation,
    )

    print(f"\nPortfolio Configuration (from config.yaml):")
    print(f"  Total Capital: ${config.total_capital:,.2f}")
    print(f"  Optimization Method: {config.optimization_method.value}")
    print(f"  Rebalance Threshold: {config.rebalance_threshold:.0%}")
    print(f"  Assets: {len(config.assets)}")

    # Initialize engine
    print("\nInitializing Multi-Asset Trading Engine...")
    engine = MultiAssetTradingEngine(
        config=config,
        data_dir=str(get_data_dir() / "portfolio"),
    )

    # Generate sample market data
    print("\nGenerating sample market data...")
    market_data = {
        "BTC/USDT": generate_sample_ohlcv("BTC", days=90, base_price=45000, volatility=0.025, trend=0.0003),
        "ETH/USDT": generate_sample_ohlcv("ETH", days=90, base_price=2500, volatility=0.03, trend=0.0002),
        "SOL/USDT": generate_sample_ohlcv("SOL", days=90, base_price=100, volatility=0.04, trend=0.0001),
        "AVAX/USDT": generate_sample_ohlcv("AVAX", days=90, base_price=35, volatility=0.045, trend=0.00005),
        "MATIC/USDT": generate_sample_ohlcv("MATIC", days=90, base_price=0.85, volatility=0.04, trend=0.0001),
    }

    # Print current prices
    print("\nCurrent Market Prices:")
    for symbol, df in market_data.items():
        print(f"  {symbol}: ${df['close'].iloc[-1]:,.2f}")

    # Run portfolio analysis
    print("\n" + "=" * 70)
    print("RUNNING PORTFOLIO ANALYSIS")
    print("=" * 70)

    decision = engine.analyze_portfolio(market_data)
    decision.print_summary()

    # Execute rebalancing if needed
    if decision.rebalance_needed:
        print("\n" + "=" * 70)
        print("EXECUTING REBALANCE")
        print("=" * 70)

        result = engine.execute_rebalance(decision)
        print(f"\nExecution Status: {result['status']}")
        print(f"New Portfolio Value: ${result.get('portfolio_value', 0):,.2f}")
        print(f"New Cash Balance: ${result.get('new_cash_balance', 0):,.2f}")

        if result.get('trades'):
            print("\nExecuted Trades:")
            for trade in result['trades']:
                if trade.get('status') == 'executed':
                    print(f"  {trade['action']} {trade['quantity']:.6f} {trade['symbol']} @ ${trade['price']:,.2f}")

    # Show portfolio status
    print("\n" + "=" * 70)
    print("PORTFOLIO STATUS")
    print("=" * 70)

    status = engine.get_portfolio_status()
    print(f"\nTotal Value: ${status['total_value']:,.2f}")
    print(f"Cash Balance: ${status['cash_balance']:,.2f}")

    if status['positions']:
        print("\nPositions:")
        for symbol, pos in status['positions'].items():
            print(f"  {symbol}: {pos['quantity']:.6f} @ ${pos['price']:,.2f} = ${pos['value']:,.2f}")

    if status['current_weights']:
        print("\nCurrent Weights:")
        for symbol, weight in status['current_weights'].items():
            print(f"  {symbol}: {weight:.2%}")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    corr_analysis = engine.get_correlation_analysis()
    if "error" not in corr_analysis:
        print(f"\nAverage Correlation: {corr_analysis['average_correlation']:.2f}")
        if corr_analysis.get('lowest_correlation'):
            low = corr_analysis['lowest_correlation']
            print(f"Lowest Correlated Pair: {low['asset1']}/{low['asset2']} ({low['correlation']:.2f})")
        if corr_analysis.get('highest_correlation'):
            high = corr_analysis['highest_correlation']
            print(f"Highest Correlated Pair: {high['asset1']}/{high['asset2']} ({high['correlation']:.2f})")

    # Compare optimization methods
    print("\n" + "=" * 70)
    print("OPTIMIZATION METHOD COMPARISON")
    print("=" * 70)

    methods = [
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.MIN_VOLATILITY,
        OptimizationMethod.MAX_SHARPE,
    ]

    # Build returns for comparison
    returns_data = {}
    for symbol, df in market_data.items():
        returns_data[symbol] = df["close"].pct_change().dropna()

    min_len = min(len(r) for r in returns_data.values())
    returns_df = pd.DataFrame({s: r.tail(min_len).values for s, r in returns_data.items()})

    print(f"\n{'Method':<20} {'Sharpe':<10} {'Volatility':<12} {'Return':<12}")
    print("-" * 54)

    for method in methods:
        result = engine.portfolio_optimizer.optimize(returns_df, method)
        m = result.metrics
        print(f"{method.value:<20} {m.sharpe_ratio:<10.2f} {m.volatility:<12.2%} {m.expected_return:<12.2%}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    return engine, decision


def run_live_portfolio_loop(
    engine: MultiAssetTradingEngine,
    market_data: dict,
    iterations: int = 5,
    interval_seconds: int = 10,
):
    """Run a simulated live trading loop."""
    print("\n" + "=" * 70)
    print("STARTING LIVE PORTFOLIO TRADING LOOP")
    print(f"Iterations: {iterations}, Interval: {interval_seconds}s")
    print("=" * 70)

    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")

        # Simulate price updates (small random changes)
        for symbol in market_data:
            last_price = market_data[symbol]["close"].iloc[-1]
            change = np.random.normal(0, 0.005)  # 0.5% volatility
            new_price = last_price * (1 + change)

            # Add new row
            new_row = pd.DataFrame({
                "open": [last_price],
                "high": [max(last_price, new_price) * 1.001],
                "low": [min(last_price, new_price) * 0.999],
                "close": [new_price],
                "volume": [np.random.uniform(1000, 5000)],
            }, index=[datetime.now()])

            market_data[symbol] = pd.concat([market_data[symbol], new_row])

        # Analyze portfolio
        decision = engine.analyze_portfolio(market_data)

        print(f"Portfolio Value: ${decision.total_value:,.2f}")
        print(f"Rebalance Needed: {'Yes' if decision.rebalance_needed else 'No'}")

        if decision.rebalance_needed and decision.trades_to_execute:
            print("Pending trades:")
            for trade in decision.trades_to_execute[:3]:
                print(f"  {trade['action']} {trade['symbol']}: ${trade.get('amount_usd', 0):,.2f}")

        if i < iterations - 1:
            print(f"Waiting {interval_seconds}s...")
            time.sleep(interval_seconds)

    print("\n" + "=" * 70)
    print("LIVE LOOP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    engine, decision = run_portfolio_demo()

    # Optionally run live loop
    # Uncomment to test live trading simulation:
    # run_live_portfolio_loop(engine, market_data, iterations=3, interval_seconds=5)
