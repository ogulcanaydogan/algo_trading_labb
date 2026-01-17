"""
Example multi-asset portfolio script.
Demonstrates portfolio management, signal aggregation, and rebalancing.
"""

import asyncio
from bot.portfolio_config import Portfolio, Asset, AssetType, PortfolioLoader
from bot.portfolio_manager import PortfolioManager
from bot.multi_asset_signals import MultiAssetSignalAggregator, AssetSignal, Signal
from datetime import datetime
import json

async def main():
    """Run example multi-asset portfolio workflow."""
    
    # 1. Create or load portfolio
    print("=" * 70)
    print("MULTI-ASSET PORTFOLIO MANAGER - EXAMPLE")
    print("=" * 70)
    
    portfolio = Portfolio(
        name="Sample Multi-Asset Portfolio",
        total_capital=100000,
        assets=[
            Asset(
                symbol="BTC/USDT",
                asset_type=AssetType.CRYPTO,
                data_source="binance",
                allocation_pct=30,
                risk_limit_pct=2.0,
            ),
            Asset(
                symbol="AAPL",
                asset_type=AssetType.EQUITY,
                data_source="yfinance",
                allocation_pct=25,
                risk_limit_pct=2.0,
            ),
            Asset(
                symbol="GC=F",
                asset_type=AssetType.COMMODITY,
                data_source="yfinance",
                allocation_pct=20,
                risk_limit_pct=2.0,
            ),
            Asset(
                symbol="SPY",
                asset_type=AssetType.ETF,
                data_source="yfinance",
                allocation_pct=25,
                risk_limit_pct=2.0,
            ),
        ]
    )
    
    # Validate
    valid, errors = portfolio.validate()
    if not valid:
        print("Portfolio validation errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    print(f"\n✓ Portfolio '{portfolio.name}' validated")
    print(f"  Total Capital: ${portfolio.total_capital:,.0f}")
    print(f"  Assets: {len(portfolio.assets)}")
    
    # 2. Initialize portfolio manager
    pm = PortfolioManager(portfolio)
    
    # 3. Simulate initial positions
    print(f"\n--- INITIAL POSITIONS ---")
    initial_prices = {
        "BTC/USDT": 43000,
        "AAPL": 185,
        "GC=F": 2050,
        "SPY": 475,
    }
    
    for symbol, price in initial_prices.items():
        quantity = pm.calculate_position_size(symbol, price)
        pm.update_position(symbol, quantity, price)
        print(f"{symbol:12} | Price: ${price:8,.2f} | Qty: {quantity:10,.4f}")
    
    # 4. Show initial portfolio state
    summary = pm.get_portfolio_summary()
    print(f"\n--- PORTFOLIO SUMMARY ---")
    print(f"Total Equity:      ${summary['total_equity']:,.2f}")
    print(f"Cash:              ${summary['cash']:,.2f}")
    print(f"Invested:          ${summary['invested']:,.2f}")
    print(f"Unrealized P&L:    ${summary['unrealized_pnl']:,.2f} ({summary['unrealized_pnl_pct']:.2f}%)")
    
    # 5. Show allocations
    print(f"\n--- ALLOCATIONS ---")
    print(f"{'Symbol':<12} | {'Target %':>10} | {'Current %':>10} | {'Drift %':>10}")
    print("-" * 50)
    
    for asset in portfolio.assets:
        current_alloc = summary['allocation'].get(asset.symbol, 0)
        drift = current_alloc - asset.allocation_pct
        print(f"{asset.symbol:<12} | {asset.allocation_pct:>9.1f}% | {current_alloc:>9.1f}% | {drift:>9.1f}%")
    
    # 6. Simulate price changes
    print(f"\n--- PRICE UPDATES (simulating market movement) ---")
    updated_prices = {
        "BTC/USDT": 44500,   # +3.5%
        "AAPL": 182,         # -1.6%
        "GC=F": 2080,        # +1.5%
        "SPY": 480,          # +1.1%
    }
    
    pm.update_prices(updated_prices)
    
    for symbol, new_price in updated_prices.items():
        old_price = initial_prices[symbol]
        change_pct = ((new_price - old_price) / old_price) * 100
        print(f"{symbol:12} | ${old_price:8,.2f} → ${new_price:8,.2f} ({change_pct:+.1f}%)")
    
    # 7. Check rebalancing needs
    needs_rebal, assets = pm.needs_rebalancing()
    print(f"\n--- REBALANCING STATUS ---")
    print(f"Needs Rebalancing: {needs_rebal}")
    if assets:
        print(f"Assets to adjust:  {', '.join(assets)}")
    
    # 8. Calculate rebalancing trades
    trades = pm.calculate_rebalancing_trades()
    if trades:
        print(f"\n--- RECOMMENDED TRADES ---")
        for symbol, qty_delta in trades.items():
            current_pos = pm.state.positions.get(symbol)
            if current_pos:
                current_qty = current_pos.quantity
                new_qty = current_qty + qty_delta
                print(f"{symbol:12} | {current_qty:10,.4f} → {new_qty:10,.4f} (Δ {qty_delta:+.4f})")
    
    # 9. Generate signals
    print(f"\n--- ASSET SIGNALS ---")
    aggregator = MultiAssetSignalAggregator()
    
    signals = [
        AssetSignal("BTC/USDT", Signal.LONG, 0.75, "EMA bullish + momentum", datetime.utcnow(), {"ema_gap": 0.5}),
        AssetSignal("AAPL", Signal.FLAT, 0.60, "Consolidation phase", datetime.utcnow(), {"rsi": 55}),
        AssetSignal("GC=F", Signal.LONG, 0.70, "Flight to safety", datetime.utcnow(), {"volatility": 2.1}),
        AssetSignal("SPY", Signal.LONG, 0.65, "Tech strength", datetime.utcnow(), {"ema_slope": 0.3}),
    ]
    
    for sig in signals:
        aggregator.add_asset_signal(sig)
        print(f"{sig.symbol:12} | {sig.signal.value:>6} | Conf: {sig.confidence:.2f} | {sig.reason}")
    
    # 10. Portfolio sentiment
    sentiment, counts = aggregator.get_portfolio_sentiment()
    consensus_sig, consensus_conf = aggregator.get_consensus_signal()
    
    print(f"\n--- PORTFOLIO SENTIMENT ---")
    print(f"Consensus Signal:   {consensus_sig.value} (Confidence: {consensus_conf:.2f})")
    print(f"Sentiment Score:    {sentiment:+.2f} (-1.0=bearish, +1.0=bullish)")
    print(f"Signal Distribution: {counts['LONG']} LONG, {counts['SHORT']} SHORT, {counts['FLAT']} FLAT")
    
    # 11. Diversification
    hhi = pm.get_portfolio_diversification()
    print(f"\n--- DIVERSIFICATION ---")
    print(f"Herfindahl-Hirschman Index: {hhi:.0f}")
    print(f"Level: {'Excellent' if hhi < 1500 else 'Good' if hhi < 2500 else 'Fair' if hhi < 5000 else 'Poor'}")
    
    # 12. Final summary
    updated_summary = pm.get_portfolio_summary()
    print(f"\n--- FINAL PORTFOLIO STATE ---")
    print(f"Total Equity:      ${updated_summary['total_equity']:,.2f}")
    print(f"Unrealized P&L:    ${updated_summary['unrealized_pnl']:,.2f} ({updated_summary['unrealized_pnl_pct']:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
