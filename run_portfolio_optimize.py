from __future__ import annotations
"""
Portfolio-wide optimizer: iterates assets from data/portfolio.json and produces
per-asset strategy files under data/portfolio/<asset>/strategy_config.json.

Uses:
- Crypto: ExchangeClient (testnet if configured) or PaperExchangeClient
- Equities/Commodities/Forex: YFinanceMarketDataClient (falls back to Paper)

This does not place orders; it only searches for parameters and writes configs.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union

from dotenv import load_dotenv

from bot.portfolio import PortfolioConfig
from bot.market_data import sanitize_symbol_for_fs
from bot.exchange import ExchangeClient, PaperExchangeClient
from bot.market_data import YFinanceMarketDataClient, MarketDataError
from bot.strategy import StrategyConfig
from bot.optimizer import random_search_optimize


def _fetch_ohlcv(asset_type: str, symbol: str, timeframe: str, lookback: int):
    # Prefer yfinance for non-crypto
    if asset_type.lower() in {"equity", "commodity", "forex", "index", "etf"}:
        try:
            yf = YFinanceMarketDataClient()
            return yf.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
        except MarketDataError:
            pass
    # Crypto or fallback
    use_testnet = os.getenv("BINANCE_TESTNET_ENABLED", "false").lower() == "true"
    if use_testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if api_key and api_secret:
            try:
                ex = ExchangeClient("binance", api_key=api_key, api_secret=api_secret, testnet=True)
                return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
            except RuntimeError:
                pass
    # Paper fallback
    ex = PaperExchangeClient(symbol=symbol, timeframe=timeframe)
    return ex.fetch_ohlcv(limit=lookback)


essentials = ("ema_fast","ema_slow","rsi_period","rsi_overbought","rsi_oversold","risk_per_trade_pct","stop_loss_pct","take_profit_pct")


def _save_params(base: StrategyConfig, params: Dict[str, Union[int, float, str]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "strategy_config.json"
    payload: Dict[str, Union[int, float, str]] = {
        "symbol": base.symbol,
        "timeframe": base.timeframe,
    }
    for key in essentials:
        if key in params:
            payload[key] = params[key]  # type: ignore[assignment]
        else:
            payload[key] = getattr(base, key)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file


def main() -> None:
    load_dotenv()
    config_path = Path(os.getenv("PORTFOLIO_CONFIG", "data/portfolio.json")).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Portfolio configuration not found: {config_path}")

    cfg = PortfolioConfig.load(config_path)
    results = []
    for asset in cfg.assets:
        symbol = asset.symbol
        timeframe = asset.timeframe or cfg.default_timeframe
        lookback = asset.lookback or cfg.default_lookback
        print(f"\n🔎 Optimizing {symbol} ({asset.asset_type}) tf={timeframe} lookback={lookback}")
        ohlcv = _fetch_ohlcv(asset.asset_type, symbol, timeframe, lookback)
        base = StrategyConfig(symbol=symbol, timeframe=timeframe)
        best, _ = random_search_optimize(
            ohlcv=ohlcv,
            base_config=base,
            n_trials=int(os.getenv("AUTO_OPTIMIZE_TRIALS", "60")),
            seed=42,
            initial_balance=float(os.getenv("STARTING_BALANCE", "10000")),
            objective=os.getenv("AUTO_OPTIMIZE_OBJECTIVE", "sharpe"),
            mdd_weight=float(os.getenv("AUTO_OPTIMIZE_MDD_WEIGHT", "0.5")),
            min_trades=int(os.getenv("AUTO_OPTIMIZE_MIN_TRADES", "5")),
        )
        asset_dir = cfg.data_dir / sanitize_symbol_for_fs(symbol)
        out = _save_params(base, best.params, asset_dir)
        print(f"✅ Wrote {out} | Sharpe={best.sharpe_ratio:.3f} PnL%={best.total_pnl_pct:.2f} Win%={best.win_rate*100:.1f} MDD%={best.max_drawdown_pct:.2f}")
        results.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "sharpe": best.sharpe_ratio,
            "pnl_pct": best.total_pnl_pct,
            "win_rate": best.win_rate,
            "mdd_pct": best.max_drawdown_pct,
        })

    # naive allocation suggestion proportional to Sharpe over positive Sharpe assets
    positive = [r for r in results if r["sharpe"] > 0]
    total_sharpe = sum(r["sharpe"] for r in positive) or 1.0
    for r in results:
        r["suggested_allocation_pct"] = (r["sharpe"] / total_sharpe * 100.0) if r["sharpe"] > 0 else 0.0

    rec_path = cfg.data_dir / "portfolio_recommendations.json"
    rec_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\n📊 Saved recommendations -> {rec_path}")


if __name__ == "__main__":
    main()
