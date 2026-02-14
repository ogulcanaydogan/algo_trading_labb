#!/usr/bin/env python3
"""
Backtest V6 Improved Models.

Tests model predictions against actual price movements with realistic trading simulation.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Feature Engineering (same as training)
# =============================================================================
def build_short_horizon_features(df: pd.DataFrame, pred_horizon: int = 3, asset_class: str = "crypto") -> pd.DataFrame:
    """Build features aligned to prediction horizon - SAME AS TRAINING."""
    # Inject synthetic volume for forex
    if df["volume"].sum() == 0:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df = df.copy()
        df["volume"] = (tr / (tr.mean() + 1e-12) * 1000).fillna(1000).astype(float)

    feat = df.copy()
    c = feat["close"]
    h = feat["high"]
    l = feat["low"]
    v = feat["volume"]

    # SHORT-HORIZON RETURNS
    feat["ret_1h"] = c.pct_change(1)
    feat["ret_2h"] = c.pct_change(2)
    feat["ret_3h"] = c.pct_change(3)
    feat["ret_6h"] = c.pct_change(6)
    feat["ret_12h"] = c.pct_change(12)
    feat["ret_24h"] = c.pct_change(24)
    
    if pred_horizon >= 12:
        feat["ret_48h"] = c.pct_change(48)
        feat["ret_72h"] = c.pct_change(72)

    # SHORT-TERM MOVING AVERAGES
    feat["ema_5"] = c.ewm(span=5).mean()
    feat["ema_10"] = c.ewm(span=10).mean()
    feat["ema_20"] = c.ewm(span=20).mean()
    feat["sma_10"] = c.rolling(10).mean()
    feat["sma_20"] = c.rolling(20).mean()
    
    if pred_horizon >= 12:
        feat["ema_50"] = c.ewm(span=50).mean()
        feat["price_vs_ema50"] = (c - feat["ema_50"]) / feat["ema_50"]

    feat["price_vs_ema5"] = (c - feat["ema_5"]) / feat["ema_5"]
    feat["price_vs_ema10"] = (c - feat["ema_10"]) / feat["ema_10"]
    feat["price_vs_ema20"] = (c - feat["ema_20"]) / feat["ema_20"]

    feat["ema_5_10_diff"] = (feat["ema_5"] - feat["ema_10"]) / feat["ema_10"]
    feat["ema_10_20_diff"] = (feat["ema_10"] - feat["ema_20"]) / feat["ema_20"]

    # MOMENTUM
    feat["momentum_3"] = c.pct_change(3)
    feat["momentum_6"] = c.pct_change(6)
    feat["momentum_12"] = c.pct_change(12)

    feat["roc_3"] = (c - c.shift(3)) / (c.shift(3) + 1e-8) * 100
    feat["roc_6"] = (c - c.shift(6)) / (c.shift(6) + 1e-8) * 100

    # RSI
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    feat["rsi_7"] = calc_rsi(c, 7)
    feat["rsi_14"] = calc_rsi(c, 14)
    feat["rsi_distance_50"] = feat["rsi_14"] - 50
    
    # MOMENTUM REGIME
    feat["momentum_regime_oversold"] = (feat["rsi_14"] < 30).astype(float)
    feat["momentum_regime_neutral"] = ((feat["rsi_14"] >= 30) & (feat["rsi_14"] <= 70)).astype(float)
    feat["momentum_regime_overbought"] = (feat["rsi_14"] > 70).astype(float)
    feat["momentum_zone"] = np.where(feat["rsi_14"] < 30, -1, 
                                     np.where(feat["rsi_14"] > 70, 1, 0))

    # BOLLINGER BANDS
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feat["bb_upper"] = bb_sma + 2 * bb_std
    feat["bb_lower"] = bb_sma - 2 * bb_std
    feat["bb_position"] = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-8)
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / bb_sma

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_ratio"] = feat["atr_14"] / c

    # ADX
    plus_dm_raw = h.diff()
    minus_dm_raw = -l.diff()
    plus_dm = pd.Series(np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0), index=feat.index)
    minus_dm = pd.Series(np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0), index=feat.index)
    
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    feat["adx"] = dx.rolling(14).mean()
    feat["plus_di"] = plus_di
    feat["minus_di"] = minus_di
    feat["trend_strength"] = feat["adx"]
    feat["trend_direction"] = np.sign(plus_di - minus_di)

    # VOLATILITY
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)
    
    # VOLATILITY REGIME
    vol_20d = feat["ret_1h"].rolling(20 * 24).std()
    feat["volatility_regime"] = feat["vol_24h"] / (vol_20d + 1e-8)
    feat["vol_expanding"] = (feat["volatility_regime"] > 1.2).astype(float)
    feat["vol_contracting"] = (feat["volatility_regime"] < 0.8).astype(float)

    # VOLUME
    feat["vol_sma_10"] = v.rolling(10).mean()
    feat["vol_sma_20"] = v.rolling(20).mean()
    feat["vol_ratio"] = v / (feat["vol_sma_20"] + 1e-8)
    feat["volume_ratio"] = feat["vol_ratio"]  # Alias for compatibility
    feat["vol_change"] = v.pct_change()

    # PRICE RANGE
    feat["range_1h"] = (h - l) / c
    feat["range_6h"] = (h.rolling(6).max() - l.rolling(6).min()) / c
    feat["range_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / c

    feat["high_12h"] = h.rolling(12).max()
    feat["low_12h"] = l.rolling(12).min()
    feat["position_in_range_12h"] = (c - feat["low_12h"]) / (feat["high_12h"] - feat["low_12h"] + 1e-8)
    
    if pred_horizon >= 12:
        feat["range_24h"] = (h.rolling(24).max() - l.rolling(24).min()) / c
        feat["high_24h"] = h.rolling(24).max()
        feat["low_24h"] = l.rolling(24).min()
        feat["position_in_range_24h"] = (c - feat["low_24h"]) / (feat["high_24h"] - feat["low_24h"] + 1e-8)

    # MACD
    ema_fast = c.ewm(span=8).mean()
    ema_slow = c.ewm(span=17).mean()
    feat["macd"] = ema_fast - ema_slow
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    feat["macd_hist_change"] = feat["macd_hist"].diff()

    # STOCHASTIC
    lowest_14 = l.rolling(14).min()
    highest_14 = h.rolling(14).max()
    feat["stoch_k"] = 100 * (c - lowest_14) / (highest_14 - lowest_14 + 1e-8)
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # TIME FEATURES
    if hasattr(feat.index, "hour"):
        feat["hour"] = feat.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
        feat["is_asian"] = ((feat["hour"] >= 0) & (feat["hour"] < 8)).astype(float)
        feat["is_european"] = ((feat["hour"] >= 8) & (feat["hour"] < 16)).astype(float)
        feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)
        
        if pred_horizon >= 12 and hasattr(feat.index, "dayofweek"):
            feat["dow"] = feat.index.dayofweek
            feat["dow_sin"] = np.sin(2 * np.pi * feat["dow"] / 7)
            feat["dow_cos"] = np.cos(2 * np.pi * feat["dow"] / 7)
            feat["is_monday"] = (feat["dow"] == 0).astype(float)
            feat["is_friday"] = (feat["dow"] == 4).astype(float)

    # Z-SCORES
    feat["zscore_10"] = (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-8)
    feat["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)

    return feat


# =============================================================================
# Data Loading
# =============================================================================
def fetch_data_yfinance(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data from yfinance."""
    import yfinance as yf
    
    # Convert symbol format
    if "_USDT" in symbol or "_USD" in symbol:
        yf_symbol = symbol.replace("_USDT", "-USD").replace("_USD", "-USD")
    elif "/" in symbol:
        yf_symbol = symbol.replace("/", "-")
    else:
        yf_symbol = symbol  # Stock symbols stay as is
    
    print(f"  Fetching {days} days for {symbol} (yfinance: {yf_symbol})...")
    
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")
    
    if df.empty:
        print(f"  [WARN] No data from yfinance for {yf_symbol}")
        return None
    
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    print(f"  Got {len(df)} candles")
    return df


def load_data_parquet(symbol: str) -> Optional[pd.DataFrame]:
    """Load data from parquet files."""
    sym = symbol.replace("/", "_")
    for name in [f"{sym}_extended.parquet", f"{sym}_1h.parquet"]:
        path = PROJECT_ROOT / "data" / "training" / name
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                if df["timestamp"].dtype == "int64" and df["timestamp"].iloc[0] > 1e12:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            print(f"  Loaded {len(df)} bars from {name}")
            return df
    return None


# =============================================================================
# Load V6 Models
# =============================================================================
def load_v6_model(symbol: str) -> Optional[Dict]:
    """Load V6 improved model for a symbol."""
    sym = symbol.replace("/", "_")
    model_dir = PROJECT_ROOT / "data" / "models_v6_improved"
    
    model_path = model_dir / f"{sym}_binary_ensemble_v6.pkl"
    scaler_path = model_dir / f"{sym}_binary_scaler_v6.pkl"
    meta_path = model_dir / f"{sym}_binary_meta_v6.json"
    features_path = model_dir / f"{sym}_selected_features_v6.json"
    
    if not all(p.exists() for p in [model_path, scaler_path, meta_path, features_path]):
        print(f"  [WARN] Missing model files for {symbol}")
        return None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            meta = json.load(f)
        with open(features_path) as f:
            features = json.load(f)
        
        print(f"  Loaded V6 model: acc={meta['metrics']['test_accuracy']:.2%}, horizon={meta['config']['horizon']}h")
        return {
            "model": model,
            "scaler": scaler,
            "meta": meta,
            "features": features,
        }
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return None


# =============================================================================
# Backtest Engine
# =============================================================================
def backtest_v6_model(
    df: pd.DataFrame,
    model_data: Dict,
    initial_capital: float = 10000.0,
    position_size: float = 0.2,
    stop_loss: float = 0.02,
    take_profit: float = 0.04,
    conf_threshold: float = 0.55,
) -> Dict:
    """Run backtest with V6 model predictions."""
    
    meta = model_data["meta"]
    horizon = meta["config"]["horizon"]
    asset_class = meta.get("asset_class", "crypto")
    selected_features = model_data["features"]
    
    # Build features
    feat = build_short_horizon_features(df, horizon, asset_class)
    
    # Ensure all selected features exist
    missing = [f for f in selected_features if f not in feat.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing[:5]}...")
        return None
    
    # Filter valid rows
    feat = feat.dropna(subset=selected_features)
    if len(feat) < 100:
        print(f"  [WARN] Only {len(feat)} valid rows after feature computation")
        return None
    
    X = feat[selected_features].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    # Scale and predict
    X_scaled = model_data["scaler"].transform(X)
    predictions = model_data["model"].predict(X_scaled)
    probabilities = model_data["model"].predict_proba(X_scaled)
    
    # Get confidence (max probability)
    confidence = np.maximum(probabilities[:, 0], probabilities[:, 1])
    
    # Add to dataframe
    feat = feat.copy()
    feat["prediction"] = predictions  # 0=DOWN, 1=UP
    feat["confidence"] = confidence
    feat["prob_up"] = probabilities[:, 1]
    
    # Trading simulation
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]
    
    prices = feat["close"].values
    
    for i in range(len(feat) - 1):
        current_price = prices[i]
        pred = feat["prediction"].iloc[i]
        conf = feat["confidence"].iloc[i]
        
        if position is None:
            # Only trade with confidence above threshold
            if conf >= conf_threshold:
                if pred == 1:  # UP -> LONG
                    position = {
                        "side": "LONG",
                        "entry_price": current_price,
                        "size": capital * position_size / current_price,
                        "entry_idx": i,
                    }
                elif pred == 0:  # DOWN -> SHORT
                    position = {
                        "side": "SHORT",
                        "entry_price": current_price,
                        "size": capital * position_size / current_price,
                        "entry_idx": i,
                    }
        else:
            # Check exit conditions
            if position["side"] == "LONG":
                pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
            else:
                pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
            
            should_exit = (
                pnl_pct <= -stop_loss or
                pnl_pct >= take_profit or
                (position["side"] == "LONG" and pred == 0 and conf >= conf_threshold) or
                (position["side"] == "SHORT" and pred == 1 and conf >= conf_threshold)
            )
            
            if should_exit:
                pnl = pnl_pct * position["size"] * position["entry_price"]
                capital += pnl
                trades.append({
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "duration": i - position["entry_idx"],
                })
                position = None
        
        equity_curve.append(capital)
    
    # Close remaining position
    if position:
        current_price = prices[-1]
        if position["side"] == "LONG":
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl_pct = (position["entry_price"] - current_price) / position["entry_price"]
        pnl = pnl_pct * position["size"] * position["entry_price"]
        capital += pnl
        trades.append({
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration": len(feat) - position["entry_idx"],
        })
    
    # Calculate metrics
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-10)
    max_drawdown = np.max(drawdown)
    
    # Sharpe ratio (annualized, assuming hourly data)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24)
    
    # Profit factor
    total_wins = sum(t["pnl"] for t in wins) if wins else 0
    total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 1e-10
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    return {
        "symbol": meta["symbol"],
        "asset_class": asset_class,
        "horizon": horizon,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital / initial_capital - 1) * 100, 2),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "avg_win_pct": round(np.mean([t["pnl_pct"] * 100 for t in wins]), 2) if wins else 0,
        "avg_loss_pct": round(np.mean([t["pnl_pct"] * 100 for t in losses]), 2) if losses else 0,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "model_accuracy": round(meta["metrics"]["test_accuracy"] * 100, 2),
        "hc_accuracy": round(meta["metrics"].get("hc_accuracy", 0) * 100, 2),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest V6 Improved Models")
    parser.add_argument("--symbols", nargs="+", default=["TSLA", "XRP_USDT", "BTC_USDT"])
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.04)
    parser.add_argument("--use-parquet", action="store_true", help="Use parquet data instead of yfinance")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  V6 IMPROVED MODEL BACKTEST")
    print("="*70)
    print(f"  Symbols: {args.symbols}")
    print(f"  Period: {args.days} days")
    print(f"  Initial Capital: ${args.capital:,.0f}")
    print(f"  Stop Loss: {args.stop_loss:.1%} | Take Profit: {args.take_profit:.1%}")
    print("="*70)
    
    results = []
    
    for symbol in args.symbols:
        print(f"\n{'='*60}")
        print(f"  Backtesting {symbol}")
        print("="*60)
        
        # Load model
        model_data = load_v6_model(symbol)
        if not model_data:
            print(f"  [SKIP] No model for {symbol}")
            continue
        
        # Load data
        if args.use_parquet:
            df = load_data_parquet(symbol)
        else:
            df = fetch_data_yfinance(symbol, args.days)
        
        if df is None or len(df) < 500:
            print(f"  [SKIP] Insufficient data for {symbol}")
            continue
        
        # Get confidence threshold from model config
        conf_thresh = model_data["meta"]["config"].get("confidence_threshold", 0.55)
        
        # Run backtest
        result = backtest_v6_model(
            df,
            model_data,
            initial_capital=args.capital,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            conf_threshold=conf_thresh,
        )
        
        if result:
            results.append(result)
            print(f"\n  RESULTS for {symbol}:")
            print(f"    Total Return:  {result['total_return_pct']:+.2f}%")
            print(f"    Sharpe Ratio:  {result['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown:  {result['max_drawdown_pct']:.2f}%")
            print(f"    Win Rate:      {result['win_rate_pct']:.1f}%")
            print(f"    Total Trades:  {result['total_trades']}")
            print(f"    Profit Factor: {result['profit_factor']:.2f}")
    
    # Summary table
    if results:
        print("\n" + "="*90)
        print("  BACKTEST SUMMARY")
        print("="*90)
        print(f"  {'Symbol':<12} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'WinRate':<10} {'Trades':<8} {'PF':<8}")
        print(f"  {'-'*80}")
        
        for r in results:
            print(f"  {r['symbol']:<12} {r['total_return_pct']:>+8.2f}% {r['sharpe_ratio']:>7.2f} {r['max_drawdown_pct']:>7.2f}% {r['win_rate_pct']:>8.1f}% {r['total_trades']:>7} {r['profit_factor']:>7.2f}")
        
        print(f"  {'-'*80}")
        
        # Aggregate stats
        avg_return = np.mean([r['total_return_pct'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_winrate = np.mean([r['win_rate_pct'] for r in results])
        
        print(f"  {'AVERAGE':<12} {avg_return:>+8.2f}% {avg_sharpe:>7.2f}          {avg_winrate:>8.1f}%")
        print("="*90)
        
        # Save results
        out_path = PROJECT_ROOT / "data" / "models_v6_improved" / "backtest_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "run_at": datetime.now().isoformat(),
                "params": {
                    "days": args.days,
                    "capital": args.capital,
                    "stop_loss": args.stop_loss,
                    "take_profit": args.take_profit,
                },
                "results": results,
            }, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
