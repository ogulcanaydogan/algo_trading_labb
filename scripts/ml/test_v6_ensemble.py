#!/usr/bin/env python3
"""
Test V6 Ensemble Predictor.

Compares individual model accuracy vs ensemble accuracy on test data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(symbol: str, data_dir: Path = Path("data")) -> pd.DataFrame | None:
    """
    Load test data for a symbol.
    Tries multiple sources.
    """
    # Try cached data
    cache_paths = [
        data_dir / f"{symbol}_hourly.csv",
        data_dir / f"{symbol.replace('_', '/')}_1h.csv",
        data_dir / "cache" / f"{symbol}_1h.pkl",
    ]
    
    for path in cache_paths:
        if path.exists():
            logger.info(f"Loading data from {path}")
            if path.suffix == ".csv":
                return pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(path, nrows=1).columns else None)
            else:
                return pd.read_pickle(path)
    
    logger.warning(f"No cached data found for {symbol}")
    return None


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features needed by V6 models."""
    df = df.copy()
    
    # Ensure proper columns
    if "close" not in df.columns:
        if "Close" in df.columns:
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    
    # Price-based features
    df["ret_2h"] = df["close"].pct_change(2)
    df["ret_6h"] = df["close"].pct_change(6)
    df["ret_12h"] = df["close"].pct_change(12)
    df["ret_24h"] = df["close"].pct_change(24)
    
    # Volatility
    df["vol_6h"] = df["close"].pct_change().rolling(6).std()
    df["vol_12h"] = df["close"].pct_change().rolling(12).std()
    df["vol_24h"] = df["close"].pct_change().rolling(24).std()
    df["vol_ratio_6_24"] = df["vol_6h"] / (df["vol_24h"] + 1e-10)
    
    # Range
    df["range_6h"] = df["high"].rolling(6).max() - df["low"].rolling(6).min()
    df["range_12h"] = df["high"].rolling(12).max() - df["low"].rolling(12).min()
    
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # RSI 7
    gain7 = (delta.where(delta > 0, 0)).rolling(7).mean()
    loss7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs7 = gain7 / (loss7 + 1e-10)
    df["rsi_7"] = 100 - (100 / (1 + rs7))
    df["rsi_distance_50"] = df["rsi_14"] - 50
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    
    # EMAs
    df["ema_5"] = df["close"].ewm(span=5).mean()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_5_10_diff"] = (df["ema_5"] - df["ema_10"]) / df["ema_10"]
    df["ema_10_20_diff"] = (df["ema_10"] - df["ema_20"]) / df["ema_20"]
    df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    
    # ADX
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    df["plus_dm"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df["minus_dm"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_14"] = atr
    df["atr_ratio"] = atr / df["close"]
    df["plus_di"] = 100 * (pd.Series(df["plus_dm"]).rolling(14).mean() / (atr + 1e-10))
    df["minus_di"] = 100 * (pd.Series(df["minus_dm"]).rolling(14).mean() / (atr + 1e-10))
    dx = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"] + 1e-10)
    df["adx"] = dx.rolling(14).mean()
    
    # Trend strength
    df["trend_strength"] = abs(df["ema_5"] - df["ema_20"]) / df["ema_20"]
    
    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # Z-score
    df["zscore_10"] = (df["close"] - df["close"].rolling(10).mean()) / (df["close"].rolling(10).std() + 1e-10)
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / (df["close"].rolling(20).std() + 1e-10)
    
    # Momentum
    df["momentum_6"] = df["close"] - df["close"].shift(6)
    df["momentum_12"] = df["close"] - df["close"].shift(12)
    
    # ROC
    df["roc_6"] = (df["close"] - df["close"].shift(6)) / (df["close"].shift(6) + 1e-10) * 100
    
    # Volatility regime (simple version)
    vol_percentile = df["vol_24h"].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df["volatility_regime"] = vol_percentile
    
    return df


def test_ensemble():
    """Test the V6 ensemble predictor."""
    from bot.ml.v6_ensemble import create_v6_ensemble, V6EnsemblePredictor
    
    print("=" * 70)
    print("V6 ENSEMBLE PREDICTOR TEST")
    print("=" * 70)
    
    # Create ensemble
    ensemble = create_v6_ensemble()
    if not ensemble:
        print("ERROR: Failed to create ensemble")
        return
    
    # Print model stats
    print("\nLoaded Models:")
    print("-" * 50)
    stats = ensemble.get_model_stats()
    for symbol, s in stats.items():
        print(f"  {symbol}:")
        print(f"    Weight (normalized WF acc): {s['weight']:.4f}")
        print(f"    Walk-Forward Accuracy:      {s['wf_accuracy']:.4f}")
        print(f"    Test Accuracy:              {s['test_accuracy']:.4f}")
        print(f"    High-Confidence Accuracy:   {s['hc_accuracy']:.4f}")
        print(f"    Features:                   {s['features']}")
    
    # Load some test data
    # Try to create synthetic test data since we may not have actual test files
    print("\n" + "-" * 50)
    print("Creating synthetic test data...")
    
    # Generate synthetic OHLCV data for testing
    np.random.seed(42)
    n_samples = 200
    
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")
    close = 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + np.random.randn(n_samples) * 0.001),
        "high": close * (1 + abs(np.random.randn(n_samples) * 0.005)),
        "low": close * (1 - abs(np.random.randn(n_samples) * 0.005)),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_samples),
    })
    
    # Compute features
    df = compute_features(df)
    df = df.dropna()
    
    print(f"Test data shape: {df.shape}")
    print(f"Features computed: {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])}")
    
    # Test predictions
    print("\n" + "-" * 50)
    print("Testing Ensemble Predictions")
    print("-" * 50)
    
    # Test different strategies
    strategies = ["weighted_avg", "majority_vote", "confidence_weighted"]
    
    for strategy in strategies:
        try:
            prediction = ensemble.predict(df, strategy=strategy)
            print(f"\n{strategy.upper()}:")
            print(f"  Signal:              {prediction.signal}")
            print(f"  Combined Probability: {prediction.combined_probability:.4f}")
            print(f"  Ensemble Confidence:  {prediction.ensemble_confidence:.4f}")
            print(f"  Agreement Score:      {prediction.agreement_score:.4f}")
            print(f"  Individual predictions:")
            for sym, pred in prediction.individual_predictions.items():
                print(f"    {sym}: {pred['signal']} (prob_up={pred['prob_up']:.3f}, conf={pred['confidence']:.3f})")
        except Exception as e:
            print(f"\n{strategy.upper()}: ERROR - {e}")
    
    # Simulate backtest to compare strategies
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON (Simulated)")
    print("=" * 70)
    
    results = {strategy: {"correct": 0, "total": 0, "signals": []} for strategy in strategies}
    
    # Generate actual future returns for comparison
    df["future_return"] = df["close"].shift(-3).pct_change(3)  # 3-period ahead return
    df["actual_direction"] = np.where(df["future_return"] > 0, "LONG", np.where(df["future_return"] < 0, "SHORT", "NEUTRAL"))
    
    # Test on last 50 samples (walk-forward style)
    test_start = len(df) - 50
    
    for i in range(test_start, len(df) - 3):  # -3 for future return calculation
        test_df = df.iloc[:i+1].copy()
        actual = df.iloc[i]["actual_direction"]
        
        for strategy in strategies:
            try:
                pred = ensemble.predict(test_df, strategy=strategy)
                results[strategy]["signals"].append(pred.signal)
                results[strategy]["total"] += 1
                if pred.signal == actual:
                    results[strategy]["correct"] += 1
            except:
                pass
    
    print("\nAccuracy Comparison:")
    print("-" * 50)
    for strategy in strategies:
        r = results[strategy]
        if r["total"] > 0:
            acc = r["correct"] / r["total"]
            print(f"{strategy:25s}: {acc:.4f} ({r['correct']}/{r['total']})")
    
    # Individual model weights
    print("\n" + "-" * 50)
    print("Model Weight Distribution:")
    print("-" * 50)
    total_wf = sum(s["wf_accuracy"] for s in stats.values())
    for symbol, s in sorted(stats.items(), key=lambda x: -x[1]["weight"]):
        bar = "█" * int(s["weight"] * 50)
        print(f"{symbol:12s}: {s['weight']:.3f} {bar}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_ensemble()
