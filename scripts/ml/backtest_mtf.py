#!/usr/bin/env python3
"""
Backtest Multi-Timeframe Strategy

Compares:
1. Single timeframe (8h) model
2. Multi-timeframe (3h + 8h + 24h) with conviction filtering

Shows how requiring model agreement improves win rate and reduces false signals.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.preprocessing import StandardScaler

from scripts.ml.train_binary_v6_improved import (
    load_data,
    build_short_horizon_features,
    get_v6_feature_cols,
)


class SingleTimeframeBacktest:
    """Backtest a single timeframe model."""
    
    def __init__(self, symbol: str, horizon: int, model_dir: str, conf_thresh: float = 0.55):
        self.symbol = symbol.replace("/", "_")
        self.horizon = horizon
        self.conf_thresh = conf_thresh
        self.model_dir = Path(model_dir)
        
        # Load model
        self.model, self.scaler, self.features = self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and features."""
        # Try horizon-specific first
        model_path = self.model_dir / f"{self.symbol}_{self.horizon}h_binary_ensemble_v6.pkl"
        scaler_path = self.model_dir / f"{self.symbol}_{self.horizon}h_binary_scaler_v6.pkl"
        features_path = self.model_dir / f"{self.symbol}_{self.horizon}h_selected_features_v6.json"
        
        # Fallback to standard naming (for existing 8h models)
        if not model_path.exists() and self.horizon == 8:
            model_path = self.model_dir / f"{self.symbol}_binary_ensemble_v6.pkl"
            scaler_path = self.model_dir / f"{self.symbol}_binary_scaler_v6.pkl"
            features_path = self.model_dir / f"{self.symbol}_selected_features_v6.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        features = []
        if features_path.exists():
            with open(features_path) as f:
                features = json.load(f)
        
        return model, scaler, features
    
    def run(self, df: pd.DataFrame, min_samples: int = 100) -> Dict:
        """Run backtest on data."""
        # Build features
        feat = build_short_horizon_features(df.copy(), self.horizon, "stock")
        feature_cols = get_v6_feature_cols(feat)
        
        # Forward return as target
        feat["actual_return"] = feat["close"].pct_change(self.horizon).shift(-self.horizon)
        feat = feat.dropna(subset=feature_cols + ["actual_return"], how="any")
        
        if len(feat) < min_samples:
            return {"status": "failed", "reason": "insufficient_data"}
        
        # Get feature matrix
        if self.features:
            missing = [f for f in self.features if f not in feat.columns]
            for f in missing:
                feat[f] = 0
            X = feat[self.features].values
        else:
            X = feat[feature_cols].values
        
        X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
        
        # Predict
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        prob_long = proba[:, 1]
        
        # Trading signals
        confidence = np.maximum(prob_long, 1 - prob_long)
        signal = np.where(prob_long > 0.5, 1, -1)
        high_conf_mask = confidence >= self.conf_thresh
        
        actual_returns = feat["actual_return"].values
        
        # Calculate PnL
        all_trades = signal * actual_returns
        hc_trades = signal[high_conf_mask] * actual_returns[high_conf_mask]
        
        # Statistics
        results = {
            "horizon": self.horizon,
            "total_bars": len(feat),
            "all_trades": {
                "count": len(all_trades),
                "win_rate": float((all_trades > 0).mean()),
                "avg_return": float(all_trades.mean()),
                "total_return": float(all_trades.sum()),
                "sharpe": float(all_trades.mean() / (all_trades.std() + 1e-8) * np.sqrt(252 * 24 / self.horizon)),
            },
            "high_conf_trades": {
                "count": int(high_conf_mask.sum()),
                "pct_of_total": float(high_conf_mask.mean()),
                "win_rate": float((hc_trades > 0).mean()) if len(hc_trades) > 0 else 0,
                "avg_return": float(hc_trades.mean()) if len(hc_trades) > 0 else 0,
                "total_return": float(hc_trades.sum()) if len(hc_trades) > 0 else 0,
                "sharpe": float(hc_trades.mean() / (hc_trades.std() + 1e-8) * np.sqrt(252 * 24 / self.horizon)) if len(hc_trades) > 0 else 0,
            },
        }
        
        return results


class MultiTimeframeBacktest:
    """Backtest multi-timeframe strategy with conviction filtering."""
    
    def __init__(
        self,
        symbol: str,
        horizons: List[int],
        model_dir: str,
        conf_thresh: float = 0.55,
        min_conviction: str = "medium",  # "high" or "medium"
    ):
        self.symbol = symbol.replace("/", "_")
        self.horizons = horizons
        self.conf_thresh = conf_thresh
        self.min_conviction = min_conviction
        self.model_dir = Path(model_dir)
        
        # Load models
        self.models = {}
        self.scalers = {}
        self.features = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models."""
        for horizon in self.horizons:
            model_path = self.model_dir / f"{self.symbol}_{horizon}h_binary_ensemble_v6.pkl"
            scaler_path = self.model_dir / f"{self.symbol}_{horizon}h_binary_scaler_v6.pkl"
            features_path = self.model_dir / f"{self.symbol}_{horizon}h_selected_features_v6.json"
            
            # Fallback for 8h
            if not model_path.exists() and horizon == 8:
                model_path = self.model_dir / f"{self.symbol}_binary_ensemble_v6.pkl"
                scaler_path = self.model_dir / f"{self.symbol}_binary_scaler_v6.pkl"
                features_path = self.model_dir / f"{self.symbol}_selected_features_v6.json"
            
            if model_path.exists():
                self.models[horizon] = joblib.load(model_path)
                self.scalers[horizon] = joblib.load(scaler_path)
                
                if features_path.exists():
                    with open(features_path) as f:
                        self.features[horizon] = json.load(f)
                else:
                    self.features[horizon] = []
                
                print(f"  Loaded {horizon}h model")
            else:
                print(f"  [WARN] {horizon}h model not found")
    
    def run(self, df: pd.DataFrame, min_samples: int = 100) -> Dict:
        """Run multi-timeframe backtest."""
        if not self.models:
            return {"status": "failed", "reason": "no_models"}
        
        # Build features for longest horizon
        max_horizon = max(self.horizons)
        feat = build_short_horizon_features(df.copy(), max_horizon, "stock")
        feature_cols = get_v6_feature_cols(feat)
        
        # Use medium horizon for actual return calculation
        target_horizon = sorted(self.horizons)[len(self.horizons) // 2]
        feat["actual_return"] = feat["close"].pct_change(target_horizon).shift(-target_horizon)
        feat = feat.dropna(subset=feature_cols + ["actual_return"], how="any")
        
        if len(feat) < min_samples:
            return {"status": "failed", "reason": "insufficient_data"}
        
        # Get predictions from each model
        predictions = {}
        for horizon in self.models:
            feats = self.features.get(horizon, feature_cols)
            
            # Handle missing features
            missing = [f for f in feats if f not in feat.columns]
            for f in missing:
                feat[f] = 0
            
            if feats:
                X = feat[feats].values
            else:
                X = feat[feature_cols].values
            
            X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
            X_scaled = self.scalers[horizon].transform(X)
            
            proba = self.models[horizon].predict_proba(X_scaled)
            predictions[horizon] = proba[:, 1]  # prob of long
        
        # Combine predictions
        n_models = len(predictions)
        actual_returns = feat["actual_return"].values
        n_samples = len(actual_returns)
        
        # Calculate agreement and signals
        signals = np.zeros(n_samples)
        convictions = np.zeros(n_samples)
        agreement_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            probs = [predictions[h][i] for h in predictions]
            directions = [1 if p > 0.5 else -1 for p in probs]
            
            n_long = sum(1 for d in directions if d == 1)
            n_short = n_models - n_long
            
            # Determine conviction
            if n_long == n_models:
                conviction = "high"
                signal = 1
                agreement = 1.0
            elif n_short == n_models:
                conviction = "high"
                signal = -1
                agreement = 1.0
            elif n_long >= n_models - 1:
                conviction = "medium"
                signal = 1
                agreement = n_long / n_models
            elif n_short >= n_models - 1:
                conviction = "medium"
                signal = -1
                agreement = n_short / n_models
            else:
                conviction = "no_trade"
                signal = 0
                agreement = max(n_long, n_short) / n_models
            
            signals[i] = signal
            convictions[i] = 2 if conviction == "high" else (1 if conviction == "medium" else 0)
            agreement_scores[i] = agreement
        
        # Filter by minimum conviction
        if self.min_conviction == "high":
            trade_mask = convictions == 2
        else:  # medium
            trade_mask = convictions >= 1
        
        # Calculate results
        all_returns = signals * actual_returns
        filtered_signals = signals[trade_mask]
        filtered_returns = actual_returns[trade_mask]
        mtf_returns = filtered_signals * filtered_returns
        
        # High conviction only
        high_conv_mask = convictions == 2
        high_conv_signals = signals[high_conv_mask]
        high_conv_returns = actual_returns[high_conv_mask]
        high_conv_pnl = high_conv_signals * high_conv_returns
        
        results = {
            "horizons": self.horizons,
            "total_bars": len(feat),
            "target_horizon": target_horizon,
            "all_trades": {
                "count": len(all_returns),
                "win_rate": float((all_returns > 0).mean()),
                "avg_return": float(all_returns.mean()),
                "total_return": float(all_returns.sum()),
            },
            "mtf_filtered": {
                "conviction_threshold": self.min_conviction,
                "count": int(trade_mask.sum()),
                "pct_of_total": float(trade_mask.mean()),
                "win_rate": float((mtf_returns > 0).mean()) if len(mtf_returns) > 0 else 0,
                "avg_return": float(mtf_returns.mean()) if len(mtf_returns) > 0 else 0,
                "total_return": float(mtf_returns.sum()) if len(mtf_returns) > 0 else 0,
                "sharpe": float(mtf_returns.mean() / (mtf_returns.std() + 1e-8) * np.sqrt(252 * 24 / target_horizon)) if len(mtf_returns) > 0 else 0,
            },
            "high_conviction_only": {
                "count": int(high_conv_mask.sum()),
                "pct_of_total": float(high_conv_mask.mean()),
                "win_rate": float((high_conv_pnl > 0).mean()) if len(high_conv_pnl) > 0 else 0,
                "avg_return": float(high_conv_pnl.mean()) if len(high_conv_pnl) > 0 else 0,
                "total_return": float(high_conv_pnl.sum()) if len(high_conv_pnl) > 0 else 0,
                "sharpe": float(high_conv_pnl.mean() / (high_conv_pnl.std() + 1e-8) * np.sqrt(252 * 24 / target_horizon)) if len(high_conv_pnl) > 0 else 0,
            },
            "agreement_stats": {
                "avg_agreement": float(agreement_scores.mean()),
                "high_agreement_pct": float((agreement_scores == 1.0).mean()),
            },
        }
        
        return results


def run_comparison(
    symbol: str,
    horizons: List[int],
    model_dir: str,
    data_lookback: int = 2000,
    conf_thresh: float = 0.55,
) -> Dict:
    """
    Run comparison backtest between single and multi-timeframe strategies.
    """
    print("\n" + "="*70)
    print(f"  MULTI-TIMEFRAME BACKTEST: {symbol}")
    print(f"  Horizons: {horizons}")
    print("="*70)
    
    # Load data
    df = load_data(symbol, data_lookback)
    if df.empty or len(df) < 500:
        print(f"  [FAIL] Not enough data: {len(df)} bars")
        return {"status": "failed"}
    
    print(f"  Loaded {len(df)} bars for backtest")
    
    results = {
        "symbol": symbol,
        "horizons": horizons,
        "data_bars": len(df),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Single timeframe backtests
    print("\n  === SINGLE TIMEFRAME RESULTS ===")
    for horizon in horizons:
        try:
            bt = SingleTimeframeBacktest(symbol, horizon, model_dir, conf_thresh)
            result = bt.run(df)
            results[f"single_{horizon}h"] = result
            
            if "all_trades" in result:
                print(f"\n  {horizon}h Model:")
                print(f"    All trades: {result['all_trades']['count']}, win={result['all_trades']['win_rate']:.1%}, ret={result['all_trades']['total_return']:.2%}")
                print(f"    High conf:  {result['high_conf_trades']['count']}, win={result['high_conf_trades']['win_rate']:.1%}, ret={result['high_conf_trades']['total_return']:.2%}")
        except FileNotFoundError as e:
            print(f"\n  {horizon}h Model: Not found")
            results[f"single_{horizon}h"] = {"status": "not_found"}
    
    # Multi-timeframe backtest
    print("\n  === MULTI-TIMEFRAME RESULTS ===")
    
    # Medium conviction (2/3 agree)
    mtf_med = MultiTimeframeBacktest(symbol, horizons, model_dir, conf_thresh, "medium")
    mtf_med_result = mtf_med.run(df)
    results["mtf_medium"] = mtf_med_result
    
    if "mtf_filtered" in mtf_med_result:
        print(f"\n  MTF (MEDIUM conviction - 2/3 agree):")
        print(f"    Trades:   {mtf_med_result['mtf_filtered']['count']} ({mtf_med_result['mtf_filtered']['pct_of_total']:.1%} of bars)")
        print(f"    Win rate: {mtf_med_result['mtf_filtered']['win_rate']:.1%}")
        print(f"    Return:   {mtf_med_result['mtf_filtered']['total_return']:.2%}")
        print(f"    Sharpe:   {mtf_med_result['mtf_filtered']['sharpe']:.2f}")
    
    # High conviction only (3/3 agree)
    if "high_conviction_only" in mtf_med_result:
        print(f"\n  MTF (HIGH conviction - 3/3 agree):")
        print(f"    Trades:   {mtf_med_result['high_conviction_only']['count']} ({mtf_med_result['high_conviction_only']['pct_of_total']:.1%} of bars)")
        print(f"    Win rate: {mtf_med_result['high_conviction_only']['win_rate']:.1%}")
        print(f"    Return:   {mtf_med_result['high_conviction_only']['total_return']:.2%}")
        print(f"    Sharpe:   {mtf_med_result['high_conviction_only']['sharpe']:.2f}")
    
    # Comparison summary
    print("\n  === COMPARISON SUMMARY ===")
    print(f"  {'Strategy':<25} {'Trades':<10} {'Win Rate':<12} {'Total Ret':<12} {'Sharpe':<10}")
    print(f"  {'-'*70}")
    
    # Single 8h baseline
    if f"single_8h" in results and "all_trades" in results["single_8h"]:
        s = results["single_8h"]["all_trades"]
        print(f"  {'Single 8h (all)':<25} {s['count']:<10} {s['win_rate']:.1%}{'':<6} {s['total_return']:.2%}{'':<6} {s.get('sharpe', 0):.2f}")
    
    if f"single_8h" in results and "high_conf_trades" in results["single_8h"]:
        s = results["single_8h"]["high_conf_trades"]
        print(f"  {'Single 8h (high conf)':<25} {s['count']:<10} {s['win_rate']:.1%}{'':<6} {s['total_return']:.2%}{'':<6} {s.get('sharpe', 0):.2f}")
    
    if "mtf_filtered" in mtf_med_result:
        s = mtf_med_result["mtf_filtered"]
        print(f"  {'MTF (2/3 agree)':<25} {s['count']:<10} {s['win_rate']:.1%}{'':<6} {s['total_return']:.2%}{'':<6} {s.get('sharpe', 0):.2f}")
    
    if "high_conviction_only" in mtf_med_result:
        s = mtf_med_result["high_conviction_only"]
        print(f"  {'MTF (3/3 agree)':<25} {s['count']:<10} {s['win_rate']:.1%}{'':<6} {s['total_return']:.2%}{'':<6} {s.get('sharpe', 0):.2f}")
    
    # Save results
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sym = symbol.replace("/", "_")
    results_path = output_dir / f"{sym}_mtf_backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest Multi-Timeframe Strategy")
    parser.add_argument("symbol", type=str, help="Symbol to backtest (e.g., TSLA)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 8, 24],
                        help="Horizons to compare (default: 3 8 24)")
    parser.add_argument("--model-dir", type=str, default="data/models_v6_improved",
                        help="Model directory")
    parser.add_argument("--lookback", type=int, default=2000,
                        help="Data lookback for backtest")
    parser.add_argument("--conf-thresh", type=float, default=0.55,
                        help="Confidence threshold for single model")
    args = parser.parse_args()
    
    run_comparison(
        symbol=args.symbol,
        horizons=args.horizons,
        model_dir=args.model_dir,
        data_lookback=args.lookback,
        conf_thresh=args.conf_thresh,
    )


if __name__ == "__main__":
    main()
