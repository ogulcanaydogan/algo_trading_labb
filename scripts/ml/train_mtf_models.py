#!/usr/bin/env python3
"""
Train Multi-Timeframe Models for a Single Asset

Trains models for 3 horizons (3h, 8h, 24h) to enable multi-timeframe
predictions with conviction-based trading.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the v6 training functions
from scripts.ml.train_binary_v6_improved import (
    load_data,
    build_short_horizon_features,
    create_v6_targets,
    get_v6_feature_cols,
    select_features_v6,
    tune_with_optuna_v6,
    build_model_v6,
    walk_forward_expanding,
    HAS_LGB,
    HAS_XGB,
)

import numpy as np
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# MTF Configuration per asset type
MTF_CONFIGS = {
    # Stocks use 3h, 8h, 24h horizons
    "stock": {
        3: {"min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55},
        8: {"min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55},
        24: {"min_move": 0.020, "vol_mult": 0.35, "conf_thresh": 0.55},
    },
    # Crypto uses shorter horizons due to higher volatility
    "crypto": {
        1: {"min_move": 0.010, "vol_mult": 0.5, "conf_thresh": 0.58},
        3: {"min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.58},
        8: {"min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58},
    },
    # Indices
    "index": {
        8: {"min_move": 0.004, "vol_mult": 0.35, "conf_thresh": 0.55},
        24: {"min_move": 0.006, "vol_mult": 0.35, "conf_thresh": 0.55},
        48: {"min_move": 0.008, "vol_mult": 0.35, "conf_thresh": 0.55},
    },
}


def train_single_horizon(
    symbol: str,
    horizon: int,
    df_raw,
    asset_class: str,
    lookback: int,
    top_n: int,
    n_optuna: int,
    config: Dict,
) -> Dict:
    """Train a single horizon model."""
    
    min_move = config.get("min_move", 0.01)
    vol_mult = config.get("vol_mult", 0.4)
    conf_thresh = config.get("conf_thresh", 0.55)
    
    print(f"\n  Training {horizon}h model...")
    print(f"  Config: min_move={min_move:.2%}, vol_mult={vol_mult}, conf={conf_thresh}")
    
    # Build features
    feat = build_short_horizon_features(df_raw.copy(), horizon, asset_class)
    feat, target_col = create_v6_targets(feat, horizon, min_move, vol_mult)
    feature_cols = get_v6_feature_cols(feat)
    
    # Filter to signal bars
    signal = feat.dropna(subset=[target_col])
    signal = signal.dropna(subset=feature_cols, how="any")
    
    if len(signal) < 500:
        print(f"  [SKIP] Only {len(signal)} samples, need 500+")
        return {"status": "failed", "reason": "few_signals", "horizon": horizon}
    
    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()}")
    
    # Split
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]
    
    # Feature selection
    selected = select_features_v6(X_train_raw, y_train, feature_cols, top_n, w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Optuna tuning
    best_params = tune_with_optuna_v6(X_train_s, y_train, n_optuna, w_train)
    
    # Build ensemble
    voters = [
        ("tuned", build_model_v6(best_params, y_train)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight="balanced", random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=6,
                                           learning_rate=0.05, random_state=42)),
    ]
    ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_s, y_train, sample_weight=w_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_s)
    y_prob = ensemble.predict_proba(X_test_s)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # High confidence accuracy
    hc_mask = np.maximum(y_prob, 1 - y_prob) >= conf_thresh
    hc_acc = accuracy_score(y_test[hc_mask], y_pred[hc_mask]) if hc_mask.sum() >= 10 else acc
    hc_count = int(hc_mask.sum())
    
    # Walk-forward
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(X_sel, y, best_params, n_splits=10,
                                 conf_thresh=conf_thresh, sample_weight=weights)
    
    print(f"  {horizon}h Results: acc={acc:.4f}, auc={auc:.4f}, hc_acc={hc_acc:.4f}, wf={wf['avg_acc']:.4f}")
    
    return {
        "status": "success",
        "horizon": horizon,
        "model": ensemble,
        "scaler": scaler,
        "selected_features": selected,
        "metrics": {
            "accuracy": float(acc),
            "auc": float(auc),
            "hc_accuracy": float(hc_acc),
            "hc_count": hc_count,
            "wf_accuracy": wf["avg_acc"],
            "wf_stable": wf["stable"],
        },
        "config": config,
        "best_params": best_params,
    }


def train_mtf_models(
    symbol: str,
    asset_class: str = "stock",
    horizons: List[int] = None,
    lookback: int = 10000,
    top_n: int = 35,
    n_optuna: int = 50,
    output_dir: str = "data/models_v6_improved",
):
    """
    Train multiple horizon models for multi-timeframe prediction.
    
    Args:
        symbol: Trading symbol (e.g., "TSLA")
        asset_class: Asset type ("stock", "crypto", "index")
        horizons: List of horizons to train (e.g., [3, 8, 24])
        lookback: Data lookback in bars
        top_n: Number of features to select
        n_optuna: Optuna trials
        output_dir: Model output directory
    """
    
    print("\n" + "="*70)
    print(f"  MULTI-TIMEFRAME MODEL TRAINING: {symbol}")
    print(f"  Asset class: {asset_class}")
    print("="*70)
    
    # Get config for asset class
    configs = MTF_CONFIGS.get(asset_class, MTF_CONFIGS["stock"])
    
    if horizons is None:
        horizons = list(configs.keys())
    
    print(f"  Horizons: {horizons}")
    print(f"  Lookback: {lookback} bars")
    
    # Load data once
    df = load_data(symbol, lookback)
    if df.empty or len(df) < 500:
        print(f"  [FAIL] Not enough data: {len(df)} bars")
        return {"symbol": symbol, "status": "failed", "reason": "no_data"}
    
    print(f"  Loaded {len(df)} bars")
    
    # Train each horizon
    results = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sym = symbol.replace("/", "_")
    
    for horizon in horizons:
        config = configs.get(horizon, {"min_move": 0.01, "vol_mult": 0.4, "conf_thresh": 0.55})
        
        result = train_single_horizon(
            symbol=symbol,
            horizon=horizon,
            df_raw=df,
            asset_class=asset_class,
            lookback=lookback,
            top_n=top_n,
            n_optuna=n_optuna,
            config=config,
        )
        
        if result["status"] == "success":
            # Save model with horizon suffix
            model_path = out_dir / f"{sym}_{horizon}h_binary_ensemble_v6.pkl"
            scaler_path = out_dir / f"{sym}_{horizon}h_binary_scaler_v6.pkl"
            features_path = out_dir / f"{sym}_{horizon}h_selected_features_v6.json"
            meta_path = out_dir / f"{sym}_{horizon}h_binary_meta_v6.json"
            
            joblib.dump(result["model"], model_path)
            joblib.dump(result["scaler"], scaler_path)
            
            with open(features_path, "w") as f:
                json.dump(result["selected_features"], f)
            
            meta = {
                "symbol": symbol,
                "horizon": horizon,
                "asset_class": asset_class,
                "version": "v6_mtf",
                "trained_at": datetime.now().isoformat(),
                "config": result["config"],
                "metrics": result["metrics"],
                "best_params": result["best_params"],
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            print(f"  Saved {horizon}h model to {model_path}")
        
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("  MULTI-TIMEFRAME TRAINING SUMMARY")
    print("="*70)
    print(f"  {'Horizon':<10} {'Accuracy':<10} {'AUC':<10} {'HC Acc':<10} {'WF Acc':<10} {'Status'}")
    print(f"  {'-'*60}")
    
    for r in results:
        if r["status"] == "success":
            m = r["metrics"]
            print(f"  {r['horizon']}h{'':<7} {m['accuracy']:<10.4f} {m['auc']:<10.4f} {m['hc_accuracy']:<10.4f} {m['wf_accuracy']:<10.4f} OK")
        else:
            print(f"  {r['horizon']}h{'':<7} {'FAILED':<10} {r.get('reason', 'unknown')}")
    
    # Save combined summary
    summary = {
        "symbol": symbol,
        "asset_class": asset_class,
        "trained_at": datetime.now().isoformat(),
        "horizons": horizons,
        "results": [
            {
                "horizon": r["horizon"],
                "status": r["status"],
                "metrics": r.get("metrics"),
            }
            for r in results
        ],
    }
    
    summary_path = out_dir / f"{sym}_mtf_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Summary saved to {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Timeframe ML Models")
    parser.add_argument("symbol", type=str, help="Symbol to train (e.g., TSLA)")
    parser.add_argument("--asset-class", type=str, default="stock",
                        choices=["stock", "crypto", "index"],
                        help="Asset class")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Horizons to train (e.g., 3 8 24)")
    parser.add_argument("--lookback", type=int, default=10000, help="Data lookback")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--output-dir", type=str, default="data/models_v6_improved",
                        help="Output directory")
    args = parser.parse_args()
    
    train_mtf_models(
        symbol=args.symbol,
        asset_class=args.asset_class,
        horizons=args.horizons,
        lookback=args.lookback,
        n_optuna=args.trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
