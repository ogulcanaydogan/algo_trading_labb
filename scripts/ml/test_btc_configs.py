#!/usr/bin/env python3
"""
BTC Configuration Testing - Systematic evaluation of different approaches
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import training functions from v6_improved
from scripts.ml.train_binary_v6_improved import (
    load_data, build_short_horizon_features, create_v6_targets,
    get_v6_feature_cols, select_features_v6, tune_with_optuna_v6,
    build_model_v6, walk_forward_expanding
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

# =============================================================================
# Test Configurations for BTC
# =============================================================================
BTC_TEST_CONFIGS = [
    # Original config for baseline
    {"name": "baseline_3h", "horizon": 3, "min_move": 0.010, "vol_mult": 0.4, "n_optuna": 50},
    
    # Longer horizons (like SPX500 success)
    {"name": "horizon_6h", "horizon": 6, "min_move": 0.010, "vol_mult": 0.4, "n_optuna": 50},
    {"name": "horizon_12h", "horizon": 12, "min_move": 0.010, "vol_mult": 0.4, "n_optuna": 50},
    {"name": "horizon_24h", "horizon": 24, "min_move": 0.010, "vol_mult": 0.4, "n_optuna": 50},
    
    # Higher min_move thresholds (filter noise)
    {"name": "minmove_015_3h", "horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "n_optuna": 50},
    {"name": "minmove_020_3h", "horizon": 3, "min_move": 0.020, "vol_mult": 0.4, "n_optuna": 50},
    
    # Combined: longer horizon + higher min_move
    {"name": "combo_12h_015", "horizon": 12, "min_move": 0.015, "vol_mult": 0.35, "n_optuna": 50},
    {"name": "combo_24h_015", "horizon": 24, "min_move": 0.015, "vol_mult": 0.35, "n_optuna": 50},
]


def train_btc_config(config: dict, lookback: int = 10000) -> dict:
    """Train BTC with a specific configuration and return metrics."""
    name = config["name"]
    horizon = config["horizon"]
    min_move = config["min_move"]
    vol_mult = config["vol_mult"]
    n_optuna = config.get("n_optuna", 50)
    
    print(f"\n{'='*70}")
    print(f"  Testing Config: {name}")
    print(f"  horizon={horizon}h, min_move={min_move:.1%}, vol_mult={vol_mult}")
    print(f"{'='*70}")
    
    # Load data
    df = load_data("BTC_USDT", lookback)
    if df.empty or len(df) < 500:
        return {"config": name, "status": "failed", "reason": "no_data"}
    
    # Build features
    feat = build_short_horizon_features(df, horizon, "crypto")
    feat, target_col = create_v6_targets(feat, horizon, min_move, vol_mult)
    feature_cols = get_v6_feature_cols(feat)
    
    # Filter to signal bars
    signal = feat.dropna(subset=[target_col])
    signal = signal.dropna(subset=feature_cols, how="any")
    
    if len(signal) < 500:
        print(f"  [FAIL] Only {len(signal)} samples, need 500+")
        return {"config": name, "status": "failed", "reason": "few_signals", "samples": len(signal)}
    
    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()}")
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]
    
    # Feature selection
    selected = select_features_v6(X_train_raw, y_train, feature_cols, 35, w_train)
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
    
    y_pred_e = ensemble.predict(X_test_s)
    y_prob_e = ensemble.predict_proba(X_test_s)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred_e)
    test_auc = roc_auc_score(y_test, y_prob_e)
    
    # High confidence accuracy
    conf_thresh = 0.58
    hc_mask = np.maximum(y_prob_e, 1 - y_prob_e) >= conf_thresh
    hc_acc = accuracy_score(y_test[hc_mask], y_pred_e[hc_mask]) if hc_mask.sum() >= 10 else test_acc
    hc_count = int(hc_mask.sum())
    
    print(f"\n  Test Results:")
    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    AUC:      {test_auc:.4f}")
    print(f"    HC Acc:   {hc_acc:.4f} (n={hc_count})")
    
    # Walk-forward
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(X_sel, y, best_params, n_splits=10,
                                 conf_thresh=conf_thresh, sample_weight=weights)
    
    return {
        "config": name,
        "status": "success",
        "horizon": horizon,
        "min_move": min_move,
        "vol_mult": vol_mult,
        "total_samples": len(X),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "hc_accuracy": float(hc_acc),
        "hc_count": hc_count,
        "wf_accuracy": float(wf["avg_acc"]),
        "wf_std": float(wf["std"]),
        "wf_stable": wf["stable"],
        "wf_valid_folds": wf["valid_folds"],
    }


def test_ensemble_seeds(horizon: int = 12, min_move: float = 0.015, lookback: int = 10000) -> dict:
    """Train 3 models with different seeds and average."""
    print(f"\n{'='*70}")
    print(f"  Ensemble Seed Test (horizon={horizon}h, min_move={min_move:.1%})")
    print(f"{'='*70}")
    
    # Load data
    df = load_data("BTC_USDT", lookback)
    feat = build_short_horizon_features(df, horizon, "crypto")
    feat, target_col = create_v6_targets(feat, horizon, min_move, 0.4)
    feature_cols = get_v6_feature_cols(feat)
    
    signal = feat.dropna(subset=[target_col])
    signal = signal.dropna(subset=feature_cols, how="any")
    
    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)
    
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]
    
    selected = select_features_v6(X_train_raw, y_train, feature_cols, 35, w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train 3 models with different seeds
    seed_probs = []
    seeds = [42, 123, 456]
    
    for seed in seeds:
        print(f"\n  Training with seed={seed}...")
        
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=-1
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=seed
            )
        
        model.fit(X_train_s, y_train, sample_weight=w_train)
        prob = model.predict_proba(X_test_s)[:, 1]
        seed_probs.append(prob)
        
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        print(f"    Seed {seed} accuracy: {acc:.4f}")
    
    # Average probabilities
    avg_prob = np.mean(seed_probs, axis=0)
    avg_pred = (avg_prob >= 0.5).astype(int)
    
    avg_acc = accuracy_score(y_test, avg_pred)
    avg_auc = roc_auc_score(y_test, avg_prob)
    
    # High confidence
    hc_mask = np.maximum(avg_prob, 1 - avg_prob) >= 0.58
    hc_acc = accuracy_score(y_test[hc_mask], avg_pred[hc_mask]) if hc_mask.sum() >= 10 else avg_acc
    
    print(f"\n  Ensemble Average Results:")
    print(f"    Accuracy: {avg_acc:.4f}")
    print(f"    AUC:      {avg_auc:.4f}")
    print(f"    HC Acc:   {hc_acc:.4f} (n={hc_mask.sum()})")
    
    return {
        "config": f"seed_ensemble_{horizon}h",
        "status": "success",
        "horizon": horizon,
        "min_move": min_move,
        "test_accuracy": float(avg_acc),
        "test_auc": float(avg_auc),
        "hc_accuracy": float(hc_acc),
        "hc_count": int(hc_mask.sum()),
        "seeds": seeds,
    }


def main():
    print("\n" + "="*70)
    print("  BTC MODEL CONFIGURATION TESTING")
    print("="*70)
    
    results = []
    
    # Test all configurations
    for config in BTC_TEST_CONFIGS:
        try:
            result = train_btc_config(config)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {config['name']}: {e}")
            results.append({"config": config["name"], "status": "error", "error": str(e)})
    
    # Test seed ensemble with best performing horizon
    try:
        seed_result = test_ensemble_seeds(horizon=12, min_move=0.015)
        results.append(seed_result)
    except Exception as e:
        print(f"  [ERROR] Seed ensemble: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("  BTC CONFIGURATION TEST RESULTS")
    print("="*70)
    print(f"  {'Config':<20} {'Horizon':<8} {'MinMove':<8} {'Test Acc':<10} {'WF Acc':<10} {'Std':<8} {'Samples':<8}")
    print(f"  {'-'*80}")
    
    best_wf = None
    best_wf_acc = 0
    
    for r in results:
        if r["status"] == "success":
            wf_acc = r.get("wf_accuracy", r.get("test_accuracy", 0))
            stable = "Y" if r.get("wf_stable", False) else "N"
            samples = r.get("total_samples", "?")
            
            print(f"  {r['config']:<20} {r.get('horizon','?')}h{'':<5} {r.get('min_move', 0):.1%}{'':<4} {r['test_accuracy']:<10.4f} {wf_acc:<10.4f} {r.get('wf_std', 0):<8.4f} {samples:<8}")
            
            if wf_acc > best_wf_acc:
                best_wf_acc = wf_acc
                best_wf = r
        else:
            print(f"  {r['config']:<20} FAILED: {r.get('reason', r.get('error', 'unknown'))}")
    
    print(f"\n  BEST CONFIG: {best_wf['config'] if best_wf else 'None'}")
    if best_wf:
        print(f"    Horizon: {best_wf.get('horizon')}h")
        print(f"    Min Move: {best_wf.get('min_move', 0):.1%}")
        print(f"    Test Accuracy: {best_wf['test_accuracy']:.4f}")
        print(f"    WF Accuracy: {best_wf.get('wf_accuracy', best_wf['test_accuracy']):.4f}")
        print(f"    HC Accuracy: {best_wf.get('hc_accuracy', 0):.4f}")
    
    # Save results
    out_dir = PROJECT_ROOT / "data" / "models_v6_improved"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(out_dir / "btc_config_test_results.json", "w") as f:
        json.dump({
            "tested_at": datetime.now().isoformat(),
            "results": results,
            "best_config": best_wf,
        }, f, indent=2)
    
    print(f"\n  Results saved to {out_dir / 'btc_config_test_results.json'}")
    
    return best_wf


if __name__ == "__main__":
    main()
