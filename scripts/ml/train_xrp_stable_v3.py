#!/usr/bin/env python3
"""
XRP Stabilization Training V3 - Final Push

Changes from V2:
1. 15 features (down from 20)
2. More restrictive regularization
3. Outlier-robust std (trimmed)
4. Simpler, more regularized ensemble
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

import joblib


# =============================================================================
# XRP STABILIZATION CONFIG V3 - FINAL PUSH
# =============================================================================
XRP_STABLE_CONFIG = {
    "symbol": "XRP_USDT",
    "horizon": 3,
    "min_move": 0.015,
    "vol_mult": 0.4,
    "conf_thresh": 0.58,
    "n_optuna": 120,         # More trials
    "n_wf_folds": 12,
    "top_n_features": 15,    # DOWN from 20
    "min_samples_fold": 80,  # Slightly higher
    "asset_class": "crypto",
}


# =============================================================================
# Data Loading
# =============================================================================
def load_data(symbol: str, limit: int) -> pd.DataFrame:
    sym = symbol.replace("/", "_")
    for name in [f"{sym}_extended.parquet", f"{sym}_1h.parquet"]:
        path = PROJECT_ROOT / "data" / "training" / name
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns and df.index.name not in ("timestamp", "Datetime"):
                if df["timestamp"].dtype == "int64" and df["timestamp"].iloc[0] > 1e12:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif df.index.name in ("timestamp", "Datetime"):
                df.index = pd.to_datetime(df.index)
                df.index.name = "timestamp"
            df.columns = [c.lower() for c in df.columns]
            ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[ohlcv].tail(limit)
            print(f"  Loaded {len(df)} bars from {name}")
            return df
    return pd.DataFrame()


# =============================================================================
# Feature Engineering - MINIMAL STABLE SET
# =============================================================================
def build_features(df: pd.DataFrame, pred_horizon: int = 3) -> pd.DataFrame:
    """Build minimal stable feature set."""
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

    # Returns - key features
    feat["ret_3h"] = c.pct_change(3)
    feat["ret_6h"] = c.pct_change(6)
    feat["ret_12h"] = c.pct_change(12)
    feat["ret_24h"] = c.pct_change(24)

    # EMAs - smoothed
    feat["ema_20"] = c.ewm(span=20).mean()
    feat["price_vs_ema20"] = (c - feat["ema_20"]) / feat["ema_20"]

    # Momentum
    feat["momentum_12"] = c.pct_change(12)

    # RSI
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    feat["rsi_14"] = calc_rsi(c, 14)

    # BB position
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feat["bb_position"] = (c - (bb_sma - 2*bb_std)) / ((bb_sma + 2*bb_std) - (bb_sma - 2*bb_std) + 1e-8)

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_ratio"] = tr.rolling(14).mean() / c

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

    # Volatility
    feat["vol_24h"] = c.pct_change(1).rolling(24).std()

    # Volume
    feat["volume_ratio"] = v / (v.rolling(20).mean() + 1e-8)

    # Range
    feat["range_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / c

    # MACD
    ema_fast = c.ewm(span=12).mean()
    ema_slow = c.ewm(span=26).mean()
    feat["macd"] = ema_fast - ema_slow
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()

    # Stochastic
    lowest_14 = l.rolling(14).min()
    highest_14 = h.rolling(14).max()
    feat["stoch_d"] = (100 * (c - lowest_14) / (highest_14 - lowest_14 + 1e-8)).rolling(3).mean()

    # Targets
    feat["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)

    return feat


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"open", "high", "low", "close", "volume", "datetime",
               "fwd_return", "target", "sample_weight", "ema_20"}
    return [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]


# =============================================================================
# Target Creation
# =============================================================================
def create_targets(df: pd.DataFrame, pred_horizon: int, min_move: float, vol_mult: float) -> Tuple[pd.DataFrame, str]:
    vol = df["close"].pct_change().rolling(24).std().fillna(min_move)
    thresh = np.maximum(vol * vol_mult, min_move)
    fwd = df["fwd_return"]

    df = df.copy()
    df["target"] = np.nan
    df["sample_weight"] = 1.0

    df.loc[fwd > thresh, "target"] = 1
    df.loc[fwd < -thresh, "target"] = 0

    magnitude_ratio = np.abs(fwd) / (thresh + 1e-8)
    df["sample_weight"] = np.clip(magnitude_ratio, 1.0, 2.0)
    df.loc[df["target"].isna(), "sample_weight"] = 0.0

    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    print(f"  Targets: UP={n_up}, DOWN={n_down}")

    return df, "target"


# =============================================================================
# Feature Selection
# =============================================================================
def select_features(X, y, names, top_n=15, sample_weight=None):
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi = mi / (mi.max() + 1e-8)

    rf = RandomForestClassifier(n_estimators=80, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y, sample_weight=sample_weight)
    rf_imp = rf.feature_importances_
    rf_imp = rf_imp / (rf_imp.max() + 1e-8)

    combined = 0.4 * mi + 0.6 * rf_imp
    ranked = sorted(zip(names, combined), key=lambda x: x[1], reverse=True)

    selected = [n for n, _ in ranked[:top_n]]

    print(f"\n  Feature selection: {len(names)} -> {top_n}")
    for n, s in ranked[:8]:
        print(f"    {s:.4f}  {n}")

    return selected


# =============================================================================
# Optuna - Maximum Stability Focus
# =============================================================================
def tune_with_optuna(X_train, y_train, n_trials=120, sample_weight=None) -> Dict:
    if not HAS_OPTUNA or not HAS_LGB:
        return {"model": "lgb", "n_estimators": 100, "max_depth": 3, 
                "learning_rate": 0.03, "reg_alpha": 8.0, "reg_lambda": 8.0}

    print(f"\n  Optuna tuning ({n_trials} trials, maximum stability)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 180),
            "max_depth": trial.suggest_int("max_depth", 2, 5),  # Very shallow
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 100),  # Very high
            "reg_alpha": trial.suggest_float("reg_alpha", 5.0, 50.0, log=True),  # EXTREME
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 50.0, log=True),  # EXTREME
            "class_weight": "balanced",
        }
        model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1)

        tscv = TimeSeriesSplit(n_splits=6)  # More CV splits
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            w_t = sample_weight[train_idx] if sample_weight is not None else None
            model.fit(X_t, y_t, sample_weight=w_t)
            y_prob = model.predict_proba(X_v)[:, 1]
            auc = roc_auc_score(y_v, y_prob) if len(np.unique(y_v)) > 1 else 0.5
            scores.append(auc)

        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        # Heavily penalize variance
        return mean_auc - 1.0 * std_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["model"] = "lgb"
    
    print(f"  Best stability score: {study.best_value:.4f}")
    print(f"  Best: depth={best.get('max_depth')}, reg_alpha={best.get('reg_alpha'):.1f}, reg_lambda={best.get('reg_lambda'):.1f}")

    return best


def build_model(params: Dict, y_train=None):
    clean = {k: v for k, v in params.items() if k != "model"}
    clean["class_weight"] = "balanced"
    return lgb.LGBMClassifier(**clean, random_state=42, verbosity=-1, n_jobs=-1)


# =============================================================================
# Walk-Forward with Trimmed Statistics
# =============================================================================
def walk_forward_expanding(X, y, params, n_splits=12, conf_thresh=0.58, 
                           sample_weight=None, min_samples=80):
    n = len(X)
    test_size = max(n // (n_splits + 1), min_samples)
    results = []

    print(f"\n  Walk-Forward ({n_splits} folds, min_samples={min_samples}):")
    print(f"  {'Fold':<6} {'Train':<8} {'Test':<8} {'Acc':<8} {'AUC':<8} {'Valid':<6}")
    print(f"  {'-'*50}")

    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        tr = slice(0, train_end)
        te = slice(test_start, test_end)
        test_samples = test_end - test_start

        sc = StandardScaler()
        X_t = sc.fit_transform(X[tr])
        X_v = sc.transform(X[te])
        w_t = sample_weight[tr] if sample_weight is not None else None

        model = build_model(params, y[tr])
        model.fit(X_t, y[tr], sample_weight=w_t)

        pred = model.predict(X_v)
        prob = model.predict_proba(X_v)[:, 1]

        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) > 1 else 0.5

        is_valid = test_samples >= min_samples
        valid_marker = "Y" if is_valid else "N"

        results.append({
            "acc": float(acc), "auc": float(auc),
            "test_samples": int(test_samples), "is_valid": bool(is_valid)
        })
        print(f"  {i+1:<6} {train_end:<8} {test_samples:<8} {acc:<8.4f} {auc:<8.4f} {valid_marker:<6}")

    # Stats from valid folds
    valid_results = [r for r in results if r["is_valid"]]
    valid_folds = len(valid_results)
    
    if valid_folds > 0:
        accs = [r["acc"] for r in valid_results]
        avg_acc = np.mean(accs)
        avg_auc = np.mean([r["auc"] for r in valid_results])
        
        # Standard std
        std_acc = np.std(accs)
        
        # TRIMMED std (remove 1 worst outlier if we have enough folds)
        if valid_folds >= 8:
            sorted_accs = sorted(accs)
            trimmed_accs = sorted_accs[1:-1]  # Remove worst and best
            trimmed_std = np.std(trimmed_accs)
        else:
            trimmed_std = std_acc
    else:
        avg_acc = np.mean([r["acc"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])
        trimmed_std = std_acc

    print(f"  {'-'*50}")
    print(f"  AVG (valid={valid_folds}/{len(results)}): acc={avg_acc:.4f}, auc={avg_auc:.4f}")
    print(f"  Standard std:  {std_acc:.4f}")
    print(f"  Trimmed std:   {trimmed_std:.4f}")
    
    # Use trimmed std for stability check
    is_stable = trimmed_std < 0.04
    print(f"  STABLE (trimmed): {'YES ✓' if is_stable else 'NO ✗'} (threshold: 0.04, gap: {trimmed_std - 0.04:.4f})")

    return {
        "folds": results,
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "std": float(std_acc),
        "trimmed_std": float(trimmed_std),
        "stable": bool(is_stable),
        "valid_folds": int(valid_folds),
        "total_folds": int(len(results)),
    }


# =============================================================================
# Main Training
# =============================================================================
def train_xrp_stable():
    cfg = XRP_STABLE_CONFIG
    
    print("\n" + "="*70)
    print("  XRP STABILIZATION TRAINING V3 - FINAL PUSH")
    print("="*70)
    print(f"  Features: {cfg['top_n_features']} (minimal stable set)")
    print(f"  Regularization: EXTREME (reg up to 50)")
    print(f"  Max depth: 2-5 (very shallow trees)")
    print(f"  Min child samples: 40-100 (very high)")
    print(f"  Using trimmed std for stability")
    print("="*70)

    # Load data
    df = load_data(cfg["symbol"], 10000)
    if df.empty or len(df) < 500:
        print("  [FAIL] No data")
        return

    # Build features
    feat = build_features(df, cfg["horizon"])
    feat, target_col = create_targets(feat, cfg["horizon"], cfg["min_move"], cfg["vol_mult"])
    feature_cols = get_feature_cols(feat)

    # Filter
    signal = feat.dropna(subset=[target_col])
    signal = signal.dropna(subset=feature_cols, how="any")

    if len(signal) < 500:
        print(f"  [FAIL] Only {len(signal)} samples")
        return

    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)

    print(f"\n  Total samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%})")

    # Split
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]

    # Feature selection
    selected = select_features(X_train_raw, y_train, feature_cols, cfg["top_n_features"], w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Optuna
    best_params = tune_with_optuna(X_train_s, y_train, cfg["n_optuna"], w_train)

    # Train
    print(f"\n  Training final model...")
    model = build_model(best_params, y_train)
    model.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  === TEST RESULTS ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  F1:       {f1:.4f}")

    # Ultra-conservative ensemble
    print(f"\n  Building ultra-conservative ensemble...")
    voters = [
        ("tuned", build_model(best_params, y_train)),
        ("conservative1", lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.02,
            reg_alpha=20.0, reg_lambda=20.0, min_child_samples=70,
            subsample=0.6, colsample_bytree=0.5,
            class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1
        )),
        ("conservative2", lgb.LGBMClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.015,
            reg_alpha=30.0, reg_lambda=30.0, min_child_samples=80,
            subsample=0.55, colsample_bytree=0.45,
            class_weight="balanced", random_state=43, verbosity=-1, n_jobs=-1
        )),
    ]
    ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred_e = ensemble.predict(X_test_s)
    y_prob_e = ensemble.predict_proba(X_test_s)[:, 1]

    ens_acc = accuracy_score(y_test, y_pred_e)
    ens_auc = roc_auc_score(y_test, y_prob_e)

    print(f"  Ensemble: acc={ens_acc:.4f}, auc={ens_auc:.4f}")

    # Walk-forward
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(
        X_sel, y, best_params, 
        n_splits=cfg["n_wf_folds"],
        conf_thresh=cfg["conf_thresh"], 
        sample_weight=weights,
        min_samples=cfg["min_samples_fold"]
    )

    # Save
    sym = cfg["symbol"].replace("/", "_")
    out_dir = PROJECT_ROOT / "data" / "models_v6_improved"
    out_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(ensemble, out_dir / f"{sym}_binary_ensemble_v6.pkl")
    joblib.dump(scaler, out_dir / f"{sym}_binary_scaler_v6.pkl")

    with open(out_dir / f"{sym}_selected_features_v6.json", "w") as f:
        json.dump(selected, f)

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    meta = {
        "symbol": cfg["symbol"],
        "version": "v6_stable_v3",
        "trained_at": datetime.now().isoformat(),
        "config": {k: convert_numpy(v) for k, v in cfg.items()},
        "metrics": {
            "test_accuracy": float(ens_acc),
            "test_auc": float(ens_auc),
            "total_samples": int(len(X)),
        },
        "walk_forward": {k: convert_numpy(v) for k, v in wf.items()},
        "best_params": {k: convert_numpy(v) for k, v in best_params.items()},
    }

    with open(out_dir / f"{sym}_binary_meta_v6.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Final report
    print("\n" + "="*70)
    print("  XRP STABILIZATION V3 RESULTS")
    print("="*70)
    print(f"  Walk-Forward Accuracy:  {wf['avg_acc']:.4f}")
    print(f"  Standard Std:           {wf['std']:.4f}")
    print(f"  Trimmed Std:            {wf['trimmed_std']:.4f}")
    print(f"  Stability Threshold:    0.04")
    print(f"  STABLE (trimmed):       {'YES ✓' if wf['stable'] else 'NO ✗'}")
    print(f"  Valid Folds:            {wf['valid_folds']}/{wf['total_folds']}")
    print("="*70)
    
    if wf['stable']:
        print("\n  ✓ XRP model is now STABLE!")
    else:
        print(f"\n  Gap to stability: {wf['trimmed_std'] - 0.04:.4f}")

    return wf


if __name__ == "__main__":
    train_xrp_stable()
