#!/usr/bin/env python3
"""
XRP Stabilization Training Script

Goal: Reduce WF std from 0.046 to < 0.04

Changes:
1. 100 Optuna trials (from 50)
2. 12 walk-forward folds (from 10)
3. 25 features (from 35)
4. Higher regularization bounds
5. Larger min samples per fold (75)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
# XRP STABILIZATION CONFIG
# =============================================================================
XRP_STABLE_CONFIG = {
    "symbol": "XRP_USDT",
    "horizon": 3,
    "min_move": 0.015,
    "vol_mult": 0.4,
    "conf_thresh": 0.58,
    "n_optuna": 100,        # UP from 50
    "n_wf_folds": 12,       # UP from 10
    "top_n_features": 25,   # DOWN from 35
    "min_samples_fold": 75, # UP from 50
    "asset_class": "crypto",
}


# =============================================================================
# Data Loading
# =============================================================================
def load_data(symbol: str, limit: int) -> pd.DataFrame:
    """Load OHLCV data from parquet files."""
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
# Feature Engineering
# =============================================================================
def build_features(df: pd.DataFrame, pred_horizon: int = 3) -> pd.DataFrame:
    """Build features for XRP."""
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

    # Returns
    feat["ret_1h"] = c.pct_change(1)
    feat["ret_2h"] = c.pct_change(2)
    feat["ret_3h"] = c.pct_change(3)
    feat["ret_6h"] = c.pct_change(6)
    feat["ret_12h"] = c.pct_change(12)
    feat["ret_24h"] = c.pct_change(24)

    # EMAs
    feat["ema_5"] = c.ewm(span=5).mean()
    feat["ema_10"] = c.ewm(span=10).mean()
    feat["ema_20"] = c.ewm(span=20).mean()
    feat["sma_10"] = c.rolling(10).mean()
    feat["sma_20"] = c.rolling(20).mean()

    feat["price_vs_ema5"] = (c - feat["ema_5"]) / feat["ema_5"]
    feat["price_vs_ema10"] = (c - feat["ema_10"]) / feat["ema_10"]
    feat["price_vs_ema20"] = (c - feat["ema_20"]) / feat["ema_20"]
    feat["ema_5_10_diff"] = (feat["ema_5"] - feat["ema_10"]) / feat["ema_10"]
    feat["ema_10_20_diff"] = (feat["ema_10"] - feat["ema_20"]) / feat["ema_20"]

    # Momentum
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
    
    # Momentum regime
    feat["momentum_regime_oversold"] = (feat["rsi_14"] < 30).astype(float)
    feat["momentum_regime_neutral"] = ((feat["rsi_14"] >= 30) & (feat["rsi_14"] <= 70)).astype(float)
    feat["momentum_regime_overbought"] = (feat["rsi_14"] > 70).astype(float)
    feat["momentum_zone"] = np.where(feat["rsi_14"] < 30, -1, np.where(feat["rsi_14"] > 70, 1, 0))

    # Bollinger Bands
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
    feat["trend_direction"] = np.sign(plus_di - minus_di)

    # Volatility
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)
    
    vol_20d = feat["ret_1h"].rolling(20 * 24).std()
    feat["volatility_regime"] = feat["vol_24h"] / (vol_20d + 1e-8)
    feat["vol_expanding"] = (feat["volatility_regime"] > 1.2).astype(float)
    feat["vol_contracting"] = (feat["volatility_regime"] < 0.8).astype(float)

    # Volume
    feat["vol_sma_10"] = v.rolling(10).mean()
    feat["vol_sma_20"] = v.rolling(20).mean()
    feat["vol_ratio"] = v / (feat["vol_sma_20"] + 1e-8)
    feat["vol_change"] = v.pct_change()

    # Price range
    feat["range_1h"] = (h - l) / c
    feat["range_6h"] = (h.rolling(6).max() - l.rolling(6).min()) / c
    feat["range_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / c

    feat["high_12h"] = h.rolling(12).max()
    feat["low_12h"] = l.rolling(12).min()
    feat["position_in_range_12h"] = (c - feat["low_12h"]) / (feat["high_12h"] - feat["low_12h"] + 1e-8)

    # MACD
    ema_fast = c.ewm(span=8).mean()
    ema_slow = c.ewm(span=17).mean()
    feat["macd"] = ema_fast - ema_slow
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    feat["macd_hist_change"] = feat["macd_hist"].diff()

    # Stochastic
    lowest_14 = l.rolling(14).min()
    highest_14 = h.rolling(14).max()
    feat["stoch_k"] = 100 * (c - lowest_14) / (highest_14 - lowest_14 + 1e-8)
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # Time features
    if hasattr(feat.index, "hour"):
        feat["hour"] = feat.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
        feat["is_asian"] = ((feat["hour"] >= 0) & (feat["hour"] < 8)).astype(float)
        feat["is_european"] = ((feat["hour"] >= 8) & (feat["hour"] < 16)).astype(float)
        feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)

    # Z-scores
    feat["zscore_10"] = (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-8)
    feat["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)

    # Targets
    feat["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)

    return feat


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns."""
    exclude = {
        "open", "high", "low", "close", "volume", "datetime",
        "fwd_return", "fwd_1h", "fwd_3h", "target", "sample_weight",
        "high_12h", "low_12h", "bb_upper", "bb_lower",
        "ema_5", "ema_10", "ema_20", "sma_10", "sma_20",
        "vol_sma_10", "vol_sma_20", "hour",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]


# =============================================================================
# Target Creation
# =============================================================================
def create_targets(df: pd.DataFrame, pred_horizon: int, min_move: float, vol_mult: float) -> Tuple[pd.DataFrame, str]:
    """Create binary targets."""
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
    print(f"  Targets: UP={n_up}, DOWN={n_down}, ratio={n_up/(n_down+1e-8):.2f}")

    return df, "target"


# =============================================================================
# Feature Selection
# =============================================================================
def select_features(X, y, names, top_n=25, sample_weight=None):
    """Feature selection using MI + RF importance."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi = mi / (mi.max() + 1e-8)

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y, sample_weight=sample_weight)
    rf_imp = rf.feature_importances_
    rf_imp = rf_imp / (rf_imp.max() + 1e-8)

    combined = 0.4 * mi + 0.6 * rf_imp
    ranked = sorted(zip(names, combined), key=lambda x: x[1], reverse=True)

    selected = [n for n, _ in ranked[:top_n]]

    print(f"\n  Feature selection: {len(names)} -> {top_n}")
    for n, s in ranked[:10]:
        print(f"    {s:.4f}  {n}")

    return selected


# =============================================================================
# Optuna Tuning (Higher Regularization)
# =============================================================================
def tune_with_optuna(X_train, y_train, n_trials=100, sample_weight=None) -> Dict:
    """Optuna tuning with HIGHER regularization ranges for stability."""
    if not HAS_OPTUNA:
        print("  [WARN] Optuna not installed")
        return {"model": "lgb" if HAS_LGB else "xgb" if HAS_XGB else "gb"}

    print(f"\n  Optuna tuning ({n_trials} trials, high regularization)...")

    def objective(trial):
        model_type = trial.suggest_categorical("model", ["lgb", "xgb", "gb"])

        if model_type == "lgb" and HAS_LGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350),
                "max_depth": trial.suggest_int("max_depth", 3, 8),  # Lower max
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),  # Higher
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),  # HIGHER
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),  # HIGHER
                "class_weight": "balanced",
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1)
        elif model_type == "xgb" and HAS_XGB:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_weight = neg_count / (pos_count + 1e-8)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350),
                "max_depth": trial.suggest_int("max_depth", 3, 7),  # Lower max
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),  # HIGHER
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),  # HIGHER
                "scale_pos_weight": scale_weight,
            }
            model = xgb.XGBClassifier(**params, objective="binary:logistic",
                                       eval_metric="logloss", random_state=42, verbosity=0)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 6),  # Lower
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 50),  # Higher
            }
            model = GradientBoostingClassifier(**params, random_state=42)

        tscv = TimeSeriesSplit(n_splits=4)  # More splits for stability
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]
            w_t = sample_weight[train_idx] if sample_weight is not None else None
            model.fit(X_t, y_t, sample_weight=w_t)
            y_prob = model.predict_proba(X_v)[:, 1]
            auc = roc_auc_score(y_v, y_prob) if len(np.unique(y_v)) > 1 else 0.5
            scores.append(auc)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best model: {study.best_params.get('model')}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def build_model(params: Dict, y_train=None):
    """Build model from params."""
    model_type = params.get("model", "lgb")
    clean = {k: v for k, v in params.items() if k != "model"}

    if model_type == "lgb" and HAS_LGB:
        clean["class_weight"] = "balanced"
        return lgb.LGBMClassifier(**clean, random_state=42, verbosity=-1, n_jobs=-1)
    elif model_type == "xgb" and HAS_XGB:
        if y_train is not None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            clean["scale_pos_weight"] = neg_count / (pos_count + 1e-8)
        return xgb.XGBClassifier(**clean, objective="binary:logistic",
                                  eval_metric="logloss", random_state=42, verbosity=0)
    else:
        return GradientBoostingClassifier(**clean, random_state=42)


# =============================================================================
# Walk-Forward Validation (12 folds, higher min samples)
# =============================================================================
def walk_forward_expanding(X, y, params, n_splits=12, conf_thresh=0.58, 
                           sample_weight=None, min_samples=75):
    """Walk-forward with 12 folds and higher min samples."""
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
            "acc": acc, "auc": auc,
            "test_samples": test_samples, "is_valid": is_valid
        })
        print(f"  {i+1:<6} {train_end:<8} {test_samples:<8} {acc:<8.4f} {auc:<8.4f} {valid_marker:<6}")

    # Stats from valid folds only
    valid_results = [r for r in results if r["is_valid"]]
    valid_folds = len(valid_results)
    
    if valid_folds > 0:
        avg_acc = np.mean([r["acc"] for r in valid_results])
        avg_auc = np.mean([r["auc"] for r in valid_results])
        std_acc = np.std([r["acc"] for r in valid_results])
    else:
        avg_acc = np.mean([r["acc"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])

    print(f"  {'-'*50}")
    print(f"  AVG (valid={valid_folds}/{len(results)}): acc={avg_acc:.4f}, auc={avg_auc:.4f}, std={std_acc:.4f}")
    
    is_stable = std_acc < 0.04
    print(f"  STABLE: {'YES ✓' if is_stable else 'NO ✗'} (threshold: 0.04)")

    return {
        "folds": results,
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "std": float(std_acc),
        "stable": is_stable,
        "valid_folds": valid_folds,
        "total_folds": len(results),
    }


# =============================================================================
# Main Training
# =============================================================================
def train_xrp_stable():
    """Train XRP with stabilization settings."""
    cfg = XRP_STABLE_CONFIG
    
    print("\n" + "="*70)
    print("  XRP STABILIZATION TRAINING")
    print("="*70)
    print(f"  Optuna trials: {cfg['n_optuna']} (up from 50)")
    print(f"  Walk-forward folds: {cfg['n_wf_folds']} (up from 10)")
    print(f"  Features: {cfg['top_n_features']} (down from 35)")
    print(f"  Min samples/fold: {cfg['min_samples_fold']} (up from 50)")
    print(f"  Target: std < 0.04")
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

    # Feature selection (25 features)
    selected = select_features(X_train_raw, y_train, feature_cols, cfg["top_n_features"], w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Optuna (100 trials)
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

    # Ensemble
    print(f"\n  Building ensemble...")
    voters = [
        ("tuned", build_model(best_params, y_train)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=6,
                                       class_weight="balanced", random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=120, max_depth=5,
                                           learning_rate=0.04, min_samples_leaf=20, random_state=42)),
    ]
    ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred_e = ensemble.predict(X_test_s)
    y_prob_e = ensemble.predict_proba(X_test_s)[:, 1]

    ens_acc = accuracy_score(y_test, y_pred_e)
    ens_auc = roc_auc_score(y_test, y_prob_e)

    print(f"  Ensemble: acc={ens_acc:.4f}, auc={ens_auc:.4f}")

    # Walk-forward (12 folds)
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

    meta = {
        "symbol": cfg["symbol"],
        "version": "v6_stable",
        "trained_at": datetime.now().isoformat(),
        "config": cfg,
        "metrics": {
            "test_accuracy": float(ens_acc),
            "test_auc": float(ens_auc),
            "total_samples": len(X),
        },
        "walk_forward": wf,
        "best_params": best_params,
        "stabilization": {
            "optuna_trials": cfg["n_optuna"],
            "wf_folds": cfg["n_wf_folds"],
            "features": cfg["top_n_features"],
            "min_samples_fold": cfg["min_samples_fold"],
        },
    }

    with open(out_dir / f"{sym}_binary_meta_v6.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Final report
    print("\n" + "="*70)
    print("  XRP STABILIZATION RESULTS")
    print("="*70)
    print(f"  Walk-Forward Accuracy:  {wf['avg_acc']:.4f}")
    print(f"  Walk-Forward Std:       {wf['std']:.4f}")
    print(f"  Stability Threshold:    0.04")
    print(f"  STABLE:                 {'YES ✓' if wf['stable'] else 'NO ✗'}")
    print(f"  Valid Folds:            {wf['valid_folds']}/{wf['total_folds']}")
    print("="*70)
    
    if wf['stable']:
        print("\n  ✓ XRP model is now STABLE!")
    else:
        gap = wf['std'] - 0.04
        print(f"\n  ✗ Still need to reduce std by {gap:.4f}")
        print("  Try: more trials, fewer features, or stronger regularization")

    return wf


if __name__ == "__main__":
    train_xrp_stable()
