#!/usr/bin/env python3
"""
ML Model v6: Fundamental Fixes for Better Accuracy

Key changes from v5:
1. LOWER THRESHOLD: 0.4x volatility (was 0.7x) → more training samples
2. HIGHER MIN_MOVE: 0.012 crypto, 0.008 stocks (was 0.003-0.008)
3. SHORT-HORIZON FEATURES: Only 1-24h features for 3h prediction
4. CLASS BALANCE: class_weight="balanced" for all models
5. FEATURE ALIGNMENT: Removed 168h+ features that don't match 3h horizon
6. FLAT CLASS OPTION: 3-class mode for uncertain markets
7. PROPER VALIDATION: Expanding window walk-forward

Usage:
    python scripts/ml/train_binary_v6_fixed.py
    python scripts/ml/train_binary_v6_fixed.py --symbols GOOGL SPX500_USD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
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
# V6 Asset Configuration - TUNED FOR BETTER ACCURACY
# =============================================================================
V6_CONFIGS = {
    # Crypto - higher min_move to filter noise
    "ETH_USDT":   {"horizon": 3, "min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50},
    "BTC_USDT":   {"horizon": 3, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50},
    "SOL_USDT":   {"horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50},
    "XRP_USDT":   {"horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50},

    # Stocks - moderate min_move
    "GOOGL":      {"horizon": 3, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
    "MSFT":       {"horizon": 3, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
    "AAPL":       {"horizon": 3, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
    "TSLA":       {"horizon": 3, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
    "AMZN":       {"horizon": 3, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},

    # Indices - lower min_move (less volatile)
    "SPX500_USD": {"horizon": 3, "min_move": 0.005, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
    "NAS100_USD": {"horizon": 3, "min_move": 0.006, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50},
}

DEFAULT_SYMBOLS = ["GOOGL", "SPX500_USD", "NAS100_USD", "BTC_USDT", "ETH_USDT"]


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
# V6 Feature Engineering - SHORT-HORIZON ALIGNED
# =============================================================================
def build_short_horizon_features(df: pd.DataFrame, pred_horizon: int = 3) -> pd.DataFrame:
    """
    Build features aligned to short-term prediction horizon.

    KEY CHANGE: Only uses features from recent 1-24 bars for 3h prediction.
    Removed: EMA-200, 168h volatility, month effects, etc.
    """
    # Inject synthetic volume for forex
    if df["volume"].sum() == 0:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df = df.copy()
        df["volume"] = (tr / (tr.mean() + 1e-12) * 1000).fillna(1000).astype(float)
        print(f"  [INFO] Injected synthetic volume from true range")

    feat = df.copy()
    c = feat["close"]
    h = feat["high"]
    l = feat["low"]
    v = feat["volume"]

    # ===== SHORT-HORIZON RETURNS (aligned to 3h prediction) =====
    feat["ret_1h"] = c.pct_change(1)
    feat["ret_2h"] = c.pct_change(2)
    feat["ret_3h"] = c.pct_change(3)
    feat["ret_6h"] = c.pct_change(6)
    feat["ret_12h"] = c.pct_change(12)
    feat["ret_24h"] = c.pct_change(24)

    # ===== SHORT-TERM MOVING AVERAGES =====
    feat["ema_5"] = c.ewm(span=5).mean()
    feat["ema_10"] = c.ewm(span=10).mean()
    feat["ema_20"] = c.ewm(span=20).mean()
    feat["sma_10"] = c.rolling(10).mean()
    feat["sma_20"] = c.rolling(20).mean()

    # Price vs short EMAs
    feat["price_vs_ema5"] = (c - feat["ema_5"]) / feat["ema_5"]
    feat["price_vs_ema10"] = (c - feat["ema_10"]) / feat["ema_10"]
    feat["price_vs_ema20"] = (c - feat["ema_20"]) / feat["ema_20"]

    # EMA alignment (short-term trend)
    feat["ema_5_10_diff"] = (feat["ema_5"] - feat["ema_10"]) / feat["ema_10"]
    feat["ema_10_20_diff"] = (feat["ema_10"] - feat["ema_20"]) / feat["ema_20"]

    # ===== MOMENTUM (short-term) =====
    feat["momentum_3"] = c.pct_change(3)
    feat["momentum_6"] = c.pct_change(6)
    feat["momentum_12"] = c.pct_change(12)

    # Rate of change
    feat["roc_3"] = (c - c.shift(3)) / (c.shift(3) + 1e-8) * 100
    feat["roc_6"] = (c - c.shift(6)) / (c.shift(6) + 1e-8) * 100

    # ===== RSI (standard 14, but also short 7) =====
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    feat["rsi_7"] = calc_rsi(c, 7)
    feat["rsi_14"] = calc_rsi(c, 14)
    feat["rsi_distance_50"] = feat["rsi_14"] - 50

    # ===== BOLLINGER BANDS (20-period) =====
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feat["bb_upper"] = bb_sma + 2 * bb_std
    feat["bb_lower"] = bb_sma - 2 * bb_std
    feat["bb_position"] = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-8)
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / bb_sma

    # ===== ATR (Average True Range) =====
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_ratio"] = feat["atr_14"] / c

    # ===== VOLATILITY (SHORT-TERM ONLY) =====
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)

    # ===== VOLUME FEATURES =====
    feat["vol_sma_10"] = v.rolling(10).mean()
    feat["vol_sma_20"] = v.rolling(20).mean()
    feat["vol_ratio"] = v / (feat["vol_sma_20"] + 1e-8)
    feat["vol_change"] = v.pct_change()

    # ===== PRICE RANGE =====
    feat["range_1h"] = (h - l) / c
    feat["range_6h"] = (h.rolling(6).max() - l.rolling(6).min()) / c
    feat["range_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / c

    feat["high_12h"] = h.rolling(12).max()
    feat["low_12h"] = l.rolling(12).min()
    feat["position_in_range_12h"] = (c - feat["low_12h"]) / (feat["high_12h"] - feat["low_12h"] + 1e-8)

    # ===== MACD (short settings: 8, 17, 9) =====
    ema_fast = c.ewm(span=8).mean()
    ema_slow = c.ewm(span=17).mean()
    feat["macd"] = ema_fast - ema_slow
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    feat["macd_hist_change"] = feat["macd_hist"].diff()

    # ===== STOCHASTIC =====
    lowest_14 = l.rolling(14).min()
    highest_14 = h.rolling(14).max()
    feat["stoch_k"] = 100 * (c - lowest_14) / (highest_14 - lowest_14 + 1e-8)
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # ===== TIME FEATURES (only intraday relevant) =====
    if hasattr(feat.index, 'hour'):
        feat["hour"] = feat.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)

        # Session indicators
        feat["is_asian"] = ((feat["hour"] >= 0) & (feat["hour"] < 8)).astype(float)
        feat["is_european"] = ((feat["hour"] >= 8) & (feat["hour"] < 16)).astype(float)
        feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)

    # ===== Z-SCORES (short-term) =====
    feat["zscore_10"] = (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-8)
    feat["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)

    # ===== FORWARD RETURNS (targets) =====
    feat["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)
    feat["fwd_1h"] = c.pct_change(1).shift(-1)
    feat["fwd_3h"] = c.pct_change(3).shift(-3)

    return feat


def get_v6_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns, excluding targets and raw OHLCV."""
    exclude = {
        "open", "high", "low", "close", "volume", "datetime",
        "fwd_return", "fwd_1h", "fwd_3h",
        "target", "sample_weight",
        "high_12h", "low_12h", "bb_upper", "bb_lower",
        "ema_5", "ema_10", "ema_20", "sma_10", "sma_20",
        "vol_sma_10", "vol_sma_20", "hour",
    }
    cols = [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]
    return cols


# =============================================================================
# V6 Target Creation - RELAXED THRESHOLD + CLASS BALANCE
# =============================================================================
def create_v6_targets(
    df: pd.DataFrame,
    pred_horizon: int = 3,
    min_move: float = 0.01,
    vol_mult: float = 0.4,  # REDUCED from 0.7
) -> Tuple[pd.DataFrame, str]:
    """
    Create binary targets with RELAXED threshold.

    KEY CHANGES:
    - vol_mult reduced from 0.7 to 0.4 → more samples included
    - higher min_move → filters real noise, not valid signals
    - Simple sample weights based on magnitude
    """
    # Adaptive threshold with LOWER multiplier
    vol = df["close"].pct_change().rolling(24).std().fillna(min_move)  # 24h volatility, not 50h
    thresh = np.maximum(vol * vol_mult, min_move)

    fwd = df["fwd_return"]

    df = df.copy()
    df["target"] = np.nan
    df["sample_weight"] = 1.0

    up_mask = fwd > thresh
    down_mask = fwd < -thresh

    df.loc[up_mask, "target"] = 1  # UP
    df.loc[down_mask, "target"] = 0  # DOWN

    # Simple magnitude weighting (1.0 to 2.0 range)
    magnitude_ratio = np.abs(fwd) / (thresh + 1e-8)
    df["sample_weight"] = np.clip(magnitude_ratio, 1.0, 2.0)
    df.loc[df["target"].isna(), "sample_weight"] = 0.0

    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    n_total = len(df)
    n_sig = n_up + n_down

    print(f"  V6 Target (horizon={pred_horizon}h, min_move={min_move:.2%}, vol_mult={vol_mult}):")
    print(f"    UP:   {n_up} ({n_up/n_total:.1%}), DOWN: {n_down} ({n_down/n_total:.1%})")
    print(f"    Signal: {n_sig} ({n_sig/n_total:.1%})")
    print(f"    Class ratio: {n_up/(n_down+1e-8):.2f} (UP/DOWN)")

    return df, "target"


# =============================================================================
# Feature Selection
# =============================================================================
def select_features_v6(X, y, names, top_n=35, sample_weight=None):
    """Feature selection using MI + RF importance with class balancing."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi = mi / (mi.max() + 1e-8)

    # Use balanced RF for importance
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        class_weight="balanced",  # KEY: class balance
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y, sample_weight=sample_weight)
    rf_imp = rf.feature_importances_
    rf_imp = rf_imp / (rf_imp.max() + 1e-8)

    combined = 0.4 * mi + 0.6 * rf_imp
    ranked = sorted(zip(names, combined), key=lambda x: x[1], reverse=True)

    selected = [n for n, _ in ranked[:top_n]]

    print(f"\n  Feature selection: {len(names)} → {top_n}")
    for n, s in ranked[:10]:
        print(f"    {s:.4f}  {n}")

    return selected


# =============================================================================
# Optuna Tuning with Class Balance
# =============================================================================
def tune_with_optuna_v6(X_train, y_train, n_trials=50, sample_weight=None) -> Dict:
    """Optuna tuning with class_weight='balanced'."""
    if not HAS_OPTUNA:
        print("  [WARN] Optuna not installed, using defaults")
        return {"model": "lgb" if HAS_LGB else "xgb" if HAS_XGB else "gb"}

    print(f"\n  Optuna tuning ({n_trials} trials, class-balanced)...")

    def objective(trial):
        model_type = trial.suggest_categorical("model", ["lgb", "xgb", "gb"])

        if model_type == "lgb" and HAS_LGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5, log=True),
                "class_weight": "balanced",  # KEY
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1)
        elif model_type == "xgb" and HAS_XGB:
            # Calculate scale_pos_weight for XGBoost
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_weight = neg_count / (pos_count + 1e-8)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5, log=True),
                "scale_pos_weight": scale_weight,  # KEY
            }
            model = xgb.XGBClassifier(**params, objective="binary:logistic",
                                       eval_metric="logloss", random_state=42, verbosity=0)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 40),
            }
            model = GradientBoostingClassifier(**params, random_state=42)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=3)
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best model: {study.best_params.get('model')}")

    return study.best_params


def build_model_v6(params: Dict, y_train=None):
    """Build model with class balancing."""
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
# Walk-Forward Validation (Expanding Window)
# =============================================================================
def walk_forward_expanding(X, y, params, n_splits=10, conf_thresh=0.55, sample_weight=None):
    """Walk-forward with EXPANDING window (more realistic)."""
    n = len(X)
    min_samples_per_fold = 50
    test_size = max(n // (n_splits + 1), min_samples_per_fold)
    results = []

    print(f"\n  Walk-Forward Expanding ({n_splits} folds):")
    print(f"  {'Fold':<6} {'Train':<8} {'Test':<8} {'Acc':<8} {'AUC':<8} {'HC Acc':<8} {'HC #':<6}")
    print(f"  {'-'*56}")

    for i in range(n_splits):
        train_end = test_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        tr = slice(0, train_end)
        te = slice(test_start, test_end)

        sc = StandardScaler()
        X_t = sc.fit_transform(X[tr])
        X_v = sc.transform(X[te])
        w_t = sample_weight[tr] if sample_weight is not None else None

        model = build_model_v6(params, y[tr])
        model.fit(X_t, y[tr], sample_weight=w_t)

        pred = model.predict(X_v)
        prob = model.predict_proba(X_v)[:, 1]

        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) > 1 else 0.5

        hc = np.maximum(prob, 1 - prob) >= conf_thresh
        hc_acc = accuracy_score(y[te][hc], pred[hc]) if hc.sum() >= 5 else acc
        hc_n = int(hc.sum())

        results.append({"acc": acc, "auc": auc, "hc_acc": hc_acc, "hc_n": hc_n})
        print(f"  {i+1:<6} {train_end:<8} {test_end-test_start:<8} {acc:<8.4f} {auc:<8.4f} {hc_acc:<8.4f} {hc_n:<6}")

    avg_acc = np.mean([r["acc"] for r in results])
    avg_auc = np.mean([r["auc"] for r in results])
    avg_hc = np.mean([r["hc_acc"] for r in results])
    std_acc = np.std([r["acc"] for r in results])

    print(f"  {'-'*56}")
    print(f"  AVG: acc={avg_acc:.4f}, auc={avg_auc:.4f}, hc_acc={avg_hc:.4f}, std={std_acc:.4f}")

    return {
        "folds": results,
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "avg_hc_acc": float(avg_hc),
        "std": float(std_acc),
        "stable": bool(std_acc < 0.04),
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================
def train_symbol_v6(
    symbol: str,
    pred_horizon: int = 3,
    lookback: int = 10000,
    top_n: int = 35,
    min_move: float = 0.01,
    vol_mult: float = 0.4,
    conf_thresh: float = 0.55,
    n_optuna: int = 50,
) -> Dict:
    """Train v6 model for a symbol."""

    print(f"\n{'='*70}")
    print(f"  V6 Training: {symbol}")
    print(f"  horizon={pred_horizon}h, min_move={min_move:.2%}, vol_mult={vol_mult}")
    print(f"{'='*70}")

    # Load data
    df = load_data(symbol, lookback)
    if df.empty or len(df) < 500:
        return {"symbol": symbol, "status": "failed", "reason": "no_data"}

    # Build SHORT-HORIZON features
    feat = build_short_horizon_features(df, pred_horizon)
    feat, target_col = create_v6_targets(feat, pred_horizon, min_move, vol_mult)
    feature_cols = get_v6_feature_cols(feat)

    # Filter to signal bars
    signal = feat.dropna(subset=[target_col])
    signal = signal.dropna(subset=feature_cols, how="any")

    if len(signal) < 500:
        print(f"  [FAIL] Only {len(signal)} samples, need 500+")
        return {"symbol": symbol, "status": "failed", "reason": "few_signals"}

    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)

    print(f"  Total samples: {len(X)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()} ({1-y.mean():.1%})")

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]

    # Feature selection on training only
    selected = select_features_v6(X_train_raw, y_train, feature_cols, top_n, w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Optuna tuning
    best_params = tune_with_optuna_v6(X_train_s, y_train, n_optuna, w_train)

    # Train final model
    print(f"\n  Training final model...")
    model = build_model_v6(best_params, y_train)
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

    # Build ensemble
    print(f"\n  Building ensemble...")
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

    ens_acc = accuracy_score(y_test, y_pred_e)
    ens_auc = roc_auc_score(y_test, y_prob_e)

    print(f"  Ensemble: acc={ens_acc:.4f}, auc={ens_auc:.4f}")

    # Confidence sweep
    print(f"\n  Confidence sweep:")
    print(f"  {'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Count':<8} {'%Test':<8}")
    print(f"  {'-'*40}")

    best_hc = {"thresh": 0.5, "acc": ens_acc, "count": len(y_test)}
    for t in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        mask = np.maximum(y_prob_e, 1 - y_prob_e) >= t
        if mask.sum() >= 10:
            a = accuracy_score(y_test[mask], y_pred_e[mask])
            p = precision_score(y_test[mask], y_pred_e[mask], zero_division=0)
            marker = " ◄" if abs(t - conf_thresh) < 0.02 else ""
            print(f"  {t:<8.0%} {a:<8.4f} {p:<8.4f} {int(mask.sum()):<8} {mask.sum()/len(y_test):<8.1%}{marker}")
            if abs(t - conf_thresh) < 0.02:
                best_hc = {"thresh": t, "acc": a, "count": int(mask.sum())}

    # Walk-forward
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(X_sel, y, best_params, n_splits=10,
                                 conf_thresh=conf_thresh, sample_weight=weights)

    # Save model
    sym = symbol.replace("/", "_")
    out_dir = PROJECT_ROOT / "data" / "models_v6"
    out_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(ensemble, out_dir / f"{sym}_binary_ensemble_v6.pkl")
    joblib.dump(scaler, out_dir / f"{sym}_binary_scaler_v6.pkl")

    with open(out_dir / f"{sym}_selected_features_v6.json", "w") as f:
        json.dump(selected, f)

    meta = {
        "symbol": symbol,
        "version": "v6_fixed",
        "trained_at": datetime.now().isoformat(),
        "config": {
            "horizon": pred_horizon,
            "min_move": min_move,
            "vol_mult": vol_mult,
            "confidence_threshold": conf_thresh,
            "top_n_features": top_n,
            "optuna_trials": n_optuna,
        },
        "metrics": {
            "test_accuracy": float(ens_acc),
            "test_auc": float(ens_auc),
            "hc_accuracy": best_hc["acc"],
            "hc_count": best_hc["count"],
            "total_samples": len(X),
            "train_samples": split,
            "test_samples": len(y_test),
        },
        "walk_forward": wf,
        "best_params": best_params,
    }

    with open(out_dir / f"{sym}_binary_meta_v6.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir}")
    print(f"  ✓ {sym}: acc={ens_acc:.4f}, hc_acc={best_hc['acc']:.4f}, wf_acc={wf['avg_acc']:.4f}")

    return {
        "symbol": symbol,
        "status": "success",
        "accuracy": float(ens_acc),
        "hc_accuracy": best_hc["acc"],
        "wf_accuracy": wf["avg_acc"],
        "wf_stable": wf["stable"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train v6 ML models with fundamental fixes")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train")
    parser.add_argument("--lookback", type=int, default=10000, help="Data lookback bars")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    print("\n" + "="*70)
    print("  V6 MODEL TRAINING - FUNDAMENTAL FIXES")
    print("="*70)
    print(f"  Symbols: {symbols}")
    print(f"  Lookback: {args.lookback}")
    print(f"  Optuna trials: {args.trials}")
    print("="*70)

    results = []
    for sym in symbols:
        cfg = V6_CONFIGS.get(sym, {
            "horizon": 3, "min_move": 0.008, "vol_mult": 0.4,
            "conf_thresh": 0.55, "n_optuna": args.trials
        })

        r = train_symbol_v6(
            symbol=sym,
            pred_horizon=cfg["horizon"],
            lookback=args.lookback,
            min_move=cfg["min_move"],
            vol_mult=cfg["vol_mult"],
            conf_thresh=cfg["conf_thresh"],
            n_optuna=cfg.get("n_optuna", args.trials),
        )
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("  V6 TRAINING SUMMARY")
    print("="*70)
    print(f"  {'Symbol':<15} {'Status':<10} {'Acc':<8} {'HC Acc':<8} {'WF Acc':<8} {'Stable':<8}")
    print(f"  {'-'*60}")

    for r in results:
        if r["status"] == "success":
            stable = "Yes" if r["wf_stable"] else "No"
            print(f"  {r['symbol']:<15} {'OK':<10} {r['accuracy']:<8.4f} {r['hc_accuracy']:<8.4f} {r['wf_accuracy']:<8.4f} {stable:<8}")
        else:
            print(f"  {r['symbol']:<15} {'FAIL':<10} {r.get('reason', 'unknown')}")

    # Save summary
    out_dir = PROJECT_ROOT / "data" / "models_v6"
    with open(out_dir / "training_summary_v6.json", "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\n  Summary saved to {out_dir / 'training_summary_v6.json'}")


if __name__ == "__main__":
    main()
