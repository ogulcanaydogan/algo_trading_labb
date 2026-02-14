#!/usr/bin/env python3
"""
ML Model v6 IMPROVED: Major Improvements

Changes from v6_fixed:
1. FILTER SMALL FOLDS: Exclude folds with < 50 samples from WF stats
2. MARKET REGIME FEATURES: ADX, volatility regime, momentum regime
3. DUAL HORIZON: Stocks 24h, Crypto 3h, Indices 12h
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
# V6 IMPROVED Asset Configuration - DUAL HORIZON STRATEGY
# =============================================================================
V6_CONFIGS = {
    # Crypto - 3h horizon (high volatility, fast moves)
    "ETH_USDT":   {"horizon": 3,  "min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    # BTC_USDT: Increased min_move from 1.0% to 1.5% - improved WF from 49.3% to 51.4%
    "BTC_USDT":   {"horizon": 3,  "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.60, "n_optuna": 50, "asset_class": "crypto"},
    "SOL_USDT":   {"horizon": 3,  "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    "XRP_USDT":   {"horizon": 3,  "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},

    # Stocks - 24h horizon (daily moves, less noise)
    "GOOGL":      {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "MSFT":       {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "AAPL":       {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "TSLA":       {"horizon": 8, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "AMZN":       {"horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},

    # Indices - 24h horizon (daily moves, institutional flow) - IMPROVED
    "SPX500_USD": {"horizon": 24, "min_move": 0.005, "vol_mult": 0.35, "conf_thresh": 0.55, "n_optuna": 80, "asset_class": "index"},
    "NAS100_USD": {"horizon": 24, "min_move": 0.006, "vol_mult": 0.35, "conf_thresh": 0.55, "n_optuna": 80, "asset_class": "index"},
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
# V6 IMPROVED Feature Engineering - WITH MARKET REGIME
# =============================================================================
def build_short_horizon_features(df: pd.DataFrame, pred_horizon: int = 3, asset_class: str = "crypto") -> pd.DataFrame:
    """
    Build features aligned to prediction horizon.
    
    IMPROVEMENTS:
    - ADX trend strength
    - Volatility regime (current vs 20d avg)
    - Momentum regime (RSI zones)
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

    # ===== SHORT-HORIZON RETURNS =====
    feat["ret_1h"] = c.pct_change(1)
    feat["ret_2h"] = c.pct_change(2)
    feat["ret_3h"] = c.pct_change(3)
    feat["ret_6h"] = c.pct_change(6)
    feat["ret_12h"] = c.pct_change(12)
    feat["ret_24h"] = c.pct_change(24)
    
    # Additional horizon-aligned returns for stocks/indices
    if pred_horizon >= 12:
        feat["ret_48h"] = c.pct_change(48)
        feat["ret_72h"] = c.pct_change(72)

    # ===== SHORT-TERM MOVING AVERAGES =====
    feat["ema_5"] = c.ewm(span=5).mean()
    feat["ema_10"] = c.ewm(span=10).mean()
    feat["ema_20"] = c.ewm(span=20).mean()
    feat["sma_10"] = c.rolling(10).mean()
    feat["sma_20"] = c.rolling(20).mean()
    
    # For longer horizons, add 50-period MA
    if pred_horizon >= 12:
        feat["ema_50"] = c.ewm(span=50).mean()
        feat["price_vs_ema50"] = (c - feat["ema_50"]) / feat["ema_50"]

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
    
    # ===== NEW: MOMENTUM REGIME (RSI zones) =====
    feat["momentum_regime_oversold"] = (feat["rsi_14"] < 30).astype(float)
    feat["momentum_regime_neutral"] = ((feat["rsi_14"] >= 30) & (feat["rsi_14"] <= 70)).astype(float)
    feat["momentum_regime_overbought"] = (feat["rsi_14"] > 70).astype(float)
    # Continuous version for gradient
    feat["momentum_zone"] = np.where(feat["rsi_14"] < 30, -1, 
                                     np.where(feat["rsi_14"] > 70, 1, 0))

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

    # ===== NEW: ADX (Trend Strength) =====
    # Calculate +DM and -DM
    plus_dm_raw = h.diff()
    minus_dm_raw = -l.diff()
    plus_dm = pd.Series(np.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), plus_dm_raw, 0), index=feat.index)
    minus_dm = pd.Series(np.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), minus_dm_raw, 0), index=feat.index)
    
    # Smooth with Wilder's smoothing (14 period)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    feat["adx"] = dx.rolling(14).mean()
    feat["plus_di"] = plus_di
    feat["minus_di"] = minus_di
    feat["trend_strength"] = feat["adx"]  # Alias
    
    # Trend direction from DI
    feat["trend_direction"] = np.sign(plus_di - minus_di)

    # ===== VOLATILITY (SHORT-TERM ONLY) =====
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)
    
    # ===== NEW: VOLATILITY REGIME =====
    vol_20d = feat["ret_1h"].rolling(20 * 24).std()  # 20-day volatility
    feat["volatility_regime"] = feat["vol_24h"] / (vol_20d + 1e-8)  # Current vs 20d avg
    feat["vol_expanding"] = (feat["volatility_regime"] > 1.2).astype(float)
    feat["vol_contracting"] = (feat["volatility_regime"] < 0.8).astype(float)

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
    
    # Extended range for longer horizons
    if pred_horizon >= 12:
        feat["range_24h"] = (h.rolling(24).max() - l.rolling(24).min()) / c
        feat["high_24h"] = h.rolling(24).max()
        feat["low_24h"] = l.rolling(24).min()
        feat["position_in_range_24h"] = (c - feat["low_24h"]) / (feat["high_24h"] - feat["low_24h"] + 1e-8)

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
    if hasattr(feat.index, "hour"):
        feat["hour"] = feat.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)

        # Session indicators
        feat["is_asian"] = ((feat["hour"] >= 0) & (feat["hour"] < 8)).astype(float)
        feat["is_european"] = ((feat["hour"] >= 8) & (feat["hour"] < 16)).astype(float)
        feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)
        
        # Day of week for longer horizons
        if pred_horizon >= 12 and hasattr(feat.index, "dayofweek"):
            feat["dow"] = feat.index.dayofweek
            feat["dow_sin"] = np.sin(2 * np.pi * feat["dow"] / 7)
            feat["dow_cos"] = np.cos(2 * np.pi * feat["dow"] / 7)
            feat["is_monday"] = (feat["dow"] == 0).astype(float)
            feat["is_friday"] = (feat["dow"] == 4).astype(float)

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
        "high_12h", "low_12h", "high_24h", "low_24h",
        "bb_upper", "bb_lower",
        "ema_5", "ema_10", "ema_20", "ema_50", "sma_10", "sma_20",
        "vol_sma_10", "vol_sma_20", "hour", "dow",
    }
    cols = [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]
    return cols
# =============================================================================
# V6 Target Creation
# =============================================================================
def create_v6_targets(
    df: pd.DataFrame,
    pred_horizon: int = 3,
    min_move: float = 0.01,
    vol_mult: float = 0.4,
) -> Tuple[pd.DataFrame, str]:
    """Create binary targets with adaptive threshold."""
    # Adaptive threshold
    vol = df["close"].pct_change().rolling(24).std().fillna(min_move)
    thresh = np.maximum(vol * vol_mult, min_move)

    fwd = df["fwd_return"]

    df = df.copy()
    df["target"] = np.nan
    df["sample_weight"] = 1.0

    up_mask = fwd > thresh
    down_mask = fwd < -thresh

    df.loc[up_mask, "target"] = 1
    df.loc[down_mask, "target"] = 0

    # Simple magnitude weighting
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
# Optuna Tuning
# =============================================================================
def tune_with_optuna_v6(X_train, y_train, n_trials=50, sample_weight=None) -> Dict:
    """Optuna tuning with class_weight=balanced."""
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
                "class_weight": "balanced",
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1)
        elif model_type == "xgb" and HAS_XGB:
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
                "scale_pos_weight": scale_weight,
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
# Walk-Forward Validation - IMPROVED WITH SMALL FOLD FILTERING
# =============================================================================
MIN_SAMPLES_PER_FOLD = 50  # Minimum samples to include in statistics

def walk_forward_expanding(X, y, params, n_splits=10, conf_thresh=0.55, sample_weight=None):
    """Walk-forward with EXPANDING window. Excludes small folds from stats."""
    n = len(X)
    test_size = max(n // (n_splits + 1), MIN_SAMPLES_PER_FOLD)
    results = []

    print(f"\n  Walk-Forward Expanding ({n_splits} folds, min_samples={MIN_SAMPLES_PER_FOLD}):")
    print(f"  {'Fold':<6} {'Train':<8} {'Test':<8} {'Acc':<8} {'AUC':<8} {'HC Acc':<8} {'HC #':<6} {'Valid':<6}")
    print(f"  {'-'*64}")

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

        model = build_model_v6(params, y[tr])
        model.fit(X_t, y[tr], sample_weight=w_t)

        pred = model.predict(X_v)
        prob = model.predict_proba(X_v)[:, 1]

        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) > 1 else 0.5

        hc = np.maximum(prob, 1 - prob) >= conf_thresh
        hc_acc = accuracy_score(y[te][hc], pred[hc]) if hc.sum() >= 5 else acc
        hc_n = int(hc.sum())
        
        # Mark if fold is valid for statistics
        is_valid = test_samples >= MIN_SAMPLES_PER_FOLD
        valid_marker = "Y" if is_valid else "N"

        results.append({
            "acc": acc, "auc": auc, "hc_acc": hc_acc, "hc_n": hc_n,
            "test_samples": test_samples, "is_valid": is_valid
        })
        print(f"  {i+1:<6} {train_end:<8} {test_samples:<8} {acc:<8.4f} {auc:<8.4f} {hc_acc:<8.4f} {hc_n:<6} {valid_marker:<6}")

    # IMPROVED: Only calculate stats from valid folds (>= MIN_SAMPLES_PER_FOLD)
    valid_results = [r for r in results if r["is_valid"]]
    valid_folds = len(valid_results)
    
    if valid_folds > 0:
        avg_acc = np.mean([r["acc"] for r in valid_results])
        avg_auc = np.mean([r["auc"] for r in valid_results])
        avg_hc = np.mean([r["hc_acc"] for r in valid_results])
        std_acc = np.std([r["acc"] for r in valid_results])
    else:
        # Fallback to all folds if none are valid
        avg_acc = np.mean([r["acc"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        avg_hc = np.mean([r["hc_acc"] for r in results])
        std_acc = np.std([r["acc"] for r in results])

    print(f"  {'-'*64}")
    print(f"  AVG (valid folds={valid_folds}/{len(results)}): acc={avg_acc:.4f}, auc={avg_auc:.4f}, hc_acc={avg_hc:.4f}, std={std_acc:.4f}")

    return {
        "folds": results,
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "avg_hc_acc": float(avg_hc),
        "std": float(std_acc),
        "stable": bool(std_acc < 0.04),
        "valid_folds": valid_folds,
        "total_folds": len(results),
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
    asset_class: str = "crypto",
) -> Dict:
    """Train v6 IMPROVED model for a symbol."""

    print(f"\n{'='*70}")
    print(f"  V6 IMPROVED Training: {symbol} [{asset_class}]")
    print(f"  horizon={pred_horizon}h, min_move={min_move:.2%}, vol_mult={vol_mult}")
    print(f"{'='*70}")

    # Load data
    df = load_data(symbol, lookback)
    if df.empty or len(df) < 500:
        return {"symbol": symbol, "status": "failed", "reason": "no_data"}

    # Build features WITH MARKET REGIME
    feat = build_short_horizon_features(df, pred_horizon, asset_class)
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
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()} ({1-y.mean():.1%})")

    # Train/test split (80/20)
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
            marker = " <" if abs(t - conf_thresh) < 0.02 else ""
            print(f"  {t:<8.0%} {a:<8.4f} {p:<8.4f} {int(mask.sum()):<8} {mask.sum()/len(y_test):<8.1%}{marker}")
            if abs(t - conf_thresh) < 0.02:
                best_hc = {"thresh": t, "acc": a, "count": int(mask.sum())}

    # Walk-forward with small fold filtering
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(X_sel, y, best_params, n_splits=10,
                                 conf_thresh=conf_thresh, sample_weight=weights)

    # Save model
    sym = symbol.replace("/", "_")
    out_dir = PROJECT_ROOT / "data" / "models_v6_improved"
    out_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(ensemble, out_dir / f"{sym}_binary_ensemble_v6.pkl")
    joblib.dump(scaler, out_dir / f"{sym}_binary_scaler_v6.pkl")

    with open(out_dir / f"{sym}_selected_features_v6.json", "w") as f:
        json.dump(selected, f)

    meta = {
        "symbol": symbol,
        "version": "v6_improved",
        "asset_class": asset_class,
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
        "improvements": [
            "market_regime_features",
            "small_fold_filtering",
            "dual_horizon_strategy",
        ],
    }

    with open(out_dir / f"{sym}_binary_meta_v6.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir}")
    print(f"  OK {sym}: acc={ens_acc:.4f}, hc_acc={best_hc['acc']:.4f}, wf_acc={wf['avg_acc']:.4f} (valid_folds={wf['valid_folds']})")

    return {
        "symbol": symbol,
        "status": "success",
        "accuracy": float(ens_acc),
        "hc_accuracy": best_hc["acc"],
        "wf_accuracy": wf["avg_acc"],
        "wf_stable": wf["stable"],
        "wf_valid_folds": wf["valid_folds"],
        "horizon": pred_horizon,
        "asset_class": asset_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Train v6 IMPROVED ML models")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train")
    parser.add_argument("--lookback", type=int, default=10000, help="Data lookback bars")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    print("\n" + "="*70)
    print("  V6 IMPROVED MODEL TRAINING")
    print("  - Market regime features (ADX, vol regime, momentum regime)")
    print("  - Dual horizon strategy (crypto 3h, stocks 24h, indices 12h)")
    print("  - Small fold filtering in walk-forward")
    print("="*70)
    print(f"  Symbols: {symbols}")
    print(f"  Lookback: {args.lookback}")
    print(f"  Optuna trials: {args.trials}")
    print("="*70)

    results = []
    for sym in symbols:
        cfg = V6_CONFIGS.get(sym, {
            "horizon": 3, "min_move": 0.008, "vol_mult": 0.4,
            "conf_thresh": 0.55, "n_optuna": args.trials, "asset_class": "crypto"
        })

        r = train_symbol_v6(
            symbol=sym,
            pred_horizon=cfg["horizon"],
            lookback=args.lookback,
            min_move=cfg["min_move"],
            vol_mult=cfg["vol_mult"],
            conf_thresh=cfg["conf_thresh"],
            n_optuna=cfg.get("n_optuna", args.trials),
            asset_class=cfg.get("asset_class", "crypto"),
        )
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("  V6 IMPROVED TRAINING SUMMARY")
    print("="*70)
    print(f"  {'Symbol':<15} {'Class':<8} {'Horizon':<8} {'Acc':<8} {'HC Acc':<8} {'WF Acc':<8} {'VFolds':<8} {'Stable':<8}")
    print(f"  {'-'*76}")

    for r in results:
        if r["status"] == "success":
            stable = "Yes" if r["wf_stable"] else "No"
            print(f"  {r['symbol']:<15} {r.get('asset_class','?'):<8} {r.get('horizon','?')}h{'':<5} {r['accuracy']:<8.4f} {r['hc_accuracy']:<8.4f} {r['wf_accuracy']:<8.4f} {r.get('wf_valid_folds','?'):<8} {stable:<8}")
        else:
            print(f"  {r['symbol']:<15} {'FAIL':<10} {r.get('reason', 'unknown')}")

    # Save summary
    out_dir = PROJECT_ROOT / "data" / "models_v6_improved"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(out_dir / "training_summary_v6_improved.json", "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "improvements": [
                "market_regime_features",
                "small_fold_filtering", 
                "dual_horizon_strategy",
            ],
            "results": results,
        }, f, indent=2)

    print(f"\n  Summary saved to {out_dir / 'training_summary_v6_improved.json'}")


if __name__ == "__main__":
    main()
