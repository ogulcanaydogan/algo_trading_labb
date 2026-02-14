#!/usr/bin/env python3
"""
V6 Extended Training - Optimized for 1% Daily Returns

NEW FEATURES:
1. Multi-horizon models: 1h, 3h, 8h for faster signals
2. Extended asset coverage: NVDA, AMD, META, more crypto
3. Kelly-based confidence sizing recommendations
4. Higher frequency signal generation
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
# V6 EXTENDED CONFIGURATIONS - Optimized for 1% Daily
# =============================================================================

# Multi-horizon configurations for each asset
V6_EXTENDED_CONFIGS = {
    # =========== CRYPTO - 24/7 trading, most opportunities ===========
    # 1h horizon - for scalping during high volatility
    "BTC_USDT_1h":   {"base": "BTC_USDT", "horizon": 1, "min_move": 0.008, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    "ETH_USDT_1h":   {"base": "ETH_USDT", "horizon": 1, "min_move": 0.010, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    "SOL_USDT_1h":   {"base": "SOL_USDT", "horizon": 1, "min_move": 0.015, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    
    # 3h horizon - standard swing (existing)
    "BTC_USDT_3h":   {"base": "BTC_USDT", "horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    "ETH_USDT_3h":   {"base": "ETH_USDT", "horizon": 3, "min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    "SOL_USDT_3h":   {"base": "SOL_USDT", "horizon": 3, "min_move": 0.018, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    "XRP_USDT_3h":   {"base": "XRP_USDT", "horizon": 3, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.58, "n_optuna": 50, "asset_class": "crypto"},
    
    # NEW crypto assets
    "DOGE_USDT_1h":  {"base": "DOGE_USDT", "horizon": 1, "min_move": 0.020, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    "LINK_USDT_1h":  {"base": "LINK_USDT", "horizon": 1, "min_move": 0.015, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    "AVAX_USDT_1h":  {"base": "AVAX_USDT", "horizon": 1, "min_move": 0.018, "vol_mult": 0.5, "conf_thresh": 0.60, "n_optuna": 40, "asset_class": "crypto"},
    
    # =========== STOCKS - High volume tech ===========
    # 1h horizon for intraday
    "NVDA_1h":       {"base": "NVDA", "horizon": 1, "min_move": 0.010, "vol_mult": 0.5, "conf_thresh": 0.58, "n_optuna": 40, "asset_class": "stock"},
    "AMD_1h":        {"base": "AMD", "horizon": 1, "min_move": 0.012, "vol_mult": 0.5, "conf_thresh": 0.58, "n_optuna": 40, "asset_class": "stock"},
    "META_1h":       {"base": "META", "horizon": 1, "min_move": 0.010, "vol_mult": 0.5, "conf_thresh": 0.58, "n_optuna": 40, "asset_class": "stock"},
    "TSLA_1h":       {"base": "TSLA", "horizon": 1, "min_move": 0.015, "vol_mult": 0.5, "conf_thresh": 0.58, "n_optuna": 40, "asset_class": "stock"},
    "COIN_1h":       {"base": "COIN", "horizon": 1, "min_move": 0.020, "vol_mult": 0.5, "conf_thresh": 0.58, "n_optuna": 40, "asset_class": "stock"},
    
    # 8h horizon - daily swing
    "NVDA_8h":       {"base": "NVDA", "horizon": 8, "min_move": 0.015, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "AMD_8h":        {"base": "AMD", "horizon": 8, "min_move": 0.018, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "META_8h":       {"base": "META", "horizon": 8, "min_move": 0.012, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "TSLA_8h":       {"base": "TSLA", "horizon": 8, "min_move": 0.020, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "GOOGL_8h":      {"base": "GOOGL", "horizon": 8, "min_move": 0.010, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "MSFT_8h":       {"base": "MSFT", "horizon": 8, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    "AAPL_8h":       {"base": "AAPL", "horizon": 8, "min_move": 0.008, "vol_mult": 0.4, "conf_thresh": 0.55, "n_optuna": 50, "asset_class": "stock"},
    
    # =========== INDICES - 24h horizon ===========
    "SPX500_USD_24h": {"base": "SPX500_USD", "horizon": 24, "min_move": 0.005, "vol_mult": 0.35, "conf_thresh": 0.55, "n_optuna": 60, "asset_class": "index"},
    "NAS100_USD_24h": {"base": "NAS100_USD", "horizon": 24, "min_move": 0.006, "vol_mult": 0.35, "conf_thresh": 0.55, "n_optuna": 60, "asset_class": "index"},
}

# Default batch for training - prioritize high-volume, high-volatility
PRIORITY_BATCH_1 = [
    # Most active crypto for 24/7 coverage
    "BTC_USDT_1h", "ETH_USDT_1h", "SOL_USDT_1h",
    "BTC_USDT_3h", "ETH_USDT_3h",
]

PRIORITY_BATCH_2 = [
    # High-vol stocks  
    "NVDA_1h", "TSLA_1h", "AMD_1h",
    "NVDA_8h", "TSLA_8h",
]

PRIORITY_BATCH_3 = [
    # More crypto + remaining stocks
    "XRP_USDT_3h", "META_1h", "META_8h",
    "GOOGL_8h", "MSFT_8h",
]


# =============================================================================
# Data Loading (reuse from v6_improved)
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
# Multi-Horizon Feature Engineering
# =============================================================================
def build_multi_horizon_features(df: pd.DataFrame, pred_horizon: int = 1, asset_class: str = "crypto") -> pd.DataFrame:
    """
    Build features optimized for specific prediction horizon.
    
    1h horizon: Focus on momentum, volatility bursts, micro-structure
    3h horizon: Add trend filters, volume profile
    8h+ horizon: Add daily patterns, session analysis
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

    feat = df.copy()
    c = feat["close"]
    h = feat["high"]
    l = feat["low"]
    v = feat["volume"]

    # ===== CORE RETURNS - Aligned to horizon =====
    for p in [1, 2, 3, 6, 12, 24]:
        feat[f"ret_{p}h"] = c.pct_change(p)
    
    # ===== MOMENTUM - Critical for short horizons =====
    for p in [3, 6, 12]:
        feat[f"momentum_{p}"] = c.pct_change(p)
        feat[f"roc_{p}"] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    # ===== RSI - Multiple timeframes =====
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    feat["rsi_5"] = calc_rsi(c, 5)   # Ultra-short for 1h
    feat["rsi_7"] = calc_rsi(c, 7)
    feat["rsi_14"] = calc_rsi(c, 14)
    feat["rsi_distance_50"] = feat["rsi_14"] - 50
    
    # RSI momentum (change in RSI)
    feat["rsi_momentum"] = feat["rsi_14"].diff(3)
    
    # ===== EMAs - Short for 1h horizon =====
    feat["ema_3"] = c.ewm(span=3).mean()
    feat["ema_5"] = c.ewm(span=5).mean()
    feat["ema_10"] = c.ewm(span=10).mean()
    feat["ema_20"] = c.ewm(span=20).mean()
    
    feat["price_vs_ema3"] = (c - feat["ema_3"]) / feat["ema_3"]
    feat["price_vs_ema5"] = (c - feat["ema_5"]) / feat["ema_5"]
    feat["price_vs_ema10"] = (c - feat["ema_10"]) / feat["ema_10"]
    feat["price_vs_ema20"] = (c - feat["ema_20"]) / feat["ema_20"]
    
    feat["ema_3_5_diff"] = (feat["ema_3"] - feat["ema_5"]) / feat["ema_5"]
    feat["ema_5_10_diff"] = (feat["ema_5"] - feat["ema_10"]) / feat["ema_10"]
    feat["ema_10_20_diff"] = (feat["ema_10"] - feat["ema_20"]) / feat["ema_20"]
    
    # EMA alignment score (trend strength)
    feat["ema_alignment"] = (
        (feat["ema_3"] > feat["ema_5"]).astype(float) +
        (feat["ema_5"] > feat["ema_10"]).astype(float) +
        (feat["ema_10"] > feat["ema_20"]).astype(float)
    ) / 3
    
    # ===== VOLATILITY - Critical for 1h =====
    feat["vol_1h"] = feat["ret_1h"].rolling(1).std()  # Instantaneous
    feat["vol_3h"] = feat["ret_1h"].rolling(3).std()
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    
    # Volatility ratios (expansion/contraction)
    feat["vol_ratio_3_12"] = feat["vol_3h"] / (feat["vol_12h"] + 1e-8)
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)
    
    # Volatility regime - use shorter window for 1h compatibility
    vol_5d = feat["ret_1h"].rolling(5 * 24).std()  # 5-day instead of 20-day
    feat["volatility_regime"] = feat["vol_6h"] / (vol_5d.fillna(feat["vol_24h"]) + 1e-8)
    feat["vol_expanding"] = (feat["volatility_regime"] > 1.3).astype(float)
    feat["vol_contracting"] = (feat["volatility_regime"] < 0.7).astype(float)
    
    # ===== ATR =====
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_7"] = tr.rolling(7).mean()
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_ratio"] = feat["atr_7"] / c
    feat["atr_expansion"] = feat["atr_7"] / (feat["atr_14"] + 1e-8)
    
    # ===== ADX (Trend Strength) =====
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
    
    # ===== BOLLINGER BANDS =====
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feat["bb_upper"] = bb_sma + 2 * bb_std
    feat["bb_lower"] = bb_sma - 2 * bb_std
    feat["bb_position"] = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-8)
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / bb_sma
    feat["bb_squeeze"] = (feat["bb_width"] < feat["bb_width"].rolling(20).quantile(0.2)).astype(float)
    
    # ===== VOLUME FEATURES =====
    feat["vol_sma_10"] = v.rolling(10).mean()
    feat["vol_sma_20"] = v.rolling(20).mean()
    feat["vol_ratio"] = v / (feat["vol_sma_20"] + 1e-8)
    feat["vol_change"] = v.pct_change()
    feat["vol_spike"] = (feat["vol_ratio"] > 2.0).astype(float)
    
    # ===== PRICE RANGE (key for 1h reversals) =====
    feat["range_1h"] = (h - l) / c
    feat["range_3h"] = (h.rolling(3).max() - l.rolling(3).min()) / c
    feat["range_6h"] = (h.rolling(6).max() - l.rolling(6).min()) / c
    feat["range_12h"] = (h.rolling(12).max() - l.rolling(12).min()) / c
    
    feat["high_12h"] = h.rolling(12).max()
    feat["low_12h"] = l.rolling(12).min()
    feat["position_in_range_12h"] = (c - feat["low_12h"]) / (feat["high_12h"] - feat["low_12h"] + 1e-8)
    
    # ===== MACD =====
    ema_fast = c.ewm(span=8).mean()
    ema_slow = c.ewm(span=17).mean()
    feat["macd"] = ema_fast - ema_slow
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    feat["macd_hist_change"] = feat["macd_hist"].diff()
    feat["macd_cross_up"] = ((feat["macd"] > feat["macd_signal"]) & (feat["macd"].shift(1) <= feat["macd_signal"].shift(1))).astype(float)
    
    # ===== STOCHASTIC =====
    lowest_14 = l.rolling(14).min()
    highest_14 = h.rolling(14).max()
    feat["stoch_k"] = 100 * (c - lowest_14) / (highest_14 - lowest_14 + 1e-8)
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()
    
    # ===== Z-SCORES =====
    feat["zscore_5"] = (c - c.rolling(5).mean()) / (c.rolling(5).std() + 1e-8)
    feat["zscore_10"] = (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-8)
    feat["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)
    
    # ===== MOMENTUM REGIME =====
    feat["momentum_regime_oversold"] = (feat["rsi_14"] < 30).astype(float)
    feat["momentum_regime_neutral"] = ((feat["rsi_14"] >= 30) & (feat["rsi_14"] <= 70)).astype(float)
    feat["momentum_regime_overbought"] = (feat["rsi_14"] > 70).astype(float)
    
    # ===== TIME FEATURES =====
    if hasattr(feat.index, "hour"):
        feat["hour"] = feat.index.hour
        feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
        
        feat["is_asian"] = ((feat["hour"] >= 0) & (feat["hour"] < 8)).astype(float)
        feat["is_european"] = ((feat["hour"] >= 8) & (feat["hour"] < 16)).astype(float)
        feat["is_us"] = ((feat["hour"] >= 14) & (feat["hour"] < 22)).astype(float)
        
        if hasattr(feat.index, "dayofweek"):
            feat["dow"] = feat.index.dayofweek
            feat["dow_sin"] = np.sin(2 * np.pi * feat["dow"] / 7)
            feat["dow_cos"] = np.cos(2 * np.pi * feat["dow"] / 7)
            feat["is_monday"] = (feat["dow"] == 0).astype(float)
            feat["is_friday"] = (feat["dow"] == 4).astype(float)
    
    # ===== FORWARD RETURNS (targets) =====
    feat["fwd_return"] = c.pct_change(pred_horizon).shift(-pred_horizon)
    
    return feat


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns, excluding targets and raw OHLCV."""
    exclude = {
        "open", "high", "low", "close", "volume", "datetime",
        "fwd_return", "fwd_1h", "fwd_3h",
        "target", "sample_weight",
        "high_12h", "low_12h", "high_24h", "low_24h",
        "bb_upper", "bb_lower",
        "ema_3", "ema_5", "ema_10", "ema_20", "ema_50", "sma_10", "sma_20",
        "vol_sma_10", "vol_sma_20", "hour", "dow",
    }
    cols = [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]
    return cols


# =============================================================================
# Target Creation
# =============================================================================
def create_targets(
    df: pd.DataFrame,
    pred_horizon: int = 1,
    min_move: float = 0.01,
    vol_mult: float = 0.5,
) -> Tuple[pd.DataFrame, str]:
    """Create binary targets with adaptive threshold."""
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

    magnitude_ratio = np.abs(fwd) / (thresh + 1e-8)
    df["sample_weight"] = np.clip(magnitude_ratio, 1.0, 2.0)
    df.loc[df["target"].isna(), "sample_weight"] = 0.0

    n_up = (df["target"] == 1).sum()
    n_down = (df["target"] == 0).sum()
    n_total = len(df)
    n_sig = n_up + n_down

    print(f"  Target (horizon={pred_horizon}h, min_move={min_move:.2%}):")
    print(f"    UP: {n_up} ({n_up/n_total:.1%}), DOWN: {n_down} ({n_down/n_total:.1%})")
    print(f"    Signal rate: {n_sig/n_total:.1%}")

    return df, "target"


# =============================================================================
# Feature Selection
# =============================================================================
def select_features(X, y, names, top_n=40, sample_weight=None):
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
    for n, s in ranked[:8]:
        print(f"    {s:.4f}  {n}")

    return selected


# =============================================================================
# Optuna Tuning
# =============================================================================
def tune_with_optuna(X_train, y_train, n_trials=40, sample_weight=None) -> Dict:
    """Optuna tuning."""
    if not HAS_OPTUNA:
        return {"model": "lgb" if HAS_LGB else "xgb" if HAS_XGB else "gb"}

    print(f"\n  Optuna tuning ({n_trials} trials)...")

    def objective(trial):
        model_type = trial.suggest_categorical("model", ["lgb", "xgb"])

        if model_type == "lgb" and HAS_LGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5, log=True),
                "class_weight": "balanced",
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1)
        else:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_weight = neg_count / (pos_count + 1e-8)

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5, log=True),
                "scale_pos_weight": scale_weight,
            }
            model = xgb.XGBClassifier(**params, objective="binary:logistic",
                                       eval_metric="logloss", random_state=42, verbosity=0)

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
    return study.best_params


def build_model(params: Dict, y_train=None):
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
# Walk-Forward Validation
# =============================================================================
MIN_SAMPLES_PER_FOLD = 50

def walk_forward_expanding(X, y, params, n_splits=8, conf_thresh=0.58, sample_weight=None):
    """Walk-forward with expanding window."""
    n = len(X)
    test_size = max(n // (n_splits + 1), MIN_SAMPLES_PER_FOLD)
    results = []

    print(f"\n  Walk-Forward ({n_splits} folds):")

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

        hc = np.maximum(prob, 1 - prob) >= conf_thresh
        hc_acc = accuracy_score(y[te][hc], pred[hc]) if hc.sum() >= 5 else acc
        
        is_valid = test_samples >= MIN_SAMPLES_PER_FOLD
        results.append({"acc": acc, "auc": auc, "hc_acc": hc_acc, "is_valid": is_valid})

    valid_results = [r for r in results if r["is_valid"]]
    
    if valid_results:
        avg_acc = np.mean([r["acc"] for r in valid_results])
        avg_auc = np.mean([r["auc"] for r in valid_results])
        avg_hc = np.mean([r["hc_acc"] for r in valid_results])
        std_acc = np.std([r["acc"] for r in valid_results])
    else:
        avg_acc = avg_auc = avg_hc = 0.5
        std_acc = 0.1

    print(f"  WF Avg: acc={avg_acc:.4f}, auc={avg_auc:.4f}, hc_acc={avg_hc:.4f}")

    return {
        "avg_acc": float(avg_acc),
        "avg_auc": float(avg_auc),
        "avg_hc_acc": float(avg_hc),
        "std": float(std_acc),
        "stable": bool(std_acc < 0.04),
        "valid_folds": len(valid_results),
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================
def train_model(config_name: str, lookback: int = 10000) -> Dict:
    """Train a model for a specific configuration."""
    
    cfg = V6_EXTENDED_CONFIGS.get(config_name)
    if not cfg:
        return {"config": config_name, "status": "failed", "reason": "unknown_config"}
    
    base_symbol = cfg["base"]
    pred_horizon = cfg["horizon"]
    min_move = cfg["min_move"]
    vol_mult = cfg["vol_mult"]
    conf_thresh = cfg["conf_thresh"]
    n_optuna = cfg["n_optuna"]
    asset_class = cfg["asset_class"]

    print(f"\n{'='*70}")
    print(f"  TRAINING: {config_name} [{asset_class}]")
    print(f"  Base: {base_symbol}, Horizon: {pred_horizon}h, MinMove: {min_move:.2%}")
    print(f"{'='*70}")

    # Load data
    df = load_data(base_symbol, lookback)
    if df.empty or len(df) < 500:
        return {"config": config_name, "status": "failed", "reason": "no_data"}

    # Build features
    feat = build_multi_horizon_features(df, pred_horizon, asset_class)
    feat, target_col = create_targets(feat, pred_horizon, min_move, vol_mult)
    feature_cols = get_feature_cols(feat)

    # Filter to signal bars - only require target, fill NaN features
    signal = feat.dropna(subset=[target_col])
    
    # Skip initial warmup period (first 100 bars have too many NaN)
    signal = signal.iloc[100:]
    
    # Fill remaining NaN with 0 (safe for normalized features)
    signal[feature_cols] = signal[feature_cols].fillna(0)

    if len(signal) < 400:
        print(f"  [FAIL] Only {len(signal)} samples")
        return {"config": config_name, "status": "failed", "reason": "few_signals"}

    X = signal[feature_cols].values
    y = signal[target_col].values.astype(int)
    weights = signal["sample_weight"].values
    X = np.nan_to_num(np.where(np.isinf(X), 0, X), nan=0.0)

    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"  Class balance: UP={y.sum()} ({y.mean():.1%}), DOWN={len(y)-y.sum()}")

    # Train/test split
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = weights[:split]

    # Feature selection
    selected = select_features(X_train_raw, y_train, feature_cols, top_n=40, sample_weight=w_train)
    sel_idx = [feature_cols.index(f) for f in selected]
    X_train = X_train_raw[:, sel_idx]
    X_test = X_test_raw[:, sel_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Optuna tuning
    best_params = tune_with_optuna(X_train_s, y_train, n_optuna, w_train)

    # Train final model
    print(f"\n  Training final model...")
    model = build_model(best_params, y_train)
    model.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  Test Results: acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}")

    # Build ensemble
    voters = [
        ("tuned", build_model(best_params, y_train)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42)),
    ]
    ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_s, y_train, sample_weight=w_train)

    y_pred_e = ensemble.predict(X_test_s)
    y_prob_e = ensemble.predict_proba(X_test_s)[:, 1]

    ens_acc = accuracy_score(y_test, y_pred_e)
    ens_auc = roc_auc_score(y_test, y_prob_e)

    print(f"  Ensemble: acc={ens_acc:.4f}, auc={ens_auc:.4f}")

    # High confidence accuracy
    hc_mask = np.maximum(y_prob_e, 1 - y_prob_e) >= conf_thresh
    hc_acc = accuracy_score(y_test[hc_mask], y_pred_e[hc_mask]) if hc_mask.sum() >= 10 else ens_acc
    hc_count = int(hc_mask.sum())

    print(f"  High Conf ({conf_thresh:.0%}): acc={hc_acc:.4f}, count={hc_count}")

    # Walk-forward
    X_sel = X[:, sel_idx]
    wf = walk_forward_expanding(X_sel, y, best_params, n_splits=8, conf_thresh=conf_thresh, sample_weight=weights)

    # Save model
    out_dir = PROJECT_ROOT / "data" / "models_v6_extended"
    out_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(ensemble, out_dir / f"{config_name}_ensemble.pkl")
    joblib.dump(scaler, out_dir / f"{config_name}_scaler.pkl")

    with open(out_dir / f"{config_name}_features.json", "w") as f:
        json.dump(selected, f)

    # Calculate Kelly sizing recommendation
    win_rate = hc_acc
    avg_win = min_move * 1.5  # Assume 1.5x min_move on wins
    avg_loss = min_move  # Assume 1x min_move on losses
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly_fraction = max(0, min(0.5, kelly_fraction))  # Cap at 50%
    recommended_position = kelly_fraction * 0.5  # Half-Kelly for safety

    meta = {
        "config": config_name,
        "base_symbol": base_symbol,
        "version": "v6_extended",
        "asset_class": asset_class,
        "trained_at": datetime.now().isoformat(),
        "config_params": {
            "horizon": pred_horizon,
            "min_move": min_move,
            "vol_mult": vol_mult,
            "confidence_threshold": conf_thresh,
        },
        "metrics": {
            "test_accuracy": float(ens_acc),
            "test_auc": float(ens_auc),
            "hc_accuracy": float(hc_acc),
            "hc_count": hc_count,
            "wf_accuracy": wf["avg_acc"],
            "wf_stable": wf["stable"],
        },
        "position_sizing": {
            "kelly_fraction": float(kelly_fraction),
            "recommended_position_pct": float(recommended_position),
            "per_trade_ev": float(win_rate * avg_win - (1 - win_rate) * avg_loss),
        },
        "best_params": best_params,
    }

    with open(out_dir / f"{config_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir / config_name}")
    print(f"  Recommended position: {recommended_position:.1%} (Half-Kelly)")

    return {
        "config": config_name,
        "status": "success",
        "accuracy": float(ens_acc),
        "hc_accuracy": float(hc_acc),
        "wf_accuracy": wf["avg_acc"],
        "recommended_position": float(recommended_position),
        "horizon": pred_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="Train V6 Extended models for 1% daily returns")
    parser.add_argument("--configs", nargs="+", default=None, help="Specific configs to train")
    parser.add_argument("--batch", type=int, default=1, choices=[1, 2, 3], help="Training batch")
    parser.add_argument("--lookback", type=int, default=10000, help="Data lookback")
    parser.add_argument("--all", action="store_true", help="Train all configs")
    args = parser.parse_args()

    if args.configs:
        configs = args.configs
    elif args.all:
        configs = list(V6_EXTENDED_CONFIGS.keys())
    elif args.batch == 1:
        configs = PRIORITY_BATCH_1
    elif args.batch == 2:
        configs = PRIORITY_BATCH_2
    else:
        configs = PRIORITY_BATCH_3

    print("\n" + "="*70)
    print("  V6 EXTENDED MODEL TRAINING")
    print("  Goal: 1% Daily Returns through more signals + better sizing")
    print("="*70)
    print(f"  Configs: {configs}")
    print("="*70)

    results = []
    for cfg_name in configs:
        try:
            r = train_model(cfg_name, args.lookback)
            results.append(r)
        except Exception as e:
            print(f"  ERROR training {cfg_name}: {e}")
            results.append({"config": cfg_name, "status": "failed", "reason": str(e)})

    # Summary
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    print(f"  {'Config':<20} {'Horizon':<8} {'Acc':<8} {'HC Acc':<8} {'WF Acc':<8} {'Position':<10}")
    print(f"  {'-'*70}")

    total_ev = 0
    for r in results:
        if r["status"] == "success":
            print(f"  {r['config']:<20} {r['horizon']}h{'':<5} {r['accuracy']:<8.4f} {r['hc_accuracy']:<8.4f} {r['wf_accuracy']:<8.4f} {r['recommended_position']*100:<8.1f}%")
            # Estimate EV contribution
            # Assuming 1 signal per day per model on average
            ev = r["recommended_position"] * (r["hc_accuracy"] * 0.03 - (1 - r["hc_accuracy"]) * 0.015)
            total_ev += ev
        else:
            print(f"  {r['config']:<20} FAILED: {r.get('reason', 'unknown')}")

    print(f"\n  Estimated daily EV from trained models: {total_ev*100:.3f}%")
    print(f"  (Assumes ~1 trade per model per day)")

    # Save summary
    out_dir = PROJECT_ROOT / "data" / "models_v6_extended"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "results": results,
            "estimated_daily_ev": total_ev,
        }, f, indent=2)


if __name__ == "__main__":
    main()
