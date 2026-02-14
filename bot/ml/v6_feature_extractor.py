"""
V6 Feature Extractor - Matches training feature engineering.

This module provides feature extraction compatible with V6 improved models.
"""
import numpy as np
import pandas as pd
from typing import List


def build_v6_features(df: pd.DataFrame, pred_horizon: int = 8, asset_class: str = "stock") -> pd.DataFrame:
    """
    Build features aligned to prediction horizon - matches training exactly.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
        pred_horizon: Prediction horizon in hours (8 for stocks, 3 for crypto)
        asset_class: "stock", "crypto", or "index"
    
    Returns:
        DataFrame with engineered features
    """
    # Inject synthetic volume for forex/indices if needed
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
    
    # ===== MOMENTUM REGIME (RSI zones) =====
    feat["momentum_regime_oversold"] = (feat["rsi_14"] < 30).astype(float)
    feat["momentum_regime_neutral"] = ((feat["rsi_14"] >= 30) & (feat["rsi_14"] <= 70)).astype(float)
    feat["momentum_regime_overbought"] = (feat["rsi_14"] > 70).astype(float)
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
    feat["trend_strength"] = feat["adx"]
    feat["trend_direction"] = np.sign(plus_di - minus_di)

    # ===== VOLATILITY (SHORT-TERM ONLY) =====
    feat["vol_6h"] = feat["ret_1h"].rolling(6).std()
    feat["vol_12h"] = feat["ret_1h"].rolling(12).std()
    feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
    feat["vol_ratio_6_24"] = feat["vol_6h"] / (feat["vol_24h"] + 1e-8)
    
    # ===== VOLATILITY REGIME =====
    # Use shorter window if not enough data (min 5 days)
    vol_window = min(20 * 24, len(feat) - 1) if len(feat) > 120 else 120
    vol_20d = feat["ret_1h"].rolling(vol_window).std()
    feat["volatility_regime"] = feat["vol_24h"] / (vol_20d + 1e-8)
    # Fill early NaN values with neutral 1.0
    feat["volatility_regime"] = feat["volatility_regime"].fillna(1.0)
    feat["vol_expanding"] = (feat["volatility_regime"] > 1.2).astype(float)
    feat["vol_contracting"] = (feat["volatility_regime"] < 0.8).astype(float)

    # ===== VOLUME FEATURES =====
    feat["vol_sma_10"] = v.rolling(10).mean()
    feat["vol_sma_20"] = v.rolling(20).mean()
    feat["vol_ratio"] = v / (feat["vol_sma_20"] + 1e-8)
    feat["volume_ratio"] = feat["vol_ratio"]  # Alias for compatibility
    feat["vol_change"] = v.pct_change()

    # ===== PRICE RANGE =====
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

    # ===== TIME FEATURES =====
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

    # ===== Z-SCORES (short-term) =====
    feat["zscore_10"] = (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-8)
    feat["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)

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
