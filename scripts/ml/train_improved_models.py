#!/usr/bin/env python3
"""
Improved Model Training with Better Labeling Strategy.

Uses trend-following labels based on:
- Multi-timeframe trend confirmation
- Volatility-adjusted thresholds
- Forward-looking smoothed returns (reduces noise)
"""

import json
import logging
import joblib
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optimal 22 features
OPTIMAL_FEATURES = [
    "ema_8_dist", "ema_21_dist", "ema_55_dist", "ema_100_dist",
    "rsi", "rsi_norm", "macd", "macd_signal", "macd_hist",
    "volatility", "volatility_ratio", "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "bb_position", "bb_width", "atr", "momentum", "momentum_acc",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 22 optimal features."""
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA distances
    for period in [8, 21, 55, 100]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"ema_{period}_dist"] = (close - ema) / ema

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))
    features["rsi_norm"] = (features["rsi"] - 50) / 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    # Volatility
    features["volatility"] = close.pct_change().rolling(20).std()
    features["volatility_ratio"] = features["volatility"] / features["volatility"].rolling(50).mean()

    # Volume ratio
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    # Returns
    for period in [1, 3, 5, 10, 20]:
        features[f"return_{period}"] = close.pct_change(period)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    features["bb_width"] = (upper - lower) / sma20

    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    # Momentum
    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    return features[OPTIMAL_FEATURES]


def create_trend_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create trend-following labels using multi-timeframe confirmation.

    This approach:
    1. Uses smoothed future returns to reduce noise
    2. Confirms trend with multiple horizons
    3. Uses volatility-adjusted thresholds
    """
    close = df["close"]

    # Smoothed future returns at different horizons
    future_4h = close.pct_change(4).shift(-4).rolling(2).mean()
    future_8h = close.pct_change(8).shift(-8).rolling(2).mean()
    future_12h = close.pct_change(12).shift(-12).rolling(2).mean()

    # Combined forward-looking return with weighting
    future_return = (future_4h * 0.5 + future_8h * 0.3 + future_12h * 0.2)

    # Volatility-adjusted threshold
    volatility = close.pct_change().rolling(20).std()
    vol_threshold = volatility * 1.5  # 1.5x volatility as threshold
    vol_threshold = vol_threshold.clip(lower=0.002, upper=0.015)  # Bounded

    # Create labels
    labels = pd.Series(1, index=df.index)  # Default: HOLD
    labels[future_return > vol_threshold] = 2   # LONG
    labels[future_return < -vol_threshold] = 0  # SHORT

    return labels


def create_momentum_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create momentum-based labels for mean reversion strategy.
    """
    close = df["close"]

    # Future return
    future_6h = close.pct_change(6).shift(-6)

    # RSI for extremes
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # BB position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_position = (close - (sma20 - 2*std20)) / (4*std20 + 1e-10)

    labels = pd.Series(1, index=df.index)  # Default: HOLD

    # Long when oversold and future is up
    labels[(rsi < 35) & (future_6h > 0.002)] = 2
    # Short when overbought and future is down
    labels[(rsi > 65) & (future_6h < -0.002)] = 0

    return labels


def fetch_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch historical data."""
    import yfinance as yf

    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days for {symbol}...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str, symbol: str, output_dir: Path) -> Dict:
    """Train a model with the given data."""

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_split=10,
            min_samples_leaf=5, max_features=0.7, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, min_samples_split=10,
            min_samples_leaf=5, learning_rate=0.05, subsample=0.8,
            random_state=42
        )
    else:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss"
            )
        except ImportError:
            logger.warning("XGBoost not available")
            return None

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)
    logger.info(f"{model_type} CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"{model_type} Test: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['SHORT', 'HOLD', 'LONG'])}")

    # Save
    symbol_safe = symbol.replace("/", "_")
    joblib.dump(model, output_dir / f"{symbol_safe}_{model_type}_model.pkl")
    joblib.dump(scaler, output_dir / f"{symbol_safe}_{model_type}_scaler.pkl")

    meta = {
        "symbol": symbol,
        "model_type": model_type,
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_features": len(OPTIMAL_FEATURES),
        "feature_names": OPTIMAL_FEATURES,
        "labeling": "trend_following",
        "trained_at": datetime.now().isoformat(),
    }
    with open(output_dir / f"{symbol_safe}_{model_type}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"accuracy": accuracy, "f1": f1, "cv_mean": cv_scores.mean()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,LINK/USDT")
    parser.add_argument("--output-dir", default="data/models")
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("IMPROVED MODEL TRAINING")
    logger.info(f"Symbols: {symbols}")
    logger.info("Using trend-following labels with MTF confirmation")
    logger.info("="*60)

    results = {}
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol}")
        logger.info("="*60)

        try:
            df = fetch_data(symbol)
            if df is None or len(df) < 1000:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            X = compute_features(df)
            y = create_trend_labels(df)

            valid = X.notna().all(axis=1) & y.notna()
            X, y = X[valid], y[valid]

            logger.info(f"Samples: {len(X)}, Classes: {y.value_counts().to_dict()}")

            results[symbol] = {}
            for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
                result = train_model(X, y, model_type, symbol, output_dir)
                if result:
                    results[symbol][model_type] = result

        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    for symbol, models in results.items():
        logger.info(f"\n{symbol}:")
        for model_type, metrics in models.items():
            logger.info(f"  {model_type}: {metrics['accuracy']:.2%} (CV: {metrics['cv_mean']:.2%})")


if __name__ == "__main__":
    main()
