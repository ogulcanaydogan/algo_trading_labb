#!/usr/bin/env python3
"""
Apply Tuned Hyperparameters to Retrain Models.

Reads the tuned_meta.json files and retrains models with optimal params.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")

FEATURES = [
    "ema_8_dist", "ema_21_dist", "ema_55_dist", "ema_100_dist",
    "rsi", "rsi_norm", "macd", "macd_signal", "macd_hist",
    "volatility", "volatility_ratio", "volume_ratio",
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "bb_position", "bb_width", "atr", "momentum", "momentum_acc",
]


def fetch_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch historical data."""
    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days of {symbol} data...")
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 22 features."""
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    for period in [8, 21, 55, 100]:
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"ema_{period}_dist"] = (close - ema) / ema

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))
    features["rsi_norm"] = (features["rsi"] - 50) / 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    features["volatility"] = close.pct_change().rolling(20).std()
    features["volatility_ratio"] = features["volatility"] / features["volatility"].rolling(50).mean()
    features["volume_ratio"] = volume / volume.rolling(20).mean()

    for period in [1, 3, 5, 10, 20]:
        features[f"return_{period}"] = close.pct_change(period)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    features["bb_width"] = (upper - lower) / sma20

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    return features[FEATURES]


def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """Create 3-class labels: DOWN=0, FLAT=1, UP=2."""
    future_return = df["close"].shift(-24) / df["close"] - 1
    labels = pd.Series(1, index=df.index)  # FLAT
    labels[future_return > threshold] = 2  # UP
    labels[future_return < -threshold] = 0  # DOWN
    return labels


def retrain_with_tuned_params(symbol: str) -> dict:
    """Retrain model with tuned hyperparameters."""
    symbol_clean = symbol.replace("/", "_")
    meta_file = MODEL_DIR / f"{symbol_clean}_tuned_meta.json"

    if not meta_file.exists():
        logger.warning(f"No tuned params for {symbol}")
        return {"symbol": symbol, "success": False, "reason": "No tuned params"}

    with open(meta_file) as f:
        meta = json.load(f)

    rf_params = meta["results"]["random_forest"]["best_params"]
    gb_params = meta["results"]["gradient_boosting"]["best_params"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Retraining {symbol} with tuned parameters")
    logger.info(f"RF params: {rf_params}")
    logger.info(f"GB params: {gb_params}")
    logger.info("=" * 60)

    # Fetch data
    df = fetch_data(symbol)
    if df.empty or len(df) < 1000:
        return {"symbol": symbol, "success": False, "reason": "Insufficient data"}

    # Features and labels
    features = compute_features(df)
    labels = create_labels(df)

    # Combine and drop NaN
    data = features.copy()
    data["target"] = labels
    data = data.dropna()

    X = data[FEATURES].values
    y = data["target"].values.astype(int)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (time-based)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    # Train Random Forest with tuned params
    logger.info("Training Random Forest with tuned params...")
    rf_model = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 200),
        max_depth=rf_params.get("max_depth", 10),
        min_samples_split=rf_params.get("min_samples_split", 5),
        min_samples_leaf=rf_params.get("min_samples_leaf", 2),
        max_features=rf_params.get("max_features", "sqrt"),
        class_weight=rf_params.get("class_weight"),
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    logger.info(f"RF Test Accuracy: {rf_acc:.4f}")
    results["random_forest"] = rf_acc

    # Save RF model
    joblib.dump(rf_model, MODEL_DIR / f"{symbol_clean}_random_forest_model.pkl")

    # Train Gradient Boosting with tuned params
    logger.info("Training Gradient Boosting with tuned params...")
    gb_model = GradientBoostingClassifier(
        n_estimators=gb_params.get("n_estimators", 100),
        max_depth=gb_params.get("max_depth", 3),
        learning_rate=gb_params.get("learning_rate", 0.1),
        min_samples_split=gb_params.get("min_samples_split", 10),
        min_samples_leaf=gb_params.get("min_samples_leaf", 5),
        subsample=gb_params.get("subsample", 0.8),
        max_features=gb_params.get("max_features", "sqrt"),
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    logger.info(f"GB Test Accuracy: {gb_acc:.4f}")
    results["gradient_boosting"] = gb_acc

    # Save GB model
    joblib.dump(gb_model, MODEL_DIR / f"{symbol_clean}_gradient_boosting_model.pkl")

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / f"{symbol_clean}_scaler.pkl")

    # Update metadata
    meta_out = {
        "symbol": symbol,
        "features": FEATURES,
        "tuned_params": {
            "random_forest": rf_params,
            "gradient_boosting": gb_params
        },
        "results": {
            "random_forest_accuracy": rf_acc,
            "gradient_boosting_accuracy": gb_acc
        },
        "trained_at": datetime.now().isoformat(),
        "data_points": len(X_scaled)
    }

    with open(MODEL_DIR / f"{symbol_clean}_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    logger.info(f"\n{symbol} Results:")
    logger.info(f"  Random Forest: {rf_acc:.4f}")
    logger.info(f"  Gradient Boosting: {gb_acc:.4f}")

    return {"symbol": symbol, "success": True, **results}


def main():
    """Apply tuned params to all symbols with tuned metadata."""
    logger.info("=" * 70)
    logger.info("APPLYING TUNED HYPERPARAMETERS")
    logger.info("=" * 70)

    # Find all tuned meta files
    tuned_files = list(MODEL_DIR.glob("*_tuned_meta.json"))
    logger.info(f"Found {len(tuned_files)} tuned configurations")

    results = []
    for meta_file in tuned_files:
        symbol_clean = meta_file.stem.replace("_tuned_meta", "")
        symbol = symbol_clean.replace("_", "/")

        result = retrain_with_tuned_params(symbol)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 70)

    for r in results:
        if r.get("success"):
            rf = r.get("random_forest", 0)
            gb = r.get("gradient_boosting", 0)
            logger.info(f"  {r['symbol']}: RF={rf:.4f}, GB={gb:.4f}")
        else:
            logger.info(f"  {r['symbol']}: FAILED - {r.get('reason', 'Unknown')}")


if __name__ == "__main__":
    main()
