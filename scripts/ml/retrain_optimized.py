#!/usr/bin/env python3
"""
Optimized Model Retraining Script.

Uses:
- Optuna-tuned hyperparameters (58% accuracy achieved)
- 22 optimal features from feature selection
- Consistent feature set across all models
- Proper metadata for inference compatibility
"""

import json
import logging
import pickle
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optuna-tuned hyperparameters (58.21% accuracy)
TUNED_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "min_samples_split": 11,
    "min_samples_leaf": 9,
    "max_features": 0.7,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}

# Gradient Boosting params (tuned for similar performance)
TUNED_GB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "min_samples_split": 10,
    "min_samples_leaf": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
}

# XGBoost params
TUNED_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "random_state": 42,
    "n_jobs": -1,
}

# 22 optimal features from feature selection (SHAP + importance analysis)
OPTIMAL_FEATURES = [
    "ema_8_dist",
    "ema_21_dist",
    "ema_55_dist",
    "ema_100_dist",
    "rsi",
    "rsi_norm",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility",
    "volatility_ratio",
    "volume_ratio",
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "bb_position",
    "bb_width",
    "atr",
    "momentum",
    "momentum_acc",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 22 optimal features for the model."""
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
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    features["atr"] = tr.rolling(14).mean() / close

    # Momentum
    features["momentum"] = close.pct_change(10)
    features["momentum_acc"] = features["momentum"] - features["momentum"].shift(5)

    # Ensure correct column order
    return features[OPTIMAL_FEATURES]


def create_labels(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.003) -> pd.Series:
    """
    Create 3-class labels for price direction prediction.

    Classes:
    - 0: DOWN (return < -threshold)
    - 1: FLAT (return between -threshold and +threshold)
    - 2: UP (return > +threshold)
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon)

    labels = pd.Series(index=df.index, dtype=int)
    labels[future_return < -threshold] = 0  # DOWN
    labels[(future_return >= -threshold) & (future_return <= threshold)] = 1  # FLAT
    labels[future_return > threshold] = 2  # UP

    return labels


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data using yfinance."""
    import yfinance as yf

    yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
    logger.info(f"Fetching {days} days for {symbol} ({yf_symbol})...")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def prepare_data(symbol: str, days: int = 365) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and prepare data for training."""
    # Fetch data using yfinance
    df = fetch_data(symbol, days)

    if df is None or len(df) < 500:
        raise ValueError(f"Insufficient data for {symbol}")

    # Compute features
    X = compute_features(df)
    y = create_labels(df)

    # Remove NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Prepared {len(X)} samples for {symbol}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y, list(X.columns)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    symbol: str,
    output_dir: Path
) -> Dict:
    """Train a single model with cross-validation."""

    # Select model and params
    if model_type == "random_forest":
        model = RandomForestClassifier(**TUNED_RF_PARAMS)
        params = TUNED_RF_PARAMS
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**TUNED_GB_PARAMS)
        params = TUNED_GB_PARAMS
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(**TUNED_XGB_PARAMS, use_label_encoder=False, eval_metric="mlogloss")
            params = TUNED_XGB_PARAMS
        except ImportError:
            logger.warning("XGBoost not available, skipping")
            return None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)

    logger.info(f"{model_type} CV scores: {cv_scores}")
    logger.info(f"{model_type} CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train/test split (last 20% for test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"{model_type} Test accuracy: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['DOWN', 'FLAT', 'UP'])}")

    # Save model
    symbol_safe = symbol.replace("/", "_")
    model_path = output_dir / f"{symbol_safe}_{model_type}_model.pkl"
    scaler_path = output_dir / f"{symbol_safe}_{model_type}_scaler.pkl"
    meta_path = output_dir / f"{symbol_safe}_{model_type}_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata with feature names for inference
    meta = {
        "symbol": symbol,
        "model_type": model_type,
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(OPTIMAL_FEATURES),
        "feature_names": OPTIMAL_FEATURES,
        "hyperparameters": {k: str(v) if v is None else v for k, v in params.items()},
        "trained_at": datetime.now().isoformat(),
        "training_version": "optimized_v2",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {model_type} model to {model_path}")

    return {
        "model_type": model_type,
        "accuracy": accuracy,
        "f1": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def retrain_all_models(symbols: List[str], output_dir: Path) -> Dict:
    """Retrain all models for all symbols."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for {symbol}")
        logger.info(f"{'='*60}")

        try:
            X, y, feature_names = prepare_data(symbol)

            results[symbol] = {}

            for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
                try:
                    result = train_model(X, y, model_type, symbol, output_dir)
                    if result:
                        results[symbol][model_type] = result
                except Exception as e:
                    logger.error(f"Failed to train {model_type} for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Failed to prepare data for {symbol}: {e}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrain optimized models")
    parser.add_argument(
        "--symbols",
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="Comma-separated symbols"
    )
    parser.add_argument(
        "--output-dir",
        default="data/models",
        help="Output directory for models"
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",")
    output_dir = Path(args.output_dir)

    logger.info("="*60)
    logger.info("OPTIMIZED MODEL RETRAINING")
    logger.info("="*60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Features: {len(OPTIMAL_FEATURES)} optimal features")
    logger.info(f"RF params: max_depth={TUNED_RF_PARAMS['max_depth']}, n_estimators={TUNED_RF_PARAMS['n_estimators']}")
    logger.info("="*60)

    results = retrain_all_models(symbols, output_dir)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)

    for symbol, models in results.items():
        logger.info(f"\n{symbol}:")
        for model_type, metrics in models.items():
            logger.info(f"  {model_type}: accuracy={metrics['accuracy']:.4f}, "
                       f"CV={metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "optimal_features": OPTIMAL_FEATURES,
            "rf_params": TUNED_RF_PARAMS,
            "gb_params": TUNED_GB_PARAMS,
            "results": {
                sym: {m: {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics.items()}
                      for m, metrics in models.items()}
                for sym, models in results.items()
            }
        }, f, indent=2)

    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
