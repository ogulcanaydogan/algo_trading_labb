#!/usr/bin/env python3
"""
Train ML models compatible with the signal generator's feature engineering.
Uses the same FeatureEngineer class that the bot uses.
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.ml.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


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


def create_labels(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.01) -> pd.Series:
    """Create training labels based on future returns."""
    future_return = df['close'].pct_change(horizon).shift(-horizon)

    labels = pd.Series(1, index=df.index)  # Default FLAT
    labels[future_return > threshold] = 2   # LONG
    labels[future_return < -threshold] = 0  # SHORT

    return labels


def train_symbol(symbol: str, days: int, output_dir: Path) -> Dict[str, float]:
    """Train all models for a symbol."""

    # Fetch data
    df = fetch_data(symbol, days)
    if df is None:
        return {}

    # Extract features using FeatureEngineer (same as signal generator)
    logger.info("  Extracting features with FeatureEngineer...")
    fe = FeatureEngineer()
    df_features = fe.extract_features(df)

    # Create labels
    df_features['target'] = create_labels(df_features)

    # Remove rows with NaN
    df_clean = df_features.dropna()

    # Get feature columns
    exclude_cols = ['target', 'target_return', 'target_direction', 'target_class',
                   'target_strong_trend', 'target_risk_adjusted',
                   'future_return_1', 'future_return_3', 'future_return_5', 'future_return_10',
                   'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]

    X = df_clean[feature_cols].values
    y = df_clean['target'].values

    # Handle infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"  Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"  Class distribution: LONG={sum(y==2)}, FLAT={sum(y==1)}, SHORT={sum(y==0)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                class_weight='balanced', n_jobs=-1, random_state=42)
    rf_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        rf.fit(X_scaled[train_idx], y[train_idx])
        rf_scores.append(accuracy_score(y[val_idx], rf.predict(X_scaled[val_idx])))
    rf_cv = np.mean(rf_scores)
    results['random_forest'] = rf_cv
    logger.info(f"    CV Accuracy: {rf_cv:.2%}")
    rf.fit(X_scaled, y)

    # Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                    min_samples_leaf=10, random_state=42)
    gb_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        gb.fit(X_scaled[train_idx], y[train_idx])
        gb_scores.append(accuracy_score(y[val_idx], gb.predict(X_scaled[val_idx])))
    gb_cv = np.mean(gb_scores)
    results['gradient_boosting'] = gb_cv
    logger.info(f"    CV Accuracy: {gb_cv:.2%}")
    gb.fit(X_scaled, y)

    # XGBoost
    if XGB_AVAILABLE:
        logger.info("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                      objective='multi:softmax', num_class=3,
                                      use_label_encoder=False, eval_metric='mlogloss',
                                      random_state=42)
        xgb_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            xgb_model.fit(X_scaled[train_idx], y[train_idx])
            xgb_scores.append(accuracy_score(y[val_idx], xgb_model.predict(X_scaled[val_idx])))
        xgb_cv = np.mean(xgb_scores)
        results['xgboost'] = xgb_cv
        logger.info(f"    CV Accuracy: {xgb_cv:.2%}")
        xgb_model.fit(X_scaled, y)

    # Save models
    symbol_clean = symbol.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, output_dir / f"{symbol_clean}_random_forest_model.pkl")
    joblib.dump(scaler, output_dir / f"{symbol_clean}_random_forest_scaler.pkl")
    joblib.dump(gb, output_dir / f"{symbol_clean}_gradient_boosting_model.pkl")
    joblib.dump(scaler, output_dir / f"{symbol_clean}_gradient_boosting_scaler.pkl")

    if XGB_AVAILABLE:
        joblib.dump(xgb_model, output_dir / f"{symbol_clean}_xgboost_model.pkl")
        joblib.dump(scaler, output_dir / f"{symbol_clean}_xgboost_scaler.pkl")

    # Save metadata with feature names
    metadata = {
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
        'feature_names': feature_cols,
        'num_features': len(feature_cols),
        'num_samples': len(X),
        'models': {
            'random_forest': {'cv_accuracy': float(rf_cv)},
            'gradient_boosting': {'cv_accuracy': float(gb_cv)},
        }
    }
    if XGB_AVAILABLE:
        metadata['models']['xgboost'] = {'cv_accuracy': float(xgb_cv)}

    # Save to both meta files for compatibility
    with open(output_dir / f"{symbol_clean}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / f"{symbol_clean}_random_forest_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Models saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output-dir", type=str, default="data/models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}

    logger.info("=" * 60)
    logger.info("COMPATIBLE ML TRAINING")
    logger.info("=" * 60)
    logger.info(f"Using FeatureEngineer for compatibility with signal generator")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Days: {args.days}")
    logger.info("=" * 60)

    for symbol in args.symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {symbol}")
        logger.info(f"{'='*40}")
        results = train_symbol(symbol, args.days, output_dir)
        if results:
            all_results[symbol] = results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        logger.info(f"\n{symbol}:")
        for model, acc in results.items():
            logger.info(f"  {model}: {acc:.2%}")

    if all_results:
        all_acc = [acc for r in all_results.values() for acc in r.values()]
        logger.info(f"\nAverage: {np.mean(all_acc):.2%}")
        logger.info(f"Best: {max(all_acc):.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("Done! Restart bot to use new models.")


if __name__ == "__main__":
    main()
