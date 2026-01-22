#!/usr/bin/env python3
"""
Hyperparameter Tuning with Optuna.

Optimizes ML model hyperparameters for better accuracy.
Uses Bayesian optimization for efficient search.

Usage:
    python scripts/ml/hyperparameter_tuning.py --symbol BTC/USDT --trials 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Run: pip install optuna")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
DATA_DIR = Path("data/training")


def fetch_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """Fetch historical data."""
    data_file = DATA_DIR / f"{symbol.replace('/', '_')}_extended.parquet"

    if data_file.exists():
        return pd.read_parquet(data_file)

    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        logger.info(f"Fetching {days} days of {symbol} data...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol.replace("/", ""), '1h', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical features."""
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))
    returns = close.pct_change()

    # EMAs
    for p in [8, 21, 55, 100]:
        ema = close.ewm(span=p).mean()
        df[f'ema_{p}_dist'] = (close - ema) / ema

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (df['rsi'] - 50) / 50

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = (ema12 - ema26) / close
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volatility
    df['volatility'] = returns.rolling(20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()

    # Volume
    df['volume_ratio'] = np.log1p(volume / (volume.rolling(20).mean() + 1))

    # Returns
    for p in [1, 3, 5, 10, 20]:
        df[f'return_{p}'] = returns.rolling(p).sum()

    # Bollinger
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)
    df['bb_width'] = (4 * bb_std) / bb_sma

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean() / close

    # Momentum
    df['momentum'] = close / close.shift(10) - 1
    df['momentum_acc'] = df['momentum'] - df['momentum'].shift(5)

    return df


def prepare_data(symbol: str, label_threshold: float = 0.01, label_horizon: int = 12):
    """Prepare data for training."""
    df = fetch_data(symbol)

    if df.empty or len(df) < 1000:
        raise ValueError(f"Insufficient data for {symbol}")

    df = create_features(df)

    # Create labels
    future_return = df['close'].shift(-label_horizon) / df['close'] - 1
    labels = pd.Series(1, index=df.index)  # FLAT
    labels[future_return > label_threshold] = 2  # UP
    labels[future_return < -label_threshold] = 0  # DOWN

    df['target'] = labels
    df = df.dropna()

    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]

    X = df[feature_cols].values
    y = df['target'].values.astype(int)

    return X, y, feature_cols


def objective_rf(trial, X, y):
    """Objective function for Random Forest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = RandomForestClassifier(**params)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)

    return scores.mean()


def objective_gb(trial, X, y):
    """Objective function for Gradient Boosting."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'random_state': 42,
    }

    model = GradientBoostingClassifier(**params)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)

    return scores.mean()


def tune_symbol(symbol: str, n_trials: int = 100, model_type: str = 'both'):
    """Tune hyperparameters for a symbol."""
    logger.info(f"Tuning {symbol} with {n_trials} trials...")

    # Prepare data
    X, y, feature_cols = prepare_data(symbol)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Data shape: {X_scaled.shape}, Classes: {np.bincount(y)}")

    results = {}
    symbol_clean = symbol.replace('/', '_')

    # Tune Random Forest
    if model_type in ['both', 'rf']:
        logger.info("Tuning Random Forest...")

        study_rf = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f'{symbol_clean}_rf'
        )

        study_rf.optimize(
            lambda trial: objective_rf(trial, X_scaled, y),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )

        logger.info(f"RF Best accuracy: {study_rf.best_value:.4f}")
        logger.info(f"RF Best params: {study_rf.best_params}")

        # Train final model with best params
        best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
        best_rf.fit(X_scaled, y)

        # Save
        joblib.dump(best_rf, MODEL_DIR / f"{symbol_clean}_random_forest_tuned_model.pkl")
        joblib.dump(scaler, MODEL_DIR / f"{symbol_clean}_random_forest_tuned_scaler.pkl")

        results['random_forest'] = {
            'best_accuracy': study_rf.best_value,
            'best_params': study_rf.best_params,
            'n_trials': n_trials
        }

    # Tune Gradient Boosting
    if model_type in ['both', 'gb']:
        logger.info("Tuning Gradient Boosting...")

        study_gb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f'{symbol_clean}_gb'
        )

        study_gb.optimize(
            lambda trial: objective_gb(trial, X_scaled, y),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )

        logger.info(f"GB Best accuracy: {study_gb.best_value:.4f}")
        logger.info(f"GB Best params: {study_gb.best_params}")

        # Train final model
        best_gb = GradientBoostingClassifier(**study_gb.best_params, random_state=42)
        best_gb.fit(X_scaled, y)

        # Save
        joblib.dump(best_gb, MODEL_DIR / f"{symbol_clean}_gradient_boosting_tuned_model.pkl")
        joblib.dump(scaler, MODEL_DIR / f"{symbol_clean}_gradient_boosting_tuned_scaler.pkl")

        results['gradient_boosting'] = {
            'best_accuracy': study_gb.best_value,
            'best_params': study_gb.best_params,
            'n_trials': n_trials
        }

    # Save metadata
    meta = {
        'symbol': symbol,
        'feature_cols': feature_cols,
        'results': results,
        'tuned_at': datetime.now().isoformat()
    }

    with open(MODEL_DIR / f"{symbol_clean}_tuned_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--model', choices=['both', 'rf', 'gb'], default='both', help='Model type')
    parser.add_argument('--all-symbols', action='store_true', help='Tune all symbols')
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not available. Install with: pip install optuna")
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.all_symbols:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT']
    else:
        symbols = [args.symbol]

    all_results = {}
    for symbol in symbols:
        try:
            results = tune_symbol(symbol, args.trials, args.model)
            all_results[symbol] = results
        except Exception as e:
            logger.error(f"Failed to tune {symbol}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        for model_type, data in results.items():
            logger.info(f"{symbol} {model_type}: {data['best_accuracy']:.2%}")


if __name__ == "__main__":
    main()
