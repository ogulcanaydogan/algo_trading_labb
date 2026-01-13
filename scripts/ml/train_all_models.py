#!/usr/bin/env python3
"""
Comprehensive ML Model Training Script.

Trains LSTM, Transformer, and traditional ML models for all configured symbols.
Supports both crypto and stock markets.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from bot.config import load_config
from bot.ml.predictor import MLPredictor
from bot.ml.feature_engineer import FeatureEngineer
from bot.ml.regime_classifier import MarketRegimeClassifier

# Try to import deep learning models
try:
    from bot.ml.models.deep_learning.lstm import LSTMModel
    from bot.ml.models.deep_learning.transformer import TransformerModel
    from bot.ml.registry import ModelRegistry
    HAS_DL = True
except ImportError as e:
    HAS_DL = False
    print(f"Deep learning models not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_training_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data for training."""
    try:
        import yfinance as yf

        # Convert symbol format
        if "/" in symbol:
            yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
        else:
            yf_symbol = symbol

        logger.info(f"Fetching {days} days of data for {symbol} ({yf_symbol})")

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.dropna()

        logger.info(f"  Got {len(df)} candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def train_traditional_models(
    symbol: str,
    ohlcv: pd.DataFrame,
    model_types: List[str] = ["random_forest", "xgboost"],
    save_dir: str = "data/models",
) -> Dict:
    """Train traditional ML models (Random Forest, XGBoost)."""
    results = {}

    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} for {symbol}...")

            predictor = MLPredictor(
                model_type=model_type,
                model_dir=save_dir,
            )

            metrics = predictor.train(ohlcv)
            model_name = f"{symbol.replace('/', '_')}_{model_type}"
            predictor.save(model_name)

            results[model_type] = {
                "accuracy": metrics.accuracy,
                "cross_val_mean": metrics.cross_val_mean,
                "cross_val_std": metrics.cross_val_std,
                "train_samples": metrics.train_samples,
                "test_samples": metrics.test_samples,
            }

            logger.info(f"  {model_type} accuracy: {metrics.accuracy:.2%}")

        except Exception as e:
            logger.error(f"  Failed to train {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results


def train_deep_learning_models(
    symbol: str,
    ohlcv: pd.DataFrame,
    model_types: List[str] = ["lstm", "transformer"],
    save_dir: str = "data/model_registry",
    epochs: int = 50,
) -> Dict:
    """Train deep learning models (LSTM, Transformer)."""
    if not HAS_DL:
        logger.warning("Deep learning models not available")
        return {}

    results = {}
    feature_engineer = FeatureEngineer()

    # Extract features
    logger.info(f"Extracting features for {symbol}...")
    df_features = feature_engineer.extract_features(ohlcv)

    if len(df_features) < 200:
        logger.warning(f"Insufficient data for deep learning ({len(df_features)} samples)")
        return {}

    # Prepare data
    exclude_cols = ["target_return", "target_direction", "target_class",
                    "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values

    # Create target (3-class classification)
    returns = df_features["close"].pct_change().shift(-1)
    threshold = returns.std() * 0.5

    y = np.where(returns > threshold, 2,  # LONG
                 np.where(returns < -threshold, 0,  # SHORT
                          1))  # FLAT

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # Handle inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data (time-series aware)
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"  Training data: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Initialize registry
    registry = ModelRegistry(registry_dir=save_dir)

    for model_type in model_types:
        try:
            logger.info(f"Training {model_type.upper()} for {symbol}...")

            if model_type == "lstm":
                model = LSTMModel(
                    input_size=X_train.shape[1],
                    hidden_size=128,
                    num_layers=2,
                    num_classes=3,
                )
                seq_len = 60
            else:  # transformer
                model = TransformerModel(
                    input_size=X_train.shape[1],
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    num_classes=3,
                )
                seq_len = 120

            # Create sequences
            def create_sequences(X, y, seq_len):
                X_seq, y_seq = [], []
                for i in range(seq_len, len(X)):
                    X_seq.append(X[i-seq_len:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)

            X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

            if len(X_train_seq) < 100:
                logger.warning(f"Insufficient sequences for {model_type}")
                continue

            # Train model
            history = model.train(
                X_train_seq, y_train_seq,
                X_test_seq, y_test_seq,
                epochs=epochs,
                batch_size=32,
            )

            # Evaluate
            predictions = model.predict(X_test_seq)
            accuracy = (predictions == y_test_seq).mean()

            # Save to registry
            safe_symbol = symbol.replace("/", "_")
            model.save(f"{save_dir}/{safe_symbol}_{model_type}")

            # Register model
            registry.register_model(
                model=model,
                symbol=symbol,
                market_type="crypto" if "USDT" in symbol else "stock",
                model_type=model_type,
                accuracy=accuracy,
            )

            results[model_type] = {
                "accuracy": accuracy,
                "train_samples": len(X_train_seq),
                "test_samples": len(X_test_seq),
                "epochs": epochs,
                "final_train_loss": history.get("train_loss", [])[-1] if history.get("train_loss") else None,
            }

            logger.info(f"  {model_type.upper()} accuracy: {accuracy:.2%}")

        except Exception as e:
            logger.error(f"  Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {"error": str(e)}

    return results


def run_backtest(
    symbol: str,
    ohlcv: pd.DataFrame,
    model_type: str = "random_forest",
    model_dir: str = "data/models",
) -> Dict:
    """Run a simple backtest using trained model."""
    try:
        logger.info(f"Running backtest for {symbol} with {model_type}...")

        predictor = MLPredictor(
            model_type=model_type,
            model_dir=model_dir,
        )

        model_name = f"{symbol.replace('/', '_')}_{model_type}"
        if not predictor.load(model_name):
            logger.warning(f"Model not found: {model_name}")
            return {}

        # Run backtest
        results = predictor.backtest_predictions(
            ohlcv,
            initial_capital=10000,
            position_size_pct=0.1,
        )

        logger.info(f"  Backtest results:")
        logger.info(f"    Total return: {results.get('total_return_pct', 0):.2f}%")
        logger.info(f"    Win rate: {results.get('win_rate', 0):.2%}")
        logger.info(f"    Max drawdown: {results.get('max_drawdown_pct', 0):.2f}%")

        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to train (default: from config)")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of historical data (default: 365)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for deep learning (default: 50)")
    parser.add_argument("--skip-traditional", action="store_true",
                        help="Skip traditional ML models")
    parser.add_argument("--skip-deep-learning", action="store_true",
                        help="Skip deep learning models")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest after training")

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = config.crypto.symbols[:4]  # Default to first 4 crypto symbols

    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days of data: {args.days}")
    logger.info(f"Deep Learning epochs: {args.epochs}")
    logger.info("=" * 60)

    all_results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*40}")

        # Fetch data
        ohlcv = fetch_training_data(symbol, args.days)
        if ohlcv is None or len(ohlcv) < 500:
            logger.warning(f"Skipping {symbol} - insufficient data")
            continue

        symbol_results = {"data_points": len(ohlcv)}

        # Train traditional models
        if not args.skip_traditional:
            traditional_results = train_traditional_models(
                symbol, ohlcv,
                model_types=["random_forest"],  # XGBoost optional
            )
            symbol_results["traditional"] = traditional_results

        # Train deep learning models
        if not args.skip_deep_learning and HAS_DL:
            dl_results = train_deep_learning_models(
                symbol, ohlcv,
                model_types=["lstm"],  # Transformer optional for speed
                epochs=args.epochs,
            )
            symbol_results["deep_learning"] = dl_results

        # Run backtest
        if args.backtest:
            backtest_results = run_backtest(symbol, ohlcv)
            symbol_results["backtest"] = backtest_results

        all_results[symbol] = symbol_results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Data points: {results.get('data_points', 0)}")

        if "traditional" in results:
            for model, metrics in results["traditional"].items():
                if "accuracy" in metrics:
                    logger.info(f"  {model}: {metrics['accuracy']:.2%}")

        if "deep_learning" in results:
            for model, metrics in results["deep_learning"].items():
                if "accuracy" in metrics:
                    logger.info(f"  {model}: {metrics['accuracy']:.2%}")

        if "backtest" in results and "total_return_pct" in results["backtest"]:
            logger.info(f"  Backtest return: {results['backtest']['total_return_pct']:.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
