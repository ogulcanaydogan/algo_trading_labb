#!/usr/bin/env python3
"""
Enhanced Model Training Pipeline.

A comprehensive training script for LSTM and Transformer models with:
- Hyperparameter optimization
- Multi-symbol training
- Data augmentation
- Progress tracking and reporting

Usage:
    python tools/train_models.py --symbols BTC,ETH --model-types lstm,transformer --epochs 100
    python tools/train_models.py --fast  # Quick training with fewer hyperparameters
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.ml.feature_engineer import FeatureEngineer
from bot.ml.models.deep_learning.lstm import LSTMModel, LSTMConfig
from bot.ml.models.deep_learning.transformer import TransformerModel, TransformerConfig
from bot.ml.registry.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "AVAX"]
DEFAULT_MODEL_TYPES = ["lstm", "transformer"]
DEFAULT_EPOCHS = 100
DEFAULT_TIMEFRAMES = ["1h", "4h"]

# Hyperparameter search space
FULL_HYPERPARAMS = {
    "hidden_sizes": [64, 128, 256],
    "learning_rates": [1e-3, 1e-4, 5e-4],
    "dropout_rates": [0.1, 0.2, 0.3],
}

FAST_HYPERPARAMS = {
    "hidden_sizes": [128],
    "learning_rates": [1e-3],
    "dropout_rates": [0.2],
}

# Rate limiting for Yahoo Finance
RATE_LIMIT_DELAY = 2.0  # seconds between API calls


@dataclass
class TrainingResult:
    """Result from training a single model."""

    symbol: str
    model_type: str
    timeframe: str
    hidden_size: int
    learning_rate: float
    dropout: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_loss: float
    val_loss: float
    epochs_trained: int
    best_epoch: int
    training_time: float
    history: Dict[str, List[float]]
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "timeframe": self.timeframe,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "train_accuracy": round(self.train_accuracy, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "training_time": round(self.training_time, 2),
            "model_path": self.model_path,
        }


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_historical_data(
    symbol: str,
    timeframe: str = "1h",
    years: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        timeframe: Data timeframe ("1h" or "4h")
        years: Number of years of historical data

    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return None

    # Map symbols to Yahoo Finance tickers
    ticker_map = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "AVAX": "AVAX-USD",
    }

    ticker = ticker_map.get(symbol, f"{symbol}-USD")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # Map timeframe to yfinance interval
    interval_map = {
        "1h": "1h",
        "4h": "1h",  # Fetch 1h and resample to 4h
    }
    interval = interval_map.get(timeframe, "1h")

    logger.info(f"Fetching {years} years of {timeframe} data for {symbol}...")

    try:
        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None

        df = df[required_cols]

        # Resample to 4h if needed
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_level: float = 0.01,
    time_warp_ratio: float = 0.1,
    augment_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to training data.

    Args:
        X: Input features (samples, sequence_length, features)
        y: Target labels
        noise_level: Standard deviation of Gaussian noise
        time_warp_ratio: Ratio of time steps to warp
        augment_factor: How many augmented copies to create

    Returns:
        Augmented X and y arrays
    """
    augmented_X = [X]
    augmented_y = [y]

    for i in range(augment_factor - 1):
        # Noise injection
        noise = np.random.normal(0, noise_level, X.shape)
        noisy_X = X + noise
        augmented_X.append(noisy_X)
        augmented_y.append(y)

        # Time warping (slightly stretch/compress sequences)
        if X.ndim == 3:
            warped_X = time_warp(X, ratio=time_warp_ratio)
            augmented_X.append(warped_X)
            augmented_y.append(y)

    return np.concatenate(augmented_X), np.concatenate(augmented_y)


def time_warp(X: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """
    Apply time warping to sequences.

    Randomly speeds up or slows down parts of the sequence.
    """
    samples, seq_len, features = X.shape
    warped = np.zeros_like(X)

    for i in range(samples):
        # Create a random warping path
        warp_amount = np.random.uniform(1 - ratio, 1 + ratio, seq_len)
        warp_path = np.cumsum(warp_amount)
        warp_path = warp_path / warp_path[-1] * (seq_len - 1)

        # Interpolate features along warped path
        for f in range(features):
            warped[i, :, f] = np.interp(
                np.arange(seq_len),
                warp_path,
                X[i, :, f],
            )

    return warped


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    augment: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Prepare data with proper train/val/test splits.

    Args:
        df: DataFrame with features and targets
        sequence_length: Length of input sequences
        train_ratio: Ratio of data for training (0.70)
        val_ratio: Ratio of data for validation (0.15)
        augment: Whether to apply data augmentation

    Returns:
        Tuple of (train_data, val_data, test_data)
        Each is a tuple of (X, y)
    """
    # Extract features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.extract_features(df)

    # Get feature columns (exclude targets and OHLCV)
    exclude_cols = ["target_return", "target_direction", "target_class",
                    "open", "high", "low", "close", "volume"]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    # Remove any columns with 'ema_' that are raw values (keep only distances)
    feature_cols = [col for col in feature_cols if not (col.startswith("ema_") and not col.endswith("_dist") and col != "ema_cross" and col != "ema_cross_signal")]

    # Prepare arrays
    X = df_features[feature_cols].values
    y = df_features["target_class"].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create sequences
    sequences_X = []
    sequences_y = []

    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length])

    X_seq = np.array(sequences_X)
    y_seq = np.array(sequences_y)

    # Split indices (time-based, no shuffling)
    n_samples = len(X_seq)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]

    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]

    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]

    # Apply augmentation to training data only
    if augment and len(X_train) > 0:
        X_train, y_train = augment_data(X_train, y_train, augment_factor=2)
        # Shuffle augmented training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

    logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================================
# Model Training
# ============================================================================

def train_model(
    model_type: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    hidden_size: int,
    learning_rate: float,
    dropout: float,
    epochs: int,
    symbol: str,
    timeframe: str,
    model_dir: str = "data/models",
) -> TrainingResult:
    """
    Train a single model with specified hyperparameters.

    Args:
        model_type: "lstm" or "transformer"
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        test_data: (X_test, y_test)
        hidden_size: Size of hidden layers
        learning_rate: Learning rate
        dropout: Dropout rate
        epochs: Maximum number of epochs
        symbol: Trading symbol
        timeframe: Data timeframe
        model_dir: Directory to save models

    Returns:
        TrainingResult with training metrics
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    input_size = X_train.shape[2]
    sequence_length = X_train.shape[1]

    start_time = time.time()

    # Create model config
    model_name = f"{symbol}_{timeframe}_{model_type}_h{hidden_size}_lr{learning_rate}_d{dropout}"

    if model_type == "lstm":
        config = LSTMConfig(
            name=model_name,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            dropout=dropout,
            epochs=epochs,
            early_stopping_patience=15,
        )
        model = LSTMModel(config=config, model_dir=model_dir)

    elif model_type == "transformer":
        # For transformer, hidden_size maps to model_dim
        config = TransformerConfig(
            name=model_name,
            sequence_length=sequence_length,
            model_dim=hidden_size,
            learning_rate=learning_rate / 10,  # Transformers need lower LR
            dropout=dropout,
            epochs=epochs,
            early_stopping_patience=15,
        )
        model = TransformerModel(config=config, model_dir=model_dir)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    logger.info(f"Training {model_type} for {symbol} ({timeframe}) - "
                f"hidden={hidden_size}, lr={learning_rate}, dropout={dropout}")

    try:
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))

        # Evaluate on test set
        test_predictions = model.predict_proba(X_test)
        test_preds = np.argmax(test_predictions, axis=1)
        test_accuracy = np.mean(test_preds == y_test)

        training_time = time.time() - start_time

        result = TrainingResult(
            symbol=symbol,
            model_type=model_type,
            timeframe=timeframe,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            dropout=dropout,
            train_accuracy=metrics.train_accuracy,
            val_accuracy=metrics.val_accuracy,
            test_accuracy=test_accuracy,
            train_loss=metrics.train_loss,
            val_loss=metrics.val_loss,
            epochs_trained=metrics.epochs_trained,
            best_epoch=metrics.best_epoch,
            training_time=training_time,
            history=metrics.history,
        )

        logger.info(f"  -> Val Acc: {metrics.val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

        return result, model

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None, None


def run_hyperparameter_search(
    symbol: str,
    timeframe: str,
    model_type: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    hyperparams: Dict[str, List],
    epochs: int,
    model_dir: str = "data/models",
) -> Tuple[TrainingResult, Any]:
    """
    Run hyperparameter search and return best model.

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        model_type: "lstm" or "transformer"
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        hyperparams: Dictionary of hyperparameter lists
        epochs: Maximum epochs
        model_dir: Model save directory

    Returns:
        Best TrainingResult and trained model
    """
    best_result = None
    best_model = None
    best_val_acc = 0.0

    all_results = []

    # Generate all combinations
    combinations = list(product(
        hyperparams["hidden_sizes"],
        hyperparams["learning_rates"],
        hyperparams["dropout_rates"],
    ))

    total_combos = len(combinations)
    logger.info(f"Running {total_combos} hyperparameter combinations for {symbol} {model_type}")

    for idx, (hidden_size, lr, dropout) in enumerate(combinations, 1):
        logger.info(f"  Combination {idx}/{total_combos}")

        result, model = train_model(
            model_type=model_type,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            hidden_size=hidden_size,
            learning_rate=lr,
            dropout=dropout,
            epochs=epochs,
            symbol=symbol,
            timeframe=timeframe,
            model_dir=model_dir,
        )

        if result is not None:
            all_results.append(result)

            if result.val_accuracy > best_val_acc:
                best_val_acc = result.val_accuracy
                best_result = result
                best_model = model
                logger.info(f"  New best validation accuracy: {best_val_acc:.4f}")

    return best_result, best_model, all_results


# ============================================================================
# Training Curves
# ============================================================================

def save_training_curves(
    results: List[TrainingResult],
    output_dir: str = "data/training_reports",
) -> None:
    """
    Save training curves as plots.

    Args:
        results: List of training results with history
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping training curves")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        if not result.history:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(result.history.get("train_loss", []), label="Train Loss")
        axes[0].plot(result.history.get("val_loss", []), label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{result.symbol} {result.model_type} - Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(result.history.get("train_acc", []), label="Train Acc")
        axes[1].plot(result.history.get("val_acc", []), label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{result.symbol} {result.model_type} - Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"{result.symbol}_{result.timeframe}_{result.model_type}_curves.png"
        plt.savefig(output_path / filename, dpi=150)
        plt.close()

    logger.info(f"Training curves saved to {output_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_summary_report(
    all_results: List[TrainingResult],
    output_dir: str = "data/training_reports",
) -> str:
    """
    Generate a summary report of all training runs.

    Args:
        all_results: List of all training results
        output_dir: Directory to save report

    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"training_report_{timestamp}.json"

    # Group results by symbol
    by_symbol = {}
    for result in all_results:
        if result.symbol not in by_symbol:
            by_symbol[result.symbol] = []
        by_symbol[result.symbol].append(result.to_dict())

    # Find best models
    best_models = []
    for symbol, results in by_symbol.items():
        if results:
            best = max(results, key=lambda x: x["val_accuracy"])
            best_models.append({
                "symbol": symbol,
                "best_model_type": best["model_type"],
                "best_val_accuracy": best["val_accuracy"],
                "best_test_accuracy": best["test_accuracy"],
                "hyperparams": {
                    "hidden_size": best["hidden_size"],
                    "learning_rate": best["learning_rate"],
                    "dropout": best["dropout"],
                },
            })

    report = {
        "timestamp": timestamp,
        "total_models_trained": len(all_results),
        "symbols_trained": list(by_symbol.keys()),
        "best_models": best_models,
        "all_results": [r.to_dict() for r in all_results],
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models trained: {len(all_results)}")
    logger.info(f"Symbols: {list(by_symbol.keys())}")
    logger.info("\nBest models per symbol:")
    for bm in best_models:
        logger.info(f"  {bm['symbol']}: {bm['best_model_type']} "
                   f"(val_acc={bm['best_val_accuracy']:.4f}, "
                   f"test_acc={bm['best_test_accuracy']:.4f})")
    logger.info(f"\nFull report saved to: {report_file}")
    logger.info("=" * 60)

    return str(report_file)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced model training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all models with full hyperparameter search
    python tools/train_models.py

    # Quick training with default hyperparameters
    python tools/train_models.py --fast

    # Train specific symbols
    python tools/train_models.py --symbols BTC,ETH

    # Train only LSTM models
    python tools/train_models.py --model-types lstm

    # Custom epochs
    python tools/train_models.py --epochs 50
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated list of symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )

    parser.add_argument(
        "--model-types",
        type=str,
        default=",".join(DEFAULT_MODEL_TYPES),
        help=f"Comma-separated list of model types (default: {','.join(DEFAULT_MODEL_TYPES)})",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick training with fewer hyperparameter combinations",
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Comma-separated list of timeframes (default: {','.join(DEFAULT_TIMEFRAMES)})",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="data/models",
        help="Directory to save models (default: data/models)",
    )

    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best models in the model registry",
    )

    args = parser.parse_args()

    # Parse arguments
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    model_types = [m.strip().lower() for m in args.model_types.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    hyperparams = FAST_HYPERPARAMS if args.fast else FULL_HYPERPARAMS
    augment = not args.no_augment

    # Create directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/training_reports").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENHANCED MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"Data augmentation: {augment}")
    logger.info(f"Hyperparameters: {hyperparams}")
    logger.info("=" * 60)

    all_results = []
    best_models_to_register = []

    # Train each symbol sequentially to avoid rate limits
    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*40}")

        for timeframe in timeframes:
            # Fetch data
            df = fetch_historical_data(symbol, timeframe=timeframe, years=2)

            if df is None or len(df) < 500:
                logger.warning(f"Insufficient data for {symbol} {timeframe}, skipping")
                continue

            # Prepare data
            try:
                train_data, val_data, test_data = prepare_training_data(
                    df,
                    sequence_length=60 if timeframe == "1h" else 30,
                    augment=augment,
                )
            except Exception as e:
                logger.error(f"Error preparing data for {symbol} {timeframe}: {e}")
                continue

            if len(train_data[0]) < 100:
                logger.warning(f"Not enough training samples for {symbol} {timeframe}")
                continue

            # Train each model type
            for model_type in model_types:
                logger.info(f"\nTraining {model_type} for {symbol} ({timeframe})")

                best_result, best_model, search_results = run_hyperparameter_search(
                    symbol=symbol,
                    timeframe=timeframe,
                    model_type=model_type,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    hyperparams=hyperparams,
                    epochs=args.epochs,
                    model_dir=args.model_dir,
                )

                if best_result is not None:
                    all_results.extend(search_results)

                    # Save best model
                    if best_model is not None:
                        model_path = best_model.save()
                        best_result.model_path = str(model_path)
                        best_models_to_register.append((best_model, symbol, timeframe))
                        logger.info(f"Best model saved to: {model_path}")

                # Rate limit between training runs
                time.sleep(1)

        # Rate limit between symbols
        time.sleep(RATE_LIMIT_DELAY)

    # Register best models if requested
    if args.register and best_models_to_register:
        logger.info("\nRegistering best models...")
        registry = ModelRegistry()

        for model, symbol, timeframe in best_models_to_register:
            try:
                key = registry.register_model(
                    model,
                    symbol=f"{symbol}/USDT",
                    market_type="crypto",
                    metadata={"timeframe": timeframe},
                )
                logger.info(f"Registered: {key}")
            except Exception as e:
                logger.error(f"Failed to register model for {symbol}: {e}")

    # Generate reports
    if all_results:
        # Save training curves
        best_results = []
        seen = set()
        for r in all_results:
            key = (r.symbol, r.timeframe, r.model_type)
            if key not in seen:
                best_for_key = max(
                    [x for x in all_results if (x.symbol, x.timeframe, x.model_type) == key],
                    key=lambda x: x.val_accuracy,
                )
                best_results.append(best_for_key)
                seen.add(key)

        save_training_curves(best_results)

        # Generate summary report
        report_path = generate_summary_report(all_results)
        logger.info(f"\nTraining complete! Report saved to: {report_path}")
    else:
        logger.warning("No models were trained successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
