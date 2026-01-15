#!/usr/bin/env python3
"""
Deep Learning Model Training Script.

Trains LSTM and Transformer models for different market types:
- Commodities (Gold, Silver, Oil)
- Stocks (AAPL, MSFT, GOOGL)
- Crypto (BTC, ETH)

Optimized for Apple Silicon (MPS) with automatic device detection.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Market configurations
MARKET_CONFIGS = {
    "commodity": {
        "symbols": {
            "GC=F": "Gold",
            "SI=F": "Silver",
            "CL=F": "Oil WTI",
        },
        "lookback_days": 365 * 3,  # 3 years of data
        "model_dir": "data/models/commodity",
    },
    "stock": {
        "symbols": {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
        },
        "lookback_days": 365 * 3,
        "model_dir": "data/models/stock",
    },
    "crypto": {
        "symbols": {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "SOL-USD": "Solana",
            "AVAX-USD": "Avalanche",
            "XRP-USD": "Ripple",
            "ADA-USD": "Cardano",
            "DOT-USD": "Polkadot",
            "LINK-USD": "Chainlink",
        },
        "lookback_days": 365 * 2,
        "model_dir": "data/models/crypto",
    },
}


def fetch_historical_data(
    symbol: str,
    days: int = 365,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        symbol: Yahoo Finance symbol
        days: Number of days of history
        interval: Data interval ('1d', '1h', etc.)

    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Normalize column names
        df.columns = df.columns.str.lower()
        df = df.rename(columns={"adj close": "adj_close"})

        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            logger.error(f"Missing required columns for {symbol}")
            return None

        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df[required]

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 60,
    validation_split: float = 0.2,
    test_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for model training.

    Args:
        df: OHLCV DataFrame
        sequence_length: Length of input sequences
        validation_split: Fraction for validation
        test_split: Fraction for testing

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    from bot.ml.feature_engineer import FeatureEngineer

    # Extract features
    logger.info("Extracting features...")
    fe = FeatureEngineer()
    features_df = fe.extract_features(df)

    if len(features_df) < sequence_length + 100:
        logger.warning(f"Insufficient data after feature extraction: {len(features_df)} rows")
        return None

    # Get feature columns (exclude targets and OHLCV)
    exclude_cols = [
        "open", "high", "low", "close", "volume",
        "target_return", "target_direction", "target_class",
    ]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Prepare features and targets
    X = features_df[feature_cols].values
    y = features_df["target_class"].values.astype(int)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    logger.info(f"Creating sequences of length {sequence_length}...")
    X_seq = []
    y_seq = []

    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Time-based split (preserve temporal order)
    n_samples = len(X_seq)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_test - n_val

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Log class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Training class distribution: {dict(zip(unique, counts))}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "lstm_model",
    model_dir: str = "data/models",
    epochs: int = 50,
) -> Dict:
    """Train LSTM model."""
    from bot.ml.models.deep_learning.lstm import LSTMModel, LSTMConfig

    logger.info(f"Training LSTM model: {model_name}")

    config = LSTMConfig(
        name=model_name,
        sequence_length=X_train.shape[1],
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=15,
    )

    model = LSTMModel(config=config, model_dir=model_dir)

    # Train
    start_time = time.time()
    metrics = model.train(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
    )
    training_time = time.time() - start_time

    # Save model
    model.save(model_name)

    result = {
        "model_name": model_name,
        "model_type": "lstm",
        "train_accuracy": metrics.train_accuracy,
        "val_accuracy": metrics.val_accuracy,
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "epochs_trained": metrics.epochs_trained,
        "training_time_seconds": training_time,
    }

    logger.info(f"LSTM training complete - Val accuracy: {metrics.val_accuracy:.4f}")
    return result


def train_transformer_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "transformer_model",
    model_dir: str = "data/models",
    epochs: int = 50,
) -> Dict:
    """Train Transformer model."""
    from bot.ml.models.deep_learning.transformer import TransformerModel, TransformerConfig

    logger.info(f"Training Transformer model: {model_name}")

    config = TransformerConfig(
        name=model_name,
        sequence_length=X_train.shape[1],
        model_dim=64,
        num_heads=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.0001,
        early_stopping_patience=15,
    )

    model = TransformerModel(config=config, model_dir=model_dir)

    # Train
    start_time = time.time()
    metrics = model.train(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
    )
    training_time = time.time() - start_time

    # Save model
    model.save(model_name)

    result = {
        "model_name": model_name,
        "model_type": "transformer",
        "train_accuracy": metrics.train_accuracy,
        "val_accuracy": metrics.val_accuracy,
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "epochs_trained": metrics.epochs_trained,
        "training_time_seconds": training_time,
    }

    logger.info(f"Transformer training complete - Val accuracy: {metrics.val_accuracy:.4f}")
    return result


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Evaluate model on test set."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Get predictions
    predictions = []
    for i in range(len(X_test)):
        pred = model.predict(X_test[i:i + 1])
        action_map = {"SHORT": 0, "FLAT": 1, "LONG": 2}
        predictions.append(action_map[pred.action])

    predictions = np.array(predictions)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["SHORT", "FLAT", "LONG"], output_dict=True)

    return {
        "test_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def train_market_models(
    market_type: str,
    epochs: int = 50,
    sequence_length: int = 60,
    skip_existing: bool = False,
) -> List[Dict]:
    """
    Train models for a specific market type.

    Args:
        market_type: 'commodity', 'stock', or 'crypto'
        epochs: Number of training epochs
        sequence_length: Length of input sequences
        skip_existing: Skip if models already exist

    Returns:
        List of training results
    """
    if market_type not in MARKET_CONFIGS:
        raise ValueError(f"Unknown market type: {market_type}")

    config = MARKET_CONFIGS[market_type]
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for symbol, name in config["symbols"].items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training models for {name} ({symbol})")
        logger.info("=" * 60)

        # Check if models already exist
        lstm_name = f"lstm_{market_type}_{symbol.replace('=', '_').replace('-', '_')}"
        transformer_name = f"transformer_{market_type}_{symbol.replace('=', '_').replace('-', '_')}"

        if skip_existing:
            lstm_path = model_dir / lstm_name / "model.pt"
            transformer_path = model_dir / transformer_name / "model.pt"
            if lstm_path.exists() and transformer_path.exists():
                logger.info(f"Skipping {symbol} - models already exist")
                continue

        # Fetch data
        df = fetch_historical_data(symbol, days=config["lookback_days"])
        if df is None:
            logger.error(f"Failed to fetch data for {symbol}")
            continue

        # Prepare data
        data = prepare_training_data(df, sequence_length=sequence_length)
        if data is None:
            logger.error(f"Failed to prepare data for {symbol}")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = data

        # Train LSTM
        try:
            lstm_result = train_lstm_model(
                X_train, y_train, X_val, y_val,
                model_name=lstm_name,
                model_dir=str(model_dir),
                epochs=epochs,
            )
            lstm_result["symbol"] = symbol
            lstm_result["market_type"] = market_type
            results.append(lstm_result)
        except Exception as e:
            logger.error(f"LSTM training failed for {symbol}: {e}")

        # Train Transformer
        try:
            transformer_result = train_transformer_model(
                X_train, y_train, X_val, y_val,
                model_name=transformer_name,
                model_dir=str(model_dir),
                epochs=epochs,
            )
            transformer_result["symbol"] = symbol
            transformer_result["market_type"] = market_type
            results.append(transformer_result)
        except Exception as e:
            logger.error(f"Transformer training failed for {symbol}: {e}")

        # Small delay between symbols to respect rate limits
        time.sleep(5)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train deep learning models for trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["commodity", "stock", "crypto", "all"],
        default=["all"],
        help="Market types to train models for",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Length of input sequences",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if models already exist",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_results.json",
        help="Output file for training results",
    )

    args = parser.parse_args()

    # Determine which markets to train
    if "all" in args.markets:
        markets = ["commodity", "stock", "crypto"]
    else:
        markets = args.markets

    logger.info(f"Training models for markets: {markets}")
    logger.info(f"Epochs: {args.epochs}, Sequence length: {args.sequence_length}")

    # Check PyTorch availability
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"PyTorch device: {device}")
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return

    # Train models for each market
    all_results = []
    training_start = time.time()

    for market_type in markets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"TRAINING MODELS FOR: {market_type.upper()}")
        logger.info("#" * 60)

        results = train_market_models(
            market_type,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            skip_existing=args.skip_existing,
        )
        all_results.extend(results)

    total_time = time.time() - training_start

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": total_time,
        "markets_trained": markets,
        "epochs": args.epochs,
        "sequence_length": args.sequence_length,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Models trained: {len(all_results)}")
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    if all_results:
        logger.info("\nModel Performance Summary:")
        logger.info("-" * 40)
        for r in all_results:
            logger.info(
                f"{r['model_type']:12} {r['symbol']:10} "
                f"Val Acc: {r['val_accuracy']:.4f} "
                f"Time: {r['training_time_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()
