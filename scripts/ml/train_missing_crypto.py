#!/usr/bin/env python3
"""
Train ML models for missing crypto symbols.
Trains LSTM and Transformer models for XRP, ADA, DOT, LINK.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from bot.ml.feature_engineer import FeatureEngineer
from bot.ml.models.deep_learning.lstm import LSTMModel, LSTMConfig
from bot.ml.models.deep_learning.transformer import TransformerModel, TransformerConfig
from bot.ml.registry.model_registry import ModelRegistry

# Ensure model registry directory exists
from pathlib import Path
Path("data/model_registry/models").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Missing symbols to train
SYMBOLS_TO_TRAIN = [
    ("XRP-USD", "XRP/USDT"),
    ("ADA-USD", "ADA/USDT"),
    ("DOT-USD", "DOT/USDT"),
    ("LINK-USD", "LINK/USDT"),
]


def fetch_data(yf_symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    logger.info(f"Fetching {days} days of hourly data for {yf_symbol}...")
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=f"{days}d", interval="1h")

    if df.empty:
        logger.error(f"No data for {yf_symbol}")
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    logger.info(f"  Got {len(df)} candles")
    return df


def prepare_training_data(df: pd.DataFrame, feature_engineer: FeatureEngineer):
    """Extract features and prepare training data."""
    df_features = feature_engineer.extract_features(df)

    if len(df_features) < 500:
        logger.warning(f"Insufficient data: {len(df_features)} samples")
        return None, None

    exclude_cols = ["target_return", "target_direction", "target_class",
                    "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X = df_features[feature_cols].values

    # Create 3-class target
    returns = df_features["close"].pct_change().shift(-1)
    threshold = returns.std() * 0.5
    y = np.where(returns > threshold, 2,
                 np.where(returns < -threshold, 0, 1))

    # Clean data
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def create_sequences(X, y, seq_len):
    """Create sequences for time-series models."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy on test set."""
    correct = 0
    total = len(X_test)
    action_map = {"SHORT": 0, "FLAT": 1, "LONG": 2}

    for i in range(total):
        pred = model.predict(X_test[i:i+1])
        pred_class = action_map.get(pred.action, 1)
        if pred_class == y_test[i]:
            correct += 1

    return correct / total if total > 0 else 0.0


def train_model(symbol_pair: tuple, feature_engineer: FeatureEngineer,
                registry: ModelRegistry, epochs: int = 30):
    """Train LSTM and Transformer models for a symbol."""
    yf_symbol, trading_symbol = symbol_pair
    safe_symbol = trading_symbol.replace("/", "_")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training models for {trading_symbol}")
    logger.info("=" * 60)

    # Fetch data
    df = fetch_data(yf_symbol)
    if df is None:
        return None

    # Prepare data
    X, y = prepare_training_data(df, feature_engineer)
    if X is None:
        return None

    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    results = {}
    model_dir = "data/model_registry/models"

    # Train LSTM
    try:
        logger.info("Training LSTM...")
        seq_len = 60
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

        if len(X_train_seq) < 100:
            logger.warning("Insufficient sequences for LSTM")
        else:
            lstm_config = LSTMConfig(
                name=f"crypto_{safe_symbol}_lstm",
                sequence_length=seq_len,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                bidirectional=True,
                use_attention=True,
                epochs=epochs,
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=10,
            )
            lstm = LSTMModel(config=lstm_config, model_dir=model_dir)

            # Train with validation data
            lstm.train(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq))

            # Evaluate
            acc = evaluate_model(lstm, X_test_seq, y_test_seq)

            # Save model
            lstm.save(f"crypto_{safe_symbol}_lstm")

            # Register
            registry.register_model(
                model=lstm,
                symbol=trading_symbol,
                market_type="crypto",
                metadata={"model_type": "lstm", "accuracy": acc},
            )

            results["lstm"] = {"accuracy": acc}
            logger.info(f"LSTM trained - accuracy: {acc:.2%}")

    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        results["lstm"] = {"error": str(e)}

    # Train Transformer
    try:
        logger.info("Training Transformer...")
        seq_len = 120
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

        if len(X_train_seq) < 100:
            logger.warning("Insufficient sequences for Transformer")
        else:
            transformer_config = TransformerConfig(
                name=f"crypto_{safe_symbol}_transformer",
                sequence_length=seq_len,
                model_dim=64,
                num_heads=4,
                num_encoder_layers=3,
                dim_feedforward=256,
                dropout=0.1,
                epochs=epochs,
                batch_size=32,
                learning_rate=0.0001,
                early_stopping_patience=10,
            )
            transformer = TransformerModel(config=transformer_config, model_dir=model_dir)

            # Train with validation data
            transformer.train(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq))

            # Evaluate
            acc = evaluate_model(transformer, X_test_seq, y_test_seq)

            # Save model
            transformer.save(f"crypto_{safe_symbol}_transformer")

            # Register
            registry.register_model(
                model=transformer,
                symbol=trading_symbol,
                market_type="crypto",
                metadata={"model_type": "transformer", "accuracy": acc},
            )

            results["transformer"] = {"accuracy": acc}
            logger.info(f"Transformer trained - accuracy: {acc:.2%}")

    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        results["transformer"] = {"error": str(e)}

    return results


def main():
    """Train all missing models."""
    logger.info("=" * 60)
    logger.info("Training Missing Crypto ML Models")
    logger.info("=" * 60)

    feature_engineer = FeatureEngineer()
    registry = ModelRegistry(registry_dir="data/model_registry")

    all_results = {}

    for symbol_pair in SYMBOLS_TO_TRAIN:
        results = train_model(symbol_pair, feature_engineer, registry, epochs=30)
        all_results[symbol_pair[1]] = results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        if results:
            lstm_acc = results.get("lstm", {}).get("accuracy", "failed")
            trans_acc = results.get("transformer", {}).get("accuracy", "failed")
            logger.info(f"{symbol}: LSTM={lstm_acc}, Transformer={trans_acc}")
        else:
            logger.info(f"{symbol}: FAILED")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
