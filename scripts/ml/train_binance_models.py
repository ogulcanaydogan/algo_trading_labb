#!/usr/bin/env python3
"""
Train ML models using Binance data for symbols not available on Yahoo Finance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import time
import ccxt
import numpy as np
import pandas as pd

from bot.ml.feature_engineer import FeatureEngineer
from bot.ml.models.deep_learning.lstm import LSTMModel, LSTMConfig
from bot.ml.models.deep_learning.transformer import TransformerModel, TransformerConfig
from bot.ml.registry.model_registry import ModelRegistry

# Ensure directories exist
Path("data/model_registry/models").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Symbols to train (Binance format)
SYMBOLS_TO_TRAIN = [
    ("MATIC/USDT", "MATIC/USDT"),
    ("UNI/USDT", "UNI/USDT"),
]


def fetch_binance_data(symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch historical data from Binance."""
    logger.info(f"Fetching {limit} candles for {symbol} from Binance...")

    exchange = ccxt.binance()
    all_data = []

    # Fetch in batches (Binance limit is 1000 per request)
    since = None
    batch_size = 1000
    total_fetched = 0

    while total_fetched < limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(batch_size, limit - total_fetched))
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            total_fetched += len(ohlcv)
            since = ohlcv[-1][0] + 1  # Next timestamp
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break

    if not all_data:
        logger.error(f"No data for {symbol}")
        return None

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)

    logger.info(f"  Got {len(df)} candles")
    return df


def create_sequences(X, y, seq_len):
    """Create sequences for LSTM/Transformer."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def evaluate_model(model, X_test, y_test) -> float:
    """Evaluate model accuracy."""
    try:
        result = model.predict(X_test)
        if result is None:
            return 0.0

        # Handle ModelPrediction object
        if hasattr(result, 'predictions'):
            predictions = result.predictions
        elif hasattr(result, 'probabilities'):
            predictions = result.probabilities
        else:
            predictions = np.array(result)

        # Convert predictions to class labels
        if len(predictions.shape) > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = (predictions > 0.5).astype(int)

        accuracy = np.mean(pred_labels == y_test)
        return accuracy
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return 0.5  # Return baseline


def train_model(symbol: str, trading_symbol: str, epochs: int = 30):
    """Train LSTM and Transformer for a symbol."""
    results = {}
    model_dir = Path("data/model_registry/models")
    registry = ModelRegistry()

    # Fetch data
    df = fetch_binance_data(symbol, limit=5000)  # Get ~200 days of hourly data
    if df is None or len(df) < 500:
        logger.error(f"Insufficient data for {symbol}")
        return results

    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.extract_features(df)
    features_df = features_df.dropna()

    if len(features_df) < 300:
        logger.error(f"Insufficient features for {symbol}")
        return results

    # Create labels (1 if price goes up, 0 otherwise)
    features_df["target"] = (features_df["close"].shift(-1) > features_df["close"]).astype(int)
    features_df = features_df.dropna()

    # Split features and target
    feature_cols = [c for c in features_df.columns if c not in ["target", "open", "high", "low", "close", "volume"]]
    X = features_df[feature_cols].values
    y = features_df["target"].values

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    safe_symbol = trading_symbol.replace("/", "_")

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
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                epochs=epochs,
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=10,
            )
            lstm = LSTMModel(config=lstm_config, model_dir=model_dir)
            lstm.train(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq))

            acc = evaluate_model(lstm, X_test_seq, y_test_seq)
            lstm.save(f"crypto_{safe_symbol}_lstm")

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
            transformer.train(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq))

            acc = evaluate_model(transformer, X_test_seq, y_test_seq)
            transformer.save(f"crypto_{safe_symbol}_transformer")

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
    logger.info("=" * 60)
    logger.info("Training ML Models from Binance Data")
    logger.info("=" * 60)

    all_results = {}

    for binance_symbol, trading_symbol in SYMBOLS_TO_TRAIN:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Training models for {trading_symbol}")
        logger.info("=" * 60)

        results = train_model(binance_symbol, trading_symbol)
        all_results[trading_symbol] = results

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    for symbol, results in all_results.items():
        logger.info(f"\n{symbol}:")
        for model_type, result in results.items():
            if "accuracy" in result:
                logger.info(f"  {model_type}: {result['accuracy']:.2%}")
            else:
                logger.info(f"  {model_type}: FAILED - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
