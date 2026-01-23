"""
LSTM Neural Network Model for Price Prediction.

Deep learning model using LSTM layers for sequence prediction.
"""

import logging
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not installed. LSTM model unavailable.")


class LSTMPredictor:
    """
    LSTM-based price direction predictor.

    Architecture:
    - 2 LSTM layers with dropout
    - Dense layers for classification
    - Binary output: UP (1) or DOWN (0)
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 5,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional[Any] = None
        self.scaler_params: Dict[str, Any] = {}
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for LSTM model")

        self.model = Sequential(
            [
                # First LSTM layer
                LSTM(
                    self.lstm_units[0],
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.n_features),
                ),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                # Second LSTM layer
                LSTM(self.lstm_units[1], return_sequences=False),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                # Dense layers
                Dense(32, activation="relu"),
                Dropout(self.dropout_rate / 2),
                Dense(16, activation="relu"),
                # Output layer
                Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"LSTM model built: {self.model.count_params()} parameters")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.

        Args:
            df: DataFrame with OHLCV data
            target_col: Name of target column

        Returns:
            Tuple of (X sequences, y targets)
        """
        # Feature columns
        feature_cols = ["open", "high", "low", "close", "volume"]

        # Normalize features
        features = df[feature_cols].copy()

        # Store scaler params
        self.scaler_params = {
            "mean": features.mean().to_dict(),
            "std": features.std().to_dict(),
        }

        # Normalize
        for col in feature_cols:
            std = self.scaler_params["std"][col]
            if std > 0:
                features[col] = (features[col] - self.scaler_params["mean"][col]) / std

        # Create sequences
        X, y = [], []
        values = features.values
        targets = df[target_col].values if target_col in df.columns else None

        for i in range(self.sequence_length, len(values)):
            X.append(values[i - self.sequence_length : i])
            if targets is not None:
                y.append(targets[i])

        return np.array(X), np.array(y) if targets is not None else None

    def create_targets(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """Create binary targets (1 if price goes up, 0 if down)."""
        df = df.copy()
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
        df["target"] = (df["future_return"] > 0).astype(int)
        return df.dropna()

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            df: DataFrame with OHLCV data
            epochs: Maximum training epochs
            batch_size: Batch size
            validation_split: Validation data fraction
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history and metrics
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for training")

        if self.model is None:
            self.build_model()

        # Create targets
        df = self.create_targets(df)

        # Prepare data
        X, y = self.prepare_data(df)

        if len(X) < 100:
            raise ValueError(f"Not enough data: {len(X)} samples")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        # Train
        logger.info(f"Training LSTM on {len(X)} samples...")

        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        val_idx = int(len(X) * (1 - validation_split))
        X_val, y_val = X[val_idx:], y[val_idx:]
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)

        self.metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "epochs_trained": len(history.history["loss"]),
            "samples": len(X),
        }

        self.is_trained = True
        logger.info(f"Training complete: accuracy={accuracy:.4f}, loss={loss:.4f}")

        return {
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "metrics": self.metrics,
        }

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction on new data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Prediction dict with probability and direction
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")

        # Prepare data (without targets)
        X, _ = self.prepare_data(df)

        if len(X) == 0:
            raise ValueError("Not enough data for prediction")

        # Get last sequence
        X_last = (
            X[-1:] if len(X.shape) == 3 else X[-1].reshape(1, self.sequence_length, self.n_features)
        )

        # Predict
        prob_up = float(self.model.predict(X_last, verbose=0)[0][0])
        prob_down = 1 - prob_up

        direction = "LONG" if prob_up > 0.5 else "SHORT"
        confidence = max(prob_up, prob_down)

        return {
            "direction": direction,
            "probability_long": prob_up,
            "probability_short": prob_down,
            "confidence": confidence,
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = path / "lstm_model.keras"
        self.model.save(model_path)

        # Save metadata
        meta = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "scaler_params": self.scaler_params,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
        }

        meta_path = path / "lstm_meta.pkl"
        joblib.dump(meta, meta_path)

        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available, cannot load LSTM model")
            return False

        path = Path(path)

        model_path = path / "lstm_model.keras"
        meta_path = path / "lstm_meta.pkl"

        if not model_path.exists() or not meta_path.exists():
            return False

        try:
            # Load metadata
            meta = joblib.load(meta_path)

            self.sequence_length = meta["sequence_length"]
            self.n_features = meta["n_features"]
            self.lstm_units = meta["lstm_units"]
            self.dropout_rate = meta["dropout_rate"]
            self.scaler_params = meta["scaler_params"]
            self.metrics = meta["metrics"]
            self.is_trained = meta["is_trained"]

            # Load model
            self.model = load_model(model_path)

            logger.info(f"LSTM model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False


def train_lstm_model(
    symbol: str,
    df: pd.DataFrame,
    save_path: str = "data/models",
    epochs: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function to train LSTM model for a symbol.

    Args:
        symbol: Trading symbol
        df: OHLCV DataFrame
        save_path: Directory to save model
        epochs: Training epochs

    Returns:
        Training results
    """
    if not HAS_TENSORFLOW:
        return {"error": "TensorFlow not installed"}

    predictor = LSTMPredictor()

    try:
        results = predictor.train(df, epochs=epochs)

        # Save model
        symbol_path = symbol.replace("/", "_")
        predictor.save(f"{save_path}/{symbol_path}_lstm")

        return {
            "symbol": symbol,
            "status": "success",
            "metrics": results["metrics"],
        }

    except Exception as e:
        logger.error(f"LSTM training failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(e),
        }
