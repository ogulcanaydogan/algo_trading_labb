"""
LSTM Model for Trading Signal Prediction.

Optimized for Apple Silicon (MPS) with bidirectional layers,
dropout regularization, and attention mechanism.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..base import BaseMLModel, ModelConfig, ModelPrediction, TrainingMetrics, get_optimal_device


@dataclass
class LSTMConfig(ModelConfig):
    """Configuration specific to LSTM models."""

    name: str = "lstm_model"
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    use_attention: bool = True
    attention_heads: int = 4

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer for LSTM outputs."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)


class LSTMNetwork(nn.Module):
    """
    Bidirectional LSTM with optional attention for time-series classification.

    Architecture:
    - Input normalization
    - Bidirectional LSTM layers
    - Optional attention mechanism
    - Fully connected classifier
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                lstm_output_size,
                num_heads=attention_heads,
                dropout=dropout,
            )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention over all timesteps
        if self.use_attention:
            lstm_out = self.attention(lstm_out)

        # Use last timestep output
        out = lstm_out[:, -1, :]

        # Classifier
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMModel(BaseMLModel):
    """
    LSTM-based model for trading signal prediction.

    Features:
    - Bidirectional LSTM layers
    - Optional attention mechanism
    - MPS (Apple Silicon) optimization
    - Early stopping and gradient clipping
    - Mixed precision training support

    Usage:
        model = LSTMModel(config=LSTMConfig(device="mps"))
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))
        prediction = model.predict(X_test)
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        model_dir: str = "data/models",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTM. Install with: pip install torch")

        # Use optimal device if not specified
        if config is None:
            config = LSTMConfig()
        if config.device == "cpu":
            config.device = get_optimal_device()

        super().__init__(config=config, model_dir=model_dir)

        self.config: LSTMConfig = config
        self._model_type = "lstm"
        self._model_name = config.name

        self.device = torch.device(config.device)
        self.network: Optional[LSTMNetwork] = None
        self.input_size: Optional[int] = None

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

        print(f"LSTM Model initialized on device: {self.device}")

    def _init_network(self, input_size: int) -> None:
        """Initialize the LSTM network with given input size."""
        self.input_size = input_size
        self.network = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_classes=3,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            use_attention=self.config.use_attention,
            attention_heads=self.config.attention_heads,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> TrainingMetrics:
        """
        Train the LSTM model.

        Args:
            X: Input features (samples, sequence_length, features)
            y: Target labels (samples,) - 0=SHORT, 1=FLAT, 2=LONG
            validation_data: Optional validation set

        Returns:
            TrainingMetrics with training results
        """
        start_time = time.time()

        # Ensure 3D input
        if X.ndim == 2:
            X = self.prepare_sequences(X)

        # Initialize network if needed
        if self.network is None:
            self._init_network(X.shape[2])

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim == 2:
                X_val = self.prepare_sequences(X_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        print(f"Training LSTM on {self.device} for {self.config.epochs} epochs...")

        for epoch in range(self.config.epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.gradient_clip,
                )

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.network.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.network(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint("best")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Loss: {avg_train_loss:.4f} - Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                )

        # Load best model
        self._load_checkpoint("best")

        training_time = time.time() - start_time
        self.is_trained = True

        self.training_metrics = TrainingMetrics(
            train_loss=history["train_loss"][-1] if history["train_loss"] else 0,
            val_loss=best_val_loss if val_loader else 0,
            train_accuracy=history["train_acc"][-1] if history["train_acc"] else 0,
            val_accuracy=history["val_acc"][best_epoch] if val_loader and history["val_acc"] else 0,
            epochs_trained=len(history["train_loss"]),
            best_epoch=best_epoch,
            training_time_seconds=training_time,
            samples_trained=len(X),
            history=history,
        )

        print(f"Training complete in {training_time:.1f}s. Best val loss: {best_val_loss:.4f}")

        return self.training_metrics

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Make prediction on input features.

        Args:
            X: Input features (can be 2D or 3D)

        Returns:
            ModelPrediction with action and probabilities
        """
        if not self.is_trained or self.network is None:
            return self.get_default_prediction()

        start_time = time.time()

        # Ensure 3D input
        if X.ndim == 2:
            X = self.prepare_sequences(X)

        # Get last sequence for prediction
        if len(X) > 1:
            X = X[-1:]

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        inference_time = (time.time() - start_time) * 1000

        # Probabilities: [SHORT, FLAT, LONG]
        prob_short, prob_flat, prob_long = probabilities

        # Determine action
        action_idx = np.argmax(probabilities)
        action_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        action = action_map[action_idx]
        confidence = float(probabilities[action_idx])

        return ModelPrediction(
            action=action,
            confidence=confidence,
            probability_long=float(prob_long),
            probability_short=float(prob_short),
            probability_flat=float(prob_flat),
            expected_return=0.0,  # Can be computed externally
            model_name=self.model_name,
            model_type=self.model_type,
            features_used=self.input_size or 0,
            sequence_length=self.config.sequence_length,
            inference_time_ms=inference_time,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for input.

        Args:
            X: Input features

        Returns:
            Array of shape (samples, 3) with [SHORT, FLAT, LONG] probabilities
        """
        if not self.is_trained or self.network is None:
            return np.array([[0.33, 0.34, 0.33]])

        if X.ndim == 2:
            X = self.prepare_sequences(X)

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def save(self, name: Optional[str] = None) -> Path:
        """Save model to disk."""
        name = name or self.model_name
        save_dir = self.model_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save network state
        if self.network is not None:
            torch.save(
                self.network.state_dict(),
                save_dir / "model.pt",
            )

        # Save optimizer state
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                save_dir / "optimizer.pt",
            )

        # Save scaler if exists
        if self.scaler is not None:
            with open(save_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        # Save metadata
        self._save_metadata(
            save_dir,
            extra={
                "input_size": self.input_size,
                "lstm_config": {
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                    "bidirectional": self.config.bidirectional,
                    "use_attention": self.config.use_attention,
                },
            },
        )

        print(f"LSTM model saved to {save_dir}")
        return save_dir

    def load(self, name: Optional[str] = None) -> bool:
        """Load model from disk."""
        name = name or self.model_name
        load_dir = self.model_dir / name

        if not load_dir.exists():
            print(f"Model directory not found: {load_dir}")
            return False

        # Load metadata
        metadata = self._load_metadata(load_dir)
        if metadata is None:
            return False

        self.input_size = metadata.get("input_size")
        self.feature_names = metadata.get("feature_names", [])
        self.is_trained = metadata.get("is_trained", False)

        # Initialize network
        if self.input_size:
            self._init_network(self.input_size)

            # Load network state
            model_path = load_dir / "model.pt"
            if model_path.exists():
                self.network.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )

        # Load scaler
        scaler_path = load_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        print(f"LSTM model loaded from {load_dir}")
        return True

    def _save_checkpoint(self, name: str) -> None:
        """Save a training checkpoint."""
        checkpoint_dir = self.model_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        torch.save(
            self.network.state_dict(),
            checkpoint_dir / f"{name}.pt",
        )

    def _load_checkpoint(self, name: str) -> bool:
        """Load a training checkpoint."""
        checkpoint_path = self.model_dir / "checkpoints" / f"{name}.pt"
        if checkpoint_path.exists() and self.network is not None:
            self.network.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            return True
        return False
