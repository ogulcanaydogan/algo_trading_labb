"""
Transformer Model for Trading Signal Prediction.

Encoder-only transformer architecture optimized for time-series classification.
Includes positional encoding and MPS optimization for Apple Silicon.
"""

from __future__ import annotations

import math
import joblib
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
class TransformerConfig(ModelConfig):
    """Configuration specific to Transformer models."""

    name: str = "transformer_model"
    sequence_length: int = 120
    model_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.0001  # Lower LR for transformers
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    warmup_steps: int = 1000
    gradient_clip: float = 1.0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    """
    Encoder-only Transformer for time-series classification.

    Architecture:
    - Input projection to model dimension
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling
    - Classification head
    """

    def __init__(
        self,
        input_size: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.model_dim = model_dim

        # Input projection
        self.input_projection = nn.Linear(input_size, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

        # CLS token for classification (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)

        # Project input to model dimension
        x = self.input_projection(x)
        x = self.input_norm(x)

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use CLS token for classification
        cls_output = x[:, 0, :]

        # Classification
        logits = self.classifier(cls_output)

        return logits


class TransformerModel(BaseMLModel):
    """
    Transformer-based model for trading signal prediction.

    Features:
    - Encoder-only architecture optimized for time-series
    - Positional encoding for temporal awareness
    - CLS token for sequence classification
    - MPS (Apple Silicon) optimization
    - Warmup learning rate scheduling

    Best for:
    - Volatile markets with complex patterns
    - Longer sequence dependencies (120+ timesteps)
    - When attention patterns are important

    Usage:
        model = TransformerModel(config=TransformerConfig(device="mps"))
        metrics = model.train(X_train, y_train, validation_data=(X_val, y_val))
        prediction = model.predict(X_test)
    """

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        model_dir: str = "data/models",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Transformer. Install with: pip install torch")

        if config is None:
            config = TransformerConfig()
        if config.device == "cpu":
            config.device = get_optimal_device()

        super().__init__(config=config, model_dir=model_dir)

        self.config: TransformerConfig = config
        self._model_type = "transformer"
        self._model_name = config.name

        self.device = torch.device(config.device)
        self.network: Optional[TransformerNetwork] = None
        self.input_size: Optional[int] = None

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

        print(f"Transformer Model initialized on device: {self.device}")

    def _init_network(self, input_size: int) -> None:
        """Initialize the Transformer network."""
        self.input_size = input_size
        self.network = TransformerNetwork(
            input_size=input_size,
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_encoder_layers=self.config.num_encoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            num_classes=3,
            max_seq_len=self.config.sequence_length + 10,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
        )

        # Warmup + cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate * 10,
            total_steps=self.config.epochs * 100,  # Approximate
            pct_start=0.1,
            anneal_strategy="cos",
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> TrainingMetrics:
        """
        Train the Transformer model.

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
            X = self.prepare_sequences(X, self.config.sequence_length)

        # Initialize network
        if self.network is None:
            self._init_network(X.shape[2])

        # Re-initialize scheduler with correct total steps
        total_steps = self.config.epochs * (len(X) // self.config.batch_size + 1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

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
                X_val = self.prepare_sequences(X_val, self.config.sequence_length)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        print(f"Training Transformer on {self.device} for {self.config.epochs} epochs...")

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
                self.scheduler.step()

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

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
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
        """Make prediction on input features."""
        if not self.is_trained or self.network is None:
            return self.get_default_prediction()

        start_time = time.time()

        # Ensure 3D input
        if X.ndim == 2:
            X = self.prepare_sequences(X, self.config.sequence_length)

        # Get last sequence
        if len(X) > 1:
            X = X[-1:]

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        inference_time = (time.time() - start_time) * 1000

        prob_short, prob_flat, prob_long = probabilities
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
            expected_return=0.0,
            model_name=self.model_name,
            model_type=self.model_type,
            features_used=self.input_size or 0,
            sequence_length=self.config.sequence_length,
            inference_time_ms=inference_time,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_trained or self.network is None:
            return np.array([[0.33, 0.34, 0.33]])

        if X.ndim == 2:
            X = self.prepare_sequences(X, self.config.sequence_length)

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

        if self.network is not None:
            torch.save(self.network.state_dict(), save_dir / "model.pt")

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")

        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.pkl")

        self._save_metadata(
            save_dir,
            extra={
                "input_size": self.input_size,
                "transformer_config": {
                    "model_dim": self.config.model_dim,
                    "num_heads": self.config.num_heads,
                    "num_encoder_layers": self.config.num_encoder_layers,
                },
            },
        )

        print(f"Transformer model saved to {save_dir}")
        return save_dir

    def load(self, name: Optional[str] = None) -> bool:
        """Load model from disk."""
        name = name or self.model_name
        load_dir = self.model_dir / name

        if not load_dir.exists():
            print(f"Model directory not found: {load_dir}")
            return False

        metadata = self._load_metadata(load_dir)
        if metadata is None:
            return False

        self.input_size = metadata.get("input_size")
        self.feature_names = metadata.get("feature_names", [])
        self.is_trained = metadata.get("is_trained", False)

        if self.input_size:
            self._init_network(self.input_size)
            model_path = load_dir / "model.pt"
            if model_path.exists():
                self.network.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )

        scaler_path = load_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        print(f"Transformer model loaded from {load_dir}")
        return True

    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.model_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_dir / f"{name}.pt")

    def _load_checkpoint(self, name: str) -> bool:
        """Load training checkpoint."""
        checkpoint_path = self.model_dir / "checkpoints" / f"{name}.pt"
        if checkpoint_path.exists() and self.network is not None:
            self.network.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            return True
        return False

    def get_attention_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract attention weights for interpretability.

        Returns attention patterns showing which timesteps influence predictions.
        """
        if not self.is_trained or self.network is None:
            return None

        if X.ndim == 2:
            X = self.prepare_sequences(X, self.config.sequence_length)

        # Note: Extracting attention weights requires hooks
        # This is a placeholder for future implementation
        return None
