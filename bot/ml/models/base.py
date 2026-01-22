"""
Base ML Model Interface.

Provides a unified interface for all ML models (ensemble and deep learning).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # Model identification
    name: str = "base_model"
    version: str = "1.0.0"

    # Training parameters
    sequence_length: int = 60
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001

    # Architecture parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # Device settings
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    use_mixed_precision: bool = False

    # Training behavior
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "device": self.device,
            "use_mixed_precision": self.use_mixed_precision,
            "early_stopping_patience": self.early_stopping_patience,
            "validation_split": self.validation_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelPrediction:
    """Unified prediction result from any ML model."""

    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    probability_long: float
    probability_short: float
    probability_flat: float
    expected_return: float
    model_name: str
    model_type: str
    features_used: int = 0
    sequence_length: int = 0
    inference_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "probability_long": round(self.probability_long, 4),
            "probability_short": round(self.probability_short, 4),
            "probability_flat": round(self.probability_flat, 4),
            "expected_return": round(self.expected_return, 6),
            "model_name": self.model_name,
            "model_type": self.model_type,
            "features_used": self.features_used,
            "sequence_length": self.sequence_length,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    epochs_trained: int
    best_epoch: int
    training_time_seconds: float
    samples_trained: int
    feature_importance: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "train_accuracy": round(self.train_accuracy, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "samples_trained": self.samples_trained,
            "feature_importance": self.feature_importance,
        }


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.

    Provides a unified interface for:
    - Ensemble models (XGBoost, RandomForest)
    - Deep learning models (LSTM, Transformer)

    All models should implement this interface for seamless integration
    with the model selector and trading engine.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_dir: str = "data/models",
    ):
        self.config = config or ModelConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self.training_metrics: Optional[TrainingMetrics] = None

        self._model_type = "base"
        self._model_name = self.config.name

    @property
    def model_type(self) -> str:
        """Return the type of model (e.g., 'lstm', 'transformer', 'xgboost')."""
        return self._model_type

    @property
    def model_name(self) -> str:
        """Return the name of this model instance."""
        return self._model_name

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> TrainingMetrics:
        """
        Train the model on prepared data.

        Args:
            X: Input features (samples, features) or (samples, sequence, features)
            y: Target labels (samples,) - 0=SHORT, 1=FLAT, 2=LONG
            validation_data: Optional validation set (X_val, y_val)

        Returns:
            TrainingMetrics with training results
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        Make prediction on prepared features.

        Args:
            X: Input features for prediction

        Returns:
            ModelPrediction with action and probabilities
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.

        Args:
            X: Input features

        Returns:
            Array of shape (samples, 3) with probabilities for [SHORT, FLAT, LONG]
        """
        pass

    @abstractmethod
    def save(self, name: Optional[str] = None) -> Path:
        """
        Save model to disk.

        Args:
            name: Optional name for saved model

        Returns:
            Path to saved model directory
        """
        pass

    @abstractmethod
    def load(self, name: Optional[str] = None) -> bool:
        """
        Load model from disk.

        Args:
            name: Optional name of saved model

        Returns:
            True if loaded successfully
        """
        pass

    def prepare_sequences(
        self,
        features: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> np.ndarray:
        """
        Prepare sequential data for time-series models.

        Args:
            features: 2D array (samples, features)
            sequence_length: Length of sequences to create

        Returns:
            3D array (samples, sequence_length, features)
        """
        seq_len = sequence_length or self.config.sequence_length

        if len(features) < seq_len:
            # Pad with zeros if not enough data
            padding = np.zeros((seq_len - len(features), features.shape[1]))
            features = np.vstack([padding, features])

        sequences = []
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])

        return np.array(sequences)

    def get_default_prediction(self) -> ModelPrediction:
        """Return default prediction when model can't predict."""
        return ModelPrediction(
            action="FLAT",
            confidence=0.33,
            probability_long=0.33,
            probability_short=0.33,
            probability_flat=0.34,
            expected_return=0.0,
            model_name=self.model_name,
            model_type=self.model_type,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "config": self.config.to_dict(),
            "feature_count": len(self.feature_names),
            "training_metrics": self.training_metrics.to_dict() if self.training_metrics else None,
        }

    def _save_metadata(self, path: Path, extra: Optional[Dict] = None) -> None:
        """Save model metadata to JSON."""
        metadata = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics.to_dict() if self.training_metrics else None,
            "saved_at": datetime.now().isoformat(),
        }
        if extra:
            metadata.update(extra)

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, path: Path) -> Optional[Dict]:
        """Load model metadata from JSON."""
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            return json.load(f)


def get_optimal_device() -> str:
    """
    Detect the optimal device for PyTorch models.

    Returns:
        'mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' otherwise
    """
    import sys

    try:
        import torch

        # Skip MPS on Python 3.14+ due to segfault in torch.backends.mps.is_available()
        # This is a known PyTorch incompatibility with Python 3.14
        if sys.version_info >= (3, 14):
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"
