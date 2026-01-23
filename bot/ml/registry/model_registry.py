"""
Model Registry for managing and loading ML models.

Provides centralized storage, versioning, and loading of trained models.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from ..models.base import BaseMLModel, ModelConfig


@dataclass
class RegisteredModel:
    """Metadata for a registered model."""

    name: str
    model_type: Literal["lstm", "transformer", "xgboost", "random_forest"]
    version: str
    market_type: str
    symbol: str
    created_at: datetime
    accuracy: float
    val_accuracy: float
    path: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "accuracy": self.accuracy,
            "val_accuracy": self.val_accuracy,
            "path": self.path,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelRegistry:
    """
    Centralized registry for ML models.

    Features:
    - Model registration and versioning
    - Symbol and market type organization
    - Model loading and instantiation
    - Best model selection by accuracy

    Usage:
        registry = ModelRegistry()
        registry.register_model(trained_model, symbol="BTC/USDT", market_type="crypto")
        model = registry.load_model("BTC/USDT", model_type="lstm")
    """

    # Model type to class mapping
    MODEL_CLASSES: Dict[str, Type[BaseMLModel]] = {}

    def __init__(self, registry_dir: str = "data/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.registered_models: Dict[str, RegisteredModel] = {}

        self._load_registry()
        self._register_model_classes()

    def _register_model_classes(self) -> None:
        """Register available model classes."""
        try:
            from ..models.deep_learning.lstm import LSTMModel

            self.MODEL_CLASSES["lstm"] = LSTMModel
        except ImportError:
            pass

        try:
            from ..models.deep_learning.transformer import TransformerModel

            self.MODEL_CLASSES["transformer"] = TransformerModel
        except ImportError:
            pass

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                data = json.load(f)
                for key, model_data in data.items():
                    self.registered_models[key] = RegisteredModel.from_dict(model_data)

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {key: model.to_dict() for key, model in self.registered_models.items()}
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_key(
        self,
        symbol: str,
        model_type: str,
        market_type: str,
        version: str,
    ) -> str:
        """Generate unique key for a model."""
        symbol_clean = symbol.replace("/", "_")
        return f"{market_type}_{symbol_clean}_{model_type}_{version}"

    def register_model(
        self,
        model: BaseMLModel,
        symbol: str,
        market_type: str = "crypto",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a trained model in the registry.

        Args:
            model: Trained model instance
            symbol: Trading symbol (e.g., "BTC/USDT")
            market_type: Market type (crypto, commodity, stock)
            version: Optional version string
            metadata: Additional metadata

        Returns:
            Registry key for the model
        """
        if not model.is_trained:
            raise ValueError("Cannot register untrained model")

        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        key = self._generate_key(symbol, model.model_type, market_type, version)

        # Create model directory
        model_path = self.models_dir / key
        model_path.mkdir(exist_ok=True)

        # Save model
        model.save(name=str(model_path / "model"))

        # Get metrics
        accuracy = model.training_metrics.train_accuracy if model.training_metrics else 0.0
        val_accuracy = model.training_metrics.val_accuracy if model.training_metrics else 0.0

        # Register
        registered = RegisteredModel(
            name=model.model_name,
            model_type=model.model_type,
            version=version,
            market_type=market_type,
            symbol=symbol,
            created_at=datetime.now(),
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            path=str(model_path),
            metadata=metadata or {},
        )

        # Deactivate previous versions of same type for this symbol
        for existing_key, existing_model in self.registered_models.items():
            if (
                existing_model.symbol == symbol
                and existing_model.model_type == model.model_type
                and existing_model.market_type == market_type
            ):
                existing_model.is_active = False

        self.registered_models[key] = registered
        self._save_registry()

        print(f"Registered model: {key}")
        return key

    def load_model(
        self,
        symbol: str,
        model_type: str,
        market_type: str = "crypto",
        version: Optional[str] = None,
    ) -> Optional[BaseMLModel]:
        """
        Load a model from the registry.

        Args:
            symbol: Trading symbol
            model_type: Type of model to load
            market_type: Market type
            version: Specific version (loads latest active if None)

        Returns:
            Loaded model instance or None
        """
        # Find matching model
        candidates = [
            (key, model)
            for key, model in self.registered_models.items()
            if model.symbol == symbol
            and model.model_type == model_type
            and model.market_type == market_type
        ]

        if not candidates:
            print(f"No registered model found for {symbol} ({model_type})")
            return None

        if version:
            # Load specific version
            matching = [(k, m) for k, m in candidates if m.version == version]
            if not matching:
                print(f"Version {version} not found")
                return None
            key, registered = matching[0]
        else:
            # Load latest active model
            active = [(k, m) for k, m in candidates if m.is_active]
            if active:
                key, registered = max(active, key=lambda x: x[1].created_at)
            else:
                # Fall back to latest inactive
                key, registered = max(candidates, key=lambda x: x[1].created_at)

        # Instantiate and load model
        model_class = self.MODEL_CLASSES.get(model_type)
        if model_class is None:
            print(f"Unknown model type: {model_type}")
            return None

        model = model_class(model_dir=registered.path)
        if model.load(name=str(Path(registered.path) / "model")):
            return model

        return None

    def get_best_model(
        self,
        symbol: str,
        market_type: str = "crypto",
        metric: str = "val_accuracy",
    ) -> Optional[RegisteredModel]:
        """
        Get the best performing model for a symbol.

        Args:
            symbol: Trading symbol
            market_type: Market type
            metric: Metric to compare (val_accuracy or accuracy)

        Returns:
            Best RegisteredModel or None
        """
        candidates = [
            model
            for model in self.registered_models.values()
            if model.symbol == symbol and model.market_type == market_type and model.is_active
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda m: getattr(m, metric, 0))

    def list_models(
        self,
        symbol: Optional[str] = None,
        market_type: Optional[str] = None,
        model_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[RegisteredModel]:
        """
        List registered models with optional filters.

        Returns:
            List of matching RegisteredModel instances
        """
        models = list(self.registered_models.values())

        if symbol:
            models = [m for m in models if m.symbol == symbol]
        if market_type:
            models = [m for m in models if m.market_type == market_type]
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        if active_only:
            models = [m for m in models if m.is_active]

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def delete_model(self, key: str, keep_files: bool = False) -> bool:
        """
        Delete a model from the registry.

        Args:
            key: Registry key
            keep_files: If True, only remove from registry but keep files

        Returns:
            True if deleted successfully
        """
        if key not in self.registered_models:
            return False

        registered = self.registered_models[key]

        if not keep_files:
            model_path = Path(registered.path)
            if model_path.exists():
                shutil.rmtree(model_path)

        del self.registered_models[key]
        self._save_registry()

        return True

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        models = list(self.registered_models.values())

        stats = {
            "total_models": len(models),
            "active_models": len([m for m in models if m.is_active]),
            "by_market_type": {},
            "by_model_type": {},
            "symbols_covered": set(),
        }

        for model in models:
            stats["symbols_covered"].add(model.symbol)

            mt = model.market_type
            stats["by_market_type"][mt] = stats["by_market_type"].get(mt, 0) + 1

            mt2 = model.model_type
            stats["by_model_type"][mt2] = stats["by_model_type"].get(mt2, 0) + 1

        stats["symbols_covered"] = list(stats["symbols_covered"])

        return stats
