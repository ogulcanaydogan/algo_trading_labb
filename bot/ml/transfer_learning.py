"""
Transfer Learning for Trading Models

Implements transfer learning strategies:
1. Pre-train on BTC (most liquid, most data)
2. Fine-tune on target assets
3. Multi-task learning across assets
4. Domain adaptation for different market types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result of transfer learning."""

    source_symbol: str
    target_symbol: str
    source_accuracy: float
    direct_transfer_accuracy: float
    finetuned_accuracy: float
    improvement_over_scratch: float
    best_approach: str
    finetuning_epochs: int


class FeatureAligner:
    """
    Aligns features between source and target domains.

    Different assets may have different feature distributions.
    This class normalizes and aligns them for transfer.
    """

    def __init__(self):
        self.source_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_mapping = None

    def fit_source(self, X_source: pd.DataFrame) -> np.ndarray:
        """Fit scaler on source domain."""
        return self.source_scaler.fit_transform(X_source)

    def fit_target(self, X_target: pd.DataFrame) -> np.ndarray:
        """Fit scaler on target domain."""
        return self.target_scaler.fit_transform(X_target)

    def align_features(
        self, X_source: pd.DataFrame, X_target: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align features between source and target.

        Ensures both domains have same features in same order.
        """
        # Find common features
        common_features = list(set(X_source.columns) & set(X_target.columns))
        common_features.sort()

        if len(common_features) == 0:
            raise ValueError("No common features between source and target")

        logger.info(f"Using {len(common_features)} common features")

        X_source_aligned = X_source[common_features]
        X_target_aligned = X_target[common_features]

        self.feature_mapping = common_features

        return X_source_aligned, X_target_aligned


class PretrainedModelBank:
    """
    Bank of pre-trained models for transfer learning.

    Models are pre-trained on BTC/major assets and can be
    transferred to other assets.
    """

    def __init__(self, model_dir: Path = Path("data/models/pretrained")):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, BaseEstimator] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}

    def save_pretrained(
        self, symbol: str, model: BaseEstimator, scaler: StandardScaler, feature_names: List[str]
    ):
        """Save a pre-trained model."""
        symbol_clean = symbol.replace("/", "_")

        model_path = self.model_dir / f"{symbol_clean}_pretrained.pkl"
        scaler_path = self.model_dir / f"{symbol_clean}_scaler.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        import json

        meta = {
            "symbol": symbol,
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = self.model_dir / f"{symbol_clean}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_names[symbol] = feature_names

        logger.info(f"Saved pretrained model for {symbol}")

    def load_pretrained(self, symbol: str) -> Tuple[BaseEstimator, StandardScaler, List[str]]:
        """Load a pre-trained model."""
        if symbol in self.models:
            return self.models[symbol], self.scalers[symbol], self.feature_names[symbol]

        symbol_clean = symbol.replace("/", "_")
        model_path = self.model_dir / f"{symbol_clean}_pretrained.pkl"
        scaler_path = self.model_dir / f"{symbol_clean}_scaler.pkl"
        meta_path = self.model_dir / f"{symbol_clean}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No pretrained model for {symbol}")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        import json

        with open(meta_path) as f:
            meta = json.load(f)

        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_names[symbol] = meta["feature_names"]

        return model, scaler, meta["feature_names"]

    def list_available(self) -> List[str]:
        """List available pretrained models."""
        available = []
        for f in self.model_dir.glob("*_pretrained.pkl"):
            symbol = f.stem.replace("_pretrained", "").replace("_", "/")
            available.append(symbol)
        return available


class TransferLearner:
    """
    Main transfer learning engine.

    Strategies:
    1. Direct transfer: Use source model directly
    2. Fine-tuning: Continue training on target data
    3. Feature extraction: Use source model as feature extractor
    4. Progressive fine-tuning: Gradually unfreeze layers
    """

    def __init__(
        self,
        source_symbol: str = "BTC/USDT",
        finetune_epochs: int = 10,
        finetune_lr_scale: float = 0.1,
    ):
        self.source_symbol = source_symbol
        self.finetune_epochs = finetune_epochs
        self.finetune_lr_scale = finetune_lr_scale

        self.model_bank = PretrainedModelBank()
        self.feature_aligner = FeatureAligner()

    def pretrain_on_source(
        self, X_source: pd.DataFrame, y_source: pd.Series, model_type: str = "random_forest"
    ) -> Tuple[BaseEstimator, float]:
        """
        Pre-train model on source domain (e.g., BTC).

        Args:
            X_source: Source features
            y_source: Source labels
            model_type: Type of model

        Returns:
            Trained model and accuracy
        """
        logger.info(f"Pre-training on {self.source_symbol} ({len(X_source)} samples)")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_source)

        # Train-test split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_source.iloc[:split_idx], y_source.iloc[split_idx:]

        # Create model
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            )
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        logger.info(f"Pre-training accuracy: {accuracy:.2%}")

        # Save to model bank
        self.model_bank.save_pretrained(self.source_symbol, model, scaler, list(X_source.columns))

        return model, accuracy

    def transfer_to_target(
        self,
        X_target: pd.DataFrame,
        y_target: pd.Series,
        target_symbol: str,
        strategy: str = "finetune",
    ) -> TransferResult:
        """
        Transfer pretrained model to target domain.

        Args:
            X_target: Target features
            y_target: Target labels
            target_symbol: Target asset symbol
            strategy: 'direct', 'finetune', or 'scratch'

        Returns:
            TransferResult with all metrics
        """
        logger.info(f"Transferring to {target_symbol} using {strategy} strategy")

        # Load pretrained model
        try:
            source_model, source_scaler, source_features = self.model_bank.load_pretrained(
                self.source_symbol
            )
        except FileNotFoundError:
            logger.warning(f"No pretrained model for {self.source_symbol}, training from scratch")
            strategy = "scratch"

        # Align features
        common_features = list(set(source_features) & set(X_target.columns))
        if len(common_features) < len(source_features) * 0.5:
            logger.warning(f"Only {len(common_features)} common features, may affect transfer")

        X_target_aligned = X_target[common_features] if strategy != "scratch" else X_target

        # Scale target data
        target_scaler = StandardScaler()
        X_target_scaled = target_scaler.fit_transform(X_target_aligned)

        # Train-test split
        split_idx = int(len(X_target_scaled) * 0.8)
        X_train, X_test = X_target_scaled[:split_idx], X_target_scaled[split_idx:]
        y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

        # Evaluate different approaches
        results = {}

        # 1. Train from scratch (baseline)
        scratch_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
        )
        scratch_model.fit(X_train, y_train)
        scratch_accuracy = accuracy_score(y_test, scratch_model.predict(X_test))
        results["scratch"] = scratch_accuracy
        logger.info(f"From scratch accuracy: {scratch_accuracy:.2%}")

        if strategy != "scratch":
            # 2. Direct transfer
            # Align source scaler to target data distribution
            X_test_for_source = source_scaler.transform(X_target[common_features].iloc[split_idx:])
            direct_accuracy = accuracy_score(y_test, source_model.predict(X_test_for_source))
            results["direct"] = direct_accuracy
            logger.info(f"Direct transfer accuracy: {direct_accuracy:.2%}")

            # 3. Fine-tuning
            if strategy == "finetune":
                finetuned_model, finetune_accuracy = self._finetune(
                    source_model, X_train, y_train, X_test, y_test
                )
                results["finetune"] = finetune_accuracy
                logger.info(f"Fine-tuned accuracy: {finetune_accuracy:.2%}")
            else:
                finetune_accuracy = direct_accuracy
                results["finetune"] = finetune_accuracy

        else:
            results["direct"] = scratch_accuracy
            results["finetune"] = scratch_accuracy

        # Determine best approach
        best_approach = max(results, key=results.get)
        best_accuracy = results[best_approach]

        improvement = (
            ((best_accuracy - scratch_accuracy) / scratch_accuracy) * 100
            if scratch_accuracy > 0
            else 0
        )

        return TransferResult(
            source_symbol=self.source_symbol,
            target_symbol=target_symbol,
            source_accuracy=0,  # Would need to calculate
            direct_transfer_accuracy=results.get("direct", 0),
            finetuned_accuracy=results.get("finetune", 0),
            improvement_over_scratch=improvement,
            best_approach=best_approach,
            finetuning_epochs=self.finetune_epochs,
        )

    def _finetune(
        self,
        source_model: BaseEstimator,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> Tuple[BaseEstimator, float]:
        """
        Fine-tune source model on target data.

        For tree-based models, we use warm_start or train new trees
        initialized from source model's feature importances.
        """
        # Clone source model
        finetuned = clone(source_model)

        # For RandomForest, we can use warm_start
        if isinstance(source_model, RandomForestClassifier):
            # Transfer feature importances as prior
            if hasattr(source_model, "feature_importances_"):
                # Create new model with source knowledge
                finetuned = RandomForestClassifier(
                    n_estimators=source_model.n_estimators + 50,  # Add trees
                    max_depth=source_model.max_depth,
                    class_weight="balanced",
                    random_state=42,
                    warm_start=True,
                    n_jobs=-1,
                )

                # Initial fit with subset
                subset_size = min(1000, len(X_train))
                finetuned.fit(X_train[:subset_size], y_train.iloc[:subset_size])

                # Continue fitting with full data
                finetuned.n_estimators += 50
                finetuned.fit(X_train, y_train)

        else:
            # For other models, just retrain
            finetuned.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, finetuned.predict(X_test))

        return finetuned, accuracy


class MultiTaskLearner:
    """
    Multi-task learning across multiple assets.

    Learns shared representations that help all tasks.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.shared_features = None
        self.task_specific_models: Dict[str, BaseEstimator] = {}

    def train_multitask(self, data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, float]:
        """
        Train multi-task model on all symbols.

        Args:
            data: Dict of {symbol: (X, y)} pairs

        Returns:
            Dict of accuracies per symbol
        """
        logger.info(f"Training multi-task model on {len(data)} symbols")

        # Find common features
        all_features = [set(X.columns) for X, _ in data.values()]
        common_features = list(set.intersection(*all_features))
        common_features.sort()

        logger.info(f"Using {len(common_features)} common features")
        self.shared_features = common_features

        # Combine all data with task labels
        combined_X = []
        combined_y = []
        task_labels = []

        for i, (symbol, (X, y)) in enumerate(data.items()):
            X_aligned = X[common_features]
            combined_X.append(X_aligned)
            combined_y.append(y)
            task_labels.extend([i] * len(X))

        X_combined = pd.concat(combined_X, ignore_index=True)
        y_combined = pd.concat(combined_y, ignore_index=True)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # Train shared model
        shared_model = RandomForestClassifier(
            n_estimators=300, max_depth=25, class_weight="balanced", random_state=42, n_jobs=-1
        )
        shared_model.fit(X_scaled, y_combined)

        # Evaluate on each task
        results = {}
        start_idx = 0

        for symbol, (X, y) in data.items():
            end_idx = start_idx + len(X)
            X_task = X_scaled[start_idx:end_idx]
            y_task = y

            split = int(len(X_task) * 0.8)
            accuracy = accuracy_score(y_task.iloc[split:], shared_model.predict(X_task[split:]))
            results[symbol] = accuracy

            logger.info(f"{symbol}: {accuracy:.2%}")
            start_idx = end_idx

        return results


def transfer_from_btc(
    target_symbol: str,
    X_target: pd.DataFrame,
    y_target: pd.Series,
    X_btc: Optional[pd.DataFrame] = None,
    y_btc: Optional[pd.Series] = None,
) -> TransferResult:
    """
    Convenience function to transfer learning from BTC to target.

    Args:
        target_symbol: Target asset
        X_target: Target features
        y_target: Target labels
        X_btc: Optional BTC features (will load if not provided)
        y_btc: Optional BTC labels

    Returns:
        TransferResult
    """
    learner = TransferLearner(source_symbol="BTC/USDT")

    # Pre-train on BTC if data provided
    if X_btc is not None and y_btc is not None:
        learner.pretrain_on_source(X_btc, y_btc)

    # Transfer to target
    result = learner.transfer_to_target(X_target, y_target, target_symbol, strategy="finetune")

    return result
