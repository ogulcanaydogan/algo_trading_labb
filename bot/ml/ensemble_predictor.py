"""
Ensemble Predictor - Combines Multiple ML Models for Higher Accuracy.

Uses weighted voting from:
- Traditional ML: Random Forest, Gradient Boosting, XGBoost
- Deep Learning: LSTM, Transformer

Achieves better generalization through model diversity.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

# Try to import PyTorch for DL models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# Deep Learning model definitions (must match training script)
if TORCH_AVAILABLE:
    class SimpleLSTM(nn.Module):
        """Simple LSTM for price prediction."""

        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out

    class SimpleTransformer(nn.Module):
        """Simple Transformer for price prediction."""

        def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=3, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size)
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.transformer(x)
            out = self.fc(x[:, -1, :])
            return out

    class RegularizedLSTM(nn.Module):
        """LSTM with strong regularization."""

        def __init__(self, input_size, hidden_size=48, num_layers=1, output_size=3, dropout=0.5):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.dropout(lstm_out[:, -1, :])
            out = self.fc(out)
            return out

    class RegularizedTransformer(nn.Module):
        """Transformer with strong regularization."""

        def __init__(self, input_size, d_model=48, nhead=4, num_layers=1, output_size=3, dropout=0.5):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.dropout1 = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.dropout2 = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.dropout1(x)
            x = self.transformer(x)
            out = self.dropout2(x[:, -1, :])
            out = self.fc(out)
            return out


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models with weighted voting.

    Supports:
    - Traditional ML: Random Forest, Gradient Boosting, XGBoost
    - Deep Learning: LSTM, Transformer

    Strategies:
    1. Majority voting (simple)
    2. Confidence-weighted voting (better)
    3. Performance-based weighting (best)
    """

    def __init__(
        self,
        model_dir: Path = Path("data/models"),
        symbol: str = "BTC/USDT",
        voting_strategy: str = "weighted",  # "majority", "weighted", "performance"
        include_dl: bool = True,  # Include deep learning models
    ):
        self.model_dir = Path(model_dir)
        self.symbol = symbol
        self.voting_strategy = voting_strategy
        self.include_dl = include_dl and TORCH_AVAILABLE

        # Traditional ML models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

        # Deep Learning models
        self.dl_models: Dict[str, Any] = {}
        self.dl_scalers: Dict[str, Any] = {}
        self.dl_meta: Dict[str, Dict] = {}
        self.dl_device = torch.device('cpu') if TORCH_AVAILABLE else None

        # Weights and accuracies
        self.model_weights: Dict[str, float] = {}
        self.model_accuracies: Dict[str, float] = {}

        # Sequence buffer for DL models
        self.feature_buffer: List[np.ndarray] = []
        self.seq_length = 30  # Default, updated when loading models
        
    def load_models(self) -> bool:
        """Load all available models for the symbol."""
        symbol_clean = self.symbol.replace('/', '_')

        # Load traditional ML models
        ml_loaded = self._load_ml_models(symbol_clean)

        # Load deep learning models
        dl_loaded = 0
        if self.include_dl:
            dl_loaded = self._load_dl_models(symbol_clean)

        total_loaded = ml_loaded + dl_loaded

        if total_loaded > 0:
            self._initialize_weights()
            logger.info(f"Ensemble loaded {total_loaded} models for {self.symbol} "
                       f"(ML: {ml_loaded}, DL: {dl_loaded})")
            return True
        else:
            logger.warning(f"No models loaded for {self.symbol}")
            return False

    def _load_ml_models(self, symbol_clean: str) -> int:
        """Load traditional ML models."""
        model_types = ['random_forest', 'gradient_boosting', 'xgboost']
        loaded_count = 0

        for model_type in model_types:
            model_path = self.model_dir / f"{symbol_clean}_{model_type}_model.pkl"
            scaler_path = self.model_dir / f"{symbol_clean}_{model_type}_scaler.pkl"
            # Try per-model meta first, then combined metadata
            per_model_meta = self.model_dir / f"{symbol_clean}_{model_type}_meta.json"
            combined_meta = self.model_dir / f"{symbol_clean}_metadata.json"

            if model_path.exists():
                try:
                    self.models[model_type] = joblib.load(model_path)

                    # Load scaler if exists
                    if scaler_path.exists():
                        self.scalers[model_type] = joblib.load(scaler_path)

                    # Load accuracy - try per-model meta first
                    accuracy = 0.4  # Default
                    if per_model_meta.exists():
                        with open(per_model_meta, 'r') as f:
                            meta = json.load(f)
                            accuracy = meta.get('accuracy', meta.get('cv_mean', 0.4))
                    elif combined_meta.exists():
                        with open(combined_meta, 'r') as f:
                            meta = json.load(f)
                            if 'models' in meta and model_type in meta['models']:
                                accuracy = meta['models'][model_type].get('cv_accuracy', 0.4)

                    self.model_accuracies[model_type] = accuracy
                    loaded_count += 1
                    logger.info(f"Loaded {model_type} for {self.symbol} (accuracy: {accuracy:.2%})")
                except Exception as e:
                    logger.warning(f"Failed to load {model_type}: {e}")

        return loaded_count

    def _load_dl_models(self, symbol_clean: str) -> int:
        """Load deep learning models (LSTM, Transformer)."""
        if not TORCH_AVAILABLE:
            return 0

        loaded_count = 0

        # Try loading different DL model variants
        dl_variants = [
            ('lstm', SimpleLSTM, {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3}),
            ('transformer', SimpleTransformer, {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3}),
            ('lstm_regularized', RegularizedLSTM, {'hidden_size': 48, 'num_layers': 1, 'dropout': 0.5}),
            ('transformer_regularized', RegularizedTransformer, {'d_model': 48, 'nhead': 4, 'num_layers': 1, 'dropout': 0.5}),
        ]

        for model_name, model_class, default_params in dl_variants:
            # Check multiple naming conventions
            if 'regularized' in model_name:
                model_paths = [
                    self.model_dir / f"{symbol_clean}_{model_name}.pt",
                    self.model_dir / f"{symbol_clean}_dl_{model_name}.pt",
                ]
                meta_paths = [
                    self.model_dir / f"{symbol_clean}_dl_regularized_meta.json",
                    self.model_dir / f"{symbol_clean}_dl_{model_name}_meta.json",
                ]
                scaler_path = self.model_dir / f"{symbol_clean}_dl_regularized_scaler.pkl"
            else:
                model_paths = [
                    self.model_dir / f"{symbol_clean}_{model_name}_model.pt",
                    self.model_dir / f"{symbol_clean}_dl_{model_name}.pt",
                ]
                meta_paths = [
                    self.model_dir / f"{symbol_clean}_dl_meta.json",
                    self.model_dir / f"{symbol_clean}_dl_{model_name}_meta.json",
                ]
                scaler_path = self.model_dir / f"{symbol_clean}_dl_scaler.pkl"

            # Find existing model file
            model_path = None
            for mp in model_paths:
                if mp.exists():
                    model_path = mp
                    break

            if model_path is None:
                continue

            # Find existing meta file
            meta_path = None
            for mtp in meta_paths:
                if mtp.exists():
                    meta_path = mtp
                    break

            try:
                # Load metadata to get input_size and seq_length
                input_size = 22  # Default to 22 optimal features
                accuracy = 0.35  # Default for DL

                if meta_path and meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        input_size = meta.get('n_features', meta.get('input_size', 22))
                        self.seq_length = meta.get('sequence_length', meta.get('seq_length', 20))
                        accuracy = meta.get('accuracy', 0.35)
                        self.dl_meta[model_name] = meta
                else:
                    # Try combined meta
                    results = {}
                    base_name = model_name.replace('_regularized', '')
                    if base_name in results:
                        accuracy = results[base_name].get('test_accuracy', 35) / 100

                # Create and load model
                model = model_class(input_size=input_size, **default_params)
                model.load_state_dict(torch.load(model_path, map_location=self.dl_device))
                model.eval()
                model.to(self.dl_device)

                self.dl_models[model_name] = model
                self.model_accuracies[model_name] = accuracy

                # Load scaler
                if scaler_path.exists():
                    self.dl_scalers[model_name] = joblib.load(scaler_path)

                loaded_count += 1
                logger.info(f"Loaded DL {model_name} for {self.symbol} (accuracy: {accuracy:.2%})")

            except Exception as e:
                logger.warning(f"Failed to load DL {model_name}: {e}")

        return loaded_count

    def add_features_to_buffer(self, features: np.ndarray) -> None:
        """Add features to sequence buffer for DL models."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        self.feature_buffer.append(features[0])

        # Keep buffer at seq_length
        if len(self.feature_buffer) > self.seq_length:
            self.feature_buffer = self.feature_buffer[-self.seq_length:]
    
    def _initialize_weights(self) -> None:
        """Initialize voting weights based on strategy."""
        # Combine all model types
        all_models = list(self.models.keys()) + list(self.dl_models.keys())
        n_models = len(all_models)

        if n_models == 0:
            return

        if self.voting_strategy == "majority":
            # Equal weights
            for model_type in all_models:
                self.model_weights[model_type] = 1.0 / n_models

        elif self.voting_strategy == "weighted":
            # Weight by confidence (starts equal, adapts during runtime)
            for model_type in all_models:
                self.model_weights[model_type] = 1.0 / n_models

        elif self.voting_strategy == "performance":
            # Weight by historical accuracy
            total_accuracy = sum(self.model_accuracies.get(m, 0.5) for m in all_models)
            if total_accuracy > 0:
                for model_type in all_models:
                    accuracy = self.model_accuracies.get(model_type, 0.5)
                    self.model_weights[model_type] = accuracy / total_accuracy
            else:
                # Fallback to equal weights
                for model_type in all_models:
                    self.model_weights[model_type] = 1.0 / n_models

        logger.info(f"Ensemble weights: {self.model_weights}")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, Dict[str, Any]]:
        """
        Generate ensemble prediction.

        Returns:
            (prediction, confidence, details)
            - prediction: 1 (long), 0 (flat), -1 (short)
            - confidence: 0.0 to 1.0
            - details: dict with individual predictions
        """
        if not self.models and not self.dl_models:
            raise ValueError("No models loaded. Call load_models() first.")

        predictions = {}

        # Add features to buffer for DL models
        self.add_features_to_buffer(features)

        # Get predictions from traditional ML models
        predictions.update(self._predict_ml(features))

        # Get predictions from DL models (if buffer is full)
        if len(self.feature_buffer) >= self.seq_length:
            predictions.update(self._predict_dl())

        if not predictions:
            return 0, 0.0, {'error': 'No predictions generated'}

        # Combine predictions using voting strategy
        final_prediction, final_confidence = self._vote(predictions)

        # Prepare details
        details = {
            'individual_predictions': predictions,
            'weights': self.model_weights,
            'voting_strategy': self.voting_strategy,
            'num_models': len(predictions),
            'ml_models': len(self.models),
            'dl_models': len(self.dl_models),
            'dl_buffer_ready': len(self.feature_buffer) >= self.seq_length
        }

        return final_prediction, final_confidence, details

    def _predict_ml(self, features: np.ndarray) -> Dict[str, Dict]:
        """Get predictions from traditional ML models."""
        predictions = {}

        for model_type, model in self.models.items():
            try:
                # Scale features if scaler available and feature count matches
                if model_type in self.scalers:
                    expected_features = self.scalers[model_type].n_features_in_
                    actual_features = features.shape[1] if features.ndim > 1 else features.shape[0]
                    features_2d = features.reshape(1, -1) if features.ndim == 1 else features
                    if expected_features == actual_features:
                        features_scaled = self.scalers[model_type].transform(features_2d)
                    else:
                        logger.debug(f"Skipping scaler for {model_type}: expects {expected_features}, got {actual_features}")
                        features_scaled = features_2d
                else:
                    features_scaled = features.reshape(1, -1) if features.ndim == 1 else features

                # Get prediction
                raw_pred = model.predict(features_scaled)[0]

                # Map model output to ensemble format
                # Models use: 0=SHORT, 1=FLAT, 2=LONG
                # Ensemble uses: -1=short, 0=flat, 1=long
                pred_map = {0: -1, 1: 0, 2: 1}
                pred = pred_map.get(int(raw_pred), 0)

                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    predictions[model_type] = {
                        'prediction': pred,
                        'confidence': float(max(proba)),
                        'type': 'ml'
                    }
                else:
                    predictions[model_type] = {
                        'prediction': pred,
                        'confidence': 0.7,
                        'type': 'ml'
                    }
            except Exception as e:
                logger.warning(f"ML prediction failed for {model_type}: {e}")

        return predictions

    def _predict_dl(self) -> Dict[str, Dict]:
        """Get predictions from deep learning models."""
        if not TORCH_AVAILABLE or len(self.feature_buffer) < self.seq_length:
            return {}

        predictions = {}

        # Build sequence from buffer
        sequence = np.array(self.feature_buffer[-self.seq_length:])

        for model_name, model in self.dl_models.items():
            try:
                # Scale if scaler available
                if model_name in self.dl_scalers:
                    scaler = self.dl_scalers[model_name]
                    # Scale each timestep
                    expected_features = scaler.n_features_in_
                    if sequence.shape[1] == expected_features:
                        sequence_scaled = scaler.transform(sequence)
                    else:
                        # Features don't match - use unscaled
                        sequence_scaled = sequence
                else:
                    sequence_scaled = sequence

                # Convert to tensor and add batch dimension
                x = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.dl_device)

                # Get prediction
                with torch.no_grad():
                    outputs = model(x)
                    proba = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    raw_pred = int(torch.argmax(outputs, dim=1).item())

                # Map: 0=DOWN, 1=FLAT, 2=UP -> -1, 0, 1
                pred_map = {0: -1, 1: 0, 2: 1}
                pred = pred_map.get(raw_pred, 0)

                predictions[model_name] = {
                    'prediction': pred,
                    'confidence': float(max(proba)),
                    'probabilities': proba.tolist(),
                    'type': 'dl'
                }

            except Exception as e:
                logger.warning(f"DL prediction failed for {model_name}: {e}")

        return predictions
    
    def _vote(self, predictions: Dict[str, Dict]) -> Tuple[int, float]:
        """Combine predictions using voting strategy."""
        if not predictions:
            return 0, 0.0  # Flat with zero confidence
        
        if self.voting_strategy == "majority":
            # Simple majority vote
            votes = [p['prediction'] for p in predictions.values()]
            from collections import Counter
            vote_counts = Counter(votes)
            winner = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[winner] / len(votes)
            return winner, confidence
        
        elif self.voting_strategy in ["weighted", "performance"]:
            # Weighted voting
            vote_scores = {1: 0.0, 0: 0.0, -1: 0.0}  # long, flat, short
            
            for model_type, pred_info in predictions.items():
                weight = self.model_weights.get(model_type, 0.0)
                pred = pred_info['prediction']
                conf = pred_info['confidence']
                
                # Weight by both model weight and prediction confidence
                vote_scores[pred] += weight * conf
            
            # Winner is highest weighted score
            winner = max(vote_scores, key=vote_scores.get)
            
            # Confidence is normalized score
            total_score = sum(vote_scores.values())
            if total_score > 0:
                confidence = vote_scores[winner] / total_score
            else:
                confidence = 0.0
            
            return winner, confidence
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def update_weights(self, model_type: str, accuracy: float) -> None:
        """Update model weight based on recent performance."""
        if self.voting_strategy == "performance":
            self.model_accuracies[model_type] = accuracy
            self._initialize_weights()
            logger.info(f"Updated {model_type} accuracy to {accuracy:.2%}")


def create_ensemble_predictor(
    symbol: str,
    model_dir: Path = Path("data/models"),
    voting_strategy: str = "performance"
) -> Optional[EnsemblePredictor]:
    """Factory function to create and load an ensemble predictor."""
    predictor = EnsemblePredictor(
        model_dir=model_dir,
        symbol=symbol,
        voting_strategy=voting_strategy
    )
    
    if predictor.load_models():
        return predictor
    else:
        return None
