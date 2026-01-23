"""
ML Training Pipeline

Phase 11: Comprehensive ML model training pipeline that integrates
technical indicators, news features, and regime data for trading predictions.

Features:
1. Feature engineering (technical + fundamental + news)
2. Multiple model architectures (XGBoost, LightGBM, Neural Networks)
3. Walk-forward training with proper train/val/test splits
4. Hyperparameter optimization
5. Model serialization and versioning
6. Performance evaluation and reporting
"""

import logging
import os
import joblib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    NEURAL_NET = "neural_net"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class PredictionTarget(Enum):
    """What the model predicts."""
    DIRECTION = "direction"      # Up/Down/Flat
    RETURN = "return"            # Continuous return
    VOLATILITY = "volatility"    # Future volatility
    REGIME = "regime"            # Market regime


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType = ModelType.XGBOOST
    target: PredictionTarget = PredictionTarget.DIRECTION

    # Data settings
    lookback_periods: int = 60
    forecast_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Feature settings
    use_technical_features: bool = True
    use_news_features: bool = True
    use_regime_features: bool = True
    use_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Training settings
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    early_stopping_rounds: int = 10
    random_state: int = 42

    # Walk-forward settings
    walk_forward_windows: int = 5
    retrain_frequency_days: int = 30

    # Neural network settings (if applicable)
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    epochs: int = 100
    batch_size: int = 32


@dataclass
class FeatureSet:
    """Container for feature data."""
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Results from model training."""
    model_id: str
    model_type: ModelType
    target: PredictionTarget
    trained_at: datetime

    # Performance metrics
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    # Feature importance
    feature_importance: Dict[str, float]

    # Metadata
    config: TrainingConfig
    data_hash: str
    model_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "target": self.target.value,
            "trained_at": self.trained_at.isoformat(),
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_importance": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),
            "model_path": self.model_path
        }


class FeatureEngineer:
    """
    Feature engineering for ML models.

    Creates features from:
    - Price data (OHLCV)
    - Technical indicators
    - News sentiment
    - Regime indicators
    """

    def __init__(self, config: TrainingConfig):
        """Initialize feature engineer."""
        self.config = config
        self.feature_names: List[str] = []

    def create_features(
        self,
        price_data: pd.DataFrame,
        news_features: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> FeatureSet:
        """
        Create feature set from raw data.

        Args:
            price_data: OHLCV data with datetime index
            news_features: News sentiment features
            regime_data: Market regime indicators

        Returns:
            FeatureSet ready for training
        """
        features_list = []
        self.feature_names = []

        # Technical features
        if self.config.use_technical_features:
            tech_features = self._create_technical_features(price_data)
            features_list.append(tech_features)

        # News features
        if self.config.use_news_features and news_features is not None:
            news_feat = self._align_news_features(price_data, news_features)
            features_list.append(news_feat)

        # Regime features
        if self.config.use_regime_features and regime_data is not None:
            regime_feat = self._align_regime_features(price_data, regime_data)
            features_list.append(regime_feat)

        # Lag features
        if self.config.use_lag_features:
            lag_features = self._create_lag_features(price_data)
            features_list.append(lag_features)

        # Combine all features
        X = pd.concat(features_list, axis=1)

        # Create target
        y = self._create_target(price_data)

        # Align and clean
        X, y, timestamps = self._align_and_clean(X, y, price_data.index)

        return FeatureSet(
            X=X.values,
            y=y.values,
            feature_names=list(X.columns),
            timestamps=timestamps,
            metadata={
                "n_samples": len(X),
                "n_features": X.shape[1],
                "target_distribution": dict(pd.Series(y).value_counts())
            }
        )

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Returns
        features['return_1d'] = close.pct_change(1)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        self.feature_names.extend(['return_1d', 'return_5d', 'return_10d', 'return_20d'])

        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}'] = close / ma - 1
            self.feature_names.append(f'ma_{period}')

        # Volatility
        features['volatility_10d'] = close.pct_change().rolling(10).std()
        features['volatility_20d'] = close.pct_change().rolling(20).std()
        self.feature_names.extend(['volatility_10d', 'volatility_20d'])

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        self.feature_names.append('rsi_14')

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])

        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = (close - (ma20 + 2 * std20)) / close
        features['bb_lower'] = (close - (ma20 - 2 * std20)) / close
        features['bb_width'] = (4 * std20) / ma20
        self.feature_names.extend(['bb_upper', 'bb_lower', 'bb_width'])

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean() / close
        self.feature_names.append('atr_14')

        # Volume features
        if 'volume' in df.columns:
            features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
            features['volume_change'] = volume.pct_change()
            self.feature_names.extend(['volume_ma_ratio', 'volume_change'])

        # Price position
        features['price_position'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())
        self.feature_names.append('price_position')

        return features

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features."""
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change()

        for lag in self.config.lag_periods:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            self.feature_names.append(f'return_lag_{lag}')

        return features

    def _align_news_features(
        self,
        price_data: pd.DataFrame,
        news_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Align news features to price data index."""
        # Forward fill news features to match price data frequency
        aligned = news_features.reindex(price_data.index, method='ffill')

        # Add feature names
        for col in aligned.columns:
            if col not in self.feature_names:
                self.feature_names.append(col)

        return aligned

    def _align_regime_features(
        self,
        price_data: pd.DataFrame,
        regime_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align regime features to price data index."""
        aligned = regime_data.reindex(price_data.index, method='ffill')

        for col in aligned.columns:
            if col not in self.feature_names:
                self.feature_names.append(col)

        return aligned

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create prediction target."""
        close = df['close']
        horizon = self.config.forecast_horizon

        if self.config.target == PredictionTarget.DIRECTION:
            # 1 = up, 0 = down
            future_return = close.shift(-horizon) / close - 1
            target = (future_return > 0).astype(int)

        elif self.config.target == PredictionTarget.RETURN:
            target = close.shift(-horizon) / close - 1

        elif self.config.target == PredictionTarget.VOLATILITY:
            returns = close.pct_change()
            target = returns.shift(-horizon).rolling(horizon).std()

        else:
            target = pd.Series(0, index=df.index)

        return target

    def _align_and_clean(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        index: pd.DatetimeIndex
    ) -> Tuple[pd.DataFrame, pd.Series, List[datetime]]:
        """Align features and target, remove NaN rows."""
        # Combine for alignment
        combined = pd.concat([X, y.rename('target')], axis=1)

        # Drop rows with NaN
        combined = combined.dropna()

        X_clean = combined.drop('target', axis=1)
        y_clean = combined['target']
        timestamps = [ts.to_pydatetime() for ts in combined.index]

        return X_clean, y_clean, timestamps


class BaseModel(ABC):
    """Base class for ML models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)."""
        pass

    @abstractmethod
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""

    def __init__(self, config: TrainingConfig):
        """Initialize XGBoost model."""
        self.config = config
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed, using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
            self.model.fit(X, y)
            return

        X_val = kwargs.get('X_val')
        y_val = kwargs.get('y_val')

        params = {
            'objective': 'binary:logistic' if self.config.target == PredictionTarget.DIRECTION else 'reg:squarederror',
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'random_state': self.config.random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss' if self.config.target == PredictionTarget.DIRECTION else 'rmse'
        }

        self.model = xgb.XGBClassifier(**params) if self.config.target == PredictionTarget.DIRECTION else xgb.XGBRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(feature_names, importance))
        return {}

    def save(self, path: str) -> None:
        """Save model."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """Load model."""
        instance = cls(TrainingConfig())
        instance.model = joblib.load(path)
        return instance


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, config: TrainingConfig):
        """Initialize LightGBM model."""
        self.config = config
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed, falling back to XGBoost")
            xgb_model = XGBoostModel(self.config)
            xgb_model.fit(X, y, **kwargs)
            self.model = xgb_model.model
            return

        X_val = kwargs.get('X_val')
        y_val = kwargs.get('y_val')

        params = {
            'objective': 'binary' if self.config.target == PredictionTarget.DIRECTION else 'regression',
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'random_state': self.config.random_state,
            'verbose': -1
        }

        self.model = lgb.LGBMClassifier(**params) if self.config.target == PredictionTarget.DIRECTION else lgb.LGBMRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X, y,
            eval_set=eval_set
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        return {}

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        instance = cls(TrainingConfig())
        instance.model = joblib.load(path)
        return instance


class NeuralNetModel(BaseModel):
    """Simple neural network model."""

    def __init__(self, config: TrainingConfig):
        """Initialize neural network."""
        self.config = config
        self.model = None
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train neural network."""
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_val = kwargs.get('X_val')
        y_val = kwargs.get('y_val')

        if self.config.target == PredictionTarget.DIRECTION:
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                max_iter=self.config.epochs,
                random_state=self.config.random_state,
                early_stopping=True if X_val is not None else False,
                validation_fraction=0.1
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                max_iter=self.config.epochs,
                random_state=self.config.random_state,
                early_stopping=True if X_val is not None else False,
                validation_fraction=0.1
            )

        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        # Neural networks don't have direct feature importance
        # Return equal weights
        return {name: 1.0 / len(feature_names) for name in feature_names}

    def save(self, path: str) -> None:
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    @classmethod
    def load(cls, path: str) -> 'NeuralNetModel':
        instance = cls(TrainingConfig())
        data = joblib.load(path)
        instance.model = data['model']
        instance.scaler = data['scaler']
        return instance


class MLTrainingPipeline:
    """
    Complete ML training pipeline.

    Handles:
    - Data preparation
    - Feature engineering
    - Model training
    - Evaluation
    - Model persistence
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_dir: str = "./data/models"
    ):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
            model_dir: Directory to save models
        """
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer(config)
        self.model: Optional[BaseModel] = None

        logger.info(f"MLTrainingPipeline initialized with {config.model_type.value} model")

    def train(
        self,
        price_data: pd.DataFrame,
        news_features: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> TrainingResult:
        """
        Train a model on the provided data.

        Args:
            price_data: OHLCV price data
            news_features: Optional news sentiment features
            regime_data: Optional regime indicator data

        Returns:
            TrainingResult with metrics and model info
        """
        logger.info("Starting training pipeline...")

        # Create features
        feature_set = self.feature_engineer.create_features(
            price_data, news_features, regime_data
        )

        logger.info(f"Created {feature_set.X.shape[1]} features from {feature_set.X.shape[0]} samples")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(feature_set)

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Create model
        self.model = self._create_model()

        # Train
        logger.info(f"Training {self.config.model_type.value} model...")
        self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Evaluate
        train_metrics = self._evaluate(X_train, y_train, "train")
        val_metrics = self._evaluate(X_val, y_val, "val")
        test_metrics = self._evaluate(X_test, y_test, "test")

        logger.info(f"Test metrics: {test_metrics}")

        # Feature importance
        feature_importance = self.model.get_feature_importance(feature_set.feature_names)

        # Generate model ID
        model_id = self._generate_model_id(feature_set)

        # Save model
        model_path = self._save_model(model_id)

        result = TrainingResult(
            model_id=model_id,
            model_type=self.config.model_type,
            target=self.config.target,
            trained_at=datetime.now(),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            config=self.config,
            data_hash=self._compute_data_hash(feature_set.X),
            model_path=model_path
        )

        # Save training result
        self._save_result(result)

        return result

    def _split_data(
        self,
        feature_set: FeatureSet
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test sets."""
        n = len(feature_set.X)

        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        X_train = feature_set.X[:train_end]
        X_val = feature_set.X[train_end:val_end]
        X_test = feature_set.X[val_end:]

        y_train = feature_set.y[:train_end]
        y_val = feature_set.y[train_end:val_end]
        y_test = feature_set.y[val_end:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_model(self) -> BaseModel:
        """Create model instance based on config."""
        model_classes = {
            ModelType.XGBOOST: XGBoostModel,
            ModelType.LIGHTGBM: LightGBMModel,
            ModelType.NEURAL_NET: NeuralNetModel,
            ModelType.RANDOM_FOREST: XGBoostModel,  # Fallback
        }

        model_class = model_classes.get(self.config.model_type, XGBoostModel)
        return model_class(self.config)

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_pred = self.model.predict(X)

        if self.config.target == PredictionTarget.DIRECTION:
            return {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, average='binary', zero_division=0)),
                "recall": float(recall_score(y, y_pred, average='binary', zero_division=0)),
                "f1": float(f1_score(y, y_pred, average='binary', zero_division=0))
            }
        else:
            return {
                "mse": float(mean_squared_error(y, y_pred)),
                "mae": float(mean_absolute_error(y, y_pred)),
                "r2": float(r2_score(y, y_pred))
            }

    def _generate_model_id(self, feature_set: FeatureSet) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.model_type.value[:3]
        target = self.config.target.value[:3]
        return f"model_{model_type}_{target}_{timestamp}"

    def _compute_data_hash(self, X: np.ndarray) -> str:
        """Compute hash of training data."""
        return hashlib.md5(X.tobytes()).hexdigest()[:12]

    def _save_model(self, model_id: str) -> str:
        """Save model to disk."""
        model_path = self.model_dir / f"{model_id}.pkl"
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def _save_result(self, result: TrainingResult) -> None:
        """Save training result to JSON."""
        result_path = self.model_dir / f"{result.model_id}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Training result saved to {result_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        model_classes = {
            ModelType.XGBOOST: XGBoostModel,
            ModelType.LIGHTGBM: LightGBMModel,
            ModelType.NEURAL_NET: NeuralNetModel,
        }

        model_class = model_classes.get(self.config.model_type, XGBoostModel)
        self.model = model_class.load(model_path)
        logger.info(f"Model loaded from {model_path}")


def create_training_pipeline(
    model_type: str = "xgboost",
    target: str = "direction",
    **kwargs
) -> MLTrainingPipeline:
    """
    Factory function to create a training pipeline.

    Args:
        model_type: Model type (xgboost, lightgbm, neural_net)
        target: Prediction target (direction, return, volatility)
        **kwargs: Additional config options

    Returns:
        Configured MLTrainingPipeline
    """
    config = TrainingConfig(
        model_type=ModelType(model_type),
        target=PredictionTarget(target),
        **kwargs
    )

    return MLTrainingPipeline(config)


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(500) * 0.5) + abs(np.random.randn(500)),
        'low': 100 + np.cumsum(np.random.randn(500) * 0.5) - abs(np.random.randn(500)),
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)

    # Fix high/low
    price_data['high'] = price_data[['open', 'high', 'close']].max(axis=1)
    price_data['low'] = price_data[['open', 'low', 'close']].min(axis=1)

    print("=== ML Training Pipeline Demo ===")

    # Create pipeline
    pipeline = create_training_pipeline(
        model_type="xgboost",
        target="direction",
        n_estimators=50,
        max_depth=4
    )

    # Train
    result = pipeline.train(price_data)

    print(f"\nModel ID: {result.model_id}")
    print(f"Test Accuracy: {result.test_metrics.get('accuracy', 0):.3f}")
    print(f"\nTop 5 Features:")
    for name, imp in list(result.feature_importance.items())[:5]:
        print(f"  {name}: {imp:.4f}")
