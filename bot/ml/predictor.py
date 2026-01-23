"""
ML Predictor Module using XGBoost and RandomForest.

Provides real machine learning predictions for trading signals.
"""

from __future__ import annotations

import json
import os
import joblib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .data_quality import (
    build_quality_report,
    get_feature_columns,
    save_quality_report,
    validate_feature_leakage,
    validate_target_alignment,
)
from .feature_engineer import FeatureEngineer, FeatureConfig


@dataclass
class PredictionResult:
    """Result from ML prediction."""
    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    probability_long: float
    probability_short: float
    probability_flat: float
    expected_return: float
    features_used: int
    model_type: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "probability_long": round(self.probability_long, 4),
            "probability_short": round(self.probability_short, 4),
            "probability_flat": round(self.probability_flat, 4),
            "expected_return": round(self.expected_return, 6),
            "features_used": self.features_used,
            "model_type": self.model_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelMetrics:
    """Training metrics for a model."""
    accuracy: float
    cross_val_mean: float
    cross_val_std: float
    train_samples: int
    test_samples: int
    feature_importance: Dict[str, float] = field(default_factory=dict)


class MLPredictor:
    """
    Machine Learning Predictor for Trading Signals.

    Supports:
    - XGBoost (if installed)
    - Random Forest
    - Gradient Boosting

    Features:
    - Automatic feature engineering
    - Model training and persistence
    - Probability-based predictions
    - Feature importance analysis
    """

    def __init__(
        self,
        model_type: Literal["xgboost", "random_forest", "gradient_boosting"] = "random_forest",
        feature_config: Optional[FeatureConfig] = None,
        model_dir: str = "data/models",
    ):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.model_type = model_type
        self.feature_engineer = FeatureEngineer(feature_config)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None

        self._init_model()

    def _init_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == "xgboost":
            if not HAS_XGBOOST:
                print("XGBoost not installed, falling back to Random Forest")
                self.model_type = "random_forest"
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=3,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                )
                return

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

    def train(
        self,
        ohlcv: pd.DataFrame,
        test_size: float = 0.2,
        validate: bool = True,
        label_horizon: int = 1,
        use_triple_barrier: bool = False,
        atr_multiplier: float = 2.0,
        min_return: float = 0.001,
        report_data_quality: bool = True,
        report_dir: str = "data/reports",
        symbol: Optional[str] = None,
    ) -> ModelMetrics:
        """
        Train the ML model on historical data.

        Args:
            ohlcv: OHLCV DataFrame
            test_size: Fraction of data to use for testing
            validate: Whether to perform cross-validation

        Returns:
            ModelMetrics with training results
        """
        print(f"Training {self.model_type} model...")

        # Extract features
        df = self.feature_engineer.extract_features(ohlcv)
        df = self.feature_engineer.build_labels(
            df,
            horizon=label_horizon,
            use_triple_barrier=use_triple_barrier,
            atr_multiplier=atr_multiplier,
            min_return=min_return,
        )
        df = df.dropna()

        # Prepare features and target
        feature_cols = get_feature_columns(df)
        leakage = validate_feature_leakage(feature_cols)
        if leakage:
            print(f"Warning: leakage columns detected and removed: {leakage}")
            feature_cols = [col for col in feature_cols if col not in leakage]

        alignment_warnings = validate_target_alignment(
            df,
            target_col="target_return",
            horizon=label_horizon,
        )
        for warning in alignment_warnings:
            print(f"Warning: {warning}")

        target_label = "target_class"
        if use_triple_barrier and "target_triple_barrier" in df.columns:
            target_label = "target_triple_barrier"
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_label].values  # 0=SHORT, 1=FLAT, 2=LONG

        if report_data_quality:
            report = build_quality_report(
                df,
                feature_cols=feature_cols,
                target_col=target_label,
                symbol=symbol,
                metadata={
                    "model_type": self.model_type,
                    "label_horizon": label_horizon,
                    "use_triple_barrier": use_triple_barrier,
                },
                alignment_warnings=alignment_warnings,
            )
            report_path = save_quality_report(report, report_dir=report_dir)
            print(f"Data quality report saved to {report_path}")

        # Split data (time-series aware - no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation (on training set only)
        cv_mean, cv_std = 0.0, 0.0
        if validate and len(X_train) > 100:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                feature_importance[name] = float(importance)

        self.metrics = ModelMetrics(
            accuracy=accuracy,
            cross_val_mean=cv_mean,
            cross_val_std=cv_std,
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_importance=feature_importance,
        )

        self.is_trained = True
        print(f"Training complete. Accuracy: {accuracy:.4f}, CV: {cv_mean:.4f} +/- {cv_std:.4f}")

        return self.metrics

    def predict(self, ohlcv: pd.DataFrame) -> PredictionResult:
        """
        Make a prediction based on current market data.

        Args:
            ohlcv: Recent OHLCV data (at least 100 bars recommended)

        Returns:
            PredictionResult with action and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load a saved model.")

        # Extract features
        df = self.feature_engineer.extract_features(ohlcv)

        if len(df) == 0:
            return self._default_prediction()

        # Get latest features
        X = df[self.feature_names].iloc[-1:].values

        # Sanitize infinity values - replace inf/-inf with NaN, then fill with 0
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.transform(X)

        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Map class indices to actions
        # 0=SHORT, 1=FLAT, 2=LONG
        prob_short = probabilities[0] if len(probabilities) > 0 else 0.33
        prob_flat = probabilities[1] if len(probabilities) > 1 else 0.34
        prob_long = probabilities[2] if len(probabilities) > 2 else 0.33

        # Determine action
        action_idx = np.argmax(probabilities)
        action_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        action = action_map.get(action_idx, "FLAT")
        confidence = float(probabilities[action_idx])

        # Estimate expected return based on recent data
        recent_returns = ohlcv["close"].pct_change().tail(20)
        expected_return = float(recent_returns.mean() * (1 if action == "LONG" else -1 if action == "SHORT" else 0))

        return PredictionResult(
            action=action,
            confidence=confidence,
            probability_long=float(prob_long),
            probability_short=float(prob_short),
            probability_flat=float(prob_flat),
            expected_return=expected_return,
            features_used=len(self.feature_names),
            model_type=self.model_type,
        )

    def _default_prediction(self) -> PredictionResult:
        """Return default prediction when model can't predict."""
        return PredictionResult(
            action="FLAT",
            confidence=0.33,
            probability_long=0.33,
            probability_short=0.33,
            probability_flat=0.34,
            expected_return=0.0,
            features_used=0,
            model_type=self.model_type,
        )

    def save(self, name: str = "ml_predictor"):
        """Save model and scaler to disk."""
        model_path = self.model_dir / f"{name}_model.pkl"
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        meta_path = self.model_dir / f"{name}_meta.json"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "cross_val_mean": self.metrics.cross_val_mean,
                "cross_val_std": self.metrics.cross_val_std,
                "train_samples": self.metrics.train_samples,
                "test_samples": self.metrics.test_samples,
            } if self.metrics else None,
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {self.model_dir}")

    def load(self, name: str = "ml_predictor") -> bool:
        """Load model and scaler from disk."""
        model_path = self.model_dir / f"{name}_model.pkl"
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        meta_path = self.model_dir / f"{name}_meta.json"

        if not all(p.exists() for p in [model_path, scaler_path, meta_path]):
            print(f"Model files not found at {self.model_dir}")
            return False

        # Try joblib first (newer models), fall back to pickle
        # joblib can load both joblib and pickle files
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.model_type = meta["model_type"]
        self.feature_names = meta["feature_names"]
        self.is_trained = meta["is_trained"]

        print(f"Model loaded from {self.model_dir}")
        return True

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if not self.metrics or not self.metrics.feature_importance:
            return []

        sorted_features = sorted(
            self.metrics.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]

    def backtest_predictions(
        self,
        ohlcv: pd.DataFrame,
        initial_balance: float = 10000.0,
    ) -> Dict:
        """
        Simple backtest of ML predictions.

        Returns metrics on how predictions would have performed.
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        df = self.feature_engineer.extract_features(ohlcv)

        if len(df) < 50:
            return {"error": "Not enough data for backtest"}

        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        actual_returns = df["target_return"].values

        balance = initial_balance
        trades = 0
        wins = 0

        for i, (pred, actual_ret) in enumerate(zip(predictions, actual_returns)):
            if np.isnan(actual_ret):
                continue

            # 0=SHORT, 1=FLAT, 2=LONG
            if pred == 2:  # LONG
                pnl = actual_ret * balance * 0.1  # 10% position size
                balance += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
            elif pred == 0:  # SHORT
                pnl = -actual_ret * balance * 0.1
                balance += pnl
                trades += 1
                if pnl > 0:
                    wins += 1

        return {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "total_return_pct": round((balance / initial_balance - 1) * 100, 2),
            "total_trades": trades,
            "win_rate": round(wins / trades * 100, 2) if trades > 0 else 0,
        }


# Global instance for singleton pattern
_predictor: Optional[MLPredictor] = None


def get_predictor(
    model_type: str = "random_forest",
    model_dir: str = "data/models",
    symbol: str = "BTC_USDT"
) -> Optional[MLPredictor]:
    """
    Get or create a predictor instance.

    Attempts to load a pre-trained model if available.

    Args:
        model_type: Type of model to use
        model_dir: Directory containing saved models
        symbol: Symbol for the model

    Returns:
        MLPredictor instance or None if unavailable
    """
    global _predictor

    if _predictor is not None:
        return _predictor

    try:
        predictor = MLPredictor(model_type=model_type, model_dir=model_dir)

        # Try to load saved model
        model_name = f"{symbol}_{model_type}"
        if predictor.load(model_name):
            _predictor = predictor
            return _predictor

        # Try without symbol prefix
        if predictor.load(model_type):
            _predictor = predictor
            return _predictor

        # Return untrained predictor
        _predictor = predictor
        return _predictor

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not create predictor: {e}")
        return None
