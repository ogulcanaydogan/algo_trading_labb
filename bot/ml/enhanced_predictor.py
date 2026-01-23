"""
Enhanced ML Predictor with improved accuracy techniques.

Features:
- Feature selection to reduce overfitting
- Binary classification option (UP/DOWN)
- Better target definition with forward returns and thresholds
- Ensemble methods combining multiple models
- Regime-aware model selection
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        VotingClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.feature_selection import (
        SelectFromModel,
        RFE,
        mutual_info_classif,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .feature_engineer import FeatureEngineer, FeatureConfig
from .regime_classifier import MarketRegimeClassifier, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPredictionResult:
    """Result from enhanced ML prediction."""
    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    probability_up: float
    probability_down: float
    probability_flat: float  # 0.0 for binary classification
    expected_return: float
    features_used: int
    features_selected: int  # After feature selection
    model_type: str
    regime: Optional[str] = None
    ensemble_agreement: float = 1.0  # Agreement among ensemble members
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "probability_up": round(self.probability_up, 4),
            "probability_down": round(self.probability_down, 4),
            "probability_flat": round(self.probability_flat, 4),
            "expected_return": round(self.expected_return, 6),
            "features_used": self.features_used,
            "features_selected": self.features_selected,
            "model_type": self.model_type,
            "regime": self.regime,
            "ensemble_agreement": round(self.ensemble_agreement, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EnhancedModelConfig:
    """Configuration for enhanced ML predictor."""
    # Classification type
    classification_type: Literal["binary", "ternary"] = "binary"

    # Target definition
    return_threshold: float = 0.002  # 0.2% threshold for UP/DOWN
    forward_periods: int = 5  # Predict 5 periods ahead

    # Feature selection
    enable_feature_selection: bool = True
    max_features: int = 30  # Maximum features to keep
    feature_selection_method: Literal["importance", "rfe", "mutual_info"] = "importance"

    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_method: Literal["voting", "stacking"] = "voting"

    # Regime-aware settings
    enable_regime_aware: bool = True

    # Model hyperparameters
    n_estimators: int = 200
    max_depth: int = 8
    min_samples_split: int = 10
    min_samples_leaf: int = 5

    # Training settings
    test_size: float = 0.2
    cv_folds: int = 5


class FeatureSelector:
    """Feature selection to reduce overfitting."""

    def __init__(
        self,
        method: Literal["importance", "rfe", "mutual_info"] = "importance",
        max_features: int = 30,
    ):
        self.method = method
        self.max_features = max_features
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        base_model=None,
    ) -> List[str]:
        """
        Select most important features.

        Returns:
            List of selected feature names
        """
        if len(feature_names) <= self.max_features:
            self.selected_features = feature_names
            return feature_names

        if self.method == "importance":
            return self._select_by_importance(X, y, feature_names, base_model)
        elif self.method == "rfe":
            return self._select_by_rfe(X, y, feature_names, base_model)
        elif self.method == "mutual_info":
            return self._select_by_mutual_info(X, y, feature_names)
        else:
            return feature_names[:self.max_features]

    def _select_by_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        base_model=None,
    ) -> List[str]:
        """Select features by model importance."""
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

        base_model.fit(X, y)
        importances = base_model.feature_importances_

        # Store scores
        for name, score in zip(feature_names, importances):
            self.feature_scores[name] = float(score)

        # Select top features
        indices = np.argsort(importances)[::-1][:self.max_features]
        self.selected_features = [feature_names[i] for i in indices]

        logger.info(f"Selected {len(self.selected_features)} features by importance")
        return self.selected_features

    def _select_by_rfe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        base_model=None,
    ) -> List[str]:
        """Select features by Recursive Feature Elimination."""
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

        rfe = RFE(base_model, n_features_to_select=self.max_features, step=10)
        rfe.fit(X, y)

        self.selected_features = [
            name for name, selected in zip(feature_names, rfe.support_)
            if selected
        ]

        # Store rankings
        for name, rank in zip(feature_names, rfe.ranking_):
            self.feature_scores[name] = 1.0 / rank

        logger.info(f"Selected {len(self.selected_features)} features by RFE")
        return self.selected_features

    def _select_by_mutual_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """Select features by mutual information."""
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Store scores
        for name, score in zip(feature_names, mi_scores):
            self.feature_scores[name] = float(score)

        # Select top features
        indices = np.argsort(mi_scores)[::-1][:self.max_features]
        self.selected_features = [feature_names[i] for i in indices]

        logger.info(f"Selected {len(self.selected_features)} features by mutual info")
        return self.selected_features

    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Transform X to only include selected features."""
        if not self.selected_features:
            return X

        indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
        return X[:, indices]


class EnhancedMLPredictor:
    """
    Enhanced Machine Learning Predictor with improved accuracy.

    Improvements over base predictor:
    1. Feature selection to reduce overfitting
    2. Binary classification option for clearer signals
    3. Better target definition with configurable thresholds
    4. Ensemble methods for more robust predictions
    5. Regime-aware model selection
    """

    def __init__(
        self,
        config: Optional[EnhancedModelConfig] = None,
        model_dir: str = "data/models",
    ):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")

        self.config = config or EnhancedModelConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector(
            method=self.config.feature_selection_method,
            max_features=self.config.max_features,
        )
        self.scaler = StandardScaler()
        self.regime_classifier = MarketRegimeClassifier() if self.config.enable_regime_aware else None

        self.model = None
        self.regime_models: Dict[str, any] = {}  # Regime-specific models
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.is_trained = False
        self.metrics: Dict = {}

        self._init_models()

    def _init_models(self):
        """Initialize ML models."""
        base_rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )

        base_gb = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators // 2,
            max_depth=self.config.max_depth - 2,
            learning_rate=0.05,
            random_state=42,
        )

        if self.config.enable_ensemble:
            if HAS_XGBOOST:
                n_classes = 2 if self.config.classification_type == "binary" else 3
                base_xgb = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=0.05,
                    objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                    num_class=n_classes if n_classes > 2 else None,
                    random_state=42,
                    n_jobs=-1,
                )
                estimators = [
                    ("rf", base_rf),
                    ("gb", base_gb),
                    ("xgb", base_xgb),
                ]
            else:
                estimators = [
                    ("rf", base_rf),
                    ("gb", base_gb),
                ]

            if self.config.ensemble_method == "voting":
                self.model = VotingClassifier(
                    estimators=estimators,
                    voting="soft",
                    n_jobs=-1,
                )
            else:  # stacking
                self.model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3,
                    n_jobs=-1,
                )
        else:
            self.model = base_rf

    def _create_targets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create target labels with improved definition.

        Uses forward returns with configurable threshold.
        """
        # Calculate forward returns
        df = df.copy()
        df["forward_return"] = df["close"].pct_change(self.config.forward_periods).shift(-self.config.forward_periods)

        if self.config.classification_type == "binary":
            # Binary: UP (1) or DOWN (0)
            df["target"] = (df["forward_return"] > 0).astype(int)
        else:
            # Ternary: UP (2), FLAT (1), DOWN (0)
            threshold = self.config.return_threshold
            df["target"] = 1  # Default FLAT
            df.loc[df["forward_return"] > threshold, "target"] = 2  # UP
            df.loc[df["forward_return"] < -threshold, "target"] = 0  # DOWN

        return df

    def train(
        self,
        ohlcv: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> Dict:
        """
        Train the enhanced ML model.

        Args:
            ohlcv: OHLCV DataFrame
            symbol: Symbol name for logging

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training enhanced model for {symbol or 'unknown'}...")

        # Extract features
        df = self.feature_engineer.extract_features(ohlcv)
        df = self._create_targets(df)
        df = df.dropna()

        if len(df) < 200:
            raise ValueError(f"Insufficient data: {len(df)} rows (need 200+)")

        # Get feature columns (exclude target, price, and future-looking columns to prevent leakage)
        exclude_cols = {
            # Target columns
            "target", "forward_return", "target_return", "target_class",
            "target_triple_barrier", "target_direction", "target_strong_trend",
            "target_risk_adjusted",
            # Price columns
            "open", "high", "low", "close", "volume",
        }
        # Also exclude any column starting with 'future_' or 'target_' (leakage prevention)
        self.feature_names = [
            c for c in df.columns
            if c not in exclude_cols
            and not c.startswith("future_")
            and not c.startswith("target_")
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        X = df[self.feature_names].values
        y = df["target"].values

        # Time-series split
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Feature selection
        if self.config.enable_feature_selection:
            self.selected_features = self.feature_selector.fit(
                X_train_scaled, y_train, self.feature_names
            )
            X_train_selected = self.feature_selector.transform(X_train_scaled, self.feature_names)
            X_test_selected = self.feature_selector.transform(X_test_scaled, self.feature_names)
        else:
            self.selected_features = self.feature_names
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled

        # Train main model
        logger.info(f"Training with {len(self.selected_features)} features...")
        self.model.fit(X_train_selected, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_selected)
        y_proba = self.model.predict_proba(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_selected, y_train,
            cv=min(self.config.cv_folds, len(y_train) // 50),
            scoring="accuracy"
        )

        # Class distribution analysis
        unique, counts = np.unique(y_test, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts.astype(int)))

        # Calculate directional accuracy (for trading relevance)
        if self.config.classification_type == "binary":
            directional_accuracy = accuracy
        else:
            # For ternary, measure if we got direction right (UP vs DOWN)
            mask = (y_test != 1) & (y_pred != 1)  # Non-FLAT
            if mask.sum() > 0:
                directional_accuracy = (y_test[mask] == y_pred[mask]).mean()
            else:
                directional_accuracy = 0.5

        self.metrics = {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "directional_accuracy": round(directional_accuracy, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_total": len(self.feature_names),
            "features_selected": len(self.selected_features),
            "class_distribution": class_dist,
            "classification_type": self.config.classification_type,
            "trained_at": datetime.now().isoformat(),
        }

        # Train regime-specific models if enabled
        if self.config.enable_regime_aware and self.regime_classifier:
            self._train_regime_models(df, X_train_scaled, y_train)

        self.is_trained = True
        logger.info(
            f"Training complete: accuracy={accuracy:.4f}, f1={f1:.4f}, "
            f"directional={directional_accuracy:.4f}, cv={cv_scores.mean():.4f}"
        )

        return self.metrics

    def _train_regime_models(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ):
        """Train separate models for different market regimes."""
        # Add regime labels to training data
        regimes = []
        for i in range(len(df)):
            if i < 200:
                regimes.append("unknown")
                continue

            window = df.iloc[max(0, i-200):i+1]
            try:
                analysis = self.regime_classifier.classify(window)
                regimes.append(analysis.regime.value)
            except Exception:
                regimes.append("unknown")

        df_temp = df.copy()
        df_temp["regime"] = regimes[:len(df)]

        # Train models for each regime
        for regime in ["bull", "bear", "sideways"]:
            mask = df_temp["regime"].str.contains(regime, case=False, na=False)
            if mask.sum() < 100:
                continue

            X_regime = X_scaled[mask.values[:len(X_scaled)]]
            y_regime = y[mask.values[:len(y)]]

            if len(X_regime) < 100:
                continue

            # Apply feature selection
            X_regime_selected = self.feature_selector.transform(X_regime, self.feature_names)

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X_regime_selected, y_regime)
            self.regime_models[regime] = model
            logger.info(f"Trained regime model: {regime} ({len(X_regime)} samples)")

    def predict(self, ohlcv: pd.DataFrame) -> EnhancedPredictionResult:
        """
        Make a prediction with the enhanced model.

        Args:
            ohlcv: Recent OHLCV data (100+ bars recommended)

        Returns:
            EnhancedPredictionResult
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features
        df = self.feature_engineer.extract_features(ohlcv)

        if len(df) == 0:
            return self._default_prediction()

        # Get current regime
        current_regime = None
        if self.config.enable_regime_aware and self.regime_classifier:
            try:
                regime_analysis = self.regime_classifier.classify(ohlcv)
                current_regime = regime_analysis.regime.value
            except Exception:
                pass

        # Prepare features
        X = df[self.feature_names].iloc[-1:].values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled, self.feature_names)

        # Get predictions from main model
        main_proba = self.model.predict_proba(X_selected)[0]

        # Get predictions from regime-specific model if available
        ensemble_agreement = 1.0
        if current_regime and current_regime in self.regime_models:
            regime_proba = self.regime_models[current_regime].predict_proba(X_selected)[0]
            # Average probabilities
            final_proba = (main_proba + regime_proba) / 2
            ensemble_agreement = 1 - np.abs(main_proba - regime_proba).mean()
        else:
            final_proba = main_proba

        # Map probabilities to action
        if self.config.classification_type == "binary":
            prob_down = final_proba[0] if len(final_proba) > 0 else 0.5
            prob_up = final_proba[1] if len(final_proba) > 1 else 0.5
            prob_flat = 0.0

            if prob_up > 0.55:
                action = "LONG"
                confidence = prob_up
            elif prob_down > 0.55:
                action = "SHORT"
                confidence = prob_down
            else:
                action = "FLAT"
                confidence = max(prob_up, prob_down)
        else:
            prob_down = final_proba[0] if len(final_proba) > 0 else 0.33
            prob_flat = final_proba[1] if len(final_proba) > 1 else 0.34
            prob_up = final_proba[2] if len(final_proba) > 2 else 0.33

            action_idx = np.argmax(final_proba)
            action_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
            action = action_map.get(action_idx, "FLAT")
            confidence = float(final_proba[action_idx])

        # Estimate expected return
        recent_returns = ohlcv["close"].pct_change().tail(20)
        expected_return = float(
            recent_returns.mean() * (1 if action == "LONG" else -1 if action == "SHORT" else 0)
        )

        return EnhancedPredictionResult(
            action=action,
            confidence=confidence,
            probability_up=float(prob_up),
            probability_down=float(prob_down),
            probability_flat=float(prob_flat),
            expected_return=expected_return,
            features_used=len(self.feature_names),
            features_selected=len(self.selected_features),
            model_type="enhanced_ensemble" if self.config.enable_ensemble else "enhanced_rf",
            regime=current_regime,
            ensemble_agreement=ensemble_agreement,
        )

    def _default_prediction(self) -> EnhancedPredictionResult:
        """Return default prediction when model can't predict."""
        return EnhancedPredictionResult(
            action="FLAT",
            confidence=0.5,
            probability_up=0.5,
            probability_down=0.5,
            probability_flat=0.0,
            expected_return=0.0,
            features_used=0,
            features_selected=0,
            model_type="default",
        )

    def save(self, name: str = "enhanced_predictor"):
        """Save model to disk."""
        import joblib

        model_path = self.model_dir / f"{name}_model.pkl"
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        meta_path = self.model_dir / f"{name}_meta.json"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save regime models
        if self.regime_models:
            for regime, model in self.regime_models.items():
                regime_path = self.model_dir / f"{name}_regime_{regime}.pkl"
                joblib.dump(model, regime_path)

        meta = {
            "config": {
                "classification_type": self.config.classification_type,
                "return_threshold": self.config.return_threshold,
                "forward_periods": self.config.forward_periods,
                "enable_feature_selection": self.config.enable_feature_selection,
                "max_features": self.config.max_features,
                "enable_ensemble": self.config.enable_ensemble,
                "enable_regime_aware": self.config.enable_regime_aware,
            },
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "feature_scores": self.feature_selector.feature_scores,
            "regime_models": list(self.regime_models.keys()),
            "metrics": self.metrics,
            "is_trained": self.is_trained,
            "saved_at": datetime.now().isoformat(),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Enhanced model saved to {self.model_dir}")

    def load(self, name: str = "enhanced_predictor") -> bool:
        """Load model from disk."""
        import joblib

        model_path = self.model_dir / f"{name}_model.pkl"
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        meta_path = self.model_dir / f"{name}_meta.json"

        if not all(p.exists() for p in [model_path, scaler_path, meta_path]):
            logger.warning(f"Model files not found at {self.model_dir}")
            return False

        # joblib can load both joblib and pickle files
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.feature_names = meta["feature_names"]
        self.selected_features = meta["selected_features"]
        self.feature_selector.selected_features = self.selected_features
        self.feature_selector.feature_scores = meta.get("feature_scores", {})
        self.metrics = meta.get("metrics", {})
        self.is_trained = meta["is_trained"]

        # Load regime models
        for regime in meta.get("regime_models", []):
            regime_path = self.model_dir / f"{name}_regime_{regime}.pkl"
            if regime_path.exists():
                self.regime_models[regime] = joblib.load(regime_path)

        logger.info(f"Enhanced model loaded from {self.model_dir}")
        return True

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if not self.feature_selector.feature_scores:
            return []

        sorted_features = sorted(
            self.feature_selector.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]

    def compare_with_baseline(
        self,
        ohlcv: pd.DataFrame,
        baseline_accuracy: float = 0.47,
    ) -> Dict:
        """
        Compare enhanced model performance with baseline.

        Args:
            ohlcv: Test data
            baseline_accuracy: Baseline model accuracy to compare against

        Returns:
            Comparison metrics
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        current_accuracy = self.metrics.get("accuracy", 0)
        improvement = current_accuracy - baseline_accuracy
        improvement_pct = (improvement / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0

        return {
            "baseline_accuracy": baseline_accuracy,
            "enhanced_accuracy": current_accuracy,
            "improvement": round(improvement, 4),
            "improvement_pct": round(improvement_pct, 2),
            "features_reduced_from": len(self.feature_names),
            "features_reduced_to": len(self.selected_features),
            "feature_reduction_pct": round(
                (1 - len(self.selected_features) / max(1, len(self.feature_names))) * 100, 2
            ),
        }


def create_enhanced_predictor(
    classification_type: Literal["binary", "ternary"] = "binary",
    enable_feature_selection: bool = True,
    enable_ensemble: bool = True,
    enable_regime_aware: bool = True,
    model_dir: str = "data/models",
) -> EnhancedMLPredictor:
    """Factory function to create enhanced predictor with common settings."""
    config = EnhancedModelConfig(
        classification_type=classification_type,
        enable_feature_selection=enable_feature_selection,
        enable_ensemble=enable_ensemble,
        enable_regime_aware=enable_regime_aware,
    )
    return EnhancedMLPredictor(config=config, model_dir=model_dir)
