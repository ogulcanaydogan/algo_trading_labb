"""
Model Interpretability - SHAP and feature importance analysis.

Provides explanations for ML model predictions to understand
what drives trading decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")


@dataclass
class FeatureImportance:
    """Feature importance result."""

    feature_name: str
    importance: float
    rank: int
    direction: str  # "positive", "negative", "mixed"

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "importance": round(self.importance, 6),
            "rank": self.rank,
            "direction": self.direction,
        }


@dataclass
class PredictionExplanation:
    """Explanation for a single prediction."""

    prediction: float
    prediction_label: str
    confidence: float
    base_value: float  # Expected value
    top_features: List[Dict[str, Any]]
    feature_contributions: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "prediction": round(self.prediction, 4),
            "prediction_label": self.prediction_label,
            "confidence": round(self.confidence, 4),
            "base_value": round(self.base_value, 4),
            "top_features": self.top_features,
            "feature_contributions": {
                k: round(v, 6) for k, v in self.feature_contributions.items()
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelInsights:
    """Overall model insights."""

    model_type: str
    feature_importances: List[FeatureImportance]
    top_positive_features: List[str]
    top_negative_features: List[str]
    interaction_effects: List[Dict[str, Any]]
    stability_score: float  # How stable are feature importances
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "model_type": self.model_type,
            "feature_importances": [f.to_dict() for f in self.feature_importances[:20]],
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "interaction_effects": self.interaction_effects[:10],
            "stability_score": round(self.stability_score, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class ModelInterpreter:
    """
    Interpret ML model predictions using SHAP and other techniques.

    Features:
    - SHAP value calculation
    - Feature importance ranking
    - Prediction explanations
    - Interaction detection
    """

    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize interpreter.

        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._shap_values_cache: Dict[str, np.ndarray] = {}
        self._background_data: Optional[np.ndarray] = None

    def set_background_data(self, X: Union[np.ndarray, pd.DataFrame], sample_size: int = 100):
        """
        Set background data for SHAP calculations.

        Args:
            X: Training data
            sample_size: Number of samples to use as background
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Sample if dataset is large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            self._background_data = X[indices]
        else:
            self._background_data = X

        # Create explainer
        self._create_explainer()

    def _create_explainer(self):
        """Create SHAP explainer based on model type."""
        if not HAS_SHAP or self._background_data is None:
            return

        model_type = type(self.model).__name__.lower()

        try:
            if (
                "tree" in model_type
                or "forest" in model_type
                or "xgb" in model_type
                or "lgb" in model_type
            ):
                # Tree-based models
                self._explainer = shap.TreeExplainer(self.model)
            elif "linear" in model_type or "logistic" in model_type:
                # Linear models
                self._explainer = shap.LinearExplainer(self.model, self._background_data)
            else:
                # Generic kernel explainer (slower)
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict,
                    self._background_data,
                )
            logger.info(f"Created SHAP explainer for {model_type}")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self._explainer = None

    def calculate_shap_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for predictions.

        Args:
            X: Input features

        Returns:
            SHAP values array
        """
        if not HAS_SHAP or self._explainer is None:
            return None

        if isinstance(X, pd.DataFrame):
            X = X.values

        try:
            shap_values = self._explainer.shap_values(X)

            # Handle multi-class output
            if isinstance(shap_values, list):
                # For binary classification, use class 1
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

            return shap_values

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None

    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_n: int = 10,
    ) -> PredictionExplanation:
        """
        Explain a single prediction.

        Args:
            X: Single sample features
            top_n: Number of top features to include

        Returns:
            PredictionExplanation with detailed breakdown
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get prediction
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            prediction = proba[1] if len(proba) == 2 else max(proba)
            pred_class = self.model.predict(X)[0]
        else:
            prediction = self.model.predict(X)[0]
            pred_class = 1 if prediction > 0.5 else 0

        # Map prediction to label
        if pred_class == 1:
            label = "BUY"
        elif pred_class == -1 or pred_class == 2:
            label = "SELL"
        else:
            label = "HOLD"

        # Calculate confidence
        confidence = abs(prediction - 0.5) * 2 if 0 <= prediction <= 1 else abs(prediction)

        # Get SHAP values
        shap_values = self.calculate_shap_values(X)

        if shap_values is not None:
            shap_values = shap_values.flatten()
            base_value = self._explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) == 2 else base_value[0]

            # Get feature contributions
            feature_contributions = dict(zip(self.feature_names, shap_values))

            # Sort by absolute importance
            sorted_features = sorted(
                zip(self.feature_names, shap_values, X.flatten()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            top_features = []
            for name, shap_val, feat_val in sorted_features[:top_n]:
                top_features.append(
                    {
                        "feature": name,
                        "value": float(feat_val),
                        "shap_value": float(shap_val),
                        "direction": "positive" if shap_val > 0 else "negative",
                    }
                )
        else:
            # Fallback to permutation importance
            base_value = 0.5
            feature_contributions = self._fallback_importance(X)
            top_features = sorted(
                [
                    {
                        "feature": k,
                        "value": 0,
                        "shap_value": v,
                        "direction": "positive" if v > 0 else "negative",
                    }
                    for k, v in feature_contributions.items()
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True,
            )[:top_n]

        return PredictionExplanation(
            prediction=float(prediction),
            prediction_label=label,
            confidence=float(confidence),
            base_value=float(base_value),
            top_features=top_features,
            feature_contributions=feature_contributions,
        )

    def _fallback_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Fallback importance when SHAP unavailable."""
        # Use model's feature importance if available
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))

        # Use coefficients for linear models
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_.flatten()
            return dict(zip(self.feature_names, coefs))

        return {name: 0.0 for name in self.feature_names}

    def get_feature_importances(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = "shap",
    ) -> List[FeatureImportance]:
        """
        Get global feature importances.

        Args:
            X: Dataset for importance calculation
            method: "shap", "permutation", or "model"

        Returns:
            List of FeatureImportance objects
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if method == "shap" and HAS_SHAP and self._explainer is not None:
            shap_values = self.calculate_shap_values(X)
            if shap_values is not None:
                # Mean absolute SHAP values
                importances = np.abs(shap_values).mean(axis=0)
                directions = np.sign(shap_values.mean(axis=0))
            else:
                return self.get_feature_importances(X, method="model")
        elif method == "model":
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                directions = np.ones(len(importances))
            elif hasattr(self.model, "coef_"):
                importances = np.abs(self.model.coef_.flatten())
                directions = np.sign(self.model.coef_.flatten())
            else:
                importances = np.zeros(len(self.feature_names))
                directions = np.zeros(len(self.feature_names))
        else:
            # Permutation importance
            importances = self._permutation_importance(X)
            directions = np.ones(len(importances))

        # Create FeatureImportance objects
        results = []
        sorted_indices = np.argsort(importances)[::-1]

        for rank, idx in enumerate(sorted_indices):
            direction = (
                "positive"
                if directions[idx] > 0
                else "negative"
                if directions[idx] < 0
                else "mixed"
            )
            results.append(
                FeatureImportance(
                    feature_name=self.feature_names[idx],
                    importance=float(importances[idx]),
                    rank=rank + 1,
                    direction=direction,
                )
            )

        return results

    def _permutation_importance(self, X: np.ndarray, n_repeats: int = 10) -> np.ndarray:
        """Calculate permutation importance."""
        if hasattr(self.model, "predict_proba"):
            base_score = self.model.predict_proba(X)[:, 1].mean()
        else:
            base_score = self.model.predict(X).mean()

        importances = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])

                if hasattr(self.model, "predict_proba"):
                    score = self.model.predict_proba(X_permuted)[:, 1].mean()
                else:
                    score = self.model.predict(X_permuted).mean()
                scores.append(abs(base_score - score))

            importances[i] = np.mean(scores)

        return importances

    def get_model_insights(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> ModelInsights:
        """
        Get comprehensive model insights.

        Args:
            X: Dataset for analysis

        Returns:
            ModelInsights with full analysis
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get feature importances
        importances = self.get_feature_importances(X)

        # Identify top positive and negative features
        top_positive = [f.feature_name for f in importances if f.direction == "positive"][:10]
        top_negative = [f.feature_name for f in importances if f.direction == "negative"][:10]

        # Calculate interaction effects (simplified)
        interactions = self._detect_interactions(X, importances[:10])

        # Calculate stability score
        stability = self._calculate_stability(X)

        return ModelInsights(
            model_type=type(self.model).__name__,
            feature_importances=importances,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            interaction_effects=interactions,
            stability_score=stability,
        )

    def _detect_interactions(
        self,
        X: np.ndarray,
        top_features: List[FeatureImportance],
    ) -> List[Dict[str, Any]]:
        """Detect feature interactions."""
        if not HAS_SHAP or self._explainer is None:
            return []

        interactions = []

        # Check for interactions between top features
        for i, f1 in enumerate(top_features[:5]):
            for f2 in top_features[i + 1 : 5]:
                idx1 = self.feature_names.index(f1.feature_name)
                idx2 = self.feature_names.index(f2.feature_name)

                # Calculate correlation between features
                corr = np.corrcoef(X[:, idx1], X[:, idx2])[0, 1]

                if abs(corr) > 0.5:
                    interactions.append(
                        {
                            "feature1": f1.feature_name,
                            "feature2": f2.feature_name,
                            "correlation": float(corr),
                            "interaction_type": "correlated",
                        }
                    )

        return interactions

    def _calculate_stability(self, X: np.ndarray, n_samples: int = 5) -> float:
        """Calculate feature importance stability across subsamples."""
        if len(X) < n_samples * 10:
            return 1.0

        importance_ranks = []

        for _ in range(n_samples):
            # Random subsample
            indices = np.random.choice(len(X), len(X) // 2, replace=False)
            X_sample = X[indices]

            importances = self.get_feature_importances(X_sample)
            ranks = {f.feature_name: f.rank for f in importances}
            importance_ranks.append(ranks)

        # Calculate rank stability (Kendall's W or similar)
        if len(importance_ranks) < 2:
            return 1.0

        # Simplified stability: correlation of top 10 ranks
        top_features = [f.feature_name for f in self.get_feature_importances(X)[:10]]

        rank_matrix = []
        for ranks in importance_ranks:
            rank_matrix.append([ranks.get(f, len(self.feature_names)) for f in top_features])

        rank_matrix = np.array(rank_matrix)

        # Average pairwise correlation
        correlations = []
        for i in range(len(rank_matrix)):
            for j in range(i + 1, len(rank_matrix)):
                corr = np.corrcoef(rank_matrix[i], rank_matrix[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 1.0

    def generate_decision_rules(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        max_depth: int = 3,
    ) -> List[str]:
        """
        Generate human-readable decision rules using a surrogate tree.

        Args:
            X: Features
            y: Labels (or model predictions)
            max_depth: Maximum tree depth

        Returns:
            List of decision rules as strings
        """
        from sklearn.tree import DecisionTreeClassifier, export_text

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Train surrogate decision tree
        surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        surrogate.fit(X, y)

        # Export rules
        rules_text = export_text(surrogate, feature_names=self.feature_names)

        # Parse into list of rules
        rules = []
        current_rule = []

        for line in rules_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if "class:" in line:
                if current_rule:
                    class_label = line.split("class:")[1].strip()
                    rules.append(f"IF {' AND '.join(current_rule)} THEN {class_label}")
                current_rule = []
            elif "<=" in line or ">" in line:
                # Clean up the condition
                condition = line.replace("|", "").replace("---", "").strip()
                if condition:
                    current_rule.append(condition)

        return rules[:20]  # Limit number of rules


def create_interpreter(model: Any, feature_names: List[str]) -> ModelInterpreter:
    """Factory function to create model interpreter."""
    return ModelInterpreter(model=model, feature_names=feature_names)
