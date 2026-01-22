"""
Model Accuracy Improvement System

Implements various techniques to improve ML model accuracy:
1. Feature selection and engineering
2. Class balancing
3. Ensemble stacking
4. Threshold optimization
5. Confidence calibration
6. Meta-learning for model selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    RFE,
    SelectFromModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    """Results from accuracy improvement."""
    original_accuracy: float
    improved_accuracy: float
    improvement_pct: float
    best_technique: str
    feature_count: int
    selected_features: List[str]
    calibrated: bool
    ensemble_used: bool


class FeatureSelector:
    """Intelligent feature selection for trading models."""

    def __init__(self, n_features: int = 30):
        self.n_features = n_features
        self.selected_features = None
        self.selector = None

    def select_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using specified method.

        Methods:
        - mutual_info: Mutual information score
        - rfe: Recursive feature elimination
        - importance: Feature importance from tree model
        - combined: Ensemble of methods
        """
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(self.n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()

        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(self.n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()

        elif method == 'importance':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            selector = SelectFromModel(model, max_features=self.n_features, prefit=True)
            X_selected = selector.transform(X)
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()

        elif method == 'combined':
            # Use voting across methods
            scores = np.zeros(X.shape[1])

            # Mutual info scores
            mi_selector = SelectKBest(mutual_info_classif, k='all')
            mi_selector.fit(X, y)
            scores += (mi_selector.scores_ - mi_selector.scores_.min()) / (mi_selector.scores_.max() - mi_selector.scores_.min() + 1e-10)

            # Tree importance
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            importance = model.feature_importances_
            scores += (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

            # Select top features by combined score
            top_idx = np.argsort(scores)[-self.n_features:]
            selected_features = X.columns[top_idx].tolist()
            X_selected = X[selected_features].values

        else:
            raise ValueError(f"Unknown method: {method}")

        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features using {method}")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features


class ClassBalancer:
    """Handles class imbalance in trading data."""

    def __init__(self):
        self.class_weights = None

    def compute_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))

        logger.info(f"Class distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"Computed weights: {self.class_weights}")

        return self.class_weights

    def resample_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'oversample'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Resample data to balance classes."""
        from collections import Counter

        class_counts = Counter(y)
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        if method == 'oversample':
            # Oversample minority classes
            X_resampled = []
            y_resampled = []

            for cls in class_counts:
                X_cls = X[y == cls]
                y_cls = y[y == cls]

                if len(X_cls) < max_count:
                    # Random oversample
                    indices = np.random.choice(len(X_cls), max_count - len(X_cls), replace=True)
                    X_extra = X_cls.iloc[indices]
                    y_extra = y_cls.iloc[indices]
                    X_resampled.append(pd.concat([X_cls, X_extra]))
                    y_resampled.append(pd.concat([y_cls, y_extra]))
                else:
                    X_resampled.append(X_cls)
                    y_resampled.append(y_cls)

            return pd.concat(X_resampled), pd.concat(y_resampled)

        elif method == 'undersample':
            # Undersample majority classes
            X_resampled = []
            y_resampled = []

            for cls in class_counts:
                X_cls = X[y == cls]
                y_cls = y[y == cls]

                if len(X_cls) > min_count:
                    indices = np.random.choice(len(X_cls), min_count, replace=False)
                    X_resampled.append(X_cls.iloc[indices])
                    y_resampled.append(y_cls.iloc[indices])
                else:
                    X_resampled.append(X_cls)
                    y_resampled.append(y_cls)

            return pd.concat(X_resampled), pd.concat(y_resampled)

        return X, y


class AdvancedEnsemble:
    """Advanced ensemble methods for trading models."""

    def __init__(self):
        self.ensemble = None
        self.base_models = None

    def create_stacking_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> StackingClassifier:
        """Create a stacking ensemble with diverse base models."""
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
        ]

        # Try to add XGBoost if available
        try:
            from xgboost import XGBClassifier
            base_estimators.append(
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                                     use_label_encoder=False, eval_metric='mlogloss'))
            )
        except ImportError:
            pass

        # Try to add LightGBM if available
        try:
            from lightgbm import LGBMClassifier
            base_estimators.append(
                ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1))
            )
        except ImportError:
            pass

        self.ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=TimeSeriesSplit(n_splits=3),
            stack_method='predict_proba',
            n_jobs=-1
        )

        self.ensemble.fit(X_train, y_train)
        self.base_models = base_estimators

        return self.ensemble

    def create_voting_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """Create a soft voting ensemble."""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]

        try:
            from xgboost import XGBClassifier
            estimators.append(
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                                     use_label_encoder=False, eval_metric='mlogloss'))
            )
        except ImportError:
            pass

        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )

        self.ensemble.fit(X_train, y_train)
        return self.ensemble


class ThresholdOptimizer:
    """Optimizes prediction thresholds for trading signals."""

    def __init__(self):
        self.optimal_thresholds = None

    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Dict[int, float]:
        """
        Find optimal probability thresholds for each class.

        This is especially important for trading where we want
        to minimize false signals.
        """
        n_classes = y_proba.shape[1]
        optimal_thresholds = {}

        for cls in range(n_classes):
            best_threshold = 0.5
            best_score = 0

            for threshold in np.arange(0.3, 0.9, 0.05):
                # Create predictions using threshold
                y_pred = np.zeros(len(y_true))
                for i in range(len(y_true)):
                    max_proba = y_proba[i].max()
                    max_class = y_proba[i].argmax()
                    if max_proba >= threshold:
                        y_pred[i] = max_class
                    else:
                        y_pred[i] = 0  # HOLD if not confident enough

                # Calculate metric
                if metric == 'f1':
                    score = f1_score(y_true == cls, y_pred == cls, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true == cls, y_pred == cls, zero_division=0)
                else:
                    score = accuracy_score(y_true == cls, y_pred == cls)

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            optimal_thresholds[cls] = best_threshold

        self.optimal_thresholds = optimal_thresholds
        logger.info(f"Optimal thresholds: {optimal_thresholds}")

        return optimal_thresholds


class ConfidenceCalibrator:
    """Calibrates model confidence scores."""

    def __init__(self):
        self.calibrator = None

    def calibrate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'isotonic'
    ) -> CalibratedClassifierCV:
        """
        Calibrate model probabilities.

        Methods:
        - isotonic: Non-parametric calibration
        - sigmoid: Platt scaling
        """
        self.calibrator = CalibratedClassifierCV(
            model,
            method=method,
            cv=TimeSeriesSplit(n_splits=3)
        )

        self.calibrator.fit(X, y)
        return self.calibrator


class AccuracyImprover:
    """
    Main class that combines all accuracy improvement techniques.
    """

    def __init__(
        self,
        target_accuracy: float = 0.65,
        use_feature_selection: bool = True,
        use_class_balancing: bool = True,
        use_ensemble: bool = True,
        use_calibration: bool = True,
        use_threshold_optimization: bool = True
    ):
        self.target_accuracy = target_accuracy
        self.use_feature_selection = use_feature_selection
        self.use_class_balancing = use_class_balancing
        self.use_ensemble = use_ensemble
        self.use_calibration = use_calibration
        self.use_threshold_optimization = use_threshold_optimization

        self.feature_selector = FeatureSelector()
        self.class_balancer = ClassBalancer()
        self.ensemble = AdvancedEnsemble()
        self.threshold_optimizer = ThresholdOptimizer()
        self.calibrator = ConfidenceCalibrator()

    def improve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_model: Optional[BaseEstimator] = None
    ) -> Tuple[BaseEstimator, ImprovementResult]:
        """
        Apply all improvement techniques and return best model.

        Args:
            X: Feature DataFrame
            y: Target Series
            base_model: Optional base model to improve

        Returns:
            Improved model and results summary
        """
        logger.info("=" * 50)
        logger.info("Starting Accuracy Improvement Pipeline")
        logger.info("=" * 50)

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get baseline accuracy
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)

        base_model.fit(X_train_scaled, y_train)
        baseline_accuracy = accuracy_score(y_test, base_model.predict(X_test_scaled))
        logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}")

        best_accuracy = baseline_accuracy
        best_model = base_model
        best_technique = "baseline"
        selected_features = list(X.columns)

        # 1. Feature Selection
        if self.use_feature_selection:
            logger.info("\n--- Feature Selection ---")
            X_selected, selected_features = self.feature_selector.select_best_features(
                X_train, y_train, method='combined'
            )

            X_train_sel = scaler.fit_transform(X_selected)
            X_test_sel = scaler.transform(X_test[selected_features])

            model_fs = clone(base_model)
            model_fs.fit(X_train_sel, y_train)
            acc_fs = accuracy_score(y_test, model_fs.predict(X_test_sel))
            logger.info(f"With feature selection: {acc_fs:.2%}")

            if acc_fs > best_accuracy:
                best_accuracy = acc_fs
                best_model = model_fs
                best_technique = "feature_selection"
                X_train_scaled = X_train_sel
                X_test_scaled = X_test_sel

        # 2. Class Balancing
        if self.use_class_balancing:
            logger.info("\n--- Class Balancing ---")
            class_weights = self.class_balancer.compute_weights(y_train)

            if hasattr(base_model, 'class_weight'):
                model_cw = clone(base_model)
                model_cw.set_params(class_weight=class_weights)
                model_cw.fit(X_train_scaled, y_train)
                acc_cw = accuracy_score(y_test, model_cw.predict(X_test_scaled))
                logger.info(f"With class weights: {acc_cw:.2%}")

                if acc_cw > best_accuracy:
                    best_accuracy = acc_cw
                    best_model = model_cw
                    best_technique = "class_balancing"

        # 3. Ensemble Methods
        if self.use_ensemble:
            logger.info("\n--- Ensemble Methods ---")

            # Stacking ensemble
            try:
                stacking = self.ensemble.create_stacking_ensemble(X_train_scaled, y_train)
                acc_stack = accuracy_score(y_test, stacking.predict(X_test_scaled))
                logger.info(f"Stacking ensemble: {acc_stack:.2%}")

                if acc_stack > best_accuracy:
                    best_accuracy = acc_stack
                    best_model = stacking
                    best_technique = "stacking_ensemble"
            except Exception as e:
                logger.warning(f"Stacking failed: {e}")

            # Voting ensemble
            try:
                voting = self.ensemble.create_voting_ensemble(X_train_scaled, y_train)
                acc_vote = accuracy_score(y_test, voting.predict(X_test_scaled))
                logger.info(f"Voting ensemble: {acc_vote:.2%}")

                if acc_vote > best_accuracy:
                    best_accuracy = acc_vote
                    best_model = voting
                    best_technique = "voting_ensemble"
            except Exception as e:
                logger.warning(f"Voting failed: {e}")

        # 4. Confidence Calibration
        calibrated = False
        if self.use_calibration and hasattr(best_model, 'predict_proba'):
            logger.info("\n--- Confidence Calibration ---")
            try:
                calibrated_model = self.calibrator.calibrate(
                    clone(best_model), X_train_scaled, y_train
                )
                acc_cal = accuracy_score(y_test, calibrated_model.predict(X_test_scaled))
                logger.info(f"After calibration: {acc_cal:.2%}")

                if acc_cal > best_accuracy:
                    best_accuracy = acc_cal
                    best_model = calibrated_model
                    best_technique += "_calibrated"
                    calibrated = True
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")

        # 5. Threshold Optimization
        if self.use_threshold_optimization and hasattr(best_model, 'predict_proba'):
            logger.info("\n--- Threshold Optimization ---")
            try:
                y_proba = best_model.predict_proba(X_test_scaled)
                optimal_thresholds = self.threshold_optimizer.optimize_thresholds(
                    y_test.values, y_proba
                )

                # Apply optimal thresholds
                y_pred_opt = np.zeros(len(y_test))
                for i in range(len(y_test)):
                    max_proba = y_proba[i].max()
                    max_class = y_proba[i].argmax()
                    threshold = optimal_thresholds.get(max_class, 0.5)
                    if max_proba >= threshold:
                        y_pred_opt[i] = max_class

                acc_opt = accuracy_score(y_test, y_pred_opt)
                logger.info(f"With optimal thresholds: {acc_opt:.2%}")

                if acc_opt > best_accuracy:
                    best_accuracy = acc_opt
                    best_technique += "_threshold_opt"
            except Exception as e:
                logger.warning(f"Threshold optimization failed: {e}")

        # Summary
        improvement_pct = ((best_accuracy - baseline_accuracy) / baseline_accuracy) * 100

        logger.info("\n" + "=" * 50)
        logger.info("IMPROVEMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}")
        logger.info(f"Best accuracy: {best_accuracy:.2%}")
        logger.info(f"Improvement: {improvement_pct:+.1f}%")
        logger.info(f"Best technique: {best_technique}")
        logger.info(f"Features used: {len(selected_features)}")

        result = ImprovementResult(
            original_accuracy=baseline_accuracy,
            improved_accuracy=best_accuracy,
            improvement_pct=improvement_pct,
            best_technique=best_technique,
            feature_count=len(selected_features),
            selected_features=selected_features,
            calibrated=calibrated,
            ensemble_used='ensemble' in best_technique
        )

        return best_model, result


def improve_model_accuracy(
    X: pd.DataFrame,
    y: pd.Series,
    base_model: Optional[BaseEstimator] = None,
    target_accuracy: float = 0.65
) -> Tuple[BaseEstimator, ImprovementResult]:
    """
    Convenience function to improve model accuracy.

    Args:
        X: Feature DataFrame
        y: Target Series
        base_model: Optional base model
        target_accuracy: Target accuracy to achieve

    Returns:
        Improved model and results
    """
    improver = AccuracyImprover(target_accuracy=target_accuracy)
    return improver.improve(X, y, base_model)
