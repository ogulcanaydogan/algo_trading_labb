"""
Tests for Enhanced ML Predictor.

Tests feature selection, binary/ternary classification, ensemble methods,
and regime-aware predictions.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from bot.ml.enhanced_predictor import (
    EnhancedMLPredictor,
    EnhancedModelConfig,
    EnhancedPredictionResult,
    FeatureSelector,
    create_enhanced_predictor,
)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_samples = 500

    # Generate realistic price data with trend
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some trend
    trend = np.linspace(0, 0.1, n_samples)
    prices = prices * (1 + trend)

    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1h")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_samples)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_samples)),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    return df


class TestFeatureSelector:
    """Tests for FeatureSelector."""

    def test_init(self):
        """Test feature selector initialization."""
        selector = FeatureSelector(method="importance", max_features=20)
        assert selector.method == "importance"
        assert selector.max_features == 20
        assert selector.selected_features == []

    def test_select_by_importance(self):
        """Test feature selection by importance."""
        np.random.seed(42)
        X = np.random.randn(200, 50)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # First 2 features are important
        feature_names = [f"feature_{i}" for i in range(50)]

        selector = FeatureSelector(method="importance", max_features=10)
        selected = selector.fit(X, y, feature_names)

        assert len(selected) == 10
        assert len(selector.feature_scores) == 50

    def test_transform(self):
        """Test feature transformation."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)
        feature_names = [f"f_{i}" for i in range(20)]

        selector = FeatureSelector(method="importance", max_features=5)
        selector.fit(X, y, feature_names)

        X_transformed = selector.transform(X, feature_names)
        assert X_transformed.shape[1] == 5


class TestEnhancedModelConfig:
    """Tests for EnhancedModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancedModelConfig()

        assert config.classification_type == "binary"
        assert config.return_threshold == 0.002
        assert config.forward_periods == 5
        assert config.enable_feature_selection is True
        assert config.max_features == 30
        assert config.enable_ensemble is True
        assert config.enable_regime_aware is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnhancedModelConfig(
            classification_type="ternary",
            return_threshold=0.005,
            max_features=50,
            enable_ensemble=False,
        )

        assert config.classification_type == "ternary"
        assert config.return_threshold == 0.005
        assert config.max_features == 50
        assert config.enable_ensemble is False


class TestEnhancedMLPredictor:
    """Tests for EnhancedMLPredictor."""

    def test_init_binary(self):
        """Test initialization with binary classification."""
        config = EnhancedModelConfig(classification_type="binary")
        predictor = EnhancedMLPredictor(config=config)

        assert predictor.config.classification_type == "binary"
        assert predictor.is_trained is False
        assert predictor.model is not None

    def test_init_ternary(self):
        """Test initialization with ternary classification."""
        config = EnhancedModelConfig(classification_type="ternary")
        predictor = EnhancedMLPredictor(config=config)

        assert predictor.config.classification_type == "ternary"

    def test_train_binary(self, sample_ohlcv):
        """Test training with binary classification."""
        config = EnhancedModelConfig(
            classification_type="binary",
            enable_ensemble=False,  # Faster for testing
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)

        metrics = predictor.train(sample_ohlcv, symbol="TEST")

        assert predictor.is_trained is True
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "features_selected" in metrics
        assert metrics["classification_type"] == "binary"
        # Binary classification should be around 50% for random data
        assert 0.3 < metrics["accuracy"] < 0.7

    def test_train_ternary(self, sample_ohlcv):
        """Test training with ternary classification."""
        config = EnhancedModelConfig(
            classification_type="ternary",
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)

        metrics = predictor.train(sample_ohlcv, symbol="TEST")

        assert predictor.is_trained is True
        assert metrics["classification_type"] == "ternary"

    def test_predict_not_trained(self, sample_ohlcv):
        """Test prediction fails when not trained."""
        predictor = EnhancedMLPredictor()

        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(sample_ohlcv)

    def test_predict_binary(self, sample_ohlcv):
        """Test prediction with binary classification."""
        config = EnhancedModelConfig(
            classification_type="binary",
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)
        predictor.train(sample_ohlcv, symbol="TEST")

        result = predictor.predict(sample_ohlcv)

        assert isinstance(result, EnhancedPredictionResult)
        assert result.action in ["LONG", "SHORT", "FLAT"]
        assert 0 <= result.confidence <= 1
        assert 0 <= result.probability_up <= 1
        assert 0 <= result.probability_down <= 1
        assert result.features_used > 0
        assert result.features_selected > 0

    def test_feature_selection_reduces_features(self, sample_ohlcv):
        """Test that feature selection reduces feature count."""
        config = EnhancedModelConfig(
            enable_feature_selection=True,
            max_features=20,
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)
        metrics = predictor.train(sample_ohlcv)

        assert metrics["features_selected"] <= 20
        assert metrics["features_selected"] < metrics["features_total"]

    def test_no_feature_leakage(self, sample_ohlcv):
        """Test that no future-looking features are included."""
        config = EnhancedModelConfig(
            enable_feature_selection=True,
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)
        predictor.train(sample_ohlcv)

        # Check no leakage features
        for feature in predictor.feature_names:
            assert not feature.startswith("future_"), f"Leakage feature found: {feature}"
            assert not feature.startswith("target_"), f"Target feature found: {feature}"

    def test_get_feature_importance(self, sample_ohlcv):
        """Test feature importance retrieval."""
        config = EnhancedModelConfig(
            enable_feature_selection=True,
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)
        predictor.train(sample_ohlcv)

        importance = predictor.get_feature_importance(top_n=10)

        assert len(importance) <= 10
        assert all(isinstance(item, tuple) for item in importance)
        assert all(len(item) == 2 for item in importance)

    def test_compare_with_baseline(self, sample_ohlcv):
        """Test baseline comparison."""
        config = EnhancedModelConfig(
            enable_ensemble=False,
            enable_regime_aware=False,
        )
        predictor = EnhancedMLPredictor(config=config)
        predictor.train(sample_ohlcv)

        comparison = predictor.compare_with_baseline(sample_ohlcv, baseline_accuracy=0.47)

        assert "baseline_accuracy" in comparison
        assert "enhanced_accuracy" in comparison
        assert "improvement" in comparison
        assert "improvement_pct" in comparison


class TestEnhancedPredictionResult:
    """Tests for EnhancedPredictionResult."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EnhancedPredictionResult(
            action="LONG",
            confidence=0.65,
            probability_up=0.65,
            probability_down=0.35,
            probability_flat=0.0,
            expected_return=0.001,
            features_used=100,
            features_selected=30,
            model_type="enhanced_ensemble",
            regime="bull",
            ensemble_agreement=0.85,
        )

        d = result.to_dict()

        assert d["action"] == "LONG"
        assert d["confidence"] == 0.65
        assert d["probability_up"] == 0.65
        assert d["probability_down"] == 0.35
        assert d["model_type"] == "enhanced_ensemble"
        assert d["regime"] == "bull"
        assert "timestamp" in d


class TestCreateEnhancedPredictor:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating predictor with defaults."""
        predictor = create_enhanced_predictor()

        assert isinstance(predictor, EnhancedMLPredictor)
        assert predictor.config.classification_type == "binary"
        assert predictor.config.enable_ensemble is True

    def test_create_custom(self):
        """Test creating predictor with custom settings."""
        predictor = create_enhanced_predictor(
            classification_type="ternary",
            enable_ensemble=False,
            enable_regime_aware=False,
        )

        assert predictor.config.classification_type == "ternary"
        assert predictor.config.enable_ensemble is False
        assert predictor.config.enable_regime_aware is False
