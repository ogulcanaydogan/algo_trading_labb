"""
Tests for ML Model Ensemble module.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

from bot.ml.ensemble import (
    ModelPrediction,
    EnsemblePrediction,
    ModelEnsemble,
    RegimeBasedEnsemble,
)


class TestModelPrediction:
    """Test ModelPrediction dataclass."""

    def test_basic_creation(self):
        """Test creating a basic model prediction."""
        pred = ModelPrediction(
            model_name="test_model",
            signal="LONG",
            confidence=0.85,
        )
        assert pred.model_name == "test_model"
        assert pred.signal == "LONG"
        assert pred.confidence == 0.85
        assert pred.probabilities == {}

    def test_with_probabilities(self):
        """Test prediction with probabilities."""
        pred = ModelPrediction(
            model_name="model_a",
            signal="SHORT",
            confidence=0.7,
            probabilities={"LONG": 0.2, "SHORT": 0.7, "FLAT": 0.1},
        )
        assert pred.probabilities["SHORT"] == 0.7
        assert sum(pred.probabilities.values()) == pytest.approx(1.0)

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        pred = ModelPrediction(
            model_name="model",
            signal="FLAT",
            confidence=0.5,
        )
        assert pred.timestamp is not None
        assert isinstance(pred.timestamp, datetime)

    def test_custom_timestamp(self):
        """Test custom timestamp."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        pred = ModelPrediction(
            model_name="model",
            signal="LONG",
            confidence=0.6,
            timestamp=ts,
        )
        assert pred.timestamp == ts


class TestEnsemblePrediction:
    """Test EnsemblePrediction dataclass."""

    def test_basic_creation(self):
        """Test creating ensemble prediction."""
        model_preds = [
            ModelPrediction(model_name="m1", signal="LONG", confidence=0.8),
            ModelPrediction(model_name="m2", signal="LONG", confidence=0.7),
        ]
        pred = EnsemblePrediction(
            signal="LONG",
            confidence=0.75,
            agreement_score=1.0,
            contributing_models=["m1", "m2"],
            model_predictions=model_preds,
            voting_method="weighted",
        )
        assert pred.signal == "LONG"
        assert pred.confidence == 0.75
        assert pred.agreement_score == 1.0
        assert len(pred.contributing_models) == 2

    def test_to_dict(self):
        """Test conversion to dict."""
        model_preds = [
            ModelPrediction(model_name="m1", signal="SHORT", confidence=0.9),
        ]
        pred = EnsemblePrediction(
            signal="SHORT",
            confidence=0.9,
            agreement_score=1.0,
            contributing_models=["m1"],
            model_predictions=model_preds,
            voting_method="majority",
        )
        d = pred.to_dict()

        assert d["signal"] == "SHORT"
        assert d["confidence"] == 0.9
        assert d["agreement_score"] == 1.0
        assert d["voting_method"] == "majority"
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)

    def test_to_dict_rounding(self):
        """Test values are properly rounded in to_dict."""
        pred = EnsemblePrediction(
            signal="FLAT",
            confidence=0.123456789,
            agreement_score=0.987654321,
            contributing_models=[],
            model_predictions=[],
            voting_method="confidence",
        )
        d = pred.to_dict()

        # Should be rounded to 4 decimal places
        assert d["confidence"] == 0.1235
        assert d["agreement_score"] == 0.9877


class TestModelEnsemble:
    """Test ModelEnsemble class."""

    @pytest.fixture
    def ensemble(self):
        """Create ensemble instance."""
        return ModelEnsemble(
            min_agreement=0.5,
            decay_factor=0.95,
            performance_window=50,
        )

    @pytest.fixture
    def mock_model_proba(self):
        """Create mock model with predict_proba."""
        model = MagicMock()
        # Returns probabilities for [SHORT, FLAT, LONG]
        model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        return model

    @pytest.fixture
    def mock_model_predict(self):
        """Create mock model with only predict."""
        model = MagicMock()
        model.predict.return_value = np.array([2])  # LONG (class 2)
        # Make sure predict_proba doesn't exist
        del model.predict_proba
        return model

    def test_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert ensemble.min_agreement == 0.5
        assert ensemble.decay_factor == 0.95
        assert ensemble.performance_window == 50
        assert ensemble.models == {}
        assert ensemble.model_weights == {}

    def test_register_model(self, ensemble, mock_model_proba):
        """Test registering a model."""
        ensemble.register_model("test_model", mock_model_proba, initial_weight=1.5)

        assert "test_model" in ensemble.models
        assert ensemble.model_weights["test_model"] == 1.5
        assert "test_model" in ensemble._model_accuracy

    def test_register_multiple_models(self, ensemble, mock_model_proba):
        """Test registering multiple models."""
        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])

        ensemble.register_model("model1", mock_model_proba)
        ensemble.register_model("model2", model2, initial_weight=2.0)

        assert len(ensemble.models) == 2
        assert ensemble.model_weights["model1"] == 1.0
        assert ensemble.model_weights["model2"] == 2.0

    def test_unregister_model(self, ensemble, mock_model_proba):
        """Test unregistering a model."""
        ensemble.register_model("model", mock_model_proba)
        assert "model" in ensemble.models

        ensemble.unregister_model("model")
        assert "model" not in ensemble.models
        assert "model" not in ensemble.model_weights

    def test_unregister_nonexistent_model(self, ensemble):
        """Test unregistering nonexistent model doesn't error."""
        ensemble.unregister_model("nonexistent")
        # Should not raise

    def test_predict_no_models_raises(self, ensemble):
        """Test predict with no models raises error."""
        features = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="No models registered"):
            ensemble.predict(features)

    def test_predict_with_proba_model(self, ensemble, mock_model_proba):
        """Test prediction with predict_proba model."""
        ensemble.register_model("model", mock_model_proba)
        features = np.array([1.0, 2.0, 3.0])

        result = ensemble.predict(features, voting_method="majority")

        assert result.signal == "LONG"  # Class 2 has highest prob (0.7)
        assert result.confidence > 0
        assert len(result.contributing_models) == 1

    def test_predict_with_predict_only_model(self, ensemble, mock_model_predict):
        """Test prediction with model that only has predict."""
        ensemble.register_model("model", mock_model_predict)
        features = np.array([1.0, 2.0, 3.0])

        result = ensemble.predict(features, voting_method="majority")

        assert result.signal == "LONG"  # Returns class 2 = LONG
        assert len(result.contributing_models) == 1

    def test_predict_model_without_predict_method(self, ensemble):
        """Test model without predict method is skipped."""
        bad_model = MagicMock(spec=[])  # No predict methods
        del bad_model.predict_proba
        del bad_model.predict

        ensemble.register_model("bad_model", bad_model)
        features = np.array([1.0, 2.0, 3.0])

        result = ensemble.predict(features)

        # Should return FLAT with no contributions
        assert result.signal == "FLAT"
        assert result.confidence == 0.0
        assert len(result.contributing_models) == 0

    def test_majority_voting(self, ensemble):
        """Test majority voting method."""
        # Create three models
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG

        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG

        model3 = MagicMock()
        model3.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])  # SHORT

        ensemble.register_model("m1", model1)
        ensemble.register_model("m2", model2)
        ensemble.register_model("m3", model3)

        result = ensemble.predict(np.array([1.0]), voting_method="majority")

        assert result.signal == "LONG"  # 2 out of 3 vote LONG
        assert result.agreement_score == pytest.approx(2 / 3, rel=0.01)

    def test_weighted_voting(self, ensemble):
        """Test weighted voting method."""
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG, conf 0.8

        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])  # SHORT, conf 0.9

        # Give model2 more weight
        ensemble.register_model("m1", model1, initial_weight=1.0)
        ensemble.register_model("m2", model2, initial_weight=2.0)

        result = ensemble.predict(np.array([1.0]), voting_method="weighted")

        # Model2 should dominate due to higher weight
        assert result.signal == "SHORT"

    def test_confidence_voting(self, ensemble):
        """Test confidence-based voting."""
        # High confidence SHORT
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.95, 0.03, 0.02]])

        # Low confidence LONG
        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.1, 0.3, 0.6]])

        ensemble.register_model("m1", model1)
        ensemble.register_model("m2", model2)

        result = ensemble.predict(np.array([1.0]), voting_method="confidence")

        # HIGH confidence SHORT should win over low confidence LONG
        assert result.signal == "SHORT"

    def test_low_agreement_returns_flat(self, ensemble):
        """Test low agreement returns FLAT signal."""
        # All different signals
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])  # SHORT

        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])  # FLAT

        model3 = MagicMock()
        model3.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG

        ensemble.min_agreement = 0.5  # Require 50% agreement
        ensemble.register_model("m1", model1)
        ensemble.register_model("m2", model2)
        ensemble.register_model("m3", model3)

        result = ensemble.predict(np.array([1.0]), voting_method="majority")

        # Each signal has 1/3 agreement, below threshold
        assert result.signal == "FLAT"

    def test_update_performance(self, ensemble, mock_model_proba):
        """Test updating model performance."""
        ensemble.register_model("model", mock_model_proba)

        predictions = [
            ModelPrediction(model_name="model", signal="LONG", confidence=0.8),
        ]

        # Model predicted LONG, actual was LONG
        ensemble.update_performance(predictions, "LONG")

        assert len(ensemble._model_accuracy["model"]) == 1
        assert ensemble._model_accuracy["model"][0] == 1.0  # Correct

        # Model predicted LONG, actual was SHORT
        ensemble.update_performance(predictions, "SHORT")

        assert len(ensemble._model_accuracy["model"]) == 2
        assert ensemble._model_accuracy["model"][1] == 0.0  # Incorrect

    def test_update_performance_adjusts_weights(self, ensemble, mock_model_proba):
        """Test that update_performance adjusts model weights."""
        ensemble.register_model("model", mock_model_proba, initial_weight=1.0)

        predictions = [
            ModelPrediction(model_name="model", signal="LONG", confidence=0.8),
        ]

        # Many correct predictions
        for _ in range(10):
            ensemble.update_performance(predictions, "LONG")

        # Weight should increase
        assert ensemble.model_weights["model"] > 1.0

    def test_update_performance_window_limit(self, ensemble, mock_model_proba):
        """Test performance window limits accuracy history."""
        ensemble.performance_window = 5
        ensemble.register_model("model", mock_model_proba)

        predictions = [
            ModelPrediction(model_name="model", signal="LONG", confidence=0.8),
        ]

        # Add more than window size
        for _ in range(10):
            ensemble.update_performance(predictions, "LONG")

        # Should be limited to window size
        assert len(ensemble._model_accuracy["model"]) == 5

    def test_get_model_stats(self, ensemble, mock_model_proba):
        """Test getting model statistics."""
        ensemble.register_model("model", mock_model_proba, initial_weight=1.5)

        predictions = [
            ModelPrediction(model_name="model", signal="LONG", confidence=0.8),
        ]
        ensemble.update_performance(predictions, "LONG")

        stats = ensemble.get_model_stats()

        assert "model" in stats
        assert stats["model"]["weight"] == pytest.approx(ensemble.model_weights["model"])
        assert stats["model"]["predictions_tracked"] == 1
        assert stats["model"]["recent_accuracy"] == 1.0

    def test_get_model_stats_no_predictions(self, ensemble, mock_model_proba):
        """Test model stats with no predictions tracked."""
        ensemble.register_model("model", mock_model_proba)

        stats = ensemble.get_model_stats()

        assert stats["model"]["predictions_tracked"] == 0
        assert stats["model"]["recent_accuracy"] == 0.0

    def test_predict_handles_model_exception(self, ensemble):
        """Test that model exceptions are handled gracefully."""
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = Exception("Model error")

        ensemble.register_model("bad", bad_model)

        result = ensemble.predict(np.array([1.0]))

        # Should return FLAT since only model failed
        assert result.signal == "FLAT"
        assert len(result.contributing_models) == 0


class TestRegimeBasedEnsemble:
    """Test RegimeBasedEnsemble class."""

    @pytest.fixture
    def regime_ensemble(self):
        """Create regime-based ensemble."""
        return RegimeBasedEnsemble(min_agreement=0.5)

    @pytest.fixture
    def bull_model(self):
        """Model good for bull markets."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG
        return model

    @pytest.fixture
    def bear_model(self):
        """Model good for bear markets."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])  # SHORT
        return model

    def test_initialization(self, regime_ensemble):
        """Test regime ensemble initialization."""
        assert "bull" in regime_ensemble.regime_models
        assert "bear" in regime_ensemble.regime_models
        assert "sideways" in regime_ensemble.regime_models
        assert "volatile" in regime_ensemble.regime_models

    def test_register_model_with_regimes(self, regime_ensemble, bull_model):
        """Test registering model with specific regimes."""
        regime_ensemble.register_model(
            "bull_specialist",
            bull_model,
            suitable_regimes=["bull"],
        )

        assert "bull_specialist" in regime_ensemble.regime_models["bull"]
        assert "bull_specialist" not in regime_ensemble.regime_models["bear"]

    def test_register_model_all_regimes(self, regime_ensemble, bull_model):
        """Test registering model for all regimes (default)."""
        regime_ensemble.register_model("general_model", bull_model)

        assert "general_model" in regime_ensemble.regime_models["bull"]
        assert "general_model" in regime_ensemble.regime_models["bear"]
        assert "general_model" in regime_ensemble.regime_models["sideways"]
        assert "general_model" in regime_ensemble.regime_models["volatile"]

    def test_predict_for_regime_filters_models(self, regime_ensemble, bull_model, bear_model):
        """Test regime prediction uses correct models."""
        regime_ensemble.register_model(
            "bull_model",
            bull_model,
            suitable_regimes=["bull"],
        )
        regime_ensemble.register_model(
            "bear_model",
            bear_model,
            suitable_regimes=["bear"],
        )

        # Predict for bull regime - should use bull_model
        bull_result = regime_ensemble.predict_for_regime(
            np.array([1.0]),
            regime="bull",
        )
        assert bull_result.signal == "LONG"
        assert "bull_model" in bull_result.contributing_models
        assert "bear_model" not in bull_result.contributing_models

        # Predict for bear regime - should use bear_model
        bear_result = regime_ensemble.predict_for_regime(
            np.array([1.0]),
            regime="bear",
        )
        assert bear_result.signal == "SHORT"
        assert "bear_model" in bear_result.contributing_models

    def test_predict_for_regime_unknown_falls_back(self, regime_ensemble, bull_model):
        """Test unknown regime falls back to all models."""
        regime_ensemble.register_model("model", bull_model)

        result = regime_ensemble.predict_for_regime(
            np.array([1.0]),
            regime="unknown_regime",
        )

        # Should use all models as fallback
        assert len(result.contributing_models) == 1

    def test_predict_for_regime_case_insensitive(self, regime_ensemble, bull_model):
        """Test regime matching is case insensitive."""
        regime_ensemble.register_model(
            "model",
            bull_model,
            suitable_regimes=["bull"],
        )

        result = regime_ensemble.predict_for_regime(
            np.array([1.0]),
            regime="BULL",  # Uppercase
        )

        assert "model" in result.contributing_models


class TestEnsembleVotingEdgeCases:
    """Test edge cases in voting methods."""

    @pytest.fixture
    def ensemble(self):
        return ModelEnsemble()

    def test_majority_vote_tie(self, ensemble):
        """Test majority voting handles ties."""
        model1 = MagicMock()
        model1.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])  # SHORT

        model2 = MagicMock()
        model2.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # LONG

        ensemble.register_model("m1", model1)
        ensemble.register_model("m2", model2)

        result = ensemble.predict(np.array([1.0]), voting_method="majority")

        # Should pick one (implementation dependent)
        assert result.signal in ["LONG", "SHORT"]

    def test_weighted_vote_zero_total(self, ensemble):
        """Test weighted vote handles zero total weight."""
        # Create predictions with zero confidence
        preds = [
            ModelPrediction(model_name="m1", signal="LONG", confidence=0.0),
        ]

        signal, conf, agreement = ensemble._weighted_vote(preds)

        assert signal == "FLAT"
        assert conf == 0.0

    def test_confidence_vote_zero_total(self, ensemble):
        """Test confidence vote handles zero total."""
        preds = [
            ModelPrediction(model_name="m1", signal="LONG", confidence=0.0),
        ]

        signal, conf, agreement = ensemble._confidence_vote(preds)

        assert signal == "FLAT"
        assert conf == 0.0

    def test_predict_returns_flat_on_empty_predictions(self, ensemble):
        """Test that empty predictions returns FLAT."""
        # Register model that returns None
        bad_model = MagicMock(spec=[])

        ensemble.register_model("bad", bad_model)
        result = ensemble.predict(np.array([1.0]))

        assert result.signal == "FLAT"
        assert result.confidence == 0.0


class TestModelPredictionSignalMapping:
    """Test signal mapping from model predictions."""

    @pytest.fixture
    def ensemble(self):
        return ModelEnsemble()

    def test_predict_proba_signal_mapping(self, ensemble):
        """Test signal mapping from predict_proba."""
        model = MagicMock()

        # Test all three classes
        # Class 0 = SHORT
        model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
        ensemble.register_model("m", model)
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "SHORT"

        # Class 1 = FLAT
        model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "FLAT"

        # Class 2 = LONG
        model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "LONG"

    def test_predict_signal_mapping_integers(self, ensemble):
        """Test signal mapping from predict (integer classes)."""
        model = MagicMock()
        del model.predict_proba

        model.predict.return_value = np.array([0])
        ensemble.register_model("m", model)
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "SHORT"

        model.predict.return_value = np.array([1])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "FLAT"

        model.predict.return_value = np.array([2])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "LONG"

    def test_predict_signal_mapping_strings(self, ensemble):
        """Test signal mapping from string predictions."""
        model = MagicMock()
        del model.predict_proba

        model.predict.return_value = np.array(["long"])
        ensemble.register_model("m", model)
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "LONG"

        model.predict.return_value = np.array(["SHORT"])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "SHORT"

        # Unknown string maps to FLAT
        model.predict.return_value = np.array(["unknown"])
        pred = ensemble._get_model_prediction("m", model, np.array([1.0]))
        assert pred.signal == "FLAT"
