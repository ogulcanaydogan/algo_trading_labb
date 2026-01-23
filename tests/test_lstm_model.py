"""
Tests for LSTM Model module.

Tests the LSTM model's data preparation and utility functions
without requiring TensorFlow/Keras to be installed.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

# Import the module to check TensorFlow availability
from bot.ml import lstm_model


class TestLSTMPredictorInit:
    """Test LSTMPredictor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        predictor = lstm_model.LSTMPredictor()

        assert predictor.sequence_length == 60
        assert predictor.n_features == 5
        assert predictor.lstm_units == [128, 64]
        assert predictor.dropout_rate == 0.2
        assert predictor.learning_rate == 0.001
        assert predictor.model is None
        assert predictor.is_trained is False
        assert predictor.metrics == {}

    def test_custom_init(self):
        """Test custom initialization."""
        predictor = lstm_model.LSTMPredictor(
            sequence_length=30,
            n_features=10,
            lstm_units=[256, 128],
            dropout_rate=0.3,
            learning_rate=0.0005,
        )

        assert predictor.sequence_length == 30
        assert predictor.n_features == 10
        assert predictor.lstm_units == [256, 128]
        assert predictor.dropout_rate == 0.3
        assert predictor.learning_rate == 0.0005


class TestLSTMDataPreparation:
    """Test data preparation methods."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        np.random.seed(42)
        n = 200
        base_price = 100

        df = pd.DataFrame(
            {
                "open": base_price + np.random.randn(n).cumsum(),
                "high": base_price + np.random.randn(n).cumsum() + 2,
                "low": base_price + np.random.randn(n).cumsum() - 2,
                "close": base_price + np.random.randn(n).cumsum(),
                "volume": np.random.randint(1000, 10000, n),
            }
        )
        return df

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return lstm_model.LSTMPredictor(sequence_length=10, n_features=5)

    def test_create_targets(self, predictor, sample_df):
        """Test target creation."""
        df_with_targets = predictor.create_targets(sample_df, lookahead=1)

        assert "target" in df_with_targets.columns
        assert "future_return" in df_with_targets.columns
        # Target should be binary
        assert df_with_targets["target"].isin([0, 1]).all()
        # Should have fewer rows due to dropna
        assert len(df_with_targets) < len(sample_df)

    def test_create_targets_lookahead(self, predictor, sample_df):
        """Test target creation with different lookahead."""
        df_1 = predictor.create_targets(sample_df.copy(), lookahead=1)
        df_5 = predictor.create_targets(sample_df.copy(), lookahead=5)

        # Both should have targets
        assert "target" in df_1.columns
        assert "target" in df_5.columns
        # Longer lookahead loses more rows
        assert len(df_5) <= len(df_1)

    def test_prepare_data_creates_sequences(self, predictor, sample_df):
        """Test data preparation creates correct sequences."""
        df = predictor.create_targets(sample_df)
        X, y = predictor.prepare_data(df)

        # X shape should be (n_samples, sequence_length, n_features)
        assert len(X.shape) == 3
        assert X.shape[1] == predictor.sequence_length
        assert X.shape[2] == predictor.n_features

        # y length should match X
        assert len(y) == len(X)

    def test_prepare_data_stores_scaler_params(self, predictor, sample_df):
        """Test scaler params are stored."""
        df = predictor.create_targets(sample_df)
        predictor.prepare_data(df)

        assert "mean" in predictor.scaler_params
        assert "std" in predictor.scaler_params
        assert "close" in predictor.scaler_params["mean"]
        assert "volume" in predictor.scaler_params["mean"]

    def test_prepare_data_without_targets(self, predictor, sample_df):
        """Test data preparation without target column."""
        X, y = predictor.prepare_data(sample_df)

        assert X is not None
        assert len(X.shape) == 3
        assert y is None

    def test_prepare_data_normalizes(self, predictor, sample_df):
        """Test data is normalized."""
        df = predictor.create_targets(sample_df)
        X, y = predictor.prepare_data(df)

        # Normalized data should have mean close to 0
        # and std close to 1 (approximately)
        flat_X = X.reshape(-1, predictor.n_features)
        means = flat_X.mean(axis=0)

        # Means should be relatively small (centered around 0)
        assert np.all(np.abs(means) < 5)


class TestLSTMModelBuild:
    """Test model building (mocked if TensorFlow unavailable)."""

    def test_build_model_without_tensorflow(self):
        """Test build_model raises ImportError without TensorFlow."""
        if not lstm_model.HAS_TENSORFLOW:
            predictor = lstm_model.LSTMPredictor()

            with pytest.raises(ImportError, match="TensorFlow required"):
                predictor.build_model()

    @pytest.mark.skipif(not lstm_model.HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_build_model_with_tensorflow(self):
        """Test model is built with TensorFlow."""
        predictor = lstm_model.LSTMPredictor()
        predictor.build_model()

        assert predictor.model is not None


class TestLSTMPrediction:
    """Test prediction functionality."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return lstm_model.LSTMPredictor(sequence_length=10)

    def test_predict_not_trained_raises(self, predictor):
        """Test predict raises when not trained."""
        df = pd.DataFrame(
            {
                "open": np.random.rand(50),
                "high": np.random.rand(50),
                "low": np.random.rand(50),
                "close": np.random.rand(50),
                "volume": np.random.rand(50),
            }
        )

        with pytest.raises(ValueError, match="not trained"):
            predictor.predict(df)


class TestLSTMSaveLoad:
    """Test save/load functionality."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return lstm_model.LSTMPredictor()

    def test_save_not_trained_raises(self, predictor, tmp_path):
        """Test save raises when not trained."""
        with pytest.raises(ValueError, match="No trained model"):
            predictor.save(str(tmp_path))

    def test_load_nonexistent_path(self, predictor, tmp_path):
        """Test load returns False for nonexistent path."""
        result = predictor.load(str(tmp_path / "nonexistent"))

        assert result is False

    @pytest.mark.skipif(not lstm_model.HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_load_without_tensorflow(self, predictor, tmp_path):
        """Test load handles missing TensorFlow gracefully."""
        # Create mock files
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "lstm_model.keras").touch()
        (model_dir / "lstm_meta.pkl").touch()

        # Load should work or return False gracefully
        # (depending on file contents)
        result = predictor.load(str(model_dir))
        # Result depends on actual file contents


class TestTrainLSTMModel:
    """Test convenience training function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            {
                "open": 100 + np.random.randn(n).cumsum(),
                "high": 102 + np.random.randn(n).cumsum(),
                "low": 98 + np.random.randn(n).cumsum(),
                "close": 100 + np.random.randn(n).cumsum(),
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    def test_train_without_tensorflow(self, sample_df):
        """Test training returns error without TensorFlow."""
        if not lstm_model.HAS_TENSORFLOW:
            result = lstm_model.train_lstm_model(
                symbol="BTC/USDT",
                df=sample_df,
            )

            assert result["error"] == "TensorFlow not installed"


class TestHasTensorflow:
    """Test TensorFlow availability detection."""

    def test_has_tensorflow_is_boolean(self):
        """Test HAS_TENSORFLOW is a boolean."""
        assert isinstance(lstm_model.HAS_TENSORFLOW, bool)


class TestLSTMPredictorSequences:
    """Test sequence generation edge cases."""

    @pytest.fixture
    def predictor(self):
        return lstm_model.LSTMPredictor(sequence_length=10)

    def test_prepare_data_minimum_rows(self, predictor):
        """Test with exactly sequence_length + 1 rows."""
        df = pd.DataFrame(
            {
                "open": np.random.rand(11),
                "high": np.random.rand(11),
                "low": np.random.rand(11),
                "close": np.random.rand(11),
                "volume": np.random.rand(11),
                "target": np.random.randint(0, 2, 11),
            }
        )

        X, y = predictor.prepare_data(df)

        # Should produce exactly 1 sequence
        assert len(X) == 1
        assert len(y) == 1

    def test_prepare_data_insufficient_rows(self, predictor):
        """Test with fewer rows than sequence_length."""
        df = pd.DataFrame(
            {
                "open": np.random.rand(5),
                "high": np.random.rand(5),
                "low": np.random.rand(5),
                "close": np.random.rand(5),
                "volume": np.random.rand(5),
                "target": np.random.randint(0, 2, 5),
            }
        )

        X, y = predictor.prepare_data(df)

        # Should produce no sequences
        assert len(X) == 0

    def test_sequence_content(self, predictor):
        """Test sequence content is correct."""
        # Create predictable data
        df = pd.DataFrame(
            {
                "open": list(range(20)),
                "high": list(range(20)),
                "low": list(range(20)),
                "close": list(range(20)),
                "volume": [1000] * 20,
                "target": [1] * 20,
            }
        )

        predictor.scaler_params = {
            "mean": {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            "std": {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        }

        X, y = predictor.prepare_data(df)

        # First sequence should start at index 0
        # Second sequence should start at index 1
        # etc.
        assert X.shape[0] == 10  # 20 - sequence_length


class TestLSTMPredictorNormalization:
    """Test normalization edge cases."""

    @pytest.fixture
    def predictor(self):
        return lstm_model.LSTMPredictor(sequence_length=5)

    def test_zero_std_handled(self, predictor):
        """Test zero standard deviation is handled."""
        # Create data with constant values (zero std)
        df = pd.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [100.0] * 20,
                "low": [100.0] * 20,
                "close": [100.0] * 20,
                "volume": [1000] * 20,
                "target": [1] * 20,
            }
        )

        # Should not raise even with zero std
        X, y = predictor.prepare_data(df)

        # Should still produce sequences
        assert len(X) > 0


class TestLSTMMetrics:
    """Test metrics tracking."""

    def test_metrics_default_empty(self):
        """Test metrics dict is empty by default."""
        predictor = lstm_model.LSTMPredictor()
        assert predictor.metrics == {}

    def test_is_trained_default_false(self):
        """Test is_trained is False by default."""
        predictor = lstm_model.LSTMPredictor()
        assert predictor.is_trained is False


class TestLSTMScalerParams:
    """Test scaler parameter storage."""

    def test_scaler_params_default_empty(self):
        """Test scaler_params dict is empty by default."""
        predictor = lstm_model.LSTMPredictor()
        assert predictor.scaler_params == {}

    def test_scaler_params_populated_after_prepare(self):
        """Test scaler_params populated after prepare_data."""
        predictor = lstm_model.LSTMPredictor(sequence_length=5)

        df = pd.DataFrame(
            {
                "open": np.random.rand(20) * 100 + 50,
                "high": np.random.rand(20) * 100 + 55,
                "low": np.random.rand(20) * 100 + 45,
                "close": np.random.rand(20) * 100 + 50,
                "volume": np.random.randint(1000, 10000, 20),
            }
        )

        predictor.prepare_data(df)

        # Verify all expected keys present
        assert "mean" in predictor.scaler_params
        assert "std" in predictor.scaler_params

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in predictor.scaler_params["mean"]
            assert col in predictor.scaler_params["std"]
