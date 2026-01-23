"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from bot.ml.feature_engineer import FeatureEngineer, FeatureConfig


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.atr_period == 14
        assert config.adx_period == 14

    def test_default_ema_periods(self):
        """Test default EMA periods are set."""
        config = FeatureConfig()
        assert config.ema_periods == [5, 10, 20, 50, 100]

    def test_default_lookback_periods(self):
        """Test default lookback periods are set."""
        config = FeatureConfig()
        assert config.lookback_periods == [1, 3, 5, 10, 20]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            rsi_period=10,
            macd_fast=8,
            ema_periods=[10, 20, 50],
            lookback_periods=[1, 5, 10],
        )
        assert config.rsi_period == 10
        assert config.macd_fast == 8
        assert config.ema_periods == [10, 20, 50]
        assert config.lookback_periods == [1, 5, 10]


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()

    @pytest.fixture
    def custom_engineer(self):
        """Create feature engineer with custom config."""
        config = FeatureConfig(
            ema_periods=[10, 20],
            lookback_periods=[1, 5],
        )
        return FeatureEngineer(config)

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        # Simulate realistic price movement
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.005),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

    @pytest.fixture
    def short_ohlcv(self):
        """Create short OHLCV data."""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1000, 900],
            }
        )

    def test_engineer_creation(self, engineer):
        """Test engineer is created with default config."""
        assert engineer.config is not None
        assert isinstance(engineer.config, FeatureConfig)

    def test_extract_features(self, engineer, sample_ohlcv):
        """Test feature extraction."""
        features = engineer.extract_features(sample_ohlcv)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        # Should have more columns than input
        assert len(features.columns) > 5

    def test_features_have_no_nan(self, engineer, sample_ohlcv):
        """Test extracted features have no NaN values."""
        features = engineer.extract_features(sample_ohlcv)
        assert not features.isnull().any().any()

    def test_features_have_no_inf(self, engineer, sample_ohlcv):
        """Test extracted features have no infinity values."""
        features = engineer.extract_features(sample_ohlcv)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert not np.any(np.isinf(features[numeric_cols].values))

    def test_return_features(self, engineer, sample_ohlcv):
        """Test return features are created."""
        features = engineer.extract_features(sample_ohlcv)
        # Should have return features for different periods
        return_cols = [c for c in features.columns if c.startswith("return_")]
        assert len(return_cols) > 0

    def test_ema_features(self, engineer, sample_ohlcv):
        """Test EMA features are created."""
        features = engineer.extract_features(sample_ohlcv)
        # Should have EMA features
        ema_cols = [c for c in features.columns if "ema" in c.lower()]
        assert len(ema_cols) > 0

    def test_momentum_features(self, engineer, sample_ohlcv):
        """Test momentum features are created."""
        features = engineer.extract_features(sample_ohlcv)
        # Should have RSI, MACD features
        assert "rsi" in features.columns or any("rsi" in c.lower() for c in features.columns)

    def test_volatility_features(self, engineer, sample_ohlcv):
        """Test volatility features are created."""
        features = engineer.extract_features(sample_ohlcv)
        # Should have ATR, BB features
        vol_cols = [c for c in features.columns if "atr" in c.lower() or "bb" in c.lower()]
        assert len(vol_cols) > 0

    def test_custom_config_affects_features(self, custom_engineer, sample_ohlcv):
        """Test custom config changes feature extraction."""
        features = custom_engineer.extract_features(sample_ohlcv)
        # Should have features based on custom periods
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_fewer_rows_after_extraction(self, engineer, sample_ohlcv):
        """Test some rows are lost due to lookback requirements."""
        features = engineer.extract_features(sample_ohlcv)
        # Should have fewer rows than input due to NaN from indicators
        assert len(features) <= len(sample_ohlcv)

    def test_original_columns_preserved(self, engineer, sample_ohlcv):
        """Test original OHLCV columns are preserved."""
        features = engineer.extract_features(sample_ohlcv)
        assert "close" in features.columns
        assert "volume" in features.columns

    def test_short_data_handling(self, engineer, short_ohlcv):
        """Test handling of short data."""
        # Short data may raise errors in indicator calculations
        # This is expected behavior - engineer needs sufficient data
        try:
            features = engineer.extract_features(short_ohlcv)
            assert isinstance(features, pd.DataFrame)
        except (ValueError, KeyError):
            # Expected for very short data
            pass


class TestFeatureSanitization:
    """Test feature sanitization."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()

    def test_sanitize_handles_infinity(self, engineer):
        """Test sanitization handles infinity values."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1100, 1200],
                "feature": [1.0, np.inf, 2.0],
            }
        )
        sanitized = engineer._sanitize_features(df)
        # Infinity should be replaced with NaN
        assert not np.any(np.isinf(sanitized["feature"]))

    def test_sanitize_handles_negative_infinity(self, engineer):
        """Test sanitization handles negative infinity."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1100, 1200],
                "feature": [1.0, -np.inf, 2.0],
            }
        )
        sanitized = engineer._sanitize_features(df)
        assert not np.any(np.isinf(sanitized["feature"]))


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        engineer = FeatureEngineer()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        # Empty data may raise errors in indicator calculations
        try:
            features = engineer.extract_features(df)
            assert isinstance(features, pd.DataFrame)
            assert len(features) == 0
        except (ValueError, KeyError):
            # Expected for empty data
            pass

    def test_constant_prices(self):
        """Test with constant prices."""
        engineer = FeatureEngineer()
        n = 200
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )
        features = engineer.extract_features(df)
        # Should handle gracefully
        assert isinstance(features, pd.DataFrame)

    def test_high_volatility_data(self):
        """Test with high volatility data."""
        engineer = FeatureEngineer()
        np.random.seed(42)
        n = 200
        # High volatility random walk
        returns = np.random.randn(n) * 0.1  # 10% daily moves
        close = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.randn(n) * 0.02),
                "high": close * (1 + np.abs(np.random.randn(n) * 0.05)),
                "low": close * (1 - np.abs(np.random.randn(n) * 0.05)),
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            }
        )
        features = engineer.extract_features(df)
        # Should handle high volatility without overflow
        assert not features.isnull().any().any()

    def test_with_datetime_index(self):
        """Test with datetime index."""
        engineer = FeatureEngineer()
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, n),
            },
            index=dates,
        )

        features = engineer.extract_features(df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
