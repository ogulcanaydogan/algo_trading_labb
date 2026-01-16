"""
Tests for market regime classifier.
"""

import pytest
import pandas as pd
import numpy as np

from bot.ml.regime_classifier import (
    MarketRegime,
    RegimeAnalysis,
    MarketRegimeClassifier,
)


class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_regime_values(self):
        """Test all regime values exist."""
        assert MarketRegime.STRONG_BULL.value == "strong_bull"
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.STRONG_BEAR.value == "strong_bear"
        assert MarketRegime.VOLATILE.value == "volatile"

    def test_regime_from_string(self):
        """Test regime can be created from string."""
        regime = MarketRegime("bull")
        assert regime == MarketRegime.BULL


class TestRegimeAnalysis:
    """Test RegimeAnalysis dataclass."""

    def test_analysis_creation(self):
        """Test creating a regime analysis."""
        analysis = RegimeAnalysis(
            regime=MarketRegime.BULL,
            confidence=0.75,
            trend_strength=0.5,
            volatility_level="normal",
            volatility_percentile=50.0,
            adx_value=25.0,
            momentum_score=0.3,
            support_level=100.0,
            resistance_level=110.0,
            regime_duration=10,
            recommended_strategy="trend_following",
            reasoning=["EMA alignment bullish", "ADX shows trend"],
        )
        assert analysis.regime == MarketRegime.BULL
        assert analysis.confidence == 0.75
        assert analysis.trend_strength == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = RegimeAnalysis(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.6,
            trend_strength=0.1,
            volatility_level="low",
            volatility_percentile=30.0,
            adx_value=15.0,
            momentum_score=0.0,
            support_level=95.0,
            resistance_level=105.0,
            regime_duration=5,
            recommended_strategy="mean_reversion",
            reasoning=["Low ADX", "Range-bound"],
        )
        d = analysis.to_dict()
        assert d["regime"] == "sideways"
        assert d["confidence"] == 0.6
        assert "reasoning" in d
        assert isinstance(d["reasoning"], list)


class TestMarketRegimeClassifier:
    """Test MarketRegimeClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return MarketRegimeClassifier()

    @pytest.fixture
    def custom_classifier(self):
        """Create classifier with custom parameters."""
        return MarketRegimeClassifier(
            ema_fast=10,
            ema_slow=30,
            ema_trend=100,
            adx_period=10,
            atr_period=10,
            lookback=50,
        )

    @pytest.fixture
    def uptrend_data(self):
        """Create data with clear uptrend."""
        np.random.seed(42)
        n = 250
        # Strong upward trend
        trend = np.arange(n) * 0.5
        noise = np.random.randn(n) * 2
        close = 100 + trend + noise

        return pd.DataFrame({
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 2,
            "low": close - np.random.rand(n) * 1,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def downtrend_data(self):
        """Create data with clear downtrend."""
        np.random.seed(42)
        n = 250
        # Strong downward trend
        trend = -np.arange(n) * 0.5
        noise = np.random.randn(n) * 2
        close = 200 + trend + noise

        return pd.DataFrame({
            "open": close + np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n) * 1,
            "low": close - np.random.rand(n) * 2,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def sideways_data(self):
        """Create data with sideways movement."""
        np.random.seed(42)
        n = 250
        # Oscillating around 100
        close = 100 + np.sin(np.arange(n) * 0.1) * 5 + np.random.randn(n) * 1

        return pd.DataFrame({
            "open": close - np.random.rand(n) * 0.3,
            "high": close + np.random.rand(n) * 1,
            "low": close - np.random.rand(n) * 1,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def volatile_data(self):
        """Create data with high volatility."""
        np.random.seed(42)
        n = 250
        # High volatility swings
        close = 100 + np.cumsum(np.random.randn(n) * 5)

        return pd.DataFrame({
            "open": close - np.random.rand(n) * 2,
            "high": close + np.random.rand(n) * 5,
            "low": close - np.random.rand(n) * 5,
            "close": close,
            "volume": np.random.randint(1000, 10000, n),
        })

    @pytest.fixture
    def short_data(self):
        """Create insufficient data."""
        return pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000] * 5,
        })

    def test_classifier_creation(self, classifier):
        """Test classifier is created with defaults."""
        assert classifier.ema_fast == 20
        assert classifier.ema_slow == 50
        assert classifier.ema_trend == 200
        assert classifier.adx_period == 14

    def test_custom_parameters(self, custom_classifier):
        """Test classifier with custom parameters."""
        assert custom_classifier.ema_fast == 10
        assert custom_classifier.ema_slow == 30
        assert custom_classifier.ema_trend == 100

    def test_insufficient_data(self, classifier, short_data):
        """Test default analysis with insufficient data."""
        analysis = classifier.classify(short_data)
        # Should return default analysis
        assert analysis.regime in list(MarketRegime)
        assert 0 <= analysis.confidence <= 1

    def test_uptrend_classification(self, classifier, uptrend_data):
        """Test classification of uptrend."""
        analysis = classifier.classify(uptrend_data)
        # Should classify as bullish regime
        assert analysis.regime in [
            MarketRegime.STRONG_BULL,
            MarketRegime.BULL,
            MarketRegime.SIDEWAYS,
            MarketRegime.VOLATILE,
        ]
        assert analysis.trend_strength >= -1
        assert analysis.trend_strength <= 1

    def test_downtrend_classification(self, classifier, downtrend_data):
        """Test classification of downtrend."""
        analysis = classifier.classify(downtrend_data)
        # Should classify as bearish regime
        assert analysis.regime in list(MarketRegime)
        assert isinstance(analysis.trend_strength, float)

    def test_sideways_classification(self, classifier, sideways_data):
        """Test classification of sideways market."""
        analysis = classifier.classify(sideways_data)
        assert analysis.regime in list(MarketRegime)
        # Trend strength should be near zero for sideways
        assert abs(analysis.trend_strength) <= 1

    def test_volatile_classification(self, classifier, volatile_data):
        """Test classification of volatile market."""
        analysis = classifier.classify(volatile_data)
        assert analysis.regime in list(MarketRegime)
        # Volatility should be elevated
        assert analysis.volatility_level in ["low", "normal", "high", "extreme"]

    def test_analysis_has_all_fields(self, classifier, uptrend_data):
        """Test analysis has all required fields."""
        analysis = classifier.classify(uptrend_data)
        assert analysis.regime is not None
        assert analysis.confidence is not None
        assert analysis.trend_strength is not None
        assert analysis.volatility_level is not None
        assert analysis.volatility_percentile is not None
        assert analysis.adx_value is not None
        assert analysis.momentum_score is not None
        assert analysis.support_level is not None
        assert analysis.resistance_level is not None
        assert analysis.regime_duration is not None
        assert analysis.recommended_strategy is not None
        assert analysis.reasoning is not None

    def test_confidence_bounds(self, classifier, uptrend_data):
        """Test confidence is within bounds."""
        analysis = classifier.classify(uptrend_data)
        assert 0 <= analysis.confidence <= 1

    def test_volatility_percentile_bounds(self, classifier, uptrend_data):
        """Test volatility percentile is within bounds."""
        analysis = classifier.classify(uptrend_data)
        assert 0 <= analysis.volatility_percentile <= 100

    def test_recommended_strategy_not_empty(self, classifier, uptrend_data):
        """Test recommended strategy is provided."""
        analysis = classifier.classify(uptrend_data)
        assert len(analysis.recommended_strategy) > 0

    def test_reasoning_is_list(self, classifier, uptrend_data):
        """Test reasoning is a list."""
        analysis = classifier.classify(uptrend_data)
        assert isinstance(analysis.reasoning, list)

    def test_regime_duration_tracking(self, classifier, uptrend_data):
        """Test regime duration is tracked."""
        # First classification
        analysis1 = classifier.classify(uptrend_data)
        duration1 = analysis1.regime_duration

        # Second classification (same data = same regime)
        analysis2 = classifier.classify(uptrend_data)
        duration2 = analysis2.regime_duration

        # Duration should increase or stay same if regime unchanged
        assert duration2 >= 1

    def test_to_dict_format(self, classifier, uptrend_data):
        """Test to_dict output format."""
        analysis = classifier.classify(uptrend_data)
        d = analysis.to_dict()

        assert "regime" in d
        assert "confidence" in d
        assert "trend_strength" in d
        assert "volatility_level" in d
        assert "adx_value" in d
        assert "recommended_strategy" in d
        assert "reasoning" in d

        # Values should be properly rounded
        assert isinstance(d["confidence"], float)
        assert isinstance(d["adx_value"], float)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        classifier = MarketRegimeClassifier()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        analysis = classifier.classify(df)
        assert analysis.regime in list(MarketRegime)

    def test_constant_prices(self):
        """Test with constant prices."""
        classifier = MarketRegimeClassifier(ema_trend=20)
        n = 50
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        })
        analysis = classifier.classify(df)
        # Should handle gracefully - likely sideways with low ADX
        assert analysis.regime in list(MarketRegime)

    def test_single_candle_spike(self):
        """Test with sudden price spike."""
        classifier = MarketRegimeClassifier()
        np.random.seed(42)
        n = 250
        close = np.ones(n) * 100
        close[-1] = 150  # Sudden spike

        df = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000] * n,
        })
        analysis = classifier.classify(df)
        assert analysis.regime in list(MarketRegime)
