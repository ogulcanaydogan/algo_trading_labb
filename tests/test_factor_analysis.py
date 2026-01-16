"""
Tests for Factor Analysis Module.

Tests the factor analysis functionality including
factor exposure calculations, regression analysis, and correlations.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict

from bot.factor_analysis import (
    FactorExposure,
    FactorAnalysisResult,
    FactorAnalyzer,
)


class TestFactorExposure:
    """Tests for FactorExposure dataclass."""

    def test_factor_exposure_creation(self):
        """Test creating a FactorExposure."""
        exposure = FactorExposure(
            name="momentum",
            beta=0.5,
            contribution=2.5,
            t_stat=2.1,
            p_value=0.04,
            is_significant=True,
        )

        assert exposure.name == "momentum"
        assert exposure.beta == 0.5
        assert exposure.contribution == 2.5
        assert exposure.t_stat == 2.1
        assert exposure.p_value == 0.04
        assert exposure.is_significant is True

    def test_factor_exposure_not_significant(self):
        """Test FactorExposure with non-significant p-value."""
        exposure = FactorExposure(
            name="volatility",
            beta=0.1,
            contribution=0.5,
            t_stat=1.2,
            p_value=0.12,
            is_significant=False,
        )

        assert exposure.is_significant is False
        assert exposure.p_value > 0.05

    def test_factor_exposure_negative_beta(self):
        """Test FactorExposure with negative beta."""
        exposure = FactorExposure(
            name="mean_reversion",
            beta=-0.3,
            contribution=-1.5,
            t_stat=-2.5,
            p_value=0.02,
            is_significant=True,
        )

        assert exposure.beta < 0
        assert exposure.contribution < 0


class TestFactorAnalysisResult:
    """Tests for FactorAnalysisResult dataclass."""

    def test_factor_analysis_result_creation(self):
        """Test creating a FactorAnalysisResult."""
        factor = FactorExposure(
            name="momentum",
            beta=0.5,
            contribution=2.5,
            t_stat=2.1,
            p_value=0.04,
            is_significant=True,
        )

        result = FactorAnalysisResult(
            timestamp=datetime.now(),
            total_return=10.5,
            factor_explained_return=8.0,
            alpha=0.02,
            r_squared=0.75,
            adjusted_r_squared=0.72,
            factors=[factor],
            factor_contributions={"momentum": 2.5},
            residual_analysis={"mean": 0.0, "std": 0.5},
        )

        assert result.total_return == 10.5
        assert result.factor_explained_return == 8.0
        assert result.alpha == 0.02
        assert result.r_squared == 0.75
        assert len(result.factors) == 1

    def test_factor_analysis_result_multiple_factors(self):
        """Test FactorAnalysisResult with multiple factors."""
        factors = [
            FactorExposure("momentum", 0.5, 2.5, 2.1, 0.04, True),
            FactorExposure("volatility", 0.3, 1.5, 1.8, 0.08, False),
            FactorExposure("trend", 0.4, 2.0, 2.3, 0.03, True),
        ]

        result = FactorAnalysisResult(
            timestamp=datetime.now(),
            total_return=12.0,
            factor_explained_return=10.0,
            alpha=0.03,
            r_squared=0.80,
            adjusted_r_squared=0.75,
            factors=factors,
            factor_contributions={"momentum": 2.5, "volatility": 1.5, "trend": 2.0},
            residual_analysis={"mean": 0.0},
        )

        assert len(result.factors) == 3
        assert len(result.factor_contributions) == 3


class TestFactorAnalyzer:
    """Tests for FactorAnalyzer class."""

    @pytest.fixture
    def sample_returns(self) -> List[float]:
        """Generate sample returns."""
        np.random.seed(42)
        return list(np.random.normal(0.001, 0.02, 100))

    @pytest.fixture
    def sample_factor_data(self) -> Dict[str, List[float]]:
        """Generate sample factor data."""
        np.random.seed(42)
        return {
            "momentum": list(np.random.normal(0.001, 0.015, 100)),
            "volatility": list(np.random.normal(0.0, 0.02, 100)),
            "trend": list(np.random.normal(0.0005, 0.01, 100)),
        }

    @pytest.fixture
    def sample_prices(self) -> np.ndarray:
        """Generate sample price data."""
        np.random.seed(42)
        prices = [100]
        for _ in range(99):
            prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))
        return np.array(prices)

    def test_analyzer_initialization(self):
        """Test FactorAnalyzer initialization."""
        analyzer = FactorAnalyzer()
        assert len(analyzer._returns) == 0
        assert len(analyzer._factor_data) == 0
        assert len(analyzer._timestamps) == 0

    def test_analyzer_factors_constant(self):
        """Test FACTORS constant is defined."""
        assert len(FactorAnalyzer.FACTORS) > 0
        assert "momentum" in FactorAnalyzer.FACTORS
        assert "volatility" in FactorAnalyzer.FACTORS
        assert "trend" in FactorAnalyzer.FACTORS

    def test_load_data(self, sample_returns, sample_factor_data):
        """Test loading returns and factor data."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        assert len(analyzer._returns) == 100
        assert "momentum" in analyzer._factor_data
        assert len(analyzer._factor_data["momentum"]) == 100

    def test_load_data_with_timestamps(self, sample_returns, sample_factor_data):
        """Test loading data with timestamps."""
        from datetime import timedelta
        analyzer = FactorAnalyzer()
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(hours=i) for i in range(100)]
        analyzer.load_data(sample_returns, sample_factor_data, timestamps)

        assert len(analyzer._timestamps) == 100

    def test_calculate_factors_from_prices(self, sample_prices):
        """Test calculating factors from price data."""
        analyzer = FactorAnalyzer()
        factors = analyzer.calculate_factors_from_prices(sample_prices)

        assert "momentum" in factors
        assert "volatility" in factors
        assert "trend" in factors
        assert "mean_reversion" in factors
        assert "market" in factors

    def test_calculate_factors_from_prices_with_volume(self, sample_prices):
        """Test calculating factors with volume data."""
        analyzer = FactorAnalyzer()
        volumes = np.random.uniform(1000, 5000, len(sample_prices))
        factors = analyzer.calculate_factors_from_prices(sample_prices, volumes)

        assert "volume" in factors
        assert len(factors["volume"]) > 0

    def test_calculate_factors_insufficient_data(self):
        """Test factors calculation with insufficient data."""
        analyzer = FactorAnalyzer()
        prices = np.array([100, 101, 102])  # Only 3 prices
        factors = analyzer.calculate_factors_from_prices(prices)

        assert factors == {}

    def test_run_analysis_insufficient_data(self):
        """Test analysis with insufficient data raises error."""
        analyzer = FactorAnalyzer()
        analyzer.load_data([0.01, 0.02], {"factor1": [0.01, 0.02]})

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.run_analysis()

    def test_run_analysis_basic(self, sample_returns, sample_factor_data):
        """Test basic factor analysis."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.run_analysis()

        assert result.timestamp is not None
        assert isinstance(result.total_return, float)
        assert isinstance(result.r_squared, float)
        assert 0 <= result.r_squared <= 1
        assert len(result.factors) > 0

    def test_run_analysis_returns_factor_exposures(self, sample_returns, sample_factor_data):
        """Test that analysis returns factor exposures."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.run_analysis()

        for factor in result.factors:
            assert isinstance(factor, FactorExposure)
            assert factor.name in sample_factor_data.keys()

    def test_run_analysis_contributions(self, sample_returns, sample_factor_data):
        """Test factor contributions are calculated."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.run_analysis()

        assert len(result.factor_contributions) == len(sample_factor_data)
        for factor_name in sample_factor_data.keys():
            assert factor_name in result.factor_contributions

    def test_run_analysis_residuals(self, sample_returns, sample_factor_data):
        """Test residual analysis is performed."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.run_analysis()

        assert "mean" in result.residual_analysis
        assert "std" in result.residual_analysis
        assert "skewness" in result.residual_analysis
        assert "kurtosis" in result.residual_analysis
        assert "autocorrelation" in result.residual_analysis

    def test_run_analysis_adjusted_r_squared(self, sample_returns, sample_factor_data):
        """Test adjusted R-squared is calculated."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.run_analysis()

        # Adjusted R-squared should be <= R-squared
        assert result.adjusted_r_squared <= result.r_squared

    def test_get_factor_correlations_empty(self):
        """Test factor correlations with no data."""
        analyzer = FactorAnalyzer()
        correlations = analyzer.get_factor_correlations()

        assert correlations == {}

    def test_get_factor_correlations(self, sample_returns, sample_factor_data):
        """Test factor correlations calculation."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        correlations = analyzer.get_factor_correlations()

        # Should have correlation matrix
        assert len(correlations) == len(sample_factor_data)
        for factor_name in sample_factor_data.keys():
            assert factor_name in correlations
            # Diagonal should be 1.0 (self-correlation)
            assert correlations[factor_name][factor_name] == 1.0

    def test_get_factor_correlations_symmetric(self, sample_returns, sample_factor_data):
        """Test that correlation matrix is symmetric."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        correlations = analyzer.get_factor_correlations()
        factors = list(correlations.keys())

        for f1 in factors:
            for f2 in factors:
                assert correlations[f1][f2] == correlations[f2][f1]

    def test_get_rolling_factor_exposure_insufficient_data(self):
        """Test rolling exposure with insufficient data."""
        analyzer = FactorAnalyzer()
        analyzer.load_data([0.01] * 30, {"factor1": [0.01] * 30})

        result = analyzer.get_rolling_factor_exposure(window=60)
        assert result == {}

    def test_get_rolling_factor_exposure(self, sample_returns, sample_factor_data):
        """Test rolling factor exposure calculation."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)

        result = analyzer.get_rolling_factor_exposure(window=20)

        # Should have rolling exposures for each factor
        assert len(result) == len(sample_factor_data)
        for factor_name in sample_factor_data.keys():
            assert factor_name in result

    def test_get_rolling_factor_exposure_structure(self, sample_returns, sample_factor_data):
        """Test rolling factor exposure structure."""
        from datetime import timedelta
        analyzer = FactorAnalyzer()
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(hours=i) for i in range(100)]
        analyzer.load_data(sample_returns, sample_factor_data, timestamps)

        result = analyzer.get_rolling_factor_exposure(window=20)

        for factor_exposures in result.values():
            if factor_exposures:
                entry = factor_exposures[0]
                assert "index" in entry
                assert "exposure" in entry

    def test_to_api_response(self, sample_returns, sample_factor_data):
        """Test API response conversion."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)
        result = analyzer.run_analysis()

        api_response = analyzer.to_api_response(result)

        assert "summary" in api_response
        assert "factors" in api_response
        assert "contributions" in api_response
        assert "residuals" in api_response
        assert "correlations" in api_response
        assert "timestamp" in api_response

    def test_to_api_response_summary(self, sample_returns, sample_factor_data):
        """Test API response summary section."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)
        result = analyzer.run_analysis()

        api_response = analyzer.to_api_response(result)
        summary = api_response["summary"]

        assert "total_return" in summary
        assert "factor_explained" in summary
        assert "alpha" in summary
        assert "r_squared" in summary
        assert "adjusted_r_squared" in summary

    def test_to_api_response_factors_sorted(self, sample_returns, sample_factor_data):
        """Test that factors in API response are sorted by contribution."""
        analyzer = FactorAnalyzer()
        analyzer.load_data(sample_returns, sample_factor_data)
        result = analyzer.run_analysis()

        api_response = analyzer.to_api_response(result)
        factors = api_response["factors"]

        # Should be sorted by absolute contribution descending
        for i in range(len(factors) - 1):
            assert abs(factors[i]["contribution"]) >= abs(factors[i + 1]["contribution"])


class TestFactorCalculations:
    """Tests for specific factor calculations."""

    def test_momentum_factor_positive_trend(self):
        """Test momentum factor with positive trend."""
        analyzer = FactorAnalyzer()
        # Steadily increasing prices
        prices = np.array([100 + i * 0.5 for i in range(50)])
        factors = analyzer.calculate_factors_from_prices(prices)

        # Momentum should be positive for upward trend
        # (after initial 20-period warmup)
        recent_momentum = factors["momentum"][-10:]
        assert np.mean(recent_momentum) > 0

    def test_momentum_factor_negative_trend(self):
        """Test momentum factor with negative trend."""
        analyzer = FactorAnalyzer()
        # Steadily decreasing prices
        prices = np.array([100 - i * 0.5 for i in range(50)])
        factors = analyzer.calculate_factors_from_prices(prices)

        # Momentum should be negative for downward trend
        recent_momentum = factors["momentum"][-10:]
        assert np.mean(recent_momentum) < 0

    def test_volatility_factor_high_volatility(self):
        """Test volatility factor with high volatility."""
        analyzer = FactorAnalyzer()
        np.random.seed(42)
        # High volatility prices
        prices = 100 + np.cumsum(np.random.normal(0, 5, 50))
        factors = analyzer.calculate_factors_from_prices(np.abs(prices))

        # Volatility should be measurable
        assert np.mean(factors["volatility"][-10:]) > 0

    def test_trend_factor(self):
        """Test trend factor calculation."""
        analyzer = FactorAnalyzer()
        # Strong uptrend
        prices = np.array([100 + i * 2 for i in range(50)])
        factors = analyzer.calculate_factors_from_prices(prices)

        # Trend should be positive (price above SMA)
        recent_trend = factors["trend"][-10:]
        assert np.mean(recent_trend) > 0

    def test_mean_reversion_opposite_of_trend(self):
        """Test mean reversion is opposite of trend."""
        analyzer = FactorAnalyzer()
        prices = np.array([100 + i * 2 for i in range(50)])
        factors = analyzer.calculate_factors_from_prices(prices)

        # Mean reversion should be negative of trend
        np.testing.assert_array_almost_equal(
            factors["mean_reversion"],
            -factors["trend"]
        )

    def test_sentiment_factor_placeholder(self):
        """Test sentiment factor is placeholder (zeros)."""
        analyzer = FactorAnalyzer()
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        factors = analyzer.calculate_factors_from_prices(prices)

        # Sentiment is placeholder zeros
        np.testing.assert_array_equal(
            factors["sentiment"],
            np.zeros(len(factors["sentiment"]))
        )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_same_returns(self):
        """Test with identical returns."""
        analyzer = FactorAnalyzer()
        returns = [0.01] * 50
        factor_data = {
            "factor1": [0.005] * 50,
            "factor2": [0.003] * 50,
        }
        analyzer.load_data(returns, factor_data)

        # Should handle without error
        result = analyzer.run_analysis()
        assert result is not None

    def test_zero_returns(self):
        """Test with zero returns."""
        analyzer = FactorAnalyzer()
        returns = [0.0] * 50
        factor_data = {"factor1": [0.0] * 50}
        analyzer.load_data(returns, factor_data)

        result = analyzer.run_analysis()
        assert result.total_return == 0

    def test_mismatched_lengths(self):
        """Test with mismatched data lengths."""
        analyzer = FactorAnalyzer()
        returns = [0.01] * 50
        factor_data = {
            "factor1": [0.005] * 40,  # Shorter
            "factor2": [0.003] * 60,  # Longer
        }
        analyzer.load_data(returns, factor_data)

        # Should handle by using minimum length
        result = analyzer.run_analysis()
        assert result is not None

    def test_single_factor(self):
        """Test with single factor."""
        np.random.seed(42)
        analyzer = FactorAnalyzer()
        returns = list(np.random.normal(0.001, 0.02, 50))
        factor_data = {"only_factor": list(np.random.normal(0.001, 0.015, 50))}
        analyzer.load_data(returns, factor_data)

        result = analyzer.run_analysis()
        assert len(result.factors) == 1
        assert result.factors[0].name == "only_factor"

    def test_many_factors(self):
        """Test with many factors."""
        np.random.seed(42)
        analyzer = FactorAnalyzer()
        returns = list(np.random.normal(0.001, 0.02, 50))
        factor_data = {
            f"factor_{i}": list(np.random.normal(0.001, 0.01, 50))
            for i in range(10)
        }
        analyzer.load_data(returns, factor_data)

        result = analyzer.run_analysis()
        assert len(result.factors) == 10

    def test_volume_factor_without_volume(self):
        """Test volume factor when no volume data provided."""
        analyzer = FactorAnalyzer()
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        factors = analyzer.calculate_factors_from_prices(prices)

        # Volume factor should be zeros when not provided
        np.testing.assert_array_equal(
            factors["volume"],
            np.zeros(len(factors["volume"]))
        )
