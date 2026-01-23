"""
Tests for Monte Carlo Simulation Module.

Tests the Monte Carlo simulation engine, including
bootstrap methods, statistics calculations, and stress testing.
"""

import pytest
import numpy as np
from typing import List

from bot.monte_carlo import (
    SimulationConfig,
    SimulationPath,
    MonteCarloResult,
    MonteCarloSimulator,
)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.num_simulations == 1000
        assert config.time_horizon_days == 252
        assert config.confidence_levels == [0.95, 0.99]
        assert config.initial_capital == 10000.0
        assert config.use_block_bootstrap is True
        assert config.block_size == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            num_simulations=500,
            time_horizon_days=126,
            confidence_levels=[0.90, 0.95],
            initial_capital=50000.0,
            use_block_bootstrap=False,
            block_size=10,
        )
        assert config.num_simulations == 500
        assert config.time_horizon_days == 126
        assert config.confidence_levels == [0.90, 0.95]
        assert config.initial_capital == 50000.0
        assert config.use_block_bootstrap is False
        assert config.block_size == 10

    def test_config_with_single_confidence_level(self):
        """Test configuration with single confidence level."""
        config = SimulationConfig(confidence_levels=[0.95])
        assert len(config.confidence_levels) == 1
        assert config.confidence_levels[0] == 0.95


class TestSimulationPath:
    """Tests for SimulationPath dataclass."""

    def test_simulation_path_creation(self):
        """Test creating a SimulationPath."""
        path = SimulationPath(
            final_value=12000.0,
            max_value=15000.0,
            min_value=8000.0,
            max_drawdown=20.0,
            total_return=20.0,
            values=[10000, 11000, 12000],
        )
        assert path.final_value == 12000.0
        assert path.max_value == 15000.0
        assert path.min_value == 8000.0
        assert path.max_drawdown == 20.0
        assert path.total_return == 20.0
        assert len(path.values) == 3

    def test_simulation_path_with_negative_return(self):
        """Test SimulationPath with negative return."""
        path = SimulationPath(
            final_value=8000.0,
            max_value=10000.0,
            min_value=7000.0,
            max_drawdown=30.0,
            total_return=-20.0,
            values=[10000, 8000],
        )
        assert path.total_return == -20.0
        assert path.final_value < 10000


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_monte_carlo_result_structure(self):
        """Test MonteCarloResult structure."""
        config = SimulationConfig()
        path = SimulationPath(
            final_value=11000,
            max_value=12000,
            min_value=9000,
            max_drawdown=10.0,
            total_return=10.0,
            values=[10000, 11000],
        )
        result = MonteCarloResult(
            config=config,
            paths=[path],
            statistics={"mean_final_value": 11000},
            percentile_curves={50: [10000, 11000]},
            var_estimates={0.95: 500},
            probability_of_loss=0.3,
            probability_of_target={1.1: 0.6},
            expected_final_value=11000,
            confidence_intervals={0.95: (9000, 13000)},
        )
        assert result.config == config
        assert len(result.paths) == 1
        assert result.expected_final_value == 11000
        assert result.probability_of_loss == 0.3


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    @pytest.fixture
    def sample_returns(self) -> List[float]:
        """Generate sample returns for testing."""
        np.random.seed(42)
        return list(np.random.normal(0.001, 0.02, 100))

    @pytest.fixture
    def sample_equity(self) -> List[float]:
        """Generate sample equity curve for testing."""
        np.random.seed(42)
        equity = [10000]
        returns = np.random.normal(0.001, 0.02, 100)
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        return equity

    def test_init_default_config(self):
        """Test initialization with default config."""
        simulator = MonteCarloSimulator()
        assert simulator.config.num_simulations == 1000
        assert len(simulator._returns) == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SimulationConfig(num_simulations=100)
        simulator = MonteCarloSimulator(config=config)
        assert simulator.config.num_simulations == 100

    def test_load_returns(self, sample_returns):
        """Test loading returns data."""
        simulator = MonteCarloSimulator()
        simulator.load_returns(sample_returns)
        assert len(simulator._returns) == 100

    def test_load_returns_filters_nan(self):
        """Test that loading returns filters NaN values."""
        simulator = MonteCarloSimulator()
        returns_with_nan = [0.01, 0.02, np.nan, 0.03, np.inf, -np.inf]
        simulator.load_returns(returns_with_nan)
        assert len(simulator._returns) == 3
        assert not np.any(np.isnan(simulator._returns))

    def test_load_equity_curve(self, sample_equity):
        """Test loading equity curve and calculating returns."""
        simulator = MonteCarloSimulator()
        simulator.load_equity_curve(sample_equity)
        assert len(simulator._returns) > 0
        # Returns should be approximately 100 (one less than equity points)
        assert len(simulator._returns) <= 100

    def test_load_equity_curve_insufficient_data(self):
        """Test loading equity curve with insufficient data."""
        simulator = MonteCarloSimulator()
        simulator.load_equity_curve([10000])  # Only one point
        assert len(simulator._returns) == 0

    def test_run_simulation_insufficient_data(self):
        """Test simulation raises error with insufficient data."""
        simulator = MonteCarloSimulator()
        simulator.load_returns([0.01, 0.02, 0.03])  # Only 3 points
        with pytest.raises(ValueError, match="Insufficient historical data"):
            simulator.run_simulation()

    def test_run_simulation_basic(self, sample_returns):
        """Test basic simulation run."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        assert result.config == config
        assert len(result.paths) <= 100  # Capped at 100
        assert result.expected_final_value > 0
        assert 0 <= result.probability_of_loss <= 1

    def test_run_simulation_statistics(self, sample_returns):
        """Test that simulation calculates statistics correctly."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        stats = result.statistics
        assert "mean_final_value" in stats
        assert "median_final_value" in stats
        assert "std_final_value" in stats
        assert "min_final_value" in stats
        assert "max_final_value" in stats
        assert "mean_return_pct" in stats
        assert "median_return_pct" in stats
        assert "mean_max_drawdown_pct" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_run_simulation_var_estimates(self, sample_returns):
        """Test that simulation calculates VaR estimates."""
        config = SimulationConfig(
            num_simulations=50, time_horizon_days=30, confidence_levels=[0.95, 0.99]
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        assert 0.95 in result.var_estimates
        assert 0.99 in result.var_estimates
        # VaR at 99% should be >= VaR at 95%
        assert result.var_estimates[0.99] >= result.var_estimates[0.95]

    def test_run_simulation_confidence_intervals(self, sample_returns):
        """Test that simulation calculates confidence intervals."""
        config = SimulationConfig(
            num_simulations=50, time_horizon_days=30, confidence_levels=[0.95]
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        assert 0.95 in result.confidence_intervals
        lower, upper = result.confidence_intervals[0.95]
        assert lower < upper

    def test_run_simulation_probability_of_target(self, sample_returns):
        """Test probability of target calculations."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        # Should have probabilities for standard targets
        assert 1.1 in result.probability_of_target  # 10% gain
        assert 1.25 in result.probability_of_target  # 25% gain
        assert 1.5 in result.probability_of_target  # 50% gain
        assert 2.0 in result.probability_of_target  # 100% gain

        # All probabilities should be between 0 and 1
        for prob in result.probability_of_target.values():
            assert 0 <= prob <= 1

    def test_run_simulation_percentile_curves(self, sample_returns):
        """Test percentile curves calculation."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()

        # Should have standard percentiles
        assert 5 in result.percentile_curves
        assert 50 in result.percentile_curves
        assert 95 in result.percentile_curves

        # 95th percentile should be >= 50th percentile at final point
        assert result.percentile_curves[95][-1] >= result.percentile_curves[50][-1]
        assert result.percentile_curves[50][-1] >= result.percentile_curves[5][-1]

    def test_simple_bootstrap(self, sample_returns):
        """Test simple bootstrap method."""
        config = SimulationConfig(
            num_simulations=10, time_horizon_days=20, use_block_bootstrap=False
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        assert len(result.paths) > 0

    def test_block_bootstrap(self, sample_returns):
        """Test block bootstrap method."""
        config = SimulationConfig(
            num_simulations=10, time_horizon_days=20, use_block_bootstrap=True, block_size=5
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        assert len(result.paths) > 0

    def test_simulate_path_metrics(self, sample_returns):
        """Test that simulated path has correct metrics."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20, initial_capital=10000)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        path = result.paths[0]

        # Path should have valid metrics
        assert path.final_value > 0
        assert path.max_value >= path.final_value
        assert path.min_value <= path.final_value
        assert path.max_value >= path.min_value
        assert 0 <= path.max_drawdown <= 100

    def test_stress_test_default_scenarios(self, sample_returns):
        """Test stress test with default scenarios."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        results = simulator.run_stress_test()

        assert "base" in results
        assert "mild_stress" in results
        assert "moderate_stress" in results
        assert "severe_stress" in results
        assert "black_swan" in results

    def test_stress_test_custom_scenarios(self, sample_returns):
        """Test stress test with custom scenarios."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        scenarios = {
            "normal": 1.0,
            "bull_market": 1.5,
            "crash": 0.2,
        }
        results = simulator.run_stress_test(scenarios)

        assert "normal" in results
        assert "bull_market" in results
        assert "crash" in results

    def test_stress_test_returns_expected_keys(self, sample_returns):
        """Test that stress test results have expected keys."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        results = simulator.run_stress_test()
        base_result = results["base"]

        assert "expected_return" in base_result
        assert "probability_of_loss" in base_result
        assert "expected_max_drawdown" in base_result
        assert "var_95" in base_result
        assert "expected_final_value" in base_result

    def test_stress_test_preserves_original_returns(self, sample_returns):
        """Test that stress test preserves original returns."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        original_returns = simulator._returns.copy()
        simulator.run_stress_test()

        np.testing.assert_array_equal(simulator._returns, original_returns)

    def test_to_api_response(self, sample_returns):
        """Test API response conversion."""
        config = SimulationConfig(num_simulations=10, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        api_response = simulator.to_api_response(result)

        assert "config" in api_response
        assert "summary" in api_response
        assert "statistics" in api_response
        assert "risk" in api_response
        assert "targets" in api_response
        assert "percentile_curves" in api_response
        assert "sample_paths" in api_response

    def test_to_api_response_config_section(self, sample_returns):
        """Test API response config section."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30, initial_capital=25000)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        api_response = simulator.to_api_response(result)

        assert api_response["config"]["num_simulations"] == 50
        assert api_response["config"]["time_horizon_days"] == 30
        assert api_response["config"]["initial_capital"] == 25000

    def test_to_api_response_summary_section(self, sample_returns):
        """Test API response summary section."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        api_response = simulator.to_api_response(result)

        summary = api_response["summary"]
        assert "expected_final_value" in summary
        assert "probability_of_loss" in summary
        assert "expected_return_pct" in summary
        assert "expected_max_drawdown_pct" in summary

    def test_to_api_response_sample_paths_limited(self, sample_returns):
        """Test that sample paths in API response are limited."""
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(sample_returns)

        result = simulator.run_simulation()
        api_response = simulator.to_api_response(result)

        # Should be limited to 10 paths
        assert len(api_response["sample_paths"]) <= 10


class TestSimulationStatistics:
    """Tests for statistics calculations."""

    @pytest.fixture
    def simulator_with_data(self) -> MonteCarloSimulator:
        """Create simulator with sample data."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(num_simulations=100, time_horizon_days=50)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)
        return simulator

    def test_statistics_values_reasonable(self, simulator_with_data):
        """Test that statistics values are reasonable."""
        result = simulator_with_data.run_simulation()
        stats = result.statistics

        # Mean and median should be relatively close
        assert abs(stats["mean_final_value"] - stats["median_final_value"]) < 5000

        # Standard deviation should be positive
        assert stats["std_final_value"] > 0

        # Min should be less than or equal to max
        assert stats["min_final_value"] <= stats["max_final_value"]

    def test_drawdown_statistics(self, simulator_with_data):
        """Test drawdown statistics."""
        result = simulator_with_data.run_simulation()
        stats = result.statistics

        # Drawdown percentages should be non-negative
        assert stats["mean_max_drawdown_pct"] >= 0
        assert stats["median_max_drawdown_pct"] >= 0
        assert stats["worst_max_drawdown_pct"] >= 0

        # Worst should be >= mean
        assert stats["worst_max_drawdown_pct"] >= stats["mean_max_drawdown_pct"]


class TestVaRCalculations:
    """Tests for Value at Risk calculations."""

    def test_var_increases_with_confidence(self):
        """Test that VaR increases with higher confidence levels."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(
            num_simulations=200, time_horizon_days=50, confidence_levels=[0.90, 0.95, 0.99]
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()

        # VaR should generally increase with confidence level
        # (higher confidence = worse case scenario)
        assert result.var_estimates[0.99] >= result.var_estimates[0.90]

    def test_var_non_negative(self):
        """Test that VaR estimates are non-negative."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(
            num_simulations=50, time_horizon_days=30, confidence_levels=[0.95, 0.99]
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()

        for var in result.var_estimates.values():
            assert var >= 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_positive_returns(self):
        """Test simulation with all positive returns."""
        returns = [0.01, 0.02, 0.015, 0.008, 0.012] * 20  # 100 positive returns
        config = SimulationConfig(num_simulations=20, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()

        # With all positive returns, probability of loss should be low
        assert result.probability_of_loss < 0.5

    def test_all_negative_returns(self):
        """Test simulation with all negative returns."""
        returns = [-0.01, -0.02, -0.015, -0.008, -0.012] * 20  # 100 negative returns
        config = SimulationConfig(num_simulations=20, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()

        # With all negative returns, probability of loss should be high
        assert result.probability_of_loss > 0.5

    def test_mixed_returns_zero_mean(self):
        """Test simulation with returns that have approximately zero mean."""
        returns = [0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 17  # ~102 returns
        config = SimulationConfig(num_simulations=20, time_horizon_days=20)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns[:100])

        result = simulator.run_simulation()

        # Probability of loss should be around 50%
        assert 0.3 <= result.probability_of_loss <= 0.7

    def test_large_number_of_simulations(self):
        """Test with larger number of simulations."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(num_simulations=500, time_horizon_days=30)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()

        # Should complete without error
        assert result.expected_final_value > 0
        # Paths should be capped at 100 for visualization
        assert len(result.paths) == 100

    def test_minimum_data_requirement(self):
        """Test with minimum required data (10 points)."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003, 0.01, -0.005]
        config = SimulationConfig(num_simulations=10, time_horizon_days=10)
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        # Should work with exactly 10 points
        result = simulator.run_simulation()
        assert result.expected_final_value > 0

    def test_small_block_size(self):
        """Test block bootstrap with small block size."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 50))
        config = SimulationConfig(
            num_simulations=10, time_horizon_days=20, use_block_bootstrap=True, block_size=2
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()
        assert result.expected_final_value > 0

    def test_large_block_size(self):
        """Test block bootstrap with larger block size."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(
            num_simulations=10, time_horizon_days=20, use_block_bootstrap=True, block_size=10
        )
        simulator = MonteCarloSimulator(config=config)
        simulator.load_returns(returns)

        result = simulator.run_simulation()
        assert result.expected_final_value > 0


class TestReproducibility:
    """Tests for reproducibility with random seed."""

    def test_reproducible_results_with_seed(self):
        """Test that results are reproducible with same seed."""
        returns = list(np.random.normal(0.001, 0.02, 100))
        config = SimulationConfig(num_simulations=50, time_horizon_days=30)

        np.random.seed(123)
        simulator1 = MonteCarloSimulator(config=config)
        simulator1.load_returns(returns)
        result1 = simulator1.run_simulation()

        np.random.seed(123)
        simulator2 = MonteCarloSimulator(config=config)
        simulator2.load_returns(returns)
        result2 = simulator2.run_simulation()

        # Results should be identical with same seed
        assert result1.expected_final_value == result2.expected_final_value
        assert result1.probability_of_loss == result2.probability_of_loss
