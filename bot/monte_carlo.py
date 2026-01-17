"""
Monte Carlo Simulation Module.

Simulates future portfolio paths based on historical
performance to estimate risk and return distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    num_simulations: int = 1000
    time_horizon_days: int = 252  # 1 year
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    initial_capital: float = 10000.0
    use_block_bootstrap: bool = True
    block_size: int = 5  # Days for block bootstrap


@dataclass
class SimulationPath:
    """A single simulation path."""
    final_value: float
    max_value: float
    min_value: float
    max_drawdown: float
    total_return: float
    values: List[float]


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation results."""
    config: SimulationConfig
    paths: List[SimulationPath]
    statistics: Dict[str, float]
    percentile_curves: Dict[int, List[float]]
    var_estimates: Dict[float, float]
    probability_of_loss: float
    probability_of_target: Dict[float, float]
    expected_final_value: float
    confidence_intervals: Dict[float, Tuple[float, float]]


class MonteCarloSimulator:
    """
    Monte Carlo portfolio simulation engine.

    Generates future portfolio paths using historical returns
    to estimate risk metrics and return distributions.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self._returns: np.ndarray = np.array([])

    def load_returns(self, returns: List[float]) -> None:
        """Load historical returns for simulation."""
        self._returns = np.array(returns)
        # Remove any NaN or infinite values
        self._returns = self._returns[np.isfinite(self._returns)]

    def load_equity_curve(self, equity: List[float]) -> None:
        """Load equity curve and calculate returns."""
        if len(equity) < 2:
            return

        equity_arr = np.array(equity)
        self._returns = np.diff(equity_arr) / equity_arr[:-1]
        self._returns = self._returns[np.isfinite(self._returns)]

    def run_simulation(self) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Returns:
            MonteCarloResult with all simulation outcomes
        """
        if len(self._returns) < 10:
            raise ValueError("Insufficient historical data for simulation")

        paths = []
        all_final_values = []
        all_max_drawdowns = []

        for _ in range(self.config.num_simulations):
            path = self._simulate_path()
            paths.append(path)
            all_final_values.append(path.final_value)
            all_max_drawdowns.append(path.max_drawdown)

        # Calculate statistics
        final_values = np.array(all_final_values)
        max_drawdowns = np.array(all_max_drawdowns)

        statistics = self._calculate_statistics(final_values, max_drawdowns)

        # Calculate percentile curves
        percentile_curves = self._calculate_percentile_curves(paths)

        # Calculate VaR estimates
        var_estimates = {}
        for conf in self.config.confidence_levels:
            var_estimates[conf] = self._calculate_var(final_values, conf)

        # Probability of loss
        prob_loss = np.mean(final_values < self.config.initial_capital)

        # Probability of reaching targets
        targets = [1.1, 1.25, 1.5, 2.0]  # 10%, 25%, 50%, 100% returns
        prob_targets = {}
        for target in targets:
            target_value = self.config.initial_capital * target
            prob_targets[target] = float(np.mean(final_values >= target_value))

        # Confidence intervals
        confidence_intervals = {}
        for conf in self.config.confidence_levels:
            lower = np.percentile(final_values, (1 - conf) / 2 * 100)
            upper = np.percentile(final_values, (1 + conf) / 2 * 100)
            confidence_intervals[conf] = (round(lower, 2), round(upper, 2))

        return MonteCarloResult(
            config=self.config,
            paths=paths[:100],  # Store only first 100 paths for visualization
            statistics=statistics,
            percentile_curves=percentile_curves,
            var_estimates=var_estimates,
            probability_of_loss=round(prob_loss, 4),
            probability_of_target=prob_targets,
            expected_final_value=round(float(np.mean(final_values)), 2),
            confidence_intervals=confidence_intervals,
        )

    def _simulate_path(self) -> SimulationPath:
        """Simulate a single portfolio path."""
        if self.config.use_block_bootstrap:
            simulated_returns = self._block_bootstrap()
        else:
            simulated_returns = self._simple_bootstrap()

        # Generate equity curve
        values = [self.config.initial_capital]
        for ret in simulated_returns:
            values.append(values[-1] * (1 + ret))

        values_arr = np.array(values)

        # Calculate metrics
        final_value = values[-1]
        max_value = np.max(values_arr)
        min_value = np.min(values_arr)

        # Max drawdown
        running_max = np.maximum.accumulate(values_arr)
        drawdowns = (running_max - values_arr) / running_max
        max_drawdown = float(np.max(drawdowns))

        total_return = (final_value / self.config.initial_capital - 1) * 100

        return SimulationPath(
            final_value=round(final_value, 2),
            max_value=round(max_value, 2),
            min_value=round(min_value, 2),
            max_drawdown=round(max_drawdown * 100, 2),
            total_return=round(total_return, 2),
            values=[round(v, 2) for v in values[::max(1, len(values) // 100)]],  # Downsample
        )

    def _simple_bootstrap(self) -> np.ndarray:
        """Generate returns using simple bootstrap."""
        indices = np.random.randint(0, len(self._returns), self.config.time_horizon_days)
        return self._returns[indices]

    def _block_bootstrap(self) -> np.ndarray:
        """Generate returns using block bootstrap (preserves autocorrelation)."""
        block_size = self.config.block_size
        n_blocks = int(np.ceil(self.config.time_horizon_days / block_size))

        simulated = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(self._returns) - block_size + 1)
            simulated.extend(self._returns[start_idx:start_idx + block_size])

        return np.array(simulated[:self.config.time_horizon_days])

    def _calculate_statistics(
        self, final_values: np.ndarray, max_drawdowns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate summary statistics."""
        value_std = float(np.std(final_values)) if len(final_values) else 0.0
        return {
            "mean_final_value": round(float(np.mean(final_values)), 2),
            "median_final_value": round(float(np.median(final_values)), 2),
            "std_final_value": round(float(np.std(final_values)), 2),
            "min_final_value": round(float(np.min(final_values)), 2),
            "max_final_value": round(float(np.max(final_values)), 2),
            "mean_return_pct": round((float(np.mean(final_values)) / self.config.initial_capital - 1) * 100, 2),
            "median_return_pct": round((float(np.median(final_values)) / self.config.initial_capital - 1) * 100, 2),
            "mean_max_drawdown_pct": round(float(np.mean(max_drawdowns)), 2),
            "median_max_drawdown_pct": round(float(np.median(max_drawdowns)), 2),
            "worst_max_drawdown_pct": round(float(np.max(max_drawdowns)), 2),
            "skewness": 0.0 if value_std < 1e-8 else round(float(stats.skew(final_values)), 2),
            "kurtosis": 0.0 if value_std < 1e-8 else round(float(stats.kurtosis(final_values)), 2),
        }

    def _calculate_percentile_curves(
        self, paths: List[SimulationPath]
    ) -> Dict[int, List[float]]:
        """Calculate percentile curves across all simulations."""
        # Align all paths to same length
        max_len = max(len(p.values) for p in paths)
        aligned = np.zeros((len(paths), max_len))

        for i, path in enumerate(paths):
            # Interpolate to common length
            x_old = np.linspace(0, 1, len(path.values))
            x_new = np.linspace(0, 1, max_len)
            aligned[i] = np.interp(x_new, x_old, path.values)

        percentiles = {}
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[pct] = [round(float(v), 2) for v in np.percentile(aligned, pct, axis=0)]

        return percentiles

    def _calculate_var(self, final_values: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        loss_values = self.config.initial_capital - final_values
        var = np.percentile(loss_values, confidence * 100)
        return round(float(max(0, var)), 2)

    def run_stress_test(
        self,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run stress tests with different market scenarios.

        Args:
            scenarios: Dict of scenario name to return multiplier

        Returns:
            Results for each scenario
        """
        if scenarios is None:
            scenarios = {
                "base": 1.0,
                "mild_stress": 0.7,
                "moderate_stress": 0.5,
                "severe_stress": 0.3,
                "black_swan": 0.1,
            }

        results = {}

        for scenario_name, multiplier in scenarios.items():
            # Adjust returns for scenario
            stressed_returns = self._returns * multiplier

            # Store original returns
            original_returns = self._returns.copy()
            self._returns = stressed_returns

            # Run simulation
            try:
                sim_result = self.run_simulation()
                results[scenario_name] = {
                    "expected_return": sim_result.statistics["mean_return_pct"],
                    "probability_of_loss": sim_result.probability_of_loss,
                    "expected_max_drawdown": sim_result.statistics["mean_max_drawdown_pct"],
                    "var_95": sim_result.var_estimates.get(0.95, 0),
                    "expected_final_value": sim_result.expected_final_value,
                }
            except Exception as e:
                logger.error(f"Stress test error for {scenario_name}: {e}")
                results[scenario_name] = {"error": str(e)}
            finally:
                # Restore original returns
                self._returns = original_returns

        return results

    def to_api_response(self, result: MonteCarloResult) -> Dict[str, Any]:
        """Convert result to API response format."""
        return {
            "config": {
                "num_simulations": result.config.num_simulations,
                "time_horizon_days": result.config.time_horizon_days,
                "initial_capital": result.config.initial_capital,
            },
            "summary": {
                "expected_final_value": result.expected_final_value,
                "probability_of_loss": result.probability_of_loss,
                "expected_return_pct": result.statistics["mean_return_pct"],
                "expected_max_drawdown_pct": result.statistics["mean_max_drawdown_pct"],
            },
            "statistics": result.statistics,
            "risk": {
                "var_estimates": {f"{int(k*100)}%": v for k, v in result.var_estimates.items()},
                "confidence_intervals": {
                    f"{int(k*100)}%": {"lower": v[0], "upper": v[1]}
                    for k, v in result.confidence_intervals.items()
                },
            },
            "targets": {
                f"{int((k-1)*100)}%_gain": round(v * 100, 1)
                for k, v in result.probability_of_target.items()
            },
            "percentile_curves": result.percentile_curves,
            "sample_paths": [
                {"values": p.values, "final": p.final_value, "max_dd": p.max_drawdown}
                for p in result.paths[:10]  # Return first 10 paths for charting
            ],
        }
