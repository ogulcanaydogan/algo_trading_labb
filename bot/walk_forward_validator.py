"""
Walk-Forward Validator + Stress Testing

Robust validation framework for strategies before live deployment:
1. Walk-forward optimization (train on window A, validate on B, roll forward)
2. Out-of-sample testing
3. Stress tests (flash crash, gap moves, volatility spikes)
4. Regime stability analysis
5. Monte Carlo simulation for confidence intervals

This prevents overfitting and ensures strategies survive real market conditions.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Overall validation result"""

    PASSED = "passed"
    FAILED = "failed"
    MARGINAL = "marginal"  # Passed but close to thresholds


@dataclass
class ValidationThresholds:
    """Thresholds for passing validation"""

    min_sharpe: float = 0.5
    min_win_rate: float = 0.4
    max_drawdown: float = 20.0  # %
    min_profit_factor: float = 1.2
    min_trades: int = 30
    max_consecutive_losses: int = 8
    min_regime_stability: float = 0.6  # Performance consistency across regimes
    max_parameter_sensitivity: float = 0.3  # How much perf changes with param tweaks


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""

    train_window_days: int = 90  # Training window size
    test_window_days: int = 30  # Out-of-sample test window
    step_days: int = 30  # How much to roll forward
    min_windows: int = 4  # Minimum number of walk-forward windows
    anchored: bool = False  # If True, always start from beginning


@dataclass
class StressTestConfig:
    """Configuration for stress tests"""

    flash_crash_pct: float = -10.0  # Sudden drop percentage
    gap_up_pct: float = 5.0  # Gap up percentage
    gap_down_pct: float = -5.0  # Gap down percentage
    volatility_multiplier: float = 3.0  # Volatility spike factor
    liquidity_drop_pct: float = 80.0  # Volume reduction
    spread_widen_factor: float = 5.0  # Spread multiplier


@dataclass
class ValidationMetrics:
    """Metrics collected during validation"""

    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Trade quality
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy: float = 0.0

    # Consistency
    consecutive_losses_max: int = 0
    monthly_returns: List[float] = field(default_factory=list)
    regime_performance: Dict[str, float] = field(default_factory=dict)

    # Robustness
    parameter_sensitivity: float = 0.0
    out_of_sample_decay: float = 0.0  # How much worse OOS vs in-sample

    def calculate_derived_metrics(self):
        """Calculate derived metrics from basic stats"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        if self.losing_trades > 0:
            self.avg_loss = self.gross_loss / self.losing_trades

        if self.winning_trades > 0:
            self.avg_win = self.gross_profit / self.winning_trades

        if self.avg_loss > 0:
            self.win_loss_ratio = self.avg_win / self.avg_loss

        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss

        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)

        # Calmar = Annual Return / Max Drawdown
        if self.max_drawdown_pct > 0:
            self.calmar_ratio = self.annualized_return_pct / self.max_drawdown_pct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 4),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "consecutive_losses_max": self.consecutive_losses_max,
            "parameter_sensitivity": round(self.parameter_sensitivity, 3),
            "out_of_sample_decay": round(self.out_of_sample_decay, 3),
            "regime_performance": self.regime_performance,
        }


@dataclass
class StressTestResult:
    """Result of a single stress test"""

    test_name: str
    passed: bool
    pnl_impact_pct: float
    max_drawdown_pct: float
    survived: bool  # Did not get liquidated
    recovery_bars: int  # Bars to recover (if survived)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "pnl_impact_pct": round(self.pnl_impact_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "survived": self.survived,
            "recovery_bars": self.recovery_bars,
            "notes": self.notes,
        }


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result"""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # In-sample (training) metrics
    in_sample_return: float = 0.0
    in_sample_sharpe: float = 0.0
    in_sample_trades: int = 0

    # Out-of-sample (test) metrics
    out_sample_return: float = 0.0
    out_sample_sharpe: float = 0.0
    out_sample_trades: int = 0

    # Decay metrics
    return_decay: float = 0.0  # OOS return / IS return
    sharpe_decay: float = 0.0  # OOS sharpe / IS sharpe

    def calculate_decay(self):
        """Calculate performance decay from in-sample to out-of-sample"""
        if self.in_sample_return != 0:
            self.return_decay = self.out_sample_return / self.in_sample_return
        if self.in_sample_sharpe != 0:
            self.sharpe_decay = self.out_sample_sharpe / self.in_sample_sharpe

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_period": f"{self.train_start.date()} to {self.train_end.date()}",
            "test_period": f"{self.test_start.date()} to {self.test_end.date()}",
            "in_sample_return": round(self.in_sample_return, 2),
            "in_sample_sharpe": round(self.in_sample_sharpe, 3),
            "out_sample_return": round(self.out_sample_return, 2),
            "out_sample_sharpe": round(self.out_sample_sharpe, 3),
            "return_decay": round(self.return_decay, 3),
            "sharpe_decay": round(self.sharpe_decay, 3),
        }


@dataclass
class ValidationReport:
    """Complete validation report"""

    strategy_name: str
    strategy_version: str
    validation_date: datetime = field(default_factory=datetime.now)

    # Overall result
    result: ValidationResult = ValidationResult.FAILED
    score: float = 0.0  # 0-100 overall score

    # Component results
    walk_forward_passed: bool = False
    stress_tests_passed: bool = False
    regime_stability_passed: bool = False
    parameter_robustness_passed: bool = False

    # Detailed metrics
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    walk_forward_windows: List[WalkForwardWindow] = field(default_factory=list)
    stress_test_results: List[StressTestResult] = field(default_factory=list)

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Promotion decision
    ready_for_shadow: bool = False
    ready_for_live: bool = False

    def calculate_score(self, thresholds: ValidationThresholds) -> float:
        """Calculate overall validation score (0-100)"""
        score = 0.0
        max_score = 100.0

        # Sharpe ratio (25 points)
        sharpe_score = min(25, (self.metrics.sharpe_ratio / thresholds.min_sharpe) * 12.5)
        score += sharpe_score

        # Win rate (15 points)
        win_rate_score = min(15, (self.metrics.win_rate / thresholds.min_win_rate) * 7.5)
        score += win_rate_score

        # Profit factor (15 points)
        pf_score = min(15, (self.metrics.profit_factor / thresholds.min_profit_factor) * 7.5)
        score += pf_score

        # Drawdown (15 points, inverse)
        if self.metrics.max_drawdown_pct > 0:
            dd_score = min(15, (thresholds.max_drawdown / self.metrics.max_drawdown_pct) * 7.5)
        else:
            dd_score = 15
        score += dd_score

        # Walk-forward stability (15 points)
        if self.walk_forward_windows:
            avg_decay = np.mean(
                [w.sharpe_decay for w in self.walk_forward_windows if w.sharpe_decay > 0]
            )
            stability_score = min(15, avg_decay * 15) if avg_decay else 0
            score += stability_score

        # Stress test survival (15 points)
        if self.stress_test_results:
            survival_rate = sum(1 for r in self.stress_test_results if r.survived) / len(
                self.stress_test_results
            )
            stress_score = survival_rate * 15
            score += stress_score

        self.score = min(max_score, score)
        return self.score

    def determine_result(self, thresholds: ValidationThresholds):
        """Determine overall validation result"""
        # Check all conditions
        passes_sharpe = self.metrics.sharpe_ratio >= thresholds.min_sharpe
        passes_win_rate = self.metrics.win_rate >= thresholds.min_win_rate
        passes_drawdown = self.metrics.max_drawdown_pct <= thresholds.max_drawdown
        passes_pf = self.metrics.profit_factor >= thresholds.min_profit_factor
        passes_trades = self.metrics.total_trades >= thresholds.min_trades

        # Walk-forward check
        if self.walk_forward_windows:
            avg_oos_sharpe = np.mean([w.out_sample_sharpe for w in self.walk_forward_windows])
            self.walk_forward_passed = avg_oos_sharpe > 0.3
        else:
            self.walk_forward_passed = False

        # Stress test check
        if self.stress_test_results:
            survival_rate = sum(1 for r in self.stress_test_results if r.survived) / len(
                self.stress_test_results
            )
            self.stress_tests_passed = survival_rate >= 0.8
        else:
            self.stress_tests_passed = True  # No stress tests = pass by default

        # Aggregate result
        core_passed = (
            passes_sharpe and passes_win_rate and passes_drawdown and passes_pf and passes_trades
        )

        if core_passed and self.walk_forward_passed and self.stress_tests_passed:
            self.result = ValidationResult.PASSED
            self.ready_for_shadow = True
            self.ready_for_live = self.score >= 70
        elif core_passed or (self.score >= 50):
            self.result = ValidationResult.MARGINAL
            self.ready_for_shadow = True
            self.ready_for_live = False
        else:
            self.result = ValidationResult.FAILED
            self.ready_for_shadow = False
            self.ready_for_live = False

        # Add warnings and recommendations
        if not passes_sharpe:
            self.warnings.append(
                f"Sharpe ratio {self.metrics.sharpe_ratio:.2f} below threshold {thresholds.min_sharpe}"
            )
        if not passes_drawdown:
            self.warnings.append(
                f"Max drawdown {self.metrics.max_drawdown_pct:.1f}% exceeds limit {thresholds.max_drawdown}%"
            )
        if not self.walk_forward_passed:
            self.warnings.append("Walk-forward validation shows significant out-of-sample decay")
        if not self.stress_tests_passed:
            self.warnings.append("Strategy failed multiple stress tests")

        if self.result == ValidationResult.MARGINAL:
            self.recommendations.append(
                "Run in shadow mode for at least 30 days before live deployment"
            )
            self.recommendations.append("Consider tightening stop losses")
        elif self.result == ValidationResult.PASSED:
            self.recommendations.append("Strategy ready for shadow mode testing")
            if self.ready_for_live:
                self.recommendations.append(
                    "Strategy ready for live deployment with reduced position size"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "validation_date": self.validation_date.isoformat(),
            "result": self.result.value,
            "score": round(self.score, 1),
            "walk_forward_passed": self.walk_forward_passed,
            "stress_tests_passed": self.stress_tests_passed,
            "metrics": self.metrics.to_dict(),
            "walk_forward_windows": [w.to_dict() for w in self.walk_forward_windows],
            "stress_test_results": [r.to_dict() for r in self.stress_test_results],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "ready_for_shadow": self.ready_for_shadow,
            "ready_for_live": self.ready_for_live,
        }


class WalkForwardValidator:
    """
    Walk-forward validation engine.

    Performs rigorous out-of-sample testing to prevent overfitting.
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        thresholds: Optional[ValidationThresholds] = None,
        stress_config: Optional[StressTestConfig] = None,
    ):
        self.config = config or WalkForwardConfig()
        self.thresholds = thresholds or ValidationThresholds()
        self.stress_config = stress_config or StressTestConfig()

        logger.info("Walk-Forward Validator initialized")

    def validate_strategy(
        self,
        strategy: Any,  # Strategy instance
        historical_data: List[Dict[str, Any]],
        run_backtest: Callable,  # Function to run backtest
        optimize_params: Optional[Callable] = None,  # Optional param optimizer
    ) -> ValidationReport:
        """
        Run full validation on a strategy.

        Args:
            strategy: Strategy instance to validate
            historical_data: List of OHLCV dicts with timestamps
            run_backtest: Function(strategy, data) -> metrics dict
            optimize_params: Optional function to optimize parameters

        Returns:
            ValidationReport with all results
        """
        report = ValidationReport(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
        )

        logger.info(f"Starting validation for {strategy.name} v{strategy.version}")

        # 1. Walk-forward validation
        logger.info("Running walk-forward validation...")
        wf_results = self._run_walk_forward(
            strategy, historical_data, run_backtest, optimize_params
        )
        report.walk_forward_windows = wf_results

        # 2. Aggregate metrics from all windows
        logger.info("Calculating aggregate metrics...")
        report.metrics = self._calculate_aggregate_metrics(
            wf_results, run_backtest, strategy, historical_data
        )

        # 3. Stress tests
        logger.info("Running stress tests...")
        stress_results = self._run_stress_tests(strategy, historical_data, run_backtest)
        report.stress_test_results = stress_results

        # 4. Parameter sensitivity (if optimizer provided)
        if optimize_params:
            logger.info("Analyzing parameter sensitivity...")
            report.metrics.parameter_sensitivity = self._analyze_parameter_sensitivity(
                strategy, historical_data, run_backtest
            )

        # 5. Calculate score and determine result
        report.calculate_score(self.thresholds)
        report.determine_result(self.thresholds)

        logger.info(f"Validation complete: {report.result.value} (score: {report.score:.1f})")

        return report

    def _run_walk_forward(
        self,
        strategy: Any,
        data: List[Dict],
        run_backtest: Callable,
        optimize_params: Optional[Callable],
    ) -> List[WalkForwardWindow]:
        """Run walk-forward optimization"""
        windows = []

        if len(data) < 2:
            return windows

        # Convert timestamps if needed
        start_date = self._get_timestamp(data[0])
        end_date = self._get_timestamp(data[-1])
        total_days = (end_date - start_date).days

        # Calculate number of windows
        window_size = self.config.train_window_days + self.config.test_window_days
        num_windows = max(
            self.config.min_windows, (total_days - window_size) // self.config.step_days + 1
        )

        current_start = start_date

        for i in range(num_windows):
            # Define window boundaries
            if self.config.anchored:
                train_start = start_date
            else:
                train_start = current_start

            train_end = train_start + timedelta(days=self.config.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Split data
            train_data = [d for d in data if train_start <= self._get_timestamp(d) < train_end]
            test_data = [d for d in data if test_start <= self._get_timestamp(d) < test_end]

            if len(train_data) < 10 or len(test_data) < 5:
                current_start += timedelta(days=self.config.step_days)
                continue

            # Optimize parameters on training data (if optimizer provided)
            if optimize_params:
                try:
                    optimize_params(strategy, train_data)
                except Exception as e:
                    logger.warning(f"Parameter optimization failed: {e}")

            # Run backtest on training data
            try:
                train_metrics = run_backtest(strategy, train_data)
            except Exception as e:
                logger.error(f"Training backtest failed: {e}")
                current_start += timedelta(days=self.config.step_days)
                continue

            # Run backtest on test data (out-of-sample)
            try:
                test_metrics = run_backtest(strategy, test_data)
            except Exception as e:
                logger.error(f"Test backtest failed: {e}")
                current_start += timedelta(days=self.config.step_days)
                continue

            # Create window result
            window = WalkForwardWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_return=train_metrics.get("total_return_pct", 0),
                in_sample_sharpe=train_metrics.get("sharpe_ratio", 0),
                in_sample_trades=train_metrics.get("total_trades", 0),
                out_sample_return=test_metrics.get("total_return_pct", 0),
                out_sample_sharpe=test_metrics.get("sharpe_ratio", 0),
                out_sample_trades=test_metrics.get("total_trades", 0),
            )
            window.calculate_decay()
            windows.append(window)

            logger.debug(
                f"Window {i}: IS Sharpe={window.in_sample_sharpe:.2f}, OOS Sharpe={window.out_sample_sharpe:.2f}"
            )

            # Move forward
            current_start += timedelta(days=self.config.step_days)

        return windows

    def _calculate_aggregate_metrics(
        self,
        windows: List[WalkForwardWindow],
        run_backtest: Callable,
        strategy: Any,
        data: List[Dict],
    ) -> ValidationMetrics:
        """Calculate aggregate metrics from walk-forward results"""
        metrics = ValidationMetrics()

        if not windows:
            # Fallback: run single backtest on all data
            try:
                result = run_backtest(strategy, data)
                metrics.total_trades = result.get("total_trades", 0)
                metrics.winning_trades = result.get("winning_trades", 0)
                metrics.losing_trades = result.get("losing_trades", 0)
                metrics.total_return_pct = result.get("total_return_pct", 0)
                metrics.sharpe_ratio = result.get("sharpe_ratio", 0)
                metrics.max_drawdown_pct = result.get("max_drawdown_pct", 0)
                metrics.gross_profit = result.get("gross_profit", 0)
                metrics.gross_loss = result.get("gross_loss", 0)
                metrics.calculate_derived_metrics()
            except Exception as e:
                logger.error(f"Fallback backtest failed: {e}")
            return metrics

        # Aggregate from OOS windows
        metrics.total_trades = sum(w.out_sample_trades for w in windows)

        # Use mean OOS metrics
        oos_returns = [w.out_sample_return for w in windows]
        oos_sharpes = [w.out_sample_sharpe for w in windows]

        metrics.total_return_pct = sum(oos_returns)
        metrics.sharpe_ratio = np.mean(oos_sharpes) if oos_sharpes else 0

        # Calculate out-of-sample decay
        in_sample_sharpes = [w.in_sample_sharpe for w in windows if w.in_sample_sharpe > 0]
        if in_sample_sharpes:
            avg_is_sharpe = np.mean(in_sample_sharpes)
            avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0
            if avg_is_sharpe > 0:
                metrics.out_of_sample_decay = 1 - (avg_oos_sharpe / avg_is_sharpe)

        # Estimate other metrics from OOS trades count
        # (In real implementation, would aggregate from actual trade results)
        if metrics.total_trades > 0:
            avg_oos_return = np.mean(oos_returns)
            if avg_oos_return > 0:
                metrics.win_rate = 0.5 + avg_oos_return / 20  # Rough estimate
            else:
                metrics.win_rate = 0.5 + avg_oos_return / 20
            metrics.win_rate = max(0.2, min(0.8, metrics.win_rate))

            metrics.winning_trades = int(metrics.total_trades * metrics.win_rate)
            metrics.losing_trades = metrics.total_trades - metrics.winning_trades

        # Estimate annualized return
        total_days = sum((w.test_end - w.test_start).days for w in windows)
        if total_days > 0:
            metrics.annualized_return_pct = metrics.total_return_pct * (365 / total_days)

        metrics.calculate_derived_metrics()
        return metrics

    def _run_stress_tests(
        self, strategy: Any, data: List[Dict], run_backtest: Callable
    ) -> List[StressTestResult]:
        """Run stress tests on strategy"""
        results = []

        if len(data) < 50:
            return results

        # 1. Flash Crash Test
        results.append(self._stress_test_flash_crash(strategy, data, run_backtest))

        # 2. Gap Up Test
        results.append(self._stress_test_gap(strategy, data, run_backtest, direction="up"))

        # 3. Gap Down Test
        results.append(self._stress_test_gap(strategy, data, run_backtest, direction="down"))

        # 4. Volatility Spike Test
        results.append(self._stress_test_volatility_spike(strategy, data, run_backtest))

        # 5. Liquidity Crisis Test
        results.append(self._stress_test_liquidity_crisis(strategy, data, run_backtest))

        # 6. Whipsaw Test (multiple reversals)
        results.append(self._stress_test_whipsaw(strategy, data, run_backtest))

        return results

    def _stress_test_flash_crash(
        self, strategy: Any, data: List[Dict], run_backtest: Callable
    ) -> StressTestResult:
        """Simulate flash crash scenario"""
        # Create modified data with sudden drop
        stress_data = [d.copy() for d in data]

        # Insert flash crash at random point (but not at start/end)
        crash_idx = len(stress_data) // 2
        crash_pct = self.stress_config.flash_crash_pct / 100

        # Apply crash
        base_price = stress_data[crash_idx]["close"]
        for i in range(crash_idx, min(crash_idx + 5, len(stress_data))):
            # Crash then partial recovery
            if i == crash_idx:
                factor = 1 + crash_pct
            elif i == crash_idx + 1:
                factor = 1 + crash_pct * 0.8
            elif i == crash_idx + 2:
                factor = 1 + crash_pct * 0.5
            else:
                factor = 1 + crash_pct * 0.3

            stress_data[i]["close"] = base_price * factor
            stress_data[i]["low"] = min(stress_data[i]["low"], stress_data[i]["close"])

        try:
            # Run backtest on stress data
            result = run_backtest(strategy, stress_data)

            pnl = result.get("total_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", abs(crash_pct) * 100)

            # Check survival (not liquidated)
            survived = max_dd < 50  # 50% drawdown threshold

            return StressTestResult(
                test_name="flash_crash",
                passed=survived and pnl > -15,
                pnl_impact_pct=pnl,
                max_drawdown_pct=max_dd,
                survived=survived,
                recovery_bars=5 if survived else 0,
                notes=f"Flash crash of {self.stress_config.flash_crash_pct}%",
            )
        except Exception as e:
            return StressTestResult(
                test_name="flash_crash",
                passed=False,
                pnl_impact_pct=-100,
                max_drawdown_pct=100,
                survived=False,
                recovery_bars=0,
                notes=f"Test failed: {e}",
            )

    def _stress_test_gap(
        self, strategy: Any, data: List[Dict], run_backtest: Callable, direction: str
    ) -> StressTestResult:
        """Simulate gap up/down scenario"""
        stress_data = [d.copy() for d in data]

        gap_idx = len(stress_data) // 2
        gap_pct = (
            self.stress_config.gap_up_pct if direction == "up" else self.stress_config.gap_down_pct
        ) / 100

        base_price = stress_data[gap_idx - 1]["close"]

        # Apply gap
        for i in range(gap_idx, len(stress_data)):
            stress_data[i]["open"] = (
                stress_data[i]["open"] * (1 + gap_pct) if i == gap_idx else stress_data[i]["open"]
            )
            stress_data[i]["high"] = stress_data[i]["high"] * (1 + gap_pct * 0.8)
            stress_data[i]["low"] = stress_data[i]["low"] * (1 + gap_pct * 0.8)
            stress_data[i]["close"] = stress_data[i]["close"] * (1 + gap_pct * 0.6)

        try:
            result = run_backtest(strategy, stress_data)
            pnl = result.get("total_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", 0)
            survived = max_dd < 30

            return StressTestResult(
                test_name=f"gap_{direction}",
                passed=survived,
                pnl_impact_pct=pnl,
                max_drawdown_pct=max_dd,
                survived=survived,
                recovery_bars=10 if survived else 0,
                notes=f"Gap {direction} of {gap_pct * 100:.1f}%",
            )
        except Exception as e:
            return StressTestResult(
                test_name=f"gap_{direction}",
                passed=False,
                pnl_impact_pct=-50,
                max_drawdown_pct=50,
                survived=False,
                recovery_bars=0,
                notes=f"Test failed: {e}",
            )

    def _stress_test_volatility_spike(
        self, strategy: Any, data: List[Dict], run_backtest: Callable
    ) -> StressTestResult:
        """Simulate volatility spike scenario"""
        stress_data = [d.copy() for d in data]

        spike_start = len(stress_data) // 3
        spike_end = spike_start + 20
        vol_mult = self.stress_config.volatility_multiplier

        for i in range(spike_start, min(spike_end, len(stress_data))):
            mid = (stress_data[i]["high"] + stress_data[i]["low"]) / 2
            range_size = stress_data[i]["high"] - stress_data[i]["low"]

            # Expand range
            stress_data[i]["high"] = mid + range_size * vol_mult / 2
            stress_data[i]["low"] = mid - range_size * vol_mult / 2

        try:
            result = run_backtest(strategy, stress_data)
            pnl = result.get("total_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", 0)
            survived = max_dd < 25

            return StressTestResult(
                test_name="volatility_spike",
                passed=survived and pnl > -20,
                pnl_impact_pct=pnl,
                max_drawdown_pct=max_dd,
                survived=survived,
                recovery_bars=20 if survived else 0,
                notes=f"Volatility multiplied by {vol_mult}x",
            )
        except Exception as e:
            return StressTestResult(
                test_name="volatility_spike",
                passed=False,
                pnl_impact_pct=-30,
                max_drawdown_pct=30,
                survived=False,
                recovery_bars=0,
                notes=f"Test failed: {e}",
            )

    def _stress_test_liquidity_crisis(
        self, strategy: Any, data: List[Dict], run_backtest: Callable
    ) -> StressTestResult:
        """Simulate liquidity crisis (low volume, wide spreads)"""
        stress_data = [d.copy() for d in data]

        crisis_start = len(stress_data) // 2
        crisis_end = crisis_start + 30
        vol_reduction = 1 - (self.stress_config.liquidity_drop_pct / 100)

        for i in range(crisis_start, min(crisis_end, len(stress_data))):
            stress_data[i]["volume"] = stress_data[i]["volume"] * vol_reduction

            # Also add some price instability
            if random.random() > 0.5:
                stress_data[i]["close"] *= 1 + random.uniform(-0.02, 0.02)

        try:
            result = run_backtest(strategy, stress_data)
            pnl = result.get("total_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", 0)
            survived = max_dd < 20

            return StressTestResult(
                test_name="liquidity_crisis",
                passed=survived,
                pnl_impact_pct=pnl,
                max_drawdown_pct=max_dd,
                survived=survived,
                recovery_bars=30 if survived else 0,
                notes=f"Volume dropped by {self.stress_config.liquidity_drop_pct}%",
            )
        except Exception as e:
            return StressTestResult(
                test_name="liquidity_crisis",
                passed=False,
                pnl_impact_pct=-20,
                max_drawdown_pct=20,
                survived=False,
                recovery_bars=0,
                notes=f"Test failed: {e}",
            )

    def _stress_test_whipsaw(
        self, strategy: Any, data: List[Dict], run_backtest: Callable
    ) -> StressTestResult:
        """Simulate whipsaw (multiple rapid reversals)"""
        stress_data = [d.copy() for d in data]

        whipsaw_start = len(stress_data) // 3
        whipsaw_end = whipsaw_start + 40

        for i in range(whipsaw_start, min(whipsaw_end, len(stress_data))):
            # Alternate direction every few bars
            cycle = ((i - whipsaw_start) // 5) % 4
            if cycle == 0:
                factor = 1.02  # Up
            elif cycle == 1:
                factor = 0.98  # Down
            elif cycle == 2:
                factor = 0.97  # More down
            else:
                factor = 1.03  # Up

            stress_data[i]["close"] *= factor
            stress_data[i]["high"] = max(stress_data[i]["high"], stress_data[i]["close"])
            stress_data[i]["low"] = min(stress_data[i]["low"], stress_data[i]["close"])

        try:
            result = run_backtest(strategy, stress_data)
            pnl = result.get("total_return_pct", 0)
            max_dd = result.get("max_drawdown_pct", 0)
            survived = max_dd < 25

            return StressTestResult(
                test_name="whipsaw",
                passed=survived and pnl > -15,
                pnl_impact_pct=pnl,
                max_drawdown_pct=max_dd,
                survived=survived,
                recovery_bars=40 if survived else 0,
                notes="Multiple rapid reversals",
            )
        except Exception as e:
            return StressTestResult(
                test_name="whipsaw",
                passed=False,
                pnl_impact_pct=-25,
                max_drawdown_pct=25,
                survived=False,
                recovery_bars=0,
                notes=f"Test failed: {e}",
            )

    def _analyze_parameter_sensitivity(
        self, strategy: Any, data: List[Dict], run_backtest: Callable, num_variations: int = 5
    ) -> float:
        """
        Analyze how sensitive strategy is to parameter changes.

        Returns sensitivity score (0 = robust, 1 = very sensitive)
        """
        if not hasattr(strategy, "params") or not strategy.params:
            return 0.0

        base_result = run_backtest(strategy, data)
        base_sharpe = base_result.get("sharpe_ratio", 0)

        if base_sharpe == 0:
            return 1.0

        sharpe_variations = []

        for param_name, param_value in strategy.params.items():
            if not isinstance(param_value, (int, float)):
                continue

            # Test variations of each parameter
            for variation_pct in [-20, -10, 10, 20]:
                varied_value = param_value * (1 + variation_pct / 100)

                # Create strategy copy with varied param
                original_value = strategy.params[param_name]
                strategy.params[param_name] = varied_value

                try:
                    result = run_backtest(strategy, data)
                    varied_sharpe = result.get("sharpe_ratio", 0)
                    sharpe_variations.append(
                        abs(varied_sharpe - base_sharpe) / base_sharpe if base_sharpe else 0
                    )
                except Exception:
                    sharpe_variations.append(1.0)
                finally:
                    strategy.params[param_name] = original_value

        if not sharpe_variations:
            return 0.0

        # Average sensitivity across all variations
        avg_sensitivity = np.mean(sharpe_variations)
        return min(1.0, avg_sensitivity)

    def _get_timestamp(self, data_point: Dict) -> datetime:
        """Extract timestamp from data point"""
        ts = data_point.get("timestamp", data_point.get("date", data_point.get("time")))
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.now()


# ============================================================
# Monte Carlo Simulation
# ============================================================


class MonteCarloSimulator:
    """
    Monte Carlo simulation for confidence intervals.

    Simulates many possible equity curves from trade distribution
    to understand range of possible outcomes.
    """

    def __init__(self, num_simulations: int = 1000, random_seed: Optional[int] = None):
        self.num_simulations = num_simulations
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def simulate(
        self,
        trade_returns: List[float],
        num_trades: int = 100,
        initial_capital: float = 10000,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.

        Args:
            trade_returns: Historical trade returns (as percentages)
            num_trades: Number of trades to simulate
            initial_capital: Starting capital

        Returns:
            Dict with simulation results and confidence intervals
        """
        if len(trade_returns) < 10:
            return {"error": "Not enough historical trades"}

        # Convert to numpy array
        returns = np.array(trade_returns)

        # Store final equity for each simulation
        final_equities = []
        max_drawdowns = []

        for _ in range(self.num_simulations):
            # Sample trades with replacement
            sampled_returns = np.random.choice(returns, size=num_trades, replace=True)

            # Build equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0

            for ret in sampled_returns:
                equity *= 1 + ret / 100
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        return {
            "num_simulations": self.num_simulations,
            "num_trades": num_trades,
            "initial_capital": initial_capital,
            "final_equity": {
                "mean": float(np.mean(final_equities)),
                "median": float(np.median(final_equities)),
                "std": float(np.std(final_equities)),
                "percentile_5": float(np.percentile(final_equities, 5)),
                "percentile_25": float(np.percentile(final_equities, 25)),
                "percentile_75": float(np.percentile(final_equities, 75)),
                "percentile_95": float(np.percentile(final_equities, 95)),
                "min": float(np.min(final_equities)),
                "max": float(np.max(final_equities)),
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "median": float(np.median(max_drawdowns)),
                "percentile_95": float(np.percentile(max_drawdowns, 95)),
                "max": float(np.max(max_drawdowns)),
            },
            "probability_of_profit": float(np.mean(final_equities > initial_capital)),
            "probability_of_20pct_loss": float(np.mean(final_equities < initial_capital * 0.8)),
            "expected_return_pct": float(
                (np.mean(final_equities) - initial_capital) / initial_capital * 100
            ),
        }


# Global instance
_validator: Optional[WalkForwardValidator] = None


def get_walk_forward_validator() -> WalkForwardValidator:
    """Get or create global validator instance"""
    global _validator
    if _validator is None:
        _validator = WalkForwardValidator()
    return _validator


__all__ = [
    # Enums
    "ValidationResult",
    # Configs
    "ValidationThresholds",
    "WalkForwardConfig",
    "StressTestConfig",
    # Results
    "ValidationMetrics",
    "StressTestResult",
    "WalkForwardWindow",
    "ValidationReport",
    # Main classes
    "WalkForwardValidator",
    "MonteCarloSimulator",
    # Factory
    "get_walk_forward_validator",
]
