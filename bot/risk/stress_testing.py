"""
Stress Testing Framework for Portfolio Risk Analysis.

Provides comprehensive stress testing scenarios including:
- Historical scenarios (2008 crisis, COVID crash, etc.)
- Hypothetical scenarios (correlation breakdown, liquidity crisis)
- Sensitivity analysis
- Monte Carlo simulations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress scenarios."""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    SENSITIVITY = "sensitivity"
    MONTE_CARLO = "monte_carlo"


class RiskFactor(Enum):
    """Risk factors for stress testing."""

    EQUITY = "equity"
    RATES = "rates"
    CREDIT = "credit"
    FX = "fx"
    COMMODITY = "commodity"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    description: str
    scenario_type: ScenarioType
    shocks: Dict[str, float]  # Factor -> shock magnitude (%)
    correlation_override: Optional[np.ndarray] = None
    liquidity_multiplier: float = 1.0
    duration_days: int = 1
    probability: float = 0.01  # Estimated probability

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "shocks": self.shocks,
            "liquidity_multiplier": self.liquidity_multiplier,
            "duration_days": self.duration_days,
            "probability": self.probability,
        }


@dataclass
class StressResult:
    """Result of a stress test."""

    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    position_impacts: Dict[str, float]
    var_impact: float
    liquidity_impact: float
    margin_impact: float
    breach_limits: List[str]
    recovery_estimate_days: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_critical(self) -> bool:
        """Check if result breaches critical thresholds."""
        return self.portfolio_pnl_pct < -0.20 or len(self.breach_limits) > 0

    def to_dict(self) -> Dict:
        return {
            "scenario_name": self.scenario_name,
            "portfolio_pnl": round(self.portfolio_pnl, 2),
            "portfolio_pnl_pct": round(self.portfolio_pnl_pct, 4),
            "position_impacts": {k: round(v, 2) for k, v in self.position_impacts.items()},
            "var_impact": round(self.var_impact, 4),
            "liquidity_impact": round(self.liquidity_impact, 4),
            "margin_impact": round(self.margin_impact, 4),
            "breach_limits": self.breach_limits,
            "recovery_estimate_days": self.recovery_estimate_days,
            "is_critical": self.is_critical,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StressTestReport:
    """Complete stress test report."""

    portfolio_value: float
    test_date: datetime
    scenarios_tested: int
    results: List[StressResult]
    worst_case: Optional[StressResult]
    expected_shortfall: float
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "portfolio_value": round(self.portfolio_value, 2),
            "test_date": self.test_date.isoformat(),
            "scenarios_tested": self.scenarios_tested,
            "results": [r.to_dict() for r in self.results],
            "worst_case": self.worst_case.to_dict() if self.worst_case else None,
            "expected_shortfall": round(self.expected_shortfall, 4),
            "recommendations": self.recommendations,
        }


@dataclass
class Position:
    """Portfolio position for stress testing."""

    symbol: str
    quantity: float
    current_price: float
    asset_class: str
    beta: float = 1.0
    correlation_to_market: float = 0.8
    liquidity_score: float = 1.0  # 0-1, higher = more liquid

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


class HistoricalScenarios:
    """Pre-defined historical stress scenarios."""

    @staticmethod
    def get_scenarios() -> List[StressScenario]:
        """Get all historical scenarios."""
        return [
            # 2008 Financial Crisis
            StressScenario(
                name="2008_financial_crisis",
                description="2008 Global Financial Crisis peak impact",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.50,
                    "credit": -0.30,
                    "volatility": 2.5,
                    "liquidity": -0.60,
                },
                liquidity_multiplier=0.3,
                duration_days=90,
                probability=0.01,
            ),
            # COVID-19 Crash (March 2020)
            StressScenario(
                name="covid_crash_2020",
                description="COVID-19 market crash March 2020",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.35,
                    "commodity": -0.40,
                    "volatility": 3.0,
                    "correlation": 0.3,  # Correlation spike
                },
                liquidity_multiplier=0.5,
                duration_days=30,
                probability=0.02,
            ),
            # Flash Crash (2010)
            StressScenario(
                name="flash_crash_2010",
                description="2010 Flash Crash",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.10,
                    "volatility": 2.0,
                    "liquidity": -0.80,
                },
                liquidity_multiplier=0.1,
                duration_days=1,
                probability=0.05,
            ),
            # Dot-com Bubble (2000-2002)
            StressScenario(
                name="dotcom_crash",
                description="Dot-com bubble burst",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.45,
                    "rates": -0.02,
                },
                duration_days=365,
                probability=0.01,
            ),
            # 2022 Crypto Winter
            StressScenario(
                name="crypto_winter_2022",
                description="2022 Crypto market crash",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.70,  # Crypto
                    "volatility": 2.0,
                    "liquidity": -0.50,
                },
                liquidity_multiplier=0.4,
                duration_days=180,
                probability=0.03,
            ),
            # Black Monday (1987)
            StressScenario(
                name="black_monday_1987",
                description="Black Monday 1987",
                scenario_type=ScenarioType.HISTORICAL,
                shocks={
                    "equity": -0.22,
                    "volatility": 4.0,
                },
                duration_days=1,
                probability=0.005,
            ),
        ]


class HypotheticalScenarios:
    """Hypothetical stress scenarios."""

    @staticmethod
    def get_scenarios() -> List[StressScenario]:
        """Get hypothetical scenarios."""
        return [
            # Correlation Breakdown
            StressScenario(
                name="correlation_breakdown",
                description="All correlations spike to 0.9",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "equity": -0.20,
                    "correlation": 0.9,
                },
                duration_days=5,
                probability=0.02,
            ),
            # Liquidity Crisis
            StressScenario(
                name="liquidity_crisis",
                description="Severe liquidity dry-up",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "liquidity": -0.90,
                    "volatility": 2.0,
                },
                liquidity_multiplier=0.1,
                duration_days=7,
                probability=0.02,
            ),
            # Stagflation
            StressScenario(
                name="stagflation",
                description="High inflation + economic stagnation",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "equity": -0.25,
                    "rates": 0.03,
                    "commodity": 0.30,
                },
                duration_days=180,
                probability=0.03,
            ),
            # Geopolitical Crisis
            StressScenario(
                name="geopolitical_crisis",
                description="Major geopolitical event",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "equity": -0.15,
                    "commodity": 0.25,
                    "fx": 0.10,
                    "volatility": 1.5,
                },
                duration_days=30,
                probability=0.05,
            ),
            # Exchange Failure
            StressScenario(
                name="exchange_failure",
                description="Major exchange goes offline",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "liquidity": -0.95,
                    "volatility": 3.0,
                },
                liquidity_multiplier=0.05,
                duration_days=3,
                probability=0.01,
            ),
            # Rate Shock
            StressScenario(
                name="rate_shock",
                description="Sudden interest rate spike",
                scenario_type=ScenarioType.HYPOTHETICAL,
                shocks={
                    "rates": 0.02,
                    "equity": -0.15,
                    "credit": -0.10,
                },
                duration_days=30,
                probability=0.04,
            ),
        ]


class StressTestEngine:
    """
    Stress Testing Engine for Portfolio Analysis.

    Capabilities:
    - Historical scenario replay
    - Hypothetical scenario analysis
    - Sensitivity analysis
    - Monte Carlo simulation
    - Liquidity stress testing
    """

    def __init__(self, risk_limits: Optional[Dict[str, float]] = None):
        self._scenarios: Dict[str, StressScenario] = {}
        self._positions: List[Position] = []
        self._portfolio_value: float = 0.0
        self._correlation_matrix: Optional[np.ndarray] = None

        # Default risk limits
        self.risk_limits = risk_limits or {
            "max_drawdown": 0.25,
            "max_position_loss": 0.50,
            "min_liquidity_ratio": 0.10,
            "max_var": 0.10,
        }

        # Load default scenarios
        self._load_default_scenarios()

    def _load_default_scenarios(self):
        """Load default historical and hypothetical scenarios."""
        for scenario in HistoricalScenarios.get_scenarios():
            self._scenarios[scenario.name] = scenario

        for scenario in HypotheticalScenarios.get_scenarios():
            self._scenarios[scenario.name] = scenario

    def add_scenario(self, scenario: StressScenario):
        """Add a custom scenario."""
        self._scenarios[scenario.name] = scenario

    def set_portfolio(
        self, positions: List[Position], correlation_matrix: Optional[np.ndarray] = None
    ):
        """Set portfolio for stress testing."""
        self._positions = positions
        self._portfolio_value = sum(p.market_value for p in positions)
        self._correlation_matrix = correlation_matrix

    def run_scenario(
        self, scenario_name: str, custom_shocks: Optional[Dict[str, float]] = None
    ) -> StressResult:
        """
        Run a single stress scenario.

        Args:
            scenario_name: Name of scenario to run
            custom_shocks: Override default shocks

        Returns:
            StressResult with impact analysis
        """
        scenario = self._scenarios.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        shocks = custom_shocks or scenario.shocks

        # Calculate position impacts
        position_impacts = {}
        total_pnl = 0.0

        for position in self._positions:
            impact = self._calculate_position_impact(position, shocks, scenario)
            position_impacts[position.symbol] = impact
            total_pnl += impact

        portfolio_pnl_pct = total_pnl / self._portfolio_value if self._portfolio_value > 0 else 0

        # Check limit breaches
        breach_limits = self._check_limit_breaches(portfolio_pnl_pct, position_impacts, scenario)

        # Calculate additional metrics
        var_impact = self._calculate_var_impact(shocks)
        liquidity_impact = 1 - scenario.liquidity_multiplier
        margin_impact = self._calculate_margin_impact(shocks)
        recovery_days = self._estimate_recovery(portfolio_pnl_pct, scenario)

        return StressResult(
            scenario_name=scenario_name,
            portfolio_pnl=total_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            position_impacts=position_impacts,
            var_impact=var_impact,
            liquidity_impact=liquidity_impact,
            margin_impact=margin_impact,
            breach_limits=breach_limits,
            recovery_estimate_days=recovery_days,
        )

    def _calculate_position_impact(
        self, position: Position, shocks: Dict[str, float], scenario: StressScenario
    ) -> float:
        """Calculate impact on a single position."""
        base_impact = 0.0

        # Equity shock
        if "equity" in shocks:
            equity_shock = shocks["equity"]
            # Adjust by beta
            position_shock = equity_shock * position.beta
            base_impact += position.market_value * position_shock

        # Volatility impact (larger positions suffer more in high vol)
        if "volatility" in shocks:
            vol_multiplier = shocks["volatility"]
            vol_impact = position.market_value * 0.01 * (vol_multiplier - 1)
            base_impact -= abs(vol_impact)

        # Liquidity impact
        if "liquidity" in shocks or scenario.liquidity_multiplier < 1:
            # Less liquid positions suffer more
            liq_penalty = (1 - position.liquidity_score) * 0.1
            base_impact -= position.market_value * liq_penalty

        # Correlation impact
        if "correlation" in shocks:
            # Higher correlation = less diversification benefit
            corr_impact = shocks["correlation"] * 0.05
            base_impact -= position.market_value * corr_impact

        return base_impact

    def _check_limit_breaches(
        self, portfolio_pnl_pct: float, position_impacts: Dict[str, float], scenario: StressScenario
    ) -> List[str]:
        """Check which risk limits are breached."""
        breaches = []

        if abs(portfolio_pnl_pct) > self.risk_limits["max_drawdown"]:
            breaches.append(
                f"max_drawdown: {portfolio_pnl_pct:.1%} > {self.risk_limits['max_drawdown']:.1%}"
            )

        for symbol, impact in position_impacts.items():
            position = next((p for p in self._positions if p.symbol == symbol), None)
            if position:
                position_pct = impact / position.market_value if position.market_value > 0 else 0
                if abs(position_pct) > self.risk_limits["max_position_loss"]:
                    breaches.append(f"position_loss_{symbol}: {position_pct:.1%}")

        if scenario.liquidity_multiplier < self.risk_limits["min_liquidity_ratio"]:
            breaches.append(
                f"liquidity: {scenario.liquidity_multiplier:.1%} < {self.risk_limits['min_liquidity_ratio']:.1%}"
            )

        return breaches

    def _calculate_var_impact(self, shocks: Dict[str, float]) -> float:
        """Calculate impact on VaR."""
        vol_multiplier = shocks.get("volatility", 1.0)
        corr_impact = shocks.get("correlation", 0)

        # Higher vol and correlation = higher VaR
        var_increase = (vol_multiplier - 1) * 0.5 + corr_impact * 0.3
        return max(0, var_increase)

    def _calculate_margin_impact(self, shocks: Dict[str, float]) -> float:
        """Calculate impact on margin requirements."""
        vol_multiplier = shocks.get("volatility", 1.0)
        # Higher volatility typically increases margin requirements
        return (vol_multiplier - 1) * 0.2

    def _estimate_recovery(self, pnl_pct: float, scenario: StressScenario) -> int:
        """Estimate days to recover from drawdown."""
        if pnl_pct >= 0:
            return 0

        # Rough estimate: need to recover |pnl_pct| with assumed daily return
        assumed_daily_return = 0.001  # 0.1% per day
        loss_to_recover = abs(pnl_pct)
        days = int(loss_to_recover / assumed_daily_return)

        # Adjust for scenario duration
        return max(days, scenario.duration_days)

    def run_all_scenarios(self) -> StressTestReport:
        """Run all registered scenarios."""
        results = []
        for name in self._scenarios:
            try:
                result = self.run_scenario(name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run scenario {name}: {e}")

        # Find worst case
        worst_case = min(results, key=lambda r: r.portfolio_pnl_pct) if results else None

        # Calculate expected shortfall
        sorted_pnls = sorted([r.portfolio_pnl_pct for r in results])
        tail_pnls = sorted_pnls[: max(1, len(sorted_pnls) // 10)]
        expected_shortfall = np.mean(tail_pnls) if tail_pnls else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(results, worst_case)

        return StressTestReport(
            portfolio_value=self._portfolio_value,
            test_date=datetime.now(),
            scenarios_tested=len(results),
            results=results,
            worst_case=worst_case,
            expected_shortfall=expected_shortfall,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, results: List[StressResult], worst_case: Optional[StressResult]
    ) -> List[str]:
        """Generate risk recommendations based on results."""
        recommendations = []

        critical_scenarios = [r for r in results if r.is_critical]
        if critical_scenarios:
            recommendations.append(
                f"CRITICAL: {len(critical_scenarios)} scenarios breach risk limits"
            )

        if worst_case and worst_case.portfolio_pnl_pct < -0.30:
            recommendations.append(
                f"Consider reducing overall exposure - worst case loss: {worst_case.portfolio_pnl_pct:.1%}"
            )

        # Check concentration
        if self._positions:
            max_position = max(self._positions, key=lambda p: p.market_value)
            concentration = (
                max_position.market_value / self._portfolio_value
                if self._portfolio_value > 0
                else 0
            )
            if concentration > 0.3:
                recommendations.append(
                    f"High concentration in {max_position.symbol} ({concentration:.1%})"
                )

        # Check liquidity
        avg_liquidity = (
            np.mean([p.liquidity_score for p in self._positions]) if self._positions else 1
        )
        if avg_liquidity < 0.5:
            recommendations.append(
                "Portfolio has low average liquidity - may face execution challenges in stress"
            )

        if not recommendations:
            recommendations.append("Portfolio stress metrics within acceptable limits")

        return recommendations

    def sensitivity_analysis(
        self, factor: str, shock_range: Tuple[float, float] = (-0.30, 0.30), steps: int = 10
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis for a single factor.

        Args:
            factor: Risk factor to stress
            shock_range: (min_shock, max_shock)
            steps: Number of steps

        Returns:
            DataFrame with shock levels and P&L impacts
        """
        shocks = np.linspace(shock_range[0], shock_range[1], steps)
        results = []

        for shock in shocks:
            scenario = StressScenario(
                name=f"sensitivity_{factor}_{shock:.2f}",
                description=f"Sensitivity test: {factor} = {shock:.1%}",
                scenario_type=ScenarioType.SENSITIVITY,
                shocks={factor: shock},
            )
            self._scenarios[scenario.name] = scenario

            result = self.run_scenario(scenario.name)
            results.append(
                {
                    "shock": shock,
                    "portfolio_pnl": result.portfolio_pnl,
                    "portfolio_pnl_pct": result.portfolio_pnl_pct,
                }
            )

        return pd.DataFrame(results)

    def monte_carlo_stress(
        self,
        num_simulations: int = 1000,
        confidence_level: float = 0.99,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo stress simulation.

        Args:
            num_simulations: Number of simulations
            confidence_level: Confidence level for VaR
            random_seed: Optional seed for reproducibility (None for true randomness)

        Returns:
            Monte Carlo results including VaR and ES
        """
        if not self._positions:
            return {"error": "No positions set"}

        # Set seed only if provided (for reproducibility in testing)
        if random_seed is not None:
            np.random.seed(random_seed)
        portfolio_returns = []

        for _ in range(num_simulations):
            # Random shocks to factors
            shocks = {
                "equity": np.random.normal(0, 0.15),
                "volatility": 1 + np.random.exponential(0.3),
                "liquidity": np.random.uniform(-0.3, 0),
            }

            total_pnl = 0
            for position in self._positions:
                equity_impact = position.market_value * shocks["equity"] * position.beta
                vol_impact = position.market_value * 0.01 * (shocks["volatility"] - 1)
                total_pnl += equity_impact - abs(vol_impact)

            portfolio_returns.append(
                total_pnl / self._portfolio_value if self._portfolio_value > 0 else 0
            )

        portfolio_returns = np.array(portfolio_returns)

        # Calculate risk metrics
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(portfolio_returns, var_percentile)
        es = np.mean(portfolio_returns[portfolio_returns <= var])

        return {
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "mean_return": float(np.mean(portfolio_returns)),
            "std_return": float(np.std(portfolio_returns)),
            "var": float(var),
            "expected_shortfall": float(es),
            "max_loss": float(np.min(portfolio_returns)),
            "max_gain": float(np.max(portfolio_returns)),
            "percentiles": {
                "1%": float(np.percentile(portfolio_returns, 1)),
                "5%": float(np.percentile(portfolio_returns, 5)),
                "10%": float(np.percentile(portfolio_returns, 10)),
                "90%": float(np.percentile(portfolio_returns, 90)),
                "95%": float(np.percentile(portfolio_returns, 95)),
                "99%": float(np.percentile(portfolio_returns, 99)),
            },
        }

    def get_scenarios(self) -> List[str]:
        """Get list of available scenarios."""
        return list(self._scenarios.keys())


def create_stress_test_engine(risk_limits: Optional[Dict[str, float]] = None) -> StressTestEngine:
    """Factory function to create stress test engine."""
    return StressTestEngine(risk_limits=risk_limits)
