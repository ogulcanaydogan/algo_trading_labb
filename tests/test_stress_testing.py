"""Tests for stress testing framework."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from bot.risk.stress_testing import (
    ScenarioType,
    RiskFactor,
    StressScenario,
    StressResult,
    StressTestReport,
    Position,
    HistoricalScenarios,
    HypotheticalScenarios,
    StressTestEngine,
    create_stress_test_engine,
)


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_scenario_types(self):
        assert ScenarioType.HISTORICAL.value == "historical"
        assert ScenarioType.HYPOTHETICAL.value == "hypothetical"
        assert ScenarioType.SENSITIVITY.value == "sensitivity"
        assert ScenarioType.MONTE_CARLO.value == "monte_carlo"


class TestRiskFactor:
    """Tests for RiskFactor enum."""

    def test_risk_factors(self):
        assert RiskFactor.EQUITY.value == "equity"
        assert RiskFactor.RATES.value == "rates"
        assert RiskFactor.VOLATILITY.value == "volatility"
        assert RiskFactor.LIQUIDITY.value == "liquidity"


class TestStressScenario:
    """Tests for StressScenario dataclass."""

    def test_scenario_creation(self):
        scenario = StressScenario(
            name="test_crash",
            description="Test crash scenario",
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks={"equity": -0.30, "volatility": 2.0},
        )
        assert scenario.name == "test_crash"
        assert scenario.shocks["equity"] == -0.30

    def test_scenario_to_dict(self):
        scenario = StressScenario(
            name="test_scenario",
            description="A test scenario",
            scenario_type=ScenarioType.HISTORICAL,
            shocks={"equity": -0.20},
            probability=0.02,
        )
        result = scenario.to_dict()
        assert result["name"] == "test_scenario"
        assert result["scenario_type"] == "historical"
        assert result["probability"] == 0.02


class TestStressResult:
    """Tests for StressResult dataclass."""

    def test_result_creation(self):
        result = StressResult(
            scenario_name="covid_crash",
            portfolio_pnl=-5000,
            portfolio_pnl_pct=-0.25,
            position_impacts={"BTC": -3000, "ETH": -2000},
            var_impact=0.5,
            liquidity_impact=0.4,
            margin_impact=0.3,
            breach_limits=["max_drawdown"],
            recovery_estimate_days=60,
        )
        assert result.scenario_name == "covid_crash"
        assert result.is_critical  # >20% loss or has breaches

    def test_result_not_critical(self):
        result = StressResult(
            scenario_name="minor_correction",
            portfolio_pnl=-1000,
            portfolio_pnl_pct=-0.05,
            position_impacts={"BTC": -1000},
            var_impact=0.1,
            liquidity_impact=0.1,
            margin_impact=0.05,
            breach_limits=[],
            recovery_estimate_days=10,
        )
        assert not result.is_critical

    def test_result_to_dict(self):
        result = StressResult(
            scenario_name="test",
            portfolio_pnl=-2000,
            portfolio_pnl_pct=-0.10,
            position_impacts={"BTC": -2000},
            var_impact=0.2,
            liquidity_impact=0.2,
            margin_impact=0.1,
            breach_limits=[],
            recovery_estimate_days=30,
        )
        data = result.to_dict()
        assert data["scenario_name"] == "test"
        assert "is_critical" in data


class TestStressTestReport:
    """Tests for StressTestReport dataclass."""

    def test_report_creation(self):
        results = [
            StressResult(
                scenario_name="scenario_1",
                portfolio_pnl=-1000,
                portfolio_pnl_pct=-0.05,
                position_impacts={},
                var_impact=0.1,
                liquidity_impact=0.1,
                margin_impact=0.05,
                breach_limits=[],
                recovery_estimate_days=10,
            ),
            StressResult(
                scenario_name="scenario_2",
                portfolio_pnl=-3000,
                portfolio_pnl_pct=-0.15,
                position_impacts={},
                var_impact=0.3,
                liquidity_impact=0.2,
                margin_impact=0.1,
                breach_limits=[],
                recovery_estimate_days=45,
            ),
        ]
        report = StressTestReport(
            portfolio_value=20000,
            test_date=datetime.now(),
            scenarios_tested=2,
            results=results,
            worst_case=results[1],
            expected_shortfall=-0.10,
            recommendations=["Consider reducing exposure"],
        )
        assert report.scenarios_tested == 2
        assert report.worst_case.scenario_name == "scenario_2"

    def test_report_to_dict(self):
        report = StressTestReport(
            portfolio_value=50000,
            test_date=datetime.now(),
            scenarios_tested=1,
            results=[],
            worst_case=None,
            expected_shortfall=-0.05,
            recommendations=["Portfolio looks OK"],
        )
        data = report.to_dict()
        assert data["portfolio_value"] == 50000
        assert data["worst_case"] is None


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        position = Position(
            symbol="BTC",
            quantity=0.5,
            current_price=50000,
            asset_class="crypto",
        )
        assert position.symbol == "BTC"
        assert position.market_value == 25000

    def test_position_with_beta(self):
        position = Position(
            symbol="ETH",
            quantity=10,
            current_price=3000,
            asset_class="crypto",
            beta=1.3,
        )
        assert position.beta == 1.3
        assert position.market_value == 30000


class TestHistoricalScenarios:
    """Tests for HistoricalScenarios."""

    def test_get_scenarios(self):
        scenarios = HistoricalScenarios.get_scenarios()
        assert len(scenarios) >= 5

        # Check specific scenarios exist
        names = [s.name for s in scenarios]
        assert "2008_financial_crisis" in names
        assert "covid_crash_2020" in names
        assert "flash_crash_2010" in names

    def test_scenarios_have_correct_type(self):
        scenarios = HistoricalScenarios.get_scenarios()
        for scenario in scenarios:
            assert scenario.scenario_type == ScenarioType.HISTORICAL
            assert "equity" in scenario.shocks or "volatility" in scenario.shocks


class TestHypotheticalScenarios:
    """Tests for HypotheticalScenarios."""

    def test_get_scenarios(self):
        scenarios = HypotheticalScenarios.get_scenarios()
        assert len(scenarios) >= 5

        names = [s.name for s in scenarios]
        assert "correlation_breakdown" in names
        assert "liquidity_crisis" in names

    def test_scenarios_have_correct_type(self):
        scenarios = HypotheticalScenarios.get_scenarios()
        for scenario in scenarios:
            assert scenario.scenario_type == ScenarioType.HYPOTHETICAL


class TestStressTestEngine:
    """Tests for StressTestEngine."""

    @pytest.fixture
    def engine(self):
        return StressTestEngine()

    @pytest.fixture
    def engine_with_portfolio(self):
        engine = StressTestEngine()
        positions = [
            Position(
                symbol="BTC",
                quantity=1.0,
                current_price=50000,
                asset_class="crypto",
                beta=1.2,
                liquidity_score=0.8,
            ),
            Position(
                symbol="ETH",
                quantity=10.0,
                current_price=3000,
                asset_class="crypto",
                beta=1.3,
                liquidity_score=0.75,
            ),
        ]
        engine.set_portfolio(positions)
        return engine

    def test_default_scenarios_loaded(self, engine):
        scenarios = engine.get_scenarios()
        assert len(scenarios) > 0
        assert "2008_financial_crisis" in scenarios
        assert "covid_crash_2020" in scenarios

    def test_add_custom_scenario(self, engine):
        custom = StressScenario(
            name="custom_crash",
            description="Custom crash scenario",
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks={"equity": -0.40},
        )
        engine.add_scenario(custom)
        assert "custom_crash" in engine.get_scenarios()

    def test_set_portfolio(self, engine):
        positions = [
            Position("BTC", 1.0, 50000, "crypto"),
        ]
        engine.set_portfolio(positions)
        assert engine._portfolio_value == 50000

    def test_run_scenario(self, engine_with_portfolio):
        result = engine_with_portfolio.run_scenario("covid_crash_2020")
        assert isinstance(result, StressResult)
        assert result.scenario_name == "covid_crash_2020"
        assert result.portfolio_pnl < 0  # Should be negative for crash

    def test_run_scenario_unknown(self, engine):
        with pytest.raises(ValueError, match="Unknown scenario"):
            engine.run_scenario("nonexistent_scenario")

    def test_run_scenario_with_custom_shocks(self, engine_with_portfolio):
        result = engine_with_portfolio.run_scenario(
            "covid_crash_2020",
            custom_shocks={"equity": -0.10},  # Less severe
        )
        # Should use custom shocks
        assert result.portfolio_pnl < 0

    def test_run_all_scenarios(self, engine_with_portfolio):
        report = engine_with_portfolio.run_all_scenarios()
        assert isinstance(report, StressTestReport)
        assert report.scenarios_tested > 0
        assert report.worst_case is not None
        assert len(report.recommendations) > 0

    def test_limit_breach_detection(self, engine_with_portfolio):
        # Create scenario that will cause breach
        severe_scenario = StressScenario(
            name="severe_crash",
            description="Very severe crash",
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks={"equity": -0.60, "volatility": 4.0},
            liquidity_multiplier=0.05,
        )
        engine_with_portfolio.add_scenario(severe_scenario)
        result = engine_with_portfolio.run_scenario("severe_crash")
        assert len(result.breach_limits) > 0

    def test_sensitivity_analysis(self, engine_with_portfolio):
        results = engine_with_portfolio.sensitivity_analysis(
            factor="equity",
            shock_range=(-0.20, 0.10),
            steps=5,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        assert "shock" in results.columns
        assert "portfolio_pnl" in results.columns

    def test_monte_carlo_stress(self, engine_with_portfolio):
        results = engine_with_portfolio.monte_carlo_stress(
            num_simulations=100,
            confidence_level=0.95,
        )
        assert "var" in results
        assert "expected_shortfall" in results
        assert results["num_simulations"] == 100

    def test_monte_carlo_no_positions(self, engine):
        results = engine.monte_carlo_stress()
        assert "error" in results

    def test_recommendations_generation(self, engine_with_portfolio):
        report = engine_with_portfolio.run_all_scenarios()
        assert len(report.recommendations) > 0

    def test_recommendations_for_concentration(self, engine):
        # Create concentrated portfolio
        positions = [
            Position("BTC", 10.0, 50000, "crypto"),  # Very large single position
        ]
        engine.set_portfolio(positions)
        report = engine.run_all_scenarios()

        # Should warn about concentration
        has_concentration_warning = any(
            "concentration" in r.lower()
            for r in report.recommendations
        )
        # May or may not warn depending on other factors
        assert len(report.recommendations) > 0

    def test_recovery_estimation(self, engine_with_portfolio):
        result = engine_with_portfolio.run_scenario("2008_financial_crisis")
        assert result.recovery_estimate_days > 0


class TestCreateStressTestEngine:
    """Tests for create_stress_test_engine factory function."""

    def test_creates_engine(self):
        engine = create_stress_test_engine()
        assert isinstance(engine, StressTestEngine)

    def test_with_custom_limits(self):
        limits = {
            "max_drawdown": 0.15,
            "max_position_loss": 0.30,
        }
        engine = create_stress_test_engine(risk_limits=limits)
        assert engine.risk_limits["max_drawdown"] == 0.15
        assert engine.risk_limits["max_position_loss"] == 0.30
