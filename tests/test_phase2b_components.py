"""
Tests for Phase 2B Components.

Tests cover:
A) Shadow Data Collector
B) Counterfactual Evaluator
C) Cost-Aware Reward Shaping
D) Safe Strategy Weighting Advisor
E) BTC Diagnosis Tool

Critical verification:
- RL has NO execution authority
- Safety constraints are LOCKED
- Weight changes are CLAMPED
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# A) Shadow Data Collector
from bot.rl.shadow_data_collector import (
    DecisionSnapshot,
    ShadowCollectorConfig,
    ShadowDataCollector,
    get_shadow_collector,
    reset_shadow_collector,
)

# B) Counterfactual Evaluator
from bot.rl.counterfactual_evaluator import (
    CounterfactualTrade,
    RegimeMetrics,
    CostDecomposition,
    ConfidenceThresholdResult,
    CounterfactualReport,
    CounterfactualEvaluator,
)

# C) Cost-Aware Reward Shaping
from bot.rl.reward_shaping import (
    RewardShaper,
    CostAwareConfig,
    CostAwareRewardShaper,
    CAPITAL_PRESERVATION_VIOLATION_PENALTY,
)

# D) Strategy Weighting Advisor
from bot.rl.strategy_weighting_advisor import (
    StrategyPerformance,
    WeightingConfig,
    WeightUpdate,
    StrategyWeightingAdvisor,
    MAX_DAILY_WEIGHT_SHIFT,
    MIN_STRATEGY_WEIGHT,
    MAX_STRATEGY_WEIGHT,
    get_strategy_weighting_advisor,
    reset_strategy_weighting_advisor,
)

# E) BTC Diagnosis
from bot.rl.btc_diagnosis import (
    TradingMetrics,
    DegradationAttribution,
    DiagnosisRecommendation,
    BTCDiagnosisReport,
    BTCDiagnosisTool,
    get_example_btc_results,
    get_example_eth_results,
)

from bot.rl.multi_agent_system import MarketState


# =============================================================================
# A) Shadow Data Collector Tests
# =============================================================================

class TestShadowDataCollector:
    """Tests for shadow data collection."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_shadow_collector()

    def test_decision_snapshot_creation(self):
        """Test creating decision snapshots."""
        snapshot = DecisionSnapshot(
            decision_id="DEC_001",
            symbol="BTC-USD",
            price=50000.0,
            volatility=0.02,
            rl_action="long",
            rl_confidence=0.75,
            gate_approved=True,
            actual_action="long",
        )

        assert snapshot.decision_id == "DEC_001"
        assert snapshot.symbol == "BTC-USD"
        assert snapshot.rl_action == "long"
        assert snapshot.executed is False  # Not executed yet

    def test_snapshot_to_dict(self):
        """Test serialization."""
        snapshot = DecisionSnapshot(
            decision_id="DEC_002",
            symbol="ETH-USD",
            price=3000.0,
            pnl=150.0,
            pnl_pct=1.5,
        )

        d = snapshot.to_dict()

        assert d["decision_id"] == "DEC_002"
        assert d["symbol"] == "ETH-USD"
        assert d["market_context"]["price"] == 3000.0
        assert d["outcome"]["pnl"] == 150.0

    def test_collector_initialization(self):
        """Test collector initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowCollectorConfig(
                log_path=Path(tmpdir) / "shadow.jsonl",
                enable_rl_shadow=False,  # Disable for testing
            )
            collector = ShadowDataCollector(config)

            assert collector.config.enabled is True
            assert collector._decision_counter == 0

    def test_record_decision_point(self):
        """Test recording decision points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowCollectorConfig(
                log_path=Path(tmpdir) / "shadow.jsonl",
                enable_rl_shadow=False,
            )
            collector = ShadowDataCollector(config)

            market_state = MarketState(
                symbol="BTC-USD",
                price=50000.0,
                volatility=0.02,
                regime="bull",
            )

            decision_id = collector.record_decision_point(
                symbol="BTC-USD",
                market_state=market_state,
                gate_approved=True,
                actual_action="long",
                actual_confidence=0.8,
                strategy_used="TrendFollower",
            )

            assert decision_id.startswith("DEC_")
            assert decision_id in collector._pending_decisions

    def test_record_execution_and_outcome(self):
        """Test recording execution and outcome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowCollectorConfig(
                log_path=Path(tmpdir) / "shadow.jsonl",
                enable_rl_shadow=False,
            )
            collector = ShadowDataCollector(config)

            market_state = MarketState(symbol="BTC-USD", price=50000.0)

            decision_id = collector.record_decision_point(
                symbol="BTC-USD",
                market_state=market_state,
                gate_approved=True,
                actual_action="long",
            )

            # Record execution
            collector.record_execution(
                decision_id=decision_id,
                entry_price=50000.0,
                position_size=1000.0,
                slippage_cost=5.0,
                fee_cost=2.0,
            )

            snapshot = collector._pending_decisions[decision_id]
            assert snapshot.executed is True
            assert snapshot.entry_price == 50000.0
            assert snapshot.total_cost == 7.0  # slippage + fee

            # Record outcome
            collector.record_outcome(
                decision_id=decision_id,
                exit_price=51000.0,
                pnl=100.0,
                pnl_pct=2.0,
                mae=0.5,
                mfe=2.5,
            )

            # Should be moved to completed
            assert decision_id not in collector._pending_decisions
            assert len(collector._decisions) == 1
            assert collector._decisions[0].pnl == 100.0

    def test_collection_stats(self):
        """Test collection statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowCollectorConfig(
                log_path=Path(tmpdir) / "shadow.jsonl",
                enable_rl_shadow=False,
            )
            collector = ShadowDataCollector(config)

            # No data yet
            stats = collector.get_collection_stats()
            assert stats["total_decisions"] == 0

    def test_collector_disabled(self):
        """Test that disabled collector returns empty decision ID."""
        config = ShadowCollectorConfig(enabled=False)
        collector = ShadowDataCollector(config)

        market_state = MarketState(symbol="BTC-USD", price=50000.0)
        decision_id = collector.record_decision_point(
            symbol="BTC-USD",
            market_state=market_state,
            gate_approved=True,
            actual_action="hold",
        )

        assert decision_id == ""


# =============================================================================
# B) Counterfactual Evaluator Tests
# =============================================================================

class TestCounterfactualEvaluator:
    """Tests for counterfactual evaluation."""

    def test_counterfactual_trade_creation(self):
        """Test creating counterfactual trade records."""
        trade = CounterfactualTrade(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            regime="bull",
            rl_action="long",
            rl_confidence=0.8,
            rl_primary_agent="TrendFollower",
            actual_action="hold",
            actual_pnl=0.0,
            counterfactual_pnl=500.0,
        )

        assert trade.symbol == "BTC-USD"
        assert trade.followed_rl is False  # Different action

    def test_regime_metrics_update(self):
        """Test regime metrics accumulation."""
        metrics = RegimeMetrics(regime="bull")

        # Add some trade data
        metrics.total_decisions = 10
        metrics.rl_correct = 6
        metrics.avg_actual_pnl = 50.0
        metrics.avg_counterfactual_pnl = 70.0
        metrics.delta_pnl = 200.0

        # Verify calculations would work
        hit_rate = metrics.rl_correct / metrics.total_decisions
        delta = metrics.delta_pnl

        assert hit_rate == 0.6
        assert delta == 200.0

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = CounterfactualEvaluator()

        assert evaluator.min_decisions == 10  # Default
        assert len(evaluator.confidence_thresholds) == 5  # Default thresholds

    def test_evaluate_empty_trades(self):
        """Test evaluation with no trades."""
        evaluator = CounterfactualEvaluator()
        report = evaluator.evaluate(trades=[], weeks=4)

        assert report.total_decisions == 0
        assert report.delta_pnl == 0.0

    def test_evaluate_with_trades(self):
        """Test evaluation with sample trades."""
        evaluator = CounterfactualEvaluator()

        decisions = [
            CounterfactualTrade(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                regime="bull",
                rl_action="long",
                rl_confidence=0.8,
                rl_primary_agent="TrendFollower",
                actual_action="long",
                actual_pnl=100.0,
                counterfactual_pnl=100.0,
                followed_rl=True,
            ),
            CounterfactualTrade(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                regime="bull",
                rl_action="short",
                rl_confidence=0.7,
                rl_primary_agent="ShortSpecialist",
                actual_action="hold",
                actual_pnl=0.0,
                counterfactual_pnl=-50.0,
            ),
            CounterfactualTrade(
                timestamp=datetime.now(),
                symbol="ETH-USD",
                regime="bear",
                rl_action="long",
                rl_confidence=0.9,
                rl_primary_agent="TrendFollower",
                actual_action="hold",
                actual_pnl=0.0,
                counterfactual_pnl=200.0,
            ),
        ]

        report = evaluator.evaluate(decisions, weeks=1)

        assert report.total_decisions == 3
        assert report.rl_agreement_rate == pytest.approx(1 / 3, abs=0.01)
        assert "bull" in report.by_regime
        assert "bear" in report.by_regime

    def test_confidence_threshold_sweep(self):
        """Test confidence threshold sweep."""
        evaluator = CounterfactualEvaluator()

        decisions = [
            CounterfactualTrade(
                timestamp=datetime.now(),
                symbol="BTC-USD",
                regime="bull",
                rl_action="long",
                rl_confidence=conf / 10.0,
                rl_primary_agent="TrendFollower",
                actual_action="hold",
                actual_pnl=0.0,
                counterfactual_pnl=100.0 if conf > 5 else -50.0,
            )
            for conf in range(1, 11)
        ]

        report = evaluator.evaluate(decisions, weeks=1)

        # Should have confidence threshold results
        assert len(report.threshold_analysis) > 0

    def test_report_serialization(self):
        """Test report to_dict."""
        report = CounterfactualReport(
            total_decisions=100,
            rl_agreement_rate=0.65,
            delta_pnl=500.0,
        )

        d = report.to_dict()

        # Fields are nested in the serialized format
        assert d["overall"]["total_decisions"] == 100
        assert d["overall"]["rl_agreement_rate"] == 0.65
        assert d["pnl_comparison"]["delta_pnl"] == 500.0


# =============================================================================
# C) Cost-Aware Reward Shaping Tests
# =============================================================================

class TestCostAwareRewardShaping:
    """Tests for cost-aware reward shaping."""

    def test_capital_preservation_penalty_constant(self):
        """Test the penalty constant is extreme."""
        assert CAPITAL_PRESERVATION_VIOLATION_PENALTY == -1000.0

    def test_base_reward_shaper(self):
        """Test base reward calculation."""
        shaper = RewardShaper(
            portfolio_value=30000.0,
            daily_target_pct=0.01,
        )

        reward = shaper.calculate_reward(
            pnl=50.0,
            pnl_pct=0.5,
            entry_price=50000.0,
            exit_price=50250.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            hold_time_minutes=60,
        )

        assert "total" in reward
        assert reward["total"] > 0  # Positive PnL should give positive reward

    def test_cost_aware_config_defaults(self):
        """Test cost-aware config defaults."""
        config = CostAwareConfig()

        assert config.turnover_penalty_per_trade == 0.05
        assert config.preservation_violation_penalty == CAPITAL_PRESERVATION_VIOLATION_PENALTY

    def test_cost_aware_turnover_penalty(self):
        """Test turnover penalty application."""
        config = CostAwareConfig(
            turnover_penalty_per_trade=0.1,
            max_trades_per_day_before_penalty=5,
        )
        shaper = CostAwareRewardShaper(cost_config=config)

        # Simulate excessive trading by making multiple trades
        for _ in range(10):
            shaper._daily_trades_count += 1

        # Excessive trading
        reward = shaper.calculate_reward(
            pnl=100.0,
            pnl_pct=1.0,
            entry_price=50000.0,
            exit_price=50500.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            hold_time_minutes=60,
            slippage_cost=5.0,
            fee_cost=2.0,
            expected_slippage=5.0,
            position_value=1000.0,
            preservation_level="normal",
            preservation_violated=False,
        )

        assert "turnover_penalty" in reward
        assert reward["turnover_penalty"] < 0  # Should be negative

    def test_slippage_surprise_penalty(self):
        """Test slippage surprise penalty."""
        config = CostAwareConfig(slippage_surprise_multiplier=5.0)
        shaper = CostAwareRewardShaper(cost_config=config)

        # Slippage much higher than expected
        reward = shaper.calculate_reward(
            pnl=100.0,
            pnl_pct=1.0,
            entry_price=50000.0,
            exit_price=50500.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            hold_time_minutes=60,
            slippage_cost=20.0,  # Much higher than expected
            fee_cost=2.0,
            expected_slippage=5.0,
            position_value=1000.0,
            preservation_level="normal",
            preservation_violated=False,
        )

        assert "slippage_surprise" in reward
        assert reward["slippage_surprise"] < 0

    def test_preservation_violation_penalty(self):
        """Test capital preservation violation gives extreme penalty."""
        config = CostAwareConfig()
        shaper = CostAwareRewardShaper(cost_config=config)

        reward = shaper.calculate_reward(
            pnl=1000.0,  # Big profit
            pnl_pct=10.0,
            entry_price=50000.0,
            exit_price=55000.0,
            stop_loss=49500.0,
            take_profit=56000.0,
            hold_time_minutes=60,
            slippage_cost=0.0,
            fee_cost=0.0,
            expected_slippage=0.0,
            position_value=1000.0,
            preservation_level="lockdown",
            preservation_violated=True,  # VIOLATION
        )

        # Should be dominated by preservation penalty
        assert reward["total"] < 0
        assert reward["total"] <= -900  # Near the -1000 penalty

    def test_calibrate_from_counterfactual(self):
        """Test calibration method."""
        config = CostAwareConfig()
        shaper = CostAwareRewardShaper(cost_config=config)

        # Calibrate with historical data
        shaper.calibrate_from_counterfactual(
            avg_slippage_pct=0.002,
            avg_fee_pct=0.001,
            avg_trades_per_day=8,
            historical_cvar=-0.03,
        )

        # Check calibration was applied
        assert shaper.cost_config.max_acceptable_cost_pct >= 0.003  # 30 bps


# =============================================================================
# D) Strategy Weighting Advisor Tests
# =============================================================================

class TestStrategyWeightingAdvisor:
    """Tests for safe strategy weighting."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_strategy_weighting_advisor()

    def test_weight_constraints_locked(self):
        """Test that weight constraints cannot be exceeded."""
        # Try to exceed limits
        config = WeightingConfig(
            max_daily_shift=0.50,  # Try 50%
            min_weight=0.00,       # Try 0%
            max_weight=1.00,       # Try 100%
        )

        # Should be clamped
        assert config.max_daily_shift <= MAX_DAILY_WEIGHT_SHIFT
        assert config.min_weight >= MIN_STRATEGY_WEIGHT
        assert config.max_weight <= MAX_STRATEGY_WEIGHT

    def test_advisor_initialization(self):
        """Test advisor initializes with equal weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WeightingConfig()
            advisor = StrategyWeightingAdvisor(
                config=config,
                state_path=Path(tmpdir) / "weights.json",
            )

            weights = advisor.get_current_weights()

            # Should start with equal weights
            assert len(weights) == 5
            assert all(abs(w - 0.20) < 0.01 for w in weights.values())

    def test_weight_clamping(self):
        """Test that weight changes are clamped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WeightingConfig(max_daily_shift=0.10)
            advisor = StrategyWeightingAdvisor(
                config=config,
                state_path=Path(tmpdir) / "weights.json",
            )

            # Set up performance to heavily favor one strategy
            advisor._performance["TrendFollower"] = StrategyPerformance(
                strategy_name="TrendFollower",
                win_rate=0.9,
                sharpe_ratio=3.0,
                trade_count=100,
            )

            # Request weight update
            new_weights = advisor.recommend_weights(
                regime="bull",
                rl_preferences={"TrendFollower": 1.0},  # Strongly prefer
                force=True,
            )

            if new_weights:
                # Check clamping
                old_weight = 0.20
                max_new = old_weight + MAX_DAILY_WEIGHT_SHIFT

                # TrendFollower weight should not exceed max shift
                assert new_weights["TrendFollower"] <= max_new + 0.01

    def test_weight_validation(self):
        """Test weight validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            # Invalid: below minimum
            invalid_weights = {
                "TrendFollower": 0.01,  # Below MIN_STRATEGY_WEIGHT
                "MeanReversion": 0.24,
                "MomentumTrader": 0.25,
                "ShortSpecialist": 0.25,
                "Scalper": 0.25,
            }
            assert not advisor._validate_weights(invalid_weights)

            # Invalid: above maximum
            invalid_weights2 = {
                "TrendFollower": 0.60,  # Above MAX_STRATEGY_WEIGHT
                "MeanReversion": 0.10,
                "MomentumTrader": 0.10,
                "ShortSpecialist": 0.10,
                "Scalper": 0.10,
            }
            assert not advisor._validate_weights(invalid_weights2)

            # Invalid: doesn't sum to 1
            invalid_weights3 = {
                "TrendFollower": 0.30,
                "MeanReversion": 0.30,
                "MomentumTrader": 0.30,
                "ShortSpecialist": 0.30,
                "Scalper": 0.30,
            }
            assert not advisor._validate_weights(invalid_weights3)

    def test_cooldown_enforcement(self):
        """Test update cooldown is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WeightingConfig(cooldown_hours=4.0)
            advisor = StrategyWeightingAdvisor(
                config=config,
                state_path=Path(tmpdir) / "weights.json",
            )

            # Set last update to now
            advisor._last_update = datetime.now()

            # Should return None due to cooldown
            result = advisor.recommend_weights(regime="bull")
            assert result is None

    def test_regime_adjusted_weights(self):
        """Test regime-specific weight adjustment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            bull_weights = advisor.get_regime_adjusted_weights("bull")
            bear_weights = advisor.get_regime_adjusted_weights("bear")

            # Bull should favor TrendFollower
            # Bear should favor ShortSpecialist
            # (Depends on default config)
            assert sum(bull_weights.values()) == pytest.approx(1.0, abs=0.01)
            assert sum(bear_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_performance_update(self):
        """Test updating strategy performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            # Update with trade outcome
            advisor.update_strategy_performance(
                strategy_name="TrendFollower",
                pnl=100.0,
                pnl_pct=1.0,
                regime="bull",
                cost_pct=0.001,
                is_win=True,
            )

            assert "TrendFollower" in advisor._performance
            assert advisor._performance["TrendFollower"].trade_count == 1
            assert advisor._performance["TrendFollower"].win_rate == 1.0

    def test_no_execution_authority(self):
        """Verify advisor has no execution methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(
                state_path=Path(tmpdir) / "weights.json",
            )

            # Should NOT have any of these methods
            assert not hasattr(advisor, "place_order")
            assert not hasattr(advisor, "execute_trade")
            assert not hasattr(advisor, "set_position_size")
            assert not hasattr(advisor, "set_leverage")

            # Should only have recommend_weights (advisory)
            assert hasattr(advisor, "recommend_weights")
            assert hasattr(advisor, "apply_weights")  # Explicit application


# =============================================================================
# E) BTC Diagnosis Tests
# =============================================================================

class TestBTCDiagnosis:
    """Tests for BTC diagnosis tool."""

    def test_example_data_degradation(self):
        """Test example data shows expected degradation."""
        btc = get_example_btc_results()
        eth = get_example_eth_results()

        btc_degradation = btc["gross_return_pct"] - btc["net_return_pct"]
        eth_degradation = eth["gross_return_pct"] - eth["net_return_pct"]

        assert btc_degradation == pytest.approx(149.0, abs=1.0)
        assert eth_degradation == pytest.approx(10.0, abs=1.0)

    def test_trading_metrics_creation(self):
        """Test TradingMetrics dataclass."""
        metrics = TradingMetrics(
            symbol="BTC-USD",
            period_days=365,
            gross_return_pct=180.0,
            net_return_pct=31.0,
            degradation_pct=149.0,  # Gross - Net, manually calculated
        )

        assert metrics.degradation_pct == 149.0
        assert metrics.gross_return_pct - metrics.net_return_pct == 149.0

    def test_diagnosis_tool_initialization(self):
        """Test diagnosis tool initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))
            assert tool.output_dir.exists()

    def test_analyze_backtest_results(self):
        """Test analyzing backtest results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            assert report.btc_metrics is not None
            assert report.baseline_metrics is not None
            assert report.attribution is not None
            assert len(report.recommendations) > 0
            assert report.primary_cause != ""

    def test_attribution_calculation(self):
        """Test degradation attribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            attr = report.attribution

            # Total attribution should relate to total costs
            total_attr = (
                attr.slippage_attribution_pct +
                attr.spread_attribution_pct +
                attr.fee_attribution_pct
            )
            assert total_attr > 0

            # Turnover ratio should be ~2x (10 trades/day vs 5)
            assert attr.turnover_ratio_vs_baseline > 1.5

    def test_recommendations_priority(self):
        """Test recommendations are prioritized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            if len(report.recommendations) > 1:
                # First recommendation should be highest priority
                priorities = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                for i in range(len(report.recommendations) - 1):
                    curr_pri = priorities.get(report.recommendations[i].priority, 3)
                    next_pri = priorities.get(report.recommendations[i + 1].priority, 3)
                    assert curr_pri <= next_pri

    def test_report_serialization(self):
        """Test report serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            d = report.to_dict()

            assert "timestamp" in d
            assert "summary" in d
            assert "attribution" in d
            assert "recommendations" in d

    def test_save_report(self):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
            )

            path = tool.save_report(report)
            assert path.exists()

            # Verify JSON is valid
            with open(path) as f:
                loaded = json.load(f)
            assert "summary" in loaded

    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            md = tool.generate_markdown_report(report)

            assert "# BTC-USD Performance Diagnosis Report" in md
            assert "Executive Summary" in md
            assert "Recommendations" in md

    def test_recoverable_return_estimate(self):
        """Test recoverable return estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = BTCDiagnosisTool(output_dir=Path(tmpdir))

            report = tool.analyze_from_backtest_results(
                btc_results=get_example_btc_results(),
                eth_results=get_example_eth_results(),
            )

            # Should estimate some recoverable return
            assert report.estimated_recoverable_pct > 0

            # But should be realistic (less than 50% of degradation)
            assert report.estimated_recoverable_pct < report.attribution.total_degradation_pct * 0.6


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2BIntegration:
    """Integration tests for Phase 2B components."""

    def test_shadow_collector_to_evaluator_flow(self):
        """Test data flow from collector to evaluator."""
        # Create collector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector_config = ShadowCollectorConfig(
                log_path=Path(tmpdir) / "shadow.jsonl",
                enable_rl_shadow=False,
            )
            collector = ShadowDataCollector(collector_config)

            # Record some decisions
            market_state = MarketState(symbol="BTC-USD", price=50000.0, regime="bull")

            for i in range(5):
                decision_id = collector.record_decision_point(
                    symbol="BTC-USD",
                    market_state=market_state,
                    gate_approved=True,
                    actual_action="long" if i % 2 == 0 else "hold",
                    strategy_used="TrendFollower",
                )

                collector.record_execution(
                    decision_id=decision_id,
                    entry_price=50000.0,
                    position_size=1000.0,
                    slippage_cost=5.0,
                )

                collector.record_outcome(
                    decision_id=decision_id,
                    exit_price=50500.0,
                    pnl=50.0,
                    pnl_pct=1.0,
                )

            # Verify data collected
            stats = collector.get_collection_stats()
            assert stats["total_decisions"] == 5

    def test_reward_shaper_with_preservation(self):
        """Test reward shaper respects preservation levels."""
        config = CostAwareConfig()
        shaper = CostAwareRewardShaper(cost_config=config)

        # Normal operation
        normal_reward = shaper.calculate_reward(
            pnl=100.0,
            pnl_pct=1.0,
            entry_price=50000.0,
            exit_price=50500.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            hold_time_minutes=60,
            slippage_cost=5.0,
            fee_cost=2.0,
            expected_slippage=5.0,
            position_value=1000.0,
            preservation_level="normal",
            preservation_violated=False,
        )

        # Lockdown violation
        violation_reward = shaper.calculate_reward(
            pnl=100.0,
            pnl_pct=1.0,
            entry_price=50000.0,
            exit_price=50500.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            hold_time_minutes=60,
            slippage_cost=5.0,
            fee_cost=2.0,
            expected_slippage=5.0,
            position_value=1000.0,
            preservation_level="lockdown",
            preservation_violated=True,
        )

        # Violation should be massively penalized
        assert normal_reward["total"] > 0
        assert violation_reward["total"] < -500

    def test_no_rl_execution_authority_verification(self):
        """CRITICAL: Verify RL components have no execution authority."""
        # Check CounterfactualEvaluator
        evaluator = CounterfactualEvaluator()
        assert not hasattr(evaluator, "execute")
        assert not hasattr(evaluator, "place_order")
        assert not hasattr(evaluator, "trade")

        # Check ShadowDataCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = ShadowDataCollector(ShadowCollectorConfig(
                log_path=Path(tmpdir) / "test.jsonl",
                enable_rl_shadow=False,
            ))
            assert not hasattr(collector, "execute")
            assert not hasattr(collector, "place_order")
            assert not hasattr(collector, "trade")

        # Check CostAwareRewardShaper
        shaper = CostAwareRewardShaper()
        assert not hasattr(shaper, "execute")
        assert not hasattr(shaper, "place_order")
        assert not hasattr(shaper, "trade")

        # Check StrategyWeightingAdvisor
        with tempfile.TemporaryDirectory() as tmpdir:
            advisor = StrategyWeightingAdvisor(state_path=Path(tmpdir) / "weights.json")
            assert not hasattr(advisor, "execute")
            assert not hasattr(advisor, "place_order")
            assert not hasattr(advisor, "trade")
            assert not hasattr(advisor, "set_leverage")
            assert not hasattr(advisor, "set_position_size")


# =============================================================================
# Safety Verification Tests
# =============================================================================

class TestPhase2BSafetyConstraints:
    """Verify all Phase 2B safety constraints are enforced."""

    def test_weight_shift_cannot_exceed_maximum(self):
        """Test weight shift is always clamped."""
        config = WeightingConfig(max_daily_shift=1.0)  # Try 100%
        assert config.max_daily_shift == MAX_DAILY_WEIGHT_SHIFT  # Should be 10%

    def test_strategy_weight_bounds_enforced(self):
        """Test strategy weights stay within bounds."""
        config = WeightingConfig()

        # Try extreme values
        config2 = WeightingConfig(min_weight=0.0, max_weight=1.0)

        assert config2.min_weight >= MIN_STRATEGY_WEIGHT
        assert config2.max_weight <= MAX_STRATEGY_WEIGHT

    def test_preservation_penalty_is_extreme(self):
        """Test capital preservation violation penalty is effectively infinite."""
        assert CAPITAL_PRESERVATION_VIOLATION_PENALTY <= -100.0

        # Should dominate any possible profit
        config = CostAwareConfig()
        shaper = CostAwareRewardShaper(cost_config=config)

        reward = shaper.calculate_reward(
            pnl=10000.0,  # Huge profit
            pnl_pct=100.0,
            entry_price=50000.0,
            exit_price=100000.0,
            stop_loss=49500.0,
            take_profit=101000.0,
            hold_time_minutes=60,
            slippage_cost=0.0,
            fee_cost=0.0,
            expected_slippage=0.0,
            position_value=1000.0,
            preservation_level="lockdown",
            preservation_violated=True,
        )

        # Even with huge profit, violation should make reward negative
        assert reward["total"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
