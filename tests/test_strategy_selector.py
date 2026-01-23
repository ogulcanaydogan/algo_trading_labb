"""Tests for regime-aware strategy selector."""

import pytest
from datetime import datetime

from bot.regime.strategy_selector import (
    StrategyType,
    StrategyConfig,
    StrategyPerformance,
    SelectionResult,
    SelectorConfig,
    RegimeStrategySelector,
    create_default_strategies,
    create_regime_strategy_selector,
)
from bot.regime.regime_detector import MarketRegime


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_strategy_types(self):
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.BREAKOUT.value == "breakout"
        assert StrategyType.VOLATILITY.value == "volatility"
        assert StrategyType.DEFENSIVE.value == "defensive"


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_strategy_config_creation(self):
        config = StrategyConfig(
            strategy_id="test_strategy",
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL, MarketRegime.STRONG_BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        assert config.strategy_id == "test_strategy"
        assert config.name == "Test Strategy"
        assert config.strategy_type == StrategyType.TREND_FOLLOWING

    def test_is_suitable_for_regime_suitable(self):
        config = StrategyConfig(
            strategy_id="test",
            name="Test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        assert config.is_suitable_for_regime(MarketRegime.BULL) is True

    def test_is_suitable_for_regime_unsuitable(self):
        config = StrategyConfig(
            strategy_id="test",
            name="Test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        assert config.is_suitable_for_regime(MarketRegime.CRASH) is False

    def test_is_suitable_for_regime_not_in_suitable(self):
        config = StrategyConfig(
            strategy_id="test",
            name="Test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        # SIDEWAYS is not in suitable_regimes
        assert config.is_suitable_for_regime(MarketRegime.SIDEWAYS) is False

    def test_strategy_config_to_dict(self):
        config = StrategyConfig(
            strategy_id="test",
            name="Test",
            strategy_type=StrategyType.MEAN_REVERSION,
            suitable_regimes={MarketRegime.SIDEWAYS},
            unsuitable_regimes={MarketRegime.CRASH},
            priority=2,
        )
        result = config.to_dict()
        assert result["strategy_id"] == "test"
        assert result["strategy_type"] == "mean_reversion"
        assert result["priority"] == 2


class TestStrategyPerformance:
    """Tests for StrategyPerformance dataclass."""

    def test_performance_creation(self):
        perf = StrategyPerformance(
            strategy_id="test_strategy",
            regime=MarketRegime.BULL,
            total_return=0.15,
            sharpe_ratio=1.5,
            win_rate=0.6,
            profit_factor=1.8,
            max_drawdown=0.1,
            trade_count=50,
            avg_holding_period=4.0,
        )
        assert perf.strategy_id == "test_strategy"
        assert perf.sharpe_ratio == 1.5

    def test_performance_score_high(self):
        perf = StrategyPerformance(
            strategy_id="good_strategy",
            regime=MarketRegime.BULL,
            total_return=0.25,
            sharpe_ratio=2.0,
            win_rate=0.65,
            profit_factor=2.0,
            max_drawdown=0.08,
            trade_count=100,
            avg_holding_period=6.0,
        )
        score = perf.score
        assert score > 0.3  # Good strategy should have positive score

    def test_performance_score_low(self):
        perf = StrategyPerformance(
            strategy_id="bad_strategy",
            regime=MarketRegime.BEAR,
            total_return=-0.1,
            sharpe_ratio=-0.5,
            win_rate=0.35,
            profit_factor=0.5,
            max_drawdown=0.35,
            trade_count=50,
            avg_holding_period=2.0,
        )
        score = perf.score
        assert score < 0  # Bad strategy should have negative score

    def test_performance_to_dict(self):
        perf = StrategyPerformance(
            strategy_id="test",
            regime=MarketRegime.SIDEWAYS,
            total_return=0.1,
            sharpe_ratio=1.2,
            win_rate=0.55,
            profit_factor=1.5,
            max_drawdown=0.12,
            trade_count=30,
            avg_holding_period=8.0,
        )
        result = perf.to_dict()
        assert result["strategy_id"] == "test"
        assert result["regime"] == "sideways"
        assert "score" in result


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_selection_result_creation(self):
        result = SelectionResult(
            selected_strategies=["strat1", "strat2"],
            regime=MarketRegime.BULL,
            confidence=0.85,
            reasoning=["Selected based on regime", "High performance score"],
            position_scale=1.2,
            risk_scale=1.0,
            excluded_strategies=["strat3"],
        )
        assert len(result.selected_strategies) == 2
        assert result.confidence == 0.85

    def test_selection_result_to_dict(self):
        result = SelectionResult(
            selected_strategies=["strat1"],
            regime=MarketRegime.CRASH,
            confidence=0.3,
            reasoning=["Defensive mode"],
            position_scale=0.3,
            risk_scale=0.2,
            excluded_strategies=["strat2", "strat3"],
        )
        data = result.to_dict()
        assert data["regime"] == "crash"
        assert data["position_scale"] == 0.3


class TestSelectorConfig:
    """Tests for SelectorConfig dataclass."""

    def test_default_config(self):
        config = SelectorConfig()
        assert config.max_concurrent_strategies == 3
        assert config.min_trade_count_for_scoring == 10
        assert config.base_position_scale == 1.0

    def test_custom_config(self):
        config = SelectorConfig(
            max_concurrent_strategies=5,
            crash_scale=0.1,
        )
        assert config.max_concurrent_strategies == 5
        assert config.crash_scale == 0.1


class TestRegimeStrategySelector:
    """Tests for RegimeStrategySelector."""

    @pytest.fixture
    def selector(self):
        return RegimeStrategySelector()

    @pytest.fixture
    def selector_with_strategies(self):
        selector = RegimeStrategySelector()
        for config in create_default_strategies():
            selector.register_strategy(config)
        return selector

    def test_register_strategy(self, selector):
        config = StrategyConfig(
            strategy_id="test_strat",
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        selector.register_strategy(config)
        assert "test_strat" in selector._strategies

    def test_unregister_strategy(self, selector):
        config = StrategyConfig(
            strategy_id="test_strat",
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            suitable_regimes={MarketRegime.BULL},
            unsuitable_regimes={MarketRegime.CRASH},
        )
        selector.register_strategy(config)
        selector.unregister_strategy("test_strat")
        assert "test_strat" not in selector._strategies

    def test_update_performance(self, selector):
        perf = StrategyPerformance(
            strategy_id="test_strat",
            regime=MarketRegime.BULL,
            total_return=0.15,
            sharpe_ratio=1.5,
            win_rate=0.6,
            profit_factor=1.8,
            max_drawdown=0.1,
            trade_count=50,
            avg_holding_period=4.0,
        )
        selector.update_performance("test_strat", MarketRegime.BULL, perf)
        assert ("test_strat", MarketRegime.BULL) in selector._performance

    def test_select_strategies_bull(self, selector_with_strategies):
        result = selector_with_strategies.select_strategies(
            MarketRegime.BULL, force_reselection=True
        )
        assert isinstance(result, SelectionResult)
        assert result.regime == MarketRegime.BULL
        assert len(result.selected_strategies) <= 3

    def test_select_strategies_crash_uses_defensive(self, selector_with_strategies):
        result = selector_with_strategies.select_strategies(
            MarketRegime.CRASH, force_reselection=True
        )
        assert result.position_scale < 1.0  # Should reduce exposure
        assert result.risk_scale < 1.0

    def test_select_strategies_with_volatility_filter(self, selector_with_strategies):
        # Register a strategy with min_volatility requirement
        config = StrategyConfig(
            strategy_id="vol_strat",
            name="Volatility Strategy",
            strategy_type=StrategyType.VOLATILITY,
            suitable_regimes={MarketRegime.HIGH_VOL},
            unsuitable_regimes=set(),
            min_volatility=0.05,
        )
        selector_with_strategies.register_strategy(config)

        # Low volatility should exclude it
        result = selector_with_strategies.select_strategies(
            MarketRegime.HIGH_VOL,
            volatility=0.02,
            force_reselection=True,
        )
        assert "vol_strat" in result.excluded_strategies

    def test_regime_transition_cooldown(self, selector_with_strategies):
        # First selection
        result1 = selector_with_strategies.select_strategies(MarketRegime.BULL)

        # Immediate second selection with different regime
        # Should be in cooldown
        result2 = selector_with_strategies.select_strategies(MarketRegime.BEAR)
        # During cooldown, keeps previous strategies
        assert result2.reasoning[0].startswith("In cooldown")

    def test_force_reselection_bypasses_cooldown(self, selector_with_strategies):
        result1 = selector_with_strategies.select_strategies(MarketRegime.BULL)
        result2 = selector_with_strategies.select_strategies(
            MarketRegime.CRASH,
            force_reselection=True,
        )
        # Should not be in cooldown with force_reselection
        assert result2.regime == MarketRegime.CRASH

    def test_get_strategy_allocation(self):
        # Create fresh selector to avoid cooldown issues
        selector = RegimeStrategySelector()
        for config in create_default_strategies():
            selector.register_strategy(config)

        # First call with force_reselection to initialize state
        selector.select_strategies(MarketRegime.BULL, force_reselection=True)

        allocations = selector.get_strategy_allocation(
            MarketRegime.BULL,
            total_capital=100000.0,
        )
        assert isinstance(allocations, dict)
        total_allocated = sum(allocations.values())
        # Should allocate something in bull market
        assert total_allocated > 0

    def test_get_current_state(self, selector_with_strategies):
        selector_with_strategies.select_strategies(MarketRegime.SIDEWAYS)
        state = selector_with_strategies.get_current_state()
        assert "current_regime" in state
        assert "active_strategies" in state
        assert state["registered_strategies"] > 0


class TestCreateDefaultStrategies:
    """Tests for create_default_strategies function."""

    def test_creates_strategies(self):
        strategies = create_default_strategies()
        assert len(strategies) > 0
        assert all(isinstance(s, StrategyConfig) for s in strategies)

    def test_includes_trend_following(self):
        strategies = create_default_strategies()
        trend_strategies = [
            s for s in strategies
            if s.strategy_type == StrategyType.TREND_FOLLOWING
        ]
        assert len(trend_strategies) >= 1

    def test_includes_defensive(self):
        strategies = create_default_strategies()
        defensive_strategies = [
            s for s in strategies
            if s.strategy_type == StrategyType.DEFENSIVE
        ]
        assert len(defensive_strategies) >= 1


class TestCreateRegimeStrategySelector:
    """Tests for create_regime_strategy_selector factory function."""

    def test_creates_selector(self):
        selector = create_regime_strategy_selector()
        assert isinstance(selector, RegimeStrategySelector)

    def test_includes_defaults(self):
        selector = create_regime_strategy_selector(include_defaults=True)
        assert len(selector._strategies) > 0

    def test_excludes_defaults(self):
        selector = create_regime_strategy_selector(include_defaults=False)
        assert len(selector._strategies) == 0

    def test_with_custom_config(self):
        config = SelectorConfig(max_concurrent_strategies=5)
        selector = create_regime_strategy_selector(config=config)
        assert selector.config.max_concurrent_strategies == 5
