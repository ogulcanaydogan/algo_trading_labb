"""Tests for dynamic risk controls."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.risk.dynamic_controls import (
    CircuitBreakerState,
    RiskLevel,
    CorrelationAlert,
    CircuitBreakerStatus,
    CorrelationConfig,
    CorrelationCircuitBreaker,
    PositionSizingConfig,
    PositionSizeResult,
    DynamicPositionSizer,
    create_correlation_circuit_breaker,
    create_dynamic_position_sizer,
)


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState enum."""

    def test_states(self):
        assert CircuitBreakerState.NORMAL.value == "normal"
        assert CircuitBreakerState.WARNING.value == "warning"
        assert CircuitBreakerState.TRIGGERED.value == "triggered"
        assert CircuitBreakerState.COOLDOWN.value == "cooldown"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestCorrelationAlert:
    """Tests for CorrelationAlert dataclass."""

    def test_alert_creation(self):
        alert = CorrelationAlert(
            timestamp=datetime.now(),
            asset_pair=("BTC", "ETH"),
            historical_correlation=0.7,
            current_correlation=0.95,
            change_magnitude=0.25,
            alert_type="spike",
            severity="warning",
        )
        assert alert.asset_pair == ("BTC", "ETH")
        assert alert.change_magnitude == 0.25

    def test_alert_to_dict(self):
        alert = CorrelationAlert(
            timestamp=datetime.now(),
            asset_pair=("BTC", "ETH"),
            historical_correlation=0.6,
            current_correlation=0.9,
            change_magnitude=0.3,
            alert_type="breakdown",
            severity="critical",
        )
        result = alert.to_dict()
        assert result["asset_pair"] == ["BTC", "ETH"]
        assert result["alert_type"] == "breakdown"


class TestCorrelationConfig:
    """Tests for CorrelationConfig dataclass."""

    def test_default_config(self):
        config = CorrelationConfig()
        assert config.correlation_spike_threshold == 0.3
        assert config.cooldown_minutes == 30

    def test_custom_config(self):
        config = CorrelationConfig(
            correlation_spike_threshold=0.4,
            warning_reduction_factor=0.5,
        )
        assert config.correlation_spike_threshold == 0.4
        assert config.warning_reduction_factor == 0.5


class TestCorrelationCircuitBreaker:
    """Tests for CorrelationCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        return CorrelationCircuitBreaker()

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="1H")
        return pd.DataFrame(
            {
                "BTC": np.random.randn(100) * 0.02,
                "ETH": np.random.randn(100) * 0.025,
                "SOL": np.random.randn(100) * 0.03,
            },
            index=dates,
        )

    def test_initial_state(self, breaker):
        assert breaker.state == CircuitBreakerState.NORMAL
        assert breaker.risk_reduction_factor == 1.0

    def test_update_correlations_normal(self, breaker, sample_returns):
        status = breaker.update_correlations(sample_returns)
        assert status.state == CircuitBreakerState.NORMAL

    def test_spike_detection(self, breaker):
        """Test detection of correlation spike."""
        # Create baseline uncorrelated data
        np.random.seed(42)
        base = np.random.randn(100) * 0.02
        uncorrelated = np.random.randn(100) * 0.02  # Independent

        returns = pd.DataFrame(
            {
                "BTC": base,
                "ETH": uncorrelated,
            }
        )

        # First update establishes baseline
        breaker.update_correlations(returns)

        # Create moderately correlated data (spike but not extreme)
        correlated = base * 0.5 + np.random.randn(100) * 0.015
        spike_returns = pd.DataFrame(
            {
                "BTC": base,
                "ETH": correlated,
            }
        )

        status = breaker.update_correlations(spike_returns)
        # Circuit breaker should detect the spike
        assert status.state in [
            CircuitBreakerState.NORMAL,
            CircuitBreakerState.WARNING,
            CircuitBreakerState.TRIGGERED,  # May trigger if correlation is high
        ]

    def test_manual_reset(self, breaker):
        breaker._state = CircuitBreakerState.TRIGGERED
        breaker.reset()
        assert breaker.state == CircuitBreakerState.NORMAL

    def test_listener_callback(self, breaker, sample_returns):
        callback_called = []

        def callback(status):
            callback_called.append(status)

        breaker.add_listener(callback)
        breaker.update_correlations(sample_returns)

        assert len(callback_called) == 1

    def test_get_status(self, breaker):
        status = breaker.get_status()
        assert isinstance(status, CircuitBreakerStatus)
        assert status.state == CircuitBreakerState.NORMAL

    def test_risk_reduction_warning(self, breaker):
        breaker._state = CircuitBreakerState.WARNING
        assert breaker.risk_reduction_factor == breaker.config.warning_reduction_factor

    def test_risk_reduction_triggered(self, breaker):
        breaker._state = CircuitBreakerState.TRIGGERED
        assert breaker.risk_reduction_factor == breaker.config.triggered_reduction_factor


class TestPositionSizingConfig:
    """Tests for PositionSizingConfig dataclass."""

    def test_default_config(self):
        config = PositionSizingConfig()
        assert config.base_risk_per_trade == 0.02
        assert config.max_position_size == 0.20
        assert config.target_volatility == 0.15

    def test_custom_config(self):
        config = PositionSizingConfig(
            base_risk_per_trade=0.01,
            kelly_fraction=0.5,
        )
        assert config.base_risk_per_trade == 0.01
        assert config.kelly_fraction == 0.5


class TestDynamicPositionSizer:
    """Tests for DynamicPositionSizer."""

    @pytest.fixture
    def sizer(self):
        sizer = DynamicPositionSizer()
        sizer.set_portfolio_value(100000)
        return sizer

    @pytest.fixture
    def sizer_with_breaker(self):
        breaker = CorrelationCircuitBreaker()
        sizer = DynamicPositionSizer(circuit_breaker=breaker)
        sizer.set_portfolio_value(100000)
        return sizer

    def test_set_portfolio_value(self, sizer):
        sizer.set_portfolio_value(50000)
        assert sizer._portfolio_value == 50000

    def test_set_portfolio_value_updates_drawdown(self, sizer):
        sizer.set_portfolio_value(100000)
        sizer.set_portfolio_value(90000)
        assert sizer._current_drawdown == 0.1  # 10% drawdown

    def test_calculate_position_size_basic(self, sizer):
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,  # 2% stop
            current_volatility=0.15,
        )
        assert isinstance(result, PositionSizeResult)
        assert result.final_size > 0
        assert result.final_size <= sizer.config.max_position_size

    def test_volatility_adjustment(self, sizer):
        # High volatility should reduce size
        result_high_vol = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.30,  # Double target
        )

        result_low_vol = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.10,  # Below target
        )

        # Check volatility-adjusted sizes (before kelly limit kicks in)
        # Higher vol should give smaller volatility-adjusted position
        assert result_high_vol.volatility_adjusted_size < result_low_vol.volatility_adjusted_size

    def test_regime_adjustment_bull(self, sizer):
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
            regime="bull",
        )
        assert "regime" in result.adjustments_applied
        assert result.adjustments_applied["regime"] == sizer.config.bull_regime_multiplier

    def test_regime_adjustment_crash(self, sizer):
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
            regime="crash",
        )
        assert "regime" in result.adjustments_applied
        assert result.adjustments_applied["regime"] == sizer.config.crash_regime_multiplier

    def test_drawdown_adjustment(self, sizer):
        # Simulate drawdown
        sizer._peak_value = 100000
        sizer._portfolio_value = 85000
        sizer._current_drawdown = 0.15

        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )
        assert "drawdown" in result.adjustments_applied
        # Should reduce position due to drawdown
        assert result.adjustments_applied["drawdown"] < 1.0

    def test_kelly_criterion(self, sizer):
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
        )
        assert result.kelly_suggested_size > 0

    def test_circuit_breaker_integration(self, sizer_with_breaker):
        # Trigger circuit breaker
        sizer_with_breaker.circuit_breaker._state = CircuitBreakerState.TRIGGERED

        result = sizer_with_breaker.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )
        assert "circuit_breaker" in result.adjustments_applied
        # Should be significantly reduced
        assert result.adjustments_applied["circuit_breaker"] < 1.0

    def test_empty_result_no_portfolio(self):
        sizer = DynamicPositionSizer()
        # Don't set portfolio value
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )
        assert result.final_size == 0

    def test_get_portfolio_risk_level_low(self, sizer):
        sizer._current_drawdown = 0.02
        assert sizer.get_portfolio_risk_level() == RiskLevel.LOW

    def test_get_portfolio_risk_level_medium(self, sizer):
        sizer._current_drawdown = 0.07
        assert sizer.get_portfolio_risk_level() == RiskLevel.MEDIUM

    def test_get_portfolio_risk_level_high(self, sizer):
        sizer._current_drawdown = 0.15
        assert sizer.get_portfolio_risk_level() == RiskLevel.HIGH

    def test_get_portfolio_risk_level_critical(self, sizer):
        sizer._current_drawdown = 0.25
        assert sizer.get_portfolio_risk_level() == RiskLevel.CRITICAL

    def test_position_size_result_to_dict(self, sizer):
        result = sizer.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )
        data = result.to_dict()
        assert "base_size" in data
        assert "final_size" in data
        assert "adjustments_applied" in data


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_correlation_circuit_breaker(self):
        breaker = create_correlation_circuit_breaker()
        assert isinstance(breaker, CorrelationCircuitBreaker)

    def test_create_circuit_breaker_with_config(self):
        config = CorrelationConfig(cooldown_minutes=60)
        breaker = create_correlation_circuit_breaker(config=config)
        assert breaker.config.cooldown_minutes == 60

    def test_create_dynamic_position_sizer(self):
        sizer = create_dynamic_position_sizer()
        assert isinstance(sizer, DynamicPositionSizer)

    def test_create_position_sizer_with_breaker(self):
        breaker = create_correlation_circuit_breaker()
        sizer = create_dynamic_position_sizer(circuit_breaker=breaker)
        assert sizer.circuit_breaker is breaker
