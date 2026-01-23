"""Tests for bot.position_sizer module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.position_sizer import (
    SizingMethod,
    SizingResult,
    PortfolioSizing,
    PositionSizer,
)


class TestSizingMethod:
    """Tests for SizingMethod enum."""

    def test_enum_values(self) -> None:
        """Test all enum values exist."""
        assert SizingMethod.FIXED_FRACTION.value == "fixed_fraction"
        assert SizingMethod.KELLY.value == "kelly"
        assert SizingMethod.HALF_KELLY.value == "half_kelly"
        assert SizingMethod.VOLATILITY_ADJUSTED.value == "volatility_adjusted"
        assert SizingMethod.RISK_PARITY.value == "risk_parity"
        assert SizingMethod.ATR_BASED.value == "atr_based"
        assert SizingMethod.DRAWDOWN_ADJUSTED.value == "drawdown_adjusted"
        assert SizingMethod.CONFIDENCE_SCALED.value == "confidence_scaled"
        assert SizingMethod.OPTIMAL_F.value == "optimal_f"


class TestSizingResult:
    """Tests for SizingResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = SizingResult(
            symbol="BTC/USDT",
            method=SizingMethod.KELLY,
            position_size=0.15,
            dollar_amount=1500.0,
            shares_or_units=0.03,
            confidence_adjustment=1.1,
            volatility_adjustment=0.9,
            drawdown_adjustment=1.0,
            reasoning="Kelly calculation",
        )

        data = result.to_dict()

        assert data["symbol"] == "BTC/USDT"
        assert data["method"] == "kelly"
        assert data["position_size"] == 0.15
        assert data["dollar_amount"] == 1500.0


class TestPortfolioSizing:
    """Tests for PortfolioSizing dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = SizingResult(
            symbol="BTC/USDT",
            method=SizingMethod.VOLATILITY_ADJUSTED,
            position_size=0.2,
            dollar_amount=2000,
            shares_or_units=0.04,
            confidence_adjustment=1.0,
            volatility_adjustment=1.0,
            drawdown_adjustment=1.0,
            reasoning="Test",
        )

        portfolio = PortfolioSizing(
            positions={"BTC/USDT": result},
            total_allocation=0.2,
            cash_reserve=0.8,
            leverage=1.0,
            risk_budget_used=0.04,
            warnings=[],
        )

        data = portfolio.to_dict()

        assert "positions" in data
        assert data["total_allocation"] == 0.2
        assert data["cash_reserve"] == 0.8


class TestPositionSizer:
    """Tests for PositionSizer class."""

    @pytest.fixture
    def sizer(self) -> PositionSizer:
        """Create position sizer with $10,000 portfolio."""
        return PositionSizer(portfolio_value=10000.0)

    def test_init_default_values(self, sizer: PositionSizer) -> None:
        """Test initialization with defaults."""
        assert sizer.portfolio_value == 10000.0
        assert sizer.max_position_size == 0.25
        assert sizer.max_portfolio_risk == 0.02
        assert sizer.min_position_size == 0.01
        assert sizer.max_leverage == 1.0

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        sizer = PositionSizer(
            portfolio_value=50000,
            max_position_size=0.30,
            max_leverage=2.0,
        )

        assert sizer.portfolio_value == 50000
        assert sizer.max_position_size == 0.30
        assert sizer.max_leverage == 2.0

    def test_update_portfolio_value_increase(self, sizer: PositionSizer) -> None:
        """Test updating portfolio value with increase."""
        sizer.update_portfolio_value(12000)

        assert sizer.portfolio_value == 12000
        assert sizer._peak_value == 12000
        assert sizer._current_drawdown == 0.0

    def test_update_portfolio_value_decrease(self, sizer: PositionSizer) -> None:
        """Test updating portfolio value with decrease (drawdown)."""
        sizer.update_portfolio_value(12000)  # New peak
        sizer.update_portfolio_value(10800)  # 10% drawdown

        assert sizer.portfolio_value == 10800
        assert sizer._peak_value == 12000
        assert abs(sizer._current_drawdown - 0.10) < 0.001

    def test_calculate_kelly_fraction_positive_edge(self, sizer: PositionSizer) -> None:
        """Test Kelly calculation with positive edge."""
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.55,
            win_loss_ratio=1.5,
        )

        # f = (0.55 * 1.5 - 0.45) / 1.5 = 0.25
        expected = (0.55 * 1.5 - 0.45) / 1.5
        assert abs(kelly - expected) < 0.001

    def test_calculate_kelly_fraction_no_edge(self, sizer: PositionSizer) -> None:
        """Test Kelly calculation with no edge."""
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.50,
            win_loss_ratio=1.0,
        )

        assert kelly == 0.0

    def test_calculate_kelly_fraction_negative_edge(self, sizer: PositionSizer) -> None:
        """Test Kelly calculation with negative edge returns 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.40,
            win_loss_ratio=1.0,
        )

        assert kelly == 0.0

    def test_calculate_kelly_fraction_invalid_inputs(self, sizer: PositionSizer) -> None:
        """Test Kelly calculation with invalid inputs."""
        assert sizer.calculate_kelly_fraction(0, 1.5) == 0.0
        assert sizer.calculate_kelly_fraction(1, 1.5) == 0.0
        assert sizer.calculate_kelly_fraction(0.55, 0) == 0.0

    def test_calculate_volatility_adjusted_size(self, sizer: PositionSizer) -> None:
        """Test volatility-adjusted sizing."""
        # With 30% volatility and 15% target, should scale to 0.5
        size = sizer.calculate_volatility_adjusted_size(
            volatility=0.30,
            target_volatility=0.15,
        )

        assert abs(size - 0.5) < 0.001

    def test_calculate_volatility_adjusted_size_low_vol(self, sizer: PositionSizer) -> None:
        """Test volatility-adjusted sizing with low volatility."""
        # With 10% volatility and 15% target, caps at 1.0
        size = sizer.calculate_volatility_adjusted_size(
            volatility=0.10,
            target_volatility=0.15,
        )

        assert size == 1.0

    def test_calculate_volatility_adjusted_size_zero_vol(self, sizer: PositionSizer) -> None:
        """Test volatility-adjusted sizing with zero volatility."""
        size = sizer.calculate_volatility_adjusted_size(volatility=0.0)

        assert size == 0.0

    def test_calculate_atr_based_size(self, sizer: PositionSizer) -> None:
        """Test ATR-based sizing."""
        size, stop_pct = sizer.calculate_atr_based_size(
            atr=100,
            price=5000,
        )

        assert size > 0
        assert stop_pct > 0
        assert stop_pct == (2 * 100) / 5000  # 2x ATR as stop

    def test_calculate_atr_based_size_invalid(self, sizer: PositionSizer) -> None:
        """Test ATR-based sizing with invalid inputs."""
        size, stop = sizer.calculate_atr_based_size(atr=0, price=5000)
        assert size == 0.0

        size, stop = sizer.calculate_atr_based_size(atr=100, price=0)
        assert size == 0.0

    def test_calculate_risk_parity_size(self, sizer: PositionSizer) -> None:
        """Test risk parity sizing."""
        volatilities = {
            "BTC/USDT": 0.60,  # Higher vol = lower weight
            "ETH/USDT": 0.80,
        }

        sizes = sizer.calculate_risk_parity_size(volatilities)

        assert "BTC/USDT" in sizes
        assert "ETH/USDT" in sizes
        # BTC should have higher weight due to lower volatility
        assert sizes["BTC/USDT"] > sizes["ETH/USDT"]

    def test_calculate_risk_parity_size_empty(self, sizer: PositionSizer) -> None:
        """Test risk parity with empty input."""
        sizes = sizer.calculate_risk_parity_size({})

        assert sizes == {}

    def test_calculate_optimal_f(self, sizer: PositionSizer) -> None:
        """Test Optimal f calculation."""
        # Generate some returns with a positive edge
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.03, 100)

        optimal_f = sizer.calculate_optimal_f(returns)

        assert 0 <= optimal_f <= 1.0

    def test_calculate_optimal_f_insufficient_data(self, sizer: PositionSizer) -> None:
        """Test Optimal f with insufficient data."""
        returns = np.array([0.01, 0.02, -0.01])

        optimal_f = sizer.calculate_optimal_f(returns)

        assert optimal_f == 0.0

    def test_calculate_optimal_f_no_losses(self, sizer: PositionSizer) -> None:
        """Test Optimal f with all positive returns."""
        returns = np.array([0.01] * 50)

        optimal_f = sizer.calculate_optimal_f(returns)

        assert optimal_f == 1.0

    def test_apply_confidence_scaling_low_confidence(self, sizer: PositionSizer) -> None:
        """Test confidence scaling with low confidence."""
        scaled, factor = sizer.apply_confidence_scaling(
            base_size=0.20,
            confidence=0.3,
            min_confidence=0.5,
        )

        assert scaled < 0.20
        assert factor < 1.0

    def test_apply_confidence_scaling_high_confidence(self, sizer: PositionSizer) -> None:
        """Test confidence scaling with high confidence."""
        scaled, factor = sizer.apply_confidence_scaling(
            base_size=0.20,
            confidence=0.9,
            min_confidence=0.5,
        )

        assert scaled > 0.20
        assert factor > 1.0

    def test_apply_confidence_scaling_normal_confidence(self, sizer: PositionSizer) -> None:
        """Test confidence scaling with normal confidence."""
        scaled, factor = sizer.apply_confidence_scaling(
            base_size=0.20,
            confidence=0.6,
            min_confidence=0.5,
        )

        assert abs(scaled - 0.20) < 0.001
        assert factor == 1.0

    def test_apply_drawdown_protection_no_drawdown(self, sizer: PositionSizer) -> None:
        """Test drawdown protection with no drawdown."""
        sizer._current_drawdown = 0.05  # 5% drawdown

        adjusted, factor = sizer.apply_drawdown_protection(
            base_size=0.20,
            drawdown_threshold=0.10,
        )

        assert adjusted == 0.20
        assert factor == 1.0

    def test_apply_drawdown_protection_in_drawdown(self, sizer: PositionSizer) -> None:
        """Test drawdown protection during drawdown."""
        sizer._current_drawdown = 0.15  # 15% drawdown

        adjusted, factor = sizer.apply_drawdown_protection(
            base_size=0.20,
            drawdown_threshold=0.10,
        )

        assert adjusted < 0.20
        assert factor < 1.0

    def test_apply_drawdown_protection_max_drawdown(self, sizer: PositionSizer) -> None:
        """Test drawdown protection at max drawdown."""
        sizer._current_drawdown = 0.25  # 25% drawdown

        adjusted, factor = sizer.apply_drawdown_protection(
            base_size=0.20,
            max_drawdown_reduction=0.5,
        )

        assert adjusted == 0.20 * 0.5
        assert factor == 0.5

    def test_calculate_size_fixed_fraction(self, sizer: PositionSizer) -> None:
        """Test fixed fraction sizing."""
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.FIXED_FRACTION,
            price=50000,
        )

        assert result.method == SizingMethod.FIXED_FRACTION
        assert result.position_size == sizer.max_position_size
        assert result.dollar_amount > 0

    def test_calculate_size_kelly(self, sizer: PositionSizer) -> None:
        """Test Kelly sizing."""
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.KELLY,
            price=50000,
            win_rate=0.55,
            win_loss_ratio=1.5,
        )

        assert result.method == SizingMethod.KELLY
        assert "Kelly" in result.reasoning

    def test_calculate_size_half_kelly(self, sizer: PositionSizer) -> None:
        """Test half Kelly sizing."""
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.HALF_KELLY,
            price=50000,
            win_rate=0.55,
            win_loss_ratio=1.5,
        )

        assert result.method == SizingMethod.HALF_KELLY
        assert "Half Kelly" in result.reasoning

    def test_calculate_size_volatility_adjusted(self, sizer: PositionSizer) -> None:
        """Test volatility-adjusted sizing."""
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.VOLATILITY_ADJUSTED,
            price=50000,
            volatility=0.60,
        )

        assert result.method == SizingMethod.VOLATILITY_ADJUSTED
        assert "Vol-adjusted" in result.reasoning

    def test_calculate_size_atr_based(self, sizer: PositionSizer) -> None:
        """Test ATR-based sizing."""
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.ATR_BASED,
            price=50000,
            atr=1000,
        )

        assert result.method == SizingMethod.ATR_BASED
        assert "ATR-based" in result.reasoning

    def test_calculate_size_respects_limits(self, sizer: PositionSizer) -> None:
        """Test that position size respects min/max limits."""
        # Very high Kelly fraction
        result = sizer.calculate_size(
            symbol="BTC/USDT",
            method=SizingMethod.KELLY,
            price=50000,
            win_rate=0.80,
            win_loss_ratio=3.0,
        )

        assert result.position_size <= sizer.max_position_size
        assert result.position_size >= sizer.min_position_size

    def test_calculate_portfolio_sizes(self, sizer: PositionSizer) -> None:
        """Test portfolio sizing for multiple assets."""
        signals = [
            {"symbol": "BTC/USDT", "price": 50000, "confidence": 0.7},
            {"symbol": "ETH/USDT", "price": 3000, "confidence": 0.6},
        ]
        volatilities = {"BTC/USDT": 0.60, "ETH/USDT": 0.80}

        portfolio = sizer.calculate_portfolio_sizes(
            signals=signals,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            volatilities=volatilities,
        )

        assert len(portfolio.positions) == 2
        assert "BTC/USDT" in portfolio.positions
        assert "ETH/USDT" in portfolio.positions
        assert portfolio.total_allocation > 0

    def test_calculate_portfolio_sizes_risk_parity(self, sizer: PositionSizer) -> None:
        """Test portfolio sizing with risk parity."""
        signals = [
            {"symbol": "BTC/USDT", "price": 50000, "confidence": 0.7},
            {"symbol": "ETH/USDT", "price": 3000, "confidence": 0.6},
        ]
        volatilities = {"BTC/USDT": 0.60, "ETH/USDT": 0.80}

        portfolio = sizer.calculate_portfolio_sizes(
            signals=signals,
            method=SizingMethod.RISK_PARITY,
            volatilities=volatilities,
        )

        assert len(portfolio.positions) == 2

    def test_calculate_portfolio_sizes_scales_down(self, sizer: PositionSizer) -> None:
        """Test portfolio sizing scales down when over-allocated."""
        # Many signals that would exceed max allocation
        signals = [{"symbol": f"ASSET{i}", "price": 100, "confidence": 0.8} for i in range(10)]

        portfolio = sizer.calculate_portfolio_sizes(
            signals=signals,
            method=SizingMethod.FIXED_FRACTION,
        )

        # Total allocation should not exceed available capital
        max_allocation = 1.0 - sizer.min_cash_reserve
        assert portfolio.total_allocation <= max_allocation + 0.001
        assert len(portfolio.warnings) > 0

    def test_suggest_sizing_method_good_win_rate(self, sizer: PositionSizer) -> None:
        """Test sizing method suggestion with good win rate."""
        method, reason = sizer.suggest_sizing_method(
            historical_win_rate=0.65,
        )

        assert method == SizingMethod.KELLY
        assert "Strong edge" in reason

    def test_suggest_sizing_method_moderate_win_rate(self, sizer: PositionSizer) -> None:
        """Test sizing method suggestion with moderate win rate."""
        method, reason = sizer.suggest_sizing_method(
            historical_win_rate=0.55,
        )

        assert method == SizingMethod.HALF_KELLY

    def test_suggest_sizing_method_with_returns(self, sizer: PositionSizer) -> None:
        """Test sizing method suggestion with returns history."""
        returns = np.random.normal(0, 0.02, 150)

        method, reason = sizer.suggest_sizing_method(
            historical_returns=returns,
        )

        assert method == SizingMethod.OPTIMAL_F

    def test_suggest_sizing_method_high_volatility(self, sizer: PositionSizer) -> None:
        """Test sizing method suggestion with high volatility."""
        method, reason = sizer.suggest_sizing_method(
            current_volatility=0.60,
        )

        assert method == SizingMethod.ATR_BASED

    def test_suggest_sizing_method_default(self, sizer: PositionSizer) -> None:
        """Test sizing method suggestion with no data."""
        method, reason = sizer.suggest_sizing_method()

        assert method == SizingMethod.CONFIDENCE_SCALED
