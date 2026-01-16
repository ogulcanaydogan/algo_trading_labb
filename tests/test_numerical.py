"""
Tests for numerical utility functions.
"""

import pytest
import numpy as np

from bot.utils.numerical import (
    safe_divide,
    safe_pct_change,
    safe_ratio,
    safe_compare,
    clip_value,
    sanitize_array,
    ensure_float,
    profit_factor,
    sharpe_ratio,
    max_drawdown,
)


class TestSafeDivide:
    """Test safe_divide function."""

    def test_normal_division(self):
        """Test normal division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(100, 4) == 25.0
        assert safe_divide(1, 3) == pytest.approx(0.333, rel=0.01)

    def test_divide_by_zero(self):
        """Test division by zero returns default."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0

    def test_negative_division(self):
        """Test negative number division."""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0
        assert safe_divide(-10, -2) == 5.0

    def test_nan_denominator(self):
        """Test NaN denominator returns default."""
        assert safe_divide(10, np.nan) == 0.0

    def test_inf_denominator(self):
        """Test inf denominator returns default."""
        assert safe_divide(10, np.inf) == 0.0

    def test_zero_numerator(self):
        """Test zero numerator."""
        assert safe_divide(0, 10) == 0.0

    def test_custom_default(self):
        """Test custom default value."""
        assert safe_divide(10, 0, default=-1.0) == -1.0


class TestSafePctChange:
    """Test safe_pct_change function."""

    def test_positive_change(self):
        """Test positive percentage change."""
        result = safe_pct_change(110, 100)
        assert result == pytest.approx(0.10, rel=0.01)

    def test_negative_change(self):
        """Test negative percentage change."""
        result = safe_pct_change(90, 100)
        assert result == pytest.approx(-0.10, rel=0.01)

    def test_no_change(self):
        """Test no change."""
        assert safe_pct_change(100, 100) == 0.0

    def test_zero_previous(self):
        """Test zero previous value."""
        assert safe_pct_change(100, 0) == 0.0

    def test_custom_default(self):
        """Test custom default."""
        assert safe_pct_change(100, 0, default=-1.0) == -1.0


class TestSafeRatio:
    """Test safe_ratio function."""

    def test_normal_ratio(self):
        """Test normal ratio calculation."""
        assert safe_ratio(10, 5) == 2.0
        assert safe_ratio(3, 4) == 0.75

    def test_zero_denominator(self):
        """Test zero denominator returns default."""
        assert safe_ratio(10, 0) == 1.0

    def test_custom_default(self):
        """Test custom default."""
        assert safe_ratio(10, 0, default=0.5) == 0.5


class TestSafeCompare:
    """Test safe_compare function."""

    def test_greater_equal(self):
        """Test >= comparison."""
        assert safe_compare(10, 5, ">=") is True
        assert safe_compare(5, 5, ">=") is True
        assert safe_compare(4, 5, ">=") is False

    def test_less_equal(self):
        """Test <= comparison."""
        assert safe_compare(5, 10, "<=") is True
        assert safe_compare(5, 5, "<=") is True
        assert safe_compare(6, 5, "<=") is False

    def test_greater_than(self):
        """Test > comparison."""
        assert safe_compare(10, 5, ">") is True
        assert safe_compare(5, 5, ">") is False

    def test_less_than(self):
        """Test < comparison."""
        assert safe_compare(5, 10, "<") is True
        assert safe_compare(5, 5, "<") is False

    def test_equal(self):
        """Test == comparison."""
        assert safe_compare(5, 5, "==") is True
        assert safe_compare(5, 6, "==") is False

    def test_none_values(self):
        """Test None values return False."""
        assert safe_compare(None, 5, ">=") is False
        assert safe_compare(5, None, ">=") is False

    def test_nan_values(self):
        """Test NaN values return False."""
        assert safe_compare(np.nan, 5, ">=") is False
        assert safe_compare(5, np.nan, ">=") is False

    def test_inf_values(self):
        """Test inf values return False."""
        assert safe_compare(np.inf, 5, ">=") is False

    def test_invalid_operator(self):
        """Test invalid operator returns False."""
        assert safe_compare(5, 5, "!=") is False

    def test_string_numbers(self):
        """Test string numbers are converted."""
        assert safe_compare("10", "5", ">") is True


class TestClipValue:
    """Test clip_value function."""

    def test_no_clipping_needed(self):
        """Test value within range."""
        assert clip_value(5, 0, 10) == 5.0

    def test_clip_to_min(self):
        """Test clipping to minimum."""
        assert clip_value(-5, 0, 10) == 0.0

    def test_clip_to_max(self):
        """Test clipping to maximum."""
        assert clip_value(15, 0, 10) == 10.0

    def test_only_min(self):
        """Test only minimum bound."""
        assert clip_value(-5, min_val=0) == 0.0
        assert clip_value(100, min_val=0) == 100.0

    def test_only_max(self):
        """Test only maximum bound."""
        assert clip_value(100, max_val=50) == 50.0
        assert clip_value(-10, max_val=50) == -10.0

    def test_nan_value(self):
        """Test NaN value returns default."""
        assert clip_value(np.nan, 0, 10) == 0.0

    def test_inf_value(self):
        """Test inf value returns default."""
        assert clip_value(np.inf, 0, 10) == 0.0

    def test_custom_default(self):
        """Test custom default."""
        assert clip_value(np.nan, 0, 10, default=5.0) == 5.0


class TestSanitizeArray:
    """Test sanitize_array function."""

    def test_normal_array(self):
        """Test normal array unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = sanitize_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_nan_replaced(self):
        """Test NaN values replaced."""
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_inf_replaced(self):
        """Test inf values replaced."""
        arr = np.array([1.0, np.inf, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_neginf_replaced(self):
        """Test negative inf replaced."""
        arr = np.array([1.0, -np.inf, 3.0])
        result = sanitize_array(arr)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_custom_replace_value(self):
        """Test custom replacement value."""
        arr = np.array([1.0, np.nan, 3.0])
        result = sanitize_array(arr, replace_val=-1.0)
        expected = np.array([1.0, -1.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestEnsureFloat:
    """Test ensure_float function."""

    def test_int_to_float(self):
        """Test integer conversion."""
        assert ensure_float(5) == 5.0

    def test_float_unchanged(self):
        """Test float unchanged."""
        assert ensure_float(3.14) == 3.14

    def test_string_number(self):
        """Test string number conversion."""
        assert ensure_float("3.14") == 3.14

    def test_nan_returns_default(self):
        """Test NaN returns default."""
        assert ensure_float(np.nan) == 0.0

    def test_inf_returns_default(self):
        """Test inf returns default."""
        assert ensure_float(np.inf) == 0.0

    def test_none_returns_default(self):
        """Test None returns default."""
        assert ensure_float(None) == 0.0

    def test_invalid_string_returns_default(self):
        """Test invalid string returns default."""
        assert ensure_float("abc") == 0.0

    def test_custom_default(self):
        """Test custom default."""
        assert ensure_float(None, default=-1.0) == -1.0


class TestProfitFactor:
    """Test profit_factor function."""

    def test_normal_profit_factor(self):
        """Test normal profit factor calculation."""
        result = profit_factor(100, 50)
        assert result == 2.0

    def test_zero_loss(self):
        """Test zero loss with profit."""
        result = profit_factor(100, 0)
        assert result == 100.0

    def test_zero_loss_zero_profit(self):
        """Test zero loss and zero profit."""
        result = profit_factor(0, 0)
        assert result == 1.0

    def test_capped_at_100(self):
        """Test profit factor capped at 100."""
        result = profit_factor(10000, 1)
        assert result == 100.0

    def test_negative_loss(self):
        """Test negative loss (invalid) returns correct value."""
        result = profit_factor(100, -10)
        assert result == 100.0


class TestSharpeRatio:
    """Test sharpe_ratio function."""

    def test_positive_sharpe(self):
        """Test positive Sharpe ratio."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025])
        result = sharpe_ratio(returns)
        assert result > 0

    def test_negative_sharpe(self):
        """Test negative Sharpe ratio."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.025])
        result = sharpe_ratio(returns)
        assert result < 0

    def test_zero_variance(self):
        """Test zero variance returns zero."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_insufficient_data(self):
        """Test insufficient data returns zero."""
        returns = np.array([0.01])
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_empty_array(self):
        """Test empty array returns zero."""
        returns = np.array([])
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_nan_in_returns(self):
        """Test NaN values handled."""
        returns = np.array([0.01, np.nan, 0.02, 0.015])
        result = sharpe_ratio(returns)
        assert np.isfinite(result)

    def test_clipped_range(self):
        """Test result clipped to reasonable range."""
        # Very high variance returns should be clipped
        returns = np.random.randn(100) * 10
        result = sharpe_ratio(returns)
        assert -10.0 <= result <= 10.0


class TestMaxDrawdown:
    """Test max_drawdown function."""

    def test_no_drawdown(self):
        """Test monotonically increasing equity."""
        equity = np.array([100, 110, 120, 130, 140])
        result = max_drawdown(equity)
        assert result == 0.0

    def test_simple_drawdown(self):
        """Test simple drawdown."""
        equity = np.array([100, 110, 100, 90, 95])
        result = max_drawdown(equity)
        # Peak of 110, trough of 90 = 18.18% drawdown
        assert result == pytest.approx(0.182, rel=0.01)

    def test_full_drawdown(self):
        """Test complete drawdown."""
        equity = np.array([100, 50, 0])
        result = max_drawdown(equity)
        assert result == 1.0

    def test_insufficient_data(self):
        """Test insufficient data."""
        equity = np.array([100])
        result = max_drawdown(equity)
        assert result == 0.0

    def test_empty_array(self):
        """Test empty array."""
        equity = np.array([])
        result = max_drawdown(equity)
        assert result == 0.0

    def test_nan_in_equity(self):
        """Test NaN values handled."""
        equity = np.array([100, np.nan, 110, 105])
        result = max_drawdown(equity)
        assert np.isfinite(result)

    def test_clipped_to_valid_range(self):
        """Test result in valid range."""
        equity = np.array([100, 80, 90, 70, 85])
        result = max_drawdown(equity)
        assert 0.0 <= result <= 1.0
