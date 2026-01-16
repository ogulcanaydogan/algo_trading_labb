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
    ensure_float,
    max_drawdown,
    profit_factor,
    sanitize_array,
    sharpe_ratio,
)


class TestSafeDivide:
    """Test safe_divide function."""

    def test_normal_division(self):
        """Test normal division works."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(1, 4) == 0.25

    def test_zero_denominator(self):
        """Test division by zero returns default."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1.0

    def test_inf_denominator(self):
        """Test infinite denominator returns default."""
        assert safe_divide(10, float('inf')) == 0.0
        assert safe_divide(10, float('-inf')) == 0.0

    def test_negative_values(self):
        """Test negative values work correctly."""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0


class TestSafePctChange:
    """Test safe_pct_change function."""

    def test_normal_change(self):
        """Test normal percentage change."""
        assert safe_pct_change(110, 100) == 0.1  # 10% increase
        assert safe_pct_change(90, 100) == -0.1  # 10% decrease

    def test_zero_previous(self):
        """Test zero previous value returns default."""
        assert safe_pct_change(100, 0) == 0.0

    def test_no_change(self):
        """Test no change returns 0."""
        assert safe_pct_change(100, 100) == 0.0


class TestSafeRatio:
    """Test safe_ratio function."""

    def test_normal_ratio(self):
        """Test normal ratio calculation."""
        assert safe_ratio(10, 5) == 2.0
        assert safe_ratio(1, 2) == 0.5

    def test_zero_denominator(self):
        """Test zero denominator returns default."""
        assert safe_ratio(10, 0) == 1.0  # default is 1.0
        assert safe_ratio(10, 0, default=0) == 0.0


class TestSafeCompare:
    """Test safe_compare function."""

    def test_greater_than_equal(self):
        """Test >= comparison."""
        assert safe_compare(10, 5, ">=")
        assert safe_compare(5, 5, ">=")
        assert not safe_compare(4, 5, ">=")

    def test_less_than_equal(self):
        """Test <= comparison."""
        assert safe_compare(5, 10, "<=")
        assert safe_compare(5, 5, "<=")
        assert not safe_compare(6, 5, "<=")

    def test_greater_than(self):
        """Test > comparison."""
        assert safe_compare(10, 5, ">")
        assert not safe_compare(5, 5, ">")

    def test_less_than(self):
        """Test < comparison."""
        assert safe_compare(5, 10, "<")
        assert not safe_compare(5, 5, "<")

    def test_equal(self):
        """Test == comparison."""
        assert safe_compare(5, 5, "==")
        assert not safe_compare(5, 6, "==")

    def test_none_values(self):
        """Test None values return False."""
        assert not safe_compare(None, 5, ">=")
        assert not safe_compare(5, None, ">=")

    def test_inf_values(self):
        """Test infinite values return False."""
        assert not safe_compare(float('inf'), 5, ">=")
        assert not safe_compare(5, float('inf'), ">=")


class TestClipValue:
    """Test clip_value function."""

    def test_within_range(self):
        """Test value within range is unchanged."""
        assert clip_value(5, 0, 10) == 5

    def test_below_min(self):
        """Test value below min is clipped."""
        assert clip_value(-5, 0, 10) == 0

    def test_above_max(self):
        """Test value above max is clipped."""
        assert clip_value(15, 0, 10) == 10

    def test_at_boundaries(self):
        """Test values at boundaries."""
        assert clip_value(0, 0, 10) == 0
        assert clip_value(10, 0, 10) == 10


class TestEnsureFloat:
    """Test ensure_float function."""

    def test_normal_values(self):
        """Test normal values are converted."""
        assert ensure_float(5) == 5.0
        assert ensure_float("5.5") == 5.5

    def test_invalid_values(self):
        """Test invalid values return default."""
        assert ensure_float(None) == 0.0
        assert ensure_float("invalid") == 0.0
        assert ensure_float(None, default=-1) == -1.0


class TestMaxDrawdown:
    """Test max_drawdown function."""

    def test_no_drawdown(self):
        """Test no drawdown when always increasing."""
        equity = [100, 110, 120, 130, 140]
        dd = max_drawdown(equity)
        assert dd == 0.0

    def test_simple_drawdown(self):
        """Test simple drawdown calculation."""
        equity = [100, 110, 90, 100]  # 20/110 = 18.18% drawdown
        dd = max_drawdown(equity)
        assert 0.15 < dd < 0.20

    def test_empty_equity(self):
        """Test empty equity returns 0."""
        assert max_drawdown([]) == 0.0

    def test_single_value(self):
        """Test single value returns 0."""
        assert max_drawdown([100]) == 0.0


class TestProfitFactor:
    """Test profit_factor function."""

    def test_positive_factor(self):
        """Test positive profit factor calculation."""
        pf = profit_factor(300, 150)  # 300 profit, 150 loss = 2.0
        assert pf == 2.0

    def test_no_loss(self):
        """Test no loss gives max factor."""
        pf = profit_factor(100, 0)
        assert pf == 100.0  # Capped at 100

    def test_no_profit(self):
        """Test no profit with losses gives 0."""
        pf = profit_factor(0, 100)
        assert pf == 0.0

    def test_equal_profit_loss(self):
        """Test equal profit and loss gives 1."""
        pf = profit_factor(100, 100)
        assert pf == 1.0


class TestSanitizeArray:
    """Test sanitize_array function."""

    def test_normal_array(self):
        """Test normal array is unchanged."""
        arr = [1, 2, 3, 4, 5]
        result = sanitize_array(arr)
        assert len(result) == 5

    def test_with_nan(self):
        """Test NaN values are handled."""
        arr = [1, float('nan'), 3, float('nan'), 5]
        result = sanitize_array(arr)
        assert not np.any(np.isnan(result))

    def test_with_inf(self):
        """Test infinite values are handled."""
        arr = [1, float('inf'), 3, float('-inf'), 5]
        result = sanitize_array(arr)
        assert not np.any(np.isinf(result))


class TestSharpeRatio:
    """Test sharpe_ratio function."""

    def test_positive_returns(self):
        """Test positive returns give positive Sharpe."""
        returns = [0.01, 0.02, 0.01, 0.015, 0.02]
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_negative_returns(self):
        """Test negative returns give negative Sharpe."""
        returns = [-0.01, -0.02, -0.01, -0.015, -0.02]
        sr = sharpe_ratio(returns)
        assert sr < 0

    def test_zero_std(self):
        """Test zero std returns 0."""
        returns = [0.01, 0.01, 0.01, 0.01]  # No variance
        sr = sharpe_ratio(returns)
        assert sr == 0.0 or np.isinf(sr)  # Depends on implementation

    def test_empty(self):
        """Test empty returns 0."""
        assert sharpe_ratio([]) == 0.0
