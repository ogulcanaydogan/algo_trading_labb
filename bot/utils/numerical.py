"""
Numerical utility functions for safe mathematical operations.

Provides functions to handle edge cases like:
- Division by zero
- NaN/Inf values
- Type mismatches in comparisons
"""

import numpy as np
from typing import Union, Optional

Number = Union[int, float]


def safe_divide(numerator: Number, denominator: Number, default: Number = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero or result is invalid.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division fails (default: 0.0)

    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or not np.isfinite(denominator):
            return float(default)
        result = numerator / denominator
        if not np.isfinite(result):
            return float(default)
        return float(result)
    except (TypeError, ValueError, ZeroDivisionError):
        return float(default)


def safe_pct_change(current: Number, previous: Number, default: Number = 0.0) -> float:
    """
    Safely calculate percentage change.

    Args:
        current: Current value
        previous: Previous value
        default: Value to return if calculation fails

    Returns:
        Percentage change as decimal (e.g., 0.05 for 5%)
    """
    return safe_divide(current - previous, previous, default)


def safe_ratio(value1: Number, value2: Number, default: Number = 1.0) -> float:
    """
    Safely calculate ratio between two values.

    Args:
        value1: First value (numerator)
        value2: Second value (denominator)
        default: Value to return if calculation fails

    Returns:
        Ratio of value1/value2
    """
    return safe_divide(value1, value2, default)


def safe_compare(a: any, b: any, op: str = ">=") -> bool:
    """
    Safely compare two values, returning False if comparison fails.

    Args:
        a: First value
        b: Second value
        op: Comparison operator (">=", "<=", ">", "<", "==")

    Returns:
        Result of comparison or False if types incompatible
    """
    try:
        # Convert to float if possible
        a_val = float(a) if a is not None else None
        b_val = float(b) if b is not None else None

        if a_val is None or b_val is None:
            return False
        if not np.isfinite(a_val) or not np.isfinite(b_val):
            return False

        if op == ">=":
            return a_val >= b_val
        elif op == "<=":
            return a_val <= b_val
        elif op == ">":
            return a_val > b_val
        elif op == "<":
            return a_val < b_val
        elif op == "==":
            return a_val == b_val
        else:
            return False
    except (TypeError, ValueError):
        return False


def clip_value(
    value: Number,
    min_val: Optional[Number] = None,
    max_val: Optional[Number] = None,
    default: Number = 0.0,
) -> float:
    """
    Clip a value to a range, handling invalid inputs.

    Args:
        value: Value to clip
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
        default: Value to return if input is invalid

    Returns:
        Clipped value
    """
    try:
        v = float(value)
        if not np.isfinite(v):
            return float(default)
        if min_val is not None:
            v = max(v, float(min_val))
        if max_val is not None:
            v = min(v, float(max_val))
        return v
    except (TypeError, ValueError):
        return float(default)


def sanitize_array(arr: np.ndarray, replace_val: float = 0.0) -> np.ndarray:
    """
    Replace NaN and Inf values in array.

    Args:
        arr: Input array
        replace_val: Value to replace invalid entries with

    Returns:
        Sanitized array
    """
    result = np.array(arr, dtype=float)
    result = np.nan_to_num(result, nan=replace_val, posinf=replace_val, neginf=replace_val)
    return result


def ensure_float(value: any, default: float = 0.0) -> float:
    """
    Ensure value is a valid float.

    Args:
        value: Value to convert
        default: Default if conversion fails

    Returns:
        Float value
    """
    try:
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def profit_factor(gross_profit: Number, gross_loss: Number) -> float:
    """
    Calculate profit factor safely.

    Args:
        gross_profit: Total profit from winning trades
        gross_loss: Total loss from losing trades (as positive number)

    Returns:
        Profit factor (profit/loss ratio), capped at 100
    """
    if gross_loss <= 0:
        return 100.0 if gross_profit > 0 else 1.0
    return min(safe_divide(gross_profit, gross_loss, 1.0), 100.0)


def sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio safely.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    try:
        returns = sanitize_array(returns)
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return <= 0 or not np.isfinite(std_return):
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return clip_value(sharpe, -10.0, 10.0, 0.0)
    except Exception:
        return 0.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown safely.

    Args:
        equity_curve: Array of equity values

    Returns:
        Maximum drawdown as positive decimal (e.g., 0.15 for 15%)
    """
    try:
        equity = sanitize_array(equity_curve)
        if len(equity) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1e-8)
        return clip_value(np.max(drawdown), 0.0, 1.0, 0.0)
    except Exception:
        return 0.0
