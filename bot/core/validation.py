"""
Data Validation Module.

Provides validation for market data, orders, and trading parameters
to ensure data quality and prevent invalid operations.

Usage:
    from bot.core.validation import validate_ohlcv, validate_order, ValidationError

    # Validate OHLCV data
    df = validate_ohlcv(raw_df)

    # Validate order parameters
    order = validate_order(symbol="BTC/USDT", side="buy", quantity=0.1, price=50000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.field:
            return f"Validation error in '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class DataQuality(Enum):
    """Data quality levels."""

    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    quality: DataQuality = DataQuality.GOOD
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        self.quality = DataQuality.INVALID

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        if self.quality == DataQuality.GOOD:
            self.quality = DataQuality.DEGRADED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality": self.quality.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class OHLCVValidator:
    """Validator for OHLCV market data."""

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(
        self,
        min_rows: int = 10,
        max_gap_hours: float = 24,
        max_price_change_pct: float = 50,
        allow_zero_volume: bool = True,
    ):
        self.min_rows = min_rows
        self.max_gap_hours = max_gap_hours
        self.max_price_change_pct = max_price_change_pct
        self.allow_zero_volume = allow_zero_volume

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check if DataFrame is empty
        if df is None or df.empty:
            result.add_error("DataFrame is empty or None")
            return result

        # Check required columns
        df_columns = {c.lower() for c in df.columns}
        missing = self.REQUIRED_COLUMNS - df_columns
        if missing:
            result.add_error(f"Missing required columns: {missing}")
            return result

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Check minimum rows
        if len(df) < self.min_rows:
            result.add_error(f"Insufficient data: {len(df)} rows (min: {self.min_rows})")
            return result

        # Check for NaN values
        nan_counts = df[list(self.REQUIRED_COLUMNS)].isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            nan_pct = total_nans / (len(df) * len(self.REQUIRED_COLUMNS)) * 100
            if nan_pct > 10:
                result.add_error(f"Too many NaN values: {nan_pct:.1f}%")
            else:
                result.add_warning(f"Contains {total_nans} NaN values ({nan_pct:.1f}%)")

        # Check OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            result.add_warning(f"Invalid OHLC relationships in {invalid_count} rows")

        # Check for negative values
        for col in ["open", "high", "low", "close"]:
            if (df[col] < 0).any():
                result.add_error(f"Negative values found in '{col}'")

        if not self.allow_zero_volume and (df["volume"] <= 0).any():
            result.add_warning("Zero or negative volume values found")

        # Check for extreme price changes
        if len(df) > 1:
            returns = df["close"].pct_change().abs()
            max_return = returns.max() * 100
            if max_return > self.max_price_change_pct:
                result.add_warning(
                    f"Extreme price change detected: {max_return:.1f}% "
                    f"(threshold: {self.max_price_change_pct}%)"
                )

        # Check timestamp gaps (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff()
            max_gap = gaps.max()
            if pd.notna(max_gap) and max_gap > timedelta(hours=self.max_gap_hours):
                result.add_warning(f"Large time gap detected: {max_gap}")

        # Collect stats
        result.stats = {
            "rows": len(df),
            "date_range": {
                "start": str(df.index[0]) if isinstance(df.index, pd.DatetimeIndex) else None,
                "end": str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else None,
            },
            "price_range": {
                "min": float(df["low"].min()),
                "max": float(df["high"].max()),
            },
            "nan_count": int(total_nans),
        }

        return result


class OrderValidator:
    """Validator for trading orders."""

    VALID_SIDES = {"buy", "sell"}
    VALID_ORDER_TYPES = {"market", "limit", "stop", "stop_limit"}

    def __init__(
        self,
        min_quantity: float = 0.0001,
        max_quantity: float = 1000000,
        min_price: float = 0.00001,
        max_price: float = 10000000,
        max_slippage_pct: float = 5.0,
    ):
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.min_price = min_price
        self.max_price = max_price
        self.max_slippage_pct = max_slippage_pct

    def validate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        stop_price: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate order parameters.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Limit price (for limit orders)
            order_type: Order type
            stop_price: Stop price (for stop orders)
            current_price: Current market price (for slippage check)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            result.add_error("Invalid symbol")
        elif "/" not in symbol and len(symbol) < 3:
            result.add_warning(f"Unusual symbol format: {symbol}")

        # Validate side
        side_lower = side.lower() if side else ""
        if side_lower not in self.VALID_SIDES:
            result.add_error(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        # Validate order type
        order_type_lower = order_type.lower() if order_type else ""
        if order_type_lower not in self.VALID_ORDER_TYPES:
            result.add_error(f"Invalid order type: {order_type}")

        # Validate quantity
        if quantity is None or not isinstance(quantity, (int, float)):
            result.add_error("Quantity must be a number")
        elif quantity <= 0:
            result.add_error("Quantity must be positive")
        elif quantity < self.min_quantity:
            result.add_error(f"Quantity {quantity} below minimum {self.min_quantity}")
        elif quantity > self.max_quantity:
            result.add_error(f"Quantity {quantity} above maximum {self.max_quantity}")

        # Validate price for limit orders
        if order_type_lower in ("limit", "stop_limit"):
            if price is None:
                result.add_error("Price required for limit orders")
            elif price <= 0:
                result.add_error("Price must be positive")
            elif price < self.min_price:
                result.add_error(f"Price {price} below minimum {self.min_price}")
            elif price > self.max_price:
                result.add_error(f"Price {price} above maximum {self.max_price}")

        # Validate stop price
        if order_type_lower in ("stop", "stop_limit"):
            if stop_price is None:
                result.add_error("Stop price required for stop orders")
            elif stop_price <= 0:
                result.add_error("Stop price must be positive")

        # Check for potential slippage issues
        if current_price and price:
            slippage_pct = abs(price - current_price) / current_price * 100
            if slippage_pct > self.max_slippage_pct:
                result.add_warning(
                    f"Large price deviation: {slippage_pct:.2f}% from current price"
                )

        return result


class PositionValidator:
    """Validator for position operations."""

    def __init__(
        self,
        max_position_pct: float = 0.25,
        max_leverage: float = 10.0,
        max_positions: int = 20,
    ):
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.max_positions = max_positions

    def validate_new_position(
        self,
        symbol: str,
        size: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        leverage: float = 1.0,
    ) -> ValidationResult:
        """
        Validate a new position against portfolio constraints.

        Args:
            symbol: Position symbol
            size: Position size in quote currency
            portfolio_value: Total portfolio value
            current_positions: Current positions {symbol: size}
            leverage: Leverage being used

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if portfolio_value <= 0:
            result.add_error("Portfolio value must be positive")
            return result

        # Check position size
        position_pct = size / portfolio_value
        if position_pct > self.max_position_pct:
            result.add_error(
                f"Position size {position_pct:.1%} exceeds maximum {self.max_position_pct:.1%}"
            )

        # Check leverage
        if leverage > self.max_leverage:
            result.add_error(f"Leverage {leverage}x exceeds maximum {self.max_leverage}x")
        elif leverage > self.max_leverage * 0.8:
            result.add_warning(f"High leverage: {leverage}x")

        # Check number of positions
        num_positions = len(current_positions)
        if symbol not in current_positions:
            num_positions += 1

        if num_positions > self.max_positions:
            result.add_error(f"Too many positions: {num_positions} (max: {self.max_positions})")

        # Check total exposure
        total_exposure = sum(abs(v) for v in current_positions.values()) + size
        exposure_pct = total_exposure / portfolio_value
        if exposure_pct > 1.0:
            result.add_warning(f"Total exposure {exposure_pct:.1%} exceeds 100%")

        result.stats = {
            "position_pct": position_pct,
            "leverage": leverage,
            "num_positions": num_positions,
            "total_exposure_pct": exposure_pct,
        }

        return result


# Convenience functions

_ohlcv_validator = OHLCVValidator()
_order_validator = OrderValidator()
_position_validator = PositionValidator()


def validate_ohlcv(
    df: pd.DataFrame,
    raise_on_error: bool = False,
) -> ValidationResult:
    """
    Validate OHLCV data.

    Args:
        df: OHLCV DataFrame
        raise_on_error: If True, raise ValidationError on failure

    Returns:
        ValidationResult

    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    result = _ohlcv_validator.validate(df)

    if not result.is_valid and raise_on_error:
        raise ValidationError("; ".join(result.errors))

    return result


def validate_order(
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    order_type: str = "market",
    raise_on_error: bool = False,
    **kwargs,
) -> ValidationResult:
    """
    Validate order parameters.

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        price: Order price (for limit orders)
        order_type: Order type
        raise_on_error: If True, raise ValidationError on failure
        **kwargs: Additional parameters passed to validator

    Returns:
        ValidationResult

    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    result = _order_validator.validate(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type=order_type,
        **kwargs,
    )

    if not result.is_valid and raise_on_error:
        raise ValidationError("; ".join(result.errors))

    return result


def validate_position(
    symbol: str,
    size: float,
    portfolio_value: float,
    current_positions: Optional[Dict[str, float]] = None,
    raise_on_error: bool = False,
    **kwargs,
) -> ValidationResult:
    """
    Validate position against portfolio constraints.

    Args:
        symbol: Position symbol
        size: Position size
        portfolio_value: Portfolio value
        current_positions: Current positions
        raise_on_error: If True, raise ValidationError on failure
        **kwargs: Additional parameters

    Returns:
        ValidationResult

    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    result = _position_validator.validate_new_position(
        symbol=symbol,
        size=size,
        portfolio_value=portfolio_value,
        current_positions=current_positions or {},
        **kwargs,
    )

    if not result.is_valid and raise_on_error:
        raise ValidationError("; ".join(result.errors))

    return result
