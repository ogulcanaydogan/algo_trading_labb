"""Tests for data validation module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from bot.core.validation import (
    DataQuality,
    OHLCVValidator,
    OrderValidator,
    PositionValidator,
    ValidationError,
    ValidationResult,
    validate_ohlcv,
    validate_order,
    validate_position,
)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_default_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.quality == DataQuality.GOOD
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")

        assert result.is_valid is False
        assert result.quality == DataQuality.INVALID
        assert "Test error" in result.errors

    def test_add_warning(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert result.quality == DataQuality.DEGRADED
        assert "Test warning" in result.warnings

    def test_to_dict(self):
        result = ValidationResult(is_valid=True, stats={"rows": 100})
        data = result.to_dict()

        assert data["is_valid"] is True
        assert data["quality"] == "good"
        assert data["stats"]["rows"] == 100


class TestOHLCVValidator:
    """Tests for OHLCV validation."""

    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV DataFrame."""
        n = 100
        dates = pd.date_range(end=datetime.now(), periods=n, freq="1h")
        np.random.seed(42)

        price = 50000
        prices = [price]
        for _ in range(n - 1):
            price = price * (1 + np.random.randn() * 0.01)
            prices.append(price)

        prices = np.array(prices)

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.001),
                "high": prices * (1 + abs(np.random.randn(n)) * 0.005),
                "low": prices * (1 - abs(np.random.randn(n)) * 0.005),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, n),
            },
            index=dates,
        )

    def test_valid_ohlcv(self, valid_ohlcv):
        result = validate_ohlcv(valid_ohlcv)
        assert result.is_valid is True
        assert result.stats["rows"] == 100

    def test_empty_dataframe(self):
        result = validate_ohlcv(pd.DataFrame())
        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_none_dataframe(self):
        result = validate_ohlcv(None)
        assert result.is_valid is False

    def test_missing_columns(self):
        df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
        result = validate_ohlcv(df)
        assert result.is_valid is False
        assert "Missing" in result.errors[0]

    def test_insufficient_rows(self):
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9],
                "close": [1, 2, 3],
                "volume": [100, 100, 100],
            }
        )
        validator = OHLCVValidator(min_rows=10)
        result = validator.validate(df)
        assert result.is_valid is False
        assert "Insufficient" in result.errors[0]

    def test_negative_prices(self):
        df = pd.DataFrame(
            {
                "open": [1, 2, -3],
                "high": [1.1, 2.1, 3.1],
                "low": [0.9, 1.9, 2.9],
                "close": [1, 2, 3],
                "volume": [100, 100, 100],
            }
        )
        validator = OHLCVValidator(min_rows=1)
        result = validator.validate(df)
        assert result.is_valid is False
        assert "Negative" in result.errors[0]

    def test_nan_warning(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.iloc[0, 0] = np.nan  # Add one NaN

        result = validate_ohlcv(df)
        assert result.is_valid is True  # Still valid with small NaN count
        assert len(result.warnings) > 0 or result.stats["nan_count"] > 0

    def test_raise_on_error(self):
        with pytest.raises(ValidationError):
            validate_ohlcv(pd.DataFrame(), raise_on_error=True)


class TestOrderValidator:
    """Tests for order validation."""

    def test_valid_market_order(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
        )
        assert result.is_valid is True

    def test_valid_limit_order(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="sell",
            quantity=0.5,
            price=50000,
            order_type="limit",
        )
        assert result.is_valid is True

    def test_invalid_side(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="invalid",
            quantity=0.1,
        )
        assert result.is_valid is False
        assert "side" in result.errors[0].lower()

    def test_invalid_quantity_negative(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=-0.1,
        )
        assert result.is_valid is False
        assert "positive" in result.errors[0].lower()

    def test_invalid_quantity_zero(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0,
        )
        assert result.is_valid is False

    def test_quantity_below_minimum(self):
        validator = OrderValidator(min_quantity=0.01)
        result = validator.validate(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.001,
        )
        assert result.is_valid is False
        assert "minimum" in result.errors[0].lower()

    def test_quantity_above_maximum(self):
        validator = OrderValidator(max_quantity=100)
        result = validator.validate(
            symbol="BTC/USDT",
            side="buy",
            quantity=1000,
        )
        assert result.is_valid is False
        assert "maximum" in result.errors[0].lower()

    def test_limit_order_missing_price(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            order_type="limit",
        )
        assert result.is_valid is False
        assert "price" in result.errors[0].lower()

    def test_stop_order_missing_stop_price(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            order_type="stop",
        )
        assert result.is_valid is False
        assert "stop" in result.errors[0].lower()

    def test_slippage_warning(self):
        result = validate_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=55000,
            order_type="limit",
            current_price=50000,
        )
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "deviation" in result.warnings[0].lower()

    def test_raise_on_error(self):
        with pytest.raises(ValidationError):
            validate_order(
                symbol="BTC/USDT",
                side="invalid",
                quantity=0.1,
                raise_on_error=True,
            )


class TestPositionValidator:
    """Tests for position validation."""

    def test_valid_position(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=1000,
            portfolio_value=10000,
        )
        assert result.is_valid is True

    def test_position_too_large(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=5000,
            portfolio_value=10000,
        )
        assert result.is_valid is False
        assert "exceeds" in result.errors[0].lower()

    def test_leverage_too_high(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=1000,
            portfolio_value=10000,
            leverage=20.0,
        )
        assert result.is_valid is False
        assert "leverage" in result.errors[0].lower()

    def test_too_many_positions(self):
        current = {f"SYM{i}/USDT": 100 for i in range(25)}

        result = validate_position(
            symbol="NEW/USDT",
            size=100,
            portfolio_value=10000,
            current_positions=current,
        )
        assert result.is_valid is False
        assert "Too many" in result.errors[0]

    def test_high_leverage_warning(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=1000,
            portfolio_value=10000,
            leverage=9.0,  # 90% of max
        )
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "High leverage" in result.warnings[0]

    def test_stats_calculated(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=2000,
            portfolio_value=10000,
            current_positions={"ETH/USDT": 1000},
            leverage=2.0,
        )

        assert "position_pct" in result.stats
        assert result.stats["position_pct"] == 0.2
        assert result.stats["leverage"] == 2.0
        assert result.stats["num_positions"] == 2

    def test_zero_portfolio_value(self):
        result = validate_position(
            symbol="BTC/USDT",
            size=1000,
            portfolio_value=0,
        )
        assert result.is_valid is False

    def test_raise_on_error(self):
        with pytest.raises(ValidationError):
            validate_position(
                symbol="BTC/USDT",
                size=5000,
                portfolio_value=10000,
                raise_on_error=True,
            )


class TestValidationError:
    """Tests for ValidationError."""

    def test_basic_error(self):
        error = ValidationError("Something went wrong")
        assert "Something went wrong" in str(error)

    def test_error_with_field(self):
        error = ValidationError("Invalid value", field="quantity")
        assert "quantity" in str(error)
        assert "Invalid value" in str(error)

    def test_error_with_value(self):
        error = ValidationError("Too small", field="price", value=0.001)
        assert error.field == "price"
        assert error.value == 0.001
