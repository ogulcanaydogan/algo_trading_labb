"""
Tests for order validation module.
"""

import pytest
from bot.order_validator import (
    ValidationError,
    ValidationResult,
    OrderValidation,
    OrderLimits,
    OrderValidator,
    get_validator,
    validate_order,
)


class TestValidationResult:
    """Test ValidationResult enum."""

    def test_all_results_exist(self):
        """Test all validation results exist."""
        assert ValidationResult.VALID.value == "valid"
        assert ValidationResult.INVALID_QUANTITY.value == "invalid_quantity"
        assert ValidationResult.INVALID_PRICE.value == "invalid_price"
        assert ValidationResult.EXCEEDS_POSITION_LIMIT.value == "exceeds_position_limit"
        assert ValidationResult.EXCEEDS_RISK_LIMIT.value == "exceeds_risk_limit"
        assert ValidationResult.INSUFFICIENT_BALANCE.value == "insufficient_balance"
        assert ValidationResult.INVALID_SYMBOL.value == "invalid_symbol"
        assert ValidationResult.STALE_DATA.value == "stale_data"


class TestOrderValidation:
    """Test OrderValidation dataclass."""

    def test_valid_order(self):
        """Test valid order validation result."""
        validation = OrderValidation(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Order validated",
        )
        assert validation.is_valid
        assert validation.result == ValidationResult.VALID
        assert validation.warnings == []

    def test_invalid_order(self):
        """Test invalid order validation result."""
        validation = OrderValidation(
            is_valid=False,
            result=ValidationResult.INVALID_QUANTITY,
            message="Quantity too small",
        )
        assert not validation.is_valid
        assert validation.result == ValidationResult.INVALID_QUANTITY

    def test_with_adjusted_quantity(self):
        """Test validation with adjusted quantity."""
        validation = OrderValidation(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Adjusted",
            adjusted_quantity=0.5,
        )
        assert validation.adjusted_quantity == 0.5

    def test_with_warnings(self):
        """Test validation with warnings."""
        validation = OrderValidation(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Valid",
            warnings=["Large order", "High slippage"],
        )
        assert len(validation.warnings) == 2


class TestOrderLimits:
    """Test OrderLimits dataclass."""

    def test_default_limits(self):
        """Test default limit values."""
        limits = OrderLimits()
        assert limits.min_quantity == 0.0001
        assert limits.max_quantity == 1_000_000
        assert limits.min_price == 0.0001
        assert limits.max_price == 1_000_000_000
        assert limits.max_position_value == 100_000
        assert limits.max_single_order_pct == 0.25
        assert limits.max_slippage_pct == 0.05

    def test_custom_limits(self):
        """Test custom limit values."""
        limits = OrderLimits(
            min_quantity=0.01,
            max_quantity=100,
            max_single_order_pct=0.1,
        )
        assert limits.min_quantity == 0.01
        assert limits.max_quantity == 100
        assert limits.max_single_order_pct == 0.1


class TestOrderValidator:
    """Test OrderValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OrderValidator()

    @pytest.fixture
    def custom_validator(self):
        """Create validator with custom limits."""
        limits = OrderLimits(
            min_quantity=0.001,
            max_quantity=1000,
            max_single_order_pct=0.5,
        )
        return OrderValidator(limits)

    def test_valid_buy_order(self, validator):
        """Test valid BUY order."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=50000.0,
        )
        assert result.is_valid
        assert result.result == ValidationResult.VALID

    def test_valid_sell_order(self, validator):
        """Test valid SELL order."""
        result = validator.validate_order(
            symbol="ETH/USDT",
            action="SELL",
            quantity=10.0,
            price=3000.0,
        )
        assert result.is_valid
        assert result.result == ValidationResult.VALID

    def test_lowercase_action(self, validator):
        """Test lowercase action is handled."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="buy",
            quantity=1.0,
            price=50000.0,
        )
        assert result.is_valid

    def test_invalid_symbol_empty(self, validator):
        """Test empty symbol is rejected."""
        result = validator.validate_order(
            symbol="",
            action="BUY",
            quantity=1.0,
            price=100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_SYMBOL

    def test_invalid_symbol_none(self, validator):
        """Test None symbol is rejected."""
        result = validator.validate_order(
            symbol=None,
            action="BUY",
            quantity=1.0,
            price=100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_SYMBOL

    def test_invalid_action(self, validator):
        """Test invalid action is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="HOLD",
            quantity=1.0,
            price=100.0,
        )
        assert not result.is_valid

    def test_negative_quantity(self, validator):
        """Test negative quantity is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=-1.0,
            price=100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_QUANTITY

    def test_zero_quantity(self, validator):
        """Test zero quantity is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=0,
            price=100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_QUANTITY

    def test_quantity_below_minimum(self, validator):
        """Test quantity below minimum is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.00001,  # Below default min 0.0001
            price=100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_QUANTITY

    def test_quantity_above_maximum(self, validator):
        """Test quantity above maximum is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=10_000_000,  # Above default max 1,000,000
            price=1.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_QUANTITY

    def test_negative_price(self, validator):
        """Test negative price is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=-100.0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_PRICE

    def test_zero_price(self, validator):
        """Test zero price is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=0,
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_PRICE

    def test_price_above_maximum(self, validator):
        """Test price above maximum is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=10_000_000_000,  # Above max
        )
        assert not result.is_valid
        assert result.result == ValidationResult.INVALID_PRICE

    def test_slippage_warning(self, validator):
        """Test slippage parameter is handled."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=52000.0,
            expected_price=50000.0,  # 4% slippage
        )
        # Should handle slippage gracefully - may or may not generate warning
        assert result.is_valid or result.result in [
            ValidationResult.INVALID_PRICE,
            ValidationResult.EXCEEDS_RISK_LIMIT,
        ]

    def test_exceeds_portfolio_percentage(self, validator):
        """Test order exceeding portfolio percentage is rejected."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=50000.0,  # $50,000 order
            portfolio_value=100000.0,  # $100,000 portfolio = 50%
        )
        assert not result.is_valid
        assert result.result == ValidationResult.EXCEEDS_RISK_LIMIT

    def test_insufficient_balance(self, validator):
        """Test insufficient balance is handled."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=10.0,
            price=50000.0,  # $500,000 order
            cash_balance=100000.0,  # Only $100,000 available
        )
        # Should either fail or adjust quantity
        if result.is_valid:
            assert result.adjusted_quantity is not None
            assert result.adjusted_quantity < 10.0
        else:
            assert result.result == ValidationResult.INSUFFICIENT_BALANCE

    def test_sell_exceeds_position(self, validator):
        """Test sell order exceeding position is adjusted."""
        result = validator.validate_order(
            symbol="BTC/USDT",
            action="SELL",
            quantity=10.0,
            price=50000.0,
            current_position=5.0,  # Only have 5
        )
        assert result.is_valid
        assert result.adjusted_quantity == 5.0

    def test_custom_limits(self, custom_validator):
        """Test custom limits are applied."""
        # Should pass with custom higher max percentage
        result = custom_validator.validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=40000.0,  # $40,000 = 40% of $100,000
            portfolio_value=100000.0,
        )
        assert result.is_valid


class TestBatchValidation:
    """Test batch order validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OrderValidator()

    def test_validate_batch(self, validator):
        """Test validating multiple orders."""
        orders = [
            {"symbol": "BTC/USDT", "action": "BUY", "quantity": 0.1, "price": 50000},
            {"symbol": "ETH/USDT", "action": "BUY", "quantity": 1.0, "price": 3000},
        ]
        results = validator.validate_batch(orders)
        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_batch_with_invalid(self, validator):
        """Test batch with some invalid orders."""
        orders = [
            {"symbol": "BTC/USDT", "action": "BUY", "quantity": 1.0, "price": 50000},
            {"symbol": "", "action": "BUY", "quantity": 1.0, "price": 100},  # Invalid
        ]
        results = validator.validate_batch(orders)
        assert len(results) == 2
        assert results[0].is_valid
        assert not results[1].is_valid

    def test_batch_tracks_balance(self, validator):
        """Test batch validation tracks remaining balance."""
        orders = [
            {"symbol": "BTC/USDT", "action": "BUY", "quantity": 0.01, "price": 50000},  # $500
            {"symbol": "ETH/USDT", "action": "BUY", "quantity": 0.1, "price": 3000},  # $300
        ]
        results = validator.validate_batch(orders, cash_balance=1000)
        assert len(results) == 2
        # Both should fit in $1000 balance
        assert results[0].is_valid
        assert results[1].is_valid


class TestGlobalValidator:
    """Test global validator functions."""

    def test_get_validator(self):
        """Test getting global validator."""
        import bot.order_validator as ov

        ov._validator = None  # Reset

        validator = get_validator()
        assert validator is not None
        assert isinstance(validator, OrderValidator)

    def test_get_validator_returns_same(self):
        """Test get_validator returns same instance."""
        import bot.order_validator as ov

        ov._validator = None  # Reset

        v1 = get_validator()
        v2 = get_validator()
        assert v1 is v2

    def test_validate_order_convenience(self):
        """Test validate_order convenience function."""
        result = validate_order(
            symbol="BTC/USDT",
            action="BUY",
            quantity=1.0,
            price=50000.0,
        )
        assert isinstance(result, OrderValidation)
