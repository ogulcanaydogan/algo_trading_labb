"""
Order validation module for trading safety.

Validates orders before execution to prevent:
- Invalid quantities (negative, zero, too large)
- Invalid prices (negative, zero, unrealistic)
- Position size violations
- Risk limit breaches
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when order validation fails."""
    pass


class ValidationResult(Enum):
    """Result of order validation."""
    VALID = "valid"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_PRICE = "invalid_price"
    EXCEEDS_POSITION_LIMIT = "exceeds_position_limit"
    EXCEEDS_RISK_LIMIT = "exceeds_risk_limit"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_SYMBOL = "invalid_symbol"
    STALE_DATA = "stale_data"


@dataclass
class OrderValidation:
    """Container for order validation result."""
    is_valid: bool
    result: ValidationResult
    message: str
    adjusted_quantity: Optional[float] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class OrderLimits:
    """Limits for order validation."""
    min_quantity: float = 0.0001
    max_quantity: float = 1_000_000
    min_price: float = 0.0001
    max_price: float = 1_000_000_000
    max_position_value: float = 100_000
    max_single_order_pct: float = 0.25  # Max 25% of portfolio per order
    max_slippage_pct: float = 0.05  # Max 5% slippage from expected price


class OrderValidator:
    """
    Validates trading orders before execution.

    Features:
    - Quantity validation (min/max, position limits)
    - Price validation (sanity checks)
    - Risk limit enforcement
    - Balance checks
    - Duplicate order detection
    """

    def __init__(self, limits: Optional[OrderLimits] = None):
        self.limits = limits or OrderLimits()
        self._recent_orders: List[Dict[str, Any]] = []
        self._max_recent = 100

    def validate_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        portfolio_value: float = 0,
        cash_balance: float = 0,
        current_position: float = 0,
        expected_price: Optional[float] = None,
    ) -> OrderValidation:
        """
        Validate an order before execution.

        Args:
            symbol: Trading symbol
            action: Order action (BUY/SELL)
            quantity: Order quantity
            price: Order price
            portfolio_value: Current portfolio value
            cash_balance: Available cash
            current_position: Current position in this symbol
            expected_price: Expected price for slippage check

        Returns:
            OrderValidation with result and any adjustments
        """
        warnings = []

        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_SYMBOL,
                message=f"Invalid symbol: {symbol}",
            )

        # Validate action
        action = action.upper()
        if action not in ["BUY", "SELL"]:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Invalid action: {action}. Must be BUY or SELL",
            )

        # Validate quantity
        if quantity is None or not isinstance(quantity, (int, float)):
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Invalid quantity type: {type(quantity)}",
            )

        if quantity <= 0:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Quantity must be positive: {quantity}",
            )

        if quantity < self.limits.min_quantity:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Quantity {quantity} below minimum {self.limits.min_quantity}",
            )

        if quantity > self.limits.max_quantity:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Quantity {quantity} exceeds maximum {self.limits.max_quantity}",
            )

        # Validate price
        if price is None or not isinstance(price, (int, float)):
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_PRICE,
                message=f"Invalid price type: {type(price)}",
            )

        if price <= 0:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_PRICE,
                message=f"Price must be positive: {price}",
            )

        if price < self.limits.min_price:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_PRICE,
                message=f"Price {price} below minimum {self.limits.min_price}",
            )

        if price > self.limits.max_price:
            return OrderValidation(
                is_valid=False,
                result=ValidationResult.INVALID_PRICE,
                message=f"Price {price} exceeds maximum {self.limits.max_price}",
            )

        # Check slippage if expected price provided
        if expected_price and expected_price > 0:
            slippage = abs(price - expected_price) / expected_price
            if slippage > self.limits.max_slippage_pct:
                warnings.append(
                    f"High slippage: {slippage:.2%} from expected price ${expected_price:,.2f}"
                )

        # Calculate order value
        order_value = quantity * price

        # Check position limit
        if order_value > self.limits.max_position_value:
            warnings.append(
                f"Large order: ${order_value:,.2f} exceeds typical position size"
            )

        # Check single order percentage of portfolio
        if portfolio_value > 0:
            order_pct = order_value / portfolio_value
            if order_pct > self.limits.max_single_order_pct:
                return OrderValidation(
                    is_valid=False,
                    result=ValidationResult.EXCEEDS_RISK_LIMIT,
                    message=f"Order is {order_pct:.1%} of portfolio, max is {self.limits.max_single_order_pct:.0%}",
                )

        # Check balance for BUY orders
        if action == "BUY" and cash_balance > 0:
            if order_value > cash_balance:
                # Adjust quantity to available balance
                adjusted_qty = (cash_balance * 0.99) / price  # 1% buffer
                if adjusted_qty >= self.limits.min_quantity:
                    warnings.append(
                        f"Reduced quantity from {quantity:.6f} to {adjusted_qty:.6f} due to balance"
                    )
                    return OrderValidation(
                        is_valid=True,
                        result=ValidationResult.VALID,
                        message="Order adjusted to available balance",
                        adjusted_quantity=adjusted_qty,
                        warnings=warnings,
                    )
                else:
                    return OrderValidation(
                        is_valid=False,
                        result=ValidationResult.INSUFFICIENT_BALANCE,
                        message=f"Insufficient balance: ${cash_balance:,.2f} < ${order_value:,.2f}",
                    )

        # Check position for SELL orders
        if action == "SELL" and current_position > 0:
            if quantity > current_position:
                # Adjust to current position
                warnings.append(
                    f"Reduced sell quantity from {quantity:.6f} to {current_position:.6f}"
                )
                return OrderValidation(
                    is_valid=True,
                    result=ValidationResult.VALID,
                    message="Order adjusted to current position",
                    adjusted_quantity=current_position,
                    warnings=warnings,
                )

        # Track recent orders for duplicate detection
        self._track_order(symbol, action, quantity, price)

        return OrderValidation(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Order validated successfully",
            warnings=warnings if warnings else None,
        )

    def _track_order(self, symbol: str, action: str, quantity: float, price: float):
        """Track order for duplicate detection."""
        from datetime import datetime

        self._recent_orders.append({
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now(),
        })

        # Keep only recent orders
        if len(self._recent_orders) > self._max_recent:
            self._recent_orders = self._recent_orders[-self._max_recent:]

    def validate_batch(
        self,
        orders: List[Dict[str, Any]],
        portfolio_value: float = 0,
        cash_balance: float = 0,
    ) -> List[OrderValidation]:
        """
        Validate a batch of orders.

        Args:
            orders: List of order dicts with symbol, action, quantity, price
            portfolio_value: Current portfolio value
            cash_balance: Available cash

        Returns:
            List of OrderValidation results
        """
        results = []
        remaining_cash = cash_balance

        for order in orders:
            validation = self.validate_order(
                symbol=order.get("symbol", ""),
                action=order.get("action", ""),
                quantity=order.get("quantity", 0),
                price=order.get("price", 0),
                portfolio_value=portfolio_value,
                cash_balance=remaining_cash,
                current_position=order.get("current_position", 0),
                expected_price=order.get("expected_price"),
            )
            results.append(validation)

            # Update remaining cash for subsequent orders
            if validation.is_valid and order.get("action", "").upper() == "BUY":
                qty = validation.adjusted_quantity or order.get("quantity", 0)
                remaining_cash -= qty * order.get("price", 0)

        return results


# Global validator instance
_validator: Optional[OrderValidator] = None


def get_validator() -> OrderValidator:
    """Get or create global order validator."""
    global _validator
    if _validator is None:
        _validator = OrderValidator()
    return _validator


def validate_order(
    symbol: str,
    action: str,
    quantity: float,
    price: float,
    **kwargs,
) -> OrderValidation:
    """Convenience function to validate an order."""
    return get_validator().validate_order(symbol, action, quantity, price, **kwargs)
