"""
Input validation utilities for trading parameters and API requests.
Provides comprehensive validation and sanitization for all trading-related inputs.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional, Union
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, validator, Field
from fastapi import HTTPException


class TradingValidationError(Exception):
    """Custom exception for trading validation failures."""

    pass


class TradeRequestValidator:
    """Validator for trading request parameters."""

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate and sanitize trading symbol."""
        if not symbol or not isinstance(symbol, str):
            raise TradingValidationError("Symbol is required and must be a string")

        symbol = symbol.strip().upper()

        # Length validation
        if len(symbol) < 1 or len(symbol) > 20:
            raise TradingValidationError("Symbol length must be between 1 and 20 characters")

        # Whitelist allowed characters (letters, numbers, slash, hyphen, underscore)
        if not re.match(r"^[A-Z0-9/\-_]+$", symbol):
            raise TradingValidationError(
                "Symbol contains invalid characters. Only letters, numbers, /, -, and _ are allowed"
            )

        # Basic format validation
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise TradingValidationError(
                    "Invalid trading pair format. Expected format: BASE/QUOTE"
                )

        return symbol

    @staticmethod
    def validate_quantity(
        quantity: Union[float, str, int], balance: float, min_quantity: float = 0.001
    ) -> float:
        """Validate trade quantity against balance and constraints."""
        try:
            # Convert to float if needed
            if isinstance(quantity, str):
                quantity = float(quantity)
            elif isinstance(quantity, int):
                quantity = float(quantity)
        except (ValueError, TypeError):
            raise TradingValidationError("Quantity must be a valid number")

        if quantity <= 0:
            raise TradingValidationError("Quantity must be positive")

        if quantity < min_quantity:
            raise TradingValidationError(f"Quantity must be at least {min_quantity}")

        # Maximum position size (95% of balance to leave room for fees)
        max_quantity = balance * 0.95
        if quantity > max_quantity:
            raise TradingValidationError(f"Quantity exceeds maximum allowed: {max_quantity:.8f}")

        # Round to appropriate precision
        return round(quantity, 8)

    @staticmethod
    def validate_price(price: Union[float, str, int]) -> float:
        """Validate price input."""
        try:
            if isinstance(price, str):
                price = float(price)
            elif isinstance(price, int):
                price = float(price)
        except (ValueError, TypeError):
            raise TradingValidationError("Price must be a valid number")

        if price <= 0:
            raise TradingValidationError("Price must be positive")

        # Check for reasonable price range (adjust as needed)
        if price > 1000000 or price < 0.000001:
            raise TradingValidationError("Price appears to be outside reasonable range")

        return round(price, 8)

    @staticmethod
    def validate_percentage(value: Union[float, str, int], field_name: str = "percentage") -> float:
        """Validate percentage values (0-100)."""
        try:
            if isinstance(value, str):
                value = float(value)
            elif isinstance(value, int):
                value = float(value)
        except (ValueError, TypeError):
            raise TradingValidationError(f"{field_name} must be a valid number")

        if value < 0 or value > 100:
            raise TradingValidationError(f"{field_name} must be between 0 and 100")

        return round(value, 2)

    @staticmethod
    def validate_order_side(side: str) -> str:
        """Validate order side."""
        if not isinstance(side, str):
            raise TradingValidationError("Order side must be a string")

        side = side.strip().upper()
        if side not in ["BUY", "SELL", "LONG", "SHORT"]:
            raise TradingValidationError("Order side must be one of: BUY, SELL, LONG, SHORT")

        # Normalize to BUY/SELL
        return "BUY" if side in ["BUY", "LONG"] else "SELL"

    @staticmethod
    def validate_order_type(order_type: str) -> str:
        """Validate order type."""
        if not isinstance(order_type, str):
            raise TradingValidationError("Order type must be a string")

        order_type = order_type.strip().upper()
        valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
        if order_type not in valid_types:
            raise TradingValidationError(f"Order type must be one of: {', '.join(valid_types)}")

        return order_type

    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """Validate timeframe string."""
        if not isinstance(timeframe, str):
            raise TradingValidationError("Timeframe must be a string")

        timeframe = timeframe.strip().lower()
        valid_timeframes = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]

        if timeframe not in valid_timeframes:
            raise TradingValidationError(f"Timeframe must be one of: {', '.join(valid_timeframes)}")

        return timeframe

    @staticmethod
    def validate_date_string(date_str: str, field_name: str = "date") -> datetime:
        """Validate date string and convert to datetime."""
        if not isinstance(date_str, str):
            raise TradingValidationError(f"{field_name} must be a string")

        try:
            # Try common date formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            raise ValueError("No matching date format found")

        except ValueError as e:
            raise TradingValidationError(f"Invalid {field_name} format: {str(e)}")


class TradeRequest(BaseModel):
    """Pydantic model for trade request validation."""

    symbol: str = Field(
        ..., min_length=1, max_length=20, description="Trading symbol (e.g., BTC/USDT)"
    )
    side: str = Field(..., description="Order side: BUY or SELL")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    order_type: str = Field(
        default="MARKET", description="Order type: MARKET, LIMIT, STOP, STOP_LIMIT"
    )
    price: Optional[float] = Field(
        None, gt=0, description="Order price (required for limit orders)"
    )
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    timeframe: Optional[str] = Field(default="1h", description="Timeframe for analysis")

    @validator("symbol")
    def validate_symbol_field(cls, v):
        return TradeRequestValidator.validate_symbol(v)

    @validator("side")
    def validate_side_field(cls, v):
        return TradeRequestValidator.validate_order_side(v)

    @validator("order_type")
    def validate_order_type_field(cls, v):
        return TradeRequestValidator.validate_order_type(v)

    @validator("quantity")
    def validate_quantity_field(cls, v):
        # Note: balance validation should be done at business logic level
        return TradeRequestValidator.validate_quantity(
            v, float("inf")
        )  # No balance limit at validation level

    @validator("price", "stop_loss", "take_profit")
    def validate_price_fields(cls, v):
        if v is not None:
            return TradeRequestValidator.validate_price(v)
        return v

    @validator("timeframe")
    def validate_timeframe_field(cls, v):
        return TradeRequestValidator.validate_timeframe(v)


class APIRequestValidator:
    """Validator for general API request parameters."""

    @staticmethod
    def validate_limit(limit: Union[int, str], default: int = 50, max_limit: int = 1000) -> int:
        """Validate and normalize limit parameter."""
        try:
            if isinstance(limit, str):
                limit = int(limit)
            elif not isinstance(limit, int):
                limit = default
        except (ValueError, TypeError):
            limit = default

        if limit < 1:
            limit = 1
        elif limit > max_limit:
            limit = max_limit

        return limit

    @staticmethod
    def validate_pagination(
        offset: Union[int, str] = 0, limit: Union[int, str] = 50
    ) -> tuple[int, int]:
        """Validate pagination parameters."""
        try:
            if isinstance(offset, str):
                offset = int(offset)
            elif not isinstance(offset, int):
                offset = 0
        except (ValueError, TypeError):
            offset = 0

        limit = APIRequestValidator.validate_limit(limit)

        return max(0, offset), limit

    @staticmethod
    def sanitize_string(value: Any, max_length: int = 255) -> str:
        """Sanitize string input."""
        if value is None:
            return ""

        if not isinstance(value, str):
            value = str(value)

        # Remove potentially harmful characters
        value = re.sub(r'[<>"\']', "", value)

        # Limit length
        value = value[:max_length].strip()

        return value


def validate_trading_request(request_data: Dict[str, Any], balance: float = 0.0) -> TradeRequest:
    """Validate a complete trading request with balance checking."""
    try:
        # Extract required fields for manual validation first
        if 'symbol' not in request_data:
            raise TradingValidationError("symbol is required")
        
        # Use individual validators first to get better error messages
        symbol = TradeRequestValidator.validate_symbol(request_data.get('symbol', ''))
        side = TradeRequestValidator.validate_order_side(request_data.get('side', ''))
        
        # Now use Pydantic for the rest
        validated_data = {
            'symbol': symbol,
            'side': side,
            'quantity': request_data.get('quantity', 0.0),
            'order_type': request_data.get('order_type', 'MARKET'),
            'price': request_data.get('price'),
            'stop_loss': request_data.get('stop_loss'),
            'take_profit': request_data.get('take_profit'),
            'timeframe': request_data.get('timeframe', '1h'),
        }
        
        trade_request = TradeRequest(**validated_data)
        
        # Additional business logic validation
        if balance > 0:
            trade_request.quantity = TradeRequestValidator.validate_quantity(
                trade_request.quantity, balance
            )
        
        # Validate price relationships
        if trade_request.price is not None:
            if trade_request.stop_loss is not None:
                if trade_request.side == 'BUY' and trade_request.stop_loss >= trade_request.price:
                    raise TradingValidationError("For BUY orders, stop loss must be below price")
                elif trade_request.side == 'SELL' and trade_request.stop_loss <= trade_request.price:
                    raise TradingValidationError("For SELL orders, stop loss must be above price")
            
            if trade_request.take_profit is not None:
                if trade_request.side == 'BUY' and trade_request.take_profit <= trade_request.price:
                    raise TradingValidationError("For BUY orders, take profit must be above price")
                elif trade_request.side == 'SELL' and trade_request.take_profit >= trade_request.price:
                    raise TradingValidationError("For SELL orders, take profit must be below price")
        
        return trade_request
        
    except TradingValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


# Middleware for FastAPI to validate all requests
class ValidationMiddleware:
    """Middleware to add validation to FastAPI requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add validation logic here if needed
            pass

        await self.app(scope, receive, send)
