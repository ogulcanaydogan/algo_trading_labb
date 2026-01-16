"""
Tests for market data module.
"""

import pytest
import pandas as pd
import numpy as np

from bot.market_data import MarketDataError, sanitize_symbol_for_fs


class TestMarketDataError:
    """Test MarketDataError exception."""

    def test_error_creation(self):
        """Test creating a MarketDataError."""
        error = MarketDataError(message="Test error message")
        assert error.message == "Test error message"

    def test_error_str(self):
        """Test error string representation."""
        error = MarketDataError(message="Failed to fetch data")
        assert str(error) == "Failed to fetch data"

    def test_error_is_runtime_error(self):
        """Test error inherits from RuntimeError."""
        error = MarketDataError(message="Test")
        assert isinstance(error, RuntimeError)


class TestSanitizeSymbolForFs:
    """Test sanitize_symbol_for_fs function."""

    def test_simple_symbol(self):
        """Test sanitizing simple symbol."""
        result = sanitize_symbol_for_fs("BTCUSDT")
        assert "/" not in result
        assert "\\" not in result
        assert ":" not in result

    def test_symbol_with_slash(self):
        """Test sanitizing symbol with slash."""
        result = sanitize_symbol_for_fs("BTC/USDT")
        assert "/" not in result
        # Common replacement is underscore
        assert "_" in result or "-" in result or result == "BTCUSDT"

    def test_symbol_with_special_chars(self):
        """Test sanitizing symbol with special characters."""
        result = sanitize_symbol_for_fs("BTC:USDT")
        assert ":" not in result

    def test_lowercase_conversion(self):
        """Test symbol case handling."""
        result = sanitize_symbol_for_fs("btc/usdt")
        # Result should be filesystem-safe
        assert isinstance(result, str)

    def test_empty_symbol(self):
        """Test empty symbol."""
        result = sanitize_symbol_for_fs("")
        assert result == ""

    def test_already_safe_symbol(self):
        """Test already safe symbol."""
        result = sanitize_symbol_for_fs("AAPL")
        assert result == "AAPL"

    def test_multiple_special_chars(self):
        """Test symbol with multiple special characters."""
        result = sanitize_symbol_for_fs("ETH/BTC:PERP")
        assert "/" not in result
        assert ":" not in result
