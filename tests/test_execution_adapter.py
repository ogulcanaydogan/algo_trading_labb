"""Tests for execution adapter module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from bot.execution_adapter import (
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Balance,
    PaperExecutionAdapter,
    create_execution_adapter,
)
from bot.core.exceptions import ExecutionError, NetworkError, ExchangeError


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1

    def test_order_with_price(self):
        """Test creating a limit order with price."""
        order = Order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=2500.0,
        )
        assert order.price == 2500.0
        assert order.order_type == OrderType.LIMIT


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_successful_order_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="test123",
            status=OrderStatus.FILLED,
            filled_quantity=0.1,
            average_price=50000.0,
            commission=5.0,
        )
        assert result.success is True
        assert result.order_id == "test123"
        assert result.status == OrderStatus.FILLED

    def test_failed_order_result(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            order_id="",
            status=OrderStatus.REJECTED,
            filled_quantity=0,
            average_price=0,
            error_message="Insufficient balance",
        )
        assert result.success is False
        assert result.error_message == "Insufficient balance"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_value(self):
        """Test position value calculation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=0.5,
            entry_price=40000.0,
            side="long",
        )
        assert position.value == 20000.0

    def test_short_position_value(self):
        """Test short position value calculation."""
        position = Position(
            symbol="ETH/USDT",
            quantity=-2.0,
            entry_price=2500.0,
            side="short",
        )
        assert position.value == 5000.0  # abs(quantity) * price


class TestPaperExecutionAdapter:
    """Tests for PaperExecutionAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a paper execution adapter."""
        return PaperExecutionAdapter(
            initial_balance=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test connecting paper adapter."""
        result = await adapter.connect()
        assert result is True
        assert adapter.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter):
        """Test disconnecting paper adapter."""
        await adapter.connect()
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """Test getting balance."""
        await adapter.connect()
        balance = await adapter.get_balance()
        assert balance.total == 10000.0
        assert balance.available == 10000.0
        assert balance.in_positions == 0.0

    @pytest.mark.asyncio
    async def test_place_market_order(self, adapter):
        """Test placing a market order."""
        await adapter.connect()
        adapter.set_price("BTC/USDT", 50000.0)
        adapter._prices["BTC/USDT_price"] = (50000.0, 0)  # Bypass cache

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        with patch.object(adapter, 'get_current_price', return_value=50000.0):
            result = await adapter.place_order(order)

        assert result.success is True
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.1
        assert result.simulated is True

    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(self, adapter):
        """Test placing order with insufficient balance."""
        await adapter.connect()

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,  # Way more than balance allows
        )

        with patch.object(adapter, 'get_current_price', return_value=50000.0):
            result = await adapter.place_order(order)

        assert result.success is False
        assert "Insufficient balance" in result.error_message

    @pytest.mark.asyncio
    async def test_get_position_empty(self, adapter):
        """Test getting position when none exists."""
        await adapter.connect()
        position = await adapter.get_position("BTC/USDT")
        assert position is None

    @pytest.mark.asyncio
    async def test_get_all_positions_empty(self, adapter):
        """Test getting all positions when none exist."""
        await adapter.connect()
        positions = await adapter.get_all_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_order_updates_balance(self, adapter):
        """Test that orders update balance correctly."""
        await adapter.connect()
        initial_balance = adapter._balance

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        with patch.object(adapter, 'get_current_price', return_value=50000.0):
            await adapter.place_order(order)

        # Balance should decrease by order value + commission
        assert adapter._balance < initial_balance

    @pytest.mark.asyncio
    async def test_buy_creates_position(self, adapter):
        """Test that buying creates a position."""
        await adapter.connect()

        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        with patch.object(adapter, 'get_current_price', return_value=50000.0):
            await adapter.place_order(order)
            position = await adapter.get_position("BTC/USDT")

        assert position is not None
        assert position.quantity == 0.1
        assert position.side == "long"

    @pytest.mark.asyncio
    async def test_sell_closes_position(self, adapter):
        """Test that selling closes a position."""
        await adapter.connect()

        # First buy
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        with patch.object(adapter, 'get_current_price', return_value=50000.0):
            await adapter.place_order(buy_order)

        # Then sell
        sell_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        with patch.object(adapter, 'get_current_price', return_value=51000.0):
            await adapter.place_order(sell_order)
            position = await adapter.get_position("BTC/USDT")

        assert position is None


class TestCreateExecutionAdapter:
    """Tests for create_execution_adapter factory function."""

    def test_create_paper_adapter(self):
        """Test creating paper adapter."""
        adapter = create_execution_adapter("paper_live_data", initial_balance=5000.0)
        assert isinstance(adapter, PaperExecutionAdapter)
        assert adapter.initial_balance == 5000.0

    def test_create_backtest_adapter(self):
        """Test creating backtest adapter (uses paper)."""
        adapter = create_execution_adapter("backtest")
        assert isinstance(adapter, PaperExecutionAdapter)

    def test_create_testnet_without_keys(self):
        """Test creating testnet adapter without keys falls back to paper."""
        adapter = create_execution_adapter("testnet")
        assert isinstance(adapter, PaperExecutionAdapter)

    def test_create_live_without_keys_raises(self):
        """Test creating live adapter without keys raises error."""
        with pytest.raises(ValueError, match="API keys required"):
            create_execution_adapter("live_limited")


class TestExceptionHandling:
    """Tests for exception handling in adapters."""

    def test_execution_error_creation(self):
        """Test ExecutionError creation."""
        error = ExecutionError("Order failed")
        assert str(error) == "Order failed"
        assert error.category.value == "execution"

    def test_network_error_creation(self):
        """Test NetworkError creation."""
        error = NetworkError("Connection timeout")
        assert str(error) == "Connection timeout"
        assert error.category.value == "network"

    def test_exchange_error_creation(self):
        """Test ExchangeError creation."""
        error = ExchangeError("API rate limit exceeded")
        assert str(error) == "API rate limit exceeded"
        assert error.category.value == "exchange"
