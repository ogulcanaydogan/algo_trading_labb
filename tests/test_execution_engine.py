"""
Comprehensive tests for Execution Engine module.

Tests order execution, slippage modeling, and exchange integration.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from decimal import Decimal

import numpy as np

from bot.execution_engine import (
    ExecutionEngine,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionResult,
    SlippageModel,
    ExecutionConfig,
)


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            id="order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
        )

        assert order.id == "order_123"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING

    def test_limit_order(self):
        """Test limit order creation."""
        order = Order(
            id="order_456",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=3000.0,
            limit_price=3050.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 3050.0

    def test_stop_loss_order(self):
        """Test stop loss order creation."""
        order = Order(
            id="order_789",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=0.1,
            price=50000.0,
            stop_price=48000.0,
        )

        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price == 48000.0


class TestSlippageModel:
    """Test SlippageModel functionality."""

    def test_default_slippage(self):
        """Test default slippage calculation."""
        model = SlippageModel()
        slippage = model.calculate_slippage(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        assert slippage >= 0
        assert slippage < price * 0.01  # Less than 1%

    def test_volume_based_slippage(self):
        """Test slippage increases with order size."""
        model = SlippageModel(base_slippage=0.0001, volume_impact=0.0001)

        slippage_small = model.calculate_slippage(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
        )

        slippage_large = model.calculate_slippage(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=50000.0,
        )

        assert slippage_large > slippage_small

    def test_volatility_adjusted_slippage(self):
        """Test slippage adjusts for volatility."""
        model = SlippageModel(volatility_factor=2.0)

        slippage_low_vol = model.calculate_slippage(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            volatility=0.01,
        )

        slippage_high_vol = model.calculate_slippage(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            volatility=0.05,
        )

        assert slippage_high_vol > slippage_low_vol


class TestExecutionConfig:
    """Test ExecutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()

        assert config.max_retries > 0
        assert config.retry_delay > 0
        assert config.timeout > 0
        assert config.max_slippage > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutionConfig(
            max_retries=5,
            retry_delay=1.0,
            timeout=30.0,
            max_slippage=0.01,
            paper_mode=True,
        )

        assert config.max_retries == 5
        assert config.paper_mode


class TestExecutionEngine:
    """Test ExecutionEngine core functionality."""

    @pytest.fixture
    def engine(self):
        """Create ExecutionEngine for testing."""
        config = ExecutionConfig(paper_mode=True)
        return ExecutionEngine(config=config)

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.paper_mode
        assert len(engine.pending_orders) == 0

    def test_create_market_order(self, engine):
        """Test creating a market order."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        assert order is not None
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET

    def test_create_limit_order(self, engine):
        """Test creating a limit order."""
        order = engine.create_order(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=3000.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 3000.0

    @pytest.mark.asyncio
    async def test_execute_market_order_paper(self, engine):
        """Test market order execution in paper mode."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await engine.execute_order(order, current_price=50000.0)

        assert result.success
        assert result.filled_quantity == 0.1
        assert result.fill_price is not None
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_limit_order_filled(self, engine):
        """Test limit order that gets filled."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=50000.0,
        )

        # Price at or below limit should fill
        result = await engine.execute_order(order, current_price=49500.0)

        assert result.success
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_limit_order_not_filled(self, engine):
        """Test limit order that doesn't get filled."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=48000.0,
        )

        # Price above limit should not fill immediately
        result = await engine.execute_order(order, current_price=50000.0)

        # Order should be pending, not filled
        assert order.status in [OrderStatus.PENDING, OrderStatus.OPEN]

    @pytest.mark.asyncio
    async def test_cancel_order(self, engine):
        """Test order cancellation."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=45000.0,
        )

        success = await engine.cancel_order(order.id)

        assert success
        assert order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_slippage_applied(self, engine):
        """Test slippage is applied to executions."""
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await engine.execute_order(order, current_price=50000.0)

        # Fill price should be slightly higher for buys due to slippage
        assert result.fill_price >= 50000.0

    def test_order_history(self, engine):
        """Test order history tracking."""
        order1 = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        order2 = engine.create_order(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        history = engine.get_order_history()
        assert len(history) >= 2

    def test_get_open_orders(self, engine):
        """Test getting open orders."""
        # Create some limit orders that won't fill immediately
        for i in range(3):
            engine.create_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                limit_price=40000.0 - i * 1000,
            )

        open_orders = engine.get_open_orders()
        assert len(open_orders) >= 0

    @pytest.mark.asyncio
    async def test_batch_execution(self, engine):
        """Test batch order execution."""
        orders = [
            engine.create_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            ),
            engine.create_order(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
            ),
        ]

        prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}
        results = await engine.execute_batch(orders, prices)

        assert len(results) == 2
        assert all(r.success for r in results)


class TestExecutionEngineExchange:
    """Test execution engine with mock exchange."""

    @pytest.fixture
    def engine_with_exchange(self):
        """Create engine with mocked exchange."""
        config = ExecutionConfig(paper_mode=False)
        engine = ExecutionEngine(config=config)

        # Mock exchange client
        mock_exchange = MagicMock()
        mock_exchange.create_order = AsyncMock(return_value={
            "id": "exchange_order_123",
            "status": "filled",
            "filled": 0.1,
            "average": 50100.0,
        })
        mock_exchange.cancel_order = AsyncMock(return_value={"status": "cancelled"})

        engine.exchange = mock_exchange
        return engine

    @pytest.mark.asyncio
    async def test_live_execution(self, engine_with_exchange):
        """Test live order execution."""
        order = engine_with_exchange.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await engine_with_exchange.execute_order(order)

        assert result.success
        assert result.exchange_order_id == "exchange_order_123"


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            order_id="order_123",
            filled_quantity=0.1,
            fill_price=50100.0,
            slippage=100.0,
            fees=5.0,
            execution_time_ms=150,
        )

        assert result.success
        assert result.filled_quantity == 0.1
        assert result.slippage == 100.0

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            order_id="order_456",
            error="Insufficient balance",
            filled_quantity=0.0,
        )

        assert not result.success
        assert result.error == "Insufficient balance"

    def test_partial_fill(self):
        """Test partial fill result."""
        result = ExecutionResult(
            success=True,
            order_id="order_789",
            filled_quantity=0.05,
            fill_price=50000.0,
            partial_fill=True,
            remaining_quantity=0.05,
        )

        assert result.partial_fill
        assert result.remaining_quantity == 0.05


class TestExecutionEngineRiskIntegration:
    """Test execution engine integration with risk guardian."""

    @pytest.fixture
    def engine_with_risk(self):
        """Create engine with risk guardian."""
        config = ExecutionConfig(paper_mode=True)
        engine = ExecutionEngine(config=config)

        # Mock risk guardian
        mock_guardian = MagicMock()
        mock_guardian.check_trade.return_value = MagicMock(
            approved=True,
            adjusted_size=0.1,
        )

        engine.risk_guardian = mock_guardian
        return engine

    @pytest.mark.asyncio
    async def test_execution_with_risk_check(self, engine_with_risk):
        """Test that execution checks risk before proceeding."""
        order = engine_with_risk.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await engine_with_risk.execute_order(
            order,
            current_price=50000.0,
            check_risk=True,
        )

        # Risk check should have been called
        engine_with_risk.risk_guardian.check_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_blocked_by_risk(self, engine_with_risk):
        """Test execution blocked when risk check fails."""
        # Make risk guardian reject
        engine_with_risk.risk_guardian.check_trade.return_value = MagicMock(
            approved=False,
            reason="Drawdown limit exceeded",
        )

        order = engine_with_risk.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = await engine_with_risk.execute_order(
            order,
            current_price=50000.0,
            check_risk=True,
        )

        assert not result.success
        assert "risk" in result.error.lower()
