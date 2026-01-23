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
    PaperExecutionEngine,
    BacktestExecutionEngine,
    LiveExecutionEngine,
    create_execution_engine,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionResult,
    SlippageModel,
    ExecutionConfig,
    ExecutionMode,
    FeeStructure,
    Fill,
    Position,
    PositionSide,
)


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            order_id="order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=50000.0,
        )

        assert order.order_id == "order_123"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING

    def test_limit_order(self):
        """Test limit order creation."""
        order = Order(
            order_id="order_456",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=3050.0,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.price == 3050.0

    def test_stop_loss_order(self):
        """Test stop loss order creation."""
        order = Order(
            order_id="order_789",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=0.1,
            price=50000.0,
            stop_price=48000.0,
        )

        assert order.order_type == OrderType.STOP_LOSS
        assert order.stop_price == 48000.0

    def test_order_is_complete(self):
        """Test order is_complete property."""
        order = Order(
            order_id="order_100",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        assert not order.is_complete
        order.status = OrderStatus.FILLED
        assert order.is_complete

    def test_order_remaining_quantity(self):
        """Test order remaining_quantity property."""
        order = Order(
            order_id="order_101",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        assert order.remaining_quantity == 1.0
        order.filled_quantity = 0.6
        assert order.remaining_quantity == 0.4


class TestSlippageModel:
    """Test SlippageModel functionality."""

    def test_default_slippage(self):
        """Test default slippage calculation."""
        model = SlippageModel()
        slippage = model.calculate_slippage(
            order_size=5000.0,  # $5000 order
            avg_daily_volume=1000000.0,  # $1M daily volume
            current_volatility=0.02,
            is_buy=True,
        )

        assert slippage >= 0
        assert slippage < 1.0  # Less than 1%

    def test_volume_based_slippage(self):
        """Test slippage increases with order size."""
        model = SlippageModel(base_slippage_pct=0.01, volume_impact_factor=0.1)

        slippage_small = model.calculate_slippage(
            order_size=1000.0, avg_daily_volume=1000000.0, current_volatility=0.02, is_buy=True
        )

        slippage_large = model.calculate_slippage(
            order_size=100000.0,  # 10x larger order
            avg_daily_volume=1000000.0,
            current_volatility=0.02,
            is_buy=True,
        )

        assert slippage_large > slippage_small

    def test_volatility_adjusted_slippage(self):
        """Test slippage adjusts for volatility."""
        model = SlippageModel(volatility_factor=2.0)

        slippage_low_vol = model.calculate_slippage(
            order_size=5000.0,
            avg_daily_volume=1000000.0,
            current_volatility=0.01,
            baseline_volatility=0.02,
            is_buy=True,
        )

        slippage_high_vol = model.calculate_slippage(
            order_size=5000.0,
            avg_daily_volume=1000000.0,
            current_volatility=0.05,
            baseline_volatility=0.02,
            is_buy=True,
        )

        assert slippage_high_vol > slippage_low_vol


class TestFeeStructure:
    """Test FeeStructure dataclass."""

    def test_default_fees(self):
        """Test default fee values."""
        fees = FeeStructure()

        assert fees.maker_fee_pct > 0
        assert fees.taker_fee_pct > 0

    def test_fee_calculation(self):
        """Test fee calculation."""
        fees = FeeStructure(maker_fee_pct=0.02, taker_fee_pct=0.04)

        # Taker fee for $10000 notional
        taker_fee = fees.calculate_fee(10000.0, is_maker=False)
        assert taker_fee == 4.0  # 0.04% of 10000

        # Maker fee
        maker_fee = fees.calculate_fee(10000.0, is_maker=True)
        assert maker_fee == 2.0  # 0.02% of 10000


class TestExecutionConfig:
    """Test ExecutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()

        assert config.max_retries > 0
        assert config.timeout_ms > 0
        assert config.max_slippage_pct > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutionConfig(
            max_retries=5,
            retry_delay_ms=500.0,
            timeout_ms=10000.0,
            max_slippage_pct=0.5,
            paper_mode=True,
        )

        assert config.max_retries == 5
        assert config.paper_mode
        assert config.mode == ExecutionMode.PAPER

    def test_mode_sync_with_paper_mode(self):
        """Test that mode syncs with paper_mode flag."""
        config_paper = ExecutionConfig(paper_mode=True)
        assert config_paper.mode == ExecutionMode.PAPER

        config_live = ExecutionConfig(paper_mode=False)
        assert config_live.mode == ExecutionMode.LIVE


class TestPaperExecutionEngine:
    """Test PaperExecutionEngine core functionality."""

    @pytest.fixture
    def engine(self):
        """Create PaperExecutionEngine for testing."""
        return PaperExecutionEngine()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.mode == ExecutionMode.PAPER
        assert len(engine.orders) == 0
        assert len(engine.pending_orders) == 0

    @pytest.mark.asyncio
    async def test_submit_market_order(self, engine):
        """Test submitting a market order."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0, "volume": 1000000}

        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        assert result.success
        assert result.order.symbol == "BTC/USDT"
        assert result.order.side == OrderSide.BUY
        assert result.order.order_type == OrderType.MARKET
        assert result.order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_submit_limit_order_filled(self, engine):
        """Test limit order that gets filled immediately."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0}

        # Buy limit at ask price should fill
        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=50020.0,  # Above ask
            market_data=market_data,
        )

        assert result.success
        assert result.order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_submit_limit_order_pending(self, engine):
        """Test limit order that goes to pending."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0}

        # Buy limit below bid shouldn't fill immediately
        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=49000.0,  # Below current price
            market_data=market_data,
        )

        assert result.success
        assert result.order.status == OrderStatus.SUBMITTED
        assert result.order.order_id in engine.pending_orders

    @pytest.mark.asyncio
    async def test_cancel_pending_order(self, engine):
        """Test cancelling a pending order."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0}

        # Create pending order
        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=45000.0,
            market_data=market_data,
        )

        order_id = result.order.order_id

        # Cancel it
        success = await engine.cancel_order(order_id)

        assert success
        assert result.order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_slippage_applied(self, engine):
        """Test slippage is applied to market executions."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0, "volume": 1000000}

        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        assert result.success
        # Fill price should be at or above ask due to slippage
        assert result.average_price >= 50010.0

    def test_get_position(self, engine):
        """Test getting position."""
        pos = engine.get_position("BTC/USDT")

        assert pos.symbol == "BTC/USDT"
        assert pos.side == PositionSide.FLAT
        assert pos.quantity == 0.0

    @pytest.mark.asyncio
    async def test_position_updated_after_fill(self, engine):
        """Test position is updated after fill."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0, "volume": 1000000}

        await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        pos = engine.get_position("BTC/USDT")
        assert pos.side == PositionSide.LONG
        assert pos.quantity > 0

    def test_execution_stats(self, engine):
        """Test execution statistics."""
        stats = engine.get_execution_stats()

        assert "mode" in stats
        assert "total_orders" in stats
        assert "successful_orders" in stats
        assert "failed_orders" in stats
        assert stats["mode"] == "paper"


class TestBacktestExecutionEngine:
    """Test BacktestExecutionEngine."""

    @pytest.fixture
    def engine(self):
        """Create BacktestExecutionEngine for testing."""
        return BacktestExecutionEngine()

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.mode == ExecutionMode.BACKTEST

    @pytest.mark.asyncio
    async def test_market_order_execution(self, engine):
        """Test market order execution in backtest."""
        market_data = {"close": 50000.0, "high": 50500.0, "low": 49500.0, "volume": 1000000}

        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        assert result.success
        assert result.order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_limit_order_filled_in_range(self, engine):
        """Test limit order that is within bar range."""
        market_data = {"close": 50000.0, "high": 51000.0, "low": 49000.0, "volume": 1000000}

        # Buy limit at 49500 - within low-high range
        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=49500.0,
            market_data=market_data,
        )

        assert result.success
        assert result.order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, engine):
        """Test stop loss order that triggers."""
        market_data = {"close": 48000.0, "high": 49000.0, "low": 47000.0, "volume": 1000000}

        # Sell stop at 47500 - should trigger since low < 47500
        result = await engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=0.1,
            stop_price=47500.0,
            market_data=market_data,
        )

        assert result.success
        assert result.order.status == OrderStatus.FILLED

    def test_set_timestamp(self, engine):
        """Test setting backtest timestamp."""
        ts = datetime(2026, 1, 15, 10, 0, 0)
        engine.set_timestamp(ts)
        assert engine.current_timestamp == ts


class TestLiveExecutionEngine:
    """Test LiveExecutionEngine with mock exchange."""

    @pytest.fixture
    def engine_with_exchange(self):
        """Create engine with mocked exchange."""
        mock_exchange = MagicMock()
        mock_exchange.create_market_order = AsyncMock(
            return_value={
                "id": "exchange_order_123",
                "status": "closed",
                "filled": 0.1,
                "average": 50100.0,
                "fee": {"cost": 2.0},
            }
        )
        mock_exchange.cancel_order = AsyncMock(return_value={"status": "cancelled"})

        return LiveExecutionEngine(exchange_client=mock_exchange)

    def test_initialization(self, engine_with_exchange):
        """Test live engine initialization."""
        assert engine_with_exchange.mode == ExecutionMode.LIVE

    @pytest.mark.asyncio
    async def test_live_market_order(self, engine_with_exchange):
        """Test live market order execution."""
        market_data = {"price": 50000.0}

        result = await engine_with_exchange.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        assert result.success
        assert result.order.exchange_order_id == "exchange_order_123"


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_successful_result_properties(self):
        """Test successful execution result properties."""
        order = Order(
            order_id="order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        fill = Fill(
            fill_id="fill_1",
            order_id="order_123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50100.0,
            fee=2.0,
        )

        result = ExecutionResult(
            success=True,
            order=order,
            fills=[fill],
        )

        assert result.success
        assert result.total_filled == 0.1
        assert result.average_price == 50100.0
        assert result.total_fees == 2.0

    def test_failed_result(self):
        """Test failed execution result."""
        order = Order(
            order_id="order_456",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )

        result = ExecutionResult(
            success=False,
            order=order,
            error_message="Insufficient balance",
        )

        assert not result.success
        assert result.error_message == "Insufficient balance"
        assert result.total_filled == 0.0


class TestCreateExecutionEngine:
    """Test factory function."""

    def test_create_backtest_engine(self):
        """Test creating backtest engine."""
        engine = create_execution_engine(ExecutionMode.BACKTEST)
        assert isinstance(engine, BacktestExecutionEngine)
        assert engine.mode == ExecutionMode.BACKTEST

    def test_create_paper_engine(self):
        """Test creating paper engine."""
        engine = create_execution_engine(ExecutionMode.PAPER)
        assert isinstance(engine, PaperExecutionEngine)
        assert engine.mode == ExecutionMode.PAPER

    def test_create_live_engine_requires_exchange(self):
        """Test that live engine requires exchange client."""
        with pytest.raises(ValueError, match="requires exchange_client"):
            create_execution_engine(ExecutionMode.LIVE)

    def test_create_live_engine_with_exchange(self):
        """Test creating live engine with exchange."""
        mock_exchange = MagicMock()
        engine = create_execution_engine(ExecutionMode.LIVE, exchange_client=mock_exchange)
        assert isinstance(engine, LiveExecutionEngine)
        assert engine.mode == ExecutionMode.LIVE

    def test_factory_uses_consistent_fees(self):
        """Test that factory creates engines with consistent fees."""
        backtest_engine = create_execution_engine(ExecutionMode.BACKTEST)
        paper_engine = create_execution_engine(ExecutionMode.PAPER)

        assert (
            backtest_engine.fee_structure.maker_fee_pct == paper_engine.fee_structure.maker_fee_pct
        )
        assert (
            backtest_engine.fee_structure.taker_fee_pct == paper_engine.fee_structure.taker_fee_pct
        )


class TestExecutionEngineRiskIntegration:
    """Test execution engine integration with risk guardian."""

    @pytest.fixture
    def engine_with_risk(self):
        """Create engine with risk guardian."""
        mock_guardian = MagicMock()
        mock_guardian.check_trade.return_value = MagicMock(
            approved=True,
            veto_reasons=[],
        )

        return PaperExecutionEngine(risk_guardian=mock_guardian)

    def test_risk_guardian_can_be_attached(self, engine_with_risk):
        """Test that risk guardian can be attached to engine."""
        assert engine_with_risk.risk_guardian is not None

    @pytest.mark.asyncio
    async def test_execution_succeeds_with_risk_guardian(self, engine_with_risk):
        """Test that execution succeeds when risk guardian is attached."""
        market_data = {"price": 50000.0, "bid": 49990.0, "ask": 50010.0, "volume": 1000000}

        result = await engine_with_risk.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )

        # Execution should succeed even with risk guardian attached
        assert result.success
        assert result.order.status == OrderStatus.FILLED
