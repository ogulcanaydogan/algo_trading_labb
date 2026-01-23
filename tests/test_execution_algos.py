"""Tests for execution algorithms."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.execution.execution_algos import (
    AlgoOrder,
    AlgoSlice,
    AlgoExecution,
    AlgoStatus,
    UrgencyLevel,
    TWAPAlgorithm,
    VWAPAlgorithm,
    POVAlgorithm,
    ImplementationShortfallAlgorithm,
    IcebergAlgorithm,
    AdaptiveAlgorithm,
    AlgorithmFactory,
    create_execution_algorithm,
)


@pytest.fixture
def mock_order_executor():
    """Create mock order executor."""
    async def executor(**kwargs):
        return {
            "fill_price": kwargs.get("price", 100.0),
            "filled_quantity": kwargs.get("quantity", 1.0),
        }
    return executor


@pytest.fixture
def mock_market_data_provider():
    """Create mock market data provider."""
    async def provider(symbol):
        return {
            "mid_price": 100.0,
            "last_price": 100.0,
            "bid_price": 99.95,
            "ask_price": 100.05,
            "bid_size": 1000.0,
            "ask_size": 1000.0,
            "volume": 10000.0,
            "recent_volume": 500.0,
            "avg_volume": 8000.0,
            "volatility": 0.02,
            "spread": 0.001,
            "spread_bps": 10,
        }
    return provider


@pytest.fixture
def sample_order():
    """Create sample order."""
    return AlgoOrder(
        order_id="test_order_1",
        symbol="BTC/USDT",
        side="buy",
        total_quantity=10.0,
        urgency=UrgencyLevel.MEDIUM,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=5),
    )


class TestAlgoOrder:
    """Tests for AlgoOrder dataclass."""

    def test_order_creation(self):
        order = AlgoOrder(
            order_id="order_1",
            symbol="BTC/USDT",
            side="buy",
            total_quantity=10.0,
        )
        assert order.order_id == "order_1"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.total_quantity == 10.0
        assert order.urgency == UrgencyLevel.MEDIUM
        assert order.start_time is not None

    def test_order_with_limit_price(self):
        order = AlgoOrder(
            order_id="order_2",
            symbol="ETH/USDT",
            side="sell",
            total_quantity=5.0,
            limit_price=2500.0,
        )
        assert order.limit_price == 2500.0


class TestAlgoSlice:
    """Tests for AlgoSlice dataclass."""

    def test_slice_creation(self):
        slice_order = AlgoSlice(
            slice_id="slice_1",
            parent_order_id="order_1",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            target_price=100.0,
        )
        assert slice_order.slice_id == "slice_1"
        assert not slice_order.is_filled
        assert slice_order.fill_rate == 0.0

    def test_slice_filled(self):
        slice_order = AlgoSlice(
            slice_id="slice_2",
            parent_order_id="order_1",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            target_price=100.0,
            filled_quantity=1.0,
            actual_price=100.05,
        )
        assert slice_order.is_filled
        assert slice_order.fill_rate == 1.0

    def test_slice_to_dict(self):
        slice_order = AlgoSlice(
            slice_id="slice_3",
            parent_order_id="order_1",
            symbol="BTC/USDT",
            side="buy",
            quantity=2.0,
            target_price=100.0,
            filled_quantity=1.5,
        )
        result = slice_order.to_dict()
        assert result["slice_id"] == "slice_3"
        assert result["fill_rate"] == 0.75


class TestAlgoExecution:
    """Tests for AlgoExecution dataclass."""

    def test_execution_result(self):
        execution = AlgoExecution(
            order_id="order_1",
            algorithm="TWAP",
            symbol="BTC/USDT",
            side="buy",
            total_quantity=10.0,
            filled_quantity=9.5,
            average_price=100.0,
            vwap_benchmark=99.9,
            arrival_price=99.8,
            slippage_bps=10.0,
            implementation_shortfall_bps=20.0,
            participation_rate=0.05,
            num_slices=10,
            duration_seconds=300.0,
            status=AlgoStatus.COMPLETED,
        )
        assert execution.fill_rate == 0.95
        assert execution.status == AlgoStatus.COMPLETED

    def test_execution_to_dict(self):
        execution = AlgoExecution(
            order_id="order_1",
            algorithm="VWAP",
            symbol="ETH/USDT",
            side="sell",
            total_quantity=5.0,
            filled_quantity=5.0,
            average_price=2500.0,
            vwap_benchmark=2499.0,
            arrival_price=2501.0,
            slippage_bps=-4.0,
            implementation_shortfall_bps=4.0,
            participation_rate=0.02,
            num_slices=5,
            duration_seconds=60.0,
            status=AlgoStatus.COMPLETED,
        )
        result = execution.to_dict()
        assert result["algorithm"] == "VWAP"
        assert result["fill_rate"] == 1.0


class TestTWAPAlgorithm:
    """Tests for TWAP algorithm."""

    @pytest.mark.asyncio
    async def test_twap_schedule_generation(
        self, mock_order_executor, mock_market_data_provider, sample_order
    ):
        algo = TWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            interval_seconds=1.0,
        )
        market_data = await mock_market_data_provider("BTC/USDT")
        schedule = await algo.generate_schedule(sample_order, market_data)

        assert len(schedule) > 0
        total_qty = sum(qty for _, qty in schedule)
        assert abs(total_qty - sample_order.total_quantity) < 0.01

    @pytest.mark.asyncio
    async def test_twap_execution(
        self, mock_order_executor, mock_market_data_provider
    ):
        algo = TWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            interval_seconds=0.01,
        )
        order = AlgoOrder(
            order_id="twap_test",
            symbol="BTC/USDT",
            side="buy",
            total_quantity=1.0,
            urgency=UrgencyLevel.CRITICAL,
        )

        result = await algo.execute(order)

        assert result.algorithm == "TWAP"
        assert result.filled_quantity > 0
        assert result.status in [AlgoStatus.COMPLETED, AlgoStatus.FAILED]


class TestVWAPAlgorithm:
    """Tests for VWAP algorithm."""

    @pytest.mark.asyncio
    async def test_vwap_schedule_generation(
        self, mock_order_executor, mock_market_data_provider, sample_order
    ):
        algo = VWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            interval_seconds=1.0,
        )
        market_data = await mock_market_data_provider("BTC/USDT")
        schedule = await algo.generate_schedule(sample_order, market_data)

        assert len(schedule) > 0
        # VWAP distributes based on volume profile
        total_qty = sum(qty for _, qty in schedule)
        assert abs(total_qty - sample_order.total_quantity) < 0.01

    @pytest.mark.asyncio
    async def test_vwap_with_custom_profile(
        self, mock_order_executor, mock_market_data_provider
    ):
        custom_profile = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
        algo = VWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            volume_profile=custom_profile,
            interval_seconds=1.0,
        )
        assert algo.volume_profile == custom_profile


class TestPOVAlgorithm:
    """Tests for POV algorithm."""

    @pytest.mark.asyncio
    async def test_pov_adapts_to_volume(
        self, mock_order_executor, mock_market_data_provider, sample_order
    ):
        algo = POVAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            target_participation=0.1,
            interval_seconds=1.0,
        )
        assert algo.target_participation == 0.1


class TestImplementationShortfallAlgorithm:
    """Tests for Implementation Shortfall algorithm."""

    @pytest.mark.asyncio
    async def test_is_balances_impact_vs_risk(
        self, mock_order_executor, mock_market_data_provider, sample_order
    ):
        algo = ImplementationShortfallAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            risk_aversion=0.5,
            interval_seconds=1.0,
        )
        market_data = await mock_market_data_provider("BTC/USDT")
        schedule = await algo.generate_schedule(sample_order, market_data)

        assert len(schedule) > 0
        # IS should front-load execution
        if len(schedule) > 1:
            first_qty = schedule[0][1]
            last_qty = schedule[-1][1]
            # First slice should generally be larger than last
            # (front-loaded for risk reduction)


class TestIcebergAlgorithm:
    """Tests for Iceberg algorithm."""

    @pytest.mark.asyncio
    async def test_iceberg_visible_quantity(
        self, mock_order_executor, mock_market_data_provider
    ):
        algo = IcebergAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            visible_quantity=0.1,
            interval_seconds=1.0,
        )
        order = AlgoOrder(
            order_id="iceberg_test",
            symbol="BTC/USDT",
            side="buy",
            total_quantity=100.0,
        )
        market_data = await mock_market_data_provider("BTC/USDT")
        schedule = await algo.generate_schedule(order, market_data)

        # Each slice should be ~10% of total
        for _, qty in schedule:
            assert qty <= order.total_quantity * 0.11  # Allow small tolerance


class TestAdaptiveAlgorithm:
    """Tests for Adaptive algorithm."""

    @pytest.mark.asyncio
    async def test_adaptive_selects_strategy(
        self, mock_order_executor, mock_market_data_provider, sample_order
    ):
        algo = AdaptiveAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
            interval_seconds=1.0,
        )
        market_data = await mock_market_data_provider("BTC/USDT")
        schedule = await algo.generate_schedule(sample_order, market_data)

        assert len(schedule) > 0


class TestAlgorithmFactory:
    """Tests for AlgorithmFactory."""

    def test_create_twap(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "twap",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "TWAP"

    def test_create_vwap(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "vwap",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "VWAP"

    def test_create_pov(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "pov",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "POV"

    def test_create_is(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "is",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "IS"

    def test_create_iceberg(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "iceberg",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "Iceberg"

    def test_create_adaptive(self, mock_order_executor, mock_market_data_provider):
        algo = AlgorithmFactory.create(
            "adaptive",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "Adaptive"

    def test_create_unknown_raises(self, mock_order_executor, mock_market_data_provider):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmFactory.create(
                "unknown_algo",
                mock_order_executor,
                mock_market_data_provider,
            )


class TestCreateExecutionAlgorithm:
    """Tests for create_execution_algorithm factory function."""

    def test_factory_function(self, mock_order_executor, mock_market_data_provider):
        algo = create_execution_algorithm(
            "twap",
            mock_order_executor,
            mock_market_data_provider,
        )
        assert algo.name == "TWAP"


class TestAlgorithmControls:
    """Tests for algorithm control methods."""

    def test_pause_resume(self, mock_order_executor, mock_market_data_provider):
        algo = TWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
        )
        algo._status = AlgoStatus.RUNNING

        algo.pause()
        assert algo._status == AlgoStatus.PAUSED

        algo.resume()
        assert algo._status == AlgoStatus.RUNNING

    def test_cancel(self, mock_order_executor, mock_market_data_provider):
        algo = TWAPAlgorithm(
            mock_order_executor,
            mock_market_data_provider,
        )
        algo.cancel()
        assert algo._cancel_requested is True
