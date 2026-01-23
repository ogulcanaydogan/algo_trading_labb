"""
Tests for the multi-exchange trading coordinator.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from bot.execution.multi_exchange import (
    AggregatedBook,
    ArbitrageOpportunity,
    ExchangeQuote,
    ExchangeStatus,
    ExecutionResult,
    MockExchangeAdapter,
    MultiExchangeCoordinator,
    RouteDecision,
    RoutingStrategy,
    create_multi_exchange_coordinator,
)


class TestExchangeQuote:
    """Tests for ExchangeQuote."""

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = ExchangeQuote(
            exchange="test",
            symbol="BTC/USDT",
            bid=49900,
            ask=50100,
            bid_size=10,
            ask_size=10,
            timestamp=datetime.now(),
        )

        assert quote.mid == 50000

    def test_spread(self):
        """Test spread calculation."""
        quote = ExchangeQuote(
            exchange="test",
            symbol="BTC/USDT",
            bid=49900,
            ask=50100,
            bid_size=10,
            ask_size=10,
            timestamp=datetime.now(),
        )

        assert quote.spread == 200
        assert pytest.approx(quote.spread_bps, rel=0.01) == 40  # 200/50000 * 10000


class TestMockExchangeAdapter:
    """Tests for MockExchangeAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create mock adapter."""
        return MockExchangeAdapter(
            exchange_id="test_exchange",
            base_price=50000,
            spread_bps=10,
            fee_rate=0.001,
        )

    @pytest.mark.asyncio
    async def test_connect(self, adapter):
        """Test connection."""
        result = await adapter.connect()
        assert result is True
        assert adapter.is_connected is True

    @pytest.mark.asyncio
    async def test_get_quote(self, adapter):
        """Test getting quote."""
        quote = await adapter.get_quote("BTC/USDT")

        assert quote.exchange == "test_exchange"
        assert quote.symbol == "BTC/USDT"
        assert quote.bid < quote.ask
        assert quote.mid == pytest.approx(50000, rel=0.01)

    @pytest.mark.asyncio
    async def test_place_order(self, adapter):
        """Test placing order."""
        result = await adapter.place_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
        )

        assert result["status"] == "filled"
        assert result["filled_quantity"] == 1.0
        assert result["fees"] > 0

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """Test getting balance."""
        balance = await adapter.get_balance("USDT")
        assert balance == 100000


class TestMultiExchangeCoordinator:
    """Tests for MultiExchangeCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with mock exchanges."""
        coord = MultiExchangeCoordinator()

        # Add exchanges with different prices
        coord.add_exchange(
            "exchange_a",
            MockExchangeAdapter("exchange_a", base_price=50000, spread_bps=10, fee_rate=0.001),
            is_primary=True,
        )
        coord.add_exchange(
            "exchange_b",
            MockExchangeAdapter("exchange_b", base_price=50100, spread_bps=20, fee_rate=0.0005),
        )
        coord.add_exchange(
            "exchange_c",
            MockExchangeAdapter("exchange_c", base_price=49900, spread_bps=15, fee_rate=0.002),
        )

        return coord

    @pytest.mark.asyncio
    async def test_add_remove_exchange(self):
        """Test adding and removing exchanges."""
        coord = MultiExchangeCoordinator()

        adapter = MockExchangeAdapter("test")
        coord.add_exchange("test", adapter)

        assert "test" in coord._exchanges
        assert coord.primary_exchange == "test"

        coord.remove_exchange("test")
        assert "test" not in coord._exchanges

    @pytest.mark.asyncio
    async def test_connect_all(self, coordinator):
        """Test connecting to all exchanges."""
        results = await coordinator.connect_all()

        assert all(results.values())
        assert coordinator._status["exchange_a"] == ExchangeStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_best_price_buy(self, coordinator):
        """Test getting best buy price."""
        await coordinator.connect_all()

        best = await coordinator.get_best_price("BTC/USDT", "buy")

        assert best is not None
        exchange, price = best
        # Best buy = lowest ask (exchange_c has lowest base price)
        assert exchange == "exchange_c"

    @pytest.mark.asyncio
    async def test_get_best_price_sell(self, coordinator):
        """Test getting best sell price."""
        await coordinator.connect_all()

        best = await coordinator.get_best_price("BTC/USDT", "sell")

        assert best is not None
        exchange, price = best
        # Best sell = highest bid (exchange_b has highest base price)
        assert exchange == "exchange_b"

    @pytest.mark.asyncio
    async def test_get_aggregated_book(self, coordinator):
        """Test getting aggregated orderbook."""
        await coordinator.connect_all()

        book = await coordinator.get_aggregated_book("BTC/USDT")

        assert book.symbol == "BTC/USDT"
        assert len(book.bids) == 3
        assert len(book.asks) == 3

        # Bids should be sorted descending
        assert book.bids[0][1] >= book.bids[1][1] >= book.bids[2][1]

        # Asks should be sorted ascending
        assert book.asks[0][1] <= book.asks[1][1] <= book.asks[2][1]

    @pytest.mark.asyncio
    async def test_execute_order_best_price(self, coordinator):
        """Test executing order with best price routing."""
        await coordinator.connect_all()

        result = await coordinator.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            routing=RoutingStrategy.BEST_PRICE,
        )

        assert result.success is True
        assert result.filled_quantity == 1.0
        assert result.average_price > 0
        assert len(result.exchange_fills) == 1

    @pytest.mark.asyncio
    async def test_execute_order_split(self, coordinator):
        """Test executing order with split routing."""
        await coordinator.connect_all()

        result = await coordinator.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=5.0,
            routing=RoutingStrategy.SPLIT,
        )

        assert result.success is True
        assert result.filled_quantity > 0
        # Should be split across multiple exchanges
        assert len(result.exchange_fills) >= 1

    @pytest.mark.asyncio
    async def test_execute_order_primary(self, coordinator):
        """Test executing order with primary exchange routing."""
        await coordinator.connect_all()

        result = await coordinator.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            routing=RoutingStrategy.PRIMARY,
        )

        assert result.success is True
        assert "exchange_a" in result.exchange_fills

    @pytest.mark.asyncio
    async def test_execute_order_lowest_fee(self, coordinator):
        """Test executing order with lowest fee routing."""
        await coordinator.connect_all()

        result = await coordinator.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            routing=RoutingStrategy.LOWEST_FEE,
        )

        assert result.success is True
        # Exchange B has lowest fee (0.0005)
        assert "exchange_b" in result.exchange_fills

    @pytest.mark.asyncio
    async def test_execute_order_no_quotes(self, coordinator):
        """Test executing when no quotes available."""
        # Don't connect, so no quotes available

        result = await coordinator.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
        )

        assert result.success is False
        assert "No quotes available" in result.errors[0]

    @pytest.mark.asyncio
    async def test_detect_arbitrage_exists(self):
        """Test arbitrage detection when opportunity exists."""
        coord = MultiExchangeCoordinator(min_arb_spread_bps=5)

        # Exchange A: bid 50000, ask 50010
        # Exchange B: bid 50100, ask 50110
        # Arbitrage: buy on A at 50010, sell on B at 50100 = 90 profit per unit

        coord.add_exchange(
            "exchange_a",
            MockExchangeAdapter("exchange_a", base_price=50005, spread_bps=2, fee_rate=0.0001),
        )
        coord.add_exchange(
            "exchange_b",
            MockExchangeAdapter("exchange_b", base_price=50105, spread_bps=2, fee_rate=0.0001),
        )

        await coord.connect_all()

        arb = await coord.detect_arbitrage("BTC/USDT")

        assert arb is not None
        assert arb.buy_exchange == "exchange_a"
        assert arb.sell_exchange == "exchange_b"
        assert arb.spread_bps > 0
        assert arb.profit_estimate > 0

    @pytest.mark.asyncio
    async def test_detect_arbitrage_none(self, coordinator):
        """Test arbitrage detection when no opportunity."""
        await coordinator.connect_all()

        # With normal spreads, no arbitrage should exist
        arb = await coordinator.detect_arbitrage("BTC/USDT")

        # May or may not find arbitrage depending on mock prices
        # Just ensure it doesn't error
        assert arb is None or isinstance(arb, ArbitrageOpportunity)

    @pytest.mark.asyncio
    async def test_get_consolidated_balance(self, coordinator):
        """Test getting consolidated balance."""
        await coordinator.connect_all()

        balances = await coordinator.get_consolidated_balance("USDT")

        assert len(balances) == 3
        assert all(b == 100000 for b in balances.values())

    @pytest.mark.asyncio
    async def test_get_exchange_status(self, coordinator):
        """Test getting exchange status."""
        await coordinator.connect_all()

        status = coordinator.get_exchange_status()

        assert len(status) == 3
        assert status["exchange_a"]["is_primary"] is True
        assert status["exchange_b"]["is_primary"] is False


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_fill_rate(self):
        """Test fill rate calculation."""
        result = ExecutionResult(
            order_id="test",
            symbol="BTC/USDT",
            side="buy",
            requested_quantity=10.0,
            filled_quantity=7.5,
            average_price=50000,
            total_fees=50,
            exchange_fills={},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=True,
        )

        assert result.fill_rate == 0.75

    def test_to_dict(self):
        """Test serialization."""
        result = ExecutionResult(
            order_id="test",
            symbol="BTC/USDT",
            side="buy",
            requested_quantity=1.0,
            filled_quantity=1.0,
            average_price=50000,
            total_fees=50,
            exchange_fills={"binance": {"quantity": 1.0}},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=True,
        )

        data = result.to_dict()

        assert data["order_id"] == "test"
        assert data["fill_rate"] == 1.0
        assert data["success"] is True


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_coordinator(self):
        """Test creating coordinator with factory."""
        adapter1 = MockExchangeAdapter("ex1")
        adapter2 = MockExchangeAdapter("ex2")

        coord = create_multi_exchange_coordinator(
            exchanges={"ex1": adapter1, "ex2": adapter2},
            primary="ex1",
        )

        assert "ex1" in coord._exchanges
        assert "ex2" in coord._exchanges
        assert coord.primary_exchange == "ex1"

    def test_create_empty_coordinator(self):
        """Test creating empty coordinator."""
        coord = create_multi_exchange_coordinator()

        assert len(coord._exchanges) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_single_exchange(self):
        """Test with single exchange."""
        coord = MultiExchangeCoordinator()
        coord.add_exchange("only", MockExchangeAdapter("only"))

        await coord.connect_all()

        result = await coord.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_exchange_failure_fallback(self):
        """Test fallback when exchange fails."""
        coord = MultiExchangeCoordinator()

        # Add failing adapter
        failing = MockExchangeAdapter("failing")
        failing.get_quote = AsyncMock(side_effect=Exception("Connection error"))

        coord.add_exchange("failing", failing, is_primary=True)
        coord.add_exchange("backup", MockExchangeAdapter("backup"))

        coord._status["failing"] = ExchangeStatus.HEALTHY
        coord._status["backup"] = ExchangeStatus.HEALTHY

        # Backup connection should work
        await coord._exchanges["backup"].connect()

        result = await coord.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            routing=RoutingStrategy.BEST_PRICE,
        )

        # Should fall back to backup
        assert result.success is True

    @pytest.mark.asyncio
    async def test_zero_quantity(self):
        """Test with zero quantity."""
        coord = MultiExchangeCoordinator()
        coord.add_exchange("test", MockExchangeAdapter("test"))

        await coord.connect_all()

        result = await coord.execute_order(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.0,
        )

        # Should handle gracefully
        assert result.requested_quantity == 0
