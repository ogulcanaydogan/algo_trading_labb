"""
Tests for Order Book Module.

Tests the order book analysis functionality including
depth processing, liquidity metrics, and visualization data.
"""

import pytest
from datetime import datetime
from typing import List

from bot.orderbook import (
    OrderBookLevel,
    OrderBookSnapshot,
    LiquidityMetrics,
    OrderBookAnalyzer,
)


class TestOrderBookLevel:
    """Tests for OrderBookLevel dataclass."""

    def test_level_creation(self):
        """Test creating an OrderBookLevel."""
        level = OrderBookLevel(
            price=50000.0,
            quantity=1.5,
            total=1.5,
            percentage=25.0,
        )

        assert level.price == 50000.0
        assert level.quantity == 1.5
        assert level.total == 1.5
        assert level.percentage == 25.0

    def test_level_with_zero_quantity(self):
        """Test level with zero quantity."""
        level = OrderBookLevel(
            price=50000.0,
            quantity=0.0,
            total=0.0,
            percentage=0.0,
        )

        assert level.quantity == 0.0


class TestOrderBookSnapshot:
    """Tests for OrderBookSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating an OrderBookSnapshot."""
        now = datetime.now()
        snapshot = OrderBookSnapshot(
            timestamp=now,
            symbol="BTC/USDT",
            bids=[],
            asks=[],
            best_bid=50000.0,
            best_ask=50010.0,
            spread=10.0,
            spread_percent=0.02,
            mid_price=50005.0,
            total_bid_volume=10.0,
            total_ask_volume=8.0,
            imbalance=0.11,
        )

        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.spread == 10.0
        assert snapshot.mid_price == 50005.0

    def test_snapshot_with_levels(self):
        """Test snapshot with order book levels."""
        bid_level = OrderBookLevel(50000, 1.0, 1.0, 50.0)
        ask_level = OrderBookLevel(50010, 0.8, 0.8, 40.0)

        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            bids=[bid_level],
            asks=[ask_level],
            best_bid=50000.0,
            best_ask=50010.0,
            spread=10.0,
            spread_percent=0.02,
            mid_price=50005.0,
            total_bid_volume=1.0,
            total_ask_volume=0.8,
            imbalance=0.11,
        )

        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating LiquidityMetrics."""
        metrics = LiquidityMetrics(
            bid_depth_1pct=5.0,
            ask_depth_1pct=4.5,
            bid_depth_5pct=20.0,
            ask_depth_5pct=18.0,
            weighted_bid_price=49990.0,
            weighted_ask_price=50020.0,
            market_impact_buy=0.04,
            market_impact_sell=0.02,
            liquidity_score=75.0,
        )

        assert metrics.bid_depth_1pct == 5.0
        assert metrics.liquidity_score == 75.0


class TestOrderBookAnalyzer:
    """Tests for OrderBookAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return OrderBookAnalyzer(depth=20)

    @pytest.fixture
    def sample_bids(self) -> List[List[float]]:
        """Create sample bid data."""
        return [
            [50000, 1.0],
            [49990, 0.8],
            [49980, 1.2],
            [49970, 0.5],
            [49960, 2.0],
        ]

    @pytest.fixture
    def sample_asks(self) -> List[List[float]]:
        """Create sample ask data."""
        return [
            [50010, 0.9],
            [50020, 1.1],
            [50030, 0.7],
            [50040, 1.5],
            [50050, 0.6],
        ]

    def test_init_default_depth(self):
        """Test default initialization."""
        analyzer = OrderBookAnalyzer()
        assert analyzer.depth == 20

    def test_init_custom_depth(self):
        """Test custom depth initialization."""
        analyzer = OrderBookAnalyzer(depth=50)
        assert analyzer.depth == 50

    def test_init_empty_state(self, analyzer):
        """Test analyzer starts with empty state."""
        assert len(analyzer._snapshots) == 0
        assert len(analyzer._spread_history) == 0

    def test_process_raw_orderbook(self, analyzer, sample_bids, sample_asks):
        """Test processing raw order book data."""
        snapshot = analyzer.process_raw_orderbook(
            bids=sample_bids,
            asks=sample_asks,
            symbol="BTC/USDT"
        )

        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.best_bid == 50000.0
        assert snapshot.best_ask == 50010.0

    def test_process_raw_orderbook_spread(self, analyzer, sample_bids, sample_asks):
        """Test spread calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        assert snapshot.spread == 10.0  # 50010 - 50000
        assert snapshot.spread_percent > 0

    def test_process_raw_orderbook_mid_price(self, analyzer, sample_bids, sample_asks):
        """Test mid price calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        expected_mid = (50000 + 50010) / 2
        assert snapshot.mid_price == expected_mid

    def test_process_raw_orderbook_volumes(self, analyzer, sample_bids, sample_asks):
        """Test volume calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        expected_bid_vol = sum(b[1] for b in sample_bids)
        expected_ask_vol = sum(a[1] for a in sample_asks)

        assert snapshot.total_bid_volume == expected_bid_vol
        assert snapshot.total_ask_volume == expected_ask_vol

    def test_process_raw_orderbook_imbalance(self, analyzer, sample_bids, sample_asks):
        """Test imbalance calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # More bids than asks = positive imbalance
        assert -1 <= snapshot.imbalance <= 1

    def test_process_raw_orderbook_levels(self, analyzer, sample_bids, sample_asks):
        """Test order book levels are processed correctly."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        assert len(snapshot.bids) == len(sample_bids)
        assert len(snapshot.asks) == len(sample_asks)

        # First bid should have cumulative = its quantity
        assert snapshot.bids[0].total == sample_bids[0][1]

    def test_process_raw_orderbook_cumulative_totals(self, analyzer, sample_bids, sample_asks):
        """Test cumulative totals are calculated correctly."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # Last bid cumulative should equal total
        assert snapshot.bids[-1].total == snapshot.total_bid_volume

    def test_process_raw_orderbook_percentages(self, analyzer, sample_bids, sample_asks):
        """Test percentage calculations."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # Sum of percentages should be ~100
        bid_pct_sum = sum(l.percentage for l in snapshot.bids)
        assert abs(bid_pct_sum - 100) < 0.1

    def test_process_empty_bids(self, analyzer, sample_asks):
        """Test processing with empty bids."""
        snapshot = analyzer.process_raw_orderbook([], sample_asks, "BTC/USDT")

        assert snapshot.best_bid == 0
        assert len(snapshot.bids) == 0

    def test_process_empty_asks(self, analyzer, sample_bids):
        """Test processing with empty asks."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, [], "BTC/USDT")

        assert snapshot.best_ask == 0
        assert len(snapshot.asks) == 0

    def test_snapshot_history_tracking(self, analyzer, sample_bids, sample_asks):
        """Test that snapshots are tracked."""
        analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        assert len(analyzer._snapshots) == 2

    def test_spread_history_tracking(self, analyzer, sample_bids, sample_asks):
        """Test that spread history is tracked."""
        analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        assert len(analyzer._spread_history) == 1

    def test_snapshot_history_limit(self, analyzer, sample_bids, sample_asks):
        """Test snapshot history is limited when exceeding threshold."""
        for _ in range(1100):
            analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # After exceeding 1000, should be trimmed to 500
        assert len(analyzer._snapshots) <= 1000
        assert len(analyzer._spread_history) <= 1000

    def test_calculate_liquidity_metrics(self, analyzer, sample_bids, sample_asks):
        """Test liquidity metrics calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot)

        assert isinstance(metrics, LiquidityMetrics)
        assert metrics.liquidity_score >= 0
        assert metrics.liquidity_score <= 100

    def test_liquidity_depth_1pct(self, analyzer, sample_bids, sample_asks):
        """Test depth within 1% calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot)

        # Should have some depth within 1%
        assert metrics.bid_depth_1pct >= 0
        assert metrics.ask_depth_1pct >= 0

    def test_liquidity_depth_5pct(self, analyzer, sample_bids, sample_asks):
        """Test depth within 5% calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot)

        # 5% depth should be >= 1% depth
        assert metrics.bid_depth_5pct >= metrics.bid_depth_1pct
        assert metrics.ask_depth_5pct >= metrics.ask_depth_1pct

    def test_liquidity_weighted_prices(self, analyzer, sample_bids, sample_asks):
        """Test weighted average prices."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot, trade_size=1.0)

        # Weighted bid should be <= best bid (fills deeper)
        assert metrics.weighted_bid_price <= snapshot.best_bid or metrics.weighted_bid_price == snapshot.best_bid
        # Weighted ask should be >= best ask (fills deeper)
        assert metrics.weighted_ask_price >= snapshot.best_ask

    def test_liquidity_market_impact(self, analyzer, sample_bids, sample_asks):
        """Test market impact calculation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot, trade_size=0.5)

        # Impact should be non-negative
        assert metrics.market_impact_buy >= 0
        assert metrics.market_impact_sell >= 0

    def test_liquidity_score_range(self, analyzer, sample_bids, sample_asks):
        """Test liquidity score is in valid range."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot)

        assert 0 <= metrics.liquidity_score <= 100

    def test_calculate_vwap_empty_levels(self, analyzer):
        """Test VWAP with empty levels."""
        result = analyzer._calculate_vwap([], 1.0)
        assert result == 0.0

    def test_calculate_vwap_small_size(self, analyzer, sample_bids, sample_asks):
        """Test VWAP for small trade size."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # Small size should mostly fill at best price
        vwap = analyzer._calculate_vwap(snapshot.bids, 0.1)
        assert vwap <= snapshot.best_bid

    def test_calculate_vwap_large_size(self, analyzer, sample_bids, sample_asks):
        """Test VWAP for large trade size."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        # Large size should have worse price
        small_vwap = analyzer._calculate_vwap(snapshot.bids, 0.5)
        large_vwap = analyzer._calculate_vwap(snapshot.bids, 3.0)

        # For bids, larger fills should have lower VWAP
        assert large_vwap <= small_vwap

    def test_liquidity_score_calculation(self, analyzer):
        """Test liquidity score components."""
        # Good spread, good depth, balanced
        score1 = analyzer._calculate_liquidity_score(0.01, 10, 0)
        # Bad spread
        score2 = analyzer._calculate_liquidity_score(1.0, 10, 0)
        # Low depth
        score3 = analyzer._calculate_liquidity_score(0.01, 1, 0)
        # High imbalance
        score4 = analyzer._calculate_liquidity_score(0.01, 10, 0.9)

        assert score1 > score2  # Better spread = higher score
        assert score1 > score3  # Better depth = higher score
        assert score1 > score4  # Better balance = higher score

    def test_get_depth_chart_data(self, analyzer, sample_bids, sample_asks):
        """Test depth chart data generation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        data = analyzer.get_depth_chart_data(snapshot)

        assert "bids" in data
        assert "asks" in data
        assert "mid_price" in data
        assert "spread" in data

    def test_get_depth_chart_data_structure(self, analyzer, sample_bids, sample_asks):
        """Test depth chart data structure."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        data = analyzer.get_depth_chart_data(snapshot)

        # Check bid structure
        for bid in data["bids"]:
            assert "price" in bid
            assert "quantity" in bid
            assert "total" in bid

    def test_get_heatmap_data(self, analyzer, sample_bids, sample_asks):
        """Test heatmap data generation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        data = analyzer.get_heatmap_data(snapshot)

        assert "levels" in data
        assert "best_bid" in data
        assert "best_ask" in data

    def test_get_heatmap_data_structure(self, analyzer, sample_bids, sample_asks):
        """Test heatmap data structure."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        data = analyzer.get_heatmap_data(snapshot)

        # Should have both bid and ask levels
        sides = set(level["side"] for level in data["levels"])
        assert "bid" in sides
        assert "ask" in sides

        # Check level structure
        for level in data["levels"]:
            assert "price" in level
            assert "quantity" in level
            assert "side" in level
            assert "intensity" in level

    def test_get_spread_history_empty(self, analyzer):
        """Test spread history when empty."""
        history = analyzer.get_spread_history()
        assert history == []

    def test_get_spread_history_with_data(self, analyzer, sample_bids, sample_asks):
        """Test spread history with data."""
        analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        history = analyzer.get_spread_history()

        assert len(history) == 2
        for entry in history:
            assert "timestamp" in entry
            assert "spread_percent" in entry

    def test_get_spread_history_limit(self, analyzer, sample_bids, sample_asks):
        """Test spread history limit."""
        for _ in range(10):
            analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")

        history = analyzer.get_spread_history(limit=5)

        assert len(history) == 5

    def test_get_imbalance_signal_buy(self, analyzer):
        """Test imbalance signal for buy."""
        # Create snapshot with positive imbalance (more bids)
        bids = [[50000, 10.0], [49990, 5.0]]
        asks = [[50010, 1.0], [50020, 1.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")
        signal = analyzer.get_imbalance_signal(snapshot)

        assert signal["signal"] == "BUY"
        assert signal["strength"] > 0

    def test_get_imbalance_signal_sell(self, analyzer):
        """Test imbalance signal for sell."""
        # Create snapshot with negative imbalance (more asks)
        bids = [[50000, 1.0], [49990, 1.0]]
        asks = [[50010, 10.0], [50020, 5.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")
        signal = analyzer.get_imbalance_signal(snapshot)

        assert signal["signal"] == "SELL"
        assert signal["strength"] > 0

    def test_get_imbalance_signal_neutral(self, analyzer, sample_bids, sample_asks):
        """Test imbalance signal for neutral."""
        # Balanced book
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        signal = analyzer.get_imbalance_signal(snapshot)

        # With similar volumes, should be neutral or weak
        assert signal["signal"] in ["NEUTRAL", "BUY", "SELL"]

    def test_get_imbalance_signal_structure(self, analyzer, sample_bids, sample_asks):
        """Test imbalance signal structure."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        signal = analyzer.get_imbalance_signal(snapshot)

        assert "signal" in signal
        assert "strength" in signal
        assert "imbalance" in signal
        assert "bid_volume" in signal
        assert "ask_volume" in signal
        assert "ratio" in signal

    def test_to_api_response(self, analyzer, sample_bids, sample_asks):
        """Test API response generation."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        response = analyzer.to_api_response(snapshot)

        assert "timestamp" in response
        assert "symbol" in response
        assert "summary" in response
        assert "volume" in response
        assert "liquidity" in response
        assert "depth_chart" in response
        assert "signal" in response

    def test_to_api_response_summary(self, analyzer, sample_bids, sample_asks):
        """Test API response summary section."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        response = analyzer.to_api_response(snapshot)

        summary = response["summary"]
        assert "best_bid" in summary
        assert "best_ask" in summary
        assert "mid_price" in summary
        assert "spread" in summary
        assert "spread_percent" in summary
        assert "imbalance" in summary

    def test_to_api_response_with_metrics(self, analyzer, sample_bids, sample_asks):
        """Test API response with provided metrics."""
        snapshot = analyzer.process_raw_orderbook(sample_bids, sample_asks, "BTC/USDT")
        metrics = analyzer.calculate_liquidity_metrics(snapshot)
        response = analyzer.to_api_response(snapshot, liquidity=metrics)

        assert response["liquidity"]["liquidity_score"] == metrics.liquidity_score


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_level_orderbook(self):
        """Test with single level on each side."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 1.0]]
        asks = [[50010, 1.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1
        assert snapshot.spread == 10

    def test_empty_orderbook(self):
        """Test with empty order book."""
        analyzer = OrderBookAnalyzer()

        snapshot = analyzer.process_raw_orderbook([], [], "BTC/USDT")

        assert snapshot.best_bid == 0
        assert snapshot.best_ask == 0
        assert snapshot.spread == 0
        assert snapshot.imbalance == 0

    def test_very_tight_spread(self):
        """Test with very tight spread."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 1.0]]
        asks = [[50000.01, 1.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        # Allow for floating point precision
        assert abs(snapshot.spread - 0.01) < 0.0001
        assert snapshot.spread_percent < 0.001

    def test_very_wide_spread(self):
        """Test with very wide spread."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 1.0]]
        asks = [[55000, 1.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        assert snapshot.spread == 5000
        assert snapshot.spread_percent == 10.0

    def test_depth_exceeds_limit(self):
        """Test with more levels than depth limit."""
        analyzer = OrderBookAnalyzer(depth=5)
        bids = [[50000 - i, 1.0] for i in range(10)]
        asks = [[50010 + i, 1.0] for i in range(10)]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        assert len(snapshot.bids) == 5
        assert len(snapshot.asks) == 5

    def test_large_imbalance_clamped(self):
        """Test that imbalance is in valid range."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 100.0]]  # Very large bid
        asks = [[50010, 0.001]]  # Tiny ask

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        assert -1 <= snapshot.imbalance <= 1

    def test_zero_volume_division(self):
        """Test handling of zero volume."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 0.0]]  # Zero quantity
        asks = [[50010, 0.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        # Should not crash, percentages should be 0
        assert snapshot.bids[0].percentage == 0

    def test_vwap_exceeds_available_volume(self):
        """Test VWAP when trade size exceeds book."""
        analyzer = OrderBookAnalyzer()
        bids = [[50000, 1.0], [49990, 1.0]]
        asks = [[50010, 1.0]]

        snapshot = analyzer.process_raw_orderbook(bids, asks, "BTC/USDT")

        # Request more than available
        vwap = analyzer._calculate_vwap(snapshot.bids, 10.0)

        # Should still return a valid price
        assert vwap > 0
