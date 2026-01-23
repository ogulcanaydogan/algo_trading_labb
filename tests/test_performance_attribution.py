"""
Tests for Performance Attribution Module.

Tests the performance attribution functionality including
trade logging, attribution analysis, and report generation.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List
import shutil

from bot.performance_attribution import (
    AttributionFactor,
    TradeRecord,
    AttributionResult,
    PerformanceReport,
    PerformanceAttributor,
)


class TestAttributionFactor:
    """Tests for AttributionFactor enum."""

    def test_factor_values(self):
        """Test AttributionFactor enum values."""
        assert AttributionFactor.STRATEGY.value == "strategy"
        assert AttributionFactor.MODEL.value == "model"
        assert AttributionFactor.ASSET.value == "asset"
        assert AttributionFactor.TIMING.value == "timing"
        assert AttributionFactor.MARKET_TYPE.value == "market_type"
        assert AttributionFactor.REGIME.value == "regime"
        assert AttributionFactor.SENTIMENT.value == "sentiment"

    def test_factor_from_string(self):
        """Test creating factor from string."""
        assert AttributionFactor("strategy") == AttributionFactor.STRATEGY
        assert AttributionFactor("model") == AttributionFactor.MODEL

    def test_all_factors_exist(self):
        """Test all expected factors exist."""
        expected = ["strategy", "model", "asset", "timing", "market_type", "regime", "sentiment"]
        actual = [f.value for f in AttributionFactor]
        assert set(expected) == set(actual)


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self):
        """Test creating a TradeRecord."""
        now = datetime.now()
        trade = TradeRecord(
            trade_id="T001",
            symbol="BTC/USDT",
            market_type="crypto",
            action="BUY",
            quantity=0.5,
            entry_price=50000,
            exit_price=52000,
            pnl=1000,
            pnl_pct=0.04,
            strategy="momentum",
            model="lstm",
            regime="trending",
            confidence=0.85,
            sentiment_score=0.6,
            entry_time=now,
            exit_time=now + timedelta(hours=2),
            holding_period_hours=2.0,
        )

        assert trade.trade_id == "T001"
        assert trade.symbol == "BTC/USDT"
        assert trade.pnl == 1000
        assert trade.confidence == 0.85

    def test_trade_record_to_dict(self):
        """Test TradeRecord to_dict method."""
        now = datetime.now()
        trade = TradeRecord(
            trade_id="T001",
            symbol="BTC/USDT",
            market_type="crypto",
            action="BUY",
            quantity=0.5,
            entry_price=50000,
            exit_price=52000,
            pnl=1000,
            pnl_pct=0.04,
            strategy="momentum",
            model="lstm",
            regime="trending",
            confidence=0.85,
            sentiment_score=0.6,
            entry_time=now,
            exit_time=now + timedelta(hours=2),
            holding_period_hours=2.0,
        )

        result = trade.to_dict()

        assert result["trade_id"] == "T001"
        assert result["pnl"] == 1000
        assert result["entry_time"] == now.isoformat()
        assert result["exit_time"] is not None

    def test_trade_record_with_none_exit(self):
        """Test TradeRecord with no exit info."""
        trade = TradeRecord(
            trade_id="T002",
            symbol="ETH/USDT",
            market_type="crypto",
            action="BUY",
            quantity=1.0,
            entry_price=3000,
            exit_price=None,
            pnl=0,
            pnl_pct=0,
            strategy="grid",
            model="default",
            regime="ranging",
            confidence=0.5,
            sentiment_score=None,
            entry_time=datetime.now(),
            exit_time=None,
            holding_period_hours=0,
        )

        result = trade.to_dict()
        assert result["exit_price"] is None
        assert result["exit_time"] is None
        assert result["sentiment_score"] is None

    def test_trade_record_with_metadata(self):
        """Test TradeRecord with metadata."""
        trade = TradeRecord(
            trade_id="T003",
            symbol="AAPL",
            market_type="stock",
            action="SELL",
            quantity=10,
            entry_price=150,
            exit_price=145,
            pnl=-50,
            pnl_pct=-0.033,
            strategy="mean_reversion",
            model="random_forest",
            regime="volatile",
            confidence=0.7,
            sentiment_score=-0.2,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            holding_period_hours=0.5,
            metadata={"reason": "stop_loss", "signal_strength": 0.8},
        )

        assert trade.metadata["reason"] == "stop_loss"


class TestAttributionResult:
    """Tests for AttributionResult dataclass."""

    def test_attribution_result_creation(self):
        """Test creating an AttributionResult."""
        result = AttributionResult(
            factor=AttributionFactor.STRATEGY,
            category="momentum",
            total_pnl=500.0,
            pnl_contribution_pct=25.0,
            trade_count=10,
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=50.0,
            profit_factor=2.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            avg_holding_period=24.0,
        )

        assert result.factor == AttributionFactor.STRATEGY
        assert result.category == "momentum"
        assert result.total_pnl == 500.0

    def test_attribution_result_to_dict(self):
        """Test AttributionResult to_dict method."""
        result = AttributionResult(
            factor=AttributionFactor.MODEL,
            category="lstm",
            total_pnl=300.0,
            pnl_contribution_pct=15.0,
            trade_count=8,
            win_rate=0.625,
            avg_win=75.0,
            avg_loss=40.0,
            profit_factor=1.875,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            avg_holding_period=12.0,
        )

        dict_result = result.to_dict()

        assert dict_result["factor"] == "model"
        assert dict_result["category"] == "lstm"
        assert dict_result["total_pnl"] == 300.0
        assert dict_result["win_rate"] == 0.625


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_performance_report_creation(self):
        """Test creating a PerformanceReport."""
        now = datetime.now()
        report = PerformanceReport(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_pnl=1500.0,
            total_pnl_pct=0.15,
            total_trades=25,
            overall_win_rate=0.56,
            overall_sharpe=1.3,
            attributions_by_strategy=[],
            attributions_by_model=[],
            attributions_by_asset=[],
            attributions_by_market=[],
            attributions_by_regime=[],
            top_performers=[],
            worst_performers=[],
            recommendations=["Consider increasing momentum strategy allocation"],
        )

        assert report.total_pnl == 1500.0
        assert report.total_trades == 25

    def test_performance_report_to_dict(self):
        """Test PerformanceReport to_dict method."""
        now = datetime.now()
        report = PerformanceReport(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_pnl=1500.0,
            total_pnl_pct=0.15,
            total_trades=25,
            overall_win_rate=0.56,
            overall_sharpe=1.3,
            attributions_by_strategy=[],
            attributions_by_model=[],
            attributions_by_asset=[],
            attributions_by_market=[],
            attributions_by_regime=[],
            top_performers=[],
            worst_performers=[],
            recommendations=[],
        )

        dict_result = report.to_dict()

        assert "period_start" in dict_result
        assert "total_pnl" in dict_result
        assert dict_result["total_trades"] == 25


class TestPerformanceAttributor:
    """Tests for PerformanceAttributor class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def attributor(self, temp_data_dir):
        """Create attributor with temp directory."""
        return PerformanceAttributor(data_dir=temp_data_dir)

    def test_init_creates_directory(self, temp_data_dir):
        """Test initialization creates data directory."""
        new_dir = Path(temp_data_dir) / "new_attr_dir"
        attributor = PerformanceAttributor(data_dir=str(new_dir))
        assert new_dir.exists()

    def test_init_default_capital(self, attributor):
        """Test default initial capital."""
        assert attributor.initial_capital == 10000

    def test_init_custom_capital(self, temp_data_dir):
        """Test custom initial capital."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir, initial_capital=50000)
        assert attributor.initial_capital == 50000

    def test_log_trade_basic(self, attributor):
        """Test basic trade logging."""
        trade = attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            strategy="momentum",
        )

        assert trade.trade_id == "T001"
        assert trade.symbol == "BTC/USDT"
        assert len(attributor.trades) == 1

    def test_log_trade_calculates_pnl_buy(self, attributor):
        """Test PnL calculation for BUY trade."""
        trade = attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=52000,
        )

        # PnL = (52000 - 50000) * 0.1 = 200
        assert trade.pnl == 200.0

    def test_log_trade_calculates_pnl_sell(self, attributor):
        """Test PnL calculation for SELL trade."""
        trade = attributor.log_trade(
            trade_id="T002",
            symbol="BTC/USDT",
            action="SELL",
            quantity=0.1,
            entry_price=50000,
            exit_price=48000,
        )

        # PnL = (50000 - 48000) * 0.1 = 200
        assert trade.pnl == 200.0

    def test_log_trade_with_explicit_pnl(self, attributor):
        """Test logging with explicit PnL."""
        trade = attributor.log_trade(
            trade_id="T003",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            pnl=150.0,  # Override calculation
        )

        assert trade.pnl == 150.0

    def test_log_trade_holding_period(self, attributor):
        """Test holding period calculation."""
        entry = datetime.now()
        exit_time = entry + timedelta(hours=5)

        trade = attributor.log_trade(
            trade_id="T004",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            entry_time=entry,
            exit_time=exit_time,
        )

        assert abs(trade.holding_period_hours - 5.0) < 0.01

    def test_log_trade_saves_to_disk(self, attributor, temp_data_dir):
        """Test that trades are saved to disk."""
        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
        )

        trades_file = Path(temp_data_dir) / "trade_history.json"
        assert trades_file.exists()

        with open(trades_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_update_trade(self, attributor):
        """Test updating a trade."""
        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
        )

        updated = attributor.update_trade("T001", exit_price=52000)

        assert updated is not None
        assert updated.exit_price == 52000
        assert updated.pnl == 200.0

    def test_update_trade_not_found(self, attributor):
        """Test updating non-existent trade returns None."""
        result = attributor.update_trade("NONEXISTENT", exit_price=50000)
        assert result is None

    def test_generate_report_empty(self, attributor):
        """Test generating report with no trades."""
        report = attributor.generate_report(days=30)

        assert report.total_pnl == 0
        assert report.total_trades == 0
        assert report.overall_win_rate == 0

    def test_generate_report_with_trades(self, attributor):
        """Test generating report with trades."""
        now = datetime.now()

        # Log some trades
        for i in range(5):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=51000 if i < 3 else 49000,  # 3 wins, 2 losses
                strategy="momentum" if i < 3 else "mean_reversion",
                model="lstm",
                regime="trending",
                entry_time=now - timedelta(days=i),
            )

        report = attributor.generate_report(days=30)

        assert report.total_trades == 5
        assert report.overall_win_rate == 0.6

    def test_generate_report_filters_by_date(self, attributor):
        """Test that report filters trades by date."""
        now = datetime.now()

        # Trade within period
        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            entry_time=now - timedelta(days=5),
        )

        # Trade outside period
        attributor.log_trade(
            trade_id="T002",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            entry_time=now - timedelta(days=60),
        )

        report = attributor.generate_report(days=30)

        assert report.total_trades == 1

    def test_generate_report_attributions_by_strategy(self, attributor):
        """Test strategy attributions in report."""
        now = datetime.now()

        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=52000,
            strategy="momentum",
            entry_time=now,
        )

        attributor.log_trade(
            trade_id="T002",
            symbol="ETH/USDT",
            action="BUY",
            quantity=1.0,
            entry_price=3000,
            exit_price=3100,
            strategy="trend",
            entry_time=now,
        )

        report = attributor.generate_report(days=30)

        assert len(report.attributions_by_strategy) == 2
        strategies = [a.category for a in report.attributions_by_strategy]
        assert "momentum" in strategies
        assert "trend" in strategies

    def test_generate_report_top_performers(self, attributor):
        """Test top performers in report."""
        now = datetime.now()

        for i in range(10):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=50000 + (i * 100),  # Increasing profit
                entry_time=now,
            )

        report = attributor.generate_report(days=30)

        assert len(report.top_performers) == 5
        # Best trade should be first
        assert report.top_performers[0]["pnl"] >= report.top_performers[1]["pnl"]

    def test_generate_report_recommendations(self, attributor):
        """Test recommendations generation."""
        now = datetime.now()

        # Log many losing trades for a strategy
        for i in range(10):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=49000,  # All losses
                strategy="bad_strategy",
                entry_time=now,
            )

        report = attributor.generate_report(days=30)

        # Should have recommendations about poor strategy
        assert len(report.recommendations) > 0

    def test_get_daily_summary_no_trades(self, attributor):
        """Test daily summary with no trades."""
        summary = attributor.get_daily_summary()

        assert summary["trade_count"] == 0
        assert summary["total_pnl"] == 0

    def test_get_daily_summary_with_trades(self, attributor):
        """Test daily summary with trades."""
        today = date.today()

        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            entry_time=datetime.combine(today, datetime.min.time()),
        )

        attributor.log_trade(
            trade_id="T002",
            symbol="ETH/USDT",
            action="BUY",
            quantity=1.0,
            entry_price=3000,
            exit_price=2900,
            entry_time=datetime.combine(today, datetime.min.time()),
        )

        summary = attributor.get_daily_summary(target_date=today)

        assert summary["trade_count"] == 2
        assert summary["total_pnl"] == 0  # 100 - 100 = 0
        assert summary["win_rate"] == 0.5

    def test_get_daily_summary_by_strategy(self, attributor):
        """Test daily summary breakdown by strategy."""
        today = date.today()

        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=52000,
            strategy="momentum",
            entry_time=datetime.combine(today, datetime.min.time()),
        )

        attributor.log_trade(
            trade_id="T002",
            symbol="ETH/USDT",
            action="BUY",
            quantity=1.0,
            entry_price=3000,
            exit_price=3200,
            strategy="trend",
            entry_time=datetime.combine(today, datetime.min.time()),
        )

        summary = attributor.get_daily_summary(target_date=today)

        assert "momentum" in summary["by_strategy"]
        assert "trend" in summary["by_strategy"]

    def test_save_report(self, attributor, temp_data_dir):
        """Test saving report to disk."""
        now = datetime.now()
        report = PerformanceReport(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_pnl=1500.0,
            total_pnl_pct=0.15,
            total_trades=25,
            overall_win_rate=0.56,
            overall_sharpe=1.3,
            attributions_by_strategy=[],
            attributions_by_model=[],
            attributions_by_asset=[],
            attributions_by_market=[],
            attributions_by_regime=[],
            top_performers=[],
            worst_performers=[],
            recommendations=[],
        )

        filepath = attributor.save_report(report)

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["total_pnl"] == 1500.0

    def test_save_report_custom_filename(self, attributor, temp_data_dir):
        """Test saving report with custom filename."""
        now = datetime.now()
        report = PerformanceReport(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_pnl=1000.0,
            total_pnl_pct=0.1,
            total_trades=10,
            overall_win_rate=0.5,
            overall_sharpe=1.0,
            attributions_by_strategy=[],
            attributions_by_model=[],
            attributions_by_asset=[],
            attributions_by_market=[],
            attributions_by_regime=[],
            top_performers=[],
            worst_performers=[],
            recommendations=[],
        )

        filepath = attributor.save_report(report, filename="custom_report.json")

        assert filepath.name == "custom_report.json"

    def test_load_trades_from_disk(self, temp_data_dir):
        """Test loading trades from disk on init."""
        # Create trades file
        trades_data = [
            {
                "trade_id": "T001",
                "symbol": "BTC/USDT",
                "market_type": "crypto",
                "action": "BUY",
                "quantity": 0.1,
                "entry_price": 50000,
                "exit_price": 51000,
                "pnl": 100,
                "pnl_pct": 0.02,
                "strategy": "momentum",
                "model": "lstm",
                "regime": "trending",
                "confidence": 0.8,
                "sentiment_score": 0.5,
                "entry_time": datetime.now().isoformat(),
                "exit_time": datetime.now().isoformat(),
                "holding_period_hours": 1.0,
                "metadata": {},
            }
        ]

        trades_file = Path(temp_data_dir) / "trade_history.json"
        with open(trades_file, "w") as f:
            json.dump(trades_data, f)

        attributor = PerformanceAttributor(data_dir=temp_data_dir)

        assert len(attributor.trades) == 1
        assert attributor.trades[0].trade_id == "T001"


class TestAttributionCalculations:
    """Tests for attribution calculation logic."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def attributor_with_trades(self, temp_data_dir):
        """Create attributor with sample trades."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        now = datetime.now()

        # Momentum strategy - 3 wins, 1 loss
        for i in range(4):
            attributor.log_trade(
                trade_id=f"MOM{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=51000 if i < 3 else 49000,
                strategy="momentum",
                model="lstm",
                regime="trending",
                entry_time=now - timedelta(days=i),
                exit_time=now - timedelta(days=i) + timedelta(hours=2),
            )

        # Mean reversion strategy - 1 win, 2 losses
        for i in range(3):
            attributor.log_trade(
                trade_id=f"MR{i:03d}",
                symbol="ETH/USDT",
                action="BUY",
                quantity=1.0,
                entry_price=3000,
                exit_price=3100 if i < 1 else 2900,
                strategy="mean_reversion",
                model="random_forest",
                regime="ranging",
                entry_time=now - timedelta(days=i),
                exit_time=now - timedelta(days=i) + timedelta(hours=4),
            )

        return attributor

    def test_win_rate_calculation(self, attributor_with_trades):
        """Test win rate calculation per strategy."""
        report = attributor_with_trades.generate_report(days=30)

        # Find momentum attribution
        momentum_attr = next(
            (a for a in report.attributions_by_strategy if a.category == "momentum"), None
        )

        assert momentum_attr is not None
        assert momentum_attr.win_rate == 0.75  # 3 wins / 4 trades

    def test_pnl_contribution_calculation(self, attributor_with_trades):
        """Test PnL contribution calculation."""
        report = attributor_with_trades.generate_report(days=30)

        # Verify contributions are calculated for each strategy
        assert len(report.attributions_by_strategy) == 2

        # Each strategy should have a calculated contribution
        for attr in report.attributions_by_strategy:
            assert attr.pnl_contribution_pct is not None

    def test_profit_factor_calculation(self, attributor_with_trades):
        """Test profit factor calculation."""
        report = attributor_with_trades.generate_report(days=30)

        momentum_attr = next(
            (a for a in report.attributions_by_strategy if a.category == "momentum"), None
        )

        # Profit factor = sum(wins) / abs(sum(losses)) = 300 / 100 = 3
        assert momentum_attr.profit_factor == 3.0

    def test_holding_period_average(self, attributor_with_trades):
        """Test average holding period calculation."""
        report = attributor_with_trades.generate_report(days=30)

        momentum_attr = next(
            (a for a in report.attributions_by_strategy if a.category == "momentum"), None
        )

        # All momentum trades have 2 hour holding period
        assert abs(momentum_attr.avg_holding_period - 2.0) < 0.1


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_all_winning_trades(self, temp_data_dir):
        """Test with all winning trades."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        now = datetime.now()

        for i in range(5):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=51000,
                entry_time=now,
            )

        report = attributor.generate_report(days=30)

        assert report.overall_win_rate == 1.0

    def test_all_losing_trades(self, temp_data_dir):
        """Test with all losing trades."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        now = datetime.now()

        for i in range(5):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=49000,
                entry_time=now,
            )

        report = attributor.generate_report(days=30)

        assert report.overall_win_rate == 0.0

    def test_zero_pnl_trades(self, temp_data_dir):
        """Test with zero PnL trades."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        now = datetime.now()

        for i in range(5):
            attributor.log_trade(
                trade_id=f"T{i:03d}",
                symbol="BTC/USDT",
                action="BUY",
                quantity=0.1,
                entry_price=50000,
                exit_price=50000,  # No profit/loss
                entry_time=now,
            )

        report = attributor.generate_report(days=30)

        assert report.total_pnl == 0

    def test_single_trade(self, temp_data_dir):
        """Test with single trade."""
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        now = datetime.now()

        attributor.log_trade(
            trade_id="T001",
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            entry_price=50000,
            exit_price=52000,
            entry_time=now,
        )

        report = attributor.generate_report(days=30)

        assert report.total_trades == 1
        assert report.overall_win_rate == 1.0

    def test_corrupt_trades_file(self, temp_data_dir):
        """Test handling corrupt trades file."""
        trades_file = Path(temp_data_dir) / "trade_history.json"
        with open(trades_file, "w") as f:
            f.write("not valid json {")

        # Should not raise, just start with empty trades
        attributor = PerformanceAttributor(data_dir=temp_data_dir)
        assert len(attributor.trades) == 0
