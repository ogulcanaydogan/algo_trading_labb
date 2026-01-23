"""Tests for structured logging configuration."""

import asyncio
import logging
import time

import pytest

from bot.core.logging_config import (
    MetricsCollector,
    OperationMetrics,
    TradeLogEntry,
    get_logger,
    log_operation,
    log_timed,
    log_timed_async,
    setup_logging,
)


class TestTradeLogEntry:
    """Tests for TradeLogEntry."""

    def test_basic_entry(self):
        entry = TradeLogEntry(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000,
        )

        data = entry.to_dict()
        assert data["symbol"] == "BTC/USDT"
        assert data["side"] == "buy"
        assert data["quantity"] == 0.1
        assert data["price"] == 50000
        assert data["notional"] == 5000

    def test_entry_with_pnl(self):
        entry = TradeLogEntry(
            symbol="ETH/USDT",
            side="sell",
            quantity=1.0,
            price=3000,
            pnl=150,
            pnl_pct=5.0,
        )

        data = entry.to_dict()
        assert data["pnl"] == 150
        assert data["pnl_pct"] == 5.0

    def test_entry_with_extra(self):
        entry = TradeLogEntry(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000,
            extra={"reason": "signal", "confidence": 0.8},
        )

        data = entry.to_dict()
        assert data["reason"] == "signal"
        assert data["confidence"] == 0.8


class TestOperationMetrics:
    """Tests for OperationMetrics."""

    def test_metrics_to_dict(self):
        metrics = OperationMetrics(
            name="fetch_data",
            duration_ms=123.45,
            success=True,
        )

        data = metrics.to_dict()
        assert data["operation"] == "fetch_data"
        assert data["duration_ms"] == 123.45
        assert data["success"] is True


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_successful_operation(self):
        with log_operation("test_op") as metrics:
            time.sleep(0.01)

        assert metrics.success is True
        assert metrics.duration_ms >= 10
        assert metrics.error is None

    def test_failed_operation(self):
        with pytest.raises(ValueError):
            with log_operation("failing_op") as metrics:
                raise ValueError("Test error")

        assert metrics.success is False
        assert "Test error" in metrics.error

    def test_records_timing(self):
        start = time.perf_counter()
        with log_operation("timed_op") as metrics:
            time.sleep(0.02)
        elapsed = (time.perf_counter() - start) * 1000

        assert metrics.duration_ms >= 20
        assert metrics.duration_ms <= elapsed + 5


class TestLogTimed:
    """Tests for log_timed decorator."""

    def test_timed_function(self):
        @log_timed()
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

    def test_preserves_function_metadata(self):
        @log_timed()
        def documented_function():
            """This is documented."""
            return True

        assert documented_function.__name__ == "documented_function"
        assert "documented" in documented_function.__doc__


class TestLogTimedAsync:
    """Tests for log_timed_async decorator."""

    @pytest.mark.asyncio
    async def test_async_timed_function(self):
        @log_timed_async()
        async def async_slow():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await async_slow()
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_timed_exception(self):
        @log_timed_async()
        async def async_fail():
            raise ValueError("Async fail")

        with pytest.raises(ValueError):
            await async_fail()


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_increment_counter(self):
        collector = MetricsCollector()
        collector.increment("requests")
        collector.increment("requests")
        collector.increment("requests", 3)

        data = collector.get_all()
        assert data["counters"]["requests"] == 5

    def test_gauge(self):
        collector = MetricsCollector()
        collector.gauge("memory_mb", 512)
        collector.gauge("memory_mb", 1024)

        data = collector.get_all()
        assert data["gauges"]["memory_mb"] == 1024

    def test_timing(self):
        collector = MetricsCollector()
        collector.timing("latency", 10)
        collector.timing("latency", 20)
        collector.timing("latency", 30)

        data = collector.get_all()
        assert data["timings"]["latency"]["count"] == 3
        assert data["timings"]["latency"]["avg_ms"] == 20
        assert data["timings"]["latency"]["min_ms"] == 10
        assert data["timings"]["latency"]["max_ms"] == 30

    def test_reset(self):
        collector = MetricsCollector()
        collector.increment("count")
        collector.gauge("value", 100)
        collector.timing("time", 50)

        collector.reset()
        data = collector.get_all()

        assert data["counters"] == {}
        assert data["gauges"] == {}
        assert data["timings"] == {}


class TestGetLogger:
    """Tests for get_logger."""

    def test_basic_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logger_with_extra(self):
        logger = get_logger("test_module", component="api", version="1.0")
        # Should return LoggerAdapter with extra context
        assert hasattr(logger, "extra")
        assert logger.extra["component"] == "api"


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_sets_level(self):
        setup_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

        # Reset
        setup_logging(level="INFO")

    def test_adds_handler(self):
        setup_logging(level="INFO")
        root = logging.getLogger()
        assert len(root.handlers) >= 1
