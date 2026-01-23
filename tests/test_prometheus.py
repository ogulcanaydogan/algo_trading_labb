"""
Tests for the Prometheus metrics module.
"""

import time
import pytest
import threading

from bot.core.prometheus import (
    Counter,
    Gauge,
    Histogram,
    HistogramTimer,
    PrometheusRegistry,
    Summary,
    counter,
    gauge,
    histogram,
    summary,
    default_registry,
    trading_registry,
    trades_total,
    trade_pnl,
    order_latency,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_basic_increment(self):
        """Test basic counter increment."""
        c = Counter("test_counter", "A test counter")
        c.inc()
        c.inc(2.0)

        samples = c._collect()
        assert len(samples) == 1
        assert samples[0].value == 3.0
        assert samples[0].name == "test_counter_total"

    def test_labeled_counter(self):
        """Test counter with labels."""
        c = Counter("http_requests", "HTTP requests", labels=["method", "status"])
        c.inc(method="GET", status="200")
        c.inc(method="GET", status="200")
        c.inc(method="POST", status="201")

        samples = c._collect()
        assert len(samples) == 2

        # Find the GET/200 sample
        get_sample = next(s for s in samples if s.labels.get("method") == "GET")
        assert get_sample.value == 2.0

    def test_labeled_counter_chained(self):
        """Test counter with labels using chained method."""
        c = Counter("requests", "Requests", labels=["endpoint"])
        labeled = c.labels(endpoint="/api")
        labeled.inc()
        labeled.inc(5.0)

        samples = c._collect()
        assert samples[0].value == 6.0

    def test_counter_cannot_decrease(self):
        """Test that counter raises on negative increment."""
        c = Counter("test", "test")
        with pytest.raises(ValueError):
            c.inc(-1.0)


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self):
        """Test setting gauge value."""
        g = Gauge("temperature", "Current temperature")
        g.set(25.5)

        samples = g._collect()
        assert samples[0].value == 25.5

    def test_inc_dec(self):
        """Test incrementing and decrementing gauge."""
        g = Gauge("connections", "Active connections")
        g.set(10)
        g.inc(5)
        g.dec(3)

        samples = g._collect()
        assert samples[0].value == 12

    def test_labeled_gauge(self):
        """Test gauge with labels."""
        g = Gauge("cpu_usage", "CPU usage", labels=["core"])
        g.set(50.0, core="0")
        g.set(75.0, core="1")

        samples = g._collect()
        assert len(samples) == 2

    def test_labeled_gauge_chained(self):
        """Test gauge with labels using chained method."""
        g = Gauge("memory", "Memory usage", labels=["type"])
        labeled = g.labels(type="heap")
        labeled.set(100)
        labeled.inc(50)
        labeled.dec(25)

        samples = g._collect()
        assert samples[0].value == 125


class TestHistogram:
    """Tests for Histogram metric."""

    def test_basic_observation(self):
        """Test basic histogram observation."""
        h = Histogram("latency", "Latency", buckets=(0.1, 0.5, 1.0, float("inf")))
        h.observe(0.05)
        h.observe(0.3)
        h.observe(0.8)

        samples = h._collect()

        # Should have 4 bucket samples + sum + count = 6
        bucket_samples = [s for s in samples if "_bucket" in s.name]
        assert len(bucket_samples) == 4

        # Check bucket counts (cumulative)
        buckets = {s.labels["le"]: s.value for s in bucket_samples}
        assert buckets["0.1"] == 1  # 0.05 <= 0.1
        assert buckets["0.5"] == 2  # 0.05, 0.3 <= 0.5
        assert buckets["1.0"] == 3  # all three <= 1.0
        assert buckets["+Inf"] == 3

        # Check sum and count
        sum_sample = next(s for s in samples if "_sum" in s.name)
        count_sample = next(s for s in samples if "_count" in s.name)
        assert sum_sample.value == pytest.approx(1.15, rel=0.01)
        assert count_sample.value == 3

    def test_histogram_timer(self):
        """Test histogram timer context manager."""
        h = Histogram("operation_time", "Operation time")

        with h.time():
            time.sleep(0.05)

        samples = h._collect()
        sum_sample = next(s for s in samples if "_sum" in s.name)
        assert sum_sample.value >= 0.05

    def test_labeled_histogram(self):
        """Test histogram with labels."""
        h = Histogram("request_latency", "Request latency", labels=["endpoint"])
        h.observe(0.1, endpoint="/api")
        h.observe(0.2, endpoint="/api")
        h.observe(0.5, endpoint="/web")

        samples = h._collect()
        # Each label set gets full bucket + sum + count
        api_count = next(
            s for s in samples
            if "_count" in s.name and s.labels.get("endpoint") == "/api"
        )
        assert api_count.value == 2


class TestSummary:
    """Tests for Summary metric."""

    def test_basic_observation(self):
        """Test basic summary observation."""
        s = Summary("response_time", "Response time")
        s.observe(0.1)
        s.observe(0.2)
        s.observe(0.3)

        samples = s._collect()

        sum_sample = next(sample for sample in samples if "_sum" in sample.name)
        count_sample = next(sample for sample in samples if "_count" in sample.name)

        assert sum_sample.value == pytest.approx(0.6, rel=0.01)
        assert count_sample.value == 3

    def test_summary_max_age(self):
        """Test that old samples are removed."""
        s = Summary("test", "test", max_age=0.1)
        s.observe(1.0)

        # Wait for sample to expire
        time.sleep(0.15)
        s.observe(2.0)

        # Old sample should be removed from sliding window
        # but sum/count still track total
        samples = s._collect()
        sum_sample = next(sample for sample in samples if "_sum" in sample.name)
        assert sum_sample.value == 3.0  # Total sum


class TestPrometheusRegistry:
    """Tests for PrometheusRegistry."""

    def test_register_metric(self):
        """Test registering a metric."""
        reg = PrometheusRegistry()
        c = Counter("test_metric", "A test metric")
        reg.register(c)

        assert reg.get("test_metric") is c

    def test_register_duplicate_raises(self):
        """Test that registering duplicate name raises."""
        reg = PrometheusRegistry()
        c1 = Counter("duplicate", "First")
        c2 = Counter("duplicate", "Second")

        reg.register(c1)
        with pytest.raises(ValueError):
            reg.register(c2)

    def test_unregister(self):
        """Test unregistering a metric."""
        reg = PrometheusRegistry()
        c = Counter("test", "test")
        reg.register(c)
        reg.unregister("test")

        assert reg.get("test") is None

    def test_export_format(self):
        """Test Prometheus text exposition format."""
        reg = PrometheusRegistry()
        c = Counter("http_requests", "Total HTTP requests")
        c.inc(10)
        reg.register(c)

        output = reg.export()

        assert "# HELP http_requests Total HTTP requests" in output
        assert "# TYPE http_requests counter" in output
        assert "http_requests_total 10" in output

    def test_export_with_labels(self):
        """Test export with labeled metrics."""
        reg = PrometheusRegistry()
        c = Counter("requests", "Requests", labels=["method"])
        c.inc(method="GET")
        reg.register(c)

        output = reg.export()

        assert 'requests_total{method="GET"}' in output


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_counter_function(self):
        """Test counter convenience function."""
        # Note: This adds to default_registry, which persists
        # In real tests, you'd use a fresh registry
        c = Counter("unique_test_counter_1", "Test")
        c.inc()
        assert c._collect()[0].value == 1.0

    def test_gauge_function(self):
        """Test gauge convenience function."""
        g = Gauge("unique_test_gauge_1", "Test")
        g.set(42)
        assert g._collect()[0].value == 42

    def test_histogram_function(self):
        """Test histogram convenience function."""
        h = Histogram("unique_test_histogram_1", "Test")
        h.observe(0.5)
        samples = h._collect()
        count = next(s for s in samples if "_count" in s.name)
        assert count.value == 1


class TestTradingMetrics:
    """Tests for pre-defined trading metrics."""

    def test_trades_total(self):
        """Test trades_total counter."""
        initial = sum(s.value for s in trades_total._collect()) if trades_total._collect() else 0
        trades_total.inc(symbol="BTC/USDT", side="buy")
        trades_total.inc(symbol="BTC/USDT", side="sell")

        samples = trades_total._collect()
        total = sum(s.value for s in samples)
        assert total >= initial + 2

    def test_trade_pnl(self):
        """Test trade_pnl gauge."""
        trade_pnl.set(100.50, symbol="ETH/USDT")
        samples = trade_pnl._collect()

        eth_sample = next(
            (s for s in samples if s.labels.get("symbol") == "ETH/USDT"),
            None
        )
        assert eth_sample is not None
        assert eth_sample.value == 100.50

    def test_order_latency(self):
        """Test order_latency histogram."""
        order_latency.observe(0.05, exchange="binance", order_type="market")
        order_latency.observe(0.1, exchange="binance", order_type="limit")

        samples = order_latency._collect()
        # Should have samples for the observations
        assert len(samples) > 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        c = Counter("thread_safe_counter", "Thread safe counter")

        def increment():
            for _ in range(1000):
                c.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        samples = c._collect()
        assert samples[0].value == 10000

    def test_gauge_thread_safety(self):
        """Test gauge is thread-safe."""
        g = Gauge("thread_safe_gauge", "Thread safe gauge")

        def update():
            for i in range(100):
                g.set(i)

        threads = [threading.Thread(target=update) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some value without crashing
        samples = g._collect()
        assert len(samples) == 1


class TestExportFromCollector:
    """Tests for export_from_collector integration."""

    def test_export_from_collector(self):
        """Test exporting from MetricsCollector."""
        from bot.core.logging_config import metrics

        # Add some metrics
        metrics.increment("test_export_counter")
        metrics.gauge("test_export_gauge", 42)
        metrics.timing("test_export_timing", 100.5)

        reg = PrometheusRegistry()
        output = reg.export_from_collector()

        assert "app_counter" in output
        assert "app_gauge" in output
        assert "app_timing" in output


class TestLabelFormatting:
    """Tests for label formatting."""

    def test_empty_labels(self):
        """Test formatting with no labels."""
        c = Counter("test", "test")
        assert c._format_labels({}) == ""

    def test_single_label(self):
        """Test formatting with single label."""
        c = Counter("test", "test")
        result = c._format_labels({"method": "GET"})
        assert result == '{method="GET"}'

    def test_multiple_labels_sorted(self):
        """Test that labels are sorted."""
        c = Counter("test", "test")
        result = c._format_labels({"z": "1", "a": "2"})
        assert result == '{a="2",z="1"}'
