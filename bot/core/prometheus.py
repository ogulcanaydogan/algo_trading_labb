"""
Prometheus Metrics Export Module.

Provides Prometheus-compatible metrics endpoint for production monitoring:
- Counter, Gauge, Histogram, Summary metrics
- HTTP endpoint handler for /metrics
- Integration with existing MetricsCollector

Usage:
    from bot.core.prometheus import PrometheusRegistry, Counter, Gauge, Histogram

    # Create metrics
    requests = Counter("http_requests_total", "Total HTTP requests")
    active_trades = Gauge("active_trades", "Number of active trades")
    order_latency = Histogram("order_latency_seconds", "Order execution latency")

    # Update metrics
    requests.inc()
    active_trades.set(5)
    order_latency.observe(0.123)

    # Export metrics
    registry = PrometheusRegistry()
    output = registry.export()
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_config import metrics as metrics_collector


@dataclass
class MetricSample:
    """A single metric sample."""

    name: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None


class Metric(ABC):
    """Base class for Prometheus metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize a metric.

        Args:
            name: Metric name (must match [a-zA-Z_:][a-zA-Z0-9_:]*)
            description: Human-readable description
            labels: List of label names
        """
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._lock = threading.Lock()

    @abstractmethod
    def _collect(self) -> List[MetricSample]:
        """Collect all samples for this metric."""
        pass

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """Return Prometheus metric type."""
        pass

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"


class Counter(Metric):
    """
    A counter metric that only goes up.

    Counters track cumulative values, like total requests.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)

    @property
    def metric_type(self) -> str:
        return "counter"

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """
        Increment the counter.

        Args:
            amount: Amount to increment by (must be positive)
            **labels: Label values
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented")

        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += amount

    def labels(self, **labels: str) -> "LabeledCounter":
        """Return a labeled counter for method chaining."""
        return LabeledCounter(self, labels)

    def _collect(self) -> List[MetricSample]:
        samples = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key)
                samples.append(MetricSample(
                    name=self.name + "_total",
                    labels=labels,
                    value=value,
                ))
        return samples


class LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: Dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, amount: float = 1.0) -> None:
        self._counter.inc(amount, **self._labels)


class Gauge(Metric):
    """
    A gauge metric that can go up or down.

    Gauges track current values, like active connections.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, labels)
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)

    @property
    def metric_type(self) -> str:
        return "gauge"

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge to a value."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment the gauge."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] += amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        """Decrement the gauge."""
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._values[label_key] -= amount

    def labels(self, **labels: str) -> "LabeledGauge":
        """Return a labeled gauge for method chaining."""
        return LabeledGauge(self, labels)

    def _collect(self) -> List[MetricSample]:
        samples = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key)
                samples.append(MetricSample(
                    name=self.name,
                    labels=labels,
                    value=value,
                ))
        return samples


class LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: Dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        self._gauge.set(value, **self._labels)

    def inc(self, amount: float = 1.0) -> None:
        self._gauge.inc(amount, **self._labels)

    def dec(self, amount: float = 1.0) -> None:
        self._gauge.dec(amount, **self._labels)


class Histogram(Metric):
    """
    A histogram metric for tracking distributions.

    Histograms track values in configurable buckets plus sum and count.
    """

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, labels)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[Tuple[str, ...], Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self._buckets}
        )
        self._sums: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._counts: Dict[Tuple[str, ...], int] = defaultdict(int)

    @property
    def metric_type(self) -> str:
        return "histogram"

    def observe(self, value: float, **labels: str) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
            **labels: Label values
        """
        label_key = tuple(sorted(labels.items()))
        with self._lock:
            self._sums[label_key] += value
            self._counts[label_key] += 1
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

    def time(self, **labels: str) -> "HistogramTimer":
        """Return a context manager for timing operations."""
        return HistogramTimer(self, labels)

    def _collect(self) -> List[MetricSample]:
        samples = []
        with self._lock:
            for label_key in set(self._counts.keys()):
                labels = dict(label_key)

                # Bucket samples (already cumulative from observe method)
                for bucket in self._buckets:
                    bucket_labels = {**labels, "le": str(bucket) if bucket != float("inf") else "+Inf"}
                    samples.append(MetricSample(
                        name=f"{self.name}_bucket",
                        labels=bucket_labels,
                        value=self._bucket_counts[label_key].get(bucket, 0),
                    ))

                # Sum and count
                samples.append(MetricSample(
                    name=f"{self.name}_sum",
                    labels=labels,
                    value=self._sums[label_key],
                ))
                samples.append(MetricSample(
                    name=f"{self.name}_count",
                    labels=labels,
                    value=self._counts[label_key],
                ))

        return samples


class HistogramTimer:
    """Context manager for timing histogram observations."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "HistogramTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self._histogram.observe(duration, **self._labels)


class Summary(Metric):
    """
    A summary metric for tracking distributions.

    Summaries calculate quantiles over a sliding time window.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        max_age: float = 60.0,
        max_samples: int = 1000,
    ):
        super().__init__(name, description, labels)
        self._max_age = max_age
        self._max_samples = max_samples
        self._samples: Dict[Tuple[str, ...], List[Tuple[float, float]]] = defaultdict(list)
        self._sums: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._counts: Dict[Tuple[str, ...], int] = defaultdict(int)

    @property
    def metric_type(self) -> str:
        return "summary"

    def observe(self, value: float, **labels: str) -> None:
        """Observe a value."""
        label_key = tuple(sorted(labels.items()))
        now = time.time()

        with self._lock:
            self._sums[label_key] += value
            self._counts[label_key] += 1

            # Add sample with timestamp
            samples = self._samples[label_key]
            samples.append((now, value))

            # Remove old samples
            cutoff = now - self._max_age
            self._samples[label_key] = [
                (t, v) for t, v in samples
                if t > cutoff
            ][-self._max_samples:]

    def _collect(self) -> List[MetricSample]:
        samples = []
        with self._lock:
            for label_key in set(self._counts.keys()):
                labels = dict(label_key)

                # Sum and count
                samples.append(MetricSample(
                    name=f"{self.name}_sum",
                    labels=labels,
                    value=self._sums[label_key],
                ))
                samples.append(MetricSample(
                    name=f"{self.name}_count",
                    labels=labels,
                    value=self._counts[label_key],
                ))

        return samples


class PrometheusRegistry:
    """
    Registry for Prometheus metrics.

    Collects and exports metrics in Prometheus text format.
    """

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def register(self, metric: Metric) -> Metric:
        """
        Register a metric.

        Args:
            metric: Metric to register

        Returns:
            The registered metric
        """
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"Metric {metric.name} already registered")
            self._metrics[metric.name] = metric
        return metric

    def unregister(self, name: str) -> None:
        """Unregister a metric by name."""
        with self._lock:
            self._metrics.pop(name, None)

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def export(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            Prometheus text exposition format string
        """
        lines = []

        with self._lock:
            for name, metric in sorted(self._metrics.items()):
                # Add HELP and TYPE
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric.metric_type}")

                # Add samples
                for sample in metric._collect():
                    label_str = metric._format_labels(sample.labels)
                    lines.append(f"{sample.name}{label_str} {sample.value}")

                lines.append("")

        return "\n".join(lines)

    def export_from_collector(self) -> str:
        """
        Export metrics from the MetricsCollector.

        Converts internal metrics to Prometheus format.
        """
        lines = []

        # Export counters
        counters = metrics_collector.get_counters()
        if counters:
            lines.append("# HELP app_counter Application counters")
            lines.append("# TYPE app_counter counter")
            for name, value in sorted(counters.items()):
                safe_name = name.replace("-", "_").replace(".", "_")
                lines.append(f'app_counter{{name="{safe_name}"}} {value}')
            lines.append("")

        # Export gauges
        gauges = metrics_collector.get_gauges()
        if gauges:
            lines.append("# HELP app_gauge Application gauges")
            lines.append("# TYPE app_gauge gauge")
            for name, value in sorted(gauges.items()):
                safe_name = name.replace("-", "_").replace(".", "_")
                lines.append(f'app_gauge{{name="{safe_name}"}} {value}')
            lines.append("")

        # Export timings as summary
        timings = metrics_collector.get_timings()
        if timings:
            lines.append("# HELP app_timing_sum Application timing sums in ms")
            lines.append("# TYPE app_timing_sum gauge")
            lines.append("# HELP app_timing_count Application timing counts")
            lines.append("# TYPE app_timing_count counter")
            for name, values in sorted(timings.items()):
                safe_name = name.replace("-", "_").replace(".", "_")
                if values:
                    total = sum(values)
                    count = len(values)
                    lines.append(f'app_timing_sum{{name="{safe_name}"}} {total}')
                    lines.append(f'app_timing_count{{name="{safe_name}"}} {count}')
            lines.append("")

        return "\n".join(lines)


# Default registry
default_registry = PrometheusRegistry()


def counter(
    name: str,
    description: str,
    labels: Optional[List[str]] = None,
) -> Counter:
    """Create and register a counter."""
    metric = Counter(name, description, labels)
    default_registry.register(metric)
    return metric


def gauge(
    name: str,
    description: str,
    labels: Optional[List[str]] = None,
) -> Gauge:
    """Create and register a gauge."""
    metric = Gauge(name, description, labels)
    default_registry.register(metric)
    return metric


def histogram(
    name: str,
    description: str,
    labels: Optional[List[str]] = None,
    buckets: Optional[Tuple[float, ...]] = None,
) -> Histogram:
    """Create and register a histogram."""
    metric = Histogram(name, description, labels, buckets)
    default_registry.register(metric)
    return metric


def summary(
    name: str,
    description: str,
    labels: Optional[List[str]] = None,
) -> Summary:
    """Create and register a summary."""
    metric = Summary(name, description, labels)
    default_registry.register(metric)
    return metric


# Pre-defined trading metrics
trading_registry = PrometheusRegistry()

# Trade metrics
trades_total = Counter("trades_total", "Total number of trades executed", ["symbol", "side"])
trade_pnl = Gauge("trade_pnl_dollars", "Realized P&L in dollars", ["symbol"])
position_size = Gauge("position_size", "Current position size", ["symbol"])
order_latency = Histogram(
    "order_latency_seconds",
    "Order execution latency",
    ["exchange", "order_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

# System metrics
data_fetch_errors = Counter("data_fetch_errors_total", "Data fetch errors", ["provider", "symbol"])
model_predictions = Counter("model_predictions_total", "ML model predictions", ["model", "signal"])
signal_strength = Gauge("signal_strength", "Current signal strength", ["symbol", "strategy"])

# Register to trading registry
for m in [trades_total, trade_pnl, position_size, order_latency,
          data_fetch_errors, model_predictions, signal_strength]:
    trading_registry.register(m)
