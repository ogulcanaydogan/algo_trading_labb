"""
Performance Benchmarks - Latency, Throughput, and System Metrics.

Provides comprehensive performance measurement and benchmarking
for trading system components.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import threading

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class LatencyStats:
    """Latency statistics."""
    name: str
    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p75_ms": round(self.p75_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "p999_ms": round(self.p999_ms, 3),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    name: str
    total_operations: int
    duration_seconds: float
    ops_per_second: float
    bytes_processed: int = 0
    bytes_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_operations": self.total_operations,
            "duration_seconds": round(self.duration_seconds, 3),
            "ops_per_second": round(self.ops_per_second, 2),
            "bytes_processed": self.bytes_processed,
            "bytes_per_second": round(self.bytes_per_second, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    name: str
    description: str
    latency: Optional[LatencyStats] = None
    throughput: Optional[ThroughputStats] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    passed: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "latency": self.latency.to_dict() if self.latency else None,
            "throughput": self.throughput.to_dict() if self.throughput else None,
            "memory_mb": round(self.memory_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "passed": self.passed,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class LatencyTracker:
    """Track latency measurements for operations."""

    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self.max_samples = max_samples
        self._samples: List[float] = []
        self._lock = threading.Lock()

    def record(self, latency_ms: float):
        """Record a latency measurement."""
        with self._lock:
            self._samples.append(latency_ms)
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples:]

    @contextmanager
    def measure(self):
        """Context manager to measure operation latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.record(latency_ms)

    def get_stats(self) -> Optional[LatencyStats]:
        """Calculate latency statistics."""
        with self._lock:
            if not self._samples:
                return None

            samples = sorted(self._samples)
            n = len(samples)

            def percentile(p: float) -> float:
                idx = int(n * p / 100)
                return samples[min(idx, n - 1)]

            return LatencyStats(
                name=self.name,
                count=n,
                mean_ms=statistics.mean(samples),
                median_ms=statistics.median(samples),
                std_ms=statistics.stdev(samples) if n > 1 else 0,
                min_ms=samples[0],
                max_ms=samples[-1],
                p50_ms=percentile(50),
                p75_ms=percentile(75),
                p90_ms=percentile(90),
                p95_ms=percentile(95),
                p99_ms=percentile(99),
                p999_ms=percentile(99.9),
            )

    def reset(self):
        """Reset all samples."""
        with self._lock:
            self._samples.clear()


class ThroughputTracker:
    """Track throughput measurements."""

    def __init__(self, name: str, window_seconds: float = 60.0):
        self.name = name
        self.window_seconds = window_seconds
        self._operations: List[tuple[float, int]] = []  # (timestamp, bytes)
        self._lock = threading.Lock()

    def record(self, bytes_processed: int = 0):
        """Record an operation."""
        now = time.time()
        with self._lock:
            self._operations.append((now, bytes_processed))
            # Remove old entries
            cutoff = now - self.window_seconds
            self._operations = [(t, b) for t, b in self._operations if t > cutoff]

    def get_stats(self) -> ThroughputStats:
        """Calculate throughput statistics."""
        with self._lock:
            if not self._operations:
                return ThroughputStats(
                    name=self.name,
                    total_operations=0,
                    duration_seconds=0,
                    ops_per_second=0,
                )

            now = time.time()
            total_ops = len(self._operations)
            total_bytes = sum(b for _, b in self._operations)

            if total_ops > 1:
                first_ts = self._operations[0][0]
                duration = now - first_ts
            else:
                duration = self.window_seconds

            duration = max(duration, 0.001)  # Avoid division by zero

            return ThroughputStats(
                name=self.name,
                total_operations=total_ops,
                duration_seconds=duration,
                ops_per_second=total_ops / duration,
                bytes_processed=total_bytes,
                bytes_per_second=total_bytes / duration,
            )

    def reset(self):
        """Reset all measurements."""
        with self._lock:
            self._operations.clear()


def latency_benchmark(tracker: LatencyTracker):
    """Decorator to benchmark function latency."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracker.measure():
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def async_latency_benchmark(tracker: LatencyTracker):
    """Decorator to benchmark async function latency."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                tracker.record(latency_ms)
        return wrapper  # type: ignore
    return decorator


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    timeout_seconds: float = 60.0
    gc_before: bool = True
    latency_threshold_ms: Optional[float] = None
    throughput_threshold_ops: Optional[float] = None


class PerformanceBenchmark:
    """
    Performance benchmarking framework.

    Measures:
    - Function/operation latency
    - System throughput
    - Memory usage
    - Comparative benchmarks
    """

    def __init__(self):
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        self._throughput_trackers: Dict[str, ThroughputTracker] = {}
        self._results: List[BenchmarkResult] = []

    def get_latency_tracker(self, name: str) -> LatencyTracker:
        """Get or create a latency tracker."""
        if name not in self._latency_trackers:
            self._latency_trackers[name] = LatencyTracker(name)
        return self._latency_trackers[name]

    def get_throughput_tracker(self, name: str) -> ThroughputTracker:
        """Get or create a throughput tracker."""
        if name not in self._throughput_trackers:
            self._throughput_trackers[name] = ThroughputTracker(name)
        return self._throughput_trackers[name]

    def run_latency_benchmark(
        self,
        name: str,
        func: Callable,
        config: Optional[BenchmarkConfig] = None,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a latency benchmark on a function.

        Args:
            name: Benchmark name
            func: Function to benchmark
            config: Benchmark configuration
            *args, **kwargs: Arguments to pass to function
        """
        config = config or BenchmarkConfig()
        tracker = LatencyTracker(name)

        try:
            # Warmup
            for _ in range(config.warmup_iterations):
                func(*args, **kwargs)

            # Collect garbage before benchmark
            if config.gc_before:
                gc.collect()

            # Run benchmark
            for _ in range(config.benchmark_iterations):
                with tracker.measure():
                    func(*args, **kwargs)

            stats = tracker.get_stats()

            # Check threshold
            passed = True
            if config.latency_threshold_ms and stats:
                passed = stats.p95_ms <= config.latency_threshold_ms

            result = BenchmarkResult(
                name=name,
                description=f"Latency benchmark: {func.__name__}",
                latency=stats,
                passed=passed,
                metadata={
                    "warmup_iterations": config.warmup_iterations,
                    "benchmark_iterations": config.benchmark_iterations,
                    "threshold_ms": config.latency_threshold_ms,
                }
            )

        except Exception as e:
            result = BenchmarkResult(
                name=name,
                description=f"Latency benchmark: {func.__name__}",
                passed=False,
                error=str(e),
            )

        self._results.append(result)
        return result

    async def run_async_latency_benchmark(
        self,
        name: str,
        func: Callable,
        config: Optional[BenchmarkConfig] = None,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Run a latency benchmark on an async function."""
        config = config or BenchmarkConfig()
        tracker = LatencyTracker(name)

        try:
            # Warmup
            for _ in range(config.warmup_iterations):
                await func(*args, **kwargs)

            if config.gc_before:
                gc.collect()

            # Benchmark
            for _ in range(config.benchmark_iterations):
                start = time.perf_counter()
                await func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                tracker.record(latency_ms)

            stats = tracker.get_stats()

            passed = True
            if config.latency_threshold_ms and stats:
                passed = stats.p95_ms <= config.latency_threshold_ms

            result = BenchmarkResult(
                name=name,
                description=f"Async latency benchmark: {func.__name__}",
                latency=stats,
                passed=passed,
                metadata={
                    "warmup_iterations": config.warmup_iterations,
                    "benchmark_iterations": config.benchmark_iterations,
                }
            )

        except Exception as e:
            result = BenchmarkResult(
                name=name,
                description=f"Async latency benchmark: {func.__name__}",
                passed=False,
                error=str(e),
            )

        self._results.append(result)
        return result

    def run_throughput_benchmark(
        self,
        name: str,
        func: Callable,
        duration_seconds: float = 10.0,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a throughput benchmark.

        Runs the function as many times as possible within duration.
        """
        tracker = ThroughputTracker(name, window_seconds=duration_seconds)

        try:
            gc.collect()
            start_time = time.time()
            end_time = start_time + duration_seconds

            while time.time() < end_time:
                result = func(*args, **kwargs)
                # Try to get bytes processed if available
                bytes_processed = 0
                if isinstance(result, (bytes, bytearray)):
                    bytes_processed = len(result)
                elif isinstance(result, str):
                    bytes_processed = len(result.encode())
                tracker.record(bytes_processed)

            stats = tracker.get_stats()

            result = BenchmarkResult(
                name=name,
                description=f"Throughput benchmark: {func.__name__}",
                throughput=stats,
                passed=True,
                metadata={"duration_seconds": duration_seconds}
            )

        except Exception as e:
            result = BenchmarkResult(
                name=name,
                description=f"Throughput benchmark: {func.__name__}",
                passed=False,
                error=str(e),
            )

        self._results.append(result)
        return result

    def compare_implementations(
        self,
        name: str,
        implementations: Dict[str, Callable],
        config: Optional[BenchmarkConfig] = None,
        *args,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple implementations.

        Returns results sorted by performance.
        """
        results = {}

        for impl_name, func in implementations.items():
            benchmark_name = f"{name}:{impl_name}"
            results[impl_name] = self.run_latency_benchmark(
                benchmark_name, func, config, *args, **kwargs
            )

        return results

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get all current statistics."""
        return {
            "latency": {
                name: tracker.get_stats().to_dict() if tracker.get_stats() else None
                for name, tracker in self._latency_trackers.items()
            },
            "throughput": {
                name: tracker.get_stats().to_dict()
                for name, tracker in self._throughput_trackers.items()
            },
            "benchmark_results": [r.to_dict() for r in self._results[-100:]],
        }

    def reset_all(self):
        """Reset all trackers and results."""
        for tracker in self._latency_trackers.values():
            tracker.reset()
        for tracker in self._throughput_trackers.values():
            tracker.reset()
        self._results.clear()


class TradingSystemBenchmarks:
    """
    Pre-built benchmarks for trading system components.

    Includes benchmarks for:
    - Order processing latency
    - Signal generation
    - Risk calculations
    - Data pipeline throughput
    """

    def __init__(self, benchmark: Optional[PerformanceBenchmark] = None):
        self.benchmark = benchmark or PerformanceBenchmark()
        self._thresholds = {
            "order_latency_ms": 10.0,
            "signal_latency_ms": 50.0,
            "risk_calc_latency_ms": 100.0,
            "data_throughput_ops": 1000.0,
        }

    def set_thresholds(self, thresholds: Dict[str, float]):
        """Update performance thresholds."""
        self._thresholds.update(thresholds)

    def benchmark_order_processing(
        self,
        order_func: Callable,
        sample_orders: List[Any]
    ) -> BenchmarkResult:
        """Benchmark order processing latency."""
        config = BenchmarkConfig(
            warmup_iterations=5,
            benchmark_iterations=len(sample_orders),
            latency_threshold_ms=self._thresholds["order_latency_ms"],
        )

        tracker = LatencyTracker("order_processing")

        for order in sample_orders[:config.warmup_iterations]:
            order_func(order)

        gc.collect()

        for order in sample_orders:
            with tracker.measure():
                order_func(order)

        stats = tracker.get_stats()
        passed = stats.p95_ms <= config.latency_threshold_ms if stats else False

        result = BenchmarkResult(
            name="order_processing",
            description="Order processing latency benchmark",
            latency=stats,
            passed=passed,
            metadata={
                "threshold_ms": config.latency_threshold_ms,
                "num_orders": len(sample_orders),
            }
        )

        self.benchmark._results.append(result)
        return result

    def benchmark_signal_generation(
        self,
        signal_func: Callable,
        market_data: Any
    ) -> BenchmarkResult:
        """Benchmark signal generation latency."""
        config = BenchmarkConfig(
            warmup_iterations=10,
            benchmark_iterations=100,
            latency_threshold_ms=self._thresholds["signal_latency_ms"],
        )

        return self.benchmark.run_latency_benchmark(
            "signal_generation",
            signal_func,
            config,
            market_data,
        )

    def benchmark_risk_calculation(
        self,
        risk_func: Callable,
        portfolio: Any
    ) -> BenchmarkResult:
        """Benchmark risk calculation latency."""
        config = BenchmarkConfig(
            warmup_iterations=5,
            benchmark_iterations=50,
            latency_threshold_ms=self._thresholds["risk_calc_latency_ms"],
        )

        return self.benchmark.run_latency_benchmark(
            "risk_calculation",
            risk_func,
            config,
            portfolio,
        )

    def benchmark_data_pipeline(
        self,
        process_func: Callable,
        data_batch: Any,
        duration_seconds: float = 10.0
    ) -> BenchmarkResult:
        """Benchmark data pipeline throughput."""
        return self.benchmark.run_throughput_benchmark(
            "data_pipeline",
            process_func,
            duration_seconds,
            data_batch,
        )

    def run_full_suite(
        self,
        order_func: Optional[Callable] = None,
        signal_func: Optional[Callable] = None,
        risk_func: Optional[Callable] = None,
        data_func: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """Run full benchmark suite."""
        results = {}

        if order_func and "sample_orders" in kwargs:
            results["order_processing"] = self.benchmark_order_processing(
                order_func, kwargs["sample_orders"]
            )

        if signal_func and "market_data" in kwargs:
            results["signal_generation"] = self.benchmark_signal_generation(
                signal_func, kwargs["market_data"]
            )

        if risk_func and "portfolio" in kwargs:
            results["risk_calculation"] = self.benchmark_risk_calculation(
                risk_func, kwargs["portfolio"]
            )

        if data_func and "data_batch" in kwargs:
            results["data_pipeline"] = self.benchmark_data_pipeline(
                data_func, kwargs["data_batch"]
            )

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        stats = self.benchmark.get_all_stats()

        # Calculate summary
        latency_results = [r for r in self.benchmark._results if r.latency]
        throughput_results = [r for r in self.benchmark._results if r.throughput]

        passed = all(r.passed for r in self.benchmark._results)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.benchmark._results),
            "passed": sum(1 for r in self.benchmark._results if r.passed),
            "failed": sum(1 for r in self.benchmark._results if not r.passed),
            "overall_status": "PASS" if passed else "FAIL",
        }

        if latency_results:
            summary["avg_p95_latency_ms"] = round(
                statistics.mean(r.latency.p95_ms for r in latency_results if r.latency),
                3
            )

        if throughput_results:
            summary["avg_throughput_ops"] = round(
                statistics.mean(r.throughput.ops_per_second for r in throughput_results if r.throughput),
                2
            )

        return {
            "summary": summary,
            "thresholds": self._thresholds,
            "details": stats,
        }


# Global benchmark instance
_benchmark = PerformanceBenchmark()


def get_benchmark() -> PerformanceBenchmark:
    """Get global benchmark instance."""
    return _benchmark


def create_trading_benchmarks(
    benchmark: Optional[PerformanceBenchmark] = None
) -> TradingSystemBenchmarks:
    """Factory function to create trading system benchmarks."""
    return TradingSystemBenchmarks(benchmark or _benchmark)
