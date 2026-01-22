"""
Observability module for performance benchmarks and distributed tracing.

Provides:
- Performance benchmarking and latency tracking
- Distributed tracing for request flows
"""

from .benchmarks import (
    PerformanceBenchmark,
    TradingSystemBenchmarks,
    LatencyTracker,
    ThroughputTracker,
    LatencyStats,
    ThroughputStats,
    BenchmarkResult,
    BenchmarkConfig,
    get_benchmark,
    create_trading_benchmarks,
    latency_benchmark,
    async_latency_benchmark,
)

from .tracing import (
    Tracer,
    TradingTracer,
    TraceAnalyzer,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    get_tracer,
    create_tracer,
    create_trading_tracer,
    trace,
    async_trace,
    ConsoleSpanProcessor,
    InMemorySpanProcessor,
    FileSpanProcessor,
)

__all__ = [
    # Benchmarks
    "PerformanceBenchmark",
    "TradingSystemBenchmarks",
    "LatencyTracker",
    "ThroughputTracker",
    "LatencyStats",
    "ThroughputStats",
    "BenchmarkResult",
    "BenchmarkConfig",
    "get_benchmark",
    "create_trading_benchmarks",
    "latency_benchmark",
    "async_latency_benchmark",
    # Tracing
    "Tracer",
    "TradingTracer",
    "TraceAnalyzer",
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "get_tracer",
    "create_tracer",
    "create_trading_tracer",
    "trace",
    "async_trace",
    "ConsoleSpanProcessor",
    "InMemorySpanProcessor",
    "FileSpanProcessor",
]
