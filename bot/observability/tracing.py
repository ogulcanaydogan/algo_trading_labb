"""
Distributed Tracing for Trading System Observability.

Provides request tracing across services for debugging,
performance analysis, and system monitoring.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(Enum):
    """Type of span."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span completion status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for trace propagation."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sampled": self.sampled,
        }

    def to_header(self) -> str:
        """Convert to W3C traceparent header format."""
        sampled_flag = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{sampled_flag}"

    @classmethod
    def from_header(cls, header: str) -> Optional["SpanContext"]:
        """Parse W3C traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) != 4:
                return None
            _, trace_id, span_id, flags = parts
            sampled = flags == "01"
            return cls(trace_id=trace_id, span_id=span_id, sampled=sampled)
        except Exception:
            return None

    @classmethod
    def generate(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Generate new span context."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=parent.span_id,
                sampled=parent.sampled,
            )
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            sampled=True,
        )


@dataclass
class SpanEvent:
    """Event within a span."""

    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A single span in a trace."""

    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds."""
        if not self.end_time:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_root(self) -> bool:
        """Check if this is a root span."""
        return self.context.parent_span_id is None

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None):
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=datetime.now(),
                attributes=attributes or {},
            )
        )

    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Set span status."""
        self.status = status
        if message:
            self.error_message = message

    def end(self, status: Optional[SpanStatus] = None):
        """End the span."""
        self.end_time = datetime.now()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "error_message": self.error_message,
        }


class SpanProcessor:
    """Process completed spans."""

    def on_start(self, span: Span):
        """Called when span starts."""
        pass

    def on_end(self, span: Span):
        """Called when span ends."""
        pass


class ConsoleSpanProcessor(SpanProcessor):
    """Output spans to console/logger."""

    def __init__(self, min_duration_ms: float = 0):
        self.min_duration_ms = min_duration_ms

    def on_end(self, span: Span):
        if span.duration_ms >= self.min_duration_ms:
            status_emoji = "✓" if span.status == SpanStatus.OK else "✗"
            logger.info(
                f"[TRACE] {status_emoji} {span.name} "
                f"| {span.duration_ms:.2f}ms "
                f"| trace={span.context.trace_id[:8]}"
            )


class InMemorySpanProcessor(SpanProcessor):
    """Store spans in memory for querying."""

    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self._spans: List[Span] = []
        self._lock = threading.Lock()

    def on_end(self, span: Span):
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self.max_spans:
                self._spans = self._spans[-self.max_spans :]

    def get_spans(
        self, trace_id: Optional[str] = None, name: Optional[str] = None, min_duration_ms: float = 0
    ) -> List[Span]:
        """Query stored spans."""
        with self._lock:
            spans = self._spans.copy()

        if trace_id:
            spans = [s for s in spans if s.context.trace_id == trace_id]
        if name:
            spans = [s for s in spans if name in s.name]
        if min_duration_ms > 0:
            spans = [s for s in spans if s.duration_ms >= min_duration_ms]

        return spans

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self.get_spans(trace_id=trace_id)

    def clear(self):
        """Clear stored spans."""
        with self._lock:
            self._spans.clear()


class FileSpanProcessor(SpanProcessor):
    """Write spans to file in JSON format."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._lock = threading.Lock()

    def on_end(self, span: Span):
        with self._lock:
            try:
                with open(self.file_path, "a") as f:
                    f.write(json.dumps(span.to_dict()) + "\n")
            except Exception as e:
                logger.error(f"Failed to write span: {e}")


# Thread-local storage for current span
_current_span = threading.local()


class Tracer:
    """
    Distributed tracer for creating and managing spans.

    Usage:
        tracer = Tracer("trading-service")

        with tracer.start_span("process_order") as span:
            span.set_attribute("order_id", "12345")
            # ... do work
    """

    def __init__(self, service_name: str, processors: Optional[List[SpanProcessor]] = None):
        self.service_name = service_name
        self._processors = processors or []
        self._sampling_rate = 1.0

    def add_processor(self, processor: SpanProcessor):
        """Add a span processor."""
        self._processors.append(processor)

    def set_sampling_rate(self, rate: float):
        """Set sampling rate (0.0 to 1.0)."""
        self._sampling_rate = max(0.0, min(1.0, rate))

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return getattr(_current_span, "span", None)

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[Union[Span, SpanContext]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a new span as a context manager.

        Args:
            name: Span name
            kind: Span kind
            parent: Parent span or context
            attributes: Initial attributes
        """
        # Determine parent context
        if parent is None:
            parent = self.get_current_span()

        if isinstance(parent, Span):
            parent_context = parent.context
        elif isinstance(parent, SpanContext):
            parent_context = parent
        else:
            parent_context = None

        # Generate context
        context = SpanContext.generate(parent_context)

        # Apply sampling
        if parent_context:
            context.sampled = parent_context.sampled
        else:
            import random

            context.sampled = random.random() < self._sampling_rate

        # Create span
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self.service_name)

        # Notify processors
        if context.sampled:
            for processor in self._processors:
                processor.on_start(span)

        # Set as current span
        previous_span = self.get_current_span()
        _current_span.span = span

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            raise
        finally:
            span.end()
            _current_span.span = previous_span

            if context.sampled:
                for processor in self._processors:
                    processor.on_end(span)

    def start_span_from_context(
        self, name: str, context: SpanContext, kind: SpanKind = SpanKind.SERVER
    ):
        """Start a span from an incoming context (e.g., from HTTP header)."""
        return self.start_span(name, kind=kind, parent=context)


def trace(
    tracer: Tracer,
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to trace a function.

    Args:
        tracer: Tracer instance
        name: Span name (defaults to function name)
        kind: Span kind
        attributes: Additional attributes
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_span(span_name, kind=kind, attributes=attributes) as span:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def async_trace(
    tracer: Tracer,
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace an async function."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_span(span_name, kind=kind, attributes=attributes) as span:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class TradingTracer:
    """
    Specialized tracer for trading operations.

    Pre-configured span types:
    - Order lifecycle
    - Signal generation
    - Risk checks
    - Data fetching
    """

    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer("trading-system")

    @contextmanager
    def trace_order(self, order_id: str, symbol: str, side: str, size: float):
        """Trace order lifecycle."""
        with self.tracer.start_span(
            "order.process",
            kind=SpanKind.INTERNAL,
            attributes={
                "order.id": order_id,
                "order.symbol": symbol,
                "order.side": side,
                "order.size": size,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_signal(self, strategy: str, symbol: str):
        """Trace signal generation."""
        with self.tracer.start_span(
            "signal.generate",
            attributes={
                "signal.strategy": strategy,
                "signal.symbol": symbol,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_risk_check(self, check_type: str):
        """Trace risk check."""
        with self.tracer.start_span(
            f"risk.{check_type}", attributes={"risk.check_type": check_type}
        ) as span:
            yield span

    @contextmanager
    def trace_data_fetch(self, source: str, symbol: str):
        """Trace data fetching."""
        with self.tracer.start_span(
            "data.fetch",
            kind=SpanKind.CLIENT,
            attributes={
                "data.source": source,
                "data.symbol": symbol,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_ml_prediction(self, model: str, symbol: str):
        """Trace ML prediction."""
        with self.tracer.start_span(
            "ml.predict",
            attributes={
                "ml.model": model,
                "ml.symbol": symbol,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_exchange_call(self, exchange: str, operation: str):
        """Trace exchange API call."""
        with self.tracer.start_span(
            f"exchange.{operation}",
            kind=SpanKind.CLIENT,
            attributes={
                "exchange.name": exchange,
                "exchange.operation": operation,
            },
        ) as span:
            yield span


class TraceAnalyzer:
    """Analyze traces for performance insights."""

    def __init__(self, processor: InMemorySpanProcessor):
        self.processor = processor

    def get_slow_spans(self, min_duration_ms: float = 100, limit: int = 10) -> List[Dict]:
        """Get slowest spans."""
        spans = self.processor.get_spans(min_duration_ms=min_duration_ms)
        spans.sort(key=lambda s: s.duration_ms, reverse=True)
        return [s.to_dict() for s in spans[:limit]]

    def get_error_spans(self, limit: int = 10) -> List[Dict]:
        """Get spans with errors."""
        spans = [s for s in self.processor.get_spans() if s.status == SpanStatus.ERROR]
        return [s.to_dict() for s in spans[:limit]]

    def get_trace_summary(self, trace_id: str) -> Dict:
        """Get summary of a trace."""
        spans = self.processor.get_trace(trace_id)
        if not spans:
            return {"error": "Trace not found"}

        root = next((s for s in spans if s.is_root), spans[0])
        total_duration = root.duration_ms
        span_count = len(spans)
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)

        # Calculate critical path
        spans_by_duration = sorted(spans, key=lambda s: s.duration_ms, reverse=True)

        return {
            "trace_id": trace_id,
            "root_span": root.name,
            "total_duration_ms": round(total_duration, 3),
            "span_count": span_count,
            "error_count": error_count,
            "slowest_spans": [
                {"name": s.name, "duration_ms": round(s.duration_ms, 3)}
                for s in spans_by_duration[:5]
            ],
        }

    def get_service_stats(self) -> Dict[str, Dict]:
        """Get statistics by service/operation."""
        spans = self.processor.get_spans()

        stats: Dict[str, List[float]] = {}
        for span in spans:
            if span.name not in stats:
                stats[span.name] = []
            stats[span.name].append(span.duration_ms)

        result = {}
        for name, durations in stats.items():
            durations.sort()
            n = len(durations)
            result[name] = {
                "count": n,
                "mean_ms": round(sum(durations) / n, 3),
                "p50_ms": round(durations[int(n * 0.5)], 3),
                "p95_ms": round(durations[int(n * 0.95)], 3) if n > 1 else durations[0],
                "p99_ms": round(durations[int(n * 0.99)], 3) if n > 1 else durations[0],
            }

        return result


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = create_tracer("trading-system")
    return _global_tracer


def create_tracer(
    service_name: str,
    enable_console: bool = True,
    enable_memory: bool = True,
    console_min_duration_ms: float = 10,
) -> Tracer:
    """
    Factory function to create a configured tracer.

    Args:
        service_name: Name of the service
        enable_console: Enable console output
        enable_memory: Enable in-memory storage
        console_min_duration_ms: Minimum duration to log to console
    """
    tracer = Tracer(service_name)

    if enable_console:
        tracer.add_processor(ConsoleSpanProcessor(min_duration_ms=console_min_duration_ms))

    if enable_memory:
        tracer.add_processor(InMemorySpanProcessor())

    return tracer


def create_trading_tracer(tracer: Optional[Tracer] = None) -> TradingTracer:
    """Factory function to create trading tracer."""
    return TradingTracer(tracer or get_tracer())
