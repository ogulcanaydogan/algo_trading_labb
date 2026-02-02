"""
Prometheus metrics exporter for the trading bot.
Exposes key trading metrics for monitoring and alerting.
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
import time

# Create a custom registry
registry = CollectorRegistry()

# Trading metrics
trades_total = Counter(
    "trades_total", "Total number of trades executed", ["mode", "symbol", "side"], registry=registry
)

trades_won = Counter(
    "trades_won_total", "Total number of winning trades", ["mode", "symbol"], registry=registry
)

trades_lost = Counter(
    "trades_lost_total", "Total number of losing trades", ["mode", "symbol"], registry=registry
)

portfolio_value = Gauge(
    "portfolio_value_usd", "Current portfolio value in USD", ["mode", "symbol"], registry=registry
)

position_size = Gauge(
    "position_size", "Current position size", ["mode", "symbol"], registry=registry
)

unrealized_pnl = Gauge(
    "unrealized_pnl_usd", "Unrealized P&L in USD", ["mode", "symbol"], registry=registry
)

daily_pnl = Gauge("daily_pnl_usd", "Daily P&L in USD", ["mode", "symbol"], registry=registry)

# Risk metrics
drawdown_pct = Gauge("drawdown_percent", "Current drawdown percentage", ["mode"], registry=registry)

stop_loss_triggered = Counter(
    "stop_loss_triggered_total",
    "Total number of stop losses triggered",
    ["mode", "symbol"],
    registry=registry,
)

margin_ratio = Gauge(
    "margin_ratio", "Current margin ratio (used/available)", ["mode"], registry=registry
)

# Signal metrics
signals_generated = Counter(
    "signals_generated_total",
    "Total signals generated",
    ["type", "symbol"],  # signal_type: LONG, SHORT, FLAT
    registry=registry,
)

signal_confidence = Gauge(
    "signal_confidence", "Latest signal confidence score", ["type", "symbol"], registry=registry
)

# AI metrics
ai_predictions_total = Counter(
    "ai_predictions_total",
    "Total AI predictions made",
    ["symbol", "outcome"],  # outcome: correct, incorrect
    registry=registry,
)

ai_prediction_accuracy = Gauge(
    "ai_prediction_accuracy",
    "AI model accuracy percentage",
    ["model_name", "symbol"],
    registry=registry,
)

# API metrics
api_requests_total = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"], registry=registry
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    registry=registry,
)

# System metrics
active_positions = Gauge(
    "active_positions", "Number of active positions", ["mode"], registry=registry
)

order_latency_ms = Histogram(
    "order_latency_milliseconds",
    "Order execution latency in milliseconds",
    ["exchange"],
    buckets=(10, 50, 100, 200, 500, 1000),
    registry=registry,
)

data_freshness_seconds = Gauge(
    "data_freshness_seconds", "Age of latest market data in seconds", ["symbol"], registry=registry
)


def get_metrics():
    """Return Prometheus metrics in text format."""
    return generate_latest(registry)
