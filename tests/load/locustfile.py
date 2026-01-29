"""
Load Testing with Locust.

Comprehensive load testing for the trading API including:
- Dashboard endpoints
- Trading state endpoints
- WebSocket connections
- Prometheus metrics

Usage:
    # Run with web UI
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Run headless
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 100 -r 10 --run-time 5m

    # Run with specific scenarios
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --tags dashboard --headless -u 50 -r 5

Configuration:
    Set LOCUST_HOST environment variable or use --host flag
"""

import json
import logging
import random
import time
from typing import Any, Dict, List

from locust import HttpUser, between, events, tag, task
from locust.exception import StopUser

logger = logging.getLogger(__name__)


class TradingAPIUser(HttpUser):
    """
    Simulates a typical user interacting with the trading API.

    This user performs a mix of operations:
    - Checking dashboard status (frequent)
    - Getting portfolio state (frequent)
    - Getting positions (moderate)
    - Checking metrics (occasional)
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Setup performed when user starts."""
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        self.markets = ["crypto", "stock", "commodity"]
        self._request_count = 0

    @tag("dashboard", "critical")
    @task(10)  # High weight - most common operation
    def get_dashboard_state(self):
        """Get unified dashboard state."""
        with self.client.get(
            "/api/dashboard/unified-state",
            catch_response=True,
            name="/api/dashboard/unified-state"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "version" not in data:
                        response.failure("Missing version field")
                    elif "total" not in data:
                        response.failure("Missing total field")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status {response.status_code}")

    @tag("status", "critical")
    @task(8)
    def get_unified_status(self):
        """Get unified trading status."""
        self.client.get(
            "/api/unified/status",
            name="/api/unified/status"
        )

    @tag("positions")
    @task(5)
    def get_positions(self):
        """Get current positions."""
        self.client.get(
            "/api/unified/positions",
            name="/api/unified/positions"
        )

    @tag("equity")
    @task(3)
    def get_equity(self):
        """Get equity data."""
        self.client.get(
            "/api/unified/equity",
            name="/api/unified/equity"
        )

    @tag("trades")
    @task(2)
    def get_trades(self):
        """Get recent trades."""
        self.client.get(
            "/api/unified/trades",
            name="/api/unified/trades"
        )

    @tag("metrics")
    @task(1)
    def get_metrics(self):
        """Get Prometheus metrics."""
        self.client.get(
            "/metrics",
            name="/metrics"
        )

    @tag("health")
    @task(2)
    def get_health(self):
        """Check API health."""
        self.client.get(
            "/health",
            name="/health"
        )

    @tag("versions")
    @task(1)
    def get_versions(self):
        """Get API version info."""
        self.client.get(
            "/api/versions",
            name="/api/versions"
        )

    @tag("websocket", "stats")
    @task(1)
    def get_websocket_stats(self):
        """Get WebSocket statistics."""
        self.client.get(
            "/ws/stats",
            name="/ws/stats"
        )


class DashboardPollingUser(HttpUser):
    """
    Simulates a dashboard constantly polling for updates.

    This user represents a typical dashboard that refreshes frequently.
    """

    wait_time = between(0.5, 2)  # More frequent polling

    @tag("dashboard", "polling")
    @task
    def poll_dashboard(self):
        """Poll dashboard state."""
        with self.client.get(
            "/api/dashboard/unified-state",
            catch_response=True,
            name="/api/dashboard/unified-state [polling]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check response time
                    if response.elapsed.total_seconds() > 1.0:
                        response.failure(f"Slow response: {response.elapsed.total_seconds():.2f}s")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON")


class HighFrequencyUser(HttpUser):
    """
    Simulates a high-frequency client making rapid requests.

    Used for stress testing endpoint capacity.
    """

    wait_time = between(0.1, 0.5)  # Very frequent requests

    def on_start(self):
        """Setup."""
        self._request_count = 0

    @tag("stress", "high-frequency")
    @task(5)
    def rapid_status_check(self):
        """Rapid status checks."""
        self.client.get(
            "/api/unified/status",
            name="/api/unified/status [rapid]"
        )
        self._request_count += 1

    @tag("stress", "high-frequency")
    @task(3)
    def rapid_positions_check(self):
        """Rapid positions checks."""
        self.client.get(
            "/api/unified/positions",
            name="/api/unified/positions [rapid]"
        )
        self._request_count += 1

    @tag("stress", "high-frequency")
    @task(2)
    def rapid_dashboard_check(self):
        """Rapid dashboard checks."""
        self.client.get(
            "/api/dashboard/unified-state",
            name="/api/dashboard/unified-state [rapid]"
        )
        self._request_count += 1


class MetricsCollectorUser(HttpUser):
    """
    Simulates Prometheus scraping metrics.

    Metrics are typically scraped every 15-60 seconds.
    """

    wait_time = between(10, 30)

    @tag("metrics", "prometheus")
    @task
    def scrape_metrics(self):
        """Scrape Prometheus metrics."""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics [scrape]"
        ) as response:
            if response.status_code == 200:
                content = response.text
                # Verify it looks like Prometheus format
                if "# HELP" in content and "# TYPE" in content:
                    response.success()
                else:
                    response.failure("Invalid Prometheus format")

    @tag("metrics", "json")
    @task
    def get_metrics_json(self):
        """Get metrics in JSON format."""
        self.client.get(
            "/metrics/json",
            name="/metrics/json"
        )


# Event hooks for custom logging
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log slow requests."""
    if response_time > 1000:  # > 1 second
        logger.warning(f"Slow request: {name} took {response_time:.0f}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start."""
    logger.info("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test summary."""
    stats = environment.stats
    logger.info(f"Load test finished. Total requests: {stats.total.num_requests}")
    logger.info(f"Requests/s: {stats.total.current_rps:.2f}")
    logger.info(f"Failures: {stats.total.num_failures}")


# Performance thresholds for CI/CD
PERFORMANCE_THRESHOLDS = {
    "max_response_time_p95_ms": 500,
    "max_response_time_p99_ms": 1000,
    "max_failure_rate_percent": 1.0,
    "min_requests_per_second": 50,
}


def check_thresholds(stats) -> List[str]:
    """Check if performance thresholds are met."""
    failures = []

    if stats.total.get_response_time_percentile(0.95) > PERFORMANCE_THRESHOLDS["max_response_time_p95_ms"]:
        failures.append(
            f"P95 response time ({stats.total.get_response_time_percentile(0.95):.0f}ms) "
            f"exceeds threshold ({PERFORMANCE_THRESHOLDS['max_response_time_p95_ms']}ms)"
        )

    if stats.total.get_response_time_percentile(0.99) > PERFORMANCE_THRESHOLDS["max_response_time_p99_ms"]:
        failures.append(
            f"P99 response time ({stats.total.get_response_time_percentile(0.99):.0f}ms) "
            f"exceeds threshold ({PERFORMANCE_THRESHOLDS['max_response_time_p99_ms']}ms)"
        )

    failure_rate = stats.total.fail_ratio * 100
    if failure_rate > PERFORMANCE_THRESHOLDS["max_failure_rate_percent"]:
        failures.append(
            f"Failure rate ({failure_rate:.2f}%) exceeds threshold "
            f"({PERFORMANCE_THRESHOLDS['max_failure_rate_percent']}%)"
        )

    if stats.total.current_rps < PERFORMANCE_THRESHOLDS["min_requests_per_second"]:
        failures.append(
            f"Requests/s ({stats.total.current_rps:.2f}) below threshold "
            f"({PERFORMANCE_THRESHOLDS['min_requests_per_second']})"
        )

    return failures
