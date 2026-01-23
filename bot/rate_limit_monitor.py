"""
API Rate Limit Dashboard Module.

Monitors API usage across all providers and displays
rate limit status and usage statistics.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an API."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    weight_per_minute: int = 1200  # For weighted rate limits (e.g., Binance)
    concurrent_limit: int = 10


@dataclass
class APIUsageStats:
    """Usage statistics for an API."""

    name: str
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    weight_last_minute: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: str = ""
    status: str = "healthy"  # healthy, warning, throttled, error


@dataclass
class RateLimitStatus:
    """Current rate limit status for an API."""

    name: str
    minute_usage: float  # Percentage of minute limit used
    hour_usage: float
    day_usage: float
    weight_usage: float
    is_throttled: bool
    seconds_until_reset: int
    warning_level: str  # none, low, medium, high, critical


class RateLimitMonitor:
    """
    Monitors API rate limits and usage across all providers.

    Tracks requests, calculates usage percentages, and provides
    warnings when approaching limits.
    """

    # Default rate limits for common providers
    DEFAULT_LIMITS = {
        "binance": RateLimitConfig(
            requests_per_minute=1200,
            requests_per_hour=10000,
            weight_per_minute=1200,
        ),
        "coinbase": RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=500,
        ),
        "polygon": RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=5000,
        ),
        "yahoo": RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=48000,
        ),
        "anthropic": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
        ),
        "newsapi": RateLimitConfig(
            requests_per_day=100,
        ),
        "finnhub": RateLimitConfig(
            requests_per_minute=60,
            requests_per_day=1500,
        ),
    }

    def __init__(self):
        self._lock = Lock()
        self._configs: Dict[str, RateLimitConfig] = dict(self.DEFAULT_LIMITS)
        self._request_log: Dict[str, List[float]] = defaultdict(list)
        self._weight_log: Dict[str, List[tuple]] = defaultdict(list)  # (timestamp, weight)
        self._stats: Dict[str, APIUsageStats] = {}
        self._throttle_until: Dict[str, float] = {}
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._errors: Dict[str, List[tuple]] = defaultdict(list)  # (timestamp, message)

    def configure_api(self, name: str, config: RateLimitConfig) -> None:
        """Configure rate limits for an API."""
        with self._lock:
            self._configs[name] = config
            if name not in self._stats:
                self._stats[name] = APIUsageStats(name=name)

    def record_request(
        self,
        api_name: str,
        weight: int = 1,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record an API request.

        Args:
            api_name: Name of the API
            weight: Request weight (for weighted rate limits)
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
        """
        with self._lock:
            now = time.time()

            # Initialize if needed
            if api_name not in self._stats:
                self._stats[api_name] = APIUsageStats(name=api_name)

            # Record request
            self._request_log[api_name].append(now)
            self._weight_log[api_name].append((now, weight))

            # Update stats
            stats = self._stats[api_name]
            stats.total_requests += 1
            stats.last_request_time = datetime.now()

            # Record latency
            if latency_ms is not None:
                self._latencies[api_name].append(latency_ms)
                # Keep only last 100 latencies
                if len(self._latencies[api_name]) > 100:
                    self._latencies[api_name] = self._latencies[api_name][-100:]
                stats.avg_latency_ms = sum(self._latencies[api_name]) / len(
                    self._latencies[api_name]
                )

            # Record error
            if error:
                stats.total_errors += 1
                stats.last_error_time = datetime.now()
                stats.last_error_message = error
                self._errors[api_name].append((now, error))
                # Keep only last 50 errors
                if len(self._errors[api_name]) > 50:
                    self._errors[api_name] = self._errors[api_name][-50:]

            # Clean old entries
            self._cleanup_old_entries(api_name, now)

    def check_rate_limit(self, api_name: str, weight: int = 1) -> tuple:
        """
        Check if request is allowed under rate limits.

        Args:
            api_name: Name of the API
            weight: Weight of the intended request

        Returns:
            Tuple of (is_allowed, wait_seconds)
        """
        with self._lock:
            now = time.time()

            # Check if throttled
            if api_name in self._throttle_until:
                if now < self._throttle_until[api_name]:
                    wait = self._throttle_until[api_name] - now
                    return False, wait

            config = self._configs.get(api_name, RateLimitConfig())
            self._cleanup_old_entries(api_name, now)

            # Check minute limit
            minute_ago = now - 60
            requests_last_minute = len([t for t in self._request_log[api_name] if t > minute_ago])

            if requests_last_minute >= config.requests_per_minute:
                oldest_in_minute = min(t for t in self._request_log[api_name] if t > minute_ago)
                wait = 60 - (now - oldest_in_minute) + 1
                return False, wait

            # Check weight limit
            weight_last_minute = sum(w for t, w in self._weight_log[api_name] if t > minute_ago)
            if weight_last_minute + weight > config.weight_per_minute:
                return False, 60

            # Check hour limit
            hour_ago = now - 3600
            requests_last_hour = len([t for t in self._request_log[api_name] if t > hour_ago])

            if requests_last_hour >= config.requests_per_hour:
                return False, 3600

            # Check day limit
            day_ago = now - 86400
            requests_last_day = len([t for t in self._request_log[api_name] if t > day_ago])

            if requests_last_day >= config.requests_per_day:
                return False, 86400

            return True, 0

    def set_throttled(self, api_name: str, seconds: int) -> None:
        """Mark an API as throttled for specified seconds."""
        with self._lock:
            self._throttle_until[api_name] = time.time() + seconds
            if api_name in self._stats:
                self._stats[api_name].status = "throttled"

    def _cleanup_old_entries(self, api_name: str, now: float) -> None:
        """Remove entries older than 24 hours."""
        day_ago = now - 86400

        self._request_log[api_name] = [t for t in self._request_log[api_name] if t > day_ago]
        self._weight_log[api_name] = [(t, w) for t, w in self._weight_log[api_name] if t > day_ago]

    def get_status(self, api_name: str) -> RateLimitStatus:
        """Get current rate limit status for an API."""
        with self._lock:
            now = time.time()
            config = self._configs.get(api_name, RateLimitConfig())

            self._cleanup_old_entries(api_name, now)

            minute_ago = now - 60
            hour_ago = now - 3600
            day_ago = now - 86400

            requests_minute = len([t for t in self._request_log[api_name] if t > minute_ago])
            requests_hour = len([t for t in self._request_log[api_name] if t > hour_ago])
            requests_day = len([t for t in self._request_log[api_name] if t > day_ago])
            weight_minute = sum(w for t, w in self._weight_log[api_name] if t > minute_ago)

            minute_usage = (
                (requests_minute / config.requests_per_minute * 100)
                if config.requests_per_minute > 0
                else 0
            )
            hour_usage = (
                (requests_hour / config.requests_per_hour * 100)
                if config.requests_per_hour > 0
                else 0
            )
            day_usage = (
                (requests_day / config.requests_per_day * 100) if config.requests_per_day > 0 else 0
            )
            weight_usage = (
                (weight_minute / config.weight_per_minute * 100)
                if config.weight_per_minute > 0
                else 0
            )

            is_throttled = api_name in self._throttle_until and now < self._throttle_until[api_name]
            seconds_until_reset = int(self._throttle_until[api_name] - now) if is_throttled else 0

            # Determine warning level
            max_usage = max(minute_usage, hour_usage, day_usage, weight_usage)
            if max_usage >= 95:
                warning_level = "critical"
            elif max_usage >= 80:
                warning_level = "high"
            elif max_usage >= 60:
                warning_level = "medium"
            elif max_usage >= 40:
                warning_level = "low"
            else:
                warning_level = "none"

            return RateLimitStatus(
                name=api_name,
                minute_usage=round(minute_usage, 1),
                hour_usage=round(hour_usage, 1),
                day_usage=round(day_usage, 1),
                weight_usage=round(weight_usage, 1),
                is_throttled=is_throttled,
                seconds_until_reset=seconds_until_reset,
                warning_level=warning_level,
            )

    def get_all_stats(self) -> Dict[str, APIUsageStats]:
        """Get usage statistics for all APIs."""
        with self._lock:
            now = time.time()

            for api_name, stats in self._stats.items():
                self._cleanup_old_entries(api_name, now)

                minute_ago = now - 60
                hour_ago = now - 3600
                day_ago = now - 86400

                stats.requests_last_minute = len(
                    [t for t in self._request_log[api_name] if t > minute_ago]
                )
                stats.requests_last_hour = len(
                    [t for t in self._request_log[api_name] if t > hour_ago]
                )
                stats.requests_last_day = len(
                    [t for t in self._request_log[api_name] if t > day_ago]
                )
                stats.weight_last_minute = sum(
                    w for t, w in self._weight_log[api_name] if t > minute_ago
                )

                # Update status
                status_obj = self.get_status(api_name)
                if status_obj.is_throttled:
                    stats.status = "throttled"
                elif status_obj.warning_level == "critical":
                    stats.status = "warning"
                elif (
                    stats.last_error_time and (datetime.now() - stats.last_error_time).seconds < 300
                ):
                    stats.status = "error"
                else:
                    stats.status = "healthy"

            return dict(self._stats)

    def get_recent_errors(self, api_name: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get recent errors."""
        with self._lock:
            errors = []

            if api_name:
                for ts, msg in self._errors.get(api_name, [])[-limit:]:
                    errors.append(
                        {
                            "api": api_name,
                            "timestamp": datetime.fromtimestamp(ts).isoformat(),
                            "message": msg,
                        }
                    )
            else:
                all_errors = []
                for name, err_list in self._errors.items():
                    for ts, msg in err_list:
                        all_errors.append((ts, name, msg))

                all_errors.sort(reverse=True)
                for ts, name, msg in all_errors[:limit]:
                    errors.append(
                        {
                            "api": name,
                            "timestamp": datetime.fromtimestamp(ts).isoformat(),
                            "message": msg,
                        }
                    )

            return errors

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        all_stats = self.get_all_stats()

        apis = []
        for name, stats in all_stats.items():
            status = self.get_status(name)
            config = self._configs.get(name, RateLimitConfig())

            apis.append(
                {
                    "name": name,
                    "status": stats.status,
                    "usage": {
                        "minute": {
                            "current": stats.requests_last_minute,
                            "limit": config.requests_per_minute,
                            "percent": status.minute_usage,
                        },
                        "hour": {
                            "current": stats.requests_last_hour,
                            "limit": config.requests_per_hour,
                            "percent": status.hour_usage,
                        },
                        "day": {
                            "current": stats.requests_last_day,
                            "limit": config.requests_per_day,
                            "percent": status.day_usage,
                        },
                        "weight": {
                            "current": stats.weight_last_minute,
                            "limit": config.weight_per_minute,
                            "percent": status.weight_usage,
                        },
                    },
                    "stats": {
                        "total_requests": stats.total_requests,
                        "total_errors": stats.total_errors,
                        "avg_latency_ms": round(stats.avg_latency_ms, 1),
                        "last_request": stats.last_request_time.isoformat()
                        if stats.last_request_time
                        else None,
                        "last_error": stats.last_error_time.isoformat()
                        if stats.last_error_time
                        else None,
                        "last_error_message": stats.last_error_message,
                    },
                    "throttle": {
                        "is_throttled": status.is_throttled,
                        "seconds_until_reset": status.seconds_until_reset,
                    },
                    "warning_level": status.warning_level,
                }
            )

        # Overall health
        health_statuses = [a["status"] for a in apis]
        if "throttled" in health_statuses:
            overall_health = "degraded"
        elif "error" in health_statuses:
            overall_health = "issues"
        elif "warning" in health_statuses:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "apis": apis,
            "recent_errors": self.get_recent_errors(limit=10),
            "timestamp": datetime.now().isoformat(),
        }


# Global instance
_monitor: Optional[RateLimitMonitor] = None


def get_rate_limit_monitor() -> RateLimitMonitor:
    """Get or create global rate limit monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = RateLimitMonitor()
    return _monitor
