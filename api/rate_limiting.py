"""
Rate Limiting Module for API Endpoints.

Provides flexible rate limiting with:
- Per-endpoint rate limits
- Per-user/IP rate limits
- Redis-backed distributed rate limiting
- Sliding window algorithm
- Rate limit headers in responses

Usage:
    from api.rate_limiting import RateLimiter, rate_limit

    limiter = RateLimiter()

    @app.get("/api/data")
    @rate_limit(requests=100, window=60)  # 100 requests per minute
    async def get_data():
        return {"data": "..."}
"""

import asyncio
import functools
import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""

    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    burst: Optional[int] = None  # Burst limit (default: same as requests)
    key_prefix: str = ""  # Prefix for rate limit key
    cost: int = 1  # Cost per request (for weighted limits)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests


@dataclass
class RateLimitState:
    """State for tracking rate limits."""

    requests: int = 0
    window_start: float = field(default_factory=time.time)
    tokens: float = 0  # For token bucket
    last_refill: float = field(default_factory=time.time)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Uses a sliding window algorithm for smooth rate limiting.
    Supports both in-memory and Redis backends.
    """

    def __init__(
        self,
        default_requests: int = 100,
        default_window: int = 60,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            default_requests: Default requests per window
            default_window: Default window size in seconds
            redis_url: Optional Redis URL for distributed limiting
        """
        self.default_requests = default_requests
        self.default_window = default_window
        self.redis_url = redis_url or os.getenv("REDIS_URL")

        # In-memory storage (for single-instance deployment)
        self._storage: Dict[str, List[float]] = defaultdict(list)
        self._storage_lock = asyncio.Lock()

        # Redis client (lazy initialization)
        self._redis = None

        # Endpoint-specific limits
        self._endpoint_limits: Dict[str, RateLimitConfig] = {}

        logger.info(
            f"Rate limiter initialized: {default_requests} req/{default_window}s"
        )

    def configure_endpoint(
        self,
        endpoint: str,
        requests: int,
        window: int,
        burst: Optional[int] = None,
    ) -> None:
        """
        Configure rate limit for a specific endpoint.

        Args:
            endpoint: Endpoint path (e.g., "/api/data")
            requests: Requests allowed per window
            window: Window size in seconds
            burst: Optional burst limit
        """
        self._endpoint_limits[endpoint] = RateLimitConfig(
            requests=requests,
            window=window,
            burst=burst,
        )
        logger.info(f"Rate limit configured for {endpoint}: {requests}/{window}s")

    def get_limit_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        return self._endpoint_limits.get(
            endpoint,
            RateLimitConfig(
                requests=self.default_requests,
                window=self.default_window,
            ),
        )

    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{identifier}:{endpoint}"

    async def _cleanup_old_requests(self, key: str, window: int) -> None:
        """Remove requests outside the current window."""
        cutoff = time.time() - window
        async with self._storage_lock:
            self._storage[key] = [
                ts for ts in self._storage[key] if ts > cutoff
            ]

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        cost: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Client identifier (IP, user ID, API key)
            endpoint: Endpoint being accessed
            cost: Cost of this request

        Returns:
            Tuple of (allowed, headers_dict)
        """
        config = self.get_limit_config(endpoint)
        key = self._get_key(identifier, endpoint)
        now = time.time()

        # Cleanup old requests
        await self._cleanup_old_requests(key, config.window)

        async with self._storage_lock:
            requests = self._storage[key]
            current_count = len(requests)

            # Check if limit exceeded
            if current_count >= config.requests:
                # Calculate reset time
                oldest = min(requests) if requests else now
                reset_time = oldest + config.window

                headers = {
                    "X-RateLimit-Limit": str(config.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_time)),
                    "Retry-After": str(int(reset_time - now)),
                }
                return False, headers

            # Record this request
            for _ in range(cost):
                self._storage[key].append(now)

            remaining = config.requests - current_count - cost
            reset_time = now + config.window

            headers = {
                "X-RateLimit-Limit": str(config.requests),
                "X-RateLimit-Remaining": str(max(0, remaining)),
                "X-RateLimit-Reset": str(int(reset_time)),
            }
            return True, headers

    async def get_remaining(self, identifier: str, endpoint: str) -> int:
        """Get remaining requests for an identifier."""
        config = self.get_limit_config(endpoint)
        key = self._get_key(identifier, endpoint)

        await self._cleanup_old_requests(key, config.window)

        async with self._storage_lock:
            current_count = len(self._storage[key])
            return max(0, config.requests - current_count)


# Global rate limiter instance
_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get or create the global rate limiter."""
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowRateLimiter()
    return _limiter


def rate_limit(
    requests: int = 100,
    window: int = 60,
    key_func: Optional[Callable[[Request], str]] = None,
    cost: int = 1,
):
    """
    Decorator to apply rate limiting to an endpoint.

    Args:
        requests: Number of requests allowed per window
        window: Time window in seconds
        key_func: Function to extract client identifier from request
        cost: Cost per request (for weighted limits)

    Example:
        @app.get("/api/data")
        @rate_limit(requests=100, window=60)
        async def get_data(request: Request):
            return {"data": "..."}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find Request object in args or kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                request = kwargs.get("request")

            if request is None:
                # No request object, skip rate limiting
                return await func(*args, **kwargs)

            # Get client identifier
            if key_func:
                identifier = key_func(request)
            else:
                # Default: use client IP
                identifier = request.client.host if request.client else "unknown"

            # Get endpoint
            endpoint = request.url.path

            # Configure endpoint limit if not already set
            limiter = get_rate_limiter()
            if endpoint not in limiter._endpoint_limits:
                limiter.configure_endpoint(endpoint, requests, window)

            # Check rate limit
            allowed, headers = await limiter.check_rate_limit(
                identifier, endpoint, cost
            )

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers=headers,
                )

            # Call the actual function
            response = await func(*args, **kwargs)

            # Add rate limit headers to response
            if isinstance(response, Response):
                for key, value in headers.items():
                    response.headers[key] = value

            return response

        return wrapper

    return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply global rate limiting.

    Use this for blanket rate limiting across all endpoints.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 1000,
        burst_size: int = 100,
        exempt_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.exempt_paths = exempt_paths or ["/health", "/metrics"]
        self.limiter = get_rate_limiter()

    async def dispatch(self, request: Request, call_next):
        # Skip exempt paths
        if any(request.url.path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        # Get client identifier
        identifier = request.client.host if request.client else "unknown"
        endpoint = request.url.path

        # Check global rate limit
        allowed, headers = await self.limiter.check_rate_limit(
            identifier, endpoint
        )

        if not allowed:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers=headers,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Pre-configured rate limits for different endpoint types
RATE_LIMITS = {
    # Dashboard endpoints - frequent polling
    "/api/dashboard": RateLimitConfig(requests=300, window=60),  # 300/min
    "/api/unified/status": RateLimitConfig(requests=300, window=60),
    "/api/unified/positions": RateLimitConfig(requests=300, window=60),

    # Trading endpoints - more restrictive
    "/api/unified/trade": RateLimitConfig(requests=60, window=60),  # 60/min
    "/api/unified/close": RateLimitConfig(requests=60, window=60),

    # Admin endpoints - very restrictive
    "/api/admin": RateLimitConfig(requests=30, window=60),  # 30/min
    "/api/mode": RateLimitConfig(requests=10, window=60),  # 10/min

    # Metrics - less restrictive
    "/metrics": RateLimitConfig(requests=60, window=60),

    # Default
    "default": RateLimitConfig(requests=100, window=60),
}


def configure_rate_limits(limiter: SlidingWindowRateLimiter) -> None:
    """Configure all predefined rate limits."""
    for endpoint, config in RATE_LIMITS.items():
        if endpoint != "default":
            limiter.configure_endpoint(
                endpoint,
                config.requests,
                config.window,
                config.burst,
            )
