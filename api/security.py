"""API Security module for authentication and rate limiting."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> Optional[str]:
    """Retrieve the configured API key from environment."""
    return os.getenv("API_KEY")


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[str]:
    """
    Verify the API key from request header.

    If API_KEY is not configured in environment, authentication is disabled.
    This allows for development mode without authentication.
    """
    configured_key = get_api_key()

    # If no API key is configured, skip authentication (development mode)
    if not configured_key:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_windows: dict[str, list[float]] = defaultdict(list)
        self._hour_windows: dict[str, list[float]] = defaultdict(list)

    def _clean_window(self, window: list[float], cutoff: float) -> list[float]:
        """Remove timestamps older than cutoff."""
        return [ts for ts in window if ts > cutoff]

    def is_allowed(self, client_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed for the given client.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries
        self._minute_windows[client_id] = self._clean_window(
            self._minute_windows[client_id], minute_ago
        )
        self._hour_windows[client_id] = self._clean_window(
            self._hour_windows[client_id], hour_ago
        )

        # Check minute limit
        if len(self._minute_windows[client_id]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check hour limit
        if len(self._hour_windows[client_id]) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

        # Record this request
        self._minute_windows[client_id].append(now)
        self._hour_windows[client_id].append(now)

        return True, None


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency to check rate limits.

    Uses client IP as the identifier. In production, consider using
    authenticated user ID or API key for more accurate limiting.
    """
    # Get client identifier (IP address or forwarded IP)
    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    limiter = get_rate_limiter()
    allowed, error_msg = limiter.is_allowed(client_ip)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_msg,
            headers={"Retry-After": "60"},
        )
