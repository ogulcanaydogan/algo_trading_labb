"""API Security module for authentication and rate limiting."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

# Initialize logger
logger = logging.getLogger(__name__)

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> Optional[str]:
    """Retrieve the configured API key from environment."""
    return os.getenv("API_KEY")


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format and strength."""
    if not api_key:
        return False

    # Minimum length for API keys
    if len(api_key) < 16:
        return False

    # Check for reasonable character variety
    has_upper = any(c.isupper() for c in api_key)
    has_lower = any(c.islower() for c in api_key)
    has_digit = any(c.isdigit() for c in api_key)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in api_key)

    # Require at least 3 of the 4 character types
    variety_score = sum([has_upper, has_lower, has_digit, has_special])
    if variety_score < 3:
        return False

    # Check for common weak patterns
    weak_patterns = [
        r"123456",
        r"password",
        r"secret",
        r"key",
        r"token",
        r"admin",
        r"root",
        r"test",
        r"demo",
    ]

    api_key_lower = api_key.lower()
    for pattern in weak_patterns:
        if re.search(pattern, api_key_lower):
            return False

    return True


async def verify_api_key_strict(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> str:
    """
    Strict API key verification - no development bypass.
    """
    configured_key = get_api_key()

    if not configured_key:
        logger.error("API authentication not configured - blocking request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication not configured. Please set API_KEY environment variable.",
        )

    if not api_key:
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key format
    if not validate_api_key_format(api_key):
        logger.warning(
            f"Invalid API key format from {request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format.",
        )

    if not validate_api_key_format(configured_key):
        logger.error("Configured API key does not meet security requirements")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error - API key format invalid.",
        )

    # Constant-time comparison to prevent timing attacks
    if not _constant_time_compare(api_key, configured_key):
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Failed API key attempt from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[str]:
    """
    Verify API key from request header.

    For backward compatibility, supports both strict and development modes.
    In production, use verify_api_key_strict instead.
    """
    # Check if we're in development mode
    development_mode = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"

    if development_mode:
        logger.warning("Running in development mode - API authentication bypassed")
        return None

    # Use strict verification in production
    return await verify_api_key_strict(request, api_key)


def _constant_time_compare(val1: str, val2: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    """
    if len(val1) != len(val2):
        return False

    result = 0
    for x, y in zip(val1, val2):
        result |= ord(x) ^ ord(y)

    return result == 0


class RateLimiter:
    """Thread-safe in-memory rate limiter using sliding window."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        max_clients: int = 10000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.max_clients = max_clients
        self._minute_windows: dict[str, list[float]] = defaultdict(list)
        self._hour_windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _clean_window(self, window: list[float], cutoff: float) -> list[float]:
        """Remove timestamps older than cutoff."""
        return [ts for ts in window if ts > cutoff]

    def _maybe_cleanup_stale_clients(self, now: float) -> None:
        """Remove stale client entries to prevent memory leaks."""
        if now - self._last_cleanup < self._cleanup_interval:
            return

        hour_ago = now - 3600

        # Remove clients with no recent activity
        stale_clients = [
            client_id for client_id, timestamps in self._hour_windows.items()
            if not timestamps or max(timestamps) < hour_ago
        ]

        for client_id in stale_clients:
            self._minute_windows.pop(client_id, None)
            self._hour_windows.pop(client_id, None)

        # If still too many clients, remove oldest ones
        if len(self._hour_windows) > self.max_clients:
            # Sort by most recent activity
            sorted_clients = sorted(
                self._hour_windows.keys(),
                key=lambda c: max(self._hour_windows[c]) if self._hour_windows[c] else 0
            )
            # Remove oldest half
            for client_id in sorted_clients[:len(sorted_clients) // 2]:
                self._minute_windows.pop(client_id, None)
                self._hour_windows.pop(client_id, None)

        self._last_cleanup = now

    def is_allowed(self, client_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed for the given client.

        Thread-safe implementation with automatic cleanup.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        with self._lock:
            # Periodic cleanup of stale clients
            self._maybe_cleanup_stale_clients(now)

            # Clean old entries for this client
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

    def get_remaining(self, client_id: str) -> dict[str, int]:
        """Get remaining requests for a client."""
        with self._lock:
            minute_used = len(self._minute_windows.get(client_id, []))
            hour_used = len(self._hour_windows.get(client_id, []))
            return {
                "minute_remaining": max(0, self.requests_per_minute - minute_used),
                "hour_remaining": max(0, self.requests_per_hour - hour_used),
            }


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
