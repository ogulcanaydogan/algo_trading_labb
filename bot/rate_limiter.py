"""
Rate Limiter and Caching utilities for API requests.

Provides:
- Token bucket rate limiting
- Exponential backoff with jitter
- In-memory and disk caching
- Request queuing
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""

    def default(self, obj: Any) -> Any:
        # Handle pandas Timestamp
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        # Handle numpy types
        if hasattr(obj, "item"):
            return obj.item()
        # Handle numpy arrays
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Rate Limiter with Token Bucket Algorithm
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""

    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    requests_per_day: Optional[int] = None
    burst_size: int = 5  # Allow burst of requests
    min_interval_seconds: float = 1.0  # Minimum time between requests
    max_retries: int = 5
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 300.0
    jitter_factor: float = 0.25  # Add randomness to backoff


class TokenBucket:
    """
    Token bucket rate limiter.

    Provides smooth rate limiting with burst capacity.
    Thread-safe implementation.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # max tokens (burst size)
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: int = 1, block: bool = True, timeout: float = None) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            block: If True, wait until tokens are available
            timeout: Maximum time to wait (only if block=True)

        Returns:
            True if tokens were acquired, False otherwise
        """
        deadline = time.monotonic() + (timeout or float('inf')) if block else time.monotonic()

        with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not block or time.monotonic() >= deadline:
                    return False

                # Calculate wait time for tokens to be available
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                wait_time = min(wait_time, deadline - time.monotonic())

                if wait_time > 0:
                    # Release lock while waiting
                    self._lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        self._lock.acquire()

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate how long to wait for tokens."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.rate


class RateLimiter:
    """
    Multi-tier rate limiter with backoff and retry support.

    Supports minute/hour/day rate limits with exponential backoff.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Create token buckets for different time windows
        self._minute_bucket = TokenBucket(
            rate=self.config.requests_per_minute / 60.0,
            capacity=self.config.burst_size,
        )

        self._hour_bucket = TokenBucket(
            rate=self.config.requests_per_hour / 3600.0,
            capacity=self.config.burst_size * 2,
        )

        if self.config.requests_per_day:
            self._day_bucket = TokenBucket(
                rate=self.config.requests_per_day / 86400.0,
                capacity=self.config.burst_size * 3,
            )
        else:
            self._day_bucket = None

        self._last_request_time = 0.0
        self._lock = threading.Lock()
        self._consecutive_errors = 0
        self._backoff_until = 0.0

    def acquire(self, block: bool = True, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a request.

        Checks all rate limit tiers and enforces minimum interval.

        Args:
            block: If True, wait until request is allowed
            timeout: Maximum time to wait

        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            # Check if we're in backoff period
            now = time.monotonic()
            if self._backoff_until > now:
                if not block:
                    return False
                wait_time = self._backoff_until - now
                if wait_time > timeout:
                    return False
                time.sleep(wait_time)

            # Enforce minimum interval
            elapsed = now - self._last_request_time
            if elapsed < self.config.min_interval_seconds:
                if not block:
                    return False
                time.sleep(self.config.min_interval_seconds - elapsed)

        # Check all buckets
        if not self._minute_bucket.acquire(block=block, timeout=timeout):
            return False
        if not self._hour_bucket.acquire(block=block, timeout=timeout):
            return False
        if self._day_bucket and not self._day_bucket.acquire(block=block, timeout=timeout):
            return False

        with self._lock:
            self._last_request_time = time.monotonic()

        return True

    def report_success(self) -> None:
        """Report successful request to reset error count."""
        with self._lock:
            self._consecutive_errors = 0
            self._backoff_until = 0.0

    def report_error(self, is_rate_limit: bool = False) -> float:
        """
        Report failed request and calculate backoff time.

        Args:
            is_rate_limit: True if the error was a rate limit (429)

        Returns:
            Backoff time in seconds
        """
        with self._lock:
            self._consecutive_errors += 1

            # Calculate exponential backoff with jitter
            backoff = self.config.base_backoff_seconds * (2 ** min(self._consecutive_errors - 1, 8))
            backoff = min(backoff, self.config.max_backoff_seconds)

            # Add jitter
            jitter = backoff * self.config.jitter_factor * (random.random() * 2 - 1)
            backoff = max(0.1, backoff + jitter)

            # For rate limit errors, use longer backoff
            if is_rate_limit:
                backoff = max(backoff, 60.0)  # At least 1 minute for rate limits

            self._backoff_until = time.monotonic() + backoff
            logger.warning(
                f"Rate limiter backoff: {backoff:.1f}s (error #{self._consecutive_errors})"
            )

            return backoff

    def get_wait_time(self) -> float:
        """Get estimated wait time for next request."""
        with self._lock:
            now = time.monotonic()
            if self._backoff_until > now:
                return self._backoff_until - now

        # Check buckets
        wait_times = [
            self._minute_bucket.wait_time(),
            self._hour_bucket.wait_time(),
        ]
        if self._day_bucket:
            wait_times.append(self._day_bucket.wait_time())

        return max(wait_times)

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        return {
            "consecutive_errors": self._consecutive_errors,
            "in_backoff": time.monotonic() < self._backoff_until,
            "backoff_remaining": max(0, self._backoff_until - time.monotonic()),
            "minute_bucket_tokens": self._minute_bucket.tokens,
            "hour_bucket_tokens": self._hour_bucket.tokens,
            "estimated_wait": self.get_wait_time(),
        }


# =============================================================================
# Caching Layer
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with metadata."""

    value: T
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.created_at + self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache(Generic[T]):
    """
    Thread-safe in-memory cache with TTL support.

    Features:
    - LRU eviction when capacity is reached
    - TTL-based expiration
    - Hit/miss statistics
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: float = 300.0,  # 5 minutes
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache if present and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: Optional[float] = None) -> None:
        """Set value in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl,
            )

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class DiskCache:
    """
    SQLite-based disk cache for persistent storage.

    Features:
    - Persists across restarts
    - Automatic cleanup of expired entries
    - Thread-safe
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        db_name: str = "market_data_cache.db",
        default_ttl_seconds: float = 3600.0,  # 1 hour
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.default_ttl = default_ttl_seconds
        self._local = threading.local()

        # Initialize database
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        conn.commit()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and not expired."""
        conn = self._get_conn()
        now = time.time()

        cursor = conn.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, now),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Update access stats
        conn.execute(
            "UPDATE cache SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
            (now, key),
        )
        conn.commit()

        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Set value in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()
        expires_at = now + ttl

        # Serialize value with custom encoder for datetime/Timestamp objects
        try:
            serialized = json.dumps(value, cls=DateTimeEncoder)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cache serialization failed: {e}")
            return  # Skip caching if serialization fails

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, created_at, expires_at, access_count, last_accessed)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (key, serialized, now, expires_at, now),
        )
        conn.commit()

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_conn()
        now = time.time()

        cursor = conn.execute("SELECT COUNT(*) as total FROM cache")
        total = cursor.fetchone()["total"]

        cursor = conn.execute("SELECT COUNT(*) as valid FROM cache WHERE expires_at > ?", (now,))
        valid = cursor.fetchone()["valid"]

        cursor = conn.execute("SELECT SUM(access_count) as accesses FROM cache")
        row = cursor.fetchone()
        accesses = row["accesses"] or 0

        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": total - valid,
            "total_accesses": accesses,
        }


class MultiLevelCache:
    """
    Multi-level cache with memory and disk layers.

    Checks memory first (fast), then disk (persistent).
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        memory_max_size: int = 500,
        memory_ttl: float = 60.0,  # 1 minute in memory
        disk_ttl: float = 3600.0,  # 1 hour on disk
    ):
        self.memory = MemoryCache(max_size=memory_max_size, default_ttl_seconds=memory_ttl)
        self.disk = DiskCache(cache_dir=cache_dir, default_ttl_seconds=disk_ttl)
        self.memory_ttl = memory_ttl
        self.disk_ttl = disk_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get from memory first, then disk."""
        # Check memory
        value = self.memory.get(key)
        if value is not None:
            return value

        # Check disk
        value = self.disk.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory.set(key, value, self.memory_ttl)
            return value

        return None

    def set(self, key: str, value: Any, memory_ttl: Optional[float] = None, disk_ttl: Optional[float] = None) -> None:
        """Set in both memory and disk."""
        self.memory.set(key, value, memory_ttl or self.memory_ttl)
        self.disk.set(key, value, disk_ttl or self.disk_ttl)

    def delete(self, key: str) -> bool:
        """Delete from both layers."""
        mem_deleted = self.memory.delete(key)
        disk_deleted = self.disk.delete(key)
        return mem_deleted or disk_deleted

    def clear(self) -> None:
        """Clear both layers."""
        self.memory.clear()
        # Don't clear disk by default to preserve historical data

    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries in both layers."""
        return {
            "memory_removed": self.memory.cleanup_expired(),
            "disk_removed": self.disk.cleanup_expired(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "memory": self.memory.get_stats(),
            "disk": self.disk.get_stats(),
        }


# =============================================================================
# Request Decorators
# =============================================================================


def rate_limited(limiter: RateLimiter):
    """Decorator to apply rate limiting to a function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire(block=True, timeout=120.0):
                raise RuntimeError("Rate limit exceeded, request blocked")

            try:
                result = func(*args, **kwargs)
                limiter.report_success()
                return result
            except Exception as e:
                # Check if it's a rate limit error
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str
                limiter.report_error(is_rate_limit=is_rate_limit)
                raise

        return wrapper

    return decorator


def cached(cache: Union[MemoryCache, DiskCache, MultiLevelCache], ttl: Optional[float] = None):
    """Decorator to cache function results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            key = hashlib.md5(key_data.encode()).hexdigest()

            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            if ttl is not None:
                cache.set(key, result, ttl)
            else:
                cache.set(key, result)

            return result

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Decorator to retry failed function calls with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay += random.uniform(0, delay * 0.1)

                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper

    return decorator


# =============================================================================
# Yahoo Finance Specific Rate Limiter
# =============================================================================


# Pre-configured rate limiter for Yahoo Finance
YAHOO_RATE_LIMIT_CONFIG = RateLimitConfig(
    requests_per_minute=20,  # Conservative limit
    requests_per_hour=500,
    requests_per_day=2000,
    burst_size=3,
    min_interval_seconds=3.0,  # At least 3 seconds between requests
    max_retries=5,
    base_backoff_seconds=5.0,
    max_backoff_seconds=300.0,
)


def create_yahoo_rate_limiter() -> RateLimiter:
    """Create a rate limiter configured for Yahoo Finance."""
    return RateLimiter(YAHOO_RATE_LIMIT_CONFIG)


# Global Yahoo Finance rate limiter (singleton)
_yahoo_limiter: Optional[RateLimiter] = None


def get_yahoo_rate_limiter() -> RateLimiter:
    """Get the global Yahoo Finance rate limiter."""
    global _yahoo_limiter
    if _yahoo_limiter is None:
        _yahoo_limiter = create_yahoo_rate_limiter()
    return _yahoo_limiter
