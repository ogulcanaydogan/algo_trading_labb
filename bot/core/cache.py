"""
Redis-based caching layer for the trading bot.

Provides caching for:
- Market data (OHLCV, prices)
- API responses
- ML predictions
- Rate limit tracking

Falls back to in-memory caching if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic cache decorators
T = TypeVar("T")

# Try to import redis
try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None


@dataclass
class CacheConfig:
    """Configuration for the cache system."""

    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    default_ttl: int = 300  # 5 minutes default TTL
    market_data_ttl: int = 10  # 10 seconds for market data
    ml_prediction_ttl: int = 60  # 1 minute for ML predictions
    api_response_ttl: int = 30  # 30 seconds for API responses
    max_memory_items: int = 10000  # Max items in memory cache
    prefix: str = "algo_trading:"
    enable_compression: bool = True


class InMemoryCache:
    """Simple in-memory LRU cache as fallback when Redis is unavailable."""

    def __init__(self, max_items: int = 10000):
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._max_items = max_items
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if expiry and time.time() > expiry:
            # Expired
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache."""
        # Evict if necessary
        while len(self._cache) >= self._max_items:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)
            else:
                break

        expiry = time.time() + ttl if ttl else None
        self._cache[key] = (value, expiry)
        self._access_order.append(key)

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern (simple wildcard support)."""
        if pattern == "*":
            return list(self._cache.keys())

        # Simple prefix matching
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._cache.keys() if k.startswith(prefix)]

        return [k for k in self._cache.keys() if k == pattern]


class CacheManager:
    """
    Unified cache manager supporting Redis with in-memory fallback.

    Usage:
        cache = CacheManager()

        # Basic operations
        cache.set("key", value, ttl=60)
        value = cache.get("key")

        # Specialized methods
        cache.cache_market_data("BTC/USDT", ohlcv_data)
        ohlcv = cache.get_market_data("BTC/USDT")
    """

    _instance: Optional["CacheManager"] = None

    def __new__(cls, config: Optional[CacheConfig] = None) -> "CacheManager":
        """Singleton pattern for cache manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[CacheConfig] = None):
        if self._initialized:
            return

        self.config = config or CacheConfig()
        self._memory_cache = InMemoryCache(self.config.max_memory_items)
        self._redis_client: Optional[Any] = None
        self._async_redis_client: Optional[Any] = None
        self._redis_available = False
        self._initialized = True

        # Try to connect to Redis
        self._connect_redis()

    def _connect_redis(self) -> None:
        """Attempt to connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.info("Redis package not installed, using in-memory cache")
            return

        try:
            self._redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            # Test connection
            self._redis_client.ping()
            self._redis_available = True
            logger.info(f"Connected to Redis at {self.config.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory cache")
            self._redis_client = None
            self._redis_available = False

    async def _get_async_redis(self) -> Optional[Any]:
        """Get async Redis client (lazy initialization)."""
        if not REDIS_AVAILABLE or not self._redis_available:
            return None

        if self._async_redis_client is None:
            try:
                self._async_redis_client = await aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=False,
                )
                await self._async_redis_client.ping()
            except Exception as e:
                logger.warning(f"Async Redis connection failed: {e}")
                self._async_redis_client = None

        return self._async_redis_client

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.config.prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        try:
            # Try JSON first (more readable)
            return json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize a cached value."""
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(data)

    # =========================================================================
    # Synchronous API
    # =========================================================================

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        full_key = self._make_key(key)

        if self._redis_available and self._redis_client:
            try:
                data = self._redis_client.get(full_key)
                if data:
                    return self._deserialize(data)
            except Exception as e:
                logger.debug(f"Redis get error: {e}, falling back to memory")

        return self._memory_cache.get(full_key)

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set a value in cache."""
        full_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        data = self._serialize(value)

        # Always set in memory cache
        self._memory_cache.set(full_key, value, ttl)

        if self._redis_available and self._redis_client:
            try:
                self._redis_client.setex(full_key, ttl, data)
                return True
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        return True

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        full_key = self._make_key(key)
        self._memory_cache.delete(full_key)

        if self._redis_available and self._redis_client:
            try:
                self._redis_client.delete(full_key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")

        return True

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        full_pattern = self._make_key(pattern)
        count = 0

        # Clear from memory
        for key in self._memory_cache.keys(full_pattern):
            self._memory_cache.delete(key)
            count += 1

        if self._redis_available and self._redis_client:
            try:
                keys = self._redis_client.keys(full_pattern)
                if keys:
                    self._redis_client.delete(*keys)
                    count = len(keys)
            except Exception as e:
                logger.debug(f"Redis clear pattern error: {e}")

        return count

    # =========================================================================
    # Async API
    # =========================================================================

    async def aget(self, key: str) -> Optional[Any]:
        """Async get a value from cache."""
        full_key = self._make_key(key)

        async_redis = await self._get_async_redis()
        if async_redis:
            try:
                data = await async_redis.get(full_key)
                if data:
                    return self._deserialize(data)
            except Exception as e:
                logger.debug(f"Async Redis get error: {e}")

        return self._memory_cache.get(full_key)

    async def aset(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Async set a value in cache."""
        full_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        data = self._serialize(value)

        # Always set in memory cache
        self._memory_cache.set(full_key, value, ttl)

        async_redis = await self._get_async_redis()
        if async_redis:
            try:
                await async_redis.setex(full_key, ttl, data)
                return True
            except Exception as e:
                logger.debug(f"Async Redis set error: {e}")

        return True

    async def adelete(self, key: str) -> bool:
        """Async delete a key from cache."""
        full_key = self._make_key(key)
        self._memory_cache.delete(full_key)

        async_redis = await self._get_async_redis()
        if async_redis:
            try:
                await async_redis.delete(full_key)
            except Exception as e:
                logger.debug(f"Async Redis delete error: {e}")

        return True

    # =========================================================================
    # Specialized Cache Methods
    # =========================================================================

    def cache_market_data(
        self, symbol: str, data: Any, timeframe: str = "1h"
    ) -> None:
        """Cache market data (OHLCV, prices)."""
        key = f"market:{symbol}:{timeframe}"
        self.set(key, data, self.config.market_data_ttl)

    def get_market_data(
        self, symbol: str, timeframe: str = "1h"
    ) -> Optional[Any]:
        """Get cached market data."""
        key = f"market:{symbol}:{timeframe}"
        return self.get(key)

    def cache_ml_prediction(
        self, symbol: str, model_name: str, prediction: Any
    ) -> None:
        """Cache an ML prediction."""
        key = f"ml:{model_name}:{symbol}"
        self.set(key, prediction, self.config.ml_prediction_ttl)

    def get_ml_prediction(
        self, symbol: str, model_name: str
    ) -> Optional[Any]:
        """Get a cached ML prediction."""
        key = f"ml:{model_name}:{symbol}"
        return self.get(key)

    def cache_api_response(
        self, endpoint: str, params: Dict[str, Any], response: Any
    ) -> None:
        """Cache an API response."""
        # Create a hash of the params for the key
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:12]
        key = f"api:{endpoint}:{params_hash}"
        self.set(key, response, self.config.api_response_ttl)

    def get_api_response(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get a cached API response."""
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:12]
        key = f"api:{endpoint}:{params_hash}"
        return self.get(key)

    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        count = 0
        full_prefix = self.config.prefix

        # Get all keys and filter those containing the symbol
        for key in list(self._memory_cache._cache.keys()):
            if symbol in key:
                self._memory_cache.delete(key)
                count += 1

        if self._redis_available and self._redis_client:
            try:
                # Use SCAN to find keys containing the symbol
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(
                        cursor, match=f"{full_prefix}*{symbol}*"
                    )
                    if keys:
                        self._redis_client.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug(f"Redis invalidate symbol error: {e}")

        return count

    # =========================================================================
    # Stats and Diagnostics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "redis_available": self._redis_available,
            "memory_cache_size": len(self._memory_cache._cache),
            "max_memory_items": self.config.max_memory_items,
        }

        if self._redis_available and self._redis_client:
            try:
                info = self._redis_client.info("memory")
                stats["redis_used_memory"] = info.get("used_memory_human", "N/A")
                stats["redis_keys"] = self._redis_client.dbsize()
            except Exception:
                pass

        return stats

    def health_check(self) -> Dict[str, str]:
        """Check cache health."""
        result = {"memory_cache": "healthy"}

        if REDIS_AVAILABLE:
            if self._redis_available and self._redis_client:
                try:
                    self._redis_client.ping()
                    result["redis"] = "healthy"
                except Exception:
                    result["redis"] = "unhealthy"
            else:
                result["redis"] = "not_connected"
        else:
            result["redis"] = "not_installed"

        return result


# =========================================================================
# Decorator for caching function results
# =========================================================================


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for the cache key
        key_builder: Optional function to build custom cache key

    Usage:
        @cached(ttl=60, key_prefix="my_func")
        def expensive_calculation(x, y):
            return x + y
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = CacheManager()

            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = ":".join(key_parts)

            # Try to get from cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


def cached_async(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async decorator for caching function results.

    Usage:
        @cached_async(ttl=60, key_prefix="my_async_func")
        async def expensive_async_calculation(x, y):
            return x + y
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            cache = CacheManager()

            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = ":".join(key_parts)

            # Try to get from cache
            cached_value = await cache.aget(key)
            if cached_value is not None:
                return cached_value

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.aset(key, result, ttl)
            return result

        return wrapper

    return decorator


# Global cache instance for convenience
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
