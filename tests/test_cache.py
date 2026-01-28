"""
Tests for the Redis caching layer.
"""

import asyncio
import time

import pytest

from bot.core.cache import (
    CacheConfig,
    CacheManager,
    InMemoryCache,
    cached,
    cached_async,
    get_cache,
)


class TestInMemoryCache:
    """Tests for the in-memory LRU cache."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        cache = InMemoryCache(max_items=100)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", {"nested": "data"})
        assert cache.get("key2") == {"nested": "data"}

    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key."""
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = InMemoryCache()

        cache.set("expiring", "value", ttl=1)
        assert cache.get("expiring") == "value"

        time.sleep(1.1)
        assert cache.get("expiring") is None

    def test_delete(self):
        """Test delete operation."""
        cache = InMemoryCache()

        cache.set("to_delete", "value")
        assert cache.get("to_delete") == "value"

        result = cache.delete("to_delete")
        assert result is True
        assert cache.get("to_delete") is None

    def test_delete_nonexistent(self):
        """Test deleting a nonexistent key."""
        cache = InMemoryCache()
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clear operation."""
        cache = InMemoryCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max items reached."""
        cache = InMemoryCache(max_items=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still there (was accessed)
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_keys_pattern_matching(self):
        """Test key pattern matching."""
        cache = InMemoryCache()

        cache.set("market:BTC:1h", "data1")
        cache.set("market:ETH:1h", "data2")
        cache.set("ml:prediction", "data3")

        all_keys = cache.keys("*")
        assert len(all_keys) == 3

        market_keys = cache.keys("market:*")
        assert len(market_keys) == 2
        assert "market:BTC:1h" in market_keys
        assert "market:ETH:1h" in market_keys


class TestCacheManager:
    """Tests for the CacheManager class."""

    def setup_method(self):
        """Reset singleton for each test."""
        CacheManager._instance = None

    def test_singleton_pattern(self):
        """Test that CacheManager is a singleton."""
        cache1 = CacheManager()
        cache2 = CacheManager()
        assert cache1 is cache2

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = CacheManager()

        cache.set("test_key", "test_value", ttl=60)
        assert cache.get("test_key") == "test_value"

        cache.delete("test_key")
        assert cache.get("test_key") is None

    def test_complex_data_types(self):
        """Test caching complex data types."""
        cache = CacheManager()

        # Dict
        cache.set("dict", {"a": 1, "b": [1, 2, 3]})
        assert cache.get("dict") == {"a": 1, "b": [1, 2, 3]}

        # List
        cache.set("list", [1, 2, {"nested": True}])
        assert cache.get("list") == [1, 2, {"nested": True}]

        # Nested structure
        complex_data = {
            "prices": [100.0, 101.5, 99.8],
            "metadata": {"symbol": "BTC", "timeframe": "1h"},
        }
        cache.set("complex", complex_data)
        assert cache.get("complex") == complex_data

    def test_market_data_caching(self):
        """Test specialized market data caching."""
        cache = CacheManager()

        ohlcv_data = [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000}
        ]

        cache.cache_market_data("BTC/USDT", ohlcv_data, "1h")
        cached = cache.get_market_data("BTC/USDT", "1h")
        assert cached == ohlcv_data

    def test_ml_prediction_caching(self):
        """Test ML prediction caching."""
        cache = CacheManager()

        prediction = {
            "action": "LONG",
            "confidence": 0.85,
            "probabilities": [0.1, 0.05, 0.85],
        }

        cache.cache_ml_prediction("BTC/USDT", "random_forest", prediction)
        cached = cache.get_ml_prediction("BTC/USDT", "random_forest")
        assert cached == prediction

    def test_api_response_caching(self):
        """Test API response caching."""
        cache = CacheManager()

        response = {"status": "success", "data": [1, 2, 3]}
        params = {"symbol": "BTC/USDT", "limit": 100}

        cache.cache_api_response("/market/data", params, response)
        cached = cache.get_api_response("/market/data", params)
        assert cached == response

        # Different params should not match
        different_params = {"symbol": "ETH/USDT", "limit": 100}
        assert cache.get_api_response("/market/data", different_params) is None

    def test_invalidate_symbol(self):
        """Test symbol invalidation."""
        cache = CacheManager()

        cache.cache_market_data("BTC/USDT", {"price": 100}, "1h")
        cache.cache_market_data("BTC/USDT", {"price": 100}, "4h")
        cache.cache_ml_prediction("BTC/USDT", "model1", {"pred": 1})

        # Should still exist
        assert cache.get_market_data("BTC/USDT", "1h") is not None

        # Invalidate
        cache.invalidate_symbol("BTC/USDT")

        # Should be gone
        assert cache.get_market_data("BTC/USDT", "1h") is None
        assert cache.get_market_data("BTC/USDT", "4h") is None
        assert cache.get_ml_prediction("BTC/USDT", "model1") is None

    def test_health_check(self):
        """Test health check."""
        cache = CacheManager()
        health = cache.health_check()

        assert "memory_cache" in health
        assert health["memory_cache"] == "healthy"

    def test_get_stats(self):
        """Test getting cache stats."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert "memory_cache_size" in stats
        assert stats["memory_cache_size"] >= 2


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    def setup_method(self):
        """Reset singleton for each test."""
        CacheManager._instance = None

    def test_cached_function(self):
        """Test caching a function result."""
        call_count = 0

        @cached(ttl=60, key_prefix="test")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call - executes function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call - returns cached result
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented

        # Different args - executes function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_cached_with_kwargs(self):
        """Test caching with keyword arguments."""
        call_count = 0

        @cached(ttl=60, key_prefix="kwargs_test")
        def func_with_kwargs(a, b=10):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = func_with_kwargs(1, b=20)
        assert result1 == 21
        assert call_count == 1

        # Same kwargs - cached
        result2 = func_with_kwargs(1, b=20)
        assert result2 == 21
        assert call_count == 1

        # Different kwargs - not cached
        result3 = func_with_kwargs(1, b=30)
        assert result3 == 31
        assert call_count == 2


class TestCachedAsyncDecorator:
    """Tests for the @cached_async decorator."""

    def setup_method(self):
        """Reset singleton for each test."""
        CacheManager._instance = None

    @pytest.mark.asyncio
    async def test_cached_async_function(self):
        """Test caching an async function result."""
        call_count = 0

        @cached_async(ttl=60, key_prefix="async_test")
        async def async_expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * y

        # First call - executes function
        result1 = await async_expensive_function(2, 3)
        assert result1 == 6
        assert call_count == 1

        # Second call - returns cached result
        result2 = await async_expensive_function(2, 3)
        assert result2 == 6
        assert call_count == 1  # Not incremented

    @pytest.mark.asyncio
    async def test_async_cache_operations(self):
        """Test async cache operations directly."""
        cache = CacheManager()

        await cache.aset("async_key", "async_value", ttl=60)
        result = await cache.aget("async_key")
        assert result == "async_value"

        await cache.adelete("async_key")
        result = await cache.aget("async_key")
        assert result is None


class TestGetCache:
    """Tests for the get_cache helper function."""

    def setup_method(self):
        """Reset singleton for each test."""
        CacheManager._instance = None

    def test_get_cache_returns_singleton(self):
        """Test that get_cache returns the singleton instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

    def test_get_cache_functional(self):
        """Test get_cache returns a functional cache."""
        cache = get_cache()
        cache.set("test", "value")
        assert cache.get("test") == "value"
