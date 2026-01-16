"""
Tests for data cache module.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

import pandas as pd

from bot.data.cache import (
    InMemoryCache,
    SQLiteCache,
    CacheManager,
    make_ohlcv_key,
    make_quote_key,
    make_symbol_key,
)


class TestInMemoryCache:
    """Test InMemoryCache class."""

    @pytest.fixture
    def cache(self):
        """Create in-memory cache."""
        return InMemoryCache(max_size=10)

    def test_basic_set_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1", ttl=60)
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self, cache):
        """Test getting non-existent key."""
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, cache):
        """Test TTL expiry."""
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_delete(self, cache):
        """Test delete operation."""
        cache.set("key1", "value1", ttl=60)
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_delete_nonexistent(self, cache):
        """Test deleting non-existent key doesn't error."""
        cache.delete("nonexistent")

    def test_clear(self, cache):
        """Test clear operation."""
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_eviction(self, cache):
        """Test LRU eviction when at capacity."""
        # Fill cache to capacity
        for i in range(10):
            cache.set(f"key{i}", f"value{i}", ttl=60)

        # Add one more (should evict oldest)
        cache.set("key10", "value10", ttl=60)

        # key0 should be evicted
        assert cache.get("key0") is None
        # key10 should exist
        assert cache.get("key10") == "value10"

    def test_lru_access_updates_order(self, cache):
        """Test that accessing a key moves it to end."""
        cache.set("key0", "value0", ttl=60)
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)

        # Access key0 to move it to end
        cache.get("key0")

        # Fill rest of cache
        for i in range(3, 10):
            cache.set(f"key{i}", f"value{i}", ttl=60)

        # Add one more (should evict key1, not key0)
        cache.set("key10", "value10", ttl=60)

        assert cache.get("key0") == "value0"
        assert cache.get("key1") is None

    def test_stats(self, cache):
        """Test stats method."""
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=1)

        time.sleep(1.1)

        stats = cache.stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "active_entries" in stats
        assert stats["max_size"] == 10


class TestSQLiteCache:
    """Test SQLiteCache class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_dir):
        """Create SQLite cache."""
        return SQLiteCache(db_path=f"{temp_dir}/cache.db")

    def test_basic_set_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", {"data": "value1"}, ttl=60)
        result = cache.get("key1")
        assert result == {"data": "value1"}

    def test_get_nonexistent(self, cache):
        """Test getting non-existent key."""
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, cache):
        """Test TTL expiry."""
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_delete(self, cache):
        """Test delete operation."""
        cache.set("key1", "value1", ttl=60)
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self, cache):
        """Test clear operation."""
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_dataframe_serialization(self, cache):
        """Test DataFrame serialization."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        cache.set("df_key", df, ttl=60)

        result = cache.get("df_key")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_dict_serialization(self, cache):
        """Test dict serialization."""
        data = {"key": "value", "nested": {"a": 1}}
        cache.set("dict_key", data, ttl=60)

        result = cache.get("dict_key")
        assert result == data

    def test_list_serialization(self, cache):
        """Test list serialization."""
        data = [1, 2, {"nested": True}]
        cache.set("list_key", data, ttl=60)

        result = cache.get("list_key")
        assert result == data

    def test_cleanup_expired(self, cache):
        """Test cleanup_expired method."""
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=60)

        time.sleep(1.1)

        cleaned = cache.cleanup_expired()
        assert cleaned == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_stats(self, cache):
        """Test stats method."""
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=1)

        time.sleep(1.1)

        stats = cache.stats()
        assert "total_entries" in stats
        assert "active_entries" in stats
        assert "expired_entries" in stats
        assert "size_bytes" in stats

    def test_replace_existing_key(self, cache):
        """Test replacing existing key."""
        cache.set("key1", "old_value", ttl=60)
        cache.set("key1", "new_value", ttl=60)

        assert cache.get("key1") == "new_value"


class TestCacheManager:
    """Test CacheManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create cache manager."""
        return CacheManager(data_dir=temp_dir)

    def test_basic_set_get(self, manager):
        """Test basic set and get operations."""
        manager.set("key1", {"data": "value1"})
        assert manager.get("key1") == {"data": "value1"}

    def test_get_nonexistent(self, manager):
        """Test getting non-existent key."""
        assert manager.get("nonexistent") is None

    def test_l1_promotion(self, manager):
        """Test L1 promotion from L2."""
        # Set with L2-only TTL config by using a custom data_type
        manager.l2.set("key1", "value1", ttl=60)

        # First get should come from L2 and promote to L1
        result = manager.get("key1")
        assert result == "value1"

        # Now it should be in L1
        assert manager.l1.get("key1") == "value1"

    def test_delete(self, manager):
        """Test delete from both levels."""
        manager.set("key1", "value1")
        manager.delete("key1")

        assert manager.l1.get("key1") is None
        assert manager.l2.get("key1") is None

    def test_clear(self, manager):
        """Test clear both levels."""
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        manager.clear()

        assert manager.get("key1") is None
        assert manager.get("key2") is None

    def test_data_type_ttls(self, manager):
        """Test different TTLs for data types."""
        # Quote type should only be in L1 (L2 TTL is 0)
        manager.set("quote_key", {"bid": 100}, data_type="quote")

        # Should be in L1
        assert manager.l1.get("quote_key") is not None

    def test_stats(self, manager):
        """Test stats method."""
        manager.set("key1", "value1")

        stats = manager.stats()
        assert "l1" in stats
        assert "l2" in stats

    def test_invalidate_symbol(self, manager):
        """Test invalidating all entries for a symbol."""
        manager.set("BTC/USDT:1h", "data1")
        manager.set("BTC/USDT:4h", "data2")
        manager.set("ETH/USDT:1h", "data3")

        count = manager.invalidate_symbol("BTC/USDT")

        assert count == 2
        assert manager.l2.get("BTC/USDT:1h") is None
        assert manager.l2.get("ETH/USDT:1h") is not None

    def test_cleanup(self, manager):
        """Test cleanup method."""
        manager.l1.set("key1", "value1", ttl=1)
        manager.l2.set("key2", "value2", ttl=1)

        time.sleep(1.1)

        result = manager.cleanup()
        assert "l1_cleaned" in result
        assert "l2_cleaned" in result


class TestCacheKeyHelpers:
    """Test cache key helper functions."""

    def test_make_ohlcv_key(self):
        """Test OHLCV key generation."""
        key = make_ohlcv_key("BTC/USDT", "1h", 100)
        assert key == "ohlcv:BTC/USDT:1h:100"

    def test_make_ohlcv_key_no_limit(self):
        """Test OHLCV key without limit."""
        key = make_ohlcv_key("ETH/USDT", "4h")
        assert key == "ohlcv:ETH/USDT:4h:0"

    def test_make_quote_key(self):
        """Test quote key generation."""
        key = make_quote_key("BTC/USDT")
        assert key == "quote:BTC/USDT"

    def test_make_symbol_key(self):
        """Test symbol key generation."""
        key = make_symbol_key("AAPL")
        assert key == "symbol:AAPL"

    def test_keys_are_unique(self):
        """Test that different inputs produce different keys."""
        key1 = make_ohlcv_key("BTC/USDT", "1h", 100)
        key2 = make_ohlcv_key("BTC/USDT", "4h", 100)
        key3 = make_ohlcv_key("ETH/USDT", "1h", 100)

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
