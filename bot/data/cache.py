"""
Multi-level caching for market data.

Provides L1 (in-memory) and L2 (SQLite) caching with automatic
TTL management and LRU eviction.
"""

import json
import io
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass


class InMemoryCache(CacheBackend):
    """
    L1 cache - fast in-memory storage with LRU eviction.

    Best for hot data with short TTLs (< 5 minutes).
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._expiry: dict = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        if key not in self._cache:
            return None
        if time.time() > self._expiry.get(key, 0):
            self.delete(key)
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value with TTL, evicting oldest if at capacity."""
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self.delete(oldest_key)

        self._cache[key] = value
        self._expiry[key] = time.time() + ttl

    def delete(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._expiry.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        active = sum(1 for exp in self._expiry.values() if exp > now)
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "active_entries": active,
            "expired_entries": len(self._cache) - active,
        }


class SQLiteCache(CacheBackend):
    """
    L2 cache - persistent SQLite storage.

    Best for data with longer TTLs (5 minutes - 1 hour).
    Survives process restarts.
    """

    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    expiry REAL NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache(expiry)")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, value_type, expiry FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                return None

            value_blob, value_type, expiry = row
            if time.time() > expiry:
                self.delete(key)
                return None

            return self._deserialize(value_blob, value_type)

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value with TTL."""
        value_blob, value_type = self._serialize(value)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, value_type, expiry, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (key, value_blob, value_type, time.time() + ttl, time.time()),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        """Remove key from cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        """Clear all cached data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE expiry < ?", (time.time(),))
            conn.commit()
            return cursor.rowcount

    def _serialize(self, value: Any) -> tuple:
        """Serialize value to bytes and type string."""
        if isinstance(value, pd.DataFrame):
            return value.to_json().encode(), "dataframe"
        elif isinstance(value, dict):
            return json.dumps(value).encode(), "dict"
        elif isinstance(value, list):
            return json.dumps(value).encode(), "list"
        else:
            return json.dumps(value).encode(), "json"

    def _deserialize(self, data: bytes, value_type: str) -> Any:
        """Deserialize bytes based on type."""
        if value_type == "dataframe":
            return pd.read_json(io.StringIO(data.decode()))
        else:
            return json.loads(data.decode())

    def stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expiry > ?", (time.time(),)
            ).fetchone()[0]
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": total - active,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024 / 1024, 2),
            }


class CacheManager:
    """
    Multi-level cache manager with write-through and read-through.

    L1 (InMemoryCache): Fast, volatile, short TTL
    L2 (SQLiteCache): Persistent, longer TTL
    """

    # Default TTLs by data type (seconds)
    DEFAULT_TTLS = {
        "ohlcv_1m": (15, 30),  # L1: 15s, L2: 30s
        "ohlcv_5m": (30, 60),  # L1: 30s, L2: 60s
        "ohlcv_15m": (60, 180),  # L1: 1m, L2: 3m
        "ohlcv_1h": (300, 1800),  # L1: 5m, L2: 30m
        "ohlcv_4h": (600, 3600),  # L1: 10m, L2: 1h
        "ohlcv_1d": (1800, 7200),  # L1: 30m, L2: 2h
        "quote": (5, 0),  # L1 only: 5s (too fast for L2)
        "symbol_info": (3600, 86400),  # L1: 1h, L2: 24h
        "default": (60, 300),  # L1: 1m, L2: 5m
    }

    def __init__(self, data_dir: str = "data"):
        self.l1 = InMemoryCache(max_size=500)
        self.l2 = SQLiteCache(db_path=f"{data_dir}/cache.db")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value with read-through strategy.

        Checks L1 first, then L2, promoting to L1 if found in L2.
        """
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1 with short TTL
            self.l1.set(key, value, ttl=60)
            return value

        return None

    def set(self, key: str, value: Any, data_type: str = "default") -> None:
        """
        Set value with write-through strategy.

        Writes to both L1 and L2 with appropriate TTLs.
        """
        l1_ttl, l2_ttl = self.DEFAULT_TTLS.get(data_type, self.DEFAULT_TTLS["default"])

        # Always write to L1
        self.l1.set(key, value, ttl=l1_ttl)

        # Only write to L2 if TTL > 0
        if l2_ttl > 0:
            self.l2.set(key, value, ttl=l2_ttl)

    def delete(self, key: str) -> None:
        """Delete from both cache levels."""
        self.l1.delete(key)
        self.l2.delete(key)

    def clear(self) -> None:
        """Clear both cache levels."""
        self.l1.clear()
        self.l2.clear()

    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        # For L1, we'd need to track keys - simplify by just clearing
        # In production, use a more sophisticated approach
        count = 0
        # L2 can be queried
        with sqlite3.connect(self.l2.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"%{symbol}%",))
            count = cursor.rowcount
            conn.commit()
        return count

    def stats(self) -> dict:
        """Get combined cache statistics."""
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
        }

    def cleanup(self) -> dict:
        """Cleanup expired entries from both levels."""
        # L1 auto-cleans on access, but we can force it
        l1_cleaned = 0
        for key in list(self.l1._cache.keys()):
            if self.l1.get(key) is None:  # This triggers cleanup
                l1_cleaned += 1

        l2_cleaned = self.l2.cleanup_expired()

        return {
            "l1_cleaned": l1_cleaned,
            "l2_cleaned": l2_cleaned,
        }


# Helper functions for cache key generation
def make_ohlcv_key(symbol: str, timeframe: str, limit: int = 0) -> str:
    """Generate cache key for OHLCV data."""
    return f"ohlcv:{symbol}:{timeframe}:{limit}"


def make_quote_key(symbol: str) -> str:
    """Generate cache key for quote data."""
    return f"quote:{symbol}"


def make_symbol_key(symbol: str) -> str:
    """Generate cache key for symbol info."""
    return f"symbol:{symbol}"
