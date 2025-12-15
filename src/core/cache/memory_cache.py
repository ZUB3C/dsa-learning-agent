"""
In-memory caching (fallback for Redis).
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class MemoryCache:
    """
    Simple in-memory LRU cache.

    Used as fallback when Redis is unavailable.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        logger.info(f"💾 Memory cache initialized (max_size={max_size})")

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key."""
        key_data = json.dumps(
            {"args": args, "kwargs": sorted(kwargs.items())}, sort_keys=True, default=str
        )

        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key not in self.cache:
            logger.debug(f"❌ Cache miss: {key}")
            return None

        # Check expiration
        entry = self.cache[key]
        if time.time() > entry["expires_at"]:
            logger.debug(f"⏰ Cache expired: {key}")
            del self.cache[key]
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)

        logger.debug(f"✅ Cache hit: {key}")
        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"🗑️ Evicted oldest: {oldest_key}")

        # Store
        self.cache[key] = {"value": value, "expires_at": time.time() + ttl}

        # Move to end
        self.cache.move_to_end(key)

        logger.debug(f"✅ Cache set: {key} (TTL: {ttl}s)")

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"🗑️ Cache deleted: {key}")

    def clear(self) -> None:
        """Clear all cache."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"🗑️ Cleared {count} cache entries")

    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [key for key, entry in self.cache.items() if now > entry["expires_at"]]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired entries")

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
        }


# Global instance
_memory_cache = MemoryCache()


def get_memory_cache() -> MemoryCache:
    """Get global memory cache instance."""
    return _memory_cache
