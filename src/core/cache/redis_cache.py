"""
Redis caching layer.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import redis.asyncio as aioredis

from src.config import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based caching layer.
    """

    def __init__(self) -> None:
        """Initialize Redis cache."""
        self.settings = get_settings()
        self.enabled = self.settings.cache.cache_enabled
        self.redis: aioredis.Redis | None = None
        self.ttl = self.settings.cache.cache_ttl_seconds

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = aioredis.from_url(
                self.settings.cache.redis_url, encoding="utf-8", decode_responses=True
            )
            logger.info(f"✅ Redis cache connected: {self.settings.cache.redis_url}")
        except Exception as e:
            logger.exception(f"❌ Redis connection failed: {e}")
            self.enabled = False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create deterministic string from args
        key_data = json.dumps(
            {"args": args, "kwargs": sorted(kwargs.items())}, sort_keys=True, default=str
        )

        # Hash it
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]

        return f"{prefix}:{key_hash}"

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.enabled or not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"✅ Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"❌ Cache miss: {key}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Cache get failed: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (seconds), defaults to config value
        """
        if not self.enabled or not self.redis:
            return

        try:
            ttl = ttl or self.ttl
            value_json = json.dumps(value, default=str)
            await self.redis.setex(key, ttl, value_json)
            logger.debug(f"✅ Cache set: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"⚠️ Cache set failed: {e}")

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self.enabled or not self.redis:
            return

        try:
            await self.redis.delete(key)
            logger.debug(f"🗑️ Cache deleted: {key}")
        except Exception as e:
            logger.warning(f"⚠️ Cache delete failed: {e}")

    async def clear_pattern(self, pattern: str) -> None:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "rag:*")
        """
        if not self.enabled or not self.redis:
            return

        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"🗑️ Cleared {len(keys)} keys matching: {pattern}")
        except Exception as e:
            logger.warning(f"⚠️ Cache clear failed: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("🔌 Redis connection closed")


def cached(prefix: str, ttl: int | None = None):
    """
    Decorator for caching function results.

    Usage:
        @cached("rag_query", ttl=3600)
        async def my_expensive_function(query: str):
            return result
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = RedisCache()

            if not cache.enabled:
                # Cache disabled, execute directly
                return await func(*args, **kwargs)

            # Generate cache key
            key = cache._generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(key, result, ttl=ttl)

            return result

        return wrapper

    return decorator
