"""Caching layer with Redis fallback to in-memory dict."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis  # type: ignore[import-untyped]
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class InMemoryCache:
    """LRU in-memory cache with TTL support, used when Redis is unavailable."""

    def __init__(self, max_size: int = 10_000, default_ttl: int = 3600) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expires_at = entry
        if expires_at < time.time():
            self._store.pop(key, None)
            self._misses += 1
            return None
        self._hits += 1
        self._store.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, expires_at)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def clear(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "memory",
            "total_keys": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "memory_usage_bytes": sum(
                len(str(k)) + len(str(v)) for k, (v, _) in self._store.items()
            ),
        }


class RedisCache:
    """Async Redis-backed cache."""

    def __init__(self, url: str = "redis://localhost:6379/0", default_ttl: int = 3600) -> None:
        self._url = url
        self._default_ttl = default_ttl
        self._client: Optional[aioredis.Redis] = None  # type: ignore[type-arg]
        self._hits = 0
        self._misses = 0

    async def connect(self) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed")
        self._client = aioredis.from_url(self._url, decode_responses=True)
        await self._client.ping()
        logger.info("Connected to Redis at %s", self._url)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    async def get(self, key: str) -> Optional[Any]:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return _deserialize(raw)
        except Exception as exc:
            logger.warning("Redis GET error: %s", exc)
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._client is None:
            return
        try:
            await self._client.set(key, _serialize(value), ex=ttl or self._default_ttl)
        except Exception as exc:
            logger.warning("Redis SET error: %s", exc)

    async def delete(self, key: str) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.delete(key))
        except Exception as exc:
            logger.warning("Redis DEL error: %s", exc)
            return False

    async def exists(self, key: str) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False

    async def clear(self) -> None:
        if self._client:
            await self._client.flushdb()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "total_keys": 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "memory_usage_bytes": 0,
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize(value: Any) -> str:
    import json
    return json.dumps(value, default=str)


def _deserialize(raw: str) -> Any:
    import json
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


# ---------------------------------------------------------------------------
# Unified Cache Manager
# ---------------------------------------------------------------------------

class CacheManager:
    """Facade that tries Redis first, falls back to in-memory cache."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
    ) -> None:
        self._redis = RedisCache(url=redis_url, default_ttl=default_ttl)
        self._memory = InMemoryCache(default_ttl=default_ttl)
        self._use_redis = False

    async def initialize(self) -> None:
        try:
            await self._redis.connect()
            self._use_redis = True
            logger.info("Cache manager using Redis backend")
        except Exception as exc:
            self._use_redis = False
            logger.warning(
                "Redis unavailable (%s), falling back to in-memory cache", exc
            )

    async def shutdown(self) -> None:
        await self._redis.close()

    async def get(self, key: str) -> Optional[Any]:
        if self._use_redis:
            val = await self._redis.get(key)
            if val is not None:
                return val
            # Promote to memory if Redis missed but key might be hot
        return await self._memory.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._use_redis:
            await self._redis.set(key, value, ttl)
        await self._memory.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        deleted = False
        if self._use_redis:
            deleted = await self._redis.delete(key)
        deleted = await self._memory.delete(key) or deleted
        return deleted

    async def clear(self) -> None:
        if self._use_redis:
            await self._redis.clear()
        await self._memory.clear()

    @property
    def stats(self) -> dict[str, Any]:
        if self._use_redis:
            return self._redis.stats
        return self._memory.stats


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

cache_manager = CacheManager()
