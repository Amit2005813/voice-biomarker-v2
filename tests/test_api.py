"""Tests for API module: health, upload, analyze, authentication, rate limiting, caching."""

from __future__ import annotations

import asyncio
import io
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from api.cache import InMemoryCache, CacheManager
from api.auth import (
    RateLimiter,
    generate_token,
    verify_token,
    _hash_password,
    _verify_password,
    register_api_key,
    verify_api_key,
)


def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ======================================================================
# Cache
# ======================================================================

class TestInMemoryCache:
    """Test in-memory LRU cache."""

    def test_set_and_get(self) -> None:
        cache = InMemoryCache(default_ttl=60)
        _run_async(cache.set("key1", {"data": 42}))
        result = _run_async(cache.get("key1"))
        assert result == {"data": 42}

    def test_get_miss(self) -> None:
        cache = InMemoryCache()
        result = _run_async(cache.get("nonexistent"))
        assert result is None

    def test_ttl_expiry(self) -> None:
        cache = InMemoryCache(default_ttl=1)
        _run_async(cache.set("key1", "value1"))
        import time
        time.sleep(1.1)
        result = _run_async(cache.get("key1"))
        assert result is None

    def test_delete(self) -> None:
        cache = InMemoryCache()
        _run_async(cache.set("key1", "value1"))
        deleted = _run_async(cache.delete("key1"))
        assert deleted is True
        assert _run_async(cache.get("key1")) is None

    def test_delete_nonexistent(self) -> None:
        cache = InMemoryCache()
        deleted = _run_async(cache.delete("nonexistent"))
        assert deleted is False

    def test_clear(self) -> None:
        cache = InMemoryCache()
        _run_async(cache.set("a", 1))
        _run_async(cache.set("b", 2))
        _run_async(cache.clear())
        assert _run_async(cache.get("a")) is None
        assert _run_async(cache.get("b")) is None

    def test_lru_eviction(self) -> None:
        cache = InMemoryCache(max_size=3, default_ttl=60)
        _run_async(cache.set("a", 1))
        _run_async(cache.set("b", 2))
        _run_async(cache.set("c", 3))
        _run_async(cache.set("d", 4))
        assert _run_async(cache.get("a")) is None
        assert _run_async(cache.get("d")) == 4

    def test_stats(self) -> None:
        cache = InMemoryCache(default_ttl=60)
        _run_async(cache.set("k", "v"))
        _run_async(cache.get("k"))
        _run_async(cache.get("miss"))
        stats = cache.stats
        assert stats["backend"] == "memory"
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_keys"] == 1


class TestCacheManager:
    """Test unified cache manager."""

    def test_fallback_to_memory(self) -> None:
        manager = CacheManager(redis_url="redis://invalid:9999/0")
        _run_async(manager.initialize())
        assert manager._use_redis is False
        _run_async(manager.set("key", "value"))
        result = _run_async(manager.get("key"))
        assert result == "value"
        _run_async(manager.shutdown())


# ======================================================================
# Authentication
# ======================================================================

class TestPasswordHashing:
    """Test password hashing utilities."""

    def test_hash_consistent(self) -> None:
        h1 = _hash_password("test123")
        h2 = _hash_password("test123")
        assert h1 == h2

    def test_verify_correct(self) -> None:
        h = _hash_password("test123")
        assert _verify_password("test123", h) is True

    def test_verify_incorrect(self) -> None:
        h = _hash_password("test123")
        assert _verify_password("wrong", h) is False


class TestJWT:
    """Test JWT token generation and verification."""

    def test_generate_and_verify(self) -> None:
        token = generate_token("user-1", "clinician")
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "user-1"
        assert payload["role"] == "clinician"

    def test_verify_invalid_token(self) -> None:
        payload = verify_token("invalid.token.here")
        assert payload is None

    def test_verify_tampered_token(self) -> None:
        token = generate_token("user-1", "clinician")
        parts = token.split(".")
        parts[2] = "TAMPERED"
        tampered = ".".join(parts)
        payload = verify_token(tampered)
        assert payload is None


class TestAPIKeys:
    """Test API key management."""

    def test_register_and_verify(self) -> None:
        register_api_key(
            "test-key-123",
            user_id="user-1",
            role="researcher",
            description="test key",
        )
        info = verify_api_key("test-key-123")
        assert info is not None
        assert info["user_id"] == "user-1"
        assert info["role"] == "researcher"

    def test_verify_nonexistent(self) -> None:
        info = verify_api_key("nonexistent-key")
        assert info is None


class TestRateLimiter:
    """Test rate limiting."""

    def test_allows_within_limit(self) -> None:
        limiter = RateLimiter()
        assert limiter.is_allowed("user1", max_requests=5, window_seconds=60) is True

    def test_blocks_over_limit(self) -> None:
        limiter = RateLimiter()
        for _ in range(5):
            limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        assert limiter.is_allowed("user1", max_requests=5, window_seconds=60) is False

    def test_separate_windows(self) -> None:
        limiter = RateLimiter()
        limiter.is_allowed("user1", max_requests=3, window_seconds=60)
        limiter.is_allowed("user1", max_requests=3, window_seconds=60)
        limiter.is_allowed("user1", max_requests=3, window_seconds=60)
        assert limiter.is_allowed("user1", max_requests=3, window_seconds=60) is False
        assert limiter.is_allowed("user2", max_requests=3, window_seconds=60) is True

    def test_remaining(self) -> None:
        limiter = RateLimiter()
        limiter.is_allowed("u", max_requests=10, window_seconds=60)
        limiter.is_allowed("u", max_requests=10, window_seconds=60)
        remaining = limiter.remaining("u", max_requests=10, window_seconds=60)
        assert remaining == 8


# ======================================================================
# API Endpoints (via TestClient)
# ======================================================================

class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check(self, api_client) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data

    def test_health_has_uptime(self, api_client) -> None:
        response = api_client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestUploadEndpoint:
    """Test /upload-audio endpoint."""

    def test_upload_valid_wav(self, api_client, sample_wav_path: Path) -> None:
        with open(sample_wav_path, "rb") as f:
            response = api_client.post(
                "/upload-audio",
                files={"file": ("test.wav", f, "audio/wav")},
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code in (200, 401)

    def test_upload_empty_file(self, api_client) -> None:
        response = api_client.post(
            "/upload-audio",
            files={"file": ("empty.wav", b"", "audio/wav")},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code in (400, 401)


class TestAnalyzeEndpoint:
    """Test /analyze endpoint."""

    def test_analyze_no_auth(self, api_client, sample_wav_path: Path) -> None:
        with open(sample_wav_path, "rb") as f:
            response = api_client.post(
                "/analyze",
                files={"file": ("test.wav", f, "audio/wav")},
            )
        assert response.status_code == 401

    def test_analyze_with_api_key(self, api_client, sample_wav_path: Path) -> None:
        register_api_key(
            "test-api-key",
            user_id="test-user",
            role="clinician",
        )
        with open(sample_wav_path, "rb") as f:
            response = api_client.post(
                "/analyze",
                files={"file": ("test.wav", f, "audio/wav")},
                headers={"X-API-Key": "test-api-key"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "status" in data


class TestAuthEndpoints:
    """Test /auth/* endpoints."""

    def test_docs_endpoint(self, api_client) -> None:
        response = api_client.get("/docs")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "endpoints" in data
