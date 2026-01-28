"""Unit tests for API security module."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from api.security import (
    RateLimiter,
    get_api_key,
    verify_api_key,
)


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_returns_env_value(self, monkeypatch):
        """Test that API key is read from environment."""
        monkeypatch.setenv("API_KEY", "test-api-key-123")
        assert get_api_key() == "test-api-key-123"

    def test_returns_none_when_not_set(self, monkeypatch):
        """Test that None is returned when API_KEY is not set."""
        monkeypatch.delenv("API_KEY", raising=False)
        assert get_api_key() is None


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_no_auth_when_key_not_configured(self, monkeypatch):
        """Test that requests are blocked when API_KEY is not configured (security measure)."""
        from fastapi import HTTPException

        monkeypatch.delenv("API_KEY", raising=False)

        request = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request, api_key=None)

        assert exc_info.value.status_code == 503
        assert "not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_key_raises_401(self, monkeypatch):
        """Test that missing API key raises 401 when auth is required."""
        from fastapi import HTTPException

        monkeypatch.setenv("API_KEY", "secret-key")

        request = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request, api_key=None)

        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_invalid_key_raises_403(self, monkeypatch):
        """Test that wrong API key raises 403."""
        from fastapi import HTTPException

        # Use a strong valid format key for the env (must have upper, lower, digit, special)
        monkeypatch.setenv("API_KEY", "Xk9#mNpQ2wRsT5uV8yAz")

        request = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            # Use a different but valid format key that doesn't match
            await verify_api_key(request, api_key="Yk9#mNpQ2wRsT5uV8yAz")

        assert exc_info.value.status_code == 403
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_key_returns_key(self, monkeypatch):
        """Test that valid API key is returned."""
        # Strong key: 20 chars, has upper, lower, digit, special, no weak patterns
        valid_key = "Xk9#mNpQ2wRsT5uV8yAz"
        monkeypatch.setenv("API_KEY", valid_key)

        request = MagicMock()
        result = await verify_api_key(request, api_key=valid_key)

        assert result == valid_key


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_requests_under_limit(self):
        """Test that requests under limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        for _ in range(5):
            allowed, error = limiter.is_allowed("client-1")
            assert allowed is True
            assert error is None

    def test_blocks_requests_over_minute_limit(self):
        """Test that requests over minute limit are blocked."""
        limiter = RateLimiter(requests_per_minute=3, requests_per_hour=100)

        # Use up the limit
        for _ in range(3):
            allowed, _ = limiter.is_allowed("client-1")
            assert allowed is True

        # Next request should be blocked
        allowed, error = limiter.is_allowed("client-1")
        assert allowed is False
        assert "per minute" in error

    def test_blocks_requests_over_hour_limit(self):
        """Test that requests over hour limit are blocked."""
        limiter = RateLimiter(requests_per_minute=100, requests_per_hour=5)

        # Use up the hourly limit
        for _ in range(5):
            allowed, _ = limiter.is_allowed("client-1")
            assert allowed is True

        # Next request should be blocked
        allowed, error = limiter.is_allowed("client-1")
        assert allowed is False
        assert "per hour" in error

    def test_separate_limits_per_client(self):
        """Test that each client has separate limits."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=100)

        # Client 1 uses their limit
        limiter.is_allowed("client-1")
        limiter.is_allowed("client-1")
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is False  # Client 1 blocked

        # Client 2 should still be allowed
        allowed, _ = limiter.is_allowed("client-2")
        assert allowed is True

    def test_window_cleanup(self):
        """Test that old requests are cleaned from window."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=100)

        # Use up the limit
        limiter.is_allowed("client-1")
        limiter.is_allowed("client-1")
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is False

        # Simulate time passing (mock the internal state)
        old_time = time.time() - 61  # 61 seconds ago
        limiter._minute_windows["client-1"] = [old_time, old_time]

        # Now requests should be allowed again
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is True

    def test_hour_window_cleanup(self):
        """Test that old requests are cleaned from hour window."""
        limiter = RateLimiter(requests_per_minute=100, requests_per_hour=2)

        # Use up the hourly limit
        limiter.is_allowed("client-1")
        limiter.is_allowed("client-1")
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is False

        # Simulate time passing
        old_time = time.time() - 3601  # 1 hour + 1 second ago
        limiter._hour_windows["client-1"] = [old_time, old_time]
        limiter._minute_windows["client-1"] = []  # Clear minute window too

        # Now requests should be allowed again
        allowed, _ = limiter.is_allowed("client-1")
        assert allowed is True
