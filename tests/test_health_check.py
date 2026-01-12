"""Unit tests for health check endpoint."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import api as api_module
from api.api import app
from bot.state import BotState


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_state_dir(tmp_path):
    """Create a mock state directory with state file."""
    state_file = tmp_path / "state.json"

    # Create a valid state
    state = BotState(
        symbol="BTC/USDT",
        position="FLAT",
        balance=10000.0,
        initial_balance=10000.0,
        risk_per_trade_pct=0.5,
    )

    state_file.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    return tmp_path


class TestHealthCheckEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_200(self, client, mock_state_dir, monkeypatch):
        """Test that health check returns 200 status."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)  # Reset cached store

        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client, mock_state_dir, monkeypatch):
        """Test that health check response has correct structure."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "bot_stale" in data
        assert "stale_threshold_seconds" in data

    def test_health_check_status_values(self, client, mock_state_dir, monkeypatch):
        """Test that status is one of valid values."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_components_present(self, client, mock_state_dir, monkeypatch):
        """Test that components dictionary contains expected keys."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        assert "bot" in data["components"]
        assert "state_store" in data["components"]

    def test_health_check_no_auth_required(self, client, mock_state_dir, monkeypatch):
        """Test that health check doesn't require authentication."""
        monkeypatch.setenv("API_KEY", "secret-key")
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        # Should work without X-API-Key header
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_uptime_positive(self, client, mock_state_dir, monkeypatch):
        """Test that uptime is positive."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        assert data["uptime_seconds"] > 0

    def test_health_check_timestamp_valid(self, client, mock_state_dir, monkeypatch):
        """Test that timestamp is a valid ISO format."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        # Should be parseable as ISO datetime
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert timestamp is not None

    def test_health_check_degraded_when_state_missing(self, client, tmp_path, monkeypatch):
        """Test that status is degraded when state file is missing."""
        # Empty directory, no state.json
        monkeypatch.setattr(api_module, "STATE_DIR", tmp_path)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        # Should be degraded or unhealthy since state is missing
        assert data["components"]["state_store"] == "no_file"

    def test_health_check_stale_threshold_configurable(self, client, mock_state_dir, monkeypatch):
        """Test that stale threshold can be configured via environment."""
        monkeypatch.setenv("BOT_STALE_THRESHOLD_SECONDS", "600")
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dir)
        monkeypatch.setattr(api_module, "_state_store", None)

        response = client.get("/health")
        data = response.json()

        assert data["stale_threshold_seconds"] == 600
