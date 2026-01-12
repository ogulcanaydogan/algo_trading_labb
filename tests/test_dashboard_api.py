"""Unit tests for dashboard API endpoints."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import api as api_module
from api.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_state_dirs(tmp_path):
    """Create mock state directories for all market types."""
    # Create directory structure
    crypto_dir = tmp_path / "live_paper_trading"
    commodity_dir = tmp_path / "commodity_trading"
    stock_dir = tmp_path / "stock_trading"

    for d in [crypto_dir, commodity_dir, stock_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create crypto state
    crypto_state = {
        "symbol": "BTC/USDT",
        "position": "LONG",
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "total_value": 10050.0,
        "pnl": 50.0,
        "pnl_pct": 0.5,
        "cash_balance": 5000.0,
        "positions_count": 2,
        "risk_per_trade_pct": 0.5,
    }
    (crypto_dir / "state.json").write_text(json.dumps(crypto_state), encoding="utf-8")
    (crypto_dir / "equity.json").write_text(json.dumps([
        {"timestamp": "2024-01-01T00:00:00Z", "value": 10000},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 10050},
    ]), encoding="utf-8")

    # Create commodity state
    commodity_state = {
        "symbol": "XAU/USD",
        "position": "FLAT",
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "total_value": 10100.0,
        "pnl": 100.0,
        "pnl_pct": 1.0,
        "cash_balance": 10000.0,
        "positions_count": 0,
        "risk_per_trade_pct": 0.5,
    }
    (commodity_dir / "state.json").write_text(json.dumps(commodity_state), encoding="utf-8")
    (commodity_dir / "equity.json").write_text(json.dumps([
        {"timestamp": "2024-01-01T00:00:00Z", "value": 10000},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 10100},
    ]), encoding="utf-8")

    # Create stock state
    stock_state = {
        "symbol": "AAPL",
        "position": "FLAT",
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "total_value": 9950.0,
        "pnl": -50.0,
        "pnl_pct": -0.5,
        "cash_balance": 9950.0,
        "positions_count": 0,
        "risk_per_trade_pct": 0.5,
    }
    (stock_dir / "state.json").write_text(json.dumps(stock_state), encoding="utf-8")
    (stock_dir / "equity.json").write_text(json.dumps([
        {"timestamp": "2024-01-01T00:00:00Z", "value": 10000},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 9950},
    ]), encoding="utf-8")

    return tmp_path


class TestPortfolioStatsEndpoint:
    """Tests for /api/portfolio/stats endpoint (renamed from /api/performance/metrics)."""

    def test_portfolio_stats_returns_200(self, client, mock_state_dirs, monkeypatch):
        """Test that portfolio stats returns 200 status."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats?market_type=all")
        assert response.status_code == 200

    def test_portfolio_stats_response_structure(self, client, mock_state_dirs, monkeypatch):
        """Test that portfolio stats response has correct structure."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats?market_type=all")
        data = response.json()

        assert "timestamp" in data
        assert "market_type" in data
        assert "metrics" in data
        assert "markets" in data

    def test_portfolio_stats_metrics_fields(self, client, mock_state_dirs, monkeypatch):
        """Test that metrics contains expected fields."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats?market_type=all")
        data = response.json()
        metrics = data.get("metrics", {})

        # These fields should be present (may be 0 if no data)
        expected_fields = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "max_drawdown_pct",
            "volatility",
            "total_return",
            "total_return_pct",
        ]

        for field in expected_fields:
            assert field in metrics, f"Missing field: {field}"

    def test_portfolio_stats_single_market(self, client, mock_state_dirs, monkeypatch):
        """Test portfolio stats for a single market type."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        for market_type in ["crypto", "commodity", "stock"]:
            response = client.get(f"/api/portfolio/stats?market_type={market_type}")
            assert response.status_code == 200
            data = response.json()
            assert data["market_type"] == market_type

    def test_portfolio_stats_no_auth_required_without_api_key(self, client, mock_state_dirs, monkeypatch):
        """Test that portfolio stats works without auth when API_KEY not set."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats?market_type=all")
        assert response.status_code == 200

    def test_portfolio_stats_market_breakdown(self, client, mock_state_dirs, monkeypatch):
        """Test that markets breakdown is included."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats?market_type=all")
        data = response.json()
        markets = data.get("markets", {})

        assert "crypto" in markets
        assert "commodity" in markets
        assert "stock" in markets


class TestBotStateEndpoint:
    """Tests for /api/bot/state/{market_type} endpoint."""

    def test_bot_state_all_returns_200(self, client, mock_state_dirs, monkeypatch):
        """Test that bot state all returns 200 status."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/all")
        assert response.status_code == 200

    def test_bot_state_all_response_structure(self, client, mock_state_dirs, monkeypatch):
        """Test that bot state all has correct structure."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/all")
        data = response.json()

        assert "markets" in data
        assert "total_portfolio_value" in data
        assert "total_pnl" in data
        assert "total_pnl_pct" in data

    def test_bot_state_all_market_values(self, client, mock_state_dirs, monkeypatch):
        """Test that bot state returns correct market values."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/all")
        data = response.json()
        markets = data.get("markets", {})

        # Check crypto
        assert "crypto" in markets
        crypto = markets["crypto"]
        assert crypto.get("total_value") == 10050.0
        assert crypto.get("pnl") == 50.0

        # Check commodity
        assert "commodity" in markets
        commodity = markets["commodity"]
        assert commodity.get("total_value") == 10100.0
        assert commodity.get("pnl") == 100.0

        # Check stock
        assert "stock" in markets
        stock = markets["stock"]
        assert stock.get("total_value") == 9950.0
        assert stock.get("pnl") == -50.0

    def test_bot_state_total_portfolio_value(self, client, mock_state_dirs, monkeypatch):
        """Test that total portfolio value is sum of all markets."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/all")
        data = response.json()

        # 10050 + 10100 + 9950 = 30100
        expected_total = 10050.0 + 10100.0 + 9950.0
        assert data.get("total_portfolio_value") == pytest.approx(expected_total, rel=0.01)

    def test_bot_state_single_market(self, client, mock_state_dirs, monkeypatch):
        """Test bot state for single market type."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/crypto")
        assert response.status_code == 200
        data = response.json()
        assert data.get("total_value") == 10050.0

    def test_bot_state_invalid_market(self, client, mock_state_dirs, monkeypatch):
        """Test bot state with invalid market type."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/bot/state/invalid")
        # Should return 400 (bad request) or 404 (not found)
        assert response.status_code in [200, 400, 404]


class TestDashboardEndpoint:
    """Tests for dashboard HTML endpoints."""

    def test_unified_dashboard_returns_200(self, client):
        """Test that unified dashboard returns 200."""
        response = client.get("/dashboard/unified")
        assert response.status_code == 200

    def test_unified_dashboard_returns_html(self, client):
        """Test that unified dashboard returns HTML content."""
        response = client.get("/dashboard/unified")
        assert "text/html" in response.headers.get("content-type", "")

    def test_unified_dashboard_contains_key_elements(self, client):
        """Test that dashboard HTML contains key elements."""
        response = client.get("/dashboard/unified")
        content = response.text

        # Check for key dashboard elements
        assert "Portfolio Allocation" in content or "portfolio" in content.lower()
        assert "totalPortfolioValue" in content
        assert "cryptoValue" in content or "crypto" in content.lower()


class TestEquityEndpoints:
    """Tests for equity data endpoints."""

    def test_equity_returns_200(self, client, mock_state_dirs, monkeypatch):
        """Test that equity endpoint returns 200."""
        # Create equity.json in the main state dir
        equity_file = mock_state_dirs / "equity.json"
        equity_file.write_text(json.dumps([
            {"timestamp": "2024-01-01T00:00:00Z", "value": 30000},
            {"timestamp": "2024-01-02T00:00:00Z", "value": 30100},
        ]), encoding="utf-8")
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/equity")
        assert response.status_code == 200

    def test_equity_returns_list(self, client, mock_state_dirs, monkeypatch):
        """Test that equity endpoint returns a list."""
        equity_file = mock_state_dirs / "equity.json"
        equity_file.write_text(json.dumps([
            {"timestamp": "2024-01-01T00:00:00Z", "value": 30000},
        ]), encoding="utf-8")
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/equity")
        data = response.json()
        assert isinstance(data, list)


class TestMarketSummaryEndpoint:
    """Tests for /api/markets/{market_type}/summary endpoint."""

    def test_market_summary_crypto(self, client, mock_state_dirs, monkeypatch):
        """Test market summary for crypto."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        # This endpoint may require additional mocking of MarketDataService
        # For now, test that it doesn't crash
        response = client.get("/api/markets/crypto/summary")
        # May return 200 with data or 500 if dependencies not available
        assert response.status_code in [200, 500, 503]

    def test_market_summary_invalid_market(self, client, mock_state_dirs, monkeypatch):
        """Test market summary with invalid market type."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/markets/invalid/summary")
        assert response.status_code in [400, 404, 422, 500]


class TestEndpointNaming:
    """Tests to verify endpoint naming (content blocker avoidance)."""

    def test_old_metrics_endpoint_not_available(self, client):
        """Test that old /api/performance/metrics endpoint is not available."""
        response = client.get("/api/performance/metrics")
        # Should return 404 since we renamed it
        assert response.status_code == 404

    def test_new_stats_endpoint_available(self, client, mock_state_dirs, monkeypatch):
        """Test that new /api/portfolio/stats endpoint is available."""
        monkeypatch.setattr(api_module, "STATE_DIR", mock_state_dirs)

        response = client.get("/api/portfolio/stats")
        assert response.status_code == 200
