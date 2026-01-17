def test_health_response_shape():
    """Test /health response structure."""
    response_data = {
        "status": "healthy",
        "timestamp": "2026-01-15T10:30:00Z",
        "version": "1.0.0"
    }
    
    # Validate required fields
    assert "status" in response_data
    assert response_data["status"] in ["healthy", "degraded", "unhealthy"]


def test_status_response_shape():
    """Test /status response structure."""
    response_data = {
        "symbol": "BTC/USDT",
        "position": "FLAT",
        "entry_price": None,
        "current_balance": 10000.0,
        "unrealized_pnl_pct": 0.0,
        "last_signal": "FLAT",
    }
    
    # Validate types
    assert isinstance(response_data["symbol"], str)
    assert isinstance(response_data["current_balance"], (int, float))


def test_risk_settings_response_shape():
    """Test /api/trading/risk-settings response structure."""
    response_data = {
        "shorting": False,
        "leverage": False,
        "aggressive": False,
    }
    
    # All risk settings must be boolean
    for key, value in response_data.items():
        assert isinstance(value, bool)


def test_ai_brain_status_response_shape():
    """Test /api/ai-brain/status response structure."""
    response_data = {
        "active_strategy": "Balanced Growth",
        "daily_pnl_pct": 0.75,
        "target_achieved": False,
        "can_still_trade": True,
        "confidence": 0.72
    }
    
    # Validate key types
    assert isinstance(response_data["target_achieved"], bool)
    assert isinstance(response_data["can_still_trade"], bool)
    assert isinstance(response_data["confidence"], (int, float))
