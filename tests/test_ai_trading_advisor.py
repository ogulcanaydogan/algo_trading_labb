"""Tests for bot.ai_trading_advisor module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from bot.ai_trading_advisor import (
    AIAdvice,
    AITradingAdvisor,
    get_advisor,
    get_ai_advice,
)


class TestAIAdvice:
    """Tests for AIAdvice dataclass."""

    def create_advice(self, **kwargs) -> AIAdvice:
        """Helper to create AIAdvice with defaults."""
        defaults = {
            "action": "HOLD",
            "confidence": 0.7,
            "reasoning": "Test reasoning",
            "risk_level": "medium",
            "position_size_pct": 10.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "market_regime": "bull",
            "warnings": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "test-model",
            "latency_ms": 100,
        }
        defaults.update(kwargs)
        return AIAdvice(**defaults)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        advice = self.create_advice(action="BUY", confidence=0.8)
        result = advice.to_dict()

        assert result["action"] == "BUY"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "Test reasoning"
        assert "timestamp" in result

    def test_should_trade_true_for_buy_high_confidence(self) -> None:
        """Test should_trade is True for BUY with high confidence."""
        advice = self.create_advice(action="BUY", confidence=0.7)
        assert advice.should_trade is True

    def test_should_trade_true_for_sell_high_confidence(self) -> None:
        """Test should_trade is True for SELL with high confidence."""
        advice = self.create_advice(action="SELL", confidence=0.6)
        assert advice.should_trade is True

    def test_should_trade_false_for_hold(self) -> None:
        """Test should_trade is False for HOLD action."""
        advice = self.create_advice(action="HOLD", confidence=0.9)
        assert advice.should_trade is False

    def test_should_trade_false_for_low_confidence(self) -> None:
        """Test should_trade is False for low confidence."""
        advice = self.create_advice(action="BUY", confidence=0.5)
        assert advice.should_trade is False

    def test_is_warning_true_for_high_risk(self) -> None:
        """Test is_warning is True for high risk level."""
        advice = self.create_advice(risk_level="high")
        assert advice.is_warning is True

    def test_is_warning_true_for_extreme_risk(self) -> None:
        """Test is_warning is True for extreme risk level."""
        advice = self.create_advice(risk_level="extreme")
        assert advice.is_warning is True

    def test_is_warning_true_for_warnings_list(self) -> None:
        """Test is_warning is True when warnings exist."""
        advice = self.create_advice(risk_level="low", warnings=["Warning 1"])
        assert advice.is_warning is True

    def test_is_warning_false_for_low_risk_no_warnings(self) -> None:
        """Test is_warning is False for low risk with no warnings."""
        advice = self.create_advice(risk_level="low", warnings=[])
        assert advice.is_warning is False


class TestAITradingAdvisor:
    """Tests for AITradingAdvisor class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        advisor = AITradingAdvisor()

        assert advisor.ollama_host == "http://localhost:11434"
        assert advisor.model == "qwen2.5:7b"
        assert advisor.timeout == 30
        assert advisor.enabled is True

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        advisor = AITradingAdvisor(
            ollama_host="http://custom:8080",
            model="llama2",
            timeout=60,
            enabled=False,
        )

        assert advisor.ollama_host == "http://custom:8080"
        assert advisor.model == "llama2"
        assert advisor.timeout == 60
        assert advisor.enabled is False

    @pytest.mark.asyncio
    async def test_check_availability_success(self) -> None:
        """Test check_availability returns True when Ollama responds."""
        advisor = AITradingAdvisor()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(advisor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await advisor.check_availability()

        assert result is True
        assert advisor._available is True

    @pytest.mark.asyncio
    async def test_check_availability_failure(self) -> None:
        """Test check_availability returns False when Ollama is down."""
        advisor = AITradingAdvisor()

        with patch.object(advisor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_get_client.return_value = mock_client

            result = await advisor.check_availability()

        assert result is False
        assert advisor._available is False

    @pytest.mark.asyncio
    async def test_check_availability_cached(self) -> None:
        """Test check_availability uses cached result."""
        advisor = AITradingAdvisor()
        advisor._available = True

        result = await advisor.check_availability()

        assert result is True

    @pytest.mark.asyncio
    async def test_get_advice_when_disabled(self) -> None:
        """Test get_advice returns default when disabled."""
        advisor = AITradingAdvisor(enabled=False)

        advice = await advisor.get_advice(
            symbol="BTC/USDT",
            current_price=50000,
            price_change_1h=1.0,
            price_change_24h=2.0,
            current_signal="LONG",
            regime="bull",
            confidence=0.8,
            portfolio_value=10000,
            position_value=1000,
            pnl_pct=5.0,
        )

        assert advice.action == "HOLD"
        assert advice.model == "default"
        assert "unavailable" in advice.reasoning.lower()

    @pytest.mark.asyncio
    async def test_get_advice_when_unavailable(self) -> None:
        """Test get_advice returns default when AI is unavailable."""
        advisor = AITradingAdvisor(enabled=True)

        with patch.object(advisor, "check_availability", return_value=False):
            advice = await advisor.get_advice(
                symbol="ETH/USDT",
                current_price=3000,
                price_change_1h=-0.5,
                price_change_24h=-1.0,
                current_signal="SHORT",
                regime="bear",
                confidence=0.6,
                portfolio_value=10000,
                position_value=500,
                pnl_pct=-2.0,
            )

        assert advice.action == "HOLD"
        assert advice.model == "default"

    @pytest.mark.asyncio
    async def test_get_advice_success(self) -> None:
        """Test successful AI advice retrieval."""
        advisor = AITradingAdvisor(enabled=True)
        advisor._available = True

        ai_response = {
            "action": "BUY",
            "confidence": 0.85,
            "reasoning": "Bullish momentum",
            "risk_level": "medium",
            "position_size_pct": 15,
            "stop_loss_pct": 2.5,
            "take_profit_pct": 5.0,
            "market_regime": "bull",
            "warnings": [],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "message": {"content": json.dumps(ai_response)}
        })

        with patch.object(advisor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            advice = await advisor.get_advice(
                symbol="BTC/USDT",
                current_price=50000,
                price_change_1h=2.0,
                price_change_24h=5.0,
                current_signal="LONG",
                regime="bull",
                confidence=0.9,
                portfolio_value=10000,
                position_value=1500,
                pnl_pct=3.0,
            )

        assert advice.action == "BUY"
        assert advice.confidence == 0.85
        assert advice.reasoning == "Bullish momentum"

    @pytest.mark.asyncio
    async def test_get_advice_api_error(self) -> None:
        """Test get_advice handles API error gracefully."""
        advisor = AITradingAdvisor(enabled=True)
        advisor._available = True

        with patch.object(advisor, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            advice = await advisor.get_advice(
                symbol="BTC/USDT",
                current_price=50000,
                price_change_1h=0.5,
                price_change_24h=1.0,
                current_signal="FLAT",
                regime="sideways",
                confidence=0.5,
                portfolio_value=10000,
                position_value=0,
                pnl_pct=0,
            )

        assert advice.action == "HOLD"
        assert advice.model == "default"

    def test_get_last_advice_none(self) -> None:
        """Test get_last_advice returns None for unknown symbol."""
        advisor = AITradingAdvisor()

        result = advisor.get_last_advice("UNKNOWN")

        assert result is None

    def test_get_all_advice_empty(self) -> None:
        """Test get_all_advice returns empty dict initially."""
        advisor = AITradingAdvisor()

        result = advisor.get_all_advice()

        assert result == {}

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test close method closes the HTTP client."""
        advisor = AITradingAdvisor()
        mock_client = AsyncMock()
        advisor._client = mock_client

        await advisor.close()

        mock_client.aclose.assert_called_once()
        assert advisor._client is None

    def test_default_advice_signal_mapping(self) -> None:
        """Test default advice maps signals correctly."""
        advisor = AITradingAdvisor()

        advice_long = advisor._default_advice("BTC/USDT", "LONG", "bull")
        advice_short = advisor._default_advice("ETH/USDT", "SHORT", "bear")
        advice_flat = advisor._default_advice("XRP/USDT", "FLAT", "sideways")

        assert advice_long.action == "HOLD"
        assert advice_short.action == "HOLD"
        assert advice_flat.action == "HOLD"

    def test_system_prompt(self) -> None:
        """Test system prompt content."""
        advisor = AITradingAdvisor()

        prompt = advisor._get_system_prompt()

        assert "trading" in prompt.lower()
        assert "json" in prompt.lower()

    def test_build_prompt(self) -> None:
        """Test prompt building."""
        advisor = AITradingAdvisor()

        prompt = advisor._build_prompt(
            symbol="BTC/USDT",
            current_price=50000,
            price_change_1h=1.5,
            price_change_24h=3.0,
            current_signal="LONG",
            regime="bull",
            confidence=0.8,
            portfolio_value=10000,
            position_value=2000,
            pnl_pct=5.0,
            asset_type="crypto",
        )

        assert "BTC/USDT" in prompt
        assert "50,000" in prompt
        assert "LONG" in prompt
        assert "bull" in prompt


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_advisor_singleton(self) -> None:
        """Test get_advisor returns singleton instance."""
        # Reset global state
        import bot.ai_trading_advisor as module
        module._advisor = None

        advisor1 = get_advisor()
        advisor2 = get_advisor()

        assert advisor1 is advisor2

    @pytest.mark.asyncio
    async def test_get_ai_advice_convenience(self) -> None:
        """Test get_ai_advice convenience function."""
        import bot.ai_trading_advisor as module
        module._advisor = None

        with patch.object(AITradingAdvisor, "get_advice") as mock_get_advice:
            mock_advice = AIAdvice(
                action="HOLD",
                confidence=0.5,
                reasoning="Test",
                risk_level="low",
                position_size_pct=5,
                stop_loss_pct=None,
                take_profit_pct=None,
                market_regime="unknown",
                warnings=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                model="test",
                latency_ms=0,
            )
            mock_get_advice.return_value = mock_advice

            advice = await get_ai_advice(
                symbol="BTC/USDT",
                current_price=50000,
            )

            mock_get_advice.assert_called_once()
            assert advice.action == "HOLD"
