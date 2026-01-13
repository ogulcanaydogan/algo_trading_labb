"""Unit tests for configuration validation schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bot.config_schemas import (
    BotConfigSchema,
    MacroEventSchema,
    MacroEventsFileSchema,
    PortfolioAssetSchema,
    PortfolioConfigSchema,
    StrategyConfigSchema,
    validate_bot_config,
    validate_portfolio_config,
    validate_strategy_config,
)


class TestStrategyConfigSchema:
    """Tests for StrategyConfigSchema validation."""

    def test_valid_default_config(self):
        """Test that default values are valid."""
        config = StrategyConfigSchema()
        assert config.symbol == "BTC/USDT"
        assert config.ema_fast == 12
        assert config.ema_slow == 26

    def test_valid_custom_config(self):
        """Test valid custom configuration."""
        config = StrategyConfigSchema(
            symbol="ETH/USDT",
            timeframe="5m",
            ema_fast=5,
            ema_slow=20,
            rsi_period=10,
            rsi_overbought=75.0,
            rsi_oversold=25.0,
        )
        assert config.symbol == "ETH/USDT"
        assert config.ema_fast == 5

    def test_invalid_ema_relationship(self):
        """Test that ema_slow must be greater than ema_fast."""
        with pytest.raises(ValidationError) as exc_info:
            StrategyConfigSchema(ema_fast=26, ema_slow=12)

        assert "ema_slow" in str(exc_info.value)

    def test_invalid_ema_equal(self):
        """Test that ema_slow cannot equal ema_fast."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(ema_fast=20, ema_slow=20)

    def test_invalid_rsi_thresholds(self):
        """Test that rsi_oversold must be less than rsi_overbought."""
        with pytest.raises(ValidationError) as exc_info:
            StrategyConfigSchema(rsi_oversold=70.0, rsi_overbought=30.0)

        assert "rsi_oversold" in str(exc_info.value)

    def test_invalid_ema_fast_too_low(self):
        """Test ema_fast minimum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(ema_fast=1)

    def test_invalid_ema_fast_too_high(self):
        """Test ema_fast maximum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(ema_fast=201, ema_slow=300)

    def test_invalid_rsi_overbought_too_low(self):
        """Test rsi_overbought minimum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(rsi_overbought=49.0)

    def test_invalid_rsi_overbought_too_high(self):
        """Test rsi_overbought maximum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(rsi_overbought=101.0)

    def test_invalid_risk_per_trade_too_low(self):
        """Test risk_per_trade_pct minimum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(risk_per_trade_pct=0.001)

    def test_invalid_risk_per_trade_too_high(self):
        """Test risk_per_trade_pct maximum value."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(risk_per_trade_pct=11.0)

    def test_invalid_timeframe_format(self):
        """Test invalid timeframe format."""
        with pytest.raises(ValidationError):
            StrategyConfigSchema(timeframe="invalid")

    def test_valid_timeframe_formats(self):
        """Test various valid timeframe formats."""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"]
        for tf in valid_timeframes:
            config = StrategyConfigSchema(timeframe=tf)
            assert config.timeframe == tf


class TestBotConfigSchema:
    """Tests for BotConfigSchema validation."""

    def test_valid_default_config(self):
        """Test that default values are valid."""
        config = BotConfigSchema()
        assert config.symbol == "BTC/USDT"
        assert config.paper_mode is True
        assert config.starting_balance == 10000.0

    def test_valid_asset_types(self):
        """Test all valid asset types."""
        valid_types = ["crypto", "equity", "stock", "etf", "index", "commodity", "forex"]
        for asset_type in valid_types:
            config = BotConfigSchema(asset_type=asset_type)
            assert config.asset_type == asset_type

    def test_invalid_asset_type(self):
        """Test invalid asset type."""
        with pytest.raises(ValidationError):
            BotConfigSchema(asset_type="invalid")

    def test_invalid_loop_interval_too_low(self):
        """Test loop_interval_seconds minimum value."""
        with pytest.raises(ValidationError):
            BotConfigSchema(loop_interval_seconds=0)

    def test_invalid_loop_interval_too_high(self):
        """Test loop_interval_seconds maximum value."""
        with pytest.raises(ValidationError):
            BotConfigSchema(loop_interval_seconds=86401)

    def test_invalid_lookback_too_low(self):
        """Test lookback minimum value."""
        with pytest.raises(ValidationError):
            BotConfigSchema(lookback=5)

    def test_invalid_starting_balance_too_low(self):
        """Test starting_balance minimum value."""
        with pytest.raises(ValidationError):
            BotConfigSchema(starting_balance=0.5)


class TestPortfolioConfigSchema:
    """Tests for PortfolioConfigSchema validation."""

    def test_valid_empty_portfolio(self):
        """Test valid empty portfolio configuration."""
        config = PortfolioConfigSchema()
        assert config.portfolio_capital == 10000.0
        assert config.assets == []

    def test_valid_portfolio_with_assets(self):
        """Test valid portfolio with assets."""
        config = PortfolioConfigSchema(
            portfolio_capital=50000.0,
            default_timeframe="1h",
            assets=[
                PortfolioAssetSchema(symbol="BTC/USDT", allocation_pct=50.0),
                PortfolioAssetSchema(symbol="ETH/USDT", allocation_pct=30.0),
                PortfolioAssetSchema(symbol="XRP/USDT", allocation_pct=20.0),
            ],
        )
        assert len(config.assets) == 3
        assert sum(a.allocation_pct or 0 for a in config.assets) == 100.0

    def test_duplicate_symbols_rejected(self):
        """Test that duplicate symbols are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioConfigSchema(
                assets=[
                    PortfolioAssetSchema(symbol="BTC/USDT"),
                    PortfolioAssetSchema(symbol="BTC/USDT"),  # duplicate
                ]
            )

        assert "Duplicate symbols" in str(exc_info.value)

    def test_case_insensitive_duplicate_detection(self):
        """Test that duplicate detection is case-insensitive."""
        with pytest.raises(ValidationError):
            PortfolioConfigSchema(
                assets=[
                    PortfolioAssetSchema(symbol="BTC/USDT"),
                    PortfolioAssetSchema(symbol="btc/usdt"),  # same symbol, different case
                ]
            )

    def test_invalid_allocation_negative(self):
        """Test that negative allocation is rejected."""
        with pytest.raises(ValidationError):
            PortfolioAssetSchema(symbol="BTC/USDT", allocation_pct=-10.0)

    def test_invalid_allocation_over_100(self):
        """Test that allocation over 100% is rejected."""
        with pytest.raises(ValidationError):
            PortfolioAssetSchema(symbol="BTC/USDT", allocation_pct=101.0)


class TestMacroEventSchema:
    """Tests for MacroEventSchema validation."""

    def test_valid_minimal_event(self):
        """Test valid event with minimal fields."""
        event = MacroEventSchema(title="Fed Rate Decision")
        assert event.title == "Fed Rate Decision"
        assert event.sentiment is None

    def test_valid_full_event(self):
        """Test valid event with all fields."""
        event = MacroEventSchema(
            title="Fed Rate Decision",
            category="monetary_policy",
            sentiment="bearish",
            impact="high",
            actor="Federal Reserve",
            interest_rate_expectation="hawkish",
            summary="Fed signals rate hikes ahead",
        )
        assert event.sentiment == "bearish"
        assert event.impact == "high"

    def test_invalid_sentiment(self):
        """Test invalid sentiment value."""
        with pytest.raises(ValidationError):
            MacroEventSchema(title="Test", sentiment="invalid")

    def test_valid_sentiments(self):
        """Test all valid sentiment values."""
        for sentiment in ["bullish", "bearish", "neutral"]:
            event = MacroEventSchema(title="Test", sentiment=sentiment)
            assert event.sentiment == sentiment

    def test_invalid_impact(self):
        """Test invalid impact value."""
        with pytest.raises(ValidationError):
            MacroEventSchema(title="Test", impact="invalid")

    def test_valid_impacts(self):
        """Test all valid impact values."""
        for impact in ["high", "medium", "low"]:
            event = MacroEventSchema(title="Test", impact=impact)
            assert event.impact == impact

    def test_empty_title_rejected(self):
        """Test that empty title is rejected."""
        with pytest.raises(ValidationError):
            MacroEventSchema(title="")


class TestMacroEventsFileSchema:
    """Tests for MacroEventsFileSchema validation."""

    def test_valid_empty_file(self):
        """Test valid empty events file."""
        file_schema = MacroEventsFileSchema()
        assert file_schema.events == []

    def test_valid_file_with_events(self):
        """Test valid file with multiple events."""
        file_schema = MacroEventsFileSchema(
            events=[
                MacroEventSchema(title="Event 1"),
                MacroEventSchema(title="Event 2", sentiment="bullish"),
            ]
        )
        assert len(file_schema.events) == 2


class TestValidationFunctions:
    """Tests for validation helper functions."""

    def test_validate_strategy_config_success(self):
        """Test successful strategy config validation."""
        config_dict = {
            "symbol": "BTC/USDT",
            "ema_fast": 10,
            "ema_slow": 20,
        }
        result = validate_strategy_config(config_dict)
        assert result.symbol == "BTC/USDT"

    def test_validate_strategy_config_failure(self):
        """Test failed strategy config validation."""
        config_dict = {
            "ema_fast": 30,
            "ema_slow": 20,  # invalid: slow < fast
        }
        with pytest.raises(ValidationError):
            validate_strategy_config(config_dict)

    def test_validate_bot_config_success(self):
        """Test successful bot config validation."""
        config_dict = {
            "symbol": "ETH/USDT",
            "paper_mode": False,
            "starting_balance": 50000.0,
        }
        result = validate_bot_config(config_dict)
        assert result.symbol == "ETH/USDT"
        assert result.paper_mode is False

    def test_validate_portfolio_config_success(self):
        """Test successful portfolio config validation."""
        config_dict = {
            "portfolio_capital": 100000.0,
            "assets": [
                {"symbol": "BTC/USDT", "allocation_pct": 60.0},
                {"symbol": "ETH/USDT", "allocation_pct": 40.0},
            ],
        }
        result = validate_portfolio_config(config_dict)
        assert result.portfolio_capital == 100000.0
        assert len(result.assets) == 2
