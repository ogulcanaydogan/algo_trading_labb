"""
Tests for configuration module.
"""

import pytest
from bot.config import (
    GeneralConfig,
    TradingConfig,
    AssetClassConfig,
    DeepLearningConfig,
    NotificationsConfig,
    APIConfig,
    StrategyConfig,
    PortfolioConfig,
    OrchestratorConfig,
    AppConfig,
)


class TestGeneralConfig:
    """Test GeneralConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = GeneralConfig()
        assert config.log_level == "INFO"
        assert config.data_dir == "./data"

    def test_custom_log_level(self):
        """Test custom log level."""
        config = GeneralConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

    def test_invalid_log_level_normalized(self):
        """Test invalid log level is normalized."""
        config = GeneralConfig(log_level="invalid")
        assert config.log_level == "INFO"

    def test_lowercase_log_level_uppercased(self):
        """Test lowercase log level is uppercased."""
        config = GeneralConfig(log_level="warning")
        assert config.log_level == "WARNING"


class TestTradingConfig:
    """Test TradingConfig dataclass."""

    def test_default_values(self):
        """Test default trading configuration."""
        config = TradingConfig()
        assert config.initial_capital == 10000.0
        assert config.rebalance_threshold == 0.05
        assert config.loop_interval == 60
        assert config.symbol_fetch_delay == 5.0
        assert config.risk_per_trade_pct == 1.0
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.04

    def test_custom_values(self):
        """Test custom trading configuration."""
        config = TradingConfig(
            initial_capital=50000,
            rebalance_threshold=0.1,
            loop_interval=120,
        )
        assert config.initial_capital == 50000
        assert config.rebalance_threshold == 0.1
        assert config.loop_interval == 120

    def test_invalid_initial_capital(self):
        """Test invalid initial capital is reset."""
        config = TradingConfig(initial_capital=-100)
        assert config.initial_capital == 10000.0

    def test_invalid_rebalance_threshold(self):
        """Test invalid rebalance threshold is reset."""
        config = TradingConfig(rebalance_threshold=5.0)
        assert config.rebalance_threshold == 0.05

    def test_invalid_loop_interval(self):
        """Test invalid loop interval is reset."""
        config = TradingConfig(loop_interval=-1)
        assert config.loop_interval == 60

    def test_invalid_symbol_fetch_delay(self):
        """Test invalid symbol fetch delay is reset."""
        config = TradingConfig(symbol_fetch_delay=-5)
        assert config.symbol_fetch_delay == 5.0

    def test_invalid_risk_per_trade(self):
        """Test invalid risk per trade is reset."""
        config = TradingConfig(risk_per_trade_pct=150)
        assert config.risk_per_trade_pct == 1.0

    def test_invalid_stop_loss(self):
        """Test invalid stop loss is reset."""
        config = TradingConfig(stop_loss_pct=2.0)
        assert config.stop_loss_pct == 0.02

    def test_invalid_take_profit(self):
        """Test invalid take profit is reset."""
        config = TradingConfig(take_profit_pct=-0.1)
        assert config.take_profit_pct == 0.04


class TestAssetClassConfig:
    """Test AssetClassConfig dataclass."""

    def test_default_values(self):
        """Test default asset class configuration."""
        config = AssetClassConfig()
        assert config.symbols == []
        assert config.enabled is True
        assert config.max_weight == 0.40
        assert config.min_weight == 0.05

    def test_custom_symbols(self):
        """Test custom symbols."""
        config = AssetClassConfig(symbols=["BTC/USDT", "ETH/USDT"])
        assert len(config.symbols) == 2
        assert "BTC/USDT" in config.symbols

    def test_disabled(self):
        """Test disabled asset class."""
        config = AssetClassConfig(enabled=False)
        assert config.enabled is False

    def test_invalid_max_weight(self):
        """Test invalid max weight is reset."""
        config = AssetClassConfig(max_weight=2.0)
        assert config.max_weight == 0.40

    def test_invalid_min_weight(self):
        """Test invalid min weight is reset."""
        config = AssetClassConfig(min_weight=-0.1)
        assert config.min_weight == 0.05

    def test_min_weight_exceeds_max(self):
        """Test min weight exceeding max is reset."""
        config = AssetClassConfig(max_weight=0.2, min_weight=0.5)
        assert config.min_weight == 0.05


class TestDeepLearningConfig:
    """Test DeepLearningConfig dataclass."""

    def test_default_values(self):
        """Test default deep learning configuration."""
        config = DeepLearningConfig()
        assert config.enabled is True
        assert config.model_selection == "regime_based"
        assert config.default_model == "lstm"
        assert config.device == "auto"

    def test_custom_model_selection(self):
        """Test custom model selection."""
        config = DeepLearningConfig(model_selection="ensemble")
        assert config.model_selection == "ensemble"

    def test_invalid_model_selection(self):
        """Test invalid model selection is reset."""
        config = DeepLearningConfig(model_selection="invalid")
        assert config.model_selection == "regime_based"

    def test_invalid_default_model(self):
        """Test invalid default model is reset."""
        config = DeepLearningConfig(default_model="invalid")
        assert config.default_model == "lstm"

    def test_invalid_device(self):
        """Test invalid device is reset."""
        config = DeepLearningConfig(device="gpu")
        assert config.device == "auto"


class TestNotificationsConfig:
    """Test NotificationsConfig dataclass."""

    def test_default_values(self):
        """Test default notification configuration."""
        config = NotificationsConfig()
        assert config.telegram_enabled is True
        assert config.ai_explanations is False
        assert config.daily_summary is True
        assert config.daily_summary_hour == 20

    def test_custom_summary_hour(self):
        """Test custom summary hour."""
        config = NotificationsConfig(daily_summary_hour=8)
        assert config.daily_summary_hour == 8

    def test_invalid_summary_hour(self):
        """Test invalid summary hour is reset."""
        config = NotificationsConfig(daily_summary_hour=25)
        assert config.daily_summary_hour == 20


class TestAPIConfig:
    """Test APIConfig dataclass."""

    def test_default_values(self):
        """Test default API configuration."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.cors_origins == ["*"]
        assert config.stale_threshold == 300

    def test_custom_port(self):
        """Test custom port."""
        config = APIConfig(port=9000)
        assert config.port == 9000

    def test_invalid_port_high(self):
        """Test invalid high port is reset."""
        config = APIConfig(port=70000)
        assert config.port == 8000

    def test_invalid_port_low(self):
        """Test invalid low port is reset."""
        config = APIConfig(port=0)
        assert config.port == 8000

    def test_invalid_stale_threshold(self):
        """Test invalid stale threshold is reset."""
        config = APIConfig(stale_threshold=0)
        assert config.stale_threshold == 300


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_default_values(self):
        """Test default strategy configuration."""
        config = StrategyConfig()
        assert config.ema_fast == 12
        assert config.ema_slow == 26
        assert config.rsi_period == 14
        assert config.rsi_overbought == 70
        assert config.rsi_oversold == 30
        assert config.timeframe == "1h"

    def test_custom_ema_periods(self):
        """Test custom EMA periods."""
        config = StrategyConfig(ema_fast=5, ema_slow=20)
        assert config.ema_fast == 5
        assert config.ema_slow == 20

    def test_invalid_ema_fast(self):
        """Test invalid EMA fast is reset."""
        config = StrategyConfig(ema_fast=-1)
        assert config.ema_fast == 12

    def test_ema_slow_less_than_fast(self):
        """Test EMA slow <= fast is adjusted."""
        config = StrategyConfig(ema_fast=20, ema_slow=10)
        assert config.ema_slow == 40  # 2x fast

    def test_invalid_rsi_period(self):
        """Test invalid RSI period is reset."""
        config = StrategyConfig(rsi_period=0)
        assert config.rsi_period == 14

    def test_invalid_rsi_overbought(self):
        """Test invalid RSI overbought is reset."""
        config = StrategyConfig(rsi_overbought=30)
        assert config.rsi_overbought == 70

    def test_invalid_rsi_oversold(self):
        """Test invalid RSI oversold is reset."""
        config = StrategyConfig(rsi_oversold=60)
        assert config.rsi_oversold == 30


class TestPortfolioConfig:
    """Test PortfolioConfig dataclass."""

    def test_default_values(self):
        """Test default portfolio configuration."""
        config = PortfolioConfig()
        assert config.optimization_method == "risk_parity"
        assert config.rebalance_frequency == "daily"
        assert config.use_correlation_filter is True
        assert config.max_correlation == 0.85

    def test_custom_optimization_method(self):
        """Test custom optimization method."""
        config = PortfolioConfig(optimization_method="equal_weight")
        assert config.optimization_method == "equal_weight"

    def test_invalid_optimization_method(self):
        """Test invalid optimization method is reset."""
        config = PortfolioConfig(optimization_method="invalid")
        assert config.optimization_method == "risk_parity"

    def test_invalid_rebalance_frequency(self):
        """Test invalid rebalance frequency is reset."""
        config = PortfolioConfig(rebalance_frequency="hourly")
        assert config.rebalance_frequency == "daily"

    def test_invalid_max_correlation(self):
        """Test invalid max correlation is reset."""
        config = PortfolioConfig(max_correlation=1.5)
        assert config.max_correlation == 0.85


class TestOrchestratorConfig:
    """Test OrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default orchestrator configuration."""
        config = OrchestratorConfig()
        assert config.health_check_interval == 60
        assert config.max_restart_attempts == 3
        assert config.restart_cooldown == 300
        assert config.auto_restart is True

    def test_invalid_health_check_interval(self):
        """Test invalid health check interval is reset."""
        config = OrchestratorConfig(health_check_interval=0)
        assert config.health_check_interval == 60

    def test_invalid_restart_attempts(self):
        """Test invalid restart attempts is reset."""
        config = OrchestratorConfig(max_restart_attempts=-1)
        assert config.max_restart_attempts == 3

    def test_invalid_restart_cooldown(self):
        """Test invalid restart cooldown is reset."""
        config = OrchestratorConfig(restart_cooldown=-100)
        assert config.restart_cooldown == 300


class TestAppConfig:
    """Test AppConfig dataclass."""

    def test_default_app_config(self):
        """Test default app configuration."""
        config = AppConfig()
        assert config.general is not None
        assert config.trading is not None
        assert config.crypto is not None
        assert config.commodities is not None
        assert config.stocks is not None
        assert config.deep_learning is not None
        assert config.notifications is not None
        assert config.api is not None
        assert config.strategy is not None
        assert config.portfolio is not None
        assert config.orchestrator is not None

    def test_default_crypto_symbols(self):
        """Test default crypto symbols."""
        config = AppConfig()
        assert len(config.crypto.symbols) == 4
        assert "BTC/USDT" in config.crypto.symbols

    def test_default_stock_symbols(self):
        """Test default stock symbols."""
        config = AppConfig()
        assert len(config.stocks.symbols) == 5
        assert "AAPL" in config.stocks.symbols

    def test_get_all_symbols(self):
        """Test get_all_symbols method."""
        config = AppConfig()
        symbols = config.get_all_symbols()
        # All asset classes enabled by default
        assert len(symbols) > 0
        assert "BTC/USDT" in symbols
        assert "AAPL" in symbols

    def test_get_all_symbols_with_disabled(self):
        """Test get_all_symbols with some disabled."""
        config = AppConfig()
        config.crypto.enabled = False
        symbols = config.get_all_symbols()
        assert "BTC/USDT" not in symbols
        assert "AAPL" in symbols

    def test_nested_config_access(self):
        """Test nested configuration access."""
        config = AppConfig()
        assert config.trading.initial_capital == 10000.0
        assert config.api.port == 8000
        assert config.strategy.rsi_period == 14
