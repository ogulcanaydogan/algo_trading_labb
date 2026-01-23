"""
Unified Configuration System for Algo Trading Lab.

This module provides a centralized configuration system that:
- Loads configuration from config.yaml
- Supports environment variable overrides
- Validates configuration values
- Provides type-safe access via dataclasses

Usage:
    from bot.config import load_config, AppConfig

    config = load_config()
    print(config.trading.initial_capital)
    print(config.crypto.symbols)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class GeneralConfig:
    """General application settings."""

    log_level: str = "INFO"
    data_dir: str = "./data"

    def __post_init__(self):
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            self.log_level = "INFO"
        else:
            self.log_level = self.log_level.upper()


@dataclass
class TradingConfig:
    """Core trading parameters."""

    initial_capital: float = 10000.0
    rebalance_threshold: float = 0.05
    loop_interval: int = 60
    symbol_fetch_delay: float = 5.0  # Delay between fetching each symbol (seconds)
    risk_per_trade_pct: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04

    def __post_init__(self):
        # Validate ranges
        if self.initial_capital <= 0:
            self.initial_capital = 10000.0
        if not 0 < self.rebalance_threshold < 1:
            self.rebalance_threshold = 0.05
        if self.loop_interval < 1:
            self.loop_interval = 60
        if self.symbol_fetch_delay < 0:
            self.symbol_fetch_delay = 5.0
        if not 0 < self.risk_per_trade_pct <= 100:
            self.risk_per_trade_pct = 1.0
        if not 0 < self.stop_loss_pct < 1:
            self.stop_loss_pct = 0.02
        if not 0 < self.take_profit_pct < 1:
            self.take_profit_pct = 0.04


@dataclass
class AssetClassConfig:
    """Configuration for an asset class (crypto, commodities, stocks)."""

    symbols: List[str] = field(default_factory=list)
    enabled: bool = True
    max_weight: float = 0.40
    min_weight: float = 0.05

    def __post_init__(self):
        # Validate weights
        if not 0 < self.max_weight <= 1:
            self.max_weight = 0.40
        if not 0 <= self.min_weight < self.max_weight:
            self.min_weight = 0.05


@dataclass
class DeepLearningConfig:
    """Deep learning and ML settings."""

    enabled: bool = True
    model_selection: str = "regime_based"
    default_model: str = "lstm"
    device: str = "auto"

    def __post_init__(self):
        # Validate model selection strategy
        valid_strategies = {"regime_based", "ensemble", "single_best"}
        if self.model_selection not in valid_strategies:
            self.model_selection = "regime_based"

        # Validate default model
        valid_models = {"lstm", "transformer", "xgboost", "random_forest"}
        if self.default_model not in valid_models:
            self.default_model = "lstm"

        # Validate device
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            self.device = "auto"


@dataclass
class NotificationsConfig:
    """Notification settings."""

    telegram_enabled: bool = True
    ai_explanations: bool = False
    daily_summary: bool = True
    daily_summary_hour: int = 20

    def __post_init__(self):
        # Validate hour
        if not 0 <= self.daily_summary_hour < 24:
            self.daily_summary_hour = 20


@dataclass
class APIConfig:
    """API server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    stale_threshold: int = 300

    def __post_init__(self):
        # Validate port
        if not 1 <= self.port <= 65535:
            self.port = 8000
        if self.stale_threshold < 1:
            self.stale_threshold = 300


@dataclass
class StrategyConfig:
    """Trading strategy parameters."""

    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    timeframe: str = "1h"

    def __post_init__(self):
        # Validate EMA periods
        if self.ema_fast < 1:
            self.ema_fast = 12
        if self.ema_slow <= self.ema_fast:
            self.ema_slow = self.ema_fast * 2

        # Validate RSI
        if not 1 <= self.rsi_period <= 100:
            self.rsi_period = 14
        if not 50 <= self.rsi_overbought <= 100:
            self.rsi_overbought = 70
        if not 0 <= self.rsi_oversold <= 50:
            self.rsi_oversold = 30


@dataclass
class PortfolioConfig:
    """Portfolio optimization settings."""

    optimization_method: str = "risk_parity"
    rebalance_frequency: str = "daily"
    use_correlation_filter: bool = True
    max_correlation: float = 0.85

    def __post_init__(self):
        # Validate optimization method
        valid_methods = {
            "equal_weight",
            "risk_parity",
            "min_volatility",
            "max_sharpe",
            "max_diversification",
            "inverse_volatility",
        }
        if self.optimization_method not in valid_methods:
            self.optimization_method = "risk_parity"

        # Validate rebalance frequency
        valid_frequencies = {"daily", "weekly", "monthly"}
        if self.rebalance_frequency not in valid_frequencies:
            self.rebalance_frequency = "daily"

        # Validate correlation
        if not 0 < self.max_correlation <= 1:
            self.max_correlation = 0.85


@dataclass
class OrchestratorConfig:
    """Multi-market orchestrator settings."""

    health_check_interval: int = 60
    max_restart_attempts: int = 3
    restart_cooldown: int = 300
    auto_restart: bool = True

    def __post_init__(self):
        if self.health_check_interval < 1:
            self.health_check_interval = 60
        if self.max_restart_attempts < 0:
            self.max_restart_attempts = 3
        if self.restart_cooldown < 0:
            self.restart_cooldown = 300


@dataclass
class AppConfig:
    """Root configuration containing all sections."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    crypto: AssetClassConfig = field(
        default_factory=lambda: AssetClassConfig(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
        )
    )
    commodities: AssetClassConfig = field(
        default_factory=lambda: AssetClassConfig(
            symbols=["XAU/USD", "XAG/USD", "USOIL/USD", "NATGAS/USD"]
        )
    )
    stocks: AssetClassConfig = field(
        default_factory=lambda: AssetClassConfig(symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"])
    )
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    def get_all_symbols(self) -> List[str]:
        """Get all enabled trading symbols across asset classes."""
        symbols = []
        if self.crypto.enabled:
            symbols.extend(self.crypto.symbols)
        if self.commodities.enabled:
            symbols.extend(self.commodities.symbols)
        if self.stocks.enabled:
            symbols.extend(self.stocks.symbols)
        return symbols

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "general": {
                "log_level": self.general.log_level,
                "data_dir": self.general.data_dir,
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "rebalance_threshold": self.trading.rebalance_threshold,
                "loop_interval": self.trading.loop_interval,
                "risk_per_trade_pct": self.trading.risk_per_trade_pct,
                "stop_loss_pct": self.trading.stop_loss_pct,
                "take_profit_pct": self.trading.take_profit_pct,
            },
            "crypto": {
                "symbols": self.crypto.symbols,
                "enabled": self.crypto.enabled,
                "max_weight": self.crypto.max_weight,
                "min_weight": self.crypto.min_weight,
            },
            "commodities": {
                "symbols": self.commodities.symbols,
                "enabled": self.commodities.enabled,
                "max_weight": self.commodities.max_weight,
                "min_weight": self.commodities.min_weight,
            },
            "stocks": {
                "symbols": self.stocks.symbols,
                "enabled": self.stocks.enabled,
                "max_weight": self.stocks.max_weight,
                "min_weight": self.stocks.min_weight,
            },
            "deep_learning": {
                "enabled": self.deep_learning.enabled,
                "model_selection": self.deep_learning.model_selection,
                "default_model": self.deep_learning.default_model,
                "device": self.deep_learning.device,
            },
            "notifications": {
                "telegram_enabled": self.notifications.telegram_enabled,
                "ai_explanations": self.notifications.ai_explanations,
                "daily_summary": self.notifications.daily_summary,
                "daily_summary_hour": self.notifications.daily_summary_hour,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors_origins": self.api.cors_origins,
                "stale_threshold": self.api.stale_threshold,
            },
            "strategy": {
                "ema_fast": self.strategy.ema_fast,
                "ema_slow": self.strategy.ema_slow,
                "rsi_period": self.strategy.rsi_period,
                "rsi_overbought": self.strategy.rsi_overbought,
                "rsi_oversold": self.strategy.rsi_oversold,
                "timeframe": self.strategy.timeframe,
            },
            "portfolio": {
                "optimization_method": self.portfolio.optimization_method,
                "rebalance_frequency": self.portfolio.rebalance_frequency,
                "use_correlation_filter": self.portfolio.use_correlation_filter,
                "max_correlation": self.portfolio.max_correlation,
            },
            "orchestrator": {
                "health_check_interval": self.orchestrator.health_check_interval,
                "max_restart_attempts": self.orchestrator.max_restart_attempts,
                "restart_cooldown": self.orchestrator.restart_cooldown,
                "auto_restart": self.orchestrator.auto_restart,
            },
        }


# =============================================================================
# Environment Variable Mapping
# =============================================================================

# Maps environment variables to config paths
# Format: ENV_VAR -> (section, key, type)
ENV_VAR_MAPPING: Dict[str, tuple] = {
    # General
    "LOG_LEVEL": ("general", "log_level", str),
    "DATA_DIR": ("general", "data_dir", str),
    # Trading
    "INITIAL_CAPITAL": ("trading", "initial_capital", float),
    "STARTING_BALANCE": ("trading", "initial_capital", float),  # Alias
    "REBALANCE_THRESHOLD": ("trading", "rebalance_threshold", float),
    "LOOP_INTERVAL": ("trading", "loop_interval", int),
    "LOOP_INTERVAL_SECONDS": ("trading", "loop_interval", int),  # Alias
    "RISK_PER_TRADE_PCT": ("trading", "risk_per_trade_pct", float),
    "STOP_LOSS_PCT": ("trading", "stop_loss_pct", float),
    "TAKE_PROFIT_PCT": ("trading", "take_profit_pct", float),
    # Deep Learning
    "USE_DEEP_LEARNING": ("deep_learning", "enabled", bool),
    "DL_MODEL_SELECTION": ("deep_learning", "model_selection", str),
    "ML_DEFAULT_MODEL": ("deep_learning", "default_model", str),
    "ML_DEVICE": ("deep_learning", "device", str),
    # Notifications
    "TELEGRAM_ENABLED": ("notifications", "telegram_enabled", bool),
    "ENABLE_AI_EXPLANATIONS": ("notifications", "ai_explanations", bool),
    "DAILY_SUMMARY_HOUR": ("notifications", "daily_summary_hour", int),
    # API
    "API_HOST": ("api", "host", str),
    "API_PORT": ("api", "port", int),
    "BOT_STALE_THRESHOLD_SECONDS": ("api", "stale_threshold", int),
    # Strategy
    "EMA_FAST": ("strategy", "ema_fast", int),
    "EMA_SLOW": ("strategy", "ema_slow", int),
    "RSI_PERIOD": ("strategy", "rsi_period", int),
    "RSI_OVERBOUGHT": ("strategy", "rsi_overbought", int),
    "RSI_OVERSOLD": ("strategy", "rsi_oversold", int),
    "TIMEFRAME": ("strategy", "timeframe", str),
    # Orchestrator
    "HEALTH_CHECK_INTERVAL": ("orchestrator", "health_check_interval", int),
    "MAX_RESTART_ATTEMPTS": ("orchestrator", "max_restart_attempts", int),
    "RESTART_COOLDOWN": ("orchestrator", "restart_cooldown", int),
}


# =============================================================================
# Configuration Loading Functions
# =============================================================================


def _parse_bool(value: Union[str, bool]) -> bool:
    """Parse a string or bool into a boolean value."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "on")


def _parse_value(value: str, target_type: type) -> Any:
    """Parse a string value to the target type."""
    if target_type == bool:
        return _parse_bool(value)
    elif target_type == int:
        return int(float(value))  # Handle "10.0" -> 10
    elif target_type == float:
        return float(value)
    return value


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config dictionary."""
    for env_var, (section, key, value_type) in ENV_VAR_MAPPING.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            if section not in config_dict:
                config_dict[section] = {}
            try:
                config_dict[section][key] = _parse_value(env_value, value_type)
            except (ValueError, TypeError):
                pass  # Keep original value if parsing fails

    # Handle CORS origins specially (comma-separated)
    cors_env = os.getenv("CORS_ORIGINS")
    if cors_env:
        if "api" not in config_dict:
            config_dict["api"] = {}
        config_dict["api"]["cors_origins"] = [o.strip() for o in cors_env.split(",")]

    return config_dict


def _find_config_file() -> Optional[Path]:
    """Find the config.yaml file."""
    # Check common locations
    search_paths = [
        Path.cwd() / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
        Path.home() / ".algo_trading_lab" / "config.yaml",
    ]

    # Also check CONFIG_FILE environment variable
    env_config = os.getenv("CONFIG_FILE")
    if env_config:
        search_paths.insert(0, Path(env_config))

    for path in search_paths:
        if path.exists():
            return path

    return None


def _load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = _find_config_file()

    if config_path is None or not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, IOError) as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


def _dict_to_config(config_dict: Dict[str, Any]) -> AppConfig:
    """Convert a configuration dictionary to AppConfig dataclass."""
    general_dict = config_dict.get("general", {})
    trading_dict = config_dict.get("trading", {})
    crypto_dict = config_dict.get("crypto", {})
    commodities_dict = config_dict.get("commodities", {})
    stocks_dict = config_dict.get("stocks", {})
    deep_learning_dict = config_dict.get("deep_learning", {})
    notifications_dict = config_dict.get("notifications", {})
    api_dict = config_dict.get("api", {})
    strategy_dict = config_dict.get("strategy", {})
    portfolio_dict = config_dict.get("portfolio", {})
    orchestrator_dict = config_dict.get("orchestrator", {})

    return AppConfig(
        general=GeneralConfig(
            log_level=general_dict.get("log_level", "INFO"),
            data_dir=general_dict.get("data_dir", "./data"),
        ),
        trading=TradingConfig(
            initial_capital=trading_dict.get("initial_capital", 10000.0),
            rebalance_threshold=trading_dict.get("rebalance_threshold", 0.05),
            loop_interval=trading_dict.get("loop_interval", 60),
            risk_per_trade_pct=trading_dict.get("risk_per_trade_pct", 1.0),
            stop_loss_pct=trading_dict.get("stop_loss_pct", 0.02),
            take_profit_pct=trading_dict.get("take_profit_pct", 0.04),
        ),
        crypto=AssetClassConfig(
            symbols=crypto_dict.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]),
            enabled=crypto_dict.get("enabled", True),
            max_weight=crypto_dict.get("max_weight", 0.40),
            min_weight=crypto_dict.get("min_weight", 0.05),
        ),
        commodities=AssetClassConfig(
            symbols=commodities_dict.get(
                "symbols", ["XAU/USD", "XAG/USD", "USOIL/USD", "NATGAS/USD"]
            ),
            enabled=commodities_dict.get("enabled", True),
            max_weight=commodities_dict.get("max_weight", 0.35),
            min_weight=commodities_dict.get("min_weight", 0.05),
        ),
        stocks=AssetClassConfig(
            symbols=stocks_dict.get("symbols", ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]),
            enabled=stocks_dict.get("enabled", True),
            max_weight=stocks_dict.get("max_weight", 0.30),
            min_weight=stocks_dict.get("min_weight", 0.05),
        ),
        deep_learning=DeepLearningConfig(
            enabled=deep_learning_dict.get("enabled", True),
            model_selection=deep_learning_dict.get("model_selection", "regime_based"),
            default_model=deep_learning_dict.get("default_model", "lstm"),
            device=deep_learning_dict.get("device", "auto"),
        ),
        notifications=NotificationsConfig(
            telegram_enabled=notifications_dict.get("telegram_enabled", True),
            ai_explanations=notifications_dict.get("ai_explanations", False),
            daily_summary=notifications_dict.get("daily_summary", True),
            daily_summary_hour=notifications_dict.get("daily_summary_hour", 20),
        ),
        api=APIConfig(
            host=api_dict.get("host", "0.0.0.0"),
            port=api_dict.get("port", 8000),
            cors_origins=api_dict.get("cors_origins", ["*"]),
            stale_threshold=api_dict.get("stale_threshold", 300),
        ),
        strategy=StrategyConfig(
            ema_fast=strategy_dict.get("ema_fast", 12),
            ema_slow=strategy_dict.get("ema_slow", 26),
            rsi_period=strategy_dict.get("rsi_period", 14),
            rsi_overbought=strategy_dict.get("rsi_overbought", 70),
            rsi_oversold=strategy_dict.get("rsi_oversold", 30),
            timeframe=strategy_dict.get("timeframe", "1h"),
        ),
        portfolio=PortfolioConfig(
            optimization_method=portfolio_dict.get("optimization_method", "risk_parity"),
            rebalance_frequency=portfolio_dict.get("rebalance_frequency", "daily"),
            use_correlation_filter=portfolio_dict.get("use_correlation_filter", True),
            max_correlation=portfolio_dict.get("max_correlation", 0.85),
        ),
        orchestrator=OrchestratorConfig(
            health_check_interval=orchestrator_dict.get("health_check_interval", 60),
            max_restart_attempts=orchestrator_dict.get("max_restart_attempts", 3),
            restart_cooldown=orchestrator_dict.get("restart_cooldown", 300),
            auto_restart=orchestrator_dict.get("auto_restart", True),
        ),
    )


# Global cached config instance
_cached_config: Optional[AppConfig] = None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> AppConfig:
    """
    Load the application configuration.

    Priority order (highest to lowest):
    1. Environment variables
    2. Config file (config.yaml)
    3. Default values

    Args:
        config_path: Optional path to config file. If None, searches default locations.
        use_cache: If True, returns cached config on subsequent calls.

    Returns:
        AppConfig instance with all configuration values.

    Example:
        >>> config = load_config()
        >>> print(config.trading.initial_capital)
        10000.0
        >>> print(config.crypto.symbols)
        ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT']
    """
    global _cached_config

    if use_cache and _cached_config is not None:
        return _cached_config

    # Convert string path to Path
    if config_path is not None:
        config_path = Path(config_path)

    # Load from YAML file
    config_dict = _load_yaml_config(config_path)

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Convert to dataclass
    config = _dict_to_config(config_dict)

    if use_cache:
        _cached_config = config

    return config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Force reload of configuration, ignoring cache.

    Args:
        config_path: Optional path to config file.

    Returns:
        Freshly loaded AppConfig instance.
    """
    global _cached_config
    _cached_config = None
    return load_config(config_path=config_path, use_cache=True)


def get_config() -> AppConfig:
    """
    Get the current configuration (alias for load_config with caching).

    Returns:
        Cached or freshly loaded AppConfig instance.
    """
    return load_config(use_cache=True)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_data_dir() -> Path:
    """Get the data directory path."""
    config = get_config()
    return Path(config.general.data_dir).expanduser().resolve()


def get_log_level() -> str:
    """Get the configured log level."""
    return get_config().general.log_level


def is_deep_learning_enabled() -> bool:
    """Check if deep learning is enabled."""
    return get_config().deep_learning.enabled


def get_crypto_symbols() -> List[str]:
    """Get enabled crypto trading symbols."""
    config = get_config()
    return config.crypto.symbols if config.crypto.enabled else []


def get_commodity_symbols() -> List[str]:
    """Get enabled commodity trading symbols."""
    config = get_config()
    return config.commodities.symbols if config.commodities.enabled else []


def get_stock_symbols() -> List[str]:
    """Get enabled stock trading symbols."""
    config = get_config()
    return config.stocks.symbols if config.stocks.enabled else []


# =============================================================================
# Module Initialization
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Log Level: {config.general.log_level}")
    print(f"Data Dir: {config.general.data_dir}")
    print(f"Initial Capital: ${config.trading.initial_capital:,.2f}")
    print(f"Crypto Symbols: {config.crypto.symbols}")
    print(f"Commodity Symbols: {config.commodities.symbols}")
    print(f"Stock Symbols: {config.stocks.symbols}")
    print(f"Deep Learning Enabled: {config.deep_learning.enabled}")
    print(f"Telegram Enabled: {config.notifications.telegram_enabled}")
    print(f"API Port: {config.api.port}")
