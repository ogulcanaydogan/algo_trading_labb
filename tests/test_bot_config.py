"""Tests for bot.bot module - BotConfig class."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.bot import BotConfig


class TestBotConfig:
    """Tests for BotConfig dataclass."""

    def test_default_values(self) -> None:
        """Test BotConfig default values."""
        config = BotConfig()

        assert config.symbol == "BTC/USDT"
        assert config.data_symbol is None
        assert config.asset_type == "crypto"
        assert config.timeframe == "1m"
        assert config.loop_interval_seconds == 60
        assert config.lookback == 250
        assert config.paper_mode is True
        assert config.exchange_id == "binance"
        assert config.starting_balance == 10_000.0
        assert config.risk_per_trade_pct == 0.5
        assert config.stop_loss_pct == 0.004
        assert config.take_profit_pct == 0.008
        assert config.data_dir == Path("./data")
        assert config.use_kelly_sizing is False
        assert config.sizing_method == "volatility_adjusted"
        assert config.max_position_pct == 0.25
        assert config.use_htf_filter is True
        assert config.htf_strict_mode is False

    def test_custom_values(self) -> None:
        """Test BotConfig with custom values."""
        config = BotConfig(
            symbol="ETH/USDT",
            data_symbol="ETH-USD",
            asset_type="crypto",
            timeframe="5m",
            lookback=500,
            paper_mode=False,
            starting_balance=50_000.0,
            use_kelly_sizing=True,
            sizing_method="kelly",
        )

        assert config.symbol == "ETH/USDT"
        assert config.data_symbol == "ETH-USD"
        assert config.timeframe == "5m"
        assert config.lookback == 500
        assert config.paper_mode is False
        assert config.starting_balance == 50_000.0
        assert config.use_kelly_sizing is True
        assert config.sizing_method == "kelly"

    def test_from_env_defaults(self) -> None:
        """Test from_env with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = BotConfig.from_env()

        assert config.symbol == "BTC/USDT"
        assert config.paper_mode is True
        assert config.starting_balance == 10_000.0

    def test_from_env_custom_symbol(self) -> None:
        """Test from_env with custom symbol."""
        with patch.dict(os.environ, {"SYMBOL": "SOL/USDT"}, clear=True):
            config = BotConfig.from_env()

        assert config.symbol == "SOL/USDT"

    def test_from_env_paper_mode_false(self) -> None:
        """Test from_env with paper mode disabled."""
        with patch.dict(os.environ, {"PAPER_MODE": "false"}, clear=True):
            config = BotConfig.from_env()

        assert config.paper_mode is False

    def test_from_env_numeric_values(self) -> None:
        """Test from_env with numeric values."""
        env = {
            "LOOP_INTERVAL_SECONDS": "120",
            "LOOKBACK": "300",
            "STARTING_BALANCE": "25000",
            "RISK_PER_TRADE_PCT": "1.0",
            "STOP_LOSS_PCT": "0.01",
            "TAKE_PROFIT_PCT": "0.02",
            "MAX_POSITION_PCT": "0.30",
        }

        with patch.dict(os.environ, env, clear=True):
            config = BotConfig.from_env()

        assert config.loop_interval_seconds == 120
        assert config.lookback == 300
        assert config.starting_balance == 25_000.0
        assert config.risk_per_trade_pct == 1.0
        assert config.stop_loss_pct == 0.01
        assert config.take_profit_pct == 0.02
        assert config.max_position_pct == 0.30

    def test_from_env_data_symbol(self) -> None:
        """Test from_env with data symbol override."""
        env = {
            "SYMBOL": "XAU/USD",
            "DATA_SYMBOL": "GC=F",
        }

        with patch.dict(os.environ, env, clear=True):
            config = BotConfig.from_env()

        assert config.symbol == "XAU/USD"
        assert config.data_symbol == "GC=F"

    def test_from_env_macro_symbol(self) -> None:
        """Test from_env with macro symbol."""
        with patch.dict(os.environ, {"MACRO_SYMBOL": "SPY"}, clear=True):
            config = BotConfig.from_env()

        assert config.macro_symbol == "SPY"

    def test_from_env_asset_type(self) -> None:
        """Test from_env with different asset types."""
        for asset_type in ["crypto", "stock", "commodity"]:
            with patch.dict(os.environ, {"ASSET_TYPE": asset_type}, clear=True):
                config = BotConfig.from_env()

            assert config.asset_type == asset_type

    def test_from_env_exchange_id(self) -> None:
        """Test from_env with different exchanges."""
        with patch.dict(os.environ, {"EXCHANGE_ID": "kraken"}, clear=True):
            config = BotConfig.from_env()

        assert config.exchange_id == "kraken"

    def test_from_env_kelly_sizing(self) -> None:
        """Test from_env with Kelly sizing enabled."""
        with patch.dict(os.environ, {"USE_KELLY_SIZING": "true"}, clear=True):
            config = BotConfig.from_env()

        assert config.use_kelly_sizing is True

    def test_from_env_sizing_method(self) -> None:
        """Test from_env with different sizing methods."""
        methods = ["fixed_fraction", "kelly", "half_kelly", "volatility_adjusted", "atr_based"]

        for method in methods:
            with patch.dict(os.environ, {"SIZING_METHOD": method}, clear=True):
                config = BotConfig.from_env()

            assert config.sizing_method == method

    def test_from_env_htf_settings(self) -> None:
        """Test from_env with HTF filter settings."""
        env = {
            "USE_HTF_FILTER": "false",
            "HTF_STRICT_MODE": "true",
        }

        with patch.dict(os.environ, env, clear=True):
            config = BotConfig.from_env()

        assert config.use_htf_filter is False
        assert config.htf_strict_mode is True

    def test_from_env_data_dir(self) -> None:
        """Test from_env with custom data directory."""
        with patch.dict(os.environ, {"DATA_DIR": "/tmp/custom_data"}, clear=True):
            config = BotConfig.from_env()

        assert config.data_dir == Path("/tmp/custom_data")

    def test_from_env_macro_events_path(self) -> None:
        """Test from_env with macro events path."""
        with patch.dict(os.environ, {"MACRO_EVENTS_PATH": "/data/events.json"}, clear=True):
            config = BotConfig.from_env()

        assert config.macro_events_path == Path("/data/events.json")

    def test_from_env_macro_events_path_empty(self) -> None:
        """Test from_env with empty macro events path."""
        with patch.dict(os.environ, {}, clear=True):
            config = BotConfig.from_env()

        assert config.macro_events_path is None

    def test_from_env_playbook_assets_path(self) -> None:
        """Test from_env with playbook assets path."""
        with patch.dict(os.environ, {"PLAYBOOK_ASSETS_PATH": "/data/assets.yaml"}, clear=True):
            config = BotConfig.from_env()

        assert config.playbook_assets_path == Path("/data/assets.yaml")

    def test_from_env_macro_refresh_seconds(self) -> None:
        """Test from_env with custom macro refresh interval."""
        with patch.dict(os.environ, {"MACRO_REFRESH_SECONDS": "600"}, clear=True):
            config = BotConfig.from_env()

        assert config.macro_refresh_seconds == 600

    def test_data_dir_path_type(self) -> None:
        """Test that data_dir is a Path object."""
        config = BotConfig(data_dir=Path("/custom/path"))

        assert isinstance(config.data_dir, Path)
        assert config.data_dir == Path("/custom/path")
