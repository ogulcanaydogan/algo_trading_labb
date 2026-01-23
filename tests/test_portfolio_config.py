"""
Tests for portfolio configuration module.
"""

import pytest
import json
import tempfile
from pathlib import Path

from bot.portfolio import PortfolioAssetConfig, PortfolioConfig


class TestPortfolioAssetConfig:
    """Test PortfolioAssetConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = PortfolioAssetConfig(symbol="BTC/USDT")
        assert config.symbol == "BTC/USDT"
        assert config.asset_type == "crypto"
        assert config.data_symbol is None
        assert config.paper_mode is None

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        config = PortfolioAssetConfig(
            symbol="ETH/USDT",
            asset_type="crypto",
            timeframe="4h",
            allocation_pct=0.25,
            stop_loss_pct=0.02,
        )
        assert config.symbol == "ETH/USDT"
        assert config.timeframe == "4h"
        assert config.allocation_pct == 0.25
        assert config.stop_loss_pct == 0.02

    def test_all_optional_fields(self):
        """Test all optional fields can be set."""
        config = PortfolioAssetConfig(
            symbol="SOL/USDT",
            data_symbol="SOL-USD",
            macro_symbol="SPY",
            timeframe="1h",
            lookback=1000,
            paper_mode=True,
            exchange_id="binance",
            starting_balance=10000.0,
            allocation_pct=0.1,
            risk_per_trade_pct=0.5,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            loop_interval_seconds=30,
            macro_events_path="/path/to/events",
            macro_refresh_seconds=600,
            data_dir="/path/to/data",
        )
        assert config.data_symbol == "SOL-USD"
        assert config.macro_symbol == "SPY"
        assert config.lookback == 1000
        assert config.paper_mode is True
        assert config.starting_balance == 10000.0


class TestPortfolioConfig:
    """Test PortfolioConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = PortfolioConfig()
        assert config.portfolio_capital == 100_000.0
        assert config.default_timeframe == "1h"
        assert config.default_loop_interval_seconds == 60
        assert config.default_lookback == 500
        assert config.default_paper_mode is True
        assert config.default_exchange_id == "binance"
        assert config.default_risk_per_trade_pct == 0.5
        assert config.default_stop_loss_pct == 0.01
        assert config.default_take_profit_pct == 0.02
        assert config.macro_refresh_seconds == 300
        assert len(config.assets) == 0

    def test_with_assets(self):
        """Test config with assets."""
        assets = [
            PortfolioAssetConfig(symbol="BTC/USDT"),
            PortfolioAssetConfig(symbol="ETH/USDT"),
        ]
        config = PortfolioConfig(assets=assets, portfolio_capital=50000.0)
        assert len(config.assets) == 2
        assert config.portfolio_capital == 50000.0
        assert config.assets[0].symbol == "BTC/USDT"

    def test_from_dict_empty(self):
        """Test from_dict with empty dict."""
        config = PortfolioConfig.from_dict({})
        assert config.portfolio_capital == 100_000.0
        assert len(config.assets) == 0

    def test_from_dict_full(self):
        """Test from_dict with full config."""
        payload = {
            "portfolio_capital": 200000.0,
            "default_timeframe": "4h",
            "default_loop_interval_seconds": 120,
            "default_lookback": 1000,
            "default_paper_mode": False,
            "default_exchange_id": "kraken",
            "default_risk_per_trade_pct": 1.0,
            "default_stop_loss_pct": 0.02,
            "default_take_profit_pct": 0.05,
            "macro_refresh_seconds": 600,
            "data_dir": "/custom/data/dir",
            "assets": [
                {"symbol": "BTC/USDT", "allocation_pct": 0.5},
                {"symbol": "ETH/USDT", "allocation_pct": 0.3},
            ],
        }
        config = PortfolioConfig.from_dict(payload)
        assert config.portfolio_capital == 200000.0
        assert config.default_timeframe == "4h"
        assert config.default_loop_interval_seconds == 120
        assert config.default_lookback == 1000
        assert config.default_paper_mode is False
        assert config.default_exchange_id == "kraken"
        assert config.default_risk_per_trade_pct == 1.0
        assert config.default_stop_loss_pct == 0.02
        assert config.default_take_profit_pct == 0.05
        assert config.macro_refresh_seconds == 600
        assert len(config.assets) == 2
        assert config.assets[0].symbol == "BTC/USDT"
        assert config.assets[0].allocation_pct == 0.5

    def test_from_dict_with_macro_path(self):
        """Test from_dict with macro events path."""
        payload = {
            "macro_events_path": "/path/to/events.json",
            "assets": [],
        }
        config = PortfolioConfig.from_dict(payload)
        assert config.macro_events_path == Path("/path/to/events.json")

    def test_load_from_file(self):
        """Test loading config from JSON file."""
        config_data = {
            "portfolio_capital": 75000.0,
            "default_timeframe": "2h",
            "assets": [
                {"symbol": "AVAX/USDT", "asset_type": "crypto"},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = PortfolioConfig.load(temp_path)
            assert config.portfolio_capital == 75000.0
            assert config.default_timeframe == "2h"
            assert len(config.assets) == 1
            assert config.assets[0].symbol == "AVAX/USDT"
        finally:
            temp_path.unlink()

    def test_data_dir_path(self):
        """Test data_dir is converted to Path."""
        payload = {"data_dir": "./custom/portfolio/data"}
        config = PortfolioConfig.from_dict(payload)
        assert isinstance(config.data_dir, Path)
        assert str(config.data_dir) == "custom/portfolio/data"


class TestConfigValidation:
    """Test configuration validation edge cases."""

    def test_empty_symbol(self):
        """Test empty symbol is allowed (no validation)."""
        config = PortfolioAssetConfig(symbol="")
        assert config.symbol == ""

    def test_negative_capital(self):
        """Test negative capital is allowed (no validation)."""
        config = PortfolioConfig(portfolio_capital=-1000.0)
        assert config.portfolio_capital == -1000.0

    def test_extreme_allocation(self):
        """Test extreme allocation values are allowed."""
        asset = PortfolioAssetConfig(symbol="BTC/USDT", allocation_pct=2.0)
        assert asset.allocation_pct == 2.0

    def test_zero_risk(self):
        """Test zero risk parameters are allowed."""
        config = PortfolioConfig(
            default_risk_per_trade_pct=0.0,
            default_stop_loss_pct=0.0,
            default_take_profit_pct=0.0,
        )
        assert config.default_risk_per_trade_pct == 0.0
