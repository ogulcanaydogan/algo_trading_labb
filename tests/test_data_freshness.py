"""
Tests for data freshness monitoring module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from bot.data_freshness import (
    FreshnessStatus,
    DataFreshnessConfig,
    DataFreshnessResult,
    DataFreshnessMonitor,
    get_monitor,
    update_data,
    is_data_fresh,
)


class TestFreshnessStatus:
    """Test FreshnessStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.STALE.value == "stale"
        assert FreshnessStatus.EXPIRED.value == "expired"
        assert FreshnessStatus.UNKNOWN.value == "unknown"


class TestDataFreshnessConfig:
    """Test DataFreshnessConfig dataclass."""

    def test_default_crypto_thresholds(self):
        """Test default crypto thresholds."""
        config = DataFreshnessConfig()
        assert config.crypto_fresh_seconds == 60
        assert config.crypto_stale_seconds == 300
        assert config.crypto_expired_seconds == 900

    def test_default_stock_thresholds(self):
        """Test default stock thresholds."""
        config = DataFreshnessConfig()
        assert config.stock_fresh_seconds == 120
        assert config.stock_stale_seconds == 600
        assert config.stock_expired_seconds == 1800

    def test_default_commodity_thresholds(self):
        """Test default commodity thresholds."""
        config = DataFreshnessConfig()
        assert config.commodity_fresh_seconds == 120
        assert config.commodity_stale_seconds == 600
        assert config.commodity_expired_seconds == 1200

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = DataFreshnessConfig(
            crypto_fresh_seconds=30,
            crypto_stale_seconds=120,
            crypto_expired_seconds=300,
        )
        assert config.crypto_fresh_seconds == 30
        assert config.crypto_stale_seconds == 120
        assert config.crypto_expired_seconds == 300


class TestDataFreshnessMonitor:
    """Test DataFreshnessMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return DataFreshnessMonitor()

    @pytest.fixture
    def custom_monitor(self):
        """Create monitor with custom config."""
        config = DataFreshnessConfig(
            crypto_fresh_seconds=10,
            crypto_stale_seconds=30,
            crypto_expired_seconds=60,
        )
        return DataFreshnessMonitor(config)

    def test_unknown_symbol(self, monitor):
        """Test unknown symbol returns UNKNOWN status."""
        result = monitor.check_freshness("UNKNOWN/SYMBOL")
        assert result.status == FreshnessStatus.UNKNOWN
        assert result.is_tradeable is False
        assert result.last_update is None

    def test_update_new_symbol(self, monitor):
        """Test updating a new symbol."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        result = monitor.check_freshness("BTC/USDT")
        assert result.status == FreshnessStatus.FRESH
        assert result.is_tradeable is True
        assert result.age_seconds < 1

    def test_update_existing_symbol(self, monitor):
        """Test updating an existing symbol."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("BTC/USDT", 51000.0, "crypto")
        result = monitor.check_freshness("BTC/USDT")
        assert result.status == FreshnessStatus.FRESH
        assert monitor._symbol_data["BTC/USDT"].update_count == 1

    def test_fresh_data(self, monitor):
        """Test fresh data detection."""
        monitor.update("ETH/USDT", 3000.0, "crypto")
        result = monitor.check_freshness("ETH/USDT")
        assert result.status == FreshnessStatus.FRESH
        assert "fresh" in result.message.lower()
        assert result.is_tradeable is True

    def test_stale_data(self, custom_monitor):
        """Test stale data detection."""
        # Add data and manipulate timestamp
        custom_monitor.update("SOL/USDT", 100.0, "crypto")
        # Manually set older timestamp
        custom_monitor._symbol_data["SOL/USDT"].last_update = datetime.now() - timedelta(seconds=20)

        result = custom_monitor.check_freshness("SOL/USDT")
        assert result.status == FreshnessStatus.STALE
        assert result.is_tradeable is True

    def test_expired_data(self, custom_monitor):
        """Test expired data detection."""
        custom_monitor.update("AVAX/USDT", 50.0, "crypto")
        # Set timestamp beyond expired threshold
        custom_monitor._symbol_data["AVAX/USDT"].last_update = datetime.now() - timedelta(seconds=120)

        result = custom_monitor.check_freshness("AVAX/USDT")
        assert result.status == FreshnessStatus.EXPIRED
        assert result.is_tradeable is False

    def test_stock_market_thresholds(self, monitor):
        """Test stock market uses different thresholds."""
        monitor.update("AAPL", 150.0, "stock")
        # Set timestamp to 90 seconds old - should be FRESH for stocks (threshold 120)
        monitor._symbol_data["AAPL"].last_update = datetime.now() - timedelta(seconds=90)

        result = monitor.check_freshness("AAPL")
        assert result.status == FreshnessStatus.FRESH
        assert result.is_tradeable is True

    def test_commodity_market_thresholds(self, monitor):
        """Test commodity market uses different thresholds."""
        monitor.update("GOLD", 2000.0, "commodity")
        result = monitor.check_freshness("GOLD")
        assert result.status == FreshnessStatus.FRESH

    def test_check_all(self, monitor):
        """Test check_all returns all symbols."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("ETH/USDT", 3000.0, "crypto")
        monitor.update("AAPL", 150.0, "stock")

        results = monitor.check_all()
        assert len(results) == 3
        assert "BTC/USDT" in results
        assert "ETH/USDT" in results
        assert "AAPL" in results

    def test_is_tradeable(self, monitor):
        """Test is_tradeable convenience method."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        assert monitor.is_tradeable("BTC/USDT") is True
        assert monitor.is_tradeable("UNKNOWN") is False

    def test_get_tradeable_symbols(self, monitor):
        """Test get_tradeable_symbols."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("ETH/USDT", 3000.0, "crypto")

        tradeable = monitor.get_tradeable_symbols()
        assert "BTC/USDT" in tradeable
        assert "ETH/USDT" in tradeable

    def test_get_stale_symbols(self, custom_monitor):
        """Test get_stale_symbols."""
        custom_monitor.update("BTC/USDT", 50000.0, "crypto")  # Fresh
        custom_monitor.update("OLD/USDT", 100.0, "crypto")
        custom_monitor._symbol_data["OLD/USDT"].last_update = datetime.now() - timedelta(seconds=25)

        stale = custom_monitor.get_stale_symbols()
        assert "OLD/USDT" in stale

    def test_get_summary(self, monitor):
        """Test get_summary returns correct structure."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("ETH/USDT", 3000.0, "crypto")

        summary = monitor.get_summary()
        assert "total_symbols" in summary
        assert "fresh" in summary
        assert "stale" in summary
        assert "expired" in summary
        assert "average_age_seconds" in summary
        assert "symbols" in summary
        assert summary["total_symbols"] == 2

    def test_reset_symbol(self, monitor):
        """Test reset clears specific symbol."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("ETH/USDT", 3000.0, "crypto")

        monitor.reset("BTC/USDT")

        result_btc = monitor.check_freshness("BTC/USDT")
        result_eth = monitor.check_freshness("ETH/USDT")

        assert result_btc.status == FreshnessStatus.UNKNOWN
        assert result_eth.status == FreshnessStatus.FRESH

    def test_reset_all(self, monitor):
        """Test reset clears all symbols."""
        monitor.update("BTC/USDT", 50000.0, "crypto")
        monitor.update("ETH/USDT", 3000.0, "crypto")

        monitor.reset()

        assert len(monitor._symbol_data) == 0
        assert len(monitor._alerts_sent) == 0


class TestGlobalMonitor:
    """Test global monitor functions."""

    def test_get_monitor_creates_instance(self):
        """Test get_monitor creates instance."""
        # Reset global
        import bot.data_freshness as df
        df._monitor = None

        monitor = get_monitor()
        assert monitor is not None
        assert isinstance(monitor, DataFreshnessMonitor)

    def test_get_monitor_returns_same_instance(self):
        """Test get_monitor returns same instance."""
        import bot.data_freshness as df
        df._monitor = None

        monitor1 = get_monitor()
        monitor2 = get_monitor()
        assert monitor1 is monitor2

    def test_update_data_convenience(self):
        """Test update_data convenience function."""
        import bot.data_freshness as df
        df._monitor = None

        update_data("TEST/USDT", 100.0, "crypto")
        assert is_data_fresh("TEST/USDT") is True

    def test_is_data_fresh_convenience(self):
        """Test is_data_fresh convenience function."""
        import bot.data_freshness as df
        df._monitor = None

        assert is_data_fresh("NONEXISTENT") is False


class TestAlertCooldown:
    """Test alert cooldown mechanism."""

    def test_alert_cooldown(self):
        """Test alerts respect cooldown period."""
        config = DataFreshnessConfig(
            crypto_fresh_seconds=10,
            crypto_stale_seconds=30,
            crypto_expired_seconds=60,
        )
        monitor = DataFreshnessMonitor(config)

        monitor.update("BTC/USDT", 50000.0, "crypto")
        # Set timestamp beyond expired threshold (60s)
        monitor._symbol_data["BTC/USDT"].last_update = datetime.now() - timedelta(seconds=120)

        # First check triggers alert (age > expired_threshold)
        result = monitor.check_freshness("BTC/USDT")
        assert result.status == FreshnessStatus.EXPIRED
        assert "BTC/USDT" in monitor._alerts_sent

        first_alert_time = monitor._alerts_sent["BTC/USDT"]

        # Second check within cooldown shouldn't update alert time
        monitor.check_freshness("BTC/USDT")
        assert monitor._alerts_sent["BTC/USDT"] == first_alert_time
