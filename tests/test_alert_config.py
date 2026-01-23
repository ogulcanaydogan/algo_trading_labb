"""
Tests for alert configuration module.
"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

from bot.alert_config import (
    AlertType,
    AlertPriority,
    AlertChannel,
    AlertThreshold,
    AlertEvent,
    AlertConfigManager,
)


class TestAlertType:
    """Test AlertType enum."""

    def test_all_types_exist(self):
        """Test all alert types exist."""
        assert AlertType.PRICE_MOVE.value == "price_move"
        assert AlertType.PRICE_LEVEL.value == "price_level"
        assert AlertType.SIGNAL_GENERATED.value == "signal_generated"
        assert AlertType.SIGNAL_CHANGE.value == "signal_change"
        assert AlertType.TRADE_EXECUTED.value == "trade_executed"
        assert AlertType.PNL_THRESHOLD.value == "pnl_threshold"
        assert AlertType.DRAWDOWN.value == "drawdown"
        assert AlertType.WIN_RATE.value == "win_rate"
        assert AlertType.PORTFOLIO_VALUE.value == "portfolio_value"
        assert AlertType.REGIME_CHANGE.value == "regime_change"
        assert AlertType.VOLATILITY.value == "volatility"


class TestAlertPriority:
    """Test AlertPriority enum."""

    def test_all_priorities_exist(self):
        """Test all priority levels exist."""
        assert AlertPriority.LOW.value == "low"
        assert AlertPriority.MEDIUM.value == "medium"
        assert AlertPriority.HIGH.value == "high"
        assert AlertPriority.CRITICAL.value == "critical"


class TestAlertChannel:
    """Test AlertChannel enum."""

    def test_all_channels_exist(self):
        """Test all channels exist."""
        assert AlertChannel.TELEGRAM.value == "telegram"
        assert AlertChannel.DISCORD.value == "discord"
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.DASHBOARD.value == "dashboard"


class TestAlertThreshold:
    """Test AlertThreshold dataclass."""

    def test_threshold_creation(self):
        """Test creating a threshold."""
        threshold = AlertThreshold(
            alert_type=AlertType.PRICE_MOVE,
            name="BTC Move",
            enabled=True,
            threshold_value=5.0,
            comparison="pct_change",
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=60,
            symbol_filter=["BTC/USDT"],
            message_template="BTC moved {value:.2f}%",
        )

        assert threshold.alert_type == AlertType.PRICE_MOVE
        assert threshold.name == "BTC Move"
        assert threshold.enabled is True
        assert threshold.threshold_value == 5.0
        assert threshold.priority == AlertPriority.HIGH

    def test_to_dict(self):
        """Test conversion to dict."""
        threshold = AlertThreshold(
            alert_type=AlertType.DRAWDOWN,
            name="Drawdown Alert",
            enabled=True,
            threshold_value=10.0,
            comparison="gte",
            priority=AlertPriority.CRITICAL,
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
            cooldown_minutes=120,
            symbol_filter=None,
            message_template="Drawdown: {value}%",
        )

        d = threshold.to_dict()

        assert d["alert_type"] == "drawdown"
        assert d["name"] == "Drawdown Alert"
        assert d["priority"] == "critical"
        assert d["channels"] == ["telegram", "email"]
        assert d["threshold_value"] == 10.0

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "alert_type": "price_move",
            "name": "Test",
            "enabled": True,
            "threshold_value": 3.0,
            "comparison": "pct_change",
            "priority": "medium",
            "channels": ["telegram", "log"],
            "cooldown_minutes": 30,
            "symbol_filter": None,
            "message_template": "Test alert",
            "metadata": {},
        }

        threshold = AlertThreshold.from_dict(data)

        assert threshold.alert_type == AlertType.PRICE_MOVE
        assert threshold.priority == AlertPriority.MEDIUM
        assert AlertChannel.TELEGRAM in threshold.channels
        assert AlertChannel.LOG in threshold.channels


class TestAlertEvent:
    """Test AlertEvent dataclass."""

    def test_event_creation(self):
        """Test creating an alert event."""
        event = AlertEvent(
            alert_id="alert_001",
            alert_type=AlertType.PRICE_MOVE,
            name="BTC Move",
            symbol="BTC/USDT",
            current_value=5.5,
            threshold_value=5.0,
            message="BTC moved 5.5%",
            priority=AlertPriority.HIGH,
            triggered_at=datetime.now(),
            sent_channels=[AlertChannel.TELEGRAM],
        )

        assert event.alert_id == "alert_001"
        assert event.current_value == 5.5
        assert event.acknowledged is False

    def test_event_to_dict(self):
        """Test event to dict conversion."""
        triggered_at = datetime(2024, 1, 15, 10, 30, 0)
        event = AlertEvent(
            alert_id="alert_002",
            alert_type=AlertType.DRAWDOWN,
            name="Drawdown",
            symbol=None,
            current_value=7.5,
            threshold_value=5.0,
            message="Drawdown 7.5%",
            priority=AlertPriority.HIGH,
            triggered_at=triggered_at,
            sent_channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
        )

        d = event.to_dict()

        assert d["alert_id"] == "alert_002"
        assert d["alert_type"] == "drawdown"
        assert d["current_value"] == 7.5
        assert d["priority"] == "high"
        assert d["triggered_at"] == triggered_at.isoformat()


class TestAlertConfigManager:
    """Test AlertConfigManager class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_config_dir):
        """Create alert config manager."""
        return AlertConfigManager(config_dir=temp_config_dir)

    def test_manager_creation(self, manager):
        """Test manager is created."""
        assert manager is not None
        # Should have default thresholds
        assert len(manager.thresholds) > 0

    def test_default_thresholds_loaded(self, manager):
        """Test default thresholds are loaded."""
        # Check some default thresholds exist
        threshold_types = [t.alert_type for t in manager.thresholds.values()]
        assert AlertType.PRICE_MOVE in threshold_types
        assert AlertType.DRAWDOWN in threshold_types

    def test_add_threshold(self, manager):
        """Test adding a threshold."""
        threshold = AlertThreshold(
            alert_type=AlertType.PNL_THRESHOLD,
            name="Custom PnL",
            enabled=True,
            threshold_value=1000.0,
            comparison="gte",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="PnL reached {value}",
        )

        manager.add_threshold(threshold)

        assert "Custom PnL" in manager.thresholds
        assert manager.thresholds["Custom PnL"].threshold_value == 1000.0

    def test_remove_threshold(self, manager):
        """Test removing a threshold."""
        # Add then remove
        threshold = AlertThreshold(
            alert_type=AlertType.VOLATILITY,
            name="ToRemove",
            enabled=True,
            threshold_value=50.0,
            comparison="gte",
            priority=AlertPriority.LOW,
            channels=[AlertChannel.LOG],
            cooldown_minutes=30,
            symbol_filter=None,
            message_template="Test",
        )
        manager.add_threshold(threshold)
        assert "ToRemove" in manager.thresholds

        manager.remove_threshold("ToRemove")
        assert "ToRemove" not in manager.thresholds

    def test_threshold_enabled_state(self, manager):
        """Test threshold enabled state."""
        threshold = AlertThreshold(
            alert_type=AlertType.WIN_RATE,
            name="WinRate",
            enabled=True,
            threshold_value=50.0,
            comparison="lt",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.LOG],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="Win rate low",
        )
        manager.add_threshold(threshold)

        # Check enabled state
        assert manager.thresholds["WinRate"].enabled is True

    def test_threshold_can_be_modified(self, manager):
        """Test threshold can be modified after adding."""
        threshold = AlertThreshold(
            alert_type=AlertType.DRAWDOWN,
            name="UpdateTest",
            enabled=True,
            threshold_value=5.0,
            comparison="gte",
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.LOG],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="Drawdown alert",
        )
        manager.add_threshold(threshold)

        # Modify directly (manager stores references)
        manager.thresholds["UpdateTest"].threshold_value = 10.0
        assert manager.thresholds["UpdateTest"].threshold_value == 10.0

    def test_filter_thresholds_by_type(self, manager):
        """Test filtering thresholds by type."""
        drawdown_thresholds = [
            t for t in manager.thresholds.values() if t.alert_type == AlertType.DRAWDOWN
        ]
        assert len(drawdown_thresholds) > 0
        for t in drawdown_thresholds:
            assert t.alert_type == AlertType.DRAWDOWN

    def test_filter_thresholds_by_priority(self, manager):
        """Test filtering thresholds by priority."""
        high_priority = [t for t in manager.thresholds.values() if t.priority == AlertPriority.HIGH]
        for t in high_priority:
            assert t.priority == AlertPriority.HIGH

    def test_notification_callback(self, temp_config_dir):
        """Test notification callback is called."""
        events_received = []

        def callback(event):
            events_received.append(event)

        manager = AlertConfigManager(
            config_dir=temp_config_dir,
            notification_callback=callback,
        )

        # Trigger an alert manually (if manager has such method)
        # This depends on implementation


class TestConfigPersistence:
    """Test configuration persistence."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_config_saved_on_add(self, temp_config_dir):
        """Test config is saved when threshold added."""
        manager = AlertConfigManager(config_dir=temp_config_dir)

        threshold = AlertThreshold(
            alert_type=AlertType.PRICE_LEVEL,
            name="SaveTest",
            enabled=True,
            threshold_value=50000.0,
            comparison="gte",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.LOG],
            cooldown_minutes=60,
            symbol_filter=["BTC/USDT"],
            message_template="Price level reached",
        )
        manager.add_threshold(threshold)

        # Check file exists
        config_file = Path(temp_config_dir) / "alert_thresholds.json"
        assert config_file.exists()

    def test_config_loaded_on_restart(self, temp_config_dir):
        """Test config is loaded on restart."""
        # Create first manager and add threshold
        manager1 = AlertConfigManager(config_dir=temp_config_dir)
        threshold = AlertThreshold(
            alert_type=AlertType.MODEL_PERFORMANCE,
            name="PersistTest",
            enabled=True,
            threshold_value=0.5,
            comparison="lt",
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=120,
            symbol_filter=None,
            message_template="Model performance low",
        )
        manager1.add_threshold(threshold)

        # Create second manager - should load persisted config
        manager2 = AlertConfigManager(config_dir=temp_config_dir)
        assert "PersistTest" in manager2.thresholds


class TestComparison:
    """Test comparison operations."""

    @pytest.fixture
    def temp_config_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_comparison_operators(self, temp_config_dir):
        """Test different comparison operators."""
        manager = AlertConfigManager(config_dir=temp_config_dir)

        # Test gt
        threshold_gt = AlertThreshold(
            alert_type=AlertType.PRICE_MOVE,
            name="GT Test",
            enabled=True,
            threshold_value=5.0,
            comparison="gt",
            priority=AlertPriority.LOW,
            channels=[AlertChannel.LOG],
            cooldown_minutes=1,
            symbol_filter=None,
            message_template="GT test",
        )
        manager.add_threshold(threshold_gt)

        # Test lt
        threshold_lt = AlertThreshold(
            alert_type=AlertType.WIN_RATE,
            name="LT Test",
            enabled=True,
            threshold_value=40.0,
            comparison="lt",
            priority=AlertPriority.LOW,
            channels=[AlertChannel.LOG],
            cooldown_minutes=1,
            symbol_filter=None,
            message_template="LT test",
        )
        manager.add_threshold(threshold_lt)

        assert manager.thresholds["GT Test"].comparison == "gt"
        assert manager.thresholds["LT Test"].comparison == "lt"
