"""
Tests for Trading System Monitoring and Alerting module.
"""

import pytest
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from bot.monitoring import (
    AlertLevel,
    AlertChannel,
    Alert,
    MetricSnapshot,
    TradingMetrics,
    AlertManager,
    SystemMonitor,
    TradingMonitor,
    MonitoringService,
)


class TestAlertLevel:
    """Test AlertLevel enum."""

    def test_alert_level_values(self):
        """Test alert level enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestAlertChannel:
    """Test AlertChannel enum."""

    def test_alert_channel_values(self):
        """Test alert channel enum values."""
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.FILE.value == "file"
        assert AlertChannel.TELEGRAM.value == "telegram"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.EMAIL.value == "email"


class TestAlert:
    """Test Alert dataclass."""

    def test_basic_alert(self):
        """Test creating a basic alert."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test message",
            source="test_module",
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test message"
        assert alert.source == "test_module"
        assert alert.acknowledged is False

    def test_alert_with_data(self):
        """Test alert with additional data."""
        alert = Alert(
            level=AlertLevel.ERROR,
            title="Error Alert",
            message="An error occurred",
            source="error_handler",
            data={"error_code": 500, "details": "Internal error"},
        )

        assert alert.data["error_code"] == 500
        assert "details" in alert.data

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test",
            message="Test",
            source="test",
        )

        assert alert.timestamp is not None
        assert isinstance(alert.timestamp, datetime)

    def test_to_dict(self):
        """Test alert to_dict conversion."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Critical Alert",
            message="System failure",
            source="system",
            data={"cpu": 99.5},
        )

        d = alert.to_dict()

        assert d["level"] == "critical"
        assert d["title"] == "Critical Alert"
        assert d["message"] == "System failure"
        assert d["source"] == "system"
        assert d["acknowledged"] is False
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)
        assert d["data"]["cpu"] == 99.5


class TestMetricSnapshot:
    """Test MetricSnapshot dataclass."""

    def test_metric_snapshot(self):
        """Test creating metric snapshot."""
        now = datetime.now()
        snapshot = MetricSnapshot(
            timestamp=now,
            cpu_percent=45.5,
            memory_percent=60.0,
            disk_percent=75.0,
            api_latency_ms=150.0,
            active_positions=3,
            daily_pnl=500.0,
            total_trades=10,
            error_count=0,
            warnings=2,
        )

        assert snapshot.timestamp == now
        assert snapshot.cpu_percent == 45.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.disk_percent == 75.0
        assert snapshot.api_latency_ms == 150.0
        assert snapshot.active_positions == 3

    def test_metric_defaults(self):
        """Test metric snapshot defaults."""
        now = datetime.now()
        snapshot = MetricSnapshot(
            timestamp=now,
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_percent=50.0,
        )

        assert snapshot.api_latency_ms is None
        assert snapshot.active_positions == 0
        assert snapshot.daily_pnl == 0.0
        assert snapshot.error_count == 0


class TestTradingMetrics:
    """Test TradingMetrics dataclass."""

    def test_trading_metrics_defaults(self):
        """Test trading metrics defaults."""
        metrics = TradingMetrics()

        assert metrics.total_pnl == 0.0
        assert metrics.daily_pnl == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
        assert metrics.risk_level == "normal"

    def test_trading_metrics_custom(self):
        """Test custom trading metrics."""
        metrics = TradingMetrics(
            total_pnl=5000.0,
            daily_pnl=250.0,
            win_rate=0.65,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            active_positions=3,
            max_drawdown=0.12,
            current_drawdown=0.05,
            sharpe_ratio=1.8,
            risk_level="low",
        )

        assert metrics.total_pnl == 5000.0
        assert metrics.win_rate == 0.65
        assert metrics.risk_level == "low"

    def test_to_dict(self):
        """Test trading metrics to_dict."""
        metrics = TradingMetrics(
            total_pnl=1234.56,
            daily_pnl=100.123,
            win_rate=0.654321,
            max_drawdown=0.123456,
            current_drawdown=0.05,
            sharpe_ratio=1.567,
        )

        d = metrics.to_dict()

        assert d["total_pnl"] == 1234.56
        assert d["daily_pnl"] == 100.12  # Rounded to 2 decimals
        assert d["win_rate"] == 65.4  # win_rate * 100, rounded to 1 decimal
        assert d["max_drawdown"] == 12.35  # * 100, rounded to 2 decimals
        assert d["sharpe_ratio"] == 1.57  # Rounded to 2 decimals


class TestAlertManager:
    """Test AlertManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return AlertManager(
            channels=[AlertChannel.LOG],
            throttle_seconds=1,  # Short for testing
            alert_history_size=100,
        )

    def test_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert AlertChannel.LOG in alert_manager.channels
        assert alert_manager.throttle_seconds == 1
        assert len(alert_manager.alert_history) == 0

    def test_default_channels(self):
        """Test default channels."""
        manager = AlertManager()
        assert AlertChannel.LOG in manager.channels
        assert AlertChannel.FILE in manager.channels

    def test_send_alert(self, alert_manager):
        """Test sending an alert."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            source="test",
        )

        result = alert_manager.send_alert(alert)

        assert result is True
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0].title == "Test Alert"

    def test_alert_throttling(self, alert_manager):
        """Test alert throttling."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Throttle Test",
            message="Message",
            source="test",
        )

        # First alert should succeed
        result1 = alert_manager.send_alert(alert)
        assert result1 is True

        # Second identical alert should be throttled
        result2 = alert_manager.send_alert(alert)
        assert result2 is False

        # Only one alert in history
        assert len(alert_manager.alert_history) == 1

    def test_throttle_expires(self, alert_manager):
        """Test throttle expires after timeout."""
        alert = Alert(
            level=AlertLevel.INFO,
            title="Expire Test",
            message="Message",
            source="test",
        )

        alert_manager.send_alert(alert)
        time.sleep(1.5)  # Wait for throttle to expire

        # Should now succeed
        result = alert_manager.send_alert(alert)
        assert result is True
        assert len(alert_manager.alert_history) == 2

    def test_different_alerts_not_throttled(self, alert_manager):
        """Test different alerts are not throttled."""
        alert1 = Alert(
            level=AlertLevel.INFO,
            title="Alert 1",
            message="Message 1",
            source="test",
        )
        alert2 = Alert(
            level=AlertLevel.INFO,
            title="Alert 2",
            message="Message 2",
            source="test",
        )

        result1 = alert_manager.send_alert(alert1)
        result2 = alert_manager.send_alert(alert2)

        assert result1 is True
        assert result2 is True
        assert len(alert_manager.alert_history) == 2

    def test_register_custom_handler(self, alert_manager):
        """Test registering custom handler."""
        received_alerts = []

        def custom_handler(alert):
            received_alerts.append(alert)

        alert_manager.channels.append(AlertChannel.WEBHOOK)
        alert_manager.register_handler(AlertChannel.WEBHOOK, custom_handler)

        alert = Alert(
            level=AlertLevel.INFO,
            title="Custom Test",
            message="Message",
            source="test",
        )
        alert_manager.send_alert(alert)

        assert len(received_alerts) == 1
        assert received_alerts[0].title == "Custom Test"

    def test_get_recent_alerts(self, alert_manager):
        """Test getting recent alerts."""
        for i in range(10):
            alert = Alert(
                level=AlertLevel.INFO,
                title=f"Alert {i}",
                message="Message",
                source="test",
            )
            alert_manager.alert_history.append(alert)

        recent = alert_manager.get_recent_alerts(limit=5)

        assert len(recent) == 5
        # Should be the last 5
        assert recent[0].title == "Alert 5"
        assert recent[4].title == "Alert 9"

    def test_get_recent_alerts_filter_by_level(self, alert_manager):
        """Test filtering alerts by level."""
        for i, level in enumerate([AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR] * 3):
            alert = Alert(
                level=level,
                title=f"Alert {i}",
                message="Message",
                source="test",
            )
            alert_manager.alert_history.append(alert)

        warnings = alert_manager.get_recent_alerts(level=AlertLevel.WARNING)

        assert all(a.level == AlertLevel.WARNING for a in warnings)
        assert len(warnings) == 3

    def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging an alert."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title="To Acknowledge",
            message="Message",
            source="test",
        )
        alert_manager.alert_history.append(alert)

        result = alert_manager.acknowledge_alert(0)

        assert result is True
        assert alert_manager.alert_history[0].acknowledged is True

    def test_acknowledge_invalid_index(self, alert_manager):
        """Test acknowledging invalid index."""
        result = alert_manager.acknowledge_alert(999)
        assert result is False

    def test_handler_exception_handled(self, alert_manager):
        """Test handler exceptions are handled."""
        def bad_handler(alert):
            raise Exception("Handler error")

        alert_manager.channels.append(AlertChannel.EMAIL)
        alert_manager.register_handler(AlertChannel.EMAIL, bad_handler)

        alert = Alert(
            level=AlertLevel.INFO,
            title="Test",
            message="Message",
            source="test",
        )

        # Should not raise
        result = alert_manager.send_alert(alert)
        assert result is True


class TestSystemMonitor:
    """Test SystemMonitor class."""

    @pytest.fixture
    def system_monitor(self):
        """Create system monitor instance."""
        return SystemMonitor(
            cpu_threshold=80.0,
            memory_threshold=80.0,
            disk_threshold=90.0,
        )

    def test_initialization(self, system_monitor):
        """Test system monitor initialization."""
        assert system_monitor.cpu_threshold == 80.0
        assert system_monitor.memory_threshold == 80.0
        assert system_monitor.disk_threshold == 90.0

    @patch("bot.monitoring.psutil")
    def test_get_system_metrics(self, mock_psutil, system_monitor):
        """Test getting system metrics."""
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value.percent = 60.0
        mock_psutil.disk_usage.return_value.percent = 70.0

        metrics = system_monitor.get_system_metrics()

        assert isinstance(metrics, MetricSnapshot)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 70.0

    @patch("bot.monitoring.psutil")
    def test_check_health_healthy(self, mock_psutil, system_monitor):
        """Test health check when healthy."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        health = system_monitor.check_health()

        assert health["healthy"] is True
        assert len(health["issues"]) == 0

    @patch("bot.monitoring.psutil")
    def test_check_health_high_cpu(self, mock_psutil, system_monitor):
        """Test health check with high CPU."""
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        health = system_monitor.check_health()

        assert health["healthy"] is False
        assert len(health["issues"]) == 1
        assert "CPU" in health["issues"][0]

    @patch("bot.monitoring.psutil")
    def test_check_health_high_memory(self, mock_psutil, system_monitor):
        """Test health check with high memory."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 90.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        health = system_monitor.check_health()

        assert health["healthy"] is False
        assert any("memory" in issue.lower() for issue in health["issues"])

    @patch("bot.monitoring.psutil")
    def test_check_health_low_disk(self, mock_psutil, system_monitor):
        """Test health check with low disk space."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 95.0

        health = system_monitor.check_health()

        assert health["healthy"] is False
        assert any("disk" in issue.lower() for issue in health["issues"])

    @patch("bot.monitoring.psutil")
    def test_start_stop_monitoring(self, mock_psutil, system_monitor):
        """Test starting and stopping monitoring."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        system_monitor.start_monitoring(interval_seconds=1)
        assert system_monitor._running is True

        system_monitor.stop_monitoring()
        assert system_monitor._running is False


class TestTradingMonitor:
    """Test TradingMonitor class."""

    @pytest.fixture
    def trading_monitor(self):
        """Create trading monitor instance."""
        return TradingMonitor(
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            max_position_size=0.3,
        )

    def test_initialization(self, trading_monitor):
        """Test trading monitor initialization."""
        assert trading_monitor.max_daily_loss_pct == 0.05
        assert trading_monitor.max_drawdown_pct == 0.15
        assert isinstance(trading_monitor.metrics, TradingMetrics)

    def test_update_metrics_basic(self, trading_monitor):
        """Test basic metrics update."""
        positions = [{"status": "open"}, {"status": "open"}]
        trades = [
            {"status": "closed", "pnl": 100},
            {"status": "closed", "pnl": -50},
        ]

        metrics = trading_monitor.update_metrics(
            current_equity=10500.0,
            positions=positions,
            trades=trades,
        )

        assert metrics.active_positions == 2
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5

    def test_update_metrics_daily_pnl(self, trading_monitor):
        """Test daily PnL calculation."""
        positions = []
        trades = []

        # First call sets daily start
        trading_monitor.update_metrics(10000.0, positions, trades)

        # Second call calculates daily pnl
        metrics = trading_monitor.update_metrics(10500.0, positions, trades)

        assert metrics.daily_pnl == 500.0

    def test_update_metrics_drawdown(self, trading_monitor):
        """Test drawdown calculation."""
        positions = []
        trades = []

        # Set peak
        trading_monitor.update_metrics(12000.0, positions, trades)

        # Drop below peak
        metrics = trading_monitor.update_metrics(10000.0, positions, trades)

        # Drawdown should be (12000 - 10000) / 12000 = 0.1667
        assert metrics.current_drawdown == pytest.approx(0.1667, rel=0.01)

    def test_risk_level_low(self, trading_monitor):
        """Test low risk level."""
        trading_monitor.metrics.current_drawdown = 0.02
        trading_monitor._update_risk_level()

        assert trading_monitor.metrics.risk_level == "low"

    def test_risk_level_normal(self, trading_monitor):
        """Test normal risk level."""
        trading_monitor.metrics.current_drawdown = 0.10
        trading_monitor._update_risk_level()

        assert trading_monitor.metrics.risk_level == "normal"

    def test_risk_level_high(self, trading_monitor):
        """Test high risk level."""
        trading_monitor.metrics.current_drawdown = 0.12
        trading_monitor._update_risk_level()

        assert trading_monitor.metrics.risk_level == "high"

    def test_risk_level_critical(self, trading_monitor):
        """Test critical risk level."""
        trading_monitor.metrics.current_drawdown = 0.20
        trading_monitor._update_risk_level()

        assert trading_monitor.metrics.risk_level == "critical"

    def test_get_metrics(self, trading_monitor):
        """Test getting metrics."""
        trading_monitor.metrics.total_pnl = 1000.0
        trading_monitor.metrics.win_rate = 0.6

        metrics_dict = trading_monitor.get_metrics()

        assert metrics_dict["total_pnl"] == 1000.0
        assert metrics_dict["win_rate"] == 60.0  # Converted to percentage


class TestMonitoringService:
    """Test MonitoringService class."""

    @pytest.fixture
    def service(self):
        """Create monitoring service."""
        return MonitoringService(config={
            "cpu_threshold": 80,
            "memory_threshold": 80,
            "disk_threshold": 90,
            "max_daily_loss_pct": 0.05,
        })

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.alert_manager is not None
        assert service.system_monitor is not None
        assert service.trading_monitor is not None
        assert service._running is False

    @patch("bot.monitoring.psutil")
    def test_start_stop(self, mock_psutil, service):
        """Test starting and stopping service."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        service.start()
        assert service._running is True

        service.stop()
        assert service._running is False

    @patch("bot.monitoring.psutil")
    def test_get_status(self, mock_psutil, service):
        """Test getting service status."""
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 40.0
        mock_psutil.disk_usage.return_value.percent = 50.0

        status = service.get_status()

        assert "running" in status
        assert "timestamp" in status
        assert "system" in status
        assert "trading" in status
        assert "recent_alerts" in status

    def test_send_alert(self, service):
        """Test sending custom alert."""
        result = service.send_alert(
            title="Test Alert",
            message="Test message",
            level=AlertLevel.WARNING,
            source="test",
            data={"key": "value"},
        )

        assert result is True

        # Check alert was stored
        alerts = service.alert_manager.get_recent_alerts(limit=1)
        assert len(alerts) == 1
        assert alerts[0].title == "Test Alert"


class TestAlertManagerFileHandler:
    """Test AlertManager file handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @patch("bot.monitoring.Path")
    def test_file_handler_creates_directory(self, mock_path):
        """Test file handler creates alert directory."""
        manager = AlertManager(channels=[AlertChannel.FILE])

        alert = Alert(
            level=AlertLevel.INFO,
            title="Test",
            message="Message",
            source="test",
        )

        # Directory creation should be called
        manager._handle_file(alert)


class TestTradingMonitorAlerts:
    """Test trading monitor alert generation."""

    @pytest.fixture
    def monitor_with_alerts(self):
        """Create trading monitor with custom alert manager."""
        alert_manager = AlertManager(
            channels=[AlertChannel.LOG],
            throttle_seconds=0,
        )
        return TradingMonitor(
            alert_manager=alert_manager,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
        )

    def test_daily_loss_alert(self, monitor_with_alerts):
        """Test daily loss alert is triggered."""
        # Set up initial equity
        monitor_with_alerts._daily_start_equity = 10000.0

        # Update with significant loss
        monitor_with_alerts._check_alerts(9000.0)  # 10% loss

        alerts = monitor_with_alerts.alert_manager.get_recent_alerts()
        assert len(alerts) > 0
        assert alerts[-1].level == AlertLevel.CRITICAL
        assert "Daily Loss" in alerts[-1].title

    def test_drawdown_alert(self, monitor_with_alerts):
        """Test drawdown alert is triggered."""
        monitor_with_alerts.metrics.current_drawdown = 0.20  # 20% drawdown

        monitor_with_alerts._check_alerts(8000.0)

        alerts = monitor_with_alerts.alert_manager.get_recent_alerts()
        assert any("Drawdown" in a.title for a in alerts)


class TestAlertManagerConcurrency:
    """Test AlertManager thread safety."""

    def test_concurrent_alerts(self):
        """Test sending alerts from multiple threads."""
        import threading

        manager = AlertManager(
            channels=[AlertChannel.LOG],
            throttle_seconds=0,
        )
        results = []

        def send_alert(i):
            alert = Alert(
                level=AlertLevel.INFO,
                title=f"Alert {i}",
                message="Message",
                source="test",
            )
            result = manager.send_alert(alert)
            results.append(result)

        threads = [threading.Thread(target=send_alert, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All alerts should have been processed
        assert len(results) == 10
        assert all(r is True for r in results)


class TestMetricEdgeCases:
    """Test edge cases in metric calculations."""

    def test_trading_metrics_zero_trades(self):
        """Test metrics with zero trades."""
        metrics = TradingMetrics(total_trades=0)
        d = metrics.to_dict()

        assert d["total_trades"] == 0
        assert d["win_rate"] == 0.0

    def test_trading_monitor_no_positions(self):
        """Test monitor with no positions."""
        monitor = TradingMonitor()

        metrics = monitor.update_metrics(
            current_equity=10000.0,
            positions=[],
            trades=[],
        )

        assert metrics.active_positions == 0

    def test_trading_monitor_all_open_positions(self):
        """Test counting only open positions."""
        monitor = TradingMonitor()

        positions = [
            {"status": "open"},
            {"status": "closed"},
            {"status": "open"},
        ]

        metrics = monitor.update_metrics(
            current_equity=10000.0,
            positions=positions,
            trades=[],
        )

        assert metrics.active_positions == 2
