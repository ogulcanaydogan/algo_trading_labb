"""
Trading System Monitoring and Alerting.

Provides comprehensive monitoring for:
- System health (CPU, memory, API latency)
- Trading metrics (P&L, positions, risk levels)
- Error tracking and alerting
- Performance dashboards
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import threading

import psutil

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Available alert channels."""

    LOG = "log"
    FILE = "file"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    EMAIL = "email"


@dataclass
class Alert:
    """Alert data structure."""

    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "acknowledged": self.acknowledged,
        }


@dataclass
class MetricSnapshot:
    """Point-in-time system metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    api_latency_ms: Optional[float] = None
    active_positions: int = 0
    daily_pnl: float = 0.0
    total_trades: int = 0
    error_count: int = 0
    warnings: int = 0


@dataclass
class TradingMetrics:
    """Trading performance metrics."""

    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    active_positions: int = 0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    risk_level: str = "normal"  # low, normal, high, critical

    def to_dict(self) -> Dict:
        return {
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "active_positions": self.active_positions,
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "current_drawdown": round(self.current_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "risk_level": self.risk_level,
        }


class AlertManager:
    """
    Manages alerts and notifications.

    Features:
    - Multi-channel alerts (log, file, telegram, webhook)
    - Alert throttling to prevent spam
    - Alert history and acknowledgment
    - Configurable thresholds
    """

    def __init__(
        self,
        channels: Optional[List[AlertChannel]] = None,
        throttle_seconds: int = 300,  # 5 minutes default
        alert_history_size: int = 1000,
    ):
        self.channels = channels or [AlertChannel.LOG, AlertChannel.FILE]
        self.throttle_seconds = throttle_seconds
        self.alert_history: deque = deque(maxlen=alert_history_size)
        self._last_alerts: Dict[str, datetime] = {}
        self._handlers: Dict[AlertChannel, Callable] = {}
        self._lock = threading.Lock()

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default alert handlers."""
        self._handlers[AlertChannel.LOG] = self._handle_log
        self._handlers[AlertChannel.FILE] = self._handle_file

    def register_handler(
        self,
        channel: AlertChannel,
        handler: Callable[[Alert], None],
    ) -> None:
        """Register a custom alert handler."""
        self._handlers[channel] = handler

    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through configured channels.

        Returns:
            True if alert was sent, False if throttled
        """
        with self._lock:
            # Check throttling
            alert_key = f"{alert.source}:{alert.title}"
            if alert_key in self._last_alerts:
                elapsed = (datetime.now() - self._last_alerts[alert_key]).total_seconds()
                if elapsed < self.throttle_seconds:
                    logger.debug(f"Alert throttled: {alert_key}")
                    return False

            # Update last alert time
            self._last_alerts[alert_key] = datetime.now()

            # Store in history
            self.alert_history.append(alert)

            # Send through all channels
            for channel in self.channels:
                if channel in self._handlers:
                    try:
                        self._handlers[channel](alert)
                    except Exception as e:
                        logger.error(f"Failed to send alert via {channel}: {e}")

            return True

    def _handle_log(self, alert: Alert) -> None:
        """Handle alert via logging."""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }
        log_level = level_map.get(alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.source}] {alert.title}: {alert.message}")

    def _handle_file(self, alert: Alert) -> None:
        """Handle alert by writing to file."""
        alert_dir = Path("data/logs/alerts")
        alert_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        alert_file = alert_dir / f"alerts_{date_str}.jsonl"

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def get_recent_alerts(
        self,
        limit: int = 50,
        level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get recent alerts, optionally filtered by level."""
        with self._lock:
            alerts = list(self.alert_history)

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts[-limit:]

    def acknowledge_alert(self, index: int) -> bool:
        """Acknowledge an alert by index."""
        try:
            self.alert_history[index].acknowledged = True
            return True
        except IndexError:
            return False


class SystemMonitor:
    """
    Monitors system health and resources.

    Tracks:
    - CPU usage
    - Memory usage
    - Disk usage
    - API latency
    - Process health
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        disk_threshold: float = 90.0,
    ):
        self.alert_manager = alert_manager or AlertManager()
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

        self._metrics_history: deque = deque(maxlen=1000)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def get_system_metrics(self) -> MetricSnapshot:
        """Get current system metrics."""
        return MetricSnapshot(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage("/").percent,
        )

    def check_health(self) -> Dict[str, Any]:
        """Check system health and return status."""
        metrics = self.get_system_metrics()
        issues = []

        # Check CPU
        if metrics.cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.WARNING,
                    title="High CPU Usage",
                    message=f"CPU usage at {metrics.cpu_percent:.1f}%",
                    source="system_monitor",
                    data={"cpu_percent": metrics.cpu_percent},
                )
            )

        # Check memory
        if metrics.memory_percent > self.memory_threshold:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.WARNING,
                    title="High Memory Usage",
                    message=f"Memory usage at {metrics.memory_percent:.1f}%",
                    source="system_monitor",
                    data={"memory_percent": metrics.memory_percent},
                )
            )

        # Check disk
        if metrics.disk_percent > self.disk_threshold:
            issues.append(f"Low disk space: {metrics.disk_percent:.1f}% used")
            self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.ERROR,
                    title="Low Disk Space",
                    message=f"Disk usage at {metrics.disk_percent:.1f}%",
                    source="system_monitor",
                    data={"disk_percent": metrics.disk_percent},
                )
            )

        return {
            "healthy": len(issues) == 0,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
            },
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
        }

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def _monitoring_loop(self, interval: int) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                metrics = self.get_system_metrics()
                self._metrics_history.append(metrics)
                self.check_health()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            time.sleep(interval)


class TradingMonitor:
    """
    Monitors trading performance and risk.

    Tracks:
    - P&L (daily, total)
    - Position exposure
    - Drawdown levels
    - Risk metrics
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_position_size: float = 0.3,
    ):
        self.alert_manager = alert_manager or AlertManager()
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size = max_position_size

        self.metrics = TradingMetrics()
        self._peak_equity = 0.0
        self._daily_start_equity = 0.0
        self._last_reset_date: Optional[datetime] = None

    def update_metrics(
        self,
        current_equity: float,
        positions: List[Dict],
        trades: List[Dict],
    ) -> TradingMetrics:
        """Update trading metrics."""
        now = datetime.now()

        # Reset daily metrics at midnight
        if self._last_reset_date is None or self._last_reset_date.date() != now.date():
            self._daily_start_equity = current_equity
            self._last_reset_date = now

        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Calculate metrics
        self.metrics.daily_pnl = current_equity - self._daily_start_equity
        self.metrics.total_pnl = current_equity - 10000  # Assuming 10k initial

        # Drawdown
        if self._peak_equity > 0:
            self.metrics.current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
            self.metrics.max_drawdown = max(
                self.metrics.max_drawdown, self.metrics.current_drawdown
            )

        # Position metrics
        self.metrics.active_positions = len([p for p in positions if p.get("status") == "open"])

        # Trade metrics
        closed_trades = [t for t in trades if t.get("status") == "closed"]
        self.metrics.total_trades = len(closed_trades)
        self.metrics.winning_trades = len([t for t in closed_trades if t.get("pnl", 0) > 0])
        self.metrics.losing_trades = len([t for t in closed_trades if t.get("pnl", 0) < 0])

        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        # Determine risk level
        self._update_risk_level()

        # Check for alerts
        self._check_alerts(current_equity)

        return self.metrics

    def _update_risk_level(self) -> None:
        """Update the current risk level."""
        if self.metrics.current_drawdown > self.max_drawdown_pct:
            self.metrics.risk_level = "critical"
        elif self.metrics.current_drawdown > self.max_drawdown_pct * 0.75:
            self.metrics.risk_level = "high"
        elif self.metrics.current_drawdown > self.max_drawdown_pct * 0.5:
            self.metrics.risk_level = "normal"
        else:
            self.metrics.risk_level = "low"

    def _check_alerts(self, current_equity: float) -> None:
        """Check for trading alerts."""
        # Daily loss alert
        if self._daily_start_equity > 0:
            daily_loss_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity
            if daily_loss_pct > self.max_daily_loss_pct:
                self.alert_manager.send_alert(
                    Alert(
                        level=AlertLevel.CRITICAL,
                        title="Daily Loss Limit Exceeded",
                        message=f"Daily loss of {daily_loss_pct * 100:.2f}% exceeds limit of {self.max_daily_loss_pct * 100:.1f}%",
                        source="trading_monitor",
                        data={"daily_loss_pct": daily_loss_pct},
                    )
                )

        # Drawdown alert
        if self.metrics.current_drawdown > self.max_drawdown_pct:
            self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.CRITICAL,
                    title="Maximum Drawdown Exceeded",
                    message=f"Drawdown of {self.metrics.current_drawdown * 100:.2f}% exceeds limit of {self.max_drawdown_pct * 100:.1f}%",
                    source="trading_monitor",
                    data={"drawdown": self.metrics.current_drawdown},
                )
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics."""
        return self.metrics.to_dict()


class MonitoringService:
    """
    Unified monitoring service that combines system and trading monitoring.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}

        # Initialize alert manager
        channels = [AlertChannel.LOG, AlertChannel.FILE]
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            channels.append(AlertChannel.TELEGRAM)

        self.alert_manager = AlertManager(
            channels=channels,
            throttle_seconds=config.get("alert_throttle", 300),
        )

        # Initialize monitors
        self.system_monitor = SystemMonitor(
            alert_manager=self.alert_manager,
            cpu_threshold=config.get("cpu_threshold", 80),
            memory_threshold=config.get("memory_threshold", 80),
            disk_threshold=config.get("disk_threshold", 90),
        )

        self.trading_monitor = TradingMonitor(
            alert_manager=self.alert_manager,
            max_daily_loss_pct=config.get("max_daily_loss_pct", 0.05),
            max_drawdown_pct=config.get("max_drawdown_pct", 0.15),
        )

        self._running = False

    def start(self) -> None:
        """Start all monitoring services."""
        self.system_monitor.start_monitoring()
        self._running = True
        logger.info("Monitoring service started")

        # Send startup alert
        self.alert_manager.send_alert(
            Alert(
                level=AlertLevel.INFO,
                title="Monitoring Service Started",
                message="All monitoring services are now active",
                source="monitoring_service",
            )
        )

    def stop(self) -> None:
        """Stop all monitoring services."""
        self.system_monitor.stop_monitoring()
        self._running = False
        logger.info("Monitoring service stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        system_health = self.system_monitor.check_health()

        return {
            "running": self._running,
            "timestamp": datetime.now().isoformat(),
            "system": system_health,
            "trading": self.trading_monitor.get_metrics(),
            "recent_alerts": [a.to_dict() for a in self.alert_manager.get_recent_alerts(10)],
        }

    def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        source: str = "manual",
        data: Optional[Dict] = None,
    ) -> bool:
        """Send a custom alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            data=data or {},
        )
        return self.alert_manager.send_alert(alert)


# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create monitoring service
    service = MonitoringService()

    # Start monitoring
    service.start()

    # Check status
    status = service.get_status()
    print(json.dumps(status, indent=2, default=str))

    # Send test alert
    service.send_alert(
        title="Test Alert",
        message="This is a test alert",
        level=AlertLevel.INFO,
    )

    # Get recent alerts
    alerts = service.alert_manager.get_recent_alerts()
    print(f"\nRecent alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert.level.value}] {alert.title}: {alert.message}")

    # Stop monitoring
    service.stop()
