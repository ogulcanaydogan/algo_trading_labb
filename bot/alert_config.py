"""
Alert Threshold Customization.

Configurable alert system for price movements, signals, portfolio,
and risk management notifications.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""

    PRICE_MOVE = "price_move"
    PRICE_LEVEL = "price_level"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CHANGE = "signal_change"
    TRADE_EXECUTED = "trade_executed"
    PNL_THRESHOLD = "pnl_threshold"
    DRAWDOWN = "drawdown"
    WIN_RATE = "win_rate"
    PORTFOLIO_VALUE = "portfolio_value"
    REGIME_CHANGE = "regime_change"
    VOLATILITY = "volatility"
    CORRELATION_SHIFT = "correlation_shift"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_STATUS = "system_status"


class AlertPriority(Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Notification channels."""

    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    DASHBOARD = "dashboard"


@dataclass
class AlertThreshold:
    """Configuration for a single alert threshold."""

    alert_type: AlertType
    name: str
    enabled: bool
    threshold_value: float
    comparison: str  # gt, lt, gte, lte, eq, pct_change
    priority: AlertPriority
    channels: List[AlertChannel]
    cooldown_minutes: int
    symbol_filter: Optional[List[str]]  # None = all symbols
    message_template: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "name": self.name,
            "enabled": self.enabled,
            "threshold_value": self.threshold_value,
            "comparison": self.comparison,
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes,
            "symbol_filter": self.symbol_filter,
            "message_template": self.message_template,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertThreshold":
        data["alert_type"] = AlertType(data["alert_type"])
        data["priority"] = AlertPriority(data["priority"])
        data["channels"] = [AlertChannel(c) for c in data["channels"]]
        return cls(**data)


@dataclass
class AlertEvent:
    """A triggered alert event."""

    alert_id: str
    alert_type: AlertType
    name: str
    symbol: Optional[str]
    current_value: float
    threshold_value: float
    message: str
    priority: AlertPriority
    triggered_at: datetime
    sent_channels: List[AlertChannel]
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "name": self.name,
            "symbol": self.symbol,
            "current_value": round(self.current_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "message": self.message,
            "priority": self.priority.value,
            "triggered_at": self.triggered_at.isoformat(),
            "sent_channels": [c.value for c in self.sent_channels],
            "acknowledged": self.acknowledged,
        }


class AlertConfigManager:
    """
    Manages alert threshold configuration.

    Features:
    - Customizable thresholds for various alert types
    - Per-symbol filtering
    - Cooldown management
    - Multi-channel delivery
    - Priority-based routing

    Usage:
        manager = AlertConfigManager()

        # Add a price alert
        manager.add_threshold(AlertThreshold(
            alert_type=AlertType.PRICE_MOVE,
            name="BTC Large Move",
            enabled=True,
            threshold_value=5.0,  # 5%
            comparison="pct_change",
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=60,
            symbol_filter=["BTC/USDT"],
            message_template="BTC moved {value:.2f}% in the last hour",
        ))

        # Check and trigger alerts
        events = manager.check_alerts(market_data)
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = [
        AlertThreshold(
            alert_type=AlertType.PRICE_MOVE,
            name="Large Price Move",
            enabled=True,
            threshold_value=3.0,
            comparison="pct_change",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            cooldown_minutes=30,
            symbol_filter=None,
            message_template="{symbol} moved {value:+.2f}% (threshold: {threshold}%)",
        ),
        AlertThreshold(
            alert_type=AlertType.SIGNAL_GENERATED,
            name="High Confidence Signal",
            enabled=True,
            threshold_value=0.75,
            comparison="gte",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="{symbol}: {action} signal with {value:.0%} confidence",
        ),
        AlertThreshold(
            alert_type=AlertType.DRAWDOWN,
            name="Drawdown Warning",
            enabled=True,
            threshold_value=5.0,
            comparison="gte",
            priority=AlertPriority.HIGH,
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
            cooldown_minutes=120,
            symbol_filter=None,
            message_template="Portfolio drawdown reached {value:.2f}% (threshold: {threshold}%)",
        ),
        AlertThreshold(
            alert_type=AlertType.DRAWDOWN,
            name="Critical Drawdown",
            enabled=True,
            threshold_value=10.0,
            comparison="gte",
            priority=AlertPriority.CRITICAL,
            channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="CRITICAL: Portfolio drawdown at {value:.2f}%!",
        ),
        AlertThreshold(
            alert_type=AlertType.PNL_THRESHOLD,
            name="Daily P&L Target",
            enabled=True,
            threshold_value=500.0,
            comparison="gte",
            priority=AlertPriority.LOW,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=1440,  # Once per day
            symbol_filter=None,
            message_template="Daily P&L target reached: ${value:.2f}",
        ),
        AlertThreshold(
            alert_type=AlertType.REGIME_CHANGE,
            name="Regime Change",
            enabled=True,
            threshold_value=1.0,  # Any change
            comparison="eq",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.TELEGRAM, AlertChannel.LOG],
            cooldown_minutes=60,
            symbol_filter=None,
            message_template="{symbol} regime changed from {old_regime} to {new_regime}",
        ),
        AlertThreshold(
            alert_type=AlertType.VOLATILITY,
            name="High Volatility",
            enabled=True,
            threshold_value=50.0,  # Volatility percentile
            comparison="gte",
            priority=AlertPriority.MEDIUM,
            channels=[AlertChannel.TELEGRAM],
            cooldown_minutes=120,
            symbol_filter=None,
            message_template="{symbol} volatility at {value:.0f}th percentile",
        ),
    ]

    def __init__(
        self,
        config_dir: str = "data/alert_config",
        notification_callback: Optional[Callable[[AlertEvent], None]] = None,
    ):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.notification_callback = notification_callback

        self.thresholds: Dict[str, AlertThreshold] = {}
        self._last_triggered: Dict[str, datetime] = {}
        self._alert_history: List[AlertEvent] = []
        self._alert_counter = 0

        self._load_config()

        # Initialize defaults if empty
        if not self.thresholds:
            for threshold in self.DEFAULT_THRESHOLDS:
                self.add_threshold(threshold)

    def _get_config_file(self) -> Path:
        return self.config_dir / "alert_thresholds.json"

    def _get_history_file(self) -> Path:
        return self.config_dir / "alert_history.json"

    def _load_config(self) -> None:
        """Load configuration from disk."""
        config_file = self._get_config_file()
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    for name, threshold_data in data.get("thresholds", {}).items():
                        self.thresholds[name] = AlertThreshold.from_dict(threshold_data)
                    self._last_triggered = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get("last_triggered", {}).items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load alert config: {e}")

        # Load history
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    # Just load last 100 events
                    for event_data in data[-100:]:
                        event_data["alert_type"] = AlertType(event_data["alert_type"])
                        event_data["priority"] = AlertPriority(event_data["priority"])
                        event_data["sent_channels"] = [
                            AlertChannel(c) for c in event_data["sent_channels"]
                        ]
                        event_data["triggered_at"] = datetime.fromisoformat(
                            event_data["triggered_at"]
                        )
                        self._alert_history.append(AlertEvent(**event_data))
            except (json.JSONDecodeError, IOError):
                pass

    def _save_config(self) -> None:
        """Save configuration to disk."""
        data = {
            "thresholds": {name: t.to_dict() for name, t in self.thresholds.items()},
            "last_triggered": {k: v.isoformat() for k, v in self._last_triggered.items()},
        }
        with open(self._get_config_file(), "w") as f:
            json.dump(data, f, indent=2)

    def _save_history(self) -> None:
        """Save alert history to disk."""
        data = [e.to_dict() for e in self._alert_history[-500:]]  # Keep last 500
        with open(self._get_history_file(), "w") as f:
            json.dump(data, f, indent=2)

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add or update a threshold."""
        self.thresholds[threshold.name] = threshold
        self._save_config()
        logger.info(f"Added alert threshold: {threshold.name}")

    def remove_threshold(self, name: str) -> bool:
        """Remove a threshold."""
        if name in self.thresholds:
            del self.thresholds[name]
            self._save_config()
            return True
        return False

    def enable_threshold(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a threshold."""
        if name in self.thresholds:
            self.thresholds[name].enabled = enabled
            self._save_config()
            return True
        return False

    def update_threshold_value(self, name: str, value: float) -> bool:
        """Update threshold value."""
        if name in self.thresholds:
            self.thresholds[name].threshold_value = value
            self._save_config()
            return True
        return False

    def get_threshold(self, name: str) -> Optional[AlertThreshold]:
        """Get a threshold by name."""
        return self.thresholds.get(name)

    def list_thresholds(
        self,
        alert_type: Optional[AlertType] = None,
        enabled_only: bool = False,
    ) -> List[AlertThreshold]:
        """List all thresholds, optionally filtered."""
        thresholds = list(self.thresholds.values())

        if alert_type:
            thresholds = [t for t in thresholds if t.alert_type == alert_type]

        if enabled_only:
            thresholds = [t for t in thresholds if t.enabled]

        return thresholds

    def _is_cooldown_active(self, threshold: AlertThreshold) -> bool:
        """Check if cooldown is active for a threshold."""
        last = self._last_triggered.get(threshold.name)
        if not last:
            return False

        elapsed = datetime.now() - last
        return elapsed < timedelta(minutes=threshold.cooldown_minutes)

    def _check_condition(
        self,
        current_value: float,
        threshold: AlertThreshold,
    ) -> bool:
        """Check if threshold condition is met."""
        tv = threshold.threshold_value
        comparison = threshold.comparison

        if comparison == "gt":
            return current_value > tv
        elif comparison == "gte":
            return current_value >= tv
        elif comparison == "lt":
            return current_value < tv
        elif comparison == "lte":
            return current_value <= tv
        elif comparison == "eq":
            return current_value == tv
        elif comparison == "pct_change":
            return abs(current_value) >= tv

        return False

    def _format_message(
        self,
        threshold: AlertThreshold,
        value: float,
        context: Dict[str, Any],
    ) -> str:
        """Format alert message with context."""
        try:
            return threshold.message_template.format(
                value=value,
                threshold=threshold.threshold_value,
                **context,
            )
        except KeyError as e:
            logger.warning(f"Missing context key for message: {e}")
            return f"{threshold.name}: value={value}, threshold={threshold.threshold_value}"

    def check_price_alerts(
        self,
        prices: Dict[str, float],
        price_changes: Dict[str, float],
    ) -> List[AlertEvent]:
        """Check price-related alerts."""
        events = []

        for threshold in self.list_thresholds(AlertType.PRICE_MOVE, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            for symbol, change in price_changes.items():
                if threshold.symbol_filter and symbol not in threshold.symbol_filter:
                    continue

                if self._check_condition(change, threshold):
                    event = self._create_alert_event(
                        threshold,
                        symbol,
                        change,
                        {"symbol": symbol, "price": prices.get(symbol, 0)},
                    )
                    events.append(event)

        for threshold in self.list_thresholds(AlertType.PRICE_LEVEL, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            for symbol, price in prices.items():
                if threshold.symbol_filter and symbol not in threshold.symbol_filter:
                    continue

                if self._check_condition(price, threshold):
                    event = self._create_alert_event(
                        threshold,
                        symbol,
                        price,
                        {"symbol": symbol},
                    )
                    events.append(event)

        return events

    def check_signal_alerts(
        self,
        signals: Dict[str, Dict[str, Any]],
    ) -> List[AlertEvent]:
        """Check signal-related alerts."""
        events = []

        for threshold in self.list_thresholds(AlertType.SIGNAL_GENERATED, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            for symbol, signal in signals.items():
                if threshold.symbol_filter and symbol not in threshold.symbol_filter:
                    continue

                confidence = signal.get("confidence", 0)
                action = signal.get("action", "FLAT")

                if action != "FLAT" and self._check_condition(confidence, threshold):
                    event = self._create_alert_event(
                        threshold,
                        symbol,
                        confidence,
                        {"symbol": symbol, "action": action},
                    )
                    events.append(event)

        return events

    def check_portfolio_alerts(
        self,
        portfolio_value: float,
        daily_pnl: float,
        drawdown: float,
    ) -> List[AlertEvent]:
        """Check portfolio-related alerts."""
        events = []

        # Drawdown alerts
        for threshold in self.list_thresholds(AlertType.DRAWDOWN, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            if self._check_condition(drawdown, threshold):
                event = self._create_alert_event(
                    threshold,
                    None,
                    drawdown,
                    {},
                )
                events.append(event)

        # P&L alerts
        for threshold in self.list_thresholds(AlertType.PNL_THRESHOLD, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            if self._check_condition(daily_pnl, threshold):
                event = self._create_alert_event(
                    threshold,
                    None,
                    daily_pnl,
                    {},
                )
                events.append(event)

        # Portfolio value alerts
        for threshold in self.list_thresholds(AlertType.PORTFOLIO_VALUE, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            if self._check_condition(portfolio_value, threshold):
                event = self._create_alert_event(
                    threshold,
                    None,
                    portfolio_value,
                    {},
                )
                events.append(event)

        return events

    def check_regime_alerts(
        self,
        regime_changes: Dict[str, Tuple[str, str]],  # symbol -> (old, new)
    ) -> List[AlertEvent]:
        """Check regime change alerts."""
        events = []

        for threshold in self.list_thresholds(AlertType.REGIME_CHANGE, enabled_only=True):
            if self._is_cooldown_active(threshold):
                continue

            for symbol, (old_regime, new_regime) in regime_changes.items():
                if threshold.symbol_filter and symbol not in threshold.symbol_filter:
                    continue

                event = self._create_alert_event(
                    threshold,
                    symbol,
                    1.0,  # Regime changed
                    {"symbol": symbol, "old_regime": old_regime, "new_regime": new_regime},
                )
                events.append(event)

        return events

    def _create_alert_event(
        self,
        threshold: AlertThreshold,
        symbol: Optional[str],
        value: float,
        context: Dict[str, Any],
    ) -> AlertEvent:
        """Create an alert event."""
        self._alert_counter += 1
        alert_id = f"ALT{self._alert_counter:06d}"

        message = self._format_message(threshold, value, context)

        event = AlertEvent(
            alert_id=alert_id,
            alert_type=threshold.alert_type,
            name=threshold.name,
            symbol=symbol,
            current_value=value,
            threshold_value=threshold.threshold_value,
            message=message,
            priority=threshold.priority,
            triggered_at=datetime.now(),
            sent_channels=threshold.channels,
        )

        # Update cooldown
        self._last_triggered[threshold.name] = datetime.now()
        self._alert_history.append(event)

        # Send notification
        if self.notification_callback:
            try:
                self.notification_callback(event)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")

        self._save_config()
        self._save_history()

        logger.info(f"Alert triggered: {event.name} - {event.message}")

        return event

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for event in self._alert_history:
            if event.alert_id == alert_id:
                event.acknowledged = True
                self._save_history()
                return True
        return False

    def get_recent_alerts(
        self,
        count: int = 20,
        unacknowledged_only: bool = False,
        priority: Optional[AlertPriority] = None,
    ) -> List[AlertEvent]:
        """Get recent alert events."""
        events = list(reversed(self._alert_history))

        if unacknowledged_only:
            events = [e for e in events if not e.acknowledged]

        if priority:
            events = [e for e in events if e.priority == priority]

        return events[:count]

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "total_thresholds": len(self.thresholds),
            "enabled_thresholds": len([t for t in self.thresholds.values() if t.enabled]),
            "by_type": {
                t.value: len(
                    [th for th in self.thresholds.values() if th.alert_type == t and th.enabled]
                )
                for t in AlertType
            },
            "recent_alerts_count": len(self._alert_history),
            "unacknowledged_count": len([e for e in self._alert_history if not e.acknowledged]),
        }

    def export_config(self) -> Dict[str, Any]:
        """Export full configuration for backup."""
        return {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "thresholds": {name: t.to_dict() for name, t in self.thresholds.items()},
        }

    def import_config(self, config: Dict[str, Any]) -> int:
        """Import configuration from backup."""
        imported = 0
        for name, threshold_data in config.get("thresholds", {}).items():
            try:
                threshold = AlertThreshold.from_dict(threshold_data)
                self.thresholds[name] = threshold
                imported += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to import threshold {name}: {e}")

        self._save_config()
        return imported
