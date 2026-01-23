"""
Real-time PnL Notifications Module.

Provides push notifications for significant PnL changes,
milestone achievements, and drawdown alerts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of PnL notifications."""

    PROFIT_MILESTONE = "profit_milestone"
    LOSS_ALERT = "loss_alert"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_RECOVERY = "drawdown_recovery"
    DAILY_SUMMARY = "daily_summary"
    TRADE_RESULT = "trade_result"
    STREAK_ALERT = "streak_alert"
    VOLATILITY_SPIKE = "volatility_spike"
    TARGET_REACHED = "target_reached"


class NotificationPriority(Enum):
    """Priority levels for notifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NotificationConfig:
    """Configuration for PnL notifications."""

    # Profit milestones (trigger notification at these % gains)
    profit_milestones: List[float] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])

    # Loss alerts (trigger notification at these % losses)
    loss_thresholds: List[float] = field(default_factory=lambda: [-1, -5, -10, -20])

    # Drawdown thresholds
    drawdown_warning: float = 5.0  # Warn at 5% drawdown
    drawdown_critical: float = 10.0  # Critical at 10% drawdown

    # Daily PnL thresholds
    daily_profit_threshold: float = 3.0  # Notify on 3%+ daily profit
    daily_loss_threshold: float = -2.0  # Notify on 2%+ daily loss

    # Streak alerts
    win_streak_threshold: int = 5  # Notify on 5+ consecutive wins
    loss_streak_threshold: int = 3  # Notify on 3+ consecutive losses

    # Cooldowns (minutes)
    cooldown_profit: int = 60
    cooldown_loss: int = 30
    cooldown_drawdown: int = 120

    # Notification channels
    telegram_enabled: bool = True
    webhook_enabled: bool = False
    webhook_url: str = ""

    # Quiet hours
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7  # 7 AM
    respect_quiet_hours: bool = False


@dataclass
class PnLNotification:
    """A single notification to be sent."""

    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    pnl_value: float
    pnl_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PnLNotificationManager:
    """
    Manages real-time PnL notifications.

    Tracks portfolio changes and sends notifications based on
    configured thresholds and rules.
    """

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
        state_file: str = "data/pnl_notification_state.json",
    ):
        self.config = config or NotificationConfig()
        self.state_file = Path(state_file)

        self._last_notifications: Dict[str, datetime] = {}
        self._notified_milestones: List[float] = []
        self._previous_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._starting_equity: float = 0.0
        self._win_streak: int = 0
        self._loss_streak: int = 0
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._last_daily_reset: Optional[datetime] = None

        self._notification_handlers: List[Callable[[PnLNotification], None]] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load notification state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    state = json.load(f)

                self._notified_milestones = state.get("notified_milestones", [])
                self._peak_equity = state.get("peak_equity", 0)
                self._starting_equity = state.get("starting_equity", 0)
                self._win_streak = state.get("win_streak", 0)
                self._loss_streak = state.get("loss_streak", 0)

                last_notif = state.get("last_notifications", {})
                self._last_notifications = {
                    k: datetime.fromisoformat(v) for k, v in last_notif.items()
                }

        except Exception as e:
            logger.error(f"Error loading notification state: {e}")

    def _save_state(self) -> None:
        """Save notification state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "notified_milestones": self._notified_milestones,
                "peak_equity": self._peak_equity,
                "starting_equity": self._starting_equity,
                "win_streak": self._win_streak,
                "loss_streak": self._loss_streak,
                "last_notifications": {
                    k: v.isoformat() for k, v in self._last_notifications.items()
                },
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving notification state: {e}")

    def add_handler(self, handler: Callable[[PnLNotification], None]) -> None:
        """Add a notification handler."""
        self._notification_handlers.append(handler)

    def set_starting_equity(self, equity: float) -> None:
        """Set the starting equity for PnL calculations."""
        self._starting_equity = equity
        self._peak_equity = equity
        self._daily_start_equity = equity
        self._last_daily_reset = datetime.now()
        self._save_state()

    def update_equity(self, current_equity: float) -> List[PnLNotification]:
        """
        Update with current equity and generate notifications.

        Args:
            current_equity: Current portfolio equity value

        Returns:
            List of notifications generated
        """
        notifications = []

        if self._starting_equity == 0:
            self._starting_equity = current_equity
            self._peak_equity = current_equity
            self._daily_start_equity = current_equity
            self._last_daily_reset = datetime.now()

        # Check for daily reset
        now = datetime.now()
        if self._last_daily_reset and now.date() > self._last_daily_reset.date():
            self._daily_start_equity = current_equity
            self._last_daily_reset = now

        # Calculate metrics
        total_pnl = current_equity - self._starting_equity
        total_pnl_pct = (
            (total_pnl / self._starting_equity * 100) if self._starting_equity > 0 else 0
        )

        daily_pnl = current_equity - self._daily_start_equity
        daily_pnl_pct = (
            (daily_pnl / self._daily_start_equity * 100) if self._daily_start_equity > 0 else 0
        )

        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Calculate drawdown
        drawdown = (
            (self._peak_equity - current_equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0
        )

        # Check quiet hours
        if self._in_quiet_hours():
            return notifications

        # Check profit milestones
        for milestone in self.config.profit_milestones:
            if total_pnl_pct >= milestone and milestone not in self._notified_milestones:
                if self._can_notify("profit"):
                    notif = PnLNotification(
                        type=NotificationType.PROFIT_MILESTONE,
                        priority=NotificationPriority.MEDIUM
                        if milestone < 25
                        else NotificationPriority.HIGH,
                        title=f"Profit Milestone: +{milestone}%",
                        message=f"Portfolio has reached +{total_pnl_pct:.1f}% total profit (${total_pnl:.2f})",
                        pnl_value=total_pnl,
                        pnl_percent=total_pnl_pct,
                        metadata={"milestone": milestone},
                    )
                    notifications.append(notif)
                    self._notified_milestones.append(milestone)
                    self._record_notification("profit")

        # Check loss thresholds
        for threshold in self.config.loss_thresholds:
            if total_pnl_pct <= threshold and threshold not in self._notified_milestones:
                if self._can_notify("loss"):
                    notif = PnLNotification(
                        type=NotificationType.LOSS_ALERT,
                        priority=NotificationPriority.HIGH
                        if threshold <= -10
                        else NotificationPriority.MEDIUM,
                        title=f"Loss Alert: {threshold}%",
                        message=f"Portfolio is down {total_pnl_pct:.1f}% (${total_pnl:.2f})",
                        pnl_value=total_pnl,
                        pnl_percent=total_pnl_pct,
                        metadata={"threshold": threshold},
                    )
                    notifications.append(notif)
                    self._notified_milestones.append(threshold)
                    self._record_notification("loss")

        # Check drawdown
        if drawdown >= self.config.drawdown_critical and self._can_notify("drawdown"):
            notif = PnLNotification(
                type=NotificationType.DRAWDOWN_WARNING,
                priority=NotificationPriority.CRITICAL,
                title=f"CRITICAL: Drawdown {drawdown:.1f}%",
                message=f"Portfolio drawdown has reached {drawdown:.1f}% from peak of ${self._peak_equity:.2f}",
                pnl_value=current_equity - self._peak_equity,
                pnl_percent=-drawdown,
                metadata={"peak_equity": self._peak_equity, "current_equity": current_equity},
            )
            notifications.append(notif)
            self._record_notification("drawdown")

        elif drawdown >= self.config.drawdown_warning and self._can_notify("drawdown"):
            notif = PnLNotification(
                type=NotificationType.DRAWDOWN_WARNING,
                priority=NotificationPriority.HIGH,
                title=f"Drawdown Warning: {drawdown:.1f}%",
                message=f"Portfolio in {drawdown:.1f}% drawdown from peak",
                pnl_value=current_equity - self._peak_equity,
                pnl_percent=-drawdown,
            )
            notifications.append(notif)
            self._record_notification("drawdown")

        # Check daily thresholds
        if daily_pnl_pct >= self.config.daily_profit_threshold and self._can_notify("daily_profit"):
            notif = PnLNotification(
                type=NotificationType.DAILY_SUMMARY,
                priority=NotificationPriority.MEDIUM,
                title=f"Strong Day: +{daily_pnl_pct:.1f}%",
                message=f"Today's profit is +${daily_pnl:.2f} ({daily_pnl_pct:.1f}%)",
                pnl_value=daily_pnl,
                pnl_percent=daily_pnl_pct,
            )
            notifications.append(notif)
            self._record_notification("daily_profit")

        if daily_pnl_pct <= self.config.daily_loss_threshold and self._can_notify("daily_loss"):
            notif = PnLNotification(
                type=NotificationType.DAILY_SUMMARY,
                priority=NotificationPriority.HIGH,
                title=f"Tough Day: {daily_pnl_pct:.1f}%",
                message=f"Today's loss is ${daily_pnl:.2f} ({daily_pnl_pct:.1f}%)",
                pnl_value=daily_pnl,
                pnl_percent=daily_pnl_pct,
            )
            notifications.append(notif)
            self._record_notification("daily_loss")

        self._previous_pnl = total_pnl
        self._save_state()

        # Send notifications
        for notif in notifications:
            self._send_notification(notif)

        return notifications

    def record_trade(self, pnl: float, pnl_percent: float) -> List[PnLNotification]:
        """
        Record a completed trade and check for streak notifications.

        Args:
            pnl: Trade PnL in base currency
            pnl_percent: Trade PnL percentage

        Returns:
            List of notifications generated
        """
        notifications = []

        if pnl > 0:
            self._win_streak += 1
            self._loss_streak = 0

            if self._win_streak >= self.config.win_streak_threshold:
                notif = PnLNotification(
                    type=NotificationType.STREAK_ALERT,
                    priority=NotificationPriority.MEDIUM,
                    title=f"Hot Streak: {self._win_streak} Wins",
                    message=f"You're on a {self._win_streak}-trade winning streak!",
                    pnl_value=pnl,
                    pnl_percent=pnl_percent,
                    metadata={"streak": self._win_streak, "type": "win"},
                )
                notifications.append(notif)

        elif pnl < 0:
            self._loss_streak += 1
            self._win_streak = 0

            if self._loss_streak >= self.config.loss_streak_threshold:
                notif = PnLNotification(
                    type=NotificationType.STREAK_ALERT,
                    priority=NotificationPriority.HIGH,
                    title=f"Cold Streak: {self._loss_streak} Losses",
                    message=f"Warning: {self._loss_streak} consecutive losing trades. Consider reviewing strategy.",
                    pnl_value=pnl,
                    pnl_percent=pnl_percent,
                    metadata={"streak": self._loss_streak, "type": "loss"},
                )
                notifications.append(notif)

        self._save_state()

        # Send notifications
        for notif in notifications:
            self._send_notification(notif)

        return notifications

    def _can_notify(self, notification_type: str) -> bool:
        """Check if notification is allowed based on cooldown."""
        last_notif = self._last_notifications.get(notification_type)
        if not last_notif:
            return True

        cooldown_map = {
            "profit": self.config.cooldown_profit,
            "loss": self.config.cooldown_loss,
            "drawdown": self.config.cooldown_drawdown,
            "daily_profit": 60 * 12,  # 12 hours
            "daily_loss": 60 * 12,
        }

        cooldown = cooldown_map.get(notification_type, 60)
        return datetime.now() - last_notif > timedelta(minutes=cooldown)

    def _record_notification(self, notification_type: str) -> None:
        """Record that a notification was sent."""
        self._last_notifications[notification_type] = datetime.now()

    def _in_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        if not self.config.respect_quiet_hours:
            return False

        hour = datetime.now().hour
        if self.config.quiet_hours_start > self.config.quiet_hours_end:
            # Spans midnight
            return hour >= self.config.quiet_hours_start or hour < self.config.quiet_hours_end
        else:
            return self.config.quiet_hours_start <= hour < self.config.quiet_hours_end

    def _send_notification(self, notification: PnLNotification) -> None:
        """Send notification via configured channels."""
        # Call registered handlers
        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")

        # Telegram
        if self.config.telegram_enabled:
            self._send_telegram(notification)

        # Webhook
        if self.config.webhook_enabled and self.config.webhook_url:
            self._send_webhook(notification)

    def _send_telegram(self, notification: PnLNotification) -> None:
        """Send notification via Telegram."""
        try:
            from .notifications import TelegramNotifier

            # Priority emoji mapping
            emoji_map = {
                NotificationPriority.LOW: "📊",
                NotificationPriority.MEDIUM: "📈",
                NotificationPriority.HIGH: "⚠️",
                NotificationPriority.CRITICAL: "🚨",
            }

            emoji = emoji_map.get(notification.priority, "📊")

            message = f"{emoji} *{notification.title}*\n\n{notification.message}"

            notifier = TelegramNotifier()
            asyncio.create_task(notifier.send_message(message))

        except Exception as e:
            logger.error(f"Telegram notification error: {e}")

    def _send_webhook(self, notification: PnLNotification) -> None:
        """Send notification via webhook."""
        try:
            import requests

            payload = {
                "type": notification.type.value,
                "priority": notification.priority.value,
                "title": notification.title,
                "message": notification.message,
                "pnl_value": notification.pnl_value,
                "pnl_percent": notification.pnl_percent,
                "timestamp": notification.timestamp.isoformat(),
                "metadata": notification.metadata,
            }

            requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=5,
            )

        except Exception as e:
            logger.error(f"Webhook notification error: {e}")

    def get_notification_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent notification history."""
        history = []

        for ntype, timestamp in sorted(
            self._last_notifications.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]:
            history.append(
                {
                    "type": ntype,
                    "timestamp": timestamp.isoformat(),
                    "time_ago": str(datetime.now() - timestamp),
                }
            )

        return history

    def reset_milestones(self) -> None:
        """Reset notified milestones (e.g., after portfolio reset)."""
        self._notified_milestones = []
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current notification status."""
        return {
            "starting_equity": self._starting_equity,
            "peak_equity": self._peak_equity,
            "notified_milestones": self._notified_milestones,
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "last_notifications": {k: v.isoformat() for k, v in self._last_notifications.items()},
            "config": {
                "profit_milestones": self.config.profit_milestones,
                "loss_thresholds": self.config.loss_thresholds,
                "drawdown_warning": self.config.drawdown_warning,
                "drawdown_critical": self.config.drawdown_critical,
            },
        }
