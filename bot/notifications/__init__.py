"""
Notifications module for alerts and external integrations.

Provides:
- Alert management (NotificationManager, Alert, AlertLevel, AlertType)
- Channel integrations (Telegram, Discord, Email, Webhook)
- Webhook delivery system
- Convenience functions for sending notifications
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

# Import from notification_system (the main alert system)
from bot.notification_system import (
    AlertLevel,
    AlertType,
    Alert,
    NotificationChannel,
    TelegramChannel,
    DiscordChannel,
    EmailChannel,
    WebhookChannel,
    NotificationManager,
)

# Import webhook infrastructure
from .webhook import (
    WebhookManager,
    WebhookEndpoint,
    WebhookDelivery,
    WebhookEventType,
    WebhookConfig,
    create_webhook_manager,
)


class TelegramNotifier:
    """
    Simple async Telegram notifier for backward compatibility.

    Used by modules that need async Telegram messaging.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        """Check if Telegram is configured."""
        return bool(self.bot_token and self.chat_id)

    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message via Telegram asynchronously."""
        if not self.is_configured():
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            # Strip markdown if parse_mode is None
            if parse_mode is None:
                message = message.replace("**", "").replace("*", "").replace("_", "")

            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }

            if parse_mode:
                payload["parse_mode"] = parse_mode

            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers={"Content-Type": "application/json"})

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urlopen(req, timeout=10)
            )

            return response.status == 200

        except (URLError, Exception) as e:
            print(f"Telegram send failed: {e}")
            return False

    def send_message_sync(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message via Telegram synchronously."""
        if not self.is_configured():
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            if parse_mode is None:
                message = message.replace("**", "").replace("*", "").replace("_", "")

            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }

            if parse_mode:
                payload["parse_mode"] = parse_mode

            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers={"Content-Type": "application/json"})

            with urlopen(req, timeout=10) as response:
                return response.status == 200

        except (URLError, Exception) as e:
            print(f"Telegram send failed: {e}")
            return False


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get or create the global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def send_notification(
    message: str,
    category: str = "system",
    level: AlertLevel = AlertLevel.INFO,
    title: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Convenience function to send a notification.

    Args:
        message: Alert message (main content)
        category: Category string (e.g., "model_retraining", "system", "trade")
        level: Alert severity level
        title: Optional title (defaults to category-based title)
        data: Optional additional data

    Returns:
        Dict mapping channel names to success status
    """
    manager = get_notification_manager()

    # Map category string to AlertType
    category_map = {
        "system": AlertType.SYSTEM,
        "trade": AlertType.TRADE_OPENED,
        "trade_opened": AlertType.TRADE_OPENED,
        "trade_closed": AlertType.TRADE_CLOSED,
        "signal": AlertType.SIGNAL_GENERATED,
        "regime": AlertType.REGIME_CHANGE,
        "error": AlertType.ERROR,
        "model_retraining": AlertType.SYSTEM,
        "daily_summary": AlertType.DAILY_SUMMARY,
    }

    alert_type = category_map.get(category.lower(), AlertType.SYSTEM)

    # Generate title from category if not provided
    if title is None:
        title_map = {
            "model_retraining": "Model Retraining",
            "system": "System Alert",
            "trade": "Trade Alert",
            "signal": "Signal Generated",
            "error": "Error",
        }
        title = title_map.get(category.lower(), category.replace("_", " ").title())

    alert = Alert(
        alert_type=alert_type,
        level=level,
        title=title,
        message=message,
        data=data or {},
    )

    return manager.send_alert(alert)


def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """
    Convenience function to send a Telegram message directly.

    Args:
        message: Message text
        parse_mode: Parse mode (Markdown, HTML, or None)

    Returns:
        True if sent successfully
    """
    notifier = TelegramNotifier()
    return notifier.send_message_sync(message, parse_mode)


__all__ = [
    # Alert system
    "AlertLevel",
    "AlertType",
    "Alert",
    "NotificationChannel",
    "TelegramChannel",
    "DiscordChannel",
    "EmailChannel",
    "WebhookChannel",
    "NotificationManager",
    # Telegram notifier
    "TelegramNotifier",
    # Webhook system
    "WebhookManager",
    "WebhookEndpoint",
    "WebhookDelivery",
    "WebhookEventType",
    "WebhookConfig",
    "create_webhook_manager",
    # Convenience functions
    "get_notification_manager",
    "send_notification",
    "send_telegram_message",
]
