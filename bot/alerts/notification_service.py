"""
Notification Service for Trading Alerts.

Supports:
- Telegram
- Discord (webhook)
- Email (SMTP)

Alerts for:
- Trade executions
- Model drift detected
- Performance warnings
- System errors
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    SIGNAL_GENERATED = "signal_generated"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE_WARNING = "performance_warning"
    SYSTEM_ERROR = "system_error"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Alert:
    """Alert message."""
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
        }


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class TelegramChannel(NotificationChannel):
    """Telegram notification channel."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _format_message(self, alert: Alert) -> str:
        """Format alert for Telegram."""
        # Emoji mapping
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        type_emoji = {
            AlertType.TRADE_OPENED: "📈",
            AlertType.TRADE_CLOSED: "📉",
            AlertType.SIGNAL_GENERATED: "🔔",
            AlertType.MODEL_DRIFT: "📊",
            AlertType.PERFORMANCE_WARNING: "⚡",
            AlertType.SYSTEM_ERROR: "🔧",
            AlertType.DAILY_SUMMARY: "📋",
        }

        emoji = type_emoji.get(alert.type, "📢")
        level = level_emoji.get(alert.level, "")

        text = f"{emoji} {level} *{alert.title}*\n\n"
        text += f"{alert.message}\n"

        if alert.data:
            text += "\n```\n"
            for key, value in alert.data.items():
                if isinstance(value, float):
                    text += f"{key}: {value:.4f}\n"
                else:
                    text += f"{key}: {value}\n"
            text += "```"

        text += f"\n_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"

        return text

    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("Telegram not configured")
            return False

        try:
            message = self._format_message(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/sendMessage",
                    json={
                        'chat_id': self.chat_id,
                        'text': message,
                        'parse_mode': 'Markdown',
                        'disable_web_page_preview': True,
                    },
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Telegram alert sent: {alert.title}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Telegram error: {response.status} - {error}")
                        return False

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class DiscordChannel(NotificationChannel):
    """Discord webhook notification channel."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def _format_embed(self, alert: Alert) -> Dict:
        """Format alert as Discord embed."""
        color_map = {
            AlertLevel.INFO: 0x3498db,  # Blue
            AlertLevel.WARNING: 0xf39c12,  # Orange
            AlertLevel.ERROR: 0xe74c3c,  # Red
            AlertLevel.CRITICAL: 0x9b59b6,  # Purple
        }

        embed = {
            'title': alert.title,
            'description': alert.message,
            'color': color_map.get(alert.level, 0x95a5a6),
            'timestamp': alert.timestamp.isoformat(),
            'footer': {'text': f"Alert Type: {alert.type.value}"},
        }

        if alert.data:
            embed['fields'] = [
                {'name': k, 'value': str(v), 'inline': True}
                for k, v in list(alert.data.items())[:25]  # Discord limit
            ]

        return embed

    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("Discord not configured")
            return False

        try:
            embed = self._format_embed(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json={'embeds': [embed]},
                    timeout=10
                ) as response:
                    if response.status in [200, 204]:
                        logger.debug(f"Discord alert sent: {alert.title}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Discord error: {response.status} - {error}")
                        return False

        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False


class NotificationService:
    """
    Central notification service.

    Manages multiple channels and alert routing.
    """

    def __init__(self):
        self.channels: List[NotificationChannel] = []
        self._setup_channels()

        # Alert filtering
        self.min_level = AlertLevel.INFO
        self.enabled_types: set = set(AlertType)

        # Rate limiting
        self._last_alerts: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Min seconds between same alert type

    def _setup_channels(self):
        """Initialize notification channels from environment."""
        # Telegram
        telegram = TelegramChannel()
        if telegram.is_configured():
            self.channels.append(telegram)
            logger.info("Telegram notifications enabled")

        # Discord
        discord = DiscordChannel()
        if discord.is_configured():
            self.channels.append(discord)
            logger.info("Discord notifications enabled")

        if not self.channels:
            logger.warning("No notification channels configured")

    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        if channel.is_configured():
            self.channels.append(channel)

    def set_min_level(self, level: AlertLevel):
        """Set minimum alert level to send."""
        self.min_level = level

    def enable_types(self, types: List[AlertType]):
        """Enable specific alert types."""
        self.enabled_types = set(types)

    def disable_types(self, types: List[AlertType]):
        """Disable specific alert types."""
        for t in types:
            self.enabled_types.discard(t)

    def _should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent."""
        # Check type
        if alert.type not in self.enabled_types:
            return False

        # Check level
        level_order = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        if level_order.index(alert.level) < level_order.index(self.min_level):
            return False

        # Rate limiting
        key = f"{alert.type.value}_{alert.title}"
        now = datetime.now()

        if key in self._last_alerts:
            elapsed = (now - self._last_alerts[key]).total_seconds()
            if elapsed < self._rate_limit_seconds:
                logger.debug(f"Rate limited: {key}")
                return False

        self._last_alerts[key] = now
        return True

    async def send(self, alert: Alert) -> bool:
        """Send alert through all configured channels."""
        if not self._should_send(alert):
            return False

        if not self.channels:
            logger.warning("No notification channels available")
            return False

        # Send to all channels concurrently
        results = await asyncio.gather(
            *[channel.send(alert) for channel in self.channels],
            return_exceptions=True
        )

        success = any(r is True for r in results)
        if not success:
            logger.warning(f"Failed to send alert: {alert.title}")

        return success

    # Convenience methods for common alerts

    async def trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        confidence: float,
        **kwargs
    ):
        """Send trade opened alert."""
        alert = Alert(
            type=AlertType.TRADE_OPENED,
            level=AlertLevel.INFO,
            title=f"Trade Opened: {symbol}",
            message=f"Opened {side.upper()} position",
            data={
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'confidence': confidence,
                **kwargs
            }
        )
        await self.send(alert)

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
        **kwargs
    ):
        """Send trade closed alert."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING

        alert = Alert(
            type=AlertType.TRADE_CLOSED,
            level=level,
            title=f"Trade Closed: {symbol}",
            message=f"Closed {side.upper()} - {'Profit' if pnl >= 0 else 'Loss'}: ${pnl:.2f} ({pnl_pct:.2f}%)",
            data={
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason,
                **kwargs
            }
        )
        await self.send(alert)

    async def model_drift(
        self,
        symbol: str,
        metric: str,
        current_value: float,
        threshold: float,
        **kwargs
    ):
        """Send model drift alert."""
        alert = Alert(
            type=AlertType.MODEL_DRIFT,
            level=AlertLevel.WARNING,
            title=f"Model Drift: {symbol}",
            message=f"{metric} degraded: {current_value:.2%} (threshold: {threshold:.2%})",
            data={
                'symbol': symbol,
                'metric': metric,
                'current_value': current_value,
                'threshold': threshold,
                **kwargs
            }
        )
        await self.send(alert)

    async def performance_warning(
        self,
        metric: str,
        value: float,
        message: str,
        **kwargs
    ):
        """Send performance warning."""
        alert = Alert(
            type=AlertType.PERFORMANCE_WARNING,
            level=AlertLevel.WARNING,
            title=f"Performance Warning: {metric}",
            message=message,
            data={'metric': metric, 'value': value, **kwargs}
        )
        await self.send(alert)

    async def system_error(self, error: str, details: Optional[str] = None):
        """Send system error alert."""
        alert = Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="System Error",
            message=error,
            data={'details': details} if details else None
        )
        await self.send(alert)

    async def daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_equity: float,
        **kwargs
    ):
        """Send daily summary."""
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        alert = Alert(
            type=AlertType.DAILY_SUMMARY,
            level=AlertLevel.INFO,
            title="Daily Trading Summary",
            message=f"Trades: {total_trades} | Win Rate: {win_rate:.1%} | P&L: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%)",
            data={
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'total_equity': total_equity,
                **kwargs
            }
        )
        await self.send(alert)


# Global instance
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get or create notification service."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


async def send_alert(
    type: AlertType,
    level: AlertLevel,
    title: str,
    message: str,
    data: Optional[Dict] = None
):
    """Convenience function to send an alert."""
    service = get_notification_service()
    alert = Alert(type=type, level=level, title=title, message=message, data=data)
    await service.send(alert)
