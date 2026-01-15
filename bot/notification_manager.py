"""
Unified Notification Manager

Phase 10: Comprehensive notification system supporting multiple channels
(Telegram, Email, Webhooks) with configurable routing and rate limiting.

Features:
- Multi-channel delivery (Telegram, Email, Webhook, Console)
- Priority-based routing
- Rate limiting to prevent spam
- Template-based message formatting
- Daily/weekly report generation
- Alert aggregation and batching
"""

import asyncio
import logging
import os
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import json
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = 1       # Informational, can be batched
    MEDIUM = 2    # Important, deliver within minutes
    HIGH = 3      # Urgent, deliver immediately
    CRITICAL = 4  # Emergency, deliver immediately to all channels


class NotificationCategory(Enum):
    """Categories for notification routing."""
    TRADE = "trade"           # Trade executions
    SIGNAL = "signal"         # Strategy signals
    RISK = "risk"             # Risk alerts
    SYSTEM = "system"         # System status
    PERFORMANCE = "performance"  # Performance updates
    REGIME = "regime"         # Regime changes
    ERROR = "error"           # Errors and exceptions
    REPORT = "report"         # Daily/weekly reports


class NotificationChannel(Enum):
    """Delivery channels."""
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    SLACK = "slack"  # Future


@dataclass
class Notification:
    """A notification to be delivered."""
    notification_id: str
    category: NotificationCategory
    priority: NotificationPriority
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional structured data
    data: Dict[str, Any] = field(default_factory=dict)

    # Targeting
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)  # email addresses, chat IDs

    # Tracking
    delivered: bool = False
    delivery_attempts: int = 0
    delivery_errors: List[str] = field(default_factory=list)

    # Formatting hints
    format_html: bool = True
    include_timestamp: bool = True


@dataclass
class NotificationConfig:
    """Configuration for notification system."""
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_ids: List[str] = field(default_factory=list)
    telegram_admin_chat_id: str = ""

    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_recipients: List[str] = field(default_factory=list)

    # Webhook
    webhook_enabled: bool = False
    webhook_urls: List[str] = field(default_factory=list)

    # Rate limiting
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 200
    batch_low_priority: bool = True
    batch_interval_seconds: int = 300  # 5 minutes

    # Routing
    route_critical_to_all: bool = True
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 7     # 7 AM
    quiet_hours_allow_critical: bool = True

    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Load configuration from environment variables."""
        return cls(
            telegram_enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_ids=os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if os.getenv("TELEGRAM_CHAT_IDS") else [],
            telegram_admin_chat_id=os.getenv("TELEGRAM_ADMIN_CHAT_ID", ""),
            email_enabled=os.getenv("EMAIL_ENABLED", "false").lower() == "true",
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            email_from=os.getenv("EMAIL_FROM", ""),
            email_recipients=os.getenv("EMAIL_RECIPIENTS", "").split(",") if os.getenv("EMAIL_RECIPIENTS") else [],
            webhook_enabled=os.getenv("WEBHOOK_ENABLED", "false").lower() == "true",
            webhook_urls=os.getenv("WEBHOOK_URLS", "").split(",") if os.getenv("WEBHOOK_URLS") else []
        )


class NotificationSender(ABC):
    """Base class for notification senders."""

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Return the channel this sender handles."""
        pass

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """
        Send a notification.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this sender is properly configured."""
        pass


class TelegramSender(NotificationSender):
    """Telegram bot notification sender."""

    def __init__(self, bot_token: str, default_chat_ids: List[str]):
        """
        Initialize Telegram sender.

        Args:
            bot_token: Telegram bot token
            default_chat_ids: Default chat IDs to send to
        """
        self.bot_token = bot_token
        self.default_chat_ids = [cid.strip() for cid in default_chat_ids if cid.strip()]
        self._http_client = None

        if self.bot_token:
            logger.info(f"TelegramSender initialized with {len(self.default_chat_ids)} chat IDs")

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.TELEGRAM

    def is_available(self) -> bool:
        return bool(self.bot_token and self.default_chat_ids)

    async def send(self, notification: Notification) -> bool:
        """Send notification via Telegram."""
        if not self.is_available():
            logger.warning("Telegram sender not configured")
            return False

        # Determine recipients
        chat_ids = notification.recipients if notification.recipients else self.default_chat_ids

        # Format message
        message = self._format_message(notification)

        # Send to each chat
        success = True
        for chat_id in chat_ids:
            try:
                await self._send_message(chat_id, message, notification.format_html)
            except Exception as e:
                logger.error(f"Failed to send Telegram to {chat_id}: {e}")
                notification.delivery_errors.append(f"Telegram {chat_id}: {e}")
                success = False

        return success

    async def _send_message(self, chat_id: str, message: str, parse_html: bool = True):
        """Send a message to a specific chat."""
        import aiohttp

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML" if parse_html else None,
            "disable_web_page_preview": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Telegram API error: {response.status} - {error_text}")

    def _format_message(self, notification: Notification) -> str:
        """Format notification for Telegram."""
        # Priority emoji
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.MEDIUM: "📊",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.CRITICAL: "🚨"
        }

        # Category emoji
        category_emoji = {
            NotificationCategory.TRADE: "💰",
            NotificationCategory.SIGNAL: "📈",
            NotificationCategory.RISK: "🛡️",
            NotificationCategory.SYSTEM: "⚙️",
            NotificationCategory.PERFORMANCE: "📊",
            NotificationCategory.REGIME: "🔄",
            NotificationCategory.ERROR: "❌",
            NotificationCategory.REPORT: "📋"
        }

        emoji = category_emoji.get(notification.category, "📌")
        priority = priority_emoji.get(notification.priority, "")

        parts = []

        # Header
        parts.append(f"{emoji} <b>{notification.title}</b> {priority}")

        # Message body
        parts.append("")
        parts.append(notification.message)

        # Data fields
        if notification.data:
            parts.append("")
            for key, value in notification.data.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                parts.append(f"• <i>{key}</i>: {value}")

        # Timestamp
        if notification.include_timestamp:
            parts.append("")
            parts.append(f"<i>{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>")

        return "\n".join(parts)


class EmailSender(NotificationSender):
    """Email notification sender via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_address: str,
        default_recipients: List[str]
    ):
        """Initialize email sender."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address
        self.default_recipients = [r.strip() for r in default_recipients if r.strip()]

        if self.smtp_user:
            logger.info(f"EmailSender initialized with {len(self.default_recipients)} recipients")

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    def is_available(self) -> bool:
        return bool(self.smtp_user and self.smtp_password and self.default_recipients)

    async def send(self, notification: Notification) -> bool:
        """Send notification via email."""
        if not self.is_available():
            logger.warning("Email sender not configured")
            return False

        recipients = notification.recipients if notification.recipients else self.default_recipients

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._format_subject(notification)
            msg["From"] = self.from_address
            msg["To"] = ", ".join(recipients)

            # Plain text version
            text_body = self._format_text_body(notification)
            msg.attach(MIMEText(text_body, "plain"))

            # HTML version
            if notification.format_html:
                html_body = self._format_html_body(notification)
                msg.attach(MIMEText(html_body, "html"))

            # Send
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_smtp,
                recipients,
                msg
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            notification.delivery_errors.append(f"Email: {e}")
            return False

    def _send_smtp(self, recipients: List[str], msg: MIMEMultipart):
        """Send email via SMTP (blocking)."""
        context = ssl.create_default_context()

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls(context=context)
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.from_address, recipients, msg.as_string())

    def _format_subject(self, notification: Notification) -> str:
        """Format email subject."""
        priority_prefix = {
            NotificationPriority.CRITICAL: "[CRITICAL] ",
            NotificationPriority.HIGH: "[ALERT] ",
            NotificationPriority.MEDIUM: "",
            NotificationPriority.LOW: ""
        }
        prefix = priority_prefix.get(notification.priority, "")
        return f"{prefix}Trading Bot: {notification.title}"

    def _format_text_body(self, notification: Notification) -> str:
        """Format plain text email body."""
        lines = [
            notification.title,
            "=" * len(notification.title),
            "",
            notification.message,
            ""
        ]

        if notification.data:
            lines.append("Details:")
            for key, value in notification.data.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                lines.append(f"  - {key}: {value}")
            lines.append("")

        lines.append(f"Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("Algo Trading Bot")

        return "\n".join(lines)

    def _format_html_body(self, notification: Notification) -> str:
        """Format HTML email body."""
        # Priority colors
        priority_colors = {
            NotificationPriority.CRITICAL: "#dc3545",
            NotificationPriority.HIGH: "#fd7e14",
            NotificationPriority.MEDIUM: "#0d6efd",
            NotificationPriority.LOW: "#6c757d"
        }
        color = priority_colors.get(notification.priority, "#6c757d")

        data_rows = ""
        if notification.data:
            for key, value in notification.data.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                data_rows += f"<tr><td style='padding: 8px; border-bottom: 1px solid #eee;'><strong>{key}</strong></td><td style='padding: 8px; border-bottom: 1px solid #eee;'>{value}</td></tr>"

        data_table = f"<table style='width: 100%; border-collapse: collapse; margin: 16px 0;'>{data_rows}</table>" if data_rows else ""

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 16px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 8px 0; color: #1a1a1a;">{notification.title}</h2>
                <p style="margin: 0; color: #666; font-size: 14px;">{notification.category.value.upper()} • {notification.priority.name}</p>
            </div>

            <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                <p style="margin: 0; line-height: 1.6;">{notification.message}</p>
            </div>

            {data_table}

            <div style="color: #999; font-size: 12px; border-top: 1px solid #eee; padding-top: 16px; margin-top: 20px;">
                <p style="margin: 0;">{notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p style="margin: 8px 0 0 0;">Algo Trading Bot</p>
            </div>
        </body>
        </html>
        """


class WebhookSender(NotificationSender):
    """Webhook notification sender."""

    def __init__(self, webhook_urls: List[str]):
        """Initialize webhook sender."""
        self.webhook_urls = [url.strip() for url in webhook_urls if url.strip()]

        if self.webhook_urls:
            logger.info(f"WebhookSender initialized with {len(self.webhook_urls)} URLs")

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.WEBHOOK

    def is_available(self) -> bool:
        return bool(self.webhook_urls)

    async def send(self, notification: Notification) -> bool:
        """Send notification via webhook."""
        if not self.is_available():
            return False

        import aiohttp

        payload = {
            "notification_id": notification.notification_id,
            "category": notification.category.value,
            "priority": notification.priority.name,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
            "timestamp": notification.timestamp.isoformat()
        }

        success = True
        async with aiohttp.ClientSession() as session:
            for url in self.webhook_urls:
                try:
                    async with session.post(url, json=payload) as response:
                        if response.status >= 400:
                            logger.warning(f"Webhook {url} returned {response.status}")
                            success = False
                except Exception as e:
                    logger.error(f"Webhook {url} failed: {e}")
                    notification.delivery_errors.append(f"Webhook {url}: {e}")
                    success = False

        return success


class ConsoleSender(NotificationSender):
    """Console/logging notification sender."""

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.CONSOLE

    def is_available(self) -> bool:
        return True

    async def send(self, notification: Notification) -> bool:
        """Log notification to console."""
        level = {
            NotificationPriority.LOW: logging.INFO,
            NotificationPriority.MEDIUM: logging.INFO,
            NotificationPriority.HIGH: logging.WARNING,
            NotificationPriority.CRITICAL: logging.ERROR
        }.get(notification.priority, logging.INFO)

        logger.log(
            level,
            f"[{notification.category.value.upper()}] {notification.title}: {notification.message}"
        )

        return True


class NotificationManager:
    """
    Central notification manager that routes notifications to appropriate channels.

    Features:
    - Multi-channel delivery
    - Priority-based routing
    - Rate limiting
    - Quiet hours
    - Notification batching
    - Deduplication
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notification manager.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig.from_env()
        self.senders: Dict[NotificationChannel, NotificationSender] = {}

        # Initialize senders based on config
        self._init_senders()

        # Rate limiting
        self._sent_timestamps: List[datetime] = []
        self._rate_limit_lock = asyncio.Lock()

        # Batching
        self._batch_queue: List[Notification] = []
        self._batch_task: Optional[asyncio.Task] = None

        # Deduplication
        self._recent_hashes: Set[str] = set()
        self._hash_expiry: Dict[str, datetime] = {}

        # Routing rules
        self._category_routes: Dict[NotificationCategory, List[NotificationChannel]] = {
            NotificationCategory.TRADE: [NotificationChannel.TELEGRAM, NotificationChannel.CONSOLE],
            NotificationCategory.SIGNAL: [NotificationChannel.CONSOLE],
            NotificationCategory.RISK: [NotificationChannel.TELEGRAM, NotificationChannel.EMAIL, NotificationChannel.CONSOLE],
            NotificationCategory.SYSTEM: [NotificationChannel.CONSOLE],
            NotificationCategory.PERFORMANCE: [NotificationChannel.CONSOLE],
            NotificationCategory.REGIME: [NotificationChannel.TELEGRAM, NotificationChannel.CONSOLE],
            NotificationCategory.ERROR: [NotificationChannel.TELEGRAM, NotificationChannel.EMAIL, NotificationChannel.CONSOLE],
            NotificationCategory.REPORT: [NotificationChannel.EMAIL, NotificationChannel.TELEGRAM]
        }

        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "total_rate_limited": 0,
            "total_deduplicated": 0
        }

        logger.info("NotificationManager initialized")

    def _init_senders(self):
        """Initialize notification senders based on config."""
        # Console sender always available
        self.senders[NotificationChannel.CONSOLE] = ConsoleSender()

        # Telegram sender
        if self.config.telegram_enabled:
            self.senders[NotificationChannel.TELEGRAM] = TelegramSender(
                bot_token=self.config.telegram_bot_token,
                default_chat_ids=self.config.telegram_chat_ids
            )

        # Email sender
        if self.config.email_enabled:
            self.senders[NotificationChannel.EMAIL] = EmailSender(
                smtp_host=self.config.smtp_host,
                smtp_port=self.config.smtp_port,
                smtp_user=self.config.smtp_user,
                smtp_password=self.config.smtp_password,
                from_address=self.config.email_from,
                default_recipients=self.config.email_recipients
            )

        # Webhook sender
        if self.config.webhook_enabled:
            self.senders[NotificationChannel.WEBHOOK] = WebhookSender(
                webhook_urls=self.config.webhook_urls
            )

        available = [ch.value for ch, sender in self.senders.items() if sender.is_available()]
        logger.info(f"Available notification channels: {available}")

    async def start(self):
        """Start background tasks."""
        if self.config.batch_low_priority:
            self._batch_task = asyncio.create_task(self._batch_processor())
            logger.info("Notification batch processor started")

    async def stop(self):
        """Stop background tasks."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Send any remaining batched notifications
        if self._batch_queue:
            await self._flush_batch()

    async def notify(
        self,
        category: NotificationCategory,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None,
        recipients: Optional[List[str]] = None,
        force: bool = False
    ) -> bool:
        """
        Send a notification.

        Args:
            category: Notification category
            title: Notification title
            message: Notification message
            priority: Priority level
            data: Optional structured data
            channels: Override default channel routing
            recipients: Override default recipients
            force: Bypass rate limiting and quiet hours

        Returns:
            True if notification was sent (or queued)
        """
        # Create notification
        notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(title)%10000}"

        notification = Notification(
            notification_id=notification_id,
            category=category,
            priority=priority,
            title=title,
            message=message,
            data=data or {},
            channels=channels or self._get_channels_for_category(category, priority),
            recipients=recipients or []
        )

        # Deduplication check
        if not force and self._is_duplicate(notification):
            self.stats["total_deduplicated"] += 1
            logger.debug(f"Deduplicated notification: {title}")
            return True

        # Quiet hours check
        if not force and self._in_quiet_hours() and priority != NotificationPriority.CRITICAL:
            if not self.config.quiet_hours_allow_critical:
                logger.debug(f"Notification suppressed during quiet hours: {title}")
                return True

        # Rate limiting check
        if not force and not await self._check_rate_limit():
            self.stats["total_rate_limited"] += 1
            logger.warning(f"Notification rate limited: {title}")
            return False

        # Batch low priority notifications
        if priority == NotificationPriority.LOW and self.config.batch_low_priority and not force:
            self._batch_queue.append(notification)
            return True

        # Send immediately
        return await self._deliver(notification)

    async def _deliver(self, notification: Notification) -> bool:
        """Deliver notification to all configured channels."""
        success = True

        # Critical notifications go to all channels if configured
        channels = notification.channels
        if notification.priority == NotificationPriority.CRITICAL and self.config.route_critical_to_all:
            channels = list(self.senders.keys())

        for channel in channels:
            sender = self.senders.get(channel)
            if not sender or not sender.is_available():
                continue

            try:
                notification.delivery_attempts += 1
                result = await sender.send(notification)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Failed to send via {channel.value}: {e}")
                notification.delivery_errors.append(f"{channel.value}: {e}")
                success = False

        notification.delivered = success

        if success:
            self.stats["total_sent"] += 1
        else:
            self.stats["total_failed"] += 1

        return success

    def _get_channels_for_category(
        self,
        category: NotificationCategory,
        priority: NotificationPriority
    ) -> List[NotificationChannel]:
        """Get default channels for a category."""
        channels = self._category_routes.get(category, [NotificationChannel.CONSOLE])

        # Add email for high priority
        if priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
            if NotificationChannel.EMAIL not in channels:
                channels = channels + [NotificationChannel.EMAIL]

        return channels

    def _is_duplicate(self, notification: Notification) -> bool:
        """Check if notification is a duplicate."""
        # Create hash of notification content
        content = f"{notification.category.value}:{notification.title}:{notification.message}"
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if seen recently
        now = datetime.now()
        if content_hash in self._recent_hashes:
            if self._hash_expiry.get(content_hash, now) > now:
                return True

        # Add to recent hashes with 5 minute expiry
        self._recent_hashes.add(content_hash)
        self._hash_expiry[content_hash] = now + timedelta(minutes=5)

        # Cleanup old hashes
        expired = [h for h, exp in self._hash_expiry.items() if exp < now]
        for h in expired:
            self._recent_hashes.discard(h)
            del self._hash_expiry[h]

        return False

    def _in_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        current_hour = datetime.now().hour
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end

        if start < end:
            return start <= current_hour < end
        else:  # Overnight quiet hours (e.g., 22-07)
            return current_hour >= start or current_hour < end

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        async with self._rate_limit_lock:
            now = datetime.now()

            # Remove old timestamps
            self._sent_timestamps = [
                ts for ts in self._sent_timestamps
                if (now - ts).total_seconds() < 3600
            ]

            # Check hourly limit
            if len(self._sent_timestamps) >= self.config.rate_limit_per_hour:
                return False

            # Check per-minute limit
            minute_ago = now - timedelta(minutes=1)
            recent_count = sum(1 for ts in self._sent_timestamps if ts > minute_ago)
            if recent_count >= self.config.rate_limit_per_minute:
                return False

            # Record timestamp
            self._sent_timestamps.append(now)
            return True

    async def _batch_processor(self):
        """Process batched low-priority notifications."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_interval_seconds)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    async def _flush_batch(self):
        """Send all batched notifications."""
        if not self._batch_queue:
            return

        # Group by category
        by_category: Dict[NotificationCategory, List[Notification]] = defaultdict(list)
        while self._batch_queue:
            notif = self._batch_queue.pop(0)
            by_category[notif.category].append(notif)

        # Send aggregated notifications
        for category, notifications in by_category.items():
            if len(notifications) == 1:
                await self._deliver(notifications[0])
            else:
                # Create aggregated notification
                titles = [n.title for n in notifications]
                aggregated = Notification(
                    notification_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    category=category,
                    priority=NotificationPriority.LOW,
                    title=f"{len(notifications)} {category.value} updates",
                    message="\n".join([f"• {t}" for t in titles[:10]]),
                    channels=notifications[0].channels
                )
                await self._deliver(aggregated)

    # =========================================================================
    # Convenience Methods for Common Notifications
    # =========================================================================

    async def notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
        pnl: Optional[float] = None
    ):
        """Send trade execution notification."""
        pnl_str = f"P&L: ${pnl:+.2f}" if pnl is not None else ""

        await self.notify(
            category=NotificationCategory.TRADE,
            title=f"{side.upper()} {symbol}",
            message=f"Executed {quantity} {symbol} @ ${price:.2f} via {strategy}. {pnl_str}".strip(),
            priority=NotificationPriority.MEDIUM,
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "strategy": strategy,
                "pnl": pnl
            }
        )

    async def notify_risk_alert(
        self,
        alert_type: str,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
        critical: bool = False
    ):
        """Send risk alert notification."""
        await self.notify(
            category=NotificationCategory.RISK,
            title=f"Risk Alert: {alert_type}",
            message=message,
            priority=NotificationPriority.CRITICAL if critical else NotificationPriority.HIGH,
            data=metrics or {}
        )

    async def notify_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float
    ):
        """Send regime change notification."""
        await self.notify(
            category=NotificationCategory.REGIME,
            title="Market Regime Change",
            message=f"Regime shifted from {old_regime} to {new_regime}",
            priority=NotificationPriority.MEDIUM,
            data={
                "previous_regime": old_regime,
                "new_regime": new_regime,
                "confidence": confidence
            }
        )

    async def notify_system_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ):
        """Send system error notification."""
        await self.notify(
            category=NotificationCategory.ERROR,
            title=f"System Error: {error_type}",
            message=error_message,
            priority=NotificationPriority.HIGH,
            data={"stack_trace": stack_trace} if stack_trace else {}
        )

    async def notify_daily_report(
        self,
        total_pnl: float,
        total_trades: int,
        win_rate: float,
        max_drawdown: float,
        top_winners: List[Dict],
        top_losers: List[Dict]
    ):
        """Send daily performance report."""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        message = f"""
Daily Performance Summary:

{pnl_emoji} Total P&L: ${total_pnl:+,.2f}
📊 Trades: {total_trades}
🎯 Win Rate: {win_rate:.1%}
📉 Max Drawdown: {max_drawdown:.2%}

Top Winners: {', '.join([f"{w['symbol']} (+${w['pnl']:.2f})" for w in top_winners[:3]])}
Top Losers: {', '.join([f"{l['symbol']} (-${abs(l['pnl']):.2f})" for l in top_losers[:3]])}
        """.strip()

        await self.notify(
            category=NotificationCategory.REPORT,
            title="Daily Trading Report",
            message=message,
            priority=NotificationPriority.LOW,
            data={
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown
            }
        )

    async def notify_kill_switch(self, action: str, reason: str):
        """Send kill switch activation notification."""
        await self.notify(
            category=NotificationCategory.SYSTEM,
            title=f"Kill Switch: {action.upper()}",
            message=f"Trading {'stopped' if action == 'stop' else 'resumed'}: {reason}",
            priority=NotificationPriority.CRITICAL,
            force=True
        )


# Factory function
def create_notification_manager(
    config: Optional[NotificationConfig] = None
) -> NotificationManager:
    """Create a notification manager with configuration."""
    return NotificationManager(config=config)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def demo():
        # Create manager with console only (no credentials)
        manager = NotificationManager()

        print("=== Notification Manager Demo ===")
        print(f"Available senders: {[ch.value for ch, s in manager.senders.items() if s.is_available()]}")

        # Send test notifications
        await manager.notify_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=45000,
            strategy="momentum",
            pnl=150.00
        )

        await manager.notify_risk_alert(
            alert_type="Drawdown Warning",
            message="Portfolio drawdown approaching 5% threshold",
            metrics={"current_drawdown": 0.045, "threshold": 0.05}
        )

        await manager.notify_regime_change(
            old_regime="trending_bullish",
            new_regime="mean_reverting",
            confidence=0.85
        )

        print(f"\nStats: {manager.stats}")

    asyncio.run(demo())
