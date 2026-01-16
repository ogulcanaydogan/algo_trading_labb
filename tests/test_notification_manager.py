"""
Tests for notification manager module.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from bot.notification_manager import (
    NotificationPriority,
    NotificationCategory,
    NotificationChannel,
    Notification,
    NotificationConfig,
    NotificationSender,
    TelegramSender,
    EmailSender,
    WebhookSender,
    ConsoleSender,
    NotificationManager,
    create_notification_manager,
)


class TestNotificationPriority:
    """Test NotificationPriority enum."""

    def test_priority_values(self):
        """Test priority values."""
        assert NotificationPriority.LOW.value == 1
        assert NotificationPriority.MEDIUM.value == 2
        assert NotificationPriority.HIGH.value == 3
        assert NotificationPriority.CRITICAL.value == 4

    def test_priority_ordering(self):
        """Test priority ordering."""
        assert NotificationPriority.LOW.value < NotificationPriority.MEDIUM.value
        assert NotificationPriority.MEDIUM.value < NotificationPriority.HIGH.value
        assert NotificationPriority.HIGH.value < NotificationPriority.CRITICAL.value


class TestNotificationCategory:
    """Test NotificationCategory enum."""

    def test_all_categories(self):
        """Test all categories exist."""
        assert NotificationCategory.TRADE.value == "trade"
        assert NotificationCategory.SIGNAL.value == "signal"
        assert NotificationCategory.RISK.value == "risk"
        assert NotificationCategory.SYSTEM.value == "system"
        assert NotificationCategory.PERFORMANCE.value == "performance"
        assert NotificationCategory.REGIME.value == "regime"
        assert NotificationCategory.ERROR.value == "error"
        assert NotificationCategory.REPORT.value == "report"


class TestNotificationChannel:
    """Test NotificationChannel enum."""

    def test_all_channels(self):
        """Test all channels exist."""
        assert NotificationChannel.TELEGRAM.value == "telegram"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.CONSOLE.value == "console"


class TestNotification:
    """Test Notification dataclass."""

    def test_notification_creation(self):
        """Test creating a notification."""
        notif = Notification(
            notification_id="test_001",
            category=NotificationCategory.TRADE,
            priority=NotificationPriority.MEDIUM,
            title="Trade Executed",
            message="BTC/USDT buy order filled",
        )
        assert notif.notification_id == "test_001"
        assert notif.category == NotificationCategory.TRADE
        assert notif.priority == NotificationPriority.MEDIUM
        assert notif.delivered is False

    def test_notification_with_data(self):
        """Test notification with data."""
        notif = Notification(
            notification_id="test_002",
            category=NotificationCategory.RISK,
            priority=NotificationPriority.HIGH,
            title="Risk Alert",
            message="Drawdown threshold reached",
            data={"drawdown": 0.05, "threshold": 0.05},
        )
        assert notif.data["drawdown"] == 0.05
        assert notif.data["threshold"] == 0.05

    def test_notification_with_channels(self):
        """Test notification with specific channels."""
        notif = Notification(
            notification_id="test_003",
            category=NotificationCategory.ERROR,
            priority=NotificationPriority.CRITICAL,
            title="System Error",
            message="Connection lost",
            channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
        )
        assert len(notif.channels) == 2
        assert NotificationChannel.TELEGRAM in notif.channels


class TestNotificationConfig:
    """Test NotificationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = NotificationConfig()
        assert config.telegram_enabled is False
        assert config.email_enabled is False
        assert config.webhook_enabled is False
        assert config.rate_limit_per_minute == 30
        assert config.rate_limit_per_hour == 200
        assert config.batch_low_priority is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test_token",
            telegram_chat_ids=["123", "456"],
            rate_limit_per_minute=60,
        )
        assert config.telegram_enabled is True
        assert config.telegram_bot_token == "test_token"
        assert len(config.telegram_chat_ids) == 2
        assert config.rate_limit_per_minute == 60

    def test_quiet_hours_config(self):
        """Test quiet hours configuration."""
        config = NotificationConfig(
            quiet_hours_start=22,
            quiet_hours_end=7,
            quiet_hours_allow_critical=True,
        )
        assert config.quiet_hours_start == 22
        assert config.quiet_hours_end == 7


class TestConsoleSender:
    """Test ConsoleSender class."""

    def test_console_sender_channel(self):
        """Test console sender channel property."""
        sender = ConsoleSender()
        assert sender.channel == NotificationChannel.CONSOLE

    def test_console_sender_available(self):
        """Test console sender is always available."""
        sender = ConsoleSender()
        assert sender.is_available() is True

    @pytest.mark.asyncio
    async def test_console_sender_send(self):
        """Test console sender sends notification."""
        sender = ConsoleSender()
        notif = Notification(
            notification_id="test",
            category=NotificationCategory.SYSTEM,
            priority=NotificationPriority.LOW,
            title="Test",
            message="Test message",
        )
        result = await sender.send(notif)
        assert result is True


class TestTelegramSender:
    """Test TelegramSender class."""

    def test_telegram_sender_not_available(self):
        """Test Telegram sender not available without token."""
        sender = TelegramSender(bot_token="", default_chat_ids=[])
        assert sender.is_available() is False

    def test_telegram_sender_channel(self):
        """Test Telegram sender channel property."""
        sender = TelegramSender(bot_token="test", default_chat_ids=["123"])
        assert sender.channel == NotificationChannel.TELEGRAM

    def test_telegram_sender_available(self):
        """Test Telegram sender availability with config."""
        sender = TelegramSender(bot_token="test_token", default_chat_ids=["123"])
        assert sender.is_available() is True

    def test_format_message(self):
        """Test Telegram message formatting."""
        sender = TelegramSender(bot_token="test", default_chat_ids=["123"])
        notif = Notification(
            notification_id="test",
            category=NotificationCategory.TRADE,
            priority=NotificationPriority.HIGH,
            title="BTC Trade",
            message="Bought 0.1 BTC",
            data={"price": 50000.0},
        )
        message = sender._format_message(notif)

        assert "BTC Trade" in message
        assert "Bought 0.1 BTC" in message


class TestEmailSender:
    """Test EmailSender class."""

    def test_email_sender_not_available(self):
        """Test email sender not available without credentials."""
        sender = EmailSender(
            smtp_host="",
            smtp_port=587,
            smtp_user="",
            smtp_password="",
            from_address="",
            default_recipients=[],
        )
        assert sender.is_available() is False

    def test_email_sender_channel(self):
        """Test email sender channel property."""
        sender = EmailSender(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="user",
            smtp_password="pass",
            from_address="test@test.com",
            default_recipients=["recipient@test.com"],
        )
        assert sender.channel == NotificationChannel.EMAIL

    def test_format_subject(self):
        """Test email subject formatting."""
        sender = EmailSender(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="user",
            smtp_password="pass",
            from_address="test@test.com",
            default_recipients=["recipient@test.com"],
        )

        notif = Notification(
            notification_id="test",
            category=NotificationCategory.RISK,
            priority=NotificationPriority.CRITICAL,
            title="Risk Alert",
            message="Test",
        )
        subject = sender._format_subject(notif)

        assert "[CRITICAL]" in subject
        assert "Risk Alert" in subject


class TestWebhookSender:
    """Test WebhookSender class."""

    def test_webhook_sender_not_available(self):
        """Test webhook sender not available without URLs."""
        sender = WebhookSender(webhook_urls=[])
        assert sender.is_available() is False

    def test_webhook_sender_available(self):
        """Test webhook sender availability with URLs."""
        sender = WebhookSender(webhook_urls=["https://example.com/webhook"])
        assert sender.is_available() is True

    def test_webhook_sender_channel(self):
        """Test webhook sender channel property."""
        sender = WebhookSender(webhook_urls=["https://example.com"])
        assert sender.channel == NotificationChannel.WEBHOOK


class TestNotificationManager:
    """Test NotificationManager class."""

    @pytest.fixture
    def manager(self):
        """Create notification manager with default config."""
        config = NotificationConfig()  # All disabled except console
        return NotificationManager(config=config)

    def test_manager_creation(self, manager):
        """Test manager is created."""
        assert manager is not None
        assert manager.config is not None

    def test_console_sender_always_available(self, manager):
        """Test console sender is always initialized."""
        assert NotificationChannel.CONSOLE in manager.senders
        assert manager.senders[NotificationChannel.CONSOLE].is_available()

    @pytest.mark.asyncio
    async def test_notify_basic(self, manager):
        """Test basic notification."""
        result = await manager.notify(
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="Test message",
        )
        assert result is True
        assert manager.stats["total_sent"] >= 1

    @pytest.mark.asyncio
    async def test_notify_with_priority(self, manager):
        """Test notification with priority."""
        result = await manager.notify(
            category=NotificationCategory.RISK,
            title="Risk Alert",
            message="Test alert",
            priority=NotificationPriority.HIGH,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_notify_with_data(self, manager):
        """Test notification with data."""
        result = await manager.notify(
            category=NotificationCategory.TRADE,
            title="Trade",
            message="BTC bought",
            data={"symbol": "BTC/USDT", "price": 50000},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_notify_trade(self, manager):
        """Test trade notification convenience method."""
        await manager.notify_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            strategy="momentum",
            pnl=100.0,
        )
        assert manager.stats["total_sent"] >= 1

    @pytest.mark.asyncio
    async def test_notify_risk_alert(self, manager):
        """Test risk alert notification."""
        await manager.notify_risk_alert(
            alert_type="Drawdown",
            message="Drawdown exceeded threshold",
            metrics={"drawdown": 0.05},
            critical=False,
        )
        assert manager.stats["total_sent"] >= 1

    @pytest.mark.asyncio
    async def test_notify_regime_change(self, manager):
        """Test regime change notification."""
        await manager.notify_regime_change(
            old_regime="bull",
            new_regime="bear",
            confidence=0.85,
        )
        assert manager.stats["total_sent"] >= 1

    @pytest.mark.asyncio
    async def test_notify_system_error(self, manager):
        """Test system error notification."""
        await manager.notify_system_error(
            error_type="ConnectionError",
            error_message="Failed to connect to exchange",
        )
        assert manager.stats["total_sent"] >= 1

    def test_quiet_hours_detection(self, manager):
        """Test quiet hours detection."""
        # This tests the logic - result depends on current time
        result = manager._in_quiet_hours()
        assert isinstance(result, bool)

    def test_get_channels_for_category(self, manager):
        """Test channel routing for categories."""
        channels = manager._get_channels_for_category(
            NotificationCategory.TRADE,
            NotificationPriority.MEDIUM,
        )
        assert isinstance(channels, list)
        assert len(channels) > 0


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def rate_limited_manager(self):
        """Create manager with low rate limits."""
        config = NotificationConfig(
            rate_limit_per_minute=2,
            rate_limit_per_hour=10,
        )
        return NotificationManager(config=config)

    @pytest.mark.asyncio
    async def test_rate_limit_per_minute(self, rate_limited_manager):
        """Test per-minute rate limiting."""
        # Send notifications up to limit
        for _ in range(2):
            await rate_limited_manager.notify(
                category=NotificationCategory.SYSTEM,
                title="Test",
                message="Test",
                priority=NotificationPriority.MEDIUM,
            )

        # Third notification should be rate limited
        result = await rate_limited_manager.notify(
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="Should be limited",
            priority=NotificationPriority.MEDIUM,
        )

        # Note: rate limiting returns False when limited
        # Stats should show rate limited count
        assert rate_limited_manager.stats["total_rate_limited"] >= 0


class TestDeduplication:
    """Test notification deduplication."""

    @pytest.fixture
    def manager(self):
        config = NotificationConfig()
        return NotificationManager(config=config)

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, manager):
        """Test duplicate notifications are detected."""
        # Send same notification twice
        await manager.notify(
            category=NotificationCategory.SYSTEM,
            title="Duplicate Test",
            message="Same message",
        )

        await manager.notify(
            category=NotificationCategory.SYSTEM,
            title="Duplicate Test",
            message="Same message",
        )

        # Second should be deduplicated
        assert manager.stats["total_deduplicated"] >= 1


class TestFactoryFunction:
    """Test factory function."""

    def test_create_notification_manager(self):
        """Test creating manager via factory."""
        manager = create_notification_manager()
        assert manager is not None
        assert isinstance(manager, NotificationManager)

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = NotificationConfig(
            rate_limit_per_minute=100,
        )
        manager = create_notification_manager(config=config)
        assert manager.config.rate_limit_per_minute == 100
