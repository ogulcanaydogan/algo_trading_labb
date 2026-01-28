"""
Notification System for Trading Alerts.

Supports:
- Telegram
- Discord
- Email (SMTP)
- Webhook (generic)
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of trading alerts."""

    SYSTEM = "system"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    TRAILING_STOP_HIT = "trailing_stop_hit"
    SIGNAL_GENERATED = "signal_generated"
    REGIME_CHANGE = "regime_change"
    DRAWDOWN_WARNING = "drawdown_warning"
    DAILY_LIMIT_WARNING = "daily_limit_warning"
    CIRCUIT_BREAKER = "circuit_breaker"
    ERROR = "error"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"


@dataclass
class Alert:
    """A trading alert."""

    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "type": self.alert_type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def format_message(self, include_data: bool = True) -> str:
        """Format alert as readable message."""
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ALERT: "🔔",
            AlertLevel.CRITICAL: "🚨",
        }
        emoji = emoji_map.get(self.level, "📢")

        msg = f"{emoji} **{self.title}**\n\n{self.message}"

        if include_data and self.data:
            msg += "\n\n**Details:**\n"
            for key, value in self.data.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                msg += f"• {key}: {value}\n"

        msg += f"\n_{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"
        return msg


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True if successful."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class TelegramChannel(NotificationChannel):
    """Telegram notification channel."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            message = alert.format_message()

            # Send as plain text to avoid formatting issues
            # Strip markdown markers
            message = message.replace("**", "").replace("_", "")

            data = json.dumps(
                {
                    "chat_id": self.chat_id,
                    "text": message,
                }
            ).encode("utf-8")

            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as response:
                return response.status == 200

        except (URLError, Exception) as e:
            print(f"Telegram send failed: {e}")
            return False


class DiscordChannel(NotificationChannel):
    """Discord webhook notification channel."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            # Discord embed colors
            color_map = {
                AlertLevel.INFO: 3447003,  # Blue
                AlertLevel.WARNING: 16776960,  # Yellow
                AlertLevel.ALERT: 15105570,  # Orange
                AlertLevel.CRITICAL: 15158332,  # Red
            }

            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": color_map.get(alert.level, 3447003),
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": f"Alert: {alert.alert_type.value}"},
            }

            if alert.data:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in list(alert.data.items())[:25]  # Discord limit
                ]

            data = json.dumps({"embeds": [embed]}).encode("utf-8")
            req = Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})

            with urlopen(req, timeout=10) as response:
                return response.status in [200, 204]

        except (URLError, Exception) as e:
            print(f"Discord send failed: {e}")
            return False


class EmailChannel(NotificationChannel):
    """Email notification channel via SMTP."""

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
    ):
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", str(smtp_port)))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("ALERT_FROM_EMAIL")
        to_env = os.getenv("ALERT_TO_EMAILS", "")
        self.to_emails = to_emails or [e.strip() for e in to_env.split(",") if e.strip()]

    def is_configured(self) -> bool:
        return all(
            [
                self.smtp_server,
                self.smtp_user,
                self.smtp_password,
                self.from_email,
                self.to_emails,
            ]
        )

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # Plain text version
            text_content = alert.format_message(include_data=True)
            text_content = text_content.replace("**", "").replace("_", "")

            # HTML version
            html_content = self._format_html(alert)

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())

            return True

        except Exception as e:
            print(f"Email send failed: {e}")
            return False

    def _format_html(self, alert: Alert) -> str:
        """Format alert as HTML email."""
        color_map = {
            AlertLevel.INFO: "#3498db",
            AlertLevel.WARNING: "#f39c12",
            AlertLevel.ALERT: "#e67e22",
            AlertLevel.CRITICAL: "#e74c3c",
        }
        color = color_map.get(alert.level, "#3498db")

        data_rows = ""
        if alert.data:
            for key, value in alert.data.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                data_rows += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{alert.title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.8;">{alert.alert_type.value}</p>
            </div>
            <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                <p>{alert.message}</p>
                {f'<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">{data_rows}</table>' if data_rows else ""}
                <p style="color: #888; font-size: 12px; margin-top: 20px;">
                    {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </div>
        </body>
        </html>
        """


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req = Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})

            with urlopen(req, timeout=10) as response:
                return response.status in [200, 201, 204]

        except (URLError, Exception) as e:
            print(f"Webhook send failed: {e}")
            return False


class NotificationManager:
    """
    Central notification manager.

    Manages multiple notification channels and alert filtering.
    """

    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self._alert_history: List[Alert] = []
        self._min_levels: Dict[str, AlertLevel] = {}

        # Auto-configure channels from environment
        self._auto_configure()

    def _auto_configure(self):
        """Auto-configure channels from environment variables."""
        telegram = TelegramChannel()
        if telegram.is_configured():
            self.add_channel("telegram", telegram)

        discord = DiscordChannel()
        if discord.is_configured():
            self.add_channel("discord", discord)

        email = EmailChannel()
        if email.is_configured():
            self.add_channel("email", email)

        webhook = WebhookChannel()
        if webhook.is_configured():
            self.add_channel("webhook", webhook)

    def add_channel(
        self,
        name: str,
        channel: NotificationChannel,
        min_level: AlertLevel = AlertLevel.INFO,
    ):
        """Add a notification channel."""
        self.channels[name] = channel
        self._min_levels[name] = min_level

    def remove_channel(self, name: str):
        """Remove a notification channel."""
        self.channels.pop(name, None)
        self._min_levels.pop(name, None)

    def has_channels(self) -> bool:
        """Check if any notification channels are configured."""
        return len(self.channels) > 0

    def set_min_level(self, channel_name: str, level: AlertLevel):
        """Set minimum alert level for a channel."""
        if channel_name in self.channels:
            self._min_levels[channel_name] = level

    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert to all configured channels.

        Returns dict of channel_name -> success status
        """
        results = {}

        for name, channel in self.channels.items():
            min_level = self._min_levels.get(name, AlertLevel.INFO)
            level_order = list(AlertLevel)

            # Skip if alert level is below channel minimum
            if level_order.index(alert.level) < level_order.index(min_level):
                results[name] = True  # Skipped, not failed
                continue

            results[name] = channel.send(alert)

        # Record alert
        self._alert_history.append(alert)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

        return results

    def notify_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Send trade opened notification."""
        alert = Alert(
            alert_type=AlertType.TRADE_OPENED,
            level=AlertLevel.INFO,
            title=f"Trade Opened: {direction} {symbol}",
            message=f"Opened {direction} position on {symbol} at ${entry_price:,.2f}",
            data={
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "size": size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        )
        return self.send_alert(alert)

    def notify_trade_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> Dict[str, bool]:
        """Send trade closed notification."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING

        alert = Alert(
            alert_type=AlertType.TRADE_CLOSED,
            level=level,
            title=f"Trade Closed: {symbol} {'✅' if pnl >= 0 else '❌'}",
            message=f"Closed {direction} position on {symbol}. P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)",
            data={
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
            },
        )
        return self.send_alert(alert)

    def notify_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        regime: str,
    ) -> Dict[str, bool]:
        """Send signal generated notification."""
        alert = Alert(
            alert_type=AlertType.SIGNAL_GENERATED,
            level=AlertLevel.INFO,
            title=f"Signal: {signal} on {symbol}",
            message=f"Generated {signal} signal for {symbol} with {confidence:.0%} confidence",
            data={
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "market_regime": regime,
            },
        )
        return self.send_alert(alert)

    def notify_regime_change(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
    ) -> Dict[str, bool]:
        """Send market regime change notification."""
        alert = Alert(
            alert_type=AlertType.REGIME_CHANGE,
            level=AlertLevel.WARNING,
            title=f"Regime Change: {symbol}",
            message=f"Market regime changed from {old_regime} to {new_regime}",
            data={
                "symbol": symbol,
                "old_regime": old_regime,
                "new_regime": new_regime,
            },
        )
        return self.send_alert(alert)

    def notify_risk_warning(
        self,
        warning_type: str,
        message: str,
        data: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Send risk warning notification."""
        alert = Alert(
            alert_type=AlertType.DRAWDOWN_WARNING,
            level=AlertLevel.ALERT,
            title=f"Risk Warning: {warning_type}",
            message=message,
            data=data,
        )
        return self.send_alert(alert)

    def notify_circuit_breaker(
        self,
        reason: str,
        duration_hours: int,
    ) -> Dict[str, bool]:
        """Send circuit breaker notification."""
        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            level=AlertLevel.CRITICAL,
            title="Circuit Breaker Triggered",
            message=f"Trading paused for {duration_hours} hours. Reason: {reason}",
            data={
                "reason": reason,
                "duration_hours": duration_hours,
            },
        )
        return self.send_alert(alert)

    def notify_daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades: int,
        win_rate: float,
    ) -> Dict[str, bool]:
        """Send daily summary notification."""
        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            level=AlertLevel.INFO,
            title="Daily Trading Summary",
            message=f"Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)",
            data={
                "balance": balance,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "trades": trades,
                "win_rate": win_rate,
            },
        )
        return self.send_alert(alert)

    def notify_error(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Send error notification."""
        alert = Alert(
            alert_type=AlertType.ERROR,
            level=AlertLevel.CRITICAL,
            title=f"Error: {error_type}",
            message=message,
            data={"details": details} if details else {},
        )
        return self.send_alert(alert)

    def send_critical(
        self,
        message: str,
        title: str = "Critical Alert",
    ) -> Dict[str, bool]:
        """Send a critical-level alert."""
        alert = Alert(
            alert_type=AlertType.ERROR,
            level=AlertLevel.CRITICAL,
            title=title,
            message=message,
        )
        return self.send_alert(alert)

    def get_configured_channels(self) -> List[str]:
        """Get list of configured channel names."""
        return list(self.channels.keys())

    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history."""
        return [a.to_dict() for a in self._alert_history[-limit:]]
