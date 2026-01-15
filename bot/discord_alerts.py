"""
Discord Webhook Alerts Module.

Send trading alerts and notifications to Discord channels.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig:
    """Configuration for Discord alerts."""

    webhook_url: Optional[str] = None
    bot_name: str = "Trading Bot"
    avatar_url: str = "https://i.imgur.com/4M34hi2.png"
    enabled: bool = True

    # Alert settings
    alert_on_trade: bool = True
    alert_on_signal: bool = False
    alert_on_error: bool = True
    alert_on_daily_summary: bool = True

    def __post_init__(self):
        if self.webhook_url is None:
            self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")


class DiscordAlerts:
    """
    Discord webhook integration for trading alerts.

    Features:
    - Trade notifications with embeds
    - Daily summaries
    - Error alerts
    - Custom formatted messages
    """

    # Colors for embeds
    COLOR_SUCCESS = 0x00FF00  # Green
    COLOR_DANGER = 0xFF0000   # Red
    COLOR_WARNING = 0xFFAA00  # Orange
    COLOR_INFO = 0x0099FF     # Blue
    COLOR_NEUTRAL = 0x808080  # Gray

    def __init__(self, config: Optional[DiscordConfig] = None):
        self.config = config or DiscordConfig()
        self._session = requests.Session()

    def _send_webhook(
        self,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Send message via webhook."""
        if not self.config.enabled or not self.config.webhook_url:
            return False

        payload = {
            "username": self.config.bot_name,
            "avatar_url": self.config.avatar_url,
        }

        if content:
            payload["content"] = content

        if embeds:
            payload["embeds"] = embeds

        try:
            response = self._session.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return False

    def send_trade_alert(
        self,
        action: str,  # "OPEN" or "CLOSE"
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Send trade notification."""
        if not self.config.alert_on_trade:
            return False

        if action == "OPEN":
            color = self.COLOR_INFO
            title = f"📈 Opened {side.upper()} Position"
            emoji = "🟢" if side.lower() == "long" else "🔴"
        else:
            if pnl and pnl > 0:
                color = self.COLOR_SUCCESS
                title = f"💰 Closed Position - Profit!"
                emoji = "✅"
            elif pnl and pnl < 0:
                color = self.COLOR_DANGER
                title = f"📉 Closed Position - Loss"
                emoji = "❌"
            else:
                color = self.COLOR_NEUTRAL
                title = f"🔄 Closed Position"
                emoji = "⏹️"

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Side", "value": side.upper(), "inline": True},
            {"name": "Quantity", "value": f"{quantity:.6f}", "inline": True},
            {"name": "Price", "value": f"${price:,.2f}", "inline": True},
        ]

        if pnl is not None:
            pnl_str = f"${pnl:+,.2f}"
            if pnl_pct is not None:
                pnl_str += f" ({pnl_pct:+.2f}%)"
            fields.append({"name": "P&L", "value": pnl_str, "inline": True})

        if reason:
            fields.append({"name": "Reason", "value": reason, "inline": True})

        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"{emoji} {self.config.bot_name}"},
        }

        return self._send_webhook(embeds=[embed])

    def send_daily_summary(
        self,
        date: str,
        starting_balance: float,
        ending_balance: float,
        trades_count: int,
        winning_trades: int,
        total_pnl: float,
        positions: List[Dict[str, Any]],
    ) -> bool:
        """Send daily performance summary."""
        if not self.config.alert_on_daily_summary:
            return False

        pnl_pct = ((ending_balance / starting_balance) - 1) * 100 if starting_balance > 0 else 0
        win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0

        if total_pnl > 0:
            color = self.COLOR_SUCCESS
            status = "📈 Profitable Day"
        elif total_pnl < 0:
            color = self.COLOR_DANGER
            status = "📉 Loss Day"
        else:
            color = self.COLOR_NEUTRAL
            status = "➖ Break Even"

        # Position summary
        pos_text = ""
        for pos in positions[:5]:  # Top 5
            pos_text += f"• {pos['symbol']}: ${pos['value']:,.2f}\n"
        if not pos_text:
            pos_text = "No open positions"

        embed = {
            "title": f"📊 Daily Summary - {date}",
            "description": status,
            "color": color,
            "fields": [
                {"name": "Starting Balance", "value": f"${starting_balance:,.2f}", "inline": True},
                {"name": "Ending Balance", "value": f"${ending_balance:,.2f}", "inline": True},
                {"name": "Day P&L", "value": f"${total_pnl:+,.2f} ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "Trades", "value": str(trades_count), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
                {"name": "Winners", "value": f"{winning_trades}/{trades_count}", "inline": True},
                {"name": "Open Positions", "value": pos_text, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": self.config.bot_name},
        }

        return self._send_webhook(embeds=[embed])

    def send_signal_alert(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reason: str,
    ) -> bool:
        """Send signal notification."""
        if not self.config.alert_on_signal:
            return False

        if signal == "BUY":
            color = self.COLOR_SUCCESS
            emoji = "🟢"
        elif signal == "SELL":
            color = self.COLOR_DANGER
            emoji = "🔴"
        else:
            color = self.COLOR_NEUTRAL
            emoji = "⚪"

        embed = {
            "title": f"{emoji} Signal: {signal}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True},
                {"name": "Reason", "value": reason, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_webhook(embeds=[embed])

    def send_error_alert(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
    ) -> bool:
        """Send error notification."""
        if not self.config.alert_on_error:
            return False

        embed = {
            "title": f"⚠️ Error: {error_type}",
            "description": message,
            "color": self.COLOR_WARNING,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if details:
            embed["fields"] = [{"name": "Details", "value": f"```{details[:1000]}```"}]

        return self._send_webhook(embeds=[embed])

    def send_custom_message(
        self,
        title: str,
        message: str,
        color: Optional[int] = None,
    ) -> bool:
        """Send custom message."""
        embed = {
            "title": title,
            "description": message,
            "color": color or self.COLOR_INFO,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_webhook(embeds=[embed])

    def send_startup_message(
        self,
        mode: str,
        balance: float,
        symbols: List[str],
    ) -> bool:
        """Send bot startup notification."""
        embed = {
            "title": "🚀 Trading Bot Started",
            "color": self.COLOR_INFO,
            "fields": [
                {"name": "Mode", "value": mode.upper(), "inline": True},
                {"name": "Balance", "value": f"${balance:,.2f}", "inline": True},
                {"name": "Symbols", "value": ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""), "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": self.config.bot_name},
        }

        return self._send_webhook(embeds=[embed])

    def send_trailing_stop_alert(
        self,
        symbol: str,
        action: str,
        old_stop: float,
        new_stop: float,
        current_price: float,
    ) -> bool:
        """Send trailing stop update notification."""
        embed = {
            "title": f"🎯 Trailing Stop {action}",
            "color": self.COLOR_INFO,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Current Price", "value": f"${current_price:,.2f}", "inline": True},
                {"name": "Old Stop", "value": f"${old_stop:,.2f}", "inline": True},
                {"name": "New Stop", "value": f"${new_stop:,.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_webhook(embeds=[embed])

    def send_dca_alert(
        self,
        symbol: str,
        dca_number: int,
        quantity: float,
        price: float,
        new_average: float,
        drawdown_pct: float,
    ) -> bool:
        """Send DCA execution notification."""
        embed = {
            "title": f"📊 DCA Order #{dca_number}",
            "color": self.COLOR_WARNING,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Drawdown", "value": f"{drawdown_pct:.1%}", "inline": True},
                {"name": "Quantity", "value": f"{quantity:.6f}", "inline": True},
                {"name": "Price", "value": f"${price:,.2f}", "inline": True},
                {"name": "New Average", "value": f"${new_average:,.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_webhook(embeds=[embed])


def create_discord_alerts(
    webhook_url: Optional[str] = None,
) -> DiscordAlerts:
    """Factory function to create Discord alerts."""
    config = DiscordConfig(webhook_url=webhook_url)
    return DiscordAlerts(config)
