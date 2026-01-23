"""
Enhanced Trade Alerts Module.

Provides rich, informative Telegram notifications for trading activities.
Uses HTML formatting for better readability.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)


class AlertCategory(Enum):
    """Alert categories for filtering."""

    TRADE = "trade"
    SIGNAL = "signal"
    RISK = "risk"
    PERFORMANCE = "performance"
    SYSTEM = "system"


@dataclass
class TradeAlertConfig:
    """Configuration for trade alerts."""

    # Alert filtering
    min_trade_value: float = 0.0  # Minimum trade value to alert on
    min_pnl_notify: float = 10.0  # Minimum P&L to send notification

    # Notification preferences
    notify_entries: bool = True
    notify_exits: bool = True
    notify_signals: bool = False  # Can be spammy
    notify_summaries: bool = True

    # Throttling (seconds)
    entry_cooldown: int = 60
    exit_cooldown: int = 0  # Always notify exits
    signal_cooldown: int = 300
    summary_cooldown: int = 3600


class TelegramTradeAlerts:
    """
    Enhanced Telegram alerts for trading.

    Features:
    - HTML formatted messages
    - Trade entry/exit notifications with rich details
    - Daily performance summaries
    - Risk alerts
    - Portfolio status updates
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        config: Optional[TradeAlertConfig] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.config = config or TradeAlertConfig()

        self._last_alerts: Dict[str, datetime] = {}
        self._daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "volume": 0.0,
        }
        self._last_daily_reset = datetime.now().date()

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats at midnight."""
        today = datetime.now().date()
        if today > self._last_daily_reset:
            self._daily_stats = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "volume": 0.0,
            }
            self._last_daily_reset = today

    def _can_send(self, alert_type: str, cooldown: int) -> bool:
        """Check if we can send this alert type."""
        if cooldown <= 0:
            return True

        last = self._last_alerts.get(alert_type)
        if not last:
            return True

        return (datetime.now() - last).total_seconds() >= cooldown

    def _record_alert(self, alert_type: str) -> None:
        """Record that an alert was sent."""
        self._last_alerts[alert_type] = datetime.now()

    def _send_html(self, html_message: str) -> bool:
        """Send HTML formatted message to Telegram."""
        if not self.is_configured():
            logger.warning("Telegram not configured")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

            data = json.dumps(
                {
                    "chat_id": self.chat_id,
                    "text": html_message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
            ).encode("utf-8")

            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as response:
                return response.status == 200

        except (URLError, Exception) as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_trade_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "",
        confidence: float = 0.0,
        market: str = "crypto",
    ) -> bool:
        """
        Send trade entry notification.

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
            confidence: Signal confidence
            market: Market type
        """
        if not self.config.notify_entries:
            return False

        alert_key = f"entry_{symbol}"
        if not self._can_send(alert_key, self.config.entry_cooldown):
            return False

        # Calculate trade value
        trade_value = entry_price * quantity
        if trade_value < self.config.min_trade_value:
            return False

        # Format side emoji
        side_emoji = "🟢" if side.upper() == "LONG" else "🔴"
        side_text = "LONG" if side.upper() == "LONG" else "SHORT"

        # Market emoji
        market_emoji = {
            "crypto": "🪙",
            "stocks": "📈",
            "forex": "💱",
            "commodities": "🛢️",
        }.get(market.lower(), "📊")

        # Build message
        html = f"""
{side_emoji} <b>Trade Opened: {side_text}</b> {market_emoji}

<b>Symbol:</b> {symbol}
<b>Entry:</b> ${entry_price:,.4f}
<b>Size:</b> {quantity:,.4f}
<b>Value:</b> ${trade_value:,.2f}
"""

        if stop_loss:
            sl_pct = abs((stop_loss - entry_price) / entry_price * 100)
            html += f"\n<b>Stop Loss:</b> ${stop_loss:,.4f} ({sl_pct:.2f}%)"

        if take_profit:
            tp_pct = abs((take_profit - entry_price) / entry_price * 100)
            html += f"\n<b>Take Profit:</b> ${take_profit:,.4f} ({tp_pct:.2f}%)"

        if strategy:
            html += f"\n<b>Strategy:</b> {strategy}"

        if confidence > 0:
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            html += f"\n<b>Confidence:</b> [{conf_bar}] {confidence:.0%}"

        html += f"\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str = "",
        duration_hours: float = 0.0,
        market: str = "crypto",
    ) -> bool:
        """
        Send trade exit notification.

        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            pnl: Profit/loss in base currency
            pnl_percent: Profit/loss percentage
            exit_reason: Reason for exit
            duration_hours: Trade duration in hours
            market: Market type
        """
        if not self.config.notify_exits:
            return False

        alert_key = f"exit_{symbol}"
        if not self._can_send(alert_key, self.config.exit_cooldown):
            return False

        # Skip small PnL notifications
        if abs(pnl) < self.config.min_pnl_notify:
            return False

        # Update daily stats
        self._reset_daily_stats_if_needed()
        self._daily_stats["trades"] += 1
        self._daily_stats["pnl"] += pnl
        self._daily_stats["volume"] += exit_price * quantity
        if pnl > 0:
            self._daily_stats["wins"] += 1
        else:
            self._daily_stats["losses"] += 1

        # Format result
        if pnl >= 0:
            result_emoji = "✅"
            result_text = "WIN"
            pnl_color = "profit"
        else:
            result_emoji = "❌"
            result_text = "LOSS"
            pnl_color = "loss"

        side_text = "LONG" if side.upper() == "LONG" else "SHORT"

        # Market emoji
        market_emoji = {
            "crypto": "🪙",
            "stocks": "📈",
            "forex": "💱",
            "commodities": "🛢️",
        }.get(market.lower(), "📊")

        # Exit reason emoji
        reason_emoji = {
            "take_profit": "🎯",
            "stop_loss": "🛑",
            "trailing_stop": "📍",
            "signal": "📶",
            "manual": "✋",
            "timeout": "⏰",
        }.get(exit_reason.lower(), "📤")

        # Build message
        html = f"""
{result_emoji} <b>Trade Closed: {result_text}</b> {market_emoji}

<b>Symbol:</b> {symbol} ({side_text})
<b>Entry:</b> ${entry_price:,.4f}
<b>Exit:</b> ${exit_price:,.4f}

<b>P&L:</b> ${pnl:+,.2f} ({pnl_percent:+.2f}%)
"""

        if exit_reason:
            html += f"\n{reason_emoji} <b>Exit:</b> {exit_reason.replace('_', ' ').title()}"

        if duration_hours > 0:
            if duration_hours < 1:
                duration_str = f"{int(duration_hours * 60)} min"
            elif duration_hours < 24:
                duration_str = f"{duration_hours:.1f} hours"
            else:
                duration_str = f"{duration_hours / 24:.1f} days"
            html += f"\n⏱️ <b>Duration:</b> {duration_str}"

        # Daily stats
        today_win_rate = (
            self._daily_stats["wins"] / self._daily_stats["trades"] * 100
            if self._daily_stats["trades"] > 0
            else 0
        )

        html += f"""

<b>Today:</b> {self._daily_stats["trades"]} trades | ${self._daily_stats["pnl"]:+,.2f} | {today_win_rate:.0f}% win rate

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_signal_alert(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        indicators: Dict[str, Any],
        strategy: str = "",
    ) -> bool:
        """
        Send signal generation alert.

        Args:
            symbol: Trading symbol
            signal: Signal type (LONG, SHORT, FLAT)
            confidence: Signal confidence
            indicators: Key indicator values
            strategy: Strategy name
        """
        if not self.config.notify_signals:
            return False

        alert_key = f"signal_{symbol}"
        if not self._can_send(alert_key, self.config.signal_cooldown):
            return False

        # Signal emoji
        signal_emoji = {
            "LONG": "📈",
            "SHORT": "📉",
            "FLAT": "➖",
        }.get(signal.upper(), "📊")

        # Confidence bar
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

        html = f"""
{signal_emoji} <b>Signal: {signal.upper()}</b>

<b>Symbol:</b> {symbol}
<b>Confidence:</b> [{conf_bar}] {confidence:.0%}
"""

        if strategy:
            html += f"<b>Strategy:</b> {strategy}\n"

        # Key indicators
        if indicators:
            html += "\n<b>Indicators:</b>\n"
            for key, value in list(indicators.items())[:5]:
                if isinstance(value, float):
                    html += f"  • {key}: {value:.4f}\n"
                else:
                    html += f"  • {key}: {value}\n"

        html += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades: int,
        wins: int,
        open_positions: int = 0,
        markets_traded: List[str] = None,
    ) -> bool:
        """
        Send daily performance summary.

        Args:
            balance: Current balance
            daily_pnl: Daily P&L
            daily_pnl_pct: Daily P&L percentage
            trades: Number of trades today
            wins: Number of winning trades
            open_positions: Number of open positions
            markets_traded: List of markets traded
        """
        if not self.config.notify_summaries:
            return False

        alert_key = "daily_summary"
        if not self._can_send(alert_key, self.config.summary_cooldown):
            return False

        # Summary emoji based on performance
        if daily_pnl_pct >= 3:
            header_emoji = "🚀"
            mood = "Excellent"
        elif daily_pnl_pct >= 1:
            header_emoji = "📈"
            mood = "Good"
        elif daily_pnl_pct >= 0:
            header_emoji = "➖"
            mood = "Flat"
        elif daily_pnl_pct >= -2:
            header_emoji = "📉"
            mood = "Tough"
        else:
            header_emoji = "⚠️"
            mood = "Rough"

        win_rate = (wins / trades * 100) if trades > 0 else 0

        html = f"""
{header_emoji} <b>Daily Summary: {mood} Day</b>

<b>Balance:</b> ${balance:,.2f}
<b>Daily P&L:</b> ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)

<b>Trades:</b> {trades}
<b>Wins:</b> {wins} ({win_rate:.0f}%)
<b>Open Positions:</b> {open_positions}
"""

        if markets_traded:
            html += f"\n<b>Markets:</b> {', '.join(markets_traded)}"

        # Performance bar
        perf_scaled = min(10, max(0, int((daily_pnl_pct + 5) / 1)))  # Scale -5% to +5%
        perf_bar = "🟢" * perf_scaled + "⚪" * (10 - perf_scaled)
        html += f"\n\n{perf_bar}"

        html += f"\n\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        metrics: Dict[str, Any] = None,
    ) -> bool:
        """
        Send risk alert notification.

        Args:
            alert_type: Type of risk alert
            message: Alert message
            severity: warning, high, critical
            metrics: Risk metrics
        """
        alert_key = f"risk_{alert_type}"
        if not self._can_send(alert_key, 300):  # 5 min cooldown
            return False

        # Severity emoji
        severity_emoji = {
            "warning": "⚠️",
            "high": "🔶",
            "critical": "🚨",
        }.get(severity.lower(), "⚠️")

        html = f"""
{severity_emoji} <b>Risk Alert: {alert_type.replace("_", " ").title()}</b>

{message}
"""

        if metrics:
            html += "\n<b>Metrics:</b>\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    html += f"  • {key}: {value:.4f}\n"
                else:
                    html += f"  • {key}: {value}\n"

        html += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_portfolio_status(
        self,
        total_balance: float,
        positions: List[Dict[str, Any]],
        daily_pnl: float,
        unrealized_pnl: float,
    ) -> bool:
        """
        Send portfolio status update.

        Args:
            total_balance: Total portfolio value
            positions: List of open positions
            daily_pnl: Realized daily P&L
            unrealized_pnl: Unrealized P&L
        """
        alert_key = "portfolio_status"
        if not self._can_send(alert_key, 3600):  # 1 hour cooldown
            return False

        total_pnl = daily_pnl + unrealized_pnl
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        html = f"""
📊 <b>Portfolio Status</b>

<b>Total Value:</b> ${total_balance:,.2f}
<b>Today's P&L:</b> ${daily_pnl:+,.2f}
<b>Unrealized:</b> ${unrealized_pnl:+,.2f}
{pnl_emoji} <b>Total:</b> ${total_pnl:+,.2f}
"""

        if positions:
            html += f"\n<b>Open Positions ({len(positions)}):</b>\n"
            for pos in positions[:5]:  # Show max 5 positions
                symbol = pos.get("symbol", "???")
                side = "L" if pos.get("side", "").upper() == "LONG" else "S"
                pnl = pos.get("unrealized_pnl", 0)
                pnl_pct = pos.get("pnl_percent", 0)
                emoji = "🟢" if pnl >= 0 else "🔴"
                html += f"  {emoji} {symbol} ({side}): ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"

            if len(positions) > 5:
                html += f"  ... and {len(positions) - 5} more\n"
        else:
            html += "\n<i>No open positions</i>\n"

        html += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        result = self._send_html(html)
        if result:
            self._record_alert(alert_key)

        return result

    def send_system_status(
        self,
        status: str,
        bots_running: int,
        uptime_hours: float,
        errors_today: int = 0,
    ) -> bool:
        """
        Send system status notification.

        Args:
            status: System status (running, stopped, error)
            bots_running: Number of active bots
            uptime_hours: System uptime in hours
            errors_today: Number of errors today
        """
        status_emoji = {
            "running": "✅",
            "stopped": "⏹️",
            "error": "❌",
            "starting": "🔄",
        }.get(status.lower(), "❓")

        # Format uptime
        if uptime_hours < 1:
            uptime_str = f"{int(uptime_hours * 60)} min"
        elif uptime_hours < 24:
            uptime_str = f"{uptime_hours:.1f} hours"
        else:
            uptime_str = f"{uptime_hours / 24:.1f} days"

        html = f"""
{status_emoji} <b>System Status: {status.title()}</b>

<b>Active Bots:</b> {bots_running}
<b>Uptime:</b> {uptime_str}
<b>Errors Today:</b> {errors_today}

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""

        return self._send_html(html)

    def send_daily_target_achieved(
        self,
        current_pnl: float,
        target_pnl: float,
        trades: int,
        wins: int,
    ) -> bool:
        """
        Send daily target achieved alert.

        Args:
            current_pnl: Current daily P&L percentage
            target_pnl: Target P&L percentage
            trades: Number of trades today
            wins: Number of winning trades
        """
        win_rate = (wins / trades * 100) if trades > 0 else 0

        html = f"""
🎯 <b>DAILY TARGET ACHIEVED!</b> 🎉

<b>Current P&L:</b> +{current_pnl:.2f}%
<b>Target:</b> {target_pnl}%
<b>Exceeded by:</b> +{current_pnl - target_pnl:.2f}%

<b>Trades:</b> {trades}
<b>Win Rate:</b> {win_rate:.0f}%

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢

<i>Consider reducing position sizes to protect gains.</i>

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""

        return self._send_html(html)

    def send_loss_limit_alert(
        self,
        current_loss: float,
        max_loss: float,
        trades: int,
        paused: bool = True,
    ) -> bool:
        """
        Send daily loss limit hit alert.

        Args:
            current_loss: Current daily loss percentage (positive number)
            max_loss: Maximum allowed loss percentage
            trades: Number of trades today
            paused: Whether trading was auto-paused
        """
        html = f"""
🚨 <b>DAILY LOSS LIMIT HIT</b> 🚨

<b>Current Loss:</b> -{current_loss:.2f}%
<b>Max Allowed:</b> -{max_loss}%

<b>Trades Today:</b> {trades}
<b>Status:</b> {"⏸️ TRADING PAUSED" if paused else "⚠️ AT RISK"}

🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴

<b>Action Required:</b>
• Review losing trades
• Identify what went wrong
• Wait until tomorrow to resume

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""

        return self._send_html(html)

    def send_auto_pause_alert(
        self,
        reason: str,
        current_pnl: float,
        recommendation: str = "",
    ) -> bool:
        """
        Send auto-pause triggered alert.

        Args:
            reason: Reason for auto-pause
            current_pnl: Current P&L percentage
            recommendation: What to do next
        """
        pnl_emoji = "📈" if current_pnl >= 0 else "📉"

        html = f"""
⏸️ <b>TRADING AUTO-PAUSED</b>

<b>Reason:</b> {reason}
{pnl_emoji} <b>Current P&L:</b> {current_pnl:+.2f}%

"""
        if recommendation:
            html += f"<b>Recommendation:</b> {recommendation}\n\n"

        html += f"""<i>Trading will remain paused until manually resumed or next trading day.</i>

<i>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
"""

        return self._send_html(html)

    def send_message(self, message: str) -> bool:
        """Send a plain text message."""
        if not self.is_configured():
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

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
            logger.error(f"Telegram send failed: {e}")
            return False


# Convenience class alias
TelegramNotifier = TelegramTradeAlerts


def create_trade_alert_manager(
    notify_signals: bool = False,
    min_pnl: float = 10.0,
) -> TelegramTradeAlerts:
    """
    Create a configured trade alert manager.

    Args:
        notify_signals: Whether to notify on signals
        min_pnl: Minimum P&L to notify on exits

    Returns:
        Configured TelegramTradeAlerts instance
    """
    config = TradeAlertConfig(
        notify_signals=notify_signals,
        min_pnl_notify=min_pnl,
    )
    return TelegramTradeAlerts(config=config)
