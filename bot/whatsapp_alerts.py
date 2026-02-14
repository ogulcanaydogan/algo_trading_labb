"""
WhatsApp Trade Alerts Module.

Writes trade alerts to a JSON file that Clawdbot can poll and deliver via WhatsApp.
This provides real-time trade notifications without requiring a direct WhatsApp API integration.

Usage:
    1. The trading engine writes alerts to data/trade_alerts.json
    2. Clawdbot polls this file periodically (via heartbeat or cron)
    3. Clawdbot sends undelivered alerts via WhatsApp
    4. Clawdbot marks alerts as delivered

Configuration:
    Set WHATSAPP_ALERTS_ENABLED=true in environment to enable
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock
import uuid

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of trade alerts."""
    SIGNAL = "signal"
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    DAILY_SUMMARY = "daily_summary"
    RISK_ALERT = "risk_alert"
    SYSTEM_STATUS = "system_status"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TradeAlert:
    """A trade alert message for WhatsApp delivery."""
    id: str
    type: str
    priority: str
    message: str
    created_at: str
    delivered: bool = False
    delivered_at: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        alert_type: AlertType,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
    ) -> "TradeAlert":
        """Create a new alert."""
        return cls(
            id=str(uuid.uuid4())[:8],
            type=alert_type.value,
            priority=priority.value,
            message=message,
            created_at=datetime.now().isoformat(),
            data=data or {},
        )


class WhatsAppAlertManager:
    """
    Manages WhatsApp trade alerts via file-based queue.
    
    Alerts are written to a JSON file that Clawdbot polls.
    This allows the trading bot to run independently while
    Clawdbot handles message delivery.
    """
    
    DEFAULT_ALERT_FILE = Path("data/trade_alerts.json")
    MAX_ALERTS = 100  # Keep last N alerts to prevent file growth
    
    def __init__(
        self,
        alert_file: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ):
        self.alert_file = alert_file or self.DEFAULT_ALERT_FILE
        self.enabled = enabled if enabled is not None else self._check_enabled()
        self._lock = Lock()
        
        # Ensure data directory exists
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if needed
        if not self.alert_file.exists():
            self._save_alerts([])
            
        logger.info(f"WhatsApp alerts {'enabled' if self.enabled else 'disabled'}")
        
    def _check_enabled(self) -> bool:
        """Check if WhatsApp alerts are enabled via environment."""
        return os.getenv("WHATSAPP_ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
    
    def _load_alerts(self) -> List[Dict[str, Any]]:
        """Load alerts from file."""
        try:
            with open(self.alert_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
            
    def _save_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Save alerts to file."""
        # Keep only last N alerts
        alerts = alerts[-self.MAX_ALERTS:]
        with open(self.alert_file, "w") as f:
            json.dump(alerts, f, indent=2)
    
    def add_alert(self, alert: TradeAlert) -> bool:
        """Add an alert to the queue."""
        if not self.enabled:
            return False
            
        with self._lock:
            alerts = self._load_alerts()
            alerts.append(asdict(alert))
            self._save_alerts(alerts)
            
        logger.info(f"WhatsApp alert queued: {alert.type} - {alert.id}")
        return True
    
    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        """Get all undelivered alerts."""
        with self._lock:
            alerts = self._load_alerts()
            return [a for a in alerts if not a.get("delivered", False)]
    
    def mark_delivered(self, alert_id: str) -> bool:
        """Mark an alert as delivered."""
        with self._lock:
            alerts = self._load_alerts()
            for alert in alerts:
                if alert.get("id") == alert_id:
                    alert["delivered"] = True
                    alert["delivered_at"] = datetime.now().isoformat()
                    self._save_alerts(alerts)
                    return True
        return False
    
    def clear_delivered(self) -> int:
        """Remove all delivered alerts. Returns count removed."""
        with self._lock:
            alerts = self._load_alerts()
            pending = [a for a in alerts if not a.get("delivered", False)]
            removed = len(alerts) - len(pending)
            self._save_alerts(pending)
            return removed
    
    # Convenience methods for common alerts
    
    def signal_generated(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        reason: str = "",
    ) -> bool:
        """Queue a signal generation alert."""
        emoji = "📈" if action.upper() in ("BUY", "LONG") else "📉" if action.upper() in ("SELL", "SHORT") else "➖"
        
        # Confidence bar
        filled = int(confidence * 10)
        conf_bar = "█" * filled + "░" * (10 - filled)
        
        message = f"""🔔 *SIGNAL GENERATED*

{emoji} *{action.upper()}* {symbol}

*Price:* ${price:,.2f}
*Confidence:* [{conf_bar}] {confidence:.0%}
"""
        if reason:
            message += f"*Reason:* {reason}\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        alert = TradeAlert.create(
            AlertType.SIGNAL,
            message,
            AlertPriority.NORMAL,
            {"symbol": symbol, "action": action, "confidence": confidence, "price": price},
        )
        return self.add_alert(alert)
    
    def trade_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: Optional[float] = None,
        reason: str = "",
    ) -> bool:
        """Queue a trade entry alert."""
        emoji = "🟢" if side.upper() in ("BUY", "LONG") else "🔴"
        side_text = "LONG" if side.upper() in ("BUY", "LONG") else "SHORT"
        trade_value = price * quantity
        
        message = f"""🔔 *TRADE ALERT*

{emoji} *{side_text}* {symbol}

*Entry:* ${price:,.4f}
*Size:* {quantity:,.4f}
*Value:* ${trade_value:,.2f}
"""
        
        if stop_loss:
            sl_pct = abs((stop_loss - price) / price * 100)
            message += f"*Stop Loss:* ${stop_loss:,.4f} ({sl_pct:.1f}%)\n"
            
        if take_profit:
            tp_pct = abs((take_profit - price) / price * 100)
            message += f"*Take Profit:* ${take_profit:,.4f} ({tp_pct:.1f}%)\n"
            
        if confidence:
            filled = int(confidence * 10)
            conf_bar = "█" * filled + "░" * (10 - filled)
            message += f"*Confidence:* [{conf_bar}] {confidence:.0%}\n"
            
        if reason:
            message += f"*Reason:* {reason[:50]}\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        alert = TradeAlert.create(
            AlertType.TRADE_ENTRY,
            message,
            AlertPriority.HIGH,
            {
                "symbol": symbol,
                "side": side_text,
                "price": price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        )
        return self.add_alert(alert)
    
    def trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        reason: str = "",
        duration_hours: float = 0,
    ) -> bool:
        """Queue a trade exit alert."""
        is_profit = pnl >= 0
        emoji = "✅" if is_profit else "❌"
        result = "WIN" if is_profit else "LOSS"
        side_text = "LONG" if side.upper() in ("BUY", "LONG") else "SHORT"
        
        message = f"""{emoji} *TRADE CLOSED - {result}*

*Symbol:* {symbol} ({side_text})
*Entry:* ${entry_price:,.4f}
*Exit:* ${exit_price:,.4f}

*P&L:* ${pnl:+,.2f} ({pnl_pct:+.2f}%)
"""
        
        if reason:
            reason_emoji = {
                "take_profit": "🎯",
                "stop_loss": "🛑",
                "trailing_stop": "📍",
                "signal": "📶",
                "manual": "✋",
            }.get(reason.lower(), "📤")
            message += f"{reason_emoji} *Exit:* {reason.replace('_', ' ').title()}\n"
            
        if duration_hours > 0:
            if duration_hours < 1:
                duration_str = f"{int(duration_hours * 60)} min"
            elif duration_hours < 24:
                duration_str = f"{duration_hours:.1f} hours"
            else:
                duration_str = f"{duration_hours / 24:.1f} days"
            message += f"⏱️ *Duration:* {duration_str}\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        priority = AlertPriority.NORMAL if is_profit else AlertPriority.HIGH
        
        alert = TradeAlert.create(
            AlertType.TRADE_EXIT,
            message,
            priority,
            {
                "symbol": symbol,
                "side": side_text,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
            },
        )
        return self.add_alert(alert)
    
    def daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_trades: int,
        winning_trades: int,
        open_positions: int = 0,
    ) -> bool:
        """Queue a daily performance summary."""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        if daily_pnl_pct >= 3:
            mood_emoji = "🚀"
            mood = "Excellent"
        elif daily_pnl_pct >= 1:
            mood_emoji = "📈"
            mood = "Good"
        elif daily_pnl_pct >= 0:
            mood_emoji = "➖"
            mood = "Flat"
        elif daily_pnl_pct >= -2:
            mood_emoji = "📉"
            mood = "Tough"
        else:
            mood_emoji = "⚠️"
            mood = "Rough"
        
        # Performance bar
        perf_scaled = min(10, max(0, int((daily_pnl_pct + 5) / 1)))
        perf_bar = "🟢" * perf_scaled + "⚪" * (10 - perf_scaled)
        
        message = f"""{mood_emoji} *DAILY SUMMARY - {mood} Day*

*Balance:* ${balance:,.2f}
*Daily P&L:* ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)

*Trades:* {total_trades}
*Wins:* {winning_trades} ({win_rate:.0f}%)
*Open Positions:* {open_positions}

{perf_bar}

_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        
        alert = TradeAlert.create(
            AlertType.DAILY_SUMMARY,
            message,
            AlertPriority.NORMAL,
            {
                "balance": balance,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
            },
        )
        return self.add_alert(alert)
    
    def risk_alert(
        self,
        alert_type: str,
        message_text: str,
        severity: str = "warning",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Queue a risk alert."""
        severity_emoji = {
            "warning": "⚠️",
            "high": "🔶",
            "critical": "🚨",
        }.get(severity.lower(), "⚠️")
        
        message = f"""{severity_emoji} *RISK ALERT: {alert_type.upper()}*

{message_text}
"""
        
        if metrics:
            message += "\n*Metrics:*\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    message += f"• {key}: {value:.4f}\n"
                else:
                    message += f"• {key}: {value}\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        priority = {
            "warning": AlertPriority.NORMAL,
            "high": AlertPriority.HIGH,
            "critical": AlertPriority.URGENT,
        }.get(severity.lower(), AlertPriority.NORMAL)
        
        alert = TradeAlert.create(
            AlertType.RISK_ALERT,
            message,
            priority,
            {"alert_type": alert_type, "severity": severity, "metrics": metrics},
        )
        return self.add_alert(alert)
    
    def system_status(
        self,
        status: str,
        details: str = "",
        is_error: bool = False,
    ) -> bool:
        """Queue a system status alert."""
        status_emoji = {
            "running": "✅",
            "started": "🚀",
            "stopped": "⏹️",
            "paused": "⏸️",
            "error": "❌",
            "warning": "⚠️",
        }.get(status.lower(), "ℹ️")
        
        message = f"""{status_emoji} *SYSTEM: {status.upper()}*
"""
        if details:
            message += f"\n{details}\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        
        priority = AlertPriority.HIGH if is_error else AlertPriority.NORMAL
        
        alert = TradeAlert.create(
            AlertType.SYSTEM_STATUS,
            message,
            priority,
            {"status": status, "is_error": is_error},
        )
        return self.add_alert(alert)


# Global instance
_alert_manager: Optional[WhatsAppAlertManager] = None


def get_whatsapp_alert_manager() -> WhatsAppAlertManager:
    """Get or create the global WhatsApp alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = WhatsAppAlertManager()
    return _alert_manager


def send_trade_entry_alert(
    symbol: str,
    side: str,
    price: float,
    quantity: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    confidence: Optional[float] = None,
    reason: str = "",
) -> bool:
    """Convenience function to send a trade entry alert."""
    return get_whatsapp_alert_manager().trade_entry(
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        reason=reason,
    )


def send_trade_exit_alert(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    pnl: float,
    pnl_pct: float,
    reason: str = "",
    duration_hours: float = 0,
) -> bool:
    """Convenience function to send a trade exit alert."""
    return get_whatsapp_alert_manager().trade_exit(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        reason=reason,
        duration_hours=duration_hours,
    )


# ============================================================================
# Clawdbot Integration Functions
# ============================================================================

def get_pending_alerts() -> List[Dict[str, Any]]:
    """
    Get pending alerts for Clawdbot to deliver.
    
    Called by Clawdbot to fetch undelivered alerts.
    Returns a list of alert dictionaries.
    """
    return get_whatsapp_alert_manager().get_pending_alerts()


def mark_alert_delivered(alert_id: str) -> bool:
    """
    Mark an alert as delivered.
    
    Called by Clawdbot after successfully sending a WhatsApp message.
    """
    return get_whatsapp_alert_manager().mark_delivered(alert_id)


def format_alert_for_whatsapp(alert: Dict[str, Any]) -> str:
    """
    Format an alert dictionary for WhatsApp delivery.
    
    Returns the message text ready to send.
    """
    return alert.get("message", "")


# ============================================================================
# Trading Engine Integration
# ============================================================================

def create_trade_callback() -> callable:
    """
    Create a callback function for the trading engine.
    
    Register with: engine.on_trade(create_trade_callback())
    """
    manager = get_whatsapp_alert_manager()
    
    def callback(action: str, symbol: str, side: str, result, pnl: float = None):
        """Handle trade events from the unified engine."""
        try:
            if action == "open":
                # Extract data from result
                price = getattr(result, "average_price", 0) or 0
                quantity = getattr(result, "filled_quantity", 0) or 0
                
                manager.trade_entry(
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=quantity,
                    reason="ML Signal",
                )
                
            elif action == "close":
                # Result contains exit details
                exit_price = getattr(result, "average_price", 0) or 0
                quantity = getattr(result, "filled_quantity", 0) or 0
                
                # PnL is passed as parameter for close events
                if pnl is not None:
                    pnl_pct = (pnl / (quantity * exit_price) * 100) if quantity > 0 and exit_price > 0 else 0
                    
                    manager.trade_exit(
                        symbol=symbol,
                        side=side,
                        entry_price=0,  # We don't have this in the callback
                        exit_price=exit_price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                    
        except Exception as e:
            logger.error(f"WhatsApp trade callback error: {e}")
    
    return callback


# ============================================================================
# CLI / Testing
# ============================================================================

def test_alerts():
    """Send test alerts to verify the system works."""
    manager = get_whatsapp_alert_manager()
    
    print("Sending test alerts...")
    
    # Test trade entry
    manager.trade_entry(
        symbol="TSLA",
        side="LONG",
        price=248.50,
        quantity=10,
        stop_loss=245.00,
        take_profit=260.00,
        confidence=0.72,
        reason="ML momentum signal",
    )
    print("✓ Trade entry alert queued")
    
    # Test trade exit
    manager.trade_exit(
        symbol="TSLA",
        side="LONG",
        entry_price=248.50,
        exit_price=255.75,
        quantity=10,
        pnl=72.50,
        pnl_pct=2.92,
        reason="take_profit",
        duration_hours=4.5,
    )
    print("✓ Trade exit alert queued")
    
    # Test daily summary
    manager.daily_summary(
        balance=10725.50,
        daily_pnl=225.50,
        daily_pnl_pct=2.15,
        total_trades=8,
        winning_trades=6,
        open_positions=2,
    )
    print("✓ Daily summary alert queued")
    
    # Show pending
    pending = manager.get_pending_alerts()
    print(f"\n📬 {len(pending)} alerts pending delivery")
    
    for alert in pending:
        print(f"\n{'='*50}")
        print(f"ID: {alert['id']} | Type: {alert['type']} | Priority: {alert['priority']}")
        print(alert['message'])
    
    return pending


if __name__ == "__main__":
    test_alerts()
