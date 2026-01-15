"""
System Event Logger

Centralized logging for tracking system state, bot status, and events.
Provides a single source of truth for what's happening in the system.
"""

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of system events."""
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"

    # Bot events
    BOT_START = "bot_start"
    BOT_STOP = "bot_stop"
    BOT_ERROR = "bot_error"
    BOT_PAUSE = "bot_pause"
    BOT_RESUME = "bot_resume"

    # AI events
    AI_BRAIN_INIT = "ai_brain_init"
    AI_BRAIN_DECISION = "ai_brain_decision"
    AI_STRATEGY_CHANGE = "ai_strategy_change"
    AI_PATTERN_LEARNED = "ai_pattern_learned"

    # ML events
    ML_MODEL_LOADED = "ml_model_loaded"
    ML_PREDICTION = "ml_prediction"
    ML_TRAINING_START = "ml_training_start"
    ML_TRAINING_COMPLETE = "ml_training_complete"

    # Trading events
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    TRADE_ERROR = "trade_error"

    # Risk events
    RISK_LIMIT_HIT = "risk_limit_hit"
    AUTO_PAUSE = "auto_pause"
    DAILY_TARGET_HIT = "daily_target_hit"

    # Config events
    CONFIG_CHANGE = "config_change"
    RISK_SETTINGS_CHANGE = "risk_settings_change"

    # Alert events
    ALERT_SENT = "alert_sent"
    ALERT_FAILED = "alert_failed"

    # Health events
    HEALTH_CHECK = "health_check"
    COMPONENT_ERROR = "component_error"


class Severity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemEvent:
    """A single system event."""
    timestamp: str
    event_type: str
    severity: str
    component: str  # Which component generated this (bot, ai_brain, ml, etc.)
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "component": self.component,
            "message": self.message,
            "details": self.details or {}
        }


class SystemLogger:
    """
    Centralized system event logger.

    Stores events in SQLite for persistence and provides
    easy querying of system state.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "system_log.db")

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

        # Track active components
        self._active_bots: Dict[str, Dict] = {}
        self._active_ai: Dict[str, Dict] = {}
        self._system_start_time: Optional[datetime] = None

    def _init_database(self):
        """Initialize the SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_component
            ON events(component)
        """)

        # Table for tracking active components
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_components (
                component_id TEXT PRIMARY KEY,
                component_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                details TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log(
        self,
        event_type: EventType,
        message: str,
        component: str = "system",
        severity: Severity = Severity.INFO,
        details: Optional[Dict] = None
    ) -> int:
        """Log a system event."""
        timestamp = datetime.now().isoformat()

        event = SystemEvent(
            timestamp=timestamp,
            event_type=event_type.value,
            severity=severity.value,
            component=component,
            message=message,
            details=details
        )

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO events (timestamp, event_type, severity, component, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.event_type,
                event.severity,
                event.component,
                event.message,
                json.dumps(event.details) if event.details else None
            ))

            event_id = cursor.lastrowid
            conn.commit()
            conn.close()

        # Also log to standard logger
        log_func = getattr(logger, severity.value, logger.info)
        log_func(f"[{component}] {message}")

        return event_id

    def register_component(
        self,
        component_id: str,
        component_type: str,
        details: Optional[Dict] = None
    ):
        """Register an active component (bot, AI, etc.)."""
        timestamp = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO active_components
                (component_id, component_type, status, started_at, last_heartbeat, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                component_id,
                component_type,
                "running",
                timestamp,
                timestamp,
                json.dumps(details) if details else None
            ))

            conn.commit()
            conn.close()

        # Track in memory too
        if component_type == "bot":
            self._active_bots[component_id] = {
                "started_at": timestamp,
                "details": details
            }
        elif component_type in ("ai_brain", "ml_model"):
            self._active_ai[component_id] = {
                "started_at": timestamp,
                "details": details
            }

    def update_heartbeat(self, component_id: str):
        """Update the heartbeat for an active component."""
        timestamp = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE active_components
                SET last_heartbeat = ?
                WHERE component_id = ?
            """, (timestamp, component_id))

            conn.commit()
            conn.close()

    def unregister_component(self, component_id: str, reason: str = "stopped"):
        """Unregister a component when it stops."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE active_components
                SET status = ?
                WHERE component_id = ?
            """, (reason, component_id))

            conn.commit()
            conn.close()

        # Remove from memory tracking
        self._active_bots.pop(component_id, None)
        self._active_ai.pop(component_id, None)

    def get_active_components(self) -> Dict[str, List[Dict]]:
        """Get all currently active components."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Consider components stale if no heartbeat in 5 minutes
            stale_threshold = (datetime.now() - timedelta(minutes=5)).isoformat()

            cursor.execute("""
                SELECT component_id, component_type, status, started_at, last_heartbeat, details
                FROM active_components
                WHERE status = 'running' AND last_heartbeat > ?
            """, (stale_threshold,))

            rows = cursor.fetchall()
            conn.close()

        result = {
            "bots": [],
            "ai": [],
            "other": []
        }

        for row in rows:
            component = {
                "id": row[0],
                "type": row[1],
                "status": row[2],
                "started_at": row[3],
                "last_heartbeat": row[4],
                "details": json.loads(row[5]) if row[5] else {}
            }

            if row[1] == "bot":
                result["bots"].append(component)
            elif row[1] in ("ai_brain", "ml_model"):
                result["ai"].append(component)
            else:
                result["other"].append(component)

        return result

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[str] = None
    ) -> List[Dict]:
        """Get recent events with optional filtering."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            if component:
                query += " AND component = ?"
                params.append(component)

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            if since:
                query += " AND timestamp > ?"
                params.append(since)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

        events = []
        for row in rows:
            events.append({
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "severity": row[3],
                "component": row[4],
                "message": row[5],
                "details": json.loads(row[6]) if row[6] else {}
            })

        return events

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active = self.get_active_components()

        # Get recent errors
        recent_errors = self.get_recent_events(
            limit=10,
            severity="error"
        )

        # Get last events by type
        last_events = {}
        for event_type in ["system_start", "bot_start", "ai_brain_init", "trade_open", "trade_close"]:
            events = self.get_recent_events(limit=1, event_type=event_type)
            if events:
                last_events[event_type] = events[0]

        # Count events in last hour
        hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        hourly_events = self.get_recent_events(limit=1000, since=hour_ago)

        event_counts = {}
        for event in hourly_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "active_bots": len(active["bots"]),
            "active_ai": len(active["ai"]),
            "bots": active["bots"],
            "ai_components": active["ai"],
            "recent_errors": recent_errors,
            "last_events": last_events,
            "hourly_event_counts": event_counts,
            "status": "healthy" if len(active["bots"]) > 0 else "no_bots_running"
        }

    def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get a summary of system activity."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        events = self.get_recent_events(limit=10000, since=since)

        # Count by type
        by_type = {}
        by_severity = {}
        by_component = {}

        for event in events:
            t = event["event_type"]
            s = event["severity"]
            c = event["component"]

            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1
            by_component[c] = by_component.get(c, 0) + 1

        return {
            "period_hours": hours,
            "total_events": len(events),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_component": by_component,
            "errors": by_severity.get("error", 0),
            "warnings": by_severity.get("warning", 0)
        }


# Global instance
_system_logger: Optional[SystemLogger] = None


def get_system_logger() -> SystemLogger:
    """Get the global system logger instance."""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    return _system_logger


# Convenience functions
def log_event(
    event_type: EventType,
    message: str,
    component: str = "system",
    severity: Severity = Severity.INFO,
    details: Optional[Dict] = None
) -> int:
    """Log a system event."""
    return get_system_logger().log(event_type, message, component, severity, details)


def log_bot_start(bot_name: str, market: str, details: Optional[Dict] = None):
    """Log bot start event."""
    syslog = get_system_logger()
    syslog.register_component(bot_name, "bot", {"market": market, **(details or {})})
    return syslog.log(
        EventType.BOT_START,
        f"Bot started: {bot_name} for {market}",
        component=bot_name,
        details={"market": market, **(details or {})}
    )


def log_bot_stop(bot_name: str, reason: str = "normal"):
    """Log bot stop event."""
    syslog = get_system_logger()
    syslog.unregister_component(bot_name, reason)
    return syslog.log(
        EventType.BOT_STOP,
        f"Bot stopped: {bot_name} ({reason})",
        component=bot_name,
        details={"reason": reason}
    )


def log_trade(
    action: str,  # "open" or "close"
    symbol: str,
    side: str,
    price: float,
    quantity: float,
    pnl: Optional[float] = None,
    component: str = "trading"
):
    """Log a trade event."""
    event_type = EventType.TRADE_OPEN if action == "open" else EventType.TRADE_CLOSE
    message = f"Trade {action}: {side} {quantity} {symbol} @ ${price:.2f}"
    if pnl is not None:
        message += f" (PnL: ${pnl:.2f})"

    return log_event(
        event_type,
        message,
        component=component,
        details={
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "pnl": pnl
        }
    )


def log_error(message: str, component: str = "system", details: Optional[Dict] = None):
    """Log an error event."""
    return log_event(
        EventType.SYSTEM_ERROR,
        message,
        component=component,
        severity=Severity.ERROR,
        details=details
    )


def heartbeat(component_id: str):
    """Send a heartbeat for a component."""
    get_system_logger().update_heartbeat(component_id)
