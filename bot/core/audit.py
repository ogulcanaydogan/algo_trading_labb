"""
Audit logging module for the trading bot.

Provides comprehensive audit logging for:
- Trading decisions and executions
- Mode transitions
- Safety controller actions
- Configuration changes
- Authentication events

Logs are structured JSON for easy parsing and analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Trading events
    TRADE_SIGNAL = "trade_signal"
    TRADE_EXECUTED = "trade_executed"
    TRADE_REJECTED = "trade_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"

    # Mode transitions
    MODE_CHANGE_REQUESTED = "mode_change_requested"
    MODE_CHANGE_APPROVED = "mode_change_approved"
    MODE_CHANGE_REJECTED = "mode_change_rejected"
    MODE_CHANGED = "mode_changed"

    # Safety events
    EMERGENCY_STOP_ACTIVATED = "emergency_stop_activated"
    EMERGENCY_STOP_CLEARED = "emergency_stop_cleared"
    DAILY_LIMIT_REACHED = "daily_limit_reached"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    SAFETY_CHECK_FAILED = "safety_check_failed"

    # Configuration events
    CONFIG_CHANGED = "config_changed"
    STRATEGY_CHANGED = "strategy_changed"
    LIMITS_UPDATED = "limits_updated"

    # Authentication events
    API_ACCESS = "api_access"
    API_AUTH_FAILED = "api_auth_failed"
    API_RATE_LIMITED = "api_rate_limited"

    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    ML_MODEL_LOADED = "ml_model_loaded"
    ML_PREDICTION_MADE = "ml_prediction_made"
    ERROR_OCCURRED = "error_occurred"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: str
    severity: AuditSeverity
    actor: str  # Who/what caused the event (user, system, bot)
    action: str  # Brief description of the action
    details: Dict[str, Any]  # Additional context
    correlation_id: Optional[str] = None  # Link related events
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    checksum: Optional[str] = None  # Integrity verification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, Enum) else self.event_type,
            "timestamp": self.timestamp,
            "severity": self.severity.value if isinstance(self.severity, Enum) else self.severity,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Thread-safe audit logger with file rotation and integrity verification.

    Usage:
        audit = AuditLogger()

        # Log a trade execution
        audit.log_trade(
            action="executed",
            symbol="BTC/USDT",
            side="BUY",
            quantity=0.1,
            price=42000.0,
            order_id="order123"
        )

        # Log a mode change
        audit.log_mode_change(
            from_mode="paper",
            to_mode="testnet",
            approver="admin@example.com"
        )
    """

    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: Optional[Path] = None) -> "AuditLogger":
        """Singleton pattern for audit logger."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_dir: Optional[Path] = None):
        if self._initialized:
            return

        self.log_dir = log_dir or Path(
            os.getenv("DATA_DIR", "./data")
        ) / "audit_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_log_file: Optional[Path] = None
        self._current_date: Optional[str] = None
        self._file_lock = threading.Lock()
        self._event_buffer: List[AuditEvent] = []
        self._buffer_lock = threading.Lock()
        self._max_buffer_size = 100
        self._correlation_context: Dict[str, str] = {}

        self._initialized = True
        logger.info(f"Audit logger initialized, logs at: {self.log_dir}")

    def _get_log_file(self) -> Path:
        """Get the current log file, rotating daily."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_date != today:
            self._current_date = today
            self._current_log_file = self.log_dir / f"audit_{today}.jsonl"

        return self._current_log_file

    def _compute_checksum(self, event: AuditEvent) -> str:
        """Compute integrity checksum for an event."""
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value if isinstance(event.event_type, Enum) else event.event_type,
            "timestamp": event.timestamp,
            "actor": event.actor,
            "action": event.action,
            "details": json.dumps(event.details, sort_keys=True, default=str),
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _write_event(self, event: AuditEvent) -> None:
        """Write an event to the log file."""
        with self._file_lock:
            log_file = self._get_log_file()
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        actor: str = "system",
        severity: AuditSeverity = AuditSeverity.INFO,
        correlation_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """
        Log an audit event.

        Returns the event ID for reference.
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            severity=severity,
            actor=actor,
            action=action,
            details=details or {},
            correlation_id=correlation_id or self._correlation_context.get("current"),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Compute integrity checksum
        event.checksum = self._compute_checksum(event)

        # Write to file
        self._write_event(event)

        # Also log to standard logger for monitoring
        log_level = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)

        logger.log(
            log_level,
            f"[AUDIT] {event_type.value}: {action}",
            extra={"audit_event_id": event_id},
        )

        return event_id

    # =========================================================================
    # Specialized logging methods
    # =========================================================================

    def log_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        strategy: Optional[str] = None,
        confidence: Optional[float] = None,
        pnl: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Log a trade event."""
        event_type = {
            "executed": AuditEventType.TRADE_EXECUTED,
            "rejected": AuditEventType.TRADE_REJECTED,
            "signal": AuditEventType.TRADE_SIGNAL,
        }.get(action.lower(), AuditEventType.TRADE_EXECUTED)

        details = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "reason": reason,
            "strategy": strategy,
            "confidence": confidence,
            "pnl": pnl,
            **kwargs,
        }

        severity = AuditSeverity.INFO
        if action.lower() == "rejected":
            severity = AuditSeverity.WARNING

        return self.log(
            event_type=event_type,
            action=f"Trade {action}: {side} {quantity} {symbol} @ {price}",
            details=details,
            actor="trading_bot",
            severity=severity,
        )

    def log_position(
        self,
        action: str,
        symbol: str,
        quantity: float,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Log a position event."""
        event_type = {
            "opened": AuditEventType.POSITION_OPENED,
            "closed": AuditEventType.POSITION_CLOSED,
            "stop_loss": AuditEventType.STOP_LOSS_HIT,
            "take_profit": AuditEventType.TAKE_PROFIT_HIT,
        }.get(action.lower(), AuditEventType.POSITION_CLOSED)

        details = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            **kwargs,
        }

        return self.log(
            event_type=event_type,
            action=f"Position {action}: {symbol}",
            details=details,
            actor="trading_bot",
        )

    def log_mode_change(
        self,
        from_mode: str,
        to_mode: str,
        status: str = "changed",
        approver: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Log a mode change event."""
        event_type = {
            "requested": AuditEventType.MODE_CHANGE_REQUESTED,
            "approved": AuditEventType.MODE_CHANGE_APPROVED,
            "rejected": AuditEventType.MODE_CHANGE_REJECTED,
            "changed": AuditEventType.MODE_CHANGED,
        }.get(status.lower(), AuditEventType.MODE_CHANGED)

        severity = AuditSeverity.WARNING if "live" in to_mode.lower() else AuditSeverity.INFO

        details = {
            "from_mode": from_mode,
            "to_mode": to_mode,
            "approver": approver,
            "reason": reason,
            **kwargs,
        }

        return self.log(
            event_type=event_type,
            action=f"Mode {status}: {from_mode} -> {to_mode}",
            details=details,
            actor=approver or "system",
            severity=severity,
        )

    def log_safety_event(
        self,
        event: str,
        reason: str,
        current_value: Optional[float] = None,
        limit: Optional[float] = None,
        action_taken: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Log a safety-related event."""
        event_type = {
            "emergency_stop": AuditEventType.EMERGENCY_STOP_ACTIVATED,
            "emergency_clear": AuditEventType.EMERGENCY_STOP_CLEARED,
            "daily_limit": AuditEventType.DAILY_LIMIT_REACHED,
            "position_limit": AuditEventType.POSITION_LIMIT_REACHED,
            "check_failed": AuditEventType.SAFETY_CHECK_FAILED,
        }.get(event.lower(), AuditEventType.SAFETY_CHECK_FAILED)

        severity = AuditSeverity.CRITICAL if "emergency" in event.lower() else AuditSeverity.WARNING

        details = {
            "reason": reason,
            "current_value": current_value,
            "limit": limit,
            "action_taken": action_taken,
            **kwargs,
        }

        return self.log(
            event_type=event_type,
            action=f"Safety event: {event}",
            details=details,
            actor="safety_controller",
            severity=severity,
        )

    def log_config_change(
        self,
        setting: str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "system",
        **kwargs,
    ) -> str:
        """Log a configuration change."""
        details = {
            "setting": setting,
            "old_value": old_value,
            "new_value": new_value,
            **kwargs,
        }

        return self.log(
            event_type=AuditEventType.CONFIG_CHANGED,
            action=f"Config changed: {setting}",
            details=details,
            actor=changed_by,
            severity=AuditSeverity.WARNING,
        )

    def log_api_access(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_key_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Log an API access event."""
        event_type = AuditEventType.API_ACCESS
        severity = AuditSeverity.INFO

        if status_code == 401:
            event_type = AuditEventType.API_AUTH_FAILED
            severity = AuditSeverity.WARNING
        elif status_code == 429:
            event_type = AuditEventType.API_RATE_LIMITED
            severity = AuditSeverity.WARNING

        details = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "api_key_id": api_key_id,
            **kwargs,
        }

        return self.log(
            event_type=event_type,
            action=f"API {method} {endpoint} -> {status_code}",
            details=details,
            actor=api_key_id or "anonymous",
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        traceback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Log an error event."""
        details = {
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
            "context": context or {},
            **kwargs,
        }

        return self.log(
            event_type=AuditEventType.ERROR_OCCURRED,
            action=f"Error: {error_type}",
            details=details,
            actor="system",
            severity=AuditSeverity.CRITICAL,
        )

    # =========================================================================
    # Correlation and context management
    # =========================================================================

    def start_correlation(self) -> str:
        """Start a new correlation context for related events."""
        correlation_id = str(uuid.uuid4())
        self._correlation_context["current"] = correlation_id
        return correlation_id

    def end_correlation(self) -> None:
        """End the current correlation context."""
        self._correlation_context.pop("current", None)

    def with_correlation(self, correlation_id: str):
        """Context manager for correlation."""
        class CorrelationContext:
            def __init__(ctx, audit_logger, cid):
                ctx.audit_logger = audit_logger
                ctx.correlation_id = cid
                ctx.previous_id = None

            def __enter__(ctx):
                ctx.previous_id = ctx.audit_logger._correlation_context.get("current")
                ctx.audit_logger._correlation_context["current"] = ctx.correlation_id
                return ctx.correlation_id

            def __exit__(ctx, *args):
                if ctx.previous_id:
                    ctx.audit_logger._correlation_context["current"] = ctx.previous_id
                else:
                    ctx.audit_logger._correlation_context.pop("current", None)

        return CorrelationContext(self, correlation_id)

    # =========================================================================
    # Query and retrieval
    # =========================================================================

    def get_events(
        self,
        date: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events from logs.

        Args:
            date: Date string (YYYY-MM-DD) or None for today
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum events to return

        Returns:
            List of audit events as dictionaries
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        log_file = self.log_dir / f"audit_{date}.jsonl"

        if not log_file.exists():
            return []

        events = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        if event_type and event.get("event_type") != event_type.value:
                            continue
                        if severity and event.get("severity") != severity.value:
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.error(f"Error reading audit log: {e}")

        return events

    def verify_integrity(self, event: Dict[str, Any]) -> bool:
        """Verify the integrity of an audit event."""
        stored_checksum = event.get("checksum")
        if not stored_checksum:
            return False

        # Recreate the checksum
        data = {
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
            "actor": event["actor"],
            "action": event["action"],
            "details": json.dumps(event.get("details", {}), sort_keys=True, default=str),
        }
        content = json.dumps(data, sort_keys=True)
        computed = hashlib.sha256(content.encode()).hexdigest()[:16]

        return computed == stored_checksum


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
