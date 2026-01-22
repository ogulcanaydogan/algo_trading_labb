"""
Audit Logging for Compliance and Security.

Provides comprehensive audit trail for all trading activities
and system changes for regulatory compliance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Trading events
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_MODIFIED = "order.modified"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_MODIFIED = "position.modified"

    # Risk events
    RISK_LIMIT_BREACH = "risk.limit_breach"
    RISK_LIMIT_WARNING = "risk.limit_warning"
    MARGIN_CALL = "risk.margin_call"
    STOP_LOSS_TRIGGERED = "risk.stop_loss"

    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config_change"
    STRATEGY_ENABLED = "system.strategy_enabled"
    STRATEGY_DISABLED = "system.strategy_disabled"

    # Security events
    LOGIN_SUCCESS = "security.login_success"
    LOGIN_FAILURE = "security.login_failure"
    API_KEY_CREATED = "security.api_key_created"
    API_KEY_REVOKED = "security.api_key_revoked"
    PERMISSION_CHANGE = "security.permission_change"

    # Data events
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    BACKUP_CREATED = "data.backup_created"

    # Compliance events
    COMPLIANCE_CHECK = "compliance.check"
    REGULATORY_REPORT = "compliance.regulatory_report"

    # Custom event
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An audit log entry."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    actor: str  # Who performed the action
    action: str  # What was done
    resource: str  # What was affected
    details: Dict[str, Any]
    outcome: str  # "success", "failure", "pending"
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        data = f"{self.id}|{self.timestamp.isoformat()}|{self.event_type.value}|{self.actor}|{self.action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "AuditEvent":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditSeverity(data["severity"]),
            actor=data["actor"],
            action=data["action"],
            resource=data["resource"],
            details=data["details"],
            outcome=data["outcome"],
            ip_address=data.get("ip_address"),
            session_id=data.get("session_id"),
            trace_id=data.get("trace_id"),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


class AuditWriter:
    """Base class for audit log writers."""

    def write(self, event: AuditEvent):
        """Write an audit event."""
        raise NotImplementedError

    def flush(self):
        """Flush any buffered events."""
        pass

    def close(self):
        """Close the writer."""
        pass


class FileAuditWriter(AuditWriter):
    """Write audit logs to file."""

    def __init__(
        self,
        log_dir: str,
        rotate_daily: bool = True,
        max_file_size_mb: int = 100
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_daily = rotate_daily
        self.max_file_size_mb = max_file_size_mb
        self._current_file = None
        self._current_date = None
        self._lock = threading.Lock()

    def _get_file(self) -> str:
        """Get current log file path."""
        today = datetime.now().date()
        if self.rotate_daily and self._current_date != today:
            self._current_date = today
            self._current_file = self.log_dir / f"audit_{today.isoformat()}.jsonl"
        elif not self._current_file:
            self._current_file = self.log_dir / "audit.jsonl"
        return str(self._current_file)

    def write(self, event: AuditEvent):
        with self._lock:
            file_path = self._get_file()
            with open(file_path, "a") as f:
                f.write(event.to_json() + "\n")

    def read_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Read events from log files."""
        events = []

        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent.from_dict(data)

                        if start_date and event.timestamp < start_date:
                            continue
                        if end_date and event.timestamp > end_date:
                            continue

                        events.append(event)
                    except Exception as e:
                        logger.warning(f"Failed to parse audit line: {e}")

        return events


class ConsoleAuditWriter(AuditWriter):
    """Write audit logs to console."""

    def __init__(self, min_severity: AuditSeverity = AuditSeverity.INFO):
        self.min_severity = min_severity
        self._severity_order = [s for s in AuditSeverity]

    def write(self, event: AuditEvent):
        if self._severity_order.index(event.severity) >= \
           self._severity_order.index(self.min_severity):
            print(
                f"[AUDIT] {event.timestamp.isoformat()} "
                f"[{event.severity.value.upper()}] "
                f"{event.event_type.value}: {event.action} "
                f"by {event.actor} -> {event.outcome}"
            )


class InMemoryAuditWriter(AuditWriter):
    """Store audit logs in memory (for testing/development)."""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()

    def write(self, event: AuditEvent):
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query stored events."""
        with self._lock:
            events = self._events.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor:
            events = [e for e in events if e.actor == actor]
        if resource:
            events = [e for e in events if resource in e.resource]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]

    def clear(self):
        """Clear all events."""
        with self._lock:
            self._events.clear()


class AuditLogger:
    """
    Main audit logging system.

    Provides:
    - Structured audit event logging
    - Multiple output targets (file, console, memory)
    - Event integrity verification
    - Query capabilities
    """

    def __init__(
        self,
        service_name: str = "trading-system",
        writers: Optional[List[AuditWriter]] = None
    ):
        self.service_name = service_name
        self._writers = writers or []
        self._context: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_writer(self, writer: AuditWriter):
        """Add an audit writer."""
        self._writers.append(writer)

    def set_context(self, **kwargs):
        """Set context that will be included in all events."""
        with self._lock:
            self._context.update(kwargs)

    def clear_context(self, *keys):
        """Clear context keys."""
        with self._lock:
            for key in keys:
                self._context.pop(key, None)

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str,
        actor: str = "system",
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: str = "success",
        details: Optional[Dict] = None,
        **kwargs
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Description of action
            resource: Resource affected
            actor: Who performed the action
            severity: Event severity
            outcome: Result of action
            details: Additional details
            **kwargs: Additional event fields
        """
        event = AuditEvent(
            id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            actor=actor,
            action=action,
            resource=resource,
            details=details or {},
            outcome=outcome,
            ip_address=kwargs.get("ip_address"),
            session_id=kwargs.get("session_id") or self._context.get("session_id"),
            trace_id=kwargs.get("trace_id") or self._context.get("trace_id"),
            metadata={
                "service": self.service_name,
                **self._context,
                **kwargs.get("metadata", {}),
            },
        )

        # Write to all writers
        for writer in self._writers:
            try:
                writer.write(event)
            except Exception as e:
                logger.error(f"Failed to write audit event: {e}")

        return event

    # Convenience methods for common events

    def log_order(
        self,
        event_type: AuditEventType,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        actor: str = "system",
        outcome: str = "success",
        **kwargs
    ):
        """Log order-related event."""
        return self.log(
            event_type=event_type,
            action=f"{event_type.value.split('.')[1]} order",
            resource=f"order:{order_id}",
            actor=actor,
            outcome=outcome,
            details={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                **kwargs,
            },
        )

    def log_risk_event(
        self,
        event_type: AuditEventType,
        description: str,
        current_value: float,
        limit_value: float,
        symbol: Optional[str] = None,
        **kwargs
    ):
        """Log risk-related event."""
        severity = AuditSeverity.WARNING
        if event_type == AuditEventType.RISK_LIMIT_BREACH:
            severity = AuditSeverity.ERROR
        elif event_type == AuditEventType.MARGIN_CALL:
            severity = AuditSeverity.CRITICAL

        return self.log(
            event_type=event_type,
            action=description,
            resource=f"risk:{symbol or 'portfolio'}",
            severity=severity,
            outcome="triggered",
            details={
                "current_value": current_value,
                "limit_value": limit_value,
                "symbol": symbol,
                "breach_percentage": (current_value / limit_value - 1) * 100 if limit_value else 0,
                **kwargs,
            },
        )

    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        actor: str = "system",
        **kwargs
    ):
        """Log configuration change."""
        return self.log(
            event_type=AuditEventType.CONFIG_CHANGE,
            action=f"Changed configuration: {config_key}",
            resource=f"config:{config_key}",
            actor=actor,
            details={
                "key": config_key,
                "old_value": str(old_value),
                "new_value": str(new_value),
                **kwargs,
            },
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        description: str,
        actor: str,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """Log security-related event."""
        severity = AuditSeverity.INFO
        if outcome == "failure":
            severity = AuditSeverity.WARNING
        if event_type in [AuditEventType.API_KEY_REVOKED, AuditEventType.PERMISSION_CHANGE]:
            severity = AuditSeverity.WARNING

        return self.log(
            event_type=event_type,
            action=description,
            resource="security",
            actor=actor,
            severity=severity,
            outcome=outcome,
            ip_address=ip_address,
            details=kwargs,
        )


class TradingAuditLogger(AuditLogger):
    """
    Specialized audit logger for trading systems.

    Pre-configured for common trading audit requirements.
    """

    def __init__(
        self,
        log_dir: str = "data/audit_logs",
        enable_console: bool = False,
        enable_file: bool = True,
        enable_memory: bool = True
    ):
        super().__init__(service_name="trading-system")

        if enable_file:
            self.add_writer(FileAuditWriter(log_dir))

        if enable_console:
            self.add_writer(ConsoleAuditWriter())

        if enable_memory:
            self._memory_writer = InMemoryAuditWriter()
            self.add_writer(self._memory_writer)
        else:
            self._memory_writer = None

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query recent events from memory."""
        if not self._memory_writer:
            return []
        return self._memory_writer.get_events(
            event_type=event_type,
            actor=actor,
            resource=resource,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def get_order_history(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get order audit history."""
        events = self.query_events(limit=limit * 4)

        # Filter for order events
        order_types = [
            AuditEventType.ORDER_PLACED,
            AuditEventType.ORDER_FILLED,
            AuditEventType.ORDER_CANCELLED,
            AuditEventType.ORDER_REJECTED,
            AuditEventType.ORDER_MODIFIED,
        ]

        order_events = [e for e in events if e.event_type in order_types]

        if order_id:
            order_events = [
                e for e in order_events
                if e.details.get("order_id") == order_id
            ]

        if symbol:
            order_events = [
                e for e in order_events
                if e.details.get("symbol") == symbol
            ]

        return [e.to_dict() for e in order_events[:limit]]

    def get_risk_events(self, limit: int = 50) -> List[Dict]:
        """Get recent risk events."""
        events = self.query_events(limit=limit * 2)

        risk_types = [
            AuditEventType.RISK_LIMIT_BREACH,
            AuditEventType.RISK_LIMIT_WARNING,
            AuditEventType.MARGIN_CALL,
            AuditEventType.STOP_LOSS_TRIGGERED,
        ]

        risk_events = [e for e in events if e.event_type in risk_types]
        return [e.to_dict() for e in risk_events[:limit]]

    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate audit report for a period."""
        events = self.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )

        # Summarize by type
        by_type: Dict[str, int] = {}
        by_actor: Dict[str, int] = {}
        by_outcome: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_actor[event.actor] = by_actor.get(event.actor, 0) + 1
            by_outcome[event.outcome] = by_outcome.get(event.outcome, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_events": len(events),
            "by_type": by_type,
            "by_actor": by_actor,
            "by_outcome": by_outcome,
            "by_severity": by_severity,
            "generated_at": datetime.now().isoformat(),
        }


def create_audit_logger(
    log_dir: str = "data/audit_logs",
    enable_console: bool = False,
    enable_file: bool = True
) -> TradingAuditLogger:
    """Factory function to create trading audit logger."""
    return TradingAuditLogger(
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file,
    )
