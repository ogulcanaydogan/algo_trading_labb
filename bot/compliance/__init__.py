"""
Compliance module for audit logging and regulatory requirements.

Provides:
- Audit logging with integrity verification
- Order and risk event tracking
- Compliance reporting
"""

from .audit_logging import (
    AuditLogger,
    TradingAuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditWriter,
    FileAuditWriter,
    ConsoleAuditWriter,
    InMemoryAuditWriter,
    create_audit_logger,
)

__all__ = [
    "AuditLogger",
    "TradingAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditWriter",
    "FileAuditWriter",
    "ConsoleAuditWriter",
    "InMemoryAuditWriter",
    "create_audit_logger",
]
