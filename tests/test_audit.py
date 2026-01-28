"""
Tests for the audit logging module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bot.core.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    get_audit_logger,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = AuditEvent(
            event_id="test-123",
            event_type=AuditEventType.TRADE_EXECUTED,
            timestamp="2024-01-15T10:30:00Z",
            severity=AuditSeverity.INFO,
            actor="system",
            action="Test action",
            details={"key": "value"},
        )

        d = event.to_dict()
        assert d["event_id"] == "test-123"
        assert d["event_type"] == "trade_executed"
        assert d["severity"] == "info"
        assert d["details"] == {"key": "value"}

    def test_to_json(self):
        """Test conversion to JSON."""
        event = AuditEvent(
            event_id="test-456",
            event_type=AuditEventType.MODE_CHANGED,
            timestamp="2024-01-15T10:30:00Z",
            severity=AuditSeverity.WARNING,
            actor="admin",
            action="Mode changed",
            details={"from": "paper", "to": "testnet"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "test-456"
        assert parsed["event_type"] == "mode_changed"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def setup_method(self):
        """Reset singleton for each test."""
        AuditLogger._instance = None

    def test_singleton_pattern(self):
        """Test that AuditLogger is a singleton."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger1 = AuditLogger(log_dir)
            logger2 = AuditLogger(log_dir)
            assert logger1 is logger2

    def test_log_basic_event(self):
        """Test logging a basic event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log(
                event_type=AuditEventType.SYSTEM_STARTED,
                action="System started",
                details={"version": "1.0.0"},
            )

            assert event_id is not None
            assert len(event_id) == 36  # UUID length

    def test_log_trade(self):
        """Test logging a trade event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log_trade(
                action="executed",
                symbol="BTC/USDT",
                side="BUY",
                quantity=0.1,
                price=42000.0,
                order_id="order123",
                strategy="momentum",
                confidence=0.85,
            )

            assert event_id is not None

            # Verify event was written
            events = audit.get_events()
            assert len(events) >= 1

            trade_event = events[-1]
            assert trade_event["event_type"] == "trade_executed"
            assert trade_event["details"]["symbol"] == "BTC/USDT"
            assert trade_event["details"]["quantity"] == 0.1

    def test_log_position(self):
        """Test logging position events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            # Open position
            event_id = audit.log_position(
                action="opened",
                symbol="ETH/USDT",
                quantity=1.5,
                entry_price=2500.0,
            )
            assert event_id is not None

            # Close position
            event_id = audit.log_position(
                action="closed",
                symbol="ETH/USDT",
                quantity=1.5,
                entry_price=2500.0,
                exit_price=2600.0,
                pnl=150.0,
                reason="Take profit",
            )
            assert event_id is not None

    def test_log_mode_change(self):
        """Test logging mode change events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log_mode_change(
                from_mode="paper",
                to_mode="testnet",
                status="approved",
                approver="admin@example.com",
                reason="Passed paper trading requirements",
            )

            assert event_id is not None

            events = audit.get_events()
            mode_event = events[-1]
            assert mode_event["event_type"] == "mode_change_approved"
            assert mode_event["details"]["from_mode"] == "paper"
            assert mode_event["details"]["to_mode"] == "testnet"

    def test_log_safety_event(self):
        """Test logging safety events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log_safety_event(
                event="emergency_stop",
                reason="Daily loss limit exceeded",
                current_value=600.0,
                limit=500.0,
                action_taken="All positions closed",
            )

            assert event_id is not None

            events = audit.get_events()
            safety_event = events[-1]
            assert safety_event["event_type"] == "emergency_stop_activated"
            assert safety_event["severity"] == "critical"

    def test_log_config_change(self):
        """Test logging configuration changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log_config_change(
                setting="max_position_size_usd",
                old_value=1000.0,
                new_value=500.0,
                changed_by="admin",
            )

            assert event_id is not None

            events = audit.get_events()
            config_event = events[-1]
            assert config_event["event_type"] == "config_changed"
            assert config_event["details"]["old_value"] == 1000.0
            assert config_event["details"]["new_value"] == 500.0

    def test_log_api_access(self):
        """Test logging API access events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            # Successful access
            event_id = audit.log_api_access(
                endpoint="/api/status",
                method="GET",
                status_code=200,
                ip_address="192.168.1.1",
                api_key_id="key123",
            )
            assert event_id is not None

            # Auth failure
            event_id = audit.log_api_access(
                endpoint="/api/trade",
                method="POST",
                status_code=401,
                ip_address="10.0.0.1",
            )
            assert event_id is not None

            events = audit.get_events()
            auth_fail_event = events[-1]
            assert auth_fail_event["event_type"] == "api_auth_failed"
            assert auth_fail_event["severity"] == "warning"

    def test_log_error(self):
        """Test logging error events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            event_id = audit.log_error(
                error_type="ConnectionError",
                message="Failed to connect to exchange",
                traceback="Traceback...",
                context={"exchange": "binance"},
            )

            assert event_id is not None

            events = audit.get_events()
            error_event = events[-1]
            assert error_event["event_type"] == "error_occurred"
            assert error_event["severity"] == "critical"

    def test_integrity_verification(self):
        """Test event integrity verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            audit.log(
                event_type=AuditEventType.SYSTEM_STARTED,
                action="Test event",
                details={"test": True},
            )

            events = audit.get_events()
            assert len(events) >= 1

            event = events[-1]
            assert audit.verify_integrity(event) is True

            # Tamper with the event
            event["action"] = "Tampered action"
            assert audit.verify_integrity(event) is False

    def test_correlation_context(self):
        """Test correlation ID management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            # Start correlation
            corr_id = audit.start_correlation()

            audit.log(
                event_type=AuditEventType.TRADE_SIGNAL,
                action="Signal generated",
            )
            audit.log(
                event_type=AuditEventType.TRADE_EXECUTED,
                action="Trade executed",
            )

            audit.end_correlation()

            events = audit.get_events()
            assert len(events) >= 2

            # Both events should have the same correlation ID
            assert events[-1]["correlation_id"] == corr_id
            assert events[-2]["correlation_id"] == corr_id

    def test_correlation_context_manager(self):
        """Test correlation context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            corr_id = "custom-correlation-id"

            with audit.with_correlation(corr_id):
                audit.log(
                    event_type=AuditEventType.TRADE_SIGNAL,
                    action="Signal in context",
                )

            # After context, correlation should be cleared
            audit.log(
                event_type=AuditEventType.SYSTEM_STARTED,
                action="Outside context",
            )

            events = audit.get_events()
            # The last event should not have the custom correlation ID
            assert events[-1].get("correlation_id") != corr_id

    def test_get_events_with_filters(self):
        """Test retrieving events with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            log_dir = Path(tmpdir)
            audit = AuditLogger(log_dir)

            # Log various events
            audit.log(
                event_type=AuditEventType.TRADE_EXECUTED,
                action="Trade 1",
                severity=AuditSeverity.INFO,
            )
            audit.log(
                event_type=AuditEventType.SAFETY_CHECK_FAILED,
                action="Safety check",
                severity=AuditSeverity.WARNING,
            )
            audit.log(
                event_type=AuditEventType.TRADE_EXECUTED,
                action="Trade 2",
                severity=AuditSeverity.INFO,
            )

            # Filter by event type
            trade_events = audit.get_events(
                event_type=AuditEventType.TRADE_EXECUTED
            )
            assert len(trade_events) == 2

            # Filter by severity
            warning_events = audit.get_events(
                severity=AuditSeverity.WARNING
            )
            assert len(warning_events) == 1


class TestGetAuditLogger:
    """Tests for get_audit_logger helper."""

    def setup_method(self):
        """Reset singleton for each test."""
        AuditLogger._instance = None

    def test_returns_singleton(self):
        """Test that get_audit_logger returns singleton."""
        # We need to reset the global too
        import bot.core.audit as audit_module
        audit_module._audit_logger = None

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2
