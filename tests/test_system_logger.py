"""
Tests for system logger module.
"""

import pytest
import tempfile
import os
from datetime import datetime

from bot.system_logger import (
    EventType,
    Severity,
    SystemEvent,
    SystemLogger,
    get_system_logger,
    log_event,
    log_bot_start,
    log_bot_stop,
    log_trade,
    log_error,
    heartbeat,
)


class TestEventType:
    """Test EventType enum."""

    def test_system_events(self):
        """Test system event types exist."""
        assert EventType.SYSTEM_START.value == "system_start"
        assert EventType.SYSTEM_STOP.value == "system_stop"
        assert EventType.SYSTEM_ERROR.value == "system_error"

    def test_bot_events(self):
        """Test bot event types exist."""
        assert EventType.BOT_START.value == "bot_start"
        assert EventType.BOT_STOP.value == "bot_stop"
        assert EventType.BOT_ERROR.value == "bot_error"
        assert EventType.BOT_PAUSE.value == "bot_pause"
        assert EventType.BOT_RESUME.value == "bot_resume"

    def test_ai_events(self):
        """Test AI event types exist."""
        assert EventType.AI_BRAIN_INIT.value == "ai_brain_init"
        assert EventType.AI_BRAIN_DECISION.value == "ai_brain_decision"
        assert EventType.AI_STRATEGY_CHANGE.value == "ai_strategy_change"

    def test_ml_events(self):
        """Test ML event types exist."""
        assert EventType.ML_MODEL_LOADED.value == "ml_model_loaded"
        assert EventType.ML_PREDICTION.value == "ml_prediction"
        assert EventType.ML_TRAINING_START.value == "ml_training_start"
        assert EventType.ML_TRAINING_COMPLETE.value == "ml_training_complete"

    def test_trade_events(self):
        """Test trade event types exist."""
        assert EventType.TRADE_OPEN.value == "trade_open"
        assert EventType.TRADE_CLOSE.value == "trade_close"
        assert EventType.TRADE_ERROR.value == "trade_error"

    def test_risk_events(self):
        """Test risk event types exist."""
        assert EventType.RISK_LIMIT_HIT.value == "risk_limit_hit"
        assert EventType.AUTO_PAUSE.value == "auto_pause"
        assert EventType.DAILY_TARGET_HIT.value == "daily_target_hit"


class TestSeverity:
    """Test Severity enum."""

    def test_all_severities(self):
        """Test all severity levels exist."""
        assert Severity.DEBUG.value == "debug"
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"


class TestSystemEvent:
    """Test SystemEvent dataclass."""

    def test_event_creation(self):
        """Test creating a system event."""
        event = SystemEvent(
            timestamp="2024-01-01T12:00:00",
            event_type="bot_start",
            severity="info",
            component="trading_bot",
            message="Bot started",
        )
        assert event.timestamp == "2024-01-01T12:00:00"
        assert event.event_type == "bot_start"
        assert event.severity == "info"
        assert event.component == "trading_bot"
        assert event.message == "Bot started"

    def test_event_with_details(self):
        """Test event with details."""
        event = SystemEvent(
            timestamp="2024-01-01T12:00:00",
            event_type="trade_open",
            severity="info",
            component="trading",
            message="Trade opened",
            details={"symbol": "BTC/USDT", "price": 50000},
        )
        assert event.details["symbol"] == "BTC/USDT"
        assert event.details["price"] == 50000

    def test_to_dict(self):
        """Test event to_dict conversion."""
        event = SystemEvent(
            timestamp="2024-01-01T12:00:00",
            event_type="bot_start",
            severity="info",
            component="bot",
            message="Started",
        )
        d = event.to_dict()
        assert "timestamp" in d
        assert "event_type" in d
        assert "severity" in d
        assert "component" in d
        assert "message" in d
        assert "details" in d

    def test_to_dict_empty_details(self):
        """Test to_dict with no details."""
        event = SystemEvent(
            timestamp="2024-01-01T12:00:00",
            event_type="bot_start",
            severity="info",
            component="bot",
            message="Started",
        )
        d = event.to_dict()
        assert d["details"] == {}


class TestSystemLogger:
    """Test SystemLogger class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def logger(self, temp_db):
        """Create logger with temp database."""
        return SystemLogger(db_path=temp_db)

    def test_logger_creation(self, logger):
        """Test logger is created."""
        assert logger is not None
        assert logger.db_path is not None

    def test_log_event(self, logger):
        """Test logging an event."""
        event_id = logger.log(
            EventType.SYSTEM_START,
            "System started",
            component="system",
            severity=Severity.INFO,
        )
        assert event_id > 0

    def test_log_with_details(self, logger):
        """Test logging with details."""
        event_id = logger.log(
            EventType.TRADE_OPEN,
            "Trade opened",
            component="trading",
            details={"symbol": "BTC/USDT"},
        )
        assert event_id > 0

    def test_register_component(self, logger):
        """Test registering a component."""
        logger.register_component("bot_1", "bot", {"market": "crypto"})
        active = logger.get_active_components()
        assert len(active["bots"]) == 1

    def test_register_ai_component(self, logger):
        """Test registering AI component."""
        logger.register_component("ai_brain_1", "ai_brain", {"model": "v1"})
        active = logger.get_active_components()
        assert len(active["ai"]) == 1

    def test_update_heartbeat(self, logger):
        """Test updating heartbeat."""
        logger.register_component("bot_1", "bot")
        logger.update_heartbeat("bot_1")
        # Should not raise any errors

    def test_unregister_component(self, logger):
        """Test unregistering a component."""
        logger.register_component("bot_1", "bot")
        logger.unregister_component("bot_1", "stopped")
        # Component should be marked as stopped
        assert "bot_1" not in logger._active_bots

    def test_get_recent_events(self, logger):
        """Test getting recent events."""
        logger.log(EventType.SYSTEM_START, "Start")
        logger.log(EventType.BOT_START, "Bot start")

        events = logger.get_recent_events(limit=10)
        assert len(events) == 2

    def test_get_events_filtered_by_type(self, logger):
        """Test getting events filtered by type."""
        logger.log(EventType.SYSTEM_START, "Start")
        logger.log(EventType.BOT_START, "Bot start")
        logger.log(EventType.BOT_START, "Another bot")

        events = logger.get_recent_events(event_type="bot_start")
        assert len(events) == 2

    def test_get_events_filtered_by_component(self, logger):
        """Test getting events filtered by component."""
        logger.log(EventType.BOT_START, "Bot 1", component="bot_1")
        logger.log(EventType.BOT_START, "Bot 2", component="bot_2")

        events = logger.get_recent_events(component="bot_1")
        assert len(events) == 1

    def test_get_events_filtered_by_severity(self, logger):
        """Test getting events filtered by severity."""
        logger.log(EventType.SYSTEM_START, "Info", severity=Severity.INFO)
        logger.log(EventType.SYSTEM_ERROR, "Error", severity=Severity.ERROR)

        events = logger.get_recent_events(severity="error")
        assert len(events) == 1

    def test_get_system_status(self, logger):
        """Test getting system status."""
        logger.register_component("bot_1", "bot")
        logger.log(EventType.SYSTEM_START, "Started")

        status = logger.get_system_status()
        assert "timestamp" in status
        assert "active_bots" in status
        assert "active_ai" in status
        assert "bots" in status
        assert "recent_errors" in status

    def test_get_summary(self, logger):
        """Test getting summary."""
        logger.log(EventType.SYSTEM_START, "Start")
        logger.log(EventType.BOT_START, "Bot")
        logger.log(EventType.SYSTEM_ERROR, "Error", severity=Severity.ERROR)

        summary = logger.get_summary(hours=1)
        assert "total_events" in summary
        assert "by_type" in summary
        assert "by_severity" in summary
        assert "by_component" in summary
        assert summary["total_events"] == 3


class TestConvenienceFunctions:
    """Test convenience logging functions."""

    @pytest.fixture(autouse=True)
    def setup_logger(self):
        """Setup temporary logger for tests."""
        import bot.system_logger as sl

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        sl._system_logger = SystemLogger(db_path=temp_path)
        yield
        os.unlink(temp_path)
        sl._system_logger = None

    def test_get_system_logger(self):
        """Test getting global logger."""
        logger = get_system_logger()
        assert logger is not None
        assert isinstance(logger, SystemLogger)

    def test_log_event_function(self):
        """Test log_event function."""
        event_id = log_event(
            EventType.SYSTEM_START,
            "System started",
        )
        assert event_id > 0

    def test_log_bot_start(self):
        """Test log_bot_start function."""
        event_id = log_bot_start("test_bot", "crypto")
        assert event_id > 0

    def test_log_bot_stop(self):
        """Test log_bot_stop function."""
        log_bot_start("test_bot", "crypto")
        event_id = log_bot_stop("test_bot", "normal")
        assert event_id > 0

    def test_log_trade_open(self):
        """Test log_trade for opening trade."""
        event_id = log_trade(
            action="open",
            symbol="BTC/USDT",
            side="buy",
            price=50000.0,
            quantity=0.1,
        )
        assert event_id > 0

    def test_log_trade_close(self):
        """Test log_trade for closing trade."""
        event_id = log_trade(
            action="close",
            symbol="BTC/USDT",
            side="sell",
            price=51000.0,
            quantity=0.1,
            pnl=100.0,
        )
        assert event_id > 0

    def test_log_error(self):
        """Test log_error function."""
        event_id = log_error(
            "Test error message",
            component="test",
            details={"code": 500},
        )
        assert event_id > 0

    def test_heartbeat(self):
        """Test heartbeat function."""
        log_bot_start("test_bot", "crypto")
        heartbeat("test_bot")
        # Should not raise errors
