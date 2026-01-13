"""Unit tests for logging configuration module."""
from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from bot.logging_config import (
    JSONFormatter,
    StructuredLogger,
    get_structured_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_format_basic_message(self):
        """Test basic message formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"

    def test_format_includes_timestamp(self):
        """Test that timestamp is included."""
        formatter = JSONFormatter(include_timestamp=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format check

    def test_format_excludes_timestamp_when_disabled(self):
        """Test that timestamp can be excluded."""
        formatter = JSONFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_format_includes_location(self):
        """Test that location info is included."""
        formatter = JSONFormatter(include_location=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_function"

        result = formatter.format(record)
        data = json.loads(result)

        assert "location" in data
        assert data["location"]["line"] == 42
        assert data["location"]["function"] == "test_function"

    def test_format_excludes_location_when_disabled(self):
        """Test that location can be excluded."""
        formatter = JSONFormatter(include_location=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "location" not in data

    def test_format_with_extra_fields(self):
        """Test that extra fields are included."""
        formatter = JSONFormatter(extra_fields={"service": "test-service", "env": "test"})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["service"] == "test-service"
        assert data["env"] == "test"

    def test_format_with_exception(self):
        """Test that exception info is formatted."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_format_with_extra_data(self):
        """Test that extra_data attribute is included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"symbol": "BTC/USDT", "price": 50000.0}

        result = formatter.format(record)
        data = json.loads(result)

        assert "data" in data
        assert data["data"]["symbol"] == "BTC/USDT"
        assert data["data"]["price"] == 50000.0


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_info_with_data(self):
        """Test info_with_data method."""
        logging.setLoggerClass(StructuredLogger)
        logger = logging.getLogger("test.structured")
        logger.setLevel(logging.DEBUG)

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            logger.info_with_data("Signal generated", symbol="BTC/USDT", decision="LONG")

            output = stream.getvalue()
            data = json.loads(output)

            assert data["message"] == "Signal generated"
            assert data["data"]["symbol"] == "BTC/USDT"
            assert data["data"]["decision"] == "LONG"
        finally:
            logger.removeHandler(handler)

    def test_debug_with_data(self):
        """Test debug_with_data method."""
        logging.setLoggerClass(StructuredLogger)
        logger = logging.getLogger("test.structured.debug")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            logger.debug_with_data("Debug info", key="value")

            output = stream.getvalue()
            data = json.loads(output)

            assert data["level"] == "DEBUG"
            assert data["data"]["key"] == "value"
        finally:
            logger.removeHandler(handler)

    def test_error_with_data(self):
        """Test error_with_data method."""
        logging.setLoggerClass(StructuredLogger)
        logger = logging.getLogger("test.structured.error")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            logger.error_with_data("Error occurred", error_code=500)

            output = stream.getvalue()
            data = json.loads(output)

            assert data["level"] == "ERROR"
            assert data["data"]["error_code"] == 500
        finally:
            logger.removeHandler(handler)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_json_format(self):
        """Test setting up JSON format logging."""
        setup_logging(level="INFO", json_format=True)

        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

        # Check that handler has JSON formatter
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_setup_text_format(self):
        """Test setting up text format logging."""
        setup_logging(level="DEBUG", json_format=False)

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

        # Check that handler has standard formatter
        handler = logger.handlers[0]
        assert not isinstance(handler.formatter, JSONFormatter)

    def test_setup_with_log_file(self, tmp_path):
        """Test setting up logging with file output."""
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", json_format=True, log_file=str(log_file))

        logger = logging.getLogger()

        # Should have both console and file handlers
        assert len(logger.handlers) >= 2

        # Log something
        logger.info("Test message")

        # Check file was created and has content
        assert log_file.exists()


class TestGetStructuredLogger:
    """Tests for get_structured_logger function."""

    def test_returns_structured_logger(self):
        """Test that function returns StructuredLogger."""
        logger = get_structured_logger("test.module")

        # Should have structured logging methods
        assert hasattr(logger, "info_with_data")
        assert hasattr(logger, "debug_with_data")
        assert hasattr(logger, "error_with_data")
        assert hasattr(logger, "warning_with_data")
        assert hasattr(logger, "critical_with_data")

    def test_logger_has_correct_name(self):
        """Test that logger has correct name."""
        logger = get_structured_logger("my.custom.logger")
        assert logger.name == "my.custom.logger"
