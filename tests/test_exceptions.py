"""
Tests for the exception handling module.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from bot.core.exceptions import (
    # Base exceptions
    TradingBotError,
    DataError,
    NetworkError,
    ExchangeError,
    ValidationError,
    ExecutionError,
    ModelError,
    ConfigurationError,
    # Additional exceptions
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitError,
    PositionLimitError,
    MarketClosedError,
    PredictionError,
    CircuitBreakerOpenError,
    # Enums
    ErrorSeverity,
    ErrorCategory,
    # Classes
    ErrorRecord,
    ErrorTracker,
    EnhancedErrorTracker,
    ErrorAlertHandler,
    # Functions
    classify_exception,
    handle_exceptions,
    handle_exceptions_async,
    safe_execution,
    log_exception,
    retry_on_exception,
    retry_on_exception_async,
    setup_error_alerts,
    get_error_summary,
    handle_critical_error,
    # Globals
    error_tracker,
    enhanced_error_tracker,
    alert_handler,
)


class TestErrorSeverity:
    def test_severity_values(self):
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    def test_category_values(self):
        assert ErrorCategory.DATA.value == "data"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.EXCHANGE.value == "exchange"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.EXECUTION.value == "execution"
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.SYSTEM.value == "system"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestTradingBotError:
    def test_basic_creation(self):
        error = TradingBotError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.UNKNOWN
        assert error.details == {}
        assert isinstance(error.timestamp, datetime)

    def test_with_severity_and_category(self):
        error = TradingBotError(
            "Critical error",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
        )
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.SYSTEM

    def test_with_details(self):
        error = TradingBotError(
            "Error with details",
            details={"key": "value", "count": 42},
        )
        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_to_dict(self):
        error = TradingBotError(
            "Test error",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            details={"extra": "info"},
        )
        result = error.to_dict()
        assert result["error_type"] == "TradingBotError"
        assert result["message"] == "Test error"
        assert result["severity"] == "high"
        assert result["category"] == "data"
        assert result["details"]["extra"] == "info"
        assert "timestamp" in result


class TestSpecializedExceptions:
    def test_data_error(self):
        error = DataError("Failed to fetch data")
        assert error.category == ErrorCategory.DATA
        assert error.severity == ErrorSeverity.MEDIUM

    def test_network_error(self):
        error = NetworkError("Connection timeout")
        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.HIGH

    def test_exchange_error(self):
        error = ExchangeError("Exchange unavailable")
        assert error.category == ErrorCategory.EXCHANGE
        assert error.severity == ErrorSeverity.HIGH

    def test_validation_error(self):
        error = ValidationError("Invalid input")
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW

    def test_execution_error(self):
        error = ExecutionError("Order failed")
        assert error.category == ErrorCategory.EXECUTION
        assert error.severity == ErrorSeverity.HIGH

    def test_model_error(self):
        error = ModelError("Model not found")
        assert error.category == ErrorCategory.MODEL
        assert error.severity == ErrorSeverity.MEDIUM

    def test_configuration_error(self):
        error = ConfigurationError("Missing config")
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.HIGH


class TestAdditionalExceptions:
    def test_insufficient_funds_error(self):
        error = InsufficientFundsError(
            "Not enough balance",
            required=1000.0,
            available=500.0,
        )
        assert error.details["required"] == 1000.0
        assert error.details["available"] == 500.0
        assert error.severity == ErrorSeverity.HIGH

    def test_order_rejected_error(self):
        error = OrderRejectedError(
            "Order rejected",
            order_id="12345",
            reason="Price too far from market",
        )
        assert error.details["order_id"] == "12345"
        assert error.details["reason"] == "Price too far from market"

    def test_rate_limit_error(self):
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60.0,
        )
        assert error.details["retry_after"] == 60.0
        assert error.severity == ErrorSeverity.MEDIUM

    def test_position_limit_error(self):
        error = PositionLimitError(
            "Position too large",
            max_allowed=10.0,
            requested=15.0,
        )
        assert error.details["max_allowed"] == 10.0
        assert error.details["requested"] == 15.0

    def test_market_closed_error(self):
        error = MarketClosedError(
            "Market closed",
            market="NYSE",
        )
        assert error.details["market"] == "NYSE"
        assert error.severity == ErrorSeverity.LOW

    def test_prediction_error(self):
        error = PredictionError(
            "Prediction failed",
            model_name="random_forest",
        )
        assert error.details["model_name"] == "random_forest"

    def test_circuit_breaker_open_error(self):
        error = CircuitBreakerOpenError(
            "Circuit breaker open",
            breaker_name="ml_prediction",
            cooldown_remaining=30.0,
        )
        assert error.details["breaker_name"] == "ml_prediction"
        assert error.details["cooldown_remaining"] == 30.0
        assert error.category == ErrorCategory.SYSTEM


class TestClassifyException:
    def test_classify_connection_error(self):
        category, severity = classify_exception(ConnectionError("Failed"))
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.HIGH

    def test_classify_timeout_error(self):
        category, severity = classify_exception(TimeoutError("Timeout"))
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.MEDIUM

    def test_classify_value_error(self):
        category, severity = classify_exception(ValueError("Invalid"))
        assert category == ErrorCategory.VALIDATION
        assert severity == ErrorSeverity.LOW

    def test_classify_key_error(self):
        category, severity = classify_exception(KeyError("missing"))
        assert category == ErrorCategory.DATA
        assert severity == ErrorSeverity.LOW

    def test_classify_trading_bot_error(self):
        error = DataError("Data error")
        category, severity = classify_exception(error)
        assert category == ErrorCategory.DATA
        assert severity == ErrorSeverity.MEDIUM

    def test_classify_unknown_error(self):
        class CustomError(Exception):
            pass
        category, severity = classify_exception(CustomError("Unknown"))
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM


class TestErrorRecord:
    def test_error_record_creation(self):
        exc = ValueError("Test error")
        record = ErrorRecord(
            exception=exc,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context="test_context",
            traceback_str="traceback...",
        )
        assert record.exception is exc
        assert record.category == ErrorCategory.VALIDATION
        assert record.severity == ErrorSeverity.LOW
        assert record.context == "test_context"
        assert record.traceback_str == "traceback..."

    def test_error_record_to_dict(self):
        exc = ValueError("Test error")
        record = ErrorRecord(
            exception=exc,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context="test_context",
        )
        result = record.to_dict()
        assert result["error_type"] == "ValueError"
        assert result["message"] == "Test error"
        assert result["category"] == "validation"
        assert result["severity"] == "low"
        assert result["context"] == "test_context"


class TestErrorTracker:
    def test_record_error(self):
        tracker = ErrorTracker()
        exc = ValueError("Test")
        record = tracker.record(exc, "test_context")

        assert record is not None
        assert record.context == "test_context"
        assert len(tracker.get_recent()) == 1

    def test_max_history(self):
        tracker = ErrorTracker(max_history=5)
        for i in range(10):
            tracker.record(ValueError(f"Error {i}"), "context")

        recent = tracker.get_recent()
        assert len(recent) == 5

    def test_get_counts(self):
        tracker = ErrorTracker()
        tracker.record(ValueError("Error 1"), "ctx1")
        tracker.record(ValueError("Error 2"), "ctx1")
        tracker.record(KeyError("Error 3"), "ctx2")

        counts = tracker.get_counts()
        assert counts["ValueError:ctx1"] == 2
        assert counts["KeyError:ctx2"] == 1

    def test_get_by_severity(self):
        tracker = ErrorTracker()
        tracker.record(ValueError("Low"), "ctx")
        tracker.record(ConnectionError("High"), "ctx")

        high_errors = tracker.get_by_severity(ErrorSeverity.HIGH)
        assert len(high_errors) == 1

    def test_clear(self):
        tracker = ErrorTracker()
        tracker.record(ValueError("Error"), "ctx")
        tracker.clear()

        assert len(tracker.get_recent()) == 0
        assert len(tracker.get_counts()) == 0


class TestEnhancedErrorTracker:
    def test_record_with_alert(self):
        tracker = EnhancedErrorTracker()
        tracker._alert_handler = MagicMock()
        tracker._alert_handler.alert = MagicMock()

        tracker.record(ValueError("Error"), "ctx", send_alert=True)
        tracker._alert_handler.alert.assert_called_once()

    def test_record_without_alert(self):
        tracker = EnhancedErrorTracker()
        tracker._alert_handler = MagicMock()
        tracker._alert_handler.alert = MagicMock()

        tracker.record(ValueError("Error"), "ctx", send_alert=False)
        tracker._alert_handler.alert.assert_not_called()

    def test_get_error_rate(self):
        tracker = EnhancedErrorTracker()
        tracker.record(ValueError("Error"), "ctx")
        tracker.record(ValueError("Error"), "ctx")

        rate = tracker.get_error_rate("ValueError", "ctx")
        assert rate == 2

    def test_get_top_errors(self):
        tracker = EnhancedErrorTracker()
        for _ in range(5):
            tracker.record(ValueError("Error"), "ctx1", send_alert=False)
        for _ in range(3):
            tracker.record(KeyError("Error"), "ctx2", send_alert=False)

        top = tracker.get_top_errors(2)
        assert len(top) == 2
        assert top[0]["count"] == 5

    def test_get_summary(self):
        tracker = EnhancedErrorTracker()
        tracker.record(ValueError("Error"), "ctx", send_alert=False)
        tracker.record(ConnectionError("Error"), "ctx", send_alert=False)

        summary = tracker.get_summary()
        assert summary["total_errors"] == 2
        assert "by_severity" in summary
        assert "by_category" in summary
        assert "top_errors" in summary


class TestErrorAlertHandler:
    def test_register_callback(self):
        handler = ErrorAlertHandler()
        callback = MagicMock()
        handler.register_callback(callback)

        assert callback in handler._alert_callbacks

    def test_set_severity_threshold(self):
        handler = ErrorAlertHandler()
        handler.set_severity_threshold(ErrorSeverity.CRITICAL)

        assert handler._severity_threshold == ErrorSeverity.CRITICAL

    def test_should_alert_below_threshold(self):
        handler = ErrorAlertHandler()
        handler.set_severity_threshold(ErrorSeverity.HIGH)

        record = ErrorRecord(
            exception=ValueError("Error"),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context="ctx",
        )

        assert not handler.should_alert(record)

    def test_should_alert_above_threshold(self):
        handler = ErrorAlertHandler()
        handler.set_severity_threshold(ErrorSeverity.MEDIUM)
        handler._error_cooldowns.clear()

        record = ErrorRecord(
            exception=ConnectionError("Error"),
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            context="ctx",
        )

        assert handler.should_alert(record)

    def test_alert_calls_callbacks(self):
        handler = ErrorAlertHandler()
        handler.set_severity_threshold(ErrorSeverity.LOW)
        handler._error_cooldowns.clear()

        callback = MagicMock()
        handler.register_callback(callback)

        record = ErrorRecord(
            exception=ValueError("Error"),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            context="test_alert",
        )

        handler.alert(record)
        callback.assert_called_once_with(record)


class TestHandleExceptionsDecorator:
    def test_returns_default_on_exception(self):
        @handle_exceptions(default=[], exceptions=(ValueError,))
        def failing_func():
            raise ValueError("Error")

        result = failing_func()
        assert result == []

    def test_returns_result_on_success(self):
        @handle_exceptions(default=[])
        def successful_func():
            return [1, 2, 3]

        result = successful_func()
        assert result == [1, 2, 3]

    def test_reraise_option(self):
        @handle_exceptions(default=None, reraise=True)
        def failing_func():
            raise ValueError("Error")

        with pytest.raises(ValueError):
            failing_func()

    def test_only_catches_specified_exceptions(self):
        @handle_exceptions(default=[], exceptions=(ValueError,))
        def failing_func():
            raise KeyError("Error")

        with pytest.raises(KeyError):
            failing_func()


class TestHandleExceptionsAsyncDecorator:
    @pytest.mark.asyncio
    async def test_returns_default_on_exception(self):
        @handle_exceptions_async(default={}, exceptions=(ValueError,))
        async def failing_func():
            raise ValueError("Error")

        result = await failing_func()
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        @handle_exceptions_async(default={})
        async def successful_func():
            return {"success": True}

        result = await successful_func()
        assert result == {"success": True}


class TestSafeExecution:
    def test_catches_exception(self):
        result = None
        with safe_execution("test_context"):
            raise ValueError("Error")
            result = "not reached"

        assert result is None

    def test_successful_execution(self):
        result = None
        with safe_execution("test_context"):
            result = "success"

        assert result == "success"


class TestRetryOnException:
    def test_succeeds_after_retry(self):
        call_count = 0

        @retry_on_exception(max_retries=3, delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_raises_after_max_retries(self):
        @retry_on_exception(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            always_fails()


class TestRetryOnExceptionAsync:
    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        call_count = 0

        @retry_on_exception_async(max_retries=3, delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 2


class TestLogException:
    def test_logs_and_records(self):
        exc = ValueError("Test error")
        record = log_exception(exc, "test_context", include_traceback=False)

        assert record is not None
        assert record.context == "test_context"


class TestConvenienceFunctions:
    def test_setup_error_alerts(self):
        callback = MagicMock()
        setup_error_alerts(
            telegram_callback=callback,
            severity_threshold=ErrorSeverity.CRITICAL,
        )

        assert callback in alert_handler._alert_callbacks
        assert alert_handler._severity_threshold == ErrorSeverity.CRITICAL

    def test_get_error_summary(self):
        summary = get_error_summary()
        assert "total_errors" in summary
        assert "by_severity" in summary
        assert "by_category" in summary

    def test_handle_critical_error(self):
        exc = MemoryError("Out of memory")
        record = handle_critical_error(exc, "memory_test")

        assert record is not None
        assert record.severity == ErrorSeverity.CRITICAL
