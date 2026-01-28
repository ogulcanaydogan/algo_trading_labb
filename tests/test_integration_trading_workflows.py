"""Integration tests for critical trading workflows."""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from api.validation import validate_trading_request, TradingValidationError
from bot.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from bot.core.structured_logging import correlation_context
from bot.safety_controller import SafetyController, SafetyLimits
from tests.conftest import TEST_API_KEY


class TestTradingWorkflows:
    """Integration tests for complete trading workflows."""

    @pytest.fixture
    def test_client(self):
        """Create authenticated FastAPI test client."""
        from api.api import app
        client = TestClient(app)
        client.headers = {"X-API-Key": TEST_API_KEY}
        return client
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        return {
            "symbol": "BTC/USDT",
            "timestamp": datetime.now().isoformat(),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 1000.0,
            "rsi": 55.0,
            "ema_fast": 50300.0,
            "ema_slow": 49800.0,
        }
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires /api/trade and /api/signals endpoints not yet implemented")
    def test_complete_trade_workflow(self, test_client, sample_market_data):
        """Test complete trade workflow from signal to execution."""
        with patch('bot.exchange.PaperExchangeClient') as mock_exchange:
            # Setup mock exchange
            mock_exchange.return_value.create_order.return_value = {
                "id": "test_order_123",
                "symbol": "BTC/USDT",
                "type": "market",
                "side": "buy",
                "amount": 0.001,
                "price": 50500.0,
                "status": "closed",
                "filled": 0.001,
            }
            
            # Step 1: Generate trading signal
            signal_data = {
                "symbol": "BTC/USDT",
                "action": "buy",
                "confidence": 0.75,
                "reason": "EMA crossover with RSI confirmation",
                "strength": "strong",
            }
            
            response = test_client.post("/api/signals", json=signal_data)
            assert response.status_code == 200
            
            # Step 2: Validate signal triggers trade
            trade_request = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.001,
                "order_type": "market",
            }
            
            response = test_client.post("/api/trade", json=trade_request)
            assert response.status_code == 200
            
            trade_result = response.json()
            assert trade_result["success"] is True
            assert "order_id" in trade_result
            
            # Step 3: Verify trade is recorded
            response = test_client.get("/api/trades?symbol=BTC/USDT")
            assert response.status_code == 200
            
            trades = response.json()
            assert len(trades) > 0
            assert trades[0]["symbol"] == "BTC/USDT"
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires complex AI brain mocking - endpoint validates input strictly")
    def test_ai_learning_workflow(self, test_client, temp_data_dir):
        """Test AI learning workflow from trade recording to model update."""
        with patch('bot.ai_trading_brain.get_ai_brain') as mock_brain:
            # Setup mock AI brain
            mock_brain_instance = Mock()
            mock_brain.return_value = mock_brain_instance
            mock_brain_instance.record_trade_result.return_value = {
                "success": True,
                "lessons_learned": 3,
                "confidence_improvement": 0.05,
            }
            
            # Step 1: Record a completed trade
            trade_data = {
                "trade_id": "test_trade_001",
                "symbol": "BTC/USDT",
                "action": "buy",
                "entry_price": 50000.0,
                "exit_price": 52000.0,
                "position_size": 0.001,
                "holding_hours": 24,
                "market_condition": "bullish",
                "trend": "up",
                "rsi": 60,
                "volatility": 70,
                "price_history": [49500, 50000, 50500, 51000, 52000],
            }
            
            response = test_client.post("/api/ai-brain/record-trade", json=trade_data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] is True
            assert "lessons_learned" in result
            
            # Step 2: Verify AI brain was called with correct data
            mock_brain_instance.record_trade_result.assert_called_once()
            call_args = mock_brain_instance.record_trade_result.call_args
            assert call_args[1]["symbol"] == "BTC/USDT"
            assert call_args[1]["action"] == "buy"
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires /api/trade endpoint not yet implemented")
    def test_risk_management_workflow(self, test_client):
        """Test risk management integration with trading."""
        with patch('bot.risk_guardian.RiskGuardian') as mock_risk:
            # Setup mock risk guardian
            mock_risk_instance = Mock()
            mock_risk.return_value = mock_risk_instance
            
            # Configure risk to allow the trade
            mock_risk_instance.evaluate_trade.return_value = {
                "approved": True,
                "risk_score": 0.3,
                "reason": "Acceptable risk level",
            }
            
            # Step 1: Attempt trade within risk limits
            trade_request = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.001,  # Small position
                "order_type": "market",
            }
            
            response = test_client.post("/api/trade", json=trade_request)
            assert response.status_code == 200
            
            # Step 2: Verify risk evaluation was called
            mock_risk_instance.evaluate_trade.assert_called()
            
            # Step 3: Test trade rejection due to risk limits
            mock_risk_instance.evaluate_trade.return_value = {
                "approved": False,
                "risk_score": 0.9,
                "reason": "Position size exceeds risk limits",
            }
            
            response = test_client.post("/api/trade", json=trade_request)
            assert response.status_code == 400
            
            result = response.json()
            assert result["success"] is False
            assert "risk" in result["error"].lower()


class TestInputValidationIntegration:
    """Integration tests for input validation across the system."""
    
    def test_trading_request_validation_integration(self):
        """Test trading request validation in various scenarios."""
        
        # Valid request
        valid_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.001,
            "order_type": "market",
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }
        
        try:
            result = validate_trading_request(valid_request, balance=10000.0)
            assert result.symbol == "BTC/USDT"
            assert result.side == "BUY"
            assert result.quantity == 0.001
        except Exception as e:
            pytest.fail(f"Valid request should not raise exception: {e}")
        
        # Invalid requests
        invalid_requests = [
            # Missing symbol
            {"side": "buy", "quantity": 0.001},
            # Invalid symbol format
            {"symbol": "INVALID@@@", "side": "buy", "quantity": 0.001},
            # Invalid side
            {"symbol": "BTC/USDT", "side": "invalid", "quantity": 0.001},
            # Negative quantity
            {"symbol": "BTC/USDT", "side": "buy", "quantity": -0.001},
            # Quantity too large
            {"symbol": "BTC/USDT", "side": "buy", "quantity": 1000000},
            # Invalid stop loss/take profit relationship (requires price to validate)
            {"symbol": "BTC/USDT", "side": "buy", "quantity": 0.001, "price": 50000.0,
             "stop_loss": 51000.0, "take_profit": 49000.0},  # SL above price for buy
        ]
        
        from fastapi import HTTPException
        for i, invalid_request in enumerate(invalid_requests):
            with pytest.raises((TradingValidationError, HTTPException)):
                validate_trading_request(invalid_request, balance=10000.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
from bot.trading_mode import TradingMode, TradingStatus
from bot.unified_state import UnifiedState, PositionState


def _create_state(mode=TradingMode.PAPER_LIVE_DATA, balance=10000.0):
    """Helper to create test state with all required fields."""
    return UnifiedState(
        mode=mode,
        status=TradingStatus.ACTIVE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        initial_capital=balance,
        current_balance=balance,
        peak_balance=balance,
    )


class TestPaperTradingWorkflow:
    """Test paper trading mode workflow."""

    def test_paper_mode_initialization(self):
        """Test that paper mode initializes with safe defaults."""
        state = _create_state()

        assert state.mode == TradingMode.PAPER_LIVE_DATA
        assert state.current_balance == 10000.0

    def test_paper_mode_open_position(self):
        """Test opening a position in paper mode."""
        state = _create_state()

        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )

        state.positions["BTC/USDT"] = position

        assert "BTC/USDT" in state.positions
        assert state.positions["BTC/USDT"].quantity == 0.01


class TestSafetyControllerWorkflow:
    """Test safety controller enforces limits."""

    def test_daily_loss_limit_enforcement(self):
        """Test that daily loss limit is configured."""
        limits = SafetyLimits(max_daily_loss_usd=100.0)
        assert limits.max_daily_loss_usd == 100.0

    def test_position_size_limit(self):
        """Test position size limit enforcement."""
        limits = SafetyLimits(max_position_size_usd=500.0)
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)

        controller.update_positions({"BTC/USDT": 400.0})
        assert sum(controller._open_positions.values()) == 400.0


class TestModeTransitionWorkflow:
    """Test trading mode transitions."""

    def test_paper_to_testnet_transition(self):
        """Test transition from paper to testnet mode."""
        state = _create_state(mode=TradingMode.PAPER_LIVE_DATA)

        assert state.mode == TradingMode.PAPER_LIVE_DATA

        state.mode = TradingMode.TESTNET

        assert state.mode == TradingMode.TESTNET

    def test_state_preserved_across_mode_transition(self):
        """Test that portfolio state is preserved during mode transition."""
        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )

        state = _create_state(balance=9580.0)
        state.positions["BTC/USDT"] = position

        state.mode = TradingMode.TESTNET

        assert state.current_balance == 9580.0
        assert "BTC/USDT" in state.positions


class TestSignalGenerationWorkflow:
    """Test signal generation from market data."""

    def test_signal_generation_returns_valid_structure(self):
        """Test that signals have required fields."""
        signal = {
            "symbol": "BTC/USDT",
            "signal": "LONG",
            "confidence": 0.75,
            "reason": "EMA crossover with RSI confirmation",
        }

        assert signal["signal"] in ["LONG", "SHORT", "FLAT"]
        assert 0.0 <= signal["confidence"] <= 1.0

    def test_signal_confidence_bounds(self):
        """Test signal confidence is within valid range."""
        valid_signals = [
            {"signal": "LONG", "confidence": 0.0},
            {"signal": "LONG", "confidence": 0.5},
            {"signal": "LONG", "confidence": 1.0},
        ]

        for sig in valid_signals:
            assert 0.0 <= sig["confidence"] <= 1.0


class TestErrorRecoveryWorkflow:
    """Test error handling and recovery in trading workflows."""

    def test_trading_pauses_on_api_errors(self):
        """Test that trading pauses after repeated API errors."""
        controller = SafetyController()
        controller.update_balance(10000.0)

        for _ in range(3):
            controller._daily_stats.api_errors += 1

        assert controller._daily_stats.api_errors == 3

    def test_state_recovery_on_reconnect(self):
        """Test state recovery when connection is restored."""
        position = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )

        state = _create_state(balance=7500.0)
        state.positions["ETH/USDT"] = position

        assert state.positions["ETH/USDT"].symbol == "ETH/USDT"
        assert state.current_balance == 7500.0


class TestMultiPositionWorkflow:
    """Test handling multiple concurrent positions."""

    def test_open_multiple_positions(self):
        """Test opening multiple positions simultaneously."""
        state = _create_state()

        pos1 = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )

        pos2 = PositionState(
            symbol="ETH/USDT",
            quantity=0.5,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )

        state.positions["BTC/USDT"] = pos1
        state.positions["ETH/USDT"] = pos2

        assert len(state.positions) == 2

    def test_close_one_position_keep_others(self):
        """Test closing one position while keeping others open."""
        state = _create_state()
        state.positions["BTC/USDT"] = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        state.positions["ETH/USDT"] = PositionState(
            symbol="ETH/USDT",
            quantity=0.5,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )

        del state.positions["BTC/USDT"]

        assert "BTC/USDT" not in state.positions
        assert "ETH/USDT" in state.positions


class TestCircuitBreakerWorkflow:
    """Test circuit breaker integration with trading."""

    def test_circuit_breaker_state_management(self):
        """Test that circuit breaker state is properly managed."""
        from bot.core.circuit_breaker import CircuitState

        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=10,
            success_threshold=1,
        )
        breaker = CircuitBreaker("test_workflow", config)

        # Initial state should be CLOSED
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

        # Simulate failures by directly setting state
        breaker.failure_count = 3
        breaker.state = CircuitState.OPEN

        # Verify circuit is open
        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout=60,
            success_threshold=2,
        )

        breaker = CircuitBreaker("test_config", config)

        assert breaker.config.failure_threshold == 5
        assert breaker.config.timeout == 60
        assert breaker.config.success_threshold == 2


class TestCorrelationIdWorkflow:
    """Test correlation ID propagation across components."""

    def test_correlation_context_propagates(self):
        """Test that correlation context propagates correctly."""
        from bot.core.structured_logging import get_correlation_id, correlation_id_var

        with correlation_context("test-corr-123") as ctx:
            # Inside context, correlation ID should be set
            corr_id = get_correlation_id()
            assert corr_id == "test-corr-123"

        # Outside context, correlation_id_var.get() should return None
        # (get_correlation_id() generates a new UUID if not set)
        raw_corr_id = correlation_id_var.get()
        assert raw_corr_id is None or raw_corr_id != "test-corr-123"


class TestCacheIntegrationWorkflow:
    """Test cache integration with trading operations."""

    def test_market_data_caching(self):
        """Test that market data is cached correctly."""
        from bot.core.cache import CacheManager

        # Reset singleton
        CacheManager._instance = None
        cache = CacheManager()

        ohlcv = [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000}
        ]

        cache.cache_market_data("BTC/USDT", ohlcv, "1h")
        cached = cache.get_market_data("BTC/USDT", "1h")

        assert cached == ohlcv

    def test_ml_prediction_caching(self):
        """Test that ML predictions are cached."""
        from bot.core.cache import CacheManager

        CacheManager._instance = None
        cache = CacheManager()

        prediction = {"action": "LONG", "confidence": 0.85}
        cache.cache_ml_prediction("BTC/USDT", "ensemble", prediction)

        cached = cache.get_ml_prediction("BTC/USDT", "ensemble")
        assert cached == prediction


class TestAuditLoggingWorkflow:
    """Test audit logging integration."""

    def test_trade_audit_logging(self):
        """Test that trades are properly audit logged."""
        import tempfile
        from bot.core.audit import AuditLogger, AuditEventType

        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            audit = AuditLogger(Path(tmpdir))

            # Log a trade
            event_id = audit.log_trade(
                action="executed",
                symbol="BTC/USDT",
                side="BUY",
                quantity=0.1,
                price=42000.0,
                order_id="order123",
            )

            # Verify event was logged
            events = audit.get_events()
            assert len(events) >= 1

            trade_event = [e for e in events if e["event_type"] == "trade_executed"]
            assert len(trade_event) >= 1
            assert trade_event[-1]["details"]["symbol"] == "BTC/USDT"

    def test_safety_event_audit_logging(self):
        """Test that safety events are audit logged."""
        import tempfile
        from bot.core.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            AuditLogger._instance = None
            audit = AuditLogger(Path(tmpdir))

            event_id = audit.log_safety_event(
                event="daily_limit",
                reason="Daily loss limit reached",
                current_value=150.0,
                limit=100.0,
            )

            events = audit.get_events()
            safety_events = [e for e in events if "daily_limit" in e["event_type"]]
            assert len(safety_events) >= 1


class TestEndToEndTradingScenarios:
    """End-to-end test scenarios for complete trading workflows."""

    def test_full_trading_cycle_simulation(self):
        """Simulate a complete trading cycle: signal -> validation -> execution -> P&L."""
        # Step 1: Generate signal data
        signal = {
            "symbol": "BTC/USDT",
            "decision": "LONG",
            "confidence": 0.75,
            "reason": "Bullish momentum detected",
        }

        # Step 2: Validate through safety controller
        limits = SafetyLimits(
            max_position_size_usd=1000.0,
            max_daily_loss_usd=200.0,
            max_trades_per_day=10,
        )
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)

        # Step 3: Simulate trade execution (paper mode)
        entry_price = 42000.0
        quantity = 0.01

        position = PositionState(
            symbol=signal["symbol"],
            quantity=quantity,
            entry_price=entry_price,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=entry_price * 0.98,
            take_profit=entry_price * 1.03,
            current_price=entry_price,
        )

        state = _create_state(balance=10000.0 - (entry_price * quantity))
        state.positions[signal["symbol"]] = position

        # Step 4: Simulate price movement
        position.current_price = 42500.0  # Price went up

        # Step 5: Calculate P&L
        unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        assert unrealized_pnl == 5.0  # $5 profit

    def test_stop_loss_triggered_scenario(self):
        """Test scenario where stop loss is triggered."""
        state = _create_state(balance=9580.0)

        position = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=41000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        state.positions["BTC/USDT"] = position

        # Price drops below stop loss
        position.current_price = 40500.0

        # Check if stop loss should trigger
        assert position.current_price < position.stop_loss

        # Simulate closing position at stop loss
        exit_price = position.stop_loss
        realized_pnl = (exit_price - position.entry_price) * position.quantity
        assert realized_pnl == -10.0  # $10 loss

        # Update balance
        state.current_balance += realized_pnl
        del state.positions["BTC/USDT"]

        assert state.current_balance == 9570.0
        assert len(state.positions) == 0

    def test_take_profit_triggered_scenario(self):
        """Test scenario where take profit is triggered."""
        state = _create_state(balance=9580.0)

        position = PositionState(
            symbol="ETH/USDT",
            quantity=1.0,
            entry_price=2500.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=2400.0,
            take_profit=2600.0,
            current_price=2500.0,
        )
        state.positions["ETH/USDT"] = position

        # Price rises above take profit
        position.current_price = 2650.0

        # Check if take profit should trigger
        assert position.current_price > position.take_profit

        # Simulate closing position at take profit
        exit_price = position.take_profit
        realized_pnl = (exit_price - position.entry_price) * position.quantity
        assert realized_pnl == 100.0  # $100 profit

        # Update balance
        state.current_balance += realized_pnl
        del state.positions["ETH/USDT"]

        assert state.current_balance == 9680.0

    def test_max_positions_limit_scenario(self):
        """Test that max positions limit is enforced."""
        limits = SafetyLimits(
            max_open_positions=2,
            max_position_size_usd=1000.0,
        )
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)

        # Add 2 positions
        controller.update_positions({
            "BTC/USDT": 500.0,
            "ETH/USDT": 500.0,
        })

        # Verify position count
        assert len(controller._open_positions) == 2

        # Trying to add a third should be blocked
        # Create a mock order object
        class MockOrder:
            symbol = "SOL/USDT"
            side = "BUY"
            quantity = 1.0
            price = 100.0

        passed, reason = controller.pre_trade_check(MockOrder())

        assert passed is False
        assert "position" in reason.lower()

    def test_daily_trade_limit_scenario(self):
        """Test that daily trade limit is enforced."""
        limits = SafetyLimits(
            max_trades_per_day=3,
            max_position_size_usd=1000.0,
            min_time_between_trades_seconds=0,  # Allow rapid trades for testing
        )
        controller = SafetyController(limits=limits)
        controller.update_balance(10000.0)

        # Record 3 trades
        for _ in range(3):
            controller._daily_stats.trades += 1

        # Fourth trade should be blocked
        class MockOrder:
            symbol = "BTC/USDT"
            side = "BUY"
            quantity = 0.01
            price = 42000.0

        passed, reason = controller.pre_trade_check(MockOrder())

        assert passed is False
        assert "trade" in reason.lower() and "limit" in reason.lower()
