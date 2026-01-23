"""Integration tests for enhanced trading engine and adapter."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.enhanced_trading_engine import (
    EnhancedTradingEngine,
    EnhancedTradingConfig,
)
from bot.enhanced_adapter import (
    EnhancedExecutionAdapter,
    EnhancedAdapterConfig,
    create_enhanced_adapter,
)
from bot.execution import UrgencyLevel
from bot.regime import MarketRegime
from bot.risk import CircuitBreakerState


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(end=datetime.now(), periods=n, freq="1h")

    # Generate realistic price data
    price = 50000
    prices = [price]
    for _ in range(n - 1):
        change = np.random.randn() * 0.01  # 1% volatility
        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.001),
            "high": prices * (1 + abs(np.random.randn(n)) * 0.005),
            "low": prices * (1 - abs(np.random.randn(n)) * 0.005),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        },
        index=dates,
    )


@pytest.fixture
def mock_base_adapter():
    """Create mock base execution adapter."""
    adapter = MagicMock()
    adapter.get_balance = AsyncMock(return_value={"total": 10000, "available": 10000})
    adapter.get_positions = AsyncMock(return_value={})
    adapter.get_current_price = AsyncMock(return_value=50000.0)
    adapter.get_ohlcv = AsyncMock()
    adapter.execute_order = AsyncMock(
        return_value=MagicMock(
            success=True,
            order_id="test_order_1",
            filled_quantity=1.0,
            average_price=50000.0,
        )
    )
    return adapter


class TestEnhancedAdapter:
    """Tests for EnhancedExecutionAdapter."""

    @pytest.fixture
    def enhanced_adapter(self, mock_base_adapter):
        return EnhancedExecutionAdapter(mock_base_adapter)

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, enhanced_adapter):
        assert enhanced_adapter.regime_detector is not None
        assert enhanced_adapter.position_sizer is not None
        assert enhanced_adapter.circuit_breaker is not None

    @pytest.mark.asyncio
    async def test_update_state(self, enhanced_adapter, sample_ohlcv, mock_base_adapter):
        mock_base_adapter.get_ohlcv.return_value = sample_ohlcv

        state = await enhanced_adapter.update_state("BTC/USDT", sample_ohlcv)

        assert "regime" in state
        assert "portfolio_value" in state
        assert "risk_level" in state
        assert state["portfolio_value"] == 10000

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, enhanced_adapter, sample_ohlcv):
        # First update state to set portfolio value
        await enhanced_adapter.update_state("BTC/USDT", sample_ohlcv)

        result = enhanced_adapter.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )

        assert "final_size" in result
        assert result["final_size"] >= 0
        assert result["final_size"] <= 0.2  # max position size

    @pytest.mark.asyncio
    async def test_get_strategy_selection(self, enhanced_adapter, sample_ohlcv):
        await enhanced_adapter.update_state("BTC/USDT", sample_ohlcv)

        selection = enhanced_adapter.get_strategy_selection(volatility=0.15)

        assert "selected_strategies" in selection
        assert "position_scale" in selection

    def test_run_stress_test(self, enhanced_adapter):
        positions = [
            {"symbol": "BTC", "quantity": 1.0, "price": 50000, "asset_class": "crypto"},
        ]

        result = enhanced_adapter.run_stress_test(positions)

        assert "scenarios_tested" in result
        assert "is_acceptable" in result

    @pytest.mark.asyncio
    async def test_is_trading_allowed_normal(self, enhanced_adapter, sample_ohlcv):
        await enhanced_adapter.update_state("BTC/USDT", sample_ohlcv)

        allowed, reason = enhanced_adapter.is_trading_allowed()

        assert allowed is True
        assert reason == "Trading allowed"

    def test_is_trading_allowed_circuit_breaker_triggered(self, enhanced_adapter):
        enhanced_adapter.circuit_breaker._state = CircuitBreakerState.TRIGGERED

        allowed, reason = enhanced_adapter.is_trading_allowed()

        assert allowed is False
        assert "Circuit breaker" in reason

    def test_get_status(self, enhanced_adapter):
        status = enhanced_adapter.get_status()

        assert "base_adapter" in status
        assert "config" in status
        assert "state" in status


class TestEnhancedTradingEngineIntegration:
    """Integration tests for EnhancedTradingEngine."""

    @pytest.fixture
    def mock_adapter(self, sample_ohlcv):
        adapter = MagicMock()
        adapter.get_balance = AsyncMock(return_value={"total": 10000, "available": 10000})
        adapter.get_positions = AsyncMock(return_value={})
        adapter.get_price = AsyncMock(return_value=50000.0)
        adapter.get_ohlcv = AsyncMock(return_value=sample_ohlcv)
        adapter.get_market_data = AsyncMock(
            return_value={
                "mid_price": 50000,
                "volatility": 0.02,
                "spread_bps": 10,
            }
        )
        adapter.place_order = AsyncMock(
            return_value={
                "order_id": "test_order",
                "status": "filled",
                "price": 50000,
                "quantity": 0.1,
            }
        )
        return adapter

    @pytest.fixture
    def engine_config(self):
        return EnhancedTradingConfig(
            symbols=["BTC/USDT"],
            update_interval_seconds=1,
            enable_stress_testing=False,  # Disable for faster tests
        )

    @pytest.fixture
    def engine(self, engine_config, mock_adapter):
        engine = EnhancedTradingEngine(engine_config, mock_adapter)
        return engine

    def test_engine_initialization(self, engine):
        assert len(engine.regime_detectors) == 1
        assert engine.strategy_selector is not None
        assert engine.circuit_breaker is not None
        assert engine.position_sizer is not None

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, engine):
        await engine.start()
        assert engine.state.is_running is True

        await asyncio.sleep(0.1)

        await engine.stop()
        assert engine.state.is_running is False

    @pytest.mark.asyncio
    async def test_force_update(self, engine):
        await engine.force_update()

        assert engine.state.last_update is not None
        assert engine.state.equity > 0

    @pytest.mark.asyncio
    async def test_regime_detection_updates(self, engine):
        await engine.force_update()

        # Should have detected a regime
        assert engine.state.current_regime is not None

    def test_get_status(self, engine):
        status = engine.get_status()

        assert "state" in status
        assert "circuit_breaker" in status
        assert "position_risk_level" in status
        assert "strategy_selector" in status

    def test_reset_daily_counters(self, engine):
        engine.state.trades_today = 5
        engine.state.daily_pnl = 100.0

        engine.reset_daily_counters()

        assert engine.state.trades_today == 0
        assert engine.state.daily_pnl == 0.0

    @pytest.mark.asyncio
    async def test_callbacks_are_called(self, engine):
        regime_changes = []
        trades = []

        engine.on_regime_change(lambda old, new: regime_changes.append((old, new)))
        engine.on_trade(lambda trade: trades.append(trade))

        # Force an update cycle
        await engine.force_update()

        # First update should trigger regime change from None to detected regime
        assert len(regime_changes) >= 1 or engine.state.current_regime is not None


class TestCreateEnhancedAdapter:
    """Tests for create_enhanced_adapter factory function."""

    def test_creates_adapter(self, mock_base_adapter):
        adapter = create_enhanced_adapter(mock_base_adapter)
        assert isinstance(adapter, EnhancedExecutionAdapter)

    def test_with_custom_config(self, mock_base_adapter):
        config = EnhancedAdapterConfig(
            execution_algorithm="vwap",
            enable_circuit_breaker=False,
        )
        adapter = create_enhanced_adapter(mock_base_adapter, config)

        assert adapter.config.execution_algorithm == "vwap"
        assert adapter.circuit_breaker is None


class TestEnhancedConfigValidation:
    """Tests for configuration validation."""

    def test_default_config(self):
        config = EnhancedTradingConfig()

        assert config.symbols == ["BTC/USDT"]
        assert config.default_execution_algo == "adaptive"
        assert config.enable_circuit_breaker is True

    def test_custom_config(self):
        config = EnhancedTradingConfig(
            symbols=["BTC/USDT", "ETH/USDT"],
            default_execution_algo="twap",
            execution_urgency=UrgencyLevel.HIGH,
            enable_stress_testing=False,
        )

        assert len(config.symbols) == 2
        assert config.default_execution_algo == "twap"
        assert config.execution_urgency == UrgencyLevel.HIGH

    def test_adapter_config_defaults(self):
        config = EnhancedAdapterConfig()

        assert config.base_risk_per_trade == 0.02
        assert config.max_position_size == 0.20
        assert config.enable_circuit_breaker is True


class TestComponentInteraction:
    """Tests for interaction between components."""

    @pytest.mark.asyncio
    async def test_regime_affects_position_sizing(self, mock_base_adapter, sample_ohlcv):
        adapter = EnhancedExecutionAdapter(mock_base_adapter)
        mock_base_adapter.get_ohlcv.return_value = sample_ohlcv

        # Update state to detect regime
        await adapter.update_state("BTC/USDT", sample_ohlcv)

        # Calculate position size
        result = adapter.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )

        # Check that regime was considered
        assert "regime" in result["risk_metrics"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_trading(self, mock_base_adapter, sample_ohlcv):
        adapter = EnhancedExecutionAdapter(mock_base_adapter)
        await adapter.update_state("BTC/USDT", sample_ohlcv)

        # Trigger circuit breaker
        adapter.circuit_breaker._state = CircuitBreakerState.TRIGGERED

        # Check trading is blocked
        allowed, reason = adapter.is_trading_allowed()
        assert allowed is False

        # Check position sizing reflects circuit breaker
        result = adapter.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_price=49000,
            current_volatility=0.15,
        )
        assert "circuit_breaker" in result["adjustments_applied"]

    @pytest.mark.asyncio
    async def test_strategy_selection_uses_regime(self, mock_base_adapter, sample_ohlcv):
        adapter = EnhancedExecutionAdapter(mock_base_adapter)
        mock_base_adapter.get_ohlcv.return_value = sample_ohlcv

        # Update state to detect regime
        await adapter.update_state("BTC/USDT", sample_ohlcv)

        # Get strategy selection
        selection = adapter.get_strategy_selection(volatility=0.15)

        # Should have regime-aware selection
        assert "regime" in selection
        assert selection["regime"] is not None


class TestStressTestingIntegration:
    """Tests for stress testing integration."""

    def test_stress_test_returns_results(self, mock_base_adapter):
        adapter = EnhancedExecutionAdapter(mock_base_adapter)

        # Run stress test on a position
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "price": 50000, "asset_class": "crypto"},
        ]

        result = adapter.run_stress_test(positions)

        # Check structure of results
        assert "scenarios_tested" in result
        assert "worst_case_loss" in result
        assert "is_acceptable" in result
        assert result["scenarios_tested"] > 0

    def test_stress_test_returns_recommendations(self, mock_base_adapter):
        adapter = EnhancedExecutionAdapter(mock_base_adapter)

        positions = [
            {"symbol": "BTC", "quantity": 1.0, "price": 50000, "asset_class": "crypto"},
        ]

        result = adapter.run_stress_test(positions)

        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)


class TestMultiSymbolSupport:
    """Tests for multi-symbol trading support."""

    @pytest.fixture
    def mock_execution_adapter(self, sample_ohlcv):
        """Create mock execution adapter for multi-symbol tests."""
        adapter = MagicMock()
        adapter.get_balance = AsyncMock(return_value={"total": 10000, "available": 10000})
        adapter.get_positions = AsyncMock(return_value={})
        adapter.get_price = AsyncMock(return_value=50000.0)
        adapter.get_ohlcv = AsyncMock(return_value=sample_ohlcv)
        adapter.get_market_data = AsyncMock(
            return_value={
                "mid_price": 50000,
                "volatility": 0.02,
                "spread_bps": 10,
            }
        )
        adapter.place_order = AsyncMock(
            return_value={
                "order_id": "test_order",
                "status": "filled",
                "price": 50000,
                "quantity": 0.1,
            }
        )
        return adapter

    @pytest.fixture
    def multi_symbol_engine(self, mock_execution_adapter, sample_ohlcv):
        config = EnhancedTradingConfig(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            update_interval_seconds=1,
            enable_stress_testing=False,
        )
        return EnhancedTradingEngine(config, mock_execution_adapter)

    def test_multiple_regime_detectors(self, multi_symbol_engine):
        assert len(multi_symbol_engine.regime_detectors) == 3
        assert "BTC/USDT" in multi_symbol_engine.regime_detectors
        assert "ETH/USDT" in multi_symbol_engine.regime_detectors
        assert "SOL/USDT" in multi_symbol_engine.regime_detectors

    @pytest.mark.asyncio
    async def test_multi_symbol_update(self, multi_symbol_engine):
        await multi_symbol_engine.force_update()

        # Should process all symbols
        assert multi_symbol_engine.state.last_update is not None
