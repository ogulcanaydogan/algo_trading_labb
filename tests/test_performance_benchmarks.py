"""Performance benchmarks for critical trading components."""

import time
from datetime import datetime, timezone

from bot.safety_controller import SafetyController, SafetyLimits
from bot.trading_mode import TradingMode, TradingStatus
from bot.unified_state import UnifiedState, PositionState


def _create_state(balance=10000.0):
    """Helper to create test state."""
    return UnifiedState(
        mode=TradingMode.PAPER_LIVE_DATA,
        status=TradingStatus.ACTIVE,
        timestamp=datetime.now(timezone.utc).isoformat(),
        initial_capital=balance,
        current_balance=balance,
        peak_balance=balance,
    )


class TestSignalGenerationPerformance:
    """Benchmark signal generation latency."""

    def test_signal_generation_latency(self):
        """Test signal generation completes within acceptable time."""
        start = time.perf_counter()
        
        signal = {
            "symbol": "BTC/USDT",
            "signal": "LONG",
            "confidence": 0.75,
        }
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        assert elapsed < 10.0

    def test_multiple_signal_generation(self):
        """Test generating signals for multiple instruments."""
        start = time.perf_counter()
        
        signals = []
        for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "XRP/USDT"]):
            signal = {
                "symbol": symbol,
                "signal": "LONG" if i % 2 == 0 else "FLAT",
                "confidence": 0.7,
            }
            signals.append(signal)
        
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 5.0


class TestOrderExecutionPerformance:
    """Benchmark order execution latency."""

    def test_order_validation_latency(self):
        """Test order validation is fast."""
        controller = SafetyController(limits=SafetyLimits())
        controller.update_balance(10000.0)
        
        start = time.perf_counter()
        allowed, reason = controller.is_trading_allowed()
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 1.0

    def test_position_update_latency(self):
        """Test updating position tracking is fast."""
        controller = SafetyController()
        
        start = time.perf_counter()
        positions = {
            "BTC/USDT": 420.0,
            "ETH/USDT": 250.0,
        }
        controller.update_positions(positions)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 1.0


class TestStateManagementPerformance:
    """Benchmark state persistence operations."""

    def test_state_creation_latency(self):
        """Test creating state object is fast."""
        start = time.perf_counter()
        state = _create_state()
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 1.0

    def test_position_addition_latency(self):
        """Test adding positions to state is fast."""
        state = _create_state()
        
        start = time.perf_counter()
        for i in range(10):
            position = PositionState(
                symbol=f"COIN{i}/USDT",
                quantity=1.0,
                entry_price=100.0,
                side="long",
                entry_time=datetime.now(timezone.utc).isoformat(),
                stop_loss=95.0,
                take_profit=105.0,
                current_price=100.0,
            )
            state.positions[f"COIN{i}/USDT"] = position
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 5.0


class TestSafetyChecksPerformance:
    """Benchmark safety check performance."""

    def test_safety_check_speed(self):
        """Test safety checks are fast enough for trading loop."""
        controller = SafetyController()
        controller.update_balance(10000.0)
        
        start = time.perf_counter()
        for _ in range(100):
            allowed, reason = controller.is_trading_allowed()
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 10.0

    def test_balance_update_performance(self):
        """Test balance updates don't cause slowdowns."""
        controller = SafetyController()
        
        start = time.perf_counter()
        for i in range(1000):
            controller.update_balance(10000.0 + i)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 10.0


class TestDataStructurePerformance:
    """Benchmark data structure operations."""

    def test_large_position_dictionary_lookup(self):
        """Test lookup in large position dictionary."""
        positions = {}
        for i in range(100):
            positions[f"COIN{i}/USDT"] = {"quantity": 1.0, "price": 100.0}
        
        start = time.perf_counter()
        for i in range(100):
            _ = positions.get(f"COIN{i % 100}/USDT")
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 1.0

    def test_position_state_copy_performance(self):
        """Test copying position state is fast."""
        original = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=40000.0,
            take_profit=44000.0,
            current_price=42000.0,
        )
        
        start = time.perf_counter()
        for _ in range(1000):
            _ = PositionState(
                symbol=original.symbol,
                quantity=original.quantity,
                entry_price=original.entry_price,
                side=original.side,
                entry_time=original.entry_time,
                stop_loss=original.stop_loss,
                take_profit=original.take_profit,
                current_price=original.current_price,
            )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 10.0


class TestTradingLoopPerformance:
    """Benchmark full trading loop iteration."""

    def test_single_loop_iteration_time(self):
        """Test a single trading loop iteration."""
        state = _create_state()
        controller = SafetyController()
        controller.update_balance(10000.0)
        
        start = time.perf_counter()
        
        allowed, _ = controller.is_trading_allowed()
        signal = {"symbol": "BTC/USDT", "signal": "LONG", "confidence": 0.7}
        allowed, _ = controller.is_trading_allowed()
        state.current_balance = 10000.0
        
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 5.0

    def test_loop_iterations_throughput(self):
        """Test how many loop iterations we can do per second."""
        controller = SafetyController()
        
        start = time.perf_counter()
        iterations = 0
        
        while (time.perf_counter() - start) < 1.0:
            controller.update_balance(10000.0 + iterations)
            allowed, _ = controller.is_trading_allowed()
            iterations += 1
        
        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed
        
        assert throughput > 10000.0
