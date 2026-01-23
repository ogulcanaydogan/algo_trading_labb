#!/usr/bin/env python3
"""
Smoke Test Suite

A single script that proves everything starts and basic functionality works.
Run this after any major changes to verify system integrity.

Usage:
    python tests/smoke_test.py
    python tests/smoke_test.py --api-only  # Skip module tests
    python tests/smoke_test.py --verbose   # Show all output
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Callable, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colored(text: str, color: str) -> str:
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.RESET}"


class SmokeTestRunner:
    """Runner for smoke tests with pretty output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results: List[Tuple[str, bool, str]] = []

    def log(self, message: str, color: str = Colors.RESET):
        """Log a message."""
        print(colored(message, color))

    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log test result."""
        self.results.append((name, passed, message))

        if passed:
            self.passed += 1
            status = colored("[PASS]", Colors.GREEN)
        else:
            self.failed += 1
            status = colored("[FAIL]", Colors.RED)

        print(f"  {status} {name}")
        if message and (not passed or self.verbose):
            print(f"         {message}")

    def log_skip(self, name: str, reason: str):
        """Log skipped test."""
        self.skipped += 1
        self.results.append((name, None, reason))
        status = colored("[SKIP]", Colors.YELLOW)
        print(f"  {status} {name}")
        if self.verbose:
            print(f"         {reason}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print(colored("SMOKE TEST SUMMARY", Colors.BOLD))
        print("=" * 60)

        total = self.passed + self.failed + self.skipped
        print(f"  Total:   {total}")
        print(f"  {colored('Passed:', Colors.GREEN)}  {self.passed}")
        print(f"  {colored('Failed:', Colors.RED)}  {self.failed}")
        print(f"  {colored('Skipped:', Colors.YELLOW)} {self.skipped}")

        if self.failed == 0:
            print("\n" + colored("All smoke tests passed!", Colors.GREEN + Colors.BOLD))
        else:
            print(
                "\n"
                + colored("Some tests failed. Review the output above.", Colors.RED + Colors.BOLD)
            )

        return self.failed == 0


# =============================================================================
# PYTEST FIXTURE
# =============================================================================

import pytest


@pytest.fixture
def runner():
    """Provide a SmokeTestRunner instance for tests."""
    return SmokeTestRunner(verbose=True)


# =============================================================================
# MODULE IMPORT TESTS
# =============================================================================


def test_core_imports(runner: SmokeTestRunner):
    """Test that core modules can be imported."""
    runner.log("\n[1/5] Testing Core Module Imports...", Colors.BLUE)

    modules = [
        ("bot.unified_engine", "UnifiedTradingEngine"),
        ("bot.unified_state", "UnifiedState"),
        ("bot.safety_controller", "SafetyController"),
        ("bot.ai_trading_brain", "get_ai_brain"),
        ("bot.ml_signal_generator", "MLSignalGenerator"),
        ("bot.ml_performance_tracker", "MLPerformanceTracker"),
        ("bot.trade_alerts", "create_trade_alert_manager"),
    ]

    for module_name, attr_name in modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, attr_name):
                runner.log_test(f"Import {module_name}.{attr_name}", True)
            else:
                runner.log_test(
                    f"Import {module_name}.{attr_name}",
                    False,
                    f"Module exists but '{attr_name}' not found",
                )
        except ImportError as e:
            runner.log_test(f"Import {module_name}", False, str(e))
        except Exception as e:
            runner.log_test(f"Import {module_name}", False, f"Unexpected error: {e}")


def test_ml_imports(runner: SmokeTestRunner):
    """Test ML module imports."""
    runner.log("\n[2/5] Testing ML Module Imports...", Colors.BLUE)

    modules = [
        "bot.ml.predictor",
        "bot.ml.feature_engineer",
        "bot.multi_timeframe",
    ]

    for module_name in modules:
        try:
            importlib.import_module(module_name)
            runner.log_test(f"Import {module_name}", True)
        except ImportError as e:
            # Some ML modules may have optional dependencies
            runner.log_skip(f"Import {module_name}", f"Optional dependency: {e}")
        except Exception as e:
            runner.log_test(f"Import {module_name}", False, str(e))


# =============================================================================
# STATE TESTS
# =============================================================================


def test_state_persistence(runner: SmokeTestRunner):
    """Test state can be created and loaded."""
    runner.log("\n[3/5] Testing State Persistence...", Colors.BLUE)

    try:
        from bot.unified_state import UnifiedStateStore, TradingMode, TradingStatus

        # Create temp directory for test
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = UnifiedStateStore(data_dir=tmpdir)

            # Test state initialization
            state = store.initialize(
                mode=TradingMode.PAPER_LIVE_DATA, initial_capital=10000.0, resume=False
            )
            runner.log_test("Create state", state is not None)

            # Test state update
            store.update_state(current_balance=10100.0)
            runner.log_test("Update state", True)

            # Test state get
            state2 = store.get_state()
            runner.log_test(
                "Get state",
                state2 is not None and state2.current_balance == 10100.0,
                f"Expected 10100.0, got {state2.current_balance if state2 else 'None'}",
            )

    except Exception as e:
        runner.log_test("State persistence", False, str(e))


# =============================================================================
# AI BRAIN TESTS
# =============================================================================


def test_ai_brain(runner: SmokeTestRunner):
    """Test AI Brain components."""
    runner.log("\n[4/5] Testing AI Brain Components...", Colors.BLUE)

    try:
        from bot.ai_trading_brain import get_ai_brain, MarketSnapshot, MarketCondition
        from datetime import datetime

        brain = get_ai_brain()
        runner.log_test("Get AI Brain instance", brain is not None)

        # Test daily target tracker
        try:
            tracker = brain.daily_tracker
            runner.log_test("Daily target tracker exists", tracker is not None)
        except Exception as e:
            runner.log_test("Daily target tracker exists", False, str(e))

        # Test market snapshot creation
        try:
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbol="BTC/USDT",
                price=42000.0,
                trend_1h="up",
                rsi=55.0,
                volatility_percentile=50.0,
                condition=MarketCondition.BULL,
            )
            runner.log_test("Create market snapshot", snapshot is not None)
        except Exception as e:
            runner.log_test("Create market snapshot", False, str(e))

        # Test strategy generator
        try:
            generator = brain.strategy_generator
            runner.log_test("Strategy generator exists", generator is not None)
        except Exception as e:
            runner.log_test("Strategy generator exists", False, str(e))

        # Test get brain status
        try:
            status = brain.get_brain_status()
            runner.log_test(
                "Get brain status", isinstance(status, dict), f"Got type: {type(status)}"
            )
        except Exception as e:
            runner.log_test("Get brain status", False, str(e))

    except Exception as e:
        runner.log_test("AI Brain import", False, str(e))


# =============================================================================
# ML PERFORMANCE TRACKER TESTS
# =============================================================================


def test_ml_tracker(runner: SmokeTestRunner):
    """Test ML Performance Tracker."""
    runner.log("\n[5/5] Testing ML Performance Tracker...", Colors.BLUE)

    try:
        from bot.ml_performance_tracker import MLPerformanceTracker
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_ml.db"
            tracker = MLPerformanceTracker(db_path=str(db_path))

            # Test prediction recording
            pred_id = tracker.record_prediction(
                model_type="test_model",
                symbol="BTC/USDT",
                prediction="buy",
                confidence=0.75,
                market_condition="bull",
                volatility=50.0,
            )
            runner.log_test("Record prediction", pred_id is not None)

            # Test outcome recording
            tracker.record_outcome(pred_id, actual_return=1.5)
            runner.log_test("Record outcome", True)

            # Test performance query
            perf = tracker.get_model_performance("test_model", days=30)
            runner.log_test("Query performance", isinstance(perf, dict), f"Got: {perf}")

            # Test ranking
            ranking = tracker.get_model_ranking(days=30)
            runner.log_test(
                "Get model ranking", isinstance(ranking, list), f"Got {len(ranking)} models"
            )

    except Exception as e:
        runner.log_test("ML Tracker", False, str(e))


# =============================================================================
# API TESTS (requires running server)
# =============================================================================


@pytest.mark.asyncio
async def test_api_endpoints(runner: SmokeTestRunner, base_url: str = "http://localhost:8000"):
    """Test API endpoints if server is running."""
    runner.log("\n[API] Testing API Endpoints...", Colors.BLUE)

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            try:
                resp = await client.get(f"{base_url}/health")
                runner.log_test(
                    "GET /health", resp.status_code == 200, f"Status: {resp.status_code}"
                )

                # Verify response shape
                data = resp.json()
                has_status = "status" in data
                runner.log_test(
                    "Health response shape", has_status, f"Has 'status' field: {has_status}"
                )
            except httpx.ConnectError:
                runner.log_skip("API endpoints", "Server not running at " + base_url)
                return

            # Risk settings
            resp = await client.get(f"{base_url}/api/trading/risk-settings")
            if resp.status_code == 200:
                data = resp.json()
                has_fields = all(k in data for k in ["shorting", "leverage", "aggressive"])
                runner.log_test("GET /api/trading/risk-settings", has_fields, f"Response: {data}")
            else:
                runner.log_test(
                    "GET /api/trading/risk-settings", False, f"Status: {resp.status_code}"
                )

            # AI Brain status
            resp = await client.get(f"{base_url}/api/ai-brain/status")
            if resp.status_code == 200:
                runner.log_test("GET /api/ai-brain/status", True)
            else:
                runner.log_test("GET /api/ai-brain/status", False, f"Status: {resp.status_code}")

            # ML model performance
            resp = await client.get(f"{base_url}/api/ml/model-performance")
            if resp.status_code == 200:
                runner.log_test("GET /api/ml/model-performance", True)
            else:
                runner.log_test(
                    "GET /api/ml/model-performance", False, f"Status: {resp.status_code}"
                )

    except ImportError:
        runner.log_skip("API endpoints", "httpx not installed")
    except Exception as e:
        runner.log_test("API endpoints", False, str(e))


# =============================================================================
# CONTRACT TESTS
# =============================================================================


def test_api_contracts(runner: SmokeTestRunner):
    """Test that API response contracts are maintained."""
    runner.log("\n[Contracts] Testing Response Contracts...", Colors.BLUE)

    # Test PositionState contract
    try:
        from bot.unified_state import PositionState

        pos = PositionState(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=42000.0,
            side="long",
            entry_time="2026-01-15T10:00:00Z",
        )

        required_fields = ["symbol", "quantity", "entry_price", "side", "entry_time"]
        has_fields = all(hasattr(pos, f) for f in required_fields)
        runner.log_test("PositionState contract", has_fields, f"Required fields: {required_fields}")
    except Exception as e:
        runner.log_test("PositionState contract", False, str(e))

    # Test TradeRecord contract
    try:
        from bot.unified_state import TradeRecord

        trade = TradeRecord(
            id="test_123",
            symbol="BTC/USDT",
            side="long",
            quantity=0.01,
            entry_price=42000.0,
            exit_price=42500.0,
            pnl=5.0,
            pnl_pct=1.19,
            entry_time="2026-01-15T08:00:00Z",
            exit_time="2026-01-15T10:00:00Z",
            exit_reason="take_profit",
            mode="paper_live_data",
        )

        required_fields = [
            "id",
            "symbol",
            "side",
            "quantity",
            "entry_price",
            "exit_price",
            "pnl",
            "pnl_pct",
            "entry_time",
            "exit_time",
            "exit_reason",
            "mode",
        ]
        has_fields = all(hasattr(trade, f) for f in required_fields)
        runner.log_test("TradeRecord contract", has_fields, f"Required fields present")
    except Exception as e:
        runner.log_test("TradeRecord contract", False, str(e))


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests")
    parser.add_argument("--api-only", action="store_true", help="Only test API endpoints")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    runner = SmokeTestRunner(verbose=args.verbose)

    print("\n" + "=" * 60)
    print(colored("ALGO TRADING LAB - SMOKE TEST SUITE", Colors.BOLD + Colors.BLUE))
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"API URL: {args.api_url}")
    print("=" * 60)

    start_time = time.time()

    if not args.api_only:
        # Run module tests
        test_core_imports(runner)
        test_ml_imports(runner)
        test_state_persistence(runner)
        test_ai_brain(runner)
        test_ml_tracker(runner)
        test_api_contracts(runner)

    # Run API tests
    asyncio.run(test_api_endpoints(runner, args.api_url))

    elapsed = time.time() - start_time

    runner.print_summary()
    print(f"\nCompleted in {elapsed:.2f}s")

    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    main()
