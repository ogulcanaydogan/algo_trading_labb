"""
Integration Tests for Shadow Data Collection in Paper Trading.

Tests cover:
1. Heartbeat is written at startup
2. heartbeat_recent becomes 1 after paper trading starts
3. /health/shadow API returns heartbeat_recent=1
4. PAPER_LIVE decisions are logged (not TEST)
"""

import json
import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockExecutionAdapter:
    """Mock execution adapter for testing."""

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_balance(self):
        class Balance:
            available = 10000.0
            in_positions = 0.0
        return Balance()

    async def get_current_price(self, symbol: str) -> float:
        return 50000.0 if "BTC" in symbol else 3000.0


class TestShadowPaperIntegration:
    """Integration tests for shadow data collection in paper trading."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directories."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "rl").mkdir()
        (data_dir / "reports").mkdir()
        return data_dir

    @pytest.fixture
    def heartbeat_path(self, temp_data_dir):
        """Return heartbeat file path."""
        return temp_data_dir / "rl" / "paper_live_heartbeat.json"

    def test_heartbeat_written_at_engine_start(self, temp_data_dir, heartbeat_path):
        """Test that heartbeat is written when paper trading engine starts."""
        from bot.rl.shadow_data_collector import (
            write_paper_live_heartbeat,
            read_paper_live_heartbeat,
            is_heartbeat_recent,
        )

        # Initially no heartbeat
        assert not heartbeat_path.exists()

        # Simulate engine startup writing heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT", "ETH/USDT"],
            total_decisions=0,
            paper_live_decisions=0,
            heartbeat_path=heartbeat_path,
        )

        # Heartbeat should now exist
        assert heartbeat_path.exists()

        # Read and verify
        heartbeat = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert heartbeat is not None
        assert heartbeat["shadow_collector_attached"] is True
        assert heartbeat["mode"] == "PAPER_LIVE"
        assert "BTC/USDT" in heartbeat["symbols"]
        assert "ETH/USDT" in heartbeat["symbols"]

    def test_heartbeat_recent_after_startup(self, temp_data_dir, heartbeat_path):
        """Test that heartbeat_recent becomes 1 after paper trading starts."""
        from bot.rl.shadow_data_collector import (
            write_paper_live_heartbeat,
            read_paper_live_heartbeat,
            is_heartbeat_recent,
        )

        # Write heartbeat (simulating engine start)
        write_paper_live_heartbeat(
            symbols=["BTC/USDT"],
            heartbeat_path=heartbeat_path,
        )

        # Check heartbeat is recent
        heartbeat = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert heartbeat is not None
        assert is_heartbeat_recent(heartbeat, max_age_hours=2.0) is True

    def test_shadow_health_api_returns_heartbeat_recent(self, temp_data_dir, heartbeat_path):
        """Test that /health/shadow returns heartbeat_recent=1 after startup."""
        from bot.rl.shadow_data_collector import write_paper_live_heartbeat
        from api.shadow_health_metrics import ShadowHealthMetricsCalculator

        # Write heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT", "ETH/USDT"],
            total_decisions=5,
            paper_live_decisions=5,
            heartbeat_path=heartbeat_path,
        )

        # Calculate metrics using the test heartbeat path
        calculator = ShadowHealthMetricsCalculator(
            reports_dir=temp_data_dir / "reports",
            heartbeat_path=heartbeat_path,
        )
        metrics = calculator.calculate_metrics()

        # heartbeat_recent should be 1
        assert metrics.heartbeat_recent == 1

    def test_paper_live_decisions_counted(self, temp_data_dir, heartbeat_path):
        """Test that PAPER_LIVE decisions are properly counted in metrics."""
        from bot.rl.shadow_data_collector import write_paper_live_heartbeat
        from api.shadow_health_metrics import ShadowHealthMetricsCalculator

        reports_dir = temp_data_dir / "reports"

        # Create a daily report with PAPER_LIVE decisions
        today = datetime.now().strftime("%Y-%m-%d")
        daily_report = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {
                "decisions_by_mode": {
                    "TEST": 2,
                    "PAPER_LIVE": 10,  # 10 PAPER_LIVE decisions
                },
            },
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump(daily_report, f)

        # Write heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT"],
            total_decisions=12,
            paper_live_decisions=10,
            heartbeat_path=heartbeat_path,
        )

        # Calculate metrics
        calculator = ShadowHealthMetricsCalculator(
            reports_dir=reports_dir,
            heartbeat_path=heartbeat_path,
        )
        metrics = calculator.calculate_metrics()

        # Verify PAPER_LIVE decisions are counted
        assert metrics.paper_live_decisions_today == 10
        assert metrics.paper_live_decisions_7d == 10
        assert metrics.heartbeat_recent == 1

    def test_decision_snapshot_uses_paper_live_mode(self):
        """Test that DecisionSnapshot defaults to PAPER_LIVE data_mode."""
        from bot.rl.shadow_data_collector import (
            DecisionSnapshot,
            DATA_MODE_PAPER_LIVE,
            DATA_MODE_TEST,
        )

        # Create a decision snapshot (as the engine would)
        snapshot = DecisionSnapshot(
            symbol="BTC/USDT",
            price=50000.0,
            actual_action="buy",
            actual_confidence=0.8,
        )

        # Should default to PAPER_LIVE
        assert snapshot.data_mode == DATA_MODE_PAPER_LIVE

        # Verify in to_dict output
        snapshot_dict = snapshot.to_dict()
        assert snapshot_dict["data_mode"] == "PAPER_LIVE"

    def test_heartbeat_updates_with_decision_counts(self, temp_data_dir, heartbeat_path):
        """Test that heartbeat updates include decision counts."""
        from bot.rl.shadow_data_collector import (
            write_paper_live_heartbeat,
            read_paper_live_heartbeat,
        )

        # Initial heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT"],
            total_decisions=0,
            paper_live_decisions=0,
            heartbeat_path=heartbeat_path,
        )

        hb1 = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert hb1["total_decisions_session"] == 0
        assert hb1["paper_live_decisions_session"] == 0

        # Updated heartbeat after some decisions
        write_paper_live_heartbeat(
            symbols=["BTC/USDT"],
            total_decisions=15,
            paper_live_decisions=15,
            heartbeat_path=heartbeat_path,
        )

        hb2 = read_paper_live_heartbeat(heartbeat_path=heartbeat_path)
        assert hb2["total_decisions_session"] == 15
        assert hb2["paper_live_decisions_session"] == 15


class TestUnifiedEngineIntegration:
    """Test shadow collector integration in UnifiedTradingEngine."""

    def test_shadow_collector_available_flag(self):
        """Test that SHADOW_COLLECTOR_AVAILABLE is True when imports succeed."""
        from bot.unified_engine import SHADOW_COLLECTOR_AVAILABLE
        assert SHADOW_COLLECTOR_AVAILABLE is True

    def test_engine_config_creates_shadow_collector(self):
        """Test that engine initialization creates shadow collector."""
        from bot.unified_engine import UnifiedTradingEngine, EngineConfig
        from bot.trading_mode import TradingMode

        config = EngineConfig(
            initial_mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=10000.0,
            symbols=["BTC/USDT"],
        )

        engine = UnifiedTradingEngine(config)

        # Shadow collector should be created
        assert engine._shadow_enabled is True
        assert engine.shadow_collector is not None

    def test_shadow_attachment_log_message(self, caplog):
        """Test that shadow attachment is logged correctly."""
        import logging
        from bot.unified_engine import UnifiedTradingEngine, EngineConfig
        from bot.trading_mode import TradingMode

        config = EngineConfig(
            initial_mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=10000.0,
            symbols=["BTC/USDT", "ETH/USDT"],
        )

        with caplog.at_level(logging.INFO):
            engine = UnifiedTradingEngine(config)

        # Check for attachment log message
        log_messages = [record.message for record in caplog.records]
        attachment_logs = [
            msg for msg in log_messages
            if "Shadow collector attached" in msg and "PAPER_LIVE" in msg
        ]
        assert len(attachment_logs) >= 1

    def test_engine_updates_heartbeat_from_shadow_stats(self, monkeypatch):
        """Test that engine heartbeat update uses shadow collector stats."""
        from bot.unified_engine import UnifiedTradingEngine, EngineConfig
        from bot.trading_mode import TradingMode
        import bot.unified_engine as unified_engine

        config = EngineConfig(
            initial_mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=10000.0,
            symbols=["BTC/USDT"],
        )
        engine = UnifiedTradingEngine(config)

        captured: dict[str, object] = {}

        def fake_write_paper_live_heartbeat(
            symbols,
            last_decision_ts=None,
            total_decisions=0,
            paper_live_decisions=0,
            heartbeat_path=None,
        ):
            captured["symbols"] = symbols
            captured["last_decision_ts"] = last_decision_ts
            captured["total"] = total_decisions
            captured["paper_live"] = paper_live_decisions

        monkeypatch.setattr(unified_engine, "write_paper_live_heartbeat", fake_write_paper_live_heartbeat)

        if engine.shadow_collector is not None:
            engine.shadow_collector.get_collection_stats = lambda: {
                "total_decisions": 2,
                "paper_live_decisions": 2,
                "last_decision_ts": "2026-01-31T00:00:00",
            }

        engine._update_shadow_heartbeat()

        assert captured["symbols"] == ["BTC/USDT"]
        assert captured["total"] == 2
        assert captured["paper_live"] == 2
        assert captured["last_decision_ts"] == "2026-01-31T00:00:00"


class TestBlockedSignalLogging:
    """Test that blocked signals are logged as HOLD decisions."""

    def test_blocked_signal_logged_as_hold(self, tmp_path):
        """Test that blocked signals (signal=None) are logged to shadow collector."""
        from bot.rl.shadow_data_collector import (
            ShadowDataCollector,
            ShadowCollectorConfig,
            DATA_MODE_PAPER_LIVE,
        )
        from bot.rl.multi_agent_system import MarketState

        # Setup shadow collector
        log_path = tmp_path / "shadow_decisions.jsonl"
        config = ShadowCollectorConfig(
            enabled=True,
            log_path=log_path,
            enable_rl_shadow=False,  # Disable RL for simpler test
        )
        collector = ShadowDataCollector(config)

        # Record a blocked signal (simulating what unified_engine does when signal=None)
        market_state = MarketState(
            symbol="BTC/USDT",
            price=85000.0,
            volatility=0.0,
            regime="unknown",
            fear_greed=50.0,
            news_sentiment=0.0,
            rsi=50.0,
            trend_strength=0.0,
        )

        decision_id = collector.record_decision_point(
            symbol="BTC/USDT",
            market_state=market_state,
            gate_approved=False,  # Blocked signals set this to False
            gate_score=0.0,
            gate_rejection_reason="signal_blocked_at_generator",
            preservation_level="normal",
            actual_action="hold",  # Blocked = HOLD
            actual_confidence=0.0,
            strategy_used="signal_blocked",  # Identifies this as a blocked signal
        )

        # Verify decision was recorded
        assert decision_id.startswith("DEC_")

        # Verify session counters incremented
        stats = collector.get_collection_stats()
        assert stats["total_decisions"] == 1
        assert stats["paper_live_decisions"] == 1

        # Verify log file was written
        assert log_path.exists()

        # Verify content of log
        with open(log_path, "r") as f:
            logged_decision = json.loads(f.readline())

        assert logged_decision["data_mode"] == "PAPER_LIVE"
        assert logged_decision["actual_decision"]["action"] == "hold"
        assert logged_decision["actual_decision"]["confidence"] == 0.0
        assert logged_decision["actual_decision"]["strategy_used"] == "signal_blocked"
        assert logged_decision["gate_decision"]["approved"] is False
        assert logged_decision["gate_decision"]["rejection_reason"] == "signal_blocked_at_generator"
        assert logged_decision["execution"]["executed"] is False

    def test_multiple_blocked_signals_increment_counters(self, tmp_path):
        """Test that multiple blocked signals properly increment heartbeat counters."""
        from bot.rl.shadow_data_collector import (
            ShadowDataCollector,
            ShadowCollectorConfig,
        )
        from bot.rl.multi_agent_system import MarketState

        log_path = tmp_path / "shadow_decisions.jsonl"
        config = ShadowCollectorConfig(
            enabled=True,
            log_path=log_path,
            enable_rl_shadow=False,
        )
        collector = ShadowDataCollector(config)

        # Simulate 3 blocked signals for different symbols
        for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            market_state = MarketState(
                symbol=symbol,
                price=1000.0,
                volatility=0.0,
                regime="unknown",
                fear_greed=50.0,
                news_sentiment=0.0,
                rsi=50.0,
                trend_strength=0.0,
            )

            collector.record_decision_point(
                symbol=symbol,
                market_state=market_state,
                gate_approved=False,
                gate_score=0.0,
                gate_rejection_reason="signal_blocked_at_generator",
                preservation_level="normal",
                actual_action="hold",
                actual_confidence=0.0,
                strategy_used="signal_blocked",
            )

        # Verify counters
        stats = collector.get_collection_stats()
        assert stats["total_decisions"] == 3
        assert stats["paper_live_decisions"] == 3
        assert stats["last_decision_ts"] is not None

        # Verify all 3 are in the log
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3


class TestShadowHealthAPIResponse:
    """Test full API response structure."""

    def test_api_response_schema_with_heartbeat(self, tmp_path):
        """Test that API returns correct schema with heartbeat data."""
        from bot.rl.shadow_data_collector import write_paper_live_heartbeat
        from api.shadow_health_metrics import get_shadow_health_metrics
        from api.schemas import ShadowHealthResponse

        # Setup test directories
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        heartbeat_path = tmp_path / "heartbeat.json"

        # Create heartbeat
        write_paper_live_heartbeat(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            total_decisions=25,
            paper_live_decisions=20,
            heartbeat_path=heartbeat_path,
        )

        # Create daily report
        today = datetime.now().strftime("%Y-%m-%d")
        daily_report = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "shadow_collection": {
                "decisions_by_mode": {"TEST": 5, "PAPER_LIVE": 20},
            },
            "summary": {"overall_health": "HEALTHY"},
        }
        with open(reports_dir / f"daily_shadow_health_{today}.json", "w") as f:
            json.dump(daily_report, f)

        # Get metrics using custom paths
        from api.shadow_health_metrics import ShadowHealthMetricsCalculator
        calculator = ShadowHealthMetricsCalculator(
            reports_dir=reports_dir,
            heartbeat_path=heartbeat_path,
        )
        metrics = calculator.calculate_metrics()

        # Create API response
        response = ShadowHealthResponse(
            paper_live_decisions_today=metrics.paper_live_decisions_today,
            paper_live_decisions_7d=metrics.paper_live_decisions_7d,
            paper_live_days_streak=metrics.paper_live_days_streak,
            paper_live_weeks_counted=metrics.paper_live_weeks_counted,
            heartbeat_recent=metrics.heartbeat_recent,
            latest_report_timestamp=metrics.latest_report_timestamp,
            gate_1_progress=metrics.gate_1_progress,
            overall_health=metrics.overall_health,
        )

        # Verify response
        assert response.paper_live_decisions_today == 20
        assert response.paper_live_decisions_7d == 20
        assert response.heartbeat_recent == 1
        assert response.overall_health == "HEALTHY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
