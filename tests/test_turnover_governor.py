"""
Tests for Turnover Governor.

Covers:
1. Per-symbol interval enforcement
2. Daily decision limits
3. EV/cost ratio checks
4. State persistence
5. Integration with unified_engine
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.turnover_governor import (
    TurnoverGovernor,
    TurnoverGovernorConfig,
    TurnoverDecision,
    SymbolTurnoverState,
    SymbolOverrideConfig,
    EffectiveSymbolConfig,
    DEFAULT_BTC_OVERRIDE,
    get_turnover_governor,
    reset_turnover_governor,
)


class TestTurnoverGovernorConfig:
    """Test turnover governor configuration."""

    def test_default_config_values(self):
        """Test that default config values are sensible."""
        config = TurnoverGovernorConfig()

        assert config.min_decision_interval_minutes == 15.0
        assert config.max_decisions_per_day == 10
        assert config.min_expected_value_multiple == 2.0
        assert config.default_fee_bps == 10.0
        assert config.default_slippage_bps == 5.0
        assert config.enabled is True

    def test_custom_config_values(self):
        """Test custom configuration."""
        config = TurnoverGovernorConfig(
            min_decision_interval_minutes=30.0,
            max_decisions_per_day=5,
            min_expected_value_multiple=3.0,
            enabled=False,
        )

        assert config.min_decision_interval_minutes == 30.0
        assert config.max_decisions_per_day == 5
        assert config.min_expected_value_multiple == 3.0
        assert config.enabled is False


class TestSymbolTurnoverState:
    """Test symbol turnover state tracking."""

    def test_state_serialization(self):
        """Test state can be serialized and deserialized."""
        state = SymbolTurnoverState(
            symbol="BTC/USDT",
            last_decision_time=datetime(2024, 1, 15, 10, 30),
            decisions_today=5,
            decisions_today_date="2024-01-15",
            blocked_today=2,
            blocked_reasons={"interval": 1, "ev_cost": 1},
            estimated_cost_drag=15.50,
        )

        # Serialize
        data = state.to_dict()

        # Deserialize
        restored = SymbolTurnoverState.from_dict(data)

        assert restored.symbol == "BTC/USDT"
        assert restored.decisions_today == 5
        assert restored.blocked_today == 2
        assert restored.blocked_reasons == {"interval": 1, "ev_cost": 1}
        assert restored.estimated_cost_drag == 15.50

    def test_state_without_last_decision_time(self):
        """Test state without last decision time."""
        state = SymbolTurnoverState(symbol="ETH/USDT")
        data = state.to_dict()

        assert data["last_decision_time"] is None

        restored = SymbolTurnoverState.from_dict(data)
        assert restored.last_decision_time is None


class TestTurnoverGovernor:
    """Test turnover governor core functionality."""

    @pytest.fixture
    def temp_state_path(self, tmp_path):
        """Create temporary state file path."""
        return tmp_path / "turnover_state.json"

    @pytest.fixture
    def governor(self, temp_state_path):
        """Create governor with temporary state."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(
            min_decision_interval_minutes=15.0,
            max_decisions_per_day=5,
            min_expected_value_multiple=2.0,
            state_path=temp_state_path,
        )
        return TurnoverGovernor(config)

    def test_hold_action_always_allowed(self, governor):
        """Test that hold/flat actions are always allowed."""
        for action in ("hold", "flat", "none", "", "HOLD", "FLAT"):
            decision = governor.evaluate_decision(
                symbol="BTC/USDT",
                action=action,
            )
            assert decision.allowed is True
            assert "Hold action" in decision.reason or "no cost" in decision.reason

    def test_first_decision_allowed(self, governor):
        """Test that first decision for a symbol is allowed."""
        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=50.0,
            position_size_usd=1000.0,
            confidence=0.8,
        )

        # Should be allowed (no prior decisions, good EV)
        assert decision.allowed is True
        assert decision.estimated_cost > 0
        assert decision.expected_value > 0

    def test_interval_enforcement(self, governor):
        """Test minimum decision interval enforcement."""
        # First decision allowed
        decision1 = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=50.0,
            position_size_usd=1000.0,
            confidence=0.8,
        )
        assert decision1.allowed is True
        governor.record_decision_taken("BTC/USDT")

        # Second decision immediately blocked by interval
        decision2 = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="sell",
            expected_pnl=50.0,
            position_size_usd=1000.0,
            confidence=0.8,
        )
        assert decision2.allowed is False
        assert decision2.blocked_by == "interval"
        assert "min until next decision" in decision2.reason

    def test_daily_limit_enforcement(self, governor):
        """Test maximum daily decisions enforcement."""
        symbol = "ETH/USDT"

        # Make 5 decisions (the limit)
        for i in range(5):
            decision = governor.evaluate_decision(
                symbol=symbol,
                action="buy",
                expected_pnl=100.0,
                position_size_usd=1000.0,
                confidence=0.9,
            )
            if decision.allowed:
                governor.record_decision_taken(symbol)

            # Clear interval block by manipulating last_decision_time
            state = governor._get_symbol_state(symbol)
            state.last_decision_time = datetime.now() - timedelta(hours=1)

        # 6th decision should be blocked by daily limit
        decision = governor.evaluate_decision(
            symbol=symbol,
            action="buy",
            expected_pnl=100.0,
            position_size_usd=1000.0,
            confidence=0.9,
        )
        assert decision.allowed is False
        assert decision.blocked_by == "daily_limit"
        assert "Daily limit reached" in decision.reason

    def test_ev_cost_enforcement(self, governor):
        """Test EV/cost ratio enforcement."""
        # Low confidence, low expected PnL = poor EV/cost ratio
        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=1.0,  # Very low expected PnL
            position_size_usd=1000.0,
            confidence=0.1,  # Very low confidence
        )

        assert decision.allowed is False
        assert decision.blocked_by == "ev_cost"
        assert "EV/cost ratio" in decision.reason

    def test_ev_cost_allowed_with_good_ratio(self, governor):
        """Test that good EV/cost ratio passes."""
        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=100.0,  # High expected PnL
            position_size_usd=1000.0,
            confidence=0.9,  # High confidence
        )

        assert decision.allowed is True
        assert decision.cost_multiple >= governor.config.min_expected_value_multiple

    def test_governor_disabled(self, temp_state_path):
        """Test that disabled governor allows all decisions."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(
            enabled=False,
            state_path=temp_state_path,
        )
        governor = TurnoverGovernor(config)

        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=0.0,  # Would normally fail EV check
            position_size_usd=1000.0,
            confidence=0.1,
        )

        assert decision.allowed is True
        assert "disabled" in decision.reason.lower()

    def test_record_decision_updates_state(self, governor):
        """Test that recording decision updates state correctly."""
        symbol = "SOL/USDT"

        # Record a decision
        governor.record_decision_taken(symbol)

        state = governor._get_symbol_state(symbol)
        assert state.decisions_today == 1
        assert state.last_decision_time is not None

        # Record another
        state.last_decision_time = datetime.now() - timedelta(hours=1)
        governor.record_decision_taken(symbol)
        assert state.decisions_today == 2

    def test_daily_stats(self, governor):
        """Test daily statistics tracking."""
        # Make some decisions
        governor.evaluate_decision("BTC/USDT", "buy", 100.0, 1000.0, 0.9)
        governor.record_decision_taken("BTC/USDT")

        # Block one (by interval)
        governor.evaluate_decision("BTC/USDT", "sell", 100.0, 1000.0, 0.9)

        stats = governor.get_daily_stats()
        assert stats["total_decisions_allowed"] == 1
        assert stats["total_decisions_blocked"] == 1
        assert stats["blocked_by_interval"] == 1
        assert "symbols_tracked" in stats

    def test_symbol_status(self, governor):
        """Test getting status for a specific symbol."""
        symbol = "AVAX/USDT"

        # Initial status
        status = governor.get_symbol_status(symbol)
        assert status["symbol"] == symbol
        assert status["decisions_today"] == 0
        assert status["decisions_remaining"] == 5  # max_decisions_per_day
        assert status["blocked_today"] == 0

        # After some activity
        governor.record_decision_taken(symbol)
        status = governor.get_symbol_status(symbol)
        assert status["decisions_today"] == 1
        assert status["decisions_remaining"] == 4

    def test_state_persistence(self, temp_state_path):
        """Test that state persists across instances."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(state_path=temp_state_path)

        # Create governor and make decisions
        gov1 = TurnoverGovernor(config)
        gov1.record_decision_taken("BTC/USDT")
        gov1.record_decision_taken("ETH/USDT")

        # Verify state file exists
        assert temp_state_path.exists()

        # Create new governor instance
        gov2 = TurnoverGovernor(config)

        # Should have loaded state
        assert len(gov2._symbol_states) >= 2
        assert "BTC/USDT" in gov2._symbol_states
        assert "ETH/USDT" in gov2._symbol_states

    def test_rl_advisory_flag(self, governor):
        """Test that RL advisory flag is included in reason messages."""
        # Strategy decision
        decision1 = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=1.0,
            position_size_usd=1000.0,
            confidence=0.1,
            is_rl_advisory=False,
        )
        assert decision1.allowed is False
        assert "Strategy blocked" in decision1.reason

        # RL advisory decision
        state = governor._get_symbol_state("ETH/USDT")
        state.last_decision_time = None  # Clear interval

        decision2 = governor.evaluate_decision(
            symbol="ETH/USDT",
            action="buy",
            expected_pnl=1.0,
            position_size_usd=1000.0,
            confidence=0.1,
            is_rl_advisory=True,
        )
        assert decision2.allowed is False
        assert "RL advisory blocked" in decision2.reason

    def test_cost_estimation(self, governor):
        """Test trade cost estimation."""
        cost = governor._estimate_trade_cost(
            symbol="BTC/USDT",
            position_size_usd=10000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
        )

        # Round-trip cost = position * (fee + slippage) * 2 / 10000
        # = 10000 * (10 + 5) * 2 / 10000 = 30
        assert cost == 30.0

    def test_custom_fee_override(self, governor):
        """Test custom fee/slippage override."""
        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=100.0,
            position_size_usd=10000.0,
            confidence=0.9,
            fee_bps=5.0,  # Lower fee
            slippage_bps=2.0,  # Lower slippage
        )

        # With lower costs, should have higher cost_multiple
        assert decision.estimated_cost == 14.0  # 10000 * (5+2) * 2 / 10000

    def test_reset_daily_stats(self, governor):
        """Test daily stats reset."""
        # Generate some activity
        governor.record_decision_taken("BTC/USDT")
        governor.evaluate_decision("BTC/USDT", "sell", 100.0, 1000.0, 0.9)

        # Reset
        governor.reset_daily_stats()

        stats = governor.get_daily_stats()
        assert stats["total_decisions_allowed"] == 0
        assert stats["total_decisions_blocked"] == 0


class TestTurnoverGovernorSingleton:
    """Test singleton pattern for turnover governor."""

    def test_singleton_returns_same_instance(self):
        """Test that get_turnover_governor returns same instance."""
        reset_turnover_governor()
        gov1 = get_turnover_governor()
        gov2 = get_turnover_governor()
        assert gov1 is gov2

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        reset_turnover_governor()
        gov1 = get_turnover_governor()
        reset_turnover_governor()
        gov2 = get_turnover_governor()
        assert gov1 is not gov2


class TestUnifiedEngineIntegration:
    """Test turnover governor integration with unified engine."""

    def test_turnover_governor_available_flag(self):
        """Test that TURNOVER_GOVERNOR_AVAILABLE is True."""
        from bot.unified_engine import TURNOVER_GOVERNOR_AVAILABLE
        assert TURNOVER_GOVERNOR_AVAILABLE is True

    def test_engine_creates_turnover_governor(self):
        """Test that engine initialization creates turnover governor."""
        from bot.unified_engine import UnifiedTradingEngine, EngineConfig
        from bot.trading_mode import TradingMode

        config = EngineConfig(
            initial_mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=10000.0,
            symbols=["BTC/USDT"],
        )

        engine = UnifiedTradingEngine(config)

        # Turnover governor should be created
        assert engine.turnover_governor is not None

    def test_turnover_check_logged(self, caplog):
        """Test that turnover governor logs when blocking decisions."""
        import logging
        from bot.unified_engine import UnifiedTradingEngine, EngineConfig
        from bot.trading_mode import TradingMode

        config = EngineConfig(
            initial_mode=TradingMode.PAPER_LIVE_DATA,
            initial_capital=10000.0,
            symbols=["BTC/USDT"],
        )

        with caplog.at_level(logging.INFO):
            engine = UnifiedTradingEngine(config)

        # Check for initialization log
        log_messages = [record.message for record in caplog.records]
        turnover_logs = [
            msg for msg in log_messages
            if "Turnover Governor" in msg or "turnover" in msg.lower()
        ]
        assert len(turnover_logs) >= 1


class TestDailyHealthReportIntegration:
    """Test turnover governor integration with daily health reports."""

    def test_health_report_includes_turnover_fields(self, tmp_path):
        """Test that daily health report includes turnover governor fields."""
        from scripts.shadow.run_daily_shadow_health import DailyHealthReport

        report = DailyHealthReport(
            date="2024-01-15",
            timestamp=datetime.now().isoformat(),
            turnover_decisions_blocked=5,
            turnover_blocked_by_interval=2,
            turnover_blocked_by_daily_limit=1,
            turnover_blocked_by_ev_cost=2,
            turnover_cost_drag_avoided=45.50,
        )

        report_dict = report.to_dict()

        # Verify turnover_governor section exists
        assert "turnover_governor" in report_dict
        tg = report_dict["turnover_governor"]
        assert tg["decisions_blocked"] == 5
        assert tg["blocked_by_interval"] == 2
        assert tg["blocked_by_daily_limit"] == 1
        assert tg["blocked_by_ev_cost"] == 2
        assert tg["cost_drag_avoided"] == 45.50


class TestSymbolOverrideConfig:
    """Test per-symbol override configuration."""

    def test_default_btc_override_values(self):
        """Test that DEFAULT_BTC_OVERRIDE has stricter values than defaults."""
        assert DEFAULT_BTC_OVERRIDE.min_interval_minutes == 30.0  # vs 15.0 default
        assert DEFAULT_BTC_OVERRIDE.max_decisions_per_day == 6  # vs 10 default
        assert DEFAULT_BTC_OVERRIDE.min_ev_cost_multiple == 2.5  # vs 2.0 default

    def test_symbol_override_serialization(self):
        """Test SymbolOverrideConfig serialization."""
        override = SymbolOverrideConfig(
            min_interval_minutes=45.0,
            max_decisions_per_day=3,
            min_ev_cost_multiple=3.0,
        )

        data = override.to_dict()
        restored = SymbolOverrideConfig.from_dict(data)

        assert restored.min_interval_minutes == 45.0
        assert restored.max_decisions_per_day == 3
        assert restored.min_ev_cost_multiple == 3.0

    def test_partial_override(self):
        """Test that partial overrides only set specified values."""
        override = SymbolOverrideConfig(
            min_interval_minutes=60.0,
            # max_decisions_per_day and min_ev_cost_multiple not set
        )

        assert override.min_interval_minutes == 60.0
        assert override.max_decisions_per_day is None
        assert override.min_ev_cost_multiple is None


class TestPerSymbolOverrides:
    """Test per-symbol turnover governor overrides."""

    @pytest.fixture
    def temp_state_path(self, tmp_path):
        """Create temporary state file path."""
        return tmp_path / "turnover_state.json"

    @pytest.fixture
    def governor_with_defaults(self, temp_state_path):
        """Create governor with default config (includes BTC overrides)."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(state_path=temp_state_path)
        return TurnoverGovernor(config)

    def test_btc_override_applies(self, governor_with_defaults):
        """Test that BTC/USDT gets stricter override rules."""
        governor = governor_with_defaults

        # Get effective config for BTC
        btc_config = governor._get_effective_config("BTC/USDT")

        # BTC should have override applied
        assert btc_config.has_override is True
        assert btc_config.min_interval_minutes == 30.0  # Stricter than default 15
        assert btc_config.max_decisions_per_day == 6  # Stricter than default 10
        assert btc_config.min_ev_cost_multiple == 2.5  # Stricter than default 2.0

    def test_eth_uses_defaults(self, governor_with_defaults):
        """Test that ETH/USDT uses default rules (no override)."""
        governor = governor_with_defaults

        # Get effective config for ETH
        eth_config = governor._get_effective_config("ETH/USDT")

        # ETH should use defaults (no override)
        assert eth_config.has_override is False
        assert eth_config.min_interval_minutes == 15.0  # Default
        assert eth_config.max_decisions_per_day == 10  # Default
        assert eth_config.min_ev_cost_multiple == 2.0  # Default

    def test_btc_stricter_than_eth_interval(self, governor_with_defaults):
        """Test that BTC interval enforcement is stricter than ETH."""
        governor = governor_with_defaults

        # BTC config
        btc_config = governor._get_effective_config("BTC/USDT")
        eth_config = governor._get_effective_config("ETH/USDT")

        # BTC should have longer minimum interval
        assert btc_config.min_interval_minutes > eth_config.min_interval_minutes

    def test_btc_stricter_than_eth_daily_limit(self, governor_with_defaults):
        """Test that BTC daily limit is stricter than ETH."""
        governor = governor_with_defaults

        btc_config = governor._get_effective_config("BTC/USDT")
        eth_config = governor._get_effective_config("ETH/USDT")

        # BTC should have fewer allowed decisions
        assert btc_config.max_decisions_per_day < eth_config.max_decisions_per_day

    def test_btc_stricter_than_eth_ev_multiple(self, governor_with_defaults):
        """Test that BTC EV/cost requirement is stricter than ETH."""
        governor = governor_with_defaults

        btc_config = governor._get_effective_config("BTC/USDT")
        eth_config = governor._get_effective_config("ETH/USDT")

        # BTC should require higher EV/cost ratio
        assert btc_config.min_ev_cost_multiple > eth_config.min_ev_cost_multiple

    def test_btc_interval_blocks_sooner(self, temp_state_path):
        """Test that BTC decisions are blocked by interval sooner than ETH."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(state_path=temp_state_path)
        governor = TurnoverGovernor(config)

        # Record decisions for both
        governor.record_decision_taken("BTC/USDT")
        governor.record_decision_taken("ETH/USDT")

        # Both should be blocked immediately after decision
        btc_decision = governor.evaluate_decision("BTC/USDT", "buy", 100.0, 1000.0, 0.9)
        eth_decision = governor.evaluate_decision("ETH/USDT", "buy", 100.0, 1000.0, 0.9)

        assert btc_decision.allowed is False
        assert btc_decision.blocked_by == "interval"
        assert eth_decision.allowed is False
        assert eth_decision.blocked_by == "interval"

        # BTC should show longer wait time (30 min vs 15 min)
        assert "30" in btc_decision.reason or "limit: 30" in btc_decision.reason

    def test_btc_daily_limit_reached_sooner(self, temp_state_path):
        """Test that BTC hits daily limit sooner than ETH would."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(state_path=temp_state_path)
        governor = TurnoverGovernor(config)

        # Make 6 BTC decisions (BTC limit)
        for i in range(6):
            state = governor._get_symbol_state("BTC/USDT")
            state.last_decision_time = datetime.now() - timedelta(hours=1)
            governor.record_decision_taken("BTC/USDT")

        # Make 6 ETH decisions (still under ETH limit of 10)
        for i in range(6):
            state = governor._get_symbol_state("ETH/USDT")
            state.last_decision_time = datetime.now() - timedelta(hours=1)
            governor.record_decision_taken("ETH/USDT")

        # Clear intervals for both
        governor._get_symbol_state("BTC/USDT").last_decision_time = datetime.now() - timedelta(hours=1)
        governor._get_symbol_state("ETH/USDT").last_decision_time = datetime.now() - timedelta(hours=1)

        # BTC should be blocked by daily limit
        btc_decision = governor.evaluate_decision("BTC/USDT", "buy", 100.0, 1000.0, 0.9)
        assert btc_decision.allowed is False
        assert btc_decision.blocked_by == "daily_limit"
        assert "(6)" in btc_decision.reason  # Shows BTC limit of 6

        # ETH should still be allowed (under 10 limit)
        eth_decision = governor.evaluate_decision("ETH/USDT", "buy", 100.0, 1000.0, 0.9)
        assert eth_decision.allowed is True

    def test_custom_symbol_override(self, temp_state_path):
        """Test adding custom symbol override."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(
            state_path=temp_state_path,
            symbol_overrides={
                "SOL/USDT": SymbolOverrideConfig(
                    min_interval_minutes=5.0,  # Faster trading for SOL
                    max_decisions_per_day=20,
                    min_ev_cost_multiple=1.5,
                ),
            },
        )
        governor = TurnoverGovernor(config)

        sol_config = governor._get_effective_config("SOL/USDT")

        assert sol_config.has_override is True
        assert sol_config.min_interval_minutes == 5.0
        assert sol_config.max_decisions_per_day == 20
        assert sol_config.min_ev_cost_multiple == 1.5

    def test_normalized_symbol_matching(self, governor_with_defaults):
        """Test that symbol matching works with different formats."""
        governor = governor_with_defaults

        # All these should match BTC override
        btc_formats = ["BTC/USDT", "BTC_USDT", "btc/usdt", "BTC-USDT"]

        for symbol in btc_formats:
            config = governor._get_effective_config(symbol)
            assert config.has_override is True, f"Failed for {symbol}"
            assert config.min_interval_minutes == 30.0, f"Wrong interval for {symbol}"

    def test_decision_includes_effective_config(self, governor_with_defaults):
        """Test that TurnoverDecision includes effective config."""
        governor = governor_with_defaults

        # Get a decision for BTC
        decision = governor.evaluate_decision(
            symbol="BTC/USDT",
            action="buy",
            expected_pnl=100.0,
            position_size_usd=1000.0,
            confidence=0.9,
        )

        # Decision should include effective config
        assert decision.effective_config is not None
        assert decision.effective_config.symbol == "BTC/USDT"
        assert decision.effective_config.has_override is True
        assert decision.effective_config.min_interval_minutes == 30.0

    def test_get_all_symbol_configs(self, governor_with_defaults):
        """Test get_all_symbol_configs returns correct data."""
        governor = governor_with_defaults

        # Track some symbols
        governor.evaluate_decision("BTC/USDT", "buy", 100.0, 1000.0, 0.9)
        governor.evaluate_decision("ETH/USDT", "buy", 100.0, 1000.0, 0.9)

        configs = governor.get_all_symbol_configs()

        # Should have default config
        assert "default_config" in configs
        assert configs["default_config"]["min_interval_minutes"] == 15.0

        # Should have override patterns
        assert "override_patterns" in configs
        assert "BTC/USDT" in configs["override_patterns"]

        # Should have tracked symbols
        assert "symbols" in configs
        assert "BTC/USDT" in configs["symbols"]
        assert "ETH/USDT" in configs["symbols"]

        # BTC should show as override, ETH should not
        assert configs["symbols"]["BTC/USDT"]["has_override"] is True
        assert configs["symbols"]["ETH/USDT"]["has_override"] is False


class TestReportIncludesPerSymbolConfig:
    """Test that reports include per-symbol config and block reasons."""

    def test_report_includes_per_symbol_configs(self, tmp_path):
        """Test that daily health report includes per-symbol turnover configs."""
        from scripts.shadow.run_daily_shadow_health import DailyHealthReport

        per_symbol_configs = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "min_interval_minutes": 30.0,
                "max_decisions_per_day": 6,
                "min_ev_cost_multiple": 2.5,
                "has_override": True,
                "override_source": "BTC/USDT",
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "min_interval_minutes": 15.0,
                "max_decisions_per_day": 10,
                "min_ev_cost_multiple": 2.0,
                "has_override": False,
                "override_source": "",
            },
        }

        report = DailyHealthReport(
            date="2024-01-15",
            timestamp=datetime.now().isoformat(),
            turnover_decisions_blocked=3,
            turnover_blocked_by_interval=2,
            turnover_blocked_by_daily_limit=1,
            turnover_blocked_by_ev_cost=0,
            turnover_cost_drag_avoided=25.00,
            turnover_per_symbol_configs=per_symbol_configs,
        )

        report_dict = report.to_dict()
        tg = report_dict["turnover_governor"]

        # Check per-symbol configs are included
        assert "per_symbol_configs" in tg
        assert "BTC/USDT" in tg["per_symbol_configs"]
        assert "ETH/USDT" in tg["per_symbol_configs"]

        # BTC should show override
        btc = tg["per_symbol_configs"]["BTC/USDT"]
        assert btc["has_override"] is True
        assert btc["min_interval_minutes"] == 30.0

        # ETH should show defaults
        eth = tg["per_symbol_configs"]["ETH/USDT"]
        assert eth["has_override"] is False
        assert eth["min_interval_minutes"] == 15.0

    def test_report_includes_per_symbol_blocked_reasons(self, tmp_path):
        """Test that report includes per-symbol block reasons."""
        from scripts.shadow.run_daily_shadow_health import DailyHealthReport

        per_symbol_blocked = {
            "BTC/USDT": {
                "blocked": 2,
                "cost_avoided": 15.50,
                "reasons": {"interval": 1, "ev_cost": 1},
            },
            "ETH/USDT": {
                "blocked": 1,
                "cost_avoided": 9.50,
                "reasons": {"daily_limit": 1},
            },
        }

        report = DailyHealthReport(
            date="2024-01-15",
            timestamp=datetime.now().isoformat(),
            turnover_decisions_blocked=3,
            turnover_per_symbol_blocked=per_symbol_blocked,
        )

        report_dict = report.to_dict()
        tg = report_dict["turnover_governor"]

        # Check per-symbol blocked stats
        assert "per_symbol_blocked" in tg
        assert "BTC/USDT" in tg["per_symbol_blocked"]
        assert "ETH/USDT" in tg["per_symbol_blocked"]

        # BTC block reasons
        btc_blocked = tg["per_symbol_blocked"]["BTC/USDT"]
        assert btc_blocked["blocked"] == 2
        assert btc_blocked["reasons"]["interval"] == 1
        assert btc_blocked["reasons"]["ev_cost"] == 1

        # ETH block reasons
        eth_blocked = tg["per_symbol_blocked"]["ETH/USDT"]
        assert eth_blocked["blocked"] == 1
        assert eth_blocked["reasons"]["daily_limit"] == 1

    def test_symbol_status_includes_effective_config(self, tmp_path):
        """Test that get_symbol_status includes effective config."""
        reset_turnover_governor()
        config = TurnoverGovernorConfig(state_path=tmp_path / "state.json")
        governor = TurnoverGovernor(config)

        # Get BTC status
        btc_status = governor.get_symbol_status("BTC/USDT")

        assert "effective_config" in btc_status
        assert btc_status["effective_config"]["has_override"] is True
        assert btc_status["effective_config"]["min_interval_minutes"] == 30.0

        # Get ETH status
        eth_status = governor.get_symbol_status("ETH/USDT")

        assert "effective_config" in eth_status
        assert eth_status["effective_config"]["has_override"] is False
        assert eth_status["effective_config"]["min_interval_minutes"] == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
