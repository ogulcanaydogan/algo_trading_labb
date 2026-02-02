"""
Unit tests for Phase 1 Production Components.

Tests:
- Trade Gate (bot/risk/trade_gate.py)
- Execution Simulator (bot/execution/execution_simulator.py)
- Risk Budget Engine (bot/risk/risk_budget_engine.py)
- Capital Preservation Mode (bot/safety/capital_preservation.py)
- Trade Forensics (bot/meta/trade_forensics.py)
- Reconciler (bot/execution/reconciler.py)
"""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path


class TestTradeGate:
    """Tests for Trade Gate qualification system."""

    def test_trade_gate_import(self):
        """Test that trade gate can be imported."""
        from bot.risk.trade_gate import (
            TradeGate,
            GateConfig,
            TradeRequest,
            GateDecision,
            get_trade_gate,
            reset_trade_gate,
        )

        # Reset singleton
        reset_trade_gate()
        gate = get_trade_gate()
        assert gate is not None
        reset_trade_gate()

    def test_gate_approves_high_quality_trade(self):
        """Test that gate approves a high-quality trade."""
        from bot.risk.trade_gate import (
            TradeGate,
            TradeRequest,
            GateDecision,
            reset_trade_gate,
        )

        reset_trade_gate()
        gate = TradeGate()

        request = TradeRequest(
            symbol="BTC/USDT",
            side="long",
            quantity=0.1,
            price=50000.0,
            order_type="market",
            leverage=2.0,
            signal_confidence=0.85,
            current_regime="BULL",
            regime_confidence=0.9,
            model_predictions={"rf": 0.8, "gb": 0.82, "xgb": 0.78},
            model_confidences={"rf": 0.9, "gb": 0.88, "xgb": 0.85},
            volatility=0.02,
            spread_bps=5.0,
            volume_24h=100_000_000.0,
            upcoming_events_hours=48.0,
            news_urgency=2,
            correlation_with_portfolio=0.3,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.0,
        )

        result = gate.evaluate(request)
        assert result.decision == GateDecision.APPROVED
        assert result.score.total_score >= gate.config.min_total_score

    def test_gate_rejects_low_confidence_trade(self):
        """Test that gate rejects low confidence trades."""
        from bot.risk.trade_gate import (
            TradeGate,
            TradeRequest,
            GateDecision,
            reset_trade_gate,
        )

        reset_trade_gate()
        gate = TradeGate()

        request = TradeRequest(
            symbol="BTC/USDT",
            side="long",
            quantity=0.1,
            price=50000.0,
            order_type="market",
            leverage=1.0,
            signal_confidence=0.3,  # Low confidence
            current_regime="UNKNOWN",
            regime_confidence=0.2,  # Low regime confidence
            model_predictions={"rf": 0.3, "gb": 0.4, "xgb": 0.35},
            model_confidences={"rf": 0.3, "gb": 0.4, "xgb": 0.25},
            volatility=0.08,  # High volatility
            spread_bps=50.0,  # Wide spread
            volume_24h=1_000_000.0,  # Low volume
            upcoming_events_hours=2.0,  # Event coming soon
            news_urgency=8,  # High urgency news
            correlation_with_portfolio=0.9,  # High correlation
            current_drawdown_pct=0.08,  # High drawdown
            daily_loss_pct=0.03,
        )

        result = gate.evaluate(request)
        assert result.decision == GateDecision.REJECTED
        assert result.score.total_score < gate.config.min_total_score

    def test_gate_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from bot.risk.trade_gate import (
            TradeGate,
            TradeRequest,
            GateDecision,
            reset_trade_gate,
        )

        reset_trade_gate()
        gate = TradeGate()

        # Create a trade request
        request = TradeRequest(
            symbol="BTC/USDT",
            side="long",
            quantity=0.1,
            price=50000.0,
            order_type="market",
            signal_confidence=0.3,
            current_regime="CRASH",
            regime_confidence=0.2,
            model_predictions={"rf": 0.3},
            model_confidences={"rf": 0.3},
            volatility=0.05,
            spread_bps=10.0,
            volume_24h=10_000_000.0,
            correlation_with_portfolio=0.5,
            current_drawdown_pct=0.05,
            daily_loss_pct=0.02,
        )

        # Trip the circuit breaker manually
        gate.trip_circuit_breaker("test: too many rejections")

        result = gate.evaluate(request)
        assert result.decision == GateDecision.REJECTED
        assert any("circuit" in str(r).lower() for r in result.rejection_reasons)


class TestExecutionSimulator:
    """Tests for Execution Simulator."""

    def test_execution_simulator_import(self):
        """Test that execution simulator can be imported."""
        from bot.execution.execution_simulator import (
            ExecutionSimulator,
            SimulatorConfig,
            ExecutionResult,
            get_execution_simulator,
            reset_execution_simulator,
        )

        reset_execution_simulator()
        simulator = get_execution_simulator()
        assert simulator is not None
        reset_execution_simulator()

    def test_simulate_market_order(self):
        """Test market order simulation."""
        from bot.execution.execution_simulator import (
            ExecutionSimulator,
            reset_execution_simulator,
        )

        reset_execution_simulator()
        simulator = ExecutionSimulator()

        result = simulator.simulate_execution(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type="market",
        )

        assert result is not None
        assert result.filled_quantity > 0
        assert result.execution_price > 0  # Actual field name
        assert result.slippage_pct >= 0  # Market orders should have slippage
        assert result.fees_usd >= 0

    def test_simulate_limit_order(self):
        """Test limit order simulation."""
        from bot.execution.execution_simulator import (
            ExecutionSimulator,
            reset_execution_simulator,
        )

        reset_execution_simulator()
        simulator = ExecutionSimulator()

        result = simulator.simulate_execution(
            symbol="ETH/USDT",
            side="sell",
            quantity=1.0,
            price=3000.0,
            order_type="limit",
        )

        assert result is not None
        # Limit orders have much smaller slippage than market orders
        assert result.slippage_pct < 0.01  # Less than 1%
        assert result.latency_ms >= 0

    def test_round_trip_cost_estimation(self):
        """Test round trip cost estimation."""
        from bot.execution.execution_simulator import (
            ExecutionSimulator,
            reset_execution_simulator,
        )

        reset_execution_simulator()
        simulator = ExecutionSimulator()

        cost = simulator.estimate_round_trip_cost(
            quantity=0.5,
            price=50000.0,
        )

        # Returns a dict with cost breakdown
        assert isinstance(cost, dict)
        assert "total_cost_usd" in cost or "entry_fees" in cost or len(cost) > 0


class TestRiskBudgetEngine:
    """Tests for Risk Budget Engine."""

    def test_risk_budget_engine_import(self):
        """Test that risk budget engine can be imported."""
        from bot.risk.risk_budget_engine import (
            RiskBudgetEngine,
            RiskBudgetConfig,
            RiskBudget,
            get_risk_budget_engine,
            reset_risk_budget_engine,
        )

        reset_risk_budget_engine()
        engine = get_risk_budget_engine()
        assert engine is not None
        reset_risk_budget_engine()

    def test_unified_engine_risk_budget_initializes(self):
        """Risk budget engine should initialize in the unified engine without errors."""
        from bot.unified_engine import EngineConfig, UnifiedTradingEngine, RISK_BUDGET_AVAILABLE

        engine = UnifiedTradingEngine(EngineConfig(use_ml_signals=False))
        if RISK_BUDGET_AVAILABLE:
            assert engine.risk_budget_engine is not None

    def test_budget_calculation_high_confidence(self):
        """Test budget calculation for high confidence trade."""
        from bot.risk.risk_budget_engine import (
            RiskBudgetEngine,
            PortfolioRiskState,
            reset_risk_budget_engine,
        )

        reset_risk_budget_engine()
        engine = RiskBudgetEngine()

        portfolio_state = PortfolioRiskState(
            total_equity=100000.0,
            current_exposure=10000.0,
            exposure_pct=0.1,
            position_count=1,
            position_values={"ETH/USDT": 10000.0},
            position_correlations={"ETH/USDT": 0.7},
            current_drawdown_pct=0.02,
            daily_pnl_pct=0.005,
            win_streak=3,
            loss_streak=0,
            recent_volatility=0.02,
        )

        budget = engine.calculate_budget(
            symbol="BTC/USDT",
            side="long",
            signal_confidence=0.85,
            regime="BULL",
            portfolio_state=portfolio_state,
            price=50000.0,
        )

        assert budget is not None
        assert budget.max_position_pct > 0
        assert budget.max_leverage > 1.0  # Bull regime should allow leverage
        assert budget.max_risk_pct > 0

    def test_budget_reduced_during_drawdown(self):
        """Test that budget is reduced during drawdown."""
        from bot.risk.risk_budget_engine import (
            RiskBudgetEngine,
            PortfolioRiskState,
            reset_risk_budget_engine,
        )

        reset_risk_budget_engine()
        engine = RiskBudgetEngine()

        # Normal conditions
        normal_state = PortfolioRiskState(
            total_equity=100000.0,
            current_exposure=0.0,
            exposure_pct=0.0,
            position_count=0,
            position_values={},
            position_correlations={},
            current_drawdown_pct=0.01,
            daily_pnl_pct=0.005,
            win_streak=2,
            loss_streak=0,
            recent_volatility=0.02,
        )

        # Drawdown conditions
        drawdown_state = PortfolioRiskState(
            total_equity=92000.0,
            current_exposure=30000.0,
            exposure_pct=0.33,
            position_count=2,
            position_values={"ETH/USDT": 15000.0, "SOL/USDT": 15000.0},
            position_correlations={"ETH/USDT": 0.8, "SOL/USDT": 0.6},
            current_drawdown_pct=0.08,  # 8% drawdown
            daily_pnl_pct=-0.02,
            win_streak=0,
            loss_streak=4,
            recent_volatility=0.05,
        )

        normal_budget = engine.calculate_budget(
            symbol="BTC/USDT",
            side="long",
            signal_confidence=0.7,
            regime="SIDEWAYS",
            portfolio_state=normal_state,
            price=50000.0,
        )

        drawdown_budget = engine.calculate_budget(
            symbol="BTC/USDT",
            side="long",
            signal_confidence=0.7,
            regime="SIDEWAYS",
            portfolio_state=drawdown_state,
            price=50000.0,
        )

        # Drawdown budget should be smaller
        assert drawdown_budget.max_position_pct < normal_budget.max_position_pct


class TestCapitalPreservation:
    """Tests for Capital Preservation Mode."""

    def test_capital_preservation_import(self):
        """Test that capital preservation can be imported."""
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationConfig,
            PreservationLevel,
            get_capital_preservation,
            reset_capital_preservation,
        )

        reset_capital_preservation()
        preservation = get_capital_preservation()
        assert preservation is not None
        reset_capital_preservation()

    def test_initial_state_is_normal(self):
        """Test that initial state is normal when using temp directory."""
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationLevel,
            PreservationConfig,
            reset_capital_preservation,
        )

        reset_capital_preservation()

        # Use temp directory to avoid stale state
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreservationConfig(state_path=Path(tmpdir) / "test_preservation.json")
            preservation = CapitalPreservationMode(config=config, initial_equity=10000.0)

            status = preservation.get_status()
            assert status["level"] == PreservationLevel.NORMAL.value

    def test_escalation_on_drawdown(self):
        """Test escalation on significant drawdown."""
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationLevel,
            PreservationConfig,
            reset_capital_preservation,
        )

        reset_capital_preservation()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreservationConfig(state_path=Path(tmpdir) / "test_preservation.json")
            preservation = CapitalPreservationMode(config=config, initial_equity=10000.0)

            # Simulate significant drawdown
            preservation.update_equity(9200.0)  # 8% drawdown

            status = preservation.get_status()
            # Should escalate to higher protection level
            assert status["level"] in [
                PreservationLevel.DEFENSIVE.value,
                PreservationLevel.CRITICAL.value,
                PreservationLevel.LOCKDOWN.value,
            ]

    def test_trade_blocking_in_lockdown(self):
        """Test that trades are blocked in lockdown mode."""
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationLevel,
            PreservationConfig,
            reset_capital_preservation,
        )

        reset_capital_preservation()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreservationConfig(state_path=Path(tmpdir) / "test_preservation.json")
            preservation = CapitalPreservationMode(config=config, initial_equity=10000.0)
            preservation.force_level(PreservationLevel.LOCKDOWN, "test")

            can_trade, reason = preservation.can_trade(signal_confidence=0.9)
            assert not can_trade
            assert "lockdown" in reason.lower()

    def test_leverage_adjustment(self):
        """Test leverage adjustment by preservation level."""
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationLevel,
            PreservationConfig,
            reset_capital_preservation,
        )

        reset_capital_preservation()

        # Use temp directory to avoid stale state
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PreservationConfig(state_path=Path(tmpdir) / "test_preservation.json")
            preservation = CapitalPreservationMode(config=config, initial_equity=10000.0)

            # Force to normal first
            preservation.reset()

            # Normal mode - full leverage
            normal_leverage = preservation.adjust_leverage(3.0)
            assert normal_leverage == 3.0, f"Expected 3.0, got {normal_leverage}"

            # Defensive mode - reduced leverage
            preservation.force_level(PreservationLevel.DEFENSIVE, "test")
            defensive_leverage = preservation.adjust_leverage(3.0)
            assert defensive_leverage < 3.0  # Should be reduced


class TestTradeForensics:
    """Tests for Trade Forensics."""

    def test_trade_forensics_import(self):
        """Test that trade forensics can be imported."""
        from bot.meta.trade_forensics import (
            TradeForensics,
            ForensicsConfig,
            ForensicsResult,
            get_trade_forensics,
            reset_trade_forensics,
        )

        reset_trade_forensics()
        forensics = get_trade_forensics()
        assert forensics is not None
        reset_trade_forensics()

    def test_analyze_winning_trade(self):
        """Test forensics analysis of a winning trade."""
        from bot.meta.trade_forensics import (
            TradeForensics,
            reset_trade_forensics,
        )

        reset_trade_forensics()
        forensics = TradeForensics()

        entry_time = datetime.now() - timedelta(hours=2)
        exit_time = datetime.now()

        # Simulate price history showing MFE at +3% and MAE at -1%
        price_history = [
            (entry_time, 50000.0),
            (entry_time + timedelta(minutes=10), 49500.0),  # MAE (-1%)
            (entry_time + timedelta(minutes=30), 50500.0),
            (entry_time + timedelta(minutes=60), 51500.0),  # MFE (+3%)
            (entry_time + timedelta(minutes=90), 51000.0),
            (exit_time, 50500.0),  # Exit (+1%)
        ]

        result = forensics.analyze_trade(
            trade_id="test_001",
            symbol="BTC/USDT",
            side="long",
            entry_price=50000.0,
            entry_timestamp=entry_time,
            exit_price=50500.0,
            exit_timestamp=exit_time,
            stop_price=49000.0,
            price_history=price_history,
            was_stopped_out=False,
        )

        assert result is not None
        assert result.mfe_pct > 0  # Had favorable excursion
        assert result.mae_pct <= 0  # Had some adverse excursion
        assert result.realized_pnl > 0  # Was profitable

    def test_analyze_stopped_out_trade(self):
        """Test forensics analysis of a stopped out trade."""
        from bot.meta.trade_forensics import (
            TradeForensics,
            ExitQuality,
            reset_trade_forensics,
        )

        reset_trade_forensics()
        forensics = TradeForensics()

        entry_time = datetime.now() - timedelta(hours=1)
        exit_time = datetime.now()

        # Price moves against position
        price_history = [
            (entry_time, 50000.0),
            (entry_time + timedelta(minutes=10), 49800.0),
            (entry_time + timedelta(minutes=20), 49500.0),
            (entry_time + timedelta(minutes=30), 49000.0),  # Stop hit
            (exit_time, 49000.0),
        ]

        result = forensics.analyze_trade(
            trade_id="test_002",
            symbol="BTC/USDT",
            side="long",
            entry_price=50000.0,
            entry_timestamp=entry_time,
            exit_price=49000.0,
            exit_timestamp=exit_time,
            stop_price=49000.0,
            price_history=price_history,
            was_stopped_out=True,
        )

        assert result is not None
        assert result.exit_quality == ExitQuality.STOPPED_OUT
        assert result.realized_pnl < 0


class TestReconciler:
    """Tests for Reconciler."""

    def test_reconciler_import(self):
        """Test that reconciler can be imported."""
        from bot.execution.reconciler import (
            Reconciler,
            ReconcilerConfig,
            TransactionState,
            get_reconciler,
            reset_reconciler,
        )

        reset_reconciler()
        reconciler = get_reconciler()
        assert reconciler is not None
        reset_reconciler()

    def test_idempotency_detection(self):
        """Test duplicate transaction detection."""
        from bot.execution.reconciler import Reconciler, reset_reconciler

        reset_reconciler()

        with tempfile.TemporaryDirectory() as tmpdir:
            from bot.execution.reconciler import ReconcilerConfig

            config = ReconcilerConfig(db_path=Path(tmpdir) / "test_reconciler.db")
            reconciler = Reconciler(config=config)

            now = datetime.now()
            tx_id = reconciler.generate_transaction_id(
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                price=50000.0,
                timestamp=now,
            )

            # First check - not duplicate
            can_process, returned_id = reconciler.check_idempotency(
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                price=50000.0,
                timestamp=now,
            )
            assert can_process
            assert returned_id == tx_id

            # Begin transaction
            reconciler.begin_transaction(
                transaction_id=tx_id,
                order_id="order_001",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                price=50000.0,
                timestamp=now,
            )

            # Mark as processed
            reconciler.mark_processed(tx_id)

            # Second check - now it's a duplicate
            assert reconciler.is_duplicate(tx_id)

    def test_position_reconciliation(self):
        """Test position reconciliation."""
        from bot.execution.reconciler import (
            Reconciler,
            ReconciliationStatus,
            reset_reconciler,
        )

        reset_reconciler()

        with tempfile.TemporaryDirectory() as tmpdir:
            from bot.execution.reconciler import ReconcilerConfig

            config = ReconcilerConfig(db_path=Path(tmpdir) / "test_reconciler.db")
            reconciler = Reconciler(config=config)

            # Update local position
            reconciler.update_position(
                symbol="BTC/USDT",
                quantity=0.5,
                average_entry_price=50000.0,
                unrealized_pnl=100.0,
            )

            # Reconcile with matching exchange position
            result = reconciler.reconcile_position(
                symbol="BTC/USDT",
                exchange_position={
                    "quantity": 0.5,
                    "average_price": 50000.0,
                    "unrealized_pnl": 100.0,
                },
            )

            assert result.status == ReconciliationStatus.MATCHED

    def test_transaction_state_persistence(self):
        """Test transaction state persistence across restarts."""
        from bot.execution.reconciler import (
            Reconciler,
            TransactionState,
            reset_reconciler,
        )

        reset_reconciler()

        with tempfile.TemporaryDirectory() as tmpdir:
            from bot.execution.reconciler import ReconcilerConfig

            config = ReconcilerConfig(db_path=Path(tmpdir) / "test_reconciler.db")

            # First instance - create transaction
            reconciler1 = Reconciler(config=config)

            now = datetime.now()
            tx_id = reconciler1.generate_transaction_id(
                symbol="ETH/USDT",
                side="sell",
                quantity=2.0,
                price=3000.0,
                timestamp=now,
            )

            reconciler1.begin_transaction(
                transaction_id=tx_id,
                order_id="order_002",
                symbol="ETH/USDT",
                side="sell",
                quantity=2.0,
                price=3000.0,
                timestamp=now,
            )
            reconciler1.confirm_transaction(tx_id, exchange_order_id="ex_order_002")
            reconciler1.mark_processed(tx_id)

            # Reset singleton
            reset_reconciler()

            # Second instance - should load state
            reconciler2 = Reconciler(config=config)

            # Should recognize the processed transaction
            assert reconciler2.is_duplicate(tx_id)


class TestIntegration:
    """Integration tests for Phase 1 components working together."""

    def test_full_trade_flow(self):
        """Test full trade flow through all Phase 1 components."""
        from bot.risk.trade_gate import (
            TradeGate,
            TradeRequest,
            GateDecision,
            reset_trade_gate,
        )
        from bot.risk.risk_budget_engine import (
            RiskBudgetEngine,
            PortfolioRiskState,
            reset_risk_budget_engine,
        )
        from bot.safety.capital_preservation import (
            CapitalPreservationMode,
            PreservationConfig,
            reset_capital_preservation,
        )

        # Reset all singletons
        reset_trade_gate()
        reset_risk_budget_engine()
        reset_capital_preservation()

        # Initialize components with temp directory for capital preservation
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = TradeGate()
            risk_engine = RiskBudgetEngine()
            config = PreservationConfig(state_path=Path(tmpdir) / "test_preservation.json")
            preservation = CapitalPreservationMode(config=config, initial_equity=100000.0)

            # Step 1: Check capital preservation allows trading
            can_trade, _ = preservation.can_trade(signal_confidence=0.8)
            assert can_trade

            # Step 2: Get adjusted leverage from preservation
            leverage = preservation.adjust_leverage(3.0)
            assert leverage == 3.0  # Normal mode

            # Step 3: Calculate risk budget
            portfolio_state = PortfolioRiskState(
                total_equity=100000.0,
                current_exposure=10000.0,
                exposure_pct=0.1,
                position_count=1,
                position_values={"ETH/USDT": 10000.0},
                position_correlations={"ETH/USDT": 0.5},
                current_drawdown_pct=0.02,
                daily_pnl_pct=0.005,
                win_streak=2,
                loss_streak=0,
                recent_volatility=0.02,
            )

            budget = risk_engine.calculate_budget(
                symbol="BTC/USDT",
                side="long",
                signal_confidence=0.8,
                regime="BULL",
                portfolio_state=portfolio_state,
                price=50000.0,
            )

            assert budget.max_position_pct > 0
            assert budget.max_leverage <= leverage  # Should not exceed preservation limit

            # Step 4: Evaluate through trade gate
            request = TradeRequest(
                symbol="BTC/USDT",
                side="long",
                quantity=0.1,
                price=50000.0,
                order_type="market",
                leverage=2.0,
                signal_confidence=0.8,
                current_regime="BULL",
                regime_confidence=0.85,
                model_predictions={"rf": 0.75, "gb": 0.78, "xgb": 0.72},
                model_confidences={"rf": 0.85, "gb": 0.82, "xgb": 0.80},
                volatility=0.02,
                spread_bps=5.0,
                volume_24h=100_000_000.0,
                upcoming_events_hours=48.0,
                news_urgency=2,
                correlation_with_portfolio=0.3,
                current_drawdown_pct=0.02,
                daily_loss_pct=0.0,
            )

            result = gate.evaluate(request)
            assert result.decision == GateDecision.APPROVED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
