"""
Feedback Orchestrator.

Central coordinator for all learning systems after EVERY trade:
1. Updates ML models (online learning)
2. Stores patterns (pattern memory)
3. Updates RL replay buffer
4. Updates regime statistics
5. Checks for drift → trigger retrain

Ensures atomic, coordinated updates across all learning subsystems.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from bot.learning.learning_database import LearningDatabase, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for feedback orchestrator."""

    # Update frequencies
    online_learning_frequency: int = 50  # Update ML models every N trades
    pattern_update_frequency: int = 1  # Store pattern every N trades
    rl_buffer_update_frequency: int = 1  # Update RL buffer every N trades
    regime_stats_update_frequency: int = 1  # Update regime stats every N trades

    # Drift detection
    drift_check_frequency: int = 20  # Check for drift every N trades
    drift_threshold: float = 0.20  # 20% deviation triggers retrain

    # Performance thresholds
    min_win_rate_for_confidence_boost: float = 0.55
    max_consecutive_losses_before_action: int = 5

    # Timeouts
    feedback_timeout_seconds: float = 5.0  # Max time for feedback loop

    # Callbacks
    on_retrain_triggered: Optional[Callable] = None
    on_drift_detected: Optional[Callable[[str, float], None]] = None
    on_streak_alert: Optional[Callable[[str, int], None]] = None


@dataclass
class TradeContext:
    """Full context for a completed trade with Phase 1 production fields."""

    # Trade identifiers
    symbol: str
    side: str  # LONG or SHORT
    trade_id: Optional[str] = None
    transaction_id: str = ""  # Idempotency key from reconciler

    # Timing
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: datetime = field(default_factory=datetime.now)

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Position
    quantity: float = 0.0
    leverage: float = 1.0

    # Performance
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_unrealized_profit_pct: float = 0.0
    max_unrealized_loss_pct: float = 0.0

    # Regime
    regime_at_entry: str = "unknown"
    regime_at_exit: str = "unknown"
    regime_confidence: float = 0.0
    volatility: float = 0.0
    trend: str = "neutral"

    # Signal
    signal_source: str = ""
    signal_confidence: float = 0.0
    signal_reason: str = ""

    # News/Sentiment
    news_sentiment: float = 0.0
    fear_greed: float = 50.0

    # Technical indicators at entry
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_position: float = 0.5
    volume_ratio: float = 1.0
    atr: float = 0.0

    # Feature vector for ML
    feature_vector: Optional[np.ndarray] = None

    # Exit reason
    exit_reason: str = ""  # TP, SL, trailing, signal_flip, manual

    # === Phase 1 Production Fields ===

    # Trade Gate context
    gate_score: float = 0.0
    gate_decision: str = ""
    gate_rejection_reason: str = ""

    # Execution Quality
    expected_entry_price: float = 0.0
    expected_exit_price: float = 0.0
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0
    total_slippage_pct: float = 0.0
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    total_fees: float = 0.0
    execution_latency_ms: float = 0.0
    partial_fill_pct: float = 1.0
    was_stopped_out: bool = False

    # Risk Budget
    risk_budget_pct: float = 0.0
    risk_budget_usd: float = 0.0
    max_leverage_allowed: float = 1.0
    kelly_fraction: float = 0.0
    var_at_entry: float = 0.0
    cvar_at_entry: float = 0.0

    # Capital Preservation
    preservation_level: str = "normal"
    leverage_multiplier_applied: float = 1.0
    confidence_threshold_at_entry: float = 0.5

    # Portfolio Context
    portfolio_equity_at_entry: float = 0.0
    portfolio_drawdown_at_entry: float = 0.0
    open_positions_at_entry: int = 0
    correlation_with_portfolio: float = 0.0

    # Price history for forensics (timestamp, price tuples)
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)


class FeedbackOrchestrator:
    """
    Coordinates all learning systems after each trade.

    Phase 1 Production Responsibilities:
    1. Record trade to learning database (with full context)
    2. Update pattern memory
    3. Update online learning buffer
    4. Update RL replay buffer
    5. Check for drift and trigger retraining
    6. Track streaks and alert on concerning patterns
    7. Run post-trade forensics (MAE/MFE analysis)
    8. Update capital preservation state
    9. Mark transaction as processed (idempotency)

    Atomic processing with <100ms latency target.
    """

    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        learning_db: Optional[LearningDatabase] = None,
        pattern_memory: Optional[Any] = None,
        online_learner: Optional[Any] = None,
        rl_buffer: Optional[Any] = None,
        trade_forensics: Optional[Any] = None,
        capital_preservation: Optional[Any] = None,
        reconciler: Optional[Any] = None,
    ):
        """
        Initialize the feedback orchestrator.

        Args:
            config: Configuration settings
            learning_db: Learning database instance
            pattern_memory: Pattern memory instance
            online_learner: Online learning manager instance
            rl_buffer: RL experience buffer instance
            trade_forensics: TradeForensics instance for MAE/MFE analysis
            capital_preservation: CapitalPreservationMode instance
            reconciler: Reconciler instance for idempotency
        """
        self.config = config or FeedbackConfig()
        self.learning_db = learning_db or LearningDatabase()

        # Optional components (can be set later)
        self.pattern_memory = pattern_memory
        self.online_learner = online_learner
        self.rl_buffer = rl_buffer

        # Phase 1 production components
        self.trade_forensics = trade_forensics
        self.capital_preservation = capital_preservation
        self.reconciler = reconciler

        # State tracking
        self._trade_count = 0
        self._win_streak = 0
        self._loss_streak = 0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_daily_reset = datetime.now().date()

        # Baseline metrics for drift detection
        self._baseline_win_rate = 0.5
        self._baseline_avg_pnl = 0.0
        self._baseline_set = False

        # Lock for thread safety
        self._lock = threading.RLock()

        # Feedback latency tracking
        self._feedback_latencies: List[float] = []

        # Latency threshold warning (100ms target)
        self._latency_warning_threshold_ms = 100.0

        logger.info("Feedback Orchestrator initialized (Phase 1 Production)")

    def set_pattern_memory(self, pattern_memory: Any):
        """Set pattern memory instance after initialization."""
        self.pattern_memory = pattern_memory

    def set_online_learner(self, online_learner: Any):
        """Set online learner instance after initialization."""
        self.online_learner = online_learner

    def set_rl_buffer(self, rl_buffer: Any):
        """Set RL buffer instance after initialization."""
        self.rl_buffer = rl_buffer

    def set_trade_forensics(self, trade_forensics: Any):
        """Set trade forensics instance after initialization."""
        self.trade_forensics = trade_forensics

    def set_capital_preservation(self, capital_preservation: Any):
        """Set capital preservation instance after initialization."""
        self.capital_preservation = capital_preservation

    def set_reconciler(self, reconciler: Any):
        """Set reconciler instance after initialization."""
        self.reconciler = reconciler

    async def process_trade_feedback(self, context: TradeContext) -> Dict[str, Any]:
        """
        Process feedback for a completed trade - ATOMIC PIPELINE.

        This is the main entry point called after every trade closes.
        Target latency: <100ms for all operations.

        Pipeline:
        1. Idempotency check (skip if already processed)
        2. Run forensics analysis (MAE/MFE)
        3. Record to learning database (with forensics)
        4. Update pattern memory
        5. Update online learning buffer
        6. Update RL replay buffer
        7. Update capital preservation state
        8. Check for drift
        9. Check streak alerts
        10. Mark transaction as processed

        Args:
            context: Full trade context

        Returns:
            Dictionary with feedback processing results
        """
        start_time = datetime.now()
        results = {
            "trade_id": None,
            "transaction_id": context.transaction_id,
            "updates_completed": [],
            "drift_detected": False,
            "retrain_triggered": False,
            "alerts": [],
            "latency_ms": 0,
            "forensics": None,
            "is_duplicate": False,
        }

        try:
            with self._lock:
                # 0. Idempotency check - skip if already processed
                if self.reconciler and context.transaction_id:
                    if self.reconciler.is_duplicate(context.transaction_id):
                        logger.info(f"Skipping duplicate transaction: {context.transaction_id}")
                        results["is_duplicate"] = True
                        results["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
                        return results

                # Reset daily counters if new day
                self._check_daily_reset()

                # Update trade count
                self._trade_count += 1
                self._daily_trades += 1
                self._daily_pnl += context.pnl

                # Update streaks
                was_profitable = context.pnl > 0
                self._update_streaks(was_profitable)

                # 1. Run forensics analysis (MAE/MFE)
                forensics_result = None
                if self.trade_forensics and context.price_history:
                    forensics_result = await self._run_forensics(context)
                    results["forensics"] = forensics_result
                    results["updates_completed"].append("forensics")

                # 2. Record to learning database (with forensics data)
                trade_id = await self._record_to_database(context, forensics_result)
                results["trade_id"] = trade_id
                results["updates_completed"].append("learning_db")

                # 3. Update pattern memory
                if self.pattern_memory and self._trade_count % self.config.pattern_update_frequency == 0:
                    await self._update_pattern_memory(context)
                    results["updates_completed"].append("pattern_memory")

                # 4. Update online learning buffer
                if self.online_learner and self._trade_count % self.config.rl_buffer_update_frequency == 0:
                    update_triggered = await self._update_online_learner(context)
                    results["updates_completed"].append("online_learner")
                    if update_triggered:
                        results["retrain_triggered"] = True

                # 5. Update RL replay buffer
                if self.rl_buffer and self._trade_count % self.config.rl_buffer_update_frequency == 0:
                    await self._update_rl_buffer(context)
                    results["updates_completed"].append("rl_buffer")

                # 6. Update capital preservation state
                if self.capital_preservation:
                    await self._update_capital_preservation(context)
                    results["updates_completed"].append("capital_preservation")

                # 7. Check for drift
                if self._trade_count % self.config.drift_check_frequency == 0:
                    drift_results = await self._check_drift()
                    if drift_results.get("drift_detected"):
                        results["drift_detected"] = True
                        results["alerts"].append(f"Drift detected: {drift_results.get('metric')}")

                        if self.config.on_drift_detected:
                            self.config.on_drift_detected(
                                drift_results.get("metric", ""),
                                drift_results.get("drift_score", 0.0)
                            )

                # 8. Check for streak alerts
                streak_alerts = self._check_streak_alerts()
                if streak_alerts:
                    results["alerts"].extend(streak_alerts)

                # 9. Mark transaction as processed (idempotency)
                if self.reconciler and context.transaction_id:
                    self.reconciler.mark_processed(context.transaction_id)
                    results["updates_completed"].append("reconciler")

        except asyncio.TimeoutError:
            logger.warning(f"Feedback timeout for trade {context.symbol}")
            results["alerts"].append("Feedback timeout")

        except Exception as e:
            logger.error(f"Feedback processing error: {e}", exc_info=True)
            results["alerts"].append(f"Error: {str(e)}")

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        results["latency_ms"] = latency_ms
        self._feedback_latencies.append(latency_ms)

        # Keep only last 100 latencies
        if len(self._feedback_latencies) > 100:
            self._feedback_latencies = self._feedback_latencies[-100:]

        # Warn if latency exceeds target
        if latency_ms > self._latency_warning_threshold_ms:
            logger.warning(
                f"Feedback latency {latency_ms:.1f}ms exceeds target {self._latency_warning_threshold_ms}ms"
            )

        logger.debug(
            f"Feedback processed for {context.symbol} in {latency_ms:.1f}ms: "
            f"{len(results['updates_completed'])} updates"
        )

        return results

    async def _run_forensics(self, context: TradeContext) -> Optional[Dict[str, Any]]:
        """Run post-trade forensics analysis."""
        if not self.trade_forensics:
            return None

        try:
            result = self.trade_forensics.analyze_trade(
                trade_id=context.trade_id or context.transaction_id or str(self._trade_count),
                symbol=context.symbol,
                side=context.side.lower(),
                entry_price=context.entry_price,
                entry_timestamp=context.entry_time,
                exit_price=context.exit_price,
                exit_timestamp=context.exit_time,
                stop_price=context.stop_loss if context.stop_loss > 0 else None,
                price_history=context.price_history,
                was_stopped_out=context.was_stopped_out,
            )

            return {
                "mae_pct": result.mae_pct,
                "mfe_pct": result.mfe_pct,
                "mae_time_seconds": int(result.time_to_mae_minutes * 60),
                "mfe_time_seconds": int(result.time_to_mfe_minutes * 60),
                "capture_ratio": result.capture_ratio,
                "pain_ratio": result.pain_ratio,
                "entry_quality_score": result.entry_quality_score,
                "exit_quality_score": result.exit_quality_score,
                "stop_quality_score": result.stop_quality_score,
                "improvements": result.improvements,
            }

        except Exception as e:
            logger.warning(f"Forensics analysis failed: {e}")
            return None

    async def _update_capital_preservation(self, context: TradeContext):
        """Update capital preservation state after trade."""
        if not self.capital_preservation:
            return

        try:
            # Record trade metrics
            self.capital_preservation.record_trade(
                pnl=context.pnl,
                expected_price=context.expected_entry_price or context.entry_price,
                actual_price=context.entry_price,
                regime_confidence=context.regime_confidence,
                signal_confidence=context.signal_confidence,
            )

        except Exception as e:
            logger.warning(f"Capital preservation update failed: {e}")

    async def _record_to_database(
        self, context: TradeContext, forensics_result: Optional[Dict[str, Any]] = None
    ) -> int:
        """Record trade to learning database with full Phase 1 context."""
        # Get streak context
        win_streak, loss_streak, daily_pnl = self.learning_db.get_streak_context()

        # Calculate hold duration
        hold_duration = 0
        if context.exit_time and context.entry_time:
            hold_duration = int((context.exit_time - context.entry_time).total_seconds())

        record = TradeRecord(
            # Core fields
            transaction_id=context.transaction_id,
            symbol=context.symbol,
            side=context.side,
            entry_time=context.entry_time,
            exit_time=context.exit_time,
            hold_duration_seconds=hold_duration,
            entry_price=context.entry_price,
            exit_price=context.exit_price,
            stop_loss=context.stop_loss,
            take_profit=context.take_profit,
            quantity=context.quantity,
            leverage=context.leverage,
            position_value=context.quantity * context.entry_price,
            pnl=context.pnl,
            pnl_pct=context.pnl_pct,
            max_profit_pct=context.max_unrealized_profit_pct,
            max_drawdown_pct=context.max_unrealized_loss_pct,
            # Regime context
            regime_at_entry=context.regime_at_entry,
            regime_at_exit=context.regime_at_exit,
            regime_confidence_at_entry=context.regime_confidence,
            volatility_at_entry=context.volatility,
            trend_at_entry=context.trend,
            # Signal context
            signal_source=context.signal_source,
            signal_confidence=context.signal_confidence,
            signal_reason=context.signal_reason,
            # News/Sentiment
            news_sentiment_score=context.news_sentiment,
            fear_greed_index=context.fear_greed,
            # Technical indicators
            rsi=context.rsi,
            macd=context.macd,
            macd_signal=context.macd_signal,
            bb_position=context.bb_position,
            volume_ratio=context.volume_ratio,
            atr=context.atr,
            # Feature vector
            feature_vector=context.feature_vector,
            # Streak context
            win_streak_at_entry=self._win_streak,
            loss_streak_at_entry=self._loss_streak,
            daily_pnl_at_entry=self._daily_pnl - context.pnl,  # PnL before this trade
            exit_reason=context.exit_reason,
            was_profitable=context.pnl > 0,
            # === Phase 1 Production Fields ===
            # Trade Gate
            gate_score=context.gate_score,
            gate_decision=context.gate_decision,
            gate_rejection_reason=context.gate_rejection_reason,
            # Execution Quality
            expected_entry_price=context.expected_entry_price,
            expected_exit_price=context.expected_exit_price,
            entry_slippage_pct=context.entry_slippage_pct,
            exit_slippage_pct=context.exit_slippage_pct,
            total_slippage_pct=context.total_slippage_pct,
            entry_fees=context.entry_fees,
            exit_fees=context.exit_fees,
            total_fees=context.total_fees,
            execution_latency_ms=context.execution_latency_ms,
            partial_fill_pct=context.partial_fill_pct,
            # Risk Budget
            risk_budget_pct=context.risk_budget_pct,
            risk_budget_usd=context.risk_budget_usd,
            max_leverage_allowed=context.max_leverage_allowed,
            kelly_fraction=context.kelly_fraction,
            var_at_entry=context.var_at_entry,
            cvar_at_entry=context.cvar_at_entry,
            # Capital Preservation
            preservation_level=context.preservation_level,
            leverage_multiplier_applied=context.leverage_multiplier_applied,
            confidence_threshold_at_entry=context.confidence_threshold_at_entry,
            # Portfolio Context
            portfolio_equity_at_entry=context.portfolio_equity_at_entry,
            portfolio_drawdown_at_entry=context.portfolio_drawdown_at_entry,
            open_positions_at_entry=context.open_positions_at_entry,
            correlation_with_portfolio=context.correlation_with_portfolio,
        )

        # Add forensics data if available
        if forensics_result:
            record.mae_pct = forensics_result.get("mae_pct", 0.0)
            record.mfe_pct = forensics_result.get("mfe_pct", 0.0)
            record.mae_time_seconds = forensics_result.get("mae_time_seconds", 0)
            record.mfe_time_seconds = forensics_result.get("mfe_time_seconds", 0)
            record.capture_ratio = forensics_result.get("capture_ratio", 0.0)
            record.pain_ratio = forensics_result.get("pain_ratio", 0.0)
            record.entry_quality_score = forensics_result.get("entry_quality_score", 0.0)
            record.exit_quality_score = forensics_result.get("exit_quality_score", 0.0)
            record.stop_quality_score = forensics_result.get("stop_quality_score", 0.0)

        trade_id = self.learning_db.record_trade(record)
        return trade_id

    async def _update_pattern_memory(self, context: TradeContext):
        """Update pattern memory with trade outcome."""
        if not self.pattern_memory:
            return

        try:
            # Import here to avoid circular dependency
            from bot.intelligence.pattern_memory import TradingPattern

            # Calculate hold duration in minutes
            hold_minutes = 0
            if context.exit_time and context.entry_time:
                hold_minutes = int((context.exit_time - context.entry_time).total_seconds() / 60)

            pattern = TradingPattern(
                symbol=context.symbol,
                regime=context.regime_at_entry,
                action=context.side,
                entry_price=context.entry_price,
                exit_price=context.exit_price,
                pnl_pct=context.pnl_pct * 100,  # Convert to percentage
                hold_duration_minutes=hold_minutes,
                confidence_at_entry=context.signal_confidence,
                rsi=context.rsi,
                macd=context.macd,
                volatility=context.volatility,
                trend_strength=0.0,  # Could be calculated from trend indicator
                was_profitable=context.pnl > 0,
                max_drawdown_pct=context.max_unrealized_loss_pct * 100,
                max_profit_pct=context.max_unrealized_profit_pct * 100,
            )

            self.pattern_memory.store_pattern(pattern)

        except Exception as e:
            logger.warning(f"Failed to update pattern memory: {e}")

    async def _update_online_learner(self, context: TradeContext) -> bool:
        """Update online learning manager and check if model update triggered."""
        if not self.online_learner:
            return False

        try:
            # Prepare features
            features = context.feature_vector
            if features is None:
                # Create basic feature vector if not provided
                features = np.array([
                    context.rsi / 100,
                    context.macd,
                    context.bb_position,
                    context.volume_ratio,
                    context.volatility,
                    context.signal_confidence,
                ], dtype=np.float32)

            # Calculate hold time in minutes
            hold_minutes = 0
            if context.exit_time and context.entry_time:
                hold_minutes = int((context.exit_time - context.entry_time).total_seconds() / 60)

            # Record trade outcome
            update_triggered = self.online_learner.record_trade_outcome(
                symbol=context.symbol,
                features=features,
                action=context.side,
                pnl=context.pnl,
                pnl_pct=context.pnl_pct,
                entry_price=context.entry_price,
                exit_price=context.exit_price,
                hold_time_minutes=hold_minutes,
                confidence=context.signal_confidence,
                regime=context.regime_at_entry,
            )

            return update_triggered

        except Exception as e:
            logger.warning(f"Failed to update online learner: {e}")
            return False

    async def _update_rl_buffer(self, context: TradeContext):
        """Update RL experience replay buffer."""
        if not self.rl_buffer:
            return

        try:
            # Prepare experience tuple
            state = context.feature_vector
            if state is None:
                return

            # Action: 0=SHORT, 1=FLAT, 2=LONG
            action = {"SHORT": 0, "FLAT": 1, "LONG": 2}.get(context.side, 1)

            # Reward based on PnL with shaping
            reward = context.pnl_pct * 100  # Base reward
            if context.pnl > 0:
                reward += 1.0  # Win bonus
            else:
                reward -= 0.5  # Loss penalty

            # Add to buffer
            if hasattr(self.rl_buffer, "add"):
                self.rl_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state,  # Simplified - could track actual next state
                    done=True,
                )
            elif hasattr(self.rl_buffer, "push"):
                self.rl_buffer.push(state, action, reward, state, True)

        except Exception as e:
            logger.warning(f"Failed to update RL buffer: {e}")

    async def _check_drift(self) -> Dict[str, Any]:
        """Check for model/performance drift."""
        result = {"drift_detected": False, "metric": None, "drift_score": 0.0}

        try:
            # Get recent metrics
            metrics = self.learning_db.get_learning_metrics(days_lookback=7)

            if metrics.total_trades < 20:
                return result

            # Set baseline if not set
            if not self._baseline_set:
                self._baseline_win_rate = metrics.win_rate
                self._baseline_avg_pnl = metrics.avg_pnl_pct
                self._baseline_set = True
                return result

            # Check win rate drift
            win_rate_drift = self.learning_db.record_drift_metric(
                metric_name="win_rate",
                current_value=metrics.win_rate,
                baseline_value=self._baseline_win_rate,
                window_size=metrics.total_trades,
            )

            if win_rate_drift:
                result["drift_detected"] = True
                result["metric"] = "win_rate"
                result["drift_score"] = abs(metrics.win_rate - self._baseline_win_rate)

                # Trigger retrain callback
                if self.config.on_retrain_triggered:
                    self.config.on_retrain_triggered()

            # Check PnL drift
            pnl_drift = self.learning_db.record_drift_metric(
                metric_name="avg_pnl",
                current_value=metrics.avg_pnl_pct,
                baseline_value=self._baseline_avg_pnl,
                window_size=metrics.total_trades,
            )

            if pnl_drift and not result["drift_detected"]:
                result["drift_detected"] = True
                result["metric"] = "avg_pnl"
                result["drift_score"] = abs(metrics.avg_pnl_pct - self._baseline_avg_pnl)

        except Exception as e:
            logger.warning(f"Drift check failed: {e}")

        return result

    def _update_streaks(self, was_profitable: bool):
        """Update win/loss streak tracking."""
        if was_profitable:
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0

    def _check_streak_alerts(self) -> List[str]:
        """Check for concerning streak patterns."""
        alerts = []

        # Check for consecutive losses
        if self._loss_streak >= self.config.max_consecutive_losses_before_action:
            alerts.append(f"ALERT: {self._loss_streak} consecutive losses")
            if self.config.on_streak_alert:
                self.config.on_streak_alert("loss", self._loss_streak)

        # Check for hot streak (could be overconfidence)
        if self._win_streak >= 10:
            alerts.append(f"INFO: {self._win_streak} consecutive wins - consider reducing risk")
            if self.config.on_streak_alert:
                self.config.on_streak_alert("win", self._win_streak)

        return alerts

    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if today > self._last_daily_reset:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_daily_reset = today

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        with self._lock:
            avg_latency = (
                sum(self._feedback_latencies) / len(self._feedback_latencies)
                if self._feedback_latencies
                else 0
            )

            return {
                "total_trades_processed": self._trade_count,
                "daily_trades": self._daily_trades,
                "daily_pnl": self._daily_pnl,
                "current_win_streak": self._win_streak,
                "current_loss_streak": self._loss_streak,
                "avg_feedback_latency_ms": avg_latency,
                "baseline_set": self._baseline_set,
                "components_active": {
                    "learning_db": self.learning_db is not None,
                    "pattern_memory": self.pattern_memory is not None,
                    "online_learner": self.online_learner is not None,
                    "rl_buffer": self.rl_buffer is not None,
                },
            }

    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations based on learning data."""
        recommendations = []

        try:
            metrics = self.learning_db.get_learning_metrics(days_lookback=14)

            # Win rate recommendations
            if metrics.win_rate < 0.45:
                recommendations.append(
                    "Win rate below 45% - consider increasing confidence threshold"
                )
            elif metrics.win_rate > 0.65:
                recommendations.append(
                    "High win rate - may be able to increase position sizes"
                )

            # Regime-specific recommendations
            for regime, stats in metrics.regime_performance.items():
                if stats["total"] >= 10:
                    if stats["win_rate"] < 0.4:
                        recommendations.append(
                            f"Poor performance in {regime} regime ({stats['win_rate']:.0%} win rate) - "
                            f"consider reducing activity in this regime"
                        )
                    elif stats["win_rate"] > 0.6:
                        recommendations.append(
                            f"Strong performance in {regime} regime ({stats['win_rate']:.0%} win rate) - "
                            f"consider increasing activity in this regime"
                        )

            # Strategy recommendations
            for strategy, stats in metrics.strategy_performance.items():
                if stats["total"] >= 10:
                    if stats["win_rate"] < 0.4:
                        recommendations.append(
                            f"{strategy} strategy underperforming ({stats['win_rate']:.0%} win rate) - "
                            f"consider disabling or retraining"
                        )

            # Consecutive loss warning
            if self._loss_streak >= 3:
                recommendations.append(
                    f"Currently on {self._loss_streak} loss streak - "
                    f"consider reducing position sizes temporarily"
                )

        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")

        return recommendations

    def reset_baseline(self):
        """Reset drift detection baseline (after retrain)."""
        with self._lock:
            self._baseline_set = False
            logger.info("Drift detection baseline reset")


def create_feedback_orchestrator(
    learning_db_path: Optional[str] = None,
    pattern_memory: Optional[Any] = None,
    online_learner: Optional[Any] = None,
    rl_buffer: Optional[Any] = None,
    config: Optional[FeedbackConfig] = None,
) -> FeedbackOrchestrator:
    """
    Factory function to create a configured FeedbackOrchestrator.

    Args:
        learning_db_path: Path to learning database
        pattern_memory: Pattern memory instance
        online_learner: Online learning manager
        rl_buffer: RL experience buffer
        config: Configuration settings

    Returns:
        Configured FeedbackOrchestrator instance
    """
    learning_db = LearningDatabase(db_path=learning_db_path) if learning_db_path else None

    return FeedbackOrchestrator(
        config=config,
        learning_db=learning_db,
        pattern_memory=pattern_memory,
        online_learner=online_learner,
        rl_buffer=rl_buffer,
    )
