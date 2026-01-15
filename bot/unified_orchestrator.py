"""
Unified Trading Orchestrator

Phase 8: Integration layer that wires all trading modules together into
a cohesive, production-ready trading system.

This orchestrator connects:
- Risk Guardian (veto power over all decisions)
- Strategy Registry (multi-strategy support)
- Meta-Allocator (capital allocation)
- Walk-Forward Validator (strategy validation)
- Promotion Gate (safe deployment)
- AI Integration (learning systems)
- News Feature Extractor (fundamental data)
- Trade Ledger (audit trail)
- Unified Execution Engine (order execution)
- WebSocket Hub (real-time streaming)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import traceback

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Operating modes for the trading system."""
    LIVE = "live"           # Real money trading
    PAPER = "paper"         # Paper trading with simulated execution
    BACKTEST = "backtest"   # Historical backtesting
    SHADOW = "shadow"       # Shadow mode - signals only, no execution


class SystemState(Enum):
    """Overall system state."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# Alias for backward compatibility
SystemStatus = SystemState


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    initial_capital: float = 10000.0
    max_positions: int = 10
    enable_notifications: bool = True
    rebalance_interval_hours: float = 24.0
    health_check_interval_seconds: float = 30.0
    max_drawdown_pct: float = 10.0
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    trading_mode: TradingMode = TradingMode.PAPER


@dataclass
class TradingDecision:
    """A trading decision with full context."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: float
    price: float

    # Source info
    strategy_name: str
    strategy_signal_strength: float
    strategy_confidence: float

    # AI contributions
    ai_sentiment: Optional[float] = None
    ai_recommendation: Optional[str] = None
    news_sentiment: Optional[float] = None

    # Risk assessment
    risk_approved: bool = False
    risk_reason: str = ""
    position_size_adjusted: bool = False
    original_quantity: Optional[float] = None

    # Allocation info
    allocation_weight: float = 1.0
    regime: str = "unknown"

    # Execution
    executed: bool = False
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    slippage: Optional[float] = None


@dataclass
class SystemHealth:
    """System health metrics."""
    state: SystemState
    uptime_seconds: float
    last_heartbeat: datetime

    # Component status
    risk_guardian_active: bool = False
    execution_engine_connected: bool = False
    data_feed_active: bool = False
    websocket_clients: int = 0

    # Performance
    decisions_today: int = 0
    trades_today: int = 0
    errors_today: int = 0

    # Risk metrics
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)


class UnifiedOrchestrator:
    """
    Central orchestrator that coordinates all trading system components.

    Responsibilities:
    1. Initialize and manage all components
    2. Route market data to strategies
    3. Aggregate signals and apply risk filters
    4. Execute approved trades
    5. Stream updates to connected clients
    6. Handle errors and recovery
    7. Provide system health monitoring
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        config: Optional["OrchestratorConfig"] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            mode: Trading mode (live, paper, backtest, shadow)
            config: Configuration object or None for defaults
        """
        self.mode = mode
        self.config = config if config is not None else OrchestratorConfig()
        self.state = SystemState.INITIALIZING
        self.start_time = datetime.now()

        # Core components (injected or created)
        self.risk_guardian = None
        self.strategy_registry = None
        self.meta_allocator = None
        self.walk_forward_validator = None
        self.promotion_gate = None
        self.ai_integration = None
        self.news_extractor = None
        self.trade_ledger = None
        self.execution_engine = None
        self.websocket_hub = None
        self.data_streamer = None

        # State tracking
        self.current_regime: str = "unknown"
        self.active_strategies: Dict[str, bool] = {}
        self.pending_decisions: List[TradingDecision] = []
        self.recent_decisions: List[TradingDecision] = []

        # Event handlers
        self._on_trade_callbacks: List[Callable] = []
        self._on_signal_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        self._on_regime_change_callbacks: List[Callable] = []

        # Background tasks
        self._main_loop_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._rebalance_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = {
            "decisions_total": 0,
            "decisions_approved": 0,
            "decisions_rejected": 0,
            "trades_executed": 0,
            "errors_total": 0
        }

        logger.info(f"UnifiedOrchestrator initialized in {mode.value} mode")

    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self.state == SystemState.RUNNING

    def inject_components(
        self,
        risk_guardian=None,
        strategy_registry=None,
        meta_allocator=None,
        walk_forward_validator=None,
        promotion_gate=None,
        ai_integration=None,
        news_extractor=None,
        trade_ledger=None,
        execution_engine=None,
        websocket_hub=None,
        data_streamer=None
    ):
        """
        Inject component dependencies.

        This allows for flexible composition and testing.
        """
        if risk_guardian:
            self.risk_guardian = risk_guardian
        if strategy_registry:
            self.strategy_registry = strategy_registry
        if meta_allocator:
            self.meta_allocator = meta_allocator
        if walk_forward_validator:
            self.walk_forward_validator = walk_forward_validator
        if promotion_gate:
            self.promotion_gate = promotion_gate
        if ai_integration:
            self.ai_integration = ai_integration
        if news_extractor:
            self.news_extractor = news_extractor
        if trade_ledger:
            self.trade_ledger = trade_ledger
        if execution_engine:
            self.execution_engine = execution_engine
        if websocket_hub:
            self.websocket_hub = websocket_hub
        if data_streamer:
            self.data_streamer = data_streamer

        logger.info("Components injected into orchestrator")

    async def start(self):
        """Start the orchestrator and all background tasks."""
        if self.state == SystemState.RUNNING:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting UnifiedOrchestrator...")

        try:
            # Validate required components
            self._validate_components()

            # Start WebSocket hub if available
            if self.websocket_hub:
                await self.websocket_hub.start()

            # Initialize strategies
            if self.strategy_registry:
                self._initialize_strategies()

            # Start background tasks
            self._main_loop_task = asyncio.create_task(self._main_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            # Start periodic rebalancing if meta-allocator is available
            if self.meta_allocator:
                self._rebalance_task = asyncio.create_task(self._rebalance_loop())

            self.state = SystemState.RUNNING
            logger.info("UnifiedOrchestrator started successfully")

            # Broadcast system status
            await self._broadcast_system_status()

        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"Failed to start orchestrator: {e}")
            raise

    async def stop(self, reason: str = "Manual stop"):
        """Stop the orchestrator gracefully."""
        logger.info(f"Stopping UnifiedOrchestrator: {reason}")

        self.state = SystemState.STOPPED

        # Cancel background tasks
        for task in [self._main_loop_task, self._health_check_task, self._rebalance_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop WebSocket hub
        if self.websocket_hub:
            await self.websocket_hub.stop()

        # Broadcast final status
        await self._broadcast_system_status()

        logger.info("UnifiedOrchestrator stopped")

    async def pause(self, reason: str = "Manual pause"):
        """Pause trading (no new trades, monitoring continues)."""
        logger.info(f"Pausing trading: {reason}")
        self.state = SystemState.PAUSED
        await self._broadcast_system_status()

    async def resume(self):
        """Resume trading after pause."""
        if self.state == SystemState.PAUSED:
            logger.info("Resuming trading")
            self.state = SystemState.RUNNING
            await self._broadcast_system_status()

    def _validate_components(self):
        """Validate that required components are available."""
        warnings = []

        if not self.risk_guardian:
            warnings.append("Risk Guardian not configured - using permissive defaults")

        if not self.strategy_registry:
            warnings.append("Strategy Registry not configured - no strategies available")

        if not self.execution_engine and self.mode in [TradingMode.LIVE, TradingMode.PAPER]:
            warnings.append("Execution Engine not configured - trades will not execute")

        if warnings:
            for w in warnings:
                logger.warning(w)

    def _initialize_strategies(self):
        """Initialize and activate strategies from registry."""
        if not self.strategy_registry:
            return

        strategies = self.strategy_registry.list_strategies()
        for strategy_name in strategies:
            self.active_strategies[strategy_name] = True
            logger.info(f"Activated strategy: {strategy_name}")

    async def process_market_update(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
        additional_data: Optional[Dict] = None
    ):
        """
        Process a market data update through the full pipeline.

        This is the main entry point for market data.

        Pipeline:
        1. Update market state
        2. Get signals from all active strategies
        3. Aggregate with AI/news sentiment
        4. Apply meta-allocation weights
        5. Check with Risk Guardian
        6. Execute approved trades
        7. Stream updates to clients
        """
        if self.state != SystemState.RUNNING:
            return

        timestamp = timestamp or datetime.now()

        try:
            # Build market state
            market_state = self._build_market_state(
                symbol, price, volume, timestamp, additional_data
            )

            # Get news features if available
            news_features = None
            if self.news_extractor:
                news_features = self.news_extractor.extract_features(timestamp)

            # Collect signals from all active strategies
            signals = await self._collect_strategy_signals(market_state)

            if not signals:
                return

            # Get AI recommendations if available
            ai_decisions = None
            if self.ai_integration:
                ai_decisions = self.ai_integration.aggregate_decisions(
                    market_state,
                    include_shadow=True
                )

            # Apply meta-allocation weights
            if self.meta_allocator and len(signals) > 1:
                signals = self._apply_allocation_weights(signals)

            # Create trading decisions
            decisions = self._create_trading_decisions(
                signals, market_state, news_features, ai_decisions
            )

            # Process each decision through risk checks and execution
            for decision in decisions:
                await self._process_decision(decision)

        except Exception as e:
            self.metrics["errors_total"] += 1
            logger.error(f"Error processing market update: {e}")
            await self._handle_error(e, {"symbol": symbol, "price": price})

    def _build_market_state(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime,
        additional_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build a market state dictionary for strategies."""
        state = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp,
            "regime": self.current_regime
        }

        if additional_data:
            state.update(additional_data)

        return state

    async def _collect_strategy_signals(
        self,
        market_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect signals from all active strategies."""
        signals = []

        if not self.strategy_registry:
            return signals

        symbol = market_state["symbol"]

        for strategy_name, is_active in self.active_strategies.items():
            if not is_active:
                continue

            try:
                strategy = self.strategy_registry.get_strategy(strategy_name)
                if not strategy:
                    continue

                # Check if strategy is suitable for current regime
                if hasattr(strategy, 'suitable_regimes'):
                    if self.current_regime not in strategy.suitable_regimes:
                        continue

                # Get signal from strategy
                signal = strategy.predict(market_state)

                if signal and signal.action != "hold":
                    signals.append({
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "action": signal.action,
                        "strength": signal.strength,
                        "confidence": signal.confidence,
                        "quantity": signal.suggested_quantity,
                        "price": market_state["price"],
                        "reasons": signal.reasons if hasattr(signal, 'reasons') else []
                    })

                    # Stream signal to clients
                    if self.data_streamer:
                        await self.data_streamer.stream_strategy_signal(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            signal=signal.action,
                            strength=signal.strength,
                            confidence=signal.confidence,
                            reasons=signal.reasons if hasattr(signal, 'reasons') else []
                        )

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} error: {e}")

        return signals

    def _apply_allocation_weights(
        self,
        signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply meta-allocation weights to signals."""
        if not self.meta_allocator:
            return signals

        # Get current allocations
        strategy_names = [s["strategy_name"] for s in signals]

        # This would use actual performance data in production
        allocations = {}
        for name in strategy_names:
            # Default equal weight if no allocation data
            allocations[name] = 1.0 / len(strategy_names)

        # Apply weights to signals
        for signal in signals:
            weight = allocations.get(signal["strategy_name"], 0.0)
            signal["allocation_weight"] = weight
            signal["quantity"] = signal["quantity"] * weight

        return signals

    def _create_trading_decisions(
        self,
        signals: List[Dict[str, Any]],
        market_state: Dict[str, Any],
        news_features: Optional[Any],
        ai_decisions: Optional[Dict]
    ) -> List[TradingDecision]:
        """Create trading decisions from signals."""
        decisions = []

        for signal in signals:
            decision_id = f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal['strategy_name'][:3]}"

            # Get news sentiment if available
            news_sentiment = None
            if news_features:
                news_sentiment = news_features.overall_sentiment

            # Get AI recommendation if available
            ai_sentiment = None
            ai_recommendation = None
            if ai_decisions:
                ai_sentiment = ai_decisions.get("aggregate_sentiment")
                ai_recommendation = ai_decisions.get("recommendation")

            decision = TradingDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                symbol=signal["symbol"],
                action=signal["action"],
                quantity=signal["quantity"],
                price=signal["price"],
                strategy_name=signal["strategy_name"],
                strategy_signal_strength=signal["strength"],
                strategy_confidence=signal["confidence"],
                ai_sentiment=ai_sentiment,
                ai_recommendation=ai_recommendation,
                news_sentiment=news_sentiment,
                allocation_weight=signal.get("allocation_weight", 1.0),
                regime=self.current_regime
            )

            decisions.append(decision)

        return decisions

    async def _process_decision(self, decision: TradingDecision):
        """Process a trading decision through risk checks and execution."""
        self.metrics["decisions_total"] += 1

        # Risk Guardian check
        risk_approved, risk_reason, adjusted_quantity = await self._check_risk(decision)

        decision.risk_approved = risk_approved
        decision.risk_reason = risk_reason

        if adjusted_quantity != decision.quantity:
            decision.position_size_adjusted = True
            decision.original_quantity = decision.quantity
            decision.quantity = adjusted_quantity

        if not risk_approved:
            self.metrics["decisions_rejected"] += 1
            logger.info(f"Decision {decision.decision_id} rejected: {risk_reason}")

            # Still record for audit
            if self.trade_ledger:
                self.trade_ledger.record_decision(decision, executed=False)

            self.recent_decisions.append(decision)
            return

        self.metrics["decisions_approved"] += 1

        # Execute if not in shadow mode
        if self.mode != TradingMode.SHADOW and self.state == SystemState.RUNNING:
            await self._execute_decision(decision)

        # Record in ledger
        if self.trade_ledger:
            self.trade_ledger.record_decision(decision, executed=decision.executed)

        self.recent_decisions.append(decision)

        # Trim recent decisions list
        if len(self.recent_decisions) > 1000:
            self.recent_decisions = self.recent_decisions[-500:]

    async def _check_risk(
        self,
        decision: TradingDecision
    ) -> Tuple[bool, str, float]:
        """
        Check decision with Risk Guardian.

        Returns:
            Tuple of (approved, reason, adjusted_quantity)
        """
        if not self.risk_guardian:
            # No risk guardian - approve with warning
            return True, "No risk guardian configured", decision.quantity

        try:
            # Build context for risk check
            context = {
                "symbol": decision.symbol,
                "action": decision.action,
                "quantity": decision.quantity,
                "price": decision.price,
                "strategy": decision.strategy_name,
                "regime": decision.regime
            }

            # Check with risk guardian
            result = self.risk_guardian.check_trade(context)

            if hasattr(result, 'approved'):
                return result.approved, result.reason, result.adjusted_quantity
            elif isinstance(result, dict):
                return (
                    result.get("approved", False),
                    result.get("reason", "Unknown"),
                    result.get("adjusted_quantity", decision.quantity)
                )
            else:
                return bool(result), "Risk check passed" if result else "Risk check failed", decision.quantity

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, f"Risk check error: {e}", decision.quantity

    async def _execute_decision(self, decision: TradingDecision):
        """Execute an approved trading decision."""
        if not self.execution_engine:
            logger.warning("No execution engine - decision not executed")
            return

        try:
            # Execute through unified execution engine
            result = await self.execution_engine.execute_order(
                symbol=decision.symbol,
                side=decision.action,
                quantity=decision.quantity,
                order_type="market",
                price=decision.price
            )

            if result and result.get("success"):
                decision.executed = True
                decision.execution_price = result.get("fill_price", decision.price)
                decision.execution_time = datetime.now()
                decision.slippage = (
                    (decision.execution_price - decision.price) / decision.price
                    if decision.execution_price else None
                )

                self.metrics["trades_executed"] += 1

                # Stream trade execution
                if self.data_streamer:
                    await self.data_streamer.stream_trade_execution(
                        trade_id=decision.decision_id,
                        symbol=decision.symbol,
                        side=decision.action,
                        quantity=decision.quantity,
                        price=decision.execution_price or decision.price,
                        strategy=decision.strategy_name
                    )

                # Notify callbacks
                for callback in self._on_trade_callbacks:
                    try:
                        await callback(decision)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")

                logger.info(
                    f"Trade executed: {decision.action} {decision.quantity} "
                    f"{decision.symbol} @ {decision.execution_price}"
                )
            else:
                logger.warning(f"Trade execution failed: {result}")

        except Exception as e:
            logger.error(f"Execution error: {e}")
            await self._handle_error(e, {"decision": decision.decision_id})

    async def update_regime(self, new_regime: str, confidence: float = 0.0):
        """Update the current market regime."""
        old_regime = self.current_regime
        self.current_regime = new_regime

        if old_regime != new_regime:
            logger.info(f"Regime changed: {old_regime} -> {new_regime}")

            # Stream regime change
            if self.data_streamer:
                await self.data_streamer.stream_regime_change(
                    previous_regime=old_regime,
                    new_regime=new_regime,
                    confidence=confidence
                )

            # Notify callbacks
            for callback in self._on_regime_change_callbacks:
                try:
                    await callback(old_regime, new_regime, confidence)
                except Exception as e:
                    logger.error(f"Regime change callback error: {e}")

    async def kill_switch(self, action: str = "stop", reason: str = "Manual"):
        """
        Emergency kill switch.

        Args:
            action: "stop" to stop trading, "close_all" to also close positions
            reason: Reason for activation
        """
        logger.warning(f"Kill switch activated: {action} - {reason}")

        # Stop all trading immediately
        await self.pause(reason=f"Kill switch: {reason}")

        if action == "close_all" and self.execution_engine:
            # Close all open positions
            try:
                await self.execution_engine.close_all_positions(reason=reason)
                logger.info("All positions closed")
            except Exception as e:
                logger.error(f"Error closing positions: {e}")

        # Stream alert
        if self.data_streamer:
            from bot.websocket_streaming import AlertLevel
            await self.data_streamer.stream_alert(
                title="Kill Switch Activated",
                message=reason,
                level=AlertLevel.EMERGENCY,
                category="kill_switch",
                action_required=True
            )

    async def adjust_risk_limit(self, limit_type: str, value: float) -> bool:
        """
        Adjust a risk limit.

        Args:
            limit_type: Type of limit to adjust
            value: New value

        Returns:
            True if successful
        """
        if not self.risk_guardian:
            logger.warning("No risk guardian to adjust")
            return False

        try:
            if hasattr(self.risk_guardian, 'update_limit'):
                self.risk_guardian.update_limit(limit_type, value)
                logger.info(f"Risk limit updated: {limit_type} = {value}")
                return True
            else:
                logger.warning("Risk guardian does not support limit updates")
                return False
        except Exception as e:
            logger.error(f"Error updating risk limit: {e}")
            return False

    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy."""
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name] = True
            logger.info(f"Strategy enabled: {strategy_name}")
            return True
        return False

    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy."""
        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name] = False
            logger.info(f"Strategy disabled: {strategy_name}")
            return True
        return False

    async def _main_loop(self):
        """Main processing loop."""
        logger.info("Main loop started")

        while self.state in [SystemState.RUNNING, SystemState.PAUSED]:
            try:
                await asyncio.sleep(1)  # Heartbeat interval

                # Process any pending work
                if self.state == SystemState.RUNNING:
                    # Could process queued data here
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await self._handle_error(e)

        logger.info("Main loop stopped")

    async def _health_check_loop(self):
        """Periodic health checking."""
        while self.state != SystemState.STOPPED:
            try:
                await asyncio.sleep(30)
                health = self.get_health()

                # Check for critical issues
                if health.current_drawdown > 0.08:  # 8% drawdown warning
                    if self.data_streamer:
                        from bot.websocket_streaming import AlertLevel
                        await self.data_streamer.stream_risk_alert(
                            alert_type="drawdown_warning",
                            level=AlertLevel.WARNING,
                            message=f"Drawdown at {health.current_drawdown:.1%}",
                            metrics={"current_drawdown": health.current_drawdown}
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _rebalance_loop(self):
        """Periodic portfolio rebalancing."""
        rebalance_interval = getattr(self.config, 'rebalance_interval_hours', 24)

        while self.state != SystemState.STOPPED:
            try:
                await asyncio.sleep(rebalance_interval * 3600)

                if self.state == SystemState.RUNNING and self.meta_allocator:
                    logger.info("Starting periodic rebalance")
                    # Rebalance logic would go here

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalance error: {e}")

    async def _broadcast_system_status(self):
        """Broadcast system status to connected clients."""
        if not self.data_streamer:
            return

        active_strats = [k for k, v in self.active_strategies.items() if v]

        await self.data_streamer.stream_system_status(
            status=self.state.value,
            trading_enabled=self.state == SystemState.RUNNING,
            connected_exchanges=["binance"],  # Would be dynamic
            active_strategies=active_strats,
            warnings=[]
        )

    async def _handle_error(self, error: Exception, context: Optional[Dict] = None):
        """Handle errors centrally."""
        error_info = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        logger.error(f"Error handled: {error_info}")

        # Notify callbacks
        for callback in self._on_error_callbacks:
            try:
                await callback(error, context)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

        # Stream error alert for critical errors
        if self.data_streamer:
            from bot.websocket_streaming import AlertLevel
            await self.data_streamer.stream_alert(
                title="System Error",
                message=str(error),
                level=AlertLevel.WARNING,
                category="error"
            )

    def get_health(self) -> SystemHealth:
        """Get current system health."""
        return SystemHealth(
            state=self.state,
            uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            last_heartbeat=datetime.now(),
            risk_guardian_active=self.risk_guardian is not None,
            execution_engine_connected=self.execution_engine is not None,
            data_feed_active=True,  # Would check actual feed
            websocket_clients=len(self.websocket_hub.clients) if self.websocket_hub else 0,
            decisions_today=self.metrics["decisions_total"],
            trades_today=self.metrics["trades_executed"],
            errors_today=self.metrics["errors_total"]
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status dictionary."""
        health = self.get_health()

        return {
            "state": self.state.value,
            "mode": self.mode.value,
            "uptime_seconds": health.uptime_seconds,
            "current_regime": self.current_regime,
            "active_strategies": [k for k, v in self.active_strategies.items() if v],
            "disabled_strategies": [k for k, v in self.active_strategies.items() if not v],
            "metrics": self.metrics,
            "components": {
                "risk_guardian": self.risk_guardian is not None,
                "strategy_registry": self.strategy_registry is not None,
                "meta_allocator": self.meta_allocator is not None,
                "ai_integration": self.ai_integration is not None,
                "news_extractor": self.news_extractor is not None,
                "execution_engine": self.execution_engine is not None,
                "websocket_hub": self.websocket_hub is not None
            },
            "recent_decisions_count": len(self.recent_decisions)
        }

    def on_trade(self, callback: Callable):
        """Register a callback for trade executions."""
        self._on_trade_callbacks.append(callback)

    def on_signal(self, callback: Callable):
        """Register a callback for strategy signals."""
        self._on_signal_callbacks.append(callback)

    def on_error(self, callback: Callable):
        """Register a callback for errors."""
        self._on_error_callbacks.append(callback)

    def on_regime_change(self, callback: Callable):
        """Register a callback for regime changes."""
        self._on_regime_change_callbacks.append(callback)


# Factory function for creating orchestrator with all components
def create_orchestrator(
    mode: TradingMode = TradingMode.PAPER,
    config: Optional[Dict[str, Any]] = None
) -> UnifiedOrchestrator:
    """
    Create an orchestrator with default component initialization.

    Args:
        mode: Trading mode
        config: Configuration dictionary

    Returns:
        Configured UnifiedOrchestrator
    """
    orchestrator = UnifiedOrchestrator(mode=mode, config=config)

    # Components would be initialized here based on config
    # For now, return empty orchestrator for injection

    return orchestrator


if __name__ == "__main__":
    # Demo usage
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def demo():
        # Create orchestrator
        orchestrator = create_orchestrator(mode=TradingMode.PAPER)

        print("=== Unified Orchestrator Demo ===")
        print(f"Status: {json.dumps(orchestrator.get_status(), indent=2)}")

        # Start orchestrator (would need components for full functionality)
        # await orchestrator.start()

        # Process a market update
        # await orchestrator.process_market_update("BTC/USDT", 45000, 1000)

        print("\nHealth:")
        health = orchestrator.get_health()
        print(f"  State: {health.state.value}")
        print(f"  Uptime: {health.uptime_seconds:.0f}s")

    asyncio.run(demo())
