"""
Enhanced Trading Engine with Advanced Features.

Integrates:
- Execution Algorithms (TWAP, VWAP, POV, IS)
- Regime-Aware Strategy Selection
- Dynamic Position Sizing & Circuit Breakers
- Stress Testing for Pre-Trade Validation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

# Regime detection
from .regime import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
    create_regime_strategy_selector,
    AdvancedRegimeStrategySelector,
    SelectionResult,
)

# Execution algorithms
from .execution import (
    AlgoOrder,
    AlgoExecution,
    UrgencyLevel,
    create_execution_algorithm,
)

# Risk controls
from .risk import (
    create_correlation_circuit_breaker,
    create_dynamic_position_sizer,
    create_stress_test_engine,
    CorrelationCircuitBreaker,
    DynamicPositionSizer,
    StressTestEngine,
    StressTestPosition,
    CircuitBreakerState,
)

# Core utilities
from .core import (
    get_logger,
    metrics,
    AsyncTaskManager,
    safe_timeout,
    validate_ohlcv,
)

logger = get_logger(__name__)


class ExecutionAdapter(Protocol):
    """Protocol for execution adapters."""

    async def get_balance(self) -> Dict[str, float]: ...
    async def get_positions(self) -> Dict[str, Dict]: ...
    async def get_price(self, symbol: str) -> float: ...
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market"
    ) -> Dict: ...
    async def get_market_data(self, symbol: str) -> Dict[str, Any]: ...


@dataclass
class EnhancedTradingConfig:
    """Configuration for enhanced trading engine."""

    # Trading basics
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    lookback_bars: int = 200
    update_interval_seconds: int = 300

    # Execution algorithm settings
    default_execution_algo: str = "adaptive"  # twap, vwap, pov, is, iceberg, adaptive
    execution_urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    max_execution_duration_minutes: int = 60

    # Risk controls
    enable_circuit_breaker: bool = True
    enable_stress_testing: bool = True
    stress_test_before_trade: bool = True
    max_stress_loss_pct: float = 0.25  # Block trades if stress test shows >25% loss

    # Position sizing
    base_risk_per_trade: float = 0.02
    max_position_size: float = 0.20
    target_volatility: float = 0.15

    # Safety
    max_daily_trades: int = 20
    min_confidence_threshold: float = 0.5

    # State persistence
    state_file: Path = Path("data/enhanced_trading_state.json")


@dataclass
class EnhancedTradingState:
    """Current state of the enhanced trading engine."""

    is_running: bool = False

    # Regime
    current_regime: Optional[MarketRegime] = None
    regime_confidence: float = 0.0

    # Strategy selection
    active_strategies: List[str] = field(default_factory=list)
    strategy_allocations: Dict[str, float] = field(default_factory=dict)

    # Risk controls
    circuit_breaker_state: str = "normal"
    position_risk_level: str = "low"
    last_stress_test_result: Optional[Dict] = None

    # Execution
    pending_executions: List[str] = field(default_factory=list)

    # Performance
    daily_pnl: float = 0.0
    trades_today: int = 0
    equity: float = 0.0

    # Timing
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "is_running": self.is_running,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "regime_confidence": self.regime_confidence,
            "active_strategies": self.active_strategies,
            "strategy_allocations": self.strategy_allocations,
            "circuit_breaker_state": self.circuit_breaker_state,
            "position_risk_level": self.position_risk_level,
            "pending_executions": self.pending_executions,
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "equity": self.equity,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class EnhancedTradingEngine:
    """
    Enhanced Trading Engine with integrated risk controls and execution algorithms.

    Features:
    - Regime-aware strategy selection
    - Advanced execution algorithms (TWAP, VWAP, POV, IS)
    - Correlation circuit breaker
    - Dynamic position sizing
    - Pre-trade stress testing
    """

    def __init__(
        self,
        config: EnhancedTradingConfig,
        execution_adapter: Optional[ExecutionAdapter] = None,
    ):
        self.config = config
        self.execution = execution_adapter

        # Initialize components
        self._init_components()

        # State
        self.state = EnhancedTradingState()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_trade: Optional[Callable] = None
        self._on_regime_change: Optional[Callable] = None
        self._on_circuit_breaker: Optional[Callable] = None

        logger.info("Enhanced trading engine initialized")

    def _init_components(self):
        """Initialize all trading components."""

        # Regime detector (per symbol)
        self.regime_detectors: Dict[str, RegimeDetector] = {
            symbol: RegimeDetector() for symbol in self.config.symbols
        }

        # Strategy selector
        self.strategy_selector: AdvancedRegimeStrategySelector = create_regime_strategy_selector(
            include_defaults=True
        )

        # Circuit breaker
        self.circuit_breaker: CorrelationCircuitBreaker = create_correlation_circuit_breaker()
        self.circuit_breaker.add_listener(self._on_circuit_breaker_change)

        # Dynamic position sizer
        self.position_sizer: DynamicPositionSizer = create_dynamic_position_sizer(
            circuit_breaker=self.circuit_breaker
        )

        # Stress test engine
        self.stress_engine: StressTestEngine = create_stress_test_engine()

        logger.info(f"Initialized components for {len(self.config.symbols)} symbols")

    def _on_circuit_breaker_change(self, status):
        """Handle circuit breaker state changes."""
        old_state = self.state.circuit_breaker_state
        new_state = status.state.value

        if old_state != new_state:
            logger.warning(f"Circuit breaker state changed: {old_state} -> {new_state}")
            self.state.circuit_breaker_state = new_state

            if self._on_circuit_breaker:
                self._on_circuit_breaker(status)

    def set_execution_adapter(self, adapter: ExecutionAdapter) -> None:
        """Set or replace the execution adapter."""
        self.execution = adapter

    def on_trade(self, callback: Callable) -> None:
        """Set callback for trade events."""
        self._on_trade = callback

    def on_regime_change(self, callback: Callable) -> None:
        """Set callback for regime changes."""
        self._on_regime_change = callback

    def on_circuit_breaker(self, callback: Callable) -> None:
        """Set callback for circuit breaker events."""
        self._on_circuit_breaker = callback

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.warning("Engine already running")
            return

        logger.info("Starting enhanced trading engine")
        self._running = True
        self.state.is_running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the trading engine."""
        if not self._running:
            return

        logger.info("Stopping enhanced trading engine")
        self._running = False
        self.state.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main trading loop."""
        logger.info("Trading loop started")

        while self._running:
            try:
                await self._update_cycle()
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)

            await asyncio.sleep(self.config.update_interval_seconds)

        logger.info("Trading loop stopped")

    async def _update_cycle(self) -> None:
        """Single update cycle."""

        if not self.execution:
            logger.warning("No execution adapter configured")
            return

        # Get account state
        balance = await self.execution.get_balance()
        positions = await self.execution.get_positions()
        self.state.equity = balance.get("total", 0)

        # Update position sizer
        self.position_sizer.set_portfolio_value(self.state.equity)

        # Process each symbol
        for symbol in self.config.symbols:
            try:
                await self._process_symbol(symbol, positions)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Update timing
        self.state.last_update = datetime.now()

    async def _process_symbol(self, symbol: str, positions: Dict) -> None:
        """Process a single symbol."""

        # Get market data
        df = await self.execution.get_ohlcv(
            symbol,
            self.config.timeframe,
            self.config.lookback_bars,
        )

        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return

        # Detect regime
        detector = self.regime_detectors.get(symbol)
        if not detector:
            detector = RegimeDetector()
            self.regime_detectors[symbol] = detector

        regime_state = detector.detect(df, symbol, self.config.timeframe)

        # Handle regime change
        if self.state.current_regime != regime_state.regime:
            logger.info(f"Regime change: {self.state.current_regime} -> {regime_state.regime}")
            if self._on_regime_change:
                self._on_regime_change(self.state.current_regime, regime_state.regime)

        self.state.current_regime = regime_state.regime
        self.state.regime_confidence = regime_state.confidence

        # Update circuit breaker with returns
        returns = df["close"].pct_change().dropna()
        if len(returns) > 20:
            returns_df = pd.DataFrame({symbol: returns})
            self.circuit_breaker.update_correlations(returns_df)

        # Check circuit breaker
        if self.circuit_breaker.state == CircuitBreakerState.TRIGGERED:
            logger.warning("Circuit breaker triggered - skipping trade evaluation")
            return

        # Select strategies
        volatility = returns.std() * np.sqrt(252 * 24) if len(returns) > 0 else 0.15
        selection = self.strategy_selector.select_strategies(
            regime_state.regime,
            volatility=volatility,
        )

        self.state.active_strategies = selection.selected_strategies

        # Get allocations
        allocations = self.strategy_selector.get_strategy_allocation(
            regime_state.regime,
            self.state.equity,
        )
        self.state.strategy_allocations = allocations

        # Get current position
        current_price = await self.execution.get_price(symbol)
        position_value = 0
        if symbol in positions:
            pos = positions[symbol]
            position_value = pos.get("quantity", 0) * current_price

        # Calculate target position
        target_allocation = selection.position_scale
        target_value = self.state.equity * target_allocation
        trade_value = target_value - position_value

        # Skip small trades
        if abs(trade_value) < 100:
            return

        # Calculate position size with dynamic sizer
        stop_loss_price = current_price * (0.97 if trade_value > 0 else 1.03)
        size_result = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
            current_volatility=volatility,
            regime=regime_state.regime.value,
        )

        # Pre-trade stress test
        if self.config.enable_stress_testing and self.config.stress_test_before_trade:
            stress_ok = await self._run_pre_trade_stress_test(
                symbol, current_price, size_result.final_size
            )
            if not stress_ok:
                logger.warning(f"Trade blocked by stress test for {symbol}")
                return

        # Check confidence threshold
        if selection.confidence < self.config.min_confidence_threshold:
            logger.debug(f"Skipping trade - low confidence: {selection.confidence:.2%}")
            return

        # Execute trade
        await self._execute_trade(
            symbol=symbol,
            side="buy" if trade_value > 0 else "sell",
            quantity=abs(trade_value) / current_price,
            price=current_price,
            regime=regime_state.regime,
            selection=selection,
        )

    async def _run_pre_trade_stress_test(
        self,
        symbol: str,
        price: float,
        proposed_size: float,
    ) -> bool:
        """
        Run stress test before executing trade.

        Returns:
            True if trade is acceptable, False if blocked
        """
        try:
            # Set up portfolio with proposed position
            positions = [
                StressTestPosition(
                    symbol=symbol,
                    quantity=proposed_size * self.state.equity / price,
                    current_price=price,
                    asset_class="crypto" if "BTC" in symbol or "ETH" in symbol else "equity",
                    beta=1.2 if "BTC" in symbol else 1.0,
                )
            ]

            self.stress_engine.set_portfolio(positions)

            # Run quick stress scenarios
            worst_loss = 0
            for scenario_name in ["covid_crash_2020", "crypto_winter_2022", "flash_crash_2010"]:
                try:
                    result = self.stress_engine.run_scenario(scenario_name)
                    if result.portfolio_pnl_pct < worst_loss:
                        worst_loss = result.portfolio_pnl_pct
                except (KeyError, ValueError, AttributeError) as e:
                    logger.debug(f"Stress scenario {scenario_name} skipped: {e}")

            self.state.last_stress_test_result = {
                "symbol": symbol,
                "worst_loss_pct": worst_loss,
                "timestamp": datetime.now().isoformat(),
            }

            # Block if worst loss exceeds threshold
            if worst_loss < -self.config.max_stress_loss_pct:
                logger.warning(
                    f"Stress test failed for {symbol}: worst case loss {worst_loss:.1%} "
                    f"exceeds threshold {-self.config.max_stress_loss_pct:.1%}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Stress test error: {e}")
            return True  # Don't block on error

    async def _execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        regime: MarketRegime,
        selection: SelectionResult,
    ) -> None:
        """Execute a trade using advanced execution algorithm."""

        # Check daily trade limit
        if self.state.trades_today >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return

        # Get market data for execution algorithm
        market_data = {}
        try:
            market_data = await self.execution.get_market_data(symbol)
        except (ConnectionError, TimeoutError, KeyError) as e:
            logger.debug(f"Using default market data for {symbol}: {e}")
            market_data = {
                "mid_price": price,
                "last_price": price,
                "volatility": 0.02,
                "spread_bps": 10,
                "volume": 1000000,
            }

        # Create order executor function
        async def order_executor(**kwargs):
            result = await self.execution.place_order(
                symbol=kwargs.get("symbol", symbol),
                side=kwargs.get("side", side),
                quantity=kwargs.get("quantity", quantity),
            )
            return {
                "fill_price": result.get("price", kwargs.get("price", price)),
                "filled_quantity": result.get("quantity", kwargs.get("quantity", quantity)),
            }

        # Create market data provider function
        async def market_data_provider(sym: str):
            try:
                return await self.execution.get_market_data(sym)
            except (ConnectionError, TimeoutError, KeyError):
                return market_data

        # Create execution algorithm
        algo = create_execution_algorithm(
            algorithm_type=self.config.default_execution_algo,
            order_executor=order_executor,
            market_data_provider=market_data_provider,
        )

        # Create order
        order = AlgoOrder(
            order_id=f"order_{datetime.now().timestamp()}",
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            urgency=self.config.execution_urgency,
        )

        # Execute
        try:
            logger.info(
                f"Executing {side.upper()} {quantity:.6f} {symbol} "
                f"using {self.config.default_execution_algo.upper()} algorithm"
            )

            self.state.pending_executions.append(order.order_id)

            execution_result: AlgoExecution = await algo.execute(order)

            self.state.pending_executions.remove(order.order_id)
            self.state.trades_today += 1

            logger.info(
                f"Execution complete: filled {execution_result.filled_quantity:.6f} @ "
                f"${execution_result.average_price:.2f} | "
                f"Slippage: {execution_result.slippage_bps:.2f}bps | "
                f"IS: {execution_result.implementation_shortfall_bps:.2f}bps"
            )

            # Callback
            if self._on_trade:
                self._on_trade(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": execution_result.filled_quantity,
                        "price": execution_result.average_price,
                        "regime": regime.value,
                        "slippage_bps": execution_result.slippage_bps,
                        "algorithm": self.config.default_execution_algo,
                    }
                )

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            if order.order_id in self.state.pending_executions:
                self.state.pending_executions.remove(order.order_id)

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "state": self.state.to_dict(),
            "circuit_breaker": self.circuit_breaker.get_status().to_dict(),
            "position_risk_level": self.position_sizer.get_portfolio_risk_level().value,
            "strategy_selector": self.strategy_selector.get_current_state(),
            "config": {
                "symbols": self.config.symbols,
                "timeframe": self.config.timeframe,
                "execution_algo": self.config.default_execution_algo,
            },
        }

    async def force_update(self) -> None:
        """Force an immediate update cycle."""
        await self._update_cycle()

    def reset_daily_counters(self) -> None:
        """Reset daily counters."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        logger.info("Daily counters reset")


def create_enhanced_trading_engine(
    symbols: List[str], execution_adapter: Optional[ExecutionAdapter] = None, **kwargs
) -> EnhancedTradingEngine:
    """
    Factory function to create enhanced trading engine.

    Args:
        symbols: List of trading symbols
        execution_adapter: Optional execution adapter
        **kwargs: Additional config options
    """
    config = EnhancedTradingConfig(symbols=symbols, **kwargs)
    return EnhancedTradingEngine(config, execution_adapter)
