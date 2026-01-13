"""
Unified Trading Engine Module.

Single entry point for all trading modes with seamless switching,
safety controls, and comprehensive state management.
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from bot.execution_adapter import (
    Balance,
    ExecutionAdapter,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    create_execution_adapter,
)
from bot.safety_controller import (
    SafetyController,
    SafetyStatus,
    create_safety_controller_for_mode,
)
from bot.trading_mode import ModeConfig, TradingMode, TradingStatus
from bot.transition_validator import TransitionValidator, create_transition_validator
from bot.unified_state import (
    EquityPoint,
    PositionState,
    TradeRecord,
    UnifiedState,
    UnifiedStateStore,
)
from bot.ml_signal_generator import MLSignalGenerator, create_signal_generator

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the unified trading engine."""

    # Mode settings
    initial_mode: TradingMode = TradingMode.PAPER_LIVE_DATA
    initial_capital: float = 10000.0

    # Trading settings
    symbols: List[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT"]
    )
    loop_interval_seconds: int = 300  # 5 minutes
    max_positions: int = 3

    # Risk settings
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit

    # Data settings
    data_dir: Path = field(default_factory=lambda: Path("data/unified_trading"))

    # API credentials (from environment)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # ML settings
    use_ml_signals: bool = True
    ml_model_type: str = "gradient_boosting"
    ml_confidence_threshold: float = 0.55

    # Signal generator (injected or auto-created)
    signal_generator: Optional[Callable] = None

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("BINANCE_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("BINANCE_API_SECRET")


class UnifiedTradingEngine:
    """
    Unified trading engine with seamless mode switching.

    Features:
    - Paper, testnet, and live trading modes
    - Safety controller with daily loss limits and position caps
    - Transition validation with strict requirements
    - Persistent state across restarts
    - Emergency stop capability
    """

    def __init__(self, config: EngineConfig):
        self.config = config

        # Core components
        self.state_store = UnifiedStateStore(data_dir=config.data_dir)
        self.safety_controller: Optional[SafetyController] = None
        self.execution_adapter: Optional[ExecutionAdapter] = None
        self.transition_validator = create_transition_validator()

        # Runtime state
        self._running = False
        self._state: Optional[UnifiedState] = None
        self._loop_task: Optional[asyncio.Task] = None

        # Signal handler
        self._signal_generator = config.signal_generator

        # Callbacks
        self._on_trade_callbacks: List[Callable] = []
        self._on_signal_callbacks: List[Callable] = []

    async def initialize(self, resume: bool = True) -> bool:
        """
        Initialize the engine.

        Args:
            resume: Whether to resume from saved state
        """
        try:
            # Initialize state
            self._state = self.state_store.initialize(
                mode=self.config.initial_mode,
                initial_capital=self.config.initial_capital,
                resume=resume,
            )

            # Create safety controller for current mode
            self.safety_controller = create_safety_controller_for_mode(
                self._state.mode.value, self.config.initial_capital
            )
            self.safety_controller.update_balance(self._state.current_balance)

            # Create execution adapter
            self.execution_adapter = create_execution_adapter(
                mode=self._state.mode.value,
                initial_balance=self._state.current_balance,
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                safety_controller=self.safety_controller,
            )

            # Connect adapter
            connected = await self.execution_adapter.connect()
            if not connected:
                logger.error("Failed to connect execution adapter")
                return False

            # Initialize ML signal generator if enabled and not already set
            if self.config.use_ml_signals and self._signal_generator is None:
                try:
                    ml_generator = create_signal_generator(
                        symbols=self.config.symbols,
                        model_type=self.config.ml_model_type,
                        confidence_threshold=self.config.ml_confidence_threshold,
                    )
                    self._signal_generator = ml_generator.generate_signal
                    logger.info(
                        f"ML signal generator initialized: {self.config.ml_model_type}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize ML signals: {e}")

            logger.info(
                f"Engine initialized: mode={self._state.mode.value}, "
                f"balance=${self._state.current_balance:.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return False

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.warning("Engine already running")
            return

        if not self._state:
            raise RuntimeError("Engine not initialized")

        self._running = True
        self._state.status = TradingStatus.ACTIVE
        self.state_store.update_state(status=TradingStatus.ACTIVE)

        logger.info(f"Starting trading engine in {self._state.mode.value} mode")

        # Start the main loop
        self._loop_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        if not self._running:
            return

        logger.info("Stopping trading engine...")
        self._running = False

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        if self.execution_adapter:
            await self.execution_adapter.disconnect()

        if self._state:
            self._state.status = TradingStatus.STOPPED
            self.state_store.update_state(status=TradingStatus.STOPPED)

        logger.info("Trading engine stopped")

    async def emergency_stop(self, reason: str) -> None:
        """Emergency stop - immediately halt all trading."""
        logger.critical(f"EMERGENCY STOP: {reason}")

        self._running = False

        if self._state:
            self._state.status = TradingStatus.EMERGENCY_STOP
            self.state_store.update_state(status=TradingStatus.EMERGENCY_STOP)

        if self.safety_controller:
            self.safety_controller.emergency_stop(reason)

        # Close all positions if in live mode
        if self._state and self._state.mode.is_live:
            await self._close_all_positions("Emergency stop")

        if self._loop_task:
            self._loop_task.cancel()

        logger.critical("Emergency stop complete - all trading halted")

    async def switch_mode(
        self, new_mode: TradingMode, force: bool = False, approver: str = ""
    ) -> tuple[bool, str]:
        """
        Switch trading mode.

        Args:
            new_mode: Target mode
            force: Force switch without validation (for downgrade only)
            approver: Name of person approving (required for live modes)

        Returns:
            (success, message)
        """
        if not self._state:
            return False, "Engine not initialized"

        current_mode = self._state.mode

        if current_mode == new_mode:
            return True, f"Already in {new_mode.value} mode"

        # Check if this is a downgrade (always allowed)
        is_downgrade = TradingMode.get_progression().index(
            new_mode
        ) < TradingMode.get_progression().index(current_mode)

        if not is_downgrade and not force:
            # Validate transition
            mode_state = self.state_store.get_mode_state()
            result = self.transition_validator.can_transition(
                current_mode, new_mode, mode_state
            )

            if not result.allowed:
                reasons = ", ".join(result.blocking_reasons)
                return False, f"Transition not allowed: {reasons}"

            if result.requires_approval and not approver:
                return False, "Approval required for this transition"

        # Pause trading during switch
        was_running = self._running
        if was_running:
            self._running = False
            await asyncio.sleep(1)  # Wait for current iteration to complete

        try:
            # Close all positions before switching to lower mode
            if is_downgrade and self._state.positions:
                await self._close_all_positions("Mode downgrade")

            # Update state
            self.state_store.change_mode(new_mode, reason="manual_switch", approver=approver)
            self._state.mode = new_mode

            # Recreate safety controller for new mode
            self.safety_controller = create_safety_controller_for_mode(
                new_mode.value, self._state.current_balance
            )
            self.safety_controller.update_balance(self._state.current_balance)

            # Recreate execution adapter
            if self.execution_adapter:
                await self.execution_adapter.disconnect()

            self.execution_adapter = create_execution_adapter(
                mode=new_mode.value,
                initial_balance=self._state.current_balance,
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                safety_controller=self.safety_controller,
            )
            await self.execution_adapter.connect()

            logger.info(
                f"Mode switched: {current_mode.value} -> {new_mode.value} "
                f"(approved by: {approver or 'system'})"
            )

            # Resume if was running
            if was_running:
                self._running = True
                self._loop_task = asyncio.create_task(self._main_loop())

            return True, f"Switched to {new_mode.value} mode"

        except Exception as e:
            logger.error(f"Failed to switch mode: {e}")
            return False, f"Mode switch failed: {e}"

    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Trading loop started")

        while self._running:
            try:
                await self._run_iteration()
                await asyncio.sleep(self.config.loop_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                if self.safety_controller:
                    self.safety_controller.record_api_error()
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Trading loop ended")

    async def _run_iteration(self) -> None:
        """Run a single trading iteration."""
        if not self._state or not self.execution_adapter:
            return

        # Check if trading is allowed
        if self.safety_controller:
            allowed, reason = self.safety_controller.is_trading_allowed()
            if not allowed:
                logger.warning(f"Trading blocked: {reason}")
                return

        # Update balance
        balance = await self.execution_adapter.get_balance()
        self._state.current_balance = balance.available + balance.in_positions
        self.state_store.update_state(current_balance=self._state.current_balance)

        if self.safety_controller:
            self.safety_controller.update_balance(self._state.current_balance)

        # Check existing positions
        await self._check_positions()

        # Generate signals for each symbol
        for symbol in self.config.symbols:
            await self._process_symbol(symbol)

        # Record equity point
        self.state_store.record_equity_point()

        # Clear API errors on successful iteration
        if self.safety_controller:
            self.safety_controller.clear_api_errors()

    async def _process_symbol(self, symbol: str) -> None:
        """Process trading signal for a symbol."""
        if not self._state or not self.execution_adapter:
            return

        # Check if we already have a position
        has_position = symbol in self._state.positions

        # Get current price
        try:
            current_price = await self.execution_adapter.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return

        # Update position price if we have one
        if has_position:
            pos = self._state.positions[symbol]
            pos.current_price = current_price
            if pos.side == "long":
                pos.unrealized_pnl = pos.quantity * (current_price - pos.entry_price)
            else:
                pos.unrealized_pnl = pos.quantity * (pos.entry_price - current_price)
            self.state_store.update_position(symbol, pos)

        # Generate signal
        signal = await self._generate_signal(symbol, current_price)

        if not signal:
            return

        # Process signal
        if signal["action"] == "BUY" and not has_position:
            await self._open_position(symbol, "long", current_price, signal)
        elif signal["action"] == "SELL" and has_position:
            await self._close_position(symbol, "Signal sell", current_price)
        elif signal["action"] == "SHORT" and not has_position:
            await self._open_position(symbol, "short", current_price, signal)

    async def _generate_signal(
        self, symbol: str, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal."""
        if self._signal_generator:
            try:
                return await self._signal_generator(symbol, current_price)
            except Exception as e:
                logger.error(f"Signal generation error: {e}")

        # Default simple signal (for testing)
        return None

    async def _open_position(
        self, symbol: str, side: str, price: float, signal: Dict[str, Any]
    ) -> None:
        """Open a new position."""
        if not self._state or not self.execution_adapter or not self.safety_controller:
            return

        # Calculate position size
        position_size = self._calculate_position_size(price)

        # Create order
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY if side == "long" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position_size,
            signal_confidence=signal.get("confidence"),
            signal_reason=signal.get("reason"),
        )

        # Safety check
        passed, reason = self.safety_controller.pre_trade_check(order)
        if not passed:
            logger.warning(f"Order blocked: {reason}")
            return

        # Execute order
        result = await self.execution_adapter.place_order(order)

        if not result.success:
            logger.error(f"Order failed: {result.error_message}")
            return

        # Create position state
        position = PositionState(
            symbol=symbol,
            quantity=result.filled_quantity,
            entry_price=result.average_price,
            side=side,
            entry_time=datetime.now().isoformat(),
            stop_loss=result.average_price * (1 - self.config.stop_loss_pct)
            if side == "long"
            else result.average_price * (1 + self.config.stop_loss_pct),
            take_profit=result.average_price * (1 + self.config.take_profit_pct)
            if side == "long"
            else result.average_price * (1 - self.config.take_profit_pct),
        )

        # Update state
        self.state_store.update_position(symbol, position)
        self._state.positions[symbol] = position

        # Update safety controller
        self.safety_controller.update_positions(
            {s: p.quantity * p.entry_price for s, p in self._state.positions.items()}
        )

        logger.info(
            f"Opened {side} position: {symbol} @ {result.average_price:.2f} "
            f"(qty: {result.filled_quantity})"
        )

        # Notify callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback("open", symbol, side, result)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    async def _close_position(
        self, symbol: str, reason: str, current_price: Optional[float] = None
    ) -> None:
        """Close an existing position."""
        if not self._state or not self.execution_adapter:
            return

        if symbol not in self._state.positions:
            return

        position = self._state.positions[symbol]

        # Create close order
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL if position.side == "long" else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )

        # Execute order
        result = await self.execution_adapter.place_order(order)

        if not result.success:
            logger.error(f"Close order failed: {result.error_message}")
            return

        # Calculate P&L
        if position.side == "long":
            pnl = position.quantity * (result.average_price - position.entry_price)
        else:
            pnl = position.quantity * (position.entry_price - result.average_price)

        pnl -= result.commission
        pnl_pct = pnl / (position.quantity * position.entry_price) * 100

        # Record trade
        trade = TradeRecord(
            id=result.order_id,
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=result.average_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            exit_reason=reason,
            commission=result.commission,
            mode=self._state.mode.value,
        )
        self.state_store.record_trade(trade)

        # Update balance
        self._state.current_balance += pnl
        self.state_store.update_state(current_balance=self._state.current_balance)

        # Remove position
        self.state_store.update_position(symbol, None)
        del self._state.positions[symbol]

        # Update safety controller
        if self.safety_controller:
            self.safety_controller.update_positions(
                {s: p.quantity * p.entry_price for s, p in self._state.positions.items()}
            )
            # Record trade result for daily stats
            result_obj = type("Result", (), {"pnl": pnl})()
            self.safety_controller.post_trade_check(result_obj)

        logger.info(
            f"Closed {position.side} position: {symbol} @ {result.average_price:.2f} "
            f"(P&L: ${pnl:.2f} / {pnl_pct:.2f}%) - {reason}"
        )

        # Notify callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback("close", symbol, position.side, result, pnl)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    async def _check_positions(self) -> None:
        """Check all positions for stop loss / take profit."""
        if not self._state or not self.execution_adapter:
            return

        for symbol, position in list(self._state.positions.items()):
            try:
                current_price = await self.execution_adapter.get_current_price(symbol)

                # Check stop loss
                if position.stop_loss:
                    if position.side == "long" and current_price <= position.stop_loss:
                        await self._close_position(symbol, "Stop loss", current_price)
                        continue
                    elif position.side == "short" and current_price >= position.stop_loss:
                        await self._close_position(symbol, "Stop loss", current_price)
                        continue

                # Check take profit
                if position.take_profit:
                    if position.side == "long" and current_price >= position.take_profit:
                        await self._close_position(symbol, "Take profit", current_price)
                        continue
                    elif position.side == "short" and current_price <= position.take_profit:
                        await self._close_position(symbol, "Take profit", current_price)
                        continue

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        if not self._state:
            return

        for symbol in list(self._state.positions.keys()):
            await self._close_position(symbol, reason)

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk parameters."""
        if not self._state:
            return 0.0

        # Risk-based sizing
        risk_amount = self._state.current_balance * self.config.risk_per_trade_pct
        stop_distance = price * self.config.stop_loss_pct
        position_value = risk_amount / self.config.stop_loss_pct

        # Apply mode-specific limits
        mode_config = ModeConfig.get_default(self._state.mode)
        max_position_value = min(
            position_value,
            mode_config.max_position_usd,
            self._state.current_balance * mode_config.max_position_size_pct
            if hasattr(mode_config, "max_position_size_pct")
            else position_value,
        )

        quantity = max_position_value / price
        return quantity

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status for dashboard."""
        if not self._state:
            return {"error": "Engine not initialized"}

        safety_status = (
            self.safety_controller.get_status() if self.safety_controller else {}
        )

        return {
            "mode": self._state.mode.value,
            "status": self._state.status.value,
            "running": self._running,
            "balance": self._state.current_balance,
            "initial_capital": self._state.initial_capital,
            "total_pnl": self._state.total_pnl,
            "total_pnl_pct": (
                self._state.total_pnl / self._state.initial_capital * 100
                if self._state.initial_capital > 0
                else 0
            ),
            "total_trades": self._state.total_trades,
            "win_rate": self._state.win_rate * 100,
            "max_drawdown": self._state.max_drawdown_pct * 100,
            "open_positions": len(self._state.positions),
            "positions": {
                s: {
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for s, p in self._state.positions.items()
            },
            "safety": safety_status,
            "daily_trades": self._state.daily_trades,
            "daily_pnl": self._state.daily_pnl,
        }

    def get_transition_status(self, target_mode: TradingMode) -> Dict[str, Any]:
        """Get transition progress to a target mode."""
        if not self._state:
            return {"error": "Engine not initialized"}

        mode_state = self.state_store.get_mode_state()
        return self.transition_validator.get_transition_progress(
            self._state.mode, target_mode, mode_state
        )

    def on_trade(self, callback: Callable) -> None:
        """Register a callback for trade events."""
        self._on_trade_callbacks.append(callback)

    def on_signal(self, callback: Callable) -> None:
        """Register a callback for signal events."""
        self._on_signal_callbacks.append(callback)


async def create_engine(config: Optional[EngineConfig] = None) -> UnifiedTradingEngine:
    """Create and initialize a trading engine."""
    if config is None:
        config = EngineConfig()

    engine = UnifiedTradingEngine(config)
    await engine.initialize()
    return engine
