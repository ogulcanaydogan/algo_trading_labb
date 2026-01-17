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
from bot.control import load_bot_control
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
from bot.trailing_stop import TrailingStopManager, create_trailing_stop_manager
from bot.dca_manager import DCAManager, create_dca_manager

# Self-learning imports
try:
    from bot.optimal_action_tracker import (
        OptimalActionTracker,
        MarketState,
        ActionOutcome,
        StateActionRecord,
        ActionType,
        get_tracker as get_action_tracker,
    )
    from bot.adaptive_risk_controller import (
        AdaptiveRiskController,
        get_adaptive_risk_controller,
    )
    SELF_LEARNING_AVAILABLE = True
except ImportError:
    SELF_LEARNING_AVAILABLE = False

# AI Trading Brain imports
try:
    from bot.ai_trading_brain import (
        AITradingBrain,
        MarketSnapshot,
        MarketCondition,
        get_ai_brain,
    )
    AI_BRAIN_AVAILABLE = True
except ImportError:
    AI_BRAIN_AVAILABLE = False

# Intelligent Trading Brain imports (new AI system)
try:
    from bot.intelligence import (
        IntelligentTradingBrain,
        BrainConfig,
        TradeOutcome,
        get_intelligent_brain,
    )
    INTELLIGENT_BRAIN_AVAILABLE = True
except ImportError:
    INTELLIGENT_BRAIN_AVAILABLE = False

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
    ml_confidence_threshold: float = 0.45  # Lowered from 0.55 for more signals

    # Signal generator (injected or auto-created)
    signal_generator: Optional[Callable] = None

    # Trailing stop settings
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.015  # 1.5% trailing distance
    trailing_activation_pct: float = 0.01  # Activate after 1% profit

    # DCA settings
    use_dca: bool = True
    max_dca_orders: int = 4
    max_position_multiplier: float = 3.0

    # Self-learning settings
    use_action_tracker: bool = True  # Track optimal actions per market state
    use_adaptive_risk: bool = True   # Auto-adjust risk settings

    # AI Trading Brain settings
    use_ai_brain: bool = True        # Use AI brain for learning and strategy generation
    daily_target_pct: float = 1.0    # Daily target gain (%)
    max_daily_loss_pct: float = 2.0  # Max daily loss before stopping (%)

    # Intelligent Trading Brain settings (new AI system)
    use_intelligent_brain: bool = True  # Use intelligent brain for explanations and learning

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.api_key is None:
            # Load testnet keys if in testnet mode, otherwise live keys
            if self.initial_mode == TradingMode.TESTNET:
                self.api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            else:
                self.api_key = os.getenv("BINANCE_API_KEY")
        if self.api_secret is None:
            if self.initial_mode == TradingMode.TESTNET:
                self.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
            else:
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

        # Trailing stop manager
        self.trailing_stop_manager: Optional[TrailingStopManager] = None
        if config.use_trailing_stop:
            self.trailing_stop_manager = create_trailing_stop_manager(
                enabled=True,
                initial_stop_pct=config.stop_loss_pct,
                trailing_pct=config.trailing_stop_pct,
                activation_profit_pct=config.trailing_activation_pct,
            )

        # DCA manager
        self.dca_manager: Optional[DCAManager] = None
        if config.use_dca:
            self.dca_manager = create_dca_manager(
                enabled=True,
                max_dca_orders=config.max_dca_orders,
                max_position_multiplier=config.max_position_multiplier,
            )

        # Self-learning components
        self.action_tracker: Optional[OptimalActionTracker] = None
        self.adaptive_risk_controller: Optional[AdaptiveRiskController] = None
        self._action_record_ids: Dict[str, int] = {}  # Track open positions for outcome recording
        self._prediction_ids: Dict[str, str] = {}  # Track ML prediction IDs for performance tracking

        if SELF_LEARNING_AVAILABLE:
            if config.use_action_tracker:
                try:
                    self.action_tracker = get_action_tracker()
                    logger.info("Optimal action tracker initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize action tracker: {e}")

            if config.use_adaptive_risk:
                try:
                    self.adaptive_risk_controller = get_adaptive_risk_controller()
                    logger.info("Adaptive risk controller initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize adaptive risk controller: {e}")

        # AI Trading Brain
        self.ai_brain: Optional[AITradingBrain] = None
        self._trade_snapshots: Dict[str, MarketSnapshot] = {}  # Store entry snapshots for learning

        if AI_BRAIN_AVAILABLE and config.use_ai_brain:
            try:
                self.ai_brain = get_ai_brain()
                logger.info("AI Trading Brain initialized - Target: 1% daily gain")
            except Exception as e:
                logger.warning(f"Could not initialize AI brain: {e}")

        # Intelligent Trading Brain (new AI system with explanations and learning)
        self.intelligent_brain: Optional[IntelligentTradingBrain] = None

        if INTELLIGENT_BRAIN_AVAILABLE and config.use_intelligent_brain:
            try:
                self.intelligent_brain = get_intelligent_brain()
                logger.info("Intelligent Trading Brain initialized - Explanations and learning enabled")
            except Exception as e:
                logger.warning(f"Could not initialize intelligent brain: {e}")

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
                import traceback
                logger.error(f"Error in trading loop: {e}\n{traceback.format_exc()}")
                if self.safety_controller:
                    self.safety_controller.record_api_error()
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Trading loop ended")

    async def _run_iteration(self) -> None:
        """Run a single trading iteration."""
        if not self._state or not self.execution_adapter:
            return

        # Check if manually paused via control.json
        control_state = load_bot_control(self.state_store.data_dir)
        if control_state.paused:
            logger.debug(f"Trading paused: {control_state.reason}")
            # Still update balance for display purposes
            try:
                balance = await self.execution_adapter.get_balance()
                self._state.current_balance = balance.available + balance.in_positions
                self.state_store.update_state(current_balance=self._state.current_balance)
            except Exception as e:
                logger.debug(f"Failed to update balance during pause: {e}")
            return

        # Check if trading is allowed
        if self.safety_controller:
            allowed, reason = self.safety_controller.is_trading_allowed()
            if not allowed:
                logger.warning(f"Trading blocked: {reason}")
                # Log this EVERY iteration so we can see what's blocking
                logger.info(f"[SAFETY BLOCK] Reason: {reason} | Allowed: {allowed}")
                return
            logger.debug(f"[SAFETY PASS] Trading allowed")

        # Update balance
        balance = await self.execution_adapter.get_balance()
        self._state.current_balance = balance.available + balance.in_positions
        self.state_store.update_state(current_balance=self._state.current_balance)

        if self.safety_controller:
            self.safety_controller.update_balance(self._state.current_balance)

        # Evaluate and adjust risk settings based on market conditions
        await self._evaluate_adaptive_risk()

        # Check existing positions
        await self._check_positions()

        # Generate signals for each symbol
        for symbol in self.config.symbols:
            await self._process_symbol(symbol)

        # Update balance with unrealized P&L from all open positions
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self._state.positions.values()
        )
        # Balance = initial capital + realized trades P&L + unrealized position P&L
        marked_to_market_balance = self._state.initial_capital + self._state.total_pnl + total_unrealized_pnl
        
        # Only update if different (to avoid constant writes)
        if abs(marked_to_market_balance - self._state.current_balance) > 0.01:
            self._state.current_balance = marked_to_market_balance
            self.state_store.update_state(current_balance=self._state.current_balance)
            if self.safety_controller:
                self.safety_controller.update_balance(self._state.current_balance)
            logger.debug(
                f"Updated balance with unrealized P&L: ${self._state.current_balance:.2f} "
                f"(unrealized: ${total_unrealized_pnl:.2f})"
            )

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
            if current_price is None:
                logger.warning(f"No price available for {symbol}")
                return
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
            logger.debug(f"No signal for {symbol}")
            return

        # Process signal
        action = signal.get("action", "FLAT")
        logger.info(f"Processing signal for {symbol}: action={action}, has_position={has_position}")
        
        if action == "BUY":
            if not has_position:
                await self._open_position(symbol, "long", current_price, signal)
            else:
                # Already have position, close if short and open long
                pos = self._state.positions[symbol]
                if pos.side == "short":
                    logger.info(f"Closing SHORT position for {symbol} due to BUY signal")
                    await self._close_position(symbol, "Signal reversal: BUY", current_price)
                else:
                    logger.debug(f"Already in LONG position for {symbol}, ignoring BUY signal")
        elif action == "SELL":
            if has_position:
                pos = self._state.positions[symbol]
                if pos.side == "long":
                    logger.info(f"Closing LONG position for {symbol} due to SELL signal")
                    await self._close_position(symbol, "Signal sell", current_price)
                else:
                    logger.debug(f"In SHORT position for {symbol}, ignoring SELL signal")
            else:
                logger.debug(f"No position to sell for {symbol}")
        elif action == "SHORT":
            if not has_position:
                await self._open_position(symbol, "short", current_price, signal)
            else:
                pos = self._state.positions[symbol]
                if pos.side == "long":
                    logger.info(f"Closing LONG position for {symbol} due to SHORT signal")
                    await self._close_position(symbol, "Signal reversal: SHORT", current_price)
                else:
                    logger.debug(f"Already in SHORT position for {symbol}, ignoring SHORT signal")
        else:
            logger.debug(f"No action for signal: {action}")

    async def _generate_signal(
        self, symbol: str, current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal."""
        if self._signal_generator:
            try:
                signal = await self._signal_generator(symbol, current_price)
                if signal:
                    logger.info(f"Signal generated for {symbol}: {signal}")
                return signal
            except Exception as e:
                logger.error(f"Signal generation error for {symbol}: {e}")

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
            price=price,  # For safety check calculations
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

        if result.average_price is None or result.filled_quantity is None:
            logger.error(f"Order incomplete: price={result.average_price}, qty={result.filled_quantity}")
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
            f"Opened {side.upper()} position: {symbol} @ ${result.average_price:.2f} "
            f"(qty: {result.filled_quantity}) | SL: ${position.stop_loss:.2f}, TP: ${position.take_profit:.2f} "
            f"| Reason: {signal.get('reason', 'unknown')}"
        )

        # Notify callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback("open", symbol, side, result)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

        # Register with trailing stop manager
        if self.trailing_stop_manager:
            self.trailing_stop_manager.add_position(
                symbol=symbol,
                entry_price=result.average_price,
                side=side,
            )
            logger.info(f"Trailing stop registered for {symbol}")

        # Register with DCA manager
        if self.dca_manager:
            self.dca_manager.add_position(
                symbol=symbol,
                entry_price=result.average_price,
                quantity=result.filled_quantity,
            )
            logger.info(f"DCA tracking registered for {symbol}")

        # Record action for learning
        self._record_action_entry(symbol, side, result.average_price, signal)

        # Store prediction ID for ML performance tracking
        prediction_id = signal.get("prediction_id")
        if prediction_id:
            self._prediction_ids[symbol] = prediction_id

        # Intelligent Brain: Explain entry and send via Telegram
        if self.intelligent_brain and INTELLIGENT_BRAIN_AVAILABLE:
            try:
                # Calculate position value
                position_value = result.filled_quantity * result.average_price
                portfolio_value = self._state.current_balance if self._state else 10000

                # Build portfolio context
                portfolio_context = {
                    "position_value": position_value,
                    "portfolio_value": portfolio_value,
                    "position_pct": (position_value / portfolio_value * 100) if portfolio_value > 0 else 0,
                }

                # Store regime for learning later
                self._last_regime = signal.get("regime", "unknown")

                explanation = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.intelligent_brain.explain_entry(
                        symbol=symbol,
                        action="BUY" if side == "long" else "SELL",
                        price=result.average_price,
                        quantity=result.filled_quantity,
                        signal=signal,
                        portfolio_context=portfolio_context,
                    )
                )
                logger.info(f"Intelligent Brain: Entry explained for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to explain entry with intelligent brain: {e}")

    async def _close_position(
        self, symbol: str, reason: str, current_price: Optional[float] = None
    ) -> None:
        """Close an existing position."""
        if not self._state or not self.execution_adapter:
            return

        if symbol not in self._state.positions:
            logger.warning(f"No position to close for {symbol}")
            return

        position = self._state.positions[symbol]
        logger.info(f"Closing {position.side.upper()} position for {symbol} - Reason: {reason}")

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
            logger.error(f"Close order failed for {symbol}: {result.error_message}")
            return

        if result.average_price is None:
            logger.error(f"Close order has no price for {symbol}")
            return

        # Calculate P&L
        if position.side == "long":
            pnl = position.quantity * (result.average_price - position.entry_price)
        else:
            pnl = position.quantity * (position.entry_price - result.average_price)

        pnl -= result.commission or 0
        pnl_pct = pnl / (position.quantity * position.entry_price) * 100
        
        logger.info(f"Position closed for {symbol}: P&L=${pnl:.2f} ({pnl_pct:.2f}%)")

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

        # Remove from trailing stop manager
        if self.trailing_stop_manager:
            self.trailing_stop_manager.remove_position(symbol)

        # Remove from DCA manager
        if self.dca_manager:
            self.dca_manager.remove_position(symbol)

        # Record action outcome for learning
        try:
            entry_time = datetime.fromisoformat(position.entry_time)
            holding_hours = (datetime.now() - entry_time).total_seconds() / 3600
        except Exception:
            holding_hours = 0.0

        await self._record_action_outcome(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=result.average_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_hours=holding_hours,
        )

        # Record ML prediction outcome for performance tracking
        if symbol in self._prediction_ids:
            try:
                from bot.ml_performance_tracker import track_outcome
                track_outcome(self._prediction_ids[symbol], pnl_pct)
                del self._prediction_ids[symbol]
            except Exception as e:
                logger.debug(f"Failed to record prediction outcome: {e}")

        # Intelligent Brain: Explain exit and learn from trade
        if self.intelligent_brain and INTELLIGENT_BRAIN_AVAILABLE:
            try:
                # Convert holding hours to minutes for the API
                hold_duration_minutes = int(holding_hours * 60)

                # Explain the exit
                explanation = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.intelligent_brain.explain_exit(
                        symbol=symbol,
                        action="SELL" if position.side == "long" else "BUY",
                        entry_price=position.entry_price,
                        exit_price=result.average_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=reason,
                        hold_duration_minutes=hold_duration_minutes,
                    )
                )

                # Learn from the trade outcome
                trade_outcome = TradeOutcome(
                    symbol=symbol,
                    action="BUY" if position.side == "long" else "SELL",
                    entry_price=position.entry_price,
                    exit_price=result.average_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_duration_minutes=hold_duration_minutes,
                    regime=getattr(self, '_last_regime', 'unknown'),
                    confidence_at_entry=getattr(position, 'signal_confidence', 0.5),
                )

                learning_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.intelligent_brain.learn_from_trade(trade_outcome)
                )

                logger.info(
                    f"Intelligent Brain: Trade learned - {symbol} "
                    f"PnL: {pnl_pct:+.2f}%, Confidence adjustment: {learning_result.confidence_adjustment:+.2f}"
                )
            except Exception as e:
                logger.warning(f"Failed to process trade with intelligent brain: {e}")

    async def _check_positions(self) -> None:
        """Check all positions for stop loss / take profit / trailing stop / DCA."""
        if not self._state or not self.execution_adapter:
            return

        for symbol, position in list(self._state.positions.items()):
            try:
                current_price = await self.execution_adapter.get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"No price for {symbol}, skipping position check")
                    continue

                # Update position current price
                position.current_price = current_price

                # Check trailing stop first (dynamic stop)
                if self.trailing_stop_manager:
                    trail_result = self.trailing_stop_manager.update(
                        symbol, current_price, position.side
                    )
                    if trail_result.get("action") == "stop_triggered":
                        await self._close_position(symbol, "Trailing stop", current_price)
                        continue
                    elif trail_result.get("action") in ("stop_raised", "moved_to_breakeven"):
                        # Update position stop loss to trailing stop level
                        new_stop = trail_result.get("new_stop")
                        if new_stop:
                            position.stop_loss = new_stop
                            self.state_store.update_position(symbol, position)
                            logger.info(f"Updated {symbol} stop to ${new_stop:.2f}")

                # Check fixed stop loss (fallback if trailing not active)
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

                # Check DCA opportunity (only for long positions in drawdown)
                if self.dca_manager and position.side == "long":
                    available_capital = self._state.current_balance * 0.1  # Max 10% per DCA
                    dca_opportunity = self.dca_manager.check_dca_opportunity(
                        symbol, current_price, available_capital
                    )
                    if dca_opportunity:
                        await self._execute_dca(symbol, dca_opportunity, current_price)

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

    async def _execute_dca(
        self, symbol: str, dca_info: Dict[str, Any], current_price: float
    ) -> None:
        """Execute a DCA order."""
        if not self._state or not self.execution_adapter or not self.dca_manager:
            return

        dca_quantity = dca_info["dca_quantity"]
        level_drawdown = dca_info["level_drawdown"]

        logger.info(
            f"DCA opportunity: {symbol} at {dca_info['drawdown_pct']*100:.1f}% drawdown, "
            f"adding {dca_quantity:.6f} @ ${current_price:.2f}"
        )

        # Create DCA order
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=dca_quantity,
            price=current_price,
            signal_reason=f"DCA at {level_drawdown*100:.0f}% drawdown",
        )

        # Execute order
        result = await self.execution_adapter.place_order(order)

        if not result.success:
            logger.error(f"DCA order failed: {result.error_message}")
            return

        if result.average_price is None or result.filled_quantity is None:
            return

        # Update DCA manager
        self.dca_manager.execute_dca(
            symbol=symbol,
            quantity=result.filled_quantity,
            price=result.average_price,
            level_drawdown=level_drawdown,
        )

        # Update position with new average
        position = self._state.positions.get(symbol)
        if position:
            # Calculate new average entry
            old_cost = position.quantity * position.entry_price
            new_cost = result.filled_quantity * result.average_price
            total_quantity = position.quantity + result.filled_quantity
            new_avg_price = (old_cost + new_cost) / total_quantity

            position.quantity = total_quantity
            position.entry_price = new_avg_price

            # Update stop loss and take profit based on new average
            position.stop_loss = new_avg_price * (1 - self.config.stop_loss_pct)
            position.take_profit = new_avg_price * (1 + self.config.take_profit_pct)

            self.state_store.update_position(symbol, position)

            # Update trailing stop with new entry
            if self.trailing_stop_manager:
                self.trailing_stop_manager.remove_position(symbol)
                self.trailing_stop_manager.add_position(symbol, new_avg_price, position.side)

            logger.info(
                f"DCA executed: {symbol} new avg=${new_avg_price:.2f}, "
                f"total qty={total_quantity:.6f}"
            )

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

    async def _evaluate_adaptive_risk(self) -> None:
        """Evaluate market conditions and adjust risk settings adaptively."""
        if not self.adaptive_risk_controller or not self._state:
            return

        try:
            # Gather market condition data
            market_regime = "unknown"
            regime_confidence = 0.5
            rsi = 50.0
            volatility = "normal"
            trend = "neutral"

            # Try to get market data from signal generator if available
            if hasattr(self, '_signal_generator') and self._signal_generator:
                # Use first symbol as market indicator
                symbol = self.config.symbols[0] if self.config.symbols else "BTC/USDT"
                try:
                    signal = await self._signal_generator(symbol, 0)
                    if signal:
                        market_regime = signal.get("regime", "unknown")
                        regime_confidence = signal.get("confidence", 0.5)
                        rsi = signal.get("rsi", 50.0)
                        volatility = signal.get("volatility", "normal")
                        trend = signal.get("trend", "neutral")
                except Exception:
                    pass

            # Calculate recent performance
            recent_performance = {
                "win_rate": self._state.win_rate,
                "total_pnl": (
                    self._state.total_pnl / self._state.initial_capital * 100
                    if self._state.initial_capital > 0
                    else 0
                ),
                "drawdown": self._state.max_drawdown_pct * 100,
            }

            # Evaluate and adjust risk settings
            current_strategy = await self.adaptive_risk_controller.evaluate_and_adjust(
                market_regime=market_regime,
                regime_confidence=regime_confidence,
                rsi=rsi,
                volatility=volatility,
                trend=trend,
                recent_performance=recent_performance,
            )

            logger.debug(
                f"Adaptive risk strategy: {current_strategy.name}, "
                f"direction: {current_strategy.expected_direction}"
            )

        except Exception as e:
            logger.warning(f"Adaptive risk evaluation failed: {e}")

    def _record_action_entry(
        self, symbol: str, side: str, price: float, signal: Dict[str, Any]
    ) -> None:
        """Record action entry for learning."""
        # Record to optimal action tracker
        if self.action_tracker and SELF_LEARNING_AVAILABLE:
            try:
                # Build market state from signal data
                state = MarketState(
                    regime=signal.get("regime", "unknown"),
                    regime_confidence=signal.get("confidence", 0.5),
                    trend_direction=signal.get("trend", "neutral"),
                    rsi=signal.get("rsi", 50.0),
                    volatility_regime=signal.get("volatility", "normal"),
                )

                # Build action outcome (entry only)
                action_type = ActionType.BUY if side == "long" else ActionType.SELL
                outcome = ActionOutcome(
                    action=action_type,
                    entry_price=price,
                )

                # Create record
                record = StateActionRecord(
                    symbol=symbol,
                    state=state,
                    outcome=outcome,
                    model_prediction=signal.get("action", ""),
                    model_confidence=signal.get("confidence", 0.0),
                    strategy_used=signal.get("strategy", ""),
                    signal_reason=signal.get("reason", ""),
                )

                # Record and save ID for later outcome update
                record_id = self.action_tracker.record_action(record)
                self._action_record_ids[symbol] = record_id

                logger.debug(f"Recorded action entry for {symbol}: {action_type.value}")

            except Exception as e:
                logger.warning(f"Failed to record action entry: {e}")

        # Create AI Brain snapshot for learning
        if self.ai_brain and AI_BRAIN_AVAILABLE:
            try:
                # Map regime to MarketCondition
                regime_map = {
                    "strong_bull": MarketCondition.STRONG_BULL,
                    "bull": MarketCondition.BULL,
                    "weak_bull": MarketCondition.WEAK_BULL,
                    "sideways": MarketCondition.SIDEWAYS,
                    "weak_bear": MarketCondition.WEAK_BEAR,
                    "bear": MarketCondition.BEAR,
                    "strong_bear": MarketCondition.STRONG_BEAR,
                    "crash": MarketCondition.CRASH,
                    "volatile": MarketCondition.VOLATILE,
                }
                regime = signal.get("regime", "sideways")
                condition = regime_map.get(regime, MarketCondition.SIDEWAYS)

                # Map volatility
                vol_str = signal.get("volatility", "normal")
                vol_percentile = {"low": 25, "normal": 50, "high": 75, "extreme": 95}.get(vol_str, 50)

                # Create snapshot for later outcome recording
                snapshot = MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    price=price,
                    trend_1h=signal.get("trend", "neutral"),
                    rsi=signal.get("rsi", 50.0),
                    volatility_percentile=vol_percentile,
                    condition=condition,
                    confidence=signal.get("confidence", 0.5)
                )

                self._trade_snapshots[symbol] = snapshot
                logger.debug(f"AI Brain: Snapshot created for {symbol} entry")

            except Exception as e:
                logger.warning(f"Failed to create AI brain snapshot: {e}")

    async def _record_action_outcome(
        self, symbol: str, entry_price: float, exit_price: float,
        pnl: float, pnl_pct: float, holding_hours: float
    ) -> None:
        """Record action outcome for learning."""
        # Record to optimal action tracker
        if self.action_tracker and SELF_LEARNING_AVAILABLE:
            try:
                record_id = self._action_record_ids.pop(symbol, None)
                if record_id:
                    outcome = ActionOutcome(
                        action=ActionType.CLOSE,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        holding_time_hours=holding_hours,
                    )

                    self.action_tracker.record_outcome(record_id, outcome)
                    logger.debug(f"Recorded action outcome for {symbol}: PnL={pnl_pct:.2f}%")
            except Exception as e:
                logger.warning(f"Failed to record action outcome: {e}")

        # Record to AI Trading Brain for learning
        if self.ai_brain and AI_BRAIN_AVAILABLE:
            try:
                entry_snapshot = self._trade_snapshots.pop(symbol, None)
                if entry_snapshot:
                    # Determine action from position side
                    position = self._state.positions.get(symbol) if self._state else None
                    action = "buy" if position and position.side == "long" else "sell"

                    result = self.ai_brain.record_trade_result(
                        trade_id=f"{symbol}_{int(datetime.now().timestamp())}",
                        symbol=symbol,
                        entry_snapshot=entry_snapshot,
                        action=action,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position.quantity if position else 1.0,
                        price_history=[],  # We don't track price history currently
                        holding_hours=holding_hours
                    )

                    # Update daily tracker
                    daily_status = self.ai_brain.daily_tracker.get_status()
                    logger.info(
                        f"AI Brain: Trade recorded - PnL {pnl_pct:.2f}%, "
                        f"Daily progress: {daily_status.get('current', '0%')}/{daily_status.get('target', '1%')}"
                    )

                    # Check for auto-pause conditions
                    should_pause, pause_reason = self.ai_brain.daily_tracker.should_auto_pause()
                    if should_pause:
                        logger.warning(f"AUTO-PAUSE TRIGGERED: {pause_reason}")
                        await self._trigger_auto_pause(pause_reason)
            except Exception as e:
                logger.warning(f"Failed to record to AI brain: {e}")

    async def _trigger_auto_pause(self, reason: str):
        """Trigger auto-pause when loss limit or target conditions met."""
        if self._state:
            self._state.status = TradingStatus.PAUSED
            logger.warning(f"Trading auto-paused: {reason}")

            # Send Telegram alert
            try:
                from bot.trade_alerts import create_trade_alert_manager
                alerts = create_trade_alert_manager()
                if alerts.is_configured():
                    daily_pnl = self._state.daily_pnl / self._state.initial_capital * 100 if self._state.initial_capital > 0 else 0
                    alerts.send_auto_pause_alert(
                        reason=reason,
                        current_pnl=daily_pnl,
                        recommendation="Review your trades and wait for next trading day."
                    )
            except Exception as e:
                logger.warning(f"Could not send auto-pause alert: {e}")

            # Notify callbacks
            for callback in self._on_trade_callbacks:
                try:
                    callback("auto_pause", "all", reason, None)
                except Exception as e:
                    logger.error(f"Auto-pause callback error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status for dashboard."""
        if not self._state:
            return {"error": "Engine not initialized"}

        safety_status = (
            self.safety_controller.get_status() if self.safety_controller else {}
        )

        # AI Brain status
        ai_brain_status = None
        if self.ai_brain and AI_BRAIN_AVAILABLE:
            try:
                daily_target = self.ai_brain.daily_tracker.get_status()
                ai_brain_status = {
                    "daily_target": daily_target.get("target", "1%"),
                    "daily_progress": daily_target.get("current", "0%"),
                    "target_achieved": daily_target.get("target_achieved", False),
                    "can_trade": daily_target.get("can_still_trade", True),
                    "recommendation": daily_target.get("recommendation", ""),
                    "patterns_learned": len(self.ai_brain.pattern_learner.profitable_patterns) + len(self.ai_brain.pattern_learner.losing_patterns),
                }
            except Exception:
                ai_brain_status = {"error": "Could not get AI Brain status"}

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
            "ai_brain": ai_brain_status,
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
