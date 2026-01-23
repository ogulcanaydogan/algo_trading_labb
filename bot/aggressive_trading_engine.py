"""
Aggressive Trading Engine - High-Frequency Profit Generation.

Integrates:
- Aggressive ML Predictor for high-confidence signals
- Profit Optimizer for optimal entry/exit timing
- Dynamic leverage based on model confidence
- Real-time learning from trade outcomes

Target: 1%+ daily returns through frequent trading.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from .ml import (
    AggressiveProfitHunter,
    AggressiveSignal,
    AggressiveConfig,
    SignalStrength,
    ProfitOptimizer,
    TradeState,
    create_aggressive_predictor,
)
from .regime import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
)
from .risk import (
    create_correlation_circuit_breaker,
    CorrelationCircuitBreaker,
    CircuitBreakerState,
)
from .core import get_logger, metrics

logger = get_logger(__name__)


class ExecutionAdapter(Protocol):
    """Protocol for execution adapters."""

    async def get_balance(self) -> Dict[str, float]: ...
    async def get_positions(self) -> Dict[str, Dict]: ...
    async def get_price(self, symbol: str) -> float: ...
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...
    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market", leverage: int = 1
    ) -> Dict: ...


@dataclass
class AggressiveTradingConfig:
    """Configuration for aggressive trading engine."""

    # Trading basics
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    lookback_bars: int = 100
    update_interval_seconds: int = 60  # Faster updates for frequent trading

    # Aggressive predictor settings
    max_leverage: float = 5.0
    enable_learning: bool = True
    min_confidence_to_trade: float = 0.55

    # Position sizing
    base_position_pct: float = 0.05  # 5% per trade
    max_position_pct: float = 0.25  # 25% max

    # Risk management
    max_daily_loss_pct: float = 0.05  # Stop trading at 5% daily loss
    max_daily_trades: int = 50  # Allow frequent trading
    max_consecutive_losses: int = 5

    # Target performance
    daily_profit_target_pct: float = 0.01  # 1% daily target

    # State persistence
    state_file: Path = Path("data/aggressive_trading_state.json")
    model_dir: Path = Path("data/models")


@dataclass
class OpenPosition:
    """Tracks an open position."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    leverage: float
    entry_time: datetime
    entry_confidence: float
    entry_signal_strength: SignalStrength
    stop_loss_pct: float
    take_profit_pct: float
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    bars_held: int = 0


@dataclass
class TradingStats:
    """Daily trading statistics."""

    date: str
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    consecutive_losses: int = 0

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return self.winning_trades / self.trades_count


@dataclass
class AggressiveTradingState:
    """Current state of aggressive trading engine."""

    is_running: bool = False

    # Account
    equity: float = 10000.0
    starting_equity: float = 10000.0
    peak_equity: float = 10000.0

    # Positions
    open_positions: Dict[str, OpenPosition] = field(default_factory=dict)

    # Performance
    daily_stats: TradingStats = field(
        default_factory=lambda: TradingStats(date=datetime.now().strftime("%Y-%m-%d"))
    )

    # Market state
    current_regime: Optional[str] = None

    # Timing
    last_update: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "is_running": self.is_running,
            "equity": self.equity,
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl_pct": self.daily_stats.total_pnl_pct,
            "trades_today": self.daily_stats.trades_count,
            "win_rate": self.daily_stats.win_rate,
            "open_positions": len(self.open_positions),
            "current_regime": self.current_regime,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


class AggressiveTradingEngine:
    """
    Aggressive Trading Engine for high-frequency profit generation.

    Features:
    - Uses AggressiveProfitHunter for ML predictions
    - ProfitOptimizer for entry/exit timing
    - Dynamic leverage based on confidence
    - Real-time learning from trade outcomes
    """

    def __init__(
        self,
        config: Optional[AggressiveTradingConfig] = None,
        execution: Optional[ExecutionAdapter] = None,
    ):
        self.config = config or AggressiveTradingConfig()
        self.execution = execution

        # Initialize state
        self.state = AggressiveTradingState()

        # Initialize ML components
        self._init_ml_components()

        # Initialize risk components
        self.circuit_breaker = create_correlation_circuit_breaker()
        self.regime_detectors: Dict[str, RegimeDetector] = {}

        # Callbacks
        self._on_trade: Optional[Callable] = None
        self._on_signal: Optional[Callable] = None

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Load state
        self._load_state()

    def _init_ml_components(self) -> None:
        """Initialize ML components."""
        # Create aggressive predictor per symbol
        self.predictors: Dict[str, AggressiveProfitHunter] = {}
        self.profit_optimizers: Dict[str, ProfitOptimizer] = {}

        for symbol in self.config.symbols:
            predictor = create_aggressive_predictor(
                max_leverage=self.config.max_leverage,
                enable_learning=self.config.enable_learning,
                model_dir=str(self.config.model_dir),
            )

            # Try to load pre-trained model
            safe_symbol = symbol.replace("/", "_")
            if predictor.load(f"aggressive_{safe_symbol}"):
                logger.info(f"Loaded pre-trained model for {symbol}")
            else:
                logger.warning(f"No pre-trained model for {symbol} - will need training")

            self.predictors[symbol] = predictor

            # Create profit optimizer
            optimizer = ProfitOptimizer(model_dir=str(self.config.model_dir))
            optimizer.load("profit_optimizer")
            self.profit_optimizers[symbol] = optimizer

    def set_execution(self, execution: ExecutionAdapter) -> None:
        """Set the execution adapter."""
        self.execution = execution

    def on_trade(self, callback: Callable) -> None:
        """Set callback for trade events."""
        self._on_trade = callback

    def on_signal(self, callback: Callable) -> None:
        """Set callback for signal events."""
        self._on_signal = callback

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            return

        self._running = True
        self.state.is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Aggressive trading engine started")

    async def stop(self) -> None:
        """Stop the trading engine."""
        self._running = False
        self.state.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("Aggressive trading engine stopped")

    async def _run_loop(self) -> None:
        """Main trading loop."""
        logger.info("Aggressive trading loop started")

        while self._running:
            try:
                # Reset daily stats if new day
                self._check_daily_reset()

                # Check if we should trade
                if self._should_stop_trading():
                    logger.info("Trading paused due to risk limits")
                    await asyncio.sleep(60)
                    continue

                # Run update cycle
                await self._update_cycle()

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)

            await asyncio.sleep(self.config.update_interval_seconds)

        logger.info("Aggressive trading loop stopped")

    def _check_daily_reset(self) -> None:
        """Reset stats at start of new trading day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.daily_stats.date != today:
            logger.info(f"New trading day: {today}")
            self.state.daily_stats = TradingStats(date=today)
            self.state.starting_equity = self.state.equity

    def _should_stop_trading(self) -> bool:
        """Check if we should stop trading due to risk limits."""
        stats = self.state.daily_stats

        # Check daily loss limit
        if stats.total_pnl_pct <= -self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit hit: {stats.total_pnl_pct:.2%}")
            return True

        # Check max trades
        if stats.trades_count >= self.config.max_daily_trades:
            return True

        # Check consecutive losses
        if stats.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"Consecutive loss limit hit: {stats.consecutive_losses}")
            return True

        return False

    async def _update_cycle(self) -> None:
        """Single update cycle."""
        if not self.execution:
            logger.warning("No execution adapter configured")
            return

        # Get account state
        try:
            balance = await self.execution.get_balance()
            positions = await self.execution.get_positions()
            self.state.equity = balance.get("total", self.state.equity)
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            return

        # Process each symbol
        for symbol in self.config.symbols:
            try:
                await self._process_symbol(symbol, positions)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

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

        # Get current price
        current_price = await self.execution.get_price(symbol)

        # Detect regime
        detector = self.regime_detectors.get(symbol)
        if not detector:
            detector = RegimeDetector()
            self.regime_detectors[symbol] = detector

        try:
            regime_state = detector.detect(df, symbol, self.config.timeframe)
            self.state.current_regime = regime_state.regime.value
        except Exception:
            self.state.current_regime = "unknown"

        # Check circuit breaker
        returns = df["close"].pct_change().dropna()
        if len(returns) > 20:
            returns_df = pd.DataFrame({symbol: returns})
            self.circuit_breaker.update_correlations(returns_df)

        if self.circuit_breaker.state == CircuitBreakerState.TRIGGERED:
            logger.warning("Circuit breaker triggered - skipping")
            return

        # Check existing position
        if symbol in self.state.open_positions:
            await self._manage_open_position(symbol, df, current_price)
        else:
            await self._evaluate_new_trade(symbol, df, current_price)

    async def _evaluate_new_trade(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> None:
        """Evaluate potential new trade."""
        predictor = self.predictors.get(symbol)
        if not predictor or not predictor.is_trained:
            logger.debug(f"Predictor not ready for {symbol}")
            return

        # Get signal
        try:
            signal = predictor.predict(df)
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return

        # Notify signal callback
        if self._on_signal:
            self._on_signal(symbol, signal)

        # Check if signal is actionable
        if signal.action == "FLAT":
            return

        if signal.confidence < self.config.min_confidence_to_trade:
            logger.debug(f"Signal confidence too low: {signal.confidence:.2%}")
            return

        # Optimize entry
        optimizer = self.profit_optimizers.get(symbol)
        if optimizer:
            entry_signal = optimizer.optimize_entry(df, signal.action, current_price)
            if not entry_signal.should_enter:
                logger.debug(f"Entry optimizer suggests waiting: {entry_signal.reason}")
                return

        # Calculate position size
        position_pct = min(signal.position_size_pct, self.config.max_position_pct)
        position_value = self.state.equity * position_pct
        quantity = position_value / current_price

        # Apply leverage
        leverage = min(signal.recommended_leverage, self.config.max_leverage)
        effective_quantity = quantity * leverage

        # Get optimal levels
        if optimizer:
            levels = optimizer.get_optimal_levels(
                current_price, signal.action, df, signal.confidence
            )
            stop_loss_pct = levels.stop_loss_pct
            take_profit_pct = levels.take_profit_pct
        else:
            stop_loss_pct = signal.stop_loss_pct
            take_profit_pct = signal.take_profit_pct

        # Execute trade
        logger.info(
            f"Opening {signal.action} on {symbol}: "
            f"qty={effective_quantity:.6f}, leverage={leverage:.1f}x, "
            f"confidence={signal.confidence:.2%}, strength={signal.strength.name}"
        )

        try:
            order = await self.execution.place_order(
                symbol=symbol,
                side="buy" if signal.action == "LONG" else "sell",
                quantity=effective_quantity,
                leverage=int(leverage),
            )

            if order.get("status") == "filled" or order.get("id"):
                # Record position
                self.state.open_positions[symbol] = OpenPosition(
                    symbol=symbol,
                    direction=signal.action,
                    entry_price=current_price,
                    quantity=effective_quantity,
                    leverage=leverage,
                    entry_time=datetime.now(),
                    entry_confidence=signal.confidence,
                    entry_signal_strength=signal.strength,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                )

                self.state.last_trade_time = datetime.now()

                if self._on_trade:
                    self._on_trade("OPEN", symbol, signal.action, effective_quantity, current_price)

                logger.info(f"Position opened: {symbol} {signal.action}")

        except Exception as e:
            logger.error(f"Order execution failed: {e}")

    async def _manage_open_position(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
    ) -> None:
        """Manage an existing open position."""
        position = self.state.open_positions[symbol]
        position.bars_held += 1

        # Calculate unrealized PnL
        if position.direction == "LONG":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price

        # Apply leverage
        pnl_pct *= position.leverage

        # Update max favorable/adverse
        position.max_favorable_pct = max(position.max_favorable_pct, pnl_pct)
        position.max_adverse_pct = min(position.max_adverse_pct, pnl_pct)

        # Check exit conditions
        should_exit = False
        exit_reason = ""

        # Stop loss
        if pnl_pct <= -position.stop_loss_pct:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Take profit
        elif pnl_pct >= position.take_profit_pct:
            should_exit = True
            exit_reason = "TAKE_PROFIT"

        # Trailing stop
        elif position.max_favorable_pct > 0.01:  # Had 1%+ profit
            trailing_stop = position.max_favorable_pct - 0.015  # 1.5% trailing
            if pnl_pct < trailing_stop:
                should_exit = True
                exit_reason = "TRAILING_STOP"

        # Check profit optimizer
        if not should_exit:
            optimizer = self.profit_optimizers.get(symbol)
            if optimizer:
                trade_state = TradeState(
                    direction=position.direction,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    bars_held=position.bars_held,
                    max_favorable=position.max_favorable_pct,
                    max_adverse=position.max_adverse_pct,
                    unrealized_pnl_pct=pnl_pct,
                )
                exit_signal = optimizer.optimize_exit(trade_state, df)
                if exit_signal.should_exit:
                    should_exit = True
                    exit_reason = exit_signal.exit_type

        # Check for signal reversal
        if not should_exit:
            predictor = self.predictors.get(symbol)
            if predictor and predictor.is_trained:
                try:
                    signal = predictor.predict(df)
                    # Close if strong opposite signal
                    if signal.action != position.direction and signal.confidence >= 0.65:
                        should_exit = True
                        exit_reason = "SIGNAL_REVERSAL"
                except Exception:
                    pass

        # Execute exit
        if should_exit:
            await self._close_position(symbol, position, current_price, pnl_pct, exit_reason)

    async def _close_position(
        self,
        symbol: str,
        position: OpenPosition,
        current_price: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Close a position and record outcome."""
        logger.info(f"Closing {position.direction} on {symbol}: pnl={pnl_pct:.2%}, reason={reason}")

        try:
            # Close the position
            close_side = "sell" if position.direction == "LONG" else "buy"
            order = await self.execution.place_order(
                symbol=symbol,
                side=close_side,
                quantity=position.quantity,
            )

            if order.get("status") == "filled" or order.get("id"):
                # Calculate actual PnL
                pnl_usd = (
                    self.state.equity
                    * (position.quantity / position.quantity)
                    * pnl_pct
                    / position.leverage
                )

                # Update stats
                self.state.daily_stats.trades_count += 1
                self.state.daily_stats.total_pnl_pct += (
                    pnl_pct / position.leverage
                )  # De-leverage for stats
                self.state.daily_stats.total_pnl_usd += pnl_usd

                if pnl_pct > 0:
                    self.state.daily_stats.winning_trades += 1
                    self.state.daily_stats.consecutive_losses = 0
                    self.state.daily_stats.best_trade_pct = max(
                        self.state.daily_stats.best_trade_pct, pnl_pct
                    )
                else:
                    self.state.daily_stats.losing_trades += 1
                    self.state.daily_stats.consecutive_losses += 1
                    self.state.daily_stats.worst_trade_pct = min(
                        self.state.daily_stats.worst_trade_pct, pnl_pct
                    )

                # Update equity
                self.state.equity += pnl_usd

                # Record for ML learning
                predictor = self.predictors.get(symbol)
                if predictor:
                    features = []  # Would need actual features
                    predictor.record_trade_outcome(
                        predicted_action=position.direction,
                        actual_pnl_pct=pnl_pct,
                        confidence=position.entry_confidence,
                        features=np.array(features) if features else np.array([]),
                        regime=self.state.current_regime or "unknown",
                        holding_period=position.bars_held,
                    )

                # Record for profit optimizer
                optimizer = self.profit_optimizers.get(symbol)
                if optimizer:
                    optimizer.record_trade(
                        entry_quality=position.entry_confidence,
                        direction=position.direction,
                        pnl_pct=pnl_pct,
                        max_favorable_pct=position.max_favorable_pct,
                        max_adverse_pct=position.max_adverse_pct,
                        bars_held=position.bars_held,
                        exit_type=reason,
                    )

                # Remove position
                del self.state.open_positions[symbol]

                if self._on_trade:
                    self._on_trade(
                        "CLOSE",
                        symbol,
                        position.direction,
                        position.quantity,
                        current_price,
                        pnl_pct,
                    )

                logger.info(
                    f"Position closed: {symbol} {position.direction}, "
                    f"PnL: {pnl_pct:.2%} (${pnl_usd:.2f})"
                )

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                "equity": self.state.equity,
                "starting_equity": self.state.starting_equity,
                "peak_equity": self.state.peak_equity,
                "daily_stats": {
                    "date": self.state.daily_stats.date,
                    "trades_count": self.state.daily_stats.trades_count,
                    "winning_trades": self.state.daily_stats.winning_trades,
                    "losing_trades": self.state.daily_stats.losing_trades,
                    "total_pnl_pct": self.state.daily_stats.total_pnl_pct,
                },
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.config.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            # Also save ML models
            for symbol, predictor in self.predictors.items():
                safe_symbol = symbol.replace("/", "_")
                predictor.save(f"aggressive_{safe_symbol}")

            for symbol, optimizer in self.profit_optimizers.items():
                optimizer.save("profit_optimizer")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            if self.config.state_file.exists():
                with open(self.config.state_file) as f:
                    data = json.load(f)

                self.state.equity = data.get("equity", 10000.0)
                self.state.starting_equity = data.get("starting_equity", 10000.0)
                self.state.peak_equity = data.get("peak_equity", 10000.0)

                logger.info(f"State loaded: equity=${self.state.equity:.2f}")

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "is_running": self.state.is_running,
            "equity": self.state.equity,
            "daily_pnl_pct": f"{self.state.daily_stats.total_pnl_pct:.2%}",
            "trades_today": self.state.daily_stats.trades_count,
            "win_rate": f"{self.state.daily_stats.win_rate:.2%}",
            "open_positions": len(self.state.open_positions),
            "current_regime": self.state.current_regime,
            "consecutive_losses": self.state.daily_stats.consecutive_losses,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
        }

    def get_performance_report(self) -> Dict:
        """Get detailed performance report."""
        stats = self.state.daily_stats

        return {
            "date": stats.date,
            "starting_equity": self.state.starting_equity,
            "current_equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "total_return_pct": (self.state.equity - self.state.starting_equity)
            / self.state.starting_equity,
            "daily_pnl_pct": stats.total_pnl_pct,
            "daily_pnl_usd": stats.total_pnl_usd,
            "trades_count": stats.trades_count,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate": stats.win_rate,
            "best_trade_pct": stats.best_trade_pct,
            "worst_trade_pct": stats.worst_trade_pct,
            "max_drawdown_pct": (self.state.peak_equity - self.state.equity)
            / self.state.peak_equity,
            "target_progress": stats.total_pnl_pct / self.config.daily_profit_target_pct
            if self.config.daily_profit_target_pct > 0
            else 0,
        }


def create_aggressive_engine(
    symbols: Optional[List[str]] = None,
    max_leverage: float = 5.0,
    daily_profit_target: float = 0.01,
    execution: Optional[ExecutionAdapter] = None,
) -> AggressiveTradingEngine:
    """Factory function to create aggressive trading engine."""
    config = AggressiveTradingConfig(
        symbols=symbols or ["BTC/USDT"],
        max_leverage=max_leverage,
        daily_profit_target_pct=daily_profit_target,
    )
    return AggressiveTradingEngine(config=config, execution=execution)
