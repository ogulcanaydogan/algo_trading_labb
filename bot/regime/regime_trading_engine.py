"""
Regime-Based Trading Engine.

The main orchestrator that combines:
- Regime detection
- Position sizing
- Risk management
- Order execution

This is the entry point for regime-based trading.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd

from .regime_detector import MarketRegime, RegimeConfig, RegimeDetector, RegimeState
from .regime_position_manager import (
    PositionAction,
    PositionRecommendation,
    PositionSizingConfig,
    RegimePositionManager,
)
from .regime_risk_engine import (
    PortfolioState,
    RegimeRiskEngine,
    RiskConfig,
    TradeRequest,
)

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode."""

    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class ExecutionAdapter(Protocol):
    """Protocol for execution adapters."""

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        ...

    async def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        ...

    async def get_price(self, symbol: str) -> float:
        """Get current price."""
        ...

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get OHLCV data."""
        ...

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
    ) -> Dict:
        """Place an order."""
        ...


@dataclass
class TradingConfig:
    """Configuration for the trading engine."""

    # Trading mode
    mode: TradingMode = TradingMode.PAPER

    # Assets to trade
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])

    # Timeframe for analysis
    timeframe: str = "1h"
    lookback_bars: int = 200

    # Update interval
    update_interval_seconds: int = 300  # 5 minutes

    # Position sizing
    position_config: PositionSizingConfig = field(default_factory=PositionSizingConfig)

    # Risk management
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Regime detection
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)

    # Safety
    max_daily_trades: int = 20
    require_confirmation: bool = False  # Require manual confirmation for live

    # Logging
    log_file: Optional[Path] = None
    state_file: Path = Path("data/regime_trading_state.json")


@dataclass
class TradingState:
    """Current state of the trading engine."""

    is_running: bool = False
    mode: TradingMode = TradingMode.PAPER

    # Regime state
    current_regime: Optional[MarketRegime] = None
    regime_confidence: float = 0.0
    regime_duration_hours: float = 0.0

    # Position state
    current_allocation: float = 0.0
    target_allocation: float = 0.0
    position_value: float = 0.0
    equity: float = 0.0

    # Performance
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0

    # Timing
    last_update: Optional[datetime] = None
    last_trade: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "is_running": self.is_running,
            "mode": self.mode.value,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "regime_confidence": self.regime_confidence,
            "regime_duration_hours": self.regime_duration_hours,
            "current_allocation": self.current_allocation,
            "target_allocation": self.target_allocation,
            "position_value": self.position_value,
            "equity": self.equity,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "trades_today": self.trades_today,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
        }


@dataclass
class TradeRecord:
    """Record of an executed trade."""

    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    value: float
    regime: MarketRegime
    reason: str
    order_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "regime": self.regime.value,
            "reason": self.reason,
            "order_id": self.order_id,
        }


class RegimeTradingEngine:
    """
    Main trading engine that orchestrates regime-based trading.

    Usage:
        engine = RegimeTradingEngine(config, execution_adapter)
        await engine.start()
        # ... engine runs automatically ...
        await engine.stop()
    """

    def __init__(
        self,
        config: TradingConfig,
        execution_adapter: Optional[ExecutionAdapter] = None,
        enable_notifications: bool = True,
    ):
        self.config = config
        self.execution = execution_adapter

        # Initialize components - separate detector per symbol to prevent cross-contamination
        self.regime_detectors: Dict[str, RegimeDetector] = {
            symbol: RegimeDetector(config.regime_config) for symbol in config.symbols
        }
        self.position_manager = RegimePositionManager(config.position_config)
        self.risk_engine = RegimeRiskEngine(
            config.risk_config,
            state_path=Path("data/regime_risk_state.json"),
        )

        # Notifications
        self._notifications_enabled = enable_notifications
        self._notifier = None
        if enable_notifications:
            try:
                from bot.notifications import NotificationManager

                self._notifier = NotificationManager()
                if self._notifier.has_channels():
                    logger.info(
                        f"Notifications enabled: {self._notifier.get_configured_channels()}"
                    )
                else:
                    logger.info("No notification channels configured")
            except ImportError:
                logger.warning("Notification module not available")

        # State
        self.state = TradingState(mode=config.mode)
        self._trade_history: List[TradeRecord] = []
        self._regime_history: List[Dict] = []
        self._last_regimes: Dict[str, MarketRegime] = {}  # Track regime per symbol

        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_regime_change: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Load state
        self._load_state()

    def set_execution_adapter(self, adapter: ExecutionAdapter) -> None:
        """Set or replace the execution adapter."""
        self.execution = adapter

    def on_regime_change(self, callback: Callable[[MarketRegime, MarketRegime], None]) -> None:
        """Set callback for regime changes."""
        self._on_regime_change = callback

    def on_trade(self, callback: Callable[[TradeRecord], None]) -> None:
        """Set callback for trades."""
        self._on_trade = callback

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.warning("Trading engine already running")
            return

        logger.info(f"Starting regime trading engine in {self.config.mode.value} mode")
        logger.info(f"Trading symbols: {self.config.symbols}")

        self._running = True
        self.state.is_running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the trading engine."""
        if not self._running:
            return

        logger.info("Stopping regime trading engine")
        self._running = False
        self.state.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._save_state()

    async def _run_loop(self) -> None:
        """Main trading loop."""
        logger.info("Trading loop started")

        while self._running:
            try:
                await self._update_cycle()
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                if self._on_error:
                    self._on_error(e)

            # Wait for next update
            await asyncio.sleep(self.config.update_interval_seconds)

        logger.info("Trading loop stopped")

    async def _update_cycle(self) -> None:
        """Single update cycle."""

        if not self.execution:
            logger.warning("No execution adapter configured")
            return

        # Get market data
        for symbol in self.config.symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Update state
        self.state.last_update = datetime.now()
        self._save_state()

    async def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol."""

        # Get OHLCV data
        df = await self.execution.get_ohlcv(
            symbol,
            self.config.timeframe,
            self.config.lookback_bars,
        )

        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return

        # Detect regime using per-symbol detector
        detector = self.regime_detectors.get(symbol)
        if not detector:
            # Create detector on-the-fly if symbol was added dynamically
            detector = RegimeDetector(self.config.regime_config)
            self.regime_detectors[symbol] = detector

        regime_state = detector.detect(df, symbol, self.config.timeframe)

        # Check for regime change (per-symbol tracking)
        old_regime = self._last_regimes.get(symbol)
        if old_regime != regime_state.regime:
            logger.info(
                f"Regime change: {old_regime} -> {regime_state.regime} "
                f"(confidence: {regime_state.confidence:.1%})"
            )
            if self._on_regime_change:
                self._on_regime_change(old_regime, regime_state.regime)

            # Send notification
            self._notify_regime_change(
                symbol=symbol,
                old_regime=old_regime,
                new_regime=regime_state.regime,
                confidence=regime_state.confidence,
            )

            self._regime_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "from": old_regime.value if old_regime else None,
                    "to": regime_state.regime.value,
                    "confidence": regime_state.confidence,
                }
            )

        # Update state (per-symbol and global)
        self._last_regimes[symbol] = regime_state.regime
        self.state.current_regime = regime_state.regime  # Global state (last updated)
        self.state.regime_confidence = regime_state.confidence

        # Get account state
        balance = await self.execution.get_balance()
        positions = await self.execution.get_positions()
        current_price = await self.execution.get_price(symbol)

        equity = balance.get("total", 0)
        position_value = 0

        if symbol in positions:
            pos = positions[symbol]
            quantity = pos.get("quantity", 0)
            position_value = quantity * current_price

        self.state.equity = equity
        self.state.position_value = position_value

        # Update risk engine
        portfolio = PortfolioState(
            equity=equity,
            available_balance=balance.get("available", equity),
            peak_equity=max(equity, getattr(self, "_peak_equity", equity)),
        )
        self.risk_engine.update_portfolio(portfolio)
        self.risk_engine.update_regime(regime_state)
        self._peak_equity = portfolio.peak_equity

        # Get position recommendation
        recommendation = self.position_manager.get_recommendation(
            regime_state,
            equity,
            position_value,
        )

        self.state.current_allocation = recommendation.current_allocation
        self.state.target_allocation = recommendation.target_allocation

        # Execute if needed
        if recommendation.should_execute:
            await self._execute_recommendation(symbol, recommendation, current_price, equity)

    async def _execute_recommendation(
        self,
        symbol: str,
        recommendation: PositionRecommendation,
        current_price: float,
        equity: float,
    ) -> None:
        """Execute a position recommendation."""

        # Check daily trade limit
        if self.state.trades_today >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return

        # Calculate trade
        target_value = equity * recommendation.target_allocation
        current_value = equity * recommendation.current_allocation
        trade_value = target_value - current_value

        if abs(trade_value) < 10:  # Minimum trade size
            return

        # Determine side
        side = "buy" if trade_value > 0 else "sell"
        quantity = abs(trade_value) / current_price

        # Check with risk engine (use 3% stop loss to stay within limits)
        trade_request = TradeRequest(
            symbol=symbol,
            direction="long" if side == "buy" else "short",
            entry_price=current_price,
            stop_loss=current_price * (0.97 if side == "buy" else 1.03),
            signal_confidence=recommendation.confidence,
        )

        risk_check = self.risk_engine.check_trade(trade_request)

        if not risk_check.is_approved and side == "buy":
            logger.warning(f"Trade blocked by risk engine: {risk_check.block_reasons}")
            return

        # Execute based on mode
        if self.config.mode == TradingMode.PAPER:
            order_result = await self._paper_execute(symbol, side, quantity, current_price)
        elif self.config.mode == TradingMode.LIVE:
            if self.config.require_confirmation:
                logger.info(f"LIVE TRADE PENDING CONFIRMATION: {side} {quantity:.4f} {symbol}")
                return
            order_result = await self.execution.place_order(symbol, side, quantity)
        else:
            return

        # Record trade
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            value=trade_value,
            regime=recommendation.regime,
            reason=recommendation.reason,
            order_id=order_result.get("order_id"),
        )

        self._trade_history.append(trade)
        self.state.trades_today += 1
        self.state.last_trade = datetime.now()

        # Update position manager
        self.position_manager.record_rebalance(recommendation)

        logger.info(
            f"Trade executed: {side.upper()} {quantity:.4f} {symbol} @ {current_price:.2f} "
            f"(value: ${abs(trade_value):.2f}) | Regime: {recommendation.regime.value}"
        )

        # Send trade notification
        self._notify_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            regime=recommendation.regime,
        )

        if self._on_trade:
            self._on_trade(trade)

    async def _paper_execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Dict:
        """Execute paper trade."""
        logger.info(f"[PAPER] {side.upper()} {quantity:.4f} {symbol} @ {price:.2f}")
        return {"order_id": f"paper_{datetime.now().timestamp()}", "status": "filled"}

    def get_status(self) -> Dict:
        """Get current engine status."""
        # Get regime stats per symbol
        regime_stats = {
            symbol: detector.get_regime_stats()
            for symbol, detector in self.regime_detectors.items()
        }
        return {
            "state": self.state.to_dict(),
            "regime_detectors": regime_stats,
            "current_regimes": {k: v.value for k, v in self._last_regimes.items()},
            "position_manager": self.position_manager.get_status(),
            "risk_engine": {
                "kill_switch_active": self.risk_engine._kill_switch_active,
            },
            "trade_count": len(self._trade_history),
            "config": {
                "mode": self.config.mode.value,
                "symbols": self.config.symbols,
                "timeframe": self.config.timeframe,
            },
        }

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        return [t.to_dict() for t in self._trade_history[-limit:]]

    def get_regime_history(self, limit: int = 100) -> List[Dict]:
        """Get recent regime history."""
        return self._regime_history[-limit:]

    # =========================================================================
    # Notification Methods
    # =========================================================================

    def _notify_regime_change(
        self,
        symbol: str,
        old_regime: Optional[MarketRegime],
        new_regime: MarketRegime,
        confidence: float,
    ) -> None:
        """Send notification for regime change."""
        if not self._notifier or not self._notifier.has_channels():
            return

        # Get regime emoji
        regime_emoji = {
            MarketRegime.BULL: "🐂",
            MarketRegime.BEAR: "🐻",
            MarketRegime.CRASH: "💥",
            MarketRegime.SIDEWAYS: "↔️",
            MarketRegime.HIGH_VOL: "🌊",
            MarketRegime.UNKNOWN: "❓",
        }

        emoji = regime_emoji.get(new_regime, "📊")
        old_name = old_regime.value if old_regime else "None"

        try:
            from bot.notifications import Alert, AlertType, AlertLevel

            alert = Alert(
                alert_type=AlertType.REGIME_CHANGE,
                level=AlertLevel.WARNING
                if new_regime in [MarketRegime.BEAR, MarketRegime.CRASH]
                else AlertLevel.INFO,
                title=f"{emoji} Regime Change: {symbol}",
                message=f"Market regime changed from {old_name.upper()} to {new_regime.value.upper()}",
                data={
                    "symbol": symbol,
                    "old_regime": old_name,
                    "new_regime": new_regime.value,
                    "confidence": f"{confidence:.1%}",
                    "action": self._get_regime_action(new_regime),
                },
            )
            self._notifier.send_alert(alert)
            logger.debug(f"Sent regime change notification for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to send regime notification: {e}")

    def _get_regime_action(self, regime: MarketRegime) -> str:
        """Get recommended action for regime."""
        actions = {
            MarketRegime.BULL: "Full exposure (100%)",
            MarketRegime.SIDEWAYS: "Reduced exposure (80%)",
            MarketRegime.BEAR: "Defensive (40%)",
            MarketRegime.HIGH_VOL: "Cautious (50%)",
            MarketRegime.CRASH: "Minimal exposure (10%)",
            MarketRegime.UNKNOWN: "Conservative (30%)",
        }
        return actions.get(regime, "Hold")

    def _notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        regime: MarketRegime,
    ) -> None:
        """Send notification for trade execution."""
        if not self._notifier or not self._notifier.has_channels():
            return

        try:
            from bot.notifications import Alert, AlertType, AlertLevel

            value = quantity * price
            emoji = "🟢" if side == "buy" else "🔴"

            alert = Alert(
                alert_type=AlertType.TRADE_OPENED if side == "buy" else AlertType.TRADE_CLOSED,
                level=AlertLevel.INFO,
                title=f"{emoji} Trade: {side.upper()} {symbol}",
                message=f"Executed {side.upper()} {quantity:.4f} {symbol} @ ${price:,.2f}",
                data={
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": f"${price:,.2f}",
                    "value": f"${value:,.2f}",
                    "regime": regime.value,
                },
            )
            self._notifier.send_alert(alert)
        except Exception as e:
            logger.warning(f"Failed to send trade notification: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if self.config.state_file.exists():
            try:
                with open(self.config.state_file, "r") as f:
                    data = json.load(f)
                    self.state.total_pnl = data.get("total_pnl", 0)
                    self._peak_equity = data.get("peak_equity", 0)
                    logger.info("Loaded trading state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.state_file, "w") as f:
                json.dump(
                    {
                        "state": self.state.to_dict(),
                        "total_pnl": self.state.total_pnl,
                        "peak_equity": getattr(self, "_peak_equity", 0),
                        "saved_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    async def force_update(self) -> None:
        """Force an immediate update cycle."""
        await self._update_cycle()

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at midnight)."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        logger.info("Daily counters reset")


# ============================================================================
# Simple Paper Trading Adapter (for testing)
# ============================================================================


class SimplePaperAdapter:
    """
    Simple paper trading adapter for testing.

    Simulates an exchange with basic functionality.
    Automatically fetches real market data for paper trading.
    """

    def __init__(self, initial_balance: float = 10000.0, auto_fetch: bool = True):
        self._balance = initial_balance
        self._positions: Dict[str, Dict] = {}
        self._prices: Dict[str, float] = {}
        self._ohlcv_data: Dict[str, pd.DataFrame] = {}
        self._auto_fetch = auto_fetch
        self._last_fetch: Dict[str, datetime] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol."""
        self._prices[symbol] = price

    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        position_value = sum(
            pos["quantity"] * self._prices.get(sym, 0) for sym, pos in self._positions.items()
        )
        return {
            "available": self._balance,
            "total": self._balance + position_value,
        }

    async def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self._positions.copy()

    async def get_price(self, symbol: str) -> float:
        """Get current price - fetch if not available."""
        if symbol not in self._prices or self._prices[symbol] <= 0:
            await self._fetch_current_price(symbol)
        return self._prices.get(symbol, 0)

    async def _fetch_current_price(self, symbol: str) -> None:
        """Fetch current price from yfinance."""
        try:
            import yfinance as yf

            yf_symbol = self._convert_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                self._prices[symbol] = float(data["Close"].iloc[-1])
                logger.info(f"Fetched price for {symbol}: ${self._prices[symbol]:,.2f}")
        except Exception as e:
            logger.warning(f"Failed to fetch price for {symbol}: {e}")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert trading symbol to yfinance format."""
        # BTC/USDT -> BTC-USD
        yf_symbol = symbol.replace("/", "-")
        if "USDT" in yf_symbol:
            yf_symbol = yf_symbol.replace("USDT", "USD")
        return yf_symbol

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Get OHLCV data - auto-fetches if not available."""
        # Check if we need to fetch new data
        should_fetch = self._auto_fetch and (
            symbol not in self._ohlcv_data
            or symbol not in self._last_fetch
            or (datetime.now() - self._last_fetch.get(symbol, datetime.min)).total_seconds() > 300
        )

        if should_fetch:
            await self._fetch_ohlcv(symbol, timeframe, limit)

        return self._ohlcv_data.get(symbol)

    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> None:
        """Fetch OHLCV data from yfinance."""
        try:
            import yfinance as yf

            yf_symbol = self._convert_symbol(symbol)

            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "4h": "1h",
                "1d": "1d",
            }
            interval = interval_map.get(timeframe, "1h")

            # Calculate period based on limit
            if interval in ["1m", "5m", "15m", "30m"]:
                period = "7d"
            elif interval == "1h":
                period = "60d"
            else:
                period = "2y"

            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=interval)

            if not data.empty:
                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Rename columns to standard format
                data = data.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                # Take last 'limit' bars
                if len(data) > limit:
                    data = data.tail(limit)

                self._ohlcv_data[symbol] = data
                self._last_fetch[symbol] = datetime.now()

                # Also update current price
                self._prices[symbol] = float(data["close"].iloc[-1])

                logger.info(f"Fetched {len(data)} bars for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch OHLCV for {symbol}: {e}")

    def set_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        """Set OHLCV data for a symbol."""
        self._ohlcv_data[symbol] = df
        self._last_fetch[symbol] = datetime.now()

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
    ) -> Dict:
        """Place an order."""
        price = self._prices.get(symbol, 0)
        if price <= 0:
            return {"status": "error", "reason": "No price"}

        if side == "buy":
            cost = quantity * price
            if cost > self._balance:
                return {"status": "error", "reason": "Insufficient balance"}

            self._balance -= cost

            if symbol not in self._positions:
                self._positions[symbol] = {"quantity": 0}
            self._positions[symbol]["quantity"] += quantity

        elif side == "sell":
            if symbol not in self._positions:
                return {"status": "error", "reason": "No position"}

            current_qty = self._positions[symbol]["quantity"]
            if quantity > current_qty:
                quantity = current_qty

            self._positions[symbol]["quantity"] -= quantity
            self._balance += quantity * price

            if self._positions[symbol]["quantity"] <= 0:
                del self._positions[symbol]

        return {
            "status": "filled",
            "order_id": f"paper_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }
