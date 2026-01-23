"""
Strategy Interface + Registry

Provides a standard interface for all trading strategies and a registry
to manage multiple strategies running in parallel.

Key concepts:
- All strategies implement the same interface
- Strategies output predictions with confidence, stops, and horizons
- Registry manages lifecycle (register, enable, disable, remove)
- Performance tracking per strategy per regime
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


class StrategyAction(Enum):
    """Possible strategy actions"""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    LONG = "long"
    SHORT = "short"


class StrategyStatus(Enum):
    """Strategy lifecycle status"""

    ACTIVE = "active"  # Running and trading
    SHADOW = "shadow"  # Running but not trading (paper mode)
    PAUSED = "paused"  # Temporarily stopped
    DISABLED = "disabled"  # Permanently disabled
    DEGRADED = "degraded"  # Performance degraded, reduced allocation


@dataclass
class StrategySignal:
    """
    Standard output from all strategies.

    This is the "contract" that all strategies must fulfill.
    """

    # Core prediction
    action: StrategyAction
    direction: str  # "long", "short", "flat"
    confidence: float  # 0.0 to 1.0

    # Position management
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_loss_pct: float = 2.0  # Default 2% stop
    take_profit: Optional[float] = None
    take_profit_pct: float = 4.0  # Default 4% take profit

    # Sizing suggestions
    position_size_pct: float = 5.0  # % of equity
    max_leverage: float = 1.0

    # Timing
    horizon_bars: int = 10  # Expected holding period in bars
    urgency: float = 0.5  # 0=can wait, 1=must execute now

    # Metadata
    strategy_name: str = ""
    strategy_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)

    # Reasoning (for audit)
    reasoning: str = ""
    features_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit": self.take_profit,
            "take_profit_pct": self.take_profit_pct,
            "position_size_pct": self.position_size_pct,
            "max_leverage": self.max_leverage,
            "horizon_bars": self.horizon_bars,
            "urgency": self.urgency,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "features_used": self.features_used,
        }


@dataclass
class MarketState:
    """
    Standard input to all strategies.

    Contains everything a strategy needs to make a decision.
    """

    # Price data
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Price history (most recent last)
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)

    # Technical indicators (pre-computed)
    indicators: Dict[str, float] = field(default_factory=dict)

    # Regime context
    regime: str = "unknown"
    regime_confidence: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0

    # Position context
    current_position: Optional[Dict[str, Any]] = None

    # Risk context
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0

    # Macro context
    macro_bias: float = 0.0  # -1 to 1
    risk_off_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ohlcv": {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            "indicators": self.indicators,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "current_position": self.current_position,
            "daily_pnl_pct": self.daily_pnl_pct,
            "drawdown_pct": self.drawdown_pct,
            "macro_bias": self.macro_bias,
            "risk_off_score": self.risk_off_score,
        }


@dataclass
class StrategyPerformance:
    """Track strategy performance metrics"""

    strategy_name: str

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # PnL metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0

    # Per-regime performance
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Recent performance (for allocation decisions)
    recent_win_rate: float = 0.0  # Last N trades
    recent_sharpe: float = 0.0
    recent_pnl_pct: float = 0.0

    # Health indicators
    consecutive_losses: int = 0
    days_since_profit: int = 0
    is_degraded: bool = False

    # Timestamps
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades

    @property
    def avg_loss(self) -> float:
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades

    def update_from_trade(self, pnl: float, pnl_pct: float, regime: str):
        """Update performance from a completed trade"""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_pnl_pct += pnl_pct

        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
            self.consecutive_losses = 0
            self.days_since_profit = 0
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
            self.consecutive_losses += 1

        # Update profit factor
        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss

        # Update per-regime stats
        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = {"trades": 0, "wins": 0, "pnl": 0.0}
        self.performance_by_regime[regime]["trades"] += 1
        self.performance_by_regime[regime]["pnl"] += pnl
        if pnl > 0:
            self.performance_by_regime[regime]["wins"] += 1

        # Check for degradation
        self.is_degraded = (
            self.consecutive_losses >= 5
            or self.days_since_profit >= 10
            or (self.total_trades >= 20 and self.win_rate < 0.35)
        )

        self.last_trade = datetime.now()
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "is_degraded": self.is_degraded,
            "performance_by_regime": self.performance_by_regime,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
        }


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - name: Unique identifier
    - version: For tracking changes
    - predict(): Generate trading signal from market state
    - suitable_regimes: Which market regimes this strategy works best in

    Optional:
    - initialize(): Setup before first prediction
    - update(): Update internal state after each bar
    - on_trade_complete(): Learn from completed trades
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.performance = StrategyPerformance(strategy_name=self.name)
        self.status = StrategyStatus.ACTIVE
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version for tracking"""
        pass

    @property
    @abstractmethod
    def suitable_regimes(self) -> List[str]:
        """List of regimes where this strategy performs best"""
        pass

    @property
    def description(self) -> str:
        """Human-readable description"""
        return f"{self.name} v{self.version}"

    @abstractmethod
    def predict(self, state: MarketState) -> StrategySignal:
        """
        Generate trading signal from market state.

        This is the core method that all strategies must implement.

        Args:
            state: Current market state with prices, indicators, regime, etc.

        Returns:
            StrategySignal with action, confidence, stops, etc.
        """
        pass

    def initialize(self, historical_data: Optional[List[Dict]] = None):
        """
        Initialize strategy with historical data.

        Called once before first prediction.
        Override to load models, warm up indicators, etc.
        """
        self._initialized = True

    def update(self, state: MarketState):
        """
        Update internal state after each bar.

        Called after each market update, regardless of whether a trade was made.
        Override to update rolling statistics, etc.
        """
        pass

    def on_trade_complete(self, trade_result: Dict[str, Any]):
        """
        Called when a trade initiated by this strategy completes.

        Use this for online learning or adaptation.

        Args:
            trade_result: Dict with pnl, pnl_pct, duration, regime, etc.
        """
        pnl = trade_result.get("pnl", 0)
        pnl_pct = trade_result.get("pnl_pct", 0)
        regime = trade_result.get("regime", "unknown")
        self.performance.update_from_trade(pnl, pnl_pct, regime)

    def get_regime_score(self, current_regime: str) -> float:
        """
        Get strategy's expected performance score for a regime.

        Returns 0.0-1.0 score based on suitability and historical performance.
        """
        # Base score from suitability
        if current_regime in self.suitable_regimes:
            base_score = 0.8
        else:
            base_score = 0.3

        # Adjust based on historical performance in this regime
        regime_perf = self.performance.performance_by_regime.get(current_regime, {})
        if regime_perf.get("trades", 0) >= 10:
            win_rate = regime_perf.get("wins", 0) / regime_perf["trades"]
            perf_adjustment = (win_rate - 0.5) * 0.4  # -0.2 to +0.2
            base_score += perf_adjustment

        return max(0.0, min(1.0, base_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": self.status.value,
            "suitable_regimes": self.suitable_regimes,
            "params": self.params,
            "performance": self.performance.to_dict(),
        }


class StrategyRegistry:
    """
    Central registry for managing multiple trading strategies.

    Features:
    - Register/unregister strategies
    - Enable/disable strategies
    - Get signals from all active strategies
    - Track performance per strategy
    - Auto-disable degraded strategies
    """

    def __init__(self, auto_disable_degraded: bool = True, min_confidence_threshold: float = 0.4):
        self.strategies: Dict[str, Strategy] = {}
        self.auto_disable_degraded = auto_disable_degraded
        self.min_confidence_threshold = min_confidence_threshold

        # Allocation weights (set by meta-allocator)
        self.allocation_weights: Dict[str, float] = {}

        # Callbacks
        self._on_signal_callbacks: List[Callable] = []
        self._on_status_change_callbacks: List[Callable] = []

        logger.info("Strategy Registry initialized")

    def register(
        self,
        strategy: Strategy,
        initial_weight: float = 1.0,
        status: StrategyStatus = StrategyStatus.ACTIVE,
    ) -> bool:
        """Register a new strategy"""
        if strategy.name in self.strategies:
            logger.warning(f"Strategy {strategy.name} already registered")
            return False

        strategy.status = status
        self.strategies[strategy.name] = strategy
        self.allocation_weights[strategy.name] = initial_weight

        logger.info(f"Registered strategy: {strategy.name} v{strategy.version} ({status.value})")
        return True

    def unregister(self, strategy_name: str) -> bool:
        """Remove a strategy from the registry"""
        if strategy_name not in self.strategies:
            return False

        del self.strategies[strategy_name]
        self.allocation_weights.pop(strategy_name, None)

        logger.info(f"Unregistered strategy: {strategy_name}")
        return True

    def set_status(self, strategy_name: str, status: StrategyStatus):
        """Change strategy status"""
        if strategy_name not in self.strategies:
            return

        old_status = self.strategies[strategy_name].status
        self.strategies[strategy_name].status = status

        logger.info(f"Strategy {strategy_name} status: {old_status.value} -> {status.value}")

        for callback in self._on_status_change_callbacks:
            try:
                callback(strategy_name, old_status, status)
            except Exception as e:
                logger.error(f"Status change callback error: {e}")

    def set_weight(self, strategy_name: str, weight: float):
        """Set allocation weight for a strategy"""
        if strategy_name in self.strategies:
            self.allocation_weights[strategy_name] = max(0.0, weight)

    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name"""
        return self.strategies.get(name)

    def get_active_strategies(self) -> List[Strategy]:
        """Get all strategies that should generate signals"""
        return [
            s
            for s in self.strategies.values()
            if s.status in [StrategyStatus.ACTIVE, StrategyStatus.SHADOW]
        ]

    def get_trading_strategies(self) -> List[Strategy]:
        """Get strategies that should execute trades (not shadow)"""
        return [s for s in self.strategies.values() if s.status == StrategyStatus.ACTIVE]

    def initialize_all(self, historical_data: Optional[List[Dict]] = None):
        """Initialize all registered strategies"""
        for strategy in self.strategies.values():
            try:
                strategy.initialize(historical_data)
            except Exception as e:
                logger.error(f"Failed to initialize {strategy.name}: {e}")
                strategy.status = StrategyStatus.DISABLED

    def get_signals(
        self, state: MarketState, include_shadow: bool = True
    ) -> List[Tuple[Strategy, StrategySignal]]:
        """
        Get signals from all active strategies.

        Args:
            state: Current market state
            include_shadow: Include shadow strategies in output

        Returns:
            List of (strategy, signal) tuples
        """
        signals = []

        strategies = (
            self.get_active_strategies() if include_shadow else self.get_trading_strategies()
        )

        for strategy in strategies:
            try:
                # Check if strategy is suitable for current regime
                regime_score = strategy.get_regime_score(state.regime)

                # Generate signal
                signal = strategy.predict(state)
                signal.strategy_name = strategy.name
                signal.strategy_version = strategy.version

                # Adjust confidence by regime suitability
                signal.confidence *= regime_score

                # Filter low confidence signals
                if signal.confidence >= self.min_confidence_threshold:
                    signals.append((strategy, signal))

                # Update strategy internal state
                strategy.update(state)

                # Notify callbacks
                for callback in self._on_signal_callbacks:
                    try:
                        callback(strategy, signal)
                    except Exception as e:
                        logger.error(f"Signal callback error: {e}")

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed to predict: {e}")
                if self.auto_disable_degraded:
                    self.set_status(strategy.name, StrategyStatus.PAUSED)

        # Check for degraded strategies
        if self.auto_disable_degraded:
            self._check_degraded_strategies()

        return signals

    def get_consensus_signal(
        self, signals: List[Tuple[Strategy, StrategySignal]]
    ) -> Optional[StrategySignal]:
        """
        Combine multiple strategy signals into a consensus.

        Uses allocation weights and confidence to weight votes.
        """
        if not signals:
            return None

        # Weight each signal
        weighted_votes = {"long": 0.0, "short": 0.0, "flat": 0.0}
        total_weight = 0.0

        for strategy, signal in signals:
            if strategy.status != StrategyStatus.ACTIVE:
                continue

            weight = self.allocation_weights.get(strategy.name, 1.0)
            vote_weight = weight * signal.confidence

            weighted_votes[signal.direction] += vote_weight
            total_weight += vote_weight

        if total_weight == 0:
            return None

        # Normalize votes
        for direction in weighted_votes:
            weighted_votes[direction] /= total_weight

        # Determine consensus direction
        consensus_direction = max(weighted_votes, key=weighted_votes.get)
        consensus_confidence = weighted_votes[consensus_direction]

        # Find matching signals for consensus direction
        matching_signals = [
            (s, sig)
            for s, sig in signals
            if sig.direction == consensus_direction and s.status == StrategyStatus.ACTIVE
        ]

        if not matching_signals:
            return None

        # Aggregate position parameters (weighted average)
        total_match_weight = sum(
            self.allocation_weights.get(s.name, 1.0) * sig.confidence for s, sig in matching_signals
        )

        avg_stop_pct = (
            sum(
                sig.stop_loss_pct * self.allocation_weights.get(s.name, 1.0) * sig.confidence
                for s, sig in matching_signals
            )
            / total_match_weight
        )

        avg_take_pct = (
            sum(
                sig.take_profit_pct * self.allocation_weights.get(s.name, 1.0) * sig.confidence
                for s, sig in matching_signals
            )
            / total_match_weight
        )

        avg_size_pct = (
            sum(
                sig.position_size_pct * self.allocation_weights.get(s.name, 1.0) * sig.confidence
                for s, sig in matching_signals
            )
            / total_match_weight
        )

        # Build consensus signal
        if consensus_direction == "long":
            action = StrategyAction.BUY
        elif consensus_direction == "short":
            action = StrategyAction.SELL
        else:
            action = StrategyAction.HOLD

        return StrategySignal(
            action=action,
            direction=consensus_direction,
            confidence=consensus_confidence,
            stop_loss_pct=avg_stop_pct,
            take_profit_pct=avg_take_pct,
            position_size_pct=avg_size_pct,
            strategy_name="consensus",
            strategy_version="1.0",
            reasoning=f"Consensus from {len(matching_signals)} strategies: {[s.name for s, _ in matching_signals]}",
        )

    def _check_degraded_strategies(self):
        """Check for and handle degraded strategies"""
        for name, strategy in self.strategies.items():
            if strategy.performance.is_degraded and strategy.status == StrategyStatus.ACTIVE:
                logger.warning(f"Strategy {name} degraded - reducing allocation")
                self.set_status(name, StrategyStatus.DEGRADED)
                self.allocation_weights[name] *= 0.5  # Reduce allocation

    def on_trade_complete(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Notify a strategy that its trade completed"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].on_trade_complete(trade_result)

    def on_signal(self, callback: Callable):
        """Register signal callback"""
        self._on_signal_callbacks.append(callback)

    def on_status_change(self, callback: Callable):
        """Register status change callback"""
        self._on_status_change_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        return {
            name: {
                "status": s.status.value,
                "weight": self.allocation_weights.get(name, 0),
                "performance": s.performance.to_dict(),
            }
            for name, s in self.strategies.items()
        }

    def get_leaderboard(self, metric: str = "sharpe_ratio") -> List[Dict[str, Any]]:
        """Get strategies ranked by a metric"""
        strategies = []
        for name, s in self.strategies.items():
            perf = s.performance
            strategies.append(
                {
                    "name": name,
                    "status": s.status.value,
                    "win_rate": perf.win_rate,
                    "total_pnl_pct": perf.total_pnl_pct,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "profit_factor": perf.profit_factor,
                    "total_trades": perf.total_trades,
                }
            )

        # Sort by metric (descending)
        strategies.sort(key=lambda x: x.get(metric, 0), reverse=True)
        return strategies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_strategies": len(self.strategies),
            "active_count": len(self.get_active_strategies()),
            "trading_count": len(self.get_trading_strategies()),
            "strategies": {name: s.to_dict() for name, s in self.strategies.items()},
            "allocation_weights": self.allocation_weights,
        }


# ============================================================
# Built-in Strategy Implementations
# ============================================================


class EMACrossoverStrategy(Strategy):
    """
    Classic EMA crossover with RSI confirmation.

    Long: Fast EMA > Slow EMA AND RSI > 50
    Short: Fast EMA < Slow EMA AND RSI < 50
    """

    @property
    def name(self) -> str:
        return "ema_crossover"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["trending", "strong_trend", "bull", "bear"]

    def predict(self, state: MarketState) -> StrategySignal:
        indicators = state.indicators

        ema_fast = indicators.get("ema_fast", indicators.get("ema_12", 0))
        ema_slow = indicators.get("ema_slow", indicators.get("ema_26", 0))
        rsi = indicators.get("rsi", indicators.get("rsi_14", 50))

        # Default to HOLD
        action = StrategyAction.HOLD
        direction = "flat"
        confidence = 0.0
        reasoning = ""

        if ema_fast > ema_slow:
            if rsi > 50:
                action = StrategyAction.BUY
                direction = "long"
                confidence = min(0.9, 0.5 + (rsi - 50) / 100 + (ema_fast - ema_slow) / ema_slow)
                reasoning = f"EMA bullish crossover (fast={ema_fast:.2f} > slow={ema_slow:.2f}), RSI={rsi:.1f}"
            else:
                confidence = 0.3
                reasoning = "EMA bullish but RSI weak"
        elif ema_fast < ema_slow:
            if rsi < 50:
                action = StrategyAction.SELL
                direction = "short"
                confidence = min(0.9, 0.5 + (50 - rsi) / 100 + (ema_slow - ema_fast) / ema_slow)
                reasoning = f"EMA bearish crossover (fast={ema_fast:.2f} < slow={ema_slow:.2f}), RSI={rsi:.1f}"
            else:
                confidence = 0.3
                reasoning = "EMA bearish but RSI strong"
        else:
            reasoning = "No clear EMA signal"

        return StrategySignal(
            action=action,
            direction=direction,
            confidence=confidence,
            stop_loss_pct=self.params.get("stop_loss_pct", 2.0),
            take_profit_pct=self.params.get("take_profit_pct", 4.0),
            position_size_pct=self.params.get("position_size_pct", 5.0),
            reasoning=reasoning,
            features_used={"ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": rsi},
        )


class MeanReversionStrategy(Strategy):
    """
    Mean reversion using Bollinger Bands and RSI extremes.

    Long: Price below lower BB AND RSI < 30
    Short: Price above upper BB AND RSI > 70
    """

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["ranging", "sideways", "low_volatility"]

    def predict(self, state: MarketState) -> StrategySignal:
        indicators = state.indicators
        price = state.close

        bb_upper = indicators.get("bb_upper", price * 1.02)
        bb_lower = indicators.get("bb_lower", price * 0.98)
        bb_middle = indicators.get("bb_middle", price)
        rsi = indicators.get("rsi", indicators.get("rsi_14", 50))

        action = StrategyAction.HOLD
        direction = "flat"
        confidence = 0.0
        reasoning = ""

        # Oversold - look for long
        if price < bb_lower and rsi < 30:
            action = StrategyAction.BUY
            direction = "long"
            # More oversold = higher confidence
            oversold_factor = (30 - rsi) / 30
            bb_factor = (bb_lower - price) / (bb_middle - bb_lower) if bb_middle != bb_lower else 0
            confidence = min(0.9, 0.5 + oversold_factor * 0.3 + bb_factor * 0.2)
            reasoning = (
                f"Oversold: Price below BB lower ({price:.2f} < {bb_lower:.2f}), RSI={rsi:.1f}"
            )

        # Overbought - look for short
        elif price > bb_upper and rsi > 70:
            action = StrategyAction.SELL
            direction = "short"
            overbought_factor = (rsi - 70) / 30
            bb_factor = (price - bb_upper) / (bb_upper - bb_middle) if bb_upper != bb_middle else 0
            confidence = min(0.9, 0.5 + overbought_factor * 0.3 + bb_factor * 0.2)
            reasoning = (
                f"Overbought: Price above BB upper ({price:.2f} > {bb_upper:.2f}), RSI={rsi:.1f}"
            )

        else:
            reasoning = f"No mean reversion signal: Price in BB range, RSI={rsi:.1f}"

        return StrategySignal(
            action=action,
            direction=direction,
            confidence=confidence,
            stop_loss_pct=self.params.get("stop_loss_pct", 1.5),  # Tighter stops for mean reversion
            take_profit_pct=self.params.get("take_profit_pct", 2.0),  # Smaller targets
            position_size_pct=self.params.get("position_size_pct", 5.0),
            reasoning=reasoning,
            features_used={
                "price": price,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "rsi": rsi,
            },
        )


class BreakoutStrategy(Strategy):
    """
    Volatility breakout strategy using ATR and price channels.

    Long: Price breaks above N-period high with expanding volatility
    Short: Price breaks below N-period low with expanding volatility
    """

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["volatile", "breakout", "expansion"]

    def predict(self, state: MarketState) -> StrategySignal:
        indicators = state.indicators
        price = state.close

        high_20 = indicators.get("high_20", indicators.get("highest_20", price))
        low_20 = indicators.get("low_20", indicators.get("lowest_20", price))
        atr = indicators.get("atr", indicators.get("atr_14", 0))
        atr_pct = (atr / price * 100) if price > 0 else 0

        # Check volatility expansion
        avg_atr_pct = indicators.get("avg_atr_pct", atr_pct)
        vol_expanding = atr_pct > avg_atr_pct * 1.2

        action = StrategyAction.HOLD
        direction = "flat"
        confidence = 0.0
        reasoning = ""

        # Upside breakout
        if price >= high_20 and vol_expanding:
            action = StrategyAction.BUY
            direction = "long"
            breakout_strength = (price - high_20) / atr if atr > 0 else 0
            confidence = min(0.85, 0.5 + breakout_strength * 0.2 + 0.1)
            reasoning = (
                f"Upside breakout: Price ({price:.2f}) >= 20-high ({high_20:.2f}), ATR expanding"
            )

        # Downside breakout
        elif price <= low_20 and vol_expanding:
            action = StrategyAction.SELL
            direction = "short"
            breakout_strength = (low_20 - price) / atr if atr > 0 else 0
            confidence = min(0.85, 0.5 + breakout_strength * 0.2 + 0.1)
            reasoning = (
                f"Downside breakout: Price ({price:.2f}) <= 20-low ({low_20:.2f}), ATR expanding"
            )

        else:
            reasoning = f"No breakout: Price in range [{low_20:.2f}, {high_20:.2f}]"

        # Use ATR-based stops
        stop_pct = atr_pct * 2 if atr_pct > 0 else 2.0
        take_pct = atr_pct * 4 if atr_pct > 0 else 4.0

        return StrategySignal(
            action=action,
            direction=direction,
            confidence=confidence,
            stop_loss_pct=stop_pct,
            take_profit_pct=take_pct,
            position_size_pct=self.params.get("position_size_pct", 5.0),
            reasoning=reasoning,
            features_used={
                "price": price,
                "high_20": high_20,
                "low_20": low_20,
                "atr_pct": atr_pct,
                "vol_expanding": vol_expanding,
            },
        )


class MomentumStrategy(Strategy):
    """
    Momentum strategy using MACD and ADX for trend strength.

    Long: MACD > Signal AND ADX > 25 (strong trend)
    Short: MACD < Signal AND ADX > 25
    """

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def suitable_regimes(self) -> List[str]:
        return ["trending", "strong_trend", "bull", "bear"]

    def predict(self, state: MarketState) -> StrategySignal:
        indicators = state.indicators

        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", macd - macd_signal)
        adx = indicators.get("adx", indicators.get("adx_14", 20))

        action = StrategyAction.HOLD
        direction = "flat"
        confidence = 0.0
        reasoning = ""

        # Strong trend filter
        strong_trend = adx > 25

        if macd > macd_signal and strong_trend:
            action = StrategyAction.BUY
            direction = "long"
            # Higher ADX and larger MACD difference = higher confidence
            trend_factor = min((adx - 25) / 25, 1.0)
            macd_factor = min(abs(macd_hist) / abs(macd_signal) if macd_signal != 0 else 0, 1.0)
            confidence = min(0.9, 0.5 + trend_factor * 0.2 + macd_factor * 0.2)
            reasoning = (
                f"Bullish momentum: MACD ({macd:.4f}) > Signal ({macd_signal:.4f}), ADX={adx:.1f}"
            )

        elif macd < macd_signal and strong_trend:
            action = StrategyAction.SELL
            direction = "short"
            trend_factor = min((adx - 25) / 25, 1.0)
            macd_factor = min(abs(macd_hist) / abs(macd_signal) if macd_signal != 0 else 0, 1.0)
            confidence = min(0.9, 0.5 + trend_factor * 0.2 + macd_factor * 0.2)
            reasoning = (
                f"Bearish momentum: MACD ({macd:.4f}) < Signal ({macd_signal:.4f}), ADX={adx:.1f}"
            )

        else:
            if not strong_trend:
                reasoning = f"Weak trend: ADX={adx:.1f} < 25"
            else:
                reasoning = "No clear MACD signal"

        return StrategySignal(
            action=action,
            direction=direction,
            confidence=confidence,
            stop_loss_pct=self.params.get("stop_loss_pct", 2.5),
            take_profit_pct=self.params.get("take_profit_pct", 5.0),
            position_size_pct=self.params.get("position_size_pct", 5.0),
            horizon_bars=15,  # Momentum trades often need more time
            reasoning=reasoning,
            features_used={
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "adx": adx,
            },
        )


# ============================================================
# Global Registry Instance
# ============================================================

_strategy_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Get or create global strategy registry"""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()

        # Register built-in strategies
        _strategy_registry.register(EMACrossoverStrategy())
        _strategy_registry.register(MeanReversionStrategy())
        _strategy_registry.register(BreakoutStrategy())
        _strategy_registry.register(MomentumStrategy())

        logger.info("Strategy Registry created with 4 built-in strategies")

    return _strategy_registry


def create_strategy_registry(
    include_builtin: bool = True, auto_disable_degraded: bool = True
) -> StrategyRegistry:
    """Create a new strategy registry"""
    registry = StrategyRegistry(auto_disable_degraded=auto_disable_degraded)

    if include_builtin:
        registry.register(EMACrossoverStrategy())
        registry.register(MeanReversionStrategy())
        registry.register(BreakoutStrategy())
        registry.register(MomentumStrategy())

    return registry


__all__ = [
    # Enums
    "StrategyAction",
    "StrategyStatus",
    # Data classes
    "StrategySignal",
    "MarketState",
    "StrategyPerformance",
    # Base class
    "Strategy",
    # Registry
    "StrategyRegistry",
    "get_strategy_registry",
    "create_strategy_registry",
    # Built-in strategies
    "EMACrossoverStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "MomentumStrategy",
]
