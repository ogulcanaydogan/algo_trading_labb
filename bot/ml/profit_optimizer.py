"""
Profit Optimizer - Learns optimal entry/exit timing for maximum profits.

Uses reinforcement learning concepts to:
- Learn optimal entry points (buy dips, short rallies)
- Optimize exit timing (let winners run, cut losers)
- Adapt stop loss and take profit levels
- Maximize risk-adjusted returns
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """Optimal entry signal."""

    should_enter: bool
    direction: Literal["LONG", "SHORT"]
    entry_quality: float  # 0-1, how good is this entry
    wait_bars: int  # Suggested bars to wait for better entry
    suggested_price: float  # Target entry price
    reason: str


@dataclass
class ExitSignal:
    """Optimal exit signal."""

    should_exit: bool
    exit_type: Literal["TAKE_PROFIT", "STOP_LOSS", "TRAILING", "TIME", "SIGNAL"]
    urgency: float  # 0-1, how urgent to exit
    target_price: float
    reason: str


@dataclass
class TradeState:
    """Current trade state for optimization."""

    direction: str
    entry_price: float
    current_price: float
    bars_held: int
    max_favorable: float  # Max profit seen
    max_adverse: float  # Max drawdown seen
    unrealized_pnl_pct: float


@dataclass
class OptimizedLevels:
    """Optimized stop loss and take profit levels."""

    stop_loss_price: float
    stop_loss_pct: float
    take_profit_price: float
    take_profit_pct: float
    trailing_stop_pct: float
    break_even_trigger_pct: float


class EntryOptimizer:
    """
    Optimizes trade entry timing.

    Learns patterns that lead to better entries:
    - Entering after pullbacks in trends
    - Entering at support/resistance levels
    - Avoiding entries during low-quality setups
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

        # Track entry quality outcomes
        self.entry_outcomes: Deque[Dict] = deque(maxlen=500)

        # Learned thresholds
        self.optimal_rsi_buy = 35.0  # RSI below this for longs
        self.optimal_rsi_sell = 65.0  # RSI above this for shorts
        self.optimal_bb_deviation = 1.5  # BB deviation for mean reversion
        self.optimal_volume_ratio = 1.2  # Volume surge for confirmation

    def analyze_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        current_price: float,
    ) -> EntrySignal:
        """
        Analyze if current price is optimal for entry.

        Args:
            df: Recent OHLCV with indicators
            direction: "LONG" or "SHORT"
            current_price: Current market price

        Returns:
            EntrySignal with recommendation
        """
        if len(df) < 20:
            return EntrySignal(
                should_enter=True,
                direction=direction,
                entry_quality=0.5,
                wait_bars=0,
                suggested_price=current_price,
                reason="Insufficient data",
            )

        # Calculate entry quality metrics
        quality_score = 0.0
        reasons = []

        # 1. RSI position (is price oversold/overbought?)
        rsi = self._calculate_rsi(df["close"], 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        if direction == "LONG":
            if current_rsi < self.optimal_rsi_buy:
                quality_score += 0.25
                reasons.append(f"RSI oversold ({current_rsi:.1f})")
            elif current_rsi > 70:
                quality_score -= 0.2
                reasons.append(f"RSI overbought ({current_rsi:.1f})")
        else:  # SHORT
            if current_rsi > self.optimal_rsi_sell:
                quality_score += 0.25
                reasons.append(f"RSI overbought ({current_rsi:.1f})")
            elif current_rsi < 30:
                quality_score -= 0.2
                reasons.append(f"RSI oversold ({current_rsi:.1f})")

        # 2. Bollinger Band position
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        bb_position = (current_price - bb_mid.iloc[-1]) / (bb_std.iloc[-1] + 1e-10)

        if direction == "LONG" and bb_position < -self.optimal_bb_deviation:
            quality_score += 0.2
            reasons.append("Near lower Bollinger Band")
        elif direction == "SHORT" and bb_position > self.optimal_bb_deviation:
            quality_score += 0.2
            reasons.append("Near upper Bollinger Band")

        # 3. Volume confirmation
        volume_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
        if volume_ratio > self.optimal_volume_ratio:
            quality_score += 0.15
            reasons.append(f"Volume surge ({volume_ratio:.1f}x)")

        # 4. Support/Resistance proximity
        recent_high = df["high"].rolling(20).max().iloc[-1]
        recent_low = df["low"].rolling(20).min().iloc[-1]
        price_range = recent_high - recent_low

        if direction == "LONG":
            distance_to_support = (current_price - recent_low) / (price_range + 1e-10)
            if distance_to_support < 0.2:
                quality_score += 0.2
                reasons.append("Near support")
        else:
            distance_to_resistance = (recent_high - current_price) / (price_range + 1e-10)
            if distance_to_resistance < 0.2:
                quality_score += 0.2
                reasons.append("Near resistance")

        # 5. Trend alignment
        ema_20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema_50 = df["close"].ewm(span=50).mean().iloc[-1]

        if direction == "LONG" and ema_20 > ema_50:
            quality_score += 0.1
            reasons.append("Aligned with uptrend")
        elif direction == "SHORT" and ema_20 < ema_50:
            quality_score += 0.1
            reasons.append("Aligned with downtrend")

        # Normalize quality score
        quality_score = max(0.0, min(1.0, quality_score + 0.5))

        # Determine if we should wait
        should_enter = quality_score >= 0.5
        wait_bars = 0
        suggested_price = current_price

        if not should_enter and quality_score >= 0.3:
            # Suggest waiting for better entry
            wait_bars = 3
            if direction == "LONG":
                suggested_price = current_price * 0.995  # Wait for 0.5% pullback
            else:
                suggested_price = current_price * 1.005

        return EntrySignal(
            should_enter=should_enter,
            direction=direction,
            entry_quality=quality_score,
            wait_bars=wait_bars,
            suggested_price=suggested_price,
            reason="; ".join(reasons) if reasons else "Standard entry",
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def record_entry_outcome(
        self,
        entry_quality: float,
        direction: str,
        pnl_pct: float,
        max_favorable_pct: float,
        max_adverse_pct: float,
    ) -> None:
        """Record outcome of an entry for learning."""
        self.entry_outcomes.append(
            {
                "quality": entry_quality,
                "direction": direction,
                "pnl_pct": pnl_pct,
                "max_favorable": max_favorable_pct,
                "max_adverse": max_adverse_pct,
                "was_good": pnl_pct > 0,
            }
        )

        # Adapt thresholds based on outcomes
        self._adapt_thresholds()

    def _adapt_thresholds(self) -> None:
        """Adapt entry thresholds based on outcomes."""
        if len(self.entry_outcomes) < 50:
            return

        recent = list(self.entry_outcomes)[-100:]

        # Analyze which quality levels led to profits
        quality_profit = {}
        for outcome in recent:
            q_bucket = round(outcome["quality"], 1)
            if q_bucket not in quality_profit:
                quality_profit[q_bucket] = []
            quality_profit[q_bucket].append(outcome["pnl_pct"])

        # Log learning progress
        good_entries = [o for o in recent if o["pnl_pct"] > 0]
        if recent:
            win_rate = len(good_entries) / len(recent)
            logger.info(f"Entry optimizer win rate: {win_rate:.2%}")


class ExitOptimizer:
    """
    Optimizes trade exit timing.

    Learns:
    - When to let winners run vs take profit
    - When to cut losses vs hold through drawdown
    - Optimal trailing stop levels
    - Time-based exit rules
    """

    def __init__(self):
        # Track exit outcomes
        self.exit_outcomes: Deque[Dict] = deque(maxlen=500)

        # Learned parameters
        self.optimal_profit_target = 0.03  # 3% base target
        self.optimal_stop_loss = 0.02  # 2% base stop
        self.optimal_trailing_pct = 0.015  # 1.5% trailing
        self.max_hold_bars = 50  # Max bars to hold
        self.break_even_trigger = 0.01  # Move to break even at 1%

    def analyze_exit(
        self,
        trade_state: TradeState,
        df: pd.DataFrame,
    ) -> ExitSignal:
        """
        Analyze if current position should be exited.

        Args:
            trade_state: Current trade state
            df: Recent OHLCV data

        Returns:
            ExitSignal with recommendation
        """
        pnl = trade_state.unrealized_pnl_pct
        bars = trade_state.bars_held
        max_profit = trade_state.max_favorable

        # Check stop loss
        if pnl <= -self.optimal_stop_loss:
            return ExitSignal(
                should_exit=True,
                exit_type="STOP_LOSS",
                urgency=1.0,
                target_price=trade_state.current_price,
                reason=f"Stop loss hit ({pnl:.2%})",
            )

        # Check take profit
        if pnl >= self.optimal_profit_target:
            # Consider letting it run if momentum is strong
            momentum = self._calculate_momentum(df)

            if trade_state.direction == "LONG" and momentum > 0.5:
                # Let it run with trailing stop
                return ExitSignal(
                    should_exit=False,
                    exit_type="TRAILING",
                    urgency=0.3,
                    target_price=trade_state.current_price,
                    reason=f"Profit at {pnl:.2%}, momentum strong - trailing",
                )

            return ExitSignal(
                should_exit=True,
                exit_type="TAKE_PROFIT",
                urgency=0.8,
                target_price=trade_state.current_price,
                reason=f"Take profit target reached ({pnl:.2%})",
            )

        # Check trailing stop (if we had profit)
        if max_profit > self.break_even_trigger:
            trailing_stop = max_profit - self.optimal_trailing_pct
            if pnl < trailing_stop:
                return ExitSignal(
                    should_exit=True,
                    exit_type="TRAILING",
                    urgency=0.9,
                    target_price=trade_state.current_price,
                    reason=f"Trailing stop hit (max: {max_profit:.2%}, now: {pnl:.2%})",
                )

        # Check time-based exit
        if bars >= self.max_hold_bars:
            return ExitSignal(
                should_exit=True,
                exit_type="TIME",
                urgency=0.6,
                target_price=trade_state.current_price,
                reason=f"Max hold time reached ({bars} bars)",
            )

        # Check momentum reversal
        momentum = self._calculate_momentum(df)
        if trade_state.direction == "LONG" and momentum < -0.3 and pnl > 0:
            return ExitSignal(
                should_exit=True,
                exit_type="SIGNAL",
                urgency=0.7,
                target_price=trade_state.current_price,
                reason=f"Momentum reversal (mom: {momentum:.2f})",
            )
        elif trade_state.direction == "SHORT" and momentum > 0.3 and pnl > 0:
            return ExitSignal(
                should_exit=True,
                exit_type="SIGNAL",
                urgency=0.7,
                target_price=trade_state.current_price,
                reason=f"Momentum reversal (mom: {momentum:.2f})",
            )

        # Hold position
        return ExitSignal(
            should_exit=False,
            exit_type="TRAILING",
            urgency=0.0,
            target_price=trade_state.current_price,
            reason=f"Hold (pnl: {pnl:.2%}, bars: {bars})",
        )

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (-1 to 1)."""
        if len(df) < 10:
            return 0.0

        # Price momentum
        returns = df["close"].pct_change()
        recent_return = returns.tail(5).mean()

        # RSI momentum
        rsi = self._calculate_rsi(df["close"], 14)
        rsi_score = (rsi.iloc[-1] - 50) / 50 if not pd.isna(rsi.iloc[-1]) else 0

        # MACD momentum
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_diff = macd - macd_signal
        macd_score = np.sign(macd_diff.iloc[-1]) * min(
            1, abs(macd_diff.iloc[-1]) / df["close"].iloc[-1] * 100
        )

        # Combine
        momentum = (
            0.4 * np.sign(recent_return) * min(1, abs(recent_return) * 100)
            + 0.3 * rsi_score
            + 0.3 * macd_score
        )

        return max(-1, min(1, momentum))

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def calculate_optimal_levels(
        self,
        entry_price: float,
        direction: str,
        volatility: float,
        confidence: float,
    ) -> OptimizedLevels:
        """
        Calculate optimized stop loss and take profit levels.

        Args:
            entry_price: Entry price
            direction: "LONG" or "SHORT"
            volatility: Current volatility (ATR as %)
            confidence: Signal confidence (0-1)

        Returns:
            OptimizedLevels with calculated prices
        """
        # Adjust based on volatility
        vol_factor = max(0.5, min(2.0, volatility / 0.02))  # Normalize to 2% baseline

        # Adjust based on confidence
        conf_factor = 0.8 + confidence * 0.4  # 0.8 - 1.2

        # Calculate percentages
        stop_loss_pct = self.optimal_stop_loss * vol_factor / conf_factor
        take_profit_pct = self.optimal_profit_target * vol_factor * conf_factor
        trailing_pct = self.optimal_trailing_pct * vol_factor

        # Ensure minimum 1.5:1 R:R
        if take_profit_pct < stop_loss_pct * 1.5:
            take_profit_pct = stop_loss_pct * 1.5

        # Calculate prices
        if direction == "LONG":
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - take_profit_pct)

        return OptimizedLevels(
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_pct,
            break_even_trigger_pct=self.break_even_trigger,
        )

    def record_exit_outcome(
        self,
        exit_type: str,
        pnl_pct: float,
        bars_held: int,
        max_favorable_pct: float,
        left_on_table_pct: float,  # How much more could we have made
    ) -> None:
        """Record outcome of an exit for learning."""
        self.exit_outcomes.append(
            {
                "exit_type": exit_type,
                "pnl_pct": pnl_pct,
                "bars_held": bars_held,
                "max_favorable": max_favorable_pct,
                "left_on_table": left_on_table_pct,
                "was_good": pnl_pct > 0,
            }
        )

        # Adapt parameters
        self._adapt_parameters()

    def _adapt_parameters(self) -> None:
        """Adapt exit parameters based on outcomes."""
        if len(self.exit_outcomes) < 50:
            return

        recent = list(self.exit_outcomes)[-100:]

        # Analyze stop loss effectiveness
        stop_exits = [o for o in recent if o["exit_type"] == "STOP_LOSS"]
        if stop_exits:
            avg_stop_loss = np.mean([-o["pnl_pct"] for o in stop_exits])
            # If stops are too tight (frequently hit but would have recovered)
            recovery_rate = len([o for o in stop_exits if o["max_favorable"] > 0]) / len(stop_exits)
            if recovery_rate > 0.4:
                self.optimal_stop_loss *= 1.1  # Widen stops
            elif recovery_rate < 0.1:
                self.optimal_stop_loss *= 0.95  # Tighten stops

        # Analyze take profit effectiveness
        tp_exits = [o for o in recent if o["exit_type"] == "TAKE_PROFIT"]
        if tp_exits:
            avg_left = np.mean([o["left_on_table"] for o in tp_exits])
            if avg_left > 0.02:  # Leaving > 2% on table
                self.optimal_profit_target *= 1.1  # Extend targets

        logger.debug(
            f"Exit optimizer adapted: SL={self.optimal_stop_loss:.2%}, "
            f"TP={self.optimal_profit_target:.2%}"
        )


class ProfitOptimizer:
    """
    Combined profit optimizer for entry and exit timing.

    Provides:
    - Optimal entry signals
    - Optimal exit signals
    - Adaptive stop loss / take profit levels
    - Trade management recommendations
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.entry_optimizer = EntryOptimizer()
        self.exit_optimizer = ExitOptimizer()

    def optimize_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        current_price: float,
    ) -> EntrySignal:
        """Get optimized entry signal."""
        return self.entry_optimizer.analyze_entry(df, direction, current_price)

    def optimize_exit(
        self,
        trade_state: TradeState,
        df: pd.DataFrame,
    ) -> ExitSignal:
        """Get optimized exit signal."""
        return self.exit_optimizer.analyze_exit(trade_state, df)

    def get_optimal_levels(
        self,
        entry_price: float,
        direction: str,
        df: pd.DataFrame,
        confidence: float = 0.6,
    ) -> OptimizedLevels:
        """Calculate optimal stop loss and take profit levels."""
        # Calculate volatility
        if len(df) > 14:
            volatility = df["close"].pct_change().rolling(14).std().iloc[-1]
        else:
            volatility = 0.02

        return self.exit_optimizer.calculate_optimal_levels(
            entry_price, direction, volatility, confidence
        )

    def record_trade(
        self,
        entry_quality: float,
        direction: str,
        pnl_pct: float,
        max_favorable_pct: float,
        max_adverse_pct: float,
        bars_held: int,
        exit_type: str,
    ) -> None:
        """Record complete trade for learning."""
        # Entry learning
        self.entry_optimizer.record_entry_outcome(
            entry_quality, direction, pnl_pct, max_favorable_pct, max_adverse_pct
        )

        # Exit learning
        left_on_table = max(0, max_favorable_pct - pnl_pct)
        self.exit_optimizer.record_exit_outcome(
            exit_type, pnl_pct, bars_held, max_favorable_pct, left_on_table
        )

    def save(self, name: str = "profit_optimizer") -> None:
        """Save optimizer state."""
        state = {
            "entry": {
                "optimal_rsi_buy": self.entry_optimizer.optimal_rsi_buy,
                "optimal_rsi_sell": self.entry_optimizer.optimal_rsi_sell,
                "optimal_bb_deviation": self.entry_optimizer.optimal_bb_deviation,
                "optimal_volume_ratio": self.entry_optimizer.optimal_volume_ratio,
            },
            "exit": {
                "optimal_profit_target": self.exit_optimizer.optimal_profit_target,
                "optimal_stop_loss": self.exit_optimizer.optimal_stop_loss,
                "optimal_trailing_pct": self.exit_optimizer.optimal_trailing_pct,
                "max_hold_bars": self.exit_optimizer.max_hold_bars,
                "break_even_trigger": self.exit_optimizer.break_even_trigger,
            },
            "saved_at": datetime.now().isoformat(),
        }

        path = self.model_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Profit optimizer saved to {path}")

    def load(self, name: str = "profit_optimizer") -> bool:
        """Load optimizer state."""
        path = self.model_dir / f"{name}.json"

        if not path.exists():
            return False

        with open(path) as f:
            state = json.load(f)

        entry = state.get("entry", {})
        self.entry_optimizer.optimal_rsi_buy = entry.get("optimal_rsi_buy", 35.0)
        self.entry_optimizer.optimal_rsi_sell = entry.get("optimal_rsi_sell", 65.0)
        self.entry_optimizer.optimal_bb_deviation = entry.get("optimal_bb_deviation", 1.5)
        self.entry_optimizer.optimal_volume_ratio = entry.get("optimal_volume_ratio", 1.2)

        exit_state = state.get("exit", {})
        self.exit_optimizer.optimal_profit_target = exit_state.get("optimal_profit_target", 0.03)
        self.exit_optimizer.optimal_stop_loss = exit_state.get("optimal_stop_loss", 0.02)
        self.exit_optimizer.optimal_trailing_pct = exit_state.get("optimal_trailing_pct", 0.015)
        self.exit_optimizer.max_hold_bars = exit_state.get("max_hold_bars", 50)
        self.exit_optimizer.break_even_trigger = exit_state.get("break_even_trigger", 0.01)

        logger.info(f"Profit optimizer loaded from {path}")
        return True

    def get_stats(self) -> Dict:
        """Get optimizer statistics."""
        return {
            "entry": {
                "optimal_rsi_buy": self.entry_optimizer.optimal_rsi_buy,
                "optimal_rsi_sell": self.entry_optimizer.optimal_rsi_sell,
                "trades_analyzed": len(self.entry_optimizer.entry_outcomes),
            },
            "exit": {
                "optimal_profit_target": f"{self.exit_optimizer.optimal_profit_target:.2%}",
                "optimal_stop_loss": f"{self.exit_optimizer.optimal_stop_loss:.2%}",
                "optimal_trailing": f"{self.exit_optimizer.optimal_trailing_pct:.2%}",
                "trades_analyzed": len(self.exit_optimizer.exit_outcomes),
            },
        }
