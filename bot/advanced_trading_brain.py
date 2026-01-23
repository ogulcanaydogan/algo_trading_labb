"""
Advanced Trading Brain - Maximum Profit Optimization.

Specialized systems for:
1. Optimal Exit Timing - Find the perfect moment to close
2. Smart Leverage - Risk-adjusted leverage based on confidence
3. Short-Selling Specialist - Expert at profiting from downturns
4. Deep Reasoning Engine - Multi-step analysis for complex decisions
5. Profit Maximizer - Dynamic targets that adapt to market conditions
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market cycle phases."""

    ACCUMULATION = "accumulation"  # Smart money buying
    MARKUP = "markup"  # Uptrend
    DISTRIBUTION = "distribution"  # Smart money selling
    MARKDOWN = "markdown"  # Downtrend - BEST FOR SHORTS
    CAPITULATION = "capitulation"  # Panic selling - END of shorts


class ExitReason(Enum):
    """Reasons for exit signals."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    MOMENTUM_REVERSAL = "momentum_reversal"
    VOLUME_DIVERGENCE = "volume_divergence"
    TIME_DECAY = "time_decay"
    RISK_LIMIT = "risk_limit"
    PATTERN_COMPLETION = "pattern_completion"
    LIQUIDITY_GRAB = "liquidity_grab"


@dataclass
class ExitSignal:
    """Signal to exit a position."""

    should_exit: bool
    urgency: float  # 0-1, how urgent
    reason: ExitReason
    suggested_price: Optional[float]
    confidence: float
    potential_remaining: float  # Estimated remaining profit potential
    risk_of_reversal: float  # 0-1, probability of adverse move
    reasoning: str


@dataclass
class LeverageRecommendation:
    """Leverage recommendation with risk analysis."""

    recommended_leverage: float
    max_safe_leverage: float
    position_size_usd: float
    liquidation_price: float
    risk_reward_ratio: float
    confidence: float
    reasoning: str
    warnings: List[str]


@dataclass
class ShortOpportunity:
    """Short selling opportunity analysis."""

    symbol: str
    score: float  # 0-100, higher = better short
    entry_price: float
    target_price: float
    stop_loss: float
    expected_profit_pct: float
    risk_reward: float
    leverage_suggestion: float
    confidence: float

    # Analysis details
    trend_strength: float  # Negative = bearish
    volume_confirmation: bool
    resistance_levels: List[float]
    support_levels: List[float]
    catalysts: List[str]
    reasoning: str


@dataclass
class ProfitTarget:
    """Dynamic profit target."""

    target_price: float
    target_pct: float
    probability: float  # Probability of reaching
    time_estimate_hours: float
    is_aggressive: bool


class ExitOptimizer:
    """
    Finds optimal exit points for maximum profit.

    Uses multiple signals:
    - Momentum exhaustion
    - Volume divergence
    - Support/resistance levels
    - Pattern completion
    - Time-based decay
    - Risk/reward optimization
    """

    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.exit_history: List[Dict] = []

        # Parameters
        self.momentum_lookback = 14
        self.volume_lookback = 20
        self.max_hold_hours = 72  # For leveraged positions

    def analyze_exit(
        self,
        symbol: str,
        side: str,  # 'long' or 'short'
        entry_price: float,
        current_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
        leverage: float,
        market_data: Dict[str, Any],
    ) -> ExitSignal:
        """
        Analyze if position should be exited.

        Returns detailed exit signal with reasoning.
        """
        # Calculate current P&L
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        pnl_pct_leveraged = pnl_pct * leverage

        # Hold time
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

        # Get indicators from market data
        rsi = market_data.get("rsi", 50)
        macd = market_data.get("macd", 0)
        macd_signal = market_data.get("macd_signal", 0)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        atr = market_data.get("atr", current_price * 0.02)
        trend = market_data.get("trend", "neutral")

        # Initialize exit analysis
        should_exit = False
        urgency = 0.0
        reason = ExitReason.TAKE_PROFIT
        confidence = 0.5
        reasoning_parts = []

        # 1. Check stop loss
        if side == "long" and current_price <= stop_loss:
            should_exit = True
            urgency = 1.0
            reason = ExitReason.STOP_LOSS
            confidence = 0.95
            reasoning_parts.append(f"Price hit stop loss ({stop_loss})")
        elif side == "short" and current_price >= stop_loss:
            should_exit = True
            urgency = 1.0
            reason = ExitReason.STOP_LOSS
            confidence = 0.95
            reasoning_parts.append(f"Price hit stop loss ({stop_loss})")

        # 2. Check take profit
        if side == "long" and current_price >= take_profit:
            should_exit = True
            urgency = 0.8
            reason = ExitReason.TAKE_PROFIT
            confidence = 0.9
            reasoning_parts.append(f"Price hit take profit ({take_profit})")
        elif side == "short" and current_price <= take_profit:
            should_exit = True
            urgency = 0.8
            reason = ExitReason.TAKE_PROFIT
            confidence = 0.9
            reasoning_parts.append(f"Price hit take profit ({take_profit})")

        # 3. Momentum reversal detection
        momentum_reversal = self._detect_momentum_reversal(side, rsi, macd, macd_signal, trend)
        if momentum_reversal["detected"] and pnl_pct > 0:
            if not should_exit:
                should_exit = True
                urgency = 0.7
                reason = ExitReason.MOMENTUM_REVERSAL
            confidence = max(confidence, momentum_reversal["confidence"])
            reasoning_parts.append(momentum_reversal["reason"])

        # 4. Volume divergence (price up but volume down = weakness)
        volume_divergence = self._detect_volume_divergence(side, pnl_pct, volume_ratio)
        if volume_divergence["detected"] and pnl_pct > 0.02:
            if not should_exit and pnl_pct > 0.03:  # Take profit on divergence if in profit
                should_exit = True
                urgency = 0.6
                reason = ExitReason.VOLUME_DIVERGENCE
            reasoning_parts.append(volume_divergence["reason"])

        # 5. Time decay for leveraged positions
        if leverage > 1 and hold_hours > self.max_hold_hours:
            if pnl_pct > 0 or hold_hours > self.max_hold_hours * 1.5:
                if not should_exit:
                    should_exit = True
                    urgency = 0.5
                    reason = ExitReason.TIME_DECAY
                reasoning_parts.append(
                    f"Position held {hold_hours:.1f}h (max recommended: {self.max_hold_hours}h)"
                )

        # 6. Risk limit check (for high leverage)
        if leverage >= 10 and pnl_pct_leveraged < -0.3:  # -30% leveraged loss
            should_exit = True
            urgency = 0.9
            reason = ExitReason.RISK_LIMIT
            confidence = 0.95
            reasoning_parts.append(f"Risk limit hit: {pnl_pct_leveraged * 100:.1f}% leveraged loss")

        # 7. Pattern completion (extended move)
        if abs(pnl_pct) > 0.1:  # 10% move
            extension_risk = self._calculate_extension_risk(pnl_pct, atr, current_price)
            if extension_risk > 0.7:
                if not should_exit and pnl_pct > 0:
                    should_exit = True
                    urgency = 0.6
                    reason = ExitReason.PATTERN_COMPLETION
                reasoning_parts.append(
                    f"Extended move ({pnl_pct * 100:.1f}%), reversal risk: {extension_risk:.0%}"
                )

        # Calculate remaining potential
        if side == "long":
            remaining_to_tp = (
                (take_profit - current_price) / current_price if current_price < take_profit else 0
            )
        else:
            remaining_to_tp = (
                (current_price - take_profit) / current_price if current_price > take_profit else 0
            )

        # Risk of reversal
        risk_of_reversal = self._calculate_reversal_risk(side, rsi, trend, pnl_pct, volume_ratio)

        return ExitSignal(
            should_exit=should_exit,
            urgency=urgency,
            reason=reason,
            suggested_price=current_price if should_exit else None,
            confidence=confidence,
            potential_remaining=remaining_to_tp,
            risk_of_reversal=risk_of_reversal,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "No exit signal",
        )

    def _detect_momentum_reversal(
        self, side: str, rsi: float, macd: float, macd_signal: float, trend: str
    ) -> Dict:
        """Detect momentum reversal signals."""
        detected = False
        confidence = 0.5
        reason = ""

        if side == "long":
            # For longs, bearish signals are reversals
            if rsi > 70:
                detected = True
                confidence = 0.6 + (rsi - 70) / 100
                reason = f"RSI overbought ({rsi:.1f})"
            if macd < macd_signal and macd > 0:  # Bearish crossover
                detected = True
                confidence = max(confidence, 0.65)
                reason = f"MACD bearish crossover"
            if trend == "down":
                confidence += 0.1
        else:  # short
            # For shorts, bullish signals are reversals
            if rsi < 30:
                detected = True
                confidence = 0.6 + (30 - rsi) / 100
                reason = f"RSI oversold ({rsi:.1f})"
            if macd > macd_signal and macd < 0:  # Bullish crossover
                detected = True
                confidence = max(confidence, 0.65)
                reason = f"MACD bullish crossover"
            if trend == "up":
                confidence += 0.1

        return {"detected": detected, "confidence": min(0.9, confidence), "reason": reason}

    def _detect_volume_divergence(self, side: str, pnl_pct: float, volume_ratio: float) -> Dict:
        """Detect volume divergence (price/volume disagreement)."""
        detected = False
        reason = ""

        if side == "long" and pnl_pct > 0:
            # Long in profit but volume declining = bearish divergence
            if volume_ratio < 0.7:
                detected = True
                reason = f"Bearish volume divergence (ratio: {volume_ratio:.2f})"
        elif side == "short" and pnl_pct > 0:
            # Short in profit but selling volume declining
            if volume_ratio < 0.7:
                detected = True
                reason = f"Bullish volume divergence (ratio: {volume_ratio:.2f})"

        return {"detected": detected, "reason": reason}

    def _calculate_extension_risk(self, pnl_pct: float, atr: float, price: float) -> float:
        """Calculate risk of mean reversion after extended move."""
        atr_pct = atr / price
        move_in_atrs = abs(pnl_pct) / atr_pct

        # Higher risk when move exceeds 3 ATRs
        if move_in_atrs > 5:
            return 0.9
        elif move_in_atrs > 3:
            return 0.7
        elif move_in_atrs > 2:
            return 0.5
        return 0.3

    def _calculate_reversal_risk(
        self, side: str, rsi: float, trend: str, pnl_pct: float, volume_ratio: float
    ) -> float:
        """Calculate probability of adverse price reversal."""
        risk = 0.3  # Base risk

        # RSI extremes increase reversal risk
        if side == "long":
            if rsi > 80:
                risk += 0.3
            elif rsi > 70:
                risk += 0.15
        else:
            if rsi < 20:
                risk += 0.3
            elif rsi < 30:
                risk += 0.15

        # Extended moves increase risk
        if abs(pnl_pct) > 0.1:
            risk += 0.2
        elif abs(pnl_pct) > 0.05:
            risk += 0.1

        # Low volume increases risk
        if volume_ratio < 0.5:
            risk += 0.15

        # Trend against position
        if (side == "long" and trend == "down") or (side == "short" and trend == "up"):
            risk += 0.1

        return min(0.95, risk)


class LeverageManager:
    """
    Smart leverage management based on:
    - Signal confidence
    - Market volatility
    - Account risk limits
    - Position correlation
    - Historical performance
    """

    def __init__(
        self,
        max_leverage: float = 20.0,
        base_risk_per_trade: float = 0.02,  # 2% of account
        max_account_risk: float = 0.10,  # 10% total exposure
    ):
        self.max_leverage = max_leverage
        self.base_risk_per_trade = base_risk_per_trade
        self.max_account_risk = max_account_risk

        # Performance tracking
        self.trade_history: List[Dict] = []
        self.leverage_performance: Dict[float, Dict] = {}  # leverage -> {wins, losses, pnl}

    def calculate_optimal_leverage(
        self,
        signal_confidence: float,
        volatility: float,  # ATR as % of price
        account_balance: float,
        current_exposure: float,  # Current open positions value
        win_rate: float,  # Recent win rate
        regime: str,
        is_short: bool = False,
    ) -> LeverageRecommendation:
        """
        Calculate optimal leverage for a trade.

        Uses Kelly Criterion modified for crypto/forex.
        """
        warnings = []

        # 1. Base leverage from confidence (1x to 5x)
        confidence_leverage = 1 + (signal_confidence - 0.5) * 8
        confidence_leverage = max(1, min(5, confidence_leverage))

        # 2. Adjust for volatility (higher vol = lower leverage)
        vol_multiplier = 1.0
        if volatility > 0.05:  # >5% daily volatility
            vol_multiplier = 0.3
            warnings.append("High volatility - leverage reduced")
        elif volatility > 0.03:
            vol_multiplier = 0.5
        elif volatility > 0.02:
            vol_multiplier = 0.7
        elif volatility < 0.01:
            vol_multiplier = 1.2  # Low vol can support higher leverage

        # 3. Adjust for win rate (Kelly-inspired)
        # f* = (bp - q) / b where b=1, p=win_rate, q=1-p
        # Simplified: leverage scales with edge
        edge = win_rate - 0.5
        kelly_multiplier = 1 + edge * 2  # Range 0.5 to 1.5
        kelly_multiplier = max(0.5, min(1.5, kelly_multiplier))

        # 4. Regime adjustments
        regime_multiplier = 1.0
        if regime in ["crash", "capitulation"]:
            regime_multiplier = 0.3
            warnings.append(f"Dangerous regime ({regime}) - leverage heavily reduced")
        elif regime in ["volatile", "bear"]:
            regime_multiplier = 0.5
        elif regime == "bull":
            regime_multiplier = 1.2

        # 5. Short positions can use slightly higher leverage in downtrends
        short_bonus = 1.0
        if is_short and regime in ["bear", "markdown", "distribution"]:
            short_bonus = 1.3
        elif is_short and regime in ["bull", "markup"]:
            short_bonus = 0.7  # Reduce leverage for shorts in uptrend
            warnings.append("Shorting in uptrend - leverage reduced")

        # 6. Calculate final leverage
        calculated_leverage = (
            confidence_leverage
            * vol_multiplier
            * kelly_multiplier
            * regime_multiplier
            * short_bonus
        )

        # 7. Apply maximum limits
        recommended_leverage = min(self.max_leverage, max(1, calculated_leverage))

        # 8. Check account risk limits
        available_risk = self.max_account_risk - (current_exposure / account_balance)
        if available_risk < self.base_risk_per_trade:
            warnings.append("Near max account exposure - position size limited")
            recommended_leverage = min(recommended_leverage, 2.0)

        # 9. Calculate position size
        risk_amount = account_balance * self.base_risk_per_trade
        position_size = risk_amount * recommended_leverage

        # 10. Calculate liquidation price (simplified)
        # For 10x leverage, 10% move = liquidation
        maintenance_margin = 0.05  # 5% typical
        liquidation_distance = (1 / recommended_leverage) - maintenance_margin

        # 11. Risk/reward calculation
        # Assuming 2:1 target
        expected_profit = position_size * 0.02 * 2  # 2% move * 2 R:R
        expected_loss = position_size * 0.02
        risk_reward = expected_profit / expected_loss if expected_loss > 0 else 0

        return LeverageRecommendation(
            recommended_leverage=round(recommended_leverage, 1),
            max_safe_leverage=min(self.max_leverage, recommended_leverage * 1.5),
            position_size_usd=round(position_size, 2),
            liquidation_price=liquidation_distance,  # As percentage
            risk_reward_ratio=round(risk_reward, 2),
            confidence=signal_confidence,
            reasoning=f"Base {confidence_leverage:.1f}x × Vol {vol_multiplier:.1f} × Kelly {kelly_multiplier:.1f} × Regime {regime_multiplier:.1f}",
            warnings=warnings,
        )

    def record_trade_result(self, leverage: float, pnl: float, won: bool):
        """Record trade result for leverage optimization."""
        self.trade_history.append(
            {"leverage": leverage, "pnl": pnl, "won": won, "timestamp": datetime.now().isoformat()}
        )

        # Update leverage performance stats
        lev_bucket = round(leverage)
        if lev_bucket not in self.leverage_performance:
            self.leverage_performance[lev_bucket] = {"wins": 0, "losses": 0, "pnl": 0}

        self.leverage_performance[lev_bucket]["pnl"] += pnl
        if won:
            self.leverage_performance[lev_bucket]["wins"] += 1
        else:
            self.leverage_performance[lev_bucket]["losses"] += 1

    def get_best_leverage_range(self) -> Tuple[float, float]:
        """Get the leverage range with best historical performance."""
        if not self.leverage_performance:
            return (2.0, 5.0)  # Default

        best_lev = None
        best_score = -float("inf")

        for lev, stats in self.leverage_performance.items():
            total = stats["wins"] + stats["losses"]
            if total < 5:
                continue

            win_rate = stats["wins"] / total
            avg_pnl = stats["pnl"] / total
            score = win_rate * 0.5 + (avg_pnl / 100) * 0.5

            if score > best_score:
                best_score = score
                best_lev = lev

        if best_lev is None:
            return (2.0, 5.0)

        return (max(1, best_lev - 1), min(self.max_leverage, best_lev + 2))


class ShortSellingSpecialist:
    """
    Expert system for short selling.

    Specializes in:
    - Identifying distribution phases
    - Detecting weakness in rallies
    - Finding optimal short entries
    - Managing short-specific risks (squeezes, funding)
    """

    def __init__(self):
        self.short_history: List[Dict] = []
        self.squeeze_detector = SqueezeDetector()

    def analyze_short_opportunity(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict[str, Any],
        orderbook_data: Optional[Dict] = None,
    ) -> ShortOpportunity:
        """
        Analyze if symbol presents good short opportunity.

        Checks:
        - Trend exhaustion
        - Distribution patterns
        - Resistance levels
        - Volume profile
        - Squeeze risk
        """
        # Extract market data
        rsi = market_data.get("rsi", 50)
        macd = market_data.get("macd", 0)
        macd_signal = market_data.get("macd_signal", 0)
        trend = market_data.get("trend", "neutral")
        volatility = market_data.get("volatility", 0.02)
        volume_ratio = market_data.get("volume_ratio", 1.0)

        # Recent price action
        high_24h = market_data.get("high_24h", current_price * 1.02)
        low_24h = market_data.get("low_24h", current_price * 0.98)
        change_24h = market_data.get("change_24h", 0)

        # Calculate scores
        scores = {}
        catalysts = []
        reasoning_parts = []

        # 1. Trend Score (bearish trend = good for shorts)
        if trend == "down":
            scores["trend"] = 80
            catalysts.append("Bearish trend")
        elif trend == "neutral":
            scores["trend"] = 50
        else:
            scores["trend"] = 20

        # 2. Momentum Score (overbought = short opportunity)
        if rsi > 70:
            scores["momentum"] = 70 + (rsi - 70)
            catalysts.append(f"Overbought RSI ({rsi:.0f})")
            reasoning_parts.append(f"RSI overbought at {rsi:.1f}")
        elif rsi > 60:
            scores["momentum"] = 50 + (rsi - 60) * 2
        else:
            scores["momentum"] = 30

        # 3. MACD Score (bearish crossover)
        if macd < macd_signal:
            scores["macd"] = 70
            if macd > 0:  # Crossover from above
                scores["macd"] = 85
                catalysts.append("MACD bearish crossover")
        else:
            scores["macd"] = 30

        # 4. Price Action Score (failed rally, lower highs)
        price_position = (
            (current_price - low_24h) / (high_24h - low_24h) if high_24h != low_24h else 0.5
        )
        if price_position > 0.8:  # Near highs
            scores["price_action"] = 75  # Good short entry
            reasoning_parts.append("Price near 24h high - potential reversal zone")
        elif price_position < 0.3:
            scores["price_action"] = 30  # Already dropped
        else:
            scores["price_action"] = 50

        # 5. Volume Score (declining on rallies = distribution)
        if change_24h > 0 and volume_ratio < 0.8:
            scores["volume"] = 75
            catalysts.append("Rising price on declining volume (distribution)")
        elif change_24h < 0 and volume_ratio > 1.2:
            scores["volume"] = 80
            catalysts.append("High volume selling")
        else:
            scores["volume"] = 50

        # 6. Squeeze Risk Score (lower = better)
        squeeze_risk = self.squeeze_detector.assess_risk(symbol, market_data, orderbook_data)
        scores["squeeze_safety"] = 100 - squeeze_risk * 100
        if squeeze_risk > 0.6:
            reasoning_parts.append(f"Warning: High squeeze risk ({squeeze_risk:.0%})")

        # Calculate overall score
        weights = {
            "trend": 0.25,
            "momentum": 0.2,
            "macd": 0.15,
            "price_action": 0.15,
            "volume": 0.15,
            "squeeze_safety": 0.1,
        }

        overall_score = sum(scores[k] * weights[k] for k in weights)

        # Determine entry, target, stop
        atr = current_price * volatility

        entry_price = current_price
        target_price = current_price * (1 - volatility * 2)  # 2 ATR target
        stop_loss = current_price * (1 + volatility * 1.5)  # 1.5 ATR stop

        expected_profit_pct = (entry_price - target_price) / entry_price
        expected_loss_pct = (stop_loss - entry_price) / entry_price
        risk_reward = expected_profit_pct / expected_loss_pct if expected_loss_pct > 0 else 0

        # Leverage suggestion based on score
        if overall_score > 75:
            leverage = 5.0
        elif overall_score > 60:
            leverage = 3.0
        elif overall_score > 50:
            leverage = 2.0
        else:
            leverage = 1.0

        # Reduce leverage if squeeze risk is high
        if squeeze_risk > 0.5:
            leverage *= 0.5

        # Confidence
        confidence = overall_score / 100

        # Calculate support/resistance
        resistance_levels = [high_24h, current_price * 1.02, current_price * 1.05]
        support_levels = [low_24h, current_price * 0.98, current_price * 0.95]

        return ShortOpportunity(
            symbol=symbol,
            score=overall_score,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_profit_pct=expected_profit_pct,
            risk_reward=risk_reward,
            leverage_suggestion=leverage,
            confidence=confidence,
            trend_strength=-1 if trend == "down" else (0 if trend == "neutral" else 1),
            volume_confirmation=volume_ratio > 1.0 or (change_24h > 0 and volume_ratio < 0.8),
            resistance_levels=resistance_levels,
            support_levels=support_levels,
            catalysts=catalysts,
            reasoning=" | ".join(reasoning_parts)
            if reasoning_parts
            else f"Score: {overall_score:.0f}/100",
        )

    def detect_market_phase(self, market_data: Dict[str, Any]) -> MarketPhase:
        """Detect current market cycle phase."""
        trend = market_data.get("trend", "neutral")
        volume_trend = market_data.get("volume_trend", "stable")
        rsi = market_data.get("rsi", 50)

        # Simplified phase detection
        if trend == "up" and volume_trend == "increasing":
            return MarketPhase.MARKUP
        elif trend == "up" and volume_trend == "decreasing":
            return MarketPhase.DISTRIBUTION  # Smart money selling
        elif trend == "down" and volume_trend == "increasing":
            return MarketPhase.MARKDOWN  # Best for shorts
        elif trend == "down" and rsi < 20:
            return MarketPhase.CAPITULATION  # End of shorts
        elif trend == "neutral" and volume_trend == "decreasing":
            return MarketPhase.ACCUMULATION
        else:
            return MarketPhase.ACCUMULATION


class SqueezeDetector:
    """Detects short squeeze potential."""

    def assess_risk(
        self, symbol: str, market_data: Dict[str, Any], orderbook_data: Optional[Dict] = None
    ) -> float:
        """
        Assess short squeeze risk (0-1).

        Factors:
        - Funding rate (high negative = many shorts)
        - Open interest changes
        - Order book imbalance
        - Recent short liquidations
        """
        risk = 0.3  # Base risk

        # Funding rate (if available)
        funding_rate = market_data.get("funding_rate", 0)
        if funding_rate < -0.01:  # Negative funding = shorts paying longs
            risk += 0.2

        # Volume surge can trigger squeezes
        volume_ratio = market_data.get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            risk += 0.15

        # RSI oversold can trigger bounces
        rsi = market_data.get("rsi", 50)
        if rsi < 25:
            risk += 0.2
        elif rsi < 30:
            risk += 0.1

        # Order book imbalance (more bids = squeeze risk)
        if orderbook_data:
            bid_volume = orderbook_data.get("bid_volume", 0)
            ask_volume = orderbook_data.get("ask_volume", 0)
            if bid_volume > ask_volume * 1.5:
                risk += 0.15

        return min(1.0, risk)


class ProfitMaximizer:
    """
    Dynamically adjusts profit targets based on market conditions.

    Features:
    - Multiple take-profit levels
    - Trailing mechanisms
    - Momentum-based extensions
    - Time-weighted targets
    """

    def __init__(self):
        self.target_history: List[Dict] = []

    def calculate_targets(
        self,
        entry_price: float,
        side: str,  # 'long' or 'short'
        confidence: float,
        volatility: float,
        regime: str,
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> List[ProfitTarget]:
        """
        Calculate multiple profit targets.

        Returns targets from conservative to aggressive.
        """
        targets = []

        # Base ATR
        atr = entry_price * volatility

        # Adjust for confidence
        confidence_multiplier = 1 + (confidence - 0.5) * 0.5  # 0.75 to 1.25

        # Adjust for regime
        if regime == "bull" and side == "long":
            regime_multiplier = 1.3
        elif regime == "bear" and side == "short":
            regime_multiplier = 1.3
        elif regime in ["crash", "volatile"]:
            regime_multiplier = 0.7
        else:
            regime_multiplier = 1.0

        # Target levels
        if side == "long":
            # Conservative target (1 ATR)
            tp1 = entry_price + atr * 1.0 * confidence_multiplier * regime_multiplier
            targets.append(
                ProfitTarget(
                    target_price=tp1,
                    target_pct=(tp1 - entry_price) / entry_price,
                    probability=0.7,
                    time_estimate_hours=4,
                    is_aggressive=False,
                )
            )

            # Standard target (2 ATR)
            tp2 = entry_price + atr * 2.0 * confidence_multiplier * regime_multiplier
            targets.append(
                ProfitTarget(
                    target_price=tp2,
                    target_pct=(tp2 - entry_price) / entry_price,
                    probability=0.5,
                    time_estimate_hours=12,
                    is_aggressive=False,
                )
            )

            # Aggressive target (3 ATR or next resistance)
            tp3 = entry_price + atr * 3.0 * confidence_multiplier * regime_multiplier
            # Check if there's a resistance level close
            for res in sorted(resistance_levels):
                if res > tp2 and res < tp3:
                    tp3 = res * 0.99  # Just below resistance
                    break

            targets.append(
                ProfitTarget(
                    target_price=tp3,
                    target_pct=(tp3 - entry_price) / entry_price,
                    probability=0.3,
                    time_estimate_hours=24,
                    is_aggressive=True,
                )
            )

        else:  # short
            # Conservative target (1 ATR down)
            tp1 = entry_price - atr * 1.0 * confidence_multiplier * regime_multiplier
            targets.append(
                ProfitTarget(
                    target_price=tp1,
                    target_pct=(entry_price - tp1) / entry_price,
                    probability=0.7,
                    time_estimate_hours=4,
                    is_aggressive=False,
                )
            )

            # Standard target (2 ATR down)
            tp2 = entry_price - atr * 2.0 * confidence_multiplier * regime_multiplier
            targets.append(
                ProfitTarget(
                    target_price=tp2,
                    target_pct=(entry_price - tp2) / entry_price,
                    probability=0.5,
                    time_estimate_hours=12,
                    is_aggressive=False,
                )
            )

            # Aggressive target (3 ATR or next support)
            tp3 = entry_price - atr * 3.0 * confidence_multiplier * regime_multiplier
            for sup in sorted(support_levels, reverse=True):
                if sup < tp2 and sup > tp3:
                    tp3 = sup * 1.01  # Just above support
                    break

            targets.append(
                ProfitTarget(
                    target_price=tp3,
                    target_pct=(entry_price - tp3) / entry_price,
                    probability=0.3,
                    time_estimate_hours=24,
                    is_aggressive=True,
                )
            )

        return targets

    def get_trailing_stop_params(
        self, current_profit_pct: float, volatility: float, side: str
    ) -> Dict[str, float]:
        """
        Get dynamic trailing stop parameters.

        As profit increases, trailing stop tightens.
        """
        atr_pct = volatility

        # Base trailing distance
        if current_profit_pct < 0.02:
            # Small profit - wide trailing
            trailing_pct = atr_pct * 2.0
            activation_pct = 0.01
        elif current_profit_pct < 0.05:
            # Medium profit - standard trailing
            trailing_pct = atr_pct * 1.5
            activation_pct = 0.02
        elif current_profit_pct < 0.10:
            # Good profit - tighter trailing
            trailing_pct = atr_pct * 1.0
            activation_pct = 0.03
        else:
            # Large profit - very tight trailing
            trailing_pct = atr_pct * 0.75
            activation_pct = 0.05

        return {
            "trailing_pct": trailing_pct,
            "activation_pct": activation_pct,
            "step_pct": atr_pct * 0.25,  # Move stop every 0.25 ATR
        }


class DeepReasoningEngine:
    """
    Multi-step reasoning for complex trading decisions.

    Performs deep analysis:
    1. Gather all data points
    2. Identify patterns and anomalies
    3. Generate hypotheses
    4. Test against historical patterns
    5. Synthesize final decision with confidence
    """

    def __init__(self):
        self.reasoning_history: List[Dict] = []

    async def deep_analyze(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict[str, Any],
        intelligence_data: Dict[str, Any],
        ml_signal: Dict[str, Any],
        existing_positions: List[Dict],
    ) -> Dict[str, Any]:
        """
        Perform deep multi-step analysis.

        Returns comprehensive trading decision with reasoning.
        """
        reasoning_steps = []

        # Step 1: Data Gathering
        step1 = {"step": "Data Collection", "findings": []}

        # Technical data
        rsi = market_data.get("rsi", 50)
        trend = market_data.get("trend", "neutral")
        volatility = market_data.get("volatility", 0.02)
        regime = market_data.get("regime", "unknown")

        step1["findings"].append(f"RSI: {rsi:.1f}")
        step1["findings"].append(f"Trend: {trend}")
        step1["findings"].append(f"Volatility: {volatility * 100:.1f}%")
        step1["findings"].append(f"Regime: {regime}")

        # Intelligence data
        sentiment = intelligence_data.get("sentiment", {})
        fear_greed = sentiment.get("fear_greed", 50)
        news_sentiment = sentiment.get("news", 0)

        step1["findings"].append(f"Fear/Greed: {fear_greed}")
        step1["findings"].append(f"News Sentiment: {news_sentiment:.2f}")

        reasoning_steps.append(step1)

        # Step 2: Pattern Recognition
        step2 = {"step": "Pattern Recognition", "patterns": []}

        # Identify patterns
        if rsi > 70 and trend == "up":
            step2["patterns"].append("Overbought in uptrend - potential reversal")
        elif rsi < 30 and trend == "down":
            step2["patterns"].append("Oversold in downtrend - potential bounce")

        if fear_greed < 25:
            step2["patterns"].append("Extreme fear - contrarian buy signal")
        elif fear_greed > 75:
            step2["patterns"].append("Extreme greed - contrarian sell signal")

        if news_sentiment < -0.5:
            step2["patterns"].append("Very negative news sentiment")
        elif news_sentiment > 0.5:
            step2["patterns"].append("Very positive news sentiment")

        reasoning_steps.append(step2)

        # Step 3: Hypothesis Generation
        step3 = {"step": "Hypothesis Generation", "hypotheses": []}

        # Generate hypotheses based on patterns
        bullish_points = 0
        bearish_points = 0

        # Trend following
        if trend == "up":
            bullish_points += 2
            step3["hypotheses"].append("H1: Trend continuation (bullish)")
        elif trend == "down":
            bearish_points += 2
            step3["hypotheses"].append("H1: Trend continuation (bearish)")

        # Mean reversion at extremes
        if rsi > 70:
            bearish_points += 1
            step3["hypotheses"].append("H2: RSI overbought - mean reversion likely")
        elif rsi < 30:
            bullish_points += 1
            step3["hypotheses"].append("H2: RSI oversold - bounce likely")

        # Sentiment contrarian
        if fear_greed < 30:
            bullish_points += 1
            step3["hypotheses"].append("H3: Extreme fear - contrarian opportunity")
        elif fear_greed > 70:
            bearish_points += 1
            step3["hypotheses"].append("H3: Extreme greed - caution warranted")

        reasoning_steps.append(step3)

        # Step 4: Decision Synthesis
        step4 = {"step": "Decision Synthesis", "analysis": []}

        # Calculate net score
        net_score = bullish_points - bearish_points

        if net_score > 2:
            action = "BUY"
            confidence = 0.6 + min(0.3, net_score * 0.05)
            step4["analysis"].append(f"Strong bullish case ({bullish_points} vs {bearish_points})")
        elif net_score < -2:
            action = "SELL"
            confidence = 0.6 + min(0.3, abs(net_score) * 0.05)
            step4["analysis"].append(f"Strong bearish case ({bearish_points} vs {bullish_points})")
        elif net_score > 0:
            action = "BUY"
            confidence = 0.5 + net_score * 0.05
            step4["analysis"].append(f"Slight bullish bias ({bullish_points} vs {bearish_points})")
        elif net_score < 0:
            action = "SELL"
            confidence = 0.5 + abs(net_score) * 0.05
            step4["analysis"].append(f"Slight bearish bias ({bearish_points} vs {bullish_points})")
        else:
            action = "HOLD"
            confidence = 0.5
            step4["analysis"].append("Neutral - no clear edge")

        # Adjust for ML signal agreement
        ml_action = ml_signal.get("action", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0.5)

        if ml_action == action:
            confidence = min(0.95, confidence + 0.1)
            step4["analysis"].append(f"ML agrees ({ml_action} @ {ml_confidence:.0%})")
        elif ml_action != "HOLD" and action != "HOLD":
            confidence = max(0.3, confidence - 0.15)
            step4["analysis"].append(f"ML disagrees ({ml_action} @ {ml_confidence:.0%})")

        # Check existing positions
        for pos in existing_positions:
            if pos.get("symbol") == symbol:
                pos_side = pos.get("side", "long")
                if (pos_side == "long" and action == "SELL") or (
                    pos_side == "short" and action == "BUY"
                ):
                    step4["analysis"].append("Consider closing existing position")

        reasoning_steps.append(step4)

        # Final decision
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "bullish_points": bullish_points,
            "bearish_points": bearish_points,
            "summary": f"{action} with {confidence:.0%} confidence based on {len(step2['patterns'])} patterns",
        }


class AdvancedTradingBrain:
    """
    Orchestrates all advanced trading systems.

    Combines:
    - Exit Optimizer
    - Leverage Manager
    - Short Specialist
    - Profit Maximizer
    - Deep Reasoning
    """

    def __init__(self, max_leverage: float = 20.0):
        self.exit_optimizer = ExitOptimizer()
        self.leverage_manager = LeverageManager(max_leverage=max_leverage)
        self.short_specialist = ShortSellingSpecialist()
        self.profit_maximizer = ProfitMaximizer()
        self.reasoning_engine = DeepReasoningEngine()

        logger.info(f"Advanced Trading Brain initialized (max leverage: {max_leverage}x)")

    async def get_complete_trade_plan(
        self,
        symbol: str,
        current_price: float,
        account_balance: float,
        current_exposure: float,
        market_data: Dict[str, Any],
        intelligence_data: Dict[str, Any],
        ml_signal: Dict[str, Any],
        existing_positions: List[Dict],
    ) -> Dict[str, Any]:
        """
        Generate complete trade plan with all parameters.
        """
        # Deep reasoning analysis
        analysis = await self.reasoning_engine.deep_analyze(
            symbol, current_price, market_data, intelligence_data, ml_signal, existing_positions
        )

        action = analysis["action"]
        confidence = analysis["confidence"]

        # If no trade, return early
        if action == "HOLD":
            return {
                "action": "HOLD",
                "confidence": confidence,
                "reasoning": analysis["summary"],
                "reasoning_steps": analysis["reasoning_steps"],
            }

        # Determine if this is a short
        is_short = action in ["SELL", "SHORT"]

        # For shorts, get specialized analysis
        short_analysis = None
        if is_short:
            short_analysis = self.short_specialist.analyze_short_opportunity(
                symbol, current_price, market_data
            )
            # Adjust confidence based on short specialist
            confidence = (confidence + short_analysis.confidence) / 2

        # Get leverage recommendation
        leverage_rec = self.leverage_manager.calculate_optimal_leverage(
            signal_confidence=confidence,
            volatility=market_data.get("volatility", 0.02),
            account_balance=account_balance,
            current_exposure=current_exposure,
            win_rate=market_data.get("recent_win_rate", 0.5),
            regime=market_data.get("regime", "unknown"),
            is_short=is_short,
        )

        # Calculate profit targets
        volatility = market_data.get("volatility", 0.02)
        support_levels = (
            short_analysis.support_levels
            if short_analysis
            else [current_price * 0.95, current_price * 0.9]
        )
        resistance_levels = (
            short_analysis.resistance_levels
            if short_analysis
            else [current_price * 1.05, current_price * 1.1]
        )

        targets = self.profit_maximizer.calculate_targets(
            entry_price=current_price,
            side="short" if is_short else "long",
            confidence=confidence,
            volatility=volatility,
            regime=market_data.get("regime", "unknown"),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
        )

        # Calculate stop loss
        atr = current_price * volatility
        if is_short:
            stop_loss = current_price + atr * 1.5
            if short_analysis:
                stop_loss = short_analysis.stop_loss
        else:
            stop_loss = current_price - atr * 1.5

        return {
            "action": action,
            "confidence": confidence,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "targets": [
                {
                    "price": t.target_price,
                    "pct": t.target_pct * 100,
                    "probability": t.probability,
                    "aggressive": t.is_aggressive,
                }
                for t in targets
            ],
            "leverage": leverage_rec.recommended_leverage,
            "position_size_usd": leverage_rec.position_size_usd,
            "leverage_warnings": leverage_rec.warnings,
            "risk_reward": leverage_rec.risk_reward_ratio,
            "reasoning": analysis["summary"],
            "reasoning_steps": analysis["reasoning_steps"],
            "short_analysis": {
                "score": short_analysis.score,
                "catalysts": short_analysis.catalysts,
                "squeeze_risk": self.short_specialist.squeeze_detector.assess_risk(
                    symbol, market_data, None
                ),
            }
            if short_analysis
            else None,
        }

    def check_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
        leverage: float,
        market_data: Dict[str, Any],
    ) -> ExitSignal:
        """Check if position should be exited."""
        return self.exit_optimizer.analyze_exit(
            symbol,
            side,
            entry_price,
            current_price,
            entry_time,
            stop_loss,
            take_profit,
            leverage,
            market_data,
        )

    def record_trade(self, leverage: float, pnl: float, won: bool):
        """Record trade result for learning."""
        self.leverage_manager.record_trade_result(leverage, pnl, won)


# Global instance
_advanced_brain: Optional[AdvancedTradingBrain] = None


def get_advanced_trading_brain(max_leverage: float = 20.0) -> AdvancedTradingBrain:
    """Get or create the Advanced Trading Brain instance."""
    global _advanced_brain
    if _advanced_brain is None:
        _advanced_brain = AdvancedTradingBrain(max_leverage=max_leverage)
    return _advanced_brain
