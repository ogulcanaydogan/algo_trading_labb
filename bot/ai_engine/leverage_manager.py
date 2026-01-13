"""
AI Leverage Manager

Intelligent leverage selection and risk management for leveraged trading.

Features:
1. Dynamic leverage optimization based on market conditions
2. Liquidation protection
3. Kelly criterion position sizing
4. Volatility-adjusted leverage
5. Learning from historical performance
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .learning_db import LearningDatabase, get_learning_db
from .leverage_rl_agent import (
    LeverageRLAgent,
    LeverageState,
    LeverageAction,
    get_leverage_rl_agent,
)

logger = logging.getLogger(__name__)


@dataclass
class LeverageDecision:
    """Result of leverage optimization."""
    recommended_leverage: float
    direction: str  # "long", "short", "none"
    position_size_pct: float  # % of capital to use
    confidence: float
    reasoning: str
    risk_metrics: Dict[str, float]
    warnings: List[str]


@dataclass
class MarginStatus:
    """Current margin account status."""
    total_margin: float
    used_margin: float
    available_margin: float
    margin_ratio: float
    liquidation_price: Optional[float]
    distance_to_liquidation: float  # % away from liquidation
    unrealized_pnl: float


class AILeverageManager:
    """
    Manages leverage decisions using AI.

    Combines:
    1. RL agent for action selection
    2. Kelly criterion for position sizing
    3. Volatility scaling for risk management
    4. Historical performance tracking
    """

    def __init__(
        self,
        rl_agent: LeverageRLAgent = None,
        db: LearningDatabase = None,
        max_leverage: float = 10.0,
        min_leverage: float = 1.0,
        base_position_size: float = 0.1,  # 10% of capital
        max_position_size: float = 0.25,  # Max 25% per trade
        volatility_scale_factor: float = 0.5,
        kelly_fraction: float = 0.25,  # Use 25% of Kelly
        liquidation_buffer: float = 0.15,  # Stay 15% away from liquidation
    ):
        self.rl_agent = rl_agent or get_leverage_rl_agent()
        self.db = db or get_learning_db()
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.volatility_scale_factor = volatility_scale_factor
        self.kelly_fraction = kelly_fraction
        self.liquidation_buffer = liquidation_buffer

        # Performance tracking
        self.leverage_history: List[Dict] = []
        self.performance_by_leverage: Dict[float, List[float]] = {
            1.0: [], 2.0: [], 3.0: [], 5.0: [], 10.0: []
        }

    def calculate_optimal_leverage(
        self,
        indicators: Dict[str, float],
        current_position: Dict[str, float] = None,
        risk_budget: float = 0.02,  # Max 2% portfolio risk per trade
        account_balance: float = 10000,
        regime: str = "neutral",
    ) -> LeverageDecision:
        """
        Calculate optimal leverage for current market conditions.

        Args:
            indicators: Technical indicators
            current_position: Current position info
            risk_budget: Max risk per trade as fraction of portfolio
            account_balance: Total account value
            regime: Current market regime

        Returns:
            LeverageDecision with recommendation
        """
        pos = current_position or {}
        warnings = []

        # Create state for RL agent
        risk_info = {
            'drawdown': pos.get('drawdown', 0),
            'consecutive_losses': pos.get('consecutive_losses', 0),
            'win_rate_recent': pos.get('win_rate_recent', 0.5),
        }

        state = LeverageState.from_market_data(
            indicators=indicators,
            position_info={
                'position': pos.get('position', 0),
                'leverage': pos.get('leverage', 1),
                'pnl': pos.get('pnl', 0),
                'duration': pos.get('duration', 0),
                'unrealized_pnl': pos.get('unrealized_pnl', 0),
                'margin_ratio': pos.get('margin_ratio', 0),
            },
            risk_info=risk_info,
        )

        # Get RL agent analysis
        analysis = self.rl_agent.get_action_analysis(state)
        best_action = analysis['best_action']
        confidence = analysis['confidence']

        # Determine direction
        if LeverageAction.is_long(best_action):
            direction = "long"
        elif LeverageAction.is_short(best_action):
            direction = "short"
        elif best_action == LeverageAction.CLOSE:
            direction = "close"
        elif best_action == LeverageAction.REDUCE_HALF:
            direction = "reduce"
        else:
            direction = "none"

        # Get base leverage from RL
        rl_leverage = LeverageAction.get_leverage(best_action)

        # Adjust leverage based on multiple factors
        adjusted_leverage = self._adjust_leverage(
            base_leverage=rl_leverage,
            volatility=indicators.get('volatility_ratio', 1),
            trend_strength=indicators.get('trend_strength', 0.5),
            adx=indicators.get('adx', 25),
            confidence=confidence,
            regime=regime,
            margin_ratio=pos.get('margin_ratio', 0),
        )

        if adjusted_leverage < rl_leverage:
            warnings.append(f"Leverage reduced from {rl_leverage}x to {adjusted_leverage}x due to market conditions")

        # Calculate position size using Kelly criterion
        position_size = self._calculate_position_size(
            leverage=adjusted_leverage,
            confidence=confidence,
            volatility=indicators.get('volatility_ratio', 1),
            risk_budget=risk_budget,
            win_rate=risk_info.get('win_rate_recent', 0.5),
        )

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(
            leverage=adjusted_leverage,
            position_size=position_size,
            volatility=indicators.get('volatility_ratio', 1),
            account_balance=account_balance,
            direction=direction,
        )

        # Additional warnings
        if risk_metrics['max_loss_pct'] > 5:
            warnings.append(f"Max potential loss: {risk_metrics['max_loss_pct']:.1f}%")
        if adjusted_leverage >= 5 and indicators.get('volatility_ratio', 1) > 1.5:
            warnings.append("High leverage in volatile market - elevated risk")
        if direction == "short" and indicators.get('trend_direction', 0) > 0.5:
            warnings.append("Shorting against uptrend - contrarian position")

        # Build reasoning
        reasoning = self._build_reasoning(
            direction=direction,
            leverage=adjusted_leverage,
            confidence=confidence,
            indicators=indicators,
            analysis=analysis,
        )

        return LeverageDecision(
            recommended_leverage=adjusted_leverage,
            direction=direction,
            position_size_pct=position_size,
            confidence=confidence,
            reasoning=reasoning,
            risk_metrics=risk_metrics,
            warnings=warnings,
        )

    def _adjust_leverage(
        self,
        base_leverage: float,
        volatility: float,
        trend_strength: float,
        adx: float,
        confidence: float,
        regime: str,
        margin_ratio: float,
    ) -> float:
        """Adjust leverage based on market conditions."""
        leverage = base_leverage

        # Volatility adjustment: reduce leverage in high volatility
        if volatility > 1.5:
            vol_factor = 1.0 / (volatility * self.volatility_scale_factor)
            leverage *= vol_factor
        elif volatility < 0.7:
            # Can increase slightly in low vol
            leverage *= 1.1

        # Trend strength adjustment
        if trend_strength > 0.7:
            leverage *= 1.1  # Increase in strong trend
        elif trend_strength < 0.3:
            leverage *= 0.8  # Reduce in weak/no trend

        # ADX adjustment
        if adx > 40:  # Strong trend
            leverage *= 1.1
        elif adx < 20:  # Weak trend
            leverage *= 0.9

        # Confidence adjustment
        if confidence < 0.4:
            leverage *= 0.7
        elif confidence > 0.7:
            leverage *= 1.1

        # Regime adjustment
        regime_multipliers = {
            'strong_bull': 1.1,
            'bull': 1.0,
            'neutral': 0.9,
            'bear': 0.9,
            'strong_bear': 0.8,  # More conservative in strong bear
            'volatile': 0.7,
        }
        leverage *= regime_multipliers.get(regime, 1.0)

        # Margin safety
        if margin_ratio > 0.6:
            leverage *= (1 - margin_ratio)

        # Clamp to limits
        leverage = max(self.min_leverage, min(self.max_leverage, leverage))

        # Round to common leverage values
        common_leverages = [1, 2, 3, 5, 10]
        leverage = min(common_leverages, key=lambda x: abs(x - leverage))

        return float(leverage)

    def _calculate_position_size(
        self,
        leverage: float,
        confidence: float,
        volatility: float,
        risk_budget: float,
        win_rate: float,
    ) -> float:
        """
        Calculate position size using modified Kelly criterion.

        Kelly fraction = (p * b - q) / b
        where:
            p = win probability
            b = win/loss ratio
            q = 1 - p
        """
        # Estimate win/loss ratio from confidence
        estimated_win_loss_ratio = 1.0 + confidence

        p = win_rate
        q = 1 - p
        b = estimated_win_loss_ratio

        # Kelly fraction
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, kelly)  # Can't be negative

        # Use fraction of Kelly (more conservative)
        kelly_adjusted = kelly * self.kelly_fraction

        # Volatility adjustment
        vol_adjusted = kelly_adjusted / max(volatility, 0.5)

        # Leverage adjustment (higher leverage = smaller position)
        leverage_adjusted = vol_adjusted / math.sqrt(leverage)

        # Risk budget cap
        risk_capped = min(leverage_adjusted, risk_budget * 2)

        # Final clamps
        position_size = max(
            0.02,  # Min 2%
            min(self.max_position_size, risk_capped)
        )

        return round(position_size, 4)

    def _calculate_risk_metrics(
        self,
        leverage: float,
        position_size: float,
        volatility: float,
        account_balance: float,
        direction: str,
    ) -> Dict[str, float]:
        """Calculate risk metrics for the position."""
        position_value = account_balance * position_size * leverage

        # Estimated daily volatility (assuming volatility_ratio is relative to normal)
        daily_vol = 0.02 * volatility  # 2% base daily vol

        # Max loss scenarios
        one_atr_move = daily_vol * 100  # As percentage
        max_adverse_move = one_atr_move * 2  # 2 ATR adverse move
        max_loss_pct = max_adverse_move * leverage * position_size

        # Liquidation distance (simplified)
        liquidation_threshold = 0.8  # 80% margin used
        margin_per_leverage = 1 / leverage
        liquidation_distance = (1 - margin_per_leverage) * 100  # As percentage

        return {
            'position_value': round(position_value, 2),
            'daily_volatility_pct': round(daily_vol * 100, 2),
            'max_adverse_move_pct': round(max_adverse_move, 2),
            'max_loss_pct': round(max_loss_pct, 2),
            'max_loss_usd': round(account_balance * max_loss_pct / 100, 2),
            'liquidation_distance_pct': round(liquidation_distance, 2),
            'effective_leverage': round(leverage * position_size, 2),
            'risk_reward_ratio': round(1 / max(max_loss_pct, 0.1), 2),
        }

    def _build_reasoning(
        self,
        direction: str,
        leverage: float,
        confidence: float,
        indicators: Dict[str, float],
        analysis: Dict,
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = []

        # Direction reasoning
        if direction == "long":
            parts.append(f"LONG position recommended at {leverage}x leverage.")
            if indicators.get('trend_direction', 0) > 0.3:
                parts.append("Bullish trend detected.")
            if indicators.get('rsi', 50) < 40:
                parts.append("RSI indicates oversold conditions.")
        elif direction == "short":
            parts.append(f"SHORT position recommended at {leverage}x leverage.")
            if indicators.get('trend_direction', 0) < -0.3:
                parts.append("Bearish trend detected.")
            if indicators.get('rsi', 50) > 60:
                parts.append("RSI indicates overbought conditions.")
        elif direction == "close":
            parts.append("Close current position recommended.")
            parts.append("Risk/reward no longer favorable.")
        elif direction == "reduce":
            parts.append("Reduce position by 50% recommended.")
            parts.append("Elevated risk conditions detected.")
        else:
            parts.append("Hold current position. No clear opportunity.")

        # Confidence
        conf_level = "high" if confidence > 0.6 else "moderate" if confidence > 0.4 else "low"
        parts.append(f"Confidence: {conf_level} ({confidence:.0%}).")

        # Market conditions
        vol = indicators.get('volatility_ratio', 1)
        if vol > 1.5:
            parts.append("Note: High volatility environment.")
        elif vol < 0.7:
            parts.append("Note: Low volatility environment.")

        return " ".join(parts)

    def get_margin_status(
        self,
        account_info: Dict[str, float],
        position_info: Dict[str, float] = None,
    ) -> MarginStatus:
        """Calculate current margin status."""
        pos = position_info or {}

        total_margin = account_info.get('total_margin', 0)
        used_margin = account_info.get('used_margin', 0)
        available_margin = total_margin - used_margin
        margin_ratio = used_margin / total_margin if total_margin > 0 else 0

        # Calculate liquidation price (simplified)
        entry_price = pos.get('entry_price')
        leverage = pos.get('leverage', 1)
        position_side = pos.get('side', 'long')

        liquidation_price = None
        distance_to_liquidation = 100.0

        if entry_price and leverage > 1:
            maintenance_margin = 0.005  # 0.5% maintenance margin
            if position_side == 'long':
                liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
                current_price = pos.get('current_price', entry_price)
                distance_to_liquidation = ((current_price - liquidation_price) / current_price) * 100
            else:
                liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
                current_price = pos.get('current_price', entry_price)
                distance_to_liquidation = ((liquidation_price - current_price) / current_price) * 100

        return MarginStatus(
            total_margin=total_margin,
            used_margin=used_margin,
            available_margin=available_margin,
            margin_ratio=margin_ratio,
            liquidation_price=liquidation_price,
            distance_to_liquidation=max(0, distance_to_liquidation),
            unrealized_pnl=pos.get('unrealized_pnl', 0),
        )

    def should_reduce_leverage(
        self,
        margin_status: MarginStatus,
        volatility: float,
        consecutive_losses: int,
    ) -> Tuple[bool, str]:
        """Check if leverage should be reduced."""
        reasons = []

        if margin_status.margin_ratio > 0.7:
            reasons.append(f"High margin usage: {margin_status.margin_ratio:.1%}")

        if margin_status.distance_to_liquidation < 10:
            reasons.append(f"Close to liquidation: {margin_status.distance_to_liquidation:.1f}%")

        if volatility > 2.0:
            reasons.append(f"Extreme volatility: {volatility:.1f}x normal")

        if consecutive_losses >= 3:
            reasons.append(f"Consecutive losses: {consecutive_losses}")

        if margin_status.unrealized_pnl < -5:
            reasons.append(f"Large unrealized loss: {margin_status.unrealized_pnl:.1f}%")

        if reasons:
            return True, "; ".join(reasons)
        return False, "Position healthy"

    def record_trade_result(
        self,
        leverage: float,
        pnl_pct: float,
        direction: str,
        hold_duration: int,
        market_conditions: Dict[str, float],
    ):
        """Record trade result for learning."""
        self.leverage_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'leverage': leverage,
            'pnl_pct': pnl_pct,
            'direction': direction,
            'hold_duration': hold_duration,
            'market_conditions': market_conditions,
        })

        if leverage in self.performance_by_leverage:
            self.performance_by_leverage[leverage].append(pnl_pct)

        # Save to database
        try:
            self.db.record_optimization_result(
                symbol=market_conditions.get('symbol', 'unknown'),
                regime=market_conditions.get('regime', 'unknown'),
                parameters={'leverage': leverage, 'direction': direction},
                score=pnl_pct,
            )
        except Exception as e:
            logger.warning(f"Failed to record leverage trade: {e}")

    def get_leverage_performance_summary(self) -> Dict[str, Any]:
        """Get summary of leverage performance."""
        summary = {}

        for leverage, pnls in self.performance_by_leverage.items():
            if pnls:
                summary[f"{leverage}x"] = {
                    'trades': len(pnls),
                    'total_pnl': round(sum(pnls), 2),
                    'avg_pnl': round(np.mean(pnls), 2),
                    'win_rate': round(len([p for p in pnls if p > 0]) / len(pnls), 2),
                    'best_trade': round(max(pnls), 2),
                    'worst_trade': round(min(pnls), 2),
                    'sharpe_approx': round(
                        np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0, 2
                    ),
                }

        # Optimal leverage based on risk-adjusted returns
        best_leverage = None
        best_sharpe = float('-inf')
        for lev_str, stats in summary.items():
            if stats.get('sharpe_approx', 0) > best_sharpe and stats['trades'] >= 10:
                best_sharpe = stats['sharpe_approx']
                best_leverage = lev_str

        return {
            'by_leverage': summary,
            'optimal_leverage': best_leverage,
            'total_trades': len(self.leverage_history),
            'rl_agent_stats': self.rl_agent.get_stats(),
        }


# Global instance
_leverage_manager: Optional[AILeverageManager] = None


def get_leverage_manager() -> AILeverageManager:
    """Get or create global leverage manager."""
    global _leverage_manager
    if _leverage_manager is None:
        _leverage_manager = AILeverageManager()
    return _leverage_manager
