"""
Master AI Controller - Unified AI Trading System.

Integrates ALL AI systems into a single intelligent trading brain:
1. ML Predictor (technical analysis)
2. AI Brain V2 (reinforcement learning)
3. Data Intelligence (news, sentiment, on-chain)
4. Continuous Learner (adaptive improvement)
5. Advanced Trading Brain (exit, leverage, shorts, profit max)

This is the main entry point for all AI-driven trading decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MasterTradeDecision:
    """Complete trading decision from Master AI."""

    symbol: str
    timestamp: datetime

    # Action
    action: str  # BUY, SELL, HOLD
    confidence: float

    # Position sizing
    position_size_pct: float
    leverage: float
    position_size_usd: float

    # Risk management
    stop_loss: float
    stop_loss_pct: float

    # Profit targets
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    take_profit_pcts: List[float]

    # Trade management
    trailing_stop_enabled: bool
    trailing_stop_pct: float
    dca_enabled: bool
    max_hold_hours: int

    # Analysis
    regime: str
    trend: str
    volatility: str
    sentiment_score: float
    fear_greed: float
    short_score: Optional[float]
    squeeze_risk: Optional[float]

    # Reasoning
    reasoning: str
    reasoning_steps: List[Dict]
    catalysts: List[str]
    warnings: List[str]

    # Source signals
    ml_action: str
    ml_confidence: float
    rl_action: str
    rl_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "confidence": self.confidence,
            "position_size_pct": self.position_size_pct,
            "leverage": self.leverage,
            "position_size_usd": self.position_size_usd,
            "stop_loss": self.stop_loss,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "dca_enabled": self.dca_enabled,
            "regime": self.regime,
            "sentiment_score": self.sentiment_score,
            "reasoning": self.reasoning,
            "warnings": self.warnings,
        }


class MasterAI:
    """
    Master AI Controller - The brain of the trading system.

    Orchestrates all AI components for optimal trading decisions.
    """

    def __init__(
        self,
        max_leverage: float = 20.0,
        enable_rl: bool = True,
        enable_intelligence: bool = True,
        enable_advanced: bool = True,
        data_dir: str = "data/master_ai",
    ):
        self.max_leverage = max_leverage
        self.enable_rl = enable_rl
        self.enable_intelligence = enable_intelligence
        self.enable_advanced = enable_advanced
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded components
        self._ai_brain = None
        self._data_intelligence = None
        self._continuous_learner = None
        self._advanced_brain = None
        self._ai_orchestrator = None

        # Performance tracking
        self.decisions_made = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        self.win_rate = 0.5

        # Load state
        self._load_state()

        logger.info(
            f"Master AI initialized (leverage: {max_leverage}x, RL: {enable_rl}, Intel: {enable_intelligence}, Advanced: {enable_advanced})"
        )

    def _load_state(self):
        """Load saved state."""
        state_path = self.data_dir / "master_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.decisions_made = state.get("decisions_made", 0)
                self.trades_executed = state.get("trades_executed", 0)
                self.total_pnl = state.get("total_pnl", 0.0)
                self.win_rate = state.get("win_rate", 0.5)
                logger.info(
                    f"Loaded Master AI state: {self.trades_executed} trades, {self.win_rate:.1%} win rate"
                )
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save state."""
        state = {
            "decisions_made": self.decisions_made,
            "trades_executed": self.trades_executed,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.data_dir / "master_state.json", "w") as f:
            json.dump(state, f, indent=2)

    @property
    def ai_brain(self):
        """Lazy load AI Brain V2."""
        if self._ai_brain is None and self.enable_rl:
            try:
                from bot.ai_brain_v2 import get_ai_brain_v2

                self._ai_brain = get_ai_brain_v2()
            except Exception as e:
                logger.warning(f"AI Brain V2 not available: {e}")
        return self._ai_brain

    @property
    def data_intelligence(self):
        """Lazy load Data Intelligence."""
        if self._data_intelligence is None and self.enable_intelligence:
            try:
                from bot.data_intelligence import get_data_intelligence

                self._data_intelligence = get_data_intelligence()
            except Exception as e:
                logger.warning(f"Data Intelligence not available: {e}")
        return self._data_intelligence

    @property
    def continuous_learner(self):
        """Lazy load Continuous Learner."""
        if self._continuous_learner is None:
            try:
                from bot.continuous_learner import get_continuous_learner

                self._continuous_learner = get_continuous_learner()
            except Exception as e:
                logger.warning(f"Continuous Learner not available: {e}")
        return self._continuous_learner

    @property
    def advanced_brain(self):
        """Lazy load Advanced Trading Brain."""
        if self._advanced_brain is None and self.enable_advanced:
            try:
                from bot.advanced_trading_brain import get_advanced_trading_brain

                self._advanced_brain = get_advanced_trading_brain(self.max_leverage)
            except Exception as e:
                logger.warning(f"Advanced Trading Brain not available: {e}")
        return self._advanced_brain

    async def get_trade_decision(
        self,
        symbol: str,
        current_price: float,
        ml_signal: Dict[str, Any],
        market_data: Dict[str, Any],
        account_balance: float,
        current_exposure: float,
        existing_positions: List[Dict],
    ) -> MasterTradeDecision:
        """
        Get complete trading decision using all AI systems.

        This is the main entry point for trading decisions.
        """
        self.decisions_made += 1

        # Extract ML signal
        ml_action = ml_signal.get("action", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0.5)
        ml_meta = ml_signal.get("signal_meta", {})

        # Get regime and indicators
        regime = ml_meta.get("regime", "unknown")
        trend = ml_meta.get("trend", "neutral")
        volatility_str = ml_meta.get("volatility", "medium")
        volatility = market_data.get("volatility", 0.02)
        rsi = ml_meta.get("rsi", 50)

        # Initialize defaults
        rl_action = ml_action
        rl_confidence = ml_confidence
        sentiment_score = 0.0
        fear_greed = 50.0
        intelligence_bias = "neutral"
        short_score = None
        squeeze_risk = None
        catalysts = []
        warnings = []
        reasoning_steps = []

        # 1. Get RL decision from AI Brain V2
        if self.ai_brain:
            try:
                rl_decision = await self.ai_brain.get_trading_decision(
                    symbol, ml_signal, market_data
                )
                rl_action = rl_decision.get("action", ml_action)
                rl_confidence = rl_decision.get("confidence", ml_confidence)
                reasoning_steps.append(
                    {
                        "step": "Reinforcement Learning",
                        "action": rl_action,
                        "confidence": rl_confidence,
                        "details": rl_decision.get("reason", ""),
                    }
                )
            except Exception as e:
                logger.warning(f"RL decision failed: {e}")

        # 2. Get intelligence data
        if self.data_intelligence:
            try:
                intel_context = await self.data_intelligence.get_full_context(symbol)
                sentiment_data = intel_context.get("sentiment", {})
                sentiment_score = sentiment_data.get("news", 0)
                fear_greed = sentiment_data.get("fear_greed", 50)
                intelligence_bias = intel_context.get("signals", {}).get("overall_bias", "neutral")

                reasoning_steps.append(
                    {
                        "step": "Market Intelligence",
                        "sentiment": sentiment_score,
                        "fear_greed": fear_greed,
                        "bias": intelligence_bias,
                    }
                )

                # Sentiment-based warnings
                if fear_greed < 20:
                    catalysts.append("Extreme fear (contrarian buy)")
                elif fear_greed > 80:
                    catalysts.append("Extreme greed (contrarian sell)")
                    warnings.append("Market may be overheated")
            except Exception as e:
                logger.warning(f"Intelligence fetch failed: {e}")

        # 3. Determine if this should be a short
        is_short_candidate = ml_action in ["SELL", "SHORT"] or rl_action in ["SELL", "SHORT"]

        # 4. Get advanced analysis
        if self.advanced_brain:
            try:
                # Short analysis
                if is_short_candidate:
                    short_analysis = self.advanced_brain.short_specialist.analyze_short_opportunity(
                        symbol,
                        current_price,
                        {
                            "rsi": rsi,
                            "trend": trend,
                            "volatility": volatility,
                            "volume_ratio": market_data.get("volume_ratio", 1.0),
                            "high_24h": market_data.get("high_24h", current_price * 1.02),
                            "low_24h": market_data.get("low_24h", current_price * 0.98),
                            "change_24h": market_data.get("change_24h", 0),
                            "regime": regime,
                        },
                    )
                    short_score = short_analysis.score
                    squeeze_risk = (
                        self.advanced_brain.short_specialist.squeeze_detector.assess_risk(
                            symbol, market_data, None
                        )
                    )
                    catalysts.extend(short_analysis.catalysts)

                    if squeeze_risk > 0.6:
                        warnings.append(f"High squeeze risk ({squeeze_risk:.0%})")

                    reasoning_steps.append(
                        {
                            "step": "Short Analysis",
                            "score": short_score,
                            "squeeze_risk": squeeze_risk,
                            "catalysts": short_analysis.catalysts,
                        }
                    )

                # Deep reasoning
                deep_analysis = await self.advanced_brain.reasoning_engine.deep_analyze(
                    symbol,
                    current_price,
                    {"rsi": rsi, "trend": trend, "volatility": volatility, "regime": regime},
                    {"sentiment": {"news": sentiment_score, "fear_greed": fear_greed}},
                    ml_signal,
                    existing_positions,
                )
                reasoning_steps.extend(deep_analysis.get("reasoning_steps", []))

            except Exception as e:
                logger.warning(f"Advanced analysis failed: {e}")

        # 5. Combine all signals to determine final action
        final_action, final_confidence = self._combine_all_signals(
            ml_action=ml_action,
            ml_confidence=ml_confidence,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            sentiment_score=sentiment_score,
            fear_greed=fear_greed,
            regime=regime,
            short_score=short_score,
        )

        # 6. Calculate position parameters
        is_short = final_action in ["SELL", "SHORT"]

        # Leverage recommendation
        leverage = 1.0
        position_size_usd = account_balance * 0.02  # Default 2% risk

        if self.advanced_brain:
            try:
                leverage_rec = self.advanced_brain.leverage_manager.calculate_optimal_leverage(
                    signal_confidence=final_confidence,
                    volatility=volatility,
                    account_balance=account_balance,
                    current_exposure=current_exposure,
                    win_rate=self.win_rate,
                    regime=regime,
                    is_short=is_short,
                )
                leverage = leverage_rec.recommended_leverage
                position_size_usd = leverage_rec.position_size_usd
                warnings.extend(leverage_rec.warnings)

                reasoning_steps.append(
                    {
                        "step": "Leverage Calculation",
                        "leverage": leverage,
                        "position_size": position_size_usd,
                        "reasoning": leverage_rec.reasoning,
                    }
                )
            except Exception as e:
                logger.warning(f"Leverage calculation failed: {e}")

        # 7. Calculate stop loss and take profits
        atr = current_price * volatility

        if is_short:
            stop_loss = current_price + atr * 1.5
            take_profit_1 = current_price - atr * 1.0
            take_profit_2 = current_price - atr * 2.0
            take_profit_3 = current_price - atr * 3.0
        else:
            stop_loss = current_price - atr * 1.5
            take_profit_1 = current_price + atr * 1.0
            take_profit_2 = current_price + atr * 2.0
            take_profit_3 = current_price + atr * 3.0

        # Use profit maximizer for better targets
        if self.advanced_brain:
            try:
                targets = self.advanced_brain.profit_maximizer.calculate_targets(
                    entry_price=current_price,
                    side="short" if is_short else "long",
                    confidence=final_confidence,
                    volatility=volatility,
                    regime=regime,
                    support_levels=[current_price * 0.95, current_price * 0.9],
                    resistance_levels=[current_price * 1.05, current_price * 1.1],
                )
                if len(targets) >= 3:
                    take_profit_1 = targets[0].target_price
                    take_profit_2 = targets[1].target_price
                    take_profit_3 = targets[2].target_price
            except Exception as e:
                logger.warning(f"Profit target calculation failed: {e}")

        # Calculate percentages
        stop_loss_pct = abs(stop_loss - current_price) / current_price
        tp_pcts = [
            abs(take_profit_1 - current_price) / current_price,
            abs(take_profit_2 - current_price) / current_price,
            abs(take_profit_3 - current_price) / current_price,
        ]

        # Position size as percentage
        position_size_pct = position_size_usd / account_balance

        # Build reasoning string
        reasoning_parts = [
            f"ML: {ml_action} ({ml_confidence:.0%})",
            f"RL: {rl_action} ({rl_confidence:.0%})",
            f"Final: {final_action} ({final_confidence:.0%})",
        ]
        if intelligence_bias != "neutral":
            reasoning_parts.append(f"Sentiment: {intelligence_bias}")
        if short_score is not None:
            reasoning_parts.append(f"Short score: {short_score:.0f}/100")

        reasoning = " | ".join(reasoning_parts)

        # Create decision
        decision = MasterTradeDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            action=final_action,
            confidence=final_confidence,
            position_size_pct=position_size_pct,
            leverage=leverage,
            position_size_usd=position_size_usd,
            stop_loss=stop_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            take_profit_pcts=tp_pcts,
            trailing_stop_enabled=volatility_str == "high" or final_confidence < 0.7,
            trailing_stop_pct=volatility * 1.5,
            dca_enabled=final_confidence < 0.65,
            max_hold_hours=72 if leverage > 5 else 168,
            regime=regime,
            trend=trend,
            volatility=volatility_str,
            sentiment_score=sentiment_score,
            fear_greed=fear_greed,
            short_score=short_score,
            squeeze_risk=squeeze_risk,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            catalysts=catalysts,
            warnings=warnings,
            ml_action=ml_action,
            ml_confidence=ml_confidence,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
        )

        # Log decision
        logger.info(
            f"[{symbol}] Master AI Decision: {final_action} @ {final_confidence:.0%}, "
            f"Leverage: {leverage}x, Size: ${position_size_usd:.0f}"
        )

        # Save state periodically
        if self.decisions_made % 10 == 0:
            self._save_state()

        return decision

    def _combine_all_signals(
        self,
        ml_action: str,
        ml_confidence: float,
        rl_action: str,
        rl_confidence: float,
        sentiment_score: float,
        fear_greed: float,
        regime: str,
        short_score: Optional[float],
    ) -> tuple[str, float]:
        """Combine all signals into final decision."""
        # Convert to numeric
        action_map = {"BUY": 1, "SELL": -1, "HOLD": 0, "LONG": 1, "SHORT": -1, "FLAT": 0}

        ml_numeric = action_map.get(ml_action.upper(), 0)
        rl_numeric = action_map.get(rl_action.upper(), 0)

        # Weights - ML gets higher priority as it's trained on historical data
        ml_weight = 0.50  # Increased from 0.35
        rl_weight = 0.20  # Decreased from 0.30 (RL needs more training)
        sentiment_weight = 0.15
        fear_greed_weight = 0.10
        short_weight = 0.05

        # Calculate weighted score
        score = 0.0

        # ML contribution
        score += ml_numeric * ml_confidence * ml_weight

        # RL contribution
        score += rl_numeric * rl_confidence * rl_weight

        # Enhanced sentiment contribution with strength multiplier
        clamped_sentiment = np.clip(sentiment_score, -1, 1)
        sentiment_strength = abs(clamped_sentiment)

        # Strong sentiment (>0.5) gets 1.5x weight
        effective_sentiment_weight = sentiment_weight
        if sentiment_strength > 0.5:
            effective_sentiment_weight = sentiment_weight * 1.5
        elif sentiment_strength > 0.7:
            effective_sentiment_weight = sentiment_weight * 2.0

        score += clamped_sentiment * effective_sentiment_weight

        # Enhanced Fear/Greed contrarian signal with stronger extremes
        if fear_greed < 15:  # Extreme fear = strong buy signal
            score += 0.8 * fear_greed_weight
        elif fear_greed < 25:
            score += 0.5 * fear_greed_weight  # Bullish
        elif fear_greed > 85:  # Extreme greed = strong sell signal
            score -= 0.8 * fear_greed_weight
        elif fear_greed > 75:
            score -= 0.5 * fear_greed_weight  # Bearish

        # Short score contribution (if analyzing short)
        if short_score is not None and short_score > 50:
            score -= (short_score - 50) / 100 * short_weight

        # Calculate confidence
        base_confidence = (ml_confidence * ml_weight + rl_confidence * rl_weight) / (
            ml_weight + rl_weight
        )

        # Agreement bonus
        if ml_action == rl_action and ml_action != "HOLD":
            base_confidence += 0.12

        # Sentiment agreement bonus
        if (ml_numeric > 0 and clamped_sentiment > 0.3) or (
            ml_numeric < 0 and clamped_sentiment < -0.3
        ):
            base_confidence += 0.05  # ML and sentiment agree

        # Conflict penalty
        if (ml_numeric > 0 and rl_numeric < 0) or (ml_numeric < 0 and rl_numeric > 0):
            base_confidence -= 0.15

        # Regime adjustment
        if regime == "crash":
            base_confidence *= 0.7
        elif regime == "volatile":
            base_confidence *= 0.85

        final_confidence = np.clip(base_confidence, 0.1, 0.95)

        # Determine action - reduced thresholds to allow more trades
        if score > 0.10:  # Reduced from 0.15
            final_action = "BUY"
        elif score < -0.10:  # Reduced from -0.15
            final_action = "SHORT"
        else:
            final_action = "HOLD"

        return final_action, final_confidence

    async def check_exit(
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
    ) -> Dict[str, Any]:
        """Check if position should be exited."""
        if self.advanced_brain:
            exit_signal = self.advanced_brain.check_exit(
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
            return {
                "should_exit": exit_signal.should_exit,
                "urgency": exit_signal.urgency,
                "reason": exit_signal.reason.value,
                "confidence": exit_signal.confidence,
                "risk_of_reversal": exit_signal.risk_of_reversal,
                "reasoning": exit_signal.reasoning,
            }

        # Fallback simple check
        if side == "long":
            if current_price <= stop_loss:
                return {"should_exit": True, "reason": "stop_loss", "urgency": 1.0}
            if current_price >= take_profit:
                return {"should_exit": True, "reason": "take_profit", "urgency": 0.8}
        else:
            if current_price >= stop_loss:
                return {"should_exit": True, "reason": "stop_loss", "urgency": 1.0}
            if current_price <= take_profit:
                return {"should_exit": True, "reason": "take_profit", "urgency": 0.8}

        return {"should_exit": False, "reason": "none", "urgency": 0.0}

    def record_trade_outcome(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        leverage: float,
        hold_time_minutes: int,
        was_stopped: bool,
        was_target_hit: bool,
        features: Optional[np.ndarray] = None,
    ):
        """Record trade outcome for all learning systems."""
        self.trades_executed += 1
        self.total_pnl += pnl

        # Update win rate
        won = pnl > 0
        self.win_rate = (
            self.win_rate * (self.trades_executed - 1) + (1 if won else 0)
        ) / self.trades_executed

        # Record to AI Brain V2
        if self.ai_brain:
            try:
                self.ai_brain.record_trade_outcome(
                    symbol,
                    action,
                    entry_price,
                    exit_price,
                    pnl,
                    pnl_pct,
                    hold_time_minutes,
                    was_stopped,
                    was_target_hit,
                )
            except Exception as e:
                logger.warning(f"AI Brain record failed: {e}")

        # Record to Continuous Learner
        if self.continuous_learner and features is not None:
            try:
                self.continuous_learner.record_prediction(
                    symbol, features, action, "WIN" if won else "LOSS", pnl, pnl_pct
                )
            except Exception as e:
                logger.warning(f"Continuous Learner record failed: {e}")

        # Record to Advanced Brain
        if self.advanced_brain:
            try:
                self.advanced_brain.record_trade(leverage, pnl, won)
            except Exception as e:
                logger.warning(f"Advanced Brain record failed: {e}")

        # Save state
        self._save_state()

        logger.info(
            f"[{symbol}] Trade recorded: {action} -> ${pnl:.2f} ({pnl_pct * 100:.2f}%), "
            f"Win rate: {self.win_rate:.1%}"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all AI systems."""
        status = {
            "master_ai": {
                "decisions_made": self.decisions_made,
                "trades_executed": self.trades_executed,
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate,
                "max_leverage": self.max_leverage,
            },
            "components": {
                "ai_brain_v2": self.ai_brain is not None,
                "data_intelligence": self.data_intelligence is not None,
                "continuous_learner": self.continuous_learner is not None,
                "advanced_brain": self.advanced_brain is not None,
            },
        }

        # AI Brain stats
        if self.ai_brain:
            try:
                status["ai_brain_v2"] = self.ai_brain.get_performance_report()
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to get AI brain report: {e}")

        # Continuous Learner stats
        if self.continuous_learner:
            try:
                status["continuous_learner"] = self.continuous_learner.get_all_summaries()
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Failed to get continuous learner report: {e}")

        return status


# Global instance
_master_ai: Optional[MasterAI] = None


def get_master_ai(
    max_leverage: float = 20.0,
    enable_rl: bool = True,
    enable_intelligence: bool = True,
    enable_advanced: bool = True,
) -> MasterAI:
    """Get or create the Master AI instance."""
    global _master_ai
    if _master_ai is None:
        _master_ai = MasterAI(
            max_leverage=max_leverage,
            enable_rl=enable_rl,
            enable_intelligence=enable_intelligence,
            enable_advanced=enable_advanced,
        )
    return _master_ai
