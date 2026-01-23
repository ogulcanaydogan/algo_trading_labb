"""
AI Integration Module - Connects all AI systems to trading engine.

This module integrates:
1. AI Brain V2 (RL-based decision making)
2. Data Intelligence (multi-source market data)
3. Continuous Learner (adaptive improvement)

With the existing:
- ML Predictor
- Intelligent Brain
- Regime Detection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with all AI inputs."""

    symbol: str
    timestamp: datetime
    action: str
    confidence: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: bool
    dca_enabled: bool
    regime: str
    trend: str
    volatility: str
    sentiment_score: float
    fear_greed: float
    reason: str
    ml_action: str
    rl_action: str
    intelligence_bias: str
    ml_confidence: float
    rl_confidence: float
    sentiment_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()
        }


class AIOrchestrator:
    """Orchestrates all AI systems for optimal trading decisions."""

    def __init__(self, enable_rl: bool = True, enable_intelligence: bool = True):
        self.enable_rl = enable_rl
        self.enable_intelligence = enable_intelligence
        self._ai_brain = None
        self._data_intelligence = None
        self._continuous_learner = None
        self.weights = {"ml": 0.4, "rl": 0.3, "sentiment": 0.15, "regime": 0.15}
        logger.info(
            f"AI Orchestrator initialized (RL: {enable_rl}, Intelligence: {enable_intelligence})"
        )

    @property
    def ai_brain(self):
        if self._ai_brain is None and self.enable_rl:
            try:
                from bot.ai_brain_v2 import get_ai_brain_v2

                self._ai_brain = get_ai_brain_v2()
            except Exception as e:
                logger.warning(f"Failed to load AI Brain V2: {e}")
        return self._ai_brain

    @property
    def data_intelligence(self):
        if self._data_intelligence is None and self.enable_intelligence:
            try:
                from bot.data_intelligence import get_data_intelligence

                self._data_intelligence = get_data_intelligence()
            except Exception as e:
                logger.warning(f"Failed to load Data Intelligence: {e}")
        return self._data_intelligence

    @property
    def continuous_learner(self):
        if self._continuous_learner is None:
            try:
                from bot.continuous_learner import get_continuous_learner

                self._continuous_learner = get_continuous_learner()
            except Exception as e:
                logger.warning(f"Failed to load Continuous Learner: {e}")
        return self._continuous_learner

    async def generate_enhanced_signal(
        self, symbol: str, ml_signal: Dict, market_data: Dict
    ) -> EnhancedSignal:
        """Generate enhanced trading signal using all AI systems."""
        ml_action = ml_signal.get("action", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0.5)
        ml_meta = ml_signal.get("signal_meta", {})

        rl_action, rl_confidence = ml_action, ml_confidence
        sentiment_score, fear_greed = 0.0, 50.0
        intelligence_bias = "neutral"

        if self.ai_brain:
            try:
                rl_decision = await self.ai_brain.get_trading_decision(
                    symbol, ml_signal, market_data
                )
                rl_action = rl_decision.get("action", ml_action)
                rl_confidence = rl_decision.get("confidence", ml_confidence)
            except Exception as e:
                logger.warning(f"RL decision failed: {e}")

        if self.data_intelligence:
            try:
                context = await self.data_intelligence.get_full_context(symbol)
                sentiment_score = context.get("sentiment", {}).get("news", 0)
                fear_greed = context.get("sentiment", {}).get("fear_greed", 50)
                intelligence_bias = context.get("signals", {}).get("overall_bias", "neutral")
            except Exception as e:
                logger.warning(f"Intelligence fetch failed: {e}")

        final_action, final_confidence = self._combine_signals(
            ml_action,
            ml_confidence,
            rl_action,
            rl_confidence,
            sentiment_score,
            fear_greed,
            ml_meta.get("regime", "unknown"),
        )

        regime_strategy = ml_meta.get("regime_strategy", {})
        position_size = regime_strategy.get("position_size_multiplier", 1.0)
        stop_loss = regime_strategy.get("stop_loss_pct", 0.02)
        take_profit = regime_strategy.get("take_profit_pct", 0.04)

        reasons = [
            f"ML: {ml_action} ({ml_confidence:.0%})",
            f"RL: {rl_action} ({rl_confidence:.0%})",
        ]
        if intelligence_bias != "neutral":
            reasons.append(f"Sentiment: {intelligence_bias}")

        return EnhancedSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            action=final_action,
            confidence=final_confidence,
            position_size_pct=min(1.0, position_size),
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            trailing_stop=ml_meta.get("volatility", "medium") == "high",
            dca_enabled=final_confidence < 0.7,
            regime=ml_meta.get("regime", "unknown"),
            trend=ml_meta.get("trend", "neutral"),
            volatility=ml_meta.get("volatility", "medium"),
            sentiment_score=sentiment_score,
            fear_greed=fear_greed,
            reason=" | ".join(reasons),
            ml_action=ml_action,
            rl_action=rl_action,
            intelligence_bias=intelligence_bias,
            ml_confidence=ml_confidence,
            rl_confidence=rl_confidence,
            sentiment_confidence=0.5 + abs(sentiment_score) * 0.3,
        )

    def _combine_signals(
        self,
        ml_action,
        ml_confidence,
        rl_action,
        rl_confidence,
        sentiment_score,
        fear_greed,
        regime,
    ):
        action_map = {"BUY": 1, "SELL": -1, "HOLD": 0, "LONG": 1, "SHORT": -1, "FLAT": 0}
        ml_numeric = action_map.get(ml_action.upper(), 0)
        rl_numeric = action_map.get(rl_action.upper(), 0)
        sentiment_numeric = np.clip(sentiment_score, -1, 1)
        fear_greed_signal = 0.5 if fear_greed < 25 else (-0.5 if fear_greed > 75 else 0)

        combined = (
            ml_numeric * ml_confidence * self.weights["ml"]
            + rl_numeric * rl_confidence * self.weights["rl"]
            + sentiment_numeric * self.weights["sentiment"]
            + fear_greed_signal * self.weights["regime"]
        )

        agreement_bonus = 0.15 if ml_action == rl_action and ml_action != "HOLD" else 0
        conflict_penalty = (
            0.2 if (ml_numeric > 0 and rl_numeric < 0) or (ml_numeric < 0 and rl_numeric > 0) else 0
        )
        base_confidence = (
            ml_confidence * self.weights["ml"] + rl_confidence * self.weights["rl"]
        ) / (self.weights["ml"] + self.weights["rl"])
        final_confidence = np.clip(base_confidence + agreement_bonus - conflict_penalty, 0.1, 0.95)

        final_action = "BUY" if combined > 0.2 else ("SELL" if combined < -0.2 else "HOLD")
        if regime == "crash" and final_action == "BUY":
            final_confidence *= 0.7

        return final_action, final_confidence

    def record_trade_outcome(
        self,
        symbol,
        action,
        entry_price,
        exit_price,
        pnl,
        pnl_pct,
        hold_time_minutes,
        was_stopped,
        was_target_hit,
        features=None,
    ):
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

        if self.continuous_learner and features is not None:
            try:
                self.continuous_learner.record_prediction(
                    symbol, features, action, "WIN" if pnl > 0 else "LOSS", pnl, pnl_pct
                )
            except Exception as e:
                logger.warning(f"Continuous learner record failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        status = {
            "ai_brain_active": self.ai_brain is not None,
            "data_intelligence_active": self.data_intelligence is not None,
            "continuous_learner_active": self.continuous_learner is not None,
            "weights": self.weights,
        }
        if self.ai_brain:
            status["ai_brain"] = self.ai_brain.get_performance_report()
        if self.continuous_learner:
            status["continuous_learner"] = self.continuous_learner.get_all_summaries()
        return status


_ai_orchestrator: Optional[AIOrchestrator] = None


def get_ai_orchestrator(enable_rl: bool = True, enable_intelligence: bool = True) -> AIOrchestrator:
    global _ai_orchestrator
    if _ai_orchestrator is None:
        _ai_orchestrator = AIOrchestrator(enable_rl, enable_intelligence)
    return _ai_orchestrator
