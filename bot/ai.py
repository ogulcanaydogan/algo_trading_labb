from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .state import BotState
from .strategy import StrategyConfig
from .macro import MacroInsight


@dataclass
class FeatureSnapshot:
    ema_gap_pct: float
    momentum_pct: float
    rsi_distance_from_mid: float
    volatility_pct: float


@dataclass
class PredictionSnapshot:
    recommended_action: str
    confidence: float
    probability_long: float
    probability_short: float
    probability_flat: float
    expected_move_pct: float
    summary: str
    features: FeatureSnapshot
    macro_bias: float = 0.0
    macro_confidence: float = 0.0
    macro_summary: str = ""
    macro_drivers: List[str] = field(default_factory=list)
    macro_interest_rate_outlook: Optional[str] = None
    macro_political_risk: Optional[str] = None


class RuleBasedAIPredictor:
    """Lightweight heuristic predictor that mimics an ML inference step."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.config = strategy_config

    def predict(
        self,
        enriched: pd.DataFrame,
        macro: Optional[MacroInsight] = None,
    ) -> PredictionSnapshot:
        if len(enriched) < 3:
            raise ValueError("Need at least three candles to produce an AI prediction.")

        last = enriched.iloc[-1]
        prev = enriched.iloc[-2]
        earlier = enriched.iloc[-3]

        ema_gap_pct = float((last["ema_fast"] - last["ema_slow"]) / max(last["close"], 1e-8))
        momentum_pct = float((last["close"] - prev["close"]) / max(prev["close"], 1e-8))
        rsi_distance_from_mid = float((last["rsi"] - 50.0) / 50.0)
        volatility_pct = float(
            (abs(last["close"] - earlier["close"]) / max(earlier["close"], 1e-8))
        )

        long_score = (
            ema_gap_pct * 8.0
            - max(rsi_distance_from_mid, 0.0) * 2.5
            + (1.0 - abs(rsi_distance_from_mid)) * 1.2
            + momentum_pct * 6.0
        )
        short_score = (
            -ema_gap_pct * 8.0
            + max(rsi_distance_from_mid, 0.0) * 2.5
            - (1.0 - abs(rsi_distance_from_mid)) * 1.2
            - momentum_pct * 6.0
        )
        flat_score = -abs(ema_gap_pct) * 4.0 - abs(momentum_pct) * 3.0 + 0.8

        macro_bias = macro.bias_score if macro else 0.0
        macro_confidence = macro.confidence if macro else 0.0
        macro_multiplier = 1.0 + macro_confidence

        if macro:
            long_score += max(0.0, macro_bias) * 3.5 * macro_multiplier
            short_score -= max(0.0, macro_bias) * 3.5 * macro_multiplier
            long_score += min(0.0, macro_bias) * 1.5
            short_score -= min(0.0, macro_bias) * 1.5
            flat_score += (1.0 - abs(macro_bias)) * (1.0 - macro_confidence) * 0.6

        probabilities = self._softmax(
            {
                "LONG": long_score,
                "SHORT": short_score,
                "FLAT": flat_score,
            }
        )

        recommended_action = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[recommended_action])
        # derive expected move as a blend of EMA gap and momentum, then apply direction
        direction = (
            1
            if recommended_action == "LONG"
            else -1
            if recommended_action == "SHORT"
            else 0
        )
        expected_move_pct = float(
            (ema_gap_pct * 0.65 + momentum_pct * 0.9) * 100.0 * direction
        )

        summary = self._summarise_prediction(
            recommended_action,
            confidence,
            expected_move_pct,
            ema_gap_pct,
            momentum_pct,
            rsi_distance_from_mid,
            macro,
        )

        features = FeatureSnapshot(
            ema_gap_pct=round(ema_gap_pct * 100.0, 4),
            momentum_pct=round(momentum_pct * 100.0, 4),
            rsi_distance_from_mid=round(rsi_distance_from_mid * 100.0, 4),
            volatility_pct=round(volatility_pct * 100.0, 4),
        )

        return PredictionSnapshot(
            recommended_action=recommended_action,
            confidence=round(confidence, 4),
            probability_long=round(probabilities["LONG"], 4),
            probability_short=round(probabilities["SHORT"], 4),
            probability_flat=round(probabilities["FLAT"], 4),
            expected_move_pct=round(expected_move_pct, 4),
            summary=summary,
            features=features,
            macro_bias=round(macro_bias, 4),
            macro_confidence=round(macro_confidence, 4),
            macro_summary=(macro.summary if macro else ""),
            macro_drivers=(macro.drivers if macro else []),
            macro_interest_rate_outlook=(
                macro.interest_rate_outlook if macro else None
            ),
            macro_political_risk=(macro.political_risk if macro else None),
        )

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        values = list(scores.values())
        max_score = max(values)
        exp_scores = {label: math.exp(value - max_score) for label, value in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {label: value / total for label, value in exp_scores.items()}

    def _summarise_prediction(
        self,
        action: str,
        confidence: float,
        expected_move_pct: float,
        ema_gap_pct: float,
        momentum_pct: float,
        rsi_distance_from_mid: float,
        macro: Optional[MacroInsight],
    ) -> str:
        direction = {
            "LONG": "upside",
            "SHORT": "downside",
            "FLAT": "sideways consolidation",
        }[action]

        drivers: List[str] = []
        if abs(ema_gap_pct) > 0.0005:
            drivers.append(f"EMA spread of {ema_gap_pct * 100:.2f}%")
        if abs(momentum_pct) > 0.0003:
            drivers.append(f"recent price momentum of {momentum_pct * 100:.2f}%")
        if abs(rsi_distance_from_mid) > 0.05:
            bias = "bullish" if rsi_distance_from_mid > 0 else "bearish"
            drivers.append(f"RSI {bias} bias of {abs(rsi_distance_from_mid) * 100:.1f}% from midline")

        driver_text = ", driven by " + ", ".join(drivers) if drivers else ""
        if action == "FLAT":
            expectation = "Model sees limited edge and prefers to stay flat"
        else:
            expectation = (
                f"Model leans {direction} with {confidence * 100:.1f}% confidence"
            )
        macro_text = ""
        if macro and macro.summary:
            macro_text = f" Macro overlay: {macro.summary}"
        return (
            f"{expectation}{driver_text}. Expected move: {expected_move_pct:.2f}%"
            f"{macro_text}"
        )


class QuestionAnsweringEngine:
    """Provide lightweight Q&A responses about the bot configuration and state."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.config = strategy_config
        self._knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[Dict[str, object]]:
        return [
            {
                "keywords": {"buy", "long", "entry"},
                "response": self._answer_buy_question,
            },
            {
                "keywords": {"sell", "short", "exit"},
                "response": self._answer_sell_question,
            },
            {
                "keywords": {"risk", "stop", "take"},
                "response": self._answer_risk_question,
            },
            {
                "keywords": {"ai", "model", "predict", "prob"},
                "response": self._answer_ai_question,
            },
            {
                "keywords": {"macro", "news", "trump", "fed", "rates", "politic"},
                "response": self._answer_macro_question,
            },
        ]

    def answer(
        self,
        question: str,
        state: Optional[BotState] = None,
        ai_snapshot: Optional[PredictionSnapshot] = None,
        macro_insight: Optional[MacroInsight] = None,
    ) -> str:
        cleaned = (question or "").strip()
        if not cleaned:
            return "Please provide a question so the assistant can help."

        lowered = cleaned.lower()
        for entry in self._knowledge_base:
            keywords: Iterable[str] = entry["keywords"]  # type: ignore[assignment]
            if any(keyword in lowered for keyword in keywords):
                responder = entry["response"]  # type: ignore[assignment]
                return responder(state, ai_snapshot, macro_insight)

        return self._fallback_answer(state, macro_insight)

    def _answer_buy_question(
        self,
        state: Optional[BotState],
        ai_snapshot: Optional[PredictionSnapshot],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        parts: List[str] = []
        parts.append(
            "Buy signals are triggered when the fast EMA crosses above the slow EMA and RSI is not overbought."
        )
        parts.append(
            f"EMA windows: {self.config.ema_fast}/{self.config.ema_slow} on {self.config.timeframe}."
        )
        if state and state.last_signal == "LONG":
            parts.append(f"Latest signal: LONG (confidence {state.confidence or 0:.2f}).")
        if ai_snapshot and ai_snapshot.recommended_action == "LONG":
            parts.append(
                f"AI leans LONG ({ai_snapshot.confidence * 100:.1f}%); expected move "
                f"{ai_snapshot.expected_move_pct:.2f}%.")
        if macro_insight:
            if macro_insight.bias_score and macro_insight.bias_score > 0.05:
                parts.append(f"Macro supportive ({macro_insight.bias_score:+.2f}).")
            elif macro_insight.bias_score and macro_insight.bias_score < -0.05:
                parts.append(f"Macro cautious ({macro_insight.bias_score:+.2f}).")
        return " ".join(parts)

    def _answer_sell_question(
        self,
        state: Optional[BotState],
        ai_snapshot: Optional[PredictionSnapshot],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        parts: List[str] = []
        parts.append(
            "Short setups occur when the fast EMA crosses below the slow EMA and RSI is not oversold."
        )
        parts.append(
            f"Oversold/overbought: {self.config.rsi_oversold}/{self.config.rsi_overbought}."
        )
        if state and state.last_signal == "SHORT":
            parts.append(f"Latest signal: SHORT (confidence {state.confidence or 0:.2f}).")
        if ai_snapshot and ai_snapshot.recommended_action == "SHORT":
            parts.append("AI leans SHORT.")
            parts.append(f"Confidence: {ai_snapshot.confidence * 100:.1f}%.")
            parts.append(f"Expected move: {ai_snapshot.expected_move_pct:.2f}%.")
        if macro_insight:
            if macro_insight.bias_score and macro_insight.bias_score < -0.05:
                parts.append(f"Macro negative ({macro_insight.bias_score:+.2f}).")
            elif macro_insight.bias_score and macro_insight.bias_score > 0.05:
                parts.append(f"Macro supportive ({macro_insight.bias_score:+.2f}).")
        return " ".join(parts)

    def _answer_risk_question(
        self,
        state: Optional[BotState],
        ai_snapshot: Optional[PredictionSnapshot],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        del ai_snapshot
        base = f"Risk per trade is capped at {self.config.risk_per_trade_pct}% of the balance."
        base += f" Stop-loss: {self.config.stop_loss_pct * 100:.2f}%"
        base += f"; take-profit: {self.config.take_profit_pct * 100:.2f}%."
        if state:
            base += f" Current unrealized PnL: {state.unrealized_pnl_pct:.2f}%"
            base += f"; position: {state.position}."
        if macro_insight and macro_insight.interest_rate_outlook:
            base += (
                f" Rate backdrop: {macro_insight.interest_rate_outlook}."
            )
        return base

    def _answer_ai_question(
        self,
        state: Optional[BotState],
        ai_snapshot: Optional[PredictionSnapshot],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        del state
        if not ai_snapshot:
            return "AI predictions become available after the bot processes live candles."
        macro_line = ""
        if macro_insight and macro_insight.summary:
            macro_line = f"Macro context: {macro_insight.summary}"

        parts: List[str] = []
        parts.append(f"AI model leans {ai_snapshot.recommended_action}.")
        parts.append(f"Confidence: {ai_snapshot.confidence * 100:.1f}%.")
        parts.append(f"Probabilities: long {ai_snapshot.probability_long * 100:.1f}%.")
        parts.append(f"Short probability: {ai_snapshot.probability_short * 100:.1f}%.")
        parts.append(f"Flat probability: {ai_snapshot.probability_flat * 100:.1f}%.")
        parts.append(f"Expected move: {ai_snapshot.expected_move_pct:.2f}%.")
        if macro_line:
            parts.append(macro_line)
        return " ".join(parts)

    def _answer_macro_question(
        self,
        state: Optional[BotState],
        ai_snapshot: Optional[PredictionSnapshot],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        del ai_snapshot
        parts: List[str] = []
        if macro_insight:
            if getattr(macro_insight, "summary", None):
                parts.append(macro_insight.summary)
            if getattr(macro_insight, "drivers", None):
                parts.append("Key drivers: " + ", ".join(macro_insight.drivers[:3]) + ".")
            if getattr(macro_insight, "interest_rate_outlook", None):
                parts.append(f"Interest-rate outlook: {macro_insight.interest_rate_outlook}.")
            if getattr(macro_insight, "political_risk", None):
                parts.append(f"Political watch: {macro_insight.political_risk}.")
        else:
            parts.append(
                "No macro catalysts registered yet. Upload a macro_events.json file"
            )
            parts.append("or set MACRO_EVENTS_PATH for richer context.")
        if state:
            parts.append(f"Strategy tracks {state.symbol} with position {state.position}.")
        return " ".join(parts)

    def _fallback_answer(
        self,
        state: Optional[BotState],
        macro_insight: Optional[MacroInsight],
    ) -> str:
        base = f"The bot monitors {self.config.symbol} on {self.config.timeframe} candles."
        base += f" EMA windows: {self.config.ema_fast}/{self.config.ema_slow}."
        base += f" RSI period: {self.config.rsi_period}."
        if state:
            base += f" Current stance: {state.position} with last signal {state.last_signal}."
        if macro_insight and macro_insight.summary:
            base += f" Macro overlay: {macro_insight.summary}."
        base += " Ask about buy, sell, risk, macro, or AI to get focused guidance."
        return base
