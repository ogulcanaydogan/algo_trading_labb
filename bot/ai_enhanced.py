"""
Enhanced AI Trading Module for GPU-Accelerated Decision Making.

Designed to run on NVIDIA Spark with 119GB RAM.
Integrates multiple AI models for superior trading decisions.
"""

from __future__ import annotations

import json
import os
import asyncio
import httpx
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AIModel(Enum):
    """Available AI models for different tasks."""
    REASONING = "llama3.1:70b"      # Deep analysis, complex reasoning
    FAST = "qwen2.5:7b"             # Quick decisions, low latency
    MULTI_EXPERT = "mixtral:8x7b"   # Multiple expert opinions
    CODE = "qwen2.5-coder:14b"      # Strategy code generation


@dataclass
class MarketContext:
    """Comprehensive market context for AI analysis."""
    symbol: str
    current_price: float
    price_change_1h: float
    price_change_24h: float
    volume_24h: float
    volume_change: float
    regime: str  # trending_up, trending_down, ranging, volatile
    regime_confidence: float
    rsi: float
    macd_signal: str
    support_levels: List[float]
    resistance_levels: List[float]
    recent_trades: List[Dict]
    news_sentiment: Optional[float] = None
    fear_greed_index: Optional[int] = None

    def to_prompt(self) -> str:
        return f"""
MARKET CONTEXT for {self.symbol}:
- Current Price: ${self.current_price:,.2f}
- 1H Change: {self.price_change_1h:+.2f}%
- 24H Change: {self.price_change_24h:+.2f}%
- 24H Volume: ${self.volume_24h:,.0f} ({self.volume_change:+.1f}% vs avg)
- Market Regime: {self.regime} (confidence: {self.regime_confidence:.0%})
- RSI(14): {self.rsi:.1f}
- MACD Signal: {self.macd_signal}
- Support Levels: {', '.join(f'${s:,.0f}' for s in self.support_levels[:3])}
- Resistance Levels: {', '.join(f'${r:,.0f}' for r in self.resistance_levels[:3])}
- Recent Trades: {len(self.recent_trades)} trades
- News Sentiment: {self.news_sentiment or 'N/A'}
- Fear & Greed: {self.fear_greed_index or 'N/A'}
"""


@dataclass
class AIDecision:
    """AI-generated trading decision with full reasoning."""
    action: Literal["LONG", "SHORT", "FLAT", "REDUCE", "ADD"]
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_pct: float  # 0-100% of available capital
    reasoning: str
    risks: List[str]
    catalysts: List[str]
    time_horizon: str  # "scalp", "intraday", "swing", "position"
    model_used: str
    generation_time_ms: int

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_pct": self.position_size_pct,
            "reasoning": self.reasoning,
            "risks": self.risks,
            "catalysts": self.catalysts,
            "time_horizon": self.time_horizon,
            "model_used": self.model_used,
            "generation_time_ms": self.generation_time_ms,
        }


@dataclass
class StrategyRecommendation:
    """AI-recommended strategy parameters."""
    strategy_name: str
    parameters: Dict[str, Any]
    expected_win_rate: float
    expected_sharpe: float
    suitable_regimes: List[str]
    reasoning: str


class EnhancedAIEngine:
    """
    GPU-Accelerated AI Trading Engine.

    Features:
    - Multi-model ensemble decisions
    - Adaptive strategy generation
    - Real-time sentiment analysis
    - Continuous learning from trades
    - Risk-adjusted position sizing
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        primary_model: str = AIModel.REASONING.value,
        fast_model: str = AIModel.FAST.value,
        timeout: int = 120,
    ):
        self.ollama_host = ollama_host
        self.primary_model = primary_model
        self.fast_model = fast_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self.decision_history: List[AIDecision] = []
        self.learning_enabled = True

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _query_ollama(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """Query Ollama with structured output."""
        client = await self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = datetime.now()

        try:
            response = await client.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": model or self.primary_model,
                    "messages": messages,
                    "stream": False,
                    "format": "json" if json_mode else None,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": 8192,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()

            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            content = result.get("message", {}).get("content", "{}")
            if json_mode:
                return {"data": json.loads(content), "elapsed_ms": elapsed_ms}
            return {"data": content, "elapsed_ms": elapsed_ms}

        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return {"data": {}, "elapsed_ms": 0, "error": str(e)}

    async def analyze_market(self, context: MarketContext) -> AIDecision:
        """
        Deep market analysis using primary reasoning model.

        Uses chain-of-thought prompting for better decisions.
        """
        system_prompt = """You are an expert quantitative trader with 20 years of experience.
You analyze markets systematically and make data-driven decisions.
Always respond in valid JSON format.
Be conservative with position sizing - capital preservation is priority #1."""

        analysis_prompt = f"""
{context.to_prompt()}

Analyze this market situation and provide a trading decision.

Think step by step:
1. What is the current trend and its strength?
2. Are we at a key support/resistance level?
3. What does volume tell us about conviction?
4. Is the regime favorable for our typical strategies?
5. What are the key risks right now?
6. What would invalidate this trade?

Respond with JSON:
{{
    "action": "LONG" | "SHORT" | "FLAT" | "REDUCE" | "ADD",
    "confidence": 0.0-1.0,
    "entry_price": number or null,
    "stop_loss": number or null,
    "take_profit": number or null,
    "position_size_pct": 0-100,
    "reasoning": "detailed explanation",
    "risks": ["risk1", "risk2"],
    "catalysts": ["what could move price"],
    "time_horizon": "scalp" | "intraday" | "swing" | "position"
}}
"""

        result = await self._query_ollama(
            analysis_prompt,
            system=system_prompt,
            temperature=0.2,
        )

        data = result.get("data", {})

        decision = AIDecision(
            action=data.get("action", "FLAT"),
            confidence=data.get("confidence", 0.0),
            entry_price=data.get("entry_price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            position_size_pct=data.get("position_size_pct", 0),
            reasoning=data.get("reasoning", "Analysis failed"),
            risks=data.get("risks", []),
            catalysts=data.get("catalysts", []),
            time_horizon=data.get("time_horizon", "intraday"),
            model_used=self.primary_model,
            generation_time_ms=result.get("elapsed_ms", 0),
        )

        self.decision_history.append(decision)
        return decision

    async def quick_decision(self, context: MarketContext) -> AIDecision:
        """
        Fast decision using lightweight model.

        For time-sensitive situations or confirmation.
        """
        prompt = f"""
{context.to_prompt()}

Quick trading decision needed. JSON response only:
{{
    "action": "LONG" | "SHORT" | "FLAT",
    "confidence": 0.0-1.0,
    "reasoning": "brief reason"
}}
"""

        result = await self._query_ollama(
            prompt,
            model=self.fast_model,
            temperature=0.1,
        )

        data = result.get("data", {})

        return AIDecision(
            action=data.get("action", "FLAT"),
            confidence=data.get("confidence", 0.0),
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            position_size_pct=0,
            reasoning=data.get("reasoning", "Quick analysis"),
            risks=[],
            catalysts=[],
            time_horizon="scalp",
            model_used=self.fast_model,
            generation_time_ms=result.get("elapsed_ms", 0),
        )

    async def ensemble_decision(self, context: MarketContext) -> AIDecision:
        """
        Multi-model ensemble for high-confidence decisions.

        Queries multiple models and aggregates their opinions.
        """
        models = [
            AIModel.REASONING.value,
            AIModel.FAST.value,
            AIModel.MULTI_EXPERT.value,
        ]

        # Query all models in parallel
        tasks = []
        for model in models:
            task = self._query_ollama(
                f"{context.to_prompt()}\n\nProvide trading decision as JSON with action, confidence, reasoning.",
                model=model,
                temperature=0.2,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate decisions
        actions = []
        confidences = []
        reasonings = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            data = result.get("data", {})
            if data.get("action"):
                actions.append(data["action"])
                confidences.append(data.get("confidence", 0.5))
                reasonings.append(f"{models[i]}: {data.get('reasoning', 'N/A')}")

        if not actions:
            return AIDecision(
                action="FLAT",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_pct=0,
                reasoning="Ensemble failed - no valid responses",
                risks=["All models failed to respond"],
                catalysts=[],
                time_horizon="intraday",
                model_used="ensemble",
                generation_time_ms=0,
            )

        # Majority voting with confidence weighting
        action_scores = {}
        for action, conf in zip(actions, confidences):
            action_scores[action] = action_scores.get(action, 0) + conf

        best_action = max(action_scores, key=action_scores.get)
        avg_confidence = sum(confidences) / len(confidences)

        # Reduce confidence if models disagree
        agreement_ratio = actions.count(best_action) / len(actions)
        final_confidence = avg_confidence * agreement_ratio

        return AIDecision(
            action=best_action,
            confidence=final_confidence,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            position_size_pct=int(final_confidence * 50),  # Max 50% on ensemble
            reasoning=f"Ensemble decision ({agreement_ratio:.0%} agreement):\n" + "\n".join(reasonings),
            risks=["Model disagreement"] if agreement_ratio < 1.0 else [],
            catalysts=[],
            time_horizon="intraday",
            model_used="ensemble",
            generation_time_ms=sum(r.get("elapsed_ms", 0) for r in results if not isinstance(r, Exception)),
        )

    async def generate_strategy(
        self,
        market_conditions: Dict[str, Any],
        performance_history: List[Dict],
    ) -> StrategyRecommendation:
        """
        Generate adaptive strategy based on current conditions.
        """
        prompt = f"""
Current Market Conditions:
{json.dumps(market_conditions, indent=2)}

Recent Performance:
{json.dumps(performance_history[-10:], indent=2)}

Generate an optimal trading strategy for these conditions.

Respond with JSON:
{{
    "strategy_name": "name",
    "parameters": {{
        "entry_type": "breakout" | "pullback" | "reversal",
        "timeframe": "1m" | "5m" | "15m" | "1h" | "4h",
        "stop_loss_atr_mult": 1.0-3.0,
        "take_profit_atr_mult": 1.5-5.0,
        "rsi_oversold": 20-40,
        "rsi_overbought": 60-80,
        "volume_filter": true | false,
        "trend_filter": true | false
    }},
    "expected_win_rate": 0.4-0.7,
    "expected_sharpe": 0.5-3.0,
    "suitable_regimes": ["trending_up", "trending_down", "ranging", "volatile"],
    "reasoning": "explanation"
}}
"""

        result = await self._query_ollama(
            prompt,
            model=AIModel.REASONING.value,
            temperature=0.4,
        )

        data = result.get("data", {})

        return StrategyRecommendation(
            strategy_name=data.get("strategy_name", "adaptive"),
            parameters=data.get("parameters", {}),
            expected_win_rate=data.get("expected_win_rate", 0.5),
            expected_sharpe=data.get("expected_sharpe", 1.0),
            suitable_regimes=data.get("suitable_regimes", ["ranging"]),
            reasoning=data.get("reasoning", "Default strategy"),
        )

    async def analyze_trade_outcome(
        self,
        trade: Dict[str, Any],
        market_context_at_entry: MarketContext,
    ) -> Dict[str, Any]:
        """
        Post-trade analysis for continuous learning.
        """
        prompt = f"""
Trade Outcome Analysis:

Entry Context:
{market_context_at_entry.to_prompt()}

Trade Details:
- Direction: {trade.get('direction')}
- Entry: ${trade.get('entry_price')}
- Exit: ${trade.get('exit_price')}
- PnL: {trade.get('pnl_pct'):+.2f}%
- Duration: {trade.get('duration_minutes')} minutes
- Exit Reason: {trade.get('exit_reason')}

Analyze what we can learn from this trade.

Respond with JSON:
{{
    "was_good_entry": true | false,
    "entry_improvements": ["suggestion1", "suggestion2"],
    "was_good_exit": true | false,
    "exit_improvements": ["suggestion1"],
    "regime_match": true | false,
    "key_lesson": "main takeaway",
    "strategy_adjustments": {{}}
}}
"""

        result = await self._query_ollama(prompt, temperature=0.3)
        return result.get("data", {})

    async def close(self):
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience function for synchronous usage
def get_ai_decision(context: MarketContext) -> AIDecision:
    """Synchronous wrapper for AI decision."""
    engine = EnhancedAIEngine()
    return asyncio.run(engine.analyze_market(context))
