"""
AI Trading Advisor - Unified AI decision making for all markets.

Connects to Ollama LLM for intelligent trading decisions.
Can be enabled/disabled per market and integrates with existing signals.
"""

from __future__ import annotations

import asyncio
import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
AI_MODEL = os.getenv("AI_MODEL", "qwen2.5:7b")  # Fast model for quick decisions
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "30"))
AI_ENABLED = os.getenv("AI_TRADING_ENABLED", "true").lower() == "true"

# State file for AI decisions
STATE_DIR = Path(os.getenv("DATA_DIR", "./data"))
AI_STATE_FILE = STATE_DIR / "ai_trading_state.json"


@dataclass
class AIAdvice:
    """AI-generated trading advice."""
    action: Literal["BUY", "SELL", "HOLD", "REDUCE"]
    confidence: float  # 0.0 - 1.0
    reasoning: str
    risk_level: Literal["low", "medium", "high", "extreme"]
    position_size_pct: float  # 0-100
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]
    market_regime: str
    warnings: List[str]
    timestamp: str
    model: str
    latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "market_regime": self.market_regime,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }

    @property
    def should_trade(self) -> bool:
        """Returns True if AI recommends taking action."""
        return self.action in ["BUY", "SELL"] and self.confidence >= 0.6

    @property
    def is_warning(self) -> bool:
        """Returns True if AI is issuing warnings."""
        return self.risk_level in ["high", "extreme"] or len(self.warnings) > 0


class AITradingAdvisor:
    """
    AI Trading Advisor for multi-market decision making.

    Features:
    - Connects to Ollama for LLM-based analysis
    - Provides buy/sell/hold recommendations
    - Risk assessment and position sizing
    - Works with any asset class (crypto, commodities, stocks)
    """

    def __init__(
        self,
        ollama_host: str = OLLAMA_HOST,
        model: str = AI_MODEL,
        timeout: int = AI_TIMEOUT,
        enabled: bool = AI_ENABLED,
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._last_advice: Dict[str, AIAdvice] = {}
        self._available = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def check_availability(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available

        try:
            client = await self._get_client()
            response = await client.get(f"{self.ollama_host}/api/tags")
            self._available = response.status_code == 200
            if self._available:
                logger.info(f"AI advisor connected to Ollama at {self.ollama_host}")
            return self._available
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
            return False

    async def get_advice(
        self,
        symbol: str,
        current_price: float,
        price_change_1h: float,
        price_change_24h: float,
        current_signal: str,
        regime: str,
        confidence: float,
        portfolio_value: float,
        position_value: float,
        pnl_pct: float,
        asset_type: str = "crypto",
    ) -> AIAdvice:
        """
        Get AI trading advice for a specific asset.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT", "XAU/USD")
            current_price: Current market price
            price_change_1h: 1-hour price change percentage
            price_change_24h: 24-hour price change percentage
            current_signal: Current strategy signal (LONG, SHORT, FLAT)
            regime: Market regime (bull, bear, sideways, volatile)
            confidence: Strategy confidence (0-1)
            portfolio_value: Total portfolio value
            position_value: Current position value in this asset
            pnl_pct: Current P&L percentage
            asset_type: Type of asset (crypto, commodity, stock)

        Returns:
            AIAdvice with recommendation
        """
        if not self.enabled:
            return self._default_advice(symbol, current_signal, regime)

        if not await self.check_availability():
            logger.warning("AI not available, using default advice")
            return self._default_advice(symbol, current_signal, regime)

        start_time = datetime.now()

        # Build context prompt
        prompt = self._build_prompt(
            symbol=symbol,
            current_price=current_price,
            price_change_1h=price_change_1h,
            price_change_24h=price_change_24h,
            current_signal=current_signal,
            regime=regime,
            confidence=confidence,
            portfolio_value=portfolio_value,
            position_value=position_value,
            pnl_pct=pnl_pct,
            asset_type=asset_type,
        )

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 4096,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()

            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            content = result.get("message", {}).get("content", "{}")
            data = json.loads(content)

            advice = AIAdvice(
                action=data.get("action", "HOLD"),
                confidence=min(1.0, max(0.0, data.get("confidence", 0.5))),
                reasoning=data.get("reasoning", "No reasoning provided"),
                risk_level=data.get("risk_level", "medium"),
                position_size_pct=min(100, max(0, data.get("position_size_pct", 10))),
                stop_loss_pct=data.get("stop_loss_pct"),
                take_profit_pct=data.get("take_profit_pct"),
                market_regime=data.get("market_regime", regime),
                warnings=data.get("warnings", []),
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=self.model,
                latency_ms=latency_ms,
            )

            self._last_advice[symbol] = advice
            logger.info(f"AI advice for {symbol}: {advice.action} (conf: {advice.confidence:.0%})")

            return advice

        except Exception as e:
            logger.error(f"AI advice failed for {symbol}: {e}")
            return self._default_advice(symbol, current_signal, regime)

    def _get_system_prompt(self) -> str:
        return """You are an expert quantitative trading AI advisor.
Your role is to analyze market conditions and provide clear trading recommendations.
Always prioritize capital preservation - never recommend aggressive positions in uncertain markets.
Respond ONLY with valid JSON matching the requested format.
Be concise and direct in your reasoning."""

    def _build_prompt(
        self,
        symbol: str,
        current_price: float,
        price_change_1h: float,
        price_change_24h: float,
        current_signal: str,
        regime: str,
        confidence: float,
        portfolio_value: float,
        position_value: float,
        pnl_pct: float,
        asset_type: str,
    ) -> str:
        position_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0

        return f"""Analyze this trading situation and provide advice:

ASSET: {symbol} ({asset_type})
CURRENT PRICE: ${current_price:,.2f}
1H CHANGE: {price_change_1h:+.2f}%
24H CHANGE: {price_change_24h:+.2f}%

STRATEGY SIGNAL: {current_signal}
MARKET REGIME: {regime}
SIGNAL CONFIDENCE: {confidence:.0%}

PORTFOLIO:
- Total Value: ${portfolio_value:,.2f}
- Position in {symbol}: ${position_value:,.2f} ({position_pct:.1f}%)
- Current P&L: {pnl_pct:+.2f}%

Based on this data, provide trading advice in JSON format:
{{
    "action": "BUY" | "SELL" | "HOLD" | "REDUCE",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "risk_level": "low" | "medium" | "high" | "extreme",
    "position_size_pct": 0-100,
    "stop_loss_pct": number or null,
    "take_profit_pct": number or null,
    "market_regime": "regime description",
    "warnings": ["any warnings"]
}}

Key considerations:
- If regime is volatile/sideways with losses, recommend REDUCE or HOLD
- If P&L is significantly negative, prioritize risk management
- Never recommend more than 20% position size in volatile markets
- Include stop_loss if recommending BUY"""

    def _default_advice(
        self,
        symbol: str,
        current_signal: str,
        regime: str,
    ) -> AIAdvice:
        """Return conservative default advice when AI is unavailable."""
        # Map strategy signals to actions
        action_map = {
            "LONG": "HOLD",  # Don't auto-buy without AI
            "SHORT": "HOLD",  # Don't auto-sell without AI
            "FLAT": "HOLD",
        }

        return AIAdvice(
            action=action_map.get(current_signal, "HOLD"),
            confidence=0.3,
            reasoning="AI unavailable - using conservative default",
            risk_level="medium",
            position_size_pct=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            market_regime=regime,
            warnings=["AI advisor unavailable - using conservative defaults"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            model="default",
            latency_ms=0,
        )

    def get_last_advice(self, symbol: str) -> Optional[AIAdvice]:
        """Get the last advice for a symbol."""
        return self._last_advice.get(symbol)

    def get_all_advice(self) -> Dict[str, Dict]:
        """Get all last advice as dict."""
        return {k: v.to_dict() for k, v in self._last_advice.items()}

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance for easy access
_advisor: Optional[AITradingAdvisor] = None


def get_advisor() -> AITradingAdvisor:
    """Get or create the global AI advisor instance."""
    global _advisor
    if _advisor is None:
        _advisor = AITradingAdvisor()
    return _advisor


async def get_ai_advice(
    symbol: str,
    current_price: float,
    price_change_1h: float = 0.0,
    price_change_24h: float = 0.0,
    current_signal: str = "FLAT",
    regime: str = "unknown",
    confidence: float = 0.5,
    portfolio_value: float = 10000.0,
    position_value: float = 0.0,
    pnl_pct: float = 0.0,
    asset_type: str = "crypto",
) -> AIAdvice:
    """
    Convenience function to get AI advice.

    Usage:
        advice = await get_ai_advice(
            symbol="BTC/USDT",
            current_price=94000,
            price_change_24h=-2.5,
            current_signal="LONG",
            regime="volatile",
            portfolio_value=10000,
            position_value=1500,
            pnl_pct=-1.5,
        )

        if advice.action == "REDUCE":
            # Reduce position
            pass
    """
    advisor = get_advisor()
    return await advisor.get_advice(
        symbol=symbol,
        current_price=current_price,
        price_change_1h=price_change_1h,
        price_change_24h=price_change_24h,
        current_signal=current_signal,
        regime=regime,
        confidence=confidence,
        portfolio_value=portfolio_value,
        position_value=position_value,
        pnl_pct=pnl_pct,
        asset_type=asset_type,
    )
