"""
LLM-based Strategy Advisor.

Uses local LLM (Ollama) to provide strategy suggestions and analysis.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .prompts import (
    STRATEGY_ADVISOR_PROMPT,
    TRADE_EXPLAINER_PROMPT,
    MARKET_ANALYSIS_PROMPT,
    STRATEGY_GENERATION_PROMPT,
)


@dataclass
class StrategyAdvice:
    """Advice from the LLM advisor."""
    assessment: str
    parameter_suggestions: Dict[str, Any]
    alternative_strategies: List[str]
    risk_recommendations: List[str]
    confidence: float
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "assessment": self.assessment,
            "parameter_suggestions": self.parameter_suggestions,
            "alternative_strategies": self.alternative_strategies,
            "risk_recommendations": self.risk_recommendations,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
        }


class LLMAdvisor:
    """
    Local LLM Advisor for Trading Strategy Improvement.

    Uses Ollama to run local models like:
    - llama3
    - mistral
    - codellama
    - phi3

    Features:
    - Strategy performance analysis
    - Parameter optimization suggestions
    - Trade explanation
    - New strategy generation
    """

    def __init__(
        self,
        model: str = "llama3",
        ollama_host: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """
        Initialize the LLM advisor.

        Args:
            model: Ollama model name (llama3, mistral, phi3, etc.)
            ollama_host: Ollama API endpoint
            timeout: Request timeout in seconds
        """
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        self._is_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._is_available is not None:
            return self._is_available

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._is_available = result.returncode == 0
            return self._is_available
        except (subprocess.SubprocessError, FileNotFoundError):
            self._is_available = False
            return False

    def get_strategy_advice(
        self,
        symbol: str,
        timeframe: str,
        regime: str,
        metrics: Dict[str, float],
        current_strategy: str,
        recent_trades: List[Dict],
    ) -> StrategyAdvice:
        """
        Get strategy improvement advice from the LLM.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            regime: Current market regime
            metrics: Performance metrics (win_rate, sharpe, etc.)
            current_strategy: Name of current strategy
            recent_trades: List of recent trade dictionaries

        Returns:
            StrategyAdvice with suggestions
        """
        if not self.is_available():
            return self._fallback_advice(metrics)

        prompt = STRATEGY_ADVISOR_PROMPT.format(
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            metrics=json.dumps(metrics, indent=2),
            current_strategy=current_strategy,
            recent_trades=json.dumps(recent_trades[-5:], indent=2),  # Last 5 trades
        )

        response = self._query_llm(prompt)

        if not response:
            return self._fallback_advice(metrics)

        return self._parse_advice_response(response, metrics)

    def explain_trade(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        strategy: str,
        indicators: Dict[str, float],
        market_conditions: str,
    ) -> str:
        """
        Get an explanation for a specific trade.

        Returns human-readable explanation of why the trade was taken.
        """
        if not self.is_available():
            return self._fallback_trade_explanation(direction, pnl)

        prompt = TRADE_EXPLAINER_PROMPT.format(
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            strategy=strategy,
            indicators=json.dumps(indicators, indent=2),
            market_conditions=market_conditions,
        )

        response = self._query_llm(prompt)
        return response or self._fallback_trade_explanation(direction, pnl)

    def analyze_market(
        self,
        symbol: str,
        price: float,
        returns: List[float],
        volatility: float,
        rsi: float,
        trend: str,
        volume_analysis: str,
    ) -> str:
        """
        Get market analysis from the LLM.

        Returns detailed market analysis text.
        """
        if not self.is_available():
            return self._fallback_market_analysis(symbol, trend, rsi)

        prompt = MARKET_ANALYSIS_PROMPT.format(
            symbol=symbol,
            price=price,
            returns=returns[-10:],  # Last 10 periods
            volatility=volatility,
            rsi=rsi,
            trend=trend,
            volume_analysis=volume_analysis,
        )

        response = self._query_llm(prompt)
        return response or self._fallback_market_analysis(symbol, trend, rsi)

    def suggest_new_strategy(
        self,
        market_analysis: str,
        performance_history: Dict,
        current_strategies: List[str],
    ) -> str:
        """
        Generate a new strategy suggestion.

        Returns strategy specification that can be implemented.
        """
        if not self.is_available():
            return "LLM not available. Consider trying: Bollinger Band breakout for volatile markets, or RSI mean reversion for sideways markets."

        prompt = STRATEGY_GENERATION_PROMPT.format(
            market_analysis=market_analysis,
            performance_history=json.dumps(performance_history, indent=2),
            current_strategies=", ".join(current_strategies),
        )

        return self._query_llm(prompt) or "Unable to generate strategy suggestion."

    def _query_llm(self, prompt: str) -> Optional[str]:
        """Query the local LLM via Ollama CLI."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Ollama error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"LLM query timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None

    def _parse_advice_response(
        self,
        response: str,
        metrics: Dict[str, float],
    ) -> StrategyAdvice:
        """Parse LLM response into structured advice."""
        # Simple parsing - extract key sections
        lines = response.split("\n")

        assessment = response[:500]  # First 500 chars as assessment
        parameter_suggestions = {}
        alternative_strategies = []
        risk_recommendations = []

        for line in lines:
            line_lower = line.lower()

            # Look for parameter suggestions
            if "ema" in line_lower and any(c.isdigit() for c in line):
                parameter_suggestions["ema_adjustment"] = line.strip()
            if "rsi" in line_lower and any(c.isdigit() for c in line):
                parameter_suggestions["rsi_adjustment"] = line.strip()
            if "stop" in line_lower and "loss" in line_lower:
                parameter_suggestions["stop_loss_adjustment"] = line.strip()

            # Look for strategy suggestions
            if "bollinger" in line_lower:
                alternative_strategies.append("Bollinger Band Strategy")
            if "macd" in line_lower:
                alternative_strategies.append("MACD Divergence Strategy")
            if "mean reversion" in line_lower:
                alternative_strategies.append("Mean Reversion Strategy")

            # Look for risk recommendations
            if "risk" in line_lower or "position size" in line_lower:
                risk_recommendations.append(line.strip())

        # Calculate confidence based on metrics
        confidence = self._calculate_advice_confidence(metrics)

        return StrategyAdvice(
            assessment=assessment,
            parameter_suggestions=parameter_suggestions,
            alternative_strategies=list(set(alternative_strategies))[:3],
            risk_recommendations=risk_recommendations[:5],
            confidence=confidence,
        )

    def _calculate_advice_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence in the advice based on data quality."""
        confidence = 0.5  # Base confidence

        # More trades = more confident advice
        total_trades = metrics.get("total_trades", 0)
        if total_trades > 100:
            confidence += 0.2
        elif total_trades > 50:
            confidence += 0.1

        # Better metrics = higher confidence in current approach
        win_rate = metrics.get("win_rate", 0)
        if win_rate > 0.6:
            confidence += 0.1
        elif win_rate < 0.4:
            confidence += 0.15  # More room for improvement

        return min(confidence, 0.95)

    def _fallback_advice(self, metrics: Dict[str, float]) -> StrategyAdvice:
        """Provide rule-based advice when LLM is unavailable."""
        win_rate = metrics.get("win_rate", 0.5)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)

        assessment = "LLM unavailable. Rule-based analysis: "
        parameter_suggestions = {}
        alternative_strategies = []
        risk_recommendations = []

        if win_rate < 0.45:
            assessment += "Low win rate suggests entry signals need refinement. "
            parameter_suggestions["entry_threshold"] = "Increase to 0.6"
            alternative_strategies.append("Consider RSI Mean Reversion")

        if sharpe < 0.5:
            assessment += "Low Sharpe ratio indicates poor risk-adjusted returns. "
            parameter_suggestions["take_profit"] = "Increase by 50%"

        if max_dd > 15:
            assessment += "High drawdown requires better risk management. "
            risk_recommendations.append("Reduce position size by 30%")
            risk_recommendations.append("Add trailing stop loss")

        if win_rate > 0.55 and sharpe > 1.0:
            assessment += "Good performance. Consider scaling up carefully."
            parameter_suggestions["position_size"] = "Can increase by 20%"

        return StrategyAdvice(
            assessment=assessment,
            parameter_suggestions=parameter_suggestions,
            alternative_strategies=alternative_strategies,
            risk_recommendations=risk_recommendations,
            confidence=0.6,
        )

    def _fallback_trade_explanation(self, direction: str, pnl: float) -> str:
        """Provide basic trade explanation when LLM unavailable."""
        outcome = "profitable" if pnl > 0 else "loss-making"
        return f"This {direction} trade was {outcome} (P&L: ${pnl:.2f}). LLM unavailable for detailed analysis."

    def _fallback_market_analysis(self, symbol: str, trend: str, rsi: float) -> str:
        """Provide basic market analysis when LLM unavailable."""
        rsi_condition = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        return f"{symbol} is in a {trend} trend with RSI {rsi_condition} at {rsi:.1f}. LLM unavailable for detailed analysis."
