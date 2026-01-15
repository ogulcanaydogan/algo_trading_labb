"""
LLM-based Strategy Advisor.

Uses local LLM (Ollama) to provide strategy suggestions and analysis.

Updated with Risk Guardian integration for risk-aware recommendations.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .prompts import (
    STRATEGY_ADVISOR_PROMPT,
    TRADE_EXPLAINER_PROMPT,
    MARKET_ANALYSIS_PROMPT,
    STRATEGY_GENERATION_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    LEVERAGE_ADVISOR_PROMPT,
    DRAWDOWN_RECOVERY_PROMPT,
    PRE_TRADE_RISK_CHECK_PROMPT,
    DAILY_RISK_REPORT_PROMPT,
)

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Risk decision from LLM"""
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE = "reduce"


@dataclass
class RiskAssessmentResult:
    """Result of LLM risk assessment"""
    decision: RiskDecision
    reasoning: str
    confidence: float
    suggested_adjustments: Dict[str, Any]
    warnings: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_adjustments": self.suggested_adjustments,
            "warnings": self.warnings,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class LeverageRecommendation:
    """Leverage recommendation from LLM"""
    recommended_leverage: float
    max_safe_leverage: float
    position_size_pct: float
    stop_loss_pct: float
    reasoning: str
    confidence: float
    warnings: List[str]

    def to_dict(self) -> Dict:
        return {
            "recommended_leverage": self.recommended_leverage,
            "max_safe_leverage": self.max_safe_leverage,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


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

    # ============================================================
    # Risk Guardian Integration Methods
    # ============================================================

    def assess_trade_risk(
        self,
        symbol: str,
        direction: str,
        size_pct: float,
        proposed_leverage: float,
        risk_guardian_status: Dict[str, Any],
        risk_events: List[Dict] = None,
    ) -> RiskAssessmentResult:
        """
        Get LLM risk assessment for a proposed trade.

        Args:
            symbol: Trading symbol
            direction: long or short
            size_pct: Position size as % of equity
            proposed_leverage: Proposed leverage level
            risk_guardian_status: Current Risk Guardian status dict
            risk_events: Recent risk events

        Returns:
            RiskAssessmentResult with decision and reasoning
        """
        if not self.is_available():
            return self._fallback_risk_assessment(
                risk_guardian_status, size_pct, proposed_leverage
            )

        prompt = RISK_ASSESSMENT_PROMPT.format(
            risk_level=risk_guardian_status.get("risk_level", "unknown"),
            daily_pnl_pct=risk_guardian_status.get("daily_pnl_pct", 0),
            daily_loss_limit=risk_guardian_status.get("daily_loss_limit", 2),
            drawdown_pct=risk_guardian_status.get("drawdown_pct", 0),
            max_drawdown=risk_guardian_status.get("max_drawdown_limit", 10),
            consecutive_losses=risk_guardian_status.get("consecutive_losses", 0),
            max_consecutive=risk_guardian_status.get("max_consecutive", 5),
            current_leverage=risk_guardian_status.get("current_leverage", 1),
            max_leverage=risk_guardian_status.get("max_leverage", 10),
            margin_usage=risk_guardian_status.get("margin_usage_pct", 0),
            kill_switch_status="ACTIVE" if risk_guardian_status.get("kill_switch_active") else "OFF",
            risk_events=json.dumps(risk_events or [], indent=2),
            symbol=symbol,
            direction=direction,
            size_pct=size_pct,
            proposed_leverage=proposed_leverage,
        )

        response = self._query_llm(prompt)

        if not response:
            return self._fallback_risk_assessment(
                risk_guardian_status, size_pct, proposed_leverage
            )

        return self._parse_risk_assessment(response)

    def get_leverage_recommendation(
        self,
        symbol: str,
        direction: str,
        volatility: float,
        avg_volatility: float,
        regime: str,
        trend_strength: float,
        equity: float,
        unrealized_pnl: float,
        margin_available: float,
        current_leverage: float,
        max_leverage: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_leverage: float,
    ) -> LeverageRecommendation:
        """
        Get LLM recommendation for leverage level.

        Returns:
            LeverageRecommendation with suggested leverage and sizing
        """
        if not self.is_available():
            return self._fallback_leverage_recommendation(
                volatility, avg_volatility, max_leverage, kelly_leverage
            )

        prompt = LEVERAGE_ADVISOR_PROMPT.format(
            symbol=symbol,
            volatility=volatility * 100,
            avg_volatility=avg_volatility * 100,
            regime=regime,
            trend_strength=trend_strength,
            equity=equity,
            unrealized_pnl=unrealized_pnl,
            margin_available=margin_available,
            current_leverage=current_leverage,
            max_leverage=max_leverage,
            win_rate=win_rate * 100,
            avg_win=avg_win * 100,
            avg_loss=avg_loss * 100,
            kelly_leverage=kelly_leverage,
            direction=direction,
        )

        response = self._query_llm(prompt)

        if not response:
            return self._fallback_leverage_recommendation(
                volatility, avg_volatility, max_leverage, kelly_leverage
            )

        return self._parse_leverage_recommendation(response, max_leverage)

    def get_drawdown_recovery_plan(
        self,
        drawdown_pct: float,
        peak_equity: float,
        current_equity: float,
        days_in_drawdown: int,
        losing_streak: int,
        win_rate: float,
        risk_actions: List[str],
        historical_drawdowns: List[Dict],
        similar_conditions: str,
    ) -> str:
        """
        Get LLM advice for recovering from a drawdown.

        Returns:
            Recovery plan text
        """
        if not self.is_available():
            return self._fallback_recovery_plan(drawdown_pct, losing_streak)

        prompt = DRAWDOWN_RECOVERY_PROMPT.format(
            drawdown_pct=drawdown_pct,
            peak_equity=peak_equity,
            current_equity=current_equity,
            days_in_drawdown=days_in_drawdown,
            losing_streak=losing_streak,
            win_rate=win_rate * 100,
            risk_actions="\n".join(f"- {action}" for action in risk_actions),
            historical_drawdowns=json.dumps(historical_drawdowns, indent=2),
            similar_conditions=similar_conditions,
        )

        response = self._query_llm(prompt)
        return response or self._fallback_recovery_plan(drawdown_pct, losing_streak)

    def pre_trade_risk_check(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        size_pct: float,
        leverage: float,
        stop_loss: float,
        stop_loss_pct: float,
        take_profit: float,
        equity: float,
        open_positions: int,
        total_exposure: float,
        daily_pnl: float,
        open_risk: float,
        risk_level: str,
        trading_allowed: bool,
        adjusted_size: float,
        regime: str,
        volatility: float,
        news_events: str = "None",
    ) -> RiskAssessmentResult:
        """
        Perform pre-trade risk check with LLM.

        Returns:
            RiskAssessmentResult with binary APPROVE/REJECT decision
        """
        if not self.is_available():
            return self._fallback_pre_trade_check(
                trading_allowed, risk_level, size_pct, adjusted_size
            )

        prompt = PRE_TRADE_RISK_CHECK_PROMPT.format(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            size_pct=size_pct,
            leverage=leverage,
            stop_loss=stop_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit=take_profit,
            equity=equity,
            open_positions=open_positions,
            total_exposure=total_exposure,
            daily_pnl=daily_pnl,
            open_risk=open_risk,
            risk_level=risk_level,
            trading_allowed="YES" if trading_allowed else "NO",
            adjusted_size=adjusted_size,
            regime=regime,
            volatility=volatility * 100,
            news_events=news_events,
        )

        response = self._query_llm(prompt)

        if not response:
            return self._fallback_pre_trade_check(
                trading_allowed, risk_level, size_pct, adjusted_size
            )

        return self._parse_risk_assessment(response)

    def generate_daily_risk_report(
        self,
        date: datetime,
        start_equity: float,
        end_equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_trades: int,
        win_rate: float,
        risk_events: List[Dict],
        positions: List[Dict],
        drawdown: float,
        max_drawdown_today: float,
        avg_leverage: float,
        margin_util: float,
    ) -> str:
        """
        Generate daily risk report using LLM.

        Returns:
            Formatted daily risk report
        """
        if not self.is_available():
            return self._fallback_daily_report(
                date, daily_pnl_pct, drawdown, win_rate
            )

        prompt = DAILY_RISK_REPORT_PROMPT.format(
            date=date.strftime("%Y-%m-%d"),
            start_equity=start_equity,
            end_equity=end_equity,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_trades=total_trades,
            win_rate=win_rate * 100,
            risk_events=json.dumps(risk_events, indent=2) if risk_events else "None",
            positions=json.dumps(positions, indent=2) if positions else "No open positions",
            drawdown=drawdown,
            max_drawdown_today=max_drawdown_today,
            avg_leverage=avg_leverage,
            margin_util=margin_util,
        )

        response = self._query_llm(prompt)
        return response or self._fallback_daily_report(
            date, daily_pnl_pct, drawdown, win_rate
        )

    # ============================================================
    # Risk-Related Fallback Methods
    # ============================================================

    def _fallback_risk_assessment(
        self,
        risk_status: Dict[str, Any],
        size_pct: float,
        leverage: float,
    ) -> RiskAssessmentResult:
        """Provide rule-based risk assessment when LLM unavailable."""
        warnings = []
        adjustments = {}

        # Check kill switch
        if risk_status.get("kill_switch_active"):
            return RiskAssessmentResult(
                decision=RiskDecision.REJECT,
                reasoning="Kill switch is active. All trading halted.",
                confidence=1.0,
                suggested_adjustments={},
                warnings=["Kill switch active - investigate before resuming"],
            )

        # Check risk level
        risk_level = risk_status.get("risk_level", "normal")
        daily_pnl = risk_status.get("daily_pnl_pct", 0)
        drawdown = risk_status.get("drawdown_pct", 0)

        if risk_level == "emergency" or risk_level == "critical":
            return RiskAssessmentResult(
                decision=RiskDecision.REJECT,
                reasoning=f"Risk level {risk_level} - no new positions allowed.",
                confidence=0.95,
                suggested_adjustments={},
                warnings=[f"Risk level: {risk_level}", f"Drawdown: {drawdown}%"],
            )

        if risk_level == "high":
            adjustments["size_reduction"] = 0.5
            warnings.append("High risk - position size should be halved")

        # Check leverage
        max_leverage = risk_status.get("max_leverage", 10)
        if leverage > max_leverage:
            adjustments["max_leverage"] = max_leverage
            warnings.append(f"Leverage {leverage}x exceeds max {max_leverage}x")

        # Check daily loss
        daily_limit = risk_status.get("daily_loss_limit", 2)
        if daily_pnl < -daily_limit * 0.8:
            warnings.append(f"Approaching daily loss limit ({daily_pnl:.1f}%)")
            adjustments["size_reduction"] = min(adjustments.get("size_reduction", 1.0), 0.25)

        # Determine decision
        if adjustments:
            decision = RiskDecision.REDUCE
            reasoning = "Trade allowed with adjustments. " + "; ".join(warnings)
        else:
            decision = RiskDecision.APPROVE
            reasoning = "Trade approved within risk limits."

        return RiskAssessmentResult(
            decision=decision,
            reasoning=reasoning,
            confidence=0.7,
            suggested_adjustments=adjustments,
            warnings=warnings,
        )

    def _fallback_leverage_recommendation(
        self,
        volatility: float,
        avg_volatility: float,
        max_leverage: float,
        kelly_leverage: float,
    ) -> LeverageRecommendation:
        """Provide rule-based leverage recommendation when LLM unavailable."""
        warnings = []

        # Volatility-adjusted leverage
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1.0

        if vol_ratio > 1.5:
            base_leverage = 1.0
            warnings.append("High volatility - using minimum leverage")
        elif vol_ratio > 1.2:
            base_leverage = 2.0
            warnings.append("Elevated volatility - reduced leverage")
        else:
            base_leverage = min(kelly_leverage * 0.5, 3.0)  # Half Kelly, max 3x

        # Cap at max leverage
        recommended = min(base_leverage, max_leverage)
        max_safe = min(base_leverage * 1.5, max_leverage)

        # Position size based on leverage
        position_size = min(10 / recommended, 5)  # Max 5% at 2x leverage
        stop_loss = 2.0 / recommended  # Tighter stop at higher leverage

        return LeverageRecommendation(
            recommended_leverage=recommended,
            max_safe_leverage=max_safe,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss,
            reasoning=f"Volatility ratio {vol_ratio:.2f}, Kelly suggests {kelly_leverage:.1f}x. Using conservative {recommended:.1f}x.",
            confidence=0.6,
            warnings=warnings,
        )

    def _fallback_recovery_plan(self, drawdown_pct: float, losing_streak: int) -> str:
        """Provide rule-based recovery plan when LLM unavailable."""
        plan = f"""DRAWDOWN RECOVERY PLAN (Rule-Based)

Current Status:
- Drawdown: {drawdown_pct:.1f}%
- Consecutive Losses: {losing_streak}

Immediate Actions:
"""
        if drawdown_pct > 15:
            plan += """- STOP TRADING for 24-48 hours
- Review all open positions for potential closure
- Reduce max position size to 25% of normal
"""
        elif drawdown_pct > 10:
            plan += """- Reduce position size to 50% of normal
- Only take A+ setups
- Review strategy performance in current regime
"""
        elif drawdown_pct > 5:
            plan += """- Reduce position size to 75% of normal
- Tighten stop losses by 20%
- Focus on high-conviction trades only
"""
        else:
            plan += """- Minor adjustments only
- Continue with caution
- Monitor for further deterioration
"""

        if losing_streak >= 5:
            plan += f"""
LOSING STREAK WARNING ({losing_streak} consecutive losses):
- Take a break from trading
- Review recent trades for pattern
- Consider strategy adjustment or regime change
"""

        plan += """
Recovery Criteria:
- Return to normal sizing after 3 consecutive winning trades
- Full position size after drawdown recovers to 50% of peak loss
"""
        return plan

    def _fallback_pre_trade_check(
        self,
        trading_allowed: bool,
        risk_level: str,
        size_pct: float,
        adjusted_size: float,
    ) -> RiskAssessmentResult:
        """Provide rule-based pre-trade check when LLM unavailable."""
        if not trading_allowed:
            return RiskAssessmentResult(
                decision=RiskDecision.REJECT,
                reasoning="Trading not allowed by Risk Guardian.",
                confidence=1.0,
                suggested_adjustments={},
                warnings=["Risk Guardian has disabled trading"],
            )

        warnings = []
        adjustments = {}

        if adjusted_size < size_pct:
            adjustments["size"] = adjusted_size
            warnings.append(f"Size reduced from {size_pct}% to {adjusted_size}%")

        if risk_level in ["high", "critical", "emergency"]:
            return RiskAssessmentResult(
                decision=RiskDecision.REJECT,
                reasoning=f"Risk level {risk_level} - trade rejected.",
                confidence=0.9,
                suggested_adjustments=adjustments,
                warnings=warnings + [f"Risk level: {risk_level}"],
            )

        if risk_level == "elevated":
            warnings.append("Elevated risk - trade with caution")

        return RiskAssessmentResult(
            decision=RiskDecision.APPROVE if not adjustments else RiskDecision.REDUCE,
            reasoning="Trade approved by rule-based check.",
            confidence=0.7,
            suggested_adjustments=adjustments,
            warnings=warnings,
        )

    def _fallback_daily_report(
        self,
        date: datetime,
        daily_pnl_pct: float,
        drawdown: float,
        win_rate: float,
    ) -> str:
        """Provide rule-based daily report when LLM unavailable."""
        # Determine status
        if daily_pnl_pct < -2 or drawdown > 10:
            status = "RED"
        elif daily_pnl_pct < -1 or drawdown > 5:
            status = "YELLOW"
        else:
            status = "GREEN"

        return f"""DAILY RISK REPORT (Rule-Based)
Date: {date.strftime("%Y-%m-%d")}
Status: {status}

Performance:
- Daily PnL: {daily_pnl_pct:+.2f}%
- Current Drawdown: {drawdown:.2f}%
- Win Rate: {win_rate*100:.1f}%

Assessment:
{"- Consider reducing position sizes" if status == "YELLOW" else ""}
{"- TRADING SHOULD BE HALTED - review strategy" if status == "RED" else ""}
{"- Performance within acceptable limits" if status == "GREEN" else ""}

Note: LLM unavailable. This is a rule-based summary.
"""

    def _parse_risk_assessment(self, response: str) -> RiskAssessmentResult:
        """Parse LLM response into RiskAssessmentResult."""
        response_lower = response.lower()

        # Determine decision
        if "reject" in response_lower[:200]:
            decision = RiskDecision.REJECT
        elif "reduce" in response_lower[:200]:
            decision = RiskDecision.REDUCE
        else:
            decision = RiskDecision.APPROVE

        # Extract warnings
        warnings = []
        for line in response.split("\n"):
            if "warning" in line.lower() or "risk" in line.lower():
                warnings.append(line.strip())

        # Extract adjustments
        adjustments = {}
        if "size" in response_lower and ("reduce" in response_lower or "decrease" in response_lower):
            adjustments["reduce_size"] = True
        if "leverage" in response_lower and ("lower" in response_lower or "reduce" in response_lower):
            adjustments["reduce_leverage"] = True

        return RiskAssessmentResult(
            decision=decision,
            reasoning=response[:500],
            confidence=0.75,
            suggested_adjustments=adjustments,
            warnings=warnings[:5],
        )

    def _parse_leverage_recommendation(
        self, response: str, max_leverage: float
    ) -> LeverageRecommendation:
        """Parse LLM response into LeverageRecommendation."""
        import re

        # Try to extract leverage numbers
        leverage_matches = re.findall(r'(\d+(?:\.\d+)?)\s*x', response.lower())
        leverages = [float(m) for m in leverage_matches if float(m) <= max_leverage]

        recommended = leverages[0] if leverages else 1.0
        max_safe = leverages[1] if len(leverages) > 1 else min(recommended * 1.5, max_leverage)

        # Extract position size
        size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%.*(?:position|size|equity)', response.lower())
        position_size = float(size_matches[0]) if size_matches else 5.0

        # Extract stop loss
        stop_matches = re.findall(r'stop.*?(\d+(?:\.\d+)?)\s*%', response.lower())
        stop_loss = float(stop_matches[0]) if stop_matches else 2.0

        # Extract warnings
        warnings = []
        for line in response.split("\n"):
            if "warning" in line.lower() or "caution" in line.lower():
                warnings.append(line.strip())

        return LeverageRecommendation(
            recommended_leverage=recommended,
            max_safe_leverage=max_safe,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss,
            reasoning=response[:300],
            confidence=0.7,
            warnings=warnings[:3],
        )
