"""
Trade Explainer - Human-Readable Decision Explanations.

Generates clear explanations for every trading decision:
- Entry reasoning
- Exit reasoning
- Risk assessment
- Market context

Uses hybrid LLM routing for cost-effective explanations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .llm_router import LLMRouter, LLMRequest, LLMResponse, RequestPriority

logger = logging.getLogger(__name__)


@dataclass
class TradeExplanation:
    """A human-readable trade explanation."""
    # Trade info
    action: str  # BUY, SELL, HOLD
    symbol: str
    price: float
    quantity: float = 0.0
    position_value: float = 0.0

    # Reasoning
    signal_reason: str = ""
    regime_context: str = ""
    news_context: str = ""
    risk_assessment: str = ""

    # Confidence
    overall_confidence: float = 0.0
    ml_confidence: float = 0.0
    pattern_confidence: float = 0.0

    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_pct: float = 0.0

    # Formatted output
    formatted_explanation: str = ""
    short_explanation: str = ""  # For Telegram

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    backend_used: str = ""
    generation_cost: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "position_value": self.position_value,
            "signal_reason": self.signal_reason,
            "regime_context": self.regime_context,
            "news_context": self.news_context,
            "risk_assessment": self.risk_assessment,
            "overall_confidence": self.overall_confidence,
            "ml_confidence": self.ml_confidence,
            "pattern_confidence": self.pattern_confidence,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_pct": self.position_size_pct,
            "formatted_explanation": self.formatted_explanation,
            "short_explanation": self.short_explanation,
            "timestamp": self.timestamp.isoformat(),
            "backend_used": self.backend_used,
            "generation_cost": self.generation_cost,
        }


class TradeExplainer:
    """
    Generates human-readable trade explanations.

    Uses LLM for detailed explanations with rule-based fallback.
    Integrates with Telegram for real-time notifications.
    """

    EXPLANATION_SYSTEM_PROMPT = """You are a professional trading analyst explaining trade decisions.
Your explanations should be:
- Clear and concise
- Focused on the reasoning behind the trade
- Include risk assessment
- Be suitable for a Telegram notification (compact format)

Always structure your response as a clear trade explanation."""

    def __init__(
        self,
        llm_router: Optional[LLMRouter] = None,
        telegram_enabled: bool = True,
    ):
        """
        Initialize the Trade Explainer.

        Args:
            llm_router: LLM router for generating explanations
            telegram_enabled: Whether to send explanations via Telegram
        """
        self.llm_router = llm_router or LLMRouter()
        self.telegram_enabled = telegram_enabled
        self._notifier = None

        logger.info("Trade Explainer initialized")

    @property
    def notifier(self):
        """Lazy load notification manager."""
        if self._notifier is None and self.telegram_enabled:
            try:
                from bot.notifications import NotificationManager, TelegramChannel
                import os

                token = os.getenv("TELEGRAM_BOT_TOKEN")
                chat_id = os.getenv("TELEGRAM_CHAT_ID")

                if token and chat_id:
                    manager = NotificationManager()
                    telegram = TelegramChannel(bot_token=token, chat_id=chat_id)
                    manager.add_channel("telegram", telegram)
                    self._notifier = manager
                    logger.info("Telegram notifications enabled for Trade Explainer")
            except Exception as e:
                logger.warning(f"Could not initialize Telegram: {e}")
                self._notifier = False
        return self._notifier if self._notifier else None

    def explain_entry(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        signal: Dict[str, Any],
        regime: str = "unknown",
        news_summary: str = "",
        portfolio_context: Dict[str, Any] = None,
        pattern_stats: Dict[str, Any] = None,
    ) -> TradeExplanation:
        """
        Generate explanation for a trade entry.

        Args:
            symbol: Trading symbol
            action: BUY or SELL
            price: Entry price
            quantity: Position quantity
            signal: Signal information (confidence, reason, etc.)
            regime: Current market regime
            news_summary: Recent news summary
            portfolio_context: Portfolio information
            pattern_stats: Historical pattern statistics

        Returns:
            TradeExplanation with formatted output
        """
        portfolio_context = portfolio_context or {}
        pattern_stats = pattern_stats or {}

        # Build context for LLM
        context = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "position_value": price * quantity,
            "confidence": signal.get("confidence", 0),
            "signal_reason": signal.get("reason", "ML signal"),
            "regime": regime,
            "portfolio_value": portfolio_context.get("total_value", 0),
            "trade_size_pct": (price * quantity) / portfolio_context.get("total_value", 1) * 100 if portfolio_context.get("total_value") else 0,
            "daily_pnl_pct": portfolio_context.get("daily_pnl_pct", 0),
            "news_summary": news_summary,
            "pattern_win_rate": pattern_stats.get("win_rate", 0),
            "pattern_count": pattern_stats.get("total_patterns", 0),
        }

        # Determine priority based on trade importance
        priority = self._determine_priority(context)

        # Generate explanation
        explanation = self._generate_explanation(context, "entry", priority)

        # Send to Telegram if enabled
        if self.notifier and explanation.short_explanation:
            self._send_telegram_notification(explanation)

        return explanation

    def explain_exit(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        hold_duration_minutes: int,
        regime: str = "unknown",
    ) -> TradeExplanation:
        """
        Generate explanation for a trade exit.

        Args:
            symbol: Trading symbol
            action: SELL (close long) or BUY (close short)
            entry_price: Original entry price
            exit_price: Exit price
            quantity: Position quantity
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            exit_reason: Why the exit occurred
            hold_duration_minutes: How long the position was held
            regime: Current market regime

        Returns:
            TradeExplanation with formatted output
        """
        context = {
            "action": f"EXIT ({action})",
            "symbol": symbol,
            "price": exit_price,
            "entry_price": entry_price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "hold_duration": hold_duration_minutes,
            "regime": regime,
            "was_profitable": pnl > 0,
        }

        priority = RequestPriority.NORMAL
        if abs(pnl_pct) > 5:
            priority = RequestPriority.HIGH

        explanation = self._generate_explanation(context, "exit", priority)

        if self.notifier and explanation.short_explanation:
            self._send_telegram_notification(explanation)

        return explanation

    def _determine_priority(self, context: Dict) -> RequestPriority:
        """Determine LLM request priority based on trade importance."""
        # High value trades
        if context.get("trade_size_pct", 0) > 5:
            return RequestPriority.HIGH

        # Approaching loss limit
        if context.get("daily_pnl_pct", 0) < -1.5:
            return RequestPriority.HIGH

        # Large portfolio
        if context.get("portfolio_value", 0) > 10000:
            return RequestPriority.HIGH

        return RequestPriority.NORMAL

    def _generate_explanation(
        self,
        context: Dict,
        explanation_type: str,
        priority: RequestPriority,
    ) -> TradeExplanation:
        """Generate explanation using LLM or rule-based fallback."""
        # Build the prompt
        if explanation_type == "entry":
            prompt = self._build_entry_prompt(context)
        else:
            prompt = self._build_exit_prompt(context)

        # Create LLM request
        request = LLMRequest(
            prompt=prompt,
            system_prompt=self.EXPLANATION_SYSTEM_PROMPT,
            priority=priority,
            purpose="trade_explanation",
            context=context,
            max_tokens=512,
            temperature=0.3,  # Lower temperature for consistency
        )

        # Get LLM response
        response = self.llm_router.route(request)

        # Build explanation object
        explanation = TradeExplanation(
            action=context.get("action", "TRADE"),
            symbol=context.get("symbol", "UNKNOWN"),
            price=context.get("price", 0),
            quantity=context.get("quantity", 0),
            position_value=context.get("position_value", context.get("price", 0) * context.get("quantity", 0)),
            signal_reason=context.get("signal_reason", ""),
            regime_context=f"Market regime: {context.get('regime', 'unknown')}",
            news_context=context.get("news_summary", ""),
            overall_confidence=context.get("confidence", 0),
            ml_confidence=context.get("confidence", 0),
            backend_used=response.backend.value,
            generation_cost=response.cost,
        )

        # Set formatted explanations
        if response.success:
            explanation.formatted_explanation = response.content
            explanation.short_explanation = self._create_short_explanation(context, explanation_type)
        else:
            # Use rule-based
            explanation.formatted_explanation = self._rule_based_explanation(context, explanation_type)
            explanation.short_explanation = self._create_short_explanation(context, explanation_type)

        return explanation

    def _build_entry_prompt(self, context: Dict) -> str:
        """Build LLM prompt for entry explanation."""
        return f"""Explain this trade entry decision:

TRADE DETAILS:
- Action: {context.get('action')} {context.get('symbol')}
- Price: ${context.get('price', 0):,.2f}
- Quantity: {context.get('quantity', 0):.4f}
- Position Value: ${context.get('position_value', 0):,.2f}

SIGNAL ANALYSIS:
- Confidence: {context.get('confidence', 0):.1%}
- Signal Reason: {context.get('signal_reason', 'ML signal')}
- Market Regime: {context.get('regime', 'unknown')}

PORTFOLIO CONTEXT:
- Portfolio Value: ${context.get('portfolio_value', 0):,.2f}
- Trade Size: {context.get('trade_size_pct', 0):.1f}% of portfolio
- Today's P&L: {context.get('daily_pnl_pct', 0):+.2f}%

PATTERN HISTORY:
- Similar patterns: {context.get('pattern_count', 0)}
- Historical win rate: {context.get('pattern_win_rate', 0):.1%}

NEWS CONTEXT:
{context.get('news_summary', 'No significant news')}

Provide a clear, concise explanation (2-3 sentences) of why this trade is being taken and the key risk factors."""

    def _build_exit_prompt(self, context: Dict) -> str:
        """Build LLM prompt for exit explanation."""
        return f"""Explain this trade exit:

TRADE DETAILS:
- Symbol: {context.get('symbol')}
- Entry Price: ${context.get('entry_price', 0):,.2f}
- Exit Price: ${context.get('price', 0):,.2f}
- P&L: ${context.get('pnl', 0):+,.2f} ({context.get('pnl_pct', 0):+.2f}%)

EXIT REASON: {context.get('exit_reason', 'Signal change')}
HOLD DURATION: {context.get('hold_duration', 0)} minutes
MARKET REGIME: {context.get('regime', 'unknown')}

Provide a brief explanation (1-2 sentences) of the exit and any lessons learned."""

    def _create_short_explanation(self, context: Dict, explanation_type: str) -> str:
        """Create a short explanation suitable for Telegram."""
        if explanation_type == "entry":
            action = context.get("action", "TRADE")
            symbol = context.get("symbol", "")
            price = context.get("price", 0)
            confidence = context.get("confidence", 0)
            regime = context.get("regime", "unknown")
            reason = context.get("signal_reason", "ML signal")

            return f"""{action} {symbol} @ ${price:,.2f}

WHY: {reason}
Regime: {regime}
Confidence: {confidence:.0%}

Position: ${context.get('position_value', 0):,.2f} ({context.get('trade_size_pct', 0):.1f}% of portfolio)"""

        else:  # exit
            symbol = context.get("symbol", "")
            entry = context.get("entry_price", 0)
            exit_price = context.get("price", 0)
            pnl = context.get("pnl", 0)
            pnl_pct = context.get("pnl_pct", 0)
            reason = context.get("exit_reason", "")
            hold = context.get("hold_duration", 0)

            emoji = "+" if pnl >= 0 else ""
            status = "PROFIT" if pnl >= 0 else "LOSS"

            return f"""CLOSED {symbol}

Entry: ${entry:,.2f}
Exit: ${exit_price:,.2f}
P&L: {emoji}${pnl:,.2f} ({pnl_pct:+.2f}%)

{status}
Reason: {reason}
Held: {hold} min"""

    def _rule_based_explanation(self, context: Dict, explanation_type: str) -> str:
        """Generate rule-based explanation when LLM unavailable."""
        if explanation_type == "entry":
            return f"""
TRADE ENTRY ANALYSIS

{context.get('action')} {context.get('symbol')} @ ${context.get('price', 0):,.2f}

Signal Details:
- Confidence: {context.get('confidence', 0):.1%}
- Trigger: {context.get('signal_reason', 'ML signal')}
- Market Regime: {context.get('regime', 'unknown')}

Risk Assessment:
- Position Size: {context.get('trade_size_pct', 0):.1f}% of portfolio
- Portfolio P&L Today: {context.get('daily_pnl_pct', 0):+.2f}%

Historical Context:
- Similar patterns in memory: {context.get('pattern_count', 0)}
- Historical win rate: {context.get('pattern_win_rate', 0):.1%}

[Rule-based analysis - LLM unavailable]
""".strip()

        else:  # exit
            result = "PROFIT" if context.get("pnl", 0) >= 0 else "LOSS"
            return f"""
TRADE EXIT ANALYSIS

{context.get('symbol')} Position Closed

Entry: ${context.get('entry_price', 0):,.2f}
Exit: ${context.get('price', 0):,.2f}
Result: {result} ${abs(context.get('pnl', 0)):,.2f} ({context.get('pnl_pct', 0):+.2f}%)

Exit Reason: {context.get('exit_reason', 'Signal change')}
Hold Duration: {context.get('hold_duration', 0)} minutes
Market Regime: {context.get('regime', 'unknown')}

[Rule-based analysis - LLM unavailable]
""".strip()

    def _send_telegram_notification(self, explanation: TradeExplanation):
        """Send explanation to Telegram."""
        if not self.notifier:
            return

        try:
            from bot.notifications import Alert, AlertType, AlertLevel

            is_entry = not explanation.action.startswith("EXIT")

            alert = Alert(
                alert_type=AlertType.TRADE_OPENED if is_entry else AlertType.TRADE_CLOSED,
                level=AlertLevel.ALERT,
                title=f"{explanation.action}: {explanation.symbol}",
                message=explanation.short_explanation,
                data=explanation.to_dict(),
            )

            self.notifier.send_alert(alert)
            logger.debug(f"Sent trade explanation to Telegram: {explanation.symbol}")

        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get explainer statistics."""
        return {
            "llm_stats": self.llm_router.get_stats(),
            "telegram_enabled": self.telegram_enabled,
            "telegram_connected": self.notifier is not None,
        }
