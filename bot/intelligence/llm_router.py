"""
LLM Router - Hybrid Claude/Ollama Routing.

Routes LLM requests to the appropriate backend based on importance:
- Claude API: For important decisions (costs money, higher quality)
- Ollama (local): For routine analysis (free, faster)

Cost management ensures Claude usage stays within budget.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for LLM requests."""
    CRITICAL = "critical"     # Breaking news, large position, regime change
    HIGH = "high"             # Trade decisions, risk assessment
    NORMAL = "normal"         # Routine explanations
    LOW = "low"               # Background analysis


class LLMBackend(Enum):
    """Available LLM backends."""
    CLAUDE_SONNET = "claude_sonnet"
    CLAUDE_HAIKU = "claude_haiku"
    OLLAMA = "ollama"
    RULE_BASED = "rule_based"


@dataclass
class LLMRequest:
    """Request to be routed to an LLM."""
    prompt: str
    system_prompt: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    purpose: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float = 0.7
    force_backend: Optional[LLMBackend] = None


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    backend: LLMBackend
    success: bool
    cost: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "backend": self.backend.value,
            "success": self.success,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMRouter:
    """
    Routes LLM requests to Claude or Ollama based on importance.

    Routing Logic:
    - CRITICAL priority -> Claude Sonnet (best quality)
    - HIGH priority -> Claude Haiku or Sonnet based on budget
    - NORMAL priority -> Ollama (free)
    - LOW priority -> Ollama or rule-based

    Additional triggers for Claude:
    - Portfolio value at risk > threshold
    - Trade size > threshold
    - Regime change detected
    - Breaking news
    - Approaching daily loss limit
    """

    # Thresholds for using Claude
    CLAUDE_TRIGGERS = {
        "portfolio_value_usd": 10000,      # Portfolio > $10k
        "trade_size_pct": 5.0,              # Trade > 5% of portfolio
        "daily_loss_pct_threshold": -1.5,   # Down > 1.5% today
        "news_urgency_score": 0.8,          # High urgency news
        "regime_change_confidence": 0.7,    # High confidence regime change
    }

    def __init__(
        self,
        daily_budget: float = 5.0,
        default_ollama_model: str = "llama3",
        enable_claude: bool = True,
        enable_ollama: bool = True,
    ):
        """
        Initialize the LLM Router.

        Args:
            daily_budget: Daily budget for Claude API (USD)
            default_ollama_model: Default Ollama model
            enable_claude: Whether to enable Claude backend
            enable_ollama: Whether to enable Ollama backend
        """
        self.daily_budget = daily_budget
        self.default_ollama_model = default_ollama_model
        self.enable_claude = enable_claude
        self.enable_ollama = enable_ollama

        # Initialize backends
        self._claude_client = None
        self._ollama_advisor = None
        self._ollama_available = None

        # Stats tracking
        self._stats = {
            "total_requests": 0,
            "claude_requests": 0,
            "ollama_requests": 0,
            "fallback_requests": 0,
            "total_cost": 0.0,
        }

        logger.info(f"LLM Router initialized: budget=${daily_budget}/day, ollama={enable_ollama}, claude={enable_claude}")

    @property
    def claude_client(self):
        """Lazy load Claude client."""
        if self._claude_client is None and self.enable_claude:
            try:
                from bot.llm.claude.client import ClaudeClient
                self._claude_client = ClaudeClient(daily_budget=self.daily_budget)
                logger.info("Claude client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
                self._claude_client = False  # Mark as unavailable
        return self._claude_client if self._claude_client else None

    @property
    def ollama_advisor(self):
        """Lazy load Ollama advisor."""
        if self._ollama_advisor is None and self.enable_ollama:
            try:
                from bot.llm.advisor import LLMAdvisor
                self._ollama_advisor = LLMAdvisor(model=self.default_ollama_model)
                self._ollama_available = self._ollama_advisor.is_available()
                if self._ollama_available:
                    logger.info(f"Ollama advisor initialized: {self.default_ollama_model}")
                else:
                    logger.warning("Ollama not available")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama advisor: {e}")
                self._ollama_advisor = False
                self._ollama_available = False
        return self._ollama_advisor if self._ollama_advisor else None

    def route(self, request: LLMRequest) -> LLMResponse:
        """
        Route a request to the appropriate LLM backend.

        Args:
            request: LLM request to route

        Returns:
            LLM response
        """
        import time
        start_time = time.time()
        self._stats["total_requests"] += 1

        # Determine backend
        if request.force_backend:
            backend = request.force_backend
        else:
            backend = self._select_backend(request)

        logger.debug(f"Routing {request.purpose} request to {backend.value}")

        # Execute request
        response = self._execute_request(request, backend)

        # Track latency
        response.latency_ms = (time.time() - start_time) * 1000

        # Track stats
        if response.backend == LLMBackend.CLAUDE_SONNET or response.backend == LLMBackend.CLAUDE_HAIKU:
            self._stats["claude_requests"] += 1
            self._stats["total_cost"] += response.cost
        elif response.backend == LLMBackend.OLLAMA:
            self._stats["ollama_requests"] += 1
        else:
            self._stats["fallback_requests"] += 1

        return response

    def _select_backend(self, request: LLMRequest) -> LLMBackend:
        """Select the appropriate backend based on request priority and context."""
        context = request.context

        # CRITICAL priority -> Claude Sonnet
        if request.priority == RequestPriority.CRITICAL:
            if self.claude_client:
                return LLMBackend.CLAUDE_SONNET
            elif self.ollama_advisor and self._ollama_available:
                return LLMBackend.OLLAMA
            return LLMBackend.RULE_BASED

        # HIGH priority -> Claude Haiku or Sonnet
        if request.priority == RequestPriority.HIGH:
            # Check if context triggers Claude
            if self._should_use_claude(context):
                if self.claude_client:
                    # Use Haiku for cost efficiency on HIGH, Sonnet on CRITICAL
                    return LLMBackend.CLAUDE_HAIKU
            # Fall back to Ollama
            if self.ollama_advisor and self._ollama_available:
                return LLMBackend.OLLAMA
            return LLMBackend.RULE_BASED

        # NORMAL/LOW priority -> Ollama
        if self.ollama_advisor and self._ollama_available:
            return LLMBackend.OLLAMA

        # Ultimate fallback
        return LLMBackend.RULE_BASED

    def _should_use_claude(self, context: Dict[str, Any]) -> bool:
        """Check if context triggers Claude usage."""
        # Portfolio value check
        if context.get("portfolio_value", 0) > self.CLAUDE_TRIGGERS["portfolio_value_usd"]:
            return True

        # Trade size check
        if context.get("trade_size_pct", 0) > self.CLAUDE_TRIGGERS["trade_size_pct"]:
            return True

        # Daily loss check
        if context.get("daily_pnl_pct", 0) < self.CLAUDE_TRIGGERS["daily_loss_pct_threshold"]:
            return True

        # News urgency check
        if context.get("news_urgency", 0) > self.CLAUDE_TRIGGERS["news_urgency_score"]:
            return True

        # Regime change check
        if context.get("regime_change", False):
            if context.get("regime_confidence", 0) > self.CLAUDE_TRIGGERS["regime_change_confidence"]:
                return True

        return False

    def _execute_request(self, request: LLMRequest, backend: LLMBackend) -> LLMResponse:
        """Execute a request on the specified backend."""
        try:
            if backend == LLMBackend.CLAUDE_SONNET:
                return self._execute_claude(request, model="sonnet")
            elif backend == LLMBackend.CLAUDE_HAIKU:
                return self._execute_claude(request, model="haiku")
            elif backend == LLMBackend.OLLAMA:
                return self._execute_ollama(request)
            else:
                return self._execute_rule_based(request)
        except Exception as e:
            logger.error(f"Error executing {backend.value} request: {e}")
            # Try fallback chain
            return self._execute_fallback(request, failed_backend=backend)

    def _execute_claude(self, request: LLMRequest, model: str = "haiku") -> LLMResponse:
        """Execute request via Claude API."""
        if not self.claude_client:
            return self._execute_fallback(request, failed_backend=LLMBackend.CLAUDE_HAIKU)

        try:
            response = self.claude_client.complete(
                prompt=request.prompt,
                system=request.system_prompt,
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                purpose=request.purpose,
            )

            if response is None:
                # Budget exceeded or API error
                logger.warning("Claude returned None, falling back")
                return self._execute_fallback(request, failed_backend=LLMBackend.CLAUDE_HAIKU)

            return LLMResponse(
                content=response.content,
                backend=LLMBackend.CLAUDE_SONNET if model == "sonnet" else LLMBackend.CLAUDE_HAIKU,
                success=True,
                cost=response.cost,
                tokens_used=response.input_tokens + response.output_tokens,
            )

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return self._execute_fallback(request, failed_backend=LLMBackend.CLAUDE_HAIKU)

    def _execute_ollama(self, request: LLMRequest) -> LLMResponse:
        """Execute request via local Ollama."""
        if not self.ollama_advisor or not self._ollama_available:
            return self._execute_fallback(request, failed_backend=LLMBackend.OLLAMA)

        try:
            # Use the underlying query method
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

            response_text = self.ollama_advisor._query_llm(full_prompt)

            if response_text:
                return LLMResponse(
                    content=response_text,
                    backend=LLMBackend.OLLAMA,
                    success=True,
                    cost=0.0,  # Free!
                )
            else:
                return self._execute_fallback(request, failed_backend=LLMBackend.OLLAMA)

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._execute_fallback(request, failed_backend=LLMBackend.OLLAMA)

    def _execute_rule_based(self, request: LLMRequest) -> LLMResponse:
        """Execute request using rule-based fallback."""
        # Generate a basic response based on context
        context = request.context
        purpose = request.purpose

        if purpose == "trade_explanation":
            content = self._generate_rule_based_explanation(context)
        elif purpose == "risk_assessment":
            content = self._generate_rule_based_risk(context)
        elif purpose == "regime_analysis":
            content = self._generate_rule_based_regime(context)
        else:
            content = "Analysis unavailable. LLM backends not accessible."

        return LLMResponse(
            content=content,
            backend=LLMBackend.RULE_BASED,
            success=True,
            cost=0.0,
        )

    def _execute_fallback(self, request: LLMRequest, failed_backend: LLMBackend) -> LLMResponse:
        """Execute fallback chain when primary backend fails."""
        # Fallback order: Claude Sonnet -> Claude Haiku -> Ollama -> Rule-based
        fallback_chain = [
            LLMBackend.CLAUDE_HAIKU,
            LLMBackend.OLLAMA,
            LLMBackend.RULE_BASED,
        ]

        for backend in fallback_chain:
            if backend == failed_backend:
                continue  # Skip the one that just failed

            try:
                if backend == LLMBackend.CLAUDE_HAIKU and self.claude_client:
                    return self._execute_claude(request, model="haiku")
                elif backend == LLMBackend.OLLAMA and self.ollama_advisor and self._ollama_available:
                    return self._execute_ollama(request)
                elif backend == LLMBackend.RULE_BASED:
                    return self._execute_rule_based(request)
            except Exception:
                continue

        # Ultimate fallback
        return LLMResponse(
            content="Unable to process request. All LLM backends unavailable.",
            backend=LLMBackend.RULE_BASED,
            success=False,
            error="All backends failed",
        )

    def _generate_rule_based_explanation(self, context: Dict) -> str:
        """Generate rule-based trade explanation."""
        action = context.get("action", "TRADE")
        symbol = context.get("symbol", "UNKNOWN")
        price = context.get("price", 0)
        confidence = context.get("confidence", 0)
        regime = context.get("regime", "unknown")
        reason = context.get("reason", "ML signal")

        return f"""
{action} {symbol} @ ${price:,.2f}

SIGNAL ANALYSIS:
- Action: {action}
- Confidence: {confidence:.1%}
- Market Regime: {regime}
- Signal Source: {reason}

Note: Detailed AI analysis unavailable. Using rule-based explanation.
""".strip()

    def _generate_rule_based_risk(self, context: Dict) -> str:
        """Generate rule-based risk assessment."""
        position_size = context.get("position_size_pct", 0)
        stop_loss = context.get("stop_loss_pct", 2)
        daily_pnl = context.get("daily_pnl_pct", 0)

        risk_level = "LOW"
        if position_size > 5:
            risk_level = "HIGH"
        elif position_size > 3:
            risk_level = "MEDIUM"

        return f"""
RISK ASSESSMENT: {risk_level}

- Position Size: {position_size:.1f}% of portfolio
- Stop Loss: {stop_loss:.1f}%
- Daily P&L: {daily_pnl:+.2f}%

Recommendation: {"PROCEED" if risk_level != "HIGH" else "REVIEW"}
""".strip()

    def _generate_rule_based_regime(self, context: Dict) -> str:
        """Generate rule-based regime analysis."""
        regime = context.get("regime", "unknown")
        confidence = context.get("confidence", 0)

        return f"""
MARKET REGIME: {regime.upper()}
Confidence: {confidence:.1%}

Strategy Recommendation: Follow regime-specific parameters.
""".strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = self._stats.copy()

        # Add Claude budget info if available
        if self.claude_client:
            try:
                claude_stats = self.claude_client.get_usage_stats()
                stats["claude_budget_remaining"] = claude_stats.get("budget_remaining", 0)
                stats["claude_cost_today"] = claude_stats.get("cost_today", 0)
            except Exception:
                pass

        return stats

    def reset_claude(self) -> bool:
        """
        Reset Claude client to allow reinitialization.

        Call this after adding/changing the API key.

        Returns:
            True if Claude initialized successfully
        """
        self._claude_client = None  # Reset to allow retry
        client = self.claude_client  # Trigger lazy load
        return client is not None

    def health_check(self) -> Dict[str, Any]:
        """Check health of all backends."""
        health = {
            "claude_available": False,
            "claude_budget_ok": False,
            "ollama_available": False,
            "rule_based_available": True,  # Always available
        }

        # Check Claude - reset if previously failed and API key now exists
        if self._claude_client is False and os.getenv("ANTHROPIC_API_KEY"):
            logger.info("API key found, resetting Claude client...")
            self._claude_client = None  # Reset to allow retry

        if self.claude_client:
            try:
                claude_health = self.claude_client.health_check()
                health["claude_available"] = claude_health.get("api_healthy", False)
                health["claude_budget_ok"] = claude_health.get("budget_remaining", 0) > 0.10
                health["claude_budget_remaining"] = claude_health.get("budget_remaining", 0)
            except Exception:
                pass

        # Check Ollama
        if self.ollama_advisor:
            health["ollama_available"] = self._ollama_available

        return health
