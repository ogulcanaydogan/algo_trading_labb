"""
RL Shadow Mode Advisor.

CRITICAL CONSTRAINTS (ENFORCED, NOT NEGOTIABLE):
1. RL CANNOT place orders
2. RL CANNOT bypass TradeGate
3. RL CANNOT override RiskBudgetEngine or Capital Preservation
4. RL CANNOT adjust leverage caps
5. RL CANNOT trade during LOCKDOWN/CRISIS
6. RL is DISABLED by default (must be explicitly enabled)

This module provides ADVISORY-ONLY outputs:
- Strategy preference distribution
- Directional bias
- Confidence estimates

All recommendations are logged for counterfactual analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from bot.rl.multi_agent_system import (
    MetaAgent,
    MarketState,
    AgentAction,
    AgentType,
    get_meta_agent,
)

logger = logging.getLogger(__name__)


class RLAdvisoryMode(Enum):
    """Operating modes for RL advisor."""
    DISABLED = "disabled"  # No RL recommendations
    SHADOW = "shadow"  # Generate recommendations, log only, no influence
    ADVISORY = "advisory"  # Recommendations influence confidence scoring only


@dataclass
class ShadowModeConfig:
    """Configuration for RL shadow mode."""

    # Master switch - default OFF
    enabled: bool = False

    # Operating mode - shadow by default even when enabled
    mode: RLAdvisoryMode = RLAdvisoryMode.SHADOW

    # Maximum influence on existing systems (when in ADVISORY mode)
    max_confidence_adjustment: float = 0.1  # +-10% max
    max_strategy_weight_adjustment: float = 0.15  # +-15% max

    # Logging
    log_all_recommendations: bool = True
    log_path: Path = field(default_factory=lambda: Path("data/rl/shadow_log.jsonl"))

    # Safety overrides - CANNOT be disabled
    respect_trade_gate: bool = True  # LOCKED
    respect_capital_preservation: bool = True  # LOCKED
    respect_risk_budget: bool = True  # LOCKED
    respect_leverage_caps: bool = True  # LOCKED

    def __post_init__(self):
        """Enforce non-negotiable constraints."""
        # These cannot be changed - always True
        self.respect_trade_gate = True
        self.respect_capital_preservation = True
        self.respect_risk_budget = True
        self.respect_leverage_caps = True


@dataclass
class RLRecommendation:
    """Advisory recommendation from RL system."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Market context
    symbol: str = ""
    regime: str = "unknown"
    preservation_level: str = "normal"

    # Strategy preferences (sums to 1.0)
    strategy_preferences: Dict[str, float] = field(default_factory=dict)

    # Directional bias
    directional_bias: str = "neutral"  # "long", "short", "neutral"
    bias_confidence: float = 0.5  # 0.0 to 1.0

    # Suggested action (ADVISORY ONLY)
    suggested_action: str = "hold"
    action_confidence: float = 0.5

    # Agent that generated the recommendation
    primary_agent: str = ""
    agent_reasoning: str = ""

    # All agent votes for logging
    agent_votes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Was this actually used?
    was_applied: bool = False
    application_reason: str = ""

    # Actual outcome (filled in after trade closes)
    actual_action: Optional[str] = None
    actual_pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime,
            "preservation_level": self.preservation_level,
            "strategy_preferences": self.strategy_preferences,
            "directional_bias": self.directional_bias,
            "bias_confidence": round(self.bias_confidence, 4),
            "suggested_action": self.suggested_action,
            "action_confidence": round(self.action_confidence, 4),
            "primary_agent": self.primary_agent,
            "agent_reasoning": self.agent_reasoning,
            "agent_votes": self.agent_votes,
            "was_applied": self.was_applied,
            "application_reason": self.application_reason,
            "actual_action": self.actual_action,
            "actual_pnl": self.actual_pnl,
        }


class RLShadowAdvisor:
    """
    Shadow mode RL advisor.

    Generates recommendations for logging and counterfactual analysis.
    CANNOT execute trades or override safety systems.
    """

    def __init__(self, config: Optional[ShadowModeConfig] = None):
        self.config = config or ShadowModeConfig()

        # Initialize meta-agent only if enabled
        self._meta_agent: Optional[MetaAgent] = None

        # Recommendation history for logging
        self._recommendations: List[RLRecommendation] = []
        self._pending_recommendations: Dict[str, RLRecommendation] = {}  # By symbol

        # Ensure log directory exists
        if self.config.log_path:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"RLShadowAdvisor initialized: enabled={self.config.enabled}, "
            f"mode={self.config.mode.value}"
        )

    def _get_meta_agent(self) -> MetaAgent:
        """Lazy-load meta agent."""
        if self._meta_agent is None:
            self._meta_agent = get_meta_agent()
        return self._meta_agent

    def is_enabled(self) -> bool:
        """Check if RL is enabled."""
        return self.config.enabled and self.config.mode != RLAdvisoryMode.DISABLED

    def get_recommendation(
        self,
        symbol: str,
        market_state: MarketState,
        preservation_level: str = "normal",
        gate_approved: bool = True,
    ) -> RLRecommendation:
        """
        Get RL recommendation for a trading opportunity.

        CRITICAL: This is ADVISORY ONLY. It cannot execute trades.

        Args:
            symbol: Trading symbol
            market_state: Current market state
            preservation_level: Current capital preservation level
            gate_approved: Whether TradeGate approved this opportunity

        Returns:
            RLRecommendation with advisory outputs
        """
        recommendation = RLRecommendation(
            symbol=symbol,
            regime=market_state.regime,
            preservation_level=preservation_level,
        )

        # If disabled, return empty recommendation
        if not self.is_enabled():
            recommendation.application_reason = "RL disabled"
            self._log_recommendation(recommendation)
            return recommendation

        # SAFETY CHECK: Do not provide recommendations in LOCKDOWN
        if preservation_level.lower() in ["lockdown", "crisis"]:
            recommendation.suggested_action = "hold"
            recommendation.action_confidence = 0.0
            recommendation.application_reason = f"Blocked: {preservation_level} mode"
            self._log_recommendation(recommendation)
            return recommendation

        # SAFETY CHECK: Do not recommend if gate rejected
        if not gate_approved:
            recommendation.suggested_action = "hold"
            recommendation.action_confidence = 0.0
            recommendation.application_reason = "Blocked: TradeGate rejected"
            self._log_recommendation(recommendation)
            return recommendation

        # Get recommendations from meta-agent
        try:
            meta_agent = self._get_meta_agent()
            best_action, all_actions = meta_agent.get_combined_action(market_state)

            # Build strategy preferences
            total_confidence = sum(a.confidence for a in all_actions)
            if total_confidence > 0:
                recommendation.strategy_preferences = {
                    a.agent_type.value: round(a.confidence / total_confidence, 4)
                    for a in all_actions
                }

            # Directional bias
            long_votes = sum(
                1 for a in all_actions if a.action in ["BUY", "LONG"]
            )
            short_votes = sum(
                1 for a in all_actions if a.action in ["SELL", "SHORT"]
            )
            total_votes = len(all_actions)

            if long_votes > short_votes:
                recommendation.directional_bias = "long"
                recommendation.bias_confidence = long_votes / total_votes
            elif short_votes > long_votes:
                recommendation.directional_bias = "short"
                recommendation.bias_confidence = short_votes / total_votes
            else:
                recommendation.directional_bias = "neutral"
                recommendation.bias_confidence = 0.5

            # Best action
            recommendation.suggested_action = best_action.action.lower()
            recommendation.action_confidence = best_action.confidence
            recommendation.primary_agent = best_action.agent_type.value
            recommendation.agent_reasoning = best_action.reasoning

            # All agent votes for logging
            recommendation.agent_votes = {
                a.agent_type.value: {
                    "action": a.action,
                    "confidence": round(a.confidence, 4),
                    "regime_match": round(a.regime_match, 4),
                    "reasoning": a.reasoning,
                }
                for a in all_actions
            }

            # Mark as applicable in shadow/advisory mode
            if self.config.mode == RLAdvisoryMode.SHADOW:
                recommendation.was_applied = False
                recommendation.application_reason = "Shadow mode: logged only"
            elif self.config.mode == RLAdvisoryMode.ADVISORY:
                recommendation.was_applied = True
                recommendation.application_reason = "Advisory mode: influences confidence"

        except Exception as e:
            logger.error(f"Error getting RL recommendation: {e}")
            recommendation.application_reason = f"Error: {str(e)}"

        # Store for later outcome recording
        self._pending_recommendations[symbol] = recommendation

        # Log recommendation
        self._log_recommendation(recommendation)

        return recommendation

    def get_confidence_adjustment(
        self,
        symbol: str,
        base_confidence: float,
        intended_action: str,
    ) -> Tuple[float, str]:
        """
        Get confidence adjustment from RL (ADVISORY mode only).

        Args:
            symbol: Trading symbol
            base_confidence: Confidence from non-RL systems
            intended_action: Action the system intends to take

        Returns:
            Tuple of (adjusted_confidence, explanation)
        """
        # If not in advisory mode, no adjustment
        if self.config.mode != RLAdvisoryMode.ADVISORY:
            return base_confidence, "RL not in advisory mode"

        # Get pending recommendation for this symbol
        rec = self._pending_recommendations.get(symbol)
        if rec is None:
            return base_confidence, "No RL recommendation available"

        # Check agreement
        rl_action = rec.suggested_action.lower()
        intended = intended_action.lower()

        # Map actions for comparison
        action_mapping = {
            "buy": "long",
            "sell": "short",
            "long": "long",
            "short": "short",
            "hold": "hold",
            "cover": "hold",
        }
        rl_mapped = action_mapping.get(rl_action, rl_action)
        intended_mapped = action_mapping.get(intended, intended)

        # Calculate adjustment
        max_adj = self.config.max_confidence_adjustment

        if rl_mapped == intended_mapped:
            # RL agrees - boost confidence proportionally
            boost = rec.action_confidence * max_adj
            adjusted = min(1.0, base_confidence + boost)
            explanation = f"RL agrees (+{boost*100:.1f}%): {rec.primary_agent}"
        elif rl_mapped == "hold" and intended_mapped != "hold":
            # RL suggests hold but system wants to trade - reduce
            reduction = rec.action_confidence * max_adj * 0.5
            adjusted = max(0.0, base_confidence - reduction)
            explanation = f"RL suggests hold (-{reduction*100:.1f}%)"
        else:
            # RL disagrees - reduce confidence
            reduction = rec.action_confidence * max_adj
            adjusted = max(0.0, base_confidence - reduction)
            explanation = f"RL disagrees (-{reduction*100:.1f}%): suggests {rl_action}"

        return adjusted, explanation

    def record_outcome(
        self,
        symbol: str,
        actual_action: str,
        pnl: float,
    ):
        """
        Record actual outcome for counterfactual analysis.

        Args:
            symbol: Trading symbol
            actual_action: Action actually taken
            pnl: Profit/loss from the trade
        """
        rec = self._pending_recommendations.pop(symbol, None)
        if rec is None:
            return

        rec.actual_action = actual_action
        rec.actual_pnl = pnl

        # Log updated recommendation
        self._log_recommendation(rec, update=True)

        # Update meta-agent if enabled
        if self.is_enabled() and self._meta_agent is not None:
            try:
                agent_type = AgentType(rec.primary_agent)
                self._meta_agent.record_outcome(
                    agent_type=agent_type,
                    pnl=pnl,
                    pnl_pct=pnl / 10000.0,  # Rough estimate
                    regime=rec.regime,
                )
            except Exception as e:
                logger.warning(f"Failed to update meta-agent: {e}")

    def _log_recommendation(self, rec: RLRecommendation, update: bool = False):
        """Log recommendation to file for analysis."""
        if not self.config.log_all_recommendations:
            return

        try:
            mode = "a" if not update else "a"
            with open(self.config.log_path, mode) as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log RL recommendation: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get RL advisor statistics."""
        stats = {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "total_recommendations": len(self._recommendations),
            "pending_recommendations": len(self._pending_recommendations),
        }

        if self._meta_agent is not None:
            stats["meta_agent_stats"] = self._meta_agent.get_stats()

        return stats

    def reset(self):
        """Reset advisor state."""
        self._recommendations = []
        self._pending_recommendations = {}
        if self._meta_agent is not None:
            self._meta_agent = None
        logger.info("RLShadowAdvisor reset")


# Singleton instance
_shadow_advisor: Optional[RLShadowAdvisor] = None


def get_shadow_advisor(config: Optional[ShadowModeConfig] = None) -> RLShadowAdvisor:
    """Get or create the shadow advisor singleton."""
    global _shadow_advisor
    if _shadow_advisor is None:
        _shadow_advisor = RLShadowAdvisor(config)
    return _shadow_advisor


def reset_shadow_advisor():
    """Reset the shadow advisor singleton (for testing)."""
    global _shadow_advisor
    if _shadow_advisor is not None:
        _shadow_advisor.reset()
    _shadow_advisor = None


def is_rl_enabled() -> bool:
    """Quick check if RL is enabled."""
    if _shadow_advisor is None:
        return False
    return _shadow_advisor.is_enabled()
